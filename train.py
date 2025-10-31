#!/usr/bin/env python3
"""
Main training script for workflow optimization.

Combines MCTS-based search with GRPO policy optimization.
Usage: python train.py --config config/training_config.yaml
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Any, List, Optional
import shutil

import torch

# Add source to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.eval import WorkflowEvaluator
from mcts import MCTSOptimizer
from grpo import QwenPolicy, GRPOTrainer, GRPOConfig
from utils.config_loader import ConfigLoader
from utils.dataset_config import get_dataset_config


# Setup logging
def setup_logging(log_path: str, level: str = 'INFO') -> None:
    """Setup logging configuration."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    logging.basicConfig(
        level=getattr(logging, level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stdout),
        ]
    )


logger = logging.getLogger(__name__)


class WorkflowTrainer:
    """
    Complete training pipeline combining MCTS + GRPO.

    Flow:
    1. Load dataset (configurable via training_config.yaml) and evaluator
    2. For each epoch:
        a. For each problem:
            - Run MCTS optimization (generate candidates)
            - Evaluate candidates (compute pass@k scores)
            - Collect trajectories
        b. Train GRPO on collected trajectories
    3. Save checkpoints and results to Drive

    Supports multiple datasets: AIME24, MATH, GSM8K, HumanEval, MBPP
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize training pipeline."""
        self.config = config
        self.device = torch.device(config['training'].get('device', 'cuda'))

        logger.info("Initializing WorkflowTrainer")
        logger.info(f"Device: {self.device}")

        # Create directories
        self._create_directories()

        # Initialize evaluator
        self.evaluator = self._init_evaluator()

        # Initialize MCTS optimizer
        self.mcts_optimizer = self._init_mcts_optimizer()

        # Initialize Qwen policy and GRPO trainer
        self.policy, self.grpo_trainer = self._init_grpo_trainer()

        # Training state
        self.epoch = 0
        self.episode = 0
        self.results = {
            'epochs': [],
            'total_mcts_problems': 0,
            'total_grpo_steps': 0,
        }

        logger.info("WorkflowTrainer initialized successfully")

    def _create_directories(self) -> None:
        """Create necessary directories."""
        paths = self.config['paths']

        dirs = [
            paths['workspace_path'],
            paths['checkpoint_path'],
            paths['log_path'],
        ]

        for dir_path in dirs:
            os.makedirs(dir_path, exist_ok=True)

    def _init_evaluator(self) -> WorkflowEvaluator:
        """Initialize evaluation system."""
        logger.info("Initializing WorkflowEvaluator")

        dataset_config = self.config['dataset']
        evaluator = WorkflowEvaluator(
            dataset_type=dataset_config['name'],
            data_path=dataset_config['data_path'],
        )

        dataset_info = evaluator.get_dataset_info()
        logger.info(f"Dataset info: {dataset_info}")

        return evaluator

    def _init_mcts_optimizer(self) -> MCTSOptimizer:
        """Initialize MCTS optimizer."""
        logger.info("Initializing MCTS Optimizer")

        paths = self.config['paths']
        mcts_config = self.config['mcts']

        # Use local AFlow by default, but allow override in config
        aflow_path = paths.get('aflow_path')
        if aflow_path == '/content/AFlow':
            # If config points to original AFlow, use local copy instead
            agentworkflow_root = os.path.dirname(os.path.abspath(__file__))
            aflow_path = os.path.join(agentworkflow_root, 'AFlow')

        optimizer = MCTSOptimizer(
            aflow_path=aflow_path,
            workspace_path=paths['workspace_path'],
            evaluator=self.evaluator,
            use_custom_llm=False,
        )

        return optimizer

    def _init_grpo_trainer(self) -> tuple:
        """Initialize Qwen policy and GRPO trainer."""
        logger.info("Initializing Qwen Policy and GRPO Trainer")

        model_config = self.config['model']
        grpo_config_dict = self.config['grpo']

        # Create policy
        policy = QwenPolicy(
            model_name=model_config['name'],
            use_lora=model_config['use_lora'],
            device=str(self.device),
        )

        # Create GRPO config
        grpo_config = GRPOConfig(
            learning_rate=grpo_config_dict['learning_rate'],
            lora_learning_rate=grpo_config_dict['lora_learning_rate'],
            num_epochs=grpo_config_dict.get('num_epochs', 1),
            batch_size=grpo_config_dict['batch_size'],
            gradient_accumulation_steps=grpo_config_dict['gradient_accumulation_steps'],
            gamma=grpo_config_dict['gamma'],
            lam=grpo_config_dict['lam'],
            entropy_coeff=grpo_config_dict['entropy_coeff'],
            value_coeff=grpo_config_dict['value_coeff'],
            max_grad_norm=grpo_config_dict['max_grad_norm'],
        )

        # Create trainer
        trainer = GRPOTrainer(
            policy=policy,
            config=grpo_config,
            device=str(self.device),
        )

        return policy, trainer

    def train(self) -> Dict[str, Any]:
        """
        Run complete training pipeline.

        Returns:
            Training results
        """
        num_epochs = self.config['training']['num_epochs']
        num_episodes = self.config['training']['num_episodes']

        logger.info(f"Starting training: {num_epochs} epochs, {num_episodes} episodes per epoch")

        try:
            for epoch in range(num_epochs):
                self.epoch = epoch
                logger.info(f"\n{'='*60}")
                logger.info(f"Epoch {epoch + 1}/{num_epochs}")
                logger.info(f"{'='*60}")

                epoch_result = self._train_epoch(num_episodes)
                self.results['epochs'].append(epoch_result)

                # Save checkpoint after each epoch
                self._save_checkpoint(epoch)

            logger.info(f"\nTraining complete!")
            self._print_summary()

            return self.results

        except Exception as e:
            logger.error(f"Error during training: {e}", exc_info=True)
            return {'success': False, 'error': str(e)}

    def _train_epoch(self, num_episodes: int) -> Dict[str, Any]:
        """Train for one epoch with MCTS + GRPO."""
        epoch_start = self.epoch
        epoch_results = {
            'epoch': epoch_start,
            'episodes': [],
            'mcts_results': [],
            'grpo_results': None,
        }

        # Collect trajectories for GRPO
        all_trajectories = []

        for episode in range(num_episodes):
            self.episode = episode
            logger.info(f"\nEpisode {episode + 1}/{num_episodes}")

            episode_result = self._train_episode()

            epoch_results['episodes'].append(episode_result)

            # Collect trajectories from MCTS
            if episode_result.get('trajectories'):
                all_trajectories.extend(episode_result['trajectories'])

        # Train GRPO on collected trajectories
        if all_trajectories:
            logger.info(f"\nTraining GRPO on {len(all_trajectories)} trajectories")

            grpo_result = self._train_grpo(all_trajectories)
            epoch_results['grpo_results'] = grpo_result

        return epoch_results

    def _train_episode(self) -> Dict[str, Any]:
        """Run one episode: MCTS optimization on sample problems."""
        episode_result = {
            'episode': self.episode,
            'mcts_results': [],
            'trajectories': [],
        }

        dataset_config = self.config['dataset']
        mcts_config = self.config['mcts']
        problems_per_episode = self.config['training'].get('problems_per_episode', 5)

        # Get sample problems from dataset (use TRAIN split, not test!)
        num_train_problems = len(self.evaluator.evaluator.train_problems)
        if num_train_problems == 0:
            logger.warning("No train problems available")
            return episode_result

        # Randomly select problems from training set
        import random
        selected_indices = random.sample(
            range(num_train_problems),
            min(problems_per_episode, num_train_problems)
        )

        for idx, problem_idx in enumerate(selected_indices):
            logger.info(f"  Problem {idx + 1}/{len(selected_indices)} (train idx: {problem_idx})")

            problem = self.evaluator.get_problem(problem_idx, split='train')

            if problem is None:
                logger.warning(f"Problem {problem_idx} not found")
                continue

            # Run MCTS optimization
            # Get dataset configuration
            dataset_name = self.config['dataset']['name']
            dataset_config = get_dataset_config(dataset_name)

            mcts_result = self.mcts_optimizer.optimize_problem(
                problem_id=problem_idx,
                problem_text=problem['question'],
                dataset_type=dataset_config.aflow_type,
                question_type=dataset_config.question_type,
                num_iterations=mcts_config['num_iterations'],
                num_samples_per_iteration=mcts_config['num_samples_per_iteration'],
                num_search_rounds=mcts_config['num_search_rounds'],
                save_outputs=True,
                split='train',  # Use training split during training
            )

            episode_result['mcts_results'].append(mcts_result)

            # Create trajectory for GRPO
            if mcts_result.get('success') and mcts_result.get('evaluation_results'):
                eval_results = mcts_result['evaluation_results']

                # Extract workflow code from files
                workflow_codes = []
                rewards = []

                for r in eval_results:
                    workflow_path = r.get('workflow_path', '')
                    if not workflow_path or not os.path.exists(workflow_path):
                        logger.warning(f"Workflow path missing or invalid: {workflow_path}")
                        continue

                    try:
                        # Read workflow code
                        with open(workflow_path, 'r', encoding='utf-8') as f:
                            workflow_code = f.read()

                        # Validate workflow code is non-empty
                        if not workflow_code.strip():
                            logger.warning(f"Empty workflow code in {workflow_path}")
                            continue

                        workflow_codes.append(workflow_code)
                        rewards.append(float(r.get('correct', False)))

                    except Exception as e:
                        logger.warning(f"Failed to read workflow from {workflow_path}: {e}")
                        continue

                # Only create trajectory if we have valid data
                if workflow_codes and len(workflow_codes) == len(rewards):
                    trajectory = {
                        'problem_id': problem_idx,
                        'problem': problem['question'],
                        'workflow_codes': workflow_codes,  # Python code strings
                        'rewards': rewards,  # 0.0 or 1.0 for incorrect/correct
                        'num_samples': len(workflow_codes),
                    }
                    episode_result['trajectories'].append(trajectory)
                    logger.info(f"Created trajectory with {len(workflow_codes)} workflow samples "
                              f"(rewards: {sum(rewards)}/{len(rewards)} correct)")
                else:
                    logger.warning(f"No valid workflows for problem {problem_idx}")

        return episode_result

    def _train_grpo(self, trajectories: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Train GRPO on collected trajectories."""
        if not trajectories:
            return {'success': False, 'error': 'No trajectories to train on'}

        logger.info(f"Training GRPO on {len(trajectories)} trajectories")

        grpo_result = self.grpo_trainer.train(
            trajectories=trajectories,
            num_epochs=self.config['grpo'].get('num_epochs', 1),
        )

        return grpo_result

    def _save_checkpoint(self, epoch: int) -> None:
        """Save training checkpoint."""
        checkpoint_dir = os.path.join(
            self.config['paths']['checkpoint_path'],
            f'epoch_{epoch}'
        )

        try:
            # Save MCTS checkpoint
            mcts_checkpoint = os.path.join(checkpoint_dir, 'mcts')
            self.mcts_optimizer.save_checkpoint(
                os.path.join(mcts_checkpoint, 'checkpoint.json')
            )

            # Save GRPO checkpoint
            grpo_checkpoint = os.path.join(checkpoint_dir, 'grpo')
            self.grpo_trainer.save_checkpoint(grpo_checkpoint)

            # Save policy
            policy_checkpoint = os.path.join(checkpoint_dir, 'policy')
            self.policy.save_checkpoint(policy_checkpoint)

            logger.info(f"Checkpoint saved to {checkpoint_dir}")

            # Copy to Drive if configured
            self._backup_to_drive(checkpoint_dir)

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")

    def _backup_to_drive(self, checkpoint_dir: str) -> None:
        """Backup checkpoint to Google Drive."""
        drive_path = self.config['paths'].get('drive_path')

        if not drive_path or not os.path.exists(drive_path):
            logger.warning(f"Drive path not available: {drive_path}")
            return

        try:
            dest_dir = os.path.join(drive_path, 'checkpoints', os.path.basename(checkpoint_dir))
            os.makedirs(dest_dir, exist_ok=True)

            # Copy files
            for root, dirs, files in os.walk(checkpoint_dir):
                for file in files:
                    src = os.path.join(root, file)
                    rel_path = os.path.relpath(src, checkpoint_dir)
                    dest = os.path.join(dest_dir, rel_path)

                    os.makedirs(os.path.dirname(dest), exist_ok=True)
                    shutil.copy2(src, dest)

            logger.info(f"Backup saved to {dest_dir}")

        except Exception as e:
            logger.warning(f"Failed to backup to Drive: {e}")

    def _print_summary(self) -> None:
        """Print training summary."""
        logger.info(f"\n{'='*60}")
        logger.info("Training Summary")
        logger.info(f"{'='*60}")

        mcts_summary = self.mcts_optimizer.get_summary()
        grpo_summary = self.grpo_trainer.get_training_summary()

        logger.info(f"MCTS: {mcts_summary}")
        logger.info(f"GRPO: {grpo_summary}")

        # Save results to file
        results_path = os.path.join(self.config['paths']['workspace_path'], 'results.json')
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)

        logger.info(f"Results saved to {results_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Train workflow optimization model')
    parser.add_argument('--config', type=str, default='config/training_config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda/cpu)')

    args = parser.parse_args()

    # Load configuration
    logger.info(f"Loading configuration from {args.config}")
    config = ConfigLoader.load(args.config)

    # Override device if specified
    if args.device:
        config['training']['device'] = args.device

    # Setup logging
    setup_logging(config['logging']['log_file'], config['logging']['level'])

    logger.info("="*60)
    logger.info("Workflow Training Pipeline")
    logger.info("="*60)

    # Create trainer
    trainer = WorkflowTrainer(config)

    # Run training
    results = trainer.train()

    if results.get('success', True):
        logger.info("Training completed successfully!")
        sys.exit(0)
    else:
        logger.error(f"Training failed: {results.get('error')}")
        sys.exit(1)


if __name__ == '__main__':
    main()

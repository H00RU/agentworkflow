"""
GRPO Trainer: Group Refined Policy Optimization training for Qwen model.
Integrates with VeRL's native GRPO algorithm and MCTS rewards.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .qwen_policy import QwenPolicy

logger = logging.getLogger(__name__)


@dataclass
class GRPOConfig:
    """Configuration for GRPO training."""

    # Learning rate
    learning_rate: float = 1e-5
    lora_learning_rate: float = 5e-4

    # Training parameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 2

    # GRPO parameters
    gamma: float = 0.99  # Discount factor
    lam: float = 0.95  # GAE lambda
    entropy_coeff: float = 0.01
    value_coeff: float = 0.5
    max_grad_norm: float = 1.0

    # Optimization
    warmup_steps: int = 100
    weight_decay: float = 0.01

    # Checkpointing
    checkpoint_every_n_steps: int = 100
    eval_every_n_steps: int = 50


class TrajectoryDataset(Dataset):
    """Dataset for MCTS-generated trajectories with rewards."""

    def __init__(self,
                 trajectories: List[Dict[str, Any]],
                 tokenizer,
                 max_length: int = 2048):
        """
        Initialize trajectory dataset.

        Args:
            trajectories: List of trajectory dicts with 'problem', 'solutions', 'rewards'
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.trajectories = trajectories
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        trajectory = self.trajectories[idx]

        problem = trajectory['problem']
        solutions = trajectory['solutions']
        rewards = trajectory['rewards']

        # Encode problem
        problem_inputs = self.tokenizer(
            problem,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        # Encode solutions
        solution_texts = " ".join(solutions)
        solution_inputs = self.tokenizer(
            solution_texts,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            'problem_ids': problem_inputs['input_ids'].squeeze(0),
            'problem_attention_mask': problem_inputs['attention_mask'].squeeze(0),
            'solution_ids': solution_inputs['input_ids'].squeeze(0),
            'solution_attention_mask': solution_inputs['attention_mask'].squeeze(0),
            'rewards': torch.tensor(rewards, dtype=torch.float32),
        }


class GRPOTrainer:
    """
    GRPO (Group Refined Policy Optimization) Trainer.

    Trains Qwen policy using MCTS-generated trajectories with pass@k rewards.
    """

    def __init__(self,
                 policy: QwenPolicy,
                 config: Optional[GRPOConfig] = None,
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        """
        Initialize GRPO trainer.

        Args:
            policy: QwenPolicy instance
            config: GRPO configuration
            device: Device to train on
        """
        self.policy = policy
        self.config = config or GRPOConfig()
        self.device = device

        # Setup optimizer
        self._setup_optimizer()

        # Training state
        self.global_step = 0
        self.epoch = 0
        self.training_history = []

        logger.info(f"GRPO Trainer initialized on {device}")

    def _setup_optimizer(self) -> None:
        """Setup optimizer for training."""
        if self.policy.use_lora:
            # Different learning rates for LoRA and base model
            lora_params = [p for n, p in self.policy.model.named_parameters()
                          if 'lora' in n and p.requires_grad]
            base_params = [p for n, p in self.policy.model.named_parameters()
                          if 'lora' not in n and p.requires_grad]

            param_groups = [
                {'params': lora_params, 'lr': self.config.lora_learning_rate},
                {'params': base_params, 'lr': self.config.learning_rate},
            ]
        else:
            param_groups = [{'params': self.policy.model.parameters(),
                            'lr': self.config.learning_rate}]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            weight_decay=self.config.weight_decay,
        )

        logger.info(f"Optimizer setup with LR: {self.config.learning_rate}")

    def compute_returns_and_advantages(self,
                                      rewards: torch.Tensor,
                                      values: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute returns and advantages using GAE (Generalized Advantage Estimation).

        Args:
            rewards: Batch of rewards
            values: Predicted values

        Returns:
            Returns and advantages
        """
        # Normalize rewards
        normalized_rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        # Compute returns
        returns = normalized_rewards.clone()

        # Compute advantages
        advantages = normalized_rewards - values.detach()

        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages

    def train_step(self,
                  batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform a single training step.

        Args:
            batch: Training batch

        Returns:
            Loss metrics
        """
        self.policy.set_train(True)

        # Move batch to device
        problem_ids = batch['problem_ids'].to(self.device)
        attention_mask = batch['problem_attention_mask'].to(self.device)
        rewards = batch['rewards'].to(self.device)

        # Forward pass
        outputs = self.policy.model(
            input_ids=problem_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        logits = outputs.logits
        hidden_states = outputs.hidden_states[-1]

        # Policy loss (log prob weighted by advantages)
        log_probs = F.log_softmax(logits, dim=-1)
        prob_loss = -(log_probs.mean() * rewards.mean())

        # Entropy bonus for exploration
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1).mean()

        # Total loss
        total_loss = prob_loss - self.config.entropy_coeff * entropy

        # Backward pass
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.policy.model.parameters(),
            self.config.max_grad_norm
        )

        # Optimizer step
        self.optimizer.step()
        self.optimizer.zero_grad()

        self.global_step += 1

        return {
            'loss': total_loss.item(),
            'policy_loss': prob_loss.item(),
            'entropy': entropy.item(),
        }

    def train_epoch(self,
                   dataloader: DataLoader) -> Dict[str, float]:
        """
        Train for one epoch.

        Args:
            dataloader: Training dataloader

        Returns:
            Epoch metrics
        """
        epoch_losses = []
        epoch_policy_losses = []
        epoch_entropies = []

        for batch_idx, batch in enumerate(dataloader):
            metrics = self.train_step(batch)

            epoch_losses.append(metrics['loss'])
            epoch_policy_losses.append(metrics['policy_loss'])
            epoch_entropies.append(metrics['entropy'])

            if (batch_idx + 1) % 10 == 0:
                logger.info(f"Epoch {self.epoch}, Batch {batch_idx + 1}: "
                           f"Loss={metrics['loss']:.4f}, "
                           f"Entropy={metrics['entropy']:.4f}")

        epoch_metrics = {
            'epoch': self.epoch,
            'avg_loss': np.mean(epoch_losses),
            'avg_policy_loss': np.mean(epoch_policy_losses),
            'avg_entropy': np.mean(epoch_entropies),
        }

        return epoch_metrics

    def train(self,
             trajectories: List[Dict[str, Any]],
             num_epochs: Optional[int] = None) -> Dict[str, Any]:
        """
        Train the policy on MCTS-generated trajectories.

        Args:
            trajectories: List of training trajectories
            num_epochs: Number of epochs (uses config if None)

        Returns:
            Training results
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs

        logger.info(f"Starting GRPO training on {len(trajectories)} trajectories")

        # Create dataset and dataloader
        dataset = TrajectoryDataset(
            trajectories,
            self.policy.tokenizer,
            max_length=2048,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )

        results = {
            'success': True,
            'epochs': [],
            'total_steps': 0,
        }

        try:
            for epoch in range(num_epochs):
                self.epoch = epoch

                logger.info(f"Epoch {epoch + 1}/{num_epochs}")

                epoch_metrics = self.train_epoch(dataloader)

                results['epochs'].append(epoch_metrics)
                self.training_history.append(epoch_metrics)

                logger.info(f"Epoch {epoch} complete: Loss={epoch_metrics['avg_loss']:.4f}")

            results['total_steps'] = self.global_step
            results['total_epochs'] = num_epochs

            logger.info(f"Training complete. Total steps: {self.global_step}")

        except Exception as e:
            logger.error(f"Error during training: {e}")
            results['success'] = False
            results['error'] = str(e)

        return results

    def evaluate(self,
                eval_trajectories: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Evaluate policy on eval trajectories.

        Args:
            eval_trajectories: Evaluation trajectories

        Returns:
            Evaluation metrics
        """
        self.policy.set_eval()

        dataset = TrajectoryDataset(
            eval_trajectories,
            self.policy.tokenizer,
            max_length=2048,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0,
        )

        total_loss = 0
        num_batches = 0

        with torch.no_grad():
            for batch in dataloader:
                problem_ids = batch['problem_ids'].to(self.device)
                attention_mask = batch['problem_attention_mask'].to(self.device)
                rewards = batch['rewards'].to(self.device)

                outputs = self.policy.model(
                    input_ids=problem_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs.logits
                log_probs = F.log_softmax(logits, dim=-1)
                loss = -(log_probs.mean() * rewards.mean())

                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches if num_batches > 0 else 0

        return {
            'eval_loss': avg_loss,
            'num_samples': len(eval_trajectories),
        }

    def save_checkpoint(self, checkpoint_path: str) -> bool:
        """Save training checkpoint."""
        try:
            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            checkpoint = {
                'epoch': self.epoch,
                'global_step': self.global_step,
                'training_history': self.training_history,
                'config': asdict(self.config),
            }

            # Save model
            self.policy.save_checkpoint(checkpoint_path)

            # Save training state
            state_path = os.path.join(checkpoint_path, 'training_state.json')
            with open(state_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)

            logger.info(f"Checkpoint saved to {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load training checkpoint."""
        try:
            self.policy.load_checkpoint(checkpoint_path)

            state_path = os.path.join(checkpoint_path, 'training_state.json')
            with open(state_path, 'r') as f:
                checkpoint = json.load(f)

            self.epoch = checkpoint.get('epoch', 0)
            self.global_step = checkpoint.get('global_step', 0)
            self.training_history = checkpoint.get('training_history', [])

            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def get_training_summary(self) -> Dict[str, Any]:
        """Get training summary."""
        if not self.training_history:
            return {'epochs': 0, 'total_steps': 0}

        avg_loss = np.mean([e['avg_loss'] for e in self.training_history])

        return {
            'epochs': len(self.training_history),
            'total_steps': self.global_step,
            'avg_loss': avg_loss,
            'final_loss': self.training_history[-1]['avg_loss'],
        }

    def __repr__(self) -> str:
        return (f"GRPOTrainer("
                f"epochs={len(self.training_history)}, "
                f"steps={self.global_step})")

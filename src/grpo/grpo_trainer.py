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


def collate_fn_pad(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function that pads sequences to the same length in a batch.

    Args:
        batch: List of samples from dataset

    Returns:
        Batched and padded tensors
    """
    # Find max lengths in this batch
    max_problem_len = max(item['problem_ids'].size(0) for item in batch)
    max_solution_len = max(item['solution_ids'].size(0) for item in batch)

    # Pad sequences
    problem_ids_list = []
    problem_attention_list = []
    solution_ids_list = []
    solution_attention_list = []
    rewards_list = []

    for item in batch:
        # Pad problem
        problem_len = item['problem_ids'].size(0)
        if problem_len < max_problem_len:
            pad_len = max_problem_len - problem_len
            problem_ids = torch.cat([item['problem_ids'], torch.zeros(pad_len, dtype=torch.long)])
            problem_attention = torch.cat([item['problem_attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
        else:
            problem_ids = item['problem_ids']
            problem_attention = item['problem_attention_mask']

        # Pad solution
        solution_len = item['solution_ids'].size(0)
        if solution_len < max_solution_len:
            pad_len = max_solution_len - solution_len
            solution_ids = torch.cat([item['solution_ids'], torch.zeros(pad_len, dtype=torch.long)])
            solution_attention = torch.cat([item['solution_attention_mask'], torch.zeros(pad_len, dtype=torch.long)])
        else:
            solution_ids = item['solution_ids']
            solution_attention = item['solution_attention_mask']

        problem_ids_list.append(problem_ids)
        problem_attention_list.append(problem_attention)
        solution_ids_list.append(solution_ids)
        solution_attention_list.append(solution_attention)
        rewards_list.append(item['rewards'])

    return {
        'problem_ids': torch.stack(problem_ids_list),
        'problem_attention_mask': torch.stack(problem_attention_list),
        'solution_ids': torch.stack(solution_ids_list),
        'solution_attention_mask': torch.stack(solution_attention_list),
        'rewards': torch.stack(rewards_list),
    }


class TrajectoryDataset(Dataset):
    """
    Dataset for GRPO training trajectories.

    Expands grouped trajectories into individual (problem, workflow_code, reward) samples.
    Each sample belongs to a group (same problem), enabling group-relative advantage computation.
    """

    def __init__(self,
                 trajectories: List[Dict[str, Any]],
                 tokenizer,
                 max_length: int = 2048):
        """
        Initialize trajectory dataset.

        Args:
            trajectories: List of trajectory dicts with 'problem', 'workflow_codes', 'rewards', 'problem_id'
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
        """
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Expand trajectories: each (problem, workflow_code, reward) becomes a sample
        self.samples = []
        for traj_idx, traj in enumerate(trajectories):
            problem = traj['problem']
            workflow_codes = traj['workflow_codes']
            rewards = traj['rewards']
            problem_id = traj['problem_id']

            # Each workflow for this problem is a sample in the same group
            for code, reward in zip(workflow_codes, rewards):
                self.samples.append({
                    'problem': problem,
                    'workflow_code': code,
                    'reward': reward,
                    'group_id': problem_id,  # For group-relative advantages
                    'traj_idx': traj_idx,
                })

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        problem = sample['problem']
        workflow_code = sample['workflow_code']
        reward = sample['reward']
        group_id = sample['group_id']

        # Create prompt: problem as instruction, workflow_code as target
        prompt = f"Generate a workflow to solve this problem:\n{problem}\n\nWorkflow:"

        # Tokenize prompt (without workflow code) to get prompt length
        prompt_inputs = self.tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_tensors="pt",
        )
        prompt_length = prompt_inputs['input_ids'].shape[1]

        # Tokenize full sequence (prompt + workflow_code)
        full_text = f"{prompt}\n{workflow_code}"
        full_inputs = self.tokenizer(
            full_text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
        )

        input_ids = full_inputs['input_ids'].squeeze(0)
        attention_mask = full_inputs['attention_mask'].squeeze(0)

        # Create response mask: 1 for workflow code tokens, 0 for prompt tokens
        response_mask = torch.zeros_like(attention_mask)
        if prompt_length < len(response_mask):
            response_mask[prompt_length:] = attention_mask[prompt_length:]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'response_mask': response_mask,  # Identifies workflow tokens for loss
            'reward': torch.tensor(reward, dtype=torch.float32),
            'group_id': torch.tensor(group_id, dtype=torch.long),
        }


def compute_grpo_advantages(
    rewards: torch.Tensor,
    response_mask: torch.Tensor,
    group_ids: torch.Tensor,
    epsilon: float = 1e-6,
    norm_by_std: bool = True
) -> torch.Tensor:
    """
    Compute group-relative advantages for GRPO.

    For each group (same problem), compute:
        advantage = (reward - group_mean) / (group_std + epsilon)

    This implements the core GRPO concept: rewards are relative to other solutions
    for the same problem, not absolute.

    Args:
        rewards: Shape (batch_size, seq_len) - token-level rewards (typically same value repeated)
        response_mask: Shape (batch_size, seq_len) - mask for response tokens
        group_ids: Shape (batch_size,) - problem/group identifier
        epsilon: Small value to prevent division by zero
        norm_by_std: Whether to normalize by std (standard GRPO)

    Returns:
        advantages: Shape (batch_size, seq_len) - token-level advantages
    """
    # Sum rewards over sequence length to get scalar reward per sample
    # Shape: (batch_size,)
    scores = (rewards * response_mask).sum(dim=-1) / (response_mask.sum(dim=-1) + epsilon)

    # Group scores by group_id
    unique_groups = torch.unique(group_ids)
    group_means = torch.zeros_like(scores)
    group_stds = torch.ones_like(scores)

    for group_id in unique_groups:
        mask = (group_ids == group_id)
        group_scores = scores[mask]

        if len(group_scores) == 1:
            # Single sample in group: advantage = 0
            group_means[mask] = 0.0
            group_stds[mask] = 1.0
        else:
            # Multiple samples: compute mean and std
            mean_val = group_scores.mean()
            std_val = group_scores.std()
            group_means[mask] = mean_val
            group_stds[mask] = std_val

    # Compute advantages
    if norm_by_std:
        # Standard GRPO: (score - mean) / (std + epsilon)
        advantages_scalar = (scores - group_means) / (group_stds + epsilon)
    else:
        # Dr.GRPO variant: (score - mean)
        advantages_scalar = scores - group_means

    # Broadcast to token-level: same advantage for all tokens in response
    # Shape: (batch_size, seq_len)
    advantages = advantages_scalar.unsqueeze(-1) * response_mask

    return advantages


def compute_grpo_loss(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    response_mask: torch.Tensor,
    advantages: torch.Tensor,
    old_log_probs: Optional[torch.Tensor] = None,
    clip_ratio: float = 0.2,
    entropy_coeff: float = 0.01
) -> Dict[str, torch.Tensor]:
    """
    Compute GRPO policy gradient loss with optional PPO clipping.

    Loss = -advantages * log_prob(actual_tokens) + entropy_bonus

    Args:
        logits: Shape (batch_size, seq_len, vocab_size) - model output logits
        input_ids: Shape (batch_size, seq_len) - actual token IDs (workflow code)
        response_mask: Shape (batch_size, seq_len) - mask for response tokens
        advantages: Shape (batch_size, seq_len) - group-relative advantages
        old_log_probs: Shape (batch_size, seq_len) - log probs from old policy (for PPO)
        clip_ratio: PPO clipping ratio
        entropy_coeff: Coefficient for entropy bonus

    Returns:
        Dict with 'loss', 'pg_loss', 'entropy', 'kl' (if old_log_probs provided)
    """
    # Shift for next-token prediction: predict token t+1 from tokens 0:t
    shift_logits = logits[:, :-1, :].contiguous()
    shift_input_ids = input_ids[:, 1:].contiguous()
    shift_response_mask = response_mask[:, 1:].contiguous()
    shift_advantages = advantages[:, 1:].contiguous()

    # Compute log probabilities of actual tokens
    # Shape: (batch_size, seq_len-1, vocab_size)
    log_probs_all = F.log_softmax(shift_logits, dim=-1)

    # Gather log_prob of actual tokens
    # Shape: (batch_size, seq_len-1)
    log_probs = log_probs_all.gather(dim=-1, index=shift_input_ids.unsqueeze(-1)).squeeze(-1)

    # Compute policy gradient loss
    # Standard PG: -A * log_prob
    pg_loss_per_token = -advantages[:, :-1] * log_probs

    # Apply response mask and average
    masked_pg_loss = pg_loss_per_token * shift_response_mask
    pg_loss = masked_pg_loss.sum() / (shift_response_mask.sum() + 1e-8)

    # Compute entropy for exploration
    probs = F.softmax(shift_logits, dim=-1)
    entropy_per_token = -(probs * log_probs_all).sum(dim=-1)
    masked_entropy = entropy_per_token * shift_response_mask
    entropy = masked_entropy.sum() / (shift_response_mask.sum() + 1e-8)

    # Total loss
    total_loss = pg_loss - entropy_coeff * entropy

    result = {
        'loss': total_loss,
        'pg_loss': pg_loss,
        'entropy': entropy,
    }

    # Optional: PPO clipping if old_log_probs provided
    if old_log_probs is not None:
        shift_old_log_probs = old_log_probs[:, 1:].contiguous()
        ratio = torch.exp(log_probs - shift_old_log_probs)
        kl = (shift_old_log_probs - log_probs) * shift_response_mask
        result['kl'] = kl.sum() / (shift_response_mask.sum() + 1e-8)

        # PPO clipped loss
        clipped_ratio = torch.clamp(ratio, 1.0 - clip_ratio, 1.0 + clip_ratio)
        pg_loss_1 = -shift_advantages * ratio
        pg_loss_2 = -shift_advantages * clipped_ratio
        pg_loss_clipped_per_token = torch.max(pg_loss_1, pg_loss_2)

        masked_pg_loss_clipped = pg_loss_clipped_per_token * shift_response_mask
        pg_loss_clipped = masked_pg_loss_clipped.sum() / (shift_response_mask.sum() + 1e-8)

        result['loss'] = pg_loss_clipped - entropy_coeff * entropy
        result['pg_loss'] = pg_loss_clipped
        result['clip_frac'] = ((ratio - 1.0).abs() > clip_ratio).float().mean()

    return result


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
        Perform a single training step with correct GRPO implementation.

        Flow:
        1. Forward pass: get logits for problem+workflow
        2. Compute group-relative advantages from rewards
        3. Compute GRPO loss: -advantages * log_prob(workflow_tokens)
        4. Backprop and update

        Args:
            batch: Training batch with 'input_ids', 'attention_mask', 'response_mask',
                   'reward', 'group_id'

        Returns:
            Loss metrics
        """
        self.policy.set_train(True)

        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        attention_mask = batch['attention_mask'].to(self.device)
        response_mask = batch['response_mask'].to(self.device)
        rewards = batch['reward'].to(self.device)
        group_ids = batch['group_id'].to(self.device)

        # Expand scalar rewards to token-level (same value for all tokens)
        # Shape: (batch_size, seq_len)
        rewards_expanded = rewards.unsqueeze(-1).expand_as(response_mask) * response_mask

        # Compute group-relative advantages
        # This is the CORE of GRPO: rewards relative to other solutions for same problem
        advantages = compute_grpo_advantages(
            rewards=rewards_expanded,
            response_mask=response_mask,
            group_ids=group_ids,
            epsilon=1e-6,
            norm_by_std=True
        )

        # Forward pass through model
        outputs = self.policy.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

        logits = outputs.logits

        # Compute GRPO loss with token-level log_prob
        loss_dict = compute_grpo_loss(
            logits=logits,
            input_ids=input_ids,
            response_mask=response_mask,
            advantages=advantages,
            old_log_probs=None,  # No reference policy yet
            clip_ratio=self.config.clip_ratio,
            entropy_coeff=self.config.entropy_coeff
        )

        total_loss = loss_dict['loss']

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

        # Return metrics
        return {
            'loss': total_loss.item(),
            'pg_loss': loss_dict['pg_loss'].item(),
            'entropy': loss_dict['entropy'].item(),
            'mean_reward': rewards.mean().item(),
            'mean_advantage': advantages[response_mask > 0].mean().item() if response_mask.sum() > 0 else 0.0,
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
            collate_fn=collate_fn_pad,  # Use custom collate function for padding
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

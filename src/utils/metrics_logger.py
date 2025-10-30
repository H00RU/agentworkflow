"""
Metrics Logger: Log and track training metrics throughout the pipeline.
Supports JSON export and visualization-ready formats.
"""

import os
import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import csv

logger = logging.getLogger(__name__)


class MetricsLogger:
    """Log and track training metrics."""

    def __init__(self, log_dir: str, experiment_name: str = None):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory to save metrics
            experiment_name: Name of experiment (timestamp if None)
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name or datetime.now().strftime("%Y%m%d_%H%M%S")

        # Create log directory
        self.exp_dir = os.path.join(log_dir, self.experiment_name)
        os.makedirs(self.exp_dir, exist_ok=True)

        # Metrics storage
        self.metrics = {
            'experiment': self.experiment_name,
            'start_time': datetime.now().isoformat(),
            'epochs': [],
            'mcts_metrics': [],
            'grpo_metrics': [],
        }

        logger.info(f"Metrics logger initialized: {self.exp_dir}")

    def log_epoch(self, epoch: int, metrics: Dict[str, Any]) -> None:
        """
        Log metrics for an epoch.

        Args:
            epoch: Epoch number
            metrics: Epoch metrics dictionary
        """
        epoch_log = {
            'epoch': epoch,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
        }

        self.metrics['epochs'].append(epoch_log)

        logger.info(f"Epoch {epoch} metrics logged")

    def log_mcts_result(self, problem_id: int, result: Dict[str, Any]) -> None:
        """
        Log MCTS optimization result.

        Args:
            problem_id: Problem ID
            result: MCTS result dictionary
        """
        mcts_log = {
            'problem_id': problem_id,
            'timestamp': datetime.now().isoformat(),
            'pass_at_k': result.get('pass_at_k', 0.0),
            'success': result.get('success', False),
            'rounds': result.get('total_rounds', 0),
        }

        self.metrics['mcts_metrics'].append(mcts_log)

        logger.info(f"MCTS result for problem {problem_id} logged: "
                   f"pass@k={mcts_log['pass_at_k']:.4f}")

    def log_grpo_step(self, step: int, loss: float, entropy: float = None) -> None:
        """
        Log GRPO training step.

        Args:
            step: Global step number
            loss: Training loss
            entropy: Entropy value (optional)
        """
        grpo_log = {
            'step': step,
            'timestamp': datetime.now().isoformat(),
            'loss': loss,
            'entropy': entropy,
        }

        self.metrics['grpo_metrics'].append(grpo_log)

        if step % 10 == 0:
            logger.info(f"GRPO step {step}: loss={loss:.4f}")

    def log_evaluation(self, eval_name: str, metrics: Dict[str, Any]) -> None:
        """
        Log evaluation metrics.

        Args:
            eval_name: Name of evaluation
            metrics: Evaluation metrics
        """
        if 'evaluations' not in self.metrics:
            self.metrics['evaluations'] = []

        eval_log = {
            'name': eval_name,
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
        }

        self.metrics['evaluations'].append(eval_log)

        logger.info(f"Evaluation '{eval_name}' logged: {metrics}")

    def save_metrics(self, filename: str = 'metrics.json') -> bool:
        """
        Save metrics to JSON file.

        Args:
            filename: Name of output file

        Returns:
            True if successful
        """
        try:
            self.metrics['end_time'] = datetime.now().isoformat()

            output_path = os.path.join(self.exp_dir, filename)

            with open(output_path, 'w') as f:
                json.dump(self.metrics, f, indent=2, default=str)

            logger.info(f"Metrics saved to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save metrics: {e}")
            return False

    def export_csv(self, output_name: str = 'metrics.csv') -> bool:
        """
        Export metrics to CSV format.

        Args:
            output_name: Name of output CSV file

        Returns:
            True if successful
        """
        try:
            output_path = os.path.join(self.exp_dir, output_name)

            # Combine all metrics into rows
            rows = []

            # Epoch metrics
            for epoch_log in self.metrics.get('epochs', []):
                row = {
                    'type': 'epoch',
                    'epoch': epoch_log['epoch'],
                    'timestamp': epoch_log['timestamp'],
                }
                row.update(epoch_log.get('metrics', {}))
                rows.append(row)

            # MCTS metrics
            for mcts_log in self.metrics.get('mcts_metrics', []):
                rows.append(mcts_log)

            # GRPO metrics
            for grpo_log in self.metrics.get('grpo_metrics', []):
                rows.append(grpo_log)

            if not rows:
                logger.warning("No metrics to export")
                return False

            # Write to CSV
            with open(output_path, 'w', newline='') as f:
                fieldnames = set()
                for row in rows:
                    fieldnames.update(row.keys())

                writer = csv.DictWriter(f, fieldnames=sorted(fieldnames))
                writer.writeheader()
                writer.writerows(rows)

            logger.info(f"Metrics exported to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to export metrics: {e}")
            return False

    def get_summary(self) -> Dict[str, Any]:
        """
        Get metrics summary.

        Returns:
            Summary statistics
        """
        summary = {
            'experiment': self.experiment_name,
            'total_epochs': len(self.metrics.get('epochs', [])),
            'total_mcts_problems': len(self.metrics.get('mcts_metrics', [])),
            'total_grpo_steps': len(self.metrics.get('grpo_metrics', [])),
        }

        # MCTS statistics
        mcts_metrics = self.metrics.get('mcts_metrics', [])
        if mcts_metrics:
            pass_at_k_values = [m.get('pass_at_k', 0.0) for m in mcts_metrics]
            summary['mcts_avg_pass_at_k'] = sum(pass_at_k_values) / len(pass_at_k_values)
            summary['mcts_max_pass_at_k'] = max(pass_at_k_values)
            summary['mcts_success_rate'] = sum(1 for m in mcts_metrics if m.get('success'))

        # GRPO statistics
        grpo_metrics = self.metrics.get('grpo_metrics', [])
        if grpo_metrics:
            losses = [m.get('loss', 0.0) for m in grpo_metrics]
            summary['grpo_avg_loss'] = sum(losses) / len(losses)
            summary['grpo_final_loss'] = losses[-1] if losses else 0.0

        return summary

    def __repr__(self) -> str:
        return f"MetricsLogger(experiment={self.experiment_name})"

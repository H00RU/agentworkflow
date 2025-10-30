"""
MCTS Optimizer: High-level interface for workflow optimization using MCTS.
Integrates with AFlow's native optimizer and WorkflowEvaluator.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Callable
from pathlib import Path

from .aflow_wrapper import AFlowOptimizerWrapper

logger = logging.getLogger(__name__)


class MCTSOptimizer:
    """
    High-level MCTS optimizer for workflow optimization.

    Provides:
    - Integration with AFlow's native MCTS
    - Evaluation of generated workflows
    - Checkpoint management
    - Pass@k score computation
    - Multi-problem optimization
    """

    def __init__(self,
                 aflow_path: Optional[str] = None,
                 workspace_path: Optional[str] = None,
                 evaluator: Optional[Any] = None,
                 use_custom_llm: bool = False,
                 custom_llm_fn: Optional[Callable] = None):
        """
        Initialize MCTSOptimizer.

        Args:
            aflow_path: Path to AFlow directory (defaults to ./AFlow)
            workspace_path: Path to output workspace (defaults to ./outputs)
            evaluator: WorkflowEvaluator instance (for evaluation)
            use_custom_llm: Whether to use custom LLM policy
            custom_llm_fn: Custom LLM function (Qwen or other)
        """
        # Get agentworkflow root for default paths
        agentworkflow_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

        if aflow_path is None:
            aflow_path = os.path.join(agentworkflow_root, 'AFlow')

        if workspace_path is None:
            workspace_path = os.path.join(agentworkflow_root, 'outputs')

        self.aflow_path = os.path.abspath(aflow_path)
        self.workspace_path = os.path.abspath(workspace_path)
        self.evaluator = evaluator
        self.use_custom_llm = use_custom_llm
        self.custom_llm_fn = custom_llm_fn

        # Create workspace
        os.makedirs(self.workspace_path, exist_ok=True)

        # Initialize AFlow wrapper (will auto-detect local AFlow)
        self.aflow_wrapper = AFlowOptimizerWrapper(
            aflow_path=self.aflow_path,
            workspace_path=self.workspace_path,
            use_custom_llm=use_custom_llm,
            custom_llm_fn=custom_llm_fn,
        )

        # Optimization history
        self.optimization_history = []
        self.problem_results = {}

    def optimize_problem(self,
                        problem_id: int,
                        problem_text: str,
                        num_iterations: int = 10,
                        num_samples_per_iteration: int = 3,
                        num_search_rounds: int = 5,
                        save_outputs: bool = True) -> Dict[str, Any]:
        """
        Run MCTS optimization on a single problem.

        Args:
            problem_id: Problem ID from dataset
            problem_text: Problem description/code
            num_iterations: MCTS iterations per round
            num_samples_per_iteration: Samples per iteration
            num_search_rounds: Number of search rounds
            save_outputs: Whether to save outputs

        Returns:
            Optimization results including best workflow and score
        """
        problem_name = f"problem_{problem_id}"

        logger.info(f"Starting MCTS optimization for problem {problem_id}")
        logger.info(f"Iterations: {num_iterations}, Samples: {num_samples_per_iteration}, "
                   f"Rounds: {num_search_rounds}")

        try:
            # Initialize optimizer for this problem
            if not self.aflow_wrapper.initialize_optimizer(
                problem_code=problem_text,
                problem_name=problem_name,
                num_iterations=num_iterations,
                num_samples_per_iteration=num_samples_per_iteration,
            ):
                return {
                    'success': False,
                    'problem_id': problem_id,
                    'error': 'Failed to initialize optimizer'
                }

            # Run optimization
            optimization_result = self.aflow_wrapper.optimize_workflow(
                initial_workflow=problem_text,
                num_rounds=num_search_rounds,
            )

            # Get generated workflows for evaluation
            generated_workflows = []
            for round_num in range(num_search_rounds):
                workflows = self.aflow_wrapper.get_generated_workflows(round_num)
                generated_workflows.extend(workflows)

            # Evaluate workflows if evaluator available
            evaluation_results = None
            pass_at_k = 0.0

            if self.evaluator and generated_workflows:
                evaluation_results = self._evaluate_workflows(
                    generated_workflows,
                    problem_id
                )

                # Compute pass@k
                correctness = [r['correct'] for r in evaluation_results]
                k = min(3, len(correctness))  # pass@3
                pass_at_k = self.evaluator.compute_pass_at_k(correctness, k)

                logger.info(f"Evaluation complete. Pass@{k}: {pass_at_k:.4f}")

            result = {
                'success': True,
                'problem_id': problem_id,
                'problem_name': problem_name,
                'best_workflow': optimization_result.get('best_workflow'),
                'best_round': optimization_result.get('best_round'),
                'total_rounds': optimization_result.get('total_rounds'),
                'generated_workflows_count': len(generated_workflows),
                'evaluation_results': evaluation_results,
                'pass_at_k': pass_at_k,
            }

            self.problem_results[problem_id] = result
            self.optimization_history.append(result)

            if save_outputs:
                self._save_optimization_result(problem_id, result)

            return result

        except Exception as e:
            logger.error(f"Error optimizing problem {problem_id}: {e}")
            return {
                'success': False,
                'problem_id': problem_id,
                'error': str(e)
            }

    def _evaluate_workflows(self,
                           workflow_paths: List[str],
                           problem_id: int) -> List[Dict[str, Any]]:
        """
        Evaluate generated workflows using WorkflowEvaluator.

        Args:
            workflow_paths: List of paths to generated workflows
            problem_id: Problem ID for evaluation

        Returns:
            List of evaluation results
        """
        if not self.evaluator:
            return []

        results = []

        for workflow_path in workflow_paths:
            try:
                # Load workflow
                with open(workflow_path, 'r') as f:
                    workflow_code = f.read()

                # Evaluate
                eval_result = self.evaluator.evaluate_workflow_response(
                    generated_text=workflow_code,
                    problem_id=problem_id,
                    split='test'
                )

                eval_result['workflow_path'] = workflow_path
                results.append(eval_result)

            except Exception as e:
                logger.warning(f"Failed to evaluate {workflow_path}: {e}")
                results.append({
                    'workflow_path': workflow_path,
                    'correct': False,
                    'error': str(e)
                })

        return results

    def optimize_batch(self,
                      problem_ids: List[int],
                      problem_texts: List[str],
                      num_iterations: int = 10,
                      num_samples_per_iteration: int = 3,
                      num_search_rounds: int = 5) -> Dict[str, Any]:
        """
        Run MCTS optimization on a batch of problems.

        Args:
            problem_ids: List of problem IDs
            problem_texts: List of problem texts
            num_iterations: MCTS iterations per round
            num_samples_per_iteration: Samples per iteration
            num_search_rounds: Number of search rounds

        Returns:
            Batch results with metrics
        """
        logger.info(f"Starting batch optimization for {len(problem_ids)} problems")

        batch_results = []
        pass_at_k_values = []

        for problem_id, problem_text in zip(problem_ids, problem_texts):
            result = self.optimize_problem(
                problem_id=problem_id,
                problem_text=problem_text,
                num_iterations=num_iterations,
                num_samples_per_iteration=num_samples_per_iteration,
                num_search_rounds=num_search_rounds,
                save_outputs=True
            )

            batch_results.append(result)

            if result.get('success'):
                pass_at_k_values.append(result.get('pass_at_k', 0.0))

        # Compute batch metrics
        avg_pass_at_k = sum(pass_at_k_values) / len(pass_at_k_values) if pass_at_k_values else 0.0

        batch_summary = {
            'total_problems': len(problem_ids),
            'successful_optimizations': sum(1 for r in batch_results if r.get('success')),
            'avg_pass_at_k': avg_pass_at_k,
            'max_pass_at_k': max(pass_at_k_values) if pass_at_k_values else 0.0,
            'min_pass_at_k': min(pass_at_k_values) if pass_at_k_values else 0.0,
            'results': batch_results,
        }

        logger.info(f"Batch optimization complete. Avg Pass@k: {avg_pass_at_k:.4f}")

        return batch_summary

    def _save_optimization_result(self, problem_id: int, result: Dict[str, Any]) -> None:
        """Save optimization result to file."""
        try:
            result_dir = os.path.join(self.workspace_path, f'results')
            os.makedirs(result_dir, exist_ok=True)

            result_file = os.path.join(result_dir, f'problem_{problem_id}_result.json')

            with open(result_file, 'w') as f:
                json.dump(result, f, indent=2, default=str)

            logger.info(f"Optimization result saved to {result_file}")

        except Exception as e:
            logger.warning(f"Failed to save optimization result: {e}")

    def save_checkpoint(self, checkpoint_path: str) -> bool:
        """Save optimization checkpoint."""
        try:
            checkpoint = {
                'aflow_path': self.aflow_path,
                'workspace_path': self.workspace_path,
                'optimization_history': self.optimization_history,
                'problem_results': self.problem_results,
            }

            os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2, default=str)

            logger.info(f"Checkpoint saved to {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return False

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """Load optimization checkpoint."""
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint = json.load(f)

            self.optimization_history = checkpoint.get('optimization_history', [])
            self.problem_results = checkpoint.get('problem_results', {})

            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def get_summary(self) -> Dict[str, Any]:
        """Get optimization summary."""
        if not self.problem_results:
            return {'total_problems': 0, 'successful': 0, 'avg_pass_at_k': 0.0}

        successful = sum(1 for r in self.problem_results.values() if r.get('success'))
        pass_at_k_values = [r.get('pass_at_k', 0.0) for r in self.problem_results.values()
                           if r.get('success')]

        return {
            'total_problems': len(self.problem_results),
            'successful': successful,
            'avg_pass_at_k': sum(pass_at_k_values) / len(pass_at_k_values)
                            if pass_at_k_values else 0.0,
            'problem_ids': list(self.problem_results.keys()),
        }

    def __repr__(self) -> str:
        return (f"MCTSOptimizer("
                f"workspace={Path(self.workspace_path).name}, "
                f"problems_optimized={len(self.problem_results)})")

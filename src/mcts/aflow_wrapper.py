"""
AFlow Optimizer Wrapper: Direct interface to AFlow's native MCTS optimizer.
Handles workflow graph loading, optimization, and state management.
"""

import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, Optional, Callable, List
import logging

logger = logging.getLogger(__name__)


class AFlowOptimizerWrapper:
    """
    Wrapper around AFlow's native Optimizer for workflow optimization.

    Provides:
    - Direct MCTS search on workflow DAGs
    - Custom LLM policy support
    - Checkpoint and state management
    - Integration with WorkflowEvaluator
    """

    def __init__(self,
                 aflow_path: Optional[str] = None,
                 workspace_path: str = None,
                 use_custom_llm: bool = False,
                 custom_llm_fn: Optional[Callable] = None):
        """
        Initialize AFlow optimizer wrapper.

        Args:
            aflow_path: Path to AFlow directory (defaults to ./AFlow)
            workspace_path: Path to output workspace (for generated workflows)
            use_custom_llm: Whether to use custom LLM policy
            custom_llm_fn: Custom function for code generation (optional)
        """
        # Use local AFlow if not specified
        if aflow_path is None:
            # Get the agentworkflow root directory
            agentworkflow_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            aflow_path = os.path.join(agentworkflow_root, 'AFlow')

        self.aflow_path = os.path.abspath(aflow_path)

        if workspace_path is None:
            agentworkflow_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            workspace_path = os.path.join(agentworkflow_root, 'outputs')

        self.workspace_path = os.path.abspath(workspace_path)
        self.use_custom_llm = use_custom_llm
        self.custom_llm_fn = custom_llm_fn

        # Add AFlow to path
        if self.aflow_path not in sys.path:
            sys.path.insert(0, self.aflow_path)

        # Add optimized path for dynamic module imports
        aflow_optimized_path = os.path.join(self.aflow_path, 'optimized')
        if aflow_optimized_path not in sys.path:
            sys.path.insert(0, aflow_optimized_path)

        # Create workspace
        os.makedirs(self.workspace_path, exist_ok=True)

        # Import AFlow components
        try:
            from scripts.optimizer import Optimizer
            self.Optimizer = Optimizer
            logger.info(f"Successfully imported AFlow Optimizer from {self.aflow_path}")
        except ImportError as e:
            logger.error(f"Failed to import AFlow Optimizer: {e}")
            raise

        self.optimizer = None
        self.current_round = 0

    def initialize_optimizer(self,
                           problem_code: str,
                           problem_name: str = "workflow",
                           num_iterations: int = 10,
                           num_samples_per_iteration: int = 3,
                           seed: int = 42) -> bool:
        """
        Initialize MCTS optimizer for a specific problem.

        Args:
            problem_code: Initial workflow code or problem statement
            problem_name: Name of the problem (used for workspace)
            num_iterations: Number of MCTS iterations
            num_samples_per_iteration: Samples per iteration
            seed: Random seed

        Returns:
            True if initialization successful
        """
        try:
            # Create problem-specific workspace
            problem_workspace = os.path.join(self.workspace_path, problem_name)
            os.makedirs(problem_workspace, exist_ok=True)

            # Initialize optimizer
            self.optimizer = self.Optimizer(
                task_name=problem_name,
                workspace=problem_workspace,
                num_iterations=num_iterations,
                num_samples_per_iteration=num_samples_per_iteration,
                seed=seed,
            )

            # Store problem code
            self.problem_code = problem_code
            self.problem_name = problem_name
            self.current_round = 0

            logger.info(f"Initialized optimizer for problem: {problem_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            return False

    def optimize_workflow(self,
                         initial_workflow: str,
                         num_rounds: int = 5) -> Dict[str, Any]:
        """
        Run MCTS optimization on workflow.

        Args:
            initial_workflow: Initial workflow code
            num_rounds: Number of optimization rounds

        Returns:
            Optimization results including best workflow and metrics
        """
        if self.optimizer is None:
            logger.error("Optimizer not initialized. Call initialize_optimizer first.")
            return {'success': False, 'error': 'Optimizer not initialized'}

        results = {
            'success': True,
            'rounds': [],
            'best_workflow': None,
            'best_score': -float('inf'),
            'best_round': -1,
        }

        try:
            for round_num in range(num_rounds):
                logger.info(f"Starting optimization round {round_num + 1}/{num_rounds}")

                round_result = self._run_optimization_round(initial_workflow, round_num)

                results['rounds'].append(round_result)

                if round_result.get('score', -float('inf')) > results['best_score']:
                    results['best_score'] = round_result['score']
                    results['best_workflow'] = round_result['best_workflow']
                    results['best_round'] = round_num

                self.current_round = round_num + 1

            results['total_rounds'] = num_rounds
            logger.info(f"Optimization complete. Best score: {results['best_score']}")

        except Exception as e:
            logger.error(f"Error during optimization: {e}")
            results['success'] = False
            results['error'] = str(e)

        return results

    def _run_optimization_round(self, workflow: str, round_num: int) -> Dict[str, Any]:
        """
        Run a single optimization round.

        Args:
            workflow: Current workflow code
            round_num: Round number

        Returns:
            Round results
        """
        try:
            # Call optimizer's optimization step
            # This uses MCTS to explore and refine the workflow
            optimized = self.optimizer.optimize(workflow)

            return {
                'round': round_num,
                'success': True,
                'best_workflow': optimized,
                'score': 0.0,  # Score would come from evaluator
            }

        except Exception as e:
            logger.error(f"Error in round {round_num}: {e}")
            return {
                'round': round_num,
                'success': False,
                'error': str(e),
            }

    def get_generated_workflows(self, round_num: Optional[int] = None) -> List[str]:
        """
        Get workflows generated in a specific round.

        Args:
            round_num: Round number (latest if None)

        Returns:
            List of workflow file paths
        """
        if round_num is None:
            round_num = self.current_round

        workflows_dir = os.path.join(
            self.workspace_path,
            self.problem_name,
            f'round_{round_num}',
            'workflows'
        )

        if not os.path.exists(workflows_dir):
            return []

        workflows = []
        for file in os.listdir(workflows_dir):
            if file.endswith('.py'):
                workflows.append(os.path.join(workflows_dir, file))

        return workflows

    def load_workflow(self, filepath: str) -> Optional[str]:
        """Load a workflow from file."""
        try:
            with open(filepath, 'r') as f:
                return f.read()
        except Exception as e:
            logger.error(f"Failed to load workflow from {filepath}: {e}")
            return None

    def save_checkpoint(self, checkpoint_path: str) -> bool:
        """Save optimization checkpoint."""
        try:
            checkpoint = {
                'problem_name': self.problem_name,
                'current_round': self.current_round,
                'aflow_path': self.aflow_path,
                'workspace_path': self.workspace_path,
            }

            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint, f, indent=2)

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

            self.problem_name = checkpoint['problem_name']
            self.current_round = checkpoint['current_round']

            logger.info(f"Checkpoint loaded from {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

    def __repr__(self) -> str:
        return (f"AFlowOptimizerWrapper("
                f"problem={self.problem_name}, "
                f"round={self.current_round})")

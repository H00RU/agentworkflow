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
                 custom_llm_fn: Optional[Callable] = None,
                 llm_config: Optional[Dict[str, Any]] = None):
        """
        Initialize AFlow optimizer wrapper.

        Args:
            aflow_path: Path to AFlow directory (defaults to ./AFlow)
            workspace_path: Path to output workspace (for generated workflows)
            use_custom_llm: Whether to use custom LLM policy
            custom_llm_fn: Custom function for code generation (optional)
            llm_config: LLM configuration dict with 'opt_model' and 'exec_model'
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
        self.llm_config = llm_config or {}

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
            from scripts.async_llm import LLMConfig
            self.Optimizer = Optimizer
            self.LLMConfig = LLMConfig
            logger.info(f"Successfully imported AFlow Optimizer from {self.aflow_path}")
        except ImportError as e:
            logger.error(f"Failed to import AFlow Optimizer: {e}")
            raise

        self.optimizer = None
        self.current_round = 0

    def _create_llm_config(self, model_name: str, api_key: str = None,
                          base_url: str = None, temperature: float = 0.7) -> Any:
        """
        Create LLM configuration for AFlow.

        Args:
            model_name: Model name (e.g., 'gpt-4o-mini')
            api_key: API key (optional, can use env var)
            base_url: Base URL for API (optional)
            temperature: Temperature for sampling

        Returns:
            LLMConfig object
        """
        config_dict = {
            "model": model_name,
            "temperature": temperature,
            "key": api_key or os.getenv("OPENAI_API_KEY"),
            "base_url": base_url or "https://api.openai.com/v1",
        }
        return self.LLMConfig(config_dict)

    def initialize_optimizer(self,
                           problem_code: str,
                           problem_name: str = "workflow",
                           dataset_type: str = "MATH",
                           question_type: str = "math",
                           num_iterations: int = 10,
                           num_samples_per_iteration: int = 3,
                           max_rounds: int = 5,
                           seed: int = 42) -> bool:
        """
        Initialize MCTS optimizer for a specific problem.

        Args:
            problem_code: Initial workflow code or problem statement
            problem_name: Name of the problem (used for workspace)
            dataset_type: Dataset type (MATH, GSM8K, HumanEval, etc.)
            question_type: Question type (math, code, qa)
            num_iterations: Number of MCTS iterations (mapped to validation_rounds)
            num_samples_per_iteration: Samples per iteration (mapped to sample)
            max_rounds: Maximum optimization rounds
            seed: Random seed

        Returns:
            True if initialization successful
        """
        try:
            # Get LLM configs from self.llm_config or create defaults
            opt_model = self.llm_config.get('opt_model', 'gpt-4o-mini')
            exec_model = self.llm_config.get('exec_model', 'gpt-4o-mini')
            api_key = self.llm_config.get('api_key', os.getenv("OPENAI_API_KEY"))
            base_url = self.llm_config.get('base_url', "https://api.openai.com/v1")

            # Create LLM configurations
            opt_llm_config = self._create_llm_config(
                model_name=opt_model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.7
            )

            exec_llm_config = self._create_llm_config(
                model_name=exec_model,
                api_key=api_key,
                base_url=base_url,
                temperature=0.0
            )

            # Define operators based on question type
            if question_type == "math":
                operators = ["Custom", "ScEnsemble", "Programmer"]
            elif question_type == "code":
                operators = ["Custom", "CustomCodeGenerate", "ScEnsemble", "Test"]
            elif question_type == "qa":
                operators = ["Custom", "AnswerGenerate", "ScEnsemble"]
            else:
                operators = ["Custom", "ScEnsemble"]

            # Initialize AFlow Optimizer with correct parameters
            # Note: optimized_path should be the workspace root, not dataset-specific
            # The Optimizer will append dataset name internally
            self.optimizer = self.Optimizer(
                dataset=dataset_type,
                question_type=question_type,
                opt_llm_config=opt_llm_config,
                exec_llm_config=exec_llm_config,
                operators=operators,
                sample=num_samples_per_iteration,  # Number of workflows to resample
                check_convergence=False,  # Disable early stopping for now
                optimized_path=self.workspace_path,  # Use workspace root, not dataset-specific path
                initial_round=1,
                max_rounds=max_rounds,
                validation_rounds=num_iterations,  # Map iterations to validation rounds
            )

            # Store problem code and metadata
            self.problem_code = problem_code
            self.problem_name = problem_name
            self.dataset_type = dataset_type
            self.question_type = question_type
            self.current_round = 0

            logger.info(f"Initialized AFlow optimizer for {dataset_type} ({question_type})")
            logger.info(f"Settings: max_rounds={max_rounds}, sample={num_samples_per_iteration}, "
                       f"validation_rounds={num_iterations}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize optimizer: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def optimize_workflow(self,
                         initial_workflow: str,
                         num_rounds: int = 5,
                         mode: str = "Graph") -> Dict[str, Any]:
        """
        Run MCTS optimization on workflow.

        Args:
            initial_workflow: Initial workflow code
            num_rounds: Number of optimization rounds (overrides max_rounds if provided)
            mode: Optimization mode ("Graph" for optimization, "Test" for testing)

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
            logger.info(f"Starting AFlow optimization in {mode} mode")

            # Run AFlow optimizer
            # Note: AFlow's optimize() method runs its own loop internally
            # We just need to call it once
            self.optimizer.optimize(mode)

            # After optimization, try to collect results
            # AFlow saves results to disk, we need to read them
            results_path = os.path.join(
                self.optimizer.root_path,
                'workflows'
            )

            if os.path.exists(results_path):
                # Try to find the best workflow from saved results
                try:
                    from scripts.optimizer_utils.data_utils import DataUtils
                    data_utils = DataUtils(self.optimizer.root_path)
                    top_rounds = data_utils.get_top_rounds(sample=1)

                    if top_rounds:
                        best_result = top_rounds[0]
                        results['best_score'] = best_result.get('score', 0.0)
                        results['best_round'] = best_result.get('round', -1)
                        results['success'] = True

                        logger.info(f"Best score: {results['best_score']} at round {results['best_round']}")
                except Exception as e:
                    logger.warning(f"Could not load optimization results: {e}")

            return results

        except Exception as e:
            logger.error(f"Optimization failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                'success': False,
                'error': str(e),
                'rounds': [],
                'best_workflow': None,
                'best_score': -float('inf'),
                'best_round': -1,
            }

    def get_generated_workflows(self, round_num: int) -> List[str]:
        """
        Get workflow file paths generated in a specific round.

        Args:
            round_num: Round number

        Returns:
            List of workflow file paths (not code content)
        """
        workflows = []

        if self.optimizer is None:
            return workflows

        try:
            # AFlow saves workflows in optimized_path/dataset/workflows/round_X/
            workflows_dir = os.path.join(
                self.optimizer.root_path,
                'workflows',
                f'round_{round_num + 1}'
            )

            if os.path.exists(workflows_dir):
                # Look for graph.py files (the actual workflow implementations)
                # AFlow generates: graph.py (workflow), prompt.py (prompts), operator.py (operators)
                # We only want to evaluate graph.py which contains the Workflow class
                graph_path = os.path.join(workflows_dir, 'graph.py')
                if os.path.exists(graph_path):
                    workflows.append(graph_path)
        except Exception as e:
            logger.warning(f"Could not load workflows from round {round_num}: {e}")

        return workflows

    def save_checkpoint(self, checkpoint_path: str) -> None:
        """
        Save optimizer state to checkpoint.

        Args:
            checkpoint_path: Path to save checkpoint
        """
        checkpoint_data = {
            'problem_name': self.problem_name,
            'problem_code': self.problem_code,
            'dataset_type': getattr(self, 'dataset_type', 'unknown'),
            'question_type': getattr(self, 'question_type', 'unknown'),
            'current_round': self.current_round,
            'workspace_path': self.workspace_path,
            'aflow_path': self.aflow_path,
        }

        os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)

        with open(checkpoint_path, 'w') as f:
            json.dump(checkpoint_data, f, indent=2)

        logger.info(f"AFlow checkpoint saved to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str) -> bool:
        """
        Load optimizer state from checkpoint.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            True if successful
        """
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)

            self.problem_name = checkpoint_data.get('problem_name')
            self.problem_code = checkpoint_data.get('problem_code')
            self.dataset_type = checkpoint_data.get('dataset_type', 'unknown')
            self.question_type = checkpoint_data.get('question_type', 'unknown')
            self.current_round = checkpoint_data.get('current_round', 0)

            logger.info(f"AFlow checkpoint loaded from {checkpoint_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return False

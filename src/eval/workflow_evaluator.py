"""
WorkflowEvaluator: Unified evaluation system for workflow optimization.
Supports multiple datasets (AIME, HumanEval) with standardized metrics.
"""

import json
import os
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np


class AnswerValidator(ABC):
    """Base class for answer validation across different dataset types."""

    @abstractmethod
    def validate(self, generated_text: str, correct_answer: Any) -> bool:
        """
        Validate if generated text contains the correct answer.

        Args:
            generated_text: Generated text from the model
            correct_answer: Ground truth answer

        Returns:
            True if answer is correct, False otherwise
        """
        pass

    @abstractmethod
    def extract_answer(self, generated_text: str) -> Optional[Any]:
        """Extract answer from generated text."""
        pass


class NumericComparisonValidator(AnswerValidator):
    """Validator for AIME (numeric answers)."""

    def extract_answer(self, generated_text: str) -> Optional[int]:
        """
        Extract numeric answer from generated text.

        Strategies (in order):
        1. Extract from <answer>...</answer> tags
        2. Extract last numeric value from text
        3. Extract numeric value from last line
        """
        # Strategy 1: <answer> tags
        answer_match = re.search(r'<answer>\s*(\d+)\s*</answer>', generated_text)
        if answer_match:
            try:
                return int(answer_match.group(1))
            except (ValueError, IndexError):
                pass

        # Strategy 2: Find all numbers and take the last one
        numbers = re.findall(r'\d+', generated_text)
        if numbers:
            try:
                return int(numbers[-1])
            except (ValueError, IndexError):
                pass

        # Strategy 3: Last line extraction
        lines = generated_text.strip().split('\n')
        for line in reversed(lines):
            numbers = re.findall(r'\d+', line)
            if numbers:
                try:
                    return int(numbers[-1])
                except (ValueError, IndexError):
                    pass

        return None

    def validate(self, generated_text: str, correct_answer: int) -> bool:
        """
        Validate numeric answer with tolerance.
        """
        extracted = self.extract_answer(generated_text)
        if extracted is None:
            return False

        # Exact match for AIME (integer answers)
        return extracted == correct_answer


class CodeExecutionValidator(AnswerValidator):
    """Validator for HumanEval (code execution based)."""

    def extract_answer(self, generated_text: str) -> Optional[str]:
        """Return the code itself as the answer."""
        return generated_text

    def validate(self, generated_text: str, correct_answer: str) -> bool:
        """
        Validate code by executing test cases.

        Args:
            generated_text: Generated Python code
            correct_answer: Test function string
        """
        try:
            # Execute generated code
            exec_globals = {}
            exec(generated_text, exec_globals)

            # Execute test
            exec(correct_answer, exec_globals)

            return True
        except Exception:
            return False


class DatasetEvaluator(ABC):
    """Abstract base class for dataset-specific evaluation."""

    def __init__(self, dataset_name: str):
        self.dataset_name = dataset_name
        self.problems: List[Dict[str, Any]] = []
        self.train_problems: List[Dict[str, Any]] = []
        self.test_problems: List[Dict[str, Any]] = []
        self.validator: Optional[AnswerValidator] = None

    @abstractmethod
    def load_dataset(self, data_path: str) -> None:
        """Load dataset from file."""
        pass

    @abstractmethod
    def split_dataset(self, train_ratio: float = 0.8) -> None:
        """Split dataset into train/test."""
        pass

    def evaluate_batch(self, generated_texts: List[str],
                      problem_indices: List[int],
                      split: str = 'test') -> Dict[str, Any]:
        """
        Evaluate a batch of generated responses.

        Args:
            generated_texts: List of generated texts
            problem_indices: Indices of problems in the split
            split: 'train' or 'test'

        Returns:
            Evaluation metrics
        """
        problems = self.test_problems if split == 'test' else self.train_problems

        correct_count = 0
        results = []

        for generated_text, idx in zip(generated_texts, problem_indices):
            problem = problems[idx]
            correct = self.validator.validate(generated_text, problem['answer'])

            if correct:
                correct_count += 1

            results.append({
                'problem_id': problem.get('pid', idx),
                'correct': correct,
                'generated': generated_text[:200],  # Store first 200 chars for debugging
            })

        accuracy = correct_count / len(generated_texts) if generated_texts else 0

        return {
            'accuracy': accuracy,
            'correct_count': correct_count,
            'total_count': len(generated_texts),
            'results': results
        }


class AIEMEvaluator(DatasetEvaluator):
    """AIME-specific evaluator following AgentFlowTarget standard."""

    def __init__(self):
        super().__init__('AIME24')
        self.validator = NumericComparisonValidator()

    def load_dataset(self, data_path: str) -> None:
        """
        Load AIME dataset from JSON file.

        Expected format:
        [
            {
                "idx": 0,
                "question": "...",
                "answer": 33,
                "pid": "0",
                ...
            },
            ...
        ]
        """
        with open(data_path, 'r') as f:
            self.problems = json.load(f)

        print(f"Loaded {len(self.problems)} AIME problems from {data_path}")

    def split_dataset(self, train_ratio: float = 0.8) -> None:
        """
        Split AIME problems into train/test (80/20 by default).
        """
        n_train = int(len(self.problems) * train_ratio)

        # Use deterministic split based on problem index
        indices = list(range(len(self.problems)))
        np.random.seed(42)  # For reproducibility
        np.random.shuffle(indices)

        train_indices = sorted(indices[:n_train])
        test_indices = sorted(indices[n_train:])

        self.train_problems = [self.problems[i] for i in train_indices]
        self.test_problems = [self.problems[i] for i in test_indices]

        print(f"Split AIME: {len(self.train_problems)} train, {len(self.test_problems)} test")

    def get_problem(self, idx: int, split: str = 'test') -> Dict[str, Any]:
        """Get a single problem."""
        problems = self.test_problems if split == 'test' else self.train_problems
        return problems[idx] if idx < len(problems) else None

    def get_batch(self, indices: List[int], split: str = 'test') -> List[Dict[str, Any]]:
        """Get a batch of problems."""
        problems = self.test_problems if split == 'test' else self.train_problems
        return [problems[i] for i in indices if i < len(problems)]


class HumanEvalEvaluator(DatasetEvaluator):
    """HumanEval-specific evaluator (code execution based)."""

    def __init__(self):
        super().__init__('HumanEval')
        self.validator = CodeExecutionValidator()

    def load_dataset(self, data_path: str) -> None:
        """Load HumanEval dataset."""
        # Placeholder - would load from actual HumanEval dataset
        pass

    def split_dataset(self, train_ratio: float = 0.8) -> None:
        """Split HumanEval dataset."""
        # Placeholder
        pass


class WorkflowEvaluator:
    """
    Unified evaluation system for workflow optimization.

    Supports multiple datasets (AIME, HumanEval) with standardized metrics.
    Follows AgentFlowTarget standard for evaluation.
    """

    def __init__(self, dataset_type: str = 'AIME24', data_path: Optional[str] = None):
        """
        Initialize WorkflowEvaluator.

        Args:
            dataset_type: 'AIME24' or 'HumanEval'
            data_path: Path to dataset file (required for AIME24)
        """
        self.dataset_type = dataset_type
        self.evaluator: Optional[DatasetEvaluator] = None

        if dataset_type == 'AIME24':
            if data_path is None:
                raise ValueError("data_path required for AIME24")

            self.evaluator = AIEMEvaluator()
            self.evaluator.load_dataset(data_path)
            self.evaluator.split_dataset(train_ratio=0.8)

        elif dataset_type == 'HumanEval':
            self.evaluator = HumanEvalEvaluator()
            if data_path:
                self.evaluator.load_dataset(data_path)

        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

    def evaluate_workflow_response(self,
                                   generated_text: str,
                                   problem_id: int,
                                   split: str = 'test') -> Dict[str, Any]:
        """
        Evaluate a single workflow optimization response.

        Args:
            generated_text: Generated response from workflow optimization
            problem_id: ID of the problem
            split: 'train' or 'test'

        Returns:
            Evaluation result with correctness and metrics
        """
        problems = (self.evaluator.test_problems
                   if split == 'test'
                   else self.evaluator.train_problems)

        if problem_id >= len(problems):
            return {'correct': False, 'error': f'Problem ID {problem_id} out of range'}

        problem = problems[problem_id]
        correct = self.evaluator.validator.validate(generated_text, problem['answer'])

        return {
            'problem_id': problem_id,
            'correct': correct,
            'expected_answer': problem['answer'],
            'problem_text': problem['question'][:200],  # First 200 chars
        }

    def evaluate_batch(self,
                      generated_texts: List[str],
                      problem_ids: List[int],
                      split: str = 'test') -> Dict[str, Any]:
        """
        Evaluate a batch of responses.

        Args:
            generated_texts: List of generated texts
            problem_ids: List of problem IDs
            split: 'train' or 'test'

        Returns:
            Batch evaluation metrics
        """
        if not generated_texts:
            return {
                'accuracy': 0.0,
                'correct_count': 0,
                'total_count': 0,
                'results': []
            }

        return self.evaluator.evaluate_batch(generated_texts, problem_ids, split)

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get information about the loaded dataset."""
        if isinstance(self.evaluator, AIEMEvaluator):
            return {
                'dataset_type': 'AIME24',
                'total_problems': len(self.evaluator.problems),
                'train_problems': len(self.evaluator.train_problems),
                'test_problems': len(self.evaluator.test_problems),
                'train_indices': [p.get('pid', i) for i, p in enumerate(self.evaluator.train_problems)],
                'test_indices': [p.get('pid', i) for i, p in enumerate(self.evaluator.test_problems)],
            }

        return {'dataset_type': self.dataset_type}

    def get_problem(self, problem_id: int, split: str = 'test') -> Optional[Dict[str, Any]]:
        """Get a specific problem."""
        problems = (self.evaluator.test_problems
                   if split == 'test'
                   else self.evaluator.train_problems)

        return problems[problem_id] if problem_id < len(problems) else None

    def get_batch_problems(self,
                          problem_ids: List[int],
                          split: str = 'test') -> List[Dict[str, Any]]:
        """Get a batch of problems."""
        problems = (self.evaluator.test_problems
                   if split == 'test'
                   else self.evaluator.train_problems)

        return [problems[i] for i in problem_ids if i < len(problems)]

    def compute_pass_at_k(self,
                         correctness: List[bool],
                         k: int = 1) -> float:
        """
        Compute pass@k metric.

        Args:
            correctness: List of boolean values indicating correctness
            k: Compute pass@k for this value of k

        Returns:
            pass@k score (0.0 to 1.0)
        """
        if not correctness:
            return 0.0

        n = len(correctness)
        c = sum(correctness)  # number of correct samples

        if c == 0:
            return 0.0

        # pass@k = 1 - C(n-c, k) / C(n, k)
        # Simplified: pass@k = 1 - ((n-c)/n * (n-c-1)/(n-1) * ... * (n-c-k+1)/(n-k+1))

        if k > n:
            return 1.0 if c == n else 0.0

        # Calculate the combination-based formula
        numerator = 1.0
        denominator = 1.0

        for i in range(k):
            numerator *= (n - c - i)
            denominator *= (n - i)

        return 1.0 - (numerator / denominator) if denominator > 0 else 1.0

    def __repr__(self) -> str:
        return f"WorkflowEvaluator(dataset_type={self.dataset_type})"

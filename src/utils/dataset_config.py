"""
Dataset Configuration System

Provides centralized configuration for all supported datasets.
Eliminates hardcoded dataset-specific logic throughout the codebase.
"""

from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class DatasetMetadata:
    """Metadata definition for a dataset"""
    name: str                    # Dataset name
    aflow_type: str             # Corresponding type in AFlow
    question_type: str          # Question type (math/code/qa)
    validator_type: str         # Validator type (numeric/code/string)
    data_path_template: str     # Data path template
    description: str = ""       # Optional description

    def get_data_path(self, base_path: str = './data') -> str:
        """Generate full data path"""
        return f"{base_path}/{self.data_path_template}"


# Dataset Registry - Central source of truth for all datasets
DATASET_REGISTRY: Dict[str, DatasetMetadata] = {
    'AIME24': DatasetMetadata(
        name='AIME24',
        aflow_type='MATH',
        question_type='math',
        validator_type='numeric',
        data_path_template='aime24/data.json',
        description='AIME 2024 competition math problems'
    ),
    'MATH': DatasetMetadata(
        name='MATH',
        aflow_type='MATH',
        question_type='math',
        validator_type='numeric',
        data_path_template='math/math_validate.jsonl',
        description='MATH dataset for mathematical problem solving'
    ),
    'GSM8K': DatasetMetadata(
        name='GSM8K',
        aflow_type='GSM8K',
        question_type='math',
        validator_type='numeric',
        data_path_template='gsm8k/gsm8k_validate.jsonl',
        description='Grade School Math 8K problems'
    ),
    'HumanEval': DatasetMetadata(
        name='HumanEval',
        aflow_type='HumanEval',
        question_type='code',
        validator_type='code',
        data_path_template='humaneval_public_test.jsonl',
        description='HumanEval benchmark for code generation'
    ),
    'MBPP': DatasetMetadata(
        name='MBPP',
        aflow_type='MBPP',
        question_type='code',
        validator_type='code',
        data_path_template='mbpp_public_test.jsonl',
        description='Mostly Basic Python Problems'
    ),
}


def get_dataset_config(dataset_name: str) -> DatasetMetadata:
    """
    Get configuration for a dataset.

    Args:
        dataset_name: Name of the dataset

    Returns:
        DatasetMetadata object

    Raises:
        ValueError: If dataset is not recognized
    """
    if dataset_name not in DATASET_REGISTRY:
        supported = get_all_supported_datasets()
        raise ValueError(
            f"Unknown dataset: '{dataset_name}'. "
            f"Supported datasets: {supported}"
        )
    return DATASET_REGISTRY[dataset_name]


def get_all_supported_datasets() -> list:
    """Get list of all supported datasets"""
    return list(DATASET_REGISTRY.keys())


def register_dataset(metadata: DatasetMetadata) -> None:
    """
    Register a new dataset.

    Args:
        metadata: DatasetMetadata object for the new dataset
    """
    DATASET_REGISTRY[metadata.name] = metadata


def is_dataset_supported(dataset_name: str) -> bool:
    """Check if a dataset is supported"""
    return dataset_name in DATASET_REGISTRY


def get_datasets_by_question_type(question_type: str) -> list:
    """Get all datasets of a specific question type"""
    return [
        name for name, config in DATASET_REGISTRY.items()
        if config.question_type == question_type
    ]


def get_datasets_by_validator_type(validator_type: str) -> list:
    """Get all datasets using a specific validator type"""
    return [
        name for name, config in DATASET_REGISTRY.items()
        if config.validator_type == validator_type
    ]

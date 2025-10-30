"""
Configuration Loader: Load and validate YAML configuration files.
Supports training configuration with MCTS and GRPO parameters.
"""

import os
import json
import logging
from typing import Dict, Any, Optional
from pathlib import Path

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False
    logging.warning("PyYAML not installed. Using JSON for configs.")

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate training configurations."""

    DEFAULT_CONFIG = {
        # Dataset
        'dataset': {
            'name': 'AIME24',
            'data_path': './data/aime24/data.json',
            'train_ratio': 0.8,
            'sample_size': None,  # Use all if None
        },

        # MCTS parameters
        'mcts': {
            'num_iterations': 10,
            'num_samples_per_iteration': 3,
            'num_search_rounds': 5,
        },

        # GRPO parameters
        'grpo': {
            'learning_rate': 1e-5,
            'lora_learning_rate': 5e-4,
            'num_epochs': 3,
            'batch_size': 4,
            'gradient_accumulation_steps': 2,
            'gamma': 0.99,
            'lam': 0.95,
            'entropy_coeff': 0.01,
            'value_coeff': 0.5,
            'max_grad_norm': 1.0,
        },

        # Model parameters
        'model': {
            'name': 'Qwen/Qwen2-7B',
            'use_lora': True,
            'lora_rank': 8,
            'lora_alpha': 16,
        },

        # Training parameters
        'training': {
            'num_epochs': 3,
            'num_episodes': 3,
            'problems_per_episode': 5,
            'device': 'cuda',
        },

        # Paths (all relative to agentworkflow root, auto-detected if not specified)
        'paths': {
            'aflow_path': './AFlow',  # Local AFlow directory
            'workspace_path': './outputs',
            'checkpoint_path': './checkpoints',
            'log_path': './logs',
            'drive_path': '/content/drive/MyDrive/agentworkflow',
        },

        # Checkpoint and logging
        'checkpoint': {
            'save_every_n_steps': 100,
            'save_every_n_epochs': 1,
        },

        'logging': {
            'level': 'INFO',
            'save_to_file': True,
            'log_file': './logs/training.log',
        },
    }

    @classmethod
    def load(cls, config_path: str) -> Dict[str, Any]:
        """
        Load configuration from file.

        Args:
            config_path: Path to config file (YAML or JSON)

        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            logger.warning(f"Config file not found: {config_path}. Using defaults.")
            return cls.DEFAULT_CONFIG.copy()

        try:
            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                if not YAML_AVAILABLE:
                    logger.error("PyYAML not installed. Cannot load YAML config.")
                    return cls.DEFAULT_CONFIG.copy()

                with open(config_path, 'r') as f:
                    config = yaml.safe_load(f)
            else:
                # Assume JSON
                with open(config_path, 'r') as f:
                    config = json.load(f)

            # Merge with defaults
            merged_config = cls._merge_configs(cls.DEFAULT_CONFIG.copy(), config)

            logger.info(f"Config loaded from {config_path}")
            return merged_config

        except Exception as e:
            logger.error(f"Failed to load config: {e}")
            return cls.DEFAULT_CONFIG.copy()

    @classmethod
    def save(cls, config: Dict[str, Any], config_path: str) -> bool:
        """
        Save configuration to file.

        Args:
            config: Configuration dictionary
            config_path: Path to save config

        Returns:
            True if successful
        """
        try:
            os.makedirs(os.path.dirname(config_path), exist_ok=True)

            if config_path.endswith('.yaml') or config_path.endswith('.yml'):
                if not YAML_AVAILABLE:
                    logger.error("PyYAML not installed. Cannot save YAML config.")
                    return False

                with open(config_path, 'w') as f:
                    yaml.safe_dump(config, f, default_flow_style=False)
            else:
                # Save as JSON
                with open(config_path, 'w') as f:
                    json.dump(config, f, indent=2)

            logger.info(f"Config saved to {config_path}")
            return True

        except Exception as e:
            logger.error(f"Failed to save config: {e}")
            return False

    @classmethod
    def _merge_configs(cls, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """
        Merge override config into base config (recursive).

        Args:
            base: Base configuration
            override: Override configuration

        Returns:
            Merged configuration
        """
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                base[key] = cls._merge_configs(base[key], value)
            else:
                base[key] = value

        return base

    @classmethod
    def validate(cls, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate configuration.

        Args:
            config: Configuration to validate

        Returns:
            (is_valid, list of error messages)
        """
        errors = []

        # Check required keys
        required_sections = ['dataset', 'mcts', 'grpo', 'model', 'training', 'paths']
        for section in required_sections:
            if section not in config:
                errors.append(f"Missing section: {section}")

        # Validate dataset
        if 'dataset' in config:
            dataset = config['dataset']
            if dataset.get('name') not in ['AIME24', 'HumanEval']:
                errors.append(f"Invalid dataset name: {dataset.get('name')}")

            if 'data_path' in dataset and not os.path.exists(dataset['data_path']):
                errors.append(f"Dataset not found: {dataset['data_path']}")

        # Validate paths
        if 'paths' in config:
            paths = config['paths']
            aflow_path = paths.get('aflow_path')
            if aflow_path and not os.path.exists(aflow_path):
                errors.append(f"AFlow path not found: {aflow_path}")

        # Validate numeric parameters
        numeric_params = [
            ('mcts.num_iterations', config.get('mcts', {}).get('num_iterations')),
            ('grpo.learning_rate', config.get('grpo', {}).get('learning_rate')),
            ('training.num_epochs', config.get('training', {}).get('num_epochs')),
        ]

        for param_name, param_value in numeric_params:
            if param_value is not None and param_value <= 0:
                errors.append(f"Invalid {param_name}: {param_value}")

        return len(errors) == 0, errors


from typing import Tuple

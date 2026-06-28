"""
Configuration management utilities.

Handles loading and validation of the main YAML configuration file.
"""

import yaml
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def load_config(config_path: str = "configs/main.yaml") -> Dict[str, Any]:
    """
    Load and validate the main configuration file.
    
    Args:
        config_path: Path to YAML configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If required keys are missing
    """
    path = Path(config_path)
    
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    
    with open(path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Validate required keys
    required_keys = ['data', 'random_state', 'results_dir']
    missing = [k for k in required_keys if k not in config]
    if missing:
        raise ValueError(f"Missing required configuration keys: {missing}")
    
    logger.info(f"Loaded configuration from {path}")
    
    return config


def get_experiment_grid(config: Dict[str, Any]):
    """
    Generate the experiment grid from configuration.
    
    Args:
        config: Configuration dictionary
        
    Yields:
        Tuples of (task, model, strategy)
    """
    experiments = config.get('experiments', {})
    tasks = experiments.get('tasks', ['binary', 'multi'])
    models = experiments.get('models', ['lr', 'rf', 'xgb'])
    strategies = experiments.get('strategies', ['s0', 's1', 's2a', 's2b'])
    
    for task in tasks:
        for model in models:
            for strategy in strategies:
                yield task, model, strategy

"""
Logging configuration and utilities.

Provides consistent logging setup across all modules.
"""

import logging
import sys
import os
from pathlib import Path
from datetime import datetime
from typing import Optional

# Set matplotlib to non-GUI backend BEFORE any matplotlib imports
# This prevents tkinter thread-safety issues
import matplotlib
matplotlib.use('Agg')


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_format: Optional[str] = None
) -> None:
    """
    Configure logging for the experiment pipeline.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional file path for logging output
        log_format: Optional custom format string
    """
    if log_format is None:
        log_format = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    
    # Create handlers with UTF-8 encoding for Windows compatibility
    # This prevents UnicodeEncodeError with special characters
    if sys.platform == 'win32':
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    stream_handler = logging.StreamHandler(sys.stdout)
    handlers = [stream_handler]
    
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        handlers.append(file_handler)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers,
        force=True
    )
    
    # Reduce verbosity of some libraries
    logging.getLogger('matplotlib').setLevel(logging.WARNING)
    logging.getLogger('sklearn').setLevel(logging.WARNING)
    

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger with the specified name.
    
    Args:
        name: Logger name (typically __name__)
        
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def log_experiment_start(experiment_id: str, task: str, model: str, strategy: str):
    """Log the start of an experiment."""
    logger = logging.getLogger('experiment')
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: {experiment_id}")
    logger.info(f"  Task:     {task}")
    logger.info(f"  Model:    {model}")
    logger.info(f"  Strategy: {strategy}")
    logger.info("=" * 60)


def log_experiment_end(experiment_id: str, metrics: dict, duration: float):
    """Log the completion of an experiment."""
    logger = logging.getLogger('experiment')
    logger.info("-" * 40)
    logger.info(f"COMPLETED: {experiment_id}")
    logger.info(f"  Accuracy: {metrics['overall']['accuracy']:.4f}")
    logger.info(f"  Macro F1: {metrics['overall']['macro_f1']:.4f}")
    logger.info(f"  G-Mean:   {metrics['overall']['g_mean']:.4f}")
    logger.info(f"  Duration: {duration:.2f}s")
    logger.info("-" * 40)

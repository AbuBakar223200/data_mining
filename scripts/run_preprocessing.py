#!/usr/bin/env python3
"""
Preprocessing Script for UNSW-NB15 Dataset
==========================================

This script runs the preprocessing pipeline independently,
useful for debugging or inspecting the preprocessed data.

Usage:
    python scripts/run_preprocessing.py
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import logging
from datetime import datetime
import numpy as np
import pandas as pd

from src.data import DataLoader, UNSWPreprocessor
from src.utils import load_config, setup_logging
from src.evaluation import plot_class_distribution

# Setup logging
setup_logging(level="INFO")
logger = logging.getLogger(__name__)


def main():
    """Run preprocessing pipeline and save diagnostics."""
    logger.info("=" * 60)
    logger.info("UNSW-NB15 PREPROCESSING PIPELINE")
    logger.info("=" * 60)
    
    # Load configuration
    config = load_config("configs/main.yaml")
    results_dir = Path(config['results_dir'])
    
    # Create output directories
    processed_dir = results_dir / 'processed'
    processed_dir.mkdir(parents=True, exist_ok=True)
    figures_dir = results_dir / 'figures'
    figures_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("\n--- Loading Data ---")
    loader = DataLoader(config)
    train_df, test_df = loader.load_all()
    
    # Run preprocessing
    logger.info("\n--- Running Preprocessing ---")
    preprocessor = UNSWPreprocessor(config)
    preprocessor.fit_transform(train_df, test_df)
    
    # Save metadata and diagnostics
    preprocessor.save_metadata(str(processed_dir / 'preprocessing_metadata.json'))
    
    # Log class distributions
    logger.info("\n--- Class Distribution (Binary) ---")
    binary_dist = preprocessor.get_class_distribution('binary')
    for split, dist in binary_dist.items():
        logger.info(f"  {split}: {dist}")
    
    logger.info("\n--- Class Distribution (Multiclass) ---")
    multi_dist = preprocessor.get_class_distribution('multi')
    for split, dist in multi_dist.items():
        logger.info(f"  {split}:")
        for cls, count in sorted(dist.items(), key=lambda x: x[1], reverse=True):
            pct = count / sum(dist.values()) * 100
            logger.info(f"    {cls}: {count:,} ({pct:.2f}%)")
    
    # Plot training class distribution
    multiclass_labels = config['data'].get('multiclass_labels', [])
    _, y_train_multi, _, _, _, _ = preprocessor.get_splits('multi')
    
    # Create class count dict
    unique, counts = np.unique(y_train_multi, return_counts=True)
    if multiclass_labels:
        class_counts = {multiclass_labels[i]: int(c) for i, c in zip(unique, counts)}
    else:
        class_counts = {str(i): int(c) for i, c in zip(unique, counts)}
    
    plot_class_distribution(
        class_counts=class_counts,
        save_path=str(figures_dir / 'training_class_distribution.png'),
        title='UNSW-NB15 Training Set Class Distribution'
    )
    
    logger.info("\n--- Feature Information ---")
    logger.info(f"Total features after preprocessing: {len(preprocessor.feature_names)}")
    logger.info(f"Numerical features: {len(preprocessor.numerical_cols)}")
    logger.info(f"Categorical features (pre-encoding): {len(preprocessor.categorical_cols)}")
    
    # Save feature names
    feature_file = processed_dir / 'feature_names.txt'
    with open(feature_file, 'w') as f:
        for i, name in enumerate(preprocessor.feature_names):
            f.write(f"{i+1}. {name}\n")
    logger.info(f"Saved feature names to {feature_file}")
    
    logger.info("\n" + "=" * 60)
    logger.info("PREPROCESSING COMPLETE")
    logger.info("=" * 60)
    

if __name__ == '__main__':
    main()

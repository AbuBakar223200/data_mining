"""
Data loading utilities for UNSW-NB15 dataset.

This module provides a clean interface for loading the official
training and testing CSV files with validation.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Load UNSW-NB15 dataset from CSV files.
    
    Handles loading of the official training and testing splits
    with basic validation to ensure file integrity.
    
    Attributes:
        train_path: Path to training CSV file
        test_path: Path to testing CSV file
    """
    
    def __init__(self, config: dict):
        """
        Initialize DataLoader with configuration.
        
        Args:
            config: Configuration dictionary containing data paths
        """
        self.train_path = Path(config['data']['train_path'])
        self.test_path = Path(config['data']['test_path'])
        self.config = config
        self._validate_paths()
    
    def _validate_paths(self) -> None:
        """Validate that required data files exist."""
        if not self.train_path.exists():
            raise FileNotFoundError(
                f"Training file not found: {self.train_path}\n"
                "Please ensure UNSW_NB15_training-set.csv is in the dataset/ directory."
            )
        if not self.test_path.exists():
            raise FileNotFoundError(
                f"Test file not found: {self.test_path}\n"
                "Please ensure UNSW_NB15_testing-set.csv is in the dataset/ directory."
            )
        logger.info("Dataset files validated successfully")
    
    def load_train(self) -> pd.DataFrame:
        """
        Load training dataset.
        
        Returns:
            DataFrame containing training data (175,341 rows expected)
        """
        logger.info(f"Loading training data from {self.train_path}")
        df = pd.read_csv(self.train_path)
        logger.info(f"Loaded {len(df):,} training samples with {len(df.columns)} columns")
        
        # Auditor T001: Schema Validation
        self._validate_schema(df, "Training Set")
        
        return df
    
    def _validate_schema(self, df: pd.DataFrame, set_name: str) -> None:
        """
        Validate that the dataframe contains all required columns from config.
        
        Args:
            df: DataFrame to validate
            set_name: Name of the dataset for logging
            
        Raises:
            ValueError: If required columns are missing
        """
        # 1. Check target columns        
        required_cols = []
        # Add targets
        required_cols.append(self.config['data']['target_binary'])
        required_cols.append(self.config['data']['target_multiclass'])
        
        # Add categorical columns
        required_cols.extend(self.config['data']['categorical_columns'])
        
        # Check for missing columns
        missing = [col for col in required_cols if col not in df.columns]
        
        if missing:
            raise ValueError(f"{set_name} schema validation failed. Missing columns: {missing}")
        
        logger.info(f"{set_name} schema validated successfully (all required columns present).")
    
    def load_test(self) -> pd.DataFrame:
        """
        Load test dataset.
        
        Note: Test set should ONLY be used for final evaluation,
        never for training, validation, or hyperparameter tuning.
        
        Returns:
            DataFrame containing test data (82,332 rows expected)
        """
        logger.info(f"Loading test data from {self.test_path}")
        df = pd.read_csv(self.test_path)
        logger.info(f"Loaded {len(df):,} test samples with {len(df.columns)} columns")
        
        # Auditor T001: Schema Validation
        self._validate_schema(df, "Test Set")
        
        return df
    
    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Load both training and test datasets.
        
        Returns:
            Tuple of (train_df, test_df)
        """
        return self.load_train(), self.load_test()

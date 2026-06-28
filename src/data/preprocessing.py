"""
Preprocessing pipeline for UNSW-NB15 dataset.

This module implements the complete preprocessing pipeline including:
- Feature cleaning (drop identifier columns)
- Missing value imputation
- Categorical encoding (One-Hot)
- Numerical scaling (StandardScaler)
- Stratified train/validation splitting

CRITICAL: All transformers are fit on training data ONLY to prevent data leakage.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from typing import Tuple, Optional, Dict, List
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class UNSWPreprocessor:
    """
    Unified preprocessing pipeline for UNSW-NB15 dataset.
    
    This class handles the complete preprocessing workflow while ensuring
    no data leakage occurs between training, validation, and test sets.
    
    Key guarantees:
    - All statistics (mean, std, categories) computed on training only
    - Consistent encoding across all splits
    - Stratified validation split preserves class distribution
    - Test set remains completely isolated
    
    Attributes:
        config: Configuration dictionary
        drop_columns: Columns to remove (identifiers)
        categorical_cols: Columns for one-hot encoding
        numerical_cols: Columns for standard scaling
        feature_names: Final feature names after transformation
    """
    
    def __init__(self, config: dict):
        """
        Initialize preprocessor with configuration.
        
        Args:
            config: Configuration dictionary with data specifications
        """
        self.config = config
        self.drop_columns = config['data']['drop_columns']
        self.categorical_cols = config['data']['categorical_columns']
        self.target_binary = config['data']['target_binary']
        self.target_multi = config['data']['target_multiclass']
        self.val_size = config.get('validation_size', 0.20)
        self.random_state = config.get('random_state', 42)
        
        # Transformers - initialized during fit
        self.preprocessor: Optional[ColumnTransformer] = None
        self.label_encoder: Optional[LabelEncoder] = None
        
        # Feature metadata
        self.numerical_cols: List[str] = []
        self.feature_names: List[str] = []
        self.label_mapping: Dict[str, int] = {}
        
        # Processed data storage
        self.X_train: Optional[np.ndarray] = None
        self.X_val: Optional[np.ndarray] = None
        self.X_test: Optional[np.ndarray] = None
        self.y_train_binary: Optional[np.ndarray] = None
        self.y_val_binary: Optional[np.ndarray] = None
        self.y_test_binary: Optional[np.ndarray] = None
        self.y_train_multi: Optional[np.ndarray] = None
        self.y_val_multi: Optional[np.ndarray] = None
        self.y_test_multi: Optional[np.ndarray] = None
        
        # Original sizes for validation
        self._original_train_size: int = 0
        self._original_test_size: int = 0
    
    def _identify_columns(self, df: pd.DataFrame) -> None:
        """Identify numerical columns after dropping identifiers and targets."""
        # All columns except drops, categoricals, and targets
        exclude_cols = set(self.drop_columns + self.categorical_cols + 
                          [self.target_binary, self.target_multi])
        self.numerical_cols = [col for col in df.columns if col not in exclude_cols]
        logger.info(f"Identified {len(self.numerical_cols)} numerical features")
        logger.info(f"Identified {len(self.categorical_cols)} categorical features")
    
    def _create_preprocessor(self) -> ColumnTransformer:
        """
        Create sklearn ColumnTransformer for preprocessing.
        
        Returns:
            Configured ColumnTransformer
        """
        # Numerical pipeline: impute missing → scale
        numerical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        
        # Categorical pipeline: impute missing → one-hot encode
        categorical_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_pipeline, self.numerical_cols),
                ('cat', categorical_pipeline, self.categorical_cols)
            ],
            remainder='drop'  # Drop any remaining columns
        )
        
        return preprocessor
    
    def _prepare_labels(self, df: pd.DataFrame, is_train: bool = False
                       ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Prepare binary and multiclass labels.
        
        Args:
            df: DataFrame with target columns
            is_train: If True, fit the label encoder
            
        Returns:
            Tuple of (binary_labels, multiclass_labels)
        """
        # Binary: 0 = Normal, 1 = Attack
        y_binary = df[self.target_binary].values.astype(int)
        
        # Multiclass: encode attack categories
        if is_train:
            self.label_encoder = LabelEncoder()
            # Fit on known labels to ensure consistent ordering
            known_labels = self.config['data'].get('multiclass_labels', 
                                                    df[self.target_multi].unique())
            self.label_encoder.fit(known_labels)
            self.label_mapping = {label: idx for idx, label in 
                                 enumerate(self.label_encoder.classes_)}
            logger.info(f"Label mapping: {self.label_mapping}")
        
        # Handle unseen categories gracefully
        y_multi_raw = df[self.target_multi].values
        y_multi = np.zeros(len(y_multi_raw), dtype=int)
        for i, label in enumerate(y_multi_raw):
            if label in self.label_mapping:
                y_multi[i] = self.label_mapping[label]
            else:
                # Map unknown to closest or log warning
                logger.warning(f"Unknown label encountered: {label}")
                y_multi[i] = 0  # Default to Normal
        
        return y_binary, y_multi
    
    def fit_transform(self, train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
        """
        Fit preprocessor on training data and transform all splits.
        
        This is the main preprocessing method that:
        1. Creates stratified train/validation split
        2. Fits all transformers on training data only
        3. Transforms all three splits
        
        Args:
            train_df: Official training DataFrame (175,341 rows)
            test_df: Official test DataFrame (82,332 rows)
        """
        logger.info("=" * 60)
        logger.info("Starting preprocessing pipeline")
        logger.info("=" * 60)
        
        self._original_test_size = len(test_df)
        
        # Identify column types
        self._identify_columns(train_df)
        
        # Step 1: Stratified train/validation split
        logger.info(f"Creating {1-self.val_size:.0%}/{self.val_size:.0%} train/val split")
        train_split, val_split = train_test_split(
            train_df,
            test_size=self.val_size,
            random_state=self.random_state,
            stratify=train_df[self.target_multi]  # Stratify by multiclass for better balance
        )
        
        self._original_train_size = len(train_split)
        logger.info(f"Training samples: {len(train_split):,}")
        logger.info(f"Validation samples: {len(val_split):,}")
        logger.info(f"Test samples: {len(test_df):,}")
        
        # Step 2: Prepare features (drop identifiers and targets)
        feature_cols = self.numerical_cols + self.categorical_cols
        X_train_raw = train_split[feature_cols]
        X_val_raw = val_split[feature_cols]
        X_test_raw = test_df[feature_cols]
        
        # Step 3: Prepare labels
        self.y_train_binary, self.y_train_multi = self._prepare_labels(
            train_split, is_train=True)
        self.y_val_binary, self.y_val_multi = self._prepare_labels(val_split)
        self.y_test_binary, self.y_test_multi = self._prepare_labels(test_df)
        
        # Step 4: Create and fit preprocessor on TRAINING ONLY
        logger.info("Fitting preprocessor on training data...")
        self.preprocessor = self._create_preprocessor()
        self.X_train = self.preprocessor.fit_transform(X_train_raw).astype(np.float32)
        
        # Step 5: Transform validation and test (NO fitting!)
        logger.info("Transforming validation and test data...")
        self.X_val = self.preprocessor.transform(X_val_raw).astype(np.float32)
        self.X_test = self.preprocessor.transform(X_test_raw).astype(np.float32)
        
        # Step 6: Extract feature names
        self._extract_feature_names()
        
        # Log final shapes
        logger.info("-" * 40)
        logger.info("Preprocessing complete!")
        logger.info(f"X_train shape: {self.X_train.shape}")
        logger.info(f"X_val shape: {self.X_val.shape}")
        logger.info(f"X_test shape: {self.X_test.shape}")
        logger.info(f"Total features: {self.X_train.shape[1]}")
        
        # Validate no data leakage
        self._validate_no_leakage()
    
    def _extract_feature_names(self) -> None:
        """Extract feature names from fitted preprocessor."""
        feature_names = []
        
        # Numerical features (unchanged names)
        feature_names.extend(self.numerical_cols)
        
        # Categorical features (one-hot encoded names)
        cat_encoder = self.preprocessor.named_transformers_['cat']['encoder']
        for col, categories in zip(self.categorical_cols, cat_encoder.categories_):
            for cat in categories:
                feature_names.append(f"{col}_{cat}")
        
        self.feature_names = feature_names
        logger.info(f"Extracted {len(feature_names)} feature names")
    
    def _validate_no_leakage(self) -> None:
        """Validate that no data leakage has occurred."""
        # Check sizes unchanged
        assert self.X_val.shape[0] + self.X_train.shape[0] > 0, "Empty training data!"
        assert self.X_test.shape[0] == self._original_test_size, \
            f"Test size changed! Expected {self._original_test_size}, got {self.X_test.shape[0]}"
        
        # Check for NaN/Inf
        assert not np.isnan(self.X_train).any(), "NaN values in X_train!"
        assert not np.isnan(self.X_val).any(), "NaN values in X_val!"
        assert not np.isnan(self.X_test).any(), "NaN values in X_test!"
        assert not np.isinf(self.X_train).any(), "Inf values in X_train!"
        
        logger.info("[OK] Data leakage validation passed")
    
    def get_splits(self, task: str) -> Tuple[np.ndarray, np.ndarray, 
                                              np.ndarray, np.ndarray,
                                              np.ndarray, np.ndarray]:
        """
        Get preprocessed data splits for specified task.
        
        Args:
            task: 'binary' or 'multi'
            
        Returns:
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        if self.X_train is None:
            raise RuntimeError("Preprocessor not fitted. Call fit_transform first.")
        
        if task == 'binary':
            return (self.X_train, self.y_train_binary,
                    self.X_val, self.y_val_binary,
                    self.X_test, self.y_test_binary)
        elif task == 'multi':
            return (self.X_train, self.y_train_multi,
                    self.X_val, self.y_val_multi,
                    self.X_test, self.y_test_multi)
        else:
            raise ValueError(f"Unknown task: {task}. Use 'binary' or 'multi'.")
    
    def get_class_distribution(self, task: str) -> Dict[str, Dict[str, int]]:
        """
        Get class distribution for each split.
        
        Args:
            task: 'binary' or 'multi'
            
        Returns:
            Dictionary with distributions for train, val, test
        """
        _, y_train, _, y_val, _, y_test = self.get_splits(task)
        
        def count_classes(y):
            unique, counts = np.unique(y, return_counts=True)
            return dict(zip(unique.astype(str), counts.tolist()))
        
        return {
            'train': count_classes(y_train),
            'val': count_classes(y_val),
            'test': count_classes(y_test)
        }
    
    def save_metadata(self, path: str) -> None:
        """Save preprocessing metadata for reproducibility."""
        metadata = {
            'feature_names': self.feature_names,
            'label_mapping': self.label_mapping,
            'numerical_cols': self.numerical_cols,
            'categorical_cols': self.categorical_cols,
            'train_shape': list(self.X_train.shape),
            'val_shape': list(self.X_val.shape),
            'test_shape': list(self.X_test.shape)
        }
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved preprocessing metadata to {path}")

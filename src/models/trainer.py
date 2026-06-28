"""
Model training framework for UNSW-NB15 experiments.

This module provides a unified interface for training all three model
families (LR, RF, XGB) with consistent handling of class weights and
sample weights across different imbalance strategies.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.base import BaseEstimator
from typing import Optional, Dict, Any
import logging
import time
import warnings
from sklearn.exceptions import ConvergenceWarning

from .config import MODEL_CONFIGS

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Unified training interface for all classification models.
    
    Handles the complexity of applying class weights and sample weights
    appropriately for each model type and strategy combination.
    
    Attributes:
        config: Configuration dictionary
        random_state: Random seed for reproducibility
    """
    
    # Model class mapping
    MODELS = {
        'lr': LogisticRegression,
        'rf': RandomForestClassifier,
        'xgb': XGBClassifier
    }
    
    def __init__(self, config: dict):
        """
        Initialize ModelTrainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.random_state = config.get('random_state', 42)
    
    def get_model(self,
                  model_name: str,
                  task: str,
                  class_weight: Optional[str] = None,
                  scale_pos_weight: Optional[float] = None,
                  num_classes: int = 2) -> BaseEstimator:
        """
        Create and configure a model instance.
        
        Args:
            model_name: 'lr', 'rf', or 'xgb'
            task: 'binary' or 'multi'
            class_weight: 'balanced' or None (for LR/RF)
            scale_pos_weight: For XGBoost binary with S1
            num_classes: Number of classes for multiclass XGBoost
            
        Returns:
            Configured but unfitted model instance
        """
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from {list(self.MODELS.keys())}")
        
        # Get base configuration
        model_config = MODEL_CONFIGS[model_name][task].copy()
        model_config['random_state'] = self.random_state
        
        # Apply class weight for LR and RF
        if model_name in ['lr', 'rf'] and class_weight is not None:
            model_config['class_weight'] = class_weight
        
        # Apply scale_pos_weight for XGBoost binary
        if model_name == 'xgb' and task == 'binary' and scale_pos_weight is not None:
            model_config['scale_pos_weight'] = scale_pos_weight
        
        # Set num_class for XGBoost multiclass
        if model_name == 'xgb' and task == 'multi':
            model_config['num_class'] = num_classes
        
        # Create model instance
        model_class = self.MODELS[model_name]
        model = model_class(**model_config)
        
        logger.debug(f"Created {model_name.upper()} for {task} task")
        return model
    
    def train(self,
              model: BaseEstimator,
              X_train: np.ndarray,
              y_train: np.ndarray,
              sample_weight: Optional[np.ndarray] = None,
              X_val: Optional[np.ndarray] = None,
              y_val: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Train a model and return training metadata with learning curves.
        
        Dispatches to specialized training methods based on model type:
        - XGBoost: Uses native early stopping and evals_result
        - Sklearn (LR/RF): Uses custom iterative warm_start loop
        """
        logger.info(f"Training {type(model).__name__} on {len(y_train):,} samples...")
        start_time = time.time()
        
        metadata = {}
        
        # Dispatch to specific handler
        if isinstance(model, XGBClassifier):
            metadata = self._train_xgboost(
                model, X_train, y_train, sample_weight, X_val, y_val
            )
        else:
            # Check if model supports warm_start for iterative curves
            if hasattr(model, 'warm_start') and model.warm_start:
                metadata = self._train_sklearn_iterative(
                    model, X_train, y_train, sample_weight, X_val, y_val
                )
            else:
                # Fallback for standard training
                self._fit_standard(model, X_train, y_train, sample_weight)
        
        training_time = time.time() - start_time
        logger.info(f"Training completed in {training_time:.2f}s")
        
        # Enforce metadata structure
        metadata.update({
            'training_time_seconds': training_time,
            'n_samples': len(y_train),
            'n_features': X_train.shape[1]
        })
        
        return metadata

    def _train_xgboost(self, model, X_train, y_train, sample_weight, X_val, y_val) -> Dict[str, Any]:
        """Handle XGBoost training with native evaluation monitoring."""
        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight
            
        if X_val is not None and y_val is not None:
             fit_params['eval_set'] = [(X_train, y_train), (X_val, y_val)]
             fit_params['verbose'] = False
             logger.info("Enabled XGBoost native tracking")

        logger.info(f"Fitting XGBoost with params: {list(fit_params.keys())}")
        model.fit(X_train, y_train, **fit_params)
        
        metadata = {}
        try:
            evals_result = model.evals_result()
            if evals_result:
                metadata['learning_curve'] = evals_result
        except Exception as e:
            logger.warning(f"Could not retrieve XGBoost learning curve: {e}")
            
        return metadata

    def _train_sklearn_iterative(self, model, X_train, y_train, sample_weight, X_val, y_val) -> Dict[str, Any]:
        """
        Train sklearn models iteratively using warm_start to generate learning curves.
        
        Uses a step size of 10 to minimize overhead (proven 1.3x vs 9x for step=1).
        """
        # Determine iteration parameter and target
        if isinstance(model, RandomForestClassifier):
            param_name = 'n_estimators'
            target_value = model.n_estimators
            model.n_estimators = 0 # Start from 0
        elif isinstance(model, LogisticRegression):
            param_name = 'max_iter'
            target_value = model.max_iter
            model.max_iter = 0
        else:
            # Should not happen given dispatch logic but safe fallback
            self._fit_standard(model, X_train, y_train, sample_weight)
            return {}

        step_size = 10
        current_step = 0
        
        # Storage for metrics: {'validation_0': {'logloss': []}, ...}
        # mimicking XGBoost structure for compatibility with plotting tools
        history = {
            'validation_0': {'score': []}, # Train
            'validation_1': {'score': []}  # Val
        }
        
        logger.info(f"Starting iterative training ({param_name}) -> {target_value}")
        
        while current_step < target_value:
            # Increment
            next_step = min(current_step + step_size, target_value)
            setattr(model, param_name, next_step)
            
            # Fit (warm_start=True ensures it builds on previous)
            # Log Reg requires unique classes check on first fit usually, but standard fit handles it
            with warnings.catch_warnings():
                # Filter benign warning about class_weight + warm_start (we use full dataset, so weights are stable)
                warnings.filterwarnings("ignore", ".*class_weight presets.*", category=UserWarning)
                # Filter convergence warnings as we are incrementally training
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
                
                if sample_weight is not None:
                    model.fit(X_train, y_train, sample_weight=sample_weight)
                else:
                    model.fit(X_train, y_train)
                
            # Log Metrics
            # Using 'score' (accuracy) as generic metric for now
            history['validation_0']['score'].append(model.score(X_train, y_train))
            if X_val is not None:
                history['validation_1']['score'].append(model.score(X_val, y_val))
            
            current_step = next_step
            
        return {'learning_curve': history}

    def _fit_standard(self, model, X_train, y_train, sample_weight):
        """Standard single-shot fit."""
        if sample_weight is not None:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        else:
            model.fit(X_train, y_train)
    
    def predict(self, model: BaseEstimator, X: np.ndarray) -> np.ndarray:
        """
        Get class predictions from trained model.
        
        Args:
            model: Trained model
            X: Features to predict
            
        Returns:
            Array of predicted class labels
        """
        return model.predict(X)
    
    def predict_proba(self, model: BaseEstimator, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions from trained model.
        
        Args:
            model: Trained model  
            X: Features to predict
            
        Returns:
            Array of class probabilities (n_samples, n_classes)
        """
        return model.predict_proba(X)
    
    def get_model_name(self, model_name: str) -> str:
        """Get human-readable model name."""
        names = {
            'lr': 'Logistic Regression',
            'rf': 'Random Forest',
            'xgb': 'XGBoost'
        }
        return names.get(model_name, model_name)

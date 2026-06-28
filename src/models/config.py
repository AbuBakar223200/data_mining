"""
Model hyperparameter configurations.

This module contains the default hyperparameter configurations for
all three model families used in the experiments:
- Logistic Regression (LR)
- Random Forest (RF)
- XGBoost (XGB)

Configurations are provided for both binary and multiclass tasks.
"""

import subprocess
import logging

logger = logging.getLogger(__name__)

def get_xgb_device():
    """Detect if NVIDIA GPU is available for XGBoost."""
    try:
        subprocess.check_output('nvidia-smi')
        logger.info("NVIDIA GPU detected. Using 'gpu_hist' for XGBoost.")
        # Explicit confirmation for Colab user
        logger.info(f"✅ XGBoost using GPU: {subprocess.check_output('nvidia-smi --query-gpu=name --format=csv,noheader', shell=True).decode().strip()}")
        return 'gpu_hist'
    except Exception:
        return 'hist'

MODEL_CONFIGS = {
    'lr': {
        'binary': {
            'C': 1.0,
            'solver': 'lbfgs',          # Optimized from saga
            'max_iter': 1000,
            'penalty': 'l2',
            'warm_start': True,         # Required for iterative learning curves
            'verbose': 0
        },
        'multi': {
            'C': 1.0,
            'solver': 'lbfgs',          # Optimized from saga
            'max_iter': 1000,
            'penalty': 'l2',
            'warm_start': True,         # Required for iterative learning curves
            'verbose': 0
        }
    },
    'rf': {
        'binary': {
            'n_estimators': 300,        # Optimized (was 100)
            'max_depth': None,          # Optimized (was 25)
            'min_samples_split': 2,     # Optimized (was 5)
            'min_samples_leaf': 1,      # Optimized (was 2)
            'max_features': 'sqrt',
            'criterion': 'gini',        # Explicit
            'class_weight': 'balanced_subsample', # Optimized
            'bootstrap': True,
            'oob_score': True,          # Added validation
            'warm_start': True,         # Required for iterative learning curves
            'n_jobs': -1,
            'verbose': 0
        },
        'multi': {
            'n_estimators': 300,        # Optimized (was 100)
            'max_depth': None,          # Optimized (was 25)
            'min_samples_split': 2,     # Optimized (was 5)
            'min_samples_leaf': 1,      # Optimized (was 2)
            'max_features': 'sqrt',
            'criterion': 'gini',        # Explicit
            'class_weight': 'balanced_subsample', # Optimized
            'bootstrap': True,
            'oob_score': True,          # Added validation
            'warm_start': True,         # Required for iterative learning curves
            'n_jobs': -1,
            'verbose': 0
        }
    },
    'xgb': {
        'binary': {
            'n_estimators': 150,        # Optimized (was 100)
            'learning_rate': 0.05,      # Optimized (was 0.1)
            'max_depth': 15,            # Optimized (was 10)
            'min_child_weight': 2,      # Optimized (was 1)
            'subsample': 0.85,          # Optimized (was 0.8)
            'colsample_bytree': 0.85,   # Optimized (was 0.8)
            'gamma': 1.0,               # New
            'reg_lambda': 1.0,          # New
            'reg_alpha': 0.5,           # New
            'use_label_encoder': False,
            'tree_method': get_xgb_device(),
            'eval_metric': ['logloss', 'error', 'auc'],
            'objective': 'binary:logistic',
            'n_jobs': -1,
            'verbosity': 0
        },
        'multi': {
            'n_estimators': 150,        # Optimized (was 100)
            'learning_rate': 0.05,      # Optimized (was 0.1)
            'max_depth': 15,            # Optimized (was 10)
            'min_child_weight': 2,      # Optimized (was 1)
            'subsample': 0.85,          # Optimized (was 0.8)
            'colsample_bytree': 0.85,   # Optimized (was 0.8)
            'gamma': 1.0,               # New
            'reg_lambda': 1.0,          # New
            'reg_alpha': 0.5,           # New
            'use_label_encoder': False,
            'tree_method': get_xgb_device(),
            'eval_metric': ['mlogloss', 'merror', 'auc'], # Multi-metric tracking
            'objective': 'multi:softprob',
            'n_jobs': -1,
            'verbosity': 0
        }
    }
}



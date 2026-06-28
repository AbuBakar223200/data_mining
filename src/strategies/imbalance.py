"""
Imbalance handling strategies for UNSW-NB15 experiments.

This module implements the three core strategies for handling class imbalance:
- S0: No balancing (baseline)
- S1: Class weighting (cost-sensitive learning)
- S2a: Random oversampling
- S2b: SMOTE (optional, with fallback)

Each strategy is applied ONLY to training data to prevent data leakage.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple, Optional, Dict
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.utils.class_weight import compute_class_weight
import logging

logger = logging.getLogger(__name__)


class ImbalanceStrategy(ABC):
    """
    Abstract base class for imbalance handling strategies.
    
    All strategies must implement the apply() method which takes
    training data and returns (potentially modified) training data.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Strategy identifier."""
        pass
    
    @abstractmethod
    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply the imbalance strategy to training data.
        
        Args:
            X: Training features
            y: Training labels
            
        Returns:
            Tuple of (X_modified, y_modified)
        """
        pass
    
    def get_class_weight(self) -> Optional[str]:
        """
        Get class_weight parameter for sklearn models.
        
        Returns:
            None for most strategies, 'balanced' for S1
        """
        return None
    
    def get_sample_weight(self, y: np.ndarray) -> Optional[np.ndarray]:
        """
        Compute sample weights for models without class_weight support.
        
        Args:
            y: Training labels
            
        Returns:
            Sample weight array or None
        """
        return None
    
    def get_scale_pos_weight(self, y: np.ndarray) -> Optional[float]:
        """
        Compute scale_pos_weight for XGBoost binary classification.
        
        Args:
            y: Binary training labels
            
        Returns:
            Ratio of negative to positive samples or None
        """
        return None


class S0_NoBalancing(ImbalanceStrategy):
    """
    Baseline strategy: No modification to training data.
    
    This serves as the baseline to demonstrate how models fail
    on minority classes without any intervention.
    
    Expected results:
    - High overall accuracy
    - Near-zero recall for rare classes (Worms, Shellcode)
    """
    
    @property
    def name(self) -> str:
        return "s0"
    
    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return data unchanged."""
        logger.info(f"S0: No balancing applied. Training size: {len(y):,}")
        return X, y


class S1_ClassWeight(ImbalanceStrategy):
    """
    Class weighting strategy: Apply inverse frequency weights during training.
    
    This penalizes misclassification of minority classes more heavily
    without modifying the actual training data.
    
    For sklearn models: Use class_weight='balanced'
    For XGBoost binary: Use scale_pos_weight
    For XGBoost multiclass: Use sample_weight
    """
    
    @property
    def name(self) -> str:
        return "s1"
    
    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return data unchanged (weights applied at training time)."""
        logger.info(f"S1: Class weighting enabled. Training size: {len(y):,}")
        return X, y
    
    def get_class_weight(self) -> str:
        """Return 'balanced' for sklearn models."""
        return 'balanced'
    
    def get_sample_weight(self, y: np.ndarray) -> np.ndarray:
        """
        Compute sample weights for models without class_weight parameter.
        
        This is particularly useful for XGBoost multiclass where
        class_weight is not directly supported.
        """
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        weight_dict = dict(zip(classes, weights))
        sample_weights = np.array([weight_dict[label] for label in y])
        
        logger.info(f"S1: Computed sample weights for {len(classes)} classes")
        logger.debug(f"Weight range: [{sample_weights.min():.2f}, {sample_weights.max():.2f}]")
        
        return sample_weights
    
    def get_scale_pos_weight(self, y: np.ndarray) -> float:
        """
        Compute scale_pos_weight for XGBoost binary classification.
        
        Formula: count(negative) / count(positive)
        """
        n_negative = np.sum(y == 0)
        n_positive = np.sum(y == 1)
        
        if n_positive == 0:
            logger.warning("No positive samples found!")
            return 1.0
        
        ratio = n_negative / n_positive
        logger.info(f"S1: XGBoost scale_pos_weight = {ratio:.2f}")
        return ratio


class S2a_RandomOverSampler(ImbalanceStrategy):
    """
    Random oversampling strategy: Duplicate minority class samples.
    
    This creates exact copies of minority class samples until all
    classes have equal representation.
    
    Post-resampling sizes:
    - Binary: ~224,000 (doubles attack class)
    - Multiclass: ~560,000 (all classes equal to majority)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize RandomOverSampler.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.sampler = RandomOverSampler(
            sampling_strategy='auto',  # Resample all minority to majority
            random_state=random_state
        )
    
    @property
    def name(self) -> str:
        return "s2a"
    
    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply random oversampling to training data.
        
        Returns:
            Tuple of (X_resampled, y_resampled) with equal class sizes
        """
        original_size = len(y)
        X_resampled, y_resampled = self.sampler.fit_resample(X, y)
        new_size = len(y_resampled)
        
        logger.info(f"S2a: RandomOverSampler applied")
        logger.info(f"     Original size: {original_size:,} -> New size: {new_size:,}")
        logger.info(f"     Expansion ratio: {new_size/original_size:.2f}x")
        
        return X_resampled, y_resampled


class S2b_SMOTE(ImbalanceStrategy):
    """
    SMOTE strategy: Generate synthetic minority samples.
    
    Creates new samples by interpolating between minority class
    neighbors in feature space.
    
    Note: Falls back to RandomOverSampler if SMOTE fails
    (e.g., when a class has fewer samples than k_neighbors).
    """
    
    def __init__(self, k_neighbors: int = 5, random_state: int = 42):
        """
        Initialize SMOTE sampler.
        
        Args:
            k_neighbors: Number of nearest neighbors for interpolation
            random_state: Random seed for reproducibility
        """
        self.k_neighbors = k_neighbors
        self.random_state = random_state
    
    @property
    def name(self) -> str:
        return "s2b"
    
    def apply(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply SMOTE with fallback to RandomOverSampler.
        
        Returns:
            Tuple of (X_resampled, y_resampled)
        """
        original_size = len(y)
        
        try:
            smote = SMOTE(
                k_neighbors=self.k_neighbors,
                random_state=self.random_state
            )
            X_resampled, y_resampled = smote.fit_resample(X, y)
            method = "SMOTE"
            
        except ValueError as e:
            # Fallback for classes with insufficient samples
            logger.warning(f"SMOTE failed: {e}")
            logger.warning("Falling back to RandomOverSampler")
            
            ros = RandomOverSampler(random_state=self.random_state)
            X_resampled, y_resampled = ros.fit_resample(X, y)
            method = "RandomOverSampler (fallback)"
        
        new_size = len(y_resampled)
        logger.info(f"S2b: {method} applied")
        logger.info(f"     Original size: {original_size:,} -> New size: {new_size:,}")
        
        return X_resampled, y_resampled


def get_strategy(name: str, random_state: int = 42) -> ImbalanceStrategy:
    """
    Factory function to get strategy by name.
    
    Args:
        name: Strategy identifier ('s0', 's1', 's2a', 's2b')
        random_state: Random seed for reproducibility
        
    Returns:
        Configured ImbalanceStrategy instance
        
    Raises:
        ValueError: If strategy name is unknown
    """
    strategies = {
        's0': S0_NoBalancing,
        's1': S1_ClassWeight,
        's2a': lambda: S2a_RandomOverSampler(random_state=random_state),
        's2b': lambda: S2b_SMOTE(random_state=random_state)
    }
    
    if name not in strategies:
        raise ValueError(f"Unknown strategy: {name}. Choose from {list(strategies.keys())}")
    
    strategy_class = strategies[name]
    return strategy_class() if callable(strategy_class) else strategy_class

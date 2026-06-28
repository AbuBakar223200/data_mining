
import pytest
import numpy as np
from src.strategies import get_strategy
from src.strategies.imbalance import S0_NoBalancing, S1_ClassWeight, S2a_RandomOverSampler, S2b_SMOTE

def test_strategy_factory():
    """Test that factory returns correct strategy instances."""
    assert isinstance(get_strategy('s0'), S0_NoBalancing)
    assert isinstance(get_strategy('s1'), S1_ClassWeight)
    assert isinstance(get_strategy('s2a', random_state=42), S2a_RandomOverSampler)
    assert isinstance(get_strategy('s2b', random_state=42), S2b_SMOTE)
    
    with pytest.raises(ValueError):
        get_strategy('invalid_strategy')

def test_s0_no_balancing(synthetic_data):
    """Test S0 strategy leaves data unchanged."""
    strategy = get_strategy('s0')
    X = synthetic_data.drop('label', axis=1).values
    y = synthetic_data['label'].values
    
    X_new, y_new = strategy.apply(X, y)
    
    np.testing.assert_array_equal(X, X_new)
    np.testing.assert_array_equal(y, y_new)

def test_s1_class_weight_logic():
    """Test S1 class weight calculations."""
    strategy = get_strategy('s1')
    
    # Create imbalanced y: 90 zeros, 10 ones
    y = np.array([0]*90 + [1]*10)
    X = np.zeros((100, 5)) # Dummy X
    
    # Apply should not change data
    X_new, y_new = strategy.apply(X, y)
    np.testing.assert_array_equal(X, X_new)
    
    # Check sklearn class_weight
    assert strategy.get_class_weight() == 'balanced'
    
    # Check sample weights
    # Minority class (1) should have higher weight than Majority (0)
    sample_weights = strategy.get_sample_weight(y)
    
    # Get weight for a class 0 sample and class 1 sample
    w0 = sample_weights[0] # First sample is 0
    w1 = sample_weights[99] # Last sample is 1
    
    assert w1 > w0, "Minority class should have higher weight"
    
    # Check XGBoost scale_pos_weight (neg/pos)
    # 90/10 = 9.0
    scale_pos = strategy.get_scale_pos_weight(y)
    assert scale_pos == 9.0

def test_s2a_random_oversampling():
    """Test S2a increases minority class samples."""
    strategy = get_strategy('s2a', random_state=42)
    
    # Create imbalanced data
    # 2 classes: 0 (Majority, 100 samples), 1 (Minority, 20 samples)
    n_maj = 100
    n_min = 20
    X = np.random.rand(n_maj + n_min, 5)
    y = np.array([0]*n_maj + [1]*n_min)
    
    X_res, y_res = strategy.apply(X, y)
    
    # After ROS, both should be equal to majority size (100)
    # Total = 200
    assert len(y_res) == n_maj * 2
    assert np.sum(y_res == 0) == n_maj
    assert np.sum(y_res == 1) == n_maj
    
    # Shapes should match
    assert X_res.shape[0] == len(y_res)

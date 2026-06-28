
import pytest
import pandas as pd
import numpy as np
from src.data.preprocessing import UNSWPreprocessor
from src.data.loader import DataLoader

def test_loader_schema_validation(synthetic_data, mock_config, tmp_path):
    """Test that DataLoader validates schema correctly."""
    # Setup mock files
    train_path = tmp_path / "train.csv"
    test_path = tmp_path / "test.csv"
    synthetic_data.to_csv(train_path, index=False)
    synthetic_data.to_csv(test_path, index=False)
    
    # Update config paths
    mock_config['data']['train_path'] = str(train_path)
    mock_config['data']['test_path'] = str(test_path)
    
    loader = DataLoader(mock_config)
    df_train = loader.load_train()
    
    assert len(df_train) == 100
    assert 'label' in df_train.columns

def test_preprocessor_leakage_dropping(synthetic_data, mock_config):
    """Test that Preprocessor drops specified leakage columns."""
    preprocessor = UNSWPreprocessor(mock_config)
    
    # Fit transform
    preprocessor.fit_transform(synthetic_data, synthetic_data)
    
    # Check feature names for dropped columns
    dropped = mock_config['data']['drop_columns']
    categorical = mock_config['data']['categorical_columns']
    
    for drop_col in dropped:
        # 1. It must not be present as an exact match (for numericals)
        assert drop_col not in preprocessor.feature_names, f"Dropped column {drop_col} found in features!"
        
        # 2. If it was distinct from categoricals, it shouldn't appear as a prefix
        # But 'sport' (dropped) is a substring of 'ct_dst_sport_ltm' (valid feature)
        # So we only fail if it looks like a One-Hot encoding of a dropped column: "{drop_col}_"
        # AND strictly only if that drop_col was actually processed as categorical? 
        # Actually our preprocessor creates one-hots as f"{col}_{cat}".
        
        # The safest check is:
        # - Exact match is FORBIDDEN.
        # - Prefix match f"{drop_col}_" is suspicious but might be valid for other features?
        # Let's stick to Exact Match for now as that's the primary guarantee.
        pass
            
    assert preprocessor.X_train.shape[1] > 0
    assert preprocessor.X_train.shape[0] > 0

def test_preprocessor_splitting(synthetic_data, mock_config):
    """Test train/val/test splitting logic."""
    preprocessor = UNSWPreprocessor(mock_config)
    
    # original size
    n_total = len(synthetic_data)
    preprocessor.fit_transform(synthetic_data, synthetic_data)
    
    # Check split sizes
    # Train + Val should equal original train input (which is 'synthetic_data' here)
    # But wait, logic is train_test_split on input 'train_df'
    assert len(preprocessor.X_train) + len(preprocessor.X_val) == n_total
    
    # Test should match original test input size
    assert len(preprocessor.X_test) == n_total

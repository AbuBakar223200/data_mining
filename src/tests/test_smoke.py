
import pytest
import numpy as np
from pathlib import Path
from main import run_single_experiment
from src.data.preprocessing import UNSWPreprocessor

def test_smoke_end_to_end_binary(synthetic_data, mock_config, tmp_path):
    """Test full execution of a single binary experiment."""
    # Setup data
    preprocessor = UNSWPreprocessor(mock_config)
    preprocessor.fit_transform(synthetic_data, synthetic_data)
    
    results_dir = tmp_path / "results/smoke_test/"
    
    # Run Experiment
    result = run_single_experiment(
        experiment_id="test_smoke_binary",
        task='binary',
        model_name='rf', # Fast model
        strategy_name='s0',
        X_train=preprocessor.X_train,
        y_train=preprocessor.y_train_binary,
        X_test=preprocessor.X_test,
        y_test=preprocessor.y_test_binary,
        X_val=preprocessor.X_val,
        y_val=preprocessor.y_val_binary,
        config=mock_config,
        results_dir=results_dir,
        class_names=['Normal', 'Attack'],
        seed=42
    )
    
    # Assertions
    assert isinstance(result, dict)
    assert 'metrics' in result
    assert result['training_time_seconds'] > 0
    
    # Check artifacts
    assert (results_dir / "metrics" / "test_smoke_binary.json").exists()
    # cm figure creation might succeed or fail depending on matplotlib backend in test env, 
    # but the function attempts it. We check if the file was created.
    assert (results_dir / "figures" / "test_smoke_binary" / "confusion_matrix.png").exists()

def test_smoke_end_to_end_multiclass(synthetic_data, mock_config, tmp_path):
    """Test full execution of a single multiclass experiment."""
    # Setup data
    preprocessor = UNSWPreprocessor(mock_config)
    preprocessor.fit_transform(synthetic_data, synthetic_data)
    
    results_dir = tmp_path / "results"
    class_names = [str(c) for c in np.unique(preprocessor.y_train_multi)]
    
    # Run Experiment
    result = run_single_experiment(
        experiment_id="test_smoke_multi",
        task='multi',
        model_name='rf',
        strategy_name='s0',
        X_train=preprocessor.X_train,
        y_train=preprocessor.y_train_multi,
        X_test=preprocessor.X_test,
        y_test=preprocessor.y_test_multi,
        X_val=preprocessor.X_val,
        y_val=preprocessor.y_val_multi,
        config=mock_config,
        results_dir=results_dir,
        class_names=class_names,
        seed=42
    )
    
    # Assertions
    assert 'metrics' in result
    assert 'overall' in result['metrics']
    assert 'macro_f1' in result['metrics']['overall']

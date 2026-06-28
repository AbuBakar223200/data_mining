#!/usr/bin/env python3
"""
Run Baseline Classifiers (ZeroR, Stratified)
============================================

This script generates baseline metrics for comparison:
1. ZeroR (Most Frequent): Predicts majority class.
2. Stratified: Predicts random class respecting training distribution.

Usage:
    python -m scripts.run_baselines
"""

import sys
import os
import json
import time
import logging
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

# Add project root to path
sys.path.append(os.getcwd())

from src.data import DataLoader, UNSWPreprocessor
from src.evaluation import compute_all_metrics, format_metrics_for_logging
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

def run_baseline(task, strategy, X_train, y_train, X_test, y_test, results_dir, class_names):
    """Run a single baseline experiment."""
    experiment_id = f"{task}_baseline_{strategy}"
    logger.info(f"Running Baseline: {experiment_id}")
    
    start_time = time.time()
    
    # Configure DummyClassifier
    if strategy == 'zeror':
        model = DummyClassifier(strategy='most_frequent', random_state=42)
    elif strategy == 'stratified':
        model = DummyClassifier(strategy='stratified', random_state=42)
    else:
        raise ValueError(f"Unknown baseline strategy: {strategy}")
        
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Compute Metrics
    metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        task=task,
        class_names=class_names
    )
    
    total_time = time.time() - start_time
    
    logger.info(f"Results ({experiment_id}):\n" + format_metrics_for_logging(metrics, task))
    
    # Compile results
    result = {
        'experiment_id': experiment_id,
        'task': task,
        'model': 'baseline',
        'strategy': strategy,
        'timestamp': datetime.now().isoformat(),
        'training_time_seconds': total_time, # Negligible
        'total_time_seconds': total_time,
        'metrics': metrics
    }
    
    # Save JSON
    metrics_path = results_dir / 'metrics' / f"{experiment_id}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        json.dump(result, f, indent=2)
        
    return result

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run Baselines")
    parser.add_argument("--config", type=str, default="configs/main.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    results_dir = Path(config['results_dir'])
    setup_logging(level="INFO", log_file=str(results_dir / 'logs' / 'baselines.log'))
    
    # Load Data
    loader = DataLoader(config)
    train_df, test_df = loader.load_all()
    
    preprocessor = UNSWPreprocessor(config)
    preprocessor.fit_transform(train_df, test_df)
    
    results = []
    
    for task in ['binary', 'multi']:
        logger.info(f"\nProcessing Task: {task}")
        X_train, y_train, _, _, X_test, y_test = preprocessor.get_splits(task)
        
        if task == 'multi':
            class_names = list(preprocessor.label_encoder.classes_)
        else:
            class_names = ['Normal', 'Attack']
            
        # Run ZeroR
        results.append(run_baseline(task, 'zeror', X_train, y_train, X_test, y_test, results_dir, class_names))
        
        # Run Stratified
        results.append(run_baseline(task, 'stratified', X_train, y_train, X_test, y_test, results_dir, class_names))
        
    # Append to experiment_log.csv
    log_path = results_dir / 'experiment_log.csv'
    
    new_rows = []
    for r in results:
        new_rows.append({
            'experiment_id': r['experiment_id'],
            'task': r['task'],
            'model': r['model'] + '_' + r['strategy'],
            'strategy': 's0', # Treated as baseline logic
            'accuracy': r['metrics']['overall']['accuracy'],
            'macro_f1': r['metrics']['overall']['macro_f1'],
            'weighted_f1': r['metrics']['overall']['weighted_f1'],
            'g_mean': r['metrics']['overall']['g_mean'],
            'roc_auc': r['metrics']['overall']['roc_auc'],
            'training_time': r['training_time_seconds'],
            'total_time': r['total_time_seconds'],
            'timestamp': r['timestamp']
        })
        
    df_new = pd.DataFrame(new_rows)
    
    if log_path.exists():
        df_old = pd.read_csv(log_path)
        # Remove existing baseline entries if any to prevent duplicates
        df_old = df_old[~df_old['experiment_id'].str.contains('baseline')]
        df_final = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df_final = df_new
        
    df_final.to_csv(log_path, index=False)
    logger.info(f"Updated {log_path} with baseline results.")

if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Main Orchestrator for UNSW-NB15 Imbalance Study
================================================

This script runs the complete 18-experiment grid:
- 2 Tasks: Binary, Multiclass
- 3 Models: Logistic Regression, Random Forest, XGBoost
- 3 Strategies: S0 (None), S1 (Class Weight), S2a (RandomOverSampler)

Usage:
    python main.py

Output:
    - results/metrics/{experiment_id}.json - Per-experiment metrics
    - results/figures/cm_{experiment_id}.png - Confusion matrices
    - results/experiment_log.csv - Master experiment tracker
"""

import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

# Local imports
from src.data import DataLoader, UNSWPreprocessor
from src.models import ModelTrainer
from src.strategies import get_strategy
from src.evaluation import (
    compute_all_metrics,
    compute_rare_class_analysis,
    format_metrics_for_logging,

    plot_confusion_matrix,
    plot_learning_curves,
    plot_roc_curve,
    plot_pr_curve,
    plot_feature_importance
)
from src.utils import load_config, setup_logging

# Configure logging
logger = logging.getLogger(__name__)


def run_single_experiment(
    experiment_id: str,
    task: str,
    model_name: str,
    strategy_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: dict,
    results_dir: Path,
    class_names: List[str],
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Execute a single experiment and save results.
    
    Args:
        experiment_id: Unique experiment identifier
        task: 'binary' or 'multi'
        model_name: 'lr', 'rf', or 'xgb'
        strategy_name: 's0', 's1', or 's2a'
        X_train, y_train: Training data
        X_test, y_test: Test data
        config: Configuration dictionary
        results_dir: Output directory path
        class_names: List of class names for reporting
        X_val, y_val: Validation data (optional)
        seed: Random seed for reproducibility
        
    Returns:
        Experiment results dictionary
    """
    start_time = time.time()
    
    # Update seed in config for this run
    run_config = config.copy()
    run_config['random_state'] = seed
    
    logger.info("=" * 60)
    logger.info(f"EXPERIMENT: {experiment_id} (Seed: {seed})")
    logger.info(f"  Task:     {task}")
    logger.info(f"  Model:    {model_name.upper()}")
    logger.info(f"  Strategy: {strategy_name.upper()}")
    logger.info("=" * 60)
    
    # Step 1: Apply imbalance strategy
    strategy = get_strategy(strategy_name, random_state=seed)
    X_train_balanced, y_train_balanced = strategy.apply(X_train, y_train)
    
    # Step 2: Configure model with appropriate weights
    trainer = ModelTrainer(run_config)
    
    # Get class weight and sample weight based on strategy
    class_weight = strategy.get_class_weight()
    sample_weight = None
    scale_pos_weight = None
    
    if strategy_name == 's1':
        if model_name == 'xgb' and task == 'binary':
            scale_pos_weight = strategy.get_scale_pos_weight(y_train)
            class_weight = None  # Use scale_pos_weight instead
        elif model_name == 'xgb' and task == 'multi':
            # XGBoost multiclass uses sample_weight
            sample_weight = strategy.get_sample_weight(y_train_balanced)
            class_weight = None
    
    # Create model instance
    num_classes = len(np.unique(y_train))
    model = trainer.get_model(
        model_name=model_name,
        task=task,
        class_weight=class_weight,
        scale_pos_weight=scale_pos_weight,
        num_classes=num_classes
    )
    
    # Step 3: Train model
    # Pass validation data if available
    training_metadata = trainer.train(
        model=model,
        X_train=X_train_balanced,
        y_train=y_train_balanced,
        sample_weight=sample_weight,
        X_val=X_val,
        y_val=y_val
    )
    
    # Step 4: Predict on TEST set
    y_pred = trainer.predict(model, X_test)
    y_pred_proba = trainer.predict_proba(model, X_test)
    
    # Step 5: Compute metrics
    metrics = compute_all_metrics(
        y_true=y_test,
        y_pred=y_pred,
        y_pred_proba=y_pred_proba,
        task=task,
        class_names=class_names if task == 'multi' else ['Normal', 'Attack']
    )
    
    # Step 6: Rare class analysis for multiclass
    rare_class_analysis = None
    if task == 'multi':
        rare_classes = config.get('rare_classes', [])
        rare_class_analysis = compute_rare_class_analysis(
            metrics=metrics,
            rare_classes=rare_classes,
            label_mapping=None  # Use class names directly
        )
    
    # Log metrics
    logger.info(f"\nResults ({experiment_id}):")
    logger.info(format_metrics_for_logging(metrics, task))
    
    total_time = time.time() - start_time
    
    # Compile results
    result = {
        'experiment_id': experiment_id,
        'seed': seed,
        'task': task,
        'model': model_name,
        'strategy': strategy_name,
        'timestamp': datetime.now().isoformat(),
        'training_time_seconds': training_metadata['training_time_seconds'],
        'total_time_seconds': total_time,
        'train_samples': len(y_train_balanced),
        'test_samples': len(y_test),
        'metrics': metrics,
        'rare_class_analysis': rare_class_analysis
    }
    
    # Propagate learning curve if available
    if 'learning_curve' in training_metadata:
        result['learning_curve'] = training_metadata['learning_curve']
    
    # Save metrics JSON
    metrics_path = results_dir / 'metrics' / f"{experiment_id}.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    with open(metrics_path, 'w') as f:
        # Exclude learning curve from main metrics file to keep it light
        result_copy = result.copy()
        if 'learning_curve' in result_copy:
            del result_copy['learning_curve']
        json.dump(result_copy, f, indent=2)
        
    # Save learning curves if available
    if 'learning_curve' in result:
        lc_dir = results_dir / 'learning_curves'
        lc_dir.mkdir(parents=True, exist_ok=True)
        
        # 1. Save JSON (Raw)
        lc_json_path = lc_dir / f"{experiment_id}.json"
        with open(lc_json_path, 'w') as f:
            json.dump(result['learning_curve'], f, indent=2)
            
        # 2. Save CSV (Processed for Visualization)
        try:
            lc_data = result['learning_curve']
            # validation_0 = Train (first in eval_set), validation_1 = Val (second)
            # We map them to friendly names
            mapping = {'validation_0': 'train', 'validation_1': 'val'}
            
            # Initialize DataFrame with epochs
            # Get length from arbitrary metric list
            first_set = next(iter(lc_data.values()))
            first_metric = next(iter(first_set.values()))
            epochs = range(1, len(first_metric) + 1)
            
            df_lc = pd.DataFrame({'epoch': epochs})
            
            for set_key, metrics_dict in lc_data.items():
                prefix = mapping.get(set_key, set_key)
                for metric_name, values in metrics_dict.items():
                    col_name = f"{prefix}_{metric_name}"
                    df_lc[col_name] = values
            
            lc_csv_path = lc_dir / f"{experiment_id}.csv"
            df_lc.to_csv(lc_csv_path, index=False)
            logger.info(f"Saved learning curve CSV: {lc_csv_path.name}")
            
        except Exception as e:
            logger.warning(f"Failed to generate learning curve CSV for {experiment_id}: {e}")
            
        # 3. Plot Learning Curves
        try:
            # Create per-experiment figure folder
            exp_fig_dir = results_dir / 'figures' / experiment_id
            exp_fig_dir.mkdir(parents=True, exist_ok=True)

            lc_plot_path = exp_fig_dir / "learning_curve.png"
            plot_learning_curves(
                learning_curve_data=result['learning_curve'],
                save_path=str(lc_plot_path),
                title=f"Learning Curves: {experiment_id} (Seed {seed})"
            )
        except Exception as e:
            logger.warning(f"Failed to plot learning curves for {experiment_id}: {e}")
    
    # Save confusion matrix plot
    cm = np.array(metrics['confusion_matrix'])
    cm_labels = class_names if task == 'multi' else ['Normal', 'Attack']
    
    # Create per-experiment figure folder (redundant check but safe if LC block skipped)
    exp_fig_dir = results_dir / 'figures' / experiment_id
    exp_fig_dir.mkdir(parents=True, exist_ok=True)
    
    cm_path = exp_fig_dir / "confusion_matrix.png"
    plot_confusion_matrix(
        cm=cm,
        labels=cm_labels,
        save_path=str(cm_path),
        title=f"Confusion Matrix: {experiment_id.replace('_', ' ').title()} (Seed {seed})"
    )
    
    # 5. ROC and PR Curves (Binary only)
    if task == 'binary':
        try:
            # ROC Curve
            roc_path = exp_fig_dir / "roc_curve.png"
            # Get positive class probabilities (assuming class 1 is Attack)
            y_scores = y_pred_proba[:, 1] if y_pred_proba.shape[1] > 1 else y_pred_proba[:, 0]
            
            plot_roc_curve(
                y_true=y_test,
                y_scores=y_scores,
                save_path=str(roc_path),
                title=f"ROC Curve: {experiment_id} (Seed {seed})"
            )
            
            # PR Curve
            pr_path = exp_fig_dir / "pr_curve.png"
            plot_pr_curve(
                y_true=y_test,
                y_scores=y_scores,
                save_path=str(pr_path),
                title=f"Precision-Recall Curve: {experiment_id} (Seed {seed})"
            )
        except Exception as e:
            logger.warning(f"Failed to plot ROC/PR curves for {experiment_id}: {e}")

    # 6. Feature Importance (Tree models only)
    if hasattr(model, 'feature_importances_'):
        try:
            fi_path = exp_fig_dir / "feature_importance.png"
            # Get feature names from preprocessor if possible, else generic
            # preprocessor is local in main(), not passed here. 
            # We construct generic names or try to retrieve.
            # For now, let's just use indices or passed names if we had them.
            # We can rely on X_train.shape[1]
            feat_names = [f"Feature_{i}" for i in range(X_train.shape[1])]
            
            # Attempt to get real names if available? 
            # We don't have easy access to preprocessor here. 
            # But the user might care. 
            # Let's check if we can pass them. 
            # For now, generic names are better than crashing.
            
            plot_feature_importance(
                importances=model.feature_importances_,
                feature_names=feat_names,
                save_path=str(fi_path),
                title=f"Feature Importances: {experiment_id} (Seed {seed})"
            )
        except Exception as e:
             logger.warning(f"Failed to plot feature importance for {experiment_id}: {e}")
    
    logger.info(f"Completed {experiment_id} in {total_time:.2f}s")
    
    return result


def main():
    """
    Main entry point for the experiment pipeline.
    
    Runs all experiments in parallel with multiple seeds.
    """
    import argparse
    parser = argparse.ArgumentParser(description="Run UNSW-NB15 Experiment Grid")
    parser.add_argument("--config", type=str, default="configs/main.yaml", help="Path to configuration file")
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config)
    
    # Setup logging
    log_path = Path(config['results_dir']) / 'logs' / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(level="INFO", log_file=str(log_path))
    
    logger.info("=" * 70)
    logger.info("UNSW-NB15 CLASS IMBALANCE STUDY (OPTIMIZED)")
    logger.info("=" * 70)
    logger.info(f"Started at: {datetime.now().isoformat()}")
    
    results_dir = Path(config['results_dir'])
    
    # Create output directories
    for subdir in ['metrics', 'figures', 'models', 'tables', 'logs', 'processed']:
        (results_dir / subdir).mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load and preprocess data
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 1: DATA PREPROCESSING")
    logger.info("=" * 50)
    
    loader = DataLoader(config)
    train_df, test_df = loader.load_all()
    
    preprocessor = UNSWPreprocessor(config)
    preprocessor.fit_transform(train_df, test_df)
    
    # Save preprocessing metadata
    preprocessor.save_metadata(str(results_dir / 'processed' / 'preprocessing_metadata.json'))
    
    # Step 2: Run experiment grid
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 2: RUNNING EXPERIMENTS")
    logger.info("=" * 50)
    
    tasks = config['experiments']['tasks']
    models = config['experiments']['models']
    strategies = config['experiments']['strategies']
    n_seeds = config['experiments'].get('n_seeds', 1)
    n_jobs = config['experiments'].get('n_jobs', 1)
    
    seeds = range(42, 42 + n_seeds)
    
    experiment_queue = []
    skipped_results = []
    
    for task in tasks:
        # Get data for this task (Large arrays, but shared in memory for threads/processes hopefully)
        # Joblib with 'loky' backend uses pickling. Large arrays might be slow to copy.
        # But 'threading' backend shares memory. sklearn models release GIL often.
        # Let's use default (loky) but be aware of memory.
        
        X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.get_splits(task)
        
        if task == 'multi':
            class_names = list(preprocessor.label_encoder.classes_)
        else:
            class_names = ['Normal', 'Attack']
        
        for model_name in models:
            for strategy_name in strategies:
                for seed in seeds:
                    experiment_id = f"{task}_{model_name}_{strategy_name}_s{seed}"
                    
                    # Check if completed
                    metrics_file = results_dir / 'metrics' / f"{experiment_id}.json"
                    if metrics_file.exists():
                        logger.info(f"Skipping {experiment_id} (found existing metrics)")
                        try:
                            with open(metrics_file, 'r') as f:
                                skipped_results.append(json.load(f))
                        except Exception:
                            logger.warning(f"Could not load {metrics_file}, re-running.")
                            pass
                        else:
                            continue

                    experiment_queue.append({
                        'experiment_id': experiment_id,
                        'task': task,
                        'model_name': model_name,
                        'strategy_name': strategy_name,
                        'X_train': X_train,
                        'y_train': y_train,
                        'X_test': X_test,
                        'y_test': y_test,
                        'X_val': X_val,
                        'y_val': y_val,
                        'config': config,
                        'results_dir': results_dir,
                        'class_names': class_names,
                        'seed': seed
                    })

    logger.info(f"Queued {len(experiment_queue)} experiments. (Skipped {len(skipped_results)})")
    
    # Run in parallel
    # Use a try-except wrapper via a helper if needed, but run_single_experiment should be robust enough?
    # Let's wrap it inline if possible or just rely on joblib handling exceptions
    
    if experiment_queue:
        logger.info(f"Starting parallel execution with n_jobs={n_jobs}...")
        parallel_results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(run_single_experiment)(**args) for args in experiment_queue
        )
        all_results = skipped_results + parallel_results
    else:
        all_results = skipped_results

    # Step 3: Generate summary reports
    logger.info("\n" + "=" * 50)
    logger.info("PHASE 3: GENERATING REPORTS")
    logger.info("=" * 50)
    
    # Save raw log
    log_data = []
    for r in all_results:
        if 'metrics' in r:
            log_data.append({
                'experiment_id': r.get('experiment_id'),
                'seed': r.get('seed'),
                'task': r.get('task'),
                'model': r.get('model'),
                'strategy': r.get('strategy'),
                'accuracy': r['metrics']['overall']['accuracy'],
                'macro_f1': r['metrics']['overall']['macro_f1'],
                'g_mean': r['metrics']['overall']['g_mean'],
                'training_time': r.get('training_time_seconds'),
                'total_time': r.get('total_time_seconds')
            })
            
    pd.DataFrame(log_data).to_csv(results_dir / 'experiment_log_detailed.csv', index=False)
    
    generate_summary_tables(all_results, results_dir, config)
    
    logger.info("Pipeline Completed.")


def generate_summary_tables(
    results: List[Dict[str, Any]],
    results_dir: Path,
    config: dict
) -> None:
    """Generate aggregated summary tables from experiment results."""
    
    tables_dir = results_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Convert to DataFrame for easy aggregation
    rows = []
    for r in results:
        if 'metrics' not in r: continue
        rows.append({
            'Task': r['task'],
            'Model': r['model'].upper(),
            'Strategy': r['strategy'].upper(),
            'Seed': r.get('seed', 0),
            'Accuracy': r['metrics']['overall']['accuracy'],
            'Macro_F1': r['metrics']['overall']['macro_f1'],
            'Weighted_F1': r['metrics']['overall']['weighted_f1'],
            'G_Mean': r['metrics']['overall']['g_mean'],
            'ROC_AUC': r['metrics']['overall']['roc_auc'],
            'Training_Time': r['training_time_seconds']
        })
    
    if not rows:
        logger.warning("No results to generate tables!")
        return

    df = pd.DataFrame(rows)
    
    # 1. Aggregated Summary (Mean +/- Std)
    summary = df.groupby(['Task', 'Model', 'Strategy']).agg({
        'Accuracy': ['mean', 'std'],
        'Macro_F1': ['mean', 'std'],
        'Weighted_F1': ['mean', 'std'],
        'G_Mean': ['mean', 'std'],
        'ROC_AUC': ['mean', 'std'],
        'Training_Time': ['mean'],
        'Seed': 'count'
    }).reset_index()
    
    # Flatten columns
    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]
    summary.to_csv(tables_dir / 'aggregated_summary.csv', index=False)
    logger.info("Generated aggregated_summary.csv")
    
    # 2. Detailed results (all seeds)
    df.to_csv(tables_dir / 'all_runs.csv', index=False)
    
    # 3. Rare Class Aggregation (if applicable)
    rare_classes = config.get('rare_classes', [])
    rare_rows = []
    for r in results:
        if r.get('task') != 'multi' or 'rare_class_analysis' not in r or not r['rare_class_analysis']:
            continue
        for cls_name, cls_metrics in r['rare_class_analysis'].items():
            if cls_name in rare_classes:
                rare_rows.append({
                    'Task': r['task'],
                    'Model': r['model'].upper(),
                    'Strategy': r['strategy'].upper(),
                    'Seed': r.get('seed', 0),
                    'Rare_Class': cls_name,
                    'Precision': cls_metrics['precision'],
                    'Recall': cls_metrics['recall'],
                    'F1': cls_metrics['f1']
                })
                
    if rare_rows:
        rare_df = pd.DataFrame(rare_rows)
        rare_summary = rare_df.groupby(['Task', 'Model', 'Strategy', 'Rare_Class']).agg({
            'Precision': ['mean', 'std'],
            'Recall': ['mean', 'std'],
            'F1': ['mean', 'std']
        }).reset_index()
        rare_summary.columns = ['_'.join(col).strip('_') for col in rare_summary.columns.values]
        rare_summary.to_csv(tables_dir / 'rare_class_aggregated.csv', index=False)
        logger.info("Generated rare_class_aggregated.csv")



if __name__ == '__main__':
    main()

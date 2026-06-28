#!/usr/bin/env python3
"""
Bootstrap Statistics Generator (N=1 Protocol)
=============================================

Implements rigorous statistical validation for Single-Seed (N=1) experiments:
1. Parametric Bootstrapping on Confusion Matrices for Metric CIs.
2. Friedman Test + Nemenyi Post-hoc on Per-Class F1 Scores.

Output:
- results/tables/metric_confidence_intervals.csv (Bootstrapped)
- results/tables/friedman_test.csv
- results/tables/nemenyi_results.csv
- results/tables/per_class_metrics_dump.csv (Intermediate)
"""

import numpy as np
import pandas as pd
from scipy import stats
try:
    import scikit_posthocs as sp
except ImportError:
    sp = None

from pathlib import Path
import json
import logging
from typing import Dict, List, Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def load_metrics(results_dir: Path) -> List[Dict]:
    """Load all JSON metric files."""
    metrics_dir = results_dir / 'metrics'
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Metrics directory not found: {metrics_dir}")
        
    data = []
    files = list(metrics_dir.glob("*.json"))
    logger.info(f"Found {len(files)} JSON files")
    
    for p in files:
        try:
            with open(p, 'r') as f:
                d = json.load(f)
            data.append(d)
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")
            
    return data

def bootstrap_cm_metrics(cm: np.array, n_bootstraps: int = 1000) -> pd.DataFrame:
    """
    Parametric Bootstrap from Confusion Matrix.
    Resample counts from Multinomial(N, CM_probs).
    Returns DataFrame of metrics for each bootstrap.
    """
    # Flatten CM to get counts and probabilities
    total_samples = np.sum(cm)
    probs = cm.flatten() / total_samples
    
    # Resample
    # shape: (n_bootstraps, n_classes * n_classes)
    resampled_counts = np.random.multinomial(total_samples, probs, size=n_bootstraps)
    
    n_classes = cm.shape[0]
    metrics = []
    
    for i in range(n_bootstraps):
        # Reshape back to CM
        current_cm = resampled_counts[i].reshape(n_classes, n_classes)
        
        # Calculate Metrics (Macro F1, G-Mean)
        # Avoid division by zero
        tp = np.diag(current_cm)
        fp = np.sum(current_cm, axis=0) - tp
        fn = np.sum(current_cm, axis=1) - tp
        tn = total_samples - (tp + fp + fn)
        
        # Per-class precision/recall
        precision = np.divide(tp, (tp + fp), out=np.zeros_like(tp, dtype=float), where=(tp + fp)!=0)
        recall = np.divide(tp, (tp + fn), out=np.zeros_like(tp, dtype=float), where=(tp + fn)!=0)
        f1 = 2 * (precision * recall) / (precision + recall)
        f1 = np.nan_to_num(f1)
        
        macro_f1 = np.mean(f1)
        
        # G-Mean (Geometric Mean of Sensitivity and Specificity - approx for multiclass is product of recalls^(1/K))
        # But standard def for imbalanced often uses Recalls product
        # Here we use the definition consistent with our other code: Geometric Mean of Per-Class Recalls
        g_mean = stats.gmean(recall + 1e-9) # Add epsilon
        
        metrics.append({
            'Macro_F1': macro_f1,
            'G_Mean': g_mean
        })
        
    return pd.DataFrame(metrics)

def compute_bootstrapped_ci(data: List[Dict], output_path: Path):
    """Generate CIs for all runs."""
    results = []
    
    for run in data:
        exp_id = run['experiment_id']
        logger.info(f"Bootstrapping {exp_id}...")
        
        cm = np.array(run['metrics']['confusion_matrix'])
        
        # Run Bootstrap
        boot_df = bootstrap_cm_metrics(cm, n_bootstraps=1000)
        
        row = {
            'Task': run['task'],
            'Model': run['model'].upper(),
            'Strategy': run['strategy'].upper(),
            'Seed': run['seed'],
            'N_Bootstraps': 1000
        }
        
        for metric in ['Macro_F1', 'G_Mean']:
            mean = boot_df[metric].mean()
            lower = np.percentile(boot_df[metric], 2.5)
            upper = np.percentile(boot_df[metric], 97.5)
            
            row[f'{metric}_Mean'] = mean
            row[f'{metric}_CI_Lower'] = lower
            row[f'{metric}_CI_Upper'] = upper
            row[f'{metric}_Error_Margin'] = (upper - lower) / 2
            
        results.append(row)
        
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    logger.info(f"Saved Bootstrapped CIs to {output_path}")

def run_friedman_test(metrics_data: List[Dict], output_dir: Path):
    """
    Run Friedman Test on Per-Class F1 Scores for Multiclass Task.
    Block = Class
    Treatment = Model_Strategy
    """
    # 1. Extract Per-Class F1s
    rows = []
    for run in metrics_data:
        if run['task'] != 'multi':
            continue
            
        model_strat = f"{run['model'].upper()}_{run['strategy'].upper()}"
        
        per_class = run['metrics']['per_class']
        for cls_name, stats_dict in per_class.items():
            if cls_name in ['macro_avg', 'weighted_avg']: continue
            
            rows.append({
                'Class': cls_name,
                'Treatment': model_strat,
                'F1': stats_dict['f1']
            })
            
    df = pd.DataFrame(rows)
    if df.empty:
        logger.warning("No multiclass data found for Friedman test.")
        return
        
    # Save intermediate for reference
    df.to_csv(output_dir / 'per_class_metrics_dump.csv', index=False)
    
    # 2. Pivot for Friedman: Index=Class, Columns=Treatment
    pivot = df.pivot(index='Class', columns='Treatment', values='F1')
    
    # 3. Friedman Test
    stat, p = stats.friedmanchisquare(*[pivot[col].values for col in pivot.columns])
    
    logger.info(f"Friedman Test: Statistic={stat:.4f}, p-value={p:.4e}")
    
    # Save Friedman Result
    pd.DataFrame([{
        'Test': 'Friedman', 
        'Statistic': stat, 
        'P_Value': p,
        'Significant': p < 0.05
    }]).to_csv(output_dir / 'friedman_test.csv', index=False)
    
    # 4. Nemenyi Post-hoc (if significant)
    if p < 0.05:
        if sp is not None:
            logger.info("Friedman significant. Running Nemenyi Post-hoc...")
            nemenyi = sp.posthoc_nemenyi_friedman(pivot)
            nemenyi.to_csv(output_dir / 'nemenyi_results.csv')
            logger.info(f"Saved {output_dir / 'nemenyi_results.csv'}")
        else:
            logger.warning("Friedman significant but scikit-posthocs not installed. Skipping Nemenyi.")

def main():
    results_dir = Path("results")
    if not results_dir.exists():
        results_dir = Path("../results") # Handle script run from scripts/
        
    tables_dir = results_dir / 'tables'
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    data = load_metrics(results_dir)
    
    # 1. Bootstrapped CIs
    try:
        compute_bootstrapped_ci(data, tables_dir / 'metric_confidence_intervals.csv')
    except Exception as e:
        logger.error(f"Bootstrapping failed: {e}")
        
    # 2. Friedman Test
    try:
        run_friedman_test(data, tables_dir)
    except Exception as e:
        logger.error(f"Friedman Test failed: {e}")

if __name__ == "__main__":
    main()

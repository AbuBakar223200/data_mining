#!/usr/bin/env python3
"""
Statistical Analysis Generator
==============================

This script implements the "Statistical Validation Protocol" defined in experiment_contract.md.
It uses the multiple seeds (N=5) available in `results/tables/all_runs.csv` to:
1. Compute 95% Confidence Intervals (CIs) for all metrics.
2. Perform Paired T-Tests (or Wilcoxon) to statistically validate strategy improvements.
3. Generate the missing artifacts:
   - results/tables/metric_confidence_intervals.csv
   - results/tables/paired_significance_tests.csv
"""

import pandas as pd
import numpy as np
from scipy import stats
from pathlib import Path
import logging
import json

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(levelname)s | %(message)s')
logger = logging.getLogger(__name__)

def load_data(results_dir: Path) -> pd.DataFrame:
    """
    Load data directly from JSON metrics files to ensure completeness.
    """
    metrics_dir = results_dir / 'metrics'
    if not metrics_dir.exists():
        raise FileNotFoundError(f"Could not find {metrics_dir}")
        
    logger.info(f"Looking for metrics in: {metrics_dir.absolute()}")
    files = list(metrics_dir.glob("*.json"))
    logger.info(f"Found {len(files)} JSON files")
    
    data = []
    for p in files:
        try:
            with open(p, 'r') as f:
                d = json.load(f)
                
            # Flatten structure
            row = {
                'Task': d.get('task'),
                'Model': d.get('model', '').upper(),
                'Strategy': d.get('strategy', '').upper(),
                'Seed': d.get('seed')
            }
            
            # Extract metrics
            overall = d.get('metrics', {}).get('overall', {})
            row.update({
                'Accuracy': overall.get('accuracy'),
                'Macro_F1': overall.get('macro_f1'),
                'Weighted_F1': overall.get('weighted_f1'),
                'G_Mean': overall.get('g_mean'),
                'ROC_AUC': overall.get('roc_auc'),
                'Training_Time': d.get('training_time_seconds')
            })
            
            data.append(row)
        except Exception as e:
            logger.warning(f"Failed to load {p}: {e}")
            
    df = pd.DataFrame(data)
    if df.empty:
        logger.error("No valid runs loaded!")
        return df
        
    logger.info(f"Loaded {len(df)} runs. Columns: {df.columns.tolist()}")
    return df


def compute_ci(df: pd.DataFrame, output_path: Path):
    """
    Compute 95% Confidence Intervals for the Mean using t-distribution.
    CI = Mean +/- t * (Std / sqrt(N))
    """
    logger.info("Computing Confidence Intervals...")
    
    # Group by experimental configuration
    # Note: Column names align with all_runs.csv output from main.py
    groups = ['Task', 'Model', 'Strategy']
    metrics = ['Macro_F1', 'G_Mean', 'ROC_AUC', 'Weighted_F1']
    
    ci_rows = []
    
    grouped = df.groupby(groups)
    for name, group in grouped:
        task, model, strategy = name
        n = len(group)
        
        row = {
            'Task': task,
            'Model': model,
            'Strategy': strategy,
            'N_Seeds': n
        }
        
        for metric in metrics:
            if metric not in group.columns:
                continue
                
            data = group[metric].dropna()
            if len(data) < 2:
                row[f'{metric}_Mean'] = data.mean() if len(data) > 0 else 0
                row[f'{metric}_CI_Lower'] = data.mean() if len(data) > 0 else 0
                row[f'{metric}_CI_Upper'] = data.mean() if len(data) > 0 else 0
                continue
                
            mean = data.mean()
            sem = stats.sem(data)
            # 95% CI using t-distribution
            interval = stats.t.interval(0.95, len(data)-1, loc=mean, scale=sem)
            
            row[f'{metric}_Mean'] = mean
            row[f'{metric}_CI_Lower'] = interval[0]
            row[f'{metric}_CI_Upper'] = interval[1]
            row[f'{metric}_Error_Margin'] = mean - interval[0]
            
        ci_rows.append(row)
        
    ci_df = pd.DataFrame(ci_rows)
    ci_df.to_csv(output_path, index=False)
    logger.info(f"Saved {output_path}")

def perform_significance_tests(df: pd.DataFrame, output_path: Path):
    """
    Perform paired statistical tests (T-Test) to compare strategies.
    Null Hypothesis: Strategy A Mean == Strategy B Mean
    """
    logger.info("Performing Significance Tests...")
    
    tasks = df['Task'].unique()
    models = df['Model'].unique()
    strategies = df['Strategy'].unique()
    
    metrics = ['Macro_F1', 'G_Mean']
    results = []
    
    # 1. Strategy Impact: Does S1/S2a improve over S0?
    baseline = 'S0'
    comparisons = [s for s in strategies if s != baseline]
    
    for task in tasks:
        for model in models:
            # Get Baseline Data (align by Seed)
            base_df = df[(df['Task'] == task) & (df['Model'] == model) & (df['Strategy'] == baseline)].sort_values('Seed')
            
            if base_df.empty:
                continue
                
            for strategy in comparisons:
                comp_df = df[(df['Task'] == task) & (df['Model'] == model) & (df['Strategy'] == strategy)].sort_values('Seed')
                
                if comp_df.empty:
                    continue
                    
                # Merge on Seed to ensure paired samples
                merged = pd.merge(base_df, comp_df, on='Seed', suffixes=('_base', '_comp'))
                
                if len(merged) < 3:
                    logger.warning(f"Not enough common seeds for {task}_{model} ({baseline} vs {strategy})")
                    continue
                
                for metric in metrics:
                    if f'{metric}_base' not in merged.columns: continue
                    a = merged[f'{metric}_base']
                    b = merged[f'{metric}_comp']
                    
                    # Paired T-Test
                    t_stat, p_val = stats.ttest_rel(b, a) # b - a (Expected improvement)
                    
                    # Cohen's d (Effect Size)
                    diff = b - a
                    d = diff.mean() / diff.std() if diff.std() > 0 else 0
                    
                    results.append({
                        'Task': task,
                        'Model': model,
                        'Comparison': f"{strategy} vs {baseline}",
                        'Metric': metric,
                        'Mean_Diff': diff.mean(),
                        'P_Value': p_val,
                        'Significant_0.05': p_val < 0.05,
                        'Significant_0.01': p_val < 0.01,
                        'Effect_Size_Cohens_D': d,
                        'N': len(merged)
                    })
                    
    # 2. Model Comparison: Does XGB improve over RF? (Fixed Strategy S1)
    # Compare XGB vs RF for each Strategy
    model_pairs = [('XGB', 'RF'), ('XGB', 'LR')]
    
    for task in tasks:
        for strategy in strategies:
            for m1, m2 in model_pairs:
                 df1 = df[(df['Task'] == task) & (df['Model'] == m1) & (df['Strategy'] == strategy)].sort_values('Seed')
                 df2 = df[(df['Task'] == task) & (df['Model'] == m2) & (df['Strategy'] == strategy)].sort_values('Seed')
                 
                 merged = pd.merge(df1, df2, on='Seed', suffixes=('_m1', '_m2'))
                 
                 if len(merged) < 3: continue
                 
                 for metric in metrics:
                    if f'{metric}_m1' not in merged.columns: continue
                    a = merged[f'{metric}_m1'] # XGB
                    b = merged[f'{metric}_m2'] # RF
                    
                    # Paired T-Test
                    t_stat, p_val = stats.ttest_rel(a, b) # a - b
                    
                    diff = a - b
                    d = diff.mean() / diff.std() if diff.std() > 0 else 0

                    results.append({
                        'Task': task,
                        'Model': f"Model Comparison ({strategy})",
                        'Comparison': f"{m1} vs {m2}",
                        'Metric': metric,
                        'Mean_Diff': diff.mean(),
                        'P_Value': p_val,
                        'Significant_0.05': p_val < 0.05,
                        'Significant_0.01': p_val < 0.01,
                        'Effect_Size_Cohens_D': d,
                        'N': len(merged)
                    })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_path, index=False)
    logger.info(f"Saved {output_path}")

def main():
    results_dir = Path("results")
    if not results_dir.exists():
        # Try relative to script if run explicitly
        results_dir = Path("../results")
        
    try:
        df = load_data(results_dir)
        
        # 1. Compute CIs
        compute_ci(df, results_dir / 'tables' / 'metric_confidence_intervals.csv')
        
        # 2. Perform Significance Tests
        perform_significance_tests(df, results_dir / 'tables' / 'paired_significance_tests.csv')
        
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()

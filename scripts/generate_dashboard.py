#!/usr/bin/env python3
"""
Results Dashboard Generator
===========================
Aggregates metrics from all 90 experiments and generates:
1. Summary Tables (CSV/Latex)
2. Heatmaps (Performance comparisons)
3. Radar Charts (Rare class analysis)
4. Stability Boxplots (Variance across seeds)

Usage:
    python scripts/generate_dashboard.py
"""

import json
import logging
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Setup Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)
STRATEGY_ORDER = ['s0', 's1', 's2a', 's2b']
STRATEGY_COLORS = {'s0': '#6C757D', 's1': '#0077B6', 's2a': '#38B000', 's2b': '#9B5DE5'}

def normalize_strategy(strategy) -> str:
    if pd.isna(strategy):
        return ''
    return str(strategy).strip().lower()

def ordered_strategies(values):
    present = {normalize_strategy(v) for v in values if normalize_strategy(v)}
    ordered = [s for s in STRATEGY_ORDER if s in present]
    extras = sorted(s for s in present if s not in STRATEGY_ORDER)
    return ordered + extras

def strategy_color(strategy) -> str:
    return STRATEGY_COLORS.get(normalize_strategy(strategy), '#333333')

def strategy_label(strategy) -> str:
    return normalize_strategy(strategy).upper()

def load_all_metrics(results_dir: Path) -> pd.DataFrame:
    """Load all JSON metrics files into a DataFrame."""
    metrics_dir = results_dir / 'metrics'
    data = []
    
    files = list(metrics_dir.glob("*.json"))
    logger.info(f"Found {len(files)} metric files.")
    
    for f in files:
        try:
            with open(f, 'r') as fp:
                d = json.load(fp)
                
            # Flatten basic metrics
            row = {
                'experiment_id': d['experiment_id'],
                'task': d['task'],
                'model': d['model'],
                'strategy': normalize_strategy(d['strategy']),
                'seed': d.get('seed', 42),
                'accuracy': d['metrics']['overall']['accuracy'],
                'macro_f1': d['metrics']['overall']['macro_f1'],
                'g_mean': d['metrics']['overall']['g_mean'],
                'roc_auc': d['metrics']['overall'].get('roc_auc', np.nan),
                'training_time': d.get('training_time_seconds', np.nan)
            }
            
            # Add Rare Class Probing (if available)
            if d.get('rare_class_analysis'):
                for cls, metrics in d['rare_class_analysis'].items():
                    row[f"recall_{cls}"] = metrics['recall']
                    row[f"f1_{cls}"] = metrics['f1']
            
            data.append(row)
        except Exception as e:
            logger.warning(f"Error loading {f.name}: {e}")
            
    return pd.DataFrame(data)

def plot_heatmaps(df: pd.DataFrame, output_dir: Path):
    """Generate Heatmaps for Strategy Impact."""
    logger.info("Generating Heatmaps...")
    
    # Filter for Primary Metric (G-Mean)
    # Average across seeds
    pivot_cols = ['task', 'model', 'strategy']
    grouped = df.groupby(pivot_cols)['g_mean'].mean().reset_index()
    
    for task in grouped['task'].unique():
        task_df = grouped[grouped['task'] == task]
        
        # Pivot: Rows=Strategy, Cols=Model
        matrix = task_df.pivot(index='strategy', columns='model', values='g_mean')
        
        # Reorder for logic
        strat_order = ordered_strategies(matrix.index)
        model_order = [m for m in ['lr', 'rf', 'xgb'] if m in matrix.columns]
        matrix = matrix.reindex(index=strat_order, columns=model_order)
        
        plt.figure(figsize=(10, 6))
        sns.heatmap(matrix, annot=True, cmap='viridis', fmt='.3f', vmin=0.5, vmax=1.0)
        plt.title(f"{task.title()} Task: G-Mean Performance")
        plt.tight_layout()
        plt.savefig(output_dir / f"heatmap_gmean_{task}.png")
        plt.close()

def plot_rare_class_radar(df: pd.DataFrame, output_dir: Path):
    """Generate Radar/Spider Chart for Rare Classes (Multiclass Only)."""
    logger.info("Generating Radar Charts...")
    
    multi_df = df[df['task'] == 'multi']
    if multi_df.empty:
        logger.warning("No multiclass data for radar chart.")
        return

    # Target Rare Classes
    rare_classes = ['Worms', 'Shellcode', 'Backdoor', 'Analysis']
    
    # Compare strategies for XGBoost (best model usually)
    # Group by Strategy, Average metrics
    subset = multi_df[multi_df['model'] == 'xgb'].groupby('strategy').mean(numeric_only=True)
    
    # Prepare Radar Data
    strategies = ordered_strategies(subset.index)
    if not strategies: return
    
    # Metrics to plot: Recall for each rare class
    labels = rare_classes
    
    try:
        # Check if columns exist
        available_labels = [l for l in labels if f"recall_{l}" in subset.columns]
        if not available_labels:
            logger.warning("Rare class metrics not found in dataframe.")
            return
        
        # Setup angles
        angles = np.linspace(0, 2*np.pi, len(available_labels), endpoint=False).tolist()
        angles += angles[:1]  # Close loop
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        for strategy in strategies:
            values = [subset.loc[strategy, f"recall_{l}"] for l in available_labels]
            values += values[:1] # Close loop
            
            ax.plot(angles, values, linewidth=2, label=strategy_label(strategy), color=strategy_color(strategy))
            ax.fill(angles, values, alpha=0.1, color=strategy_color(strategy))
            
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_labels)
        ax.set_ylim(0, 1.0)
        plt.title("XGBoost: Rare Class Recall by Strategy")
        plt.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        plt.savefig(output_dir / "radar_rare_class_xgb.png")
        plt.close()
        
    except Exception as e:
        logger.error(f"Failed to plot radar chart: {e}")

def plot_stability_boxplots(df: pd.DataFrame, output_dir: Path):
    """Plot variance across seeds to show stability."""
    logger.info("Generating Stability Boxplots...")
    
    for task in df['task'].unique():
        plt.figure(figsize=(12, 6))
        
        # Create a combined column for x-axis
        df['config'] = df['model'].str.upper() + "_" + df['strategy'].str.upper()
        
        # Sort key to keep order logical
        df = df.sort_values(['model', 'strategy'])
        
        sns.boxplot(data=df[df['task'] == task], x='config', y='macro_f1', hue='model')
        plt.title(f"{task.title()} Stability (Macro F1 across 5 seeds)")
        plt.xticks(rotation=45)
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_dir / f"stability_{task}.png")
        plt.close()

def main():
    config = load_config("configs/main.yaml")
    results_dir = Path(config['results_dir'])
    
    # Setup Output
    dashboard_dir = results_dir / 'dashboard'
    dashboard_dir.mkdir(exist_ok=True, parents=True)
    setup_logging(level="INFO", log_file=str(dashboard_dir / "dashboard_gen.log"))
    
    logger.info("Starting Dashboard Generation...")
    
    # 1. Load Data
    df = load_all_metrics(results_dir)
    if df.empty:
        logger.error("No data found!")
        return

    # 2. Generate Tables
    # Summary Table
    summary = df.groupby(['task', 'model', 'strategy']).agg({
        'g_mean': ['mean', 'std'],
        'macro_f1': ['mean', 'std'],
        'accuracy': ['mean', 'std'],
        'seed': 'count'
    }).round(4)
    
    summary.to_csv(dashboard_dir / "final_summary_table.csv")
    logger.info(f"Saved summary table to {dashboard_dir}")

    # Rare Class Table
    if 'task' in df.columns and 'recall_Worms' in df.columns:
        rare_cols = [c for c in df.columns if c.startswith('recall_') or c.startswith('f1_')]
        rare_summary = df[df['task']=='multi'].groupby(['model', 'strategy'])[rare_cols].mean().round(4)
        rare_summary.to_csv(dashboard_dir / "rare_class_summary.csv")
    
    # 3. Generate Plots
    plot_heatmaps(df, dashboard_dir)
    plot_rare_class_radar(df, dashboard_dir)
    plot_stability_boxplots(df, dashboard_dir)
    
    logger.info("Dashboard Generation Complete.")

if __name__ == "__main__":
    main()


import sys
import os
import json
import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.evaluation.visualizer import PublicationVisualizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def main():
    logger.info("Starting Report Generation...")
    
    # Paths
    results_dir = Path("results")
    log_path = results_dir / "experiment_log_detailed.csv"
    output_dir = Path("reports")
    figure_dir = results_dir / "figures_final"
    
    output_dir.mkdir(exist_ok=True)
    figure_dir.mkdir(exist_ok=True, parents=True)
    
    # Load Data
    if not log_path.exists():
        logger.error(f"Log file not found at {log_path}")
        return
        
    df = pd.read_csv(log_path)
    logger.info(f"Loaded {len(df)} runs from log.")

    tasks = sorted(df['task'].dropna().unique().tolist())
    models = sorted(df['model'].dropna().unique().tolist())
    strategies = sorted(df['strategy'].dropna().unique().tolist())
    seeds = sorted(df['seed'].dropna().unique().tolist()) if 'seed' in df.columns else [None]
    grid_info = {
        'task_count': len(tasks),
        'model_count': len(models),
        'strategy_count': len(strategies),
        'seed_count': len(seeds),
    }
    grid_info['configuration_count'] = (
        grid_info['task_count'] * grid_info['model_count'] * grid_info['strategy_count']
    )
    grid_info['expected_runs'] = grid_info['configuration_count'] * grid_info['seed_count']
    
    # Initialize Visualizer
    viz = PublicationVisualizer()
    
    # 1. Radar Chart: Best Model Comparison (Binary)
    # Get best of each model type for binary task
    best_binary = df[df['task'] == 'binary'].sort_values('g_mean', ascending=False).groupby('model').first()
    
    radar_data = {}
    metrics = ['accuracy', 'macro_f1', 'g_mean']
    
    for model, row in best_binary.iterrows():
        radar_data[model] = {m: row[m] for m in metrics}
        
    viz.plot_radar_chart(
        radar_data, 
        metrics, 
        str(figure_dir / "radar_binary_best.png"),
        title="Best Binary Models Comparison"
    )
    
    # 2. Strategy Comparison (Rank Plot)
    # Calculate average rank of strategies across all tasks/models
    # Lower rank is better
    df['rank'] = df.groupby(['task', 'model'])['g_mean'].rank(ascending=False)
    avg_ranks = df.groupby('strategy')['rank'].mean().to_dict()
    
    viz.plot_critical_difference_proxy(
        avg_ranks,
        str(figure_dir / "strategy_ranks.png")
    )

    # 3. Generate Markdown Report
    report_content = generate_markdown(df, avg_ranks, grid_info)
    
    with open(output_dir / "final_results.md", "w") as f:
        f.write(report_content)
        
    logger.info(f"Report generated at {output_dir / 'final_results.md'}")

def generate_markdown(df, ranks, grid_info):
    best_overall = df.loc[df['g_mean'].idxmax()]
    strategy_labels = {
        's0': 'No Balancing',
        's1': 'Class Weighting',
        's2a': 'Random Oversampling',
        's2b': 'SMOTE',
    }
    strategy_order = [s for s in ['s0', 's1', 's2a', 's2b'] if s in ranks]
    if not strategy_order:
        strategy_order = sorted(ranks.keys())
    strategy_summary = ", ".join(
        f"{strategy_labels.get(s, s.upper())} ({s.upper()})" for s in strategy_order
    )
    
    md = f"""# Final Research Results: UNSW-NB15 Imbalance Study

## Executive Summary
This study systematically evaluated the impact of class imbalance strategies on ID systems.
The rigorous {grid_info['configuration_count']}-configuration grid across {grid_info['seed_count']} seeds ({grid_info['expected_runs']} total runs) confirms that **{best_overall['strategy'].upper()}** using **{best_overall['model'].upper()}** achieves the state-of-the-art performance with a G-Mean of **{best_overall['g_mean']:.4f}**.

## Key Findings

### 1. Strategy Ranking
We compared {strategy_summary}.
The average rankings (lower is better, across all tasks) are:

![Strategy Ranks](../results/figures_final/strategy_ranks.png)

| Strategy | Avg Rank | Description |
|----------|----------|-------------|
"""
    for strat, rank in sorted(ranks.items(), key=lambda x: x[1]):
        md += f"| {strat.upper()} | {rank:.2f} | ... |\n"
        
    md += """
### 2. Model Performance Analysis
The radar chart below illustrates the trade-offs between Accuracy, F1, and G-Mean for the top models in the Binary task.

![Radar Chart](../results/figures_final/radar_binary_best.png)

## Detailed Results Table
"""
    # Add table of top 5 runs
    top5 = df.sort_values('g_mean', ascending=False).head(5)[['task', 'model', 'strategy', 'g_mean', 'macro_f1']]
    md += top5.to_markdown(index=False)
    
    return md

if __name__ == "__main__":
    main()

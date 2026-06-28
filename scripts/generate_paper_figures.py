import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import pi
from pathlib import Path

STRATEGY_ORDER = ['S0', 'S1', 'S2A', 'S2B']
STRATEGY_COLORS = {'S0': '#e74c3c', 'S1': '#3498db', 'S2A': '#2ecc71', 'S2B': '#9b59b6'}

def normalize_strategy(strategy):
    if pd.isna(strategy):
        return ''
    return str(strategy).strip().upper()

def ordered_strategies(values):
    present = {normalize_strategy(v) for v in values if normalize_strategy(v)}
    ordered = [s for s in STRATEGY_ORDER if s in present]
    extras = sorted(s for s in present if s not in STRATEGY_ORDER)
    return ordered + extras

def plot_radar_chart(df, task, model, output_path):
    """
    Generate Radar Chart for Model Strategy Comparison.
    Metrics: Macro_F1, G_Mean, Accuracy, Weighted_F1, ROC_AUC
    """
    # Filter by task
    task_df = df[df['Task'] == task].copy()
    if task_df.empty:
        return

    task_df['Strategy_norm'] = task_df['Strategy'].apply(normalize_strategy)
    strategies = ordered_strategies(task_df['Strategy_norm'].unique())
    
    metrics = ['Macro_F1_mean', 'G_Mean_mean', 'Accuracy_mean', 'Weighted_F1_mean', 'ROC_AUC_mean']
    labels = ['Macro F1', 'G-Mean', 'Accuracy', 'Weighted F1', 'ROC-AUC']
    
    # Create background
    categories = labels
    N = len(categories)
    
    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    plt.xticks(angles[:-1], categories, color='grey', size=10)
    ax.set_rlabel_position(0)
    plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=7)
    plt.ylim(0, 1.05)
    
    for strategy in strategies:
        row = task_df[(task_df['Model'] == model) & (task_df['Strategy_norm'] == strategy)]
        if row.empty: continue
        
        values = row[metrics].values.flatten().tolist()
        values += values[:1]
        
        ax.plot(angles, values, linewidth=2, linestyle='solid', label=f"{strategy}", color=STRATEGY_COLORS.get(strategy, '#7f8c8d'))
        ax.fill(angles, values, color=STRATEGY_COLORS.get(strategy, '#7f8c8d'), alpha=0.1)
        
    plt.title(f"{model} Performance: Strategy Comparison ({task.title()})", size=15, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved {output_path}")
    plt.close()

def plot_rare_class_bar(df, metric, output_path):
    """
    Bar chart for Rare Class Metrics (Multi-class strategy comparison).
    Intended for 'Worms', 'Shellcode', 'Backdoor', 'Analysis'
    """
    rare_classes = ['Worms', 'Shellcode', 'Backdoor', 'Analysis']
    model = 'XGB' # Focus on XGB for per-class analysis as it's usually best
    
    # Filter
    df = df[(df['Task'] == 'multi') & (df['Model'] == model) & (df['Rare_Class'].isin(rare_classes))]
    if df.empty: return

    # Pivot: Index=Class, Cols=Strategy, Val=Metric
    # Metric column name e.g. 'Recall_mean'
    col_name = f"{metric}_mean"
    if col_name not in df.columns:
        print(f"Column {col_name} not found in dataframe.")
        return

    pivot = df.pivot(index='Rare_Class', columns='Strategy', values=col_name)
    pivot.columns = [normalize_strategy(c) for c in pivot.columns]
    
    # Reorder if needed
    available_strategies = ordered_strategies(pivot.columns)
    if not available_strategies:
        return
    pivot = pivot[available_strategies]
    
    colors = [STRATEGY_COLORS.get(s, '#7f8c8d') for s in available_strategies]
    ax = pivot.plot(kind='bar', figsize=(10, 6), width=0.8, color=colors)
    
    plt.title(f"Rare Class {metric} Improvement ({model})", fontsize=14)
    plt.ylabel(f"{metric} Score", fontsize=12)
    plt.xlabel("Attack Category", fontsize=12)
    plt.xticks(rotation=45)
    plt.ylim(0, 1.0)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.legend(title='Strategy')
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f', padding=3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Saved {output_path}")
    plt.close()

def main():
    results_dir = Path('results')
    tables_dir = results_dir / 'tables'
    fig_dir = results_dir / 'figures'
    fig_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Radar Chart Data
    summary_path = tables_dir / 'aggregated_summary.csv'
    if summary_path.exists():
        df_summary = pd.read_csv(summary_path)
        
        # Loop through all models and tasks
        tasks = ['binary', 'multi']
        models = ['LR', 'RF', 'XGB']
        
        for task in tasks:
            for model in models:
                output_file = fig_dir / f"radar_{task}_{model.lower()}.png"
                plot_radar_chart(df_summary, task, model, output_file)

    else:
        print("aggregated_summary.csv not found!")

    # 2. Rare Class Bar Chart Data
    rare_path = tables_dir / 'rare_class_aggregated.csv'
    if rare_path.exists():
        df_rare = pd.read_csv(rare_path)
        
        # Loop through metrics
        metrics = ['Precision', 'Recall', 'F1']
        for metric in metrics:
             output_file = fig_dir / f"rare_class_{metric.lower()}_comparison.png"
             plot_rare_class_bar(df_rare, metric, output_file)
    else:
        print("rare_class_aggregated.csv not found!")

if __name__ == "__main__":
    main()

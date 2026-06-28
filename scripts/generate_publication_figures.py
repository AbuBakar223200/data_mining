import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import math

# ==========================================
# 1. CONFIGURATION & STANDARDS
# ==========================================

# Colors
TASK_COLORS = {'binary': '#2E86AB', 'multi': '#A23B72'}
MODEL_COLORS = {'lr': '#E63946', 'rf': '#06A77D', 'xgb': '#F77F00', 'LR': '#E63946', 'RF': '#06A77D', 'XGB': '#F77F00'}
STRATEGY_ORDER = ['s0', 's1', 's2a', 's2b']
STRATEGY_LABELS = {'s0': 'S0', 's1': 'S1', 's2a': 'S2A', 's2b': 'S2B'}
STRATEGY_DESCRIPTIONS = {'s0': 'Baseline', 's1': 'Weight', 's2a': 'ROS', 's2b': 'SMOTE'}
STRATEGY_COLORS = {'s0': '#6C757D', 's1': '#0077B6', 's2a': '#38B000', 's2b': '#9B5DE5'}
CLASS_COLORS = {
    'Normal': '#2A9D8F', 'Generic': '#E76F51', 'Exploits': '#F4A261',
    'Fuzzers': '#E9C46A', 'DoS': '#8338EC', 'Reconnaissance': '#3A86FF',
    'Analysis': '#FB5607', 'Backdoor': '#FF006E', 'Shellcode': '#FFBE0B',
    'Worms': '#D62828'
}

# Fonts & Sizes
plt.rcParams['font.family'] = 'DejaVu Sans'
FONT_SIZES = {'title': 16, 'axis_label': 14, 'tick_label': 12, 'legend': 11, 'annotation': 10}

# Output Directory
OUTPUT_DIR = 'results/figures/comprehensive'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def normalize_strategy(strategy):
    if pd.isna(strategy):
        return ''
    return str(strategy).strip().lower()

def ordered_strategies(values):
    present = {normalize_strategy(v) for v in values if normalize_strategy(v)}
    ordered = [s for s in STRATEGY_ORDER if s in present]
    extras = sorted(s for s in present if s not in STRATEGY_ORDER)
    return ordered + extras

def strategy_color(strategy):
    return STRATEGY_COLORS.get(normalize_strategy(strategy), '#333')

def strategy_label(strategy):
    key = normalize_strategy(strategy)
    return STRATEGY_LABELS.get(key, str(strategy).upper())

# Load Data
def load_data():
    df_summary = pd.read_csv('results/tables/final_summary_tables.csv')
    df_class = pd.read_csv('results/tables/per_class_metrics.csv')
    df_rare = pd.read_csv('results/tables/rare_class_report.csv')
    df_log = pd.read_csv('results/experiment_log.csv')
    try:
        df_dump = pd.read_csv('results/tables/per_class_metrics_dump.csv')
    except FileNotFoundError:
        df_dump = pd.DataFrame()
    return df_summary, df_class, df_rare, df_log, df_dump

# Helper: Save Figure
def save_fig(fig, name):
    path = os.path.join(OUTPUT_DIR, f"{name}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"Saved: {path}")
    plt.close(fig)

# ==========================================
# 2. PLOTTING FUNCTIONS
# ==========================================

def plot_performance_comparison(df):
    """Grouped Bar Charts for Macro-F1 and G-Mean"""
    metrics = [('Macro_F1', 'Macro-F1 Score'), ('G_Mean', 'G-Mean Score'), ('ROC_AUC', 'ROC-AUC Score')]
    df = df.copy()
    df['Strategy_norm'] = df['Strategy'].apply(normalize_strategy)
    
    for metric, label in metrics:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Prepare data for plotting
        models = df['Model'].unique()
        strategies = ordered_strategies(df['Strategy_norm'].unique())
        if not strategies:
            plt.close(fig)
            continue
        bar_width = min(0.25, 0.8 / len(strategies))
        
        # Position of bars on x-axis
        r = np.arange(len(models))
        
        for i, strategy in enumerate(strategies):
            subset = df[df['Strategy_norm'] == strategy]
            # Ensure order matches 'models'
            values = [subset[subset['Model'] == m][metric].values[0] if not subset[subset['Model'] == m].empty else 0 for m in models]
            
            pos = r + (i - (len(strategies) - 1) / 2) * bar_width
            ax.bar(pos, values, color=strategy_color(strategy), width=bar_width, 
                   edgecolor='white', label=strategy_label(strategy))

        # Formatting
        ax.set_xlabel('Model', fontsize=FONT_SIZES['axis_label'])
        ax.set_ylabel(label, fontsize=FONT_SIZES['axis_label'])
        ax.set_title(f'{label} Comparison Across Strategies', fontsize=FONT_SIZES['title'], fontweight='bold')
        ax.set_xticks(r)
        ax.set_xticklabels(models, fontsize=FONT_SIZES['tick_label'])
        ax.legend(title='Strategy', fontsize=FONT_SIZES['legend'])
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        
        # Remove top/right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        save_fig(fig, f"bar_comparison_{metric.lower()}")

def plot_training_efficiency(df):
    """Scatter Plot: Training Time vs Macro-F1"""
    fig, ax = plt.subplots(figsize=(10, 6))
    df = df.copy()
    df['strategy_norm'] = df['strategy'].apply(normalize_strategy)
    
    # Map markers to models
    markers = {'lr': 'o', 'rf': 's', 'xgb': '^'}
    
    for idx, row in df.iterrows():
        strategy = row['strategy_norm']
        model = row['model']
        
        ax.scatter(row['training_time'], row['macro_f1'], 
                   color=strategy_color(strategy),
                   marker=markers.get(model, 'o'),
                   s=150, alpha=0.8, edgecolors='black', linewidth=0.5,
                   label=f"{model.upper()}-{strategy_label(strategy)}")

    ax.set_xlabel('Training Time (seconds)', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel('Macro-F1 Score', fontsize=FONT_SIZES['axis_label'])
    ax.set_title('Efficiency Frontier: Time vs Performance', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Custom Legend
    # We can't use the default legend easily because of the mix, so we'll simplify
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', label='LR', markersize=10),
        Line2D([0], [0], marker='s', color='w', markerfacecolor='gray', label='RF', markersize=10),
        Line2D([0], [0], marker='^', color='w', markerfacecolor='gray', label='XGB', markersize=10),
    ]
    for strategy in ordered_strategies(df['strategy_norm'].unique()):
        desc = STRATEGY_DESCRIPTIONS.get(strategy, 'Strategy')
        legend_elements.append(
            Line2D([0], [0], color=strategy_color(strategy), lw=4, label=f"{strategy_label(strategy)} ({desc})")
        )
    ax.legend(handles=legend_elements, loc='lower right')
    
    save_fig(fig, "scatter_efficiency")

def plot_class_heatmap(df):
    """Heatmap of F1 Score per Class per Experiment"""
    # Pivot: Index=Class, Columns=Experiment(Model_Strategy), Values=F1
    # Aggregate first to avoid duplicate keys across runs/tasks.
    df = df.copy()
    df['Experiment'] = df['Model'] + '_' + df['Strategy']
    aggregated = (
        df.groupby(['Class', 'Experiment'], as_index=False)['F1']
        .mean()
    )
    pivot = aggregated.pivot(index='Class', columns='Experiment', values='F1')
    
    # Sort Index by standard class order/frequency if possible
    class_order = ['Normal', 'Generic', 'Exploits', 'Fuzzers', 'DoS', 'Reconnaissance', 
                   'Analysis', 'Backdoor', 'Shellcode', 'Worms']
    # Filter to existing classes
    class_order = [c for c in class_order if c in pivot.index]
    pivot = pivot.reindex(class_order)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlGnBu', 
                linewidths=.5, ax=ax, vmin=0, vmax=1.0)
    
    ax.set_title('Class-Wise F1-Score Heatmap', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax.set_xlabel('Experiment Configuration', fontsize=FONT_SIZES['axis_label'])
    ax.set_ylabel('Class', fontsize=FONT_SIZES['axis_label'])
    plt.xticks(rotation=45, ha='right')
    
    save_fig(fig, "heatmap_class_f1")

def plot_rare_trajectory(df):
    """Line Chart: Recall Trajectory for Rare Classes"""
    # Filter for Rare Classes
    rare_classes = ['Worms', 'Shellcode', 'Backdoor', 'Analysis']
    df_rare = df[df['Class'].isin(rare_classes)].copy()
    
    strategy_order = ordered_strategies(df_rare['Strategy'].unique())
    if not strategy_order:
        print("Warning: No strategies found for rare trajectory.")
        return
    x_labels = [strategy_label(s) for s in strategy_order]
            
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for cls in rare_classes:
        subset = df_rare[df_rare['Class'] == cls]
        # Aggregate if multiple models (take mean) or plot specific model
        # Let's plot mean across models to show general strategy effect
        means = []
        for strat in strategy_order:
            val = subset[subset['Strategy'].apply(normalize_strategy) == strat]['Recall'].mean()
            means.append(val)
            
        ax.plot(x_labels, means, marker='o', linewidth=3, 
                label=cls, color=CLASS_COLORS.get(cls, 'black'))

    ax.set_title('Rare Class Recall Trajectory (Avg across Models)', fontsize=FONT_SIZES['title'], fontweight='bold')
    ax.set_ylabel('Recall Score', fontsize=FONT_SIZES['axis_label'])
    ax.set_xlabel('Strategy', fontsize=FONT_SIZES['axis_label'])
    ax.legend(title='Rare Class')
    ax.grid(True, alpha=0.3)
    
    save_fig(fig, "line_rare_trajectory")

def plot_radar(df):
    """Radar Chart for multiclass strategy comparison."""
    # Filter for Multiclass task and one model
    df_multi = df[df['Task'].astype(str).str.lower() == 'multi'].copy()
    df_multi['Model_norm'] = df_multi['Model'].astype(str).str.upper()
    model = 'RF'
    candidate_metrics = ['Accuracy', 'Macro_F1', 'Weighted_F1', 'G_Mean', 'ROC_AUC']
    metrics = [m for m in candidate_metrics if m in df_multi.columns]
    if len(metrics) < 3:
        print("Warning: Not enough metrics available for radar chart.")
        return
    
    strategies = ordered_strategies(df_multi[df_multi['Model_norm'] == model]['Strategy'].unique())
    if not strategies:
        print(f"Warning: No data for Radar {model}")
        return
    
    # Setup data
    labels = metrics
    num_vars = len(labels)
    
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]
    
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    for strat in strategies:
        row = df_multi[
            (df_multi['Model_norm'] == model)
            & (df_multi['Strategy'].apply(normalize_strategy) == strat)
        ]
        if row.empty: 
            print(f"Warning: No data for Radar {model} {strat}")
            continue
        
        # Take the first match if multiple exist (though should be unique per task/model/strategy)
        values = row[metrics].iloc[0].values.flatten().tolist()
        values += values[:1]
        
        ax.plot(
            angles,
            values,
            linewidth=2,
            linestyle='solid',
            label=strategy_label(strat),
            color=strategy_color(strat)
        )
        ax.fill(angles, values, alpha=0.1, color=strategy_color(strat))

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels)
    
    ax.set_title(f'Radar Chart: {model} Strategy Comparison (Multiclass)', fontsize=FONT_SIZES['title'], fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
    
    save_fig(fig, "radar_rf_s0_s2a_multi")

def plot_rank_comparison(df):
    """Horizontal Bar Chart of Average Ranks (Friedman Test Visualization)"""
    if df.empty:
        print("Skipping Rank Comparison (No Data)")
        return
        
    from scipy.stats import rankdata
    
    # Pivot: Index=Class, Columns=Treatment, Values=F1
    pivot = df.pivot(index='Class', columns='Treatment', values='F1')
    
    # Rank: "Higher F1 is better". Rankdata gives 1 to smallest.
    # So we want Rank 1 to be best.
    # Let's rank descending: -F1
    ranks = pivot.apply(lambda x: rankdata(-x), axis=1, result_type='expand')
    ranks.columns = pivot.columns
    
    # Calculate Average Rank per Treatment
    avg_ranks = ranks.mean().sort_values()
    
    # Prepare Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    y_pos = np.arange(len(avg_ranks))
    colors = []
    
    for idx in avg_ranks.index:
        strat = idx.split('_')[-1] # Extract strategy (e.g. S0, S1)
        colors.append(strategy_color(strat))
        
    ax.barh(y_pos, avg_ranks.values, color=colors, edgecolor='black')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(avg_ranks.index, fontsize=FONT_SIZES['tick_label'])
    ax.invert_yaxis()  # Best rank (1) at top
    
    ax.set_xlabel('Average Rank (Lower is Better)', fontsize=FONT_SIZES['axis_label'])
    ax.set_title('Critical Difference: Average Ranks (Friedman Analysis)', fontsize=FONT_SIZES['title'], fontweight='bold')
    
    # Add counts
    for i, v in enumerate(avg_ranks.values):
        ax.text(v + 0.1, i, f"{v:.2f}", va='center', fontweight='bold')
        
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    
    # Add Friedman Check Note
    friedman_file = 'results/tables/friedman_test.csv'
    if os.path.exists(friedman_file):
        try:
            f_res = pd.read_csv(friedman_file)
            p_val = f_res['P_Value'].iloc[0]
            sig = "Significant" if p_val < 0.05 else "Not Significant"
            ax.text(0.95, 0.05, f"Friedman Test: p={p_val:.1e}\n({sig})", 
                    transform=ax.transAxes, ha='right', bbox=dict(facecolor='white', alpha=0.8))
        except:
            pass
            
    save_fig(fig, "rank_comparison")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

if __name__ == "__main__":
    print("Loading data...")
    df_sum, df_cls, df_rare, df_log, df_dump = load_data()
    
    print("Generating figures...")
    
    # 0. Rank Comparison (New)
    plot_rank_comparison(df_dump)
    
    # 1. Summary Bars
    plot_performance_comparison(df_sum)
    
    # 2. Efficiency Scatter
    # Normalize log column names if needed
    if 'task' in df_log.columns: # It's lowercase in log csv
        plot_training_efficiency(df_log)
        
    # 3. Class Heatmap
    # Normalize per_class csv
    # Check if 'Model' and 'Strategy' exist or need extraction
    if 'Model' in df_cls.columns:
        plot_class_heatmap(df_cls)
        
    # 4. Rare Trajectory
    # Check if rare_class_report has data
    if not df_rare.empty:
        # Use full per_class for trajectory to be safe if rare report is partial
        # But rare_report is specifically for this.
        # Let's use per_class df but filter for rare
        plot_rare_trajectory(df_cls)
        
    # 5. Radar
    plot_radar(df_sum)
    
    print("Done! All figures saved to results/figures/comprehensive/")

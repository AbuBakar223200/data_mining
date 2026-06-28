"""
Professional visualization functions for UNSW-NB15 experiment results.

This module provides publication-quality visualizations including:
- Confusion matrix heatmaps
- Strategy comparison bar charts
- Rare class recall analysis
- Class distribution plots

All figures use consistent professional styling suitable for research papers.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend to prevent tkinter warnings
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional, Any
import logging

logger = logging.getLogger(__name__)

# Professional color palettes
STRATEGY_COLORS = {
    's0': '#E74C3C',    # Red - baseline
    's1': '#3498DB',    # Blue - class weight
    's2a': '#2ECC71',   # Green - oversampling
    's2b': '#9B59B6'    # Purple - SMOTE
}

MODEL_COLORS = {
    'lr': '#1ABC9C',    # Teal
    'rf': '#E67E22',    # Orange
    'xgb': '#9B59B6'    # Purple
}


def set_plot_style():
    """Set consistent professional plot styling."""
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'figure.dpi': 150,
        'savefig.dpi': 300,
        'font.size': 10,
        'axes.titlesize': 12,
        'axes.labelsize': 10,
        'xtick.labelsize': 9,
        'ytick.labelsize': 9,
        'legend.fontsize': 9,
        'figure.figsize': (10, 8),
        'axes.spines.top': False,
        'axes.spines.right': False
    })


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: List[str],
    save_path: str,
    title: str = 'Confusion Matrix',
    normalize: bool = True,
    figsize: tuple = (10, 8),
    cmap: str = 'Blues'
) -> None:
    """
    Generate professional confusion matrix heatmap.
    
    Args:
        cm: Confusion matrix array (n_classes, n_classes)
        labels: Class labels
        save_path: Path to save the figure
        title: Plot title
        normalize: If True, show percentages; if False, show counts
        figsize: Figure dimensions
        cmap: Colormap name
    """
    set_plot_style()
    fig, ax = plt.subplots(figsize=figsize)
    
    # Normalize if requested
    if normalize:
        # Avoid division by zero
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        cm_display = cm.astype('float') / row_sums
        fmt = '.1%'
        vmax = 1.0
    else:
        cm_display = cm
        fmt = 'd'
        vmax = None
    
    # Create heatmap
    sns.heatmap(
        cm_display,
        annot=True,
        fmt=fmt,
        cmap=cmap,
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        vmin=0,
        vmax=vmax,
        cbar_kws={'label': 'Proportion' if normalize else 'Count'},
        square=True,
        linewidths=0.5
    )
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
    
    # Rotate x-axis labels for multiclass
    if len(labels) > 4:
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
    
    plt.tight_layout()
    
    # Save figure
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {save_path}")


def plot_strategy_comparison(
    results: Dict[str, Dict[str, float]],
    metric: str,
    save_path: str,
    title: Optional[str] = None
) -> None:
    """
    Create grouped bar chart comparing strategies across models.
    
    Args:
        results: Nested dict {model: {strategy: metric_value}}
        metric: Name of metric being compared
        save_path: Path to save figure
        title: Optional custom title
    """
    set_plot_style()
    
    models = list(results.keys())
    strategies = list(results[models[0]].keys())
    
    x = np.arange(len(models))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for i, strategy in enumerate(strategies):
        values = [results[model][strategy] for model in models]
        offset = (i - len(strategies)/2 + 0.5) * width
        bars = ax.bar(
            x + offset, values, width,
            label=strategy.upper(),
            color=STRATEGY_COLORS.get(strategy, f'C{i}'),
            edgecolor='white',
            linewidth=0.5
        )
        
        # Add value labels on bars
        for bar, val in zip(bars, values):
            ax.annotate(
                f'{val:.3f}',
                xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                ha='center', va='bottom',
                fontsize=8
            )
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel(metric.replace('_', ' ').title(), fontweight='bold')
    ax.set_title(title or f'{metric.replace("_", " ").title()} by Model and Strategy',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models])
    ax.legend(title='Strategy', loc='lower right')
    ax.set_ylim(0, 1.1)
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved strategy comparison to {save_path}")


def plot_rare_class_recall(
    results: Dict[str, Dict[str, Dict[str, float]]],
    rare_classes: List[str],
    save_path: str
) -> None:
    """
    Create heatmap showing rare class recall across experiments.
    
    Args:
        results: Nested dict {experiment_id: {class_name: {'recall': value}}}
        rare_classes: List of rare class names
        save_path: Path to save figure
    """
    set_plot_style()
    
    # Build recall matrix
    experiments = list(results.keys())
    recall_matrix = np.zeros((len(rare_classes), len(experiments)))
    
    for j, exp in enumerate(experiments):
        for i, cls in enumerate(rare_classes):
            if cls in results[exp]:
                recall_matrix[i, j] = results[exp][cls].get('recall', 0)
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    sns.heatmap(
        recall_matrix,
        annot=True,
        fmt='.2f',
        cmap='RdYlGn',
        xticklabels=experiments,
        yticklabels=rare_classes,
        ax=ax,
        vmin=0,
        vmax=1,
        cbar_kws={'label': 'Recall'},
        linewidths=0.5
    )
    
    ax.set_xlabel('Experiment', fontweight='bold')
    ax.set_ylabel('Rare Class', fontweight='bold')
    ax.set_title('Rare Class Recall Across Experiments',
                 fontsize=14, fontweight='bold', pad=20)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved rare class recall heatmap to {save_path}")


def plot_class_distribution(
    class_counts: Dict[str, int],
    save_path: str,
    title: str = 'Class Distribution'
) -> None:
    """
    Create bar chart showing class distribution.
    
    Args:
        class_counts: Dictionary {class_name: count}
        save_path: Path to save figure
        title: Plot title
    """
    set_plot_style()
    
    classes = list(class_counts.keys())
    counts = list(class_counts.values())
    
    # Sort by count descending
    sorted_pairs = sorted(zip(classes, counts), key=lambda x: x[1], reverse=True)
    classes, counts = zip(*sorted_pairs)
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Color rare classes differently
    colors = ['#E74C3C' if c < 3000 else '#3498DB' for c in counts]
    
    bars = ax.bar(classes, counts, color=colors, edgecolor='white', linewidth=0.5)
    
    # Add count labels
    for bar, count in zip(bars, counts):
        ax.annotate(
            f'{count:,}',
            xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
            ha='center', va='bottom',
            fontsize=9,
            fontweight='bold'
        )
    
    ax.set_xlabel('Attack Category', fontweight='bold')
    ax.set_ylabel('Sample Count', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_yscale('log')  # Log scale for better visibility of rare classes
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved class distribution to {save_path}")


def plot_metric_comparison_grid(
    results: List[Dict[str, Any]],
    metrics: List[str],
    save_path: str
) -> None:
    """
    Create a grid of comparison plots for multiple metrics.
    
    Args:
        results: List of experiment result dictionaries
        metrics: List of metric names to plot
        save_path: Path to save figure
    """
    set_plot_style()
    
    n_metrics = len(metrics)
    fig, axes = plt.subplots(2, (n_metrics + 1) // 2, figsize=(14, 10))
    axes = axes.flatten()
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        # Group results by task
        for task in ['binary', 'multi']:
            task_results = [r for r in results if r['task'] == task]
            models = [r['model'] for r in task_results]
            values = [r['metrics']['overall'].get(metric, 0) for r in task_results]
            
            ax.scatter(range(len(values)), values, label=task, alpha=0.7)
        
        ax.set_title(metric.replace('_', ' ').title())
        ax.set_ylim(0, 1.1)
        ax.legend()
    
    # Hide unused axes
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)
    
    plt.suptitle('Metric Comparison Across Experiments', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()


def plot_learning_curves(
    learning_curve_data: Dict[str, Dict[str, List[float]]],
    save_path: str,
    title: str = 'Learning Curves'
) -> None:
    """
    Plot learning curves (Training vs Validation) for available metrics.

    Args:
        learning_curve_data: Nested dict {set_name: {metric_name: [values]}}
                             e.g. {'validation_0': {'logloss': [...]}, ...}
        save_path: Path to save figure
        title: Plot title
    """
    set_plot_style()
    
    # Identify metrics available
    first_set = next(iter(learning_curve_data.values()))
    metrics = list(first_set.keys())
    
    fig, axes = plt.subplots(1, len(metrics), figsize=(7 * len(metrics), 5))
    if len(metrics) == 1:
        axes = [axes]
        
    mapping = {'validation_0': 'Train', 'validation_1': 'Validation'}
    colors = {'validation_0': '#3498DB', 'validation_1': '#E74C3C'}
    
    for i, metric in enumerate(metrics):
        ax = axes[i]
        
        for set_key, metrics_dict in learning_curve_data.items():
            if metric in metrics_dict:
                values = metrics_dict[metric]
                epochs = range(1, len(values) + 1)
                label = mapping.get(set_key, set_key)
                
                ax.plot(epochs, values, label=label, linewidth=2, 
                        color=colors.get(set_key, f'C{i}'))
        
        ax.set_xlabel('Epochs')
        ax.set_ylabel(metric.title())
        ax.set_title(f'{metric.title()}')
        ax.legend()
        ax.grid(True, linestyle='--', alpha=0.3)

    plt.suptitle(title, fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    
    logger.info(f"Saved learning curve plot to {save_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str,
    title: str = 'ROC Curve'
) -> None:
    """
    Plot Receiver Operating Characteristic (ROC) curve.
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities for the positive class
        save_path: Path to save figure
        title: Plot title
    """
    from sklearn.metrics import roc_curve, auc
    
    set_plot_style()
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='#2ECC71', lw=2,
            label=f'ROC curve (area = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontweight='bold')
    ax.set_ylabel('True Positive Rate', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower right")
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved ROC curve to {save_path}")


def plot_pr_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str,
    title: str = 'Precision-Recall Curve'
) -> None:
    """
    Plot Precision-Recall curve (crucial for imbalanced data).
    
    Args:
        y_true: True binary labels
        y_scores: Predicted probabilities for the positive class
        save_path: Path to save figure
        title: Plot title
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    set_plot_style()
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    average_precision = average_precision_score(y_true, y_scores)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(recall, precision, color='#9B59B6', lw=2,
            label=f'AP = {average_precision:.3f}')
    
    ax.set_xlabel('Recall', fontweight='bold')
    ax.set_ylabel('Precision', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(loc="lower left")
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved PR curve to {save_path}")


def plot_feature_importance(
    importances: np.ndarray,
    feature_names: List[str],
    save_path: str,
    title: str = 'Feature Importance',
    top_n: int = 20
) -> None:
    """
    Plot top N feature importances.
    
    Args:
        importances: Array of feature importance scores
        feature_names: List of feature names
        save_path: Path to save figure
        title: Plot title
        top_n: Number of top features to show
    """
    set_plot_style()
    
    # Sort importances
    indices = np.argsort(importances)[::-1]
    
    # Select top N
    top_indices = indices[:top_n]
    top_importances = importances[top_indices]
    top_names = [feature_names[i] for i in top_indices]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    y_pos = np.arange(len(top_names))
    ax.barh(y_pos, top_importances, align='center', color='#34495E')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()  # labels read top-to-bottom
    
    ax.set_xlabel('Importance Score', fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', facecolor='white')
    plt.close()
    logger.info(f"Saved feature importance plot to {save_path}")

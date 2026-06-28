"""Evaluation module for metrics and visualization."""
from .metrics import (
    compute_all_metrics,
    compute_rare_class_analysis,
    format_metrics_for_logging
)
from .plots import (
    plot_confusion_matrix,
    plot_strategy_comparison,
    plot_rare_class_recall,
    plot_class_distribution,
    set_plot_style,
    plot_learning_curves,
    plot_roc_curve,
    plot_pr_curve,
    plot_feature_importance
)

__all__ = [
    'compute_all_metrics',
    'compute_rare_class_analysis',
    'format_metrics_for_logging',
    'plot_confusion_matrix',
    'plot_strategy_comparison',
    'plot_rare_class_recall',
    'plot_class_distribution',
    'set_plot_style',
    'plot_learning_curves',
    'plot_roc_curve',
    'plot_pr_curve',
    'plot_feature_importance'
]

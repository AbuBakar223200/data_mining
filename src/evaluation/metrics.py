"""
Evaluation metrics computation for UNSW-NB15 experiments.

This module provides functions to compute comprehensive metrics including:
- Overall metrics: Accuracy, Macro F1, Weighted F1, G-Mean, ROC-AUC
- Per-class metrics: Precision, Recall, F1, Support
- Rare class analysis: Special focus on Worms, Shellcode, Backdoor, Analysis
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, classification_report
)
from imblearn.metrics import geometric_mean_score
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: np.ndarray,
    task: str,
    class_names: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    Compute comprehensive metrics for a single experiment.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted class labels
        y_pred_proba: Prediction probabilities (n_samples, n_classes)
        task: 'binary' or 'multi'
        class_names: Optional list of class names for reporting
        
    Returns:
        Dictionary containing:
        - overall: Accuracy, Macro F1, Weighted F1, G-Mean, ROC-AUC
        - per_class: Precision, Recall, F1, Support per class
        - confusion_matrix: Raw confusion matrix
    """
    metrics = {
        'overall': {},
        'per_class': {},
        'confusion_matrix': None
    }
    
    # Overall metrics
    metrics['overall']['accuracy'] = float(accuracy_score(y_true, y_pred))
    metrics['overall']['macro_f1'] = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
    metrics['overall']['weighted_f1'] = float(f1_score(y_true, y_pred, average='weighted', zero_division=0))
    
    # G-Mean (primary metric for imbalanced data)
    try:
        metrics['overall']['g_mean'] = float(geometric_mean_score(y_true, y_pred, average='macro'))
    except Exception as e:
        logger.warning(f"G-Mean calculation failed: {e}")
        metrics['overall']['g_mean'] = 0.0
    
    # ROC-AUC
    try:
        if task == 'binary':
            # For binary, use probability of positive class
            if y_pred_proba.ndim == 2:
                proba = y_pred_proba[:, 1]
            else:
                proba = y_pred_proba
            metrics['overall']['roc_auc'] = float(roc_auc_score(y_true, proba))
        else:
            # For multiclass, use One-vs-Rest with macro averaging
            metrics['overall']['roc_auc'] = float(roc_auc_score(
                y_true, y_pred_proba,
                multi_class='ovr',
                average='macro'
            ))
    except ValueError as e:
        # Can fail if a class has no samples in y_true
        logger.warning(f"ROC-AUC calculation failed: {e}")
        metrics['overall']['roc_auc'] = None
    
    # Per-class metrics using classification_report
    report = classification_report(
        y_true, y_pred,
        output_dict=True,
        zero_division=0,
        target_names=class_names
    )
    
    for class_key, class_metrics in report.items():
        # Skip summary rows
        if class_key in ['accuracy', 'macro avg', 'weighted avg']:
            continue
        
        metrics['per_class'][str(class_key)] = {
            'precision': float(class_metrics['precision']),
            'recall': float(class_metrics['recall']),
            'f1': float(class_metrics['f1-score']),
            'support': int(class_metrics['support'])
        }
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    return metrics


def compute_rare_class_analysis(
    metrics: Dict[str, Any],
    rare_classes: List[str],
    label_mapping: Optional[Dict[str, int]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Extract performance metrics specifically for rare attack classes.
    
    This is a key differentiator of the study - focusing on classes
    that are typically ignored in aggregate metrics.
    
    Args:
        metrics: Full metrics dictionary from compute_all_metrics
        rare_classes: List of rare class names (e.g., ['Worms', 'Shellcode'])
        label_mapping: Optional mapping from class name to index
        
    Returns:
        Dictionary with metrics for each rare class:
        {
            'Worms': {'precision': 0.xx, 'recall': 0.xx, 'f1': 0.xx, 'support': n},
            ...
        }
    """
    rare_analysis = {}
    
    for cls in rare_classes:
        # Try to find the class in per_class metrics
        # It might be stored by name or by index
        cls_key = str(cls)
        if label_mapping and cls in label_mapping:
            cls_idx_key = str(label_mapping[cls])
        else:
            cls_idx_key = cls_key
        
        # Check both name and index keys
        if cls_key in metrics['per_class']:
            rare_analysis[cls] = metrics['per_class'][cls_key]
        elif cls_idx_key in metrics['per_class']:
            rare_analysis[cls] = metrics['per_class'][cls_idx_key]
        else:
            logger.warning(f"Rare class '{cls}' not found in per_class metrics")
            rare_analysis[cls] = {
                'precision': 0.0,
                'recall': 0.0,
                'f1': 0.0,
                'support': 0
            }
    
    return rare_analysis


def format_metrics_for_logging(metrics: Dict[str, Any], task: str) -> str:
    """
    Format metrics as a readable string for logging.
    
    Args:
        metrics: Metrics dictionary
        task: 'binary' or 'multi'
        
    Returns:
        Formatted string for logging
    """
    overall = metrics['overall']
    
    lines = [
        f"Accuracy:    {overall['accuracy']:.4f}",
        f"Macro F1:    {overall['macro_f1']:.4f}",
        f"Weighted F1: {overall['weighted_f1']:.4f}",
        f"G-Mean:      {overall['g_mean']:.4f}",
    ]
    
    if overall['roc_auc'] is not None:
        lines.append(f"ROC-AUC:     {overall['roc_auc']:.4f}")
    else:
        lines.append("ROC-AUC:     N/A")
    
    return '\n'.join(lines)


def aggregate_experiment_results(
    results: List[Dict[str, Any]]
) -> Dict[str, List[Any]]:
    """
    Aggregate results from multiple experiments into a summary table.
    
    Args:
        results: List of experiment result dictionaries
        
    Returns:
        Dictionary suitable for DataFrame conversion
    """
    aggregated = {
        'experiment_id': [],
        'task': [],
        'model': [],
        'strategy': [],
        'accuracy': [],
        'macro_f1': [],
        'weighted_f1': [],
        'g_mean': [],
        'roc_auc': [],
        'training_time': []
    }
    
    for result in results:
        aggregated['experiment_id'].append(result['experiment_id'])
        aggregated['task'].append(result['task'])
        aggregated['model'].append(result['model'])
        aggregated['strategy'].append(result['strategy'])
        aggregated['accuracy'].append(result['metrics']['overall']['accuracy'])
        aggregated['macro_f1'].append(result['metrics']['overall']['macro_f1'])
        aggregated['weighted_f1'].append(result['metrics']['overall']['weighted_f1'])
        aggregated['g_mean'].append(result['metrics']['overall']['g_mean'])
        aggregated['roc_auc'].append(result['metrics']['overall']['roc_auc'])
        aggregated['training_time'].append(result.get('training_time_seconds', 0))
    
    return aggregated


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import logging
from math import pi

logger = logging.getLogger(__name__)

class PublicationVisualizer:
    """
    Generates "Wow"-factor, publication-ready visualizations for ML papers.
    Focuses on aesthetics, clarity, and deep insight.
    """
    
    def __init__(self, style: str = 'seaborn-v0_8-whitegrid'):
        self.style = style
        plt.style.use(style)
        self._set_custom_rc()
        
        # Premium pairings
        self.colors = {
            'primary': '#2C3E50',    # Dark Blue
            'secondary': '#E74C3C',  # Red
            'accent': '#3498DB',     # Bright Blue
            'success': '#27AE60',    # Green
            'purple': '#8E44AD',     # Wisteria
            'orange': '#D35400',     # Pumpkin
        }
        
    def _set_custom_rc(self):
        """Configure matplotlib for high-DPI, professional output."""
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['Arial', 'DejaVu Sans', 'Liberation Sans'],
            'font.size': 11,
            'axes.titlesize': 14,
            'axes.titleweight': 'bold',
            'axes.labelsize': 12,
            'axes.labelweight': 'bold',
            'xtick.labelsize': 10,
            'ytick.labelsize': 10,
            'legend.fontsize': 10,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'lines.linewidth': 2.5,
            'lines.markersize': 8,
            'axes.grid': True,
            'grid.alpha': 0.3,
            'axes.spines.top': False,
            'axes.spines.right': False,
        })

    def plot_radar_chart(self, 
                        data: Dict[str, Dict[str, float]], 
                        metrics: List[str], 
                        save_path: str,
                        title: str = "Model Comparison"):
        """
        Create a radar chart comparing multiple models across metrics.
        
        Args:
            data: {model_name: {metric: value}}
            metrics: List of metrics to plot axes for
            save_path: Output file path
        """
        N = len(metrics)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Close the loop
        
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw={'projection': 'polar'})
        
        # Draw one axe per variable + labels
        plt.xticks(angles[:-1], [m.replace('_', ' ').title() for m in metrics], color='black', size=10)
        
        # Draw ylabels
        ax.set_rlabel_position(0)
        plt.yticks([0.2, 0.4, 0.6, 0.8, 1.0], ["0.2", "0.4", "0.6", "0.8", "1.0"], color="grey", size=8)
        plt.ylim(0, 1)
        
        # Plot data
        palette = sns.color_palette("husl", len(data))
        
        for idx, (label, metrics_dict) in enumerate(data.items()):
            values = [metrics_dict.get(m, 0.0) for m in metrics]
            values += values[:1]  # Close loop
            
            ax.plot(angles, values, linewidth=2, linestyle='solid', label=label, color=palette[idx])
            ax.fill(angles, values, color=palette[idx], alpha=0.1)
            
        plt.title(title, size=16, color=self.colors['primary'], y=1.1)
        plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved radar chart to {save_path}")

    def plot_critical_difference_proxy(self, 
                                     ranks: Dict[str, float], 
                                     save_path: str):
        """
        Visualizes average rank of strategies (Proxy for Critical Difference Diagram).
        Real CD diagrams require statistical tests, this creates a beautiful rank comparison.
        """
        sorted_ranks = sorted(ranks.items(), key=lambda item: item[1])
        names = [x[0] for x in sorted_ranks]
        values = [x[1] for x in sorted_ranks]
        
        fig, ax = plt.subplots(figsize=(10, 4))
        
        # Create a horizontal line
        ax.hlines(y=1, xmin=0.5, xmax=len(ranks)+0.5, color='gray', alpha=0.5, linewidth=2)
        
        # Plot points on the line
        # We normalize ranks to map onto the line nicely, actually let's just do a bar chart rank
        # A simple bar chart is clearer if we aren't doing the Nemenyi test cliques
        plt.close()
        
        # Fallback to horizontal bar chart for ranks (Lower is better)
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(names, values, color=self.colors['accent'])
        
        ax.set_xlabel('Average Rank (Lower is Better)')
        ax.set_title("Strategy Ranking Comparison", fontsize=14)
        ax.invert_yaxis()  # Best on top
        
        # Value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(width + 0.05, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left', va='center')
            
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved rank comparison to {save_path}")

    def plot_rare_class_heatmap_grid(self, 
                                     df_results: pd.DataFrame, 
                                     rare_classes: List[str],
                                     save_path: str):
        """
        Creates a faceted heatmap specifically for rare classes.
        Input DF must have columns: model, strategy, [class_names...]
        """
        # Pivot functionality would be specific to how we parse the CSV.
        # This is a placeholder for the logic in generate_report.py
        pass
        
    def plot_confusion_matrix_styled(self, cm, classes, save_path, title='Confusion Matrix'):
        """
        Premium styled confusion matrix.
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Normalize
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(cm_norm, annot=True, fmt='.1%', cmap='RdPu', 
                    xticklabels=classes, yticklabels=classes,
                    cbar=True, square=True, linewidths=.5, linecolor='white')
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.title(title, pad=20)
        
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved styled CM to {save_path}")

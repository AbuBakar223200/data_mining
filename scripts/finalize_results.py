#!/usr/bin/env python3
"""
Finalize Results Tables
=======================

Generates the specific artifacts required by the Experiment Contract:
1. `results/tables/final_summary_tables.csv`
2. `results/tables/rare_class_report.csv`

It reads from `results/metrics/*.json` to ensure full fidelity.
"""

import sys
import os
import json
import logging
import pandas as pd
from pathlib import Path
from glob import glob

# Add project root to path
sys.path.append(os.getcwd())
from src.utils import load_config, setup_logging

logger = logging.getLogger(__name__)

def main():
    config = load_config("configs/main.yaml")
    results_dir = Path(config['results_dir'])
    setup_logging(level="INFO", log_file=str(results_dir / 'logs' / 'finalize.log'))
    
    # Load all metrics JSONs
    json_files = glob(str(results_dir / 'metrics' / '*.json'))
    logger.info(f"Found {len(json_files)} metric files.")
    
    all_results = []
    rare_class_rows = []
    
    rare_classes = ['Worms', 'Shellcode', 'Backdoor', 'Analysis']
    
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            
        # Overall Summary
        row = {
            'Experiment_ID': data.get('experiment_id'),
            'Task': data.get('task'),
            'Model': data.get('model'),
            'Strategy': data.get('strategy'),
            'Seed': data.get('seed', 0), # Baselines might not have seed
            'Accuracy': data['metrics']['overall']['accuracy'],
            'Macro_F1': data['metrics']['overall']['macro_f1'],
            'G_Mean': data['metrics']['overall']['g_mean'],
            'ROC_AUC': data['metrics']['overall']['roc_auc'],
            'Training_Time': data.get('training_time_seconds', 0)
        }
        all_results.append(row)
        
        # Rare Class Analysis
        if data.get('task') == 'multi':
            # Check if rare_class_analysis exists (baselines need manual extraction)
            rc_analysis = data.get('rare_class_analysis')
            
            # If missing (e.g. baseline), try to extract from per_class
            if not rc_analysis and 'per_class' in data['metrics']:
                rc_analysis = {}
                for rc in rare_classes:
                    if rc in data['metrics']['per_class']:
                        rc_analysis[rc] = data['metrics']['per_class'][rc]
                    else:
                         rc_analysis[rc] = {'precision': 0, 'recall': 0, 'f1': 0, 'support': 0}
            
            if rc_analysis:
                for cls_name, metrics in rc_analysis.items():
                    if cls_name in rare_classes:
                        r_row = {
                            'Experiment_ID': data.get('experiment_id'),
                            'Model': data.get('model'),
                            'Strategy': data.get('strategy'),
                            'Class': cls_name,
                            'Precision': metrics['precision'],
                            'Recall': metrics['recall'],
                            'F1': metrics['f1'],
                            'Support': metrics['support']
                        }
                        rare_class_rows.append(r_row)

    # 1. Final Summary Table
    df_summary = pd.DataFrame(all_results)
    # Sort for readability
    if not df_summary.empty:
        df_summary.sort_values(by=['Task', 'Model', 'Strategy'], inplace=True)
        out_summary = results_dir / 'tables' / 'final_summary_tables.csv'
        df_summary.to_csv(out_summary, index=False)
        logger.info(f"Saved {out_summary}")
    
    # 2. Rare Class Report
    if rare_class_rows:
        df_rare = pd.DataFrame(rare_class_rows)
        df_rare.sort_values(by=['Class', 'Model', 'Strategy'], inplace=True)
        
        out_rare = results_dir / 'tables' / 'rare_class_report.csv'
        df_rare.to_csv(out_rare, index=False)
        logger.info(f"Saved {out_rare}")
    else:
        logger.warning("No rare class data found!")

if __name__ == "__main__":
    main()

import pandas as pd
from pathlib import Path
import shutil
import os

def main():
    src_dir = Path("results/single_seed/tables")
    dest_dir = Path("results/tables")
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Final Summary Tables (Aggregated)
    if (src_dir / 'aggregated_summary.csv').exists():
        df = pd.read_csv(src_dir / 'aggregated_summary.csv')
        # Rename columns to remove _mean suffix if present
        rename_map = {c: c.replace('_mean', '') for c in df.columns if '_mean' in c}
        df = df.rename(columns=rename_map)
        df.to_csv(dest_dir / 'final_summary_tables.csv', index=False)
        print(f"Created {dest_dir / 'final_summary_tables.csv'}")
        
    # 2. Rare Class Report
    if (src_dir / 'rare_class_aggregated.csv').exists():
        df = pd.read_csv(src_dir / 'rare_class_aggregated.csv')
        # Check if column names need mapping. Usually 'recall_mean' -> 'Recall'
        rename_map = {c: c.replace('_mean', '').capitalize() for c in df.columns if '_mean' in c}
        # Also ensure 'class' -> 'Class' etc
        rename_map.update({c: c.capitalize() for c in df.columns if c.lower() in ['class', 'model', 'strategy']})
        df = df.rename(columns=rename_map)
        df.to_csv(dest_dir / 'rare_class_report.csv', index=False)
        print(f"Created {dest_dir / 'rare_class_report.csv'}")
       
    # 3. Per Class Metrics
    # We ideally want the full per-class metrics.
    # If per_class_metrics_dump.csv exists (from my stats script), use it or standard one.
    # generate_publication_figures uses per_class_metrics.csv for Heatmap.
    # The stats script made per_class_metrics_dump.csv with columns: Class, Treatment, F1.
    # The plotter expects: 'Model', 'Strategy', 'Class', 'F1'.
    
    if (dest_dir / 'per_class_metrics_dump.csv').exists():
        df = pd.read_csv(dest_dir / 'per_class_metrics_dump.csv')
        # Parse Treatment -> Model, Strategy
        # Treatment is "MODEL_STRATEGY" (e.g. LR_S0)
        def parse_treatment(t):
            parts = t.split('_')
            return parts[0], parts[1]
            
        df[['Model', 'Strategy']] = df['Treatment'].apply(lambda x: pd.Series(parse_treatment(x)))
        df.to_csv(dest_dir / 'per_class_metrics.csv', index=False)
        print(f"Created {dest_dir / 'per_class_metrics.csv'}")
        
    # 4. Logs (copy if needed)
    # The plotter loads results/experiment_log.csv.
    # Ensure it exists.
    if not Path("results/experiment_log.csv").exists():
        # Try to find one
        logs = list(Path("results/single_seed").glob("experiment_log*.csv"))
        if logs:
            shutil.copy(logs[0], "results/experiment_log.csv")
            print(f"Copied {logs[0]} to results/experiment_log.csv")

if __name__ == "__main__":
    main()


import json
import csv
import os
import glob
import pandas as pd
from pathlib import Path

RESULTS_DIR = r"e:\University\ML\ML_PAPER_REVIEW\results"
METRICS_DIR = os.path.join(RESULTS_DIR, "metrics")
LOG_FILE = os.path.join(RESULTS_DIR, "experiment_log.csv")

def verify_artifacts():
    print("Starting Comprehensive Artifact Verification...")
    
    # 1. Load the master log
    if not os.path.exists(LOG_FILE):
        print(f"CRITICAL: Master log not found at {LOG_FILE}")
        return
    
    df_log = pd.read_csv(LOG_FILE)
    print(f"Loaded experiment_log.csv with {len(df_log)} entries.")
    
    # 2. Scan all JSON metrics
    json_files = glob.glob(os.path.join(METRICS_DIR, "*.json"))
    print(f"Found {len(json_files)} JSON metric files.")
    
    inconsistencies = []
    verified_count = 0
    
    for json_path in json_files:
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            exp_id = data.get('experiment_id')
            if not exp_id:
                inconsistencies.append(f"{os.path.basename(json_path)}: Missing experiment_id")
                continue
                
            # Find matching row in log
            log_row = df_log[df_log['experiment_id'] == exp_id]
            
            if log_row.empty:
                inconsistencies.append(f"{exp_id}: Found in JSON but missing from experiment_log.csv")
                continue
            
            # Cross-verify key metrics (JSON vs CSV)
            # Allow for small floating point differences
            json_f1 = data['metrics']['overall']['macro_f1']
            csv_f1 = log_row.iloc[0]['macro_f1']
            
            if abs(json_f1 - csv_f1) > 0.0001:
                inconsistencies.append(f"{exp_id}: F1 Mismatch (JSON={json_f1:.4f}, CSV={csv_f1:.4f})")
            
            json_gmean = data['metrics']['overall']['g_mean']
            csv_gmean = log_row.iloc[0]['g_mean']
            
            if abs(json_gmean - csv_gmean) > 0.0001:
                 inconsistencies.append(f"{exp_id}: G-Mean Mismatch (JSON={json_gmean:.4f}, CSV={csv_gmean:.4f})")

            # Verify Data Contract (Train/Test split sizes)
            # Standard UNSW-NB15 split: Train ~175k (but here we used a subset or split?), Test ~82k
            # data content says: train_samples": 140272, "test_samples": 82332
            # Let's verify this consistency across all files
            if data['test_samples'] != 82332:
                 inconsistencies.append(f"{exp_id}: Unexpected test set size {data['test_samples']} (Expected 82332)")
            
            verified_count += 1
            
        except Exception as e:
            inconsistencies.append(f"{os.path.basename(json_path)}: Error reading/parsing - {str(e)}")

    print(f"\nVerification Complete.")
    print(f"Verified {verified_count} experiments successfully against the log.")
    
    if inconsistencies:
        print("\n!!! INCONSISTENCIES FOUND !!!")
        for issue in inconsistencies:
            print(f" - {issue}")
    else:
        print("\n✅ Data Integrity Confirmed: All JSON metrics match the Experiment Log exactly.")
        print("✅ Test Set Size Consistency: All experiments used exactly 82,332 test samples.")

if __name__ == "__main__":
    verify_artifacts()

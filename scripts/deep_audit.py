
import os
import glob
from pathlib import Path
import json

RESULTS_DIR = Path("results")
REQUIRED_TASKS = ['binary', 'multi']
REQUIRED_MODELS = ['lr', 'rf', 'xgb']
REQUIRED_STRATEGIES = ['s0', 's1', 's2a', 's2b']
REQUIRED_SEEDS = [42, 43, 44, 45, 46]

def audit_all():
    print(f"Scanning {RESULTS_DIR.absolute()} recursively...")
    all_json_files = glob.glob(str(RESULTS_DIR / "**/*.json"), recursive=True)
    
    found_experiments = {} # exp_id -> [list of paths]
    
    for fpath in all_json_files:
        path = Path(fpath)
        # simplistic filtering for metrics files
        if "metrics" in str(path) or "learning_curves" not in str(path):
            # Check if filename matches our grid pattern
            name = path.stem
            # Verify pattern broadly
            if any(t in name for t in REQUIRED_TASKS):
                 if name not in found_experiments:
                     found_experiments[name] = []
                 found_experiments[name].append(str(path))

    print(f"\nFound {len(found_experiments)} unique experiment IDs across all folders.")
    
    # Check against Grid
    found_ids = set()
    grid_total = 0
    missing = []
    
    for task in REQUIRED_TASKS:
        for model in REQUIRED_MODELS:
            for strategy in REQUIRED_STRATEGIES:
                for seed in REQUIRED_SEEDS:
                    grid_total += 1
                    exp_id = f"{task}_{model}_{strategy}_s{seed}"
                    if exp_id in found_experiments:
                        found_ids.add(exp_id)
                    else:
                        missing.append(exp_id)
                        
    print(f"Grid Coverage: {len(found_ids)}/{grid_total}")
    
    if len(found_ids) < grid_total:
        print(f"MISSING ({len(missing)}):")
        for m in missing[:10]:
            print(f" - {m}")
        if len(missing)>10: print("...")

    # Duplicates Analysis
    duplicates = {k:v for k,v in found_experiments.items() if len(v) > 1}
    if duplicates:
        print(f"\nDuplicate/Redundant Results Found: {len(duplicates)}")
        for k, v in list(duplicates.items())[:5]:
            print(f" - {k}:")
            for p in v:
                print(f"   -> {p}")

    # Generate Python List for Rescue
    print("\n--- RESCUE DATA ---")
    print(sorted(list(found_ids)))

if __name__ == "__main__":
    audit_all()

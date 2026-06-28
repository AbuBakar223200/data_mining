import os
import shutil
import json
from pathlib import Path

# Paths
BASE_DIR = Path(r"e:\University\ML\ML_PAPER_REVIEW")
COLAB_DIR = BASE_DIR / "results/results-from-colab-drive/run_20260123_072608"
DEST_DIR = BASE_DIR / "results"

TASKS = ['binary', 'multi']
MODELS = ['lr', 'rf', 'xgb']
STRATEGIES = ['s0', 's1', 's2a', 's2b']
SEEDS = [42, 43, 44, 45, 46]

def merge_folders(src, dst):
    if not src.exists():
        print(f"Source not found: {src}")
        return 0
    
    count = 0
    dst.mkdir(parents=True, exist_ok=True)
    for item in src.glob("*"):
        if item.is_file():
            dst_file = dst / item.name
            if not dst_file.exists():
                shutil.copy2(item, dst_file)
                count += 1
                print(f"Copied: {item.name}")
            else:
                print(f"Skipped (Exists): {item.name}")
    return count

def audit_experiments():
    metrics_dir = DEST_DIR / "metrics"
    if not metrics_dir.exists():
        print("No metrics directory found.")
        return

    expected_total = len(TASKS) * len(MODELS) * len(STRATEGIES) * len(SEEDS)
    experiments = set()
    for f in metrics_dir.glob("*.json"):
        # filenames are like: binary_xgb_s2a_s42.json
        experiments.add(f.stem)

    print(f"\nTotal Experiments Found: {len(experiments)} / {expected_total}")

    # Check for missing
    missing = []
    for t in TASKS:
        for m in MODELS:
            for s in STRATEGIES:
                for seed in SEEDS:
                    exp_id = f"{t}_{m}_{s}_s{seed}"
                    if exp_id not in experiments:
                        missing.append(exp_id)
    
    if missing:
        print(f"Missing {len(missing)} experiments:")
        for m in missing[:10]:
            print(f" - {m}")
        if len(missing) > 10:
            print(f" ... and {len(missing)-10} more.")
    else:
        print("ALL EXPERIMENTS COMPLETE!")

if __name__ == "__main__":
    print("--- Merging Metrics ---")
    m_count = merge_folders(COLAB_DIR / "metrics", DEST_DIR / "metrics")
    print(f"Merged {m_count} metric files.")
    
    print("\n--- Merging Figures ---")
    f_count = merge_folders(COLAB_DIR / "figures", DEST_DIR / "figures")
    print(f"Merged {f_count} figure files.")

    print("\n--- Auditing Results ---")
    audit_experiments()

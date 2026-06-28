import os
import shutil
from pathlib import Path

# Paths
BASE_DIR = Path(r"e:\University\ML\ML_PAPER_REVIEW")
SOURCE_ROOT = BASE_DIR / "results/results-from-colab-drive"
DEST_METRICS = BASE_DIR / "results/metrics"
DEST_FIGURES = BASE_DIR / "results/figures"

TASKS = ['binary', 'multi']
MODELS = ['lr', 'rf', 'xgb']
STRATEGIES = ['s0', 's1', 's2a', 's2b']
SEEDS = [42, 43, 44, 45, 46]

def consolidate():
    print(f"Scanning {SOURCE_ROOT} recursively...")
    
    unique_metrics = set()
    copied_metrics = 0
    copied_figures = 0
    
    # Ensure destinations exist
    DEST_METRICS.mkdir(parents=True, exist_ok=True)
    DEST_FIGURES.mkdir(parents=True, exist_ok=True)
    
    # Recursive walk
    for root, dirs, files in os.walk(SOURCE_ROOT):
        root_path = Path(root)
        
        for file in files:
            # Check for Metrics
            if file.endswith(".json") and "learning_curves" not in str(root_path): # simplistic check
                # Verify it looks like a metric file (e.g. contains task_model_strategy)
                if any(x in file for x in ['binary_', 'multi_']):
                    src_file = root_path / file
                    dest_file = DEST_METRICS / file
                    
                    # Size check to avoid copying empty/corrupt files
                    if src_file.stat().st_size > 100:
                        if not dest_file.exists() or dest_file.stat().st_size < src_file.stat().st_size:
                            shutil.copy2(src_file, dest_file)
                            copied_metrics += 1
                        unique_metrics.add(file)

            # Check for Figures
            if file.endswith(".png"):
                 if "cm_" in file or "roc_" in file or "pr_" in file:
                    src_file = root_path / file
                    dest_file = DEST_FIGURES / file
                    if not dest_file.exists():
                        shutil.copy2(src_file, dest_file)
                        copied_figures += 1

    print(f"--- Consolidation Report ---")
    print(f"Unique Metrics Found: {len(unique_metrics)}")
    print(f"New Metrics Copied: {copied_metrics}")
    print(f"New Figures Copied: {copied_figures}")
    
    return unique_metrics

def audit(found_files):
    print("\n--- Final Gap Analysis ---")
    expected_total = len(TASKS) * len(MODELS) * len(STRATEGIES) * len(SEEDS)
    
    missing = []
    for t in TASKS:
        for m in MODELS:
            for s in STRATEGIES:
                for seed in SEEDS:
                    exp_id = f"{t}_{m}_{s}_s{seed}.json"
                    if exp_id not in found_files:
                        missing.append(exp_id)
    
    if missing:
        print(f"STILL MISSING ({len(missing)}/{expected_total}):")
        for m in missing[:10]:
            print(f" - {m}")
        if len(missing) > 10:
            print(f" ... and {len(missing)-10} more.")
    else:
        print(f"✅ 100% COMPLETION! All {expected_total}/{expected_total} experiments found.")

if __name__ == "__main__":
    found = consolidate()
    audit(found)

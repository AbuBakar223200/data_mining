#!/usr/bin/env python3
"""
Full Grid Runner (Optimized & Forced)
=====================================

Runs the complete 90-experiment grid from scratch.

Features:
1.  **Optimized Execution**: Temporarily sets `n_jobs: 1` to run experiments sequentially.
    -   This prevents "Fork Bomb" thrashing on Colab (where 20+ threads fight for 2 cores).
    -   Allows XGBoost/RF to use 100% of the CPU/GPU per run.
2.  **Force Run**: Clears `results/metrics` to ensure every experiment runs from zero.

Usage:
    python run_full_grid.py
"""

import os
import sys
import yaml
import shutil
import subprocess
from pathlib import Path

def main():
    print("🚀 [run_full_grid.py] Initializing Full System Run...")
    
    # ---------------------------------------------------------
    # 1. OPTIMIZE CONFIGURATION (n_jobs = 1)
    # ---------------------------------------------------------
    config_path = "configs/main.yaml"
    temp_config_path = "configs/temp_optimized_full_run.yaml"
    
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
    except Exception as e:
        print(f"❌ Error reading {config_path}: {e}")
        sys.exit(1)

    original_jobs = config.get('experiments', {}).get('n_jobs', 'Unknown')
    print(f"   ℹ️  Original configuration: n_jobs={original_jobs}")
    
    # Force sequential execution for the PIPELINE, but allow models to use full power
    # This is the key fix for the "5 hour" runtime.
    config['experiments']['n_jobs'] = 1 
    print(f"   ⚡ Optimization applied:   n_jobs=1 (Sequential Pipeline -> Max Model Performance)")
    
    try:
        with open(temp_config_path, "w") as f:
            yaml.dump(config, f)
    except Exception as e:
        print(f"❌ Error writing temp config: {e}")
        sys.exit(1)

    # ---------------------------------------------------------
    # 2. FORCE CLEAN STATE
    # ---------------------------------------------------------
    results_dir = Path(config.get("results_dir", "results"))
    metrics_dir = results_dir / "metrics"
    logs_dir = results_dir / "logs"
    
    print(f"   🧹 Cleaning previous results in {results_dir}...")
    
    # Clean metrics to force Main.py to re-run everything
    if metrics_dir.exists():
        try:
            shutil.rmtree(metrics_dir)
            print(f"      - {metrics_dir} cleared.")
        except Exception as e:
            print(f"      ⚠️  Warning: Could not clear {metrics_dir}: {e}")

    # ---------------------------------------------------------
    # 3. EXECUTE MAIN PIPELINE
    # ---------------------------------------------------------
    print("=" * 60)
    print("   STARTING SUBPROCESS: main.py")
    print("=" * 60)
    
    cmd = [sys.executable, "main.py", "--config", temp_config_path]
    
    exit_code = 0
    try:
        # Stream output to console
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Pipeline failed with exit code {e.returncode}")
        exit_code = e.returncode
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user.")
        exit_code = 130
    finally:
        # ---------------------------------------------------------
        # 4. CLEANUP
        # ---------------------------------------------------------
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)
            print(f"\n   🗑️  Cleaned up temp config: {temp_config_path}")

    sys.exit(exit_code)

if __name__ == "__main__":
    main()

import sys
import os
sys.path.append(os.getcwd())
import json
from pathlib import Path
import pandas as pd
from src.utils import load_config
from main import generate_summary_tables

def main():
    config = load_config('configs/main.yaml')
    results_dir = Path('results')
    metrics_dir = results_dir / 'metrics'
    
    results = []
    for p in metrics_dir.glob('*.json'):
        with open(p, 'r') as f:
            results.append(json.load(f))
            
    print(f"Loaded {len(results)} metrics files.")
    generate_summary_tables(results, results_dir, config)
    print("Tables generated in results/tables/")

if __name__ == "__main__":
    main()

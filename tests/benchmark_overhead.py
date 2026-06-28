
import time
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

def benchmark_rf():
    print("Generating synthetic data (approx UNSW-NB15 size)...")
    # UNSW-NB15 Train is ~175k rows, ~40 cols effective numericals
    X, y = make_classification(n_samples=50000, n_features=40, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    
    print("\n--- Sceario A: Current Baseline (Batch Fit) ---")
    start = time.time()
    rf_batch = RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
    rf_batch.fit(X_train, y_train)
    batch_time = time.time() - start
    print(f"Batch Training Time: {batch_time:.2f}s")
    
    print("\n--- Scenario B: Proposed Iterative (Step=1) ---")
    start = time.time()
    rf_iter = RandomForestClassifier(n_estimators=0, warm_start=True, n_jobs=-1, random_state=42)
    for i in range(100):
        rf_iter.n_estimators += 1
        rf_iter.fit(X_train, y_train)
        # Simulate scoring overhead
        _ = rf_iter.score(X_val, y_val)
    iter_step1_time = time.time() - start
    print(f"Iterative (Step=1) Time: {iter_step1_time:.2f}s")
    print(f"Overhead Factor: {iter_step1_time / batch_time:.1f}x")

    print("\n--- Scenario C: Optimized Iterative (Step=10) ---")
    start = time.time()
    rf_iter_opt = RandomForestClassifier(n_estimators=0, warm_start=True, n_jobs=-1, random_state=42)
    for i in range(10): # 10 steps of 10 trees = 100 trees
        rf_iter_opt.n_estimators += 10
        rf_iter_opt.fit(X_train, y_train)
        _ = rf_iter_opt.score(X_val, y_val)
    iter_step10_time = time.time() - start
    print(f"Iterative (Step=10) Time: {iter_step10_time:.2f}s")
    print(f"Overhead Factor: {iter_step10_time / batch_time:.1f}x")

if __name__ == "__main__":
    benchmark_rf()

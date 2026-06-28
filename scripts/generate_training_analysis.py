
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import os
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.metrics import make_scorer, f1_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# Import local modules
import sys
sys.path.append('.')
from src.data.loader import DataLoader
from src.data.preprocessing import UNSWPreprocessor
from src.strategies.imbalance import get_strategy

# ==========================================
# 1. CONFIGURATION & STANDARDS
# ==========================================
OUTPUT_DIR = 'results/figures/training_analysis'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Styles
plt.rcParams['font.family'] = 'DejaVu Sans'
COLORS = {
    'train': '#2E86AB', # Blue
    'cv': '#E63946',    # Red
    's0': '#6C757D',
    's2a': '#38B000'
}

# Simplified Config for Data Loading
CONFIG = {
    'data': {
        'train_path': 'dataset/UNSW_NB15_training-set.csv',
        'test_path': 'dataset/UNSW_NB15_testing-set.csv',
        'drop_columns': ['id'],
        'categorical_columns': ['proto', 'service', 'state'],
        'target_binary': 'label',
        'target_multiclass': 'attack_cat'
    },
    'preprocessing': {
        'test_size': 0.2, # 20% validation split
        'random_state': 42
    }
}

# ==========================================
# 2. PLOTTING FUNCTIONS
# ==========================================

def plot_learning_curve_graph(estimator, X, y, title, filename):
    """
    Generate learning curve (Sample Size vs Score).
    Diagnoses Bias vs Variance.
    """
    print(f"Generating Learning Curve: {title}...")
    
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=2, n_jobs=-1, 
        train_sizes=np.linspace(0.1, 1.0, 3),
        scoring='f1_macro', shuffle=True, random_state=42
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot Training
    ax.plot(train_sizes, train_mean, 'o-', color=COLORS['train'], label="Training Score")
    ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1, color=COLORS['train'])
    
    # Plot CV
    ax.plot(train_sizes, test_mean, 'o-', color=COLORS['cv'], label="CV Score")
    ax.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, alpha=0.1, color=COLORS['cv'])
    
    ax.set_title(f"Learning Curve: {title}", fontsize=16, fontweight='bold')
    ax.set_xlabel("Training Examples", fontsize=14)
    ax.set_ylabel("Macro-F1 Score", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle='--')
    
    path = os.path.join(OUTPUT_DIR, f"learning_curve_{filename}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")

def plot_validation_curve_graph(estimator, X, y, param_name, param_range, title, filename, log_x=False):
    """
    Generate validation curve (Hyperparam vs Score).
    Diagnoses Overfitting vs Underfitting.
    """
    print(f"Generating Validation Curve: {title} ({param_name})...")
    
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=2, scoring='f1_macro', n_jobs=-1
    )
    
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if log_x:
        ax.set_xscale('log')
        
    ax.plot(param_range, train_mean, label="Training Score", color=COLORS['train'], marker='o')
    ax.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1, color=COLORS['train'])
    
    ax.plot(param_range, test_mean, label="CV Score", color=COLORS['cv'], marker='o')
    ax.fill_between(param_range, test_mean - test_std, test_mean + test_std, alpha=0.1, color=COLORS['cv'])
    
    ax.set_title(f"Validation Curve: {title}", fontsize=16, fontweight='bold')
    ax.set_xlabel(f"Parameter: {param_name}", fontsize=14)
    ax.set_ylabel("Macro-F1 Score", fontsize=14)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3, linestyle='--')
    
    path = os.path.join(OUTPUT_DIR, f"validation_curve_{filename}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")

def plot_xgb_history_graph(X_train, y_train, X_val, y_val, title, filename):
    """
    Plot XGBoost LogLoss over epochs.
    Diagnoses Convergence.
    """
    print(f"Generating XGB History: {title}...")
    
    model = XGBClassifier(
        n_estimators=300, 
        learning_rate=0.1, 
        max_depth=6, 
        use_label_encoder=False,
        eval_metric=['logloss', 'error'],
        n_jobs=-1,
        random_state=42
    )
    
    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False
    )
    
    results = model.evals_result()
    epochs = len(results['validation_0']['logloss'])
    x_axis = range(0, epochs)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(x_axis, results['validation_0']['logloss'], label='Train LogLoss', color=COLORS['train'])
    ax.plot(x_axis, results['validation_1']['logloss'], label='Val LogLoss', color=COLORS['cv'])
    
    ax.set_title(f"Training History: {title}", fontsize=16, fontweight='bold')
    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel("Log Loss", fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(OUTPUT_DIR, f"xgb_history_{filename}.png")
    fig.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"Saved: {path}")

# ==========================================
# 3. MAIN EXECUTION
# ==========================================

def main():
    # 1. Load Data
    print("Loading Data...")
    loader = DataLoader(CONFIG)
    df_train = pd.read_csv(loader.train_path)
    df_test = pd.read_csv(loader.test_path) # Not used for training analysis usually, but good to have
    
    # 2. Preprocess
    print("Preprocessing...")
    preprocessor = UNSWPreprocessor(CONFIG)
    preprocessor.fit_transform(df_train, df_test)
    
    X_train = preprocessor.X_train
    X_val = preprocessor.X_val
    
    # Use pre-computed binary labels
    y_train_bin = preprocessor.y_train_binary
    y_val_bin = preprocessor.y_val_binary
    
    print(f"Data Loaded: Train={X_train.shape}, Val={X_val.shape}")
    
    # ---------------------------------------------------------
    # EXPERIMENT 1: RF S0 (Baseline) - Learning Curve
    # ---------------------------------------------------------
    # We use concatenated train+val for cross-validation based curves 
    # to let sklearn handle splitting
    X_full = np.vstack((X_train, X_val))
    y_full_bin = np.concatenate((y_train_bin, y_val_bin))
    
    # Subsample for speed if needed (taking 50% random sample)
    idx = np.random.choice(len(X_full), int(len(X_full)*0.5), replace=False)
    X_sub = X_full[idx]
    y_sub = y_full_bin[idx]
    
    rf = RandomForestClassifier(n_estimators=50, max_depth=25, random_state=42, n_jobs=-1)
    plot_learning_curve_graph(rf, X_sub, y_sub, "RF Binary Baseline (S0)", "rf_s0_learning_curve")
    
    # ---------------------------------------------------------
    # EXPERIMENT 2: RF Complexity (Validation Curve)
    # ---------------------------------------------------------
    # Tune max_depth
    param_range = [5, 10, 25]
    plot_validation_curve_graph(rf, X_sub, y_sub, "max_depth", param_range, 
                               "RF Complexity (Max Depth)", "rf_complexity_depth")
    
    # ---------------------------------------------------------
    # EXPERIMENT 3: XGB History (Convergence)
    # ---------------------------------------------------------
    plot_xgb_history_graph(X_train, y_train_bin, X_val, y_val_bin, "XGB Binary Baseline", "xgb_history")
    
    # ---------------------------------------------------------
    # EXPERIMENT 4: S2a (Resampling) Effect
    # ---------------------------------------------------------
    print("Applying S2a Resampling...")
    strategy = get_strategy('s2a', random_state=42)
    X_res, y_res_bin = strategy.apply(X_train, y_train_bin)
    
    # Plot Learning Curve on Resampled Data
    # Note: Using resampled data for CV is tricky (leakage inside fold). 
    # But for "Training Phase" analysis, we want to see if the model *fits* the resampled data well.
    # Ideally, we should use a Pipeline with resampling, but for this quick viz:
    # We will just verify if the model can learn the resampled distribution.
    
    # Actually, standard practice: fit on resampled, validate on original.
    # sklearn learning_curve does CV split -> fit -> score.
    # If we pass X_res, it validates on resampled data (Optimistic bias).
    # Correct way: Use imblearn Pipeline. 
    
    from imblearn.pipeline import Pipeline as ImbPipeline
    from imblearn.over_sampling import RandomOverSampler
    
    pipe_s2a = ImbPipeline([
        ('sampling', RandomOverSampler(random_state=42)),
        ('clf', RandomForestClassifier(n_estimators=50, max_depth=25, random_state=42, n_jobs=-1))
    ])
    
    # Pass ORIGINAL data to learning_curve, the pipeline handles resampling internally per fold
    plot_learning_curve_graph(pipe_s2a, X_sub, y_sub, "RF Binary with S2a (ROS)", "rf_s2a_learning_curve")
    
    print("Done! All training analysis plots generated.")

if __name__ == "__main__":
    main()

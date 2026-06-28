
# Handling Class Imbalance in UNSW-NB15: A Reproducible Baseline

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Data](https://img.shields.io/badge/dataset-UNSW--NB15-orange)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ihmorol/unsw-nb15-handling-binary-multiclass-ids/blob/main/UNSW_NB15_Full_Grid.ipynb)

> **Baseline for Intrusion Detection Systems (IDS)**
> A rigorous, reproducible study evaluating the impact of class imbalance strategies (Class Weighting, Random OverSampling, SMOTE) on Binary and Multiclass IDS performance.

---

## Key Features

*   **Rigorous 18-Experiment Grid**: Systematic evaluation of 2 Tasks (Binary/Multi) × 3 Models (LR, RF, XGB) × 3 Strategies (S0, S1, S2a).
*   **Rare Class Focus**: Explicit analysis of critical minority classes (**Worms**: 0.07%, **Shellcode**: 0.65%), moving beyond misleading "accuracy" metrics.
*   **Leakage-Proof Pipeline**: Strict separation of training/validation/test splits with preprocessing fit *only* on training data.
*   **Publication-Quality Visualizations**: Automatic generation of Radar Charts, Critical Difference proxies, and Faceted Heatmaps.
*   **Reproducible**: Guaranteed reproducibility with fixed seeds, exact config snapshots, and Docker-ready structure.

---

## Repository Structure

```
├── configs/               # Configuration files
│   ├── main.yaml          #    Master experiment configuration
│   ├── fast.yaml          #    Fast configuration for quick testing
│   └── smoke_test.yaml    #    Minimal configuration for validation
├── docs/                  # Documentation (MkDocs)
│   ├── getting-started/   #    Installation and setup guides
│   ├── experiments/       #    Experiment methodology and reproducibility
│   ├── research/          #    Research findings and methodology
│   └── contracts/         #    Data and experiment contracts
├── notebooks/             # Jupyter notebooks (legacy)
│   └── UNSW_NB15_Full_Grid.ipynb  # Legacy Colab notebook (use root version)
├── reports/               # Generated reports and findings
│   ├── final_results.md   #    Executive summary with key findings
│   ├── final_review.md    #    Comprehensive experimental review
│   └── strategic_audit_report.md    # Project audit results
├── results/               # Experiment outputs
│   ├── tables/            #    CSV tables with aggregated results
│   ├── figures/           #    Generated plots and visualizations
│   └── metrics/           #    Raw JSON metrics per experiment
├── scripts/               # Execution scripts
│   ├── colab_full_grid.py #    Full experiment grid runner for Colab
│   └── colab_notebook_cell.py     # Colab cell execution helper
├── UNSW_NB15_Full_Grid.ipynb  # Complete Colab notebook (root)
├── src/                   # Source code modules
│   ├── data/              #    Data loading and preprocessing
│   │   ├── loader.py      #    Dataset loading utilities
│   │   └── preprocessing.py # Data cleaning and feature engineering
│   ├── models/            #    Model definitions and training
│   │   ├── config.py      #    Model hyperparameters
│   │   └── trainer.py     #    Training orchestration
│   ├── evaluation/        #    Metrics and visualization
│   │   ├── metrics.py     #    Performance metrics calculation
│   │   ├── plots.py       #    Plotting utilities
│   │   └── visualizer.py  #    Advanced visualizations
│   ├── strategies/        #    Imbalance handling strategies
│   ├── utils/             #    Utilities and helpers
│   │   ├── config.py      #    Configuration management
│   │   └── logging.py     #    Logging setup
│   ├── tests/             #    Unit tests and validation
│   └── visualization/     #    Additional visualization tools
├── tests/                 # Integration tests
│   └── benchmark_overhead.py     # Performance benchmarking
├── .opencode/             # AI-generated workflow documentation
├── requirements.txt       # Python dependencies
├── main.py                # Main experiment orchestrator
├── task.md                # Project task checklist
├── implementation_plan.md # Implementation roadmap
├── prompts.md             # AI interaction prompts
├── pyrightconfig.json     # Python type checking config
├── mkdocs.yml             # Documentation configuration
└── README.md              # This file
```

---

## Installation

### Prerequisites
- Python 3.8+
- pip package manager
- (Optional) Google Colab account for cloud execution

### Local Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/ihmorol/unsw-nb15-handling-binary-multiclass-ids.git
   cd ML_PAPER_REVIEW
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download the UNSW-NB15 dataset** (not included in repo due to size):
   - Visit the [UNSW-NB15 dataset page](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
   - Download `UNSW_NB15_training-set.csv` and `UNSW_NB15_testing-set.csv`
   - Create a `dataset/` directory and place the files there

### Key Dependencies
- **pandas** (2.0.0+): Data manipulation and analysis
- **scikit-learn** (1.3.0+): Machine learning algorithms and metrics
- **xgboost** (2.0.0+): Gradient boosting implementation
- **imbalanced-learn** (0.11.0+): Imbalance handling techniques
- **matplotlib** & **seaborn**: Data visualization
- **PyYAML** (6.0.0+): Configuration file parsing
- **mkdocs**: Documentation generation

---

## Quick Start

### Option A: Google Colab (Recommended for First-Time Users)

The easiest way to run experiments without local setup:

1. Click the **Open in Colab** badge above or visit the [notebook directly](https://colab.research.google.com/github/ihmorol/unsw-nb15-handling-binary-multiclass-ids/blob/main/UNSW_NB15_Full_Grid.ipynb)
2. Follow the notebook cells to:
   - Mount Google Drive for result storage
   - Install dependencies automatically
   - Run the complete 18-experiment grid
   - Generate visualizations and reports
3. Results are automatically saved to your Google Drive

**Benefits:** Zero local setup, GPU acceleration, persistent storage

### Option B: Local Execution

#### Basic Run (18 Experiments)
```bash
# Run the complete experiment grid (45-60 minutes)
python main.py --config configs/main.yaml
```

#### Fast Test Run (Quick Validation)
```bash
# Run minimal configuration for testing (5-10 minutes)
python main.py --config configs/fast.yaml
```

#### Smoke Test (Validation Only)
```bash
# Run basic validation tests (1-2 minutes)
python main.py --config configs/smoke_test.yaml
```

### Option C: Advanced Execution Scripts

#### Full Colab Grid Runner
For comprehensive cloud execution with automatic syncing:
```bash
# Designed for Google Colab environment
python colab_full_grid.py
```
**Features:**
- Automatic Google Drive mounting and syncing
- Optimized for Colab's resource constraints
- Periodic result backups every 60 seconds
- Handles up to 90 experiments (with multiple seeds)

#### Custom Configuration
```bash
# Run with custom config file
python main.py --config path/to/your/config.yaml
```

---

## Scripts and Tools

### Core Execution Scripts

#### `main.py` - Main Experiment Orchestrator
**Purpose:** The primary entry point for running the complete experiment pipeline locally.

**What it does:**
- Loads and preprocesses the UNSW-NB15 dataset
- Executes the experiment grid in parallel (configurable via `n_jobs`)
- Applies different imbalance strategies to training data
- Trains models and evaluates on test set
- Generates comprehensive metrics and visualizations
- Saves results to structured output directories

**Usage:**
```bash
python main.py --config configs/main.yaml
```

**Benefits:**
- Parallel execution for faster completion
- Robust error handling and logging
- Automatic result aggregation and reporting
- Supports incremental runs (skips completed experiments)

#### `colab_full_grid.py` - Colab-Optimized Runner
**Purpose:** Specialized script for running large experiment grids in Google Colab environment.

**What it does:**
- Automatically mounts Google Drive for persistent storage
- Clones/pulls the latest code from GitHub
- Installs dependencies from requirements.txt
- Optimizes configuration for Colab constraints (sequential pipeline execution)
- Runs experiments with periodic Drive syncing
- Generates timestamped result directories

**Usage:**
```bash
# Run directly in Colab cell
python colab_full_grid.py
```

**Benefits:**
- Designed for cloud execution with limited resources
- Automatic backup prevents data loss
- Optimized for Colab's 2-core, high-memory environment
- Handles long-running experiments (4-6 hours) reliably

### Utility Scripts

#### `colab_notebook_cell.py` - Colab Helper
**Purpose:** Lightweight helper for running individual experiments in Colab cells.

**What it does:**
- Installs dependencies
- Sets up basic environment
- Provides functions for single experiment execution
- Integrates with Colab notebook workflow

**Usage:**
```python
# In Colab cell
!python colab_notebook_cell.py
```

#### `UNSW_NB15_Full_Grid.ipynb` - Complete Colab Notebook
**Purpose:** Self-contained Jupyter notebook for full experiment execution (also available in repository root).

**What it does:**
- Interactive Colab environment with step-by-step execution
- Integrated visualization and result analysis
- User-friendly interface with forms and controls
- Automatic report generation

**Benefits:**
- No coding required - just run cells sequentially
- Interactive debugging and monitoring
- Built-in result visualization
- Educational format showing methodology

### Source Code Modules

#### `src/data/` - Data Pipeline
- **`loader.py`**: Handles dataset downloading, validation, and basic loading
- **`preprocessing.py`**: Implements data cleaning, feature engineering, and preprocessing pipeline

**Purpose:** Ensures consistent, leakage-proof data preparation across all experiments.

#### `src/models/` - Model Management
- **`config.py`**: Defines hyperparameters and model configurations for LR, RF, XGBoost
- **`trainer.py`**: Orchestrates model training with appropriate weighting strategies

**Purpose:** Provides unified interface for training different model types with various imbalance handling.

#### `src/evaluation/` - Metrics and Visualization
- **`metrics.py`**: Computes comprehensive metrics including G-Mean, Macro-F1, rare class analysis
- **`plots.py`**: Generates confusion matrices, ROC curves, learning curves
- **`visualizer.py`**: Creates publication-quality visualizations and dashboards

**Purpose:** Standardized evaluation framework focusing on rare class performance.

#### `src/strategies/` - Imbalance Handling
Implements three core strategies:
- **S0 (Baseline)**: No imbalance handling
- **S1 (Class Weighting)**: Automatic weight calculation based on class frequencies
- **S2a (Random Oversampling)**: SMOTE-based oversampling of minority classes

#### `src/utils/` - Utilities
- **`config.py`**: YAML configuration loading and validation
- **`logging.py`**: Structured logging setup with timestamps and levels

#### `src/tests/` - Validation Suite
- **`test_smoke.py`**: Basic functionality tests
- **`test_strategies.py`**: Strategy validation
- **`test_data_pipeline.py`**: Data processing verification

**Purpose:** Ensures code reliability and catches regressions.

---

## Configuration Guide

### Configuration Files

The experiment behavior is controlled by YAML configuration files in the `configs/` directory:

#### `configs/main.yaml` - Full Experiment Suite
```yaml
# Complete 18-experiment grid (2 tasks × 3 models × 3 strategies × 1 seed)
experiments:
  n_seeds: 1          # Number of random seeds for reproducibility
  n_jobs: -1          # Parallel jobs (-1 = all cores)
  tasks: [binary, multi]
  models: [lr, rf, xgb]
  strategies: [s0, s1, s2a]
```

#### `configs/fast.yaml` - Quick Testing
```yaml
# Reduced configuration for rapid iteration
experiments:
  n_seeds: 1
  n_jobs: 1           # Sequential execution
  tasks: [binary]     # Only binary classification
  models: [lr, rf]    # Skip XGBoost
  strategies: [s0, s1]
```

#### `configs/smoke_test.yaml` - Validation
```yaml
# Minimal test for code validation
experiments:
  n_seeds: 1
  n_jobs: 1
  tasks: [binary]
  models: [lr]
  strategies: [s0]
```

### Customizing Experiments

#### Adding New Models
1. Add model name to `experiments.models` in config
2. Implement model creation in `src/models/config.py`
3. Add training logic in `src/models/trainer.py`

#### Modifying Strategies
1. Define new strategy class in `src/strategies/`
2. Implement `apply()` and `get_class_weight()` methods
3. Add to strategy mapping in `src/strategies/__init__.py`

#### Changing Metrics
1. Add new metrics in `src/evaluation/metrics.py`
2. Update `compute_all_metrics()` function
3. Metrics automatically included in results and reports

### Data Configuration

```yaml
data:
  train_path: "dataset/UNSW_NB15_training-set.csv"
  test_path: "dataset/UNSW_NB15_testing-set.csv"
  drop_columns: [id, srcip, dstip, sport, dsport, stime, ltime]
  categorical_columns: [proto, state, service, target_binary, target_multiclass]
```

**Why these columns are dropped:** They cause data leakage by providing temporal or network-level information not available during prediction.

---

## Outputs and Results

### Directory Structure After Execution

```
results/
├── tables/                    # Aggregated results
│   ├── aggregated_summary.csv # Mean/std metrics across seeds
│   ├── all_runs.csv          # Individual experiment results
│   └── rare_class_aggregated.csv # Rare class performance
├── figures/                   # Per-experiment visualizations
│   └── {experiment_id}/
│       ├── confusion_matrix.png
│       ├── roc_curve.png      # Binary only
│       ├── pr_curve.png       # Binary only
│       └── feature_importance.png # Tree models only
├── metrics/                   # Raw experiment data
│   └── {experiment_id}.json   # Complete results per experiment
├── learning_curves/           # Training progress
│   ├── {experiment_id}.json   # Raw learning curve data
│   └── {experiment_id}.csv    # Processed for plotting
└── logs/                      # Execution logs
    └── run_{timestamp}.log
```

### Key Output Files

#### Experiment Metrics JSON (`results/metrics/{experiment_id}.json`)
Contains complete results for each experiment:
```json
{
  "experiment_id": "binary_lr_s0_s42",
  "seed": 42,
  "task": "binary",
  "model": "lr",
  "strategy": "s0",
  "metrics": {
    "overall": {
      "accuracy": 0.89,
      "macro_f1": 0.78,
      "weighted_f1": 0.91,
      "g_mean": 0.82,
      "roc_auc": 0.85
    },
    "per_class": {...},
    "confusion_matrix": [[...]]
  },
  "rare_class_analysis": {...},
  "training_time_seconds": 45.2
}
```

#### Aggregated Tables (`results/tables/`)
- **`aggregated_summary.csv`**: Mean and standard deviation across seeds
- **`all_runs.csv`**: Raw results from all experiments
- **`rare_class_aggregated.csv`**: Performance on minority classes (Worms, Shellcode, etc.)

### Understanding Metrics

#### Primary Metrics (Focus on these for IDS evaluation)
- **G-Mean**: Geometric mean of sensitivity and specificity - treats all classes equally
- **Macro-F1**: Average F1-score across classes - penalizes poor minority performance
- **ROC-AUC**: Area under ROC curve - measures ranking quality

#### Why Accuracy is Misleading
Traditional accuracy can be high (>90%) by simply predicting "Normal" for all samples, missing critical attacks. Our evaluation focuses on balanced metrics that reflect true IDS effectiveness.

#### Rare Class Analysis
Special attention to classes with <3% representation:
- **Worms**: 0.07% (130 samples)
- **Shellcode**: 0.65% (1,133 samples)
- **Backdoor**: 1.00% (1,746 samples)

### Generated Reports

#### `reports/final_results.md`
Auto-generated executive summary with:
- Key findings and performance comparisons
- Statistical significance tests
- Publication-ready tables and figures
- Methodology documentation

#### Visualization Gallery
- Confusion matrices for each experiment
- ROC/PR curves for binary classification
- Feature importance plots for tree models
- Learning curves showing training progress

---

## Contributing

### Ways to Contribute

1. **Bug Reports**: Open issues for bugs or unexpected behavior
2. **Feature Requests**: Suggest new imbalance strategies, models, or metrics
3. **Code Contributions**: Submit pull requests with improvements
4. **Documentation**: Improve guides, add examples, or clarify explanations
5. **Testing**: Add test cases or validate results on different datasets

### Development Setup

```bash
# Fork and clone the repository
git clone https://github.com/yourusername/ML_PAPER_REVIEW.git
cd ML_PAPER_REVIEW

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -r requirements.txt
pip install -e .

# Run tests
python -m pytest src/tests/

# Run type checking
pyright
```

### Code Standards

- **Python**: Follow PEP 8 style guidelines
- **Documentation**: Use Google-style docstrings
- **Testing**: Maintain >80% test coverage
- **Commits**: Use conventional commit format

### Pull Request Process

1. Create a feature branch from `main`
2. Add tests for new functionality
3. Ensure all tests pass
4. Update documentation if needed
5. Submit PR with clear description

### Research Contributions

This is a research codebase. When contributing:
- Cite relevant papers in docstrings
- Include statistical validation for claims
- Document experimental methodology
- Share reproducible results

---

## License and Citation

### License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### Citation

If you use this codebase in your research, please cite:

```bibtex
@misc{ml_paper_review_2024,
  title={Handling Class Imbalance in UNSW-NB15: A Reproducible Baseline},
  author={Morol, Emon and Contributors},
  year={2024},
  publisher={GitHub},
  url={https://github.com/ihmorol/unsw-nb15-handling-binary-multiclass-ids}
}
```

### Acknowledgments

- **Dataset**: UNSW-NB15 dataset by the University of New South Wales
- **Research**: Based on methodologies from class imbalance literature
- **Tools**: Built with scikit-learn, XGBoost, and imbalanced-learn

### Contact

- **Issues**: [GitHub Issues](https://github.com/ihmorol/unsw-nb15-handling-binary-multiclass-ids/issues)
- **Discussions**: [GitHub Discussions](https://github.com/ihmorol/unsw-nb15-handling-binary-multiclass-ids/discussions)
- **Email**: Research inquiries welcome

---

## Related Resources

- [UNSW-NB15 Dataset](https://research.unsw.edu.au/projects/unsw-nb15-dataset)
- [Scikit-learn Documentation](https://scikit-learn.org/)
- [Imbalanced-learn Library](https://imbalanced-learn.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)

---

*Built for reproducible machine learning research*


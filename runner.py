import argparse
import sys
import yaml
import logging
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent))

from src.data.preprocessing import UNSWPreprocessor
from src.models.trainer import ModelTrainer
from main import run_single_experiment, setup_logging

def get_args():
    parser = argparse.ArgumentParser(description='Run single ML IDS experiment')
    parser.add_argument('--task', type=str, required=True, choices=['binary', 'multi'],
                        help='Classification task')
    parser.add_argument('--model', type=str, required=True, choices=['lr', 'rf', 'xgb'],
                        help='Model architecture')
    parser.add_argument('--strategy', type=str, required=True, choices=['s0', 's1', 's2', 's2a', 's2b'],
                        help='Imbalance strategy (s2 kept as alias for s2a)')
    parser.add_argument('--config', type=str, default='configs/main.yaml',
                        help='Path to config file')
    return parser.parse_args()

def main():
    args = get_args()
    
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup logging
    log_path = Path(config['results_dir']) / 'logs' / f"single_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    setup_logging(level="INFO", log_file=str(log_path))
    logger = logging.getLogger(__name__)
    
    try:
        logger.info(f"Preparing data for task: {args.task}")
        # Initialize components
        from src.data.loader import DataLoader
        
        # Load and preprocess
        loader = DataLoader(config)
        train_df, test_df = loader.load_all()
        
        preprocessor = UNSWPreprocessor(config)
        preprocessor.fit_transform(train_df, test_df)
        
        # Get splits
        X_train, y_train, X_val, y_val, X_test, y_test = preprocessor.get_splits(args.task)
        
        # Get class names
        class_names = config['data'].get('multiclass_labels', 
                                       list(preprocessor.label_mapping.keys()))
        
        # Keep backward compatibility: s2 aliases to s2a.
        strategy_map = {
            's0': 's0',
            's1': 's1',
            's2': 's2a',
            's2a': 's2a',
            's2b': 's2b',
        }
        strategy_name = strategy_map[args.strategy]
            
        logger.info(f"Starting Single Experiment: Task={args.task}, Model={args.model}, Strategy={strategy_name}")
        
        # Generate Run ID
        run_id = f"{args.task}_{args.model}_{strategy_name}_single"
        
        # Run Experiment
        metrics = run_single_experiment(
            experiment_id=run_id,
            task=args.task,
            model_name=args.model,
            strategy_name=strategy_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            config=config,
            results_dir=Path(config['results_dir']),
            class_names=class_names,
            X_val=X_val,
            y_val=y_val
        )
        
        logger.info("Experiment Completed Successfully.")
        print(f"Artifacts saved to {config['results_dir']}/runs/{run_id}")
        
    except Exception as e:
        logger.exception(f"Experiment failed: {str(e)}")
        raise

if __name__ == '__main__':
    main()

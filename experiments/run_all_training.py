"""
Master Training Script for MDDM Pipeline

Uruchamia wszystkie treningi modeli BEZ preprocessingu.
Zakłada, że dane są już przetworzone (spectrogramy, instances, features).

Użycie:
    # Pełny pipeline z Optuna
    python experiments/run_all_training.py --n-trials 50
    
    # Szybki test bez Optuna
    python experiments/run_all_training.py --no-optuna
    
    # Tylko wybrane modele
    python experiments/run_all_training.py --models xgboost mil
"""

import sys
import argparse
import time
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import CONFIG
from experiments.experiment_tracker import ExperimentTracker


def check_data_availability():
    """Check if required data files exist"""
    print("\n" + "="*60)
    print("Checking data availability...")
    print("="*60)
    
    required_files = {
        'Labels CSV': CONFIG.LABELS_CSV,
        'Fused Features': CONFIG.FUSED_FEATURES_CSV,
        'Instances CSV': CONFIG.INSTANCES_CSV,
    }
    
    optional_files = {
        'X_train.npy': CONFIG.X_TRAIN_NPY,
        'X_val.npy': CONFIG.X_VAL_NPY,
        'X_test.npy': CONFIG.X_TEST_NPY,
        'Spectrograms dir': CONFIG.SPECTROGRAMS_DIR,
    }
    
    missing_required = []
    for name, path in required_files.items():
        exists = path.exists()
        status = "[OK]" if exists else "[MISSING]"
        print(f"  {status} {name}: {path}")
        if not exists:
            missing_required.append(name)
    
    print("\nOptional files:")
    for name, path in optional_files.items():
        if path.is_dir():
            exists = path.exists() and len(list(path.glob("*"))) > 0
        else:
            exists = path.exists()
        status = "[OK]" if exists else "[--]"
        print(f"  {status} {name}: {path}")
    
    if missing_required:
        print(f"\n[ERROR] Missing required files: {', '.join(missing_required)}")
        print("   Run the preprocessing pipeline first!")
        return False
    
    print("\n[OK] All required data available")
    return True


def train_xgboost(use_optuna: bool = True, n_trials: int = 30, tracker: ExperimentTracker = None):
    """Train XGBoost model"""
    print("\n" + "="*60)
    print("PHASE 1: XGBoost Baseline")
    print("="*60)
    
    from experiments.run_baselines import run_xgboost_baseline
    from baselines.optuna_tuning import load_best_params
    
    start_time = time.time()
    
    if tracker:
        tracker.start_run("XGBoost (Early Fusion)")
    
    try:
        baseline, report = run_xgboost_baseline(
            use_tuning=use_optuna,
            n_trials=n_trials
        )
        
        metrics = {
            'f1': report.metrics['f1'],
            'auroc': report.metrics.get('auroc', 0),
            'precision': report.metrics['precision'],
            'recall': report.metrics['recall'],
            'mcc': report.metrics.get('mcc', 0),
            'accuracy': report.metrics['accuracy'],
        }
        
        params = {
            'n_estimators': baseline.n_estimators,
            'max_depth': baseline.max_depth,
            'learning_rate': baseline.learning_rate,
            'n_features': baseline.n_features,
            'best_threshold': baseline.best_threshold,
        }
        
        if tracker:
            tracker.log_params(params)
            tracker.log_metrics(metrics)
            tracker.log_artifact(CONFIG.XGBOOST_MODEL)
            tracker.end_run()
        
        duration = time.time() - start_time
        print(f"\n[OK] XGBoost completed in {duration:.1f}s")
        print(f"  F1: {metrics['f1']:.4f}, AUROC: {metrics['auroc']:.4f}")
        
        return {'params': params, 'metrics': metrics, 'model': baseline, 'report': report}
    
    except Exception as e:
        print(f"\n[ERROR] XGBoost training failed: {e}")
        if tracker and tracker._current_run:
            tracker.end_run()
        return None


def train_mil_text(use_optuna: bool = True, n_trials: int = 30, tracker: ExperimentTracker = None):
    """Train MIL Text model"""
    print("\n" + "="*60)
    print("PHASE 2: MIL Text Model")
    print("="*60)
    
    start_time = time.time()
    
    if tracker:
        tracker.start_run("MIL Text (attention)")
    
    try:
        if use_optuna:
            from experiments.optuna_mil_text import run_mil_optimization, train_with_best_params
            
            best_params, study = run_mil_optimization(
                n_trials=n_trials,
                timeout=CONFIG.OPTUNA_TIMEOUT
            )
            model, metrics = train_with_best_params(best_params)
            params = best_params
        else:
            from experiments.optuna_mil_text import train_with_best_params
            model, metrics = train_with_best_params()
            params = {
                'hidden_dim': 256,
                'dropout_rate': 0.3,
                'learning_rate': 2e-4,
                'pos_weight': 2.0,
                'unfreeze_layers': 2,
                'pooling_strategy': 'attention'
            }
        
        if tracker:
            tracker.log_params(params)
            tracker.log_metrics(metrics)
            tracker.log_artifact(CONFIG.MIL_TEXT_MODEL)
            tracker.end_run()
        
        duration = time.time() - start_time
        print(f"\n[OK] MIL Text completed in {duration:.1f}s")
        print(f"  F1: {metrics['f1']:.4f}, AUROC: {metrics['auroc']:.4f}")
        
        return {'params': params, 'metrics': metrics, 'model': model}
    
    except Exception as e:
        print(f"\n[ERROR] MIL Text training failed: {e}")
        import traceback
        traceback.print_exc()
        if tracker and tracker._current_run:
            tracker.end_run()
        return None


def train_multimodal(use_optuna: bool = True, n_trials: int = 30, tracker: ExperimentTracker = None):
    """Train Multimodal Fusion model"""
    print("\n" + "="*60)
    print("PHASE 3: Multimodal Fusion")
    print("="*60)
    
    start_time = time.time()
    
    if tracker:
        tracker.start_run("Multimodal Late Fusion")
    
    try:
        if use_optuna:
            from experiments.optuna_multimodal import run_multimodal_optimization, train_with_best_params
            
            best_params, study = run_multimodal_optimization(
                n_trials=n_trials,
                timeout=CONFIG.OPTUNA_TIMEOUT
            )
            model, metrics = train_with_best_params(best_params)
            params = best_params
        else:
            from experiments.optuna_multimodal import train_with_best_params
            model, metrics = train_with_best_params()
            params = {
                'fusion_hidden': 64,
                'dropout_rate': 0.3,
                'learning_rate': 1e-3,
                'pos_weight': 2.5
            }
        
        if tracker:
            tracker.log_params(params)
            tracker.log_metrics(metrics)
            tracker.log_artifact(CONFIG.MULTIMODAL_MODEL)
            tracker.end_run()
        
        duration = time.time() - start_time
        print(f"\n[OK] Multimodal completed in {duration:.1f}s")
        print(f"  F1: {metrics['f1']:.4f}, AUROC: {metrics['auroc']:.4f}")
        
        return {'params': params, 'metrics': metrics, 'model': model}
    
    except Exception as e:
        print(f"\n[ERROR] Multimodal training failed: {e}")
        import traceback
        traceback.print_exc()
        if tracker and tracker._current_run:
            tracker.end_run()
        return None


def generate_final_report(tracker: ExperimentTracker, results: dict):
    """Generate final comparison report"""
    print("\n" + "="*60)
    print("FINAL COMPARISON REPORT")
    print("="*60)
    
    # Print summary table
    print(f"\n{'Model':<30} {'F1':>10} {'AUROC':>10} {'Prec':>8} {'Recall':>8}")
    print("-" * 70)
    
    sorted_results = sorted(
        [(k, v) for k, v in results.items() if v is not None],
        key=lambda x: x[1]['metrics']['f1'],
        reverse=True
    )
    
    for name, result in sorted_results:
        m = result['metrics']
        print(f"{name:<30} {m['f1']:>10.4f} {m.get('auroc', 0):>10.4f} "
              f"{m.get('precision', 0):>8.4f} {m.get('recall', 0):>8.4f}")
    
    print("-" * 70)
    
    if sorted_results:
        best_name, best_result = sorted_results[0]
        print(f"\n*** Best model: {best_name}")
        print(f"   F1: {best_result['metrics']['f1']:.4f}")
        print(f"   AUROC: {best_result['metrics'].get('auroc', 0):.4f}")
    
    # Save tracker summary
    tracker.print_summary()
    tracker.export_comparison_csv()
    
    # Save final report
    report_path = CONFIG.RESULTS_DIR / "training_report.txt"
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("MDDM TRAINING REPORT\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write("="*70 + "\n\n")
        
        f.write("RESULTS SUMMARY\n")
        f.write("-"*70 + "\n")
        f.write(f"{'Model':<30} {'F1':>10} {'AUROC':>10} {'Prec':>8} {'Recall':>8}\n")
        f.write("-"*70 + "\n")
        
        for name, result in sorted_results:
            m = result['metrics']
            f.write(f"{name:<30} {m['f1']:>10.4f} {m.get('auroc', 0):>10.4f} "
                   f"{m.get('precision', 0):>8.4f} {m.get('recall', 0):>8.4f}\n")
        
        f.write("-"*70 + "\n\n")
        
        if sorted_results:
            f.write(f"Best model: {sorted_results[0][0]}\n")
            f.write(f"Best F1: {sorted_results[0][1]['metrics']['f1']:.4f}\n")
    
    print(f"\nReport saved to: {report_path}")


def run_all(
    use_optuna: bool = True,
    n_trials: int = 30,
    models: list = None
):
    """
    Run all training pipelines.
    
    Args:
        use_optuna: Whether to use Optuna hyperparameter optimization
        n_trials: Number of Optuna trials per model
        models: List of models to train ['xgboost', 'mil', 'multimodal'] or ['all']
    """
    print("\n" + "="*70)
    print("  MDDM - MULTIMODAL DEPRESSION DETECTION MODEL")
    print("  Full Training Pipeline")
    print("="*70)
    print(f"\n  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Optuna: {'Enabled' if use_optuna else 'Disabled'}")
    if use_optuna:
        print(f"  Trials per model: {n_trials}")
    print(f"  Device: {CONFIG.DEVICE}")
    
    # Check data
    if not check_data_availability():
        return None
    
    # Set seed
    CONFIG.set_seed()
    
    # Determine which models to train
    if models is None or 'all' in models:
        models = ['xgboost', 'mil', 'multimodal']
    
    print(f"\n  Models to train: {', '.join(models)}")
    
    # Create tracker
    experiment_name = f"full_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    tracker = ExperimentTracker(experiment_name)
    
    total_start = time.time()
    results = {}
    
    # Train XGBoost
    if 'xgboost' in models:
        results['XGBoost'] = train_xgboost(use_optuna, n_trials, tracker)
    
    # Train MIL Text
    if 'mil' in models:
        results['MIL Text'] = train_mil_text(use_optuna, n_trials, tracker)
    
    # Train Multimodal
    if 'multimodal' in models:
        results['Multimodal'] = train_multimodal(use_optuna, n_trials, tracker)
    
    # Generate final report
    generate_final_report(tracker, results)
    
    total_duration = time.time() - total_start
    
    print("\n" + "="*70)
    print(f"  TRAINING COMPLETE")
    print(f"  Total duration: {total_duration/60:.1f} minutes")
    print(f"  Results saved to: {CONFIG.RESULTS_DIR}")
    print(f"  Logs saved to: {tracker.experiment_dir}")
    print("="*70 + "\n")
    
    return tracker, results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='MDDM Master Training Script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full pipeline with Optuna (recommended)
  python experiments/run_all_training.py --n-trials 50
  
  # Quick test without Optuna
  python experiments/run_all_training.py --no-optuna
  
  # Only specific models
  python experiments/run_all_training.py --models xgboost mil
  
  # Only Multimodal with more trials
  python experiments/run_all_training.py --models multimodal --n-trials 100
        """
    )
    
    parser.add_argument('--no-optuna', action='store_true',
                       help='Disable Optuna hyperparameter optimization')
    parser.add_argument('--n-trials', type=int, default=30,
                       help='Number of Optuna trials per model (default: 30)')
    parser.add_argument('--models', nargs='+',
                       choices=['xgboost', 'mil', 'multimodal', 'all'],
                       default=['all'],
                       help='Models to train (default: all)')
    
    args = parser.parse_args()
    
    tracker, results = run_all(
        use_optuna=not args.no_optuna,
        n_trials=args.n_trials,
        models=args.models
    )
    
    if results:
        print("\n[SUCCESS] All training completed successfully!")
    else:
        print("\n[FAILED] Training failed - check data availability")

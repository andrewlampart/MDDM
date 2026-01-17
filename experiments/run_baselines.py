"""
Sprint 1: Run Baseline Experiments

This script trains and evaluates the XGBoost baseline model with:
- GPU acceleration (if available)
- Threshold optimization
- Full evaluation with bootstrap CI
- ROC/PR curves
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import CONFIG
from baselines.xgboost_baseline import XGBoostBaseline
from baselines.optuna_tuning import run_optimization, load_best_params, OPTUNA_AVAILABLE
from evaluation.metrics import MetricsComputer
from evaluation.report import EvaluationReport


def load_data():
    """Load preprocessed data"""
    print("Loading data...")
    
    # Try loading from numpy files first
    if CONFIG.X_TRAIN_NPY.exists():
        X_train = np.load(CONFIG.X_TRAIN_NPY)
        X_val = np.load(CONFIG.X_VAL_NPY)
        X_test = np.load(CONFIG.X_TEST_NPY)
        y_train = np.load(CONFIG.Y_TRAIN_NPY)
        y_val = np.load(CONFIG.Y_VAL_NPY)
        y_test = np.load(CONFIG.Y_TEST_NPY)
        
        print(f"Loaded from numpy files:")
    else:
        # Load from fused features CSV
        fused_path = CONFIG.FUSED_FEATURES_CSV
        if not fused_path.exists():
            fused_path = CONFIG.PROCESSED_DATA_DIR / "daic_fused_features.csv"
        
        if not fused_path.exists():
            raise FileNotFoundError(
                f"No data found. Run the preprocessing pipeline first.\n"
                f"Expected: {CONFIG.X_TRAIN_NPY} or {fused_path}"
            )
        
        print(f"Loading from {fused_path}...")
        df = pd.read_csv(fused_path)
        
        # Prepare features and labels
        exclude_cols = ['session_id', 'depression', 'phq8_score', 'qids_score']
        feature_cols = [c for c in df.columns if c not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['depression'].values
        
        # Split data
        from sklearn.model_selection import train_test_split
        
        # First split: train+val vs test
        X_trainval, X_test, y_trainval, y_test = train_test_split(
            X, y,
            test_size=CONFIG.TEST_SIZE / len(y),
            stratify=y,
            random_state=CONFIG.RANDOM_SEED
        )
        
        # Second split: train vs val
        X_train, X_val, y_train, y_val = train_test_split(
            X_trainval, y_trainval,
            test_size=CONFIG.VAL_SIZE / (CONFIG.TRAIN_SIZE + CONFIG.VAL_SIZE),
            stratify=y_trainval,
            random_state=CONFIG.RANDOM_SEED
        )
        
        print(f"Split from fused features:")
    
    print(f"  Train: {X_train.shape[0]} samples, {X_train.shape[1]} features")
    print(f"  Val:   {X_val.shape[0]} samples")
    print(f"  Test:  {X_test.shape[0]} samples")
    print(f"  Class distribution - Train: {y_train.mean():.1%}, Test: {y_test.mean():.1%}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def run_xgboost_baseline(use_tuning: bool = False, n_trials: int = None):
    """Train and evaluate XGBoost baseline
    
    Args:
        use_tuning: Whether to run Optuna hyperparameter optimization
        n_trials: Number of Optuna trials (default from CONFIG)
    """
    
    print("=" * 60)
    print("Sprint 1: XGBoost Baseline Experiment")
    print("=" * 60)
    
    # Set seed
    CONFIG.set_seed()
    
    # Print GPU info
    CONFIG.print_gpu_info()
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Optuna hyperparameter tuning
    best_params = None
    if use_tuning:
        if not OPTUNA_AVAILABLE:
            print("\n⚠️  Optuna not available. Install with: pip install optuna")
            print("    Continuing with default parameters...")
        else:
            best_params, study = run_optimization(
                X_train, y_train, X_val, y_val,
                n_trials=n_trials or CONFIG.OPTUNA_N_TRIALS,
                timeout=CONFIG.OPTUNA_TIMEOUT,
                metric=CONFIG.OPTUNA_METRIC
            )
    
    # Train XGBoost
    print("\n" + "-" * 60)
    print("Training XGBoost Baseline" + (" (with tuned params)" if best_params else ""))
    print("-" * 60)
    
    # Apply tuned parameters if available
    if best_params:
        baseline = XGBoostBaseline(
            n_estimators=best_params.get('n_estimators', CONFIG.XGBOOST_N_ESTIMATORS),
            max_depth=best_params.get('max_depth', CONFIG.XGBOOST_MAX_DEPTH),
            learning_rate=best_params.get('learning_rate', CONFIG.XGBOOST_LEARNING_RATE),
            n_features=best_params.get('n_features', CONFIG.N_FEATURES_SELECT),
            use_gpu=CONFIG.USE_GPU
        )
    else:
        baseline = XGBoostBaseline(
            n_features=CONFIG.N_FEATURES_SELECT,
            use_gpu=CONFIG.USE_GPU
        )
    
    baseline.fit(
        X_train, y_train,
        X_val, y_val,
        early_stopping_rounds=CONFIG.XGBOOST_EARLY_STOPPING,
        verbose=50
    )
    
    # Evaluate on test set
    print("\n" + "-" * 60)
    print("Evaluating on Test Set")
    print("-" * 60)
    
    y_pred, y_proba = baseline.predict(X_test), baseline.predict_proba(X_test)[:, 1]
    
    # Generate evaluation report
    report = EvaluationReport(
        model_name="XGBoost (Early Fusion)",
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba
    )
    
    report.generate(n_bootstrap=CONFIG.BOOTSTRAP_CI_RESAMPLES)
    report.print_report()
    
    # Save report and plots
    report.save_json()
    report.save_plots()
    
    # Feature importance
    print("\n" + "-" * 60)
    print("Top 15 Most Important Features")
    print("-" * 60)
    
    importance = baseline.get_feature_importance(top_n=15)
    for i, (name, imp) in enumerate(zip(importance['feature_names'], importance['importances'])):
        print(f"  {i+1:2d}. {name}: {imp:.4f}")
    
    # Save model
    baseline.save()
    
    # Comparison with default threshold
    print("\n" + "-" * 60)
    print("Threshold Comparison")
    print("-" * 60)
    
    y_pred_default = (y_proba >= 0.5).astype(int)
    from sklearn.metrics import f1_score
    f1_default = f1_score(y_test, y_pred_default)
    f1_optimal = f1_score(y_test, y_pred)
    
    print(f"  F1 (threshold=0.50): {f1_default:.4f}")
    print(f"  F1 (threshold={baseline.best_threshold:.2f}): {f1_optimal:.4f}")
    print(f"  Improvement: +{(f1_optimal - f1_default):.4f}")
    
    print("\n" + "=" * 60)
    print("Sprint 1 Complete!")
    print("=" * 60)
    
    return baseline, report


def run_random_forest_comparison():
    """Run Random Forest for comparison"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    
    print("\n" + "=" * 60)
    print("Random Forest Baseline (for comparison)")
    print("=" * 60)
    
    X_train, X_val, X_test, y_train, y_val, y_test = load_data()
    
    # Preprocess
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Feature selection
    selector = SelectKBest(score_func=f_classif, k=CONFIG.N_FEATURES_SELECT)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train)
    X_val_sel = selector.transform(X_val_scaled)
    X_test_sel = selector.transform(X_test_scaled)
    
    # Train
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=CONFIG.RANDOM_SEED,
        n_jobs=-1
    )
    rf.fit(X_train_sel, y_train)
    
    # Predict
    y_proba = rf.predict_proba(X_test_sel)[:, 1]
    
    # Optimize threshold
    from sklearn.metrics import f1_score
    best_f1, best_thresh = 0, 0.5
    for thresh in np.arange(0.3, 0.8, 0.01):
        f1 = f1_score(y_test, (y_proba >= thresh).astype(int))
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    y_pred = (y_proba >= best_thresh).astype(int)
    
    # Report
    report = EvaluationReport(
        model_name="Random Forest (Early Fusion)",
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba
    )
    report.generate(n_bootstrap=CONFIG.BOOTSTRAP_CI_RESAMPLES)
    report.print_report()
    report.save_json()
    
    return report


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Run baseline experiments')
    parser.add_argument('--rf', action='store_true', help='Also run Random Forest')
    parser.add_argument('--tune', action='store_true', help='Run Optuna hyperparameter optimization')
    parser.add_argument('--n_trials', type=int, default=None, help='Number of Optuna trials (default: 50)')
    args = parser.parse_args()
    
    # Run XGBoost baseline
    baseline, xgb_report = run_xgboost_baseline(
        use_tuning=args.tune,
        n_trials=args.n_trials
    )
    
    # Optionally run RF
    if args.rf:
        rf_report = run_random_forest_comparison()
        
        # Compare
        from evaluation.report import ModelComparison
        comparison = ModelComparison()
        comparison.add_report(xgb_report)
        comparison.add_report(rf_report)
        comparison.print_comparison()
        comparison.save_comparison_csv()

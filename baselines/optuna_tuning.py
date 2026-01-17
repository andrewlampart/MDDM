"""
Optuna Hyperparameter Optimization for XGBoost Baseline

Sprint 1 Enhancement: Automatic hyperparameter tuning with Optuna.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
import warnings

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna")

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG


def create_objective(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    metric: str = "f1"
):
    """
    Create Optuna objective function for XGBoost hyperparameter optimization.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        metric: Metric to optimize ("f1", "auroc", "mcc", "auprc")
    
    Returns:
        Objective function for Optuna
    """
    
    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter space (zoptymalizowane dla małego datasetu)
        params = {
            'n_estimators': trial.suggest_int('n_estimators', 100, 800, step=50),
            'max_depth': trial.suggest_int('max_depth', 3, 8),  # Ograniczone dla mniejszego overfittingu
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.15, log=True),  # Max 0.15 zamiast 0.3
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),  # Min 0.6 dla stabilności
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),  # Min 0.6
            'reg_alpha': trial.suggest_float('reg_alpha', 0.01, 10.0, log=True),  # Min 0.01 zamiast 1e-8
            'reg_lambda': trial.suggest_float('reg_lambda', 0.01, 10.0, log=True),  # Min 0.01 zamiast 1e-8
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),  # Ograniczone
            'gamma': trial.suggest_float('gamma', 0.0, 5.0),  # Dodatkowa regularyzacja
        }
        
        # Feature selection
        n_features = trial.suggest_int('n_features', 20, min(100, X_train.shape[1]))
        
        # Preprocessing
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # Feature selection
        if n_features < X_train_scaled.shape[1]:
            selector = SelectKBest(score_func=f_classif, k=n_features)
            X_train_sel = selector.fit_transform(X_train_scaled, y_train)
            X_val_sel = selector.transform(X_val_scaled)
        else:
            X_train_sel = X_train_scaled
            X_val_sel = X_val_scaled
        
        # Sample weights for imbalanced classes
        weights = compute_sample_weight('balanced', y_train)
        
        # Create model (XGBoost 2.0+ requires early_stopping_rounds in constructor)
        model = XGBClassifier(
            **params,
            tree_method='hist',
            device='cuda' if CONFIG.USE_GPU else 'cpu',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=CONFIG.RANDOM_SEED,
            n_jobs=-1,
            early_stopping_rounds=CONFIG.XGBOOST_EARLY_STOPPING  # Użyj z config (30)
        )
        
        # Train with early stopping
        model.fit(
            X_train_sel, y_train,
            sample_weight=weights,
            eval_set=[(X_val_sel, y_val)],
            verbose=False
        )
        
        # Predict
        y_proba = model.predict_proba(X_val_sel)[:, 1]
        
        # Optimize threshold for F1
        best_f1 = 0
        best_thresh = 0.5
        for thresh in np.arange(0.1, 0.9, 0.02):
            y_pred = (y_proba >= thresh).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
        
        y_pred = (y_proba >= best_thresh).astype(int)
        
        # Calculate metric
        if metric == "f1":
            score = f1_score(y_val, y_pred, zero_division=0)
        elif metric == "auroc":
            score = roc_auc_score(y_val, y_proba)
        elif metric == "mcc":
            score = matthews_corrcoef(y_val, y_pred)
        elif metric == "auprc":
            from sklearn.metrics import average_precision_score
            score = average_precision_score(y_val, y_proba)
        else:
            score = f1_score(y_val, y_pred, zero_division=0)
        
        # Store threshold in trial
        trial.set_user_attr('best_threshold', best_thresh)
        trial.set_user_attr('best_iteration', model.best_iteration if hasattr(model, 'best_iteration') else params['n_estimators'])
        
        return score
    
    return objective


def run_optimization(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    n_trials: int = None,
    timeout: int = None,
    metric: str = None,
    study_name: str = "xgboost_optimization",
    save_path: Path = None
) -> Tuple[Dict, optuna.Study]:
    """
    Run Optuna hyperparameter optimization for XGBoost.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        n_trials: Number of trials (default from CONFIG)
        timeout: Timeout in seconds (default from CONFIG)
        metric: Metric to optimize (default from CONFIG)
        study_name: Name for the Optuna study
        save_path: Path to save best parameters
    
    Returns:
        Tuple of (best_params dict, optuna.Study)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required. Install with: pip install optuna")
    
    # Defaults from config
    n_trials = n_trials or getattr(CONFIG, 'OPTUNA_N_TRIALS', 50)
    timeout = timeout or getattr(CONFIG, 'OPTUNA_TIMEOUT', 3600)
    metric = metric or getattr(CONFIG, 'OPTUNA_METRIC', 'f1')
    save_path = save_path or CONFIG.RESULTS_DIR / "optuna_best_params.json"
    
    print(f"\n{'='*60}")
    print(f"Optuna Hyperparameter Optimization")
    print(f"{'='*60}")
    print(f"  Metric: {metric}")
    print(f"  Max trials: {n_trials}")
    print(f"  Timeout: {timeout}s")
    print(f"  Train samples: {len(y_train)}")
    print(f"  Validation samples: {len(y_val)}")
    
    # Create study
    sampler = TPESampler(seed=CONFIG.RANDOM_SEED)
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10)
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    
    # Create objective
    objective = create_objective(X_train, y_train, X_val, y_val, metric=metric)
    
    # Suppress Optuna logs during optimization
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Run optimization with progress callback
    def callback(study, trial):
        if trial.number % 10 == 0:
            print(f"  Trial {trial.number}: {metric}={trial.value:.4f} (best: {study.best_value:.4f})")
    
    print(f"\nStarting optimization...")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[callback],
        show_progress_bar=True
    )
    
    # Get best parameters
    best_params = study.best_params.copy()
    best_params['best_threshold'] = study.best_trial.user_attrs.get('best_threshold', 0.5)
    best_params['best_iteration'] = study.best_trial.user_attrs.get('best_iteration', best_params['n_estimators'])
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Optimization Complete!")
    print(f"{'='*60}")
    print(f"  Best {metric}: {study.best_value:.4f}")
    print(f"  Best trial: #{study.best_trial.number}")
    print(f"\nBest hyperparameters:")
    for key, value in best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.6f}")
        else:
            print(f"  {key}: {value}")
    
    # Save best parameters
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump({
            'best_params': best_params,
            'best_score': study.best_value,
            'metric': metric,
            'n_trials': len(study.trials),
            'study_name': study_name
        }, f, indent=2)
    print(f"\nBest parameters saved to: {save_path}")
    
    return best_params, study


def load_best_params(path: Path = None) -> Optional[Dict]:
    """Load best parameters from JSON file."""
    path = path or CONFIG.RESULTS_DIR / "optuna_best_params.json"
    
    if not path.exists():
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data.get('best_params')


if __name__ == '__main__':
    """Quick test of Optuna optimization"""
    print("Optuna Tuning Module")
    print(f"Optuna available: {OPTUNA_AVAILABLE}")
    
    if OPTUNA_AVAILABLE:
        # Load data
        if CONFIG.X_TRAIN_NPY.exists():
            X_train = np.load(CONFIG.X_TRAIN_NPY)
            X_val = np.load(CONFIG.X_VAL_NPY)
            y_train = np.load(CONFIG.Y_TRAIN_NPY)
            y_val = np.load(CONFIG.Y_VAL_NPY)
            
            print(f"Data loaded: {X_train.shape[0]} train, {X_val.shape[0]} val")
            
            # Quick test with few trials
            best_params, study = run_optimization(
                X_train, y_train, X_val, y_val,
                n_trials=5,
                timeout=120
            )
        else:
            print("No data found. Run preprocessing first.")

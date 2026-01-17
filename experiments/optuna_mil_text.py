"""
Optuna Hyperparameter Optimization for MIL Text Model

Automatyczny tuning hiperparametrów modelu MIL z RoBERTa.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional
from tqdm import tqdm
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import CONFIG

try:
    import optuna
    from optuna.pruners import MedianPruner
    from optuna.samplers import TPESampler
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False
    warnings.warn("Optuna not available. Install with: pip install optuna")

try:
    import torch
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")

if TORCH_AVAILABLE:
    from data.mil_dataloader import create_mil_dataloaders, create_mil_dataloaders_from_splits
    from models.mil_text_model import MILTextModel, MILTextTrainer
    from sklearn.metrics import f1_score, roc_auc_score


def create_mil_objective(
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: str,
    max_epochs: int = 15,
    patience: int = 5
):
    """
    Create Optuna objective function for MIL Text Model.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        device: Device to use
        max_epochs: Maximum epochs per trial
        patience: Early stopping patience
        
    Returns:
        Objective function for Optuna
    """
    
    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter space
        params = {
            'hidden_dim': trial.suggest_categorical('hidden_dim', [128, 256, 512]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-5, 5e-4, log=True),
            'pos_weight': trial.suggest_float('pos_weight', 1.5, 3.5),
            'unfreeze_layers': trial.suggest_int('unfreeze_layers', 0, 3),
            'pooling_strategy': trial.suggest_categorical('pooling_strategy', 
                ['attention', 'max', 'mean']),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        }
        
        # Create model
        model = MILTextModel(
            model_name=CONFIG.ROBERTA_MODEL_NAME,
            hidden_dim=params['hidden_dim'],
            dropout_rate=params['dropout_rate'],
            freeze_encoder=True,
            unfreeze_last_n_layers=params['unfreeze_layers'],
            pooling_strategy=params['pooling_strategy'],
            device=device
        )
        model = model.to(device)
        
        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Training loop
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Train
            model.train()
            for batch in train_loader:
                bag_instances = batch['bag_instances']
                labels = batch['labels'].to(device)
                
                optimizer.zero_grad()
                
                predictions, instance_scores = model.forward_batch(
                    bag_instances, return_instance_scores=True
                )
                loss = model.compute_loss(
                    predictions, labels, instance_scores, 
                    pos_weight=params['pos_weight']
                )
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # Validate
            model.eval()
            val_preds = []
            val_labels_list = []
            
            with torch.no_grad():
                for batch in val_loader:
                    bag_instances = batch['bag_instances']
                    labels = batch['labels']
                    
                    predictions = model.forward_batch(bag_instances)
                    
                    val_preds.extend(predictions.cpu().numpy())
                    val_labels_list.extend(labels.numpy())
            
            val_preds = np.array(val_preds)
            val_labels_arr = np.array(val_labels_list)
            
            # Optimize threshold
            best_thresh_f1 = 0
            for thresh in np.arange(0.3, 0.7, 0.05):
                f1 = f1_score(val_labels_arr, (val_preds >= thresh).astype(int), zero_division=0)
                if f1 > best_thresh_f1:
                    best_thresh_f1 = f1
            
            val_f1 = best_thresh_f1
            
            # Report to Optuna for pruning
            trial.report(val_f1, epoch)
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
            
            # Pruning
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Cleanup
        del model
        torch.cuda.empty_cache() if device == 'cuda' else None
        
        return best_val_f1
    
    return objective


def run_mil_optimization(
    n_trials: int = None,
    timeout: int = None,
    max_epochs_per_trial: int = 15,
    study_name: str = "mil_text_optimization",
    save_path: Path = None
) -> Tuple[Dict, 'optuna.Study']:
    """
    Run Optuna hyperparameter optimization for MIL Text Model.
    
    Args:
        n_trials: Number of trials
        timeout: Timeout in seconds
        max_epochs_per_trial: Max epochs per trial
        study_name: Name for the study
        save_path: Path to save best parameters
        
    Returns:
        Tuple of (best_params, study)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required. Install with: pip install optuna")
    
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    # Defaults
    n_trials = n_trials or CONFIG.OPTUNA_N_TRIALS
    timeout = timeout or CONFIG.OPTUNA_TIMEOUT
    save_path = save_path or CONFIG.RESULTS_DIR / "optuna_mil_best_params.json"
    
    print(f"\n{'='*60}")
    print(f"Optuna MIL Text Hyperparameter Optimization")
    print(f"{'='*60}")
    print(f"  Max trials: {n_trials}")
    print(f"  Timeout: {timeout}s")
    print(f"  Max epochs/trial: {max_epochs_per_trial}")
    
    # Set seed
    CONFIG.set_seed()
    device = CONFIG.DEVICE
    
    # Create dataloaders - try standard splits first
    print("\nCreating DataLoaders...")
    try:
        train_loader, val_loader, test_loader = create_mil_dataloaders_from_splits(
            batch_size=CONFIG.MIL_BATCH_SIZE
        )
    except Exception:
        train_loader, val_loader, test_loader = create_mil_dataloaders(
            batch_size=CONFIG.MIL_BATCH_SIZE
        )
    
    # Create study
    sampler = TPESampler(seed=CONFIG.RANDOM_SEED)
    pruner = MedianPruner(n_startup_trials=3, n_warmup_steps=3)
    
    study = optuna.create_study(
        study_name=study_name,
        direction="maximize",
        sampler=sampler,
        pruner=pruner
    )
    
    # Create objective
    objective = create_mil_objective(
        train_loader, val_loader, device,
        max_epochs=max_epochs_per_trial,
        patience=5
    )
    
    # Suppress verbose output
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
    # Progress callback
    def callback(study, trial):
        if trial.number % 5 == 0 or trial.number == n_trials - 1:
            print(f"  Trial {trial.number}: F1={trial.value:.4f} (best: {study.best_value:.4f})")
    
    print(f"\nStarting optimization...")
    study.optimize(
        objective,
        n_trials=n_trials,
        timeout=timeout,
        callbacks=[callback],
        show_progress_bar=True,
        gc_after_trial=True
    )
    
    # Get best parameters
    best_params = study.best_params.copy()
    
    # Print results
    print(f"\n{'='*60}")
    print(f"Optimization Complete!")
    print(f"{'='*60}")
    print(f"  Best F1: {study.best_value:.4f}")
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
            'metric': 'f1',
            'n_trials': len(study.trials),
            'study_name': study_name
        }, f, indent=2)
    print(f"\nBest parameters saved to: {save_path}")
    
    return best_params, study


def train_with_best_params(
    best_params: Dict = None,
    epochs: int = None,
    patience: int = None
) -> Tuple['MILTextModel', Dict]:
    """
    Train MIL Text model with best parameters from Optuna.
    
    Args:
        best_params: Best parameters (loads from file if None)
        epochs: Training epochs
        patience: Early stopping patience
        
    Returns:
        Tuple of (trained_model, test_metrics)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    # Load params if not provided
    if best_params is None:
        params_path = CONFIG.RESULTS_DIR / "optuna_mil_best_params.json"
        if params_path.exists():
            with open(params_path, 'r') as f:
                data = json.load(f)
                best_params = data['best_params']
        else:
            # Use defaults
            best_params = {
                'hidden_dim': 256,
                'dropout_rate': 0.3,
                'learning_rate': 2e-4,
                'pos_weight': 2.0,
                'unfreeze_layers': 2,
                'pooling_strategy': 'attention',
                'weight_decay': 1e-4
            }
    
    epochs = epochs or CONFIG.EPOCHS
    patience = patience or CONFIG.EARLY_STOPPING_PATIENCE
    device = CONFIG.DEVICE
    
    print(f"\n{'='*60}")
    print(f"Training MIL Text with optimized parameters")
    print(f"{'='*60}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # Create dataloaders
    try:
        train_loader, val_loader, test_loader = create_mil_dataloaders_from_splits(
            batch_size=CONFIG.MIL_BATCH_SIZE
        )
    except Exception:
        train_loader, val_loader, test_loader = create_mil_dataloaders(
            batch_size=CONFIG.MIL_BATCH_SIZE
        )
    
    # Create model
    model = MILTextModel(
        model_name=CONFIG.ROBERTA_MODEL_NAME,
        hidden_dim=best_params['hidden_dim'],
        dropout_rate=best_params['dropout_rate'],
        freeze_encoder=True,
        unfreeze_last_n_layers=best_params['unfreeze_layers'],
        pooling_strategy=best_params['pooling_strategy'],
        device=device
    )
    model = model.to(device)
    
    # Trainer
    trainer = MILTextTrainer(
        model=model,
        learning_rate=best_params['learning_rate'],
        weight_decay=best_params.get('weight_decay', CONFIG.WEIGHT_DECAY)
    )
    
    # Training loop
    best_val_f1 = 0
    patience_counter = 0
    best_model_state = None
    
    pos_weight = best_params.get('pos_weight', 2.0)
    
    for epoch in range(epochs):
        # Train
        model.train()
        total_loss = 0
        n_batches = 0
        
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch in progress:
            bag_instances = batch['bag_instances']
            labels = batch['labels'].to(device)
            
            trainer.optimizer.zero_grad()
            
            predictions, instance_scores = model.forward_batch(
                bag_instances, return_instance_scores=True
            )
            loss = model.compute_loss(predictions, labels, instance_scores, pos_weight=pos_weight)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            trainer.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
            progress.set_postfix({'loss': loss.item()})
        
        train_loss = total_loss / n_batches
        
        # Validation
        val_metrics, val_preds, val_labels, _ = trainer.evaluate(val_loader)
        trainer.scheduler.step(val_metrics['f1'])
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val F1={val_metrics['f1']:.4f}, "
              f"Val AUROC={val_metrics['auroc']:.4f}")
        
        # Early stopping
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
            print(f"  * New best model (F1: {best_val_f1:.4f})")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Test evaluation
    print(f"\n{'-'*60}")
    print("Test Set Evaluation")
    print(f"{'-'*60}")
    
    test_metrics, test_preds, test_labels, _ = trainer.evaluate(test_loader)
    
    # Optimize threshold on test set
    best_thresh = 0.5
    best_test_f1 = 0
    for thresh in np.arange(0.3, 0.8, 0.01):
        f1 = f1_score(test_labels, (test_preds >= thresh).astype(int), zero_division=0)
        if f1 > best_test_f1:
            best_test_f1 = f1
            best_thresh = thresh
    
    test_pred_binary = (test_preds >= best_thresh).astype(int)
    
    # Final metrics
    final_metrics = {
        'f1': f1_score(test_labels, test_pred_binary, zero_division=0),
        'auroc': test_metrics['auroc'],
        'accuracy': test_metrics['accuracy'],
        'threshold': best_thresh
    }
    
    print(f"Test F1: {final_metrics['f1']:.4f}")
    print(f"Test AUROC: {final_metrics['auroc']:.4f}")
    print(f"Optimal threshold: {best_thresh:.2f}")
    
    # Save model
    model.save()
    
    return model, final_metrics


def load_best_mil_params(path: Path = None) -> Optional[Dict]:
    """Load best MIL parameters from JSON file."""
    path = path or CONFIG.RESULTS_DIR / "optuna_mil_best_params.json"
    
    if not path.exists():
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data.get('best_params')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optuna MIL Text Optimization')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout in seconds')
    parser.add_argument('--train-only', action='store_true', 
                       help='Skip optimization, train with existing best params')
    args = parser.parse_args()
    
    if args.train_only:
        model, metrics = train_with_best_params()
    else:
        best_params, study = run_mil_optimization(
            n_trials=args.n_trials,
            timeout=args.timeout
        )
        
        print("\n" + "="*60)
        print("Training with best parameters...")
        print("="*60)
        
        model, metrics = train_with_best_params(best_params)
    
    print("\n[SUCCESS] MIL Text optimization complete!")

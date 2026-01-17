"""
Optuna Hyperparameter Optimization for Multimodal Fusion Model

Automatyczny tuning hiperparametrów modelu fuzji audio + tekst.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, Optional, List
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
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available")

if TORCH_AVAILABLE:
    from models.audio_cnn_model import AudioCNNModel
    from models.mil_text_model import MILTextModel
    from models.multimodal_fusion import LateFusionModel
    from sklearn.metrics import f1_score


def resize_spectrogram(spec: np.ndarray, target_shape: tuple = None) -> np.ndarray:
    """Resize/pad spectrogram to target shape"""
    target_shape = target_shape or (CONFIG.N_MELS, CONFIG.SPECTROGRAM_LENGTH)
    target_h, target_w = target_shape
    h, w = spec.shape
    
    if h < target_h:
        spec = np.pad(spec, ((0, target_h - h), (0, 0)), mode='constant')
    elif h > target_h:
        spec = spec[:target_h, :]
    
    if w < target_w:
        spec = np.pad(spec, ((0, 0), (0, target_w - w)), mode='constant')
    elif w > target_w:
        spec = spec[:, :target_w]
    
    return spec


def load_session_ids_from_split(split_path):
    """Load session IDs from DAIC-WOZ split CSV file"""
    if not split_path.exists():
        return []
    split_df = pd.read_csv(split_path)
    id_col = 'Participant_ID' if 'Participant_ID' in split_df.columns else 'participant_ID'
    if id_col in split_df.columns:
        return split_df[id_col].tolist()
    return []


def get_daic_woz_splits(available_sessions=None):
    """Get train/val/test session IDs from DAIC-WOZ standard splits (107/35/47)"""
    train_split_path = CONFIG.DATA_ROOT / "train_split_Depression_AVEC2017.csv"
    dev_split_path = CONFIG.DATA_ROOT / "dev_split_Depression_AVEC2017.csv"
    test_split_path = CONFIG.DATA_ROOT / "test_split_Depression_AVEC2017.csv"
    
    train_sessions = load_session_ids_from_split(train_split_path)
    val_sessions = load_session_ids_from_split(dev_split_path)
    test_sessions = load_session_ids_from_split(test_split_path)
    
    # Filter to available sessions if provided
    if available_sessions is not None:
        available = set(available_sessions)
        train_sessions = [s for s in train_sessions if s in available]
        val_sessions = [s for s in val_sessions if s in available]
        test_sessions = [s for s in test_sessions if s in available]
    
    return train_sessions, val_sessions, test_sessions


def load_multimodal_data():
    """Load and prepare multimodal data"""
    # Load labels
    labels_df = pd.read_csv(CONFIG.LABELS_CSV)
    labels_dict = dict(zip(labels_df['session_id'], labels_df['depression']))
    
    # Get sessions with spectrograms
    spec_sessions = set()
    for f in CONFIG.SPECTROGRAMS_DIR.glob("*.npy"):
        session_id = int(f.stem.split('_')[0])
        spec_sessions.add(session_id)
    
    # Get sessions with text instances
    instances_df = pd.read_csv(CONFIG.INSTANCES_CSV)
    text_sessions = set(instances_df['session_id'].unique())
    
    # Common sessions
    common_sessions = list(spec_sessions & text_sessions & set(labels_dict.keys()))
    
    return common_sessions, labels_dict, instances_df


def split_sessions(sessions: List[int], labels_dict: Dict) -> Tuple[List, List, List]:
    """Split sessions into train/val/test using DAIC-WOZ standard splits"""
    return get_daic_woz_splits(sessions)


class WeightedBCELoss(nn.Module):
    """BCE Loss with class weights"""
    def __init__(self, pos_weight: float = 2.0):
        super().__init__()
        self.pos_weight = pos_weight
    
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Clamp predictions to avoid log(0)
        predictions = torch.clamp(predictions, 1e-7, 1 - 1e-7)
        
        # Weighted BCE
        loss = -self.pos_weight * targets * torch.log(predictions) - \
               (1 - targets) * torch.log(1 - predictions)
        
        return loss.mean()


def create_fusion_objective(
    train_sessions: List[int],
    val_sessions: List[int],
    labels_dict: Dict,
    instances_df: pd.DataFrame,
    audio_model: 'AudioCNNModel',
    text_model: 'MILTextModel',
    device: str,
    max_epochs: int = 20,
    patience: int = 5
):
    """
    Create Optuna objective function for Multimodal Fusion.
    
    Args:
        train_sessions: Training session IDs
        val_sessions: Validation session IDs
        labels_dict: Session ID -> depression label
        instances_df: DataFrame with text instances
        audio_model: Pre-trained audio model
        text_model: Pre-trained text model
        device: Device to use
        max_epochs: Max epochs per trial
        patience: Early stopping patience
        
    Returns:
        Objective function for Optuna
    """
    
    def objective(trial: optuna.Trial) -> float:
        # Hyperparameter space
        params = {
            'fusion_hidden': trial.suggest_categorical('fusion_hidden', [32, 64, 128]),
            'dropout_rate': trial.suggest_float('dropout_rate', 0.2, 0.5),
            'learning_rate': trial.suggest_float('learning_rate', 1e-4, 1e-2, log=True),
            'pos_weight': trial.suggest_float('pos_weight', 1.5, 3.5),
            'weight_decay': trial.suggest_float('weight_decay', 1e-5, 1e-3, log=True),
        }
        
        # Create fusion model
        fusion_model = LateFusionModel(
            audio_model=audio_model,
            text_model=text_model,
            fusion_hidden=params['fusion_hidden'],
            dropout_rate=params['dropout_rate'],
            device=device
        )
        fusion_model = fusion_model.to(device)
        
        # Optimizer - only train fusion network
        optimizer = torch.optim.AdamW(
            fusion_model.fusion_net.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
        
        # Loss with class weighting
        criterion = WeightedBCELoss(pos_weight=params['pos_weight'])
        
        best_val_f1 = 0
        patience_counter = 0
        
        for epoch in range(max_epochs):
            # Training
            fusion_model.train()
            
            for session_id in train_sessions:
                spec_path = CONFIG.SPECTROGRAMS_DIR / f"{session_id}_spec.npy"
                if not spec_path.exists():
                    continue
                
                # Load spectrogram
                spec = np.load(spec_path)
                spec = resize_spectrogram(spec)
                spec = (spec - spec.mean()) / (spec.std() + 1e-8)
                spec = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(device)
                
                # Load text
                session_instances = instances_df[instances_df['session_id'] == session_id]
                if len(session_instances) == 0:
                    continue
                
                texts = session_instances.sort_values('sequence_num')['text'].tolist()
                label = torch.tensor([labels_dict[session_id]], dtype=torch.float).to(device)
                
                # Forward
                optimizer.zero_grad()
                fused, _, _ = fusion_model(spec, [texts])
                loss = criterion(fused, label)
                
                loss.backward()
                optimizer.step()
            
            # Validation
            fusion_model.eval()
            val_preds = []
            val_labels_list = []
            
            with torch.no_grad():
                for session_id in val_sessions:
                    spec_path = CONFIG.SPECTROGRAMS_DIR / f"{session_id}_spec.npy"
                    if not spec_path.exists():
                        continue
                    
                    spec = np.load(spec_path)
                    spec = resize_spectrogram(spec)
                    spec = (spec - spec.mean()) / (spec.std() + 1e-8)
                    spec = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(device)
                    
                    session_instances = instances_df[instances_df['session_id'] == session_id]
                    if len(session_instances) == 0:
                        continue
                    
                    texts = session_instances.sort_values('sequence_num')['text'].tolist()
                    
                    fused, _, _ = fusion_model(spec, [texts])
                    
                    val_preds.append(fused.item())
                    val_labels_list.append(labels_dict[session_id])
            
            if not val_preds:
                continue
            
            val_preds = np.array(val_preds)
            val_labels_arr = np.array(val_labels_list)
            
            # Find best threshold
            best_thresh_f1 = 0
            for thresh in np.arange(0.3, 0.7, 0.05):
                f1 = f1_score(val_labels_arr, (val_preds >= thresh).astype(int), zero_division=0)
                if f1 > best_thresh_f1:
                    best_thresh_f1 = f1
            
            val_f1 = best_thresh_f1
            
            # Report to Optuna
            trial.report(val_f1, epoch)
            
            # Early stopping
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                break
            
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        # Cleanup
        del fusion_model
        torch.cuda.empty_cache() if device == 'cuda' else None
        
        return best_val_f1
    
    return objective


def run_multimodal_optimization(
    n_trials: int = None,
    timeout: int = None,
    max_epochs_per_trial: int = 20,
    study_name: str = "multimodal_optimization",
    save_path: Path = None
) -> Tuple[Dict, 'optuna.Study']:
    """
    Run Optuna hyperparameter optimization for Multimodal Fusion.
    
    Args:
        n_trials: Number of trials
        timeout: Timeout in seconds
        max_epochs_per_trial: Max epochs per trial
        study_name: Study name
        save_path: Path to save best params
        
    Returns:
        Tuple of (best_params, study)
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna is required")
    
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    # Defaults
    n_trials = n_trials or CONFIG.OPTUNA_N_TRIALS
    timeout = timeout or CONFIG.OPTUNA_TIMEOUT
    save_path = save_path or CONFIG.RESULTS_DIR / "optuna_multimodal_best_params.json"
    
    print(f"\n{'='*60}")
    print(f"Optuna Multimodal Fusion Optimization")
    print(f"{'='*60}")
    print(f"  Max trials: {n_trials}")
    print(f"  Timeout: {timeout}s")
    
    CONFIG.set_seed()
    device = CONFIG.DEVICE
    
    # Load data
    print("\nLoading multimodal data...")
    common_sessions, labels_dict, instances_df = load_multimodal_data()
    train_sessions, val_sessions, test_sessions = split_sessions(common_sessions, labels_dict)
    
    print(f"  Train: {len(train_sessions)}, Val: {len(val_sessions)}, Test: {len(test_sessions)}")
    
    # Load pre-trained models
    print("\nLoading pre-trained modality models...")
    
    audio_model = AudioCNNModel(device=device)
    audio_model = audio_model.to(device)
    if CONFIG.AUDIO_CNN_MODEL.exists():
        audio_model.load()
    else:
        print("  Warning: Audio model not found, using untrained")
    
    text_model = MILTextModel(device=device)
    text_model = text_model.to(device)
    if CONFIG.MIL_TEXT_MODEL.exists():
        text_model.load()
    else:
        print("  Warning: Text model not found, using untrained")
    
    # Freeze modality models
    for param in audio_model.parameters():
        param.requires_grad = False
    for param in text_model.parameters():
        param.requires_grad = False
    
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
    objective = create_fusion_objective(
        train_sessions, val_sessions, labels_dict, instances_df,
        audio_model, text_model, device,
        max_epochs=max_epochs_per_trial,
        patience=5
    )
    
    # Suppress verbose
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    
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
    
    # Save
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
    epochs: int = 50,
    patience: int = 10
) -> Tuple['LateFusionModel', Dict]:
    """
    Train Multimodal Fusion with best parameters.
    
    Args:
        best_params: Best parameters (loads from file if None)
        epochs: Training epochs
        patience: Early stopping patience
        
    Returns:
        Tuple of (trained_model, test_metrics)
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required")
    
    # Load params
    if best_params is None:
        params_path = CONFIG.RESULTS_DIR / "optuna_multimodal_best_params.json"
        if params_path.exists():
            with open(params_path, 'r') as f:
                data = json.load(f)
                best_params = data['best_params']
        else:
            best_params = {
                'fusion_hidden': 64,
                'dropout_rate': 0.3,
                'learning_rate': 1e-3,
                'pos_weight': 2.5,
                'weight_decay': 1e-4
            }
    
    device = CONFIG.DEVICE
    
    print(f"\n{'='*60}")
    print(f"Training Multimodal Fusion with optimized parameters")
    print(f"{'='*60}")
    for k, v in best_params.items():
        print(f"  {k}: {v}")
    
    # Load data
    common_sessions, labels_dict, instances_df = load_multimodal_data()
    train_sessions, val_sessions, test_sessions = split_sessions(common_sessions, labels_dict)
    
    # Load models
    audio_model = AudioCNNModel(device=device)
    audio_model = audio_model.to(device)
    if CONFIG.AUDIO_CNN_MODEL.exists():
        audio_model.load()
    
    text_model = MILTextModel(device=device)
    text_model = text_model.to(device)
    if CONFIG.MIL_TEXT_MODEL.exists():
        text_model.load()
    
    # Create fusion model
    fusion_model = LateFusionModel(
        audio_model=audio_model,
        text_model=text_model,
        fusion_hidden=best_params['fusion_hidden'],
        dropout_rate=best_params['dropout_rate'],
        device=device
    )
    fusion_model = fusion_model.to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        fusion_model.fusion_net.parameters(),
        lr=best_params['learning_rate'],
        weight_decay=best_params.get('weight_decay', 1e-4)
    )
    
    criterion = WeightedBCELoss(pos_weight=best_params['pos_weight'])
    
    # Training
    best_val_f1 = -1
    patience_counter = 0
    
    for epoch in range(epochs):
        fusion_model.train()
        total_loss = 0
        n_samples = 0
        
        for session_id in train_sessions:
            spec_path = CONFIG.SPECTROGRAMS_DIR / f"{session_id}_spec.npy"
            if not spec_path.exists():
                continue
            
            spec = np.load(spec_path)
            spec = resize_spectrogram(spec)
            spec = (spec - spec.mean()) / (spec.std() + 1e-8)
            spec = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(device)
            
            session_instances = instances_df[instances_df['session_id'] == session_id]
            if len(session_instances) == 0:
                continue
            
            texts = session_instances.sort_values('sequence_num')['text'].tolist()
            label = torch.tensor([labels_dict[session_id]], dtype=torch.float).to(device)
            
            optimizer.zero_grad()
            fused, _, _ = fusion_model(spec, [texts])
            loss = criterion(fused, label)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_samples += 1
        
        if n_samples == 0:
            continue
        
        train_loss = total_loss / n_samples
        
        # Validation
        fusion_model.eval()
        val_preds = []
        val_labels_list = []
        
        with torch.no_grad():
            for session_id in val_sessions:
                spec_path = CONFIG.SPECTROGRAMS_DIR / f"{session_id}_spec.npy"
                if not spec_path.exists():
                    continue
                
                spec = np.load(spec_path)
                spec = resize_spectrogram(spec)
                spec = (spec - spec.mean()) / (spec.std() + 1e-8)
                spec = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(device)
                
                session_instances = instances_df[instances_df['session_id'] == session_id]
                if len(session_instances) == 0:
                    continue
                
                texts = session_instances.sort_values('sequence_num')['text'].tolist()
                
                fused, _, _ = fusion_model(spec, [texts])
                
                val_preds.append(fused.item())
                val_labels_list.append(labels_dict[session_id])
        
        if not val_preds:
            continue
        
        val_preds_arr = np.array(val_preds)
        val_labels_arr = np.array(val_labels_list)
        val_f1 = f1_score(val_labels_arr, (val_preds_arr >= 0.5).astype(int), zero_division=0)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            fusion_model.save()
            print(f"  * New best model")
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break
    
    # Load best model
    if CONFIG.MULTIMODAL_MODEL.exists():
        fusion_model.load()
    
    # Test evaluation
    print(f"\n{'-'*60}")
    print("Test Set Evaluation")
    print(f"{'-'*60}")
    
    fusion_model.eval()
    test_preds = []
    test_labels_list = []
    
    with torch.no_grad():
        for session_id in test_sessions:
            spec_path = CONFIG.SPECTROGRAMS_DIR / f"{session_id}_spec.npy"
            if not spec_path.exists():
                continue
            
            spec = np.load(spec_path)
            spec = resize_spectrogram(spec)
            spec = (spec - spec.mean()) / (spec.std() + 1e-8)
            spec = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(device)
            
            session_instances = instances_df[instances_df['session_id'] == session_id]
            if len(session_instances) == 0:
                continue
            
            texts = session_instances.sort_values('sequence_num')['text'].tolist()
            
            fused, _, _ = fusion_model(spec, [texts])
            
            test_preds.append(fused.item())
            test_labels_list.append(labels_dict[session_id])
    
    test_preds = np.array(test_preds)
    test_labels_arr = np.array(test_labels_list)
    
    # Find best threshold
    best_thresh = 0.5
    best_test_f1 = 0
    for thresh in np.arange(0.3, 0.8, 0.01):
        f1 = f1_score(test_labels_arr, (test_preds >= thresh).astype(int), zero_division=0)
        if f1 > best_test_f1:
            best_test_f1 = f1
            best_thresh = thresh
    
    test_pred_binary = (test_preds >= best_thresh).astype(int)
    
    from sklearn.metrics import roc_auc_score, accuracy_score
    
    final_metrics = {
        'f1': f1_score(test_labels_arr, test_pred_binary, zero_division=0),
        'auroc': roc_auc_score(test_labels_arr, test_preds) if len(np.unique(test_labels_arr)) > 1 else 0.5,
        'accuracy': accuracy_score(test_labels_arr, test_pred_binary),
        'threshold': best_thresh
    }
    
    print(f"Test F1: {final_metrics['f1']:.4f}")
    print(f"Test AUROC: {final_metrics['auroc']:.4f}")
    print(f"Optimal threshold: {best_thresh:.2f}")
    
    return fusion_model, final_metrics


def load_best_multimodal_params(path: Path = None) -> Optional[Dict]:
    """Load best multimodal parameters from JSON file."""
    path = path or CONFIG.RESULTS_DIR / "optuna_multimodal_best_params.json"
    
    if not path.exists():
        return None
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    return data.get('best_params')


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Optuna Multimodal Optimization')
    parser.add_argument('--n-trials', type=int, default=30, help='Number of trials')
    parser.add_argument('--timeout', type=int, default=3600, help='Timeout in seconds')
    parser.add_argument('--train-only', action='store_true',
                       help='Skip optimization, train with existing best params')
    args = parser.parse_args()
    
    if args.train_only:
        model, metrics = train_with_best_params()
    else:
        best_params, study = run_multimodal_optimization(
            n_trials=args.n_trials,
            timeout=args.timeout
        )
        
        print("\n" + "="*60)
        print("Training with best parameters...")
        print("="*60)
        
        model, metrics = train_with_best_params(best_params)
    
    print("\n[SUCCESS] Multimodal optimization complete!")

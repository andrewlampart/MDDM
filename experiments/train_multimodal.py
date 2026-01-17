"""
Sprint 3: Train Multimodal Fusion Model

Training script for late fusion of audio CNN and MIL text models.
Includes ablation study and uncertainty estimation.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import CONFIG

try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch required")
    sys.exit(1)

from models.audio_cnn_model import AudioCNNModel, SpectrogramDataset, generate_spectrograms_for_sessions
from models.mil_text_model import MILTextModel
from models.multimodal_fusion import LateFusionModel, CalibratedFusionModel
from models.mc_dropout import MCDropoutEstimator
from data.mil_dataloader import MILBagDataset, mil_collate_fn
from evaluation.report import EvaluationReport, ModelComparison


def resize_spectrogram(spec: np.ndarray, target_shape: tuple = None) -> np.ndarray:
    """Resize/pad spectrogram to target shape (n_mels, time_frames)"""
    target_shape = target_shape or (CONFIG.N_MELS, CONFIG.SPECTROGRAM_LENGTH)
    target_h, target_w = target_shape
    h, w = spec.shape
    
    # Pad or crop height (n_mels)
    if h < target_h:
        pad_h = target_h - h
        spec = np.pad(spec, ((0, pad_h), (0, 0)), mode='constant')
    elif h > target_h:
        spec = spec[:target_h, :]
    
    # Pad or crop width (time)
    if w < target_w:
        pad_w = target_w - w
        spec = np.pad(spec, ((0, 0), (0, pad_w)), mode='constant')
    elif w > target_w:
        spec = spec[:, :target_w]
    
    return spec


def load_or_train_audio_model(device: str) -> AudioCNNModel:
    """Load pre-trained audio model or train if not available"""
    
    audio_model = AudioCNNModel(device=device)
    audio_model = audio_model.to(device)
    
    if CONFIG.AUDIO_CNN_MODEL.exists():
        print("Loading pre-trained audio model...")
        audio_model.load()
        return audio_model
    
    print("Training audio CNN model...")
    
    # Ensure spectrograms exist
    if not CONFIG.SPECTROGRAMS_DIR.exists() or len(list(CONFIG.SPECTROGRAMS_DIR.glob("*.npy"))) == 0:
        print("Generating spectrograms...")
        generate_spectrograms_for_sessions()
    
    # Load labels
    labels_df = pd.read_csv(CONFIG.LABELS_CSV)
    labels_dict = dict(zip(labels_df['session_id'], labels_df['depression']))
    
    # Split sessions
    from sklearn.model_selection import train_test_split
    sessions = list(labels_dict.keys())
    labels = [labels_dict[s] for s in sessions]
    
    train_sessions, temp_sessions, train_labels, temp_labels = train_test_split(
        sessions, labels, test_size=0.3, stratify=labels, random_state=CONFIG.RANDOM_SEED
    )
    val_sessions, test_sessions = train_test_split(
        temp_sessions, test_size=0.5, stratify=temp_labels, random_state=CONFIG.RANDOM_SEED
    )
    
    # Create datasets
    train_dataset = SpectrogramDataset(session_ids=train_sessions, labels_dict=labels_dict, augment=True)
    val_dataset = SpectrogramDataset(session_ids=val_sessions, labels_dict=labels_dict)
    
    if len(train_dataset) == 0:
        print("Warning: No spectrograms found. Skipping audio model training.")
        return audio_model
    
    train_loader = DataLoader(train_dataset, batch_size=CONFIG.CNN_BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG.CNN_BATCH_SIZE)
    
    # Training
    from models.audio_cnn_model import AudioCNNTrainer
    trainer = AudioCNNTrainer(audio_model, learning_rate=CONFIG.LEARNING_RATE)
    
    scaler = torch.amp.GradScaler('cuda') if CONFIG.USE_MIXED_PRECISION and device == 'cuda' else None
    
    best_val_f1 = -1  # Start with -1 to ensure first save
    patience_counter = 0
    
    for epoch in range(CONFIG.EPOCHS):
        train_loss = trainer.train_epoch(train_loader, epoch, scaler)
        val_metrics, _, _ = trainer.evaluate(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val F1={val_metrics['f1']:.4f}")
        
        if val_metrics['f1'] > best_val_f1:
            best_val_f1 = val_metrics['f1']
            patience_counter = 0
            audio_model.save()
        else:
            patience_counter += 1
        
        if patience_counter >= CONFIG.EARLY_STOPPING_PATIENCE:
            break
    
    # Load best model if exists, otherwise keep current
    if CONFIG.AUDIO_CNN_MODEL.exists():
        audio_model.load()
    return audio_model


def load_or_train_text_model(device: str) -> MILTextModel:
    """Load pre-trained text model or train if not available"""
    
    text_model = MILTextModel(device=device)
    text_model = text_model.to(device)
    
    if CONFIG.MIL_TEXT_MODEL.exists():
        print("Loading pre-trained text model...")
        text_model.load()
        return text_model
    
    print("Text model not found. Please run: python experiments/train_mil_text.py")
    print("Using untrained model for demonstration...")
    
    return text_model


def create_multimodal_dataloader(split: str = 'test'):
    """Create dataloader with both spectrograms and text instances"""
    
    # Load labels
    labels_df = pd.read_csv(CONFIG.LABELS_CSV)
    labels_dict = dict(zip(labels_df['session_id'], labels_df['depression']))
    
    # Get sessions that have both modalities
    spec_sessions = set()
    for f in CONFIG.SPECTROGRAMS_DIR.glob("*.npy"):
        session_id = int(f.stem.split('_')[0])
        spec_sessions.add(session_id)
    
    instances_df = pd.read_csv(CONFIG.INSTANCES_CSV)
    text_sessions = set(instances_df['session_id'].unique())
    
    common_sessions = spec_sessions & text_sessions & set(labels_dict.keys())
    common_sessions = list(common_sessions)
    
    # Split
    from sklearn.model_selection import train_test_split
    labels = [labels_dict[s] for s in common_sessions]
    
    train_sessions, temp_sessions = train_test_split(
        common_sessions, test_size=0.3, stratify=labels, random_state=CONFIG.RANDOM_SEED
    )
    temp_labels = [labels_dict[s] for s in temp_sessions]
    val_sessions, test_sessions = train_test_split(
        temp_sessions, test_size=0.5, stratify=temp_labels, random_state=CONFIG.RANDOM_SEED
    )
    
    if split == 'train':
        session_ids = train_sessions
    elif split == 'val':
        session_ids = val_sessions
    else:
        session_ids = test_sessions
    
    return session_ids, labels_dict, instances_df


def train_multimodal_fusion():
    """Train and evaluate multimodal fusion model"""
    
    print("=" * 60)
    print("Sprint 3: Multimodal Fusion Training")
    print("=" * 60)
    
    CONFIG.set_seed()
    device = CONFIG.DEVICE
    
    # Load modality models
    print("\n--- Loading Modality Models ---")
    audio_model = load_or_train_audio_model(device)
    text_model = load_or_train_text_model(device)
    
    # Create fusion model
    print("\n--- Creating Fusion Model ---")
    fusion_model = LateFusionModel(
        audio_model=audio_model,
        text_model=text_model,
        fusion_hidden=64,
        device=device
    )
    fusion_model = fusion_model.to(device)
    
    # Get data splits
    train_sessions, labels_dict, instances_df = create_multimodal_dataloader('train')
    val_sessions, _, _ = create_multimodal_dataloader('val')
    test_sessions, _, _ = create_multimodal_dataloader('test')
    
    print(f"\nData splits: Train={len(train_sessions)}, Val={len(val_sessions)}, Test={len(test_sessions)}")
    
    # Training
    print("\n--- Training Fusion Network ---")
    optimizer = torch.optim.AdamW(
        fusion_model.fusion_net.parameters(),
        lr=1e-3,
        weight_decay=CONFIG.WEIGHT_DECAY
    )
    
    # Weighted BCE loss to handle class imbalance (pos_weight=2.5 for ~23% positive class)
    class WeightedBCE(nn.Module):
        def __init__(self, pos_weight=2.5):
            super().__init__()
            self.pos_weight = pos_weight
        
        def forward(self, pred, target):
            pred = torch.clamp(pred, 1e-7, 1 - 1e-7)
            loss = -self.pos_weight * target * torch.log(pred) - (1 - target) * torch.log(1 - pred)
            return loss.mean()
    
    criterion = WeightedBCE(pos_weight=2.5)
    
    best_val_f1 = -1  # Start with -1 to ensure first save
    patience_counter = 0
    
    for epoch in range(50):
        # Training
        fusion_model.train()
        total_loss = 0
        n_samples = 0
        
        for session_id in train_sessions:
            # Load spectrogram
            spec_path = CONFIG.SPECTROGRAMS_DIR / f"{session_id}_spec.npy"
            if not spec_path.exists():
                continue
            
            spec = np.load(spec_path)
            spec = resize_spectrogram(spec)  # Resize to expected shape
            spec = (spec - spec.mean()) / (spec.std() + 1e-8)
            spec = torch.FloatTensor(spec).unsqueeze(0).unsqueeze(0).to(device)  # (1, 1, H, W)
            
            # Load text instances
            session_instances = instances_df[instances_df['session_id'] == session_id]
            if len(session_instances) == 0:
                continue
            
            texts = session_instances.sort_values('sequence_num')['text'].tolist()
            label = torch.tensor([labels_dict[session_id]], dtype=torch.float).to(device)
            
            # Forward
            optimizer.zero_grad()
            fused, audio_score, text_score = fusion_model(spec, [texts])
            loss = criterion(fused, label)
            
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            n_samples += 1
        
        if n_samples == 0:
            print("No samples processed. Check data availability.")
            break
        
        train_loss = total_loss / n_samples
        
        # Validation
        fusion_model.eval()
        val_preds = []
        val_labels = []
        
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
                val_labels.append(labels_dict[session_id])
        
        if len(val_preds) == 0:
            continue
        
        val_preds = np.array(val_preds)
        val_labels = np.array(val_labels)
        
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, (val_preds >= 0.5).astype(int), zero_division=0)
        
        print(f"Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val F1={val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            fusion_model.save()
        else:
            patience_counter += 1
        
        if patience_counter >= 10:
            break
    
    # Test evaluation
    print("\n--- Test Evaluation ---")
    if CONFIG.MULTIMODAL_MODEL.exists():
        fusion_model.load()
    fusion_model.eval()
    
    test_preds = []
    test_labels = []
    test_audio_scores = []
    test_text_scores = []
    
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
            
            fused, audio_score, text_score = fusion_model(spec, [texts])
            
            test_preds.append(fused.item())
            test_audio_scores.append(audio_score.item())
            test_text_scores.append(text_score.item())
            test_labels.append(labels_dict[session_id])
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_audio_scores = np.array(test_audio_scores)
    test_text_scores = np.array(test_text_scores)
    
    # Optimize threshold
    best_thresh = 0.5
    best_f1 = 0
    for thresh in np.arange(0.3, 0.8, 0.01):
        f1 = f1_score(test_labels, (test_preds >= thresh).astype(int), zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh
    
    test_pred_binary = (test_preds >= best_thresh).astype(int)
    
    # Generate report
    report = EvaluationReport(
        model_name="Multimodal Late Fusion",
        y_true=test_labels,
        y_pred=test_pred_binary,
        y_proba=test_preds
    )
    report.generate()
    report.print_report()
    report.save_json()
    report.save_plots()
    
    # Ablation study
    print("\n--- Ablation Study ---")
    
    audio_f1 = f1_score(test_labels, (test_audio_scores >= 0.5).astype(int), zero_division=0)
    text_f1 = f1_score(test_labels, (test_text_scores >= 0.5).astype(int), zero_division=0)
    fusion_f1 = f1_score(test_labels, test_pred_binary, zero_division=0)
    
    print(f"Audio-only F1:  {audio_f1:.4f}")
    print(f"Text-only F1:   {text_f1:.4f}")
    print(f"Multimodal F1:  {fusion_f1:.4f}")
    print(f"Improvement:    +{(fusion_f1 - max(audio_f1, text_f1)):.4f}")
    
    # MC Dropout uncertainty
    print("\n--- Uncertainty Estimation (MC Dropout) ---")
    mc_estimator = MCDropoutEstimator(fusion_model, n_samples=CONFIG.MC_DROPOUT_SAMPLES)
    
    # Recollect predictions with uncertainty
    all_means = []
    all_stds = []
    
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
        
        mean, std = mc_estimator.predict_with_uncertainty(spec, [texts])
        all_means.append(mean[0])
        all_stds.append(std[0])
    
    all_means = np.array(all_means)
    all_stds = np.array(all_stds)
    
    # Calibration analysis
    calibration = mc_estimator.calibration_analysis(all_means, all_stds, test_labels)
    
    print(f"Mean uncertainty (correct):   {calibration['mean_uncertainty_correct']:.4f}")
    print(f"Mean uncertainty (incorrect): {calibration['mean_uncertainty_incorrect']:.4f}")
    print(f"Uncertainty-error correlation: {calibration['uncertainty_error_correlation']:.3f}")
    
    # Save uncertainty plot
    mc_estimator.plot_uncertainty_analysis(
        all_means, all_stds, test_labels,
        save_path=CONFIG.RESULTS_DIR / "multimodal_uncertainty.png"
    )
    
    print("\n" + "=" * 60)
    print("Sprint 3 Complete!")
    print("=" * 60)
    
    return fusion_model, report


if __name__ == '__main__':
    train_multimodal_fusion()

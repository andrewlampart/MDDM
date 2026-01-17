"""
Audio CNN + LSTM Model for Depression Detection

Architecture:
1. 3 convolutional blocks with BatchNorm and MaxPool
2. LSTM on flattened conv output for temporal modeling
3. Classification head with dropout

Input: Mel-spectrogram (1, 128, 400) - (channels, n_mels, time_frames)
Output: Depression probability [0, 1]
"""

import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Audio CNN disabled.")


if TORCH_AVAILABLE:
    
    class SpectrogramDataset(Dataset):
        """Dataset for mel-spectrograms"""
        
        def __init__(self, 
                     spectrogram_dir: Path = None,
                     session_ids: List[int] = None,
                     labels_dict: Dict[int, int] = None,
                     target_shape: Tuple[int, int] = None,
                     augment: bool = False):
            """
            Args:
                spectrogram_dir: Directory with .npy spectrogram files
                session_ids: List of session IDs to include
                labels_dict: Mapping session_id -> label
                target_shape: (n_mels, time_frames) for resizing
                augment: Whether to apply augmentation
            """
            self.spectrogram_dir = Path(spectrogram_dir or CONFIG.SPECTROGRAMS_DIR)
            self.target_shape = target_shape or (CONFIG.N_MELS, CONFIG.SPECTROGRAM_LENGTH)
            self.augment = augment
            
            # Load labels if not provided
            if labels_dict is None:
                import pandas as pd
                labels_df = pd.read_csv(CONFIG.LABELS_CSV)
                labels_dict = dict(zip(labels_df['session_id'], labels_df['depression']))
            
            self.labels_dict = labels_dict
            
            # Find available spectrograms
            self.samples = []
            
            if session_ids is None:
                # Use all available
                for spec_file in self.spectrogram_dir.glob("*.npy"):
                    session_id = int(spec_file.stem.split('_')[0])
                    if session_id in labels_dict:
                        self.samples.append({
                            'session_id': session_id,
                            'path': spec_file,
                            'label': labels_dict[session_id]
                        })
            else:
                for session_id in session_ids:
                    spec_file = self.spectrogram_dir / f"{session_id}_spec.npy"
                    if spec_file.exists() and session_id in labels_dict:
                        self.samples.append({
                            'session_id': session_id,
                            'path': spec_file,
                            'label': labels_dict[session_id]
                        })
            
            print(f"SpectrogramDataset: {len(self.samples)} samples")
        
        def __len__(self):
            return len(self.samples)
        
        def __getitem__(self, idx):
            sample = self.samples[idx]
            
            # Load spectrogram
            spec = np.load(sample['path'])
            
            # Resize to target shape
            spec = self._resize_spectrogram(spec)
            
            # Augmentation
            if self.augment:
                spec = self._augment(spec)
            
            # Normalize
            spec = (spec - spec.mean()) / (spec.std() + 1e-8)
            
            # Add channel dimension
            spec = np.expand_dims(spec, axis=0)  # (1, H, W)
            
            return {
                'spectrogram': torch.FloatTensor(spec),
                'label': torch.tensor(sample['label'], dtype=torch.float),
                'session_id': sample['session_id']
            }
        
        def _resize_spectrogram(self, spec: np.ndarray) -> np.ndarray:
            """Resize spectrogram to target shape"""
            h, w = spec.shape
            target_h, target_w = self.target_shape
            
            # Pad or crop width (time dimension)
            if w < target_w:
                pad_width = target_w - w
                spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant',
                             constant_values=spec.min())
            elif w > target_w:
                # Random crop for augmentation, center crop otherwise
                if self.augment:
                    start = np.random.randint(0, w - target_w)
                else:
                    start = (w - target_w) // 2
                spec = spec[:, start:start + target_w]
            
            # Handle height if needed
            if h != target_h:
                from scipy.ndimage import zoom
                zoom_factor = (target_h / h, 1.0)
                spec = zoom(spec, zoom_factor, order=1)
            
            return spec
        
        def _augment(self, spec: np.ndarray) -> np.ndarray:
            """Apply augmentation"""
            # Time masking
            if np.random.random() < 0.5:
                t = np.random.randint(0, 40)
                t0 = np.random.randint(0, spec.shape[1] - t)
                spec[:, t0:t0+t] = spec.min()
            
            # Frequency masking
            if np.random.random() < 0.5:
                f = np.random.randint(0, 20)
                f0 = np.random.randint(0, spec.shape[0] - f)
                spec[f0:f0+f, :] = spec.min()
            
            # Small noise
            if np.random.random() < 0.3:
                noise = np.random.randn(*spec.shape) * 0.01
                spec = spec + noise
            
            return spec
    
    
    class AudioCNNModel(nn.Module):
        """
        CNN + LSTM model for spectrogram-based depression detection.
        
        Architecture:
        - 3 conv blocks: Conv2d -> BatchNorm -> ReLU -> MaxPool -> Dropout
        - Flatten -> LSTM for temporal modeling
        - Classification head with dropout
        """
        
        def __init__(self,
                     input_channels: int = 1,
                     n_mels: int = None,
                     time_frames: int = None,
                     hidden_dim: int = 256,
                     lstm_layers: int = 2,
                     dropout_rate: float = None,
                     device: str = None):
            """
            Args:
                input_channels: Number of input channels (1 for mono)
                n_mels: Number of mel bands
                time_frames: Number of time frames
                hidden_dim: Hidden dimension for LSTM and MLP
                lstm_layers: Number of LSTM layers
                dropout_rate: Dropout rate
                device: Device to use
            """
            super().__init__()
            
            self.n_mels = n_mels or CONFIG.N_MELS
            self.time_frames = time_frames or CONFIG.SPECTROGRAM_LENGTH
            self.hidden_dim = hidden_dim
            self.dropout_rate = dropout_rate or CONFIG.DROPOUT_RATE
            self.device = device or CONFIG.DEVICE
            
            # Conv Block 1
            self.conv1 = nn.Sequential(
                nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, kernel_size=3, padding=1),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.25)
            )
            
            # Conv Block 2
            self.conv2 = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, kernel_size=3, padding=1),
                nn.BatchNorm2d(64),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.25)
            )
            
            # Conv Block 3
            self.conv3 = nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout2d(0.25)
            )
            
            # Calculate flattened size after conv layers
            # After 3x MaxPool2d(2,2): (128, time/8, mels/8)
            conv_h = self.n_mels // 8
            conv_w = self.time_frames // 8
            self.conv_output_size = 128 * conv_h * conv_w
            
            # LSTM for temporal modeling
            self.lstm = nn.LSTM(
                input_size=self.conv_output_size,
                hidden_size=hidden_dim,
                num_layers=lstm_layers,
                dropout=self.dropout_rate if lstm_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
            
            # Classification head (outputs logits, no sigmoid - use BCEWithLogitsLoss)
            self.classifier = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            # Feature extractor (for late fusion)
            self.feature_dim = hidden_dim // 2
        
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            """
            Forward pass.
            
            Args:
                x: Spectrogram tensor (batch, 1, n_mels, time_frames)
                
            Returns:
                Depression probability (batch,)
            """
            batch_size = x.shape[0]
            
            # Conv layers
            x = self.conv1(x)  # (batch, 32, n_mels/2, time/2)
            x = self.conv2(x)  # (batch, 64, n_mels/4, time/4)
            x = self.conv3(x)  # (batch, 128, n_mels/8, time/8)
            
            # Flatten for LSTM
            x = x.view(batch_size, 1, -1)  # (batch, 1, conv_output_size)
            
            # LSTM
            lstm_out, (h_n, c_n) = self.lstm(x)
            x = h_n[-1]  # Last hidden state (batch, hidden_dim)
            
            # Classify
            output = self.classifier(x)  # (batch, 1)
            
            return output.squeeze(-1)  # (batch,)
        
        def get_features(self, x: torch.Tensor) -> torch.Tensor:
            """Get features before final classifier (for late fusion)"""
            batch_size = x.shape[0]
            
            # Conv layers
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            
            # Flatten for LSTM
            x = x.view(batch_size, 1, -1)
            
            # LSTM
            lstm_out, (h_n, c_n) = self.lstm(x)
            x = h_n[-1]
            
            # First layer of classifier (features)
            x = self.classifier[0](x)  # Linear
            x = self.classifier[1](x)  # ReLU
            
            return x  # (batch, hidden_dim//2)
        
        def save(self, path: Path = None):
            """Save model weights"""
            path = path or CONFIG.AUDIO_CNN_MODEL
            path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': {
                    'n_mels': self.n_mels,
                    'time_frames': self.time_frames,
                    'hidden_dim': self.hidden_dim,
                    'dropout_rate': self.dropout_rate,
                }
            }, path)
            print(f"Model saved to {path}")
        
        def load(self, path: Path = None):
            """Load model weights"""
            path = path or CONFIG.AUDIO_CNN_MODEL
            
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {path}")
            
            return self
    
    
    class AudioCNNTrainer:
        """Trainer for Audio CNN model"""
        
        def __init__(self, model: AudioCNNModel,
                    learning_rate: float = None,
                    weight_decay: float = None):
            self.model = model
            self.device = model.device
            
            learning_rate = learning_rate or CONFIG.LEARNING_RATE
            weight_decay = weight_decay or CONFIG.WEIGHT_DECAY
            
            self.optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
            
            # BCEWithLogitsLoss with pos_weight for class imbalance (77.3% / 22.7% ≈ 3.4)
            pos_weight = torch.tensor([3.4], device=model.device)
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
            
            self.history = {
                'train_loss': [], 'val_loss': [],
                'val_f1': [], 'val_auroc': []
            }
        
        def train_epoch(self, train_loader, epoch: int, scaler=None) -> float:
            """Train for one epoch"""
            self.model.train()
            total_loss = 0
            n_batches = 0
            
            for batch in train_loader:
                specs = batch['spectrogram'].to(self.device)
                labels = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                
                if scaler:
                    with torch.amp.autocast('cuda'):
                        logits = self.model(specs)
                        loss = self.criterion(logits, labels)  # BCEWithLogitsLoss is AMP-safe
                    
                    scaler.scale(loss).backward()
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                else:
                    logits = self.model(specs)
                    loss = self.criterion(logits, labels)
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            return total_loss / n_batches
        
        def evaluate(self, loader) -> Dict:
            """Evaluate on a dataset"""
            self.model.eval()
            
            all_preds = []
            all_labels = []
            total_loss = 0
            n_batches = 0
            
            with torch.no_grad():
                for batch in loader:
                    specs = batch['spectrogram'].to(self.device)
                    labels = batch['label'].to(self.device)
                    
                    logits = self.model(specs)
                    loss = self.criterion(logits, labels)
                    
                    total_loss += loss.item()
                    n_batches += 1
                    
                    # Apply sigmoid to logits for probability
                    probs = torch.sigmoid(logits)
                    all_preds.extend(probs.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
            
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            pred_binary = (all_preds >= 0.5).astype(int)
            
            from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
            
            metrics = {
                'loss': total_loss / n_batches,
                'accuracy': accuracy_score(all_labels, pred_binary),
                'f1': f1_score(all_labels, pred_binary, zero_division=0),
            }
            
            try:
                metrics['auroc'] = roc_auc_score(all_labels, all_preds)
            except:
                metrics['auroc'] = 0.5
            
            return metrics, all_preds, all_labels


def generate_spectrograms_for_sessions():
    """Generate spectrograms for all sessions if not already done"""
    import librosa
    import pandas as pd
    
    CONFIG.SPECTROGRAMS_DIR.mkdir(parents=True, exist_ok=True)
    
    labels_df = pd.read_csv(CONFIG.LABELS_CSV)
    
    generated = 0
    skipped = 0
    
    for _, row in labels_df.iterrows():
        session_id = row['session_id']
        audio_path = Path(row['audio_path'])
        output_path = CONFIG.SPECTROGRAMS_DIR / f"{session_id}_spec.npy"
        
        if output_path.exists():
            skipped += 1
            continue
        
        if not audio_path.exists():
            continue
        
        try:
            # Load audio
            y, sr = librosa.load(str(audio_path), sr=CONFIG.SAMPLE_RATE, mono=True)
            
            # Generate mel spectrogram
            S = librosa.feature.melspectrogram(
                y=y, sr=sr, n_mels=CONFIG.N_MELS, n_fft=2048, hop_length=512
            )
            S_db = librosa.power_to_db(S, ref=np.max)
            
            # Save
            np.save(output_path, S_db)
            generated += 1
            
        except Exception as e:
            print(f"Error processing {session_id}: {e}")
    
    print(f"Generated: {generated}, Skipped (existing): {skipped}")


if __name__ == '__main__':
    if not TORCH_AVAILABLE:
        print("PyTorch required")
    else:
        print("Testing Audio CNN Model...")
        
        model = AudioCNNModel(device='cpu')
        model.to('cpu')
        
        # Test with random input
        x = torch.randn(2, 1, 128, 400)
        output = model(x)
        print(f"Output shape: {output.shape}")
        print(f"Output: {output}")
        
        features = model.get_features(x)
        print(f"Features shape: {features.shape}")

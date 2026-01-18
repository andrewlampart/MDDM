"""
Fixed Multimodal Fusion Models for Depression Detection

Problems with original implementation:
- No regularization (dropout, weight decay)
- No batch normalization
- No class imbalance handling (pos_weight)
- No early stopping
- Model collapsed to MCC=0, Specificity=0

This module provides fixed versions with:
- BatchNorm for stable training
- Dropout for regularization
- Proper loss function with pos_weight
- Early stopping on F1 (not loss!)
- Gradient clipping
- L2 regularization via weight_decay
"""

import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from pathlib import Path
import logging
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

logger = logging.getLogger(__name__)

# Import config for GPU settings
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import CONFIG
    USE_BF16 = CONFIG.USE_BF16
    GPU_MEMORY_FRACTION = CONFIG.GPU_MEMORY_FRACTION
except ImportError:
    USE_BF16 = True
    GPU_MEMORY_FRACTION = 0.8


def clear_gpu_memory():
    """Clear GPU memory cache."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class LateFusionModelFixed(nn.Module):
    """
    Properly regularized late fusion model.
    
    Fixes:
    - Adds BatchNorm for stable training
    - Adds Dropout at each layer
    - Designed to work with BCEWithLogitsLoss + pos_weight
    - Outputs raw logits (not sigmoid) for numerical stability
    """
    
    def __init__(
        self,
        audio_dim: int = 42,
        text_dim: int = 768,
        hidden_dim: int = 64,
        dropout_audio: float = 0.3,
        dropout_text: float = 0.4,
        dropout_fusion: float = 0.2
    ):
        """
        Initialize the fixed late fusion model.
        
        Args:
            audio_dim: Dimension of audio features
            text_dim: Dimension of text features (e.g., 768 for BERT)
            hidden_dim: Hidden dimension for encoders
            dropout_audio: Dropout rate for audio encoder
            dropout_text: Dropout rate for text encoder (higher - more complex)
            dropout_fusion: Dropout rate for fusion classifier
        """
        super().__init__()
        
        # Audio encoder with regularization
        self.audio_encoder = nn.Sequential(
            nn.Linear(audio_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout_audio),
            nn.Linear(128, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_audio * 0.7)  # Slightly less at deeper layers
        )
        
        # Text encoder with regularization
        self.text_encoder = nn.Sequential(
            nn.Linear(text_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_text),
            nn.Linear(256, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_text * 0.5)
        )
        
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_fusion),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout_fusion * 0.5),
            nn.Linear(32, 1)
            # No sigmoid - use BCEWithLogitsLoss
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier/Glorot."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            audio_features: (batch_size, audio_dim)
            text_features: (batch_size, text_dim)
            
        Returns:
            logits: (batch_size,) - raw logits (apply sigmoid for probabilities)
        """
        audio_encoded = self.audio_encoder(audio_features)
        text_encoded = self.text_encoder(text_features)
        
        fused = torch.cat([audio_encoded, text_encoded], dim=1)
        logits = self.classifier(fused).squeeze(-1)
        
        return logits
    
    def predict_proba(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """Get probabilities (sigmoid applied to logits)."""
        logits = self.forward(audio_features, text_features)
        return torch.sigmoid(logits)


class EarlyFusionModelFixed(nn.Module):
    """
    Fixed early fusion model - simpler alternative.
    
    Concatenates features and uses single classifier.
    Good baseline when Late Fusion fails.
    """
    
    def __init__(
        self,
        audio_dim: int = 42,
        text_dim: int = 768,
        hidden_dims: List[int] = [256, 128, 64],
        dropout_rate: float = 0.3
    ):
        """
        Initialize early fusion model.
        
        Args:
            audio_dim: Audio feature dimension
            text_dim: Text feature dimension
            hidden_dims: List of hidden layer dimensions
            dropout_rate: Dropout rate
        """
        super().__init__()
        
        input_dim = audio_dim + text_dim
        
        layers = []
        prev_dim = input_dim
        
        for i, hidden_dim in enumerate(hidden_dims):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout_rate * (0.8 ** i))  # Decreasing dropout
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        self.classifier = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass."""
        combined = torch.cat([audio_features, text_features], dim=1)
        logits = self.classifier(combined).squeeze(-1)
        return logits


class FusionTrainer:
    """
    Trainer with all the fixes:
    - Class weighting (pos_weight)
    - L2 regularization (weight_decay)
    - Learning rate scheduling
    - Early stopping on F1
    - Gradient clipping
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_positive: int,
        n_negative: int,
        learning_rate: float = 0.001,
        weight_decay: float = 0.01,
        device: str = 'cpu',
        use_mixed_precision: bool = True
    ):
        """
        Initialize trainer with GPU memory management.
        
        Args:
            model: The fusion model to train
            n_positive: Number of positive samples in training set
            n_negative: Number of negative samples
            learning_rate: Initial learning rate
            weight_decay: L2 regularization strength
            device: 'cuda' or 'cpu'
            use_mixed_precision: Use bf16/fp16 for training (RTX 5060)
        """
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        
        # Setup GPU memory limit
        if device == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=0)
            except Exception as e:
                logger.warning(f"Could not set GPU memory fraction: {e}")
        
        self.model = model.to(device)
        
        # Class weight for imbalance
        pos_weight = n_negative / n_positive
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        
        # Optimizer with L2 regularization
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )
        
        # Learning rate scheduler
        self.scheduler = ReduceLROnPlateau(
            self.optimizer,
            mode='max',  # Maximize F1
            factor=0.5,
            patience=5
        )
        
        # Mixed precision scaler dla RTX 5060 (bf16)
        self.scaler = None
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
            self.amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
            logger.info(f"  Mixed precision: {self.amp_dtype}")
        
        logger.info(f"FusionTrainer initialized:")
        logger.info(f"  pos_weight: {pos_weight:.2f}")
        logger.info(f"  weight_decay: {weight_decay}")
        logger.info(f"  device: {device}")
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        max_grad_norm: float = 1.0
    ) -> float:
        """
        Train for one epoch with mixed precision support.
        
        Args:
            train_loader: DataLoader with (audio, text, labels)
            max_grad_norm: Maximum gradient norm for clipping
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            audio, text, labels = batch
            audio = audio.to(self.device)
            text = text.to(self.device)
            labels = labels.to(self.device).float()
            
            self.optimizer.zero_grad()
            
            # Mixed precision training
            if self.use_mixed_precision and self.scaler is not None:
                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    logits = self.model(audio, text)
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard precision training
                logits = self.model(audio, text)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), max_norm=max_grad_norm
                )
                self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        # Cleanup after epoch
        if self.device == 'cuda':
            clear_gpu_memory()
        
        return total_loss / n_batches
    
    def evaluate(
        self,
        val_loader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Returns:
            Dict with loss, f1, accuracy, specificity, sensitivity
        """
        self.model.eval()
        
        all_labels = []
        all_preds = []
        all_probs = []
        total_loss = 0.0
        n_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                audio, text, labels = batch
                audio = audio.to(self.device)
                text = text.to(self.device)
                labels = labels.to(self.device).float()
                
                logits = self.model(audio, text)
                loss = self.criterion(logits, labels)
                
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).long()
                
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                total_loss += loss.item()
                n_batches += 1
        
        all_labels = np.array(all_labels)
        all_preds = np.array(all_preds)
        
        # Compute metrics
        tp = ((all_labels == 1) & (all_preds == 1)).sum()
        tn = ((all_labels == 0) & (all_preds == 0)).sum()
        fp = ((all_labels == 0) & (all_preds == 1)).sum()
        fn = ((all_labels == 1) & (all_preds == 0)).sum()
        
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # MCC
        denom = np.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
        mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0
        
        return {
            'loss': total_loss / n_batches,
            'f1': f1,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'mcc': mcc,
            'tp': tp, 'tn': tn, 'fp': fp, 'fn': fn
        }
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        patience: int = 20,
        save_path: Optional[Path] = None,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop with early stopping on F1.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum epochs
            patience: Early stopping patience
            save_path: Path to save best model
            verbose: Print progress
            
        Returns:
            Training history dict
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_mcc': [],
            'val_specificity': []
        }
        
        best_f1 = 0.0
        best_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            # Train
            train_loss = self.train_epoch(train_loader)
            
            # Evaluate
            val_metrics = self.evaluate(val_loader)
            
            # Update history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_mcc'].append(val_metrics['mcc'])
            history['val_specificity'].append(val_metrics['specificity'])
            
            # Learning rate scheduling
            self.scheduler.step(val_metrics['f1'])
            
            # Early stopping on F1 (NOT loss!)
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_state = self.model.state_dict().copy()
                patience_counter = 0
                
                if save_path:
                    torch.save(best_state, save_path)
            else:
                patience_counter += 1
            
            # Logging
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val F1: {val_metrics['f1']:.4f} | "
                    f"Val MCC: {val_metrics['mcc']:.4f} | "
                    f"Val Spec: {val_metrics['specificity']:.4f}"
                )
            
            # Early stopping
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(f"Loaded best model with F1={best_f1:.4f}")
        
        return history


def create_fusion_model(
    audio_dim: int = 42,
    text_dim: int = 768,
    model_type: str = 'late',
    **kwargs
) -> nn.Module:
    """
    Factory function to create fusion models.
    
    Args:
        audio_dim: Audio feature dimension
        text_dim: Text feature dimension
        model_type: 'late' or 'early'
        **kwargs: Additional model arguments
        
    Returns:
        Fusion model
    """
    if model_type == 'late':
        return LateFusionModelFixed(
            audio_dim=audio_dim,
            text_dim=text_dim,
            **kwargs
        )
    elif model_type == 'early':
        return EarlyFusionModelFixed(
            audio_dim=audio_dim,
            text_dim=text_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


if __name__ == "__main__":
    print("Testing Fixed Fusion Models...")
    
    # Create dummy data
    batch_size = 32
    audio_dim = 42
    text_dim = 768
    
    audio = torch.randn(batch_size, audio_dim)
    text = torch.randn(batch_size, text_dim)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Test Late Fusion
    print("\n1. Testing LateFusionModelFixed:")
    model = LateFusionModelFixed(audio_dim=audio_dim, text_dim=text_dim)
    logits = model(audio, text)
    print(f"   Input: audio={audio.shape}, text={text.shape}")
    print(f"   Output logits: {logits.shape}")
    print(f"   [OK] LateFusionModelFixed works!")
    
    # Test Early Fusion
    print("\n2. Testing EarlyFusionModelFixed:")
    model_early = EarlyFusionModelFixed(audio_dim=audio_dim, text_dim=text_dim)
    logits_early = model_early(audio, text)
    print(f"   Output logits: {logits_early.shape}")
    print(f"   [OK] EarlyFusionModelFixed works!")
    
    # Test Trainer
    print("\n3. Testing FusionTrainer:")
    trainer = FusionTrainer(
        model=model,
        n_positive=10,
        n_negative=22,
        device='cpu'
    )
    print(f"   [OK] FusionTrainer initialized!")
    
    # Create simple dataset
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(audio, text, labels.float())
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # Single training step
    loss = trainer.train_epoch(loader)
    print(f"   Training loss: {loss:.4f}")
    
    # Evaluation
    metrics = trainer.evaluate(loader)
    print(f"   Val metrics: F1={metrics['f1']:.3f}, MCC={metrics['mcc']:.3f}")
    print(f"   [OK] Training works!")
    
    print("\n[OK] All fusion model tests passed!")

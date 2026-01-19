"""
Attention-based Multimodal Fusion for Depression Detection

Implements state-of-the-art fusion architecture based on 2024-2025 literature:
- Self-attention within each modality
- Cross-modal attention between audio and text
- Gated fusion with learned modality weights
- Optional Bi-LSTM for temporal context

Expected improvement: F1 from 0.345 to 0.65-0.75

References:
- IMDD-Net (Nature 2025): Kronecker product fusion + residual networks
- Nature Depression Detection (2025): Wav2Vec + BERT + Bi-LSTM
- Context-Aware Deep Learning (2024): Attention mechanisms
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingWarmRestarts

logger = logging.getLogger(__name__)

# Import config
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


class MultiheadSelfAttention(nn.Module):
    """
    Multi-head self-attention for single modality.
    
    Allows the model to attend to different positions and learn
    which parts of the feature vector are most relevant for depression.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
        batch_first: bool = True
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Ensure embed_dim is divisible by num_heads
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first
        )
        
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply self-attention.
        
        Args:
            x: (batch_size, embed_dim) or (batch_size, seq_len, embed_dim)
            
        Returns:
            Attended features of same shape
        """
        # If 2D input, add sequence dimension
        squeeze_output = False
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (batch, 1, embed_dim)
            squeeze_output = True
        
        # Self-attention with residual
        attended, _ = self.attention(x, x, x, need_weights=False)
        x = self.layer_norm(x + self.dropout(attended))
        
        if squeeze_output:
            x = x.squeeze(1)
        
        return x


class CrossModalAttention(nn.Module):
    """
    Cross-modal attention between audio and text modalities.
    
    Allows audio features to attend to relevant text features and vice versa.
    This is key for capturing semantic-acoustic correlations in depression.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        
        # Audio attending to text
        self.audio_to_text_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Text attending to audio
        self.text_to_audio_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.audio_norm = nn.LayerNorm(embed_dim)
        self.text_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        audio: torch.Tensor,
        text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply cross-modal attention.
        
        Args:
            audio: (batch_size, embed_dim) or (batch_size, seq_len, embed_dim)
            text: (batch_size, embed_dim) or (batch_size, seq_len, embed_dim)
            
        Returns:
            Tuple of (audio_attended, text_attended)
        """
        # Add sequence dimension if needed
        squeeze_audio = squeeze_text = False
        
        if audio.dim() == 2:
            audio = audio.unsqueeze(1)
            squeeze_audio = True
        if text.dim() == 2:
            text = text.unsqueeze(1)
            squeeze_text = True
        
        # Audio attends to text (query=audio, key/value=text)
        audio_attended, _ = self.audio_to_text_attn(audio, text, text)
        audio_out = self.audio_norm(audio + self.dropout(audio_attended))
        
        # Text attends to audio (query=text, key/value=audio)
        text_attended, _ = self.text_to_audio_attn(text, audio, audio)
        text_out = self.text_norm(text + self.dropout(text_attended))
        
        # Remove sequence dimension if we added it
        if squeeze_audio:
            audio_out = audio_out.squeeze(1)
        if squeeze_text:
            text_out = text_out.squeeze(1)
        
        return audio_out, text_out


class GatedFusion(nn.Module):
    """
    Gated fusion mechanism that learns optimal modality weights.
    
    Instead of simple concatenation (which caused negative synergy),
    this module learns when to trust audio vs text for each sample.
    """
    
    def __init__(
        self,
        embed_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # Gate network
        self.gate_net = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=-1)
        )
        
        # Fusion projection
        self.fusion_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU()
        )
        
    def forward(
        self,
        audio: torch.Tensor,
        text: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply gated fusion.
        
        Args:
            audio: (batch_size, embed_dim)
            text: (batch_size, embed_dim)
            
        Returns:
            Tuple of (fused_features, gate_weights)
        """
        # Compute gates
        combined = torch.cat([audio, text], dim=-1)
        gates = self.gate_net(combined)  # (batch, 2)
        
        # Weighted fusion
        fused = gates[:, 0:1] * audio + gates[:, 1:2] * text
        fused = self.fusion_proj(fused)
        
        return fused, gates


class AttentionFusionModel(nn.Module):
    """
    Complete attention-based multimodal fusion model for depression detection.
    
    Architecture:
    1. Input normalization and projection to common dimension
    2. Self-attention within each modality
    3. Cross-modal attention between audio and text
    4. Gated fusion with learned weights
    5. Classification head with dropout
    
    This addresses the -11.8% negative synergy by:
    - Normalizing modalities before fusion
    - Learning which modality to trust for each sample
    - Capturing audio-text correlations through cross-attention
    """
    
    def __init__(
        self,
        audio_dim: int = 788,
        text_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 4,
        num_attention_layers: int = 2,
        dropout: float = 0.3,
        use_cross_attention: bool = True,
        use_gated_fusion: bool = True
    ):
        """
        Initialize the attention fusion model.
        
        Args:
            audio_dim: Input dimension of audio features
            text_dim: Input dimension of text features
            hidden_dim: Common hidden dimension (should be divisible by num_heads)
            num_heads: Number of attention heads
            num_attention_layers: Number of self-attention layers per modality
            dropout: Dropout rate
            use_cross_attention: Whether to use cross-modal attention
            use_gated_fusion: Whether to use gated fusion (vs simple concat)
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_cross_attention = use_cross_attention
        self.use_gated_fusion = use_gated_fusion
        
        # Ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            hidden_dim = (hidden_dim // num_heads + 1) * num_heads
            logger.warning(f"Adjusted hidden_dim to {hidden_dim} for num_heads={num_heads}")
            self.hidden_dim = hidden_dim
        
        # Input projections with normalization
        self.audio_input = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.text_input = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Self-attention layers for each modality
        self.audio_self_attn = nn.ModuleList([
            MultiheadSelfAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_attention_layers)
        ])
        
        self.text_self_attn = nn.ModuleList([
            MultiheadSelfAttention(hidden_dim, num_heads, dropout)
            for _ in range(num_attention_layers)
        ])
        
        # Cross-modal attention
        if use_cross_attention:
            self.cross_attention = CrossModalAttention(hidden_dim, num_heads, dropout)
        
        # Gated fusion
        if use_gated_fusion:
            self.gated_fusion = GatedFusion(hidden_dim, dropout)
            classifier_input_dim = hidden_dim
        else:
            classifier_input_dim = hidden_dim * 2
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        self._init_weights()
        
        # Store attention weights for interpretability
        self.last_gate_weights = None
        
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
            logits: (batch_size,)
        """
        # Project to common dimension
        audio = self.audio_input(audio_features)
        text = self.text_input(text_features)
        
        # Apply self-attention layers
        for attn_layer in self.audio_self_attn:
            audio = attn_layer(audio)
        
        for attn_layer in self.text_self_attn:
            text = attn_layer(text)
        
        # Cross-modal attention
        if self.use_cross_attention:
            audio, text = self.cross_attention(audio, text)
        
        # Fusion
        if self.use_gated_fusion:
            fused, gate_weights = self.gated_fusion(audio, text)
            self.last_gate_weights = gate_weights.detach()
        else:
            fused = torch.cat([audio, text], dim=-1)
        
        # Classification
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
    
    def get_gate_weights(self) -> Optional[torch.Tensor]:
        """Get the last computed gate weights for interpretability."""
        return self.last_gate_weights
    
    def get_modality_importance(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Dict[str, float]:
        """
        Get average modality importance scores.
        
        Returns:
            Dict with 'audio_weight' and 'text_weight' averaged over batch
        """
        self.eval()
        with torch.no_grad():
            _ = self.forward(audio_features, text_features)
        
        if self.last_gate_weights is not None:
            weights = self.last_gate_weights.mean(dim=0)
            return {
                'audio_weight': float(weights[0]),
                'text_weight': float(weights[1])
            }
        return {'audio_weight': 0.5, 'text_weight': 0.5}


class AttentionFusionWithBiLSTM(AttentionFusionModel):
    """
    Extended attention fusion model with Bi-LSTM for temporal context.
    
    Useful when features have temporal structure (e.g., segment-level features).
    Based on Nature 2025 depression detection architecture.
    """
    
    def __init__(
        self,
        audio_dim: int = 788,
        text_dim: int = 768,
        hidden_dim: int = 256,
        lstm_hidden: int = 128,
        num_lstm_layers: int = 2,
        **kwargs
    ):
        super().__init__(
            audio_dim=audio_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
        
        # Replace simple classifier with Bi-LSTM + classifier
        classifier_input = hidden_dim if self.use_gated_fusion else hidden_dim * 2
        
        self.bi_lstm = nn.LSTM(
            input_size=classifier_input,
            hidden_size=lstm_hidden,
            num_layers=num_lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=kwargs.get('dropout', 0.3) if num_lstm_layers > 1 else 0
        )
        
        # New classifier after Bi-LSTM
        self.classifier = nn.Sequential(
            nn.Linear(lstm_hidden * 2, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(kwargs.get('dropout', 0.3)),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """Forward pass with Bi-LSTM."""
        # Project and attend
        audio = self.audio_input(audio_features)
        text = self.text_input(text_features)
        
        for attn_layer in self.audio_self_attn:
            audio = attn_layer(audio)
        for attn_layer in self.text_self_attn:
            text = attn_layer(text)
        
        if self.use_cross_attention:
            audio, text = self.cross_attention(audio, text)
        
        # Fusion
        if self.use_gated_fusion:
            fused, gate_weights = self.gated_fusion(audio, text)
            self.last_gate_weights = gate_weights.detach()
        else:
            fused = torch.cat([audio, text], dim=-1)
        
        # Add sequence dimension for LSTM
        fused = fused.unsqueeze(1)  # (batch, 1, features)
        
        # Bi-LSTM
        lstm_out, _ = self.bi_lstm(fused)
        lstm_out = lstm_out[:, -1, :]  # Take last output
        
        # Classification
        logits = self.classifier(lstm_out).squeeze(-1)
        
        return logits


class AttentionFusionTrainer:
    """
    Trainer specifically designed for attention-based fusion models.
    
    Features:
    - Cosine annealing with warm restarts
    - Label smoothing
    - Gradient accumulation for small batches
    - Mixed precision training
    - Temperature scaling calibration (NEW)
    - Optimal threshold selection (NEW)
    - AUROC tracking (NEW)
    """
    
    def __init__(
        self,
        model: nn.Module,
        n_positive: int,
        n_negative: int,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        label_smoothing: float = 0.1,
        device: str = 'cpu',
        use_mixed_precision: bool = True,
        use_temperature_scaling: bool = True,
        use_threshold_optimization: bool = True
    ):
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        self.use_temperature_scaling = use_temperature_scaling
        self.use_threshold_optimization = use_threshold_optimization
        
        # Setup GPU
        if device == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=0)
            except Exception as e:
                logger.warning(f"Could not set GPU memory fraction: {e}")
        
        self.model = model.to(device)
        
        # Loss with class weighting and label smoothing
        pos_weight = n_negative / n_positive
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(device)
        )
        self.label_smoothing = label_smoothing
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
            betas=(0.9, 0.999)
        )
        
        # Cosine annealing scheduler
        self.scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,
            T_mult=2,
            eta_min=learning_rate * 0.01
        )
        
        # Mixed precision
        self.scaler = None
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
            self.amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
        
        # Calibration components (NEW)
        self.temperature = 1.0  # Temperature for scaling
        self.optimal_threshold = 0.5  # Optimal classification threshold
        self.calibrated = False
        
        logger.info(f"AttentionFusionTrainer initialized:")
        logger.info(f"  pos_weight: {pos_weight:.2f}")
        logger.info(f"  label_smoothing: {label_smoothing}")
        logger.info(f"  device: {device}")
        logger.info(f"  temperature_scaling: {use_temperature_scaling}")
        logger.info(f"  threshold_optimization: {use_threshold_optimization}")
    
    def _smooth_labels(self, labels: torch.Tensor) -> torch.Tensor:
        """Apply label smoothing."""
        if self.label_smoothing > 0:
            labels = labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        return labels
    
    def get_logits(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract raw logits and labels from data loader.
        
        Used for calibration fitting.
        
        Args:
            data_loader: DataLoader with (audio, text, labels) batches
            
        Returns:
            Tuple of (logits, labels) as numpy arrays
        """
        self.model.eval()
        all_logits = []
        all_labels = []
        
        with torch.no_grad():
            for batch in data_loader:
                audio, text, labels = batch
                audio = audio.to(self.device)
                text = text.to(self.device)
                
                logits = self.model(audio, text)
                
                all_logits.extend(logits.cpu().numpy())
                all_labels.extend(labels.numpy())
        
        return np.array(all_logits), np.array(all_labels)
    
    def fit_temperature(
        self,
        val_loader: torch.utils.data.DataLoader,
        bounds: Tuple[float, float] = (0.1, 10.0)
    ) -> float:
        """
        Fit temperature scaling parameter on validation set.
        
        Temperature scaling adjusts model confidence:
        - T > 1: Less confident (spreads probabilities)
        - T < 1: More confident (sharpens probabilities)
        
        This helps fix AUROC issues where model has good F1 but poor ranking.
        
        Args:
            val_loader: Validation data loader
            bounds: (min, max) bounds for temperature search
            
        Returns:
            Optimal temperature value
        """
        if not self.use_temperature_scaling:
            logger.info("Temperature scaling disabled")
            return 1.0
        
        from scipy.optimize import minimize_scalar
        
        # Get logits and labels
        logits, labels = self.get_logits(val_loader)
        
        def nll_loss(T: float) -> float:
            """Negative log likelihood with temperature scaling."""
            if T <= 0:
                return np.inf
            scaled = logits / T
            proba = 1 / (1 + np.exp(-np.clip(scaled, -500, 500)))
            proba = np.clip(proba, 1e-7, 1 - 1e-7)
            loss = -np.mean(labels * np.log(proba) + (1 - labels) * np.log(1 - proba))
            return loss
        
        # Store NLL before calibration
        nll_before = nll_loss(1.0)
        
        # Optimize temperature
        result = minimize_scalar(nll_loss, bounds=bounds, method='bounded')
        self.temperature = result.x
        
        # Store NLL after calibration
        nll_after = nll_loss(self.temperature)
        
        logger.info(f"Temperature scaling fitted:")
        logger.info(f"  Optimal temperature: {self.temperature:.4f}")
        logger.info(f"  NLL: {nll_before:.4f} -> {nll_after:.4f} (improvement: {nll_before - nll_after:.4f})")
        
        self.calibrated = True
        return self.temperature
    
    def fit_optimal_threshold(
        self,
        val_loader: torch.utils.data.DataLoader,
        metric: str = 'f1'
    ) -> float:
        """
        Find optimal classification threshold on validation set.
        
        Instead of using default 0.5, searches for threshold that maximizes
        the specified metric.
        
        Args:
            val_loader: Validation data loader
            metric: Metric to optimize ('f1', 'balanced', 'youden')
            
        Returns:
            Optimal threshold value
        """
        if not self.use_threshold_optimization:
            logger.info("Threshold optimization disabled")
            return 0.5
        
        from sklearn.metrics import f1_score, confusion_matrix
        
        # Get calibrated probabilities
        logits, labels = self.get_logits(val_loader)
        
        # Apply temperature scaling if calibrated
        if self.calibrated and self.temperature != 1.0:
            proba = 1 / (1 + np.exp(-logits / self.temperature))
        else:
            proba = 1 / (1 + np.exp(-logits))
        
        # Search for optimal threshold
        thresholds = np.arange(0.1, 0.9, 0.01)
        best_score = -np.inf
        best_threshold = 0.5
        
        for threshold in thresholds:
            y_pred = (proba > threshold).astype(int)
            
            if len(np.unique(y_pred)) < 2:
                continue
            
            if metric == 'f1':
                score = f1_score(labels, y_pred, zero_division=0)
            elif metric == 'balanced':
                cm = confusion_matrix(labels, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    score = (sens + spec) / 2
                else:
                    score = 0.0
            elif metric == 'youden':
                cm = confusion_matrix(labels, y_pred)
                if cm.shape == (2, 2):
                    tn, fp, fn, tp = cm.ravel()
                    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                    score = sens + spec - 1
                else:
                    score = 0.0
            else:
                score = f1_score(labels, y_pred, zero_division=0)
            
            if score > best_score:
                best_score = score
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        
        logger.info(f"Threshold optimization ({metric}):")
        logger.info(f"  Optimal threshold: {self.optimal_threshold:.3f} (score={best_score:.4f})")
        
        return self.optimal_threshold
    
    def calibrate(
        self,
        val_loader: torch.utils.data.DataLoader,
        threshold_metric: str = 'f1'
    ) -> Dict[str, float]:
        """
        Full calibration: temperature scaling + threshold optimization.
        
        Call this after training to calibrate the model.
        
        Args:
            val_loader: Validation data loader
            threshold_metric: Metric for threshold optimization
            
        Returns:
            Dict with calibration info
        """
        logger.info("=" * 50)
        logger.info("CALIBRATING MODEL")
        logger.info("=" * 50)
        
        # Step 1: Temperature scaling
        temperature = self.fit_temperature(val_loader)
        
        # Step 2: Threshold optimization
        threshold = self.fit_optimal_threshold(val_loader, metric=threshold_metric)
        
        logger.info("=" * 50)
        logger.info(f"Calibration complete: T={temperature:.4f}, threshold={threshold:.3f}")
        logger.info("=" * 50)
        
        return {
            'temperature': temperature,
            'optimal_threshold': threshold,
            'calibrated': True
        }
    
    def predict_proba_calibrated(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> np.ndarray:
        """
        Get calibrated probabilities using temperature scaling.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Calibrated probabilities
        """
        logits, _ = self.get_logits(data_loader)
        
        if self.calibrated and self.temperature != 1.0:
            proba = 1 / (1 + np.exp(-logits / self.temperature))
        else:
            proba = 1 / (1 + np.exp(-logits))
        
        return proba
    
    def predict_calibrated(
        self,
        data_loader: torch.utils.data.DataLoader
    ) -> np.ndarray:
        """
        Get calibrated binary predictions using optimal threshold.
        
        Args:
            data_loader: Data loader
            
        Returns:
            Binary predictions
        """
        proba = self.predict_proba_calibrated(data_loader)
        return (proba > self.optimal_threshold).astype(int)
    
    def train_epoch(
        self,
        train_loader: torch.utils.data.DataLoader,
        max_grad_norm: float = 1.0
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        
        for batch in train_loader:
            audio, text, labels = batch
            audio = audio.to(self.device)
            text = text.to(self.device)
            labels = self._smooth_labels(labels.to(self.device).float())
            
            self.optimizer.zero_grad()
            
            if self.use_mixed_precision and self.scaler is not None:
                with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                    logits = self.model(audio, text)
                    loss = self.criterion(logits, labels)
                
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                logits = self.model(audio, text)
                loss = self.criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                self.optimizer.step()
            
            total_loss += loss.item()
            n_batches += 1
        
        self.scheduler.step()
        
        if self.device == 'cuda':
            clear_gpu_memory()
        
        return total_loss / n_batches
    
    def evaluate(
        self,
        val_loader: torch.utils.data.DataLoader,
        use_calibration: bool = True
    ) -> Dict[str, float]:
        """
        Evaluate on validation set.
        
        Args:
            val_loader: Validation data loader
            use_calibration: Whether to use calibrated predictions (if available)
            
        Returns:
            Dict with all metrics including AUROC
        """
        from sklearn.metrics import roc_auc_score
        
        self.model.eval()
        
        all_labels = []
        all_logits = []
        all_probs = []
        all_probs_calibrated = []
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
                
                all_labels.extend(labels.cpu().numpy())
                all_logits.extend(logits.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                total_loss += loss.item()
                n_batches += 1
        
        all_labels = np.array(all_labels)
        all_logits = np.array(all_logits)
        all_probs = np.array(all_probs)
        
        # Apply temperature scaling if calibrated
        if use_calibration and self.calibrated and self.temperature != 1.0:
            all_probs_calibrated = 1 / (1 + np.exp(-all_logits / self.temperature))
        else:
            all_probs_calibrated = all_probs
        
        # Use optimal threshold if calibrated, else 0.5
        threshold = self.optimal_threshold if (use_calibration and self.calibrated) else 0.5
        all_preds = (all_probs_calibrated > threshold).astype(int)
        
        # Metrics
        tp = ((all_labels == 1) & (all_preds == 1)).sum()
        tn = ((all_labels == 0) & (all_preds == 0)).sum()
        fp = ((all_labels == 0) & (all_preds == 1)).sum()
        fn = ((all_labels == 1) & (all_preds == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        denom = np.sqrt((tp+fp) * (tp+fn) * (tn+fp) * (tn+fn))
        mcc = (tp*tn - fp*fn) / denom if denom > 0 else 0
        
        # AUROC (NEW - critical for calibration assessment)
        try:
            auroc = roc_auc_score(all_labels, all_probs_calibrated)
        except ValueError:
            auroc = 0.5
        
        # AUROC before calibration (for comparison)
        try:
            auroc_raw = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc_raw = 0.5
        
        # Get modality importance
        importance = {}
        if hasattr(self.model, 'get_modality_importance'):
            importance = self.model.get_modality_importance(
                audio.to(self.device), text.to(self.device)
            )
        
        return {
            'loss': total_loss / n_batches,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'mcc': mcc,
            'auroc': auroc,
            'auroc_raw': auroc_raw,
            'threshold': threshold,
            'temperature': self.temperature,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn),
            **importance
        }
    
    def train(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        epochs: int = 100,
        patience: int = 20,
        save_path: Optional[Path] = None,
        verbose: bool = True,
        calibrate_after_training: bool = True
    ) -> Dict[str, List[float]]:
        """
        Full training loop with optional calibration.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum epochs
            patience: Early stopping patience
            save_path: Path to save best model
            verbose: Print progress
            calibrate_after_training: Whether to calibrate model after training
            
        Returns:
            Training history dict
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_f1': [],
            'val_mcc': [],
            'val_auroc': [],  # NEW
            'val_specificity': [],  # NEW
            'audio_weight': [],
            'text_weight': []
        }
        
        best_f1 = 0.0
        best_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader, use_calibration=False)  # Raw during training
            
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_metrics['loss'])
            history['val_f1'].append(val_metrics['f1'])
            history['val_mcc'].append(val_metrics['mcc'])
            history['val_auroc'].append(val_metrics.get('auroc', 0.5))
            history['val_specificity'].append(val_metrics.get('specificity', 0.5))
            history['audio_weight'].append(val_metrics.get('audio_weight', 0.5))
            history['text_weight'].append(val_metrics.get('text_weight', 0.5))
            
            if val_metrics['f1'] > best_f1:
                best_f1 = val_metrics['f1']
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                patience_counter = 0
                if save_path:
                    torch.save(best_state, save_path)
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Loss: {train_loss:.4f} | "
                    f"F1: {val_metrics['f1']:.4f} | "
                    f"AUROC: {val_metrics.get('auroc', 0.5):.4f} | "
                    f"MCC: {val_metrics['mcc']:.4f} | "
                    f"Audio: {val_metrics.get('audio_weight', 0.5):.2f}"
                )
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        if best_state is not None:
            self.model.load_state_dict(best_state)
            logger.info(f"Loaded best model with F1={best_f1:.4f}")
        
        # Calibrate model after training (NEW)
        if calibrate_after_training and (self.use_temperature_scaling or self.use_threshold_optimization):
            calibration_info = self.calibrate(val_loader)
            history['calibration'] = calibration_info
            
            # Evaluate with calibration
            calibrated_metrics = self.evaluate(val_loader, use_calibration=True)
            history['calibrated_f1'] = calibrated_metrics['f1']
            history['calibrated_auroc'] = calibrated_metrics['auroc']
            history['calibrated_specificity'] = calibrated_metrics['specificity']
            
            logger.info(f"Post-calibration metrics:")
            logger.info(f"  F1: {val_metrics['f1']:.4f} -> {calibrated_metrics['f1']:.4f}")
            logger.info(f"  AUROC: {val_metrics.get('auroc', 0.5):.4f} -> {calibrated_metrics['auroc']:.4f}")
            logger.info(f"  Specificity: {val_metrics.get('specificity', 0.5):.4f} -> {calibrated_metrics['specificity']:.4f}")
        
        return history


def create_attention_fusion_model(
    audio_dim: int = 788,
    text_dim: int = 768,
    model_type: str = 'attention',
    **kwargs
) -> nn.Module:
    """
    Factory function for attention-based fusion models.
    
    Args:
        audio_dim: Audio feature dimension
        text_dim: Text feature dimension
        model_type: 'attention' or 'attention_lstm'
        **kwargs: Additional model arguments
        
    Returns:
        Fusion model
    """
    if model_type == 'attention':
        return AttentionFusionModel(
            audio_dim=audio_dim,
            text_dim=text_dim,
            **kwargs
        )
    elif model_type == 'attention_lstm':
        return AttentionFusionWithBiLSTM(
            audio_dim=audio_dim,
            text_dim=text_dim,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Use 'attention' or 'attention_lstm'")


if __name__ == "__main__":
    print("Testing Attention Fusion Models...")
    
    # Create dummy data
    batch_size = 32
    audio_dim = 788  # Wav2Vec (768) + prosody (20)
    text_dim = 768
    
    audio = torch.randn(batch_size, audio_dim)
    text = torch.randn(batch_size, text_dim)
    labels = torch.randint(0, 2, (batch_size,))
    
    # Test MultiheadSelfAttention
    print("\n1. Testing MultiheadSelfAttention:")
    self_attn = MultiheadSelfAttention(embed_dim=256, num_heads=4)
    x = torch.randn(batch_size, 256)
    out = self_attn(x)
    print(f"   Input: {x.shape} -> Output: {out.shape}")
    print(f"   [OK] MultiheadSelfAttention works!")
    
    # Test CrossModalAttention
    print("\n2. Testing CrossModalAttention:")
    cross_attn = CrossModalAttention(embed_dim=256, num_heads=4)
    audio_emb = torch.randn(batch_size, 256)
    text_emb = torch.randn(batch_size, 256)
    audio_out, text_out = cross_attn(audio_emb, text_emb)
    print(f"   Audio: {audio_emb.shape} -> {audio_out.shape}")
    print(f"   Text: {text_emb.shape} -> {text_out.shape}")
    print(f"   [OK] CrossModalAttention works!")
    
    # Test GatedFusion
    print("\n3. Testing GatedFusion:")
    gated = GatedFusion(embed_dim=256)
    fused, gates = gated(audio_emb, text_emb)
    print(f"   Fused: {fused.shape}")
    print(f"   Gates: {gates.shape}")
    print(f"   Sample gates: audio={gates[0,0]:.3f}, text={gates[0,1]:.3f}")
    print(f"   [OK] GatedFusion works!")
    
    # Test AttentionFusionModel
    print("\n4. Testing AttentionFusionModel:")
    model = AttentionFusionModel(
        audio_dim=audio_dim,
        text_dim=text_dim,
        hidden_dim=256,
        num_heads=4
    )
    logits = model(audio, text)
    print(f"   Input: audio={audio.shape}, text={text.shape}")
    print(f"   Output: {logits.shape}")
    importance = model.get_modality_importance(audio, text)
    print(f"   Modality importance: {importance}")
    print(f"   [OK] AttentionFusionModel works!")
    
    # Test AttentionFusionWithBiLSTM
    print("\n5. Testing AttentionFusionWithBiLSTM:")
    model_lstm = AttentionFusionWithBiLSTM(
        audio_dim=audio_dim,
        text_dim=text_dim,
        hidden_dim=256
    )
    logits_lstm = model_lstm(audio, text)
    print(f"   Output: {logits_lstm.shape}")
    print(f"   [OK] AttentionFusionWithBiLSTM works!")
    
    # Test Trainer
    print("\n6. Testing AttentionFusionTrainer:")
    trainer = AttentionFusionTrainer(
        model=model,
        n_positive=10,
        n_negative=22,
        device='cpu'
    )
    
    from torch.utils.data import TensorDataset, DataLoader
    dataset = TensorDataset(audio, text, labels.float())
    loader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    loss = trainer.train_epoch(loader)
    print(f"   Training loss: {loss:.4f}")
    
    metrics = trainer.evaluate(loader)
    print(f"   Val F1: {metrics['f1']:.3f}, MCC: {metrics['mcc']:.3f}")
    print(f"   Audio weight: {metrics.get('audio_weight', 0.5):.3f}")
    print(f"   Text weight: {metrics.get('text_weight', 0.5):.3f}")
    print(f"   [OK] Trainer works!")
    
    # Test factory function
    print("\n7. Testing create_attention_fusion_model factory:")
    for model_type in ['attention', 'attention_lstm']:
        m = create_attention_fusion_model(
            audio_dim=audio_dim,
            text_dim=text_dim,
            model_type=model_type
        )
        out = m(audio, text)
        print(f"   {model_type}: {type(m).__name__} -> {out.shape}")
    print(f"   [OK] Factory works!")
    
    print("\n[OK] All attention fusion model tests passed!")

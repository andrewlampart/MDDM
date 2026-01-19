"""
Teacher-Student Knowledge Distillation for Multimodal Depression Detection

Based on SOTA 2024-2025 literature (Gan et al. 2025, F1=99.1%):
- Phase 1: Train separate "teacher" models for audio and text
- Phase 2: Train "student" fusion model with soft labels from teachers

Architecture:
    TeacherAudio: Wav2Vec features (768) -> MLP -> p(depressed)
    TeacherText: BERT features (768) -> MLP -> p(depressed)
    StudentFusion: Audio + Text -> Multi-Head Attention -> Fusion -> p(depressed)

Hybrid Loss:
    L = α * KL(student || teacher_ensemble) + (1-α) * BCE(student, y_true)
    where α = 0.7 (balance parameter)

Expected improvement: F1 from 0.52 to 0.85-0.93
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import gc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau
from torch.utils.data import DataLoader, TensorDataset

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


# =============================================================================
# TEACHER MODELS
# =============================================================================

class TeacherModel(nn.Module):
    """
    Single-modality teacher model.
    
    Simple but effective MLP architecture for classification.
    Trains on single modality (audio OR text) to provide soft labels.
    
    Architecture (based on Gan et al. 2025):
        Input (768) -> Linear(128) -> ReLU -> Dropout
                    -> Linear(64) -> ReLU -> Dropout
                    -> Linear(1) -> Sigmoid
    """
    
    def __init__(
        self,
        input_dim: int = 768,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.3
    ):
        """
        Initialize teacher model.
        
        Args:
            input_dim: Input feature dimension (768 for Wav2Vec/BERT)
            hidden_dims: List of hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights using Xavier."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input features (batch_size, input_dim)
            
        Returns:
            Probability of depression (batch_size,)
        """
        logits = self.mlp(x).squeeze(-1)
        return torch.sigmoid(logits)
    
    def get_logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get raw logits before sigmoid."""
        return self.mlp(x).squeeze(-1)


# =============================================================================
# STUDENT FUSION MODEL
# =============================================================================

class StudentFusionModel(nn.Module):
    """
    Multi-head attention fusion student model.
    
    Learns from both modalities AND soft labels from teachers.
    
    Architecture (based on Gan et al. 2025):
        1. Project audio/text to common dimension (256)
        2. Multi-Head Attention (audio queries, text keys/values)
        3. Fusion: [Attention_output; h_text] concatenation
        4. Classification MLP
    
    Key insight: Audio attends to text to find semantic-acoustic correlations.
    """
    
    def __init__(
        self,
        audio_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3,
        use_bidirectional_attention: bool = True
    ):
        """
        Initialize student fusion model.
        
        Args:
            audio_dim: Audio feature dimension
            text_dim: Text feature dimension
            hidden_dim: Common hidden dimension (must be divisible by num_heads)
            num_heads: Number of attention heads
            dropout: Dropout rate
            use_bidirectional_attention: If True, also text attends to audio
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.use_bidirectional = use_bidirectional_attention
        
        # Ensure hidden_dim is divisible by num_heads
        if hidden_dim % num_heads != 0:
            hidden_dim = (hidden_dim // num_heads + 1) * num_heads
            logger.warning(f"Adjusted hidden_dim to {hidden_dim} for num_heads={num_heads}")
            self.hidden_dim = hidden_dim
        
        # Projection layers to common dimension
        self.audio_proj = nn.Sequential(
            nn.LayerNorm(audio_dim),
            nn.Linear(audio_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        self.text_proj = nn.Sequential(
            nn.LayerNorm(text_dim),
            nn.Linear(text_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout * 0.5)
        )
        
        # Multi-head attention: audio queries, text keys/values
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Optional: bidirectional attention (text -> audio)
        if use_bidirectional_attention:
            self.reverse_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            )
            fusion_input_dim = hidden_dim * 4  # attended_audio + audio + attended_text + text
        else:
            fusion_input_dim = hidden_dim * 2  # attended_audio + text
        
        # Layer norms for attention outputs
        self.audio_norm = nn.LayerNorm(hidden_dim)
        self.text_norm = nn.LayerNorm(hidden_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(hidden_dim // 4, 1)
        )
        
        self._init_weights()
        
        # Store attention weights for interpretability
        self.last_attention_weights = None
    
    def _init_weights(self):
        """Initialize weights using Xavier."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            audio_features: Audio features (batch_size, audio_dim)
            text_features: Text features (batch_size, text_dim)
            
        Returns:
            Tuple of (probabilities, attention_weights)
        """
        # Project to common dimension
        h_audio = self.audio_proj(audio_features)  # (B, hidden_dim)
        h_text = self.text_proj(text_features)      # (B, hidden_dim)
        
        # Add sequence dimension for attention
        h_audio_seq = h_audio.unsqueeze(1)  # (B, 1, hidden_dim)
        h_text_seq = h_text.unsqueeze(1)    # (B, 1, hidden_dim)
        
        # Cross-attention: audio queries text
        attended_audio, attn_weights = self.cross_attention(
            h_audio_seq, h_text_seq, h_text_seq,
            need_weights=True
        )
        attended_audio = attended_audio.squeeze(1)  # (B, hidden_dim)
        attended_audio = self.audio_norm(attended_audio + h_audio)  # Residual
        
        self.last_attention_weights = attn_weights.detach()
        
        # Fusion
        if self.use_bidirectional:
            # Reverse attention: text queries audio
            attended_text, _ = self.reverse_attention(
                h_text_seq, h_audio_seq, h_audio_seq
            )
            attended_text = attended_text.squeeze(1)
            attended_text = self.text_norm(attended_text + h_text)
            
            # Concatenate all representations
            fused = torch.cat([attended_audio, h_audio, attended_text, h_text], dim=-1)
        else:
            # Simple concatenation
            fused = torch.cat([attended_audio, h_text], dim=-1)
        
        # Fusion layer
        fused = self.fusion(fused)
        
        # Classification
        logits = self.classifier(fused).squeeze(-1)
        proba = torch.sigmoid(logits)
        
        return proba, attn_weights
    
    def get_logits(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor
    ) -> torch.Tensor:
        """Get raw logits before sigmoid."""
        h_audio = self.audio_proj(audio_features)
        h_text = self.text_proj(text_features)
        
        h_audio_seq = h_audio.unsqueeze(1)
        h_text_seq = h_text.unsqueeze(1)
        
        attended_audio, _ = self.cross_attention(h_audio_seq, h_text_seq, h_text_seq)
        attended_audio = attended_audio.squeeze(1)
        attended_audio = self.audio_norm(attended_audio + h_audio)
        
        if self.use_bidirectional:
            attended_text, _ = self.reverse_attention(h_text_seq, h_audio_seq, h_audio_seq)
            attended_text = attended_text.squeeze(1)
            attended_text = self.text_norm(attended_text + h_text)
            fused = torch.cat([attended_audio, h_audio, attended_text, h_text], dim=-1)
        else:
            fused = torch.cat([attended_audio, h_text], dim=-1)
        
        fused = self.fusion(fused)
        logits = self.classifier(fused).squeeze(-1)
        
        return logits


# =============================================================================
# HYBRID LOSS FUNCTION
# =============================================================================

class HybridKDLoss(nn.Module):
    """
    Hybrid Knowledge Distillation Loss.
    
    Combines KL divergence (soft labels from teachers) with 
    cross-entropy (hard labels from ground truth).
    
    L_total = α * KL(p_student || p_teacher_ensemble) + (1-α) * BCE(p_student, y_true)
    
    Based on:
    - Hinton et al. (2015): Knowledge Distillation
    - Gan et al. (2025): Teacher-Student for Depression Detection
    """
    
    def __init__(
        self,
        alpha: float = 0.7,
        temperature: float = 1.0,
        label_smoothing: float = 0.1
    ):
        """
        Initialize hybrid loss.
        
        Args:
            alpha: Balance between KL (soft) and BCE (hard) loss
                   Higher alpha = more trust in teachers
            temperature: Temperature for softening probabilities
            label_smoothing: Label smoothing for BCE
        """
        super().__init__()
        
        self.alpha = alpha
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        
        logger.info(f"HybridKDLoss: α={alpha}, T={temperature}, smoothing={label_smoothing}")
    
    def forward(
        self,
        student_proba: torch.Tensor,
        teacher_proba: torch.Tensor,
        hard_labels: torch.Tensor,
        teacher_audio_proba: Optional[torch.Tensor] = None,
        teacher_text_proba: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute hybrid loss.
        
        Args:
            student_proba: Student predictions (batch_size,)
            teacher_proba: Ensemble teacher predictions (batch_size,)
            hard_labels: Ground truth labels (batch_size,)
            teacher_audio_proba: Optional audio teacher predictions
            teacher_text_proba: Optional text teacher predictions
            
        Returns:
            Tuple of (total_loss, loss_components_dict)
        """
        # Apply temperature scaling to soften distributions
        student_soft = student_proba / self.temperature
        teacher_soft = teacher_proba / self.temperature
        
        # Clamp probabilities to avoid log(0)
        eps = 1e-7
        student_soft = torch.clamp(student_soft, eps, 1 - eps)
        teacher_soft = torch.clamp(teacher_soft, eps, 1 - eps)
        
        # KL Divergence loss (soft targets)
        # For binary: KL = p * log(p/q) + (1-p) * log((1-p)/(1-q))
        kl_loss = teacher_soft * torch.log(teacher_soft / student_soft) + \
                  (1 - teacher_soft) * torch.log((1 - teacher_soft) / (1 - student_soft))
        kl_loss = kl_loss.mean()
        
        # BCE loss (hard targets) with label smoothing
        if self.label_smoothing > 0:
            smooth_labels = hard_labels * (1 - self.label_smoothing) + 0.5 * self.label_smoothing
        else:
            smooth_labels = hard_labels
        
        bce_loss = F.binary_cross_entropy(
            student_proba.clamp(eps, 1 - eps),
            smooth_labels.float()
        )
        
        # Combined loss
        total_loss = self.alpha * kl_loss + (1 - self.alpha) * bce_loss
        
        # Loss components for logging
        components = {
            'total': total_loss.item(),
            'kl': kl_loss.item(),
            'bce': bce_loss.item()
        }
        
        # Optional: agreement between teachers
        if teacher_audio_proba is not None and teacher_text_proba is not None:
            teacher_agreement = 1 - torch.abs(teacher_audio_proba - teacher_text_proba).mean()
            components['teacher_agreement'] = teacher_agreement.item()
        
        return total_loss, components


# =============================================================================
# TEACHER-STUDENT TRAINER
# =============================================================================

class TeacherStudentTrainer:
    """
    Complete training pipeline for Teacher-Student Knowledge Distillation.
    
    Two-phase training:
    1. Train teachers independently on each modality
    2. Train student with soft labels from teachers + hard labels
    
    Features:
    - Cosine annealing learning rate
    - Mixed precision training
    - Temperature scaling calibration
    - Threshold optimization
    - Early stopping
    """
    
    def __init__(
        self,
        audio_dim: int = 768,
        text_dim: int = 768,
        hidden_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.3,
        teacher_lr: float = 6.25e-4,
        student_lr: float = 1e-4,
        alpha: float = 0.7,
        temperature: float = 1.0,
        weight_decay: float = 0.01,
        device: str = 'cpu',
        use_mixed_precision: bool = True
    ):
        """
        Initialize trainer.
        
        Args:
            audio_dim: Audio feature dimension
            text_dim: Text feature dimension
            hidden_dim: Hidden dimension for student
            num_heads: Number of attention heads
            dropout: Dropout rate
            teacher_lr: Learning rate for teachers (SOTA: 6.25e-4)
            student_lr: Learning rate for student (SOTA: 1e-4)
            alpha: KL vs BCE balance (SOTA: 0.7)
            temperature: Temperature for KD
            weight_decay: Weight decay for optimizer
            device: 'cuda' or 'cpu'
            use_mixed_precision: Use AMP for training
        """
        self.device = device
        self.use_mixed_precision = use_mixed_precision and device == 'cuda'
        self.alpha = alpha
        
        # Initialize models
        self.teacher_audio = TeacherModel(input_dim=audio_dim, dropout=dropout).to(device)
        self.teacher_text = TeacherModel(input_dim=text_dim, dropout=dropout).to(device)
        self.student = StudentFusionModel(
            audio_dim=audio_dim,
            text_dim=text_dim,
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        ).to(device)
        
        # Optimizers
        self.teacher_audio_opt = torch.optim.AdamW(
            self.teacher_audio.parameters(),
            lr=teacher_lr,
            weight_decay=weight_decay
        )
        self.teacher_text_opt = torch.optim.AdamW(
            self.teacher_text.parameters(),
            lr=teacher_lr,
            weight_decay=weight_decay
        )
        self.student_opt = torch.optim.AdamW(
            self.student.parameters(),
            lr=student_lr,
            weight_decay=weight_decay
        )
        
        # Schedulers
        self.teacher_audio_scheduler = CosineAnnealingWarmRestarts(
            self.teacher_audio_opt, T_0=10, T_mult=2
        )
        self.teacher_text_scheduler = CosineAnnealingWarmRestarts(
            self.teacher_text_opt, T_0=10, T_mult=2
        )
        self.student_scheduler = CosineAnnealingWarmRestarts(
            self.student_opt, T_0=10, T_mult=2
        )
        
        # Loss functions
        self.teacher_criterion = nn.BCELoss()
        self.hybrid_loss = HybridKDLoss(alpha=alpha, temperature=temperature)
        
        # Mixed precision
        self.scaler = None
        if self.use_mixed_precision:
            self.scaler = torch.amp.GradScaler('cuda')
            self.amp_dtype = torch.bfloat16 if USE_BF16 else torch.float16
        
        # Calibration
        self.temperature_scaling = 1.0
        self.optimal_threshold = 0.5
        
        logger.info(f"TeacherStudentTrainer initialized:")
        logger.info(f"  Device: {device}")
        logger.info(f"  Teacher LR: {teacher_lr}, Student LR: {student_lr}")
        logger.info(f"  Alpha (KL weight): {alpha}")
        logger.info(f"  Hidden dim: {hidden_dim}, Heads: {num_heads}")
    
    def _compute_class_weight(self, labels: torch.Tensor) -> torch.Tensor:
        """Compute positive class weight for imbalanced data."""
        n_positive = labels.sum().item()
        n_negative = len(labels) - n_positive
        if n_positive > 0:
            return torch.tensor(n_negative / n_positive, device=self.device)
        return torch.tensor(1.0, device=self.device)
    
    def train_teachers(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 10,
        patience: int = 5,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Phase 1: Train teacher models independently.
        
        Args:
            train_loader: Training data (audio, text, labels)
            val_loader: Validation data
            epochs: Number of training epochs
            patience: Early stopping patience
            verbose: Print progress
            
        Returns:
            Training history dict
        """
        logger.info("=" * 50)
        logger.info("PHASE 1: Training Teacher Models")
        logger.info("=" * 50)
        
        history = {
            'audio_train_loss': [], 'audio_val_f1': [],
            'text_train_loss': [], 'text_val_f1': []
        }
        
        # Get class weight from first batch
        for batch in train_loader:
            _, _, labels = batch
            pos_weight = self._compute_class_weight(labels.to(self.device))
            break
        
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        best_audio_f1 = 0
        best_text_f1 = 0
        best_audio_state = None
        best_text_state = None
        patience_audio = patience_text = 0
        
        for epoch in range(epochs):
            # Train audio teacher
            self.teacher_audio.train()
            audio_loss = 0
            for batch in train_loader:
                audio, text, labels = [b.to(self.device) for b in batch]
                
                self.teacher_audio_opt.zero_grad()
                logits = self.teacher_audio.get_logits(audio)
                loss = criterion(logits, labels.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.teacher_audio.parameters(), 1.0)
                self.teacher_audio_opt.step()
                audio_loss += loss.item()
            
            # Train text teacher
            self.teacher_text.train()
            text_loss = 0
            for batch in train_loader:
                audio, text, labels = [b.to(self.device) for b in batch]
                
                self.teacher_text_opt.zero_grad()
                logits = self.teacher_text.get_logits(text)
                loss = criterion(logits, labels.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.teacher_text.parameters(), 1.0)
                self.teacher_text_opt.step()
                text_loss += loss.item()
            
            # Step schedulers
            self.teacher_audio_scheduler.step()
            self.teacher_text_scheduler.step()
            
            # Evaluate teachers
            audio_metrics = self._evaluate_teacher(self.teacher_audio, val_loader, 'audio')
            text_metrics = self._evaluate_teacher(self.teacher_text, val_loader, 'text')
            
            history['audio_train_loss'].append(audio_loss / len(train_loader))
            history['audio_val_f1'].append(audio_metrics['f1'])
            history['text_train_loss'].append(text_loss / len(train_loader))
            history['text_val_f1'].append(text_metrics['f1'])
            
            # Early stopping
            if audio_metrics['f1'] > best_audio_f1:
                best_audio_f1 = audio_metrics['f1']
                best_audio_state = {k: v.cpu().clone() for k, v in self.teacher_audio.state_dict().items()}
                patience_audio = 0
            else:
                patience_audio += 1
            
            if text_metrics['f1'] > best_text_f1:
                best_text_f1 = text_metrics['f1']
                best_text_state = {k: v.cpu().clone() for k, v in self.teacher_text.state_dict().items()}
                patience_text = 0
            else:
                patience_text += 1
            
            if verbose and (epoch + 1) % 2 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Audio F1: {audio_metrics['f1']:.3f} | "
                    f"Text F1: {text_metrics['f1']:.3f}"
                )
            
            # Stop if both teachers converged
            if patience_audio >= patience and patience_text >= patience:
                logger.info(f"Teachers converged at epoch {epoch+1}")
                break
        
        # Load best states
        if best_audio_state:
            self.teacher_audio.load_state_dict(best_audio_state)
        if best_text_state:
            self.teacher_text.load_state_dict(best_text_state)
        
        logger.info(f"Best Audio Teacher F1: {best_audio_f1:.3f}")
        logger.info(f"Best Text Teacher F1: {best_text_f1:.3f}")
        
        return history
    
    def _evaluate_teacher(
        self,
        model: nn.Module,
        data_loader: DataLoader,
        modality: str
    ) -> Dict[str, float]:
        """Evaluate a single teacher model."""
        model.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                audio, text, labels = [b.to(self.device) for b in batch]
                features = audio if modality == 'audio' else text
                probs = model(features)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        all_preds = (all_probs > 0.5).astype(int)
        
        # Compute metrics
        tp = ((all_labels == 1) & (all_preds == 1)).sum()
        fp = ((all_labels == 0) & (all_preds == 1)).sum()
        fn = ((all_labels == 1) & (all_preds == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        return {'f1': f1, 'precision': precision, 'recall': recall}
    
    def train_student(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        patience: int = 10,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Phase 2: Train student with knowledge distillation.
        
        Args:
            train_loader: Training data (audio, text, labels)
            val_loader: Validation data
            epochs: Number of training epochs
            patience: Early stopping patience
            verbose: Print progress
            
        Returns:
            Training history dict
        """
        logger.info("=" * 50)
        logger.info("PHASE 2: Training Student with Knowledge Distillation")
        logger.info("=" * 50)
        
        # Freeze teachers
        self.teacher_audio.eval()
        self.teacher_text.eval()
        for param in self.teacher_audio.parameters():
            param.requires_grad = False
        for param in self.teacher_text.parameters():
            param.requires_grad = False
        
        history = {
            'train_loss': [], 'train_kl': [], 'train_bce': [],
            'val_f1': [], 'val_auroc': [], 'val_mcc': []
        }
        
        best_f1 = 0
        best_state = None
        patience_counter = 0
        
        for epoch in range(epochs):
            self.student.train()
            epoch_loss = 0
            epoch_kl = 0
            epoch_bce = 0
            
            for batch in train_loader:
                audio, text, labels = [b.to(self.device) for b in batch]
                
                # Get soft labels from teachers
                with torch.no_grad():
                    teacher_audio_proba = self.teacher_audio(audio)
                    teacher_text_proba = self.teacher_text(text)
                    teacher_ensemble = (teacher_audio_proba + teacher_text_proba) / 2
                
                # Student forward
                self.student_opt.zero_grad()
                
                if self.use_mixed_precision and self.scaler is not None:
                    with torch.amp.autocast('cuda', dtype=self.amp_dtype):
                        student_proba, _ = self.student(audio, text)
                        loss, components = self.hybrid_loss(
                            student_proba, teacher_ensemble, labels,
                            teacher_audio_proba, teacher_text_proba
                        )
                    
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.student_opt)
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    self.scaler.step(self.student_opt)
                    self.scaler.update()
                else:
                    student_proba, _ = self.student(audio, text)
                    loss, components = self.hybrid_loss(
                        student_proba, teacher_ensemble, labels,
                        teacher_audio_proba, teacher_text_proba
                    )
                    
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
                    self.student_opt.step()
                
                epoch_loss += components['total']
                epoch_kl += components['kl']
                epoch_bce += components['bce']
            
            self.student_scheduler.step()
            
            n_batches = len(train_loader)
            history['train_loss'].append(epoch_loss / n_batches)
            history['train_kl'].append(epoch_kl / n_batches)
            history['train_bce'].append(epoch_bce / n_batches)
            
            # Evaluate student
            metrics = self.evaluate(val_loader)
            history['val_f1'].append(metrics['f1'])
            history['val_auroc'].append(metrics['auroc'])
            history['val_mcc'].append(metrics['mcc'])
            
            # Early stopping
            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_state = {k: v.cpu().clone() for k, v in self.student.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
            
            if verbose and (epoch + 1) % 5 == 0:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} | "
                    f"Loss: {epoch_loss/n_batches:.4f} (KL: {epoch_kl/n_batches:.4f}, BCE: {epoch_bce/n_batches:.4f}) | "
                    f"F1: {metrics['f1']:.3f} | AUROC: {metrics['auroc']:.3f}"
                )
            
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best state
        if best_state:
            self.student.load_state_dict(best_state)
        
        logger.info(f"Best Student F1: {best_f1:.3f}")
        
        # Clear GPU memory
        if self.device == 'cuda':
            clear_gpu_memory()
        
        return history
    
    def train_full(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        teacher_epochs: int = 10,
        student_epochs: int = 20,
        teacher_patience: int = 5,
        student_patience: int = 10,
        calibrate: bool = True,
        verbose: bool = True
    ) -> Dict[str, any]:
        """
        Full training pipeline (Phase 1 + Phase 2 + Calibration).
        
        Args:
            train_loader: Training data
            val_loader: Validation data
            teacher_epochs: Epochs for teacher training
            student_epochs: Epochs for student training
            teacher_patience: Early stopping for teachers
            student_patience: Early stopping for student
            calibrate: Whether to calibrate after training
            verbose: Print progress
            
        Returns:
            Complete training history
        """
        # Phase 1: Train teachers
        teacher_history = self.train_teachers(
            train_loader, val_loader,
            epochs=teacher_epochs,
            patience=teacher_patience,
            verbose=verbose
        )
        
        # Phase 2: Train student
        student_history = self.train_student(
            train_loader, val_loader,
            epochs=student_epochs,
            patience=student_patience,
            verbose=verbose
        )
        
        # Calibration
        calibration_info = {}
        if calibrate:
            calibration_info = self.calibrate(val_loader)
        
        return {
            'teacher_history': teacher_history,
            'student_history': student_history,
            'calibration': calibration_info
        }
    
    def calibrate(self, val_loader: DataLoader) -> Dict[str, float]:
        """
        Calibrate model using temperature scaling and threshold optimization.
        
        Args:
            val_loader: Validation data
            
        Returns:
            Calibration parameters
        """
        logger.info("Calibrating model...")
        
        # Get predictions
        all_labels = []
        all_probs = []
        
        self.student.eval()
        with torch.no_grad():
            for batch in val_loader:
                audio, text, labels = [b.to(self.device) for b in batch]
                probs, _ = self.student(audio, text)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Temperature scaling
        from scipy.optimize import minimize_scalar
        
        def nll_loss(T):
            if T <= 0:
                return np.inf
            scaled = all_probs / T
            scaled = np.clip(scaled, 1e-7, 1 - 1e-7)
            loss = -np.mean(all_labels * np.log(scaled) + (1 - all_labels) * np.log(1 - scaled))
            return loss
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature_scaling = result.x
        
        # Apply temperature scaling
        calibrated_probs = all_probs / self.temperature_scaling
        calibrated_probs = np.clip(calibrated_probs, 1e-7, 1 - 1e-7)
        
        # Threshold optimization
        from sklearn.metrics import f1_score
        
        best_f1 = 0
        best_threshold = 0.5
        for threshold in np.arange(0.1, 0.9, 0.01):
            preds = (calibrated_probs > threshold).astype(int)
            f1 = f1_score(all_labels, preds, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        
        logger.info(f"Calibration complete: T={self.temperature_scaling:.3f}, threshold={self.optimal_threshold:.3f}")
        
        return {
            'temperature': self.temperature_scaling,
            'threshold': self.optimal_threshold,
            'calibrated_f1': best_f1
        }
    
    def evaluate(
        self,
        data_loader: DataLoader,
        use_calibration: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate student model.
        
        Args:
            data_loader: Data to evaluate on
            use_calibration: Apply temperature scaling and threshold
            
        Returns:
            Dictionary of metrics
        """
        from sklearn.metrics import roc_auc_score, matthews_corrcoef
        
        self.student.eval()
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in data_loader:
                audio, text, labels = [b.to(self.device) for b in batch]
                probs, _ = self.student(audio, text)
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Apply calibration if requested
        if use_calibration:
            all_probs = all_probs / self.temperature_scaling
            all_probs = np.clip(all_probs, 1e-7, 1 - 1e-7)
            threshold = self.optimal_threshold
        else:
            threshold = 0.5
        
        all_preds = (all_probs > threshold).astype(int)
        
        # Metrics
        tp = ((all_labels == 1) & (all_preds == 1)).sum()
        tn = ((all_labels == 0) & (all_preds == 0)).sum()
        fp = ((all_labels == 0) & (all_preds == 1)).sum()
        fn = ((all_labels == 1) & (all_preds == 0)).sum()
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        try:
            auroc = roc_auc_score(all_labels, all_probs)
        except ValueError:
            auroc = 0.5
        
        mcc = matthews_corrcoef(all_labels, all_preds)
        
        return {
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'auroc': auroc,
            'mcc': mcc,
            'threshold': threshold,
            'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
        }
    
    def predict_proba(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        use_calibration: bool = True
    ) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            audio_features: Audio features
            text_features: Text features
            use_calibration: Apply temperature scaling
            
        Returns:
            Probability array
        """
        self.student.eval()
        with torch.no_grad():
            audio = audio_features.to(self.device)
            text = text_features.to(self.device)
            probs, _ = self.student(audio, text)
            probs = probs.cpu().numpy()
        
        if use_calibration:
            probs = probs / self.temperature_scaling
            probs = np.clip(probs, 1e-7, 1 - 1e-7)
        
        return probs
    
    def predict(
        self,
        audio_features: torch.Tensor,
        text_features: torch.Tensor,
        use_calibration: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get binary predictions.
        
        Args:
            audio_features: Audio features
            text_features: Text features
            use_calibration: Apply calibration
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        probs = self.predict_proba(audio_features, text_features, use_calibration)
        threshold = self.optimal_threshold if use_calibration else 0.5
        preds = (probs > threshold).astype(int)
        return preds, probs


# =============================================================================
# TESTING
# =============================================================================

if __name__ == "__main__":
    print("Testing Teacher-Student Knowledge Distillation...")
    
    # Create dummy data
    batch_size = 32
    audio_dim = 768
    text_dim = 768
    
    audio = torch.randn(batch_size, audio_dim)
    text = torch.randn(batch_size, text_dim)
    labels = torch.randint(0, 2, (batch_size,)).float()
    
    # Test TeacherModel
    print("\n1. Testing TeacherModel:")
    teacher = TeacherModel(input_dim=audio_dim)
    probs = teacher(audio)
    print(f"   Input: {audio.shape} -> Output: {probs.shape}")
    print(f"   Sample probs: {probs[:3].detach().numpy()}")
    print("   [OK] TeacherModel works!")
    
    # Test StudentFusionModel
    print("\n2. Testing StudentFusionModel:")
    student = StudentFusionModel(
        audio_dim=audio_dim,
        text_dim=text_dim,
        hidden_dim=256,
        num_heads=8
    )
    probs, attn = student(audio, text)
    print(f"   Audio: {audio.shape}, Text: {text.shape}")
    print(f"   Output probs: {probs.shape}")
    print(f"   Attention weights: {attn.shape}")
    print("   [OK] StudentFusionModel works!")
    
    # Test HybridKDLoss
    print("\n3. Testing HybridKDLoss:")
    loss_fn = HybridKDLoss(alpha=0.7)
    
    student_proba = torch.rand(batch_size)
    teacher_proba = torch.rand(batch_size)
    
    loss, components = loss_fn(student_proba, teacher_proba, labels)
    print(f"   Total loss: {loss.item():.4f}")
    print(f"   KL loss: {components['kl']:.4f}")
    print(f"   BCE loss: {components['bce']:.4f}")
    print("   [OK] HybridKDLoss works!")
    
    # Test TeacherStudentTrainer
    print("\n4. Testing TeacherStudentTrainer:")
    
    # Create data loaders
    from torch.utils.data import TensorDataset, DataLoader
    
    dataset = TensorDataset(audio, text, labels)
    train_loader = DataLoader(dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(dataset, batch_size=8)
    
    trainer = TeacherStudentTrainer(
        audio_dim=audio_dim,
        text_dim=text_dim,
        hidden_dim=256,
        num_heads=8,
        device='cpu'
    )
    
    # Quick training test
    history = trainer.train_full(
        train_loader, val_loader,
        teacher_epochs=2,
        student_epochs=2,
        calibrate=True,
        verbose=True
    )
    
    print(f"   Final student F1: {history['student_history']['val_f1'][-1]:.3f}")
    print(f"   Calibration: T={history['calibration']['temperature']:.3f}")
    print("   [OK] TeacherStudentTrainer works!")
    
    # Test prediction
    print("\n5. Testing Prediction:")
    preds, probs = trainer.predict(audio[:5], text[:5])
    print(f"   Predictions: {preds}")
    print(f"   Probabilities: {probs}")
    print("   [OK] Prediction works!")
    
    print("\n[OK] All Teacher-Student tests passed!")

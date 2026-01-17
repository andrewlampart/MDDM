"""
Multi-Instance Learning Text Model for Depression Detection

Based on the Nature paper methodology:
1. Each instance (patient response) gets embedded via RoBERTa
2. Instance-level MLP classifier scores each instance
3. MIL pooling aggregates instance scores to bag-level prediction

Parameters α (alpha) and β (beta) control the MIL decision:
- α: threshold for individual instance scores
- β: threshold for aggregated bag score
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import warnings

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModel
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch/Transformers not available. MIL model disabled.")


if TORCH_AVAILABLE:
    
    class MILTextModel(nn.Module):
        """
        Multi-Instance Learning model for depression detection from text.
        
        Architecture:
        1. RoBERTa encoder (frozen or fine-tunable)
        2. Instance-level MLP: embedding -> depression score per instance
        3. MIL pooling: aggregate instance scores to bag prediction
        
        Loss: Binary cross-entropy on bag-level predictions
        """
        
        def __init__(self,
                     model_name: str = None,
                     hidden_dim: int = 256,
                     dropout_rate: float = None,
                     alpha: float = None,
                     beta: float = None,
                     freeze_encoder: bool = True,
                     unfreeze_last_n_layers: int = 0,
                     pooling_strategy: str = 'attention',
                     device: str = None):
            """
            Args:
                model_name: HuggingFace model name (default: roberta-base)
                hidden_dim: Hidden dimension for MLP
                dropout_rate: Dropout rate
                alpha: Threshold for instance scores
                beta: Threshold for aggregated score
                freeze_encoder: Whether to freeze RoBERTa weights
                unfreeze_last_n_layers: Number of last encoder layers to unfreeze (0 = all frozen)
                pooling_strategy: 'max', 'mean', 'attention', or 'alpha_beta'
                device: Device to use
            """
            super().__init__()
            
            self.model_name = model_name or CONFIG.ROBERTA_MODEL_NAME
            self.hidden_dim = hidden_dim
            self.dropout_rate = dropout_rate or CONFIG.DROPOUT_RATE
            self.alpha = alpha if alpha is not None else CONFIG.MIL_ALPHA
            self.beta = beta if beta is not None else CONFIG.MIL_BETA
            self.freeze_encoder = freeze_encoder
            self.unfreeze_last_n_layers = unfreeze_last_n_layers
            self.pooling_strategy = pooling_strategy
            self.device = device or CONFIG.DEVICE
            
            # Load RoBERTa
            print(f"Loading {self.model_name}...")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.encoder = AutoModel.from_pretrained(self.model_name)
            self.encoder_dim = self.encoder.config.hidden_size  # 768 for roberta-base
            
            # Freeze encoder if specified
            if self.freeze_encoder:
                for param in self.encoder.parameters():
                    param.requires_grad = False
                
                # Optionally unfreeze last N layers for fine-tuning
                if self.unfreeze_last_n_layers > 0:
                    # RoBERTa has 12 layers
                    total_layers = len(self.encoder.encoder.layer)
                    for i in range(total_layers - self.unfreeze_last_n_layers, total_layers):
                        for param in self.encoder.encoder.layer[i].parameters():
                            param.requires_grad = True
                    print(f"Encoder: frozen except last {self.unfreeze_last_n_layers} layers")
                else:
                    print("Encoder weights fully frozen")
            
            # Instance-level MLP
            self.instance_mlp = nn.Sequential(
                nn.Linear(self.encoder_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim // 2, 1)
            )
            
            # Attention pooling (if used)
            if pooling_strategy == 'attention':
                self.attention_weights = nn.Sequential(
                    nn.Linear(hidden_dim // 2, hidden_dim // 4),
                    nn.Tanh(),
                    nn.Linear(hidden_dim // 4, 1)
                )
                # Get features before final layer
                self.feature_extractor = nn.Sequential(
                    nn.Linear(self.encoder_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(self.dropout_rate),
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                )
        
        def _encode_texts(self, texts: List[str], max_length: int = None) -> torch.Tensor:
            """
            Encode list of texts using RoBERTa.
            
            Args:
                texts: List of text strings
                max_length: Maximum sequence length
                
            Returns:
                Tensor of shape (n_texts, encoder_dim)
            """
            max_length = max_length or CONFIG.MAX_TEXT_LENGTH
            
            # Tokenize
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Encode
            with torch.no_grad() if self.freeze_encoder else torch.enable_grad():
                outputs = self.encoder(**inputs)
                # Use [CLS] token embedding
                embeddings = outputs.last_hidden_state[:, 0, :]
            
            return embeddings
        
        def forward_instances(self, texts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            Forward pass for a list of instances.
            
            Args:
                texts: List of instance texts
                
            Returns:
                Tuple of (instance_scores, instance_features)
            """
            # Encode
            embeddings = self._encode_texts(texts)  # (n_instances, encoder_dim)
            
            if self.pooling_strategy == 'attention':
                # Get features before final layer
                features = self.feature_extractor(embeddings)  # (n_instances, hidden_dim//2)
                # Get scores
                scores = self.instance_mlp[4:](features)  # Final layers
            else:
                features = embeddings
                scores = self.instance_mlp(embeddings)  # (n_instances, 1)
            
            scores = torch.sigmoid(scores.squeeze(-1))  # (n_instances,)
            
            return scores, features
        
        def forward(self, bag_instances: List[str], 
                   return_instance_scores: bool = False) -> torch.Tensor:
            """
            Forward pass for a single bag (interview).
            
            Args:
                bag_instances: List of instance texts from one interview
                return_instance_scores: Whether to return instance-level scores
                
            Returns:
                Bag-level prediction (and optionally instance scores)
            """
            # Get instance scores
            instance_scores, features = self.forward_instances(bag_instances)
            
            # MIL pooling
            if self.pooling_strategy == 'max':
                bag_score = instance_scores.max()
                
            elif self.pooling_strategy == 'mean':
                bag_score = instance_scores.mean()
                
            elif self.pooling_strategy == 'attention':
                # Attention weights
                attn_logits = self.attention_weights(features).squeeze(-1)  # (n_instances,)
                attn_weights = F.softmax(attn_logits, dim=0)  # (n_instances,)
                
                # Weighted sum of scores
                bag_score = (attn_weights * instance_scores).sum()
                
            elif self.pooling_strategy == 'alpha_beta':
                # Nature paper's α-β strategy
                n_instances = len(bag_instances)
                
                # Count instances above α threshold
                n_positive = (instance_scores > self.alpha).sum().float()
                proportion_positive = n_positive / n_instances
                
                # Mean score
                mean_score = instance_scores.mean()
                
                # Combine: bag is positive if both conditions met
                # Soft version for differentiability
                bag_score = torch.sigmoid(
                    (proportion_positive - self.beta) * 10 + 
                    (mean_score - self.alpha) * 10
                )
            else:
                bag_score = instance_scores.max()
            
            if return_instance_scores:
                return bag_score.unsqueeze(0), instance_scores
            else:
                return bag_score.unsqueeze(0)
        
        def forward_batch(self, batch_instances: List[List[str]],
                         return_instance_scores: bool = False):
            """
            Forward pass for a batch of bags.
            
            Args:
                batch_instances: List of bags, each bag is a list of instance texts
                return_instance_scores: Whether to return instance scores
                
            Returns:
                Batch predictions (and optionally instance scores per bag)
            """
            predictions = []
            all_instance_scores = []
            
            for bag_instances in batch_instances:
                if return_instance_scores:
                    pred, inst_scores = self.forward(bag_instances, return_instance_scores=True)
                    all_instance_scores.append(inst_scores)
                else:
                    pred = self.forward(bag_instances)
                
                predictions.append(pred)
            
            predictions = torch.cat(predictions, dim=0)  # (batch_size,)
            
            if return_instance_scores:
                return predictions, all_instance_scores
            return predictions
        
        def compute_loss(self, predictions: torch.Tensor, labels: torch.Tensor,
                        instance_scores: List[torch.Tensor] = None,
                        pos_weight: float = None) -> torch.Tensor:
            """
            Compute MIL loss with class weighting for imbalanced data.
            
            Args:
                predictions: Bag-level predictions (batch_size,)
                labels: Bag-level labels (batch_size,)
                instance_scores: Optional instance scores for regularization
                pos_weight: Weight for positive class (default: 3.4 for DAIC-WOZ imbalance)
                
            Returns:
                Loss value
            """
            # Default pos_weight based on DAIC-WOZ class imbalance
            # Reduced from 3.4 to 2.0 to prevent over-prediction of positive class
            if pos_weight is None:
                pos_weight = 2.0
            
            # Main BCE loss with class weights - disable autocast for BCE compatibility
            with torch.amp.autocast(device_type='cuda', enabled=False):
                labels = labels.float()
                predictions = predictions.float()
                
                # Weighted BCE: weight positive samples more heavily
                weights = torch.where(labels > 0.5, 
                                     torch.tensor(pos_weight, device=self.device),
                                     torch.tensor(1.0, device=self.device))
                loss = F.binary_cross_entropy(predictions, labels, weight=weights)
                
                # Optional: Instance-level regularization
                # Encourage positive bags to have at least some high-scoring instances
                if instance_scores is not None:
                    for i, (scores, label) in enumerate(zip(instance_scores, labels)):
                        if label > 0.5:  # Positive bag
                            # Encourage max score to be high
                            max_score = scores.max()
                            loss += 0.1 * F.binary_cross_entropy(max_score.float().unsqueeze(0), 
                                                                 torch.ones(1, device=self.device))
            
            return loss
        
        def save(self, path: Path = None):
            """Save model weights"""
            path = path or CONFIG.MIL_TEXT_MODEL
            path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'model_state_dict': self.state_dict(),
                'config': {
                    'model_name': self.model_name,
                    'hidden_dim': self.hidden_dim,
                    'dropout_rate': self.dropout_rate,
                    'alpha': self.alpha,
                    'beta': self.beta,
                    'pooling_strategy': self.pooling_strategy,
                }
            }, path)
            print(f"Model saved to {path}")
        
        def load(self, path: Path = None):
            """Load model weights"""
            path = path or CONFIG.MIL_TEXT_MODEL
            
            checkpoint = torch.load(path, map_location=self.device)
            self.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded from {path}")
            
            return self


    class MILTextTrainer:
        """Trainer for MIL Text Model"""
        
        def __init__(self, model: MILTextModel, 
                    learning_rate: float = None,
                    weight_decay: float = None):
            """
            Args:
                model: MILTextModel instance
                learning_rate: Learning rate
                weight_decay: Weight decay for regularization
            """
            self.model = model
            self.device = model.device
            
            learning_rate = learning_rate or CONFIG.MIL_LEARNING_RATE
            weight_decay = weight_decay or CONFIG.WEIGHT_DECAY
            
            # Only optimize unfrozen parameters
            trainable_params = [p for p in model.parameters() if p.requires_grad]
            self.optimizer = torch.optim.AdamW(
                trainable_params,
                lr=learning_rate,
                weight_decay=weight_decay
            )
            
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode='max', factor=0.5, patience=5
            )
            
            self.history = {
                'train_loss': [], 'val_loss': [],
                'val_f1': [], 'val_auroc': []
            }
        
        def train_epoch(self, train_loader, epoch: int) -> float:
            """Train for one epoch"""
            self.model.train()
            total_loss = 0
            n_batches = 0
            
            for batch in train_loader:
                bag_instances = batch['bag_instances']
                labels = batch['labels'].to(self.device)
                
                self.optimizer.zero_grad()
                
                # Forward
                predictions, instance_scores = self.model.forward_batch(
                    bag_instances, return_instance_scores=True
                )
                
                # Loss
                loss = self.model.compute_loss(predictions, labels, instance_scores)
                
                # Backward
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            self.history['train_loss'].append(avg_loss)
            
            return avg_loss
        
        def evaluate(self, loader) -> Dict:
            """Evaluate on a dataset"""
            self.model.eval()
            
            all_preds = []
            all_labels = []
            all_instance_scores = []
            total_loss = 0
            n_batches = 0
            
            with torch.no_grad():
                for batch in loader:
                    bag_instances = batch['bag_instances']
                    labels = batch['labels'].to(self.device)
                    
                    predictions, instance_scores = self.model.forward_batch(
                        bag_instances, return_instance_scores=True
                    )
                    
                    loss = self.model.compute_loss(predictions, labels)
                    total_loss += loss.item()
                    n_batches += 1
                    
                    all_preds.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    all_instance_scores.extend([s.cpu().numpy() for s in instance_scores])
            
            all_preds = np.array(all_preds)
            all_labels = np.array(all_labels)
            
            # Metrics
            from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
            
            pred_binary = (all_preds >= 0.5).astype(int)
            
            metrics = {
                'loss': total_loss / n_batches,
                'accuracy': accuracy_score(all_labels, pred_binary),
                'f1': f1_score(all_labels, pred_binary, zero_division=0),
            }
            
            try:
                metrics['auroc'] = roc_auc_score(all_labels, all_preds)
            except:
                metrics['auroc'] = 0.5
            
            return metrics, all_preds, all_labels, all_instance_scores


if __name__ == '__main__':
    if not TORCH_AVAILABLE:
        print("PyTorch required")
    else:
        print("Testing MIL Text Model...")
        
        model = MILTextModel(device='cpu')
        model.to('cpu')
        
        # Test with sample texts
        sample_bag = [
            "I feel tired all the time and have no energy.",
            "Nothing really makes me happy anymore.",
            "I've been sleeping too much lately."
        ]
        
        pred, scores = model(sample_bag, return_instance_scores=True)
        print(f"Bag prediction: {pred.item():.4f}")
        print(f"Instance scores: {scores.detach().numpy()}")

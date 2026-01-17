"""
Multimodal Fusion Models for Depression Detection

Fusion strategies:
1. Late Fusion: Separate models for audio/text, combine predictions
2. Early Calibrated Fusion: Learned weights for each modality
3. Feature Fusion: Concatenate features, single classifier
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
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. Fusion models disabled.")


if TORCH_AVAILABLE:
    
    class LateFusionModel(nn.Module):
        """
        Late Fusion: Separate models for audio and text, 
        combine their predictions through a fusion network.
        
        Audio model outputs: audio_score
        Text model outputs: text_score
        Fusion network: [audio_score, text_score] -> final_prediction
        """
        
        def __init__(self,
                     audio_model: nn.Module,
                     text_model: nn.Module,
                     fusion_hidden: int = 64,
                     dropout_rate: float = None,
                     device: str = None):
            """
            Args:
                audio_model: Pre-trained audio CNN model
                text_model: Pre-trained MIL text model
                fusion_hidden: Hidden dimension for fusion network
                dropout_rate: Dropout rate
                device: Device to use
            """
            super().__init__()
            
            self.audio_model = audio_model
            self.text_model = text_model
            self.dropout_rate = dropout_rate or CONFIG.DROPOUT_RATE
            self.device = device or CONFIG.DEVICE
            
            # Freeze modality models (train only fusion network)
            for param in self.audio_model.parameters():
                param.requires_grad = False
            for param in self.text_model.parameters():
                param.requires_grad = False
            
            # Fusion network
            # Input: [audio_score, text_score] = 2 values
            self.fusion_net = nn.Sequential(
                nn.Linear(2, fusion_hidden),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(fusion_hidden, fusion_hidden // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(fusion_hidden // 2, 1),
                nn.Sigmoid()
            )
        
        def forward(self, spectrograms: torch.Tensor, 
                   bag_instances: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            """
            Forward pass.
            
            Args:
                spectrograms: Audio spectrograms (batch, 1, n_mels, time)
                bag_instances: List of text instances per sample
                
            Returns:
                Tuple of (fused_output, audio_scores, text_scores)
            """
            batch_size = spectrograms.shape[0]
            
            # Audio forward (model returns logits, apply sigmoid for probabilities)
            self.audio_model.eval()
            with torch.no_grad():
                audio_logits = self.audio_model(spectrograms)
                audio_scores = torch.sigmoid(audio_logits)  # (batch,)
            
            # Text forward
            self.text_model.eval()
            text_scores_list = []
            with torch.no_grad():
                for instances in bag_instances:
                    text_score = self.text_model(instances)  # (1,)
                    text_scores_list.append(text_score)
            
            text_scores = torch.cat(text_scores_list, dim=0)  # (batch,)
            
            # Fusion
            combined = torch.stack([audio_scores, text_scores], dim=1)  # (batch, 2)
            fused_output = self.fusion_net(combined).squeeze(-1)  # (batch,)
            
            return fused_output, audio_scores, text_scores
        
        def save(self, path: Path = None):
            """Save fusion network weights"""
            path = path or CONFIG.MULTIMODAL_MODEL
            path.parent.mkdir(parents=True, exist_ok=True)
            
            torch.save({
                'fusion_state_dict': self.fusion_net.state_dict(),
            }, path)
            print(f"Fusion model saved to {path}")
        
        def load(self, path: Path = None):
            """Load fusion network weights"""
            path = path or CONFIG.MULTIMODAL_MODEL
            
            checkpoint = torch.load(path, map_location=self.device)
            self.fusion_net.load_state_dict(checkpoint['fusion_state_dict'])
            print(f"Fusion model loaded from {path}")
            
            return self
    
    
    class CalibratedFusionModel(nn.Module):
        """
        Calibrated Fusion: Learn optimal weights for each modality.
        
        final_score = w_audio * audio_score + w_text * text_score
        where w_audio + w_text = 1 (normalized via softmax)
        """
        
        def __init__(self,
                     audio_model: nn.Module,
                     text_model: nn.Module,
                     device: str = None):
            super().__init__()
            
            self.audio_model = audio_model
            self.text_model = text_model
            self.device = device or CONFIG.DEVICE
            
            # Freeze modality models
            for param in self.audio_model.parameters():
                param.requires_grad = False
            for param in self.text_model.parameters():
                param.requires_grad = False
            
            # Learnable weights (before softmax)
            self.w_audio_raw = nn.Parameter(torch.tensor(0.5))
            self.w_text_raw = nn.Parameter(torch.tensor(0.5))
        
        def get_weights(self) -> Tuple[float, float]:
            """Get normalized weights"""
            weights = F.softmax(torch.stack([self.w_audio_raw, self.w_text_raw]), dim=0)
            return weights[0].item(), weights[1].item()
        
        def forward(self, spectrograms: torch.Tensor,
                   bag_instances: List[List[str]]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict]:
            """Forward pass"""
            # Get modality scores
            self.audio_model.eval()
            with torch.no_grad():
                audio_scores = self.audio_model(spectrograms)
            
            self.text_model.eval()
            text_scores_list = []
            with torch.no_grad():
                for instances in bag_instances:
                    text_scores_list.append(self.text_model(instances))
            
            text_scores = torch.cat(text_scores_list, dim=0)
            
            # Weighted combination
            weights = F.softmax(torch.stack([self.w_audio_raw, self.w_text_raw]), dim=0)
            w_audio, w_text = weights[0], weights[1]
            
            fused_output = w_audio * audio_scores + w_text * text_scores
            
            weight_info = {
                'w_audio': w_audio.item(),
                'w_text': w_text.item()
            }
            
            return fused_output, audio_scores, text_scores, weight_info
    
    
    class FeatureFusionModel(nn.Module):
        """
        Feature Fusion: Concatenate features from audio and text models,
        then use a single classifier.
        """
        
        def __init__(self,
                     audio_model: nn.Module,
                     text_model: nn.Module,
                     audio_feature_dim: int = 128,
                     text_feature_dim: int = 128,
                     hidden_dim: int = 128,
                     dropout_rate: float = None,
                     device: str = None):
            super().__init__()
            
            self.audio_model = audio_model
            self.text_model = text_model
            self.device = device or CONFIG.DEVICE
            self.dropout_rate = dropout_rate or CONFIG.DROPOUT_RATE
            
            # Freeze modality models
            for param in self.audio_model.parameters():
                param.requires_grad = False
            for param in self.text_model.parameters():
                param.requires_grad = False
            
            # Feature dimensions
            fused_dim = audio_feature_dim + text_feature_dim
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(fused_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout_rate),
                nn.Linear(hidden_dim // 2, 1),
                nn.Sigmoid()
            )
        
        def forward(self, spectrograms: torch.Tensor,
                   bag_instances: List[List[str]]) -> torch.Tensor:
            """Forward pass"""
            # Get features from audio model
            self.audio_model.eval()
            with torch.no_grad():
                audio_features = self.audio_model.get_features(spectrograms)
            
            # Get features from text model (need to implement get_features in MIL)
            # For now, use the instance-level features averaged
            self.text_model.eval()
            text_features_list = []
            with torch.no_grad():
                for instances in bag_instances:
                    _, features = self.text_model.forward_instances(instances)
                    # Average pooling of instance features
                    text_feat = features.mean(dim=0, keepdim=True)
                    text_features_list.append(text_feat)
            
            text_features = torch.cat(text_features_list, dim=0)
            
            # Concatenate
            fused_features = torch.cat([audio_features, text_features], dim=1)
            
            # Classify
            output = self.classifier(fused_features).squeeze(-1)
            
            return output

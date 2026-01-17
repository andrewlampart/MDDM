"""CNN model for spectrogram-based depression detection

Zgodnie z IRP: "CNN architectures for audiogram analysis"
Ten moduł implementuje CNN do analizy mel-spektrogramów.
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, List
import pickle
import warnings

# PyTorch imports
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. CNN models will be disabled.")

from ..utils.config import Config


if TORCH_AVAILABLE:
    
    class SpectrogramDataset(Dataset):
        """Dataset dla spektrogramów"""
        
        def __init__(self, spectrograms: List[np.ndarray], labels: np.ndarray,
                     target_shape: Tuple[int, int] = (128, 256)):
            """
            Args:
                spectrograms: Lista mel-spektrogramów
                labels: Array z etykietami (0/1)
                target_shape: Docelowy rozmiar (n_mels, time_frames)
            """
            self.spectrograms = spectrograms
            self.labels = labels
            self.target_shape = target_shape
        
        def __len__(self):
            return len(self.labels)
        
        def __getitem__(self, idx):
            spec = self.spectrograms[idx]
            label = self.labels[idx]
            
            # Resize/pad to target shape
            spec = self._resize_spectrogram(spec, self.target_shape)
            
            # Add channel dimension
            spec = np.expand_dims(spec, axis=0)  # (1, H, W)
            
            return torch.FloatTensor(spec), torch.LongTensor([label])[0]
        
        def _resize_spectrogram(self, spec: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
            """Resize spektrogramu do docelowego rozmiaru"""
            h, w = spec.shape
            target_h, target_w = target_shape
            
            # Pad or truncate width (time dimension)
            if w < target_w:
                # Pad with minimum value
                pad_width = target_w - w
                spec = np.pad(spec, ((0, 0), (0, pad_width)), mode='constant',
                             constant_values=spec.min())
            elif w > target_w:
                # Take center crop
                start = (w - target_w) // 2
                spec = spec[:, start:start + target_w]
            
            # Height should be constant (n_mels), but handle edge cases
            if h != target_h:
                # Simple resize using interpolation
                from scipy.ndimage import zoom
                zoom_factor = (target_h / h, target_w / spec.shape[1])
                spec = zoom(spec, zoom_factor, order=1)
            
            return spec
    
    
    class DepressionCNN(nn.Module):
        """CNN dla detekcji depresji na podstawie spektrogramów
        
        Architektura inspirowana VGGNet - prosta ale skuteczna.
        """
        
        def __init__(self, input_channels: int = 1, n_classes: int = 2,
                     dropout: float = 0.5):
            super(DepressionCNN, self).__init__()
            
            # Convolutional layers
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
            
            # Global average pooling
            self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
            
            # Fully connected layers
            self.fc = nn.Sequential(
                nn.Linear(128, 64),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(64, n_classes)
            )
        
        def forward(self, x):
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = self.fc(x)
            
            return x
        
        def get_features(self, x):
            """Zwróć features przed klasyfikatorem (dla late fusion)"""
            x = self.conv1(x)
            x = self.conv2(x)
            x = self.conv3(x)
            
            x = self.global_pool(x)
            x = x.view(x.size(0), -1)
            
            # Return features before final classifier
            x = self.fc[0](x)  # Linear 128->64
            x = self.fc[1](x)  # ReLU
            
            return x  # Shape: (batch, 64)


class CNNTrainer:
    """Trener dla CNN modelu"""
    
    def __init__(self, device: str = None, lr: float = 0.001,
                 batch_size: int = 16, n_epochs: int = 50):
        """
        Args:
            device: 'cuda' lub 'cpu'
            lr: Learning rate
            batch_size: Rozmiar batcha
            n_epochs: Liczba epok
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for CNN training")
        
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = lr
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        
        self.model = None
        self.history = {'train_loss': [], 'val_loss': [], 'val_f1': []}
    
    def train(self, train_specs: List[np.ndarray], train_labels: np.ndarray,
              val_specs: List[np.ndarray] = None, val_labels: np.ndarray = None,
              early_stopping_patience: int = 10) -> Dict:
        """Trenuj model CNN
        
        Args:
            train_specs: Lista spektrogramów treningowych
            train_labels: Etykiety treningowe
            val_specs: Spektrogramy walidacyjne (opcjonalne)
            val_labels: Etykiety walidacyjne (opcjonalne)
            early_stopping_patience: Liczba epok bez poprawy przed zatrzymaniem
            
        Returns:
            Historia treningu
        """
        # Create datasets
        train_dataset = SpectrogramDataset(train_specs, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size,
                                 shuffle=True, num_workers=0)
        
        val_loader = None
        if val_specs is not None and val_labels is not None:
            val_dataset = SpectrogramDataset(val_specs, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size,
                                   shuffle=False, num_workers=0)
        
        # Initialize model
        self.model = DepressionCNN().to(self.device)
        
        # Class weights for imbalanced data
        n_pos = train_labels.sum()
        n_neg = len(train_labels) - n_pos
        weights = torch.FloatTensor([n_pos / len(train_labels), n_neg / len(train_labels)]).to(self.device)
        
        criterion = nn.CrossEntropyLoss(weight=weights)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=5)
        
        best_val_f1 = 0
        patience_counter = 0
        best_model_state = None
        
        print(f"Training CNN on {self.device}...")
        print(f"Train: {len(train_labels)} samples, Positive: {n_pos} ({n_pos/len(train_labels):.1%})")
        
        for epoch in range(self.n_epochs):
            # Training
            self.model.train()
            train_loss = 0
            
            for batch_specs, batch_labels in train_loader:
                batch_specs = batch_specs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_specs)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
            
            train_loss /= len(train_loader)
            self.history['train_loss'].append(train_loss)
            
            # Validation
            if val_loader is not None:
                val_loss, val_f1 = self._evaluate(val_loader, criterion)
                self.history['val_loss'].append(val_loss)
                self.history['val_f1'].append(val_f1)
                
                scheduler.step(val_f1)
                
                # Early stopping
                if val_f1 > best_val_f1:
                    best_val_f1 = val_f1
                    patience_counter = 0
                    best_model_state = self.model.state_dict().copy()
                else:
                    patience_counter += 1
                
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs}: "
                          f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val F1: {val_f1:.4f}")
                
                if patience_counter >= early_stopping_patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
            else:
                if (epoch + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.n_epochs}: Train Loss: {train_loss:.4f}")
        
        # Load best model
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)
            print(f"Loaded best model with Val F1: {best_val_f1:.4f}")
        
        return self.history
    
    def _evaluate(self, loader, criterion) -> Tuple[float, float]:
        """Ewaluacja na zbiorze walidacyjnym"""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_specs, batch_labels in loader:
                batch_specs = batch_specs.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_specs)
                loss = criterion(outputs, batch_labels)
                total_loss += loss.item()
                
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(batch_labels.cpu().numpy())
        
        avg_loss = total_loss / len(loader)
        
        from sklearn.metrics import f1_score
        f1 = f1_score(all_labels, all_preds, zero_division=0)
        
        return avg_loss, f1
    
    def predict(self, spectrograms: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        """Predykcja dla nowych spektrogramów
        
        Returns:
            (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        # Create dummy labels for dataset
        dummy_labels = np.zeros(len(spectrograms))
        dataset = SpectrogramDataset(spectrograms, dummy_labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_preds = []
        all_probs = []
        
        with torch.no_grad():
            for batch_specs, _ in loader:
                batch_specs = batch_specs.to(self.device)
                outputs = self.model(batch_specs)
                probs = F.softmax(outputs, dim=1)
                
                preds = outputs.argmax(dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of positive class
        
        return np.array(all_preds), np.array(all_probs)
    
    def get_features(self, spectrograms: List[np.ndarray]) -> np.ndarray:
        """Ekstrahuj features z CNN (dla late fusion)
        
        Returns:
            Feature array (n_samples, 64)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        self.model.eval()
        
        dummy_labels = np.zeros(len(spectrograms))
        dataset = SpectrogramDataset(spectrograms, dummy_labels)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)
        
        all_features = []
        
        with torch.no_grad():
            for batch_specs, _ in loader:
                batch_specs = batch_specs.to(self.device)
                features = self.model.get_features(batch_specs)
                all_features.append(features.cpu().numpy())
        
        return np.vstack(all_features)
    
    def save(self, path: Path):
        """Zapisz model"""
        if self.model is None:
            raise ValueError("No model to save")
        
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'history': self.history
        }, path)
        print(f"CNN model saved to {path}")
    
    def load(self, path: Path):
        """Załaduj model"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required")
        
        checkpoint = torch.load(path, map_location=self.device)
        self.model = DepressionCNN().to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.history = checkpoint.get('history', {})
        print(f"CNN model loaded from {path}")

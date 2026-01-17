"""Late Fusion and Attention-based Fusion for multimodal depression detection

Strategie fuzji:
1. Early Fusion: konkatenacja cech (już zaimplementowane)
2. Late Fusion: osobne modele + kombinacja predykcji
3. Attention Fusion: ważona kombinacja z learned weights
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import f1_score
import warnings


class LateFusionClassifier(BaseEstimator, ClassifierMixin):
    """Late Fusion: trenuj osobne modele dla każdej modalności, 
    połącz predykcje na końcu.
    
    Strategie łączenia:
    - 'average': średnia prawdopodobieństw
    - 'max': maksimum prawdopodobieństw (optymistyczne)
    - 'weighted': ważona średnia (wagi oparte na wydajności)
    - 'stacking': meta-classifier na prawdopodobieństwach
    """
    
    def __init__(self, 
                 fusion_strategy: str = 'weighted',
                 n_features_per_modality: int = 50,
                 random_state: int = 42):
        """
        Args:
            fusion_strategy: 'average', 'max', 'weighted', 'stacking'
            n_features_per_modality: Liczba cech do wybrania per modalność
            random_state: Random seed
        """
        self.fusion_strategy = fusion_strategy
        self.n_features_per_modality = n_features_per_modality
        self.random_state = random_state
        
        # Modele dla każdej modalności
        self.models = {}
        self.scalers = {}
        self.selectors = {}
        self.modality_weights = {}
        self.meta_classifier = None
        self.optimal_threshold = 0.5
    
    def _create_model(self):
        """Stwórz model dla modalności"""
        return RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def fit(self, X_dict: Dict[str, np.ndarray], y: np.ndarray,
            X_val_dict: Dict[str, np.ndarray] = None, y_val: np.ndarray = None):
        """Trenuj modele dla każdej modalności
        
        Args:
            X_dict: Dictionary {'audio': X_audio, 'text': X_text}
            y: Labels
            X_val_dict: Validation data (dla weighted fusion)
            y_val: Validation labels
        """
        modalities = list(X_dict.keys())
        
        for modality in modalities:
            X = X_dict[modality]
            
            # Scaling
            self.scalers[modality] = StandardScaler()
            X_scaled = self.scalers[modality].fit_transform(X)
            
            # Feature selection
            n_feat = min(self.n_features_per_modality, X_scaled.shape[1])
            self.selectors[modality] = SelectKBest(score_func=f_classif, k=n_feat)
            X_selected = self.selectors[modality].fit_transform(X_scaled, y)
            
            # Train model
            self.models[modality] = self._create_model()
            self.models[modality].fit(X_selected, y)
            
            print(f"  Trained {modality} model: {X_selected.shape[1]} features")
        
        # Calculate modality weights based on validation performance
        if self.fusion_strategy == 'weighted' and X_val_dict is not None:
            self._calculate_weights(X_val_dict, y_val)
        elif self.fusion_strategy == 'stacking' and X_val_dict is not None:
            self._train_meta_classifier(X_val_dict, y_val)
        else:
            # Equal weights
            for modality in modalities:
                self.modality_weights[modality] = 1.0 / len(modalities)
        
        # Optimize threshold on validation set
        if X_val_dict is not None and y_val is not None:
            self._optimize_threshold(X_val_dict, y_val)
        
        return self
    
    def _transform_modality(self, modality: str, X: np.ndarray) -> np.ndarray:
        """Transform features for a modality"""
        X_scaled = self.scalers[modality].transform(X)
        X_selected = self.selectors[modality].transform(X_scaled)
        return X_selected
    
    def _calculate_weights(self, X_val_dict: Dict[str, np.ndarray], y_val: np.ndarray):
        """Calculate modality weights based on validation F1"""
        f1_scores = {}
        
        for modality, X in X_val_dict.items():
            X_transformed = self._transform_modality(modality, X)
            y_pred = self.models[modality].predict(X_transformed)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            f1_scores[modality] = f1
        
        # Softmax weights
        total = sum(f1_scores.values())
        if total > 0:
            for modality, f1 in f1_scores.items():
                self.modality_weights[modality] = f1 / total
        else:
            for modality in f1_scores:
                self.modality_weights[modality] = 1.0 / len(f1_scores)
        
        print(f"  Modality weights: {self.modality_weights}")
    
    def _train_meta_classifier(self, X_val_dict: Dict[str, np.ndarray], y_val: np.ndarray):
        """Train meta-classifier for stacking"""
        # Get probabilities from each model
        probas = []
        for modality in sorted(X_val_dict.keys()):
            X = X_val_dict[modality]
            X_transformed = self._transform_modality(modality, X)
            proba = self.models[modality].predict_proba(X_transformed)[:, 1]
            probas.append(proba.reshape(-1, 1))
        
        X_meta = np.hstack(probas)
        
        # Train simple logistic regression as meta-classifier
        self.meta_classifier = LogisticRegression(
            random_state=self.random_state,
            class_weight='balanced'
        )
        self.meta_classifier.fit(X_meta, y_val)
        print(f"  Meta-classifier trained on {X_meta.shape[1]} modality probabilities")
    
    def _optimize_threshold(self, X_val_dict: Dict[str, np.ndarray], y_val: np.ndarray):
        """Find optimal threshold on validation set"""
        probas = self.predict_proba(X_val_dict)[:, 1]
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (probas >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        print(f"  Optimal threshold: {self.optimal_threshold:.2f} (F1: {best_f1:.4f})")
    
    def predict_proba(self, X_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Get probabilities using late fusion
        
        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        modalities = sorted(X_dict.keys())
        all_probas = []
        
        for modality in modalities:
            if modality not in self.models:
                continue
            X = X_dict[modality]
            X_transformed = self._transform_modality(modality, X)
            proba = self.models[modality].predict_proba(X_transformed)[:, 1]
            all_probas.append(proba)
        
        all_probas = np.array(all_probas)  # Shape: (n_modalities, n_samples)
        
        # Combine probabilities based on strategy
        if self.fusion_strategy == 'average':
            fused_proba = np.mean(all_probas, axis=0)
        
        elif self.fusion_strategy == 'max':
            fused_proba = np.max(all_probas, axis=0)
        
        elif self.fusion_strategy == 'weighted':
            weights = np.array([self.modality_weights.get(m, 1.0/len(modalities)) 
                               for m in modalities])
            fused_proba = np.average(all_probas, axis=0, weights=weights)
        
        elif self.fusion_strategy == 'stacking' and self.meta_classifier is not None:
            X_meta = all_probas.T  # Shape: (n_samples, n_modalities)
            fused_proba = self.meta_classifier.predict_proba(X_meta)[:, 1]
        
        else:
            fused_proba = np.mean(all_probas, axis=0)
        
        # Return (n_samples, 2) array
        return np.column_stack([1 - fused_proba, fused_proba])
    
    def predict(self, X_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict using late fusion with optimized threshold"""
        probas = self.predict_proba(X_dict)[:, 1]
        return (probas >= self.optimal_threshold).astype(int)


class AttentionFusionClassifier(BaseEstimator, ClassifierMixin):
    """Attention-based Fusion: learn attention weights for each modality
    
    Używa prostego mechanizmu attention:
    attention_weights = softmax(W @ modality_features)
    fused = sum(attention_weights * modality_features)
    """
    
    def __init__(self, 
                 n_features_per_modality: int = 50,
                 attention_dim: int = 32,
                 random_state: int = 42):
        """
        Args:
            n_features_per_modality: Liczba cech per modalność
            attention_dim: Wymiar warstwy attention
            random_state: Random seed
        """
        self.n_features_per_modality = n_features_per_modality
        self.attention_dim = attention_dim
        self.random_state = random_state
        
        self.scalers = {}
        self.selectors = {}
        self.attention_weights = None
        self.classifier = None
        self.optimal_threshold = 0.5
    
    def fit(self, X_dict: Dict[str, np.ndarray], y: np.ndarray,
            X_val_dict: Dict[str, np.ndarray] = None, y_val: np.ndarray = None):
        """Train attention-based fusion model"""
        modalities = sorted(X_dict.keys())
        
        # Process each modality
        processed_features = []
        for modality in modalities:
            X = X_dict[modality]
            
            # Scaling
            self.scalers[modality] = StandardScaler()
            X_scaled = self.scalers[modality].fit_transform(X)
            
            # Feature selection - same dimension for all modalities
            n_feat = min(self.n_features_per_modality, X_scaled.shape[1])
            self.selectors[modality] = SelectKBest(score_func=f_classif, k=n_feat)
            X_selected = self.selectors[modality].fit_transform(X_scaled, y)
            
            processed_features.append(X_selected)
        
        # Compute simple attention weights (based on feature correlation with target)
        modality_scores = []
        for i, modality in enumerate(modalities):
            X_feat = processed_features[i]
            # Średnia korelacja z target
            correlations = []
            for j in range(X_feat.shape[1]):
                corr = np.corrcoef(X_feat[:, j], y)[0, 1]
                if not np.isnan(corr):
                    correlations.append(abs(corr))
            avg_corr = np.mean(correlations) if correlations else 0
            modality_scores.append(avg_corr)
        
        # Softmax attention weights
        scores = np.array(modality_scores)
        exp_scores = np.exp(scores - np.max(scores))  # Stability
        self.attention_weights = exp_scores / exp_scores.sum()
        
        print(f"  Attention weights: {dict(zip(modalities, self.attention_weights))}")
        
        # Weighted concatenation of features
        weighted_features = []
        for i, X_feat in enumerate(processed_features):
            weighted_features.append(X_feat * self.attention_weights[i])
        
        X_fused = np.hstack(weighted_features)
        
        # Train final classifier
        self.classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        self.classifier.fit(X_fused, y)
        
        # Optimize threshold
        if X_val_dict is not None and y_val is not None:
            X_val_fused = self._transform(X_val_dict)
            y_proba = self.classifier.predict_proba(X_val_fused)[:, 1]
            
            best_f1 = 0
            best_threshold = 0.5
            for threshold in np.arange(0.1, 0.9, 0.05):
                y_pred = (y_proba >= threshold).astype(int)
                f1 = f1_score(y_val, y_pred, zero_division=0)
                if f1 > best_f1:
                    best_f1 = f1
                    best_threshold = threshold
            
            self.optimal_threshold = best_threshold
            print(f"  Optimal threshold: {self.optimal_threshold:.2f}")
        
        return self
    
    def _transform(self, X_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Transform and fuse features with attention weights"""
        modalities = sorted(X_dict.keys())
        weighted_features = []
        
        for i, modality in enumerate(modalities):
            X = X_dict[modality]
            X_scaled = self.scalers[modality].transform(X)
            X_selected = self.selectors[modality].transform(X_scaled)
            weighted_features.append(X_selected * self.attention_weights[i])
        
        return np.hstack(weighted_features)
    
    def predict_proba(self, X_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Get class probabilities"""
        X_fused = self._transform(X_dict)
        return self.classifier.predict_proba(X_fused)
    
    def predict(self, X_dict: Dict[str, np.ndarray]) -> np.ndarray:
        """Predict with optimized threshold"""
        probas = self.predict_proba(X_dict)[:, 1]
        return (probas >= self.optimal_threshold).astype(int)

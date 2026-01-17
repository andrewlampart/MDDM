"""
XGBoost Baseline Model for Depression Detection

Sprint 1: Baseline with GPU acceleration, threshold optimization, and early stopping.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict
import warnings

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    warnings.warn("XGBoost not available. Install with: pip install xgboost")

from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG


class XGBoostBaseline:
    """
    XGBoost baseline classifier with GPU support and threshold optimization.
    
    Features:
    - GPU acceleration (CUDA) if available
    - Sample weight computation for imbalanced classes
    - Optimal threshold search on validation set
    - Early stopping on validation loss
    - Feature selection (optional)
    """
    
    def __init__(self, 
                 n_estimators: int = None,
                 max_depth: int = None,
                 learning_rate: float = None,
                 use_gpu: bool = None,
                 n_features: int = None,
                 random_state: int = None):
        """
        Initialize XGBoost baseline.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            use_gpu: Whether to use GPU acceleration
            n_features: Number of features to select (None = use all)
            random_state: Random seed for reproducibility
        """
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost is required. Install with: pip install xgboost")
        
        self.n_estimators = n_estimators or CONFIG.XGBOOST_N_ESTIMATORS
        self.max_depth = max_depth or CONFIG.XGBOOST_MAX_DEPTH
        self.learning_rate = learning_rate or CONFIG.XGBOOST_LEARNING_RATE
        self.use_gpu = use_gpu if use_gpu is not None else CONFIG.USE_GPU
        self.n_features = n_features or CONFIG.N_FEATURES_SELECT
        self.random_state = random_state or CONFIG.RANDOM_SEED
        
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.best_threshold = 0.5
        self.is_fitted = False
        
        # Training history
        self.best_iteration = None
        self.eval_results = {}
    
    def _create_model(self, early_stopping_rounds: int = None) -> XGBClassifier:
        """Create XGBClassifier with appropriate settings"""
        params = {
            'n_estimators': self.n_estimators,
            'max_depth': self.max_depth,
            'learning_rate': self.learning_rate,
            'subsample': CONFIG.XGBOOST_SUBSAMPLE,
            'colsample_bytree': CONFIG.XGBOOST_COLSAMPLE_BYTREE,
            'random_state': self.random_state,
            'n_jobs': -1,
            'eval_metric': 'logloss',
            'use_label_encoder': False,
        }
        
        # Early stopping (XGBoost 2.0+ requires this in constructor)
        if early_stopping_rounds:
            params['early_stopping_rounds'] = early_stopping_rounds
        
        # GPU settings
        if self.use_gpu:
            params['tree_method'] = 'hist'
            params['device'] = 'cuda'
        else:
            params['tree_method'] = 'hist'
            params['device'] = 'cpu'
        
        return XGBClassifier(**params)
    
    def compute_class_weights(self, y: np.ndarray) -> np.ndarray:
        """Compute sample weights for imbalanced classes"""
        return compute_sample_weight('balanced', y)
    
    def _preprocess(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """Scale and select features"""
        # Scaling
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        # Feature selection
        if self.n_features and self.n_features < X_scaled.shape[1]:
            if fit:
                # This will be done during fit with labels
                return X_scaled
            elif self.feature_selector is not None:
                X_scaled = self.feature_selector.transform(X_scaled)
        
        return X_scaled
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            early_stopping_rounds: int = None,
            verbose: int = 50) -> 'XGBoostBaseline':
        """
        Train XGBoost model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (for early stopping)
            y_val: Validation labels
            early_stopping_rounds: Rounds without improvement before stopping
            verbose: Print progress every N rounds
            
        Returns:
            self
        """
        early_stopping_rounds = early_stopping_rounds or CONFIG.XGBOOST_EARLY_STOPPING
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Feature selection
        if self.n_features and self.n_features < X_train_scaled.shape[1]:
            self.feature_selector = SelectKBest(score_func=f_classif, k=self.n_features)
            X_train_scaled = self.feature_selector.fit_transform(X_train_scaled, y_train)
            self.selected_features = self.feature_selector.get_support(indices=True)
            print(f"Selected {len(self.selected_features)} features from {X_train.shape[1]}")
        
        # Compute sample weights
        weights = self.compute_class_weights(y_train)
        
        # Prepare evaluation set
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            if self.feature_selector is not None:
                X_val_scaled = self.feature_selector.transform(X_val_scaled)
            eval_set = [(X_val_scaled, y_val)]
        
        # Create model (with early stopping if eval_set provided)
        self.model = self._create_model(
            early_stopping_rounds=early_stopping_rounds if eval_set else None
        )
        
        # Train
        print(f"\nTraining XGBoost {'(GPU)' if self.use_gpu else '(CPU)'}...")
        print(f"  n_estimators: {self.n_estimators}")
        print(f"  max_depth: {self.max_depth}")
        print(f"  learning_rate: {self.learning_rate}")
        print(f"  early_stopping: {early_stopping_rounds if eval_set else 'disabled'}")
        print(f"  Training samples: {len(y_train)}")
        print(f"  Positive class: {y_train.sum()} ({y_train.mean():.1%})")
        
        self.model.fit(
            X_train_scaled, y_train,
            sample_weight=weights,
            eval_set=eval_set,
            verbose=verbose
        )
        
        self.best_iteration = self.model.best_iteration if hasattr(self.model, 'best_iteration') else self.n_estimators
        self.is_fitted = True
        
        print(f"\nTraining complete. Best iteration: {self.best_iteration}")
        
        # Optimize threshold if validation set provided
        if X_val is not None and y_val is not None:
            self.optimize_threshold(X_val, y_val)
        
        return self
    
    def optimize_threshold(self, X_val: np.ndarray, y_val: np.ndarray,
                          threshold_range: Tuple[float, float] = (0.1, 0.9),
                          step: float = 0.01) -> float:
        """
        Find optimal classification threshold based on F1 score.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            threshold_range: (min, max) threshold range to search
            step: Step size for threshold search
            
        Returns:
            Optimal threshold
        """
        y_proba = self.predict_proba(X_val)[:, 1]
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(threshold_range[0], threshold_range[1], step):
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.best_threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.4f})")
        
        return best_threshold
    
    def predict(self, X: np.ndarray, use_optimal_threshold: bool = True) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features
            use_optimal_threshold: Whether to use optimized threshold
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        y_proba = self.predict_proba(X)[:, 1]
        
        if use_optimal_threshold:
            return (y_proba >= self.best_threshold).astype(int)
        else:
            return (y_proba >= 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features
            
        Returns:
            Array of shape (n_samples, 2) with class probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_scaled = self.scaler.transform(X)
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        
        return self.model.predict_proba(X_scaled)
    
    def get_feature_importance(self, feature_names: list = None, top_n: int = 20) -> Dict:
        """
        Get feature importance from trained model.
        
        Args:
            feature_names: List of feature names
            top_n: Number of top features to return
            
        Returns:
            Dictionary with feature importance info
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        importances = self.model.feature_importances_
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        elif self.selected_features is not None:
            feature_names = [feature_names[i] for i in self.selected_features]
        
        # Sort by importance
        indices = np.argsort(importances)[::-1][:top_n]
        
        return {
            'feature_names': [feature_names[i] for i in indices],
            'importances': importances[indices].tolist(),
            'indices': indices.tolist()
        }
    
    def save(self, path: Path = None):
        """Save model to file"""
        path = path or CONFIG.XGBOOST_MODEL
        path.parent.mkdir(parents=True, exist_ok=True)
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'best_threshold': self.best_threshold,
            'best_iteration': self.best_iteration,
            'config': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate,
                'n_features': self.n_features,
            }
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {path}")
    
    def load(self, path: Path = None) -> 'XGBoostBaseline':
        """Load model from file"""
        path = path or CONFIG.XGBOOST_MODEL
        
        with open(path, 'rb') as f:
            model_data = pickle.load(f)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_selector = model_data.get('feature_selector')
        self.selected_features = model_data.get('selected_features')
        self.best_threshold = model_data.get('best_threshold', 0.5)
        self.best_iteration = model_data.get('best_iteration')
        self.is_fitted = True
        
        print(f"Model loaded from {path}")
        return self


if __name__ == '__main__':
    # Quick test
    print("XGBoost Baseline Module")
    print(f"XGBoost available: {XGBOOST_AVAILABLE}")
    print(f"GPU enabled: {CONFIG.USE_GPU}")
    
    if XGBOOST_AVAILABLE:
        baseline = XGBoostBaseline()
        print(f"Model created with {baseline.n_estimators} estimators")

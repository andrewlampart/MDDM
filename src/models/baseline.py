"""Baseline classifier model for depression detection"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Dict, Optional
import pickle

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, precision_recall_curve
)

from ..utils.config import Config


class BaselineClassifier:
    """Baseline classifier dla diagnozy depresji z optymalizacją"""
    
    def __init__(self, n_estimators: int = 100, max_depth: Optional[int] = 10,
                 random_state: int = 42, n_features: int = 50):
        """Initialize BaselineClassifier
        
        Args:
            n_estimators: Liczba drzew w Random Forest
            max_depth: Maksymalna głębokość drzew
            random_state: Random seed dla reproducibility
            n_features: Liczba features do selekcji
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.random_state = random_state
        self.n_features = n_features
        self.optimal_threshold = 0.5
        
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=5,
            min_samples_split=10,
            random_state=random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.selected_features = None
        self.is_fitted = False
    
    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """Przygotuj dane do treningu
        
        Args:
            df: DataFrame z features i labels
            
        Returns:
            Tuple (X, y) gdzie X to features, y to labels
        """
        # Usuń kolumny nie-feature
        exclude_cols = ['session_id', 'depression', 'phq8_score', 'qids_score']
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        X = df[feature_cols].values
        y = df['depression'].values
        
        return X, y
    
    def train_test_split_stratified(
        self, 
        df: pd.DataFrame,
        train_size: int = 107,
        val_size: int = 35,
        test_size: int = 47
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Stratified train/val/test split zgodnie z DAIC-WOZ standardem
        
        Args:
            df: DataFrame z danymi
            train_size: Liczba próbek w train set
            val_size: Liczba próbek w validation set
            test_size: Liczba próbek w test set
            
        Returns:
            Tuple (train_df, val_df, test_df)
        """
        # Najpierw podziel na train+val i test
        test_ratio = test_size / len(df)
        train_val_df, test_df = train_test_split(
            df,
            test_size=test_ratio,
            stratify=df['depression'],
            random_state=self.random_state
        )
        
        # Potem podziel train+val na train i val
        val_ratio = val_size / (train_size + val_size)
        train_df, val_df = train_test_split(
            train_val_df,
            test_size=val_ratio,
            stratify=train_val_df['depression'],
            random_state=self.random_state
        )
        
        return train_df, val_df, test_df
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: np.ndarray = None, y_val: np.ndarray = None,
            optimize_threshold: bool = True):
        """Trenuj model z selekcją cech i optymalizacją progu
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (dla optymalizacji progu)
            y_val: Validation labels
            optimize_threshold: Czy optymalizować próg decyzyjny
        """
        # Normalizacja
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Selekcja cech - wybierz najważniejsze
        n_features = min(self.n_features, X_train_scaled.shape[1])
        self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        X_train_selected = self.feature_selector.fit_transform(X_train_scaled, y_train)
        
        # Zapisz indeksy wybranych cech
        self.selected_features = self.feature_selector.get_support(indices=True)
        print(f"Selected {len(self.selected_features)} features from {X_train_scaled.shape[1]}")
        
        # Trening
        self.model.fit(X_train_selected, y_train)
        self.is_fitted = True
        
        # Optymalizacja progu na validation set
        if optimize_threshold and X_val is not None and y_val is not None:
            self._optimize_threshold(X_val, y_val)
    
    def _optimize_threshold(self, X_val: np.ndarray, y_val: np.ndarray):
        """Znajdź optymalny próg decyzyjny maksymalizujący F1"""
        y_proba = self.predict_proba(X_val)[:, 1]
        
        best_f1 = 0
        best_threshold = 0.5
        
        # Przeszukaj progi od 0.1 do 0.9
        for threshold in np.arange(0.1, 0.9, 0.05):
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred, zero_division=0)
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.optimal_threshold = best_threshold
        print(f"Optimal threshold: {self.optimal_threshold:.2f} (F1: {best_f1:.4f})")
    
    def _transform_features(self, X: np.ndarray) -> np.ndarray:
        """Normalizacja i selekcja cech"""
        X_scaled = self.scaler.transform(X)
        if self.feature_selector is not None:
            X_scaled = self.feature_selector.transform(X_scaled)
        return X_scaled
    
    def predict(self, X: np.ndarray, use_optimal_threshold: bool = True) -> np.ndarray:
        """Predykcja z opcjonalnym optymalnym progiem
        
        Args:
            X: Features
            use_optimal_threshold: Użyj zoptymalizowanego progu
            
        Returns:
            Predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if use_optimal_threshold and self.optimal_threshold != 0.5:
            y_proba = self.predict_proba(X)[:, 1]
            return (y_proba >= self.optimal_threshold).astype(int)
        
        X_transformed = self._transform_features(X)
        return self.model.predict(X_transformed)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predykcja prawdopodobieństw
        
        Args:
            X: Features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X_transformed = self._transform_features(X)
        return self.model.predict_proba(X_transformed)
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, set_name: str = "Test") -> Dict:
        """Ewaluacja modelu
        
        Args:
            X: Features
            y: True labels
            set_name: Nazwa setu (dla logowania)
            
        Returns:
            Dictionary z metrykami
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]  # Prawdopodobieństwo klasy pozytywnej
        
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0),
            'confusion_matrix': confusion_matrix(y, y_pred).tolist()
        }
        
        print(f"\n=== {set_name} Set Evaluation ===")
        print(f"Accuracy:  {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall:    {metrics['recall']:.4f}")
        print(f"F1 Score:  {metrics['f1']:.4f}")
        print(f"\nConfusion Matrix:")
        print(f"  TN: {metrics['confusion_matrix'][0][0]}, FP: {metrics['confusion_matrix'][0][1]}")
        print(f"  FN: {metrics['confusion_matrix'][1][0]}, TP: {metrics['confusion_matrix'][1][1]}")
        
        print(f"\nClassification Report:")
        print(classification_report(y, y_pred, target_names=['No Depression', 'Depression']))
        
        return metrics
    
    def save(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        """Zapisz model, scaler, feature selector i próg
        
        Args:
            model_path: Ścieżka do zapisu modelu
            scaler_path: Ścieżka do zapisu scalera
        """
        model_path = model_path or Config.MODEL_PKL
        scaler_path = scaler_path or Config.SCALER_PKL
        
        model_path.parent.mkdir(parents=True, exist_ok=True)
        scaler_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Zapisz wszystko w jednym słowniku
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_selector': self.feature_selector,
            'selected_features': self.selected_features,
            'optimal_threshold': self.optimal_threshold
        }
        
        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Dla kompatybilności wstecznej - sam scaler
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        print(f"Model saved to {model_path}")
        print(f"Scaler saved to {scaler_path}")
    
    def load(self, model_path: Optional[Path] = None, scaler_path: Optional[Path] = None):
        """Załaduj model i scaler
        
        Args:
            model_path: Ścieżka do modelu
            scaler_path: Ścieżka do scalera
        """
        model_path = model_path or Config.MODEL_PKL
        scaler_path = scaler_path or Config.SCALER_PKL
        
        with open(model_path, 'rb') as f:
            data = pickle.load(f)
        
        # Obsłuż nowy i stary format
        if isinstance(data, dict):
            self.model = data['model']
            self.scaler = data['scaler']
            self.feature_selector = data.get('feature_selector')
            self.selected_features = data.get('selected_features')
            self.optimal_threshold = data.get('optimal_threshold', 0.5)
        else:
            self.model = data
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        
        self.is_fitted = True
        print(f"Model loaded from {model_path}")
    
    def grid_search(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict:
        """Grid search dla najlepszych hiperparametrów
        
        Args:
            X_train: Training features (już przeskalowane)
            y_train: Training labels
            
        Returns:
            Najlepsze parametry
        """
        # Przygotuj dane
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Selekcja cech przed grid search
        n_features = min(self.n_features, X_scaled.shape[1])
        self.feature_selector = SelectKBest(score_func=f_classif, k=n_features)
        X_selected = self.feature_selector.fit_transform(X_scaled, y_train)
        self.selected_features = self.feature_selector.get_support(indices=True)
        
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [5, 10, 15, None],
            'min_samples_leaf': [3, 5, 10],
            'min_samples_split': [5, 10, 15]
        }
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        grid = GridSearchCV(
            RandomForestClassifier(class_weight='balanced', random_state=self.random_state, n_jobs=-1),
            param_grid,
            cv=cv,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        print("Running grid search...")
        grid.fit(X_selected, y_train)
        
        print(f"\nBest parameters: {grid.best_params_}")
        print(f"Best CV F1: {grid.best_score_:.4f}")
        
        # Użyj najlepszego modelu
        self.model = grid.best_estimator_
        self.is_fitted = True
        
        return grid.best_params_

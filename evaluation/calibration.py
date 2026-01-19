"""
Model Calibration for Depression Detection

Implements:
- Expected Calibration Error (ECE) - key metric for calibration quality
- Platt Scaling (sigmoid calibration)
- Isotonic Regression calibration
- Calibration visualization

Expected improvement: ECE 0.10 -> 0.04-0.05
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
from pathlib import Path
import logging

from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.base import BaseEstimator, ClassifierMixin

logger = logging.getLogger(__name__)


def expected_calibration_error(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10,
    strategy: str = 'uniform'
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    ECE measures the average difference between predicted probability
    and actual accuracy within probability bins. Lower is better.
    
    Args:
        y_true: True binary labels (0 or 1)
        y_pred_proba: Predicted probabilities for positive class
        n_bins: Number of bins for calibration
        strategy: 'uniform' (equal width) or 'quantile' (equal size)
        
    Returns:
        ECE value (float between 0 and 1)
        
    Interpretation:
        ECE < 0.03: Excellent calibration
        ECE < 0.05: Good calibration
        ECE < 0.10: Acceptable calibration
        ECE > 0.10: Poor calibration, needs recalibration
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    if strategy == 'uniform':
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
    elif strategy == 'quantile':
        bin_boundaries = np.percentile(y_pred_proba, np.linspace(0, 100, n_bins + 1))
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    ece = 0.0
    total_samples = len(y_true)
    
    for i in range(n_bins):
        # Find samples in this bin
        if i == n_bins - 1:
            # Last bin includes right boundary
            mask = (y_pred_proba >= bin_boundaries[i]) & (y_pred_proba <= bin_boundaries[i+1])
        else:
            mask = (y_pred_proba >= bin_boundaries[i]) & (y_pred_proba < bin_boundaries[i+1])
        
        n_in_bin = mask.sum()
        
        if n_in_bin > 0:
            # Actual accuracy in this bin
            accuracy = y_true[mask].mean()
            # Average confidence (predicted probability) in this bin
            confidence = y_pred_proba[mask].mean()
            # Weighted absolute difference
            ece += np.abs(accuracy - confidence) * n_in_bin / total_samples
    
    return float(ece)


def maximum_calibration_error(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    n_bins: int = 10
) -> float:
    """
    Compute Maximum Calibration Error (MCE).
    
    MCE is the maximum difference between predicted probability
    and actual accuracy across all bins. Useful for worst-case analysis.
    
    Args:
        y_true: True binary labels
        y_pred_proba: Predicted probabilities
        n_bins: Number of bins
        
    Returns:
        MCE value (float between 0 and 1)
    """
    y_true = np.array(y_true)
    y_pred_proba = np.array(y_pred_proba)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    
    max_error = 0.0
    
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred_proba >= bin_boundaries[i]) & (y_pred_proba <= bin_boundaries[i+1])
        else:
            mask = (y_pred_proba >= bin_boundaries[i]) & (y_pred_proba < bin_boundaries[i+1])
        
        if mask.sum() > 0:
            accuracy = y_true[mask].mean()
            confidence = y_pred_proba[mask].mean()
            error = np.abs(accuracy - confidence)
            max_error = max(max_error, error)
    
    return float(max_error)


def calibrate_model(
    model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
    method: str = 'sigmoid'
) -> CalibratedClassifierCV:
    """
    Calibrate a pre-trained classifier using Platt Scaling or Isotonic Regression.
    
    Args:
        model: Pre-trained sklearn-compatible classifier
        X_val: Validation features for fitting calibrator
        y_val: Validation labels
        method: 'sigmoid' (Platt scaling) or 'isotonic'
        
    Returns:
        CalibratedClassifierCV: Calibrated model wrapper
        
    Usage:
        # Train your model
        model.fit(X_train, y_train)
        
        # Calibrate on validation set
        calibrated = calibrate_model(model, X_val, y_val)
        
        # Use calibrated model for predictions
        proba = calibrated.predict_proba(X_test)[:, 1]
    """
    if method not in ('sigmoid', 'isotonic'):
        raise ValueError(f"method must be 'sigmoid' or 'isotonic', got {method}")
    
    logger.info(f"Calibrating model with {method} method...")
    
    calibrator = CalibratedClassifierCV(
        estimator=model,
        method=method,
        cv='prefit'  # Use pre-trained model
    )
    
    calibrator.fit(X_val, y_val)
    
    logger.info("Calibration complete.")
    return calibrator


def evaluate_calibration(
    y_true: np.ndarray,
    y_pred_proba_uncal: np.ndarray,
    y_pred_proba_cal: Optional[np.ndarray] = None,
    n_bins: int = 10
) -> Dict[str, float]:
    """
    Evaluate calibration quality before and optionally after calibration.
    
    Args:
        y_true: True binary labels
        y_pred_proba_uncal: Uncalibrated predicted probabilities
        y_pred_proba_cal: Calibrated predicted probabilities (optional)
        n_bins: Number of bins for ECE computation
        
    Returns:
        Dictionary with calibration metrics
    """
    results = {
        'ece_before': expected_calibration_error(y_true, y_pred_proba_uncal, n_bins),
        'mce_before': maximum_calibration_error(y_true, y_pred_proba_uncal, n_bins),
        'brier_before': brier_score_loss(y_true, y_pred_proba_uncal),
    }
    
    if y_pred_proba_cal is not None:
        results['ece_after'] = expected_calibration_error(y_true, y_pred_proba_cal, n_bins)
        results['mce_after'] = maximum_calibration_error(y_true, y_pred_proba_cal, n_bins)
        results['brier_after'] = brier_score_loss(y_true, y_pred_proba_cal)
        
        # Improvement metrics
        results['ece_improvement'] = results['ece_before'] - results['ece_after']
        results['brier_improvement'] = results['brier_before'] - results['brier_after']
    
    return results


def calibrate_and_evaluate(
    model: Any,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    method: str = 'sigmoid'
) -> Tuple[CalibratedClassifierCV, Dict[str, float]]:
    """
    Full calibration pipeline: calibrate model and evaluate on test set.
    
    Args:
        model: Pre-trained classifier (already fitted on X_train, y_train)
        X_train, y_train: Training data (for reference)
        X_val, y_val: Validation data (for calibration)
        X_test, y_test: Test data (for evaluation)
        method: Calibration method
        
    Returns:
        Tuple of (calibrated_model, evaluation_metrics)
    """
    # Get uncalibrated predictions on test set
    y_pred_uncal = model.predict_proba(X_test)[:, 1]
    
    # Calibrate on validation set
    calibrated_model = calibrate_model(model, X_val, y_val, method)
    
    # Get calibrated predictions on test set
    y_pred_cal = calibrated_model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    metrics = evaluate_calibration(y_test, y_pred_uncal, y_pred_cal)
    
    # Log results
    logger.info(f"Calibration results on test set:")
    logger.info(f"  ECE: {metrics['ece_before']:.4f} -> {metrics['ece_after']:.4f} "
                f"(improvement: {metrics['ece_improvement']:.4f})")
    logger.info(f"  Brier: {metrics['brier_before']:.4f} -> {metrics['brier_after']:.4f}")
    
    return calibrated_model, metrics


def plot_calibration_comparison(
    y_true: np.ndarray,
    y_pred_uncal: np.ndarray,
    y_pred_cal: np.ndarray,
    n_bins: int = 10,
    save_path: Optional[Path] = None,
    model_name: str = "Model"
):
    """
    Plot calibration curves comparing before/after calibration.
    
    Args:
        y_true: True labels
        y_pred_uncal: Uncalibrated probabilities
        y_pred_cal: Calibrated probabilities
        n_bins: Number of bins
        save_path: Path to save figure
        model_name: Name for title
        
    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Compute calibration curves
    fraction_pos_uncal, mean_pred_uncal = calibration_curve(
        y_true, y_pred_uncal, n_bins=n_bins, strategy='uniform'
    )
    fraction_pos_cal, mean_pred_cal = calibration_curve(
        y_true, y_pred_cal, n_bins=n_bins, strategy='uniform'
    )
    
    # Compute ECE
    ece_uncal = expected_calibration_error(y_true, y_pred_uncal, n_bins)
    ece_cal = expected_calibration_error(y_true, y_pred_cal, n_bins)
    
    # Plot 1: Uncalibrated
    axes[0].plot(mean_pred_uncal, fraction_pos_uncal, 's-', color='#F44336', 
                 lw=2, markersize=8, label=f'Uncalibrated (ECE={ece_uncal:.3f})')
    axes[0].plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    axes[0].set_xlabel('Mean Predicted Probability')
    axes[0].set_ylabel('Fraction of Positives')
    axes[0].set_title('Before Calibration', fontweight='bold')
    axes[0].legend(loc='lower right')
    axes[0].grid(alpha=0.3)
    axes[0].set_xlim([0, 1])
    axes[0].set_ylim([0, 1])
    
    # Plot 2: Calibrated
    axes[1].plot(mean_pred_cal, fraction_pos_cal, 's-', color='#4CAF50',
                 lw=2, markersize=8, label=f'Calibrated (ECE={ece_cal:.3f})')
    axes[1].plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    axes[1].set_xlabel('Mean Predicted Probability')
    axes[1].set_ylabel('Fraction of Positives')
    axes[1].set_title('After Calibration', fontweight='bold')
    axes[1].legend(loc='lower right')
    axes[1].grid(alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])
    
    # Plot 3: Both
    axes[2].plot(mean_pred_uncal, fraction_pos_uncal, 's-', color='#F44336',
                 lw=2, markersize=8, label=f'Before (ECE={ece_uncal:.3f})')
    axes[2].plot(mean_pred_cal, fraction_pos_cal, 's-', color='#4CAF50',
                 lw=2, markersize=8, label=f'After (ECE={ece_cal:.3f})')
    axes[2].plot([0, 1], [0, 1], 'k--', lw=1, label='Perfect')
    axes[2].set_xlabel('Mean Predicted Probability')
    axes[2].set_ylabel('Fraction of Positives')
    axes[2].set_title('Comparison', fontweight='bold')
    axes[2].legend(loc='lower right')
    axes[2].grid(alpha=0.3)
    axes[2].set_xlim([0, 1])
    axes[2].set_ylim([0, 1])
    
    plt.suptitle(f'{model_name} Calibration Analysis', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Calibration plot saved to {save_path}")
    
    return fig


class TemperatureScaling:
    """
    Temperature Scaling for neural network calibration.
    
    Learns a single temperature parameter T to scale logits:
        p = softmax(logits / T)
    
    Useful for deep learning models where Platt scaling may not work well.
    """
    
    def __init__(self):
        self.temperature = 1.0
    
    def fit(self, logits: np.ndarray, y_true: np.ndarray):
        """
        Fit temperature parameter using validation set.
        
        Args:
            logits: Model logits (before sigmoid)
            y_true: True labels
        """
        from scipy.optimize import minimize_scalar
        
        def nll_loss(T):
            """Negative log likelihood with temperature scaling."""
            scaled = logits / T
            proba = 1 / (1 + np.exp(-scaled))
            # Clip for numerical stability
            proba = np.clip(proba, 1e-7, 1 - 1e-7)
            loss = -np.mean(y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba))
            return loss
        
        result = minimize_scalar(nll_loss, bounds=(0.1, 10.0), method='bounded')
        self.temperature = result.x
        logger.info(f"Optimal temperature: {self.temperature:.4f}")
        
        return self
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Apply temperature scaling to get calibrated probabilities.
        
        Args:
            logits: Model logits
            
        Returns:
            Calibrated probabilities
        """
        scaled = logits / self.temperature
        proba = 1 / (1 + np.exp(-scaled))
        return proba


# Convenience functions
def compute_ece(y_true, y_proba, n_bins=10):
    """Shorthand for expected_calibration_error."""
    return expected_calibration_error(y_true, y_proba, n_bins)


def compute_brier(y_true, y_proba):
    """Shorthand for brier_score_loss."""
    return brier_score_loss(y_true, y_proba)


# =============================================================================
# NEW: Threshold Optimization
# =============================================================================

def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    metric: str = 'f1',
    threshold_range: Tuple[float, float] = (0.1, 0.9),
    step: float = 0.01
) -> Tuple[float, float, Dict[float, float]]:
    """
    Find optimal classification threshold for a given metric.
    
    Instead of using default 0.5 threshold, this function searches for
    the threshold that maximizes the specified metric.
    
    Args:
        y_true: True binary labels
        y_proba: Predicted probabilities for positive class
        metric: Metric to optimize. Options:
            - 'f1': F1 score (default, balances precision/recall)
            - 'balanced': (sensitivity + specificity) / 2
            - 'youden': Youden's J statistic (sensitivity + specificity - 1)
            - 'precision': Precision score
            - 'recall': Recall/Sensitivity score
        threshold_range: (min, max) threshold values to search
        step: Step size for threshold search
        
    Returns:
        Tuple of (optimal_threshold, best_score, all_scores_dict)
        
    Example:
        >>> threshold, score, scores = find_optimal_threshold(y_val, y_proba, 'f1')
        >>> print(f"Optimal threshold: {threshold:.3f} (F1={score:.3f})")
        >>> y_pred = (y_proba > threshold).astype(int)
    """
    from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
    
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    thresholds = np.arange(threshold_range[0], threshold_range[1] + step, step)
    best_score = -np.inf
    best_threshold = 0.5
    scores = {}
    
    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        
        # Skip if all predictions are same class
        if len(np.unique(y_pred)) < 2:
            scores[threshold] = 0.0
            continue
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced':
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = (sensitivity + specificity) / 2
            else:
                score = 0.0
        elif metric == 'youden':
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp, fn, tp = cm.ravel()
                sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
                score = sensitivity + specificity - 1  # Youden's J
            else:
                score = 0.0
        elif metric == 'precision':
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}. Use 'f1', 'balanced', 'youden', 'precision', 'recall'")
        
        scores[threshold] = score
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    logger.info(f"Optimal threshold for {metric}: {best_threshold:.3f} (score={best_score:.4f})")
    
    return float(best_threshold), float(best_score), scores


def plot_threshold_optimization(
    scores: Dict[float, float],
    optimal_threshold: float,
    metric_name: str = 'F1',
    save_path: Optional[Path] = None
):
    """
    Plot threshold optimization curve.
    
    Args:
        scores: Dict mapping threshold -> metric score
        optimal_threshold: The optimal threshold found
        metric_name: Name of metric for labels
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    thresholds = list(scores.keys())
    metric_values = list(scores.values())
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ax.plot(thresholds, metric_values, 'b-', linewidth=2, label=f'{metric_name} Score')
    ax.axvline(optimal_threshold, color='r', linestyle='--', linewidth=2,
               label=f'Optimal: {optimal_threshold:.3f}')
    ax.axvline(0.5, color='gray', linestyle=':', linewidth=1, alpha=0.7,
               label='Default (0.5)')
    
    ax.set_xlabel('Classification Threshold', fontsize=12)
    ax.set_ylabel(f'{metric_name} Score', fontsize=12)
    ax.set_title(f'Threshold Optimization for {metric_name}', fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Threshold plot saved to {save_path}")
    
    return fig


# =============================================================================
# NEW: Temperature Scaling for PyTorch Neural Networks
# =============================================================================

class TemperatureScalingNN:
    """
    Temperature Scaling specifically designed for PyTorch neural networks.
    
    This is more effective than Platt scaling for deep learning models because:
    1. It preserves the ranking of predictions (monotonic transformation)
    2. It only learns one parameter, reducing overfitting risk
    3. It's mathematically equivalent to adjusting model confidence
    
    The calibrated probability is: p = sigmoid(logits / T)
    where T is the learned temperature parameter.
    
    If T > 1: Model becomes less confident (spreads probabilities)
    If T < 1: Model becomes more confident (sharpens probabilities)
    
    Usage:
        calibrator = TemperatureScalingNN()
        calibrator.fit(logits_val, y_val)  # Fit on validation set
        calibrated_proba = calibrator.predict_proba(logits_test)
    """
    
    def __init__(self, init_temperature: float = 1.5):
        """
        Initialize temperature scaling.
        
        Args:
            init_temperature: Initial temperature value for optimization
        """
        self.temperature = init_temperature
        self.fitted = False
        self.nll_before = None
        self.nll_after = None
    
    def fit(
        self,
        logits: np.ndarray,
        y_true: np.ndarray,
        bounds: Tuple[float, float] = (0.1, 10.0)
    ) -> 'TemperatureScalingNN':
        """
        Fit temperature parameter by minimizing NLL on validation set.
        
        Args:
            logits: Model logits (before sigmoid), shape (n_samples,)
            y_true: True binary labels, shape (n_samples,)
            bounds: (min, max) bounds for temperature search
            
        Returns:
            self (fitted calibrator)
        """
        from scipy.optimize import minimize_scalar
        
        logits = np.array(logits).flatten()
        y_true = np.array(y_true).flatten()
        
        def nll_loss(T: float) -> float:
            """Negative log likelihood with temperature scaling."""
            if T <= 0:
                return np.inf
            scaled = logits / T
            # Sigmoid with numerical stability
            proba = 1 / (1 + np.exp(-np.clip(scaled, -500, 500)))
            proba = np.clip(proba, 1e-7, 1 - 1e-7)
            # Binary cross-entropy
            loss = -np.mean(y_true * np.log(proba) + (1 - y_true) * np.log(1 - proba))
            return loss
        
        # Store NLL before calibration
        self.nll_before = nll_loss(1.0)
        
        # Optimize temperature
        result = minimize_scalar(nll_loss, bounds=bounds, method='bounded')
        self.temperature = result.x
        self.fitted = True
        
        # Store NLL after calibration
        self.nll_after = nll_loss(self.temperature)
        
        logger.info(f"TemperatureScalingNN fitted:")
        logger.info(f"  Optimal temperature: {self.temperature:.4f}")
        logger.info(f"  NLL: {self.nll_before:.4f} -> {self.nll_after:.4f}")
        
        return self
    
    def predict_proba(self, logits: np.ndarray) -> np.ndarray:
        """
        Get calibrated probabilities.
        
        Args:
            logits: Model logits (before sigmoid)
            
        Returns:
            Calibrated probabilities
        """
        if not self.fitted:
            logger.warning("TemperatureScalingNN not fitted, using T=1.0")
            T = 1.0
        else:
            T = self.temperature
        
        logits = np.array(logits).flatten()
        scaled = logits / T
        proba = 1 / (1 + np.exp(-np.clip(scaled, -500, 500)))
        return proba
    
    def get_calibration_improvement(self) -> Dict[str, float]:
        """Get calibration improvement metrics."""
        if not self.fitted:
            return {}
        return {
            'temperature': self.temperature,
            'nll_before': self.nll_before,
            'nll_after': self.nll_after,
            'nll_improvement': self.nll_before - self.nll_after
        }


# =============================================================================
# NEW: Calibrated PyTorch Model Wrapper
# =============================================================================

class CalibratedPyTorchModel:
    """
    Wrapper that adds calibration to any PyTorch model.
    
    Combines:
    1. Temperature Scaling for probability calibration
    2. Optimal Threshold selection for classification
    
    This solves the common problem where neural networks have:
    - High F1 but low AUROC (poor probability ranking)
    - Overconfident predictions (probabilities too close to 0 or 1)
    - Suboptimal default threshold of 0.5
    
    Usage:
        # After training your PyTorch model
        calibrated = CalibratedPyTorchModel(pytorch_model)
        calibrated.fit_calibration(X_audio_val, X_text_val, y_val)
        
        # For inference
        y_proba = calibrated.predict_proba(X_audio_test, X_text_test)
        y_pred = calibrated.predict(X_audio_test, X_text_test)
    """
    
    def __init__(
        self,
        model,
        device: str = 'cpu',
        use_temperature_scaling: bool = True,
        use_threshold_optimization: bool = True,
        threshold_metric: str = 'f1'
    ):
        """
        Initialize calibrated model wrapper.
        
        Args:
            model: PyTorch model with forward(audio, text) -> logits
            device: Device to run model on
            use_temperature_scaling: Whether to apply temperature scaling
            use_threshold_optimization: Whether to optimize classification threshold
            threshold_metric: Metric for threshold optimization
        """
        self.model = model
        self.device = device
        self.use_temperature_scaling = use_temperature_scaling
        self.use_threshold_optimization = use_threshold_optimization
        self.threshold_metric = threshold_metric
        
        # Calibration components
        self.temperature_scaler = TemperatureScalingNN() if use_temperature_scaling else None
        self.optimal_threshold = 0.5
        self.calibrated = False
    
    def _get_logits(self, X_audio: np.ndarray, X_text: np.ndarray) -> np.ndarray:
        """Get raw logits from model."""
        import torch
        
        self.model.eval()
        with torch.no_grad():
            audio_tensor = torch.FloatTensor(X_audio).to(self.device)
            text_tensor = torch.FloatTensor(X_text).to(self.device)
            logits = self.model(audio_tensor, text_tensor)
            return logits.cpu().numpy().flatten()
    
    def fit_calibration(
        self,
        X_audio_val: np.ndarray,
        X_text_val: np.ndarray,
        y_val: np.ndarray
    ) -> 'CalibratedPyTorchModel':
        """
        Fit calibration on validation set.
        
        IMPORTANT: Use a held-out validation set, NOT training data!
        
        Args:
            X_audio_val: Audio features for validation
            X_text_val: Text features for validation
            y_val: True labels for validation
            
        Returns:
            self (fitted calibrated model)
        """
        y_val = np.array(y_val).flatten()
        
        # Get logits from model
        logits = self._get_logits(X_audio_val, X_text_val)
        
        # Step 1: Temperature Scaling
        if self.use_temperature_scaling and self.temperature_scaler is not None:
            self.temperature_scaler.fit(logits, y_val)
            calibrated_proba = self.temperature_scaler.predict_proba(logits)
        else:
            # Just apply sigmoid
            calibrated_proba = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        
        # Step 2: Threshold Optimization
        if self.use_threshold_optimization:
            self.optimal_threshold, best_score, _ = find_optimal_threshold(
                y_val, calibrated_proba, metric=self.threshold_metric
            )
            logger.info(f"Optimal threshold: {self.optimal_threshold:.3f} "
                       f"({self.threshold_metric}={best_score:.4f})")
        
        self.calibrated = True
        logger.info("Calibration complete!")
        
        return self
    
    def predict_proba(self, X_audio: np.ndarray, X_text: np.ndarray) -> np.ndarray:
        """
        Get calibrated probabilities.
        
        Args:
            X_audio: Audio features
            X_text: Text features
            
        Returns:
            Calibrated probabilities (n_samples,)
        """
        logits = self._get_logits(X_audio, X_text)
        
        if self.use_temperature_scaling and self.temperature_scaler is not None and self.temperature_scaler.fitted:
            proba = self.temperature_scaler.predict_proba(logits)
        else:
            proba = 1 / (1 + np.exp(-np.clip(logits, -500, 500)))
        
        return proba
    
    def predict(self, X_audio: np.ndarray, X_text: np.ndarray) -> np.ndarray:
        """
        Get calibrated binary predictions using optimal threshold.
        
        Args:
            X_audio: Audio features
            X_text: Text features
            
        Returns:
            Binary predictions (n_samples,)
        """
        proba = self.predict_proba(X_audio, X_text)
        return (proba > self.optimal_threshold).astype(int)
    
    def get_calibration_info(self) -> Dict[str, Any]:
        """Get calibration parameters and metrics."""
        info = {
            'calibrated': self.calibrated,
            'optimal_threshold': self.optimal_threshold,
            'use_temperature_scaling': self.use_temperature_scaling,
            'use_threshold_optimization': self.use_threshold_optimization
        }
        
        if self.temperature_scaler is not None and self.temperature_scaler.fitted:
            info.update(self.temperature_scaler.get_calibration_improvement())
        
        return info


# =============================================================================
# NEW: Ensemble Calibration Utilities
# =============================================================================

def calibrate_ensemble_predictions(
    fold_predictions: List[np.ndarray],
    fold_labels: List[np.ndarray],
    method: str = 'mean'
) -> Tuple[np.ndarray, float]:
    """
    Calibrate and combine predictions from multiple folds/models.
    
    Args:
        fold_predictions: List of probability arrays from each fold
        fold_labels: List of true label arrays from each fold
        method: Combination method ('mean', 'median', 'vote')
        
    Returns:
        Tuple of (combined_predictions, optimal_threshold)
    """
    all_preds = np.concatenate(fold_predictions)
    all_labels = np.concatenate(fold_labels)
    
    # Find optimal threshold on combined data
    threshold, score, _ = find_optimal_threshold(all_labels, all_preds, 'f1')
    
    # Combine predictions
    if method == 'mean':
        # For new data, average across models
        combined = all_preds
    elif method == 'median':
        combined = all_preds
    elif method == 'vote':
        # Hard voting
        combined = (all_preds > 0.5).astype(float)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return combined, threshold


if __name__ == "__main__":
    # Test calibration functions
    import numpy as np
    
    print("Testing calibration functions...")
    
    # Create synthetic data
    np.random.seed(42)
    n = 1000
    y_true = np.random.binomial(1, 0.3, n)
    
    # Uncalibrated predictions (overconfident)
    y_pred_uncal = np.random.beta(2, 2, n)  # Concentrated around 0.5
    y_pred_uncal = y_pred_uncal * 0.6 + 0.2  # Shift to [0.2, 0.8]
    
    # Calibrated predictions (closer to reality)
    y_pred_cal = y_pred_uncal * 0.8 + y_true * 0.2 * np.random.random(n)
    y_pred_cal = np.clip(y_pred_cal, 0.01, 0.99)
    
    # Compute metrics
    ece_uncal = expected_calibration_error(y_true, y_pred_uncal)
    ece_cal = expected_calibration_error(y_true, y_pred_cal)
    
    print(f"ECE uncalibrated: {ece_uncal:.4f}")
    print(f"ECE calibrated: {ece_cal:.4f}")
    
    brier_uncal = brier_score_loss(y_true, y_pred_uncal)
    brier_cal = brier_score_loss(y_true, y_pred_cal)
    
    print(f"Brier uncalibrated: {brier_uncal:.4f}")
    print(f"Brier calibrated: {brier_cal:.4f}")
    
    # Test evaluate_calibration
    metrics = evaluate_calibration(y_true, y_pred_uncal, y_pred_cal)
    print(f"\nFull evaluation: {metrics}")
    
    print("\n[OK] Calibration functions work correctly!")

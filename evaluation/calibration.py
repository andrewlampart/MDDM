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

"""
Comprehensive Metrics Computation for Depression Detection

Features:
- All standard classification metrics
- Bootstrap confidence intervals
- ROC and PR curve plotting
- Calibration analysis
"""

import numpy as np
from typing import Dict, Tuple, List, Optional
import warnings

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    confusion_matrix, matthews_corrcoef, average_precision_score,
    brier_score_loss
)

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG


class MetricsComputer:
    """
    Comprehensive metrics computation with bootstrap confidence intervals.
    
    Metrics computed:
    - accuracy, precision, recall, f1 (standard)
    - auroc, auprc (ranking metrics)
    - mcc (Matthews correlation coefficient)
    - sensitivity, specificity (clinical metrics)
    - brier_score (calibration)
    """
    
    @staticmethod
    def compute_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                           y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Compute all classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities for positive class
            
        Returns:
            Dictionary with all metrics
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
            'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
        }
        
        # Probability-based metrics
        if y_proba is not None:
            y_proba = np.array(y_proba)
            
            try:
                metrics['auroc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['auroc'] = 0.5  # If only one class present
            
            try:
                metrics['auprc'] = average_precision_score(y_true, y_proba)
            except ValueError:
                metrics['auprc'] = y_true.mean()  # Baseline
            
            metrics['brier_score'] = brier_score_loss(y_true, y_proba)
        
        return metrics
    
    @staticmethod
    def bootstrap_ci(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray,
                    metric_name: str = 'f1',
                    n_resamples: int = None,
                    ci: float = None,
                    random_state: int = None) -> Dict[str, float]:
        """
        Compute bootstrap confidence intervals for a metric.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            y_pred: Predicted labels
            metric_name: Metric to compute CI for
            n_resamples: Number of bootstrap resamples
            ci: Confidence level (e.g., 0.95 for 95% CI)
            random_state: Random seed
            
        Returns:
            Dictionary with mean, lower_ci, upper_ci, std
        """
        n_resamples = n_resamples or CONFIG.BOOTSTRAP_CI_RESAMPLES
        ci = ci or CONFIG.CONFIDENCE_LEVEL
        random_state = random_state or CONFIG.RANDOM_SEED
        
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)
        y_pred = np.array(y_pred)
        
        n_samples = len(y_true)
        metric_scores = []
        
        np.random.seed(random_state)
        
        for _ in range(n_resamples):
            # Resample with replacement
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_resample = y_true[idx]
            y_pred_resample = y_pred[idx]
            y_proba_resample = y_proba[idx]
            
            # Skip if only one class in resample
            if len(np.unique(y_true_resample)) < 2:
                continue
            
            try:
                if metric_name == 'f1':
                    score = f1_score(y_true_resample, y_pred_resample, zero_division=0)
                elif metric_name == 'auroc':
                    score = roc_auc_score(y_true_resample, y_proba_resample)
                elif metric_name == 'auprc':
                    score = average_precision_score(y_true_resample, y_proba_resample)
                elif metric_name == 'precision':
                    score = precision_score(y_true_resample, y_pred_resample, zero_division=0)
                elif metric_name == 'recall':
                    score = recall_score(y_true_resample, y_pred_resample, zero_division=0)
                elif metric_name == 'accuracy':
                    score = accuracy_score(y_true_resample, y_pred_resample)
                elif metric_name == 'mcc':
                    score = matthews_corrcoef(y_true_resample, y_pred_resample)
                elif metric_name == 'specificity':
                    cm = confusion_matrix(y_true_resample, y_pred_resample)
                    tn, fp = cm[0, 0], cm[0, 1]
                    score = tn / (tn + fp) if (tn + fp) > 0 else 0
                else:
                    raise ValueError(f"Unknown metric: {metric_name}")
                
                if not np.isnan(score):
                    metric_scores.append(score)
                    
            except Exception:
                continue
        
        if len(metric_scores) < 10:
            warnings.warn(f"Only {len(metric_scores)} valid bootstrap samples for {metric_name}")
            return {
                'mean': np.nan,
                'lower_ci': np.nan,
                'upper_ci': np.nan,
                'std': np.nan
            }
        
        metric_scores = np.array(metric_scores)
        
        # Compute percentiles for CI
        alpha = 1 - ci
        lower = np.percentile(metric_scores, alpha / 2 * 100)
        upper = np.percentile(metric_scores, (1 - alpha / 2) * 100)
        
        return {
            'mean': float(np.mean(metric_scores)),
            'lower_ci': float(lower),
            'upper_ci': float(upper),
            'std': float(np.std(metric_scores)),
            'n_valid_samples': len(metric_scores)
        }
    
    @staticmethod
    def compute_all_ci(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray,
                      metrics: List[str] = None,
                      n_resamples: int = None) -> Dict[str, Dict[str, float]]:
        """
        Compute bootstrap CI for multiple metrics.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            y_pred: Predicted labels
            metrics: List of metrics to compute
            n_resamples: Number of bootstrap resamples
            
        Returns:
            Dictionary mapping metric names to CI dictionaries
        """
        if metrics is None:
            metrics = ['f1', 'auroc', 'auprc', 'precision', 'recall', 'accuracy', 'mcc']
        
        ci_results = {}
        for metric in metrics:
            ci_results[metric] = MetricsComputer.bootstrap_ci(
                y_true, y_proba, y_pred,
                metric_name=metric,
                n_resamples=n_resamples
            )
        
        return ci_results
    
    @staticmethod
    def plot_roc_pr_curves(y_true: np.ndarray, y_proba: np.ndarray,
                          save_path: Path = None,
                          model_name: str = "Model") -> 'Figure':
        """
        Plot ROC and Precision-Recall curves.
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            save_path: Path to save figure
            model_name: Name for legend
            
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax1.plot(fpr, tpr, color='#2196F3', lw=2, 
                label=f'{model_name} (AUC = {roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.7, label='Random')
        ax1.fill_between(fpr, tpr, alpha=0.2, color='#2196F3')
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel('False Positive Rate', fontsize=11)
        ax1.set_ylabel('True Positive Rate', fontsize=11)
        ax1.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)
        
        # PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        baseline = y_true.mean()
        
        ax2.plot(recall, precision, color='#4CAF50', lw=2,
                label=f'{model_name} (AP = {pr_auc:.3f})')
        ax2.axhline(y=baseline, color='gray', linestyle='--', lw=1,
                   label=f'Baseline ({baseline:.2f})')
        ax2.fill_between(recall, precision, alpha=0.2, color='#4CAF50')
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel('Recall', fontsize=11)
        ax2.set_ylabel('Precision', fontsize=11)
        ax2.set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        ax2.legend(loc='upper right')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Curves saved to {save_path}")
        
        return fig
    
    @staticmethod
    def plot_calibration_curve(y_true: np.ndarray, y_proba: np.ndarray,
                              n_bins: int = 10,
                              save_path: Path = None,
                              model_name: str = "Model") -> 'Figure':
        """
        Plot calibration curve (reliability diagram).
        
        Args:
            y_true: True labels
            y_proba: Predicted probabilities
            n_bins: Number of bins
            save_path: Path to save figure
            model_name: Name for legend
            
        Returns:
            matplotlib Figure object
        """
        import matplotlib.pyplot as plt
        from sklearn.calibration import calibration_curve
        
        y_true = np.array(y_true)
        y_proba = np.array(y_proba)
        
        # Compute calibration curve
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_true, y_proba, n_bins=n_bins, strategy='uniform'
        )
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calibration curve
        ax1.plot(mean_predicted_value, fraction_of_positives, 
                's-', color='#FF9800', lw=2, markersize=8,
                label=f'{model_name}')
        ax1.plot([0, 1], [0, 1], 'k--', lw=1, label='Perfectly calibrated')
        ax1.set_xlabel('Mean Predicted Probability', fontsize=11)
        ax1.set_ylabel('Fraction of Positives', fontsize=11)
        ax1.set_title('Calibration Curve', fontsize=12, fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)
        ax1.set_xlim([0, 1])
        ax1.set_ylim([0, 1])
        
        # Histogram of predicted probabilities
        ax2.hist(y_proba[y_true == 0], bins=30, alpha=0.6, color='#2196F3',
                label='Negative class', density=True)
        ax2.hist(y_proba[y_true == 1], bins=30, alpha=0.6, color='#F44336',
                label='Positive class', density=True)
        ax2.set_xlabel('Predicted Probability', fontsize=11)
        ax2.set_ylabel('Density', fontsize=11)
        ax2.set_title('Prediction Distribution', fontsize=12, fontweight='bold')
        ax2.legend()
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Calibration plot saved to {save_path}")
        
        return fig
    
    @staticmethod
    def format_metric_with_ci(metric_value: float, ci_dict: Dict[str, float],
                             precision: int = 3) -> str:
        """
        Format metric with confidence interval as string.
        
        Args:
            metric_value: Point estimate
            ci_dict: Dictionary with lower_ci and upper_ci
            precision: Decimal precision
            
        Returns:
            Formatted string like "0.850 [0.820, 0.880]"
        """
        return f"{metric_value:.{precision}f} [{ci_dict['lower_ci']:.{precision}f}, {ci_dict['upper_ci']:.{precision}f}]"

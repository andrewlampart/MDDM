"""
Monte Carlo Dropout for Uncertainty Estimation

MC Dropout estimates model uncertainty by running multiple forward passes
with dropout enabled during inference.

High uncertainty may indicate:
- Out-of-distribution samples
- Ambiguous cases
- Potential misclassification
"""

import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    warnings.warn("PyTorch not available. MC Dropout disabled.")


if TORCH_AVAILABLE:
    
    class MCDropoutEstimator:
        """
        Monte Carlo Dropout for uncertainty estimation.
        
        Runs multiple forward passes with dropout enabled to estimate
        predictive uncertainty.
        """
        
        def __init__(self, model: nn.Module, n_samples: int = None,
                    device: str = None):
            """
            Args:
                model: PyTorch model with dropout layers
                n_samples: Number of MC samples
                device: Device to use
            """
            self.model = model
            self.n_samples = n_samples or CONFIG.MC_DROPOUT_SAMPLES
            self.device = device or CONFIG.DEVICE
        
        def enable_dropout(self):
            """Enable dropout layers during evaluation"""
            for module in self.model.modules():
                if isinstance(module, nn.Dropout) or isinstance(module, nn.Dropout2d):
                    module.train()
        
        def predict_with_uncertainty(self, *args, **kwargs) -> Tuple[np.ndarray, np.ndarray]:
            """
            Make predictions with uncertainty estimation.
            
            Args:
                *args, **kwargs: Arguments to model.forward()
                
            Returns:
                Tuple of (mean_predictions, std_uncertainty)
            """
            self.model.eval()
            self.enable_dropout()
            
            predictions = []
            
            with torch.no_grad():
                for _ in range(self.n_samples):
                    output = self.model(*args, **kwargs)
                    
                    # Handle tuple outputs (e.g., from fusion models)
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    predictions.append(output.cpu().numpy())
            
            predictions = np.array(predictions)  # (n_samples, batch_size)
            
            mean_pred = predictions.mean(axis=0)
            std_pred = predictions.std(axis=0)
            
            return mean_pred, std_pred
        
        def predict_with_full_info(self, *args, **kwargs) -> Dict:
            """
            Get full prediction information including uncertainty metrics.
            
            Returns:
                Dictionary with mean, std, percentiles, etc.
            """
            self.model.eval()
            self.enable_dropout()
            
            predictions = []
            
            with torch.no_grad():
                for _ in range(self.n_samples):
                    output = self.model(*args, **kwargs)
                    
                    if isinstance(output, tuple):
                        output = output[0]
                    
                    predictions.append(output.cpu().numpy())
            
            predictions = np.array(predictions)
            
            return {
                'mean': predictions.mean(axis=0),
                'std': predictions.std(axis=0),
                'median': np.median(predictions, axis=0),
                'percentile_5': np.percentile(predictions, 5, axis=0),
                'percentile_95': np.percentile(predictions, 95, axis=0),
                'min': predictions.min(axis=0),
                'max': predictions.max(axis=0),
                'all_samples': predictions
            }
        
        def calibration_analysis(self, predictions: np.ndarray,
                                uncertainties: np.ndarray,
                                labels: np.ndarray) -> Dict:
            """
            Analyze calibration of uncertainty estimates.
            
            A well-calibrated model should be more uncertain when wrong.
            
            Args:
                predictions: Mean predictions
                uncertainties: Prediction uncertainties (std)
                labels: True labels
                
            Returns:
                Calibration metrics
            """
            pred_binary = (predictions >= 0.5).astype(int)
            correct = (pred_binary == labels).astype(int)
            
            # Mean uncertainty for correct vs incorrect predictions
            mean_unc_correct = uncertainties[correct == 1].mean() if (correct == 1).any() else 0
            mean_unc_incorrect = uncertainties[correct == 0].mean() if (correct == 0).any() else 0
            
            # Correlation between uncertainty and error
            from scipy.stats import spearmanr
            try:
                corr, p_value = spearmanr(uncertainties, 1 - correct)
            except:
                corr, p_value = 0, 1
            
            # Selective prediction: accuracy when filtering high-uncertainty
            n_samples = len(predictions)
            selective_metrics = {}
            
            for threshold_percentile in [90, 80, 70, 50]:
                unc_threshold = np.percentile(uncertainties, threshold_percentile)
                mask = uncertainties <= unc_threshold
                
                if mask.sum() > 0:
                    selective_metrics[f'acc_below_p{threshold_percentile}'] = {
                        'accuracy': (pred_binary[mask] == labels[mask]).mean(),
                        'coverage': mask.mean(),
                        'n_samples': mask.sum()
                    }
            
            return {
                'mean_uncertainty_correct': mean_unc_correct,
                'mean_uncertainty_incorrect': mean_unc_incorrect,
                'uncertainty_error_correlation': corr,
                'correlation_p_value': p_value,
                'selective_prediction': selective_metrics
            }
        
        def plot_uncertainty_analysis(self, predictions: np.ndarray,
                                      uncertainties: np.ndarray,
                                      labels: np.ndarray,
                                      save_path: Path = None):
            """
            Create uncertainty analysis plots.
            
            Args:
                predictions: Mean predictions
                uncertainties: Prediction uncertainties
                labels: True labels
                save_path: Path to save figure
            """
            import matplotlib.pyplot as plt
            
            pred_binary = (predictions >= 0.5).astype(int)
            correct = (pred_binary == labels).astype(int)
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))
            
            # 1. Uncertainty distribution by correctness
            ax1 = axes[0, 0]
            ax1.hist(uncertainties[correct == 1], bins=20, alpha=0.6, 
                    color='#4CAF50', label='Correct', density=True)
            ax1.hist(uncertainties[correct == 0], bins=20, alpha=0.6,
                    color='#F44336', label='Incorrect', density=True)
            ax1.set_xlabel('Uncertainty (Std)', fontsize=11)
            ax1.set_ylabel('Density', fontsize=11)
            ax1.set_title('Uncertainty Distribution', fontsize=12, fontweight='bold')
            ax1.legend()
            ax1.grid(alpha=0.3)
            
            # 2. Prediction vs Uncertainty scatter
            ax2 = axes[0, 1]
            colors = ['#4CAF50' if c else '#F44336' for c in correct]
            ax2.scatter(predictions, uncertainties, c=colors, alpha=0.6, s=50)
            ax2.axhline(y=np.median(uncertainties), color='gray', linestyle='--', alpha=0.7)
            ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
            ax2.set_xlabel('Mean Prediction', fontsize=11)
            ax2.set_ylabel('Uncertainty (Std)', fontsize=11)
            ax2.set_title('Prediction vs Uncertainty', fontsize=12, fontweight='bold')
            ax2.grid(alpha=0.3)
            
            # Add legend manually
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='#4CAF50', alpha=0.6, label='Correct'),
                Patch(facecolor='#F44336', alpha=0.6, label='Incorrect')
            ]
            ax2.legend(handles=legend_elements)
            
            # 3. Selective prediction accuracy curve
            ax3 = axes[1, 0]
            coverages = []
            accuracies = []
            
            for percentile in range(0, 101, 5):
                if percentile == 0:
                    mask = np.ones(len(predictions), dtype=bool)
                else:
                    unc_threshold = np.percentile(uncertainties, percentile)
                    mask = uncertainties <= unc_threshold
                
                if mask.sum() > 0:
                    coverages.append(mask.mean())
                    accuracies.append((pred_binary[mask] == labels[mask]).mean())
            
            ax3.plot(coverages, accuracies, 'o-', color='#2196F3', linewidth=2)
            ax3.set_xlabel('Coverage (fraction of data)', fontsize=11)
            ax3.set_ylabel('Accuracy', fontsize=11)
            ax3.set_title('Selective Prediction Curve', fontsize=12, fontweight='bold')
            ax3.grid(alpha=0.3)
            
            # 4. Calibration metrics table
            ax4 = axes[1, 1]
            ax4.axis('off')
            
            calibration = self.calibration_analysis(predictions, uncertainties, labels)
            
            text = f"""Calibration Analysis
            
Mean Uncertainty (Correct):   {calibration['mean_uncertainty_correct']:.4f}
Mean Uncertainty (Incorrect): {calibration['mean_uncertainty_incorrect']:.4f}

Uncertainty-Error Correlation: {calibration['uncertainty_error_correlation']:.3f}
(p-value: {calibration['correlation_p_value']:.4f})

Selective Prediction:
"""
            for key, val in calibration['selective_prediction'].items():
                text += f"\n{key}: Acc={val['accuracy']:.3f}, Coverage={val['coverage']:.1%}"
            
            ax4.text(0.1, 0.9, text, transform=ax4.transAxes, fontsize=11,
                    verticalalignment='top', fontfamily='monospace')
            
            plt.tight_layout()
            
            if save_path:
                save_path = Path(save_path)
                save_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Uncertainty analysis saved to {save_path}")
            
            return fig


if __name__ == '__main__':
    if not TORCH_AVAILABLE:
        print("PyTorch required")
    else:
        print("MC Dropout module loaded successfully")
        
        # Quick test with dummy model
        class DummyModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(10, 1)
                self.dropout = nn.Dropout(0.5)
            
            def forward(self, x):
                return torch.sigmoid(self.fc(self.dropout(x)))
        
        model = DummyModel()
        estimator = MCDropoutEstimator(model, n_samples=10, device='cpu')
        
        x = torch.randn(5, 10)
        mean, std = estimator.predict_with_uncertainty(x)
        
        print(f"Mean predictions: {mean}")
        print(f"Uncertainty (std): {std}")

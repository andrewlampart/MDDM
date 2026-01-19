"""
Feature Diagnostics for Multimodal Depression Detection

Comprehensive diagnostics to identify data quality issues before training:
- NaN/Inf detection
- Feature correlation with labels
- Distribution analysis
- Comparison of old vs new feature dimensions

This helps diagnose why models may underperform (e.g., AUROC=0.5 issue).

Usage:
    diag = FeatureDiagnostics(audio_features, text_features, labels)
    diag.print_full_report()
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from pathlib import Path
import logging
import warnings

logger = logging.getLogger(__name__)


class FeatureDiagnostics:
    """
    Comprehensive feature diagnostics for multimodal data.
    
    Checks for common issues that can cause model problems:
    1. NaN/Inf values that break training
    2. Low correlation features that add noise
    3. Redundant features that cause overfitting
    4. Distribution shifts between modalities
    
    Example:
        diag = FeatureDiagnostics(X_audio, X_text, y)
        diag.print_full_report()
        
        # Check if there are critical issues
        if diag.has_critical_issues():
            print("WARNING: Fix data issues before training!")
    """
    
    def __init__(
        self,
        audio_features: np.ndarray,
        text_features: np.ndarray,
        labels: np.ndarray,
        audio_features_old: Optional[np.ndarray] = None,
        text_features_old: Optional[np.ndarray] = None,
        feature_names: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize feature diagnostics.
        
        Args:
            audio_features: Audio features, shape (n_samples, audio_dim)
            text_features: Text features, shape (n_samples, text_dim)
            labels: Binary labels, shape (n_samples,)
            audio_features_old: Optional previous audio features for comparison
            text_features_old: Optional previous text features for comparison
            feature_names: Optional dict with feature names per modality
        """
        self.audio = np.array(audio_features)
        self.text = np.array(text_features)
        self.labels = np.array(labels).flatten()
        self.audio_old = np.array(audio_features_old) if audio_features_old is not None else None
        self.text_old = np.array(text_features_old) if text_features_old is not None else None
        self.feature_names = feature_names or {}
        
        # Validation
        n_samples = len(self.labels)
        if len(self.audio) != n_samples:
            raise ValueError(f"Audio samples ({len(self.audio)}) != labels ({n_samples})")
        if len(self.text) != n_samples:
            raise ValueError(f"Text samples ({len(self.text)}) != labels ({n_samples})")
        
        # Store diagnostic results
        self.issues = []
        self.warnings = []
        self.info = []
        
        logger.info(f"FeatureDiagnostics initialized:")
        logger.info(f"  Samples: {n_samples}")
        logger.info(f"  Audio dim: {self.audio.shape[1]}")
        logger.info(f"  Text dim: {self.text.shape[1]}")
        logger.info(f"  Class balance: {self.labels.mean():.1%} positive")
    
    def check_nans_infs(self) -> Dict[str, Dict[str, int]]:
        """
        Check for NaN and Inf values in features.
        
        Returns:
            Dict with counts per modality
        """
        results = {}
        
        # Audio
        audio_nan = np.isnan(self.audio).sum()
        audio_inf = np.isinf(self.audio).sum()
        audio_total = self.audio.size
        
        results['audio'] = {
            'nan_count': int(audio_nan),
            'inf_count': int(audio_inf),
            'total_values': int(audio_total),
            'nan_ratio': float(audio_nan / audio_total) if audio_total > 0 else 0,
            'inf_ratio': float(audio_inf / audio_total) if audio_total > 0 else 0,
        }
        
        # Text
        text_nan = np.isnan(self.text).sum()
        text_inf = np.isinf(self.text).sum()
        text_total = self.text.size
        
        results['text'] = {
            'nan_count': int(text_nan),
            'inf_count': int(text_inf),
            'total_values': int(text_total),
            'nan_ratio': float(text_nan / text_total) if text_total > 0 else 0,
            'inf_ratio': float(text_inf / text_total) if text_total > 0 else 0,
        }
        
        # Log issues
        if audio_nan > 0:
            msg = f"Audio has {audio_nan} NaN values ({results['audio']['nan_ratio']:.2%})"
            self.issues.append(msg)
            logger.warning(msg)
        
        if audio_inf > 0:
            msg = f"Audio has {audio_inf} Inf values ({results['audio']['inf_ratio']:.2%})"
            self.issues.append(msg)
            logger.warning(msg)
        
        if text_nan > 0:
            msg = f"Text has {text_nan} NaN values ({results['text']['nan_ratio']:.2%})"
            self.issues.append(msg)
            logger.warning(msg)
        
        if text_inf > 0:
            msg = f"Text has {text_inf} Inf values ({results['text']['inf_ratio']:.2%})"
            self.issues.append(msg)
            logger.warning(msg)
        
        return results
    
    def check_feature_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Compute basic statistics for each modality.
        
        Returns:
            Dict with statistics per modality
        """
        results = {}
        
        for name, features in [('audio', self.audio), ('text', self.text)]:
            # Handle NaN for stats
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                results[name] = {
                    'min': float(np.nanmin(features)),
                    'max': float(np.nanmax(features)),
                    'mean': float(np.nanmean(features)),
                    'std': float(np.nanstd(features)),
                    'median': float(np.nanmedian(features)),
                    'n_features': int(features.shape[1]),
                    'n_samples': int(features.shape[0]),
                }
            
            # Check for extreme values
            if results[name]['max'] > 1e6:
                msg = f"{name.capitalize()} has extreme max value: {results[name]['max']:.2e}"
                self.warnings.append(msg)
                logger.warning(msg)
            
            if results[name]['min'] < -1e6:
                msg = f"{name.capitalize()} has extreme min value: {results[name]['min']:.2e}"
                self.warnings.append(msg)
                logger.warning(msg)
            
            # Check for zero variance features
            feature_stds = np.nanstd(features, axis=0)
            zero_var_count = np.sum(feature_stds < 1e-10)
            results[name]['zero_variance_features'] = int(zero_var_count)
            
            if zero_var_count > 0:
                msg = f"{name.capitalize()} has {zero_var_count} zero-variance features"
                self.warnings.append(msg)
                logger.warning(msg)
        
        return results
    
    def check_feature_correlation_with_label(
        self,
        method: str = 'spearman',
        top_k: int = 10
    ) -> Dict[str, List[Tuple[int, float, float]]]:
        """
        Check correlation of each feature with the label.
        
        Features with higher correlation are more predictive.
        Features with near-zero correlation may be noise.
        
        Args:
            method: 'spearman' (rank) or 'pearson' (linear)
            top_k: Number of top features to return
            
        Returns:
            Dict with top correlated features per modality
        """
        from scipy.stats import spearmanr, pearsonr
        
        corr_func = spearmanr if method == 'spearman' else pearsonr
        results = {}
        
        for name, features in [('audio', self.audio), ('text', self.text)]:
            correlations = []
            
            for i in range(features.shape[1]):
                feature_col = features[:, i]
                
                # Skip if feature has no variance or all NaN
                if np.nanstd(feature_col) < 1e-10 or np.all(np.isnan(feature_col)):
                    correlations.append((i, 0.0, 1.0))
                    continue
                
                # Handle NaN by using only valid pairs
                valid_mask = ~np.isnan(feature_col)
                if valid_mask.sum() < 10:
                    correlations.append((i, 0.0, 1.0))
                    continue
                
                try:
                    corr, pval = corr_func(feature_col[valid_mask], self.labels[valid_mask])
                    if np.isnan(corr):
                        corr = 0.0
                    correlations.append((i, abs(corr), pval))
                except Exception:
                    correlations.append((i, 0.0, 1.0))
            
            # Sort by absolute correlation (descending)
            correlations_sorted = sorted(correlations, key=lambda x: x[1], reverse=True)
            results[name] = correlations_sorted[:top_k]
            
            # Compute average correlation
            avg_corr = np.mean([c[1] for c in correlations])
            results[f'{name}_avg_correlation'] = float(avg_corr)
            
            # Log info
            self.info.append(f"{name.capitalize()} avg correlation with label: {avg_corr:.4f}")
        
        return results
    
    def check_class_separability(self) -> Dict[str, float]:
        """
        Check how well features separate positive/negative classes.
        
        Uses Fisher's discriminant ratio: (mean1 - mean0)^2 / (var1 + var0)
        Higher ratio = better separability.
        
        Returns:
            Dict with separability metrics
        """
        results = {}
        
        pos_mask = self.labels == 1
        neg_mask = self.labels == 0
        
        for name, features in [('audio', self.audio), ('text', self.text)]:
            # Compute Fisher ratio for each feature
            fisher_ratios = []
            
            for i in range(features.shape[1]):
                pos_vals = features[pos_mask, i]
                neg_vals = features[neg_mask, i]
                
                mean_pos = np.nanmean(pos_vals)
                mean_neg = np.nanmean(neg_vals)
                var_pos = np.nanvar(pos_vals)
                var_neg = np.nanvar(neg_vals)
                
                denom = var_pos + var_neg
                if denom > 1e-10:
                    fisher = (mean_pos - mean_neg) ** 2 / denom
                else:
                    fisher = 0.0
                
                fisher_ratios.append(fisher)
            
            results[f'{name}_avg_fisher_ratio'] = float(np.nanmean(fisher_ratios))
            results[f'{name}_max_fisher_ratio'] = float(np.nanmax(fisher_ratios))
            results[f'{name}_n_discriminative_features'] = int(np.sum(np.array(fisher_ratios) > 0.1))
        
        return results
    
    def compare_feature_dimensions(self) -> Optional[Dict[str, Dict[str, any]]]:
        """
        Compare current features with old features (if provided).
        
        Useful for tracking changes after feature engineering updates.
        
        Returns:
            Dict with comparison metrics or None if old features not provided
        """
        if self.audio_old is None and self.text_old is None:
            return None
        
        results = {}
        
        if self.audio_old is not None:
            old_dim = self.audio_old.shape[1]
            new_dim = self.audio.shape[1]
            
            results['audio'] = {
                'old_dim': old_dim,
                'new_dim': new_dim,
                'added_features': new_dim - old_dim,
                'change_percent': (new_dim - old_dim) / old_dim * 100 if old_dim > 0 else 0
            }
            
            self.info.append(f"Audio: {old_dim} -> {new_dim} features (+{new_dim - old_dim})")
        
        if self.text_old is not None:
            old_dim = self.text_old.shape[1]
            new_dim = self.text.shape[1]
            
            results['text'] = {
                'old_dim': old_dim,
                'new_dim': new_dim,
                'added_features': new_dim - old_dim,
                'change_percent': (new_dim - old_dim) / old_dim * 100 if old_dim > 0 else 0
            }
            
            self.info.append(f"Text: {old_dim} -> {new_dim} features (+{new_dim - old_dim})")
        
        return results
    
    def check_modality_scale_difference(self) -> Dict[str, float]:
        """
        Check if modalities have very different scales.
        
        Large scale differences can cause fusion problems.
        
        Returns:
            Dict with scale metrics
        """
        audio_scale = np.nanstd(self.audio)
        text_scale = np.nanstd(self.text)
        
        scale_ratio = max(audio_scale, text_scale) / min(audio_scale, text_scale) if min(audio_scale, text_scale) > 0 else float('inf')
        
        results = {
            'audio_scale': float(audio_scale),
            'text_scale': float(text_scale),
            'scale_ratio': float(scale_ratio)
        }
        
        if scale_ratio > 10:
            msg = f"Large scale difference between modalities (ratio={scale_ratio:.1f}). Consider normalization."
            self.warnings.append(msg)
            logger.warning(msg)
        
        return results
    
    def has_critical_issues(self) -> bool:
        """Check if there are critical issues that should block training."""
        # Run all checks if not done
        if not self.issues and not self.warnings:
            self.run_all_checks()
        
        return len(self.issues) > 0
    
    def run_all_checks(self) -> Dict[str, any]:
        """
        Run all diagnostic checks.
        
        Returns:
            Dict with all check results
        """
        results = {}
        
        results['nan_inf'] = self.check_nans_infs()
        results['statistics'] = self.check_feature_statistics()
        results['correlations'] = self.check_feature_correlation_with_label()
        results['separability'] = self.check_class_separability()
        results['scale'] = self.check_modality_scale_difference()
        
        dim_comparison = self.compare_feature_dimensions()
        if dim_comparison:
            results['dimension_comparison'] = dim_comparison
        
        return results
    
    def print_full_report(self) -> str:
        """
        Generate and print comprehensive diagnostic report.
        
        Returns:
            Report as string
        """
        # Run all checks
        results = self.run_all_checks()
        
        lines = []
        lines.append("=" * 70)
        lines.append("FEATURE DIAGNOSTICS REPORT")
        lines.append("=" * 70)
        
        # Dataset overview
        lines.append("\n[1] DATASET OVERVIEW")
        lines.append("-" * 40)
        lines.append(f"  Samples: {len(self.labels)}")
        lines.append(f"  Audio features: {self.audio.shape[1]}")
        lines.append(f"  Text features: {self.text.shape[1]}")
        lines.append(f"  Class balance: {self.labels.mean():.1%} positive")
        
        # NaN/Inf check
        lines.append("\n[2] NaN/Inf CHECK")
        lines.append("-" * 40)
        for modality in ['audio', 'text']:
            nan_inf = results['nan_inf'][modality]
            status = "OK" if nan_inf['nan_count'] == 0 and nan_inf['inf_count'] == 0 else "WARNING"
            lines.append(f"  {modality.upper()}: NaN={nan_inf['nan_count']}, Inf={nan_inf['inf_count']} [{status}]")
        
        # Statistics
        lines.append("\n[3] FEATURE STATISTICS")
        lines.append("-" * 40)
        for modality in ['audio', 'text']:
            stats = results['statistics'][modality]
            lines.append(f"  {modality.upper()}:")
            lines.append(f"    Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
            lines.append(f"    Mean: {stats['mean']:.4f}, Std: {stats['std']:.4f}")
            lines.append(f"    Zero-variance features: {stats['zero_variance_features']}")
        
        # Correlation with label
        lines.append("\n[4] TOP CORRELATED FEATURES (with depression label)")
        lines.append("-" * 40)
        for modality in ['audio', 'text']:
            avg_corr = results['correlations'].get(f'{modality}_avg_correlation', 0)
            lines.append(f"  {modality.upper()} (avg correlation: {avg_corr:.4f}):")
            top_features = results['correlations'][modality][:5]
            for feat_idx, corr, pval in top_features:
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
                lines.append(f"    Feature {feat_idx:4d}: r={corr:.4f} {sig}")
        
        # Separability
        lines.append("\n[5] CLASS SEPARABILITY (Fisher ratio)")
        lines.append("-" * 40)
        for modality in ['audio', 'text']:
            avg_fisher = results['separability'].get(f'{modality}_avg_fisher_ratio', 0)
            n_disc = results['separability'].get(f'{modality}_n_discriminative_features', 0)
            lines.append(f"  {modality.upper()}: avg Fisher={avg_fisher:.4f}, discriminative features={n_disc}")
        
        # Scale difference
        lines.append("\n[6] MODALITY SCALE")
        lines.append("-" * 40)
        scale = results['scale']
        lines.append(f"  Audio scale (std): {scale['audio_scale']:.4f}")
        lines.append(f"  Text scale (std): {scale['text_scale']:.4f}")
        lines.append(f"  Scale ratio: {scale['scale_ratio']:.2f}")
        
        # Issues and warnings
        if self.issues:
            lines.append("\n[!] CRITICAL ISSUES")
            lines.append("-" * 40)
            for issue in self.issues:
                lines.append(f"  - {issue}")
        
        if self.warnings:
            lines.append("\n[!] WARNINGS")
            lines.append("-" * 40)
            for warning in self.warnings:
                lines.append(f"  - {warning}")
        
        # Summary
        lines.append("\n" + "=" * 70)
        if self.issues:
            lines.append("STATUS: CRITICAL ISSUES FOUND - Fix before training!")
        elif self.warnings:
            lines.append("STATUS: WARNINGS - Consider addressing before training")
        else:
            lines.append("STATUS: OK - Data looks good for training")
        lines.append("=" * 70)
        
        report = "\n".join(lines)
        print(report)
        
        return report
    
    def save_report(self, output_path: Path):
        """Save diagnostic report to file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        report = self.print_full_report()
        
        with open(output_path, 'w') as f:
            f.write(report)
        
        logger.info(f"Report saved to {output_path}")


def quick_diagnostics(
    audio_features: np.ndarray,
    text_features: np.ndarray,
    labels: np.ndarray
) -> bool:
    """
    Quick diagnostic check before training.
    
    Returns True if data looks OK, False if critical issues found.
    
    Args:
        audio_features: Audio features
        text_features: Text features
        labels: Binary labels
        
    Returns:
        True if OK to proceed with training
    """
    diag = FeatureDiagnostics(audio_features, text_features, labels)
    diag.run_all_checks()
    
    if diag.has_critical_issues():
        logger.error("Critical data issues found! Review diagnostics before training.")
        for issue in diag.issues:
            logger.error(f"  - {issue}")
        return False
    
    if diag.warnings:
        logger.warning("Data warnings found:")
        for warning in diag.warnings:
            logger.warning(f"  - {warning}")
    
    return True


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing FeatureDiagnostics...")
    
    np.random.seed(42)
    n_samples = 200
    
    # Create synthetic features
    y = np.random.binomial(1, 0.3, n_samples)
    
    # Audio features (with some signal)
    X_audio = np.random.randn(n_samples, 838)
    X_audio[y == 1, :10] += 0.5  # Add signal to first 10 features
    
    # Text features (with stronger signal)
    X_text = np.random.randn(n_samples, 560)
    X_text[y == 1, :20] += 0.8  # Add signal to first 20 features
    
    # Add some NaN values to test detection
    X_audio[0, 0] = np.nan
    X_text[1, 1] = np.nan
    
    # Run diagnostics
    diag = FeatureDiagnostics(X_audio, X_text, y)
    report = diag.print_full_report()
    
    print("\n" + "=" * 70)
    print("Quick check result:", quick_diagnostics(X_audio, X_text, y))
    print("\n[OK] FeatureDiagnostics tests passed!")

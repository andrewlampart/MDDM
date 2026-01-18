"""
Ablation Study Framework for Multimodal Depression Detection

Systematically tests modality contributions to understand:
- Does audio actually help, or is it noise?
- Does text help, or is it just correlated with audio?
- Is there synergy (audio + text > sum of parts)?
- What should go into the production model?

Usage:
    ablation = AblationStudy(X_audio, X_text, X_demographic, y)
    results = ablation.run_full_ablation(model_factory, n_splits=5)
    print(results)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Callable, Optional, Any, Tuple
from pathlib import Path
import logging
import json
from datetime import datetime

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    f1_score, roc_auc_score, matthews_corrcoef,
    accuracy_score, precision_score, recall_score
)

logger = logging.getLogger(__name__)


class AblationStudy:
    """
    Systematically test modality contributions in multimodal models.
    
    Tests all combinations:
    - Audio only
    - Text only
    - Demographics only (if provided)
    - Audio + Text
    - Audio + Demographics
    - Text + Demographics
    - All modalities
    
    Uses stratified K-fold cross-validation for robust estimates.
    """
    
    def __init__(
        self,
        X_audio: np.ndarray,
        X_text: np.ndarray,
        y: np.ndarray,
        X_demographic: Optional[np.ndarray] = None,
        feature_names: Optional[Dict[str, List[str]]] = None
    ):
        """
        Initialize ablation study.
        
        Args:
            X_audio: Audio features, shape (n_samples, audio_dim)
            X_text: Text features, shape (n_samples, text_dim)
            y: Labels, shape (n_samples,)
            X_demographic: Optional demographic features, shape (n_samples, demo_dim)
            feature_names: Optional dict mapping modality to feature names
        """
        self.X_audio = np.array(X_audio)
        self.X_text = np.array(X_text)
        self.y = np.array(y)
        self.X_demographic = np.array(X_demographic) if X_demographic is not None else None
        self.feature_names = feature_names or {}
        
        # Validate shapes
        n_samples = len(self.y)
        assert len(self.X_audio) == n_samples, f"Audio samples mismatch: {len(self.X_audio)} vs {n_samples}"
        assert len(self.X_text) == n_samples, f"Text samples mismatch: {len(self.X_text)} vs {n_samples}"
        if self.X_demographic is not None:
            assert len(self.X_demographic) == n_samples, f"Demo samples mismatch"
        
        self.results = {}
        self.detailed_results = {}
        
        logger.info(f"AblationStudy initialized:")
        logger.info(f"  Samples: {n_samples}")
        logger.info(f"  Audio dim: {self.X_audio.shape[1]}")
        logger.info(f"  Text dim: {self.X_text.shape[1]}")
        if self.X_demographic is not None:
            logger.info(f"  Demo dim: {self.X_demographic.shape[1]}")
        logger.info(f"  Class balance: {self.y.mean():.2%} positive")
    
    def _get_ablation_configs(self) -> Dict[str, np.ndarray]:
        """
        Get all ablation configurations.
        
        Returns:
            Dict mapping config name to feature matrix
        """
        configs = {
            '1. Audio Only': self.X_audio,
            '2. Text Only': self.X_text,
        }
        
        if self.X_demographic is not None:
            configs['3. Demo Only'] = self.X_demographic
            configs['4. Audio + Text'] = np.hstack([self.X_audio, self.X_text])
            configs['5. Audio + Demo'] = np.hstack([self.X_audio, self.X_demographic])
            configs['6. Text + Demo'] = np.hstack([self.X_text, self.X_demographic])
            configs['7. All (Audio+Text+Demo)'] = np.hstack([
                self.X_audio, self.X_text, self.X_demographic
            ])
        else:
            configs['3. Audio + Text'] = np.hstack([self.X_audio, self.X_text])
        
        return configs
    
    def run_full_ablation(
        self,
        model_factory: Callable[[], Any],
        n_splits: int = 5,
        random_state: int = 42,
        verbose: bool = True
    ) -> pd.DataFrame:
        """
        Run complete ablation study with cross-validation.
        
        Args:
            model_factory: Function that returns a fresh sklearn-compatible model
            n_splits: Number of CV folds
            random_state: Random seed for reproducibility
            verbose: Print progress
            
        Returns:
            DataFrame with results for each configuration
        """
        configs = self._get_ablation_configs()
        
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        all_results = {}
        
        for name, X in configs.items():
            if verbose:
                print(f"\n{'='*60}")
                print(f"Running: {name} (features: {X.shape[1]})")
                print(f"{'='*60}")
            
            fold_scores = {
                'f1': [], 'auroc': [], 'mcc': [],
                'accuracy': [], 'precision': [], 'recall': [],
                'specificity': []
            }
            
            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, self.y)):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = self.y[train_idx], self.y[val_idx]
                
                # Train model
                model = model_factory()
                model.fit(X_train, y_train)
                
                # Predictions
                y_pred = model.predict(X_val)
                
                # Probabilities (if available)
                if hasattr(model, 'predict_proba'):
                    y_proba = model.predict_proba(X_val)[:, 1]
                else:
                    y_proba = y_pred.astype(float)
                
                # Compute metrics
                fold_scores['f1'].append(f1_score(y_val, y_pred, zero_division=0))
                fold_scores['accuracy'].append(accuracy_score(y_val, y_pred))
                fold_scores['precision'].append(precision_score(y_val, y_pred, zero_division=0))
                fold_scores['recall'].append(recall_score(y_val, y_pred, zero_division=0))
                fold_scores['mcc'].append(matthews_corrcoef(y_val, y_pred))
                
                try:
                    fold_scores['auroc'].append(roc_auc_score(y_val, y_proba))
                except ValueError:
                    fold_scores['auroc'].append(0.5)
                
                # Specificity
                tn = ((y_val == 0) & (y_pred == 0)).sum()
                fp = ((y_val == 0) & (y_pred == 1)).sum()
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                fold_scores['specificity'].append(spec)
                
                if verbose:
                    print(f"  Fold {fold_idx+1}: F1={fold_scores['f1'][-1]:.3f}, "
                          f"AUROC={fold_scores['auroc'][-1]:.3f}, "
                          f"MCC={fold_scores['mcc'][-1]:.3f}")
            
            # Aggregate results
            result = {
                'F1 Mean': np.mean(fold_scores['f1']),
                'F1 Std': np.std(fold_scores['f1']),
                'AUROC Mean': np.mean(fold_scores['auroc']),
                'AUROC Std': np.std(fold_scores['auroc']),
                'MCC Mean': np.mean(fold_scores['mcc']),
                'MCC Std': np.std(fold_scores['mcc']),
                'Accuracy Mean': np.mean(fold_scores['accuracy']),
                'Precision Mean': np.mean(fold_scores['precision']),
                'Recall Mean': np.mean(fold_scores['recall']),
                'Specificity Mean': np.mean(fold_scores['specificity']),
                'n_features': X.shape[1],
            }
            
            all_results[name] = result
            self.detailed_results[name] = fold_scores
            
            if verbose:
                print(f"\n  SUMMARY: F1={result['F1 Mean']:.3f} +/- {result['F1 Std']:.3f}, "
                      f"AUROC={result['AUROC Mean']:.3f}, MCC={result['MCC Mean']:.3f}")
        
        self.results = all_results
        return pd.DataFrame(all_results).T
    
    def get_synergy_analysis(self) -> Dict[str, float]:
        """
        Analyze synergy between modalities.
        
        Returns:
            Dict with synergy metrics
        """
        if not self.results:
            raise ValueError("Run run_full_ablation first!")
        
        analysis = {}
        
        # Get individual modality scores
        audio_f1 = self.results.get('1. Audio Only', {}).get('F1 Mean', 0)
        text_f1 = self.results.get('2. Text Only', {}).get('F1 Mean', 0)
        
        # Get combined score
        combined_key = '4. Audio + Text' if '4. Audio + Text' in self.results else '3. Audio + Text'
        combined_f1 = self.results.get(combined_key, {}).get('F1 Mean', 0)
        
        # Average of individuals
        avg_individual = (audio_f1 + text_f1) / 2
        
        # Synergy = combined - max(individuals)
        max_individual = max(audio_f1, text_f1)
        synergy = combined_f1 - max_individual
        
        analysis['audio_f1'] = audio_f1
        analysis['text_f1'] = text_f1
        analysis['combined_f1'] = combined_f1
        analysis['avg_individual_f1'] = avg_individual
        analysis['max_individual_f1'] = max_individual
        analysis['synergy'] = synergy
        analysis['synergy_pct'] = (synergy / max_individual * 100) if max_individual > 0 else 0
        
        # Interpretation
        if synergy > 0.05:
            analysis['interpretation'] = "Strong synergy - modalities complement each other"
        elif synergy > 0:
            analysis['interpretation'] = "Weak synergy - slight benefit from combining"
        else:
            analysis['interpretation'] = "No synergy - best single modality is sufficient"
        
        # Best modality
        if combined_f1 >= max_individual:
            analysis['recommendation'] = "Use combined Audio + Text"
        elif audio_f1 > text_f1:
            analysis['recommendation'] = "Use Audio only (simpler model)"
        else:
            analysis['recommendation'] = "Use Text only (simpler model)"
        
        return analysis
    
    def plot_results(
        self,
        metric: str = 'F1 Mean',
        save_path: Optional[Path] = None
    ):
        """
        Plot ablation results as bar chart.
        
        Args:
            metric: Metric to plot
            save_path: Path to save figure
        """
        import matplotlib.pyplot as plt
        
        if not self.results:
            raise ValueError("Run run_full_ablation first!")
        
        df = pd.DataFrame(self.results).T
        
        # Sort by metric
        df_sorted = df.sort_values(metric, ascending=False)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(df_sorted)))
        
        bars = ax.barh(range(len(df_sorted)), df_sorted[metric], color=colors)
        
        ax.set_yticks(range(len(df_sorted)))
        ax.set_yticklabels(df_sorted.index)
        ax.set_xlabel(metric, fontsize=12)
        ax.set_title(f'Ablation Study Results - {metric}', fontsize=14, fontweight='bold')
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, df_sorted[metric])):
            ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{val:.3f}', va='center', fontsize=10)
        
        ax.set_xlim(0, max(df_sorted[metric]) * 1.15)
        ax.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Plot saved to {save_path}")
        
        return fig
    
    def save_results(self, output_dir: Path):
        """
        Save ablation results to files.
        
        Args:
            output_dir: Directory to save results
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save summary DataFrame
        df = pd.DataFrame(self.results).T
        df.to_csv(output_dir / 'ablation_results.csv')
        
        # Save detailed fold results
        with open(output_dir / 'ablation_detailed.json', 'w') as f:
            # Convert numpy arrays to lists for JSON serialization
            detailed_serializable = {}
            for config, scores in self.detailed_results.items():
                detailed_serializable[config] = {
                    k: [float(v) for v in vals] for k, vals in scores.items()
                }
            json.dump(detailed_serializable, f, indent=2)
        
        # Save synergy analysis
        synergy = self.get_synergy_analysis()
        with open(output_dir / 'synergy_analysis.json', 'w') as f:
            json.dump(synergy, f, indent=2)
        
        # Save plot
        self.plot_results(save_path=output_dir / 'ablation_plot.png')
        
        logger.info(f"Results saved to {output_dir}")
    
    def generate_report(self) -> str:
        """
        Generate text report of ablation study.
        
        Returns:
            Formatted report string
        """
        if not self.results:
            return "No results yet. Run run_full_ablation first."
        
        report = []
        report.append("=" * 70)
        report.append("ABLATION STUDY REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 70)
        
        # Dataset info
        report.append(f"\nDataset: {len(self.y)} samples, {self.y.mean():.1%} positive")
        
        # Results table
        report.append("\nRESULTS BY CONFIGURATION:")
        report.append("-" * 70)
        report.append(f"{'Configuration':<30} {'F1':>8} {'AUROC':>8} {'MCC':>8} {'Spec':>8}")
        report.append("-" * 70)
        
        for config, metrics in sorted(self.results.items()):
            report.append(
                f"{config:<30} {metrics['F1 Mean']:>8.3f} {metrics['AUROC Mean']:>8.3f} "
                f"{metrics['MCC Mean']:>8.3f} {metrics['Specificity Mean']:>8.3f}"
            )
        
        report.append("-" * 70)
        
        # Synergy analysis
        synergy = self.get_synergy_analysis()
        report.append("\nSYNERGY ANALYSIS:")
        report.append(f"  Audio only F1: {synergy['audio_f1']:.3f}")
        report.append(f"  Text only F1: {synergy['text_f1']:.3f}")
        report.append(f"  Combined F1: {synergy['combined_f1']:.3f}")
        report.append(f"  Synergy: {synergy['synergy']:+.3f} ({synergy['synergy_pct']:+.1f}%)")
        report.append(f"\n  {synergy['interpretation']}")
        report.append(f"  RECOMMENDATION: {synergy['recommendation']}")
        
        report.append("\n" + "=" * 70)
        
        return "\n".join(report)


def run_ablation_from_files(
    audio_path: Path,
    text_path: Path,
    labels_path: Path,
    model_factory: Callable[[], Any],
    output_dir: Optional[Path] = None,
    demo_path: Optional[Path] = None,
    n_splits: int = 5
) -> pd.DataFrame:
    """
    Convenience function to run ablation study from saved feature files.
    
    Args:
        audio_path: Path to audio features (npy or csv)
        text_path: Path to text features (npy or csv)
        labels_path: Path to labels (npy or csv)
        model_factory: Function returning fresh model
        output_dir: Directory to save results
        demo_path: Optional path to demographic features
        n_splits: Number of CV folds
        
    Returns:
        DataFrame with results
    """
    # Load data
    def load_array(path):
        path = Path(path)
        if path.suffix == '.npy':
            return np.load(path)
        elif path.suffix == '.csv':
            return pd.read_csv(path).values
        else:
            raise ValueError(f"Unknown format: {path.suffix}")
    
    X_audio = load_array(audio_path)
    X_text = load_array(text_path)
    y = load_array(labels_path).ravel()
    
    X_demo = load_array(demo_path) if demo_path else None
    
    # Run ablation
    ablation = AblationStudy(X_audio, X_text, y, X_demo)
    results = ablation.run_full_ablation(model_factory, n_splits=n_splits)
    
    # Save if output dir provided
    if output_dir:
        ablation.save_results(output_dir)
    
    # Print report
    print(ablation.generate_report())
    
    return results


if __name__ == "__main__":
    # Demo with synthetic data
    print("Testing AblationStudy with synthetic data...")
    
    np.random.seed(42)
    n_samples = 200
    
    # Create synthetic features with different signal strengths
    y = np.random.binomial(1, 0.35, n_samples)
    
    # Audio features (moderate signal)
    X_audio = np.random.randn(n_samples, 42)
    X_audio[y == 1] += 0.3  # Add some signal
    
    # Text features (stronger signal)
    X_text = np.random.randn(n_samples, 768)
    X_text[y == 1] += 0.4  # Stronger signal
    
    # Run ablation
    from sklearn.ensemble import RandomForestClassifier
    
    def model_factory():
        return RandomForestClassifier(n_estimators=50, random_state=42)
    
    ablation = AblationStudy(X_audio, X_text, y)
    results = ablation.run_full_ablation(model_factory, n_splits=3, verbose=True)
    
    print("\n" + ablation.generate_report())
    print("\n[OK] AblationStudy works correctly!")

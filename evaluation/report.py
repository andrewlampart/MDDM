"""
Evaluation Report Generation for Depression Detection Models

Features:
- Comprehensive metrics report with CI
- JSON export for model comparison
- Formatted console output
- LaTeX table generation
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import CONFIG
from evaluation.metrics import MetricsComputer


class EvaluationReport:
    """
    Generate comprehensive evaluation reports for depression detection models.
    
    Features:
    - Point estimates for all metrics
    - Bootstrap confidence intervals
    - JSON export
    - Formatted printing
    - Comparison tables
    """
    
    def __init__(self, model_name: str, y_true: np.ndarray, 
                 y_pred: np.ndarray, y_proba: np.ndarray):
        """
        Initialize evaluation report.
        
        Args:
            model_name: Name of the model being evaluated
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Predicted probabilities (for positive class)
        """
        self.model_name = model_name
        self.y_true = np.array(y_true)
        self.y_pred = np.array(y_pred)
        self.y_proba = np.array(y_proba)
        
        self.metrics = None
        self.ci_results = {}
        self.generated_at = None
        self.n_bootstrap = None
    
    def generate(self, n_bootstrap: int = None, 
                metrics_for_ci: List[str] = None) -> 'EvaluationReport':
        """
        Generate complete evaluation report with CI.
        
        Args:
            n_bootstrap: Number of bootstrap resamples
            metrics_for_ci: Which metrics to compute CI for
            
        Returns:
            self
        """
        n_bootstrap = n_bootstrap or CONFIG.BOOTSTRAP_CI_RESAMPLES
        self.n_bootstrap = n_bootstrap
        
        if metrics_for_ci is None:
            metrics_for_ci = ['f1', 'auroc', 'auprc', 'precision', 'recall', 'accuracy', 'mcc']
        
        # Compute point estimates
        self.metrics = MetricsComputer.compute_all_metrics(
            self.y_true, self.y_pred, self.y_proba
        )
        
        # Compute bootstrap CI
        print(f"Computing bootstrap CI ({n_bootstrap} resamples)...")
        self.ci_results = MetricsComputer.compute_all_ci(
            self.y_true, self.y_proba, self.y_pred,
            metrics=metrics_for_ci,
            n_resamples=n_bootstrap
        )
        
        self.generated_at = datetime.now().isoformat()
        
        return self
    
    def print_report(self):
        """Print formatted evaluation report to console"""
        if self.metrics is None:
            raise ValueError("Report not generated. Call generate() first.")
        
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT: {self.model_name}")
        print(f"{'='*60}")
        print(f"Generated: {self.generated_at}")
        print(f"Test samples: {len(self.y_true)}")
        print(f"Positive class: {self.y_true.sum()} ({self.y_true.mean():.1%})")
        
        print(f"\n{'-'*60}")
        print("Point Estimates:")
        print(f"{'-'*60}")
        
        primary_metrics = ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'auprc', 'mcc']
        for metric in primary_metrics:
            if metric in self.metrics:
                value = self.metrics[metric]
                print(f"  {metric:15s}: {value:.4f}")
        
        print(f"\n{'-'*60}")
        print(f"Bootstrap {int(CONFIG.CONFIDENCE_LEVEL*100)}% Confidence Intervals:")
        print(f"{'-'*60}")
        
        for metric, ci in self.ci_results.items():
            formatted = MetricsComputer.format_metric_with_ci(
                self.metrics.get(metric, ci['mean']), ci
            )
            print(f"  {metric:15s}: {formatted}")
        
        print(f"\n{'-'*60}")
        print("Confusion Matrix:")
        print(f"{'-'*60}")
        print(f"  TN: {self.metrics['tn']:4d}  |  FP: {self.metrics['fp']:4d}")
        print(f"  FN: {self.metrics['fn']:4d}  |  TP: {self.metrics['tp']:4d}")
        
        print(f"\n{'-'*60}")
        print("Clinical Metrics:")
        print(f"{'-'*60}")
        print(f"  Sensitivity: {self.metrics['sensitivity']:.4f}")
        print(f"  Specificity: {self.metrics['specificity']:.4f}")
        if 'brier_score' in self.metrics:
            print(f"  Brier Score: {self.metrics['brier_score']:.4f}")
        
        print(f"\n{'='*60}\n")
    
    def to_dict(self) -> Dict:
        """Convert report to dictionary"""
        return {
            'model_name': self.model_name,
            'generated_at': self.generated_at,
            'n_samples': len(self.y_true),
            'n_positive': int(self.y_true.sum()),
            'n_bootstrap': self.n_bootstrap,
            'metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v 
                       for k, v in self.metrics.items()},
            'bootstrap_ci': self.ci_results,
            'predictions': {
                'y_true': self.y_true.tolist(),
                'y_pred': self.y_pred.tolist(),
                'y_proba': self.y_proba.tolist()
            }
        }
    
    def save_json(self, path: Path = None):
        """Save report as JSON file"""
        if self.metrics is None:
            raise ValueError("Report not generated. Call generate() first.")
        
        if path is None:
            safe_name = self.model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
            path = CONFIG.RESULTS_DIR / f"{safe_name}_results.json"
        
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        report_dict = self.to_dict()
        # Remove predictions to reduce file size
        report_dict_lite = {k: v for k, v in report_dict.items() if k != 'predictions'}
        
        with open(path, 'w') as f:
            json.dump(report_dict_lite, f, indent=2)
        
        print(f"Report saved to {path}")
    
    def save_plots(self, output_dir: Path = None):
        """Save ROC/PR curves and calibration plot"""
        if self.metrics is None:
            raise ValueError("Report not generated. Call generate() first.")
        
        output_dir = Path(output_dir or CONFIG.RESULTS_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        safe_name = self.model_name.lower().replace(' ', '_').replace('(', '').replace(')', '')
        
        # ROC/PR curves
        MetricsComputer.plot_roc_pr_curves(
            self.y_true, self.y_proba,
            save_path=output_dir / f"{safe_name}_roc_pr.png",
            model_name=self.model_name
        )
        
        # Calibration curve
        MetricsComputer.plot_calibration_curve(
            self.y_true, self.y_proba,
            save_path=output_dir / f"{safe_name}_calibration.png",
            model_name=self.model_name
        )
    
    @staticmethod
    def load_json(path: Path) -> Dict:
        """Load report from JSON file"""
        with open(path, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def compare_reports(report_paths: List[Path]) -> str:
        """
        Generate comparison table from multiple report JSON files.
        
        Args:
            report_paths: List of paths to JSON report files
            
        Returns:
            Formatted comparison table string
        """
        reports = []
        for path in report_paths:
            with open(path, 'r') as f:
                reports.append(json.load(f))
        
        # Header
        lines = []
        lines.append("=" * 90)
        lines.append("MODEL COMPARISON")
        lines.append("=" * 90)
        lines.append("")
        
        # Table header
        header = f"{'Model':<30} {'F1':>12} {'AUROC':>12} {'Precision':>10} {'Recall':>10}"
        lines.append(header)
        lines.append("-" * 90)
        
        # Sort by F1
        reports.sort(key=lambda r: r['metrics']['f1'], reverse=True)
        
        for report in reports:
            name = report['model_name'][:28]
            metrics = report['metrics']
            ci = report.get('bootstrap_ci', {})
            
            # Format F1 with CI if available
            if 'f1' in ci:
                f1_str = f"{metrics['f1']:.3f} [{ci['f1']['lower_ci']:.3f},{ci['f1']['upper_ci']:.3f}]"
            else:
                f1_str = f"{metrics['f1']:.3f}"
            
            # Format AUROC with CI if available
            if 'auroc' in ci:
                auroc_str = f"{metrics['auroc']:.3f} [{ci['auroc']['lower_ci']:.3f},{ci['auroc']['upper_ci']:.3f}]"
            else:
                auroc_str = f"{metrics.get('auroc', 0):.3f}"
            
            line = f"{name:<30} {f1_str:>12} {auroc_str:>12} {metrics['precision']:>10.3f} {metrics['recall']:>10.3f}"
            lines.append(line)
        
        lines.append("-" * 90)
        lines.append("")
        
        return "\n".join(lines)
    
    def to_latex(self) -> str:
        """Generate LaTeX table row for the report"""
        if self.metrics is None:
            raise ValueError("Report not generated. Call generate() first.")
        
        def format_with_ci(metric_name):
            value = self.metrics.get(metric_name, 0)
            if metric_name in self.ci_results:
                ci = self.ci_results[metric_name]
                return f"${value:.3f}$ $[{ci['lower_ci']:.3f}, {ci['upper_ci']:.3f}]$"
            return f"${value:.3f}$"
        
        row = f"{self.model_name} & "
        row += f"{format_with_ci('f1')} & "
        row += f"{format_with_ci('auroc')} & "
        row += f"{format_with_ci('precision')} & "
        row += f"{format_with_ci('recall')} \\\\"
        
        return row


class ModelComparison:
    """Compare multiple models and generate comparison reports"""
    
    def __init__(self):
        self.reports = {}
    
    def add_report(self, report: EvaluationReport):
        """Add a report to comparison"""
        self.reports[report.model_name] = report
    
    def add_from_json(self, path: Path, model_name: str = None):
        """Add report from JSON file"""
        data = EvaluationReport.load_json(path)
        if model_name is None:
            model_name = data['model_name']
        self.reports[model_name] = data
    
    def print_comparison(self):
        """Print comparison table"""
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        
        # Table header
        print(f"\n{'Model':<25} {'F1':>10} {'AUROC':>10} {'Prec':>8} {'Recall':>8} {'MCC':>8}")
        print("-" * 80)
        
        # Sort by F1
        sorted_reports = sorted(
            self.reports.items(),
            key=lambda x: x[1].metrics['f1'] if isinstance(x[1], EvaluationReport) else x[1]['metrics']['f1'],
            reverse=True
        )
        
        for name, report in sorted_reports:
            if isinstance(report, EvaluationReport):
                m = report.metrics
            else:
                m = report['metrics']
            
            print(f"{name[:24]:<25} {m['f1']:>10.4f} {m.get('auroc', 0):>10.4f} "
                  f"{m['precision']:>8.4f} {m['recall']:>8.4f} {m.get('mcc', 0):>8.4f}")
        
        print("-" * 80)
        
        # Best model
        best_name, best_report = sorted_reports[0]
        if isinstance(best_report, EvaluationReport):
            best_f1 = best_report.metrics['f1']
        else:
            best_f1 = best_report['metrics']['f1']
        
        print(f"\n*** Best model: {best_name} (F1: {best_f1:.4f})")
        print("=" * 80 + "\n")
    
    def save_comparison_csv(self, path: Path = None):
        """Save comparison as CSV"""
        import pandas as pd
        
        rows = []
        for name, report in self.reports.items():
            if isinstance(report, EvaluationReport):
                m = report.metrics
                ci = report.ci_results
            else:
                m = report['metrics']
                ci = report.get('bootstrap_ci', {})
            
            row = {
                'model': name,
                'f1': m['f1'],
                'f1_ci_lower': ci.get('f1', {}).get('lower_ci', np.nan),
                'f1_ci_upper': ci.get('f1', {}).get('upper_ci', np.nan),
                'auroc': m.get('auroc', np.nan),
                'auroc_ci_lower': ci.get('auroc', {}).get('lower_ci', np.nan),
                'auroc_ci_upper': ci.get('auroc', {}).get('upper_ci', np.nan),
                'precision': m['precision'],
                'recall': m['recall'],
                'mcc': m.get('mcc', np.nan),
            }
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('f1', ascending=False)
        
        path = path or CONFIG.RESULTS_DIR / 'model_comparison.csv'
        df.to_csv(path, index=False)
        print(f"Comparison saved to {path}")
        
        return df

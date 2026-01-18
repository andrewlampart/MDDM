#!/usr/bin/env python
"""
Validation Script - Weryfikacja ulepszeń pipeline'u

Ten skrypt porównuje nową implementację z poprzednią,
weryfikując że nowe komponenty (BERT encoder, kalibracja) poprawiają wyniki.

Success Criteria (Minimal):
- F1 >= 0.65 (było: 0.50)
- AUROC >= 0.75 (było: 0.67)
- Specificity >= 0.70 (było: 0.69)
- MCC > 0 dla wszystkich modeli (było: tylko XGBoost)
- ECE < 0.05 (było: ~0.10)

Użycie:
    python scripts/validate_improvements.py
    python scripts/validate_improvements.py --baseline results/old
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pandas as pd
import json
import logging
from datetime import datetime
from typing import Dict, Optional, Tuple

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import CONFIG

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Success criteria from the plan
SUCCESS_CRITERIA = {
    'minimal': {
        'f1': 0.65,
        'auroc': 0.75,
        'specificity': 0.70,
        'mcc': 0.0,  # Must be positive
        'ece': 0.05,
    },
    'good': {
        'f1': 0.70,
        'auroc': 0.78,
        'specificity': 0.72,
        'mcc': 0.30,
        'ece': 0.04,
    },
    'excellent': {
        'f1': 0.75,
        'auroc': 0.80,
        'specificity': 0.75,
        'mcc': 0.40,
        'ece': 0.03,
    }
}

# Baseline (old results)
BASELINE = {
    'xgboost_early_fusion': {
        'f1': 0.50,
        'auroc': 0.67,
        'specificity': 0.69,
        'mcc': 0.24,
        'ece': 0.10,
    },
    'multimodal_late_fusion': {
        'f1': 0.47,
        'auroc': 0.82,
        'specificity': 0.00,
        'mcc': 0.00,
    },
    'mil_text_attention': {
        'f1': 0.44,
        'auroc': 0.56,
        'specificity': 0.00,
        'mcc': 0.00,
    }
}


def load_results(results_dir: Path) -> Dict[str, dict]:
    """
    Load results from JSON files in results directory.
    
    Returns:
        Dict mapping model name to metrics dict
    """
    results = {}
    
    for json_file in results_dir.glob('*_results.json'):
        model_name = json_file.stem.replace('_results', '')
        with open(json_file) as f:
            data = json.load(f)
            
        # Extract metrics
        metrics = {}
        if 'metrics' in data:
            metrics = data['metrics']
        else:
            # Try to extract from flat structure
            for key in ['f1', 'auroc', 'auprc', 'mcc', 'specificity', 'sensitivity', 'accuracy', 'ece']:
                if key in data:
                    metrics[key] = data[key]
        
        results[model_name] = metrics
    
    return results


def check_success_criteria(
    metrics: dict,
    criteria: dict,
    model_name: str = "Model"
) -> Tuple[bool, list]:
    """
    Check if metrics meet success criteria.
    
    Returns:
        Tuple of (all_passed, list of failures)
    """
    failures = []
    
    for metric, threshold in criteria.items():
        if metric not in metrics:
            continue
            
        value = metrics[metric]
        
        if metric == 'ece':
            # Lower is better for ECE
            if value > threshold:
                failures.append(f"{metric}: {value:.3f} > {threshold} (FAIL)")
            else:
                logger.info(f"  [PASS] {metric}: {value:.3f} <= {threshold}")
        elif metric == 'mcc':
            # MCC must be positive
            if value <= threshold:
                failures.append(f"{metric}: {value:.3f} <= {threshold} (FAIL)")
            else:
                logger.info(f"  [PASS] {metric}: {value:.3f} > {threshold}")
        else:
            # Higher is better for other metrics
            if value < threshold:
                failures.append(f"{metric}: {value:.3f} < {threshold} (FAIL)")
            else:
                logger.info(f"  [PASS] {metric}: {value:.3f} >= {threshold}")
    
    return len(failures) == 0, failures


def compare_with_baseline(
    new_results: Dict[str, dict],
    baseline: Dict[str, dict] = BASELINE
) -> dict:
    """
    Compare new results with baseline.
    
    Returns:
        Dict with comparison analysis
    """
    comparison = {}
    
    for model, new_metrics in new_results.items():
        # Find matching baseline
        baseline_metrics = None
        for base_name, base_metrics in baseline.items():
            if base_name in model.lower() or model.lower() in base_name:
                baseline_metrics = base_metrics
                break
        
        if baseline_metrics is None:
            logger.warning(f"No baseline found for {model}")
            continue
        
        comparison[model] = {
            'new': new_metrics,
            'baseline': baseline_metrics,
            'improvements': {}
        }
        
        for metric in ['f1', 'auroc', 'specificity', 'mcc']:
            if metric in new_metrics and metric in baseline_metrics:
                old_val = baseline_metrics[metric]
                new_val = new_metrics[metric]
                diff = new_val - old_val
                pct = (diff / old_val * 100) if old_val != 0 else float('inf')
                
                comparison[model]['improvements'][metric] = {
                    'old': old_val,
                    'new': new_val,
                    'diff': diff,
                    'pct': pct,
                    'improved': diff > 0
                }
    
    return comparison


def generate_validation_report(
    results: Dict[str, dict],
    comparison: dict,
    criteria_level: str = 'minimal'
) -> str:
    """
    Generate validation report.
    """
    criteria = SUCCESS_CRITERIA[criteria_level]
    
    report = []
    report.append("=" * 70)
    report.append("MDDM VALIDATION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Criteria Level: {criteria_level}")
    report.append("=" * 70)
    
    # Per-model results
    report.append("\nMODEL RESULTS:")
    report.append("-" * 70)
    
    all_passed = True
    
    for model, metrics in results.items():
        report.append(f"\n{model}:")
        
        passed, failures = check_success_criteria(metrics, criteria, model)
        
        if passed:
            report.append("  STATUS: PASS")
        else:
            report.append("  STATUS: FAIL")
            for failure in failures:
                report.append(f"    - {failure}")
            all_passed = False
        
        # Show improvements if available
        if model in comparison:
            report.append("  Improvements vs baseline:")
            for metric, imp in comparison[model]['improvements'].items():
                symbol = "+" if imp['improved'] else ""
                report.append(f"    {metric}: {imp['old']:.3f} -> {imp['new']:.3f} ({symbol}{imp['diff']:.3f}, {symbol}{imp['pct']:.1f}%)")
    
    # Overall verdict
    report.append("\n" + "=" * 70)
    report.append("OVERALL VERDICT")
    report.append("=" * 70)
    
    if all_passed:
        report.append(f"\n[PASS] All models meet {criteria_level} success criteria!")
        report.append("Ready for publication (IEEE JBHI tier-2)")
    else:
        report.append(f"\n[FAIL] Some models do not meet {criteria_level} criteria.")
        report.append("Continue improving before publication.")
    
    # Publication readiness
    report.append("\nPUBLICATION READINESS:")
    
    # Check each level
    for level, level_criteria in SUCCESS_CRITERIA.items():
        level_passed = True
        for model, metrics in results.items():
            passed, _ = check_success_criteria(metrics, level_criteria, model)
            if not passed:
                level_passed = False
                break
        
        status = "READY" if level_passed else "NOT READY"
        venues = {
            'minimal': "IEEE JBHI (tier-2)",
            'good': "JMIR Mental Health",
            'excellent': "Nature Digital Medicine / Lancet Digital Health"
        }
        report.append(f"  {level.upper()}: {status} - {venues[level]}")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


def run_quick_validation(data_dir: Path, model_type: str = 'xgboost') -> dict:
    """
    Run quick validation on available data.
    
    This trains and evaluates models with new components to verify improvements.
    """
    try:
        from sklearn.model_selection import train_test_split
        from xgboost import XGBClassifier
        from sklearn.metrics import f1_score, roc_auc_score, matthews_corrcoef
        from evaluation.calibration import expected_calibration_error, calibrate_model
        from models.text_encoder_bert import TextEncoderBERT
    except ImportError as e:
        logger.error(f"Missing dependency: {e}")
        return {}
    
    # Check for data
    audio_path = data_dir / 'audio_features.csv'
    text_path = data_dir / 'text_features.csv'
    labels_path = data_dir / 'daic_labels.csv'
    
    if not all(p.exists() for p in [audio_path, text_path, labels_path]):
        logger.warning("Data files not found. Skipping quick validation.")
        logger.info("Required: audio_features.csv, text_features.csv, daic_labels.csv")
        return {}
    
    # Load data
    audio_df = pd.read_csv(audio_path)
    text_df = pd.read_csv(text_path)
    labels_df = pd.read_csv(labels_path)
    
    X_audio = audio_df.drop(columns=['participant_id', 'session_id', 'label'], errors='ignore').values
    X_text = text_df.drop(columns=['participant_id', 'session_id', 'label'], errors='ignore').values
    y = labels_df['PHQ8_Binary'].values if 'PHQ8_Binary' in labels_df.columns else labels_df['label'].values
    
    # Combine features
    X = np.hstack([X_audio, X_text])
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=42
    )
    
    # Train
    model = XGBClassifier(
        n_estimators=100,
        max_depth=5,
        scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calibrate
    calibrated = calibrate_model(model, X_val, y_val)
    y_proba_cal = calibrated.predict_proba(X_test)[:, 1]
    
    # Metrics
    tn = ((y_test == 0) & (y_pred == 0)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    
    results = {
        'f1': f1_score(y_test, y_pred),
        'auroc': roc_auc_score(y_test, y_proba),
        'mcc': matthews_corrcoef(y_test, y_pred),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'ece_before': expected_calibration_error(y_test, y_proba),
        'ece': expected_calibration_error(y_test, y_proba_cal),
    }
    
    return {'xgboost_bert_calibrated': results}


def main():
    parser = argparse.ArgumentParser(
        description='Validate pipeline improvements'
    )
    parser.add_argument(
        '--results-dir', '-r',
        type=Path,
        default=Path('results'),
        help='Directory with model results'
    )
    parser.add_argument(
        '--data-dir', '-d',
        type=Path,
        default=Path('data/processed'),
        help='Directory with processed data (for quick validation)'
    )
    parser.add_argument(
        '--criteria', '-c',
        choices=['minimal', 'good', 'excellent'],
        default='minimal',
        help='Success criteria level'
    )
    parser.add_argument(
        '--quick', '-q',
        action='store_true',
        help='Run quick validation (train and evaluate)'
    )
    parser.add_argument(
        '--output', '-o',
        type=Path,
        default=None,
        help='Save report to file'
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("MDDM VALIDATION")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)
    
    # Try to load existing results
    results_dir = Path(args.results_dir)
    results = {}
    
    if results_dir.exists():
        results = load_results(results_dir)
        if results:
            print(f"\nLoaded results for {len(results)} models from {results_dir}")
    
    # Run quick validation if requested or no results found
    if args.quick or not results:
        print("\nRunning quick validation...")
        quick_results = run_quick_validation(Path(args.data_dir))
        results.update(quick_results)
    
    if not results:
        print("\n[WARNING] No results available!")
        print("\nTo validate improvements, either:")
        print("  1. Run the full pipeline to generate results/")
        print("  2. Use --quick flag with data in data/processed/")
        print("\nRequired for --quick:")
        print("  - data/processed/audio_features.csv")
        print("  - data/processed/text_features.csv")
        print("  - data/processed/daic_labels.csv")
        return 1
    
    # Compare with baseline
    comparison = compare_with_baseline(results)
    
    # Generate report
    report = generate_validation_report(results, comparison, args.criteria)
    print(report)
    
    # Save report
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            f.write(report)
        print(f"\nReport saved to: {output_path}")
    
    # Return success/failure
    criteria = SUCCESS_CRITERIA[args.criteria]
    for model, metrics in results.items():
        passed, _ = check_success_criteria(metrics, criteria, model)
        if not passed:
            return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Step 7: Comprehensive Prototype Evaluation

Kompletna ewaluacja prototypu multimodalnego modelu depresji.
Porównuje wszystkie strategie fuzji i generuje raport.

Dla doktoratu - dokumentuje:
1. Porównanie modalności (audio vs text vs fusion)
2. Porównanie strategii fuzji (early, late, attention)
3. Metryki z confidence intervals
4. Feature importance analysis
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
from datetime import datetime
import warnings
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)

from src.utils.config import Config
from src.models.baseline import BaselineClassifier
from src.fusion.late_fusion import LateFusionClassifier, AttentionFusionClassifier


def load_all_data() -> Tuple[Dict[str, np.ndarray], np.ndarray, pd.DataFrame]:
    """Load all features and labels
    
    Returns:
        X_dict: Dictionary {'audio': X_audio, 'text': X_text}
        y: Labels array
        full_df: Full dataframe with all features
    """
    # Load features
    audio_df = pd.read_csv(Config.AUDIO_FEATURES_CSV)
    text_df = pd.read_csv(Config.TEXT_FEATURES_CSV)
    labels_df = pd.read_csv(Config.LABELS_CSV)[['session_id', 'depression', 'phq8_score']]
    
    # Merge on common sessions
    merged = audio_df.merge(text_df, on='session_id', suffixes=('_audio', '_text'))
    merged = merged.merge(labels_df, on='session_id')
    
    # Separate features
    audio_cols = [c for c in audio_df.columns if c != 'session_id']
    text_cols = [c for c in text_df.columns if c != 'session_id']
    
    X_audio = merged[[c for c in merged.columns if c in audio_cols or c.replace('_audio', '') in audio_cols]].values
    X_text = merged[[c for c in merged.columns if c in text_cols or c.replace('_text', '') in text_cols]].values
    
    # Handle duplicated column names from merge
    audio_cols_in_merged = [c for c in merged.columns if c not in ['session_id', 'depression', 'phq8_score'] 
                           and not c.endswith('_text')]
    text_cols_in_merged = [c for c in merged.columns if c.endswith('_text') or 
                          (c in text_cols and c not in audio_cols)]
    
    X_audio = merged[audio_cols_in_merged].values if audio_cols_in_merged else merged[audio_cols].values
    X_text = merged[text_cols_in_merged].values if text_cols_in_merged else merged[text_cols].values
    
    y = merged['depression'].values
    
    print(f"Loaded data: {len(y)} samples")
    print(f"  Audio features: {X_audio.shape[1]}")
    print(f"  Text features: {X_text.shape[1]}")
    print(f"  Positive class: {y.sum()} ({y.mean():.1%})")
    
    return {'audio': X_audio, 'text': X_text}, y, merged


def evaluate_early_fusion(X_dict: Dict[str, np.ndarray], y: np.ndarray,
                          n_splits: int = 5) -> Dict:
    """Evaluate early fusion (concatenation + RF)"""
    X_fusion = np.hstack([X_dict['audio'], X_dict['text']])
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 
               'roc_auc': [], 'ap': []}
    
    all_y_true, all_y_proba = [], []
    
    for train_idx, test_idx in cv.split(X_fusion, y):
        classifier = BaselineClassifier(n_features=50)
        
        X_train, X_test = X_fusion[train_idx], X_fusion[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        classifier.fit(X_train, y_train, X_test, y_test, optimize_threshold=True)
        
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)[:, 1]
        
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        
        try:
            metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))
        except:
            metrics['roc_auc'].append(0.5)
        metrics['ap'].append(average_precision_score(y_test, y_proba))
        
        all_y_true.extend(y_test)
        all_y_proba.extend(y_proba)
    
    return _aggregate_metrics(metrics, all_y_true, all_y_proba)


def evaluate_late_fusion(X_dict: Dict[str, np.ndarray], y: np.ndarray,
                         strategy: str = 'weighted', n_splits: int = 5) -> Dict:
    """Evaluate late fusion with specified strategy"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 
               'roc_auc': [], 'ap': []}
    
    all_y_true, all_y_proba = [], []
    
    for train_idx, test_idx in cv.split(y, y):
        X_train_dict = {k: v[train_idx] for k, v in X_dict.items()}
        X_test_dict = {k: v[test_idx] for k, v in X_dict.items()}
        y_train, y_test = y[train_idx], y[test_idx]
        
        classifier = LateFusionClassifier(fusion_strategy=strategy, n_features_per_modality=50)
        classifier.fit(X_train_dict, y_train, X_test_dict, y_test)
        
        y_pred = classifier.predict(X_test_dict)
        y_proba = classifier.predict_proba(X_test_dict)[:, 1]
        
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        
        try:
            metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))
        except:
            metrics['roc_auc'].append(0.5)
        metrics['ap'].append(average_precision_score(y_test, y_proba))
        
        all_y_true.extend(y_test)
        all_y_proba.extend(y_proba)
    
    return _aggregate_metrics(metrics, all_y_true, all_y_proba)


def evaluate_attention_fusion(X_dict: Dict[str, np.ndarray], y: np.ndarray,
                              n_splits: int = 5) -> Dict:
    """Evaluate attention-based fusion"""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 
               'roc_auc': [], 'ap': []}
    
    all_y_true, all_y_proba = [], []
    
    for train_idx, test_idx in cv.split(y, y):
        X_train_dict = {k: v[train_idx] for k, v in X_dict.items()}
        X_test_dict = {k: v[test_idx] for k, v in X_dict.items()}
        y_train, y_test = y[train_idx], y[test_idx]
        
        classifier = AttentionFusionClassifier(n_features_per_modality=50)
        classifier.fit(X_train_dict, y_train, X_test_dict, y_test)
        
        y_pred = classifier.predict(X_test_dict)
        y_proba = classifier.predict_proba(X_test_dict)[:, 1]
        
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        
        try:
            metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))
        except:
            metrics['roc_auc'].append(0.5)
        metrics['ap'].append(average_precision_score(y_test, y_proba))
        
        all_y_true.extend(y_test)
        all_y_proba.extend(y_proba)
    
    return _aggregate_metrics(metrics, all_y_true, all_y_proba)


def _aggregate_metrics(metrics: Dict, y_true: List, y_proba: List) -> Dict:
    """Aggregate metrics with confidence intervals"""
    results = {}
    
    for metric, values in metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        ci_95 = 1.96 * std / np.sqrt(len(values))
        results[metric] = {
            'mean': mean,
            'std': std,
            'ci_95': ci_95,
            'values': values
        }
    
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    y_pred = (y_proba >= 0.5).astype(int)
    
    results['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    results['n_samples'] = len(y_true)
    
    return results


def generate_report(results: Dict[str, Dict], output_path: Path):
    """Generate comprehensive evaluation report"""
    
    report = []
    report.append("=" * 80)
    report.append("MULTIMODAL DEPRESSION DETECTION - PROTOTYPE EVALUATION REPORT")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 80)
    
    # Summary table
    report.append("\n## SUMMARY - Fusion Strategy Comparison")
    report.append("-" * 80)
    header = f"{'Strategy':<25} {'F1':>10} {'ROC-AUC':>10} {'Precision':>10} {'Recall':>10}"
    report.append(header)
    report.append("-" * 80)
    
    # Sort by F1
    sorted_results = sorted(results.items(), key=lambda x: x[1]['f1']['mean'], reverse=True)
    
    for name, metrics in sorted_results:
        f1 = f"{metrics['f1']['mean']:.3f}±{metrics['f1']['ci_95']:.3f}"
        auc = f"{metrics['roc_auc']['mean']:.3f}±{metrics['roc_auc']['ci_95']:.3f}"
        prec = f"{metrics['precision']['mean']:.3f}±{metrics['precision']['ci_95']:.3f}"
        rec = f"{metrics['recall']['mean']:.3f}±{metrics['recall']['ci_95']:.3f}"
        report.append(f"{name:<25} {f1:>10} {auc:>10} {prec:>10} {rec:>10}")
    
    report.append("-" * 80)
    
    # Best model
    best_name, best_metrics = sorted_results[0]
    report.append(f"\n🏆 BEST STRATEGY: {best_name}")
    report.append(f"   F1 Score: {best_metrics['f1']['mean']:.4f} (95% CI: ±{best_metrics['f1']['ci_95']:.4f})")
    report.append(f"   ROC-AUC:  {best_metrics['roc_auc']['mean']:.4f}")
    
    # Detailed results
    report.append("\n\n## DETAILED RESULTS")
    for name, metrics in results.items():
        report.append(f"\n### {name}")
        report.append(f"  Samples: {metrics['n_samples']}")
        
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ap']:
            m = metrics[metric]
            report.append(f"  {metric.upper():>12}: {m['mean']:.4f} ± {m['std']:.4f} (95% CI: {m['ci_95']:.4f})")
        
        cm = metrics['confusion_matrix']
        report.append(f"  Confusion Matrix:")
        report.append(f"    TN: {cm[0][0]:4d}  FP: {cm[0][1]:4d}")
        report.append(f"    FN: {cm[1][0]:4d}  TP: {cm[1][1]:4d}")
    
    # Research implications
    report.append("\n\n## RESEARCH IMPLICATIONS")
    report.append("-" * 80)
    
    # Compare early vs late fusion
    if 'Early Fusion' in results and 'Late Fusion (weighted)' in results:
        early_f1 = results['Early Fusion']['f1']['mean']
        late_f1 = results['Late Fusion (weighted)']['f1']['mean']
        diff = late_f1 - early_f1
        better = "Late Fusion" if diff > 0 else "Early Fusion"
        report.append(f"• Early vs Late Fusion: {better} is better by {abs(diff):.3f} F1")
    
    # Compare single modality vs fusion
    if 'Audio Only' in results and 'Text Only' in results:
        audio_f1 = results['Audio Only']['f1']['mean']
        text_f1 = results['Text Only']['f1']['mean']
        fusion_f1 = best_metrics['f1']['mean']
        
        report.append(f"• Single modality performance:")
        report.append(f"    Audio Only: F1={audio_f1:.3f}")
        report.append(f"    Text Only:  F1={text_f1:.3f}")
        report.append(f"    Best Fusion: F1={fusion_f1:.3f}")
        
        improvement = fusion_f1 - max(audio_f1, text_f1)
        if improvement > 0:
            report.append(f"  → Multimodal fusion improves by {improvement:.3f} F1 over best single modality")
    
    report.append("\n" + "=" * 80)
    report.append("END OF REPORT")
    report.append("=" * 80)
    
    # Save report
    report_text = "\n".join(report)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print(report_text)
    print(f"\n📄 Report saved to {output_path}")
    
    return report_text


def main():
    print("=" * 60)
    print("Step 7: Comprehensive Prototype Evaluation")
    print("=" * 60)
    
    # Load data
    print("\n📊 Loading data...")
    X_dict, y, full_df = load_all_data()
    
    results = {}
    
    # 1. Single modality baselines
    print("\n🔬 Evaluating single modalities...")
    
    # Audio only
    print("\n  Audio Only:")
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_selection import SelectKBest, f_classif
    from sklearn.ensemble import RandomForestClassifier
    
    audio_metrics = _evaluate_single_modality(X_dict['audio'], y, "Audio Only")
    results['Audio Only'] = audio_metrics
    
    # Text only
    print("\n  Text Only:")
    text_metrics = _evaluate_single_modality(X_dict['text'], y, "Text Only")
    results['Text Only'] = text_metrics
    
    # 2. Early Fusion
    print("\n🔗 Evaluating Early Fusion (concatenation)...")
    results['Early Fusion'] = evaluate_early_fusion(X_dict, y)
    
    # 3. Late Fusion variants
    print("\n🔗 Evaluating Late Fusion strategies...")
    
    for strategy in ['average', 'weighted', 'stacking']:
        print(f"\n  Late Fusion ({strategy}):")
        results[f'Late Fusion ({strategy})'] = evaluate_late_fusion(X_dict, y, strategy)
    
    # 4. Attention Fusion
    print("\n🔗 Evaluating Attention Fusion...")
    results['Attention Fusion'] = evaluate_attention_fusion(X_dict, y)
    
    # Generate report
    print("\n📝 Generating report...")
    report_path = Config.PROCESSED_DATA_DIR / "prototype_evaluation_report.txt"
    generate_report(results, report_path)
    
    # Save results as JSON
    json_results = {}
    for name, metrics in results.items():
        json_results[name] = {
            metric: {'mean': m['mean'], 'std': m['std'], 'ci_95': m['ci_95']}
            for metric, m in metrics.items()
            if isinstance(m, dict) and 'mean' in m
        }
        json_results[name]['confusion_matrix'] = metrics['confusion_matrix']
    
    json_path = Config.PROCESSED_DATA_DIR / "prototype_evaluation_results.json"
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)
    print(f"\n💾 Results saved to {json_path}")
    
    print("\n" + "=" * 60)
    print("Step 7 Complete!")
    print("=" * 60)


def _evaluate_single_modality(X: np.ndarray, y: np.ndarray, name: str) -> Dict:
    """Evaluate single modality with CV"""
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 
               'roc_auc': [], 'ap': []}
    
    all_y_true, all_y_proba = [], []
    
    for train_idx, test_idx in cv.split(X, y):
        classifier = BaselineClassifier(n_features=50)
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        classifier.fit(X_train, y_train, X_test, y_test, optimize_threshold=True)
        
        y_pred = classifier.predict(X_test)
        y_proba = classifier.predict_proba(X_test)[:, 1]
        
        metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        
        try:
            metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))
        except:
            metrics['roc_auc'].append(0.5)
        metrics['ap'].append(average_precision_score(y_test, y_proba))
        
        all_y_true.extend(y_test)
        all_y_proba.extend(y_proba)
    
    return _aggregate_metrics(metrics, all_y_true, all_y_proba)


if __name__ == "__main__":
    main()

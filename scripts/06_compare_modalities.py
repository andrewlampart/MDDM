"""Step 6: Compare modalities - Audio-only, Text-only, Fusion

Porównanie skuteczności różnych modalności i strategii fuzji.
Kluczowe dla publikacji i dokumentacji prototypu.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Tuple, List
import warnings

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt

from src.utils.config import Config


def load_features() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Załaduj cechy audio i tekstowe osobno"""
    
    # Audio features
    audio_df = pd.read_csv(Config.AUDIO_FEATURES_CSV)
    print(f"Audio features: {len(audio_df)} samples, {len(audio_df.columns)-1} features")
    
    # Text features
    text_df = pd.read_csv(Config.TEXT_FEATURES_CSV)
    print(f"Text features: {len(text_df)} samples, {len(text_df.columns)-1} features")
    
    # Labels
    labels_df = pd.read_csv(Config.LABELS_CSV)[['session_id', 'depression', 'phq8_score']]
    print(f"Labels: {len(labels_df)} samples")
    
    return audio_df, text_df, labels_df


def prepare_modality_data(
    audio_df: pd.DataFrame,
    text_df: pd.DataFrame,
    labels_df: pd.DataFrame
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Przygotuj dane dla każdej modalności
    
    Returns:
        Dict z kluczami: 'audio', 'text', 'fusion' -> (X, y)
    """
    # Merge z labels
    audio_merged = audio_df.merge(labels_df, on='session_id')
    text_merged = text_df.merge(labels_df, on='session_id')
    
    # Fusion - inner join aby mieć wszystkie modalności
    fusion_df = audio_df.merge(text_df, on='session_id', suffixes=('_audio', '_text'))
    fusion_df = fusion_df.merge(labels_df, on='session_id')
    
    # Audio only
    audio_cols = [c for c in audio_merged.columns if c not in ['session_id', 'depression', 'phq8_score']]
    X_audio = audio_merged[audio_cols].values
    y_audio = audio_merged['depression'].values
    
    # Text only
    text_cols = [c for c in text_merged.columns if c not in ['session_id', 'depression', 'phq8_score']]
    X_text = text_merged[text_cols].values
    y_text = text_merged['depression'].values
    
    # Fusion (early fusion - concatenation)
    fusion_cols = [c for c in fusion_df.columns if c not in ['session_id', 'depression', 'phq8_score']]
    X_fusion = fusion_df[fusion_cols].values
    y_fusion = fusion_df['depression'].values
    
    print(f"\nPrepared data:")
    print(f"  Audio: {X_audio.shape[0]} samples, {X_audio.shape[1]} features")
    print(f"  Text:  {X_text.shape[0]} samples, {X_text.shape[1]} features")
    print(f"  Fusion: {X_fusion.shape[0]} samples, {X_fusion.shape[1]} features")
    
    return {
        'audio': (X_audio, y_audio),
        'text': (X_text, y_text),
        'fusion': (X_fusion, y_fusion)
    }


def evaluate_modality(
    X: np.ndarray,
    y: np.ndarray,
    n_features: int = 50,
    n_splits: int = 5,
    random_state: int = 42
) -> Dict:
    """Ewaluacja modalności z cross-validation
    
    Returns:
        Dictionary z metrykami i confidence intervals
    """
    # Feature selection i scaling w pipeline
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Feature selection
    n_feat = min(n_features, X_scaled.shape[1])
    selector = SelectKBest(score_func=f_classif, k=n_feat)
    X_selected = selector.fit_transform(X_scaled, y)
    
    # Model
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_leaf=5,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # Collect metrics per fold
    fold_metrics = {
        'accuracy': [], 'precision': [], 'recall': [],
        'f1': [], 'roc_auc': [], 'ap': []
    }
    
    all_y_true = []
    all_y_pred = []
    all_y_proba = []
    
    for train_idx, test_idx in cv.split(X_selected, y):
        X_train, X_test = X_selected[train_idx], X_selected[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
        fold_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
        fold_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        fold_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
        
        try:
            fold_metrics['roc_auc'].append(roc_auc_score(y_test, y_proba))
        except:
            fold_metrics['roc_auc'].append(0.5)
        
        fold_metrics['ap'].append(average_precision_score(y_test, y_proba))
        
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)
        all_y_proba.extend(y_proba)
    
    # Aggregate metrics with 95% CI
    results = {}
    for metric, values in fold_metrics.items():
        mean = np.mean(values)
        std = np.std(values)
        ci_95 = 1.96 * std / np.sqrt(len(values))
        results[metric] = {
            'mean': mean,
            'std': std,
            'ci_95': ci_95,
            'values': values
        }
    
    # Overall confusion matrix
    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    all_y_proba = np.array(all_y_proba)
    
    results['confusion_matrix'] = confusion_matrix(all_y_true, all_y_pred)
    results['y_true'] = all_y_true
    results['y_proba'] = all_y_proba
    results['n_features_selected'] = n_feat
    results['n_samples'] = len(y)
    
    return results


def print_results(results: Dict[str, Dict], title: str = "Modality Comparison"):
    """Wyświetl wyniki porównania"""
    print("\n" + "=" * 70)
    print(f"{title}")
    print("=" * 70)
    
    # Header
    print(f"\n{'Modality':<12} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10} {'AP':>10}")
    print("-" * 70)
    
    for modality, metrics in results.items():
        acc = f"{metrics['accuracy']['mean']:.3f}±{metrics['accuracy']['ci_95']:.3f}"
        prec = f"{metrics['precision']['mean']:.3f}±{metrics['precision']['ci_95']:.3f}"
        rec = f"{metrics['recall']['mean']:.3f}±{metrics['recall']['ci_95']:.3f}"
        f1 = f"{metrics['f1']['mean']:.3f}±{metrics['f1']['ci_95']:.3f}"
        auc = f"{metrics['roc_auc']['mean']:.3f}±{metrics['roc_auc']['ci_95']:.3f}"
        ap = f"{metrics['ap']['mean']:.3f}±{metrics['ap']['ci_95']:.3f}"
        
        print(f"{modality:<12} {acc:>10} {prec:>10} {rec:>10} {f1:>10} {auc:>10} {ap:>10}")
    
    print("-" * 70)
    
    # Detailed view
    print("\n📊 Detailed Results:")
    for modality, metrics in results.items():
        print(f"\n  {modality.upper()}:")
        print(f"    Samples: {metrics['n_samples']}, Features selected: {metrics['n_features_selected']}")
        print(f"    Confusion Matrix:")
        cm = metrics['confusion_matrix']
        print(f"      TN: {cm[0,0]:3d}  FP: {cm[0,1]:3d}")
        print(f"      FN: {cm[1,0]:3d}  TP: {cm[1,1]:3d}")


def plot_comparison(results: Dict[str, Dict], output_path: Path = None):
    """Wizualizacja porównania modalności"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    modalities = list(results.keys())
    colors = {'audio': '#2196F3', 'text': '#4CAF50', 'fusion': '#FF9800'}
    
    # 1. Bar plot - główne metryki
    ax1 = axes[0, 0]
    metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    x = np.arange(len(metrics))
    width = 0.25
    
    for i, modality in enumerate(modalities):
        values = [results[modality][m]['mean'] for m in metrics]
        errors = [results[modality][m]['ci_95'] for m in metrics]
        ax1.bar(x + i * width, values, width, yerr=errors, label=modality.capitalize(),
                color=colors.get(modality, '#999'), capsize=3)
    
    ax1.set_ylabel('Score')
    ax1.set_title('Modality Comparison - Key Metrics (5-fold CV)')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'])
    ax1.legend()
    ax1.set_ylim(0, 1)
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. ROC Curves
    ax2 = axes[0, 1]
    for modality in modalities:
        y_true = results[modality]['y_true']
        y_proba = results[modality]['y_proba']
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = results[modality]['roc_auc']['mean']
        ax2.plot(fpr, tpr, color=colors.get(modality, '#999'),
                label=f"{modality.capitalize()} (AUC={auc:.3f})")
    
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.legend()
    ax2.grid(alpha=0.3)
    
    # 3. Precision-Recall Curves
    ax3 = axes[1, 0]
    for modality in modalities:
        y_true = results[modality]['y_true']
        y_proba = results[modality]['y_proba']
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = results[modality]['ap']['mean']
        ax3.plot(recall, precision, color=colors.get(modality, '#999'),
                label=f"{modality.capitalize()} (AP={ap:.3f})")
    
    # Baseline (proportion of positive class)
    baseline = np.mean(results['fusion']['y_true'])
    ax3.axhline(y=baseline, color='gray', linestyle='--', alpha=0.5, label=f'Baseline ({baseline:.2f})')
    
    ax3.set_xlabel('Recall')
    ax3.set_ylabel('Precision')
    ax3.set_title('Precision-Recall Curves')
    ax3.legend()
    ax3.grid(alpha=0.3)
    
    # 4. F1 Score Distribution (box plot)
    ax4 = axes[1, 1]
    f1_data = [results[m]['f1']['values'] for m in modalities]
    bp = ax4.boxplot(f1_data, labels=[m.capitalize() for m in modalities], patch_artist=True)
    
    for patch, modality in zip(bp['boxes'], modalities):
        patch.set_facecolor(colors.get(modality, '#999'))
        patch.set_alpha(0.7)
    
    ax4.set_ylabel('F1 Score')
    ax4.set_title('F1 Score Distribution (5-fold CV)')
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\n📈 Plot saved to {output_path}")
    
    plt.show()


def main():
    print("=" * 60)
    print("Step 6: Modality Comparison")
    print("=" * 60)
    
    # Load data
    print("\nLoading features...")
    audio_df, text_df, labels_df = load_features()
    
    # Prepare data for each modality
    print("\nPreparing modality data...")
    data = prepare_modality_data(audio_df, text_df, labels_df)
    
    # Evaluate each modality
    print("\n🔄 Evaluating modalities with 5-fold stratified CV...")
    results = {}
    
    for modality, (X, y) in data.items():
        print(f"\n  Evaluating {modality}...")
        results[modality] = evaluate_modality(X, y, n_features=50)
    
    # Print results
    print_results(results)
    
    # Plot comparison
    output_path = Config.PROCESSED_DATA_DIR / "modality_comparison.png"
    try:
        plot_comparison(results, output_path)
    except Exception as e:
        print(f"\nWarning: Could not create plot: {e}")
    
    # Save results to CSV
    results_df = []
    for modality, metrics in results.items():
        row = {
            'modality': modality,
            'n_samples': metrics['n_samples'],
            'n_features': metrics['n_features_selected'],
        }
        for metric in ['accuracy', 'precision', 'recall', 'f1', 'roc_auc', 'ap']:
            row[f'{metric}_mean'] = metrics[metric]['mean']
            row[f'{metric}_std'] = metrics[metric]['std']
            row[f'{metric}_ci95'] = metrics[metric]['ci_95']
        results_df.append(row)
    
    results_df = pd.DataFrame(results_df)
    results_csv = Config.PROCESSED_DATA_DIR / "modality_comparison_results.csv"
    results_df.to_csv(results_csv, index=False)
    print(f"\n📄 Results saved to {results_csv}")
    
    print("\n" + "=" * 60)
    print("Step 6 Complete!")
    print("=" * 60)
    
    # Summary
    best_modality = max(results.keys(), key=lambda m: results[m]['f1']['mean'])
    print(f"\n🏆 Best modality by F1: {best_modality.upper()}")
    print(f"   F1: {results[best_modality]['f1']['mean']:.3f} ± {results[best_modality]['f1']['ci_95']:.3f}")
    print(f"   ROC-AUC: {results[best_modality]['roc_auc']['mean']:.3f}")


if __name__ == "__main__":
    main()

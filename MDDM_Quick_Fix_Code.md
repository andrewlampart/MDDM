# 🔧 QUICK FIX CODE - Calibration + Diagnostyka
## Ready-to-use kod do wdrożenia DZISIAJ

---

## CZĘŚĆ 1: Calibration dla Attention Fusion

```python
# calibration_fix.py
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn

class CalibratedAttentionFusion:
    """
    Wrapper dla Attention Fusion z post-hoc calibracją
    Rozwiązuje problem AUROC=0.5 i niskiej specificity
    """
    
    def __init__(self, attention_model):
        self.attention_model = attention_model
        self.calibrator = None
        self.threshold_optimal = 0.5
        
    def fit_calibration(self, X_val_audio, X_val_text, y_val):
        """
        Fit calibrator na validation secie
        WAŻNE: X_val musi być INNA niż treningowy zbiór
        """
        # Uzyskaj raw predictions z attention model
        with torch.no_grad():
            X_audio_tensor = torch.tensor(
                X_val_audio, dtype=torch.float32, device='cuda'
            )
            X_text_tensor = torch.tensor(
                X_val_text, dtype=torch.float32, device='cuda'
            )
            y_proba_raw = self.attention_model(X_audio_tensor, X_text_tensor)
            y_proba_raw = y_proba_raw.cpu().numpy().flatten()
        
        # Fit calibrator (sigmoid lub isotonic)
        self.calibrator = CalibratedClassifierCV(
            DummyClassifier(),  # dummy, bo już mamy predictions
            method='sigmoid',   # lub 'isotonic'
            cv='prefit'
        )
        
        # Fit on validation set
        self.calibrator.fit(
            np.column_stack([y_proba_raw]),  # fake X
            y_val
        )
        
        print("✓ Calibration fitted")
        return self
    
    def predict_proba_calibrated(self, X_audio, X_text):
        """
        Predykcja z kalibracją
        Zwraca (N, 2) array: [[p_negative, p_positive], ...]
        """
        # Raw predictions
        with torch.no_grad():
            X_audio_tensor = torch.tensor(
                X_audio, dtype=torch.float32, device='cuda'
            )
            X_text_tensor = torch.tensor(
                X_text, dtype=torch.float32, device='cuda'
            )
            y_proba_raw = self.attention_model(X_audio_tensor, X_text_tensor)
            y_proba_raw = y_proba_raw.cpu().numpy().flatten()
        
        # Calibrate
        if self.calibrator:
            # Sigmoid calibration
            y_proba_cal = self.calibrator.predict_proba(
                np.column_stack([y_proba_raw])
            )[:, 1]
        else:
            y_proba_cal = y_proba_raw
        
        # Return (N, 2) format
        return np.column_stack([1 - y_proba_cal, y_proba_cal])
    
    def predict_with_optimal_threshold(self, X_audio, X_text):
        """
        Predykcja z optymalnym thresholdem
        """
        y_proba = self.predict_proba_calibrated(X_audio, X_text)[:, 1]
        y_pred = (y_proba > self.threshold_optimal).astype(int)
        return y_pred, y_proba

class DummyClassifier:
    """Dummy classifier dla calibrator"""
    pass

# ============================================================================
# UŻYCIE: W pętli CV
# ============================================================================

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score, recall_score, confusion_matrix

def train_with_calibration(
    audio_features,
    text_features,
    labels,
    attention_model_class,
    n_splits=5
):
    """
    5-fold CV z kalibracją
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    results = {
        'f1_before': [],
        'f1_after': [],
        'auroc_before': [],
        'auroc_after': [],
        'recall_before': [],
        'recall_after': [],
        'specificity_before': [],
        'specificity_after': [],
    }
    
    fold = 1
    for train_idx, test_idx in skf.split(audio_features, labels):
        print(f"\n{'='*60}")
        print(f"FOLD {fold}/{n_splits}")
        print(f"{'='*60}")
        
        # Split data
        X_audio_train = audio_features[train_idx]
        X_text_train = text_features[train_idx]
        y_train = labels[train_idx]
        
        X_audio_test = audio_features[test_idx]
        X_text_test = text_features[test_idx]
        y_test = labels[test_idx]
        
        # Split train into train/val (80/20)
        n_val = int(0.2 * len(X_audio_train))
        val_idx = np.random.choice(len(X_audio_train), n_val, replace=False)
        train_idx_final = np.array([i for i in range(len(X_audio_train)) if i not in val_idx])
        
        X_audio_val = X_audio_train[val_idx]
        X_text_val = X_text_train[val_idx]
        y_val = y_train[val_idx]
        
        X_audio_train_final = X_audio_train[train_idx_final]
        X_text_train_final = X_text_train[train_idx_final]
        y_train_final = y_train[train_idx_final]
        
        # Train attention model
        model = attention_model_class()  # Twoja AttentionFusion
        train(model, X_audio_train_final, X_text_train_final, y_train_final)
        
        # Raw predictions (PRZED calibracją)
        y_proba_raw_test = model.predict_proba(X_audio_test, X_text_test)[:, 1]
        y_pred_raw_test = (y_proba_raw_test > 0.5).astype(int)
        
        f1_before = f1_score(y_test, y_pred_raw_test)
        auroc_before = roc_auc_score(y_test, y_proba_raw_test)
        recall_before = recall_score(y_test, y_pred_raw_test)
        cm_before = confusion_matrix(y_test, y_pred_raw_test)
        specificity_before = cm_before[0, 0] / (cm_before[0, 0] + cm_before[0, 1])
        
        print(f"PRZED Calibracją:")
        print(f"  F1:         {f1_before:.4f}")
        print(f"  AUROC:      {auroc_before:.4f}")
        print(f"  Recall:     {recall_before:.4f}")
        print(f"  Specificity:{specificity_before:.4f}")
        
        # Fit calibrator
        calibrated_model = CalibratedAttentionFusion(model)
        calibrated_model.fit_calibration(X_audio_val, X_text_val, y_val)
        
        # Calibrated predictions
        y_proba_cal_test = calibrated_model.predict_proba_calibrated(
            X_audio_test, X_text_test
        )[:, 1]
        y_pred_cal_test = (y_proba_cal_test > 0.5).astype(int)
        
        f1_after = f1_score(y_test, y_pred_cal_test)
        auroc_after = roc_auc_score(y_test, y_proba_cal_test)
        recall_after = recall_score(y_test, y_pred_cal_test)
        cm_after = confusion_matrix(y_test, y_pred_cal_test)
        specificity_after = cm_after[0, 0] / (cm_after[0, 0] + cm_after[0, 1])
        
        print(f"\nPO Calibracją:")
        print(f"  F1:         {f1_after:.4f} ({f1_after-f1_before:+.4f})")
        print(f"  AUROC:      {auroc_after:.4f} ({auroc_after-auroc_before:+.4f}) ★")
        print(f"  Recall:     {recall_after:.4f} ({recall_after-recall_before:+.4f})")
        print(f"  Specificity:{specificity_after:.4f} ({specificity_after-specificity_before:+.4f}) ★")
        
        # Collect results
        results['f1_before'].append(f1_before)
        results['f1_after'].append(f1_after)
        results['auroc_before'].append(auroc_before)
        results['auroc_after'].append(auroc_after)
        results['recall_before'].append(recall_before)
        results['recall_after'].append(recall_after)
        results['specificity_before'].append(specificity_before)
        results['specificity_after'].append(specificity_after)
        
        fold += 1
    
    # Summary
    print(f"\n{'='*60}")
    print("PODSUMOWANIE (5 foldów)")
    print(f"{'='*60}")
    print(f"\nF1:")
    print(f"  Przed:  {np.mean(results['f1_before']):.4f} ± {np.std(results['f1_before']):.4f}")
    print(f"  Po:     {np.mean(results['f1_after']):.4f} ± {np.std(results['f1_after']):.4f}")
    print(f"  Zmiana: {np.mean(results['f1_after']) - np.mean(results['f1_before']):+.4f}")
    
    print(f"\nAUROC:")
    print(f"  Przed:  {np.mean(results['auroc_before']):.4f} ± {np.std(results['auroc_before']):.4f}")
    print(f"  Po:     {np.mean(results['auroc_after']):.4f} ± {np.std(results['auroc_after']):.4f}")
    print(f"  Zmiana: {np.mean(results['auroc_after']) - np.mean(results['auroc_before']):+.4f} ★")
    
    print(f"\nRecall:")
    print(f"  Przed:  {np.mean(results['recall_before']):.4f} ± {np.std(results['recall_before']):.4f}")
    print(f"  Po:     {np.mean(results['recall_after']):.4f} ± {np.std(results['recall_after']):.4f}")
    
    print(f"\nSpecificity:")
    print(f"  Przed:  {np.mean(results['specificity_before']):.4f} ± {np.std(results['specificity_before']):.4f}")
    print(f"  Po:     {np.mean(results['specificity_after']):.4f} ± {np.std(results['specificity_after']):.4f}")
    print(f"  Zmiana: {np.mean(results['specificity_after']) - np.mean(results['specificity_before']):+.4f} ★")
    
    return results

# UŻYCIE:
if __name__ == "__main__":
    from fusion_model import AttentionFusion
    
    # Załaduj swoje dane
    # audio_features: (N, 838)
    # text_features: (N, 560)
    # labels: (N,)
    
    results = train_with_calibration(
        audio_features,
        text_features,
        labels,
        attention_model_class=AttentionFusion,
        n_splits=5
    )
```

---

## CZĘŚĆ 2: Diagnostyka Features

```python
# feature_diagnostics.py
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
import matplotlib.pyplot as plt

class FeatureDiagnostics:
    """
    Zdiagnozuj czy nowe wymiary (838 audio, 560 text) pomagają
    czy są szumem
    """
    
    def __init__(self, audio_features, text_features, labels, 
                 audio_features_old=None, text_features_old=None):
        self.audio = audio_features
        self.text = text_features
        self.labels = labels
        self.audio_old = audio_features_old
        self.text_old = text_features_old
    
    def check_nans_infs(self):
        """Szukaj NaN i Inf wartości"""
        print("=" * 60)
        print("NAN / INF CHECK")
        print("=" * 60)
        
        print(f"\nAudio features ({self.audio.shape[1]} dims):")
        print(f"  NaN count:  {np.isnan(self.audio).sum()}")
        print(f"  Inf count:  {np.isinf(self.audio).sum()}")
        print(f"  Min value:  {np.nanmin(self.audio):.6f}")
        print(f"  Max value:  {np.nanmax(self.audio):.6f}")
        print(f"  Mean value: {np.nanmean(self.audio):.6f}")
        print(f"  Std value:  {np.nanstd(self.audio):.6f}")
        
        print(f"\nText features ({self.text.shape[1]} dims):")
        print(f"  NaN count:  {np.isnan(self.text).sum()}")
        print(f"  Inf count:  {np.isinf(self.text).sum()}")
        print(f"  Min value:  {np.nanmin(self.text):.6f}")
        print(f"  Max value:  {np.nanmax(self.text):.6f}")
        print(f"  Mean value: {np.nanmean(self.text):.6f}")
        print(f"  Std value:  {np.nanstd(self.text):.6f}")
    
    def check_feature_correlation_with_label(self):
        """Którym features korelują z depresją?"""
        print("\n" + "=" * 60)
        print("FEATURE CORRELATION WITH DEPRESSION")
        print("=" * 60)
        
        # Audio features
        audio_corrs = []
        for i in range(self.audio.shape[1]):
            corr, pval = spearmanr(self.audio[:, i], self.labels)
            audio_corrs.append((i, abs(corr), pval))
        
        audio_corrs_sorted = sorted(audio_corrs, key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 Audio Features (Spearman correlation):")
        for i, (feat_idx, corr, pval) in enumerate(audio_corrs_sorted[:10]):
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {i+1}. Feature {feat_idx:3d}: r={corr:+.4f} (p={pval:.4f}) {sig}")
        
        # Text features
        text_corrs = []
        for i in range(self.text.shape[1]):
            corr, pval = spearmanr(self.text[:, i], self.labels)
            text_corrs.append((i, abs(corr), pval))
        
        text_corrs_sorted = sorted(text_corrs, key=lambda x: x[1], reverse=True)
        
        print(f"\nTop 10 Text Features (Spearman correlation):")
        for i, (feat_idx, corr, pval) in enumerate(text_corrs_sorted[:10]):
            sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else ""
            print(f"  {i+1}. Feature {feat_idx:3d}: r={corr:+.4f} (p={pval:.4f}) {sig}")
        
        return audio_corrs_sorted, text_corrs_sorted
    
    def compare_old_vs_new_features(self):
        """Porównaj stare (786/512) vs nowe (838/560) wymiary"""
        if self.audio_old is None or self.text_old is None:
            print("\n⚠️ Old features not provided - skipping comparison")
            return
        
        print("\n" + "=" * 60)
        print("OLD vs NEW FEATURES COMPARISON")
        print("=" * 60)
        
        # Audio
        print(f"\nAudio:")
        print(f"  Old shape: {self.audio_old.shape}")
        print(f"  New shape: {self.audio.shape}")
        print(f"  Added:     {self.audio.shape[1] - self.audio_old.shape[1]} features")
        
        # Czy overlap?
        audio_overlap = self.audio[:, :self.audio_old.shape[1]]
        audio_new_only = self.audio[:, self.audio_old.shape[1]:]
        
        # Correlation z labelem
        corr_old_audio = [abs(spearmanr(audio_overlap[:, i], self.labels)[0]) 
                         for i in range(audio_overlap.shape[1])]
        corr_new_audio = [abs(spearmanr(audio_new_only[:, i], self.labels)[0]) 
                         for i in range(audio_new_only.shape[1])]
        
        print(f"  Avg correlation (old audio features):  {np.mean(corr_old_audio):.4f}")
        print(f"  Avg correlation (new audio features):  {np.mean(corr_new_audio):.4f}")
        print(f"  Verdict: {'✓ NEW FEATURES HELP' if np.mean(corr_new_audio) > np.mean(corr_old_audio) * 0.8 else '✗ NEW FEATURES HURT'}")
        
        # Text
        print(f"\nText:")
        print(f"  Old shape: {self.text_old.shape}")
        print(f"  New shape: {self.text.shape}")
        print(f"  Added:     {self.text.shape[1] - self.text_old.shape[1]} features")
        
        text_overlap = self.text[:, :self.text_old.shape[1]]
        text_new_only = self.text[:, self.text_old.shape[1]:]
        
        corr_old_text = [abs(spearmanr(text_overlap[:, i], self.labels)[0]) 
                        for i in range(text_overlap.shape[1])]
        corr_new_text = [abs(spearmanr(text_new_only[:, i], self.labels)[0]) 
                        for i in range(text_new_only.shape[1])]
        
        print(f"  Avg correlation (old text features):   {np.mean(corr_old_text):.4f}")
        print(f"  Avg correlation (new text features):   {np.mean(corr_new_text):.4f}")
        print(f"  Verdict: {'✓ NEW FEATURES HELP' if np.mean(corr_new_text) > np.mean(corr_old_text) * 0.8 else '✗ NEW FEATURES HURT'}")
    
    def print_full_report(self):
        """Kompletu raport"""
        self.check_nans_infs()
        audio_corrs, text_corrs = self.check_feature_correlation_with_label()
        self.compare_old_vs_new_features()

# UŻYCIE:
if __name__ == "__main__":
    diag = FeatureDiagnostics(
        audio_features,      # (189, 838)
        text_features,       # (189, 560)
        labels,              # (189,)
        # Opcjonalnie:
        # audio_features_old,  # (189, 786)
        # text_features_old    # (189, 512)
    )
    
    diag.print_full_report()
```

---

## CZĘŚĆ 3: Threshold Optimization

```python
# threshold_optimization.py
from sklearn.metrics import f1_score, precision_recall_curve, auc
import numpy as np

def find_optimal_threshold(y_true, y_proba, metric='f1'):
    """
    Znajdź optymalny threshold dla Twojego modelu
    """
    thresholds = np.arange(0.0, 1.01, 0.01)
    best_metric = -np.inf
    best_threshold = 0.5
    scores = {}
    
    for threshold in thresholds:
        y_pred = (y_proba > threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == 'balanced':
            # Balance between sensitivity and specificity
            cm = confusion_matrix(y_true, y_pred)
            sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            score = (sensitivity + specificity) / 2
        elif metric == 'youden':
            # Youden's J statistic
            cm = confusion_matrix(y_true, y_pred)
            sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0]) if (cm[1, 1] + cm[1, 0]) > 0 else 0
            specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1]) if (cm[0, 0] + cm[0, 1]) > 0 else 0
            score = sensitivity + specificity - 1
        
        scores[threshold] = score
        if score > best_metric:
            best_metric = score
            best_threshold = threshold
    
    return best_threshold, best_metric, scores

# UŻYCIE:
if __name__ == "__main__":
    from sklearn.metrics import confusion_matrix
    
    # Na validation secie
    best_threshold, best_score, scores = find_optimal_threshold(
        y_val, 
        y_proba_val, 
        metric='f1'  # lub 'balanced', 'youden'
    )
    
    print(f"Optimal threshold: {best_threshold:.3f}")
    print(f"Best score: {best_score:.4f}")
    
    # Plot
    plt.figure(figsize=(10, 6))
    thresholds = list(scores.keys())
    metric_values = list(scores.values())
    plt.plot(thresholds, metric_values, 'b-', linewidth=2)
    plt.axvline(best_threshold, color='r', linestyle='--', label=f'Optimal: {best_threshold:.3f}')
    plt.xlabel('Threshold')
    plt.ylabel('F1 Score')
    plt.title('Threshold Optimization')
    plt.legend()
    plt.grid(True)
    plt.savefig('threshold_optimization.png', dpi=150, bbox_inches='tight')
    print("✓ Plot saved: threshold_optimization.png")
```

---

## CZĘŚĆ 4: Quick Integration

```python
# main_quick_fix.py - Integracja wszystkiego

import numpy as np
from calibration_fix import CalibratedAttentionFusion, train_with_calibration
from feature_diagnostics import FeatureDiagnostics
from threshold_optimization import find_optimal_threshold

# Step 1: Diagnostyka
print("STEP 1: Feature Diagnostics")
print("=" * 60)
diag = FeatureDiagnostics(audio_features, text_features, labels)
diag.print_full_report()

# Step 2: Trenuj z calibracją
print("\nSTEP 2: Train with Calibration")
print("=" * 60)
results = train_with_calibration(
    audio_features,
    text_features,
    labels,
    attention_model_class=AttentionFusion,
    n_splits=5
)

# Step 3: Czy poprawiło się?
print("\nSTEP 3: Did it improve?")
print("=" * 60)
improvement_auroc = (
    np.mean(results['auroc_after']) - 
    np.mean(results['auroc_before'])
)
improvement_f1 = (
    np.mean(results['f1_after']) - 
    np.mean(results['f1_before'])
)

print(f"\nAUROC Improvement: {improvement_auroc:+.4f}")
print(f"F1 Improvement:   {improvement_f1:+.4f}")

if improvement_auroc > 0.05:
    print("✓ CALIBRATION HELPED!")
else:
    print("✗ Calibration nie pomogła - problem głębszy")
```

---

**To wdróż dzisiaj!**

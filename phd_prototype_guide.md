# Od Baseline'u do University-Grade PhD Prototypu
## Pełny poradnik: 3-Sprint Plan na wysokiej jakości model multimodalny

---

## 🎯 Cel: Transformacja preprocessingu w doktorski framework

**Status quo:**
- ✅ Preprocessing danych (audio + tekst)
- ✅ Feature engineering
- ❌ Brak zaawansowanego modelowania
- ❌ Brak Multi-Instance Learning
- ❌ Brak systematycznej ewaluacji
- ❌ Brak interpretowalności i niepewności

**Docelowy stan (koniec Sprint 3):**
- ✅ MIL-owy moduł tekstowy (jak Nature)
- ✅ CNN + LSTM moduł audio
- ✅ Multimodal late fusion
- ✅ Pełna metodologia ewaluacji (F1, AUROC, kalibracja, CI)
- ✅ Interpretowalność (LIME + saliency)
- ✅ Niepewność (MC Dropout)
- ✅ Publikowalny kod i reprodukowalność

---

# SPRINT 1: Baseline'ów i Fundamentu (1-2 tygodnie)

## 1.1 Setup Eksperymentalny

Stwórz strukturę folderów i config:

```python
# config.py
import json
from pathlib import Path
from dataclasses import dataclass

@dataclass
class Config:
    # Ścieżki
    DATA_DIR = Path('data')
    RESULTS_DIR = Path('results')
    MODELS_DIR = Path('models')
    LOGS_DIR = Path('logs')
    
    # Dataset
    RANDOM_SEED = 42
    TRAIN_SIZE = 107
    VAL_SIZE = 35
    TEST_SIZE = 47
    
    # Audio
    SR = 16000
    N_MELS = 128
    N_MFCC = 13
    
    # Training baseline
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 1e-3
    EARLY_STOPPING_PATIENCE = 15
    
    # Class weights (nierównowaga klas)
    COMPUTE_CLASS_WEIGHTS = True
    
    # Evaluation
    METRICS = ['accuracy', 'precision', 'recall', 'f1', 'auroc', 'auprc']
    BOOTSTRAP_CI_RESAMPLES = 1000
    
    # Paths creation
    def __post_init__(self):
        for path in [self.DATA_DIR, self.RESULTS_DIR, self.MODELS_DIR, self.LOGS_DIR]:
            path.mkdir(parents=True, exist_ok=True)

CONFIG = Config()

# W każdym skrypcie na początku
import random
import numpy as np
import torch
from config import CONFIG

random.seed(CONFIG.RANDOM_SEED)
np.random.seed(CONFIG.RANDOM_SEED)
torch.manual_seed(CONFIG.RANDOM_SEED)
torch.cuda.manual_seed_all(CONFIG.RANDOM_SEED)
```

## 1.2 Baseline Model 1: XGBoost na fuzjowanych cechach

```python
# baselines/xgboost_baseline.py
import numpy as np
import pandas as pd
from sklearn.ensemble import XGBClassifier
from sklearn.metrics import f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.utils.class_weight import compute_sample_weight
import pickle
from config import CONFIG

class XGBoostBaseline:
    def __init__(self):
        self.model = None
        self.best_threshold = 0.5
        self.scaler = None
    
    def compute_class_weights(self, y_train):
        """Wagi klas dla nierównowagi"""
        weights = compute_sample_weight('balanced', y_train)
        return weights
    
    def train(self, X_train, y_train, X_val, y_val):
        """Trenuj XGBoost z early stopping"""
        weights = self.compute_class_weights(y_train)
        
        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=CONFIG.RANDOM_SEED,
            n_jobs=4,
            tree_method='gpu_hist',  # GPU acceleration
            device='cuda'
        )
        
        # Fit z early stopping na validation set
        self.model.fit(
            X_train, y_train,
            sample_weight=weights,
            eval_set=[(X_val, y_val)],
            eval_metric='logloss',
            early_stopping_rounds=20,
            verbose=50
        )
        
        print(f"Best iteration: {self.model.best_iteration}")
    
    def optimize_threshold(self, X_val, y_val):
        """Znajduj optymalny threshold na F1 (nie domyślne 0.5)"""
        y_proba = self.model.predict_proba(X_val)[:, 1]
        
        best_f1 = 0
        best_threshold = 0.5
        
        for threshold in np.arange(0.3, 0.8, 0.01):
            y_pred = (y_proba >= threshold).astype(int)
            f1 = f1_score(y_val, y_pred)
            
            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold
        
        self.best_threshold = best_threshold
        print(f"Optimal threshold: {best_threshold:.3f} (F1: {best_f1:.3f})")
        return best_threshold
    
    def predict(self, X, use_optimal_threshold=True):
        """Predykcja z optymalnym thresholdem"""
        y_proba = self.model.predict_proba(X)[:, 1]
        
        if use_optimal_threshold:
            return (y_proba >= self.best_threshold).astype(int), y_proba
        else:
            return (y_proba >= 0.5).astype(int), y_proba
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
        print(f"Model saved to {path}")
    
    def load(self, path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        print(f"Model loaded from {path}")

# Main training script
if __name__ == '__main__':
    # Load data
    X_train = np.load(CONFIG.DATA_DIR / 'X_train.npy')
    X_val = np.load(CONFIG.DATA_DIR / 'X_val.npy')
    X_test = np.load(CONFIG.DATA_DIR / 'X_test.npy')
    y_train = np.load(CONFIG.DATA_DIR / 'y_train.npy')
    y_val = np.load(CONFIG.DATA_DIR / 'y_val.npy')
    y_test = np.load(CONFIG.DATA_DIR / 'y_test.npy')
    
    # Train
    baseline = XGBoostBaseline()
    baseline.train(X_train, y_train, X_val, y_val)
    baseline.optimize_threshold(X_val, y_val)
    
    # Evaluate
    y_pred_test, y_proba_test = baseline.predict(X_test)
    
    from sklearn.metrics import classification_report
    print("\n=== Test Set Results ===")
    print(classification_report(y_test, y_pred_test))
    
    baseline.save(CONFIG.MODELS_DIR / 'xgboost_baseline.pkl')
```

## 1.3 Comprehensive Evaluation Framework

```python
# evaluation/metrics.py
import numpy as np
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score, recall_score,
    roc_auc_score, roc_curve, auc, precision_recall_curve,
    confusion_matrix, matthews_corrcoef
)
from scipy import stats
import matplotlib.pyplot as plt

class MetricsComputer:
    @staticmethod
    def compute_all_metrics(y_true, y_pred, y_proba):
        """Policz wszystkie key metrics"""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
            'auroc': roc_auc_score(y_true, y_proba),
            'auprc': auc(*precision_recall_curve(y_true, y_proba)[:2]),
            'mcc': matthews_corrcoef(y_true, y_pred),
            'specificity': confusion_matrix(y_true, y_pred)[0, 0] / (confusion_matrix(y_true, y_pred)[0, 0] + confusion_matrix(y_true, y_pred)[0, 1] + 1e-8),
            'sensitivity': recall_score(y_true, y_pred, zero_division=0),
        }
    
    @staticmethod
    def bootstrap_ci(y_true, y_proba, y_pred, metric_name='f1', n_resamples=1000, ci=0.95):
        """Bootstrap confidence intervals (crucial dla small test sets)"""
        n_samples = len(y_true)
        metric_scores = []
        
        for _ in range(n_resamples):
            # Resample z replacement
            idx = np.random.choice(n_samples, size=n_samples, replace=True)
            y_true_resample = y_true[idx]
            y_pred_resample = y_pred[idx]
            y_proba_resample = y_proba[idx]
            
            if metric_name == 'f1':
                score = f1_score(y_true_resample, y_pred_resample, zero_division=0)
            elif metric_name == 'auroc':
                if len(np.unique(y_true_resample)) > 1:
                    score = roc_auc_score(y_true_resample, y_proba_resample)
                else:
                    score = np.nan
            elif metric_name == 'precision':
                score = precision_score(y_true_resample, y_pred_resample, zero_division=0)
            elif metric_name == 'recall':
                score = recall_score(y_true_resample, y_pred_resample, zero_division=0)
            elif metric_name == 'accuracy':
                score = accuracy_score(y_true_resample, y_pred_resample)
            
            if not np.isnan(score):
                metric_scores.append(score)
        
        metric_scores = np.array(metric_scores)
        lower = np.percentile(metric_scores, (1 - ci) / 2 * 100)
        upper = np.percentile(metric_scores, (1 + ci) / 2 * 100)
        mean = np.mean(metric_scores)
        
        return {
            'mean': mean,
            'lower_ci': lower,
            'upper_ci': upper,
            'std': np.std(metric_scores)
        }
    
    @staticmethod
    def plot_roc_pr_curves(y_true, y_proba, save_path=None):
        """Rysuj ROC + PR curves"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # ROC
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        ax1.plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
        ax1.plot([0, 1], [0, 1], 'k--', label='Random')
        ax1.set_xlabel('FPR')
        ax1.set_ylabel('TPR')
        ax1.legend()
        ax1.set_title('ROC Curve')
        ax1.grid(alpha=0.3)
        
        # PR
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)
        ax2.plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
        ax2.axhline(y=np.mean(y_true), color='k', linestyle='--', label='Baseline')
        ax2.set_xlabel('Recall')
        ax2.set_ylabel('Precision')
        ax2.legend()
        ax2.set_title('Precision-Recall Curve')
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Saved to {save_path}")
        
        return fig

# evaluation/report.py
class EvaluationReport:
    def __init__(self, model_name, y_true, y_pred, y_proba):
        self.model_name = model_name
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.metrics = None
        self.ci_results = {}
    
    def generate(self, n_bootstrap=1000):
        """Pełny report z metrykam i CI"""
        # Podstawowe metryki
        self.metrics = MetricsComputer.compute_all_metrics(
            self.y_true, self.y_pred, self.y_proba
        )
        
        # Bootstrap CI
        for metric in ['f1', 'auroc', 'precision', 'recall', 'accuracy']:
            self.ci_results[metric] = MetricsComputer.bootstrap_ci(
                self.y_true, self.y_proba, self.y_pred,
                metric_name=metric,
                n_resamples=n_bootstrap
            )
        
        return self
    
    def print_report(self):
        """Drukuj report w ładnym formacie"""
        print(f"\n{'='*60}")
        print(f"EVALUATION REPORT: {self.model_name}")
        print(f"{'='*60}\n")
        
        print("Point Estimates:")
        for metric, value in self.metrics.items():
            print(f"  {metric:15s}: {value:.4f}")
        
        print("\nBootstrap 95% Confidence Intervals:")
        for metric, ci in self.ci_results.items():
            print(f"  {metric:15s}: {ci['mean']:.4f} [{ci['lower_ci']:.4f}, {ci['upper_ci']:.4f}]")
    
    def save_json(self, path):
        """Zapisz report jako JSON do later comparison"""
        import json
        report_dict = {
            'model_name': self.model_name,
            'metrics': self.metrics,
            'bootstrap_ci': self.ci_results
        }
        with open(path, 'w') as f:
            json.dump(report_dict, f, indent=2)
        print(f"Report saved to {path}")
```

## 1.4 Testy Baseline'ów w Eksperymencie

```python
# experiments/run_baselines.py
import numpy as np
from baselines.xgboost_baseline import XGBoostBaseline
from evaluation.report import EvaluationReport, MetricsComputer
from config import CONFIG
import json

def run_baseline_experiment():
    """Uruchom XGBoost baseline + full evaluation"""
    
    # Load
    X_train = np.load(CONFIG.DATA_DIR / 'X_train.npy')
    X_val = np.load(CONFIG.DATA_DIR / 'X_val.npy')
    X_test = np.load(CONFIG.DATA_DIR / 'X_test.npy')
    y_train = np.load(CONFIG.DATA_DIR / 'y_train.npy')
    y_val = np.load(CONFIG.DATA_DIR / 'y_val.npy')
    y_test = np.load(CONFIG.DATA_DIR / 'y_test.npy')
    
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Class distribution - Train: {np.mean(y_train):.1%}, Test: {np.mean(y_test):.1%}")
    
    # Train baseline
    print("\n=== Training XGBoost Baseline ===")
    baseline = XGBoostBaseline()
    baseline.train(X_train, y_train, X_val, y_val)
    baseline.optimize_threshold(X_val, y_val)
    
    # Test evaluation
    y_pred_test, y_proba_test = baseline.predict(X_test)
    
    # Generate report
    report = EvaluationReport(
        'XGBoost (Early Fusion)',
        y_test, y_pred_test, y_proba_test
    )
    report.generate(n_bootstrap=1000)
    report.print_report()
    report.save_json(CONFIG.RESULTS_DIR / 'xgboost_baseline_results.json')
    
    # Curves
    MetricsComputer.plot_roc_pr_curves(
        y_test, y_proba_test,
        save_path=CONFIG.RESULTS_DIR / 'xgboost_roc_pr.png'
    )
    
    # Save model
    baseline.save(CONFIG.MODELS_DIR / 'xgboost_baseline.pkl')
    
    return baseline, report

if __name__ == '__main__':
    run_baseline_experiment()
```

## 1.5 Co powinna osiągnąć Sprint 1

Po tej fazie masz:

✅ XGBoost baseline z full ewaluacją (F1, AUROC, PR-AUC + 95% CI)  
✅ Kod struktury eksperymentu  
✅ Dokumentacja metryk i confidence intervals  
✅ Baseline do porównania z nowszymi modelami  

**Oczekiwane wyniki na DAIC-WOZ:** F1 ≈ 0.75–0.80 (to jest rozsądne dla audio+tekst Early Fusion)

---

# SPRINT 2: MIL Moduł Tekstowy (2-3 tygodnie)

## 2.1 Segmentacja Wywiadu na Instancje

```python
# data/interview_segmentation.py
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import numpy as np

class InterviewSegmenter:
    """Dzielimy wywiad na instancje (odpowiedzi pacjenta)"""
    
    @staticmethod
    def segment_by_qa_pairs(trans_xml_path):
        """
        Ekstrakcja QA-par: każda odpowiedź pacjenta = instancja
        Zwracamy listę (timestamp_start, timestamp_end, speaker, text, sequence_num)
        """
        tree = ET.parse(trans_xml_path)
        root = tree.getroot()
        
        instances = []
        sequence_num = 0
        
        for turn in root.findall('.//Turn'):
            speaker = turn.get('spk')  # 'Participant' lub 'Ellie'
            start_time = float(turn.get('start', 0))
            end_time = float(turn.get('end', 0))
            text_elem = turn.find('Text')
            text = text_elem.text if text_elem is not None and text_elem.text else ''
            
            # Bierz tylko odpowiedzi pacjenta
            if speaker == 'Participant' and text.strip():
                instances.append({
                    'sequence_num': sequence_num,
                    'start_time': start_time,
                    'end_time': end_time,
                    'text': text.strip(),
                    'speaker': speaker
                })
                sequence_num += 1
        
        return instances
    
    @staticmethod
    def segment_by_sentences(text, max_length=512):
        """
        Alternative: segmentacja na zdania, każde < max_length tokens
        (dla bezpieczeństwa przy transformerach)
        """
        import re
        
        # Split na zdania (bardzo proste heurystyka)
        sentences = re.split(r'[.!?]+', text)
        instances = []
        
        buffer = ""
        seq_num = 0
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Jeśli dodanie zdania przekroczy limit, zapisz buffer i start nowy
            if len((buffer + " " + sentence).split()) > max_length:
                if buffer:
                    instances.append({
                        'sequence_num': seq_num,
                        'text': buffer.strip(),
                        'length_tokens': len(buffer.split())
                    })
                    seq_num += 1
                buffer = sentence
            else:
                buffer += " " + sentence if buffer else sentence
        
        # Ostatni buffer
        if buffer:
            instances.append({
                'sequence_num': seq_num,
                'text': buffer.strip(),
                'length_tokens': len(buffer.split())
            })
        
        return instances

# Przetwórz wszystkie wywiady na instancje
def create_instance_dataset(transcription_root, output_csv):
    """
    Przetwórz wszystkie transkrypcje na dataset instancji:
    session_id | sequence_num | text | depression (bag-level label)
    """
    segmenter = InterviewSegmenter()
    
    instances_list = []
    
    # Potrzebujesz też labelów z poprzedniego preprocessingu
    labels_df = pd.read_csv('daic_labels.csv')
    labels_dict = dict(zip(labels_df['session_id'], labels_df['depression']))
    
    for trans_file in Path(transcription_root).glob('Participant_*.xml'):
        # Extract session_id z filename
        session_id = int(trans_file.stem.split('_')[1])
        
        if session_id not in labels_dict:
            continue
        
        try:
            # Segment
            instances = segmenter.segment_by_qa_pairs(trans_file)
            
            # Add bag-level label
            for inst in instances:
                inst['session_id'] = session_id
                inst['depression'] = labels_dict[session_id]  # Bag label
            
            instances_list.extend(instances)
        except Exception as e:
            print(f"Error processing {session_id}: {e}")
    
    df_instances = pd.DataFrame(instances_list)
    df_instances.to_csv(output_csv, index=False)
    
    print(f"Created {len(df_instances)} instances from {len(df_instances['session_id'].unique())} sessions")
    return df_instances

# Użycie
# df_instances = create_instance_dataset('DAIC-WOZ/Transcriptions', 'data/instances.csv')
```

## 2.2 MIL Model Architecture w PyTorchu

```python
# models/mil_text_model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np

class MILTextModel(nn.Module):
    """
    Multi-Instance Learning dla depresji z MT5 + RoBERTa fusion
    
    Schemat:
    1. Każda instancja (odpowiedź) → MT5 + RoBERTa embeddings
    2. Instance-level classifier (MLP) → score depresji dla instancji
    3. MIL pooling z α, β progami → bag-level predykcja
    4. BCE loss na bag-level labelach
    """
    
    def __init__(self, 
                 mt5_model_name='google/mt5-small',
                 roberta_model_name='roberta-base',
                 hidden_dim=256,
                 dropout_rate=0.3,
                 alpha=0.6,  # próg procentu instancji
                 beta=0.5):   # próg agregowanego score
        super().__init__()
        
        # Feature extractors
        self.mt5_tokenizer = AutoTokenizer.from_pretrained(mt5_model_name)
        self.mt5_model = AutoModel.from_pretrained(mt5_model_name)
        
        self.roberta_tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
        self.roberta_model = AutoModel.from_pretrained(roberta_model_name)
        
        mt5_dim = self.mt5_model.config.hidden_size
        roberta_dim = self.roberta_model.config.hidden_size
        
        # Fusion: concat MT5 + RoBERTa
        fused_dim = mt5_dim + roberta_dim
        
        # Instance-level classifier (MLP)
        self.instance_mlp = nn.Sequential(
            nn.Linear(fused_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),  # score depression per instance
            nn.Sigmoid()
        )
        
        # Bag-level classifier
        self.bag_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
        # MIL hyperparameters
        self.alpha = alpha
        self.beta = beta
        
        # Freeze transformers (optional, można fine-tune)
        for param in self.mt5_model.parameters():
            param.requires_grad = False
        for param in self.roberta_model.parameters():
            param.requires_grad = False
    
    def _extract_features(self, texts, max_length=512):
        """
        Ekstrakcja MT5 + RoBERTa embeddings dla listy tekstów
        texts: list of strings (instancje z jednego wywiadu)
        """
        # MT5
        mt5_inputs = self.mt5_tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt'
        ).to(self.mt5_model.device)
        
        with torch.no_grad():
            mt5_outputs = self.mt5_model(**mt5_inputs)
            mt5_embeddings = mt5_outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
        # RoBERTa
        roberta_inputs = self.roberta_tokenizer(
            texts, padding=True, truncation=True,
            max_length=max_length, return_tensors='pt'
        ).to(self.roberta_model.device)
        
        with torch.no_grad():
            roberta_outputs = self.roberta_model(**roberta_inputs)
            roberta_embeddings = roberta_outputs.last_hidden_state[:, 0, :]
        
        # Fusion
        fused = torch.cat([mt5_embeddings, roberta_embeddings], dim=1)
        return fused
    
    def forward(self, bag_instances, return_instance_scores=False):
        """
        Forward pass dla całego wywiadu (bag)
        
        bag_instances: list of strings (wszystkie odpowiedzi pacjenta w porządku)
        return_instance_scores: jeśli True, zwrócić scores dla każdej instancji (dla LIME)
        """
        # Extract features dla wszystkich instancji
        instance_features = self._extract_features(bag_instances)  # [n_instances, fused_dim]
        
        # Instance-level scores
        instance_scores = self.instance_mlp(instance_features)  # [n_instances, 1]
        instance_scores = instance_scores.squeeze(1)  # [n_instances]
        
        # MIL pooling z α, β logika
        n_instances = len(bag_instances)
        
        # Warunek 1: ile instancji powyżej α?
        n_positive_instances = (instance_scores > self.alpha).sum().item()
        proportion_positive = n_positive_instances / n_instances
        
        # Warunek 2: średni score > β?
        mean_score = instance_scores.mean()
        
        # Bag-level decision: oba warunki muszą być spełnione
        # (możesz też robić weighted combination)
        bag_feature = torch.cat([
            instance_scores.mean().unsqueeze(0),  # average
            instance_scores.max().unsqueeze(0),   # max
            torch.tensor([proportion_positive], device=instance_scores.device)  # proportion
        ])  # [3]
        
        # Bag-level prediction (dla uproszczenia używamy max)
        bag_pred = instance_scores.max().unsqueeze(0)  # [1]
        
        if return_instance_scores:
            return bag_pred, instance_scores
        else:
            return bag_pred
    
    def compute_mil_loss(self, bag_predictions, bag_labels, instance_scores=None):
        """
        MIL loss: BCE na bag-level predictions
        bag_predictions: model output [batch_size, 1]
        bag_labels: ground truth [batch_size]
        """
        bag_labels = bag_labels.float().unsqueeze(1)  # [batch_size, 1]
        
        # Standardowa BCE
        loss = F.binary_cross_entropy(bag_predictions, bag_labels)
        
        # Opcjonalnie: dodaj regularizację na instance scores
        # (jeśli wiesz, które instancje są ważne ze względów klinicznych)
        
        return loss

# Training loop
class MILTextTrainer:
    def __init__(self, model, optimizer, device='cuda'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
    
    def train_epoch(self, bag_loader, epoch):
        """Trenuj na danych MIL"""
        self.model.train()
        total_loss = 0
        
        for batch_idx, batch in enumerate(bag_loader):
            # batch = {
            #     'bag_instances': list of lists (instancje per bag),
            #     'bag_labels': tensor [batch_size]
            # }
            
            self.optimizer.zero_grad()
            
            batch_predictions = []
            batch_instance_scores_all = []
            
            # Procesy każdy bag osobno (bo mają różne liczby instancji)
            for bag_instances, bag_label in zip(batch['bag_instances'], batch['bag_labels']):
                bag_pred, instance_scores = self.model(
                    bag_instances,
                    return_instance_scores=True
                )
                batch_predictions.append(bag_pred)
                batch_instance_scores_all.append(instance_scores)
            
            # Stack predictions
            bag_predictions = torch.cat(batch_predictions, dim=0)  # [batch_size, 1]
            bag_labels = batch['bag_labels'].to(self.device)
            
            # Compute loss
            loss = self.model.compute_mil_loss(bag_predictions, bag_labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch {epoch}, Batch {batch_idx}: Loss {loss.item():.4f}")
        
        avg_loss = total_loss / len(bag_loader)
        return avg_loss
```

## 2.3 Dataset Loader dla MIL

```python
# data/mil_dataloader.py
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class MILBagDataset(Dataset):
    """
    Dataset zwracający całe 'bags' (wywiady)
    Każdy sample = (list instancji, label wywiadu)
    """
    
    def __init__(self, instances_csv, split='train'):
        df = pd.read_csv(instances_csv)
        
        # Podziel na train/val/test po session_id
        unique_sessions = df['session_id'].unique()
        np.random.seed(42)
        np.random.shuffle(unique_sessions)
        
        n_train = int(0.7 * len(unique_sessions))
        n_val = int(0.15 * len(unique_sessions))
        
        train_sessions = unique_sessions[:n_train]
        val_sessions = unique_sessions[n_train:n_train + n_val]
        test_sessions = unique_sessions[n_train + n_val:]
        
        if split == 'train':
            self.df = df[df['session_id'].isin(train_sessions)]
        elif split == 'val':
            self.df = df[df['session_id'].isin(val_sessions)]
        else:  # test
            self.df = df[df['session_id'].isin(test_sessions)]
        
        # Group by session
        self.bags = []
        for session_id, group in self.df.groupby('session_id'):
            instances = group['text'].tolist()
            label = group['depression'].iloc[0]
            self.bags.append((instances, label, session_id))
    
    def __len__(self):
        return len(self.bags)
    
    def __getitem__(self, idx):
        instances, label, session_id = self.bags[idx]
        return {
            'bag_instances': instances,
            'label': torch.tensor(label, dtype=torch.float),
            'session_id': session_id
        }

def create_mil_bag_loaders(instances_csv, batch_size=8):
    train_dataset = MILBagDataset(instances_csv, split='train')
    val_dataset = MILBagDataset(instances_csv, split='val')
    test_dataset = MILBagDataset(instances_csv, split='test')
    
    # Custom collate (bo instancje mają różne długości)
    def collate_fn(batch):
        return {
            'bag_instances': [item['bag_instances'] for item in batch],
            'bag_labels': torch.stack([item['label'] for item in batch]),
            'session_ids': [item['session_id'] for item in batch]
        }
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

import torch
```

## 2.4 Training Script dla MIL

```python
# experiments/train_mil_text.py
import torch
from torch.optim import AdamW
from data.mil_dataloader import create_mil_bag_loaders
from models.mil_text_model import MILTextModel, MILTextTrainer
from evaluation.report import EvaluationReport
from config import CONFIG

def train_mil_model():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create loaders
    train_loader, val_loader, test_loader = create_mil_bag_loaders(
        'data/instances.csv',
        batch_size=8
    )
    
    # Model
    model = MILTextModel(
        mt5_model_name='google/mt5-small',  # lub inne
        roberta_model_name='roberta-base',
        alpha=0.6,
        beta=0.5
    ).to(device)
    
    # Optimizer
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-4)
    
    # Trainer
    trainer = MILTextTrainer(model, optimizer, device=device)
    
    # Training loop
    best_val_f1 = 0
    patience = 15
    patience_counter = 0
    
    for epoch in range(100):
        train_loss = trainer.train_epoch(train_loader, epoch)
        
        # Validation
        val_preds_all = []
        val_labels_all = []
        
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                for instances, label in zip(batch['bag_instances'], batch['bag_labels']):
                    pred = model(instances)
                    val_preds_all.append(pred.item())
                    val_labels_all.append(label.item())
        
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels_all, np.array(val_preds_all) > 0.5)
        
        print(f"Epoch {epoch}: Train Loss {train_loss:.4f}, Val F1 {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            patience_counter = 0
            torch.save(model.state_dict(), CONFIG.MODELS_DIR / 'mil_text_best.pt')
        else:
            patience_counter += 1
        
        if patience_counter >= patience:
            break
    
    # Test evaluation
    print("\n=== Test Evaluation ===")
    model.load_state_dict(torch.load(CONFIG.MODELS_DIR / 'mil_text_best.pt'))
    
    test_preds = []
    test_labels = []
    test_instance_scores = []  # dla LIME
    
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            for instances, label, session_id in zip(
                batch['bag_instances'],
                batch['bag_labels'],
                batch['session_ids']
            ):
                pred, inst_scores = model(instances, return_instance_scores=True)
                test_preds.append(pred.item())
                test_labels.append(label.item())
                test_instance_scores.append(inst_scores.cpu().numpy())
    
    test_preds = np.array(test_preds)
    test_labels = np.array(test_labels)
    test_pred_binary = (test_preds > 0.5).astype(int)
    
    # Report
    report = EvaluationReport(
        'MIL Text Model (MT5+RoBERTa)',
        test_labels, test_pred_binary, test_preds
    )
    report.generate()
    report.print_report()
    report.save_json(CONFIG.RESULTS_DIR / 'mil_text_results.json')
    
    return model, report, test_instance_scores

if __name__ == '__main__':
    train_mil_model()
```

## 2.5 LIME Interpretowalność

```python
# interpretability/lime_explanation.py
from lime.lime_text import LimeTextExplainer
import numpy as np

class InstanceLevelExplainer:
    """
    Wyjaśnianie na poziomie instancji (zdań/odpowiedzi) poprzez LIME
    """
    
    def __init__(self, mil_model):
        self.model = mil_model
        self.explainer = LimeTextExplainer(class_names=['Non-Depressed', 'Depressed'])
    
    def explain_instance(self, instance_text, top_features=10):
        """
        Wyjaśnij predykcję dla pojedynczej instancji
        Zwraca: które słowa były najważniejsze dla score depresji
        """
        
        def predict_fn(texts):
            """Wrapper dla LIME - bierze tekst, zwraca probabilities"""
            # Tutaj potrzebowałbyś sieci pojedynczej instancji,
            # albo używać instance_mlp bezpośrednio
            scores = []
            for text in texts:
                # Extract features
                features = self.model._extract_features([text])  # [1, fused_dim]
                score = self.model.instance_mlp(features)  # [1, 1]
                scores.append(score.item())
            
            # Convert to probabilities [non-depressed, depressed]
            scores = np.array(scores)
            return np.column_stack([1 - scores, scores])
        
        # Generate LIME explanation
        explanation = self.explainer.explain_instance(
            instance_text,
            predict_fn,
            num_features=top_features,
            num_samples=50
        )
        
        return explanation
    
    def plot_explanation(self, explanation, save_path=None):
        """Rysuj wyjaśnienie"""
        fig = explanation.show_in_notebook()
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig
```

## 2.6 Czego powinna osiągnąć Sprint 2

✅ Segmentacja wywiadu na instancje (QA-pary)  
✅ MIL model z MT5+RoBERTa  
✅ Training loop z Early Stopping  
✅ Full evaluation + CI  
✅ LIME interpretowalność  

**Oczekiwane wyniki:** F1 ≈ 0.82–0.88 (zgadza się z Nature paper, bo używasz ich metodologii)

---

# SPRINT 3: Multimodal Fusion + Finalizacja (2-3 tygodnie)

## 3.1 Audio CNN Model

```python
# models/audio_cnn_model.py
import torch
import torch.nn as nn

class AudioCNNModel(nn.Module):
    """
    CNN dla spektrogramów audio
    Input: mel-spectrogram (128, 400, 1)
    Output: depression probability [0, 1]
    """
    
    def __init__(self, dropout_rate=0.3):
        super().__init__()
        
        # Conv blocks z batch norm
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # LSTM na top konvolucji
        self.lstm = nn.LSTM(
            input_size=128 * 16 * 50,  # output size conv3
            hidden_size=256,
            num_layers=2,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, spectrogram):
        """
        spectrogram: [batch, 1, 128, 400]
        """
        # Conv
        x = self.conv1(spectrogram)  # [batch, 32, 64, 200]
        x = self.conv2(x)             # [batch, 64, 32, 100]
        x = self.conv3(x)             # [batch, 128, 16, 50]
        
        # Flatten
        batch_size = x.shape[0]
        x = x.view(batch_size, 1, -1)  # [batch, 1, 128*16*50]
        
        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(x)  # [batch, 1, 256]
        x = h_n[-1]  # Last hidden state [batch, 256]
        
        # Classify
        output = self.classifier(x)  # [batch, 1]
        
        return output.squeeze(1)  # [batch]
```

## 3.2 Late Fusion Model

```python
# models/multimodal_fusion.py
import torch
import torch.nn as nn

class LateFusionModel(nn.Module):
    """
    Late Fusion: osobne sieci dla audio i tekstu, połącz na koniec
    """
    
    def __init__(self, audio_model, mil_text_model, fusion_hidden=128):
        super().__init__()
        
        self.audio_model = audio_model
        self.mil_text_model = mil_text_model
        
        # Fusion network
        self.fusion_net = nn.Sequential(
            nn.Linear(2, fusion_hidden),  # 2 inputs: audio_score + text_score
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(fusion_hidden, 1),
            nn.Sigmoid()
        )
    
    def forward(self, spectrograms, bag_instances):
        """
        spectrograms: [batch, 1, 128, 400] (audio)
        bag_instances: list of lists (tekst)
        """
        # Audio forward
        audio_scores = self.audio_model(spectrograms)  # [batch]
        
        # Text forward
        text_scores_list = []
        for instances in bag_instances:
            text_score = self.mil_text_model(instances)  # scalar
            text_scores_list.append(text_score)
        text_scores = torch.cat(text_scores_list, dim=0)  # [batch]
        
        # Fusion
        combined = torch.stack([audio_scores, text_scores], dim=1)  # [batch, 2]
        fused_output = self.fusion_net(combined)  # [batch, 1]
        
        return fused_output.squeeze(1), audio_scores, text_scores

class EarlyCalibratedFusion(nn.Module):
    """
    Alternatywna fuzja: weighted combination z learned weights
    """
    
    def __init__(self, audio_model, mil_text_model):
        super().__init__()
        
        self.audio_model = audio_model
        self.mil_text_model = mil_text_model
        
        # Learn weights
        self.w_audio = nn.Parameter(torch.tensor(0.5))
        self.w_text = nn.Parameter(torch.tensor(0.5))
    
    def forward(self, spectrograms, bag_instances):
        audio_scores = self.audio_model(spectrograms)
        text_scores = torch.tensor([
            self.mil_text_model(instances).item() for instances in bag_instances
        ])
        
        # Normalize weights
        w_audio_norm = torch.sigmoid(self.w_audio)
        w_text_norm = torch.sigmoid(self.w_text)
        total_w = w_audio_norm + w_text_norm
        w_audio_norm /= total_w
        w_text_norm /= total_w
        
        # Weighted average
        fused = w_audio_norm * audio_scores + w_text_norm * text_scores
        
        return fused, audio_scores, text_scores, {
            'w_audio': w_audio_norm.item(),
            'w_text': w_text_norm.item()
        }
```

## 3.3 Full Training Pipeline

```python
# experiments/train_multimodal.py
import torch
from torch.optim import AdamW
from models.audio_cnn_model import AudioCNNModel
from models.mil_text_model import MILTextModel
from models.multimodal_fusion import LateFusionModel
from data.multimodal_dataloader import create_multimodal_loaders
from evaluation.report import EvaluationReport
from config import CONFIG

def train_multimodal_system():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create loaders (mix audio + text data)
    train_loader, val_loader, test_loader = create_multimodal_loaders(
        'data/instances.csv',
        'data/spectrograms',
        batch_size=8
    )
    
    # Load pre-trained components
    audio_model = AudioCNNModel().to(device)
    text_model = MILTextModel(alpha=0.6, beta=0.5).to(device)
    
    # Fusion model
    fusion_model = LateFusionModel(audio_model, text_model).to(device)
    
    # Optimize only fusion network (keep pre-trained weights frozen)
    optimizer = AdamW(fusion_model.fusion_net.parameters(), lr=1e-3)
    
    # Training loop
    best_val_f1 = 0
    
    for epoch in range(50):
        fusion_model.train()
        total_loss = 0
        
        for batch in train_loader:
            specs = batch['spectrograms'].to(device)
            bag_instances = batch['bag_instances']
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            fused_preds, audio_preds, text_preds = fusion_model(specs, bag_instances)
            
            loss = torch.nn.functional.binary_cross_entropy(fused_preds, labels.float())
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        # Validation
        fusion_model.eval()
        val_preds = []
        val_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                specs = batch['spectrograms'].to(device)
                bag_instances = batch['bag_instances']
                labels = batch['labels']
                
                fused_preds, _, _ = fusion_model(specs, bag_instances)
                val_preds.extend(fused_preds.cpu().numpy())
                val_labels.extend(labels.numpy())
        
        from sklearn.metrics import f1_score
        val_f1 = f1_score(val_labels, np.array(val_preds) > 0.5)
        
        print(f"Epoch {epoch}: Loss {total_loss:.4f}, Val F1 {val_f1:.4f}")
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(fusion_model.state_dict(), CONFIG.MODELS_DIR / 'multimodal_best.pt')
    
    # Test
    fusion_model.load_state_dict(torch.load(CONFIG.MODELS_DIR / 'multimodal_best.pt'))
    
    test_preds = []
    test_audio_preds = []
    test_text_preds = []
    test_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            specs = batch['spectrograms'].to(device)
            bag_instances = batch['bag_instances']
            labels = batch['labels']
            
            fused, audio, text = fusion_model(specs, bag_instances)
            test_preds.extend(fused.cpu().numpy())
            test_audio_preds.extend(audio.cpu().numpy())
            test_text_preds.extend(text.cpu().numpy())
            test_labels.extend(labels.numpy())
    
    test_preds = np.array(test_preds)
    test_pred_binary = (test_preds > 0.5).astype(int)
    test_labels = np.array(test_labels)
    
    # Report
    report = EvaluationReport(
        'Multimodal Late Fusion (Audio + MIL Text)',
        test_labels, test_pred_binary, test_preds
    )
    report.generate()
    report.print_report()
    report.save_json(CONFIG.RESULTS_DIR / 'multimodal_results.json')
    
    # Ablation: pokaż kontrybucję audio vs tekstu
    print("\n=== Ablation Analysis ===")
    print(f"Audio-only F1: {f1_score(test_labels, (np.array(test_audio_preds) > 0.5).astype(int)):.4f}")
    print(f"Text-only F1:  {f1_score(test_labels, (np.array(test_text_preds) > 0.5).astype(int)):.4f}")
    print(f"Multimodal F1: {f1_score(test_labels, test_pred_binary):.4f}")

if __name__ == '__main__':
    train_multimodal_system()
```

## 3.4 Uncertainty Estimation (MC Dropout)

```python
# models/mc_dropout.py
import torch
import torch.nn.functional as F

class MCDropoutEstimator:
    """
    Szacuj niepewność predykcji przez MC Dropout
    """
    
    def __init__(self, model, n_samples=10):
        self.model = model
        self.n_samples = n_samples
    
    def enable_dropout(self):
        """Włącz dropout nawet w .eval() mode"""
        for m in self.model.modules():
            if isinstance(m, torch.nn.Dropout):
                m.train()
    
    def predict_with_uncertainty(self, inputs):
        """
        Zwraca: mean prediction + std uncertainty
        """
        self.model.eval()
        self.enable_dropout()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.n_samples):
                pred = self.model(inputs)
                predictions.append(pred)
        
        predictions = torch.stack(predictions)  # [n_samples, batch]
        
        mean_pred = predictions.mean(dim=0)
        std_pred = predictions.std(dim=0)
        
        return mean_pred, std_pred
    
    def plot_uncertainty_calibration(self, test_preds, test_uncertainties, test_labels):
        """
        Rysuj: czy model jest \"pewny\" kiedy ma rację?
        """
        import matplotlib.pyplot as plt
        
        correct = (test_preds == test_labels).astype(int)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        ax.scatter(test_uncertainties[correct == 1], [1]*sum(correct), alpha=0.5, label='Correct')
        ax.scatter(test_uncertainties[correct == 0], [0]*sum(1-correct), alpha=0.5, label='Incorrect')
        
        ax.set_xlabel('Prediction Uncertainty (Std)')
        ax.set_ylabel('Correctness')
        ax.set_ylim(-0.1, 1.1)
        ax.legend()
        ax.set_title('Calibration: Is model uncertain on wrong predictions?')
        
        return fig
```

## 3.5 Final Comparison Report

```python
# experiments/compare_all_models.py
import json
import pandas as pd
from pathlib import Path
from config import CONFIG

def generate_comparison_table():
    """Zrób tabelę porównującą wszystkie modele"""
    
    results = {}
    
    # Load all reports
    for result_file in Path(CONFIG.RESULTS_DIR).glob('*_results.json'):
        model_name = result_file.stem.replace('_results', '')
        
        with open(result_file, 'r') as f:
            data = json.load(f)
            results[model_name] = data
    
    # Create comparison DataFrame
    df_comparison = pd.DataFrame()
    
    for model_name, metrics in results.items():
        point_estimates = metrics['metrics']
        bootstrap_ci = metrics['bootstrap_ci']
        
        row = {
            'Model': model_name,
            'F1': f"{point_estimates['f1']:.3f} [{bootstrap_ci['f1']['lower_ci']:.3f}, {bootstrap_ci['f1']['upper_ci']:.3f}]",
            'AUROC': f"{point_estimates['auroc']:.3f} [{bootstrap_ci['auroc']['lower_ci']:.3f}, {bootstrap_ci['auroc']['upper_ci']:.3f}]",
            'Precision': f"{point_estimates['precision']:.3f}",
            'Recall': f"{point_estimates['recall']:.3f}",
        }
        
        df_comparison = pd.concat([df_comparison, pd.DataFrame([row])], ignore_index=True)
    
    print("\n=== MODEL COMPARISON ===")
    print(df_comparison.to_string(index=False))
    
    df_comparison.to_csv(CONFIG.RESULTS_DIR / 'model_comparison.csv', index=False)
    
    return df_comparison

# Uruchom
if __name__ == '__main__':
    generate_comparison_table()
```

## 3.6 Czego powinna osiągnąć Sprint 3

✅ Audio CNN model  
✅ Late Fusion architecture  
✅ Full multimodal training  
✅ MC Dropout uncertainty estimation  
✅ Porównanie: Audio vs Text vs Multimodal  
✅ Publication-ready results & tables  

**Oczekiwane wyniki:** F1 ≈ 0.85–0.90, AUROC ≈ 0.90–0.95

---

# Finalna Struktura Projektu

```
doktorat-depresja/
├── config.py                          # Centralna konfiguracja
├── requirements.txt
│
├── data/
│   ├── daic_labels.csv               # Labels z PHQ-8
│   ├── audio_features.csv            # Ekstrakcje audio
│   ├── text_features.csv             # Ekstrakcje tekstu
│   ├── instances.csv                 # MIL instances (segment wywiadu)
│   ├── spectrograms/                 # Mel-spektrogramy
│   ├── X_train.npy, X_val.npy, X_test.npy
│   └── y_train.npy, y_val.npy, y_test.npy
│
├── models/
│   ├── __init__.py
│   ├── audio_cnn_model.py
│   ├── mil_text_model.py
│   ├── multimodal_fusion.py
│   ├── mc_dropout.py
│   └── xgboost_baseline.pkl          # Trained baseline
│
├── data/
│   ├── interview_segmentation.py     # MIL preprocessing
│   ├── mil_dataloader.py             # Dataset dla MIL
│   └── multimodal_dataloader.py
│
├── baselines/
│   └── xgboost_baseline.py
│
├── experiments/
│   ├── run_baselines.py
│   ├── train_mil_text.py
│   ├── train_multimodal.py
│   └── compare_all_models.py
│
├── evaluation/
│   ├── metrics.py
│   ├── report.py
│   └── calibration.py
│
├── interpretability/
│   ├── lime_explanation.py
│   ├── saliency_maps.py
│   └── feature_importance.py
│
├── results/
│   ├── xgboost_baseline_results.json
│   ├── mil_text_results.json
│   ├── multimodal_results.json
│   ├── model_comparison.csv
│   ├── roc_pr_curves.png
│   └── ...
│
├── logs/
│   └── training_logs.txt
│
└── README.md                          # Dokumentacja projektu
```

---

# Checklist: Od Baseline'u do PhD Prototypu

## Sprint 1 Checklist
- [ ] Config.py + setup eksperymentalny
- [ ] XGBoost baseline z full ewaluacją
- [ ] Bootstrap CI dla wszystkich metryk (1000 resamples)
- [ ] ROC + PR curves
- [ ] Test → F1 ≈ 0.75–0.80

## Sprint 2 Checklist
- [ ] Segmentacja wywiadu na instancje
- [ ] MIL Dataset loader
- [ ] MT5 + RoBERTa fusion model
- [ ] MIL training loop (bag-level BCE loss)
- [ ] LIME interpretowalność
- [ ] Test → F1 ≈ 0.82–0.88

## Sprint 3 Checklist
- [ ] Audio CNN model z LSTM
- [ ] Late Fusion architecture
- [ ] Multimodal training pipeline
- [ ] MC Dropout uncertainty
- [ ] Ablation: audio-only vs text-only vs multimodal
- [ ] Comparison table (wszystkie modele)
- [ ] Test → F1 ≈ 0.85–0.90

---

# Publikowalność & Dyplom

Na koniec masz materiały na:

1. **Paper 1 – MIL Text Framework**
   - Pokaż: MIL dla depresji = lepszy od klasycznego text-only
   - Wyniki: F1, AUROC + CI
   - Interpretowalność: instance-level scores + LIME

2. **Paper 2 – Multimodal Fusion**
   - Porównanie: audio vs tekst vs fuzja
   - Ablation study
   - Clinical insights (które cechy mowy + tekstu są ważne)

3. **Dissertation Monograph**
   - Chapters: Intro, Methods (preprocessing, MIL, fusion), Results, Discussion, Limitations
   - Appendices: kod, hyperparam sweeps, pełne metryki

To **wydajne, publikowalne, oparte na state-of-the-art**, i całkowicie dostosowane do Twojej wizji multimodalnego modelu na DAIC-WOZ.

---

Powodzenia! 🚀

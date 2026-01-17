# Pełny Opis Techniczny Projektu MDDM
## Multimodal Depression Detection Model

---

## 1. Wprowadzenie

### 1.1 Cel Projektu

MDDM (Multimodal Depression Detection Model) to prototyp systemu do automatycznej detekcji depresji na podstawie danych audio i tekstowych z wywiadów klinicznych. Projekt wykorzystuje dataset DAIC-WOZ (Distress Analysis Interview Corpus - Wizard of Oz) i implementuje zaawansowane metody uczenia maszynowego, w tym Multi-Instance Learning (MIL) oraz fuzję multimodalną.

### 1.2 Zakres Funkcjonalności

- **Ekstrakcja cech**: Audio (spektrogramy, MFCC, openSMILE) i tekstowe (transformery)
- **Modelowanie**: Baseline (XGBoost), MIL dla tekstu, CNN dla audio, fuzja multimodalna
- **Ewaluacja**: Kompleksowe metryki z bootstrap confidence intervals
- **Interpretowalność**: LIME, saliency maps, feature importance
- **Niepewność**: MC Dropout dla estymacji niepewności predykcji

---

## 2. Architektura Systemu

### 2.1 Struktura Projektu

```
MDDM/
├── config.py                    # Centralna konfiguracja
├── requirements.txt            # Zależności (CPU)
├── requirements_gpu.txt       # Zależności (GPU)
│
├── data/                       # Przetwarzanie danych
│   ├── interview_segmentation.py    # Segmentacja wywiadów na instancje
│   ├── mil_dataloader.py            # DataLoader dla MIL
│   └── processed/                  # Przetworzone dane
│       ├── daic_labels.csv
│       ├── audio_features.csv
│       ├── text_features.csv
│       ├── instances.csv
│       └── spectrograms/
│
├── models/                     # Modele głębokiego uczenia
│   ├── audio_cnn_model.py     # CNN + LSTM dla audio
│   ├── mil_text_model.py       # MIL dla tekstu (MT5 + RoBERTa)
│   ├── multimodal_fusion.py   # Fuzja late/early/calibrated
│   └── mc_dropout.py          # Estymacja niepewności
│
├── baselines/                  # Modele baseline
│   ├── xgboost_baseline.py    # XGBoost z GPU
│   └── optuna_tuning.py       # Optymalizacja hyperparametrów
│
├── experiments/               # Skrypty treningowe
│   ├── run_baselines.py
│   ├── train_mil_text.py
│   ├── train_multimodal.py
│   └── compare_all_models.py
│
├── evaluation/                # Ewaluacja
│   ├── metrics.py             # Metryki + bootstrap CI
│   └── report.py             # Generowanie raportów
│
├── src/                       # Moduły pomocnicze
│   ├── features/              # Ekstraktory cech
│   ├── fusion/                # Fuzja cech
│   └── utils/                 # Narzędzia
│
└── results/                   # Wyniki eksperymentów
    ├── *.json                 # Raporty JSON
    ├── *.png                  # Wykresy (ROC, PR, calibration)
    └── model_comparison.csv
```

### 2.2 Przepływ Danych (Pipeline)

```
1. Ekstrakcja Metadanych
   └─> DAIC-WOZ ZIP files → daic_labels.csv

2. Ekstrakcja Cech Audio
   └─> WAV files → Spektrogramy (mel) → audio_features.csv

3. Ekstrakcja Cech Tekstowych
   └─> XML transkrypcje → Instancje (QA-pary) → instances.csv

4. Trening Modeli
   ├─> Baseline: XGBoost na fuzjowanych cechach
   ├─> MIL Text: MT5 + RoBERTa → bag-level predictions
   ├─> Audio CNN: Spektrogramy → CNN + LSTM → predictions
   └─> Multimodal: Late Fusion (audio + text)

5. Ewaluacja
   └─> Metryki + Bootstrap CI + Wykresy
```

---

## 3. Technologie i Narzędzia

### 3.1 Główne Biblioteki

**Deep Learning:**
- `torch >= 2.6.0` - Framework głębokiego uczenia
- `torchvision >= 0.21.0` - Narzędzia wizualne
- `torchaudio >= 2.6.0` - Przetwarzanie audio

**Transformery NLP:**
- `transformers >= 4.40.0` - Hugging Face transformers
- `sentence-transformers >= 2.2.0` - Embeddingi tekstowe
- `tokenizers >= 0.19.0` - Tokenizacja

**Machine Learning:**
- `xgboost >= 2.0.0` - Gradient boosting (GPU support)
- `scikit-learn >= 1.4.0` - Klasyczne ML
- `optuna >= 3.5.0` - Optymalizacja hyperparametrów

**Przetwarzanie Audio:**
- `librosa >= 0.10.0` - Analiza sygnałów audio
- `soundfile >= 0.12.0` - I/O plików audio
- `opensmile >= 2.5.0` - Ekstrakcja cech audio (opcjonalne)

**Narzędzia:**
- `numpy >= 1.26.0` - Obliczenia numeryczne
- `pandas >= 2.2.0` - Manipulacja danymi
- `matplotlib >= 3.8.0` - Wizualizacja
- `tqdm >= 4.66.0` - Progress bars

### 3.2 Wymagania Sprzętowe

**Minimalne:**
- CPU: 4+ rdzenie
- RAM: 8GB
- Dysk: ~10GB wolnego miejsca
- Python: 3.8+

**Zalecane (dla GPU):**
- GPU: NVIDIA z CUDA support (RTX 3060+)
- RAM: 16GB+
- CUDA: 11.8+
- cuDNN: 8.0+

---

## 4. Dataset i Preprocessing

### 4.1 DAIC-WOZ Dataset

**Charakterystyka:**
- **Źródło**: USC Institute for Creative Technologies
- **Licencja**: Academic Use Agreement (wymagana)
- **Rozmiar**: ~189 wywiadów klinicznych
- **Format**: Audio (WAV) + Transkrypcje (XML)
- **Etykiety**: PHQ-8 scores (depresja: 0/1)

**Split Standardowy:**
- Train: 107 sesji
- Validation: 35 sesji
- Test: 47 sesji

**Problematyczne Sesje (pomijane):**
- 451, 458, 480 - brakujące transkrypcje
- 409 - błąd etykietowania

### 4.2 Preprocessing Audio

**Kroki:**
1. **Wczytanie**: WAV → numpy array (sample_rate=16000 Hz)
2. **Normalizacja**: Amplitude normalization
3. **Spektrogram Mel**: 
   - `n_mels=128` (częstotliwości)
   - `n_fft=2048`, `hop_length=512`
   - Długość: `SPECTROGRAM_LENGTH=400` frames
4. **MFCC**: 13 współczynników (opcjonalnie)
5. **openSMILE**: 6373 cechy (ComParE, opcjonalnie)

**Output:**
- Spektrogramy: `data/processed/spectrograms/{session_id}_spec.npy`
- Cechy: `data/processed/audio_features.csv`

### 4.3 Preprocessing Tekstu

**Segmentacja:**
- **Metoda**: QA-pary (odpowiedzi pacjenta = instancje)
- **Format**: XML transkrypcje → lista instancji
- **Filtrowanie**: Tylko wypowiedzi `Participant` (nie `Ellie`)

**Embeddingi:**
- **Model 1**: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- **Model 2**: `roberta-base` (768-dim)
- **Fuzja**: Concatenation → 1152-dim

**Output:**
- Instancje: `data/processed/instances.csv`
  - Kolumny: `session_id`, `sequence_num`, `text`, `depression` (bag label)

---

## 5. Modele i Architektury

### 5.1 Baseline: XGBoost

**Architektura:**
- Gradient Boosting Decision Trees
- GPU acceleration (CUDA)
- Early stopping na validation set
- Feature selection (SelectKBest, k=50)

**Hyperparametry:**
```python
n_estimators: 500
max_depth: 6
learning_rate: 0.05
subsample: 0.8
colsample_bytree: 0.8
early_stopping_rounds: 30
```

**Funkcje:**
- Sample weights dla niezbalansowanych klas
- Optymalizacja threshold (F1-based)
- Feature importance analysis

**Input:** Fuzjowane cechy audio+tekst (early fusion)
**Output:** Prawdopodobieństwo depresji [0, 1]

### 5.2 MIL Text Model

**Architektura Multi-Instance Learning:**

```
Input: Bag (wywiad) = Lista instancji (odpowiedzi)
  ↓
1. Ekstrakcja cech per instancja:
   - MT5 embeddings (512-dim)
   - RoBERTa embeddings (768-dim)
   - Concatenation → 1280-dim
  ↓
2. Instance-level classifier (MLP):
   - Input: 1280-dim
   - Hidden: 256 → 128
   - Output: instance_score [0, 1]
  ↓
3. MIL Pooling:
   - Max pooling: max(instance_scores)
   - Mean pooling: mean(instance_scores)
   - Proportion: % instances > α threshold
  ↓
4. Bag-level prediction:
   - Output: depression probability [0, 1]
```

**Hyperparametry MIL:**
```python
alpha: 0.6  # Próg dla instancji
beta: 0.5   # Próg dla agregowanego score
hidden_dim: 256
dropout_rate: 0.3
```

**Training:**
- Loss: Binary Cross-Entropy (bag-level)
- Optimizer: AdamW (lr=2e-4)
- Early stopping: patience=15
- Freeze transformers (feature extraction only)

### 5.3 Audio CNN Model

**Architektura:**

```
Input: Mel-spectrogram (1, 128, 400)
  ↓
Conv Block 1:
  - Conv2d(1→32, kernel=3, padding=1)
  - BatchNorm + ReLU
  - MaxPool2d(2,2) → (32, 64, 200)
  ↓
Conv Block 2:
  - Conv2d(32→64, kernel=3, padding=1)
  - BatchNorm + ReLU
  - MaxPool2d(2,2) → (64, 32, 100)
  ↓
Conv Block 3:
  - Conv2d(64→128, kernel=3, padding=1)
  - BatchNorm + ReLU
  - MaxPool2d(2,2) → (128, 16, 50)
  ↓
Flatten → (128*16*50 = 102400)
  ↓
LSTM:
  - Input: 102400
  - Hidden: 256, layers=2
  - Output: last hidden state (256-dim)
  ↓
Classifier:
  - Linear(256 → 128) + ReLU + Dropout(0.3)
  - Linear(128 → 1) + Sigmoid
  ↓
Output: depression probability [0, 1]
```

**Training:**
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr=1e-3)
- Batch size: 16
- Data augmentation: Time masking, frequency masking

### 5.4 Multimodal Fusion

**Strategie Fuzji:**

#### 5.4.1 Late Fusion

```
Audio Model → audio_score [0, 1]
Text Model  → text_score [0, 1]
  ↓
Fusion Network:
  - Input: [audio_score, text_score] (2-dim)
  - Hidden: 64 → 32
  - Output: fused_score [0, 1]
```

**Training:**
- Freeze modality models
- Train only fusion network
- Optimizer: AdamW (lr=1e-3)

#### 5.4.2 Calibrated Fusion

```
Audio Model → audio_score
Text Model  → text_score
  ↓
Learned weights:
  w_audio, w_text (softmax-normalized)
  ↓
Output: w_audio * audio_score + w_text * text_score
```

#### 5.4.3 Feature Fusion

```
Audio Model → audio_features (128-dim)
Text Model  → text_features (128-dim)
  ↓
Concatenation → fused_features (256-dim)
  ↓
Classifier:
  - Linear(256 → 128) + ReLU + Dropout
  - Linear(128 → 1) + Sigmoid
```

---

## 6. Ewaluacja

### 6.1 Metryki

**Klasyfikacja:**
- Accuracy, Precision, Recall, F1-Score
- AUROC (Area Under ROC Curve)
- AUPRC (Area Under Precision-Recall Curve)
- MCC (Matthews Correlation Coefficient)

**Kliniczne:**
- Sensitivity (Recall)
- Specificity
- Brier Score (calibration)

**Niepewność:**
- MC Dropout: mean + std predictions
- Calibration analysis: uncertainty-error correlation

### 6.2 Bootstrap Confidence Intervals

**Metoda:**
- Resampling z replacement (n=1000)
- 95% confidence intervals dla wszystkich metryk
- Percentile method

**Użycie:**
```python
from evaluation.metrics import MetricsComputer

ci = MetricsComputer.bootstrap_ci(
    y_true, y_proba, y_pred,
    metric_name='f1',
    n_resamples=1000,
    ci=0.95
)
# Output: {'mean': 0.85, 'lower_ci': 0.82, 'upper_ci': 0.88, 'std': 0.015}
```

### 6.3 Wizualizacje

**Generowane wykresy:**
1. **ROC Curve**: FPR vs TPR + AUC
2. **Precision-Recall Curve**: Recall vs Precision + AP
3. **Calibration Curve**: Predicted vs Actual probabilities
4. **Uncertainty Analysis**: Uncertainty vs correctness

**Format:** PNG (300 DPI), zapisane w `results/`

### 6.4 Raporty

**Format JSON:**
```json
{
  "model_name": "Multimodal Late Fusion",
  "metrics": {
    "f1": 0.852,
    "auroc": 0.901,
    "precision": 0.834,
    "recall": 0.871
  },
  "bootstrap_ci": {
    "f1": {
      "mean": 0.850,
      "lower_ci": 0.820,
      "upper_ci": 0.880,
      "std": 0.015
    }
  }
}
```

**Porównanie modeli:**
- CSV: `results/model_comparison.csv`
- Console output: Tabela porównawcza
- LaTeX: Format dla publikacji

---

## 7. Konfiguracja

### 7.1 Plik `config.py`

**Struktura:**
- `Config` dataclass z wszystkimi parametrami
- Automatyczne tworzenie katalogów
- Seed management dla reproducibility
- GPU detection

**Główne sekcje:**

```python
# Dataset
RANDOM_SEED: 42
TRAIN_SIZE: 107
VAL_SIZE: 35
TEST_SIZE: 47

# Audio
SAMPLE_RATE: 16000
N_MELS: 128
N_MFCC: 13
SPECTROGRAM_LENGTH: 400
USE_OPENSMILE: True

# Text
TEXT_MODEL_NAME: "sentence-transformers/all-MiniLM-L6-v2"
ROBERTA_MODEL_NAME: "roberta-base"
MAX_TEXT_LENGTH: 512

# Training
BATCH_SIZE: 32
MIL_BATCH_SIZE: 8
CNN_BATCH_SIZE: 16
EPOCHS: 100
LEARNING_RATE: 1e-3
EARLY_STOPPING_PATIENCE: 15
DROPOUT_RATE: 0.3

# GPU
USE_GPU: True (if CUDA available)
DEVICE: "cuda" or "cpu"
USE_MIXED_PRECISION: True

# Evaluation
BOOTSTRAP_CI_RESAMPLES: 1000
CONFIDENCE_LEVEL: 0.95
MC_DROPOUT_SAMPLES: 10
```

### 7.2 Ścieżki

Wszystkie ścieżki są automatycznie tworzone przez `Config.__post_init__()`:

- `DATA_ROOT`: `DAIC-WOZ-Dataset/
- `RAW_DATA_DIR`: `data/raw/`
- `PROCESSED_DATA_DIR`: `data/processed/`
- `MODELS_DIR`: `data/models/`
- `RESULTS_DIR`: `results/`
- `LOGS_DIR`: `logs/`

---

## 8. Instalacja i Użycie

### 8.1 Instalacja

```bash
# 1. Virtual environment
python -m venv venv_daic
venv_daic\Scripts\activate  # Windows
# source venv_daic/bin/activate  # Linux/Mac

# 2. Zainstaluj zależności
pip install -r requirements.txt  # CPU
# pip install -r requirements_gpu.txt  # GPU

# 3. Przygotuj dane
# Umieść pliki ZIP z DAIC-WOZ w DAIC-WOZ-Dataset/
```

### 8.2 Pipeline

**Interaktywne Menu (Zalecane):**
```bash
python scripts/pipeline_menu.py
```

**Pełny Pipeline:**
```bash
python scripts/run_full_pipeline.py
```

**Krok po Kroku:**
```bash
# 1. Ekstrakcja metadanych
python scripts/01_extract_metadata.py

# 2. Ekstrakcja cech audio
python scripts/02_extract_audio_features.py

# 3. Ekstrakcja cech tekstowych
python scripts/03_extract_text_features.py

# 4. Segmentacja na instancje (MIL)
python data/interview_segmentation.py

# 5. Trening baseline
python experiments/run_baselines.py

# 6. Trening MIL text
python experiments/train_mil_text.py

# 7. Trening multimodal
python experiments/train_multimodal.py

# 8. Porównanie modeli
python experiments/compare_all_models.py
```

### 8.3 Przykłady Użycia

**XGBoost Baseline:**
```python
from baselines.xgboost_baseline import XGBoostBaseline
import numpy as np

# Load data
X_train = np.load('data/processed/X_train.npy')
y_train = np.load('data/processed/y_train.npy')
X_val = np.load('data/processed/X_val.npy')
y_val = np.load('data/processed/y_val.npy')

# Train
baseline = XGBoostBaseline()
baseline.fit(X_train, y_train, X_val, y_val)

# Predict
y_pred, y_proba = baseline.predict(X_test, return_proba=True)
```

**MIL Text Model:**
```python
from models.mil_text_model import MILTextModel
from data.mil_dataloader import create_mil_dataloaders

# Create loaders
train_loader, val_loader, test_loader = create_mil_dataloaders()

# Model
model = MILTextModel(device='cuda')
model = model.to('cuda')

# Training loop (see experiments/train_mil_text.py)
```

**Multimodal Fusion:**
```python
from models.multimodal_fusion import LateFusionModel
from models.audio_cnn_model import AudioCNNModel
from models.mil_text_model import MILTextModel

# Load pre-trained models
audio_model = AudioCNNModel().load()
text_model = MILTextModel().load()

# Fusion
fusion = LateFusionModel(audio_model, text_model)
# Training (see experiments/train_multimodal.py)
```

**Ewaluacja:**
```python
from evaluation.report import EvaluationReport

report = EvaluationReport(
    model_name="My Model",
    y_true=y_test,
    y_pred=y_pred,
    y_proba=y_proba
)
report.generate(n_bootstrap=1000)
report.print_report()
report.save_json()
report.save_plots()
```

---

## 9. Reproducibility

### 9.1 Seed Management

Wszystkie losowe operacje używają `CONFIG.RANDOM_SEED=42`:

```python
CONFIG.set_seed()  # Sets seeds for random, numpy, torch, CUDA
```

**Zapewnia:**
- Reprodukowalne train/val/test splits
- Deterministic training (CUDA)
- Reprodukowalne bootstrap CI

### 9.2 Zapisywanie Modeli

**Format:**
- PyTorch: `.pt` (state_dict)
- XGBoost: `.pkl` (pickle)
- Scaler/Selector: `.pkl`

**Lokalizacja:**
- `data/models/{model_name}.pt` lub `.pkl`

### 9.3 Logi

**Training logs:**
- Console output z progress bars
- Metryki per epoch
- Best model checkpointing

**Evaluation logs:**
- JSON reports z timestamp
- Wykresy z nazwą modelu
- Comparison tables

---

## 10. Rozszerzenia i Przyszłe Prace

### 10.1 Zaimplementowane

✅ Baseline XGBoost z GPU  
✅ MIL Text Model (MT5 + RoBERTa)  
✅ Audio CNN + LSTM  
✅ Late/Calibrated/Feature Fusion  
✅ Bootstrap CI  
✅ MC Dropout uncertainty  
✅ LIME interpretability  

### 10.2 Możliwe Rozszerzenia

**Modelowanie:**
- [ ] Attention mechanisms w MIL
- [ ] Transformer dla audio (Wav2Vec2)
- [ ] Cross-modal attention
- [ ] Self-supervised pre-training

**Ewaluacja:**
- [ ] Cross-validation
- [ ] External validation dataset
- [ ] Clinical interpretability metrics

**Deployment:**
- [ ] REST API
- [ ] Real-time inference
- [ ] Model serving (TorchServe)

---

## 11. Licencja i Zastrzeżenia

**Dataset DAIC-WOZ:**
- Wymaga podpisania Academic Use Agreement
- Dostęp: https://dcapswoz.ict.usc.edu/

**Kod:**
- Do użytku edukacyjnego i badawczego
- Autor: Prototyp na podstawie dokumentacji w `daic_preprocessing_guide.md`

**Ograniczenia:**
- Dataset: ~189 sesji (mały rozmiar)
- Brak walidacji zewnętrznej
- Model nie jest narzędziem diagnostycznym (tylko badawczy)

---

## 12. Referencje

**Dataset:**
- Gratch, J., et al. (2014). "The Distress Analysis Interview Corpus of human and computer interviews." LREC.

**Metody:**
- Multi-Instance Learning dla depresji (Nature papers)
- Late Fusion w multimodalnych systemach
- Bootstrap CI dla małych test sets

**Narzędzia:**
- PyTorch: https://pytorch.org/
- Hugging Face: https://huggingface.co/
- XGBoost: https://xgboost.readthedocs.io/

---

**Wersja dokumentu:** 1.0  
**Data:** 2024  
**Autor:** MDDM Development Team

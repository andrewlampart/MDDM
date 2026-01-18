# MDDM - Multimodal Depression Detection Model

Projekt detekcji depresji na podstawie analizy multimodalnej (audio + tekst) z wykorzystaniem DAIC-WOZ dataset.

## Szybki start

```bash
# Instalacja zależności
pip install -r requirements.txt

# Uruchomienie pełnego eksperymentu
python scripts/run_full_experiment.py

# Opcje:
#   --skip-audio    Pomiń Wav2Vec (tylko prosody features)
#   --quick         Szybki test (3-fold CV zamiast 5-fold)
```

## Struktura projektu

```
MDDM/
├── config.py                 # Konfiguracja projektu
├── requirements.txt          # Zależności
│
├── models/                   # Modele ML
│   ├── text_encoder_bert.py     # BERT text encoder (512-dim)
│   ├── audio_encoder_advanced.py # Wav2Vec + prosody (~786-dim)
│   └── fusion_fixed.py          # Late/Early fusion z regularyzacją
│
├── evaluation/               # Ewaluacja
│   ├── metrics.py               # Metryki (F1, AUROC, MCC, etc.)
│   ├── calibration.py           # Platt Scaling, ECE
│   └── ablation.py              # Ablation study framework
│
├── scripts/                  # Skrypty
│   ├── run_full_experiment.py   # GŁÓWNY SKRYPT - pełny pipeline
│   ├── ablation_study.py        # Standalone ablation
│   ├── validate_improvements.py # Walidacja wyników
│   └── cleanup_old_results.py   # Czyszczenie starych danych
│
├── data/                     # Dane
│   ├── raw/                     # Surowe dane DAIC-WOZ
│   └── processed/               # Przetworzone features
│
├── DAIC-WOZ-Dataset/         # Split files + dokumentacja
├── results/                  # Wyniki eksperymentów
├── logs/                     # Logi treningów
└── docs/                     # Dokumentacja
    └── methods_section.md       # Sekcja Methods do publikacji
```

## Pipeline

Skrypt `run_full_experiment.py` wykonuje:

1. **Walidacja danych** - sprawdza dostępność plików audio i transkryptów
2. **Ładowanie etykiet** - PHQ-8 binary labels z split files
3. **Ekstrakcja features**:
   - Text: BERT SentenceTransformer (512-dim)
   - Audio: Wav2Vec 2.0 (768-dim) + prosody (18-dim)
4. **Trening modeli** (5-fold CV):
   - XGBoost Early Fusion (audio + text)
   - XGBoost Text-only
   - XGBoost Audio-only
5. **Ablation Study** - systematyczne testowanie modalności
6. **Raport** - wyniki w `results/`

## Wymagania

- Python 3.10+
- PyTorch 2.0+
- transformers (Wav2Vec 2.0)
- sentence-transformers (BERT)
- xgboost
- librosa (audio processing)
- scikit-learn

## Success Criteria

| Metryka | Minimal | Good | Excellent |
|---------|---------|------|-----------|
| F1 | ≥ 0.65 | ≥ 0.70 | ≥ 0.75 |
| AUROC | ≥ 0.75 | ≥ 0.78 | ≥ 0.80 |
| MCC | > 0 | ≥ 0.30 | ≥ 0.40 |
| Specificity | ≥ 0.70 | ≥ 0.72 | ≥ 0.75 |
| ECE | < 0.05 | < 0.04 | < 0.03 |

## Dataset

DAIC-WOZ (Distress Analysis Interview Corpus):
- 189 uczestników (107 train, 35 dev, 47 test)
- Semi-structured clinical interviews
- PHQ-8 depression questionnaire labels

## Autor

Jakub Pawłowski - PhD Research, GUMed

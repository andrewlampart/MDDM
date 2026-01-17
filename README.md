# Multimodal Depression Detection Model (MDDM)

Prototyp multimodalnego modelu do diagnozowania depresji na podstawie danych audio i tekstowych z datasetu DAIC-WOZ.

## Struktura Projektu

```
MDDM/
├── src/                    # Kod źródłowy
│   ├── data/              # Ekstrakcja i ładowanie danych
│   ├── features/          # Ekstrakcja cech (audio + tekst)
│   ├── fusion/            # Fuzja modalności
│   ├── models/            # Modele klasyfikacji
│   └── utils/             # Narzędzia pomocnicze
├── scripts/                # Skrypty pipeline
├── data/                   # Dane (raw, processed, models)
└── DAIC-WOZ-Dataset/       # Dataset (pliki ZIP)
```

## Wymagania

- Python 3.8+
- CUDA (opcjonalne, dla GPU)
- ~10GB wolnego miejsca na dane

## Instalacja

### 1. Utwórz virtual environment

```bash
python -m venv venv_daic
# Windows
venv_daic\Scripts\activate
# Linux/Mac
source venv_daic/bin/activate
```

### 2. Zainstaluj zależności

```bash
# Windows (jeśli pip nie jest w PATH)
python -m pip install -r requirements.txt

# Lub jeśli pip działa:
pip install -r requirements.txt
```

**Uwaga:** openSMILE jest opcjonalne. Jeśli nie jest zainstalowane, system użyje tylko low-level audio features.

### 3. Przygotuj dane

Umieść pliki ZIP z datasetu DAIC-WOZ w folderze `DAIC-WOZ-Dataset/`. Dataset można pobrać z:
https://dcapswoz.ict.usc.edu/ (wymaga podpisania umowy Academic Use Agreement)

## Użycie

### Interaktywne Menu (Zalecane)

Najprostszy sposób na uruchamianie pipeline - menu z checkboxami pokazującymi postęp:

```bash
python scripts/pipeline_menu.py
```

Menu pokazuje które kroki są już wykonane (✓) i pozwala wybrać który krok uruchomić.

### Pełny Pipeline

Uruchomienie całego pipeline'u od ekstrakcji danych do treningu modelu:

```bash
python scripts/run_full_pipeline.py
```

### Krok po Kroku

Możesz też uruchomić każdy krok osobno:

1. **Ekstrakcja metadanych:**
```bash
python scripts/01_extract_metadata.py
```

2. **Ekstrakcja cech audio:**
```bash
python scripts/02_extract_audio_features.py
```

3. **Ekstrakcja cech tekstowych:**
```bash
python scripts/03_extract_text_features.py
```

4. **Fuzja cech:**
```bash
python scripts/04_fusion.py
```

5. **Trening modelu:**
```bash
python scripts/05_train_model.py
```

## Konfiguracja

Parametry można zmienić w `src/utils/config.py`:

- `SAMPLE_RATE`: Sample rate dla audio (domyślnie 16000)
- `USE_OPENSMILE`: Czy używać openSMILE (domyślnie True)
- `USE_TEXT`: Czy używać cech tekstowych (domyślnie True)
- `RANDOM_SEED`: Seed dla reproducibility (domyślnie 42)

## Output

Po uruchomieniu pipeline'u, w folderze `data/processed/` znajdziesz:

- `daic_labels.csv` - Metadane i labels
- `audio_features.csv` - Cechy audio
- `text_features.csv` - Cechy tekstowe
- `daic_fused_features.h5` - Połączone cechy (HDF5)
- `X_train.npy`, `X_val.npy`, `X_test.npy` - Features dla train/val/test
- `y_train.npy`, `y_val.npy`, `y_test.npy` - Labels dla train/val/test

W folderze `data/models/`:

- `baseline_model.pkl` - Wytrenowany model
- `scaler.pkl` - Scaler do normalizacji

## Model

Baseline model używa **Random Forest Classifier** z:
- 100 drzew
- Balanced class weights (dla obsługi niezbalansowanych klas)
- StandardScaler normalization
- Stratified train/val/test split (107/35/47)

## Metryki Ewaluacji

Model jest ewaluowany na validation i test set z następującymi metrykami:
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

## Znane Problemy w Datasecie

Następujące sesje są automatycznie pomijane:
- 451, 458, 480 - brakujące transkrypcje
- 409 - błąd etykietowania

## Troubleshooting

### openSMILE nie zainstalowane
System automatycznie użyje tylko low-level features. To jest OK dla prototypu.

### Brak pamięci GPU
Model transformers (MiniLM) działa też na CPU, choć wolniej.

### Błąd przy ekstrakcji ZIP
Upewnij się, że masz wystarczająco miejsca na dysku (~10GB).

## Licencja

Dataset DAIC-WOZ wymaga podpisania Academic Use Agreement. Kod w tym repozytorium jest dostępny do użytku edukacyjnego i badawczego.

## Autor

Prototyp stworzony na podstawie dokumentacji w `daic_preprocessing_guide.md`.

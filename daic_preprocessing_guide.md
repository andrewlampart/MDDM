# DAIC-WOZ Preprocessing & Audio+Text Feature Extraction
## Prototyp dla multimodalnego modelu diagnozowania depresji

### 1. Setup Środowiska

```bash
# Utwórz virtual environment
python -m venv venv_daic
source venv/Scripts/activate  # Windows
# lub: source venv/bin/activate  # Linux/Mac

# Zainstaluj zależności
pip install librosa numpy scipy pandas scikit-learn matplotlib
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install tensorboard h5py tqdm
pip install opensmile  # Dla paralinguistic features z audio
```

### 2. Pobieranie i Organizacja DAIC-WOZ Dataset

```bash
# Pobierz dataset ze strony: https://dcapswoz.ict.usc.edu/
# Wymaga podpisania umowy (Academic Use Agreement)
# Dataset struktura po rozpkowaniu:
# DAIC-WOZ/
# ├── Interviews_video/  (opcjonalne - nie potrzebne dla audio+tekst)
# ├── Transcriptions/
# │   ├── Participant_*.xml
# │   └── ...
# └── Session_Data/
#     ├── 401/
#     │   ├── 401_AUDIO.wav
#     │   └── 401.xml
#     └── ...

# Sprawdź znane problemy w datasecie
# Brakujące transkrypcje: 451, 458, 480
# Niezsynchronizowane: 318, 321, 341, 362
# Błąd etykietowania: 409 (PHQ-8 = 10, ale depresja = True?)
```

### 3. Preprocessing Dataset - Krok po Kroku

#### 3.1 Ekstrakcja Metadanych i PHQ-8 Labels

```python
import xml.etree.ElementTree as ET
import pandas as pd
from pathlib import Path

def extract_labels_from_xml(xml_path):
    """Ekstrakcja PHQ-8 score oraz binary depression label"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    session = root.find('Session')
    phq8_score = int(session.find('PHQ8_Score').text)
    depression = phq8_score >= 10  # PHQ-8 >= 10 = depresja
    
    return {
        'phq8_score': phq8_score,
        'depression': int(depression),
        'qids_score': int(session.find('QIDS_Score').text) if session.find('QIDS_Score') is not None else None
    }

def load_transcription(xml_path):
    """Ekstrakcja tekstu z pliku transkrypcji"""
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    dialogues = []
    for turn in root.findall('.//Turn'):
        speaker = turn.get('spk')  # 'Participant' lub 'Ellie'
        text = turn.find('Text').text if turn.find('Text') is not None else ''
        
        if speaker == 'Participant' and text:
            dialogues.append(text)
    
    return ' '.join(dialogues)

# Załaduj labels
data_root = Path('path/to/DAIC-WOZ/Session_Data')
metadata = []

for session_dir in data_root.iterdir():
    if session_dir.is_dir() and session_dir.name.isdigit():
        session_id = int(session_dir.name)
        
        # Pomiń znane problemy
        if session_id in [451, 458, 480, 409]:
            continue
        
        xml_path = session_dir / f'{session_id}.xml'
        trans_path = Path('path/to/DAIC-WOZ/Transcriptions') / f'Participant_{session_id}.xml'
        audio_path = session_dir / f'{session_id}_AUDIO.wav'
        
        if xml_path.exists() and audio_path.exists():
            try:
                labels = extract_labels_from_xml(xml_path)
                
                # Tekst jest opcjonalny (może brakować transkrypcji)
                text = extract_transcription(trans_path) if trans_path.exists() else None
                
                metadata.append({
                    'session_id': session_id,
                    'audio_path': str(audio_path),
                    'text': text,
                    **labels
                })
            except Exception as e:
                print(f"Error processing session {session_id}: {e}")

df_labels = pd.DataFrame(metadata)
df_labels.to_csv('daic_labels.csv', index=False)
print(f"Loaded {len(df_labels)} sessions")
print(f"Positive class (depression): {df_labels['depression'].sum()}")
```

#### 3.2 Feature Extraction - Audio Modalność

```python
import librosa
import numpy as np
from scipy.fftpack import fft
import opensmile

class AudioFeatureExtractor:
    """Ekstrakcja cech akustycznych dla diagnozy depresji"""
    
    def __init__(self, sr=16000):
        self.sr = sr
        self.smile = opensmile.Smile(
            feature_set=opensmile.FeatureSet.eGeMAPSv02,
            feature_level=opensmile.FeatureLevel.Functionals
        )
    
    def extract_low_level_features(self, audio, sr):
        """Ekstrahuj spectral i temporal features"""
        features = {}
        
        # 1. Mel-frequency cepstral coefficients (MFCC)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        
        # 2. Spectral Centroid - gdzie skupiona jest energia
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['spec_centroid_mean'] = np.mean(spec_centroid)
        features['spec_centroid_std'] = np.std(spec_centroid)
        
        # 3. Zero Crossing Rate - częstość przejść przez zero
        zcr = librosa.feature.zero_crossing_rate(audio)
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 4. Spectral Rolloff - frekvencja zawierająca 85% energii
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['spec_rolloff_mean'] = np.mean(spec_rolloff)
        features['spec_rolloff_std'] = np.std(spec_rolloff)
        
        # 5. Chromatogram - energia w każdym pitch class
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        
        # 6. Tempogram - rytm
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        features['onset_strength_mean'] = np.mean(onset_env)
        
        return features
    
    def extract_high_level_features(self, audio_path):
        """Użyj openSMILE do ekstrakcji paralinguistic features"""
        # eGeMAPS v02 zawiera 88 funkcjonalnych deskryptorów:
        # - pitch features (F0, voicing)
        # - spectral features
        # - mfcc features
        # - voice quality descriptors
        # - prosody features
        result = self.smile.process_file(audio_path)
        return result.values[0]
    
    def get_spectrogram(self, audio, sr):
        """Zwróć spectrogramu do CNN (dla innej sieci)"""
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db  # Shape: (128, time_frames)
    
    def process_audio(self, audio_path):
        """Pipeline: załaduj audio → ekstrahuj cechy"""
        audio, sr = librosa.load(audio_path, sr=self.sr, mono=True)
        
        # Normalizacja
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        low_level = self.extract_low_level_features(audio, sr)
        high_level = self.extract_high_level_features(audio_path)
        spectrogram = self.get_spectrogram(audio, sr)
        
        return {
            'low_level_features': low_level,
            'high_level_features': high_level,
            'spectrogram': spectrogram,
            'duration': len(audio) / sr
        }

# Użycie
extractor = AudioFeatureExtractor(sr=16000)

# Przetwórz wszystkie audio pliki
df = pd.read_csv('daic_labels.csv')
audio_features = []

for idx, row in df.iterrows():
    audio_path = row['audio_path']
    try:
        features = extractor.process_audio(audio_path)
        
        # Flatten low-level features
        flat_features = {}
        for key, val in features['low_level_features'].items():
            if isinstance(val, np.ndarray):
                for i, v in enumerate(val):
                    flat_features[f'{key}_{i}'] = v
            else:
                flat_features[key] = val
        
        # Dodaj high-level
        for i, v in enumerate(features['high_level_features']):
            flat_features[f'opensmile_feat_{i}'] = v
        
        flat_features['session_id'] = row['session_id']
        audio_features.append(flat_features)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")

df_audio_features = pd.DataFrame(audio_features)
df_audio_features.to_csv('audio_features.csv', index=False)
print(f"Extracted audio features for {len(df_audio_features)} sessions")
```

#### 3.3 Feature Extraction - Text Modalność

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoTokenizer, AutoModel
import torch

class TextFeatureExtractor:
    """Ekstrakcja cech tekstowych z transkrypcji"""
    
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def extract_linguistic_features(self, text):
        """Ekstrakcja cech lingwistycznych"""
        if not text or len(text.strip()) == 0:
            return None
        
        words = text.split()
        sentences = text.split('.')
        
        features = {
            'word_count': len(words),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
            'unique_words': len(set(words)),
            'vocabulary_richness': len(set(words)) / max(len(words), 1),
            'text_length': len(text),
            'avg_sentence_length': len(words) / max(len([s for s in sentences if s.strip()]), 1),
        }
        
        # Sentiment-like markers (prosty heurystyk)
        negative_words = ['nie', 'nigdy', 'żaden', 'nikt', 'nic', 'smutny', 'zły']
        features['negative_word_ratio'] = sum(1 for w in words if w.lower() in negative_words) / max(len(words), 1)
        
        # Pronoun usage patterns (mogą wskazywać na depresję)
        first_person = sum(1 for w in words if w.lower() in ['ja', 'mnie', 'mi', 'mój', 'moja'])
        features['first_person_ratio'] = first_person / max(len(words), 1)
        
        return features
    
    def get_embedding(self, text):
        """Zwróć sentence embedding z transformera"""
        if not text or len(text.strip()) == 0:
            return np.zeros(384)  # Default embedding size dla MiniLM
        
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # Mean pooling
            embeddings = outputs.last_hidden_state.mean(dim=1)
        
        return embeddings[0].numpy()
    
    def process_text(self, text):
        """Pipeline: tekst → cechy lingwistyczne + embedding"""
        if not text:
            return None
        
        linguistic = self.extract_linguistic_features(text)
        embedding = self.get_embedding(text)
        
        return {
            'linguistic_features': linguistic,
            'embedding': embedding
        }

# Użycie
text_extractor = TextFeatureExtractor()

df = pd.read_csv('daic_labels.csv')
text_features = []

for idx, row in df.iterrows():
    text = row['text']
    if pd.isna(text) or not text:
        continue
    
    try:
        features = text_extractor.process_text(text)
        
        flat_features = {'session_id': row['session_id']}
        
        # Linguistic features
        if features['linguistic_features']:
            flat_features.update(features['linguistic_features'])
        
        # Embedding
        for i, val in enumerate(features['embedding']):
            flat_features[f'text_embedding_{i}'] = val
        
        text_features.append(flat_features)
    except Exception as e:
        print(f"Error processing text for session {row['session_id']}: {e}")

df_text_features = pd.DataFrame(text_features)
df_text_features.to_csv('text_features.csv', index=False)
print(f"Extracted text features for {len(df_text_features)} sessions")
```

#### 3.4 Fuzja Modalności

```python
# Połącz audio + tekst + labels
df_audio = pd.read_csv('audio_features.csv')
df_text = pd.read_csv('text_features.csv')
df_labels_final = df.groupby('session_id').agg({
    'depression': 'first',
    'phq8_score': 'first'
}).reset_index()

# Merge
df_fused = df_labels_final.merge(df_audio, on='session_id', how='inner')
df_fused = df_fused.merge(df_text, on='session_id', how='left')

# Obsłuż braki w tekstowych (dla sesji bez transkrypcji)
text_cols = df_text.columns.tolist()
text_cols.remove('session_id')
for col in text_cols:
    df_fused[col].fillna(0, inplace=True)

print(f"Final fused dataset: {len(df_fused)} samples")
print(f"Features: {len(df_fused.columns) - 2} (depression + phq8_score + session_id)")
print(f"Class distribution:\n{df_fused['depression'].value_counts()}")

# Zapisz w formacie HDF5 dla szybszego ładowania
df_fused.to_hdf('daic_fused_features.h5', key='data', mode='w')
```

### 4. Train/Val/Test Split - Standardowy dla DAIC-WOZ

```python
from sklearn.model_selection import train_test_split

df = pd.read_hdf('daic_fused_features.h5', key='data')

# Stratified split - zachowaj balans klasy
# DAIC-WOZ standardowo: train=107, val=35, test=47
train_ids, test_ids = train_test_split(
    df['session_id'],
    test_size=47/len(df),
    stratify=df['depression'],
    random_state=42
)

train_val_ids = df[df['session_id'].isin(train_ids)]['session_id']
train_ids, val_ids = train_test_split(
    train_val_ids,
    test_size=35/(107+35),
    stratify=df[df['session_id'].isin(train_val_ids)]['depression'],
    random_state=42
)

# Normalizacja (fit na train set!)
from sklearn.preprocessing import StandardScaler

X_train = df[df['session_id'].isin(train_ids)].drop(['session_id', 'depression', 'phq8_score'], axis=1)
X_val = df[df['session_id'].isin(val_ids)].drop(['session_id', 'depression', 'phq8_score'], axis=1)
X_test = df[df['session_id'].isin(test_ids)].drop(['session_id', 'depression', 'phq8_score'], axis=1)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Zapisz
np.save('X_train.npy', X_train_scaled)
np.save('X_val.npy', X_val_scaled)
np.save('X_test.npy', X_test_scaled)

y_train = df[df['session_id'].isin(train_ids)]['depression'].values
y_val = df[df['session_id'].isin(val_ids)]['depression'].values
y_test = df[df['session_id'].isin(test_ids)]['depression'].values

np.save('y_train.npy', y_train)
np.save('y_val.npy', y_val)
np.save('y_test.npy', y_test)

print(f"Train: {len(X_train)} samples ({y_train.sum()} positive)")
print(f"Val: {len(X_val)} samples ({y_val.sum()} positive)")
print(f"Test: {len(X_test)} samples ({y_test.sum()} positive)")
```

### 5. Spectrogram Preprocessing dla CNN

```python
import cv2

class SpectrogramPreprocessor:
    """Przygotowanie spektrogramów dla CNN (dla osobnej sieci audio)"""
    
    def __init__(self, target_size=(128, 400)):
        self.target_size = target_size
    
    def process_spectrogram(self, spectrogram):
        """Resize i normalizacja spektrogramu"""
        # spectrogram shape: (n_mels, time_frames)
        
        # Resize do stałego rozmiaru
        spec_resized = cv2.resize(spectrogram, self.target_size)
        
        # Normalizacja [0, 1]
        spec_norm = (spec_resized - spec_resized.min()) / (spec_resized.max() - spec_resized.min() + 1e-8)
        
        # Dodaj channel dimension dla CNN: (height, width, 1)
        spec_final = np.expand_dims(spec_norm, axis=-1)
        
        return spec_final
    
    def save_spectrograms_to_folder(self, audio_features_dict, output_dir):
        """Zapisz wszystkie spektrogramy"""
        Path(output_dir).mkdir(exist_ok=True)
        
        for session_id, features in audio_features_dict.items():
            spec = features['spectrogram']
            spec_processed = self.process_spectrogram(spec)
            
            # Zapisz jako .npy
            np.save(
                f"{output_dir}/spec_{session_id}.npy",
                spec_processed
            )

# Użycie - przetwórz spektrogramy wszystkich sesji
# (Audio features zawierają raw spectrogramy)
spec_processor = SpectrogramPreprocessor()
# spec_processor.save_spectrograms_to_folder(all_spectrograms, 'data/spectrograms')
```

### 6. Sprawdzenie Jakości Preprocessingu

```python
# Zaladuj preprocessowany dataset
X_train = np.load('X_train.npy')
X_val = np.load('X_val.npy')
X_test = np.load('X_test.npy')
y_train = np.load('y_train.npy')
y_val = np.load('y_val.npy')
y_test = np.load('y_test.npy')

print("=== Dataset Quality Report ===")
print(f"Train set: {X_train.shape} - {np.mean(y_train):.1%} positive")
print(f"Val set:   {X_val.shape} - {np.mean(y_val):.1%} positive")
print(f"Test set:  {X_test.shape} - {np.mean(y_test):.1%} positive")

# Sprawdź brakujące wartości
print(f"\nMissing values in train: {np.isnan(X_train).sum()}")
print(f"Missing values in val: {np.isnan(X_val).sum()}")
print(f"Missing values in test: {np.isnan(X_test).sum()}")

# Statystyka features
print(f"\nFeature statistics (train set):")
print(f"Mean: {np.mean(X_train, axis=0)[:5]}...")
print(f"Std: {np.std(X_train, axis=0)[:5]}...")
print(f"Min: {np.min(X_train, axis=0)[:5]}...")
print(f"Max: {np.max(X_train, axis=0)[:5]}...")
```

---

## 7. Optymalizacja dla Twojego Sprzętu

Masz **RTX 5060 8GB** + **Intel Core Ultra 7 (NPU 13 TOPS)** - to doskonała konfiguracja!

### Memory & Batch Size Tuning

```python
# RTX 5060 8GB - maksymalny batch size do feature extraction
BATCH_SIZE = 64
NUM_WORKERS = 4  # Twojego procesora (8 P-core + 12 E-core)

# Dla preprocessingu - można równolegle
from concurrent.futures import ThreadPoolExecutor

# Przetwarzaj sesje równolegle (unika GIL dla I/O)
with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
    futures = [
        executor.submit(extractor.process_audio, row['audio_path'])
        for _, row in df.iterrows()
    ]
    results = [f.result() for f in futures]
```

### PyTorch Optimized Settings

```python
# W train loop
torch.backends.cudnn.benchmark = True  # Auto-optimize convolutions
torch.backends.cudnn.enabled = True

# Mixed precision - zmniejsza memory, przyspieszą training
from torch.cuda.amp import autocast

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

### Inference Optymalizacja

```python
# Kwantyzacja modelu do FP16 - szybsze inference
model = model.half()  # Zmień do float16
inputs = inputs.half()

# Lub ONNX export
import torch.onnx
torch.onnx.export(model, dummy_input, "model.onnx")
# Potem inference z ONNX Runtime - szybsze, mniejsze RAM
```

---

## 8. Gotowe do Modelowania!

Po wykonaniu wszystkich kroków masz:

```
data/
├── X_train.npy         (107 samples, audio+text features)
├── X_val.npy           (35 samples)
├── X_test.npy          (47 samples)
├── y_train.npy         (labels)
├── y_val.npy
├── y_test.npy
├── spectrograms/       (dla CNN audio sieci)
│   ├── spec_401.npy
│   ├── spec_402.npy
│   └── ...
├── daic_fused_features.h5
├── audio_features.csv
├── text_features.csv
└── daic_labels.csv

Gotowe do:
1. Fusion strategy (połączenie audio + text features)
2. CNN training na spektrogramach
3. Transformer fine-tuning na tekstach
4. Ewaluacja na test set
```

---

## Dodatki

### Debugging Preprocessing
```python
# Sprawdź konkretną sesję
session_id = 401
audio_path = f'DAIC-WOZ/Session_Data/{session_id}/{session_id}_AUDIO.wav'

# Audio
audio, sr = librosa.load(audio_path, sr=16000)
print(f"Audio duration: {len(audio)/sr:.1f}s")
print(f"Audio range: [{audio.min():.3f}, {audio.max():.3f}]")

# Tekst
print(f"Text length: {len(df[df['session_id']==session_id]['text'].values[0])}")
```

### Reproducibility
```python
import random

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
```

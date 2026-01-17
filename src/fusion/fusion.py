"""Feature fusion module for combining audio and text features"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional

from ..utils.config import Config


def fuse_features(
    audio_features_path: Optional[Path] = None,
    text_features_path: Optional[Path] = None,
    labels_path: Optional[Path] = None,
    output_path: Optional[Path] = None
) -> pd.DataFrame:
    """Połącz audio + tekst + labels w jeden dataset
    
    Args:
        audio_features_path: Ścieżka do CSV z cechami audio
        text_features_path: Ścieżka do CSV z cechami tekstowymi
        labels_path: Ścieżka do CSV z labels (opcjonalne, jeśli już w audio/text)
        output_path: Ścieżka do zapisu połączonego datasetu (HDF5)
        
    Returns:
        DataFrame z połączonymi features
    """
    audio_features_path = audio_features_path or Config.AUDIO_FEATURES_CSV
    text_features_path = text_features_path or Config.TEXT_FEATURES_CSV
    labels_path = labels_path or Config.LABELS_CSV
    output_path = output_path or Config.FUSED_FEATURES_H5
    
    # Załaduj audio features
    if not audio_features_path.exists():
        raise FileNotFoundError(f"Audio features not found: {audio_features_path}")
    
    df_audio = pd.read_csv(audio_features_path)
    print(f"Loaded audio features: {len(df_audio)} sessions, {len(df_audio.columns)-1} features")
    
    # Załaduj text features (opcjonalne)
    df_text = None
    if text_features_path.exists():
        df_text = pd.read_csv(text_features_path)
        print(f"Loaded text features: {len(df_text)} sessions, {len(df_text.columns)-1} features")
    else:
        print("Text features not found, continuing without text")
    
    # Załaduj labels
    df_labels = None
    if labels_path.exists():
        df_labels = pd.read_csv(labels_path)
        # Użyj tylko potrzebnych kolumn
        label_cols = ['session_id', 'depression', 'phq8_score']
        available_cols = [col for col in label_cols if col in df_labels.columns]
        df_labels = df_labels[available_cols]
        print(f"Loaded labels: {len(df_labels)} sessions")
    else:
        print("Labels not found, checking if in feature files...")
    
    # Merge audio + text
    df_fused = df_audio.copy()
    
    if df_text is not None:
        # Merge text features (left join - niektóre sesje mogą nie mieć tekstu)
        df_fused = df_fused.merge(df_text, on='session_id', how='left', suffixes=('', '_text'))
        
        # Obsłuż braki w tekstowych (dla sesji bez transkrypcji)
        text_cols = [col for col in df_text.columns if col != 'session_id']
        for col in text_cols:
            if col in df_fused.columns:
                df_fused[col].fillna(0, inplace=True)
    
    # Merge labels jeśli osobny plik
    if df_labels is not None:
        # Sprawdź czy labels już są w df_fused
        if 'depression' not in df_fused.columns:
            df_fused = df_fused.merge(df_labels, on='session_id', how='inner')
        else:
            # Aktualizuj tylko jeśli brakuje
            for col in df_labels.columns:
                if col != 'session_id' and col not in df_fused.columns:
                    df_fused = df_fused.merge(df_labels[['session_id', col]], 
                                            on='session_id', how='left')
    
    # Upewnij się że mamy session_id i depression
    if 'session_id' not in df_fused.columns:
        raise ValueError("session_id column missing in fused dataset")
    
    if 'depression' not in df_fused.columns:
        raise ValueError("depression label missing in fused dataset")
    
    # Usuń duplikaty kolumn (jeśli były)
    df_fused = df_fused.loc[:, ~df_fused.columns.duplicated()]
    
    # Sortuj po session_id
    df_fused = df_fused.sort_values('session_id').reset_index(drop=True)
    
    print(f"\nFinal fused dataset: {len(df_fused)} samples")
    print(f"Total features: {len(df_fused.columns) - 2} (excluding session_id and depression)")
    
    if 'depression' in df_fused.columns:
        print(f"Class distribution:")
        print(df_fused['depression'].value_counts().to_dict())
        print(f"Positive class: {df_fused['depression'].sum()} ({df_fused['depression'].mean():.1%})")
    
    # Sprawdź brakujące wartości
    missing = df_fused.isnull().sum()
    if missing.sum() > 0:
        print(f"\nWarning: Missing values found:")
        print(missing[missing > 0])
        # Wypełnij zerami
        df_fused.fillna(0, inplace=True)
        print("Filled missing values with 0")
    
    # Zapisz jako CSV
    output_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path = output_path.with_suffix('.csv')
    df_fused.to_csv(csv_path, index=False)
    print(f"\nFused features saved to {csv_path}")
    
    # Spróbuj zapisać do HDF5 dla szybszego ładowania (opcjonalne)
    try:
        df_fused.to_hdf(str(output_path), key='data', mode='w', format='table')
        print(f"Also saved as HDF5: {output_path}")
    except ImportError:
        print("Note: pytables not installed. Skipping HDF5 format (CSV is sufficient).")
    except Exception as e:
        print(f"Warning: Could not save HDF5 format: {e}")
    
    return df_fused

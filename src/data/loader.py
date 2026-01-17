"""Data loading and ZIP extraction utilities"""

import zipfile
import pandas as pd
from pathlib import Path
from typing import List, Optional, Dict
from tqdm import tqdm

from ..utils.config import Config
from .extractor import extract_labels_from_xml, load_transcription
import csv


class DataLoader:
    """Klasa do ładowania i ekstrakcji danych z DAIC-WOZ dataset"""
    
    def __init__(self, data_root: Optional[Path] = None):
        """Initialize DataLoader
        
        Args:
            data_root: Ścieżka do folderu z plikami ZIP. Jeśli None, używa Config.DATA_ROOT
        """
        self.data_root = data_root or Config.DATA_ROOT
        self.zip_files = list(self.data_root.glob("*.zip"))
    
    def extract_zip(self, zip_path: Path, target_dir: Path, session_id: Optional[int] = None):
        """Rozpakuj pojedynczy plik ZIP
        
        Args:
            zip_path: Ścieżka do pliku ZIP
            target_dir: Katalog docelowy
            session_id: Opcjonalny ID sesji (jeśli znany z nazwy pliku)
        """
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(target_dir)
        except Exception as e:
            print(f"Error extracting {zip_path}: {e}")
    
    def extract_all_zips(self, target_dir: Optional[Path] = None, force: bool = False):
        """Rozpakuj wszystkie pliki ZIP z datasetu
        
        Args:
            target_dir: Katalog docelowy (domyślnie Config.RAW_DATA_DIR)
            force: Jeśli True, rozpakuje ponownie nawet jeśli już istnieją
        """
        target_dir = target_dir or Config.RAW_DATA_DIR
        target_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"Found {len(self.zip_files)} ZIP files")
        
        for zip_path in tqdm(self.zip_files, desc="Extracting ZIP files"):
            # Sprawdź czy już rozpakowane
            if not force:
                # Sprawdź czy Session_Data już istnieje
                if (target_dir / "Session_Data").exists():
                    # Sprawdź czy wszystkie sesje są już rozpakowane
                    session_dirs = [d for d in (target_dir / "Session_Data").iterdir() 
                                  if d.is_dir() and d.name.isdigit()]
                    if len(session_dirs) > 100:  # Przybliżona liczba sesji
                        print("Data already extracted. Use force=True to re-extract.")
                        return
            
            self.extract_zip(zip_path, target_dir)
        
        print(f"Extraction complete. Data in {target_dir}")
    
    def load_metadata(self) -> pd.DataFrame:
        """Załaduj wszystkie metadane z sesji i utwórz DataFrame
        
        Returns:
            DataFrame z kolumnami: session_id, audio_path, text, phq8_score, depression, qids_score
        """
        Config.create_directories()
        
        session_data_dir = Config.SESSION_DATA_DIR
        raw_data_dir = Config.RAW_DATA_DIR
        transcriptions_dir = Config.TRANSCRIPTIONS_DIR
        
        metadata = []
        
        # Sprawdź czy mamy strukturę z folderami Session_Data/300/ czy pliki bezpośrednio w raw/
        if session_data_dir.exists():
            # Struktura z folderami: Session_Data/300/300_AUDIO.wav
            session_dirs = [d for d in session_data_dir.iterdir() 
                           if d.is_dir() and d.name.isdigit()]
            
            # Jeśli Session_Data istnieje ale jest pusty, użyj płaskiej struktury
            if len(session_dirs) == 0:
                print("Session_Data directory is empty, searching in raw/ directly")
                search_dir = raw_data_dir
                use_subdirs = False
                
                # Znajdź wszystkie pliki audio
                audio_files = list(raw_data_dir.glob("*_AUDIO.wav"))
                session_ids = set()
                for audio_file in audio_files:
                    try:
                        session_id = int(audio_file.stem.replace("_AUDIO", ""))
                        session_ids.add(session_id)
                    except:
                        continue
                session_dirs = sorted(session_ids)
                print(f"Found {len(session_dirs)} sessions from audio files")
            else:
                print(f"Found {len(session_dirs)} session directories in Session_Data")
                search_dir = session_data_dir
                use_subdirs = True
        else:
            # Struktura płaska: raw/300_AUDIO.wav
            print("Session_Data directory not found, searching in raw/ directly")
            search_dir = raw_data_dir
            use_subdirs = False
            
            # Znajdź wszystkie pliki audio
            audio_files = list(raw_data_dir.glob("*_AUDIO.wav"))
            session_ids = set()
            for audio_file in audio_files:
                try:
                    session_id = int(audio_file.stem.replace("_AUDIO", ""))
                    session_ids.add(session_id)
                except:
                    continue
            session_dirs = sorted(session_ids)
            print(f"Found {len(session_dirs)} sessions from audio files")
        
        for session_item in tqdm(session_dirs, desc="Loading metadata"):
            try:
                if use_subdirs:
                    session_id = int(session_item.name)
                    session_dir = session_item
                    xml_path = session_dir / f"{session_id}.xml"
                    audio_path = session_dir / f"{session_id}_AUDIO.wav"
                else:
                    session_id = session_item
                    xml_path = raw_data_dir / f"{session_id}.xml"
                    audio_path = raw_data_dir / f"{session_id}_AUDIO.wav"
                
                # Pomiń znane problemy
                if session_id in Config.PROBLEMATIC_SESSIONS:
                    continue
                
                if not audio_path.exists():
                    continue
                
                # Ekstrakcja labels - spróbuj z XML, jeśli nie ma to z CSV
                labels = None
                if xml_path.exists():
                    try:
                        labels = extract_labels_from_xml(xml_path)
                    except:
                        pass
                
                # Jeśli nie ma XML, użyj CSV z metadanymi
                if labels is None:
                    labels = self._load_labels_from_csv(session_id)
                    if labels is None:
                        continue  # Pomiń jeśli nie ma metadanych
                
                # Ekstrakcja tekstu (opcjonalne)
                # Sprawdź w Transcriptions/ lub bezpośrednio w raw/
                trans_path = transcriptions_dir / f"Participant_{session_id}.xml"
                text = None
                
                if trans_path.exists():
                    text = load_transcription(trans_path)
                else:
                    # Sprawdź CSV transcript w raw/
                    trans_csv = raw_data_dir / f"{session_id}_TRANSCRIPT.csv"
                    if trans_csv.exists():
                        try:
                            # CSV transcript - zwykle format: start_time,stop_time,speaker,value
                            # Spróbuj z różnymi separatorami
                            trans_df = None
                            for sep in ['\t', ',']:
                                try:
                                    trans_df = pd.read_csv(trans_csv, sep=sep)
                                    if len(trans_df.columns) > 1:  # Prawidłowy separator
                                        break
                                except:
                                    continue
                            
                            if trans_df is not None and 'speaker' in trans_df.columns and 'value' in trans_df.columns:
                                # Filtruj tylko wypowiedzi uczestnika
                                participant_texts = trans_df[
                                    trans_df['speaker'].str.contains('Participant', case=False, na=False)
                                ]['value'].dropna()
                                text = ' '.join(participant_texts.astype(str).tolist()) if len(participant_texts) > 0 else None
                        except Exception as e:
                            pass
                
                metadata.append({
                    'session_id': session_id,
                    'audio_path': str(audio_path),
                    'text': text,
                    **labels
                })
            except Exception as e:
                session_name = str(session_item) if use_subdirs else str(session_item)
                print(f"Error processing session {session_name}: {e}")
                continue
        
        df = pd.DataFrame(metadata)
        
        if len(df) > 0:
            print(f"\nLoaded {len(df)} sessions")
            print(f"Positive class (depression): {df['depression'].sum()} ({df['depression'].mean():.1%})")
            print(f"Sessions with transcriptions: {df['text'].notna().sum()}")
        else:
            print("Warning: No metadata loaded. Check if data is extracted correctly.")
        
        return df
    
    def _load_labels_from_csv(self, session_id: int) -> Optional[Dict]:
        """Załaduj labels z plików CSV split (train/dev/test)"""
        csv_files = [
            Config.DATA_ROOT / "train_split_Depression_AVEC2017.csv",
            Config.DATA_ROOT / "dev_split_Depression_AVEC2017.csv",
            Config.DATA_ROOT / "test_split_Depression_AVEC2017.csv",
        ]
        
        for csv_file in csv_files:
            if not csv_file.exists():
                continue
            
            try:
                df = pd.read_csv(csv_file)
                
                # Różne formaty CSV - sprawdź kolumny
                id_col = None
                if 'Participant_ID' in df.columns:
                    id_col = 'Participant_ID'
                elif 'participant_ID' in df.columns:
                    id_col = 'participant_ID'
                
                if id_col is None:
                    continue
                
                row = df[df[id_col] == session_id]
                if len(row) > 0:
                    row = row.iloc[0]
                    
                    # PHQ8_Score
                    if 'PHQ8_Score' in df.columns:
                        phq8_score = int(row['PHQ8_Score']) if pd.notna(row['PHQ8_Score']) else 0
                    else:
                        phq8_score = 0
                    
                    # PHQ8_Binary (jeśli dostępne, użyj tego zamiast obliczać)
                    if 'PHQ8_Binary' in df.columns:
                        depression = int(row['PHQ8_Binary']) if pd.notna(row['PHQ8_Binary']) else int(phq8_score >= 10)
                    else:
                        depression = int(phq8_score >= 10)
                    
                    return {
                        'phq8_score': phq8_score,
                        'depression': depression,
                        'qids_score': None  # Nie ma w CSV
                    }
            except Exception as e:
                continue
        
        return None
    
    def save_metadata(self, output_path: Optional[Path] = None):
        """Załaduj i zapisz metadane do CSV
        
        Args:
            output_path: Ścieżka do pliku wyjściowego (domyślnie Config.LABELS_CSV)
        """
        output_path = output_path or Config.LABELS_CSV
        df = self.load_metadata()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Metadata saved to {output_path}")

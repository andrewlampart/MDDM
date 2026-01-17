"""Configuration module for paths and parameters"""

from pathlib import Path
import os

class Config:
    """Configuration class for dataset paths and parameters"""
    
    # Base paths
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_ROOT = PROJECT_ROOT / "DAIC-WOZ-Dataset"
    RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
    PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
    MODELS_DIR = PROJECT_ROOT / "data" / "models"
    
    # Dataset paths
    SESSION_DATA_DIR = RAW_DATA_DIR / "Session_Data"
    TRANSCRIPTIONS_DIR = RAW_DATA_DIR / "Transcriptions"
    
    # Output files
    LABELS_CSV = PROCESSED_DATA_DIR / "daic_labels.csv"
    AUDIO_FEATURES_CSV = PROCESSED_DATA_DIR / "audio_features.csv"
    TEXT_FEATURES_CSV = PROCESSED_DATA_DIR / "text_features.csv"
    FUSED_FEATURES_H5 = PROCESSED_DATA_DIR / "daic_fused_features.h5"
    
    # Train/val/test splits
    X_TRAIN_NPY = PROCESSED_DATA_DIR / "X_train.npy"
    X_VAL_NPY = PROCESSED_DATA_DIR / "X_val.npy"
    X_TEST_NPY = PROCESSED_DATA_DIR / "X_test.npy"
    Y_TRAIN_NPY = PROCESSED_DATA_DIR / "y_train.npy"
    Y_VAL_NPY = PROCESSED_DATA_DIR / "y_val.npy"
    Y_TEST_NPY = PROCESSED_DATA_DIR / "y_test.npy"
    
    # Model files
    MODEL_PKL = MODELS_DIR / "baseline_model.pkl"
    SCALER_PKL = MODELS_DIR / "scaler.pkl"
    
    # Audio parameters
    SAMPLE_RATE = 16000
    N_MELS = 128
    N_MFCC = 13
    
    # Text parameters
    TEXT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
    MAX_TEXT_LENGTH = 512
    
    # Training parameters
    RANDOM_SEED = 42
    TRAIN_SIZE = 107
    VAL_SIZE = 35
    TEST_SIZE = 47
    
    # Feature extraction flags
    USE_OPENSMILE = True  # Set to False if openSMILE not installed
    USE_TEXT = True
    
    # Known problematic sessions (skip these)
    PROBLEMATIC_SESSIONS = [451, 458, 480, 409]
    
    # Processing parameters
    BATCH_SIZE = 64
    NUM_WORKERS = 4
    
    @classmethod
    def create_directories(cls):
        """Create all necessary directories"""
        directories = [
            cls.RAW_DATA_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.MODELS_DIR,
            cls.SESSION_DATA_DIR,
            cls.TRANSCRIPTIONS_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    @classmethod
    def get_session_audio_path(cls, session_id: int) -> Path:
        """Get path to audio file for a session"""
        return cls.SESSION_DATA_DIR / str(session_id) / f"{session_id}_AUDIO.wav"
    
    @classmethod
    def get_session_xml_path(cls, session_id: int) -> Path:
        """Get path to XML metadata file for a session"""
        return cls.SESSION_DATA_DIR / str(session_id) / f"{session_id}.xml"
    
    @classmethod
    def get_transcription_path(cls, session_id: int) -> Path:
        """Get path to transcription XML file for a session"""
        return cls.TRANSCRIPTIONS_DIR / f"Participant_{session_id}.xml"

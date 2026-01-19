"""
Central Configuration for MDDM PhD Pipeline
Multimodal Depression Detection Model

This config replaces src/utils/config.py as the main configuration file.
"""

import random
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
import numpy as np

# Try to import torch for GPU detection
try:
    import torch
    TORCH_AVAILABLE = True
    CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    CUDA_AVAILABLE = False


@dataclass
class Config:
    """Central configuration for all pipeline components"""
    
    # ===================
    # Paths
    # ===================
    PROJECT_ROOT: Path = field(default_factory=lambda: Path(__file__).parent)
    
    # Data directories
    DATA_ROOT: Path = field(default_factory=lambda: Path(__file__).parent / "DAIC-WOZ-Dataset")
    RAW_DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "raw")
    PROCESSED_DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed")
    
    # Output directories
    MODELS_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "models")
    RESULTS_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "results")
    LOGS_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "logs")
    
    # Session data
    SESSION_DATA_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "raw" / "Session_Data")
    TRANSCRIPTIONS_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "raw" / "Transcriptions")
    SPECTROGRAMS_DIR: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed" / "spectrograms")
    
    # Output files
    LABELS_CSV: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed" / "daic_labels.csv")
    AUDIO_FEATURES_CSV: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed" / "audio_features.csv")
    TEXT_FEATURES_CSV: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed" / "text_features.csv")
    FUSED_FEATURES_CSV: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed" / "daic_fused_features.csv")
    INSTANCES_CSV: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed" / "instances.csv")
    
    # Train/val/test splits
    X_TRAIN_NPY: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed" / "X_train.npy")
    X_VAL_NPY: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed" / "X_val.npy")
    X_TEST_NPY: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed" / "X_test.npy")
    Y_TRAIN_NPY: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed" / "y_train.npy")
    Y_VAL_NPY: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed" / "y_val.npy")
    Y_TEST_NPY: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "processed" / "y_test.npy")
    
    # Model files
    MODEL_PKL: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "models" / "baseline_model.pkl")
    SCALER_PKL: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "models" / "scaler.pkl")
    XGBOOST_MODEL: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "models" / "xgboost_baseline.pkl")
    MIL_TEXT_MODEL: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "models" / "mil_text_best.pt")
    AUDIO_CNN_MODEL: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "models" / "audio_cnn_best.pt")
    MULTIMODAL_MODEL: Path = field(default_factory=lambda: Path(__file__).parent / "data" / "models" / "multimodal_best.pt")
    
    # ===================
    # Dataset parameters
    # ===================
    RANDOM_SEED: int = 42
    TRAIN_SIZE: int = 107  # DAIC-WOZ standard
    VAL_SIZE: int = 35
    TEST_SIZE: int = 47
    
    # Known problematic sessions (skip these)
    PROBLEMATIC_SESSIONS: List[int] = field(default_factory=lambda: [451, 458, 480, 409])
    
    # ===================
    # Audio parameters
    # ===================
    SAMPLE_RATE: int = 16000
    N_MELS: int = 128
    N_MFCC: int = 13
    SPECTROGRAM_LENGTH: int = 400  # Time frames for CNN input
    USE_OPENSMILE: bool = True
    
    # ===================
    # Text parameters
    # ===================
    TEXT_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"
    ROBERTA_MODEL_NAME: str = "roberta-base"
    MAX_TEXT_LENGTH: int = 512
    
    # MIL parameters
    MIL_ALPHA: float = 0.6  # Threshold for instance score
    MIL_BETA: float = 0.5   # Threshold for aggregated score
    
    # ===================
    # Training parameters
    # ===================
    # General
    BATCH_SIZE: int = 32
    MIL_BATCH_SIZE: int = 8  # Smaller for MIL due to variable bag sizes
    CNN_BATCH_SIZE: int = 16
    
    EPOCHS: int = 100
    LEARNING_RATE: float = 1e-3
    MIL_LEARNING_RATE: float = 2e-4
    
    EARLY_STOPPING_PATIENCE: int = 15
    
    # Regularization
    DROPOUT_RATE: float = 0.3
    WEIGHT_DECAY: float = 1e-4
    
    # Class weights
    COMPUTE_CLASS_WEIGHTS: bool = True
    
    # Feature selection
    N_FEATURES_SELECT: int = 50
    
    # ===================
    # GPU parameters - RTX 5060 8GB Blackwell
    # ===================
    USE_GPU: bool = CUDA_AVAILABLE
    DEVICE: str = "cuda" if CUDA_AVAILABLE else "cpu"
    USE_MIXED_PRECISION: bool = True
    USE_BF16: bool = True  # Blackwell preferuje bf16 nad fp16
    GPU_MEMORY_FRACTION: float = 0.8  # Zostawić 20% VRAM wolne dla systemu
    
    # ===================
    # Windows-specific settings
    # ===================
    NUM_WORKERS: int = 0  # Windows wymaga 0 dla DataLoader (multiprocessing issues)
    PIN_MEMORY: bool = False  # Windows nie obsługuje dobrze pin_memory
    
    # ===================
    # Audio processing limits (dla stabilności GPU)
    # ===================
    MAX_AUDIO_LENGTH_SEC: int = 60  # Chunk długiego audio na 60s fragmenty
    AUDIO_CHUNK_OVERLAP_SEC: int = 5  # Overlap między chunkami
    AUDIO_BATCH_SIZE: int = 4  # Umiarkowany batch (opcja A)
    
    # ===================
    # Text processing limits
    # ===================
    TEXT_BATCH_SIZE: int = 32  # Zwiększony batch dla BERT (opcja A)
    
    # ===================
    # Evaluation parameters
    # ===================
    METRICS: List[str] = field(default_factory=lambda: [
        'accuracy', 'precision', 'recall', 'f1', 
        'auroc', 'auprc', 'mcc', 'sensitivity', 'specificity'
    ])
    BOOTSTRAP_CI_RESAMPLES: int = 1000
    CONFIDENCE_LEVEL: float = 0.95
    
    # MC Dropout
    MC_DROPOUT_SAMPLES: int = 10
    
    # ===================
    # XGBoost parameters
    # ===================
    XGBOOST_N_ESTIMATORS: int = 500
    XGBOOST_MAX_DEPTH: int = 6
    XGBOOST_LEARNING_RATE: float = 0.05
    XGBOOST_SUBSAMPLE: float = 0.8
    XGBOOST_COLSAMPLE_BYTREE: float = 0.8
    XGBOOST_EARLY_STOPPING: int = 30
    
    # ===================
    # Optuna parameters
    # ===================
    OPTUNA_N_TRIALS: int = 50
    OPTUNA_TIMEOUT: int = 3600  # 1 hour max
    OPTUNA_METRIC: str = "f1"
    
    # ===================
    # Calibration parameters (NEW)
    # ===================
    # Temperature Scaling - fixes AUROC issues in neural networks
    USE_TEMPERATURE_SCALING: bool = True
    TEMPERATURE_BOUNDS: tuple = (0.1, 10.0)  # Search range for temperature
    
    # Threshold Optimization - finds optimal classification threshold
    USE_THRESHOLD_OPTIMIZATION: bool = True
    OPTIMAL_THRESHOLD_METRIC: str = 'f1'  # Options: 'f1', 'balanced', 'youden'
    THRESHOLD_SEARCH_RANGE: tuple = (0.1, 0.9)  # Search range for threshold
    THRESHOLD_SEARCH_STEP: float = 0.01
    
    # ===================
    # Ensemble parameters (NEW)
    # ===================
    USE_ENSEMBLE_VOTING: bool = True
    ENSEMBLE_N_MODELS: int = 5  # Number of models in ensemble (same as n_splits)
    ENSEMBLE_METHOD: str = 'mean'  # Options: 'mean', 'median', 'vote'
    
    # ===================
    # Feature diagnostics (NEW)
    # ===================
    RUN_FEATURE_DIAGNOSTICS: bool = True
    WARN_ON_HIGH_NAN_RATIO: float = 0.01  # Warn if >1% NaN
    MIN_FEATURE_CORRELATION: float = 0.05  # Features below this may be noise
    
    # ===================
    # Attention Fusion Model parameters (NEW)
    # ===================
    ATTENTION_HIDDEN_DIM: int = 128
    ATTENTION_NUM_HEADS: int = 4
    ATTENTION_NUM_LAYERS: int = 1
    ATTENTION_DROPOUT: float = 0.4
    ATTENTION_USE_CROSS: bool = True
    ATTENTION_USE_GATED: bool = True
    ATTENTION_LEARNING_RATE: float = 1e-4
    ATTENTION_WEIGHT_DECAY: float = 0.01
    ATTENTION_EPOCHS: int = 50
    ATTENTION_PATIENCE: int = 10
    
    def __post_init__(self):
        """Create all necessary directories"""
        directories = [
            self.RAW_DATA_DIR,
            self.PROCESSED_DATA_DIR,
            self.MODELS_DIR,
            self.RESULTS_DIR,
            self.LOGS_DIR,
            self.SESSION_DATA_DIR,
            self.TRANSCRIPTIONS_DIR,
            self.SPECTROGRAMS_DIR,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def set_seed(self):
        """Set random seeds for reproducibility"""
        random.seed(self.RANDOM_SEED)
        np.random.seed(self.RANDOM_SEED)
        
        if TORCH_AVAILABLE:
            import torch
            torch.manual_seed(self.RANDOM_SEED)
            if CUDA_AVAILABLE:
                torch.cuda.manual_seed_all(self.RANDOM_SEED)
                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False
    
    def get_device(self):
        """Get torch device"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available")
        import torch
        return torch.device(self.DEVICE)
    
    def print_gpu_info(self):
        """Print GPU information"""
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            import torch
            print(f"PyTorch version: {torch.__version__}")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            print(f"Memory fraction limit: {self.GPU_MEMORY_FRACTION * 100:.0f}%")
            print(f"BF16 enabled: {self.USE_BF16}")
        else:
            print("GPU not available, using CPU")
    
    def setup_gpu_memory_limit(self):
        """Setup GPU memory limit to prevent OOM crashes"""
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            import torch
            # Limit memory fraction dla stabilności
            torch.cuda.set_per_process_memory_fraction(
                self.GPU_MEMORY_FRACTION, 
                device=0
            )
            # Włącz memory efficient attention jeśli dostępne
            if hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'):
                torch.backends.cuda.enable_mem_efficient_sdp(True)
            print(f"GPU memory limit set to {self.GPU_MEMORY_FRACTION * 100:.0f}%")
    
    def clear_gpu_memory(self):
        """Clear GPU memory cache"""
        if TORCH_AVAILABLE and CUDA_AVAILABLE:
            import torch
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()


# Global config instance
CONFIG = Config()

# For backward compatibility with existing scripts
def create_directories():
    """Create all necessary directories"""
    CONFIG.__post_init__()


# Backward compatibility aliases
class LegacyConfig:
    """Backward compatibility with src/utils/config.py"""
    PROJECT_ROOT = CONFIG.PROJECT_ROOT
    DATA_ROOT = CONFIG.DATA_ROOT
    RAW_DATA_DIR = CONFIG.RAW_DATA_DIR
    PROCESSED_DATA_DIR = CONFIG.PROCESSED_DATA_DIR
    MODELS_DIR = CONFIG.MODELS_DIR
    SESSION_DATA_DIR = CONFIG.SESSION_DATA_DIR
    TRANSCRIPTIONS_DIR = CONFIG.TRANSCRIPTIONS_DIR
    
    LABELS_CSV = CONFIG.LABELS_CSV
    AUDIO_FEATURES_CSV = CONFIG.AUDIO_FEATURES_CSV
    TEXT_FEATURES_CSV = CONFIG.TEXT_FEATURES_CSV
    FUSED_FEATURES_H5 = CONFIG.PROCESSED_DATA_DIR / "daic_fused_features.h5"
    
    X_TRAIN_NPY = CONFIG.X_TRAIN_NPY
    X_VAL_NPY = CONFIG.X_VAL_NPY
    X_TEST_NPY = CONFIG.X_TEST_NPY
    Y_TRAIN_NPY = CONFIG.Y_TRAIN_NPY
    Y_VAL_NPY = CONFIG.Y_VAL_NPY
    Y_TEST_NPY = CONFIG.Y_TEST_NPY
    
    MODEL_PKL = CONFIG.MODEL_PKL
    SCALER_PKL = CONFIG.SCALER_PKL
    
    SAMPLE_RATE = CONFIG.SAMPLE_RATE
    N_MELS = CONFIG.N_MELS
    N_MFCC = CONFIG.N_MFCC
    
    TEXT_MODEL_NAME = CONFIG.TEXT_MODEL_NAME
    MAX_TEXT_LENGTH = CONFIG.MAX_TEXT_LENGTH
    
    RANDOM_SEED = CONFIG.RANDOM_SEED
    TRAIN_SIZE = CONFIG.TRAIN_SIZE
    VAL_SIZE = CONFIG.VAL_SIZE
    TEST_SIZE = CONFIG.TEST_SIZE
    
    USE_OPENSMILE = CONFIG.USE_OPENSMILE
    USE_TEXT = True
    
    PROBLEMATIC_SESSIONS = CONFIG.PROBLEMATIC_SESSIONS
    
    BATCH_SIZE = CONFIG.BATCH_SIZE
    NUM_WORKERS = CONFIG.NUM_WORKERS  # 0 dla Windows
    PIN_MEMORY = CONFIG.PIN_MEMORY
    
    @classmethod
    def create_directories(cls):
        create_directories()
    
    @classmethod
    def get_session_audio_path(cls, session_id: int) -> Path:
        return cls.SESSION_DATA_DIR / str(session_id) / f"{session_id}_AUDIO.wav"
    
    @classmethod
    def get_session_xml_path(cls, session_id: int) -> Path:
        return cls.SESSION_DATA_DIR / str(session_id) / f"{session_id}.xml"
    
    @classmethod
    def get_transcription_path(cls, session_id: int) -> Path:
        return cls.TRANSCRIPTIONS_DIR / f"Participant_{session_id}.xml"

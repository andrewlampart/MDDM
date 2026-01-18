"""
Advanced Audio Encoder for Depression Detection

Combines:
1. Wav2Vec 2.0 self-supervised embeddings (768-dim)
2. Depression-specific prosody features (~20-dim)

Depression-specific acoustic markers:
- Fundamental Frequency (F0): Lower in depression, reduced variation
- Energy Contour: Flatter, less dynamic
- Speech Rate: Slower, more pauses
- Voice Quality: Changes in formants, spectral features

Expected improvement: F1 +0.08-0.12 vs basic MFCC
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging
import warnings
import gc

warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# Import config for GPU settings
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import CONFIG
    MAX_AUDIO_LENGTH_SEC = CONFIG.MAX_AUDIO_LENGTH_SEC
    AUDIO_CHUNK_OVERLAP_SEC = CONFIG.AUDIO_CHUNK_OVERLAP_SEC
    GPU_MEMORY_FRACTION = CONFIG.GPU_MEMORY_FRACTION
except ImportError:
    MAX_AUDIO_LENGTH_SEC = 60
    AUDIO_CHUNK_OVERLAP_SEC = 5
    GPU_MEMORY_FRACTION = 0.8


class AudioEncoderHybrid:
    """
    Hybrid audio encoder combining Wav2Vec 2.0 with manual prosody features.
    
    Wav2Vec 2.0: Self-supervised learning on 960h LibriSpeech
    - Learns acoustic patterns without task-specific training
    - 768-dimensional embeddings
    
    Prosody Features: Depression-specific markers
    - Interpretable for clinicians
    - F0 dynamics, energy, speech rate
    
    Usage:
        encoder = AudioEncoderHybrid()
        embedding = encoder.extract_all('audio.wav')
        # embedding.shape = (~788,) - 768 Wav2Vec + 20 prosody
    """
    
    def __init__(
        self,
        use_wav2vec: bool = True,
        use_prosody: bool = True,
        wav2vec_model: str = "facebook/wav2vec2-base-960h",
        device: Optional[str] = None,
        sample_rate: int = 16000,
        max_audio_length_sec: int = MAX_AUDIO_LENGTH_SEC,
        chunk_overlap_sec: int = AUDIO_CHUNK_OVERLAP_SEC
    ):
        """
        Initialize the audio encoder.
        
        Args:
            use_wav2vec: Whether to extract Wav2Vec embeddings
            use_prosody: Whether to extract prosody features
            wav2vec_model: HuggingFace model name for Wav2Vec
            device: 'cuda', 'cpu', or None (auto-detect)
            sample_rate: Audio sample rate (Wav2Vec requires 16kHz)
            max_audio_length_sec: Max audio length before chunking (default 60s)
            chunk_overlap_sec: Overlap between chunks (default 5s)
        """
        self.use_wav2vec = use_wav2vec
        self.use_prosody = use_prosody
        self.sample_rate = sample_rate
        self.max_audio_length_sec = max_audio_length_sec
        self.chunk_overlap_sec = chunk_overlap_sec
        
        self.wav2vec_dim = 768 if use_wav2vec else 0
        self.prosody_dim = 20 if use_prosody else 0
        
        if use_wav2vec:
            self._init_wav2vec(wav2vec_model, device)
        else:
            self.device = 'cpu'
        
        logger.info(f"AudioEncoderHybrid initialized:")
        logger.info(f"  Wav2Vec: {self.wav2vec_dim}-dim" if use_wav2vec else "  Wav2Vec: disabled")
        logger.info(f"  Prosody: {self.prosody_dim}-dim" if use_prosody else "  Prosody: disabled")
        logger.info(f"  Total: {self.get_embedding_dim()}-dim")
        logger.info(f"  Max audio length: {max_audio_length_sec}s (chunked if longer)")
    
    def _init_wav2vec(self, model_name: str, device: Optional[str]):
        """Initialize Wav2Vec 2.0 model with memory management."""
        try:
            import torch
            from transformers import Wav2Vec2Processor, Wav2Vec2Model
        except ImportError:
            raise ImportError(
                "transformers and torch required for Wav2Vec. Run:\n"
                "pip install transformers torch"
            )
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        
        # Setup GPU memory limit dla RTX 5060 8GB
        if device == 'cuda':
            try:
                torch.cuda.set_per_process_memory_fraction(GPU_MEMORY_FRACTION, device=0)
                logger.info(f"GPU memory limit set to {GPU_MEMORY_FRACTION * 100:.0f}%")
            except Exception as e:
                logger.warning(f"Could not set GPU memory fraction: {e}")
        
        logger.info(f"Loading Wav2Vec2: {model_name}")
        self.wav2vec_processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.wav2vec_model = Wav2Vec2Model.from_pretrained(model_name).to(device)
        self.wav2vec_model.eval()
        
        # Enable gradient checkpointing dla oszczędności pamięci
        if hasattr(self.wav2vec_model, 'gradient_checkpointing_enable'):
            self.wav2vec_model.gradient_checkpointing_enable()
        
        logger.info(f"Wav2Vec loaded on {device}")
    
    def extract_wav2vec_embeddings(
        self,
        audio: Union[np.ndarray, str, Path],
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract Wav2Vec 2.0 embeddings with chunked processing for long audio.
        
        Args:
            audio: Audio waveform array or path to audio file
            sr: Sample rate (if audio is array)
            
        Returns:
            numpy array of shape (768,)
        """
        import torch
        import librosa
        
        # Load audio if path
        if isinstance(audio, (str, Path)):
            y, sr = librosa.load(str(audio), sr=self.sample_rate)
        else:
            y = audio
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
        
        # Check if audio needs chunking
        audio_length_sec = len(y) / self.sample_rate
        if audio_length_sec > self.max_audio_length_sec:
            logger.info(f"Audio length {audio_length_sec:.1f}s > {self.max_audio_length_sec}s, using chunked processing")
            return self._extract_wav2vec_chunked(y)
        
        try:
            embedding = self._extract_wav2vec_single(y)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"GPU OOM, falling back to chunked processing")
                self._clear_gpu_memory()
                return self._extract_wav2vec_chunked(y)
            raise
        
        # Cleanup GPU memory
        if self.device == 'cuda':
            self._clear_gpu_memory()
        
        return embedding
    
    def _extract_wav2vec_single(self, y: np.ndarray) -> np.ndarray:
        """Extract embeddings from single audio chunk."""
        import torch
        
        # Process
        inputs = self.wav2vec_processor(
            y, sampling_rate=self.sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Extract embeddings
        with torch.no_grad():
            outputs = self.wav2vec_model(**inputs)
        
        # Mean pooling over time dimension
        embedding = outputs.last_hidden_state.mean(dim=1)[0].cpu().numpy()
        
        # Clear intermediate tensors
        del inputs, outputs
        
        return embedding  # (768,)
    
    def _extract_wav2vec_chunked(self, y: np.ndarray) -> np.ndarray:
        """
        Extract embeddings from long audio by processing in chunks.
        
        Splits audio into max_audio_length_sec chunks with overlap,
        extracts embeddings from each, then averages them.
        """
        import torch
        
        chunk_samples = int(self.max_audio_length_sec * self.sample_rate)
        overlap_samples = int(self.chunk_overlap_sec * self.sample_rate)
        step_samples = chunk_samples - overlap_samples
        
        chunk_embeddings = []
        start = 0
        
        while start < len(y):
            end = min(start + chunk_samples, len(y))
            chunk = y[start:end]
            
            # Skip very short chunks
            if len(chunk) < self.sample_rate:  # < 1 second
                break
            
            try:
                embedding = self._extract_wav2vec_single(chunk)
                chunk_embeddings.append(embedding)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM on chunk, clearing memory and retrying on CPU")
                    self._clear_gpu_memory()
                    # Fallback to CPU for this chunk
                    original_device = self.device
                    self.device = 'cpu'
                    self.wav2vec_model = self.wav2vec_model.cpu()
                    embedding = self._extract_wav2vec_single(chunk)
                    chunk_embeddings.append(embedding)
                    # Try to move back to GPU
                    if original_device == 'cuda' and torch.cuda.is_available():
                        try:
                            self._clear_gpu_memory()
                            self.wav2vec_model = self.wav2vec_model.cuda()
                            self.device = 'cuda'
                        except:
                            logger.warning("Could not move model back to GPU, staying on CPU")
                else:
                    raise
            
            # Clear memory after each chunk
            if self.device == 'cuda':
                self._clear_gpu_memory()
            
            start += step_samples
        
        if not chunk_embeddings:
            logger.warning("No valid chunks extracted, returning zeros")
            return np.zeros(self.wav2vec_dim)
        
        # Average embeddings from all chunks
        return np.mean(chunk_embeddings, axis=0)
    
    def _clear_gpu_memory(self):
        """Clear GPU memory cache."""
        import torch
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    
    def extract_prosody_features(
        self,
        audio: Union[np.ndarray, str, Path],
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract depression-specific prosody features.
        
        Features extracted:
        1. F0 (Fundamental Frequency) dynamics - intonation
        2. Energy contour - prosodic emphasis
        3. Speech rate - temporal dynamics
        4. Spectral features - voice quality
        5. MFCC variation - voice quality stability
        
        Args:
            audio: Audio waveform or path
            sr: Sample rate
            
        Returns:
            numpy array of shape (~20,)
        """
        import librosa
        
        # Load audio if path
        if isinstance(audio, (str, Path)):
            y, sr = librosa.load(str(audio), sr=self.sample_rate)
        else:
            y = audio
            sr = sr or self.sample_rate
            if sr != self.sample_rate:
                y = librosa.resample(y, orig_sr=sr, target_sr=self.sample_rate)
        
        features = {}
        
        # 1. Fundamental Frequency (F0) - Intonation
        # Lower F0 and reduced variation indicate depression
        try:
            f0, voiced_flag, voiced_probs = librosa.pyin(
                y, fmin=80, fmax=400, sr=sr
            )
            f0_valid = f0[~np.isnan(f0)]
            
            if len(f0_valid) > 0:
                features['f0_mean'] = np.mean(f0_valid)
                features['f0_std'] = np.std(f0_valid)
                features['f0_range'] = np.ptp(f0_valid)  # Peak-to-peak
                features['f0_median'] = np.median(f0_valid)
                # Voiced ratio (% of speech that is voiced)
                features['voiced_ratio'] = np.sum(voiced_flag) / len(voiced_flag)
            else:
                features['f0_mean'] = 0
                features['f0_std'] = 0
                features['f0_range'] = 0
                features['f0_median'] = 0
                features['voiced_ratio'] = 0
        except Exception as e:
            logger.warning(f"F0 extraction failed: {e}")
            features.update({
                'f0_mean': 0, 'f0_std': 0, 'f0_range': 0,
                'f0_median': 0, 'voiced_ratio': 0
            })
        
        # 2. Energy Contour - Prosodic emphasis
        # Flatter energy in depression
        rms = librosa.feature.rms(y=y)[0]
        features['energy_mean'] = np.mean(rms)
        features['energy_std'] = np.std(rms)
        features['energy_range'] = np.ptp(rms)
        features['energy_skew'] = self._safe_skew(rms)
        
        # 3. Speech Rate - Temporal dynamics
        # Slower speech and more pauses in depression
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        features['speech_rate_mean'] = np.mean(zcr)
        features['speech_rate_std'] = np.std(zcr)
        
        # Pause detection (low energy segments)
        energy_threshold = np.percentile(rms, 20)
        features['pause_ratio'] = np.mean(rms < energy_threshold)
        
        # 4. Spectral Features - Voice quality
        spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        features['spectral_centroid_mean'] = np.mean(spec_cent)
        features['spectral_centroid_std'] = np.std(spec_cent)
        
        spec_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        features['spectral_rolloff_mean'] = np.mean(spec_rolloff)
        
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        features['spectral_bandwidth_mean'] = np.mean(spec_bandwidth)
        
        # 5. MFCC variation - Voice quality stability
        # Higher variation may indicate irregular voice (depression marker)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        mfcc_delta = np.abs(np.diff(mfcc, axis=1))
        features['mfcc_delta_mean'] = np.mean(mfcc_delta)
        features['mfcc_delta_std'] = np.std(mfcc_delta)
        
        # Convert to array
        prosody_features = np.array(list(features.values()), dtype=np.float32)
        
        # Handle NaN/Inf
        prosody_features = np.nan_to_num(prosody_features, nan=0.0, posinf=0.0, neginf=0.0)
        
        return prosody_features
    
    def _safe_skew(self, x: np.ndarray) -> float:
        """Safely compute skewness."""
        from scipy import stats
        try:
            return stats.skew(x)
        except:
            return 0.0
    
    def extract_all(
        self,
        audio: Union[np.ndarray, str, Path],
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract all features (Wav2Vec + Prosody).
        
        Args:
            audio: Audio waveform or path
            sr: Sample rate
            
        Returns:
            Combined feature vector of shape (~788,)
        """
        features = []
        
        if self.use_wav2vec:
            wav2vec = self.extract_wav2vec_embeddings(audio, sr)
            features.append(wav2vec)
        
        if self.use_prosody:
            prosody = self.extract_prosody_features(audio, sr)
            features.append(prosody)
        
        if not features:
            raise ValueError("Both use_wav2vec and use_prosody are False!")
        
        combined = np.concatenate(features)
        return combined
    
    def extract_batch(
        self,
        audio_paths: List[Union[str, Path]],
        show_progress: bool = True,
        cleanup_every: int = 5
    ) -> np.ndarray:
        """
        Extract features for multiple audio files with memory management.
        
        Args:
            audio_paths: List of paths to audio files
            show_progress: Show progress bar
            cleanup_every: Clear GPU memory every N files (default 5)
            
        Returns:
            numpy array of shape (n_samples, embedding_dim)
        """
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(audio_paths, desc="Extracting audio features")
            except ImportError:
                iterator = audio_paths
        else:
            iterator = audio_paths
        
        embeddings = []
        for i, path in enumerate(iterator):
            try:
                emb = self.extract_all(path)
                embeddings.append(emb)
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    logger.warning(f"OOM on {path}, clearing memory and using zeros")
                    self._clear_gpu_memory()
                    embeddings.append(np.zeros(self.get_embedding_dim()))
                else:
                    logger.error(f"Failed to process {path}: {e}")
                    embeddings.append(np.zeros(self.get_embedding_dim()))
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                embeddings.append(np.zeros(self.get_embedding_dim()))
            
            # Periodic memory cleanup
            if (i + 1) % cleanup_every == 0 and self.use_wav2vec and self.device == 'cuda':
                self._clear_gpu_memory()
                logger.debug(f"Memory cleanup after {i+1} files")
        
        # Final cleanup
        if self.use_wav2vec and self.device == 'cuda':
            self._clear_gpu_memory()
        
        return np.array(embeddings)
    
    def get_embedding_dim(self) -> int:
        """Return total embedding dimension."""
        return self.wav2vec_dim + self.prosody_dim
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        names = []
        
        if self.use_wav2vec:
            names.extend([f'wav2vec_{i}' for i in range(self.wav2vec_dim)])
        
        if self.use_prosody:
            prosody_names = [
                'f0_mean', 'f0_std', 'f0_range', 'f0_median', 'voiced_ratio',
                'energy_mean', 'energy_std', 'energy_range', 'energy_skew',
                'speech_rate_mean', 'speech_rate_std', 'pause_ratio',
                'spectral_centroid_mean', 'spectral_centroid_std',
                'spectral_rolloff_mean', 'spectral_bandwidth_mean',
                'mfcc_delta_mean', 'mfcc_delta_std'
            ]
            names.extend(prosody_names)
        
        return names
    
    def __repr__(self):
        return (f"AudioEncoderHybrid(wav2vec={self.wav2vec_dim}, "
                f"prosody={self.prosody_dim}, total={self.get_embedding_dim()})")


class AudioEncoderProsodyOnly:
    """
    Lightweight prosody-only encoder (no deep learning).
    
    Use this when Wav2Vec is too slow or resources are limited.
    Still extracts depression-specific features.
    """
    
    def __init__(self, sample_rate: int = 16000):
        self.sample_rate = sample_rate
        self._hybrid = AudioEncoderHybrid(
            use_wav2vec=False,
            use_prosody=True,
            sample_rate=sample_rate
        )
        self.embedding_dim = self._hybrid.get_embedding_dim()
    
    def extract(self, audio, sr=None):
        return self._hybrid.extract_all(audio, sr)
    
    def extract_batch(self, paths, show_progress=True):
        return self._hybrid.extract_batch(paths, show_progress)
    
    def get_embedding_dim(self):
        return self.embedding_dim


def extract_audio_features(
    audio_path: Union[str, Path],
    use_wav2vec: bool = True,
    use_prosody: bool = True
) -> np.ndarray:
    """
    Convenience function to extract audio features.
    
    Args:
        audio_path: Path to audio file
        use_wav2vec: Include Wav2Vec embeddings
        use_prosody: Include prosody features
        
    Returns:
        Feature vector
    """
    encoder = AudioEncoderHybrid(use_wav2vec=use_wav2vec, use_prosody=use_prosody)
    return encoder.extract_all(audio_path)


if __name__ == "__main__":
    import sys
    
    print("Testing AudioEncoderHybrid...")
    
    # Test prosody-only first (doesn't require GPU)
    print("\n1. Testing Prosody-only encoder:")
    try:
        import librosa
        import numpy as np
        
        # Create synthetic audio
        sr = 16000
        duration = 3.0
        t = np.linspace(0, duration, int(sr * duration))
        # Simple sine wave with some variation
        audio = 0.5 * np.sin(2 * np.pi * 200 * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))
        
        encoder_prosody = AudioEncoderHybrid(use_wav2vec=False, use_prosody=True)
        prosody_features = encoder_prosody.extract_all(audio, sr=sr)
        print(f"  Prosody features shape: {prosody_features.shape}")
        print(f"  Feature names: {encoder_prosody.get_feature_names()[:5]}...")
        print("  [OK] Prosody extraction works!")
    except Exception as e:
        print(f"  [ERROR] Prosody extraction failed: {e}")
        sys.exit(1)
    
    # Test full encoder (requires transformers)
    print("\n2. Testing Full encoder (Wav2Vec + Prosody):")
    try:
        encoder_full = AudioEncoderHybrid(use_wav2vec=True, use_prosody=True)
        full_features = encoder_full.extract_all(audio, sr=sr)
        print(f"  Full features shape: {full_features.shape}")
        print(f"  Expected: ~{768 + 20} dimensions")
        print("  [OK] Full extraction works!")
    except ImportError as e:
        print(f"  [SKIP] Wav2Vec not available: {e}")
        print("  Install with: pip install transformers torch")
    except Exception as e:
        print(f"  [ERROR] Full extraction failed: {e}")
    
    print("\n[OK] AudioEncoderHybrid tests complete!")

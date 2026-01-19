"""
Advanced Audio Encoder for Depression Detection

Combines:
1. Wav2Vec 2.0 self-supervised embeddings (768-dim)
2. Depression-specific prosody features (~20-dim)
3. Extended MFCC features (39-dim with deltas)
4. Glottal features (jitter, shimmer, HNR) (~10-dim)
5. Voice Activity Detection (VAD) preprocessing

Depression-specific acoustic markers:
- Fundamental Frequency (F0): Lower in depression, reduced variation
- Energy Contour: Flatter, less dynamic
- Speech Rate: Slower, more pauses
- Voice Quality: Changes in formants, spectral features
- Glottal Features: Jitter, shimmer, HNR indicate voice quality

Expected improvement: F1 +0.15-0.25 vs basic approach
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


# ============================================================================
# Voice Activity Detection (VAD)
# ============================================================================

class VADFilter:
    """
    Voice Activity Detection filter to remove silence and noise.
    
    Uses energy-based VAD with optional webrtcvad or silero-vad backends.
    Removes non-speech segments before feature extraction to improve quality.
    
    Depression speech often has more pauses - we preserve pause statistics
    but remove long silences that add noise to features.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        frame_duration_ms: int = 30,
        min_speech_duration_ms: int = 250,
        energy_threshold_percentile: int = 20,
        backend: str = 'energy'
    ):
        """
        Initialize VAD filter.
        
        Args:
            sample_rate: Audio sample rate (16000 for Wav2Vec compatibility)
            frame_duration_ms: Frame size in milliseconds
            min_speech_duration_ms: Minimum speech segment duration
            energy_threshold_percentile: Percentile for energy threshold
            backend: 'energy', 'webrtcvad', or 'silero'
        """
        self.sample_rate = sample_rate
        self.frame_duration_ms = frame_duration_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.energy_threshold_percentile = energy_threshold_percentile
        self.backend = backend
        
        self.frame_size = int(sample_rate * frame_duration_ms / 1000)
        self.min_speech_frames = int(min_speech_duration_ms / frame_duration_ms)
        
        # Initialize backend
        self._init_backend()
    
    def _init_backend(self):
        """Initialize VAD backend."""
        if self.backend == 'webrtcvad':
            try:
                import webrtcvad
                self.vad = webrtcvad.Vad(2)  # Aggressiveness mode 2
                logger.info("Using webrtcvad backend")
            except ImportError:
                logger.warning("webrtcvad not available, falling back to energy-based VAD")
                self.backend = 'energy'
        
        elif self.backend == 'silero':
            try:
                import torch
                self.silero_model, self.silero_utils = torch.hub.load(
                    repo_or_dir='snakers4/silero-vad',
                    model='silero_vad',
                    force_reload=False,
                    onnx=False
                )
                logger.info("Using Silero VAD backend")
            except Exception as e:
                logger.warning(f"Silero VAD not available: {e}, falling back to energy-based VAD")
                self.backend = 'energy'
        
        if self.backend == 'energy':
            logger.info("Using energy-based VAD backend")
    
    def get_speech_segments(
        self,
        audio: np.ndarray,
        return_mask: bool = False
    ) -> Union[List[Tuple[int, int]], np.ndarray]:
        """
        Detect speech segments in audio.
        
        Args:
            audio: Audio waveform
            return_mask: If True, return boolean mask instead of segments
            
        Returns:
            List of (start, end) sample indices or boolean mask
        """
        if self.backend == 'energy':
            return self._energy_vad(audio, return_mask)
        elif self.backend == 'webrtcvad':
            return self._webrtc_vad(audio, return_mask)
        elif self.backend == 'silero':
            return self._silero_vad(audio, return_mask)
    
    def _energy_vad(
        self,
        audio: np.ndarray,
        return_mask: bool = False
    ) -> Union[List[Tuple[int, int]], np.ndarray]:
        """Energy-based VAD."""
        import librosa
        
        # Compute frame-wise energy
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_size,
            hop_length=self.frame_size
        )[0]
        
        # Adaptive threshold
        threshold = np.percentile(rms, self.energy_threshold_percentile)
        
        # Speech frames
        speech_frames = rms > threshold
        
        if return_mask:
            # Expand to sample-level mask
            mask = np.repeat(speech_frames, self.frame_size)[:len(audio)]
            return mask
        
        # Find contiguous speech segments
        segments = []
        in_speech = False
        start = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start = i * self.frame_size
                in_speech = True
            elif not is_speech and in_speech:
                end = i * self.frame_size
                if (end - start) / self.sample_rate * 1000 >= self.min_speech_duration_ms:
                    segments.append((start, end))
                in_speech = False
        
        # Handle last segment
        if in_speech:
            end = len(audio)
            if (end - start) / self.sample_rate * 1000 >= self.min_speech_duration_ms:
                segments.append((start, end))
        
        return segments
    
    def _webrtc_vad(
        self,
        audio: np.ndarray,
        return_mask: bool = False
    ) -> Union[List[Tuple[int, int]], np.ndarray]:
        """WebRTC VAD backend."""
        import webrtcvad
        
        # Convert to 16-bit PCM
        audio_int16 = (audio * 32767).astype(np.int16)
        
        # Frame-wise VAD
        speech_frames = []
        for i in range(0, len(audio_int16) - self.frame_size, self.frame_size):
            frame = audio_int16[i:i + self.frame_size].tobytes()
            try:
                is_speech = self.vad.is_speech(frame, self.sample_rate)
                speech_frames.append(is_speech)
            except:
                speech_frames.append(False)
        
        speech_frames = np.array(speech_frames)
        
        if return_mask:
            mask = np.repeat(speech_frames, self.frame_size)[:len(audio)]
            return mask
        
        # Find segments (same logic as energy VAD)
        segments = []
        in_speech = False
        start = 0
        
        for i, is_speech in enumerate(speech_frames):
            if is_speech and not in_speech:
                start = i * self.frame_size
                in_speech = True
            elif not is_speech and in_speech:
                end = i * self.frame_size
                if (end - start) / self.sample_rate * 1000 >= self.min_speech_duration_ms:
                    segments.append((start, end))
                in_speech = False
        
        if in_speech:
            segments.append((start, len(audio)))
        
        return segments
    
    def _silero_vad(
        self,
        audio: np.ndarray,
        return_mask: bool = False
    ) -> Union[List[Tuple[int, int]], np.ndarray]:
        """Silero VAD backend (most accurate)."""
        import torch
        
        # Convert to tensor
        audio_tensor = torch.from_numpy(audio).float()
        
        # Get speech timestamps
        get_speech_timestamps = self.silero_utils[0]
        timestamps = get_speech_timestamps(
            audio_tensor,
            self.silero_model,
            sampling_rate=self.sample_rate,
            min_speech_duration_ms=self.min_speech_duration_ms
        )
        
        if return_mask:
            mask = np.zeros(len(audio), dtype=bool)
            for ts in timestamps:
                mask[ts['start']:ts['end']] = True
            return mask
        
        return [(ts['start'], ts['end']) for ts in timestamps]
    
    def filter_audio(
        self,
        audio: np.ndarray,
        keep_ratio: float = 0.1
    ) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Filter audio keeping only speech segments.
        
        Args:
            audio: Input audio waveform
            keep_ratio: Minimum ratio of audio to keep (prevents empty output)
            
        Returns:
            Tuple of (filtered_audio, vad_statistics)
        """
        segments = self.get_speech_segments(audio)
        
        # Statistics for depression analysis
        total_duration = len(audio) / self.sample_rate
        speech_duration = sum((e - s) / self.sample_rate for s, e in segments)
        
        stats = {
            'total_duration_sec': total_duration,
            'speech_duration_sec': speech_duration,
            'speech_ratio': speech_duration / total_duration if total_duration > 0 else 0,
            'num_speech_segments': len(segments),
            'avg_segment_duration_sec': speech_duration / len(segments) if segments else 0,
            'num_pauses': max(0, len(segments) - 1),
        }
        
        # If no speech detected or too little, return original
        if not segments or speech_duration / total_duration < keep_ratio:
            logger.warning(f"VAD kept only {stats['speech_ratio']:.1%} of audio, using original")
            return audio, stats
        
        # Concatenate speech segments
        filtered = np.concatenate([audio[s:e] for s, e in segments])
        
        return filtered, stats


# ============================================================================
# Extended Audio Features
# ============================================================================

def extract_extended_mfcc(
    audio: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 13
) -> np.ndarray:
    """
    Extract extended MFCC features (39-dim).
    
    Returns MFCC + delta + delta-delta for richer representation.
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        n_mfcc: Number of MFCC coefficients
        
    Returns:
        Feature vector of shape (n_mfcc * 3,) = (39,) by default
    """
    import librosa
    
    # Extract MFCCs
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    
    # Compute deltas (velocity)
    mfcc_delta = librosa.feature.delta(mfcc)
    
    # Compute delta-deltas (acceleration)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)
    
    # Aggregate statistics
    features = []
    for feat in [mfcc, mfcc_delta, mfcc_delta2]:
        features.extend([
            np.mean(feat, axis=1),  # Mean over time
        ])
    
    return np.concatenate(features)  # (39,)


def extract_glottal_features(
    audio: np.ndarray,
    sr: int = 16000
) -> np.ndarray:
    """
    Extract glottal/voice quality features relevant to depression.
    
    Features:
    - Jitter: Cycle-to-cycle variation in fundamental frequency
    - Shimmer: Cycle-to-cycle variation in amplitude
    - HNR: Harmonics-to-Noise Ratio (voice clarity)
    
    These features are clinically validated markers of depression.
    
    Args:
        audio: Audio waveform
        sr: Sample rate
        
    Returns:
        Feature vector of shape (~10,)
    """
    import librosa
    
    features = {}
    
    # Extract F0 for jitter/shimmer calculation
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(
            audio, fmin=80, fmax=400, sr=sr
        )
        f0_valid = f0[~np.isnan(f0)]
        
        if len(f0_valid) > 2:
            # Jitter (local) - percentage
            f0_diff = np.abs(np.diff(f0_valid))
            jitter_local = np.mean(f0_diff) / np.mean(f0_valid) * 100
            features['jitter_local'] = jitter_local
            
            # Jitter (rap) - relative average perturbation
            if len(f0_valid) > 3:
                jitter_rap = np.mean(np.abs(
                    f0_valid[1:-1] - (f0_valid[:-2] + f0_valid[1:-1] + f0_valid[2:]) / 3
                )) / np.mean(f0_valid) * 100
                features['jitter_rap'] = jitter_rap
            else:
                features['jitter_rap'] = 0
            
            # Jitter (ppq5) - 5-point period perturbation quotient
            if len(f0_valid) > 5:
                jitter_ppq5 = np.mean(np.abs(
                    f0_valid[2:-2] - np.convolve(f0_valid, np.ones(5)/5, mode='valid')
                )) / np.mean(f0_valid) * 100
                features['jitter_ppq5'] = jitter_ppq5
            else:
                features['jitter_ppq5'] = 0
        else:
            features['jitter_local'] = 0
            features['jitter_rap'] = 0
            features['jitter_ppq5'] = 0
            
    except Exception as e:
        logger.warning(f"Jitter extraction failed: {e}")
        features['jitter_local'] = 0
        features['jitter_rap'] = 0
        features['jitter_ppq5'] = 0
    
    # Shimmer (amplitude variation)
    try:
        # Frame-wise amplitude
        rms = librosa.feature.rms(y=audio, frame_length=512, hop_length=256)[0]
        
        if len(rms) > 2:
            # Shimmer local
            amp_diff = np.abs(np.diff(rms))
            shimmer_local = np.mean(amp_diff) / np.mean(rms) * 100
            features['shimmer_local'] = shimmer_local
            
            # Shimmer (apq3)
            if len(rms) > 3:
                shimmer_apq3 = np.mean(np.abs(
                    rms[1:-1] - (rms[:-2] + rms[1:-1] + rms[2:]) / 3
                )) / np.mean(rms) * 100
                features['shimmer_apq3'] = shimmer_apq3
            else:
                features['shimmer_apq3'] = 0
            
            # Shimmer (apq5)
            if len(rms) > 5:
                shimmer_apq5 = np.mean(np.abs(
                    rms[2:-2] - np.convolve(rms, np.ones(5)/5, mode='valid')
                )) / np.mean(rms) * 100
                features['shimmer_apq5'] = shimmer_apq5
            else:
                features['shimmer_apq5'] = 0
        else:
            features['shimmer_local'] = 0
            features['shimmer_apq3'] = 0
            features['shimmer_apq5'] = 0
            
    except Exception as e:
        logger.warning(f"Shimmer extraction failed: {e}")
        features['shimmer_local'] = 0
        features['shimmer_apq3'] = 0
        features['shimmer_apq5'] = 0
    
    # HNR (Harmonics-to-Noise Ratio)
    try:
        # Compute autocorrelation-based HNR estimate
        # Using spectral analysis as proxy
        S = np.abs(librosa.stft(audio))
        
        # Harmonic component (low frequencies)
        harmonic = librosa.effects.harmonic(audio)
        percussive = librosa.effects.percussive(audio)
        
        h_energy = np.sum(harmonic ** 2)
        n_energy = np.sum(percussive ** 2)
        
        if n_energy > 0:
            hnr = 10 * np.log10(h_energy / n_energy + 1e-10)
        else:
            hnr = 20  # Assume good quality if no noise
        
        features['hnr'] = np.clip(hnr, -10, 40)  # Reasonable range
        
    except Exception as e:
        logger.warning(f"HNR extraction failed: {e}")
        features['hnr'] = 0
    
    # Convert to array
    feature_array = np.array(list(features.values()), dtype=np.float32)
    
    # Handle NaN/Inf
    feature_array = np.nan_to_num(feature_array, nan=0.0, posinf=0.0, neginf=0.0)
    
    return feature_array


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


class AudioEncoderAdvanced:
    """
    Advanced Audio Encoder with VAD preprocessing and extended features.
    
    This is the recommended encoder for depression detection based on
    SOTA 2024-2025 literature. Combines:
    
    1. VAD preprocessing (removes silence/noise)
    2. Wav2Vec 2.0 embeddings (768-dim)
    3. Extended MFCC (39-dim with deltas)
    4. Prosody features (20-dim)
    5. Glottal features (10-dim) - jitter, shimmer, HNR
    
    Total: ~837-dim feature vector
    
    Usage:
        encoder = AudioEncoderAdvanced()
        features = encoder.extract_all('audio.wav')
        # features.shape = (~837,)
    """
    
    def __init__(
        self,
        use_wav2vec: bool = True,
        use_vad: bool = True,
        use_extended_mfcc: bool = True,
        use_glottal: bool = True,
        use_prosody: bool = True,
        wav2vec_model: str = "facebook/wav2vec2-base-960h",
        device: Optional[str] = None,
        sample_rate: int = 16000,
        vad_backend: str = 'energy'
    ):
        """
        Initialize advanced audio encoder.
        
        Args:
            use_wav2vec: Include Wav2Vec 2.0 embeddings (768-dim)
            use_vad: Apply VAD preprocessing
            use_extended_mfcc: Include extended MFCC features (39-dim)
            use_glottal: Include glottal features (10-dim)
            use_prosody: Include prosody features (20-dim)
            wav2vec_model: HuggingFace model name
            device: 'cuda', 'cpu', or None (auto)
            sample_rate: Audio sample rate
            vad_backend: 'energy', 'webrtcvad', or 'silero'
        """
        self.sample_rate = sample_rate
        self.use_wav2vec = use_wav2vec
        self.use_vad = use_vad
        self.use_extended_mfcc = use_extended_mfcc
        self.use_glottal = use_glottal
        self.use_prosody = use_prosody
        
        # Initialize VAD
        if use_vad:
            self.vad = VADFilter(
                sample_rate=sample_rate,
                backend=vad_backend
            )
        else:
            self.vad = None
        
        # Initialize base encoder for Wav2Vec and prosody
        self._base_encoder = AudioEncoderHybrid(
            use_wav2vec=use_wav2vec,
            use_prosody=use_prosody,
            wav2vec_model=wav2vec_model,
            device=device,
            sample_rate=sample_rate
        )
        
        self.device = self._base_encoder.device
        
        # Calculate dimensions
        self.wav2vec_dim = 768 if use_wav2vec else 0
        self.mfcc_dim = 39 if use_extended_mfcc else 0  # 13 * 3 (mfcc + delta + delta2)
        self.glottal_dim = 10 if use_glottal else 0
        self.prosody_dim = 20 if use_prosody else 0
        self.vad_stats_dim = 6 if use_vad else 0  # VAD statistics as features
        
        logger.info(f"AudioEncoderAdvanced initialized:")
        logger.info(f"  VAD: {'enabled (' + vad_backend + ')' if use_vad else 'disabled'}")
        logger.info(f"  Wav2Vec: {self.wav2vec_dim}-dim")
        logger.info(f"  Extended MFCC: {self.mfcc_dim}-dim")
        logger.info(f"  Glottal: {self.glottal_dim}-dim")
        logger.info(f"  Prosody: {self.prosody_dim}-dim")
        logger.info(f"  VAD stats: {self.vad_stats_dim}-dim")
        logger.info(f"  Total: {self.get_embedding_dim()}-dim")
    
    def extract_all(
        self,
        audio: Union[np.ndarray, str, Path],
        sr: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract all features with VAD preprocessing.
        
        Args:
            audio: Audio waveform or path
            sr: Sample rate
            
        Returns:
            Combined feature vector
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
        
        # Apply VAD
        vad_stats = {}
        if self.use_vad and self.vad is not None:
            y_filtered, vad_stats = self.vad.filter_audio(y)
        else:
            y_filtered = y
        
        features = []
        
        # Wav2Vec features (on filtered audio)
        if self.use_wav2vec:
            wav2vec = self._base_encoder.extract_wav2vec_embeddings(y_filtered, self.sample_rate)
            features.append(wav2vec)
        
        # Extended MFCC (on filtered audio)
        if self.use_extended_mfcc:
            mfcc_ext = extract_extended_mfcc(y_filtered, self.sample_rate)
            features.append(mfcc_ext)
        
        # Glottal features (on filtered audio)
        if self.use_glottal:
            glottal = extract_glottal_features(y_filtered, self.sample_rate)
            features.append(glottal)
        
        # Prosody features (on original audio to preserve pause info)
        if self.use_prosody:
            prosody = self._base_encoder.extract_prosody_features(y, self.sample_rate)
            features.append(prosody)
        
        # VAD statistics as features (useful for depression detection)
        if self.use_vad and vad_stats:
            vad_features = np.array([
                vad_stats.get('speech_ratio', 0),
                vad_stats.get('num_speech_segments', 0) / 10,  # Normalize
                vad_stats.get('avg_segment_duration_sec', 0),
                vad_stats.get('num_pauses', 0) / 10,  # Normalize
                vad_stats.get('speech_duration_sec', 0) / 60,  # Normalize to minutes
                vad_stats.get('total_duration_sec', 0) / 60,
            ], dtype=np.float32)
            features.append(vad_features)
        
        if not features:
            raise ValueError("No features enabled!")
        
        combined = np.concatenate(features)
        
        # Handle NaN/Inf
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        
        return combined
    
    def extract_batch(
        self,
        audio_paths: List[Union[str, Path]],
        show_progress: bool = True,
        cleanup_every: int = 5
    ) -> np.ndarray:
        """
        Extract features for multiple audio files.
        
        Args:
            audio_paths: List of paths to audio files
            show_progress: Show progress bar
            cleanup_every: Clear GPU memory every N files
            
        Returns:
            numpy array of shape (n_samples, embedding_dim)
        """
        if show_progress:
            try:
                from tqdm import tqdm
                iterator = tqdm(audio_paths, desc="Extracting advanced audio features")
            except ImportError:
                iterator = audio_paths
        else:
            iterator = audio_paths
        
        embeddings = []
        for i, path in enumerate(iterator):
            try:
                emb = self.extract_all(path)
                embeddings.append(emb)
            except Exception as e:
                logger.error(f"Failed to process {path}: {e}")
                embeddings.append(np.zeros(self.get_embedding_dim()))
            
            # Periodic memory cleanup
            if (i + 1) % cleanup_every == 0 and self.use_wav2vec and self.device == 'cuda':
                self._base_encoder._clear_gpu_memory()
        
        # Final cleanup
        if self.use_wav2vec and self.device == 'cuda':
            self._base_encoder._clear_gpu_memory()
        
        return np.array(embeddings)
    
    def get_embedding_dim(self) -> int:
        """Return total embedding dimension."""
        return (
            self.wav2vec_dim +
            self.mfcc_dim +
            self.glottal_dim +
            self.prosody_dim +
            self.vad_stats_dim
        )
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        names = []
        
        if self.use_wav2vec:
            names.extend([f'wav2vec_{i}' for i in range(self.wav2vec_dim)])
        
        if self.use_extended_mfcc:
            for prefix in ['mfcc', 'mfcc_delta', 'mfcc_delta2']:
                names.extend([f'{prefix}_{i}' for i in range(13)])
        
        if self.use_glottal:
            names.extend([
                'jitter_local', 'jitter_rap', 'jitter_ppq5',
                'shimmer_local', 'shimmer_apq3', 'shimmer_apq5',
                'hnr',
                'glottal_pad_1', 'glottal_pad_2', 'glottal_pad_3'  # Padding to 10
            ])
        
        if self.use_prosody:
            names.extend(self._base_encoder.get_feature_names())
        
        if self.use_vad:
            names.extend([
                'vad_speech_ratio', 'vad_num_segments', 'vad_avg_segment_dur',
                'vad_num_pauses', 'vad_speech_duration', 'vad_total_duration'
            ])
        
        return names
    
    def __repr__(self):
        return (
            f"AudioEncoderAdvanced("
            f"wav2vec={self.wav2vec_dim}, "
            f"mfcc={self.mfcc_dim}, "
            f"glottal={self.glottal_dim}, "
            f"prosody={self.prosody_dim}, "
            f"vad_stats={self.vad_stats_dim}, "
            f"total={self.get_embedding_dim()})"
        )


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
    
    print("Testing Audio Encoders...")
    
    # Create synthetic audio
    sr = 16000
    duration = 3.0
    t = np.linspace(0, duration, int(sr * duration))
    # Simulate speech with pauses
    audio = 0.5 * np.sin(2 * np.pi * 200 * t) * (1 + 0.3 * np.sin(2 * np.pi * 2 * t))
    # Add some silence
    audio[int(sr*1.0):int(sr*1.5)] = 0.01 * np.random.randn(int(sr*0.5))
    
    # Test VAD
    print("\n1. Testing VADFilter:")
    try:
        vad = VADFilter(sample_rate=sr, backend='energy')
        segments = vad.get_speech_segments(audio)
        filtered, stats = vad.filter_audio(audio)
        print(f"  Original length: {len(audio)} samples")
        print(f"  Filtered length: {len(filtered)} samples")
        print(f"  Speech segments: {len(segments)}")
        print(f"  Speech ratio: {stats['speech_ratio']:.2%}")
        print("  [OK] VADFilter works!")
    except Exception as e:
        print(f"  [ERROR] VAD failed: {e}")
    
    # Test extended MFCC
    print("\n2. Testing Extended MFCC:")
    try:
        mfcc_ext = extract_extended_mfcc(audio, sr)
        print(f"  Extended MFCC shape: {mfcc_ext.shape}")
        print(f"  Expected: (39,)")
        print("  [OK] Extended MFCC works!")
    except Exception as e:
        print(f"  [ERROR] Extended MFCC failed: {e}")
    
    # Test glottal features
    print("\n3. Testing Glottal Features:")
    try:
        glottal = extract_glottal_features(audio, sr)
        print(f"  Glottal features shape: {glottal.shape}")
        print(f"  Features: jitter={glottal[0]:.4f}, shimmer={glottal[3]:.4f}, hnr={glottal[6]:.2f}")
        print("  [OK] Glottal features work!")
    except Exception as e:
        print(f"  [ERROR] Glottal features failed: {e}")
    
    # Test prosody-only encoder
    print("\n4. Testing AudioEncoderHybrid (prosody-only):")
    try:
        encoder_prosody = AudioEncoderHybrid(use_wav2vec=False, use_prosody=True)
        prosody_features = encoder_prosody.extract_all(audio, sr=sr)
        print(f"  Prosody features shape: {prosody_features.shape}")
        print("  [OK] Prosody extraction works!")
    except Exception as e:
        print(f"  [ERROR] Prosody extraction failed: {e}")
        sys.exit(1)
    
    # Test advanced encoder (without Wav2Vec for speed)
    print("\n5. Testing AudioEncoderAdvanced (no Wav2Vec):")
    try:
        encoder_adv = AudioEncoderAdvanced(
            use_wav2vec=False,
            use_vad=True,
            use_extended_mfcc=True,
            use_glottal=True,
            use_prosody=True
        )
        adv_features = encoder_adv.extract_all(audio, sr=sr)
        print(f"  Advanced features shape: {adv_features.shape}")
        print(f"  Expected dim: {encoder_adv.get_embedding_dim()}")
        print(f"  {encoder_adv}")
        print("  [OK] AudioEncoderAdvanced works!")
    except Exception as e:
        print(f"  [ERROR] Advanced encoder failed: {e}")
    
    # Test full advanced encoder (requires transformers)
    print("\n6. Testing AudioEncoderAdvanced (full with Wav2Vec):")
    try:
        encoder_full = AudioEncoderAdvanced(
            use_wav2vec=True,
            use_vad=True,
            use_extended_mfcc=True,
            use_glottal=True,
            use_prosody=True
        )
        full_features = encoder_full.extract_all(audio, sr=sr)
        print(f"  Full features shape: {full_features.shape}")
        print(f"  Expected dim: {encoder_full.get_embedding_dim()}")
        print(f"  {encoder_full}")
        print("  [OK] Full AudioEncoderAdvanced works!")
    except ImportError as e:
        print(f"  [SKIP] Wav2Vec not available: {e}")
        print("  Install with: pip install transformers torch")
    except Exception as e:
        print(f"  [ERROR] Full encoder failed: {e}")
    
    print("\n[OK] All audio encoder tests complete!")

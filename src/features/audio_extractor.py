"""Audio feature extraction for depression detection"""

import librosa
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import warnings

# Opcjonalny import openSMILE
try:
    import opensmile
    OPENSMILE_AVAILABLE = True
except ImportError:
    OPENSMILE_AVAILABLE = False
    warnings.warn("openSMILE not available. High-level features will be skipped.")

# Import prosody extractor
try:
    from .prosody_extractor import ProsodyExtractor
    PROSODY_AVAILABLE = True
except ImportError:
    PROSODY_AVAILABLE = False


class AudioFeatureExtractor:
    """Ekstrakcja cech akustycznych dla diagnozy depresji"""
    
    def __init__(self, sr: int = 16000, use_opensmile: bool = True, use_prosody: bool = True):
        """Initialize AudioFeatureExtractor
        
        Args:
            sr: Sample rate dla audio
            use_opensmile: Czy używać openSMILE (jeśli dostępne)
            use_prosody: Czy ekstrahować cechy prozodyczne (pausowanie, pitch)
        """
        self.sr = sr
        self.use_opensmile = use_opensmile and OPENSMILE_AVAILABLE
        self.use_prosody = use_prosody and PROSODY_AVAILABLE
        
        if self.use_opensmile:
            try:
                self.smile = opensmile.Smile(
                    feature_set=opensmile.FeatureSet.eGeMAPSv02,
                    feature_level=opensmile.FeatureLevel.Functionals
                )
            except Exception as e:
                warnings.warn(f"Failed to initialize openSMILE: {e}")
                self.use_opensmile = False
        else:
            self.smile = None
        
        # Initialize prosody extractor
        if self.use_prosody:
            self.prosody_extractor = ProsodyExtractor(sr=sr)
        else:
            self.prosody_extractor = None
    
    def extract_low_level_features(self, audio: np.ndarray, sr: int) -> Dict:
        """Ekstrahuj spectral i temporal features
        
        Args:
            audio: Sygnał audio (1D numpy array)
            sr: Sample rate
            
        Returns:
            Dictionary z low-level features
        """
        features = {}
        
        # 1. Mel-frequency cepstral coefficients (MFCC)
        mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=13)
        features['mfcc_mean'] = np.mean(mfcc, axis=1)
        features['mfcc_std'] = np.std(mfcc, axis=1)
        
        # 2. Spectral Centroid - gdzie skupiona jest energia
        spec_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)[0]
        features['spec_centroid_mean'] = np.mean(spec_centroid)
        features['spec_centroid_std'] = np.std(spec_centroid)
        
        # 3. Zero Crossing Rate - częstość przejść przez zero
        zcr = librosa.feature.zero_crossing_rate(audio)[0]
        features['zcr_mean'] = np.mean(zcr)
        features['zcr_std'] = np.std(zcr)
        
        # 4. Spectral Rolloff - frekvencja zawierająca 85% energii
        spec_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)[0]
        features['spec_rolloff_mean'] = np.mean(spec_rolloff)
        features['spec_rolloff_std'] = np.std(spec_rolloff)
        
        # 5. Chromatogram - energia w każdym pitch class
        chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        features['chroma_std'] = np.std(chroma, axis=1)
        
        # 6. Tempogram - rytm
        onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
        features['onset_strength_mean'] = np.mean(onset_env)
        features['onset_strength_std'] = np.std(onset_env)
        
        # 7. Spectral bandwidth
        spec_bandwidth = librosa.feature.spectral_bandwidth(y=audio, sr=sr)[0]
        features['spec_bandwidth_mean'] = np.mean(spec_bandwidth)
        features['spec_bandwidth_std'] = np.std(spec_bandwidth)
        
        # 8. Tempo estimation
        try:
            tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
            features['tempo'] = tempo
        except:
            features['tempo'] = 0.0
        
        return features
    
    def extract_high_level_features(self, audio_path: Path) -> Optional[np.ndarray]:
        """Użyj openSMILE do ekstrakcji paralinguistic features
        
        Args:
            audio_path: Ścieżka do pliku audio
            
        Returns:
            Array z 88 eGeMAPS features lub None jeśli openSMILE niedostępne
        """
        if not self.use_opensmile or self.smile is None:
            return None
        
        try:
            result = self.smile.process_file(str(audio_path))
            return result.values[0]  # Shape: (88,)
        except Exception as e:
            warnings.warn(f"openSMILE processing failed for {audio_path}: {e}")
            return None
    
    def get_spectrogram(self, audio: np.ndarray, sr: int) -> np.ndarray:
        """Zwróć spectrogram do CNN (dla innej sieci)
        
        Args:
            audio: Sygnał audio
            sr: Sample rate
            
        Returns:
            Mel spectrogram w dB, shape: (n_mels, time_frames)
        """
        S = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=128)
        S_db = librosa.power_to_db(S, ref=np.max)
        return S_db
    
    def process_audio(self, audio_path: Path) -> Dict:
        """Pipeline: załaduj audio → ekstrahuj cechy
        
        Args:
            audio_path: Ścieżka do pliku audio
            
        Returns:
            Dictionary z low_level_features, high_level_features, prosody_features, spectrogram, duration
        """
        if not audio_path.exists():
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Załaduj audio
        audio, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
        
        # Normalizacja
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        # Ekstrakcja cech
        low_level = self.extract_low_level_features(audio, sr)
        high_level = self.extract_high_level_features(audio_path)
        spectrogram = self.get_spectrogram(audio, sr)
        
        # Ekstrakcja cech prozodycznych (pausowanie, pitch, intensity)
        # Używamy już załadowanego audio zamiast ładować ponownie
        prosody_features = None
        if self.prosody_extractor is not None:
            try:
                prosody_features = self.prosody_extractor.extract_all_features_from_audio(audio)
            except Exception as e:
                warnings.warn(f"Prosody extraction failed for {audio_path}: {e}")
        
        return {
            'low_level_features': low_level,
            'high_level_features': high_level,
            'prosody_features': prosody_features,
            'spectrogram': spectrogram,
            'duration': len(audio) / sr
        }
    
    def flatten_features(self, features: Dict) -> Dict:
        """Spłaszcz features do formatu dla DataFrame
        
        Args:
            features: Output z process_audio()
            
        Returns:
            Flattened dictionary z wszystkimi cechami jako scalars
        """
        flat = {}
        
        # Low-level features
        for key, val in features['low_level_features'].items():
            if isinstance(val, np.ndarray):
                for i, v in enumerate(val):
                    flat[f'{key}_{i}'] = float(v)
            else:
                flat[key] = float(val)
        
        # High-level features (openSMILE)
        if features['high_level_features'] is not None:
            for i, v in enumerate(features['high_level_features']):
                flat[f'opensmile_feat_{i}'] = float(v)
        else:
            # Wypełnij zerami jeśli brak openSMILE
            for i in range(88):  # eGeMAPS v02 ma 88 features
                flat[f'opensmile_feat_{i}'] = 0.0
        
        # Prosody features (pausowanie, pitch, intensity)
        if features.get('prosody_features') is not None:
            flat.update(features['prosody_features'])
        else:
            # Wypełnij zerami jeśli brak prosody
            prosody_keys = [
                'prosody_pause_ratio', 'prosody_speech_ratio', 'prosody_pause_count',
                'prosody_pause_rate_per_min', 'prosody_pause_duration_mean',
                'prosody_pause_duration_std', 'prosody_pause_duration_max',
                'prosody_pause_duration_median', 'prosody_speech_segment_count',
                'prosody_speech_segment_mean', 'prosody_speech_segment_std',
                'prosody_speech_rate', 'prosody_f0_mean', 'prosody_f0_std',
                'prosody_f0_range', 'prosody_f0_cv', 'prosody_voiced_fraction',
                'prosody_f0_slope', 'prosody_intensity_mean', 'prosody_intensity_std',
                'prosody_intensity_range', 'prosody_intensity_cv'
            ]
            for key in prosody_keys:
                flat[key] = 0.0
        
        flat['duration'] = float(features['duration'])
        
        return flat

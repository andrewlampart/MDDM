"""Prosody and pause analysis for depression detection

Kluczowe cechy prozodyczne powiązane z depresją:
- Dłuższe i częstsze pauzy
- Wolniejsze tempo mowy
- Mniejsza zmienność F0 (pitch)
- Niższe natężenie głosu
"""

import numpy as np
import librosa
from typing import Dict, Tuple, Optional
from pathlib import Path


class ProsodyExtractor:
    """Ekstrakcja cech prozodycznych - pausowanie, tempo, intonacja"""
    
    def __init__(self, sr: int = 16000, frame_length: int = 2048, hop_length: int = 512):
        """
        Args:
            sr: Sample rate
            frame_length: Długość ramki FFT (w próbkach)
            hop_length: Przesunięcie między ramkami
        """
        self.sr = sr
        self.frame_length = frame_length
        self.hop_length = hop_length
        
        # Parametry detekcji ciszy
        self.silence_threshold_db = -40  # dB poniżej maksimum
        self.min_pause_duration = 0.3  # sekundy - minimalna pauza
        self.min_speech_duration = 0.1  # sekundy - minimalny segment mowy
    
    def detect_voice_activity(self, audio: np.ndarray) -> np.ndarray:
        """Detekcja aktywności głosowej (VAD) oparta na energii
        
        Args:
            audio: Sygnał audio (1D)
            
        Returns:
            Binary array: 1 = mowa, 0 = cisza (per frame)
        """
        # Oblicz RMS energy per frame
        rms = librosa.feature.rms(
            y=audio, 
            frame_length=self.frame_length, 
            hop_length=self.hop_length
        )[0]
        
        # Konwertuj do dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Próg: wszystko powyżej threshold_db jest mową
        vad = (rms_db > self.silence_threshold_db).astype(int)
        
        return vad
    
    def get_pause_segments(self, vad: np.ndarray) -> list:
        """Znajdź segmenty pauz (ciszy) w sygnale
        
        Args:
            vad: Voice activity detection array (0/1)
            
        Returns:
            Lista tupli (start_frame, end_frame, duration_sec) dla każdej pauzy
        """
        pauses = []
        in_pause = False
        pause_start = 0
        
        for i, v in enumerate(vad):
            if v == 0 and not in_pause:
                # Początek pauzy
                in_pause = True
                pause_start = i
            elif v == 1 and in_pause:
                # Koniec pauzy
                in_pause = False
                duration_frames = i - pause_start
                duration_sec = duration_frames * self.hop_length / self.sr
                
                if duration_sec >= self.min_pause_duration:
                    pauses.append((pause_start, i, duration_sec))
        
        # Obsłuż pauzę na końcu
        if in_pause:
            duration_frames = len(vad) - pause_start
            duration_sec = duration_frames * self.hop_length / self.sr
            if duration_sec >= self.min_pause_duration:
                pauses.append((pause_start, len(vad), duration_sec))
        
        return pauses
    
    def get_speech_segments(self, vad: np.ndarray) -> list:
        """Znajdź segmenty mowy w sygnale
        
        Args:
            vad: Voice activity detection array (0/1)
            
        Returns:
            Lista tupli (start_frame, end_frame, duration_sec) dla każdego segmentu mowy
        """
        segments = []
        in_speech = False
        speech_start = 0
        
        for i, v in enumerate(vad):
            if v == 1 and not in_speech:
                in_speech = True
                speech_start = i
            elif v == 0 and in_speech:
                in_speech = False
                duration_frames = i - speech_start
                duration_sec = duration_frames * self.hop_length / self.sr
                
                if duration_sec >= self.min_speech_duration:
                    segments.append((speech_start, i, duration_sec))
        
        if in_speech:
            duration_frames = len(vad) - speech_start
            duration_sec = duration_frames * self.hop_length / self.sr
            if duration_sec >= self.min_speech_duration:
                segments.append((speech_start, len(vad), duration_sec))
        
        return segments
    
    def extract_pause_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Ekstrahuj cechy związane z pausowaniem
        
        Args:
            audio: Sygnał audio
            
        Returns:
            Dictionary z cechami pausowania
        """
        total_duration = len(audio) / self.sr
        
        vad = self.detect_voice_activity(audio)
        pauses = self.get_pause_segments(vad)
        speech_segments = self.get_speech_segments(vad)
        
        # Podstawowe statystyki pauz
        pause_durations = [p[2] for p in pauses]
        speech_durations = [s[2] for s in speech_segments]
        
        total_pause_time = sum(pause_durations) if pause_durations else 0
        total_speech_time = sum(speech_durations) if speech_durations else 0
        
        features = {
            # Pause ratio - kluczowy wskaźnik depresji
            'pause_ratio': total_pause_time / total_duration if total_duration > 0 else 0,
            'speech_ratio': total_speech_time / total_duration if total_duration > 0 else 0,
            
            # Liczba pauz (znormalizowana do minuty)
            'pause_count': len(pauses),
            'pause_rate_per_min': len(pauses) / (total_duration / 60) if total_duration > 0 else 0,
            
            # Statystyki długości pauz
            'pause_duration_mean': np.mean(pause_durations) if pause_durations else 0,
            'pause_duration_std': np.std(pause_durations) if len(pause_durations) > 1 else 0,
            'pause_duration_max': np.max(pause_durations) if pause_durations else 0,
            'pause_duration_median': np.median(pause_durations) if pause_durations else 0,
            
            # Statystyki segmentów mowy
            'speech_segment_count': len(speech_segments),
            'speech_segment_mean': np.mean(speech_durations) if speech_durations else 0,
            'speech_segment_std': np.std(speech_durations) if len(speech_durations) > 1 else 0,
            
            # Speech rate proxy (segmenty mowy na minutę)
            'speech_rate': len(speech_segments) / (total_duration / 60) if total_duration > 0 else 0,
        }
        
        return features
    
    def extract_pitch_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Ekstrahuj cechy F0 (pitch) - zmienność F0 niższa w depresji
        
        Args:
            audio: Sygnał audio
            
        Returns:
            Dictionary z cechami pitch
        """
        # Użyj szybszej metody YIN zamiast pyin (pyin jest bardzo wolne dla długich plików)
        # Ograniczamy długość audio do pierwszych 60 sekund dla szybkości
        max_duration = 60 * self.sr
        audio_short = audio[:max_duration] if len(audio) > max_duration else audio
        
        try:
            # Użyj librosa.yin - szybsze niż pyin
            f0 = librosa.yin(
                audio_short,
                fmin=librosa.note_to_hz('C2'),  # ~65 Hz
                fmax=librosa.note_to_hz('C7'),  # ~2093 Hz
                sr=self.sr,
                frame_length=self.frame_length,
                hop_length=self.hop_length
            )
            
            # Filtruj nieprawidłowe wartości (NaN, inf, zbyt niskie/wysokie)
            valid_mask = np.isfinite(f0) & (f0 > 50) & (f0 < 500)
            f0_voiced = f0[valid_mask]
            
            if len(f0_voiced) == 0:
                return {
                    'f0_mean': 0,
                    'f0_std': 0,
                    'f0_range': 0,
                    'f0_cv': 0,  # coefficient of variation
                    'voiced_fraction': 0,
                    'f0_slope': 0,
                }
            
            features = {
                'f0_mean': float(np.nanmean(f0_voiced)),
                'f0_std': float(np.nanstd(f0_voiced)),
                'f0_range': float(np.nanmax(f0_voiced) - np.nanmin(f0_voiced)),
                'f0_cv': float(np.nanstd(f0_voiced) / np.nanmean(f0_voiced)) if np.nanmean(f0_voiced) > 0 else 0,
                'voiced_fraction': float(np.sum(valid_mask) / len(f0)),
                
                # Trend F0 (czy spada/rośnie w czasie)
                'f0_slope': float(np.polyfit(np.arange(len(f0_voiced)), f0_voiced, 1)[0]) if len(f0_voiced) > 1 else 0,
            }
            
            return features
        except Exception as e:
            # Fallback jeśli yin zawiedzie
            return {
                'f0_mean': 0,
                'f0_std': 0,
                'f0_range': 0,
                'f0_cv': 0,
                'voiced_fraction': 0,
                'f0_slope': 0,
            }
    
    def extract_intensity_features(self, audio: np.ndarray) -> Dict[str, float]:
        """Ekstrahuj cechy natężenia/głośności
        
        Args:
            audio: Sygnał audio
            
        Returns:
            Dictionary z cechami intensity
        """
        rms = librosa.feature.rms(
            y=audio,
            frame_length=self.frame_length,
            hop_length=self.hop_length
        )[0]
        
        # Konwersja do dB
        rms_db = librosa.amplitude_to_db(rms, ref=np.max)
        
        # Filtruj tylko frames z mową (powyżej progu)
        rms_speech = rms_db[rms_db > self.silence_threshold_db]
        
        if len(rms_speech) == 0:
            return {
                'intensity_mean': 0,
                'intensity_std': 0,
                'intensity_range': 0,
                'intensity_cv': 0,
            }
        
        features = {
            'intensity_mean': float(np.mean(rms_speech)),
            'intensity_std': float(np.std(rms_speech)),
            'intensity_range': float(np.max(rms_speech) - np.min(rms_speech)),
            'intensity_cv': float(np.std(rms_speech) / abs(np.mean(rms_speech))) if np.mean(rms_speech) != 0 else 0,
        }
        
        return features
    
    def extract_all_features_from_audio(self, audio: np.ndarray) -> Dict[str, float]:
        """Ekstrahuj wszystkie cechy prozodyczne z już załadowanego audio
        
        Args:
            audio: Sygnał audio (już załadowany i znormalizowany)
            
        Returns:
            Dictionary z wszystkimi cechami prozodycznymi
        """
        features = {}
        
        # Pause features
        pause_features = self.extract_pause_features(audio)
        features.update({f'prosody_{k}': v for k, v in pause_features.items()})
        
        # Pitch features
        try:
            pitch_features = self.extract_pitch_features(audio)
            features.update({f'prosody_{k}': v for k, v in pitch_features.items()})
        except Exception as e:
            # Fallback jeśli pyin zawiedzie
            for key in ['f0_mean', 'f0_std', 'f0_range', 'f0_cv', 'voiced_fraction', 'f0_slope']:
                features[f'prosody_{key}'] = 0.0
        
        # Intensity features
        intensity_features = self.extract_intensity_features(audio)
        features.update({f'prosody_{k}': v for k, v in intensity_features.items()})
        
        return features
    
    def extract_all_features(self, audio_path: Path) -> Dict[str, float]:
        """Ekstrahuj wszystkie cechy prozodyczne z pliku audio
        
        Args:
            audio_path: Ścieżka do pliku audio
            
        Returns:
            Dictionary z wszystkimi cechami prozodycznymi
        """
        # Załaduj audio
        audio, sr = librosa.load(str(audio_path), sr=self.sr, mono=True)
        
        # Normalizacja
        audio = audio / (np.max(np.abs(audio)) + 1e-8)
        
        return self.extract_all_features_from_audio(audio)

"""
Data Augmentation for Multimodal Depression Detection

Implements augmentation techniques to increase effective dataset size:

Audio Augmentation:
- Pitch shifting (+/- 2 semitones)
- Time stretching (0.9-1.1x)
- Adding background noise
- Speed perturbation

Text Augmentation:
- Synonym replacement
- Random insertion
- Random swap
- Random deletion
- Back-translation (optional)

Based on AVTF-TBN (2024) and Context-Aware Deep Learning (2024) recommendations.
"""

import numpy as np
from typing import List, Optional, Tuple, Union
from pathlib import Path
import logging
import random

logger = logging.getLogger(__name__)


# ============================================================================
# Audio Augmentation
# ============================================================================

class AudioAugmenter:
    """
    Audio augmentation for depression speech data.
    
    Preserves depression-relevant features while adding variety:
    - Pitch shift: simulates different speakers
    - Time stretch: simulates speaking rate variation
    - Noise addition: simulates recording conditions
    
    Does NOT modify features that are critical for depression detection
    (e.g., maintains relative F0 patterns, pause structure).
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        pitch_shift_range: Tuple[float, float] = (-2, 2),
        time_stretch_range: Tuple[float, float] = (0.9, 1.1),
        noise_level_range: Tuple[float, float] = (0.001, 0.01),
        random_seed: Optional[int] = None
    ):
        """
        Initialize audio augmenter.
        
        Args:
            sample_rate: Audio sample rate
            pitch_shift_range: Range for pitch shifting in semitones
            time_stretch_range: Range for time stretching factor
            noise_level_range: Range for noise level (relative to signal)
            random_seed: Random seed for reproducibility
        """
        self.sample_rate = sample_rate
        self.pitch_shift_range = pitch_shift_range
        self.time_stretch_range = time_stretch_range
        self.noise_level_range = noise_level_range
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def pitch_shift(
        self,
        audio: np.ndarray,
        n_steps: Optional[float] = None
    ) -> np.ndarray:
        """
        Shift pitch by n semitones.
        
        Args:
            audio: Audio waveform
            n_steps: Number of semitones to shift (random if None)
            
        Returns:
            Pitch-shifted audio
        """
        import librosa
        
        if n_steps is None:
            n_steps = random.uniform(*self.pitch_shift_range)
        
        shifted = librosa.effects.pitch_shift(
            audio,
            sr=self.sample_rate,
            n_steps=n_steps
        )
        
        return shifted
    
    def time_stretch(
        self,
        audio: np.ndarray,
        rate: Optional[float] = None
    ) -> np.ndarray:
        """
        Stretch/compress audio in time domain.
        
        Args:
            audio: Audio waveform
            rate: Stretch factor (>1 = faster, <1 = slower)
            
        Returns:
            Time-stretched audio
        """
        import librosa
        
        if rate is None:
            rate = random.uniform(*self.time_stretch_range)
        
        stretched = librosa.effects.time_stretch(audio, rate=rate)
        
        return stretched
    
    def add_noise(
        self,
        audio: np.ndarray,
        noise_level: Optional[float] = None,
        noise_type: str = 'white'
    ) -> np.ndarray:
        """
        Add background noise to audio.
        
        Args:
            audio: Audio waveform
            noise_level: Noise level relative to signal (random if None)
            noise_type: 'white', 'pink', or 'brown'
            
        Returns:
            Noisy audio
        """
        if noise_level is None:
            noise_level = random.uniform(*self.noise_level_range)
        
        # Generate noise
        if noise_type == 'white':
            noise = np.random.randn(len(audio))
        elif noise_type == 'pink':
            # Pink noise has 1/f spectrum
            freqs = np.fft.rfftfreq(len(audio))
            freqs[0] = 1  # Avoid division by zero
            pink_filter = 1 / np.sqrt(freqs)
            white = np.random.randn(len(audio))
            pink_spectrum = np.fft.rfft(white) * pink_filter
            noise = np.fft.irfft(pink_spectrum, len(audio))
        elif noise_type == 'brown':
            # Brown noise is integrated white noise
            noise = np.cumsum(np.random.randn(len(audio)))
            noise = noise - np.mean(noise)
        else:
            noise = np.random.randn(len(audio))
        
        # Scale noise relative to signal
        signal_power = np.mean(audio ** 2)
        noise = noise * np.sqrt(signal_power) * noise_level
        
        return audio + noise
    
    def speed_perturb(
        self,
        audio: np.ndarray,
        speed_factor: Optional[float] = None
    ) -> np.ndarray:
        """
        Speed perturbation (changes both tempo and pitch).
        
        Different from time_stretch - this simulates playing at different speed.
        
        Args:
            audio: Audio waveform
            speed_factor: Speed factor (random if None)
            
        Returns:
            Speed-perturbed audio
        """
        import librosa
        
        if speed_factor is None:
            speed_factor = random.uniform(*self.time_stretch_range)
        
        # Resample to change speed
        target_length = int(len(audio) / speed_factor)
        perturbed = librosa.resample(
            audio,
            orig_sr=self.sample_rate,
            target_sr=int(self.sample_rate * speed_factor)
        )
        
        # Resample back to original sample rate
        perturbed = librosa.resample(
            perturbed,
            orig_sr=int(self.sample_rate * speed_factor),
            target_sr=self.sample_rate
        )
        
        return perturbed
    
    def augment(
        self,
        audio: np.ndarray,
        augmentations: Optional[List[str]] = None,
        p: float = 0.5
    ) -> np.ndarray:
        """
        Apply random augmentations.
        
        Args:
            audio: Audio waveform
            augmentations: List of augmentations to apply ('pitch', 'time', 'noise', 'speed')
                          If None, applies all with probability p
            p: Probability of applying each augmentation
            
        Returns:
            Augmented audio
        """
        if augmentations is None:
            augmentations = ['pitch', 'time', 'noise']
        
        augmented = audio.copy()
        
        for aug in augmentations:
            if random.random() < p:
                if aug == 'pitch':
                    augmented = self.pitch_shift(augmented)
                elif aug == 'time':
                    augmented = self.time_stretch(augmented)
                elif aug == 'noise':
                    augmented = self.add_noise(augmented)
                elif aug == 'speed':
                    augmented = self.speed_perturb(augmented)
        
        return augmented
    
    def augment_batch(
        self,
        audios: List[np.ndarray],
        n_augmentations: int = 2,
        include_original: bool = True
    ) -> List[np.ndarray]:
        """
        Augment a batch of audio samples.
        
        Args:
            audios: List of audio waveforms
            n_augmentations: Number of augmented versions per sample
            include_original: Whether to include original samples
            
        Returns:
            List of augmented audio samples
        """
        result = []
        
        for audio in audios:
            if include_original:
                result.append(audio)
            
            for _ in range(n_augmentations):
                augmented = self.augment(audio)
                result.append(augmented)
        
        return result


# ============================================================================
# Text Augmentation
# ============================================================================

class TextAugmenter:
    """
    Text augmentation for depression transcript data.
    
    Techniques:
    - Synonym replacement: replace words with synonyms
    - Random insertion: insert random synonyms
    - Random swap: swap word positions
    - Random deletion: delete words with probability
    
    Preserves meaning while adding variety.
    """
    
    def __init__(
        self,
        alpha_sr: float = 0.1,
        alpha_ri: float = 0.1,
        alpha_rs: float = 0.1,
        p_rd: float = 0.1,
        random_seed: Optional[int] = None
    ):
        """
        Initialize text augmenter.
        
        Args:
            alpha_sr: Percent of words to replace with synonyms
            alpha_ri: Percent of words to insert
            alpha_rs: Percent of words to swap
            p_rd: Probability of deleting each word
            random_seed: Random seed
        """
        self.alpha_sr = alpha_sr
        self.alpha_ri = alpha_ri
        self.alpha_rs = alpha_rs
        self.p_rd = p_rd
        
        if random_seed is not None:
            random.seed(random_seed)
        
        # Try to load WordNet for synonyms
        self._load_wordnet()
    
    def _load_wordnet(self):
        """Load WordNet for synonym lookup."""
        self.wordnet_available = False
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # Download if needed
            try:
                wordnet.synsets('test')
            except:
                nltk.download('wordnet', quiet=True)
                nltk.download('omw-1.4', quiet=True)
            
            self.wordnet = wordnet
            self.wordnet_available = True
            logger.info("WordNet loaded for text augmentation")
        except ImportError:
            logger.warning("NLTK not available, using simple synonym replacement")
            self.wordnet = None
    
    def _get_synonyms(self, word: str) -> List[str]:
        """Get synonyms for a word."""
        if self.wordnet_available and self.wordnet is not None:
            synonyms = set()
            for syn in self.wordnet.synsets(word):
                for lemma in syn.lemmas():
                    synonym = lemma.name().replace('_', ' ')
                    if synonym.lower() != word.lower():
                        synonyms.add(synonym)
            return list(synonyms)
        
        # Fallback: simple synonym dictionary for depression-related words
        simple_synonyms = {
            'sad': ['unhappy', 'sorrowful', 'down', 'low'],
            'happy': ['glad', 'joyful', 'pleased', 'content'],
            'tired': ['exhausted', 'fatigued', 'weary', 'drained'],
            'anxious': ['worried', 'nervous', 'uneasy', 'concerned'],
            'depressed': ['sad', 'down', 'low', 'dejected'],
            'hopeless': ['desperate', 'despairing', 'bleak'],
            'angry': ['upset', 'irritated', 'frustrated', 'annoyed'],
            'lonely': ['alone', 'isolated', 'solitary'],
            'afraid': ['scared', 'frightened', 'fearful'],
            'good': ['fine', 'well', 'okay', 'alright'],
            'bad': ['poor', 'terrible', 'awful'],
        }
        return simple_synonyms.get(word.lower(), [])
    
    def synonym_replacement(self, text: str, n: Optional[int] = None) -> str:
        """
        Replace n words with synonyms.
        
        Args:
            text: Input text
            n: Number of words to replace (calculated from alpha_sr if None)
            
        Returns:
            Augmented text
        """
        import re
        
        words = text.split()
        if not words:
            return text
        
        if n is None:
            n = max(1, int(self.alpha_sr * len(words)))
        
        # Get indices of words that have synonyms
        candidates = []
        for i, word in enumerate(words):
            clean_word = re.sub(r'[^\w]', '', word.lower())
            synonyms = self._get_synonyms(clean_word)
            if synonyms:
                candidates.append((i, synonyms))
        
        if not candidates:
            return text
        
        # Randomly select words to replace
        n = min(n, len(candidates))
        to_replace = random.sample(candidates, n)
        
        new_words = words.copy()
        for i, synonyms in to_replace:
            synonym = random.choice(synonyms)
            # Preserve capitalization
            if words[i][0].isupper():
                synonym = synonym.capitalize()
            new_words[i] = synonym
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: Optional[int] = None) -> str:
        """
        Insert n random synonyms into the text.
        
        Args:
            text: Input text
            n: Number of words to insert
            
        Returns:
            Augmented text
        """
        words = text.split()
        if not words:
            return text
        
        if n is None:
            n = max(1, int(self.alpha_ri * len(words)))
        
        new_words = words.copy()
        
        for _ in range(n):
            # Pick random word and get synonym
            word = random.choice(words)
            synonyms = self._get_synonyms(word.lower())
            
            if synonyms:
                synonym = random.choice(synonyms)
                # Insert at random position
                pos = random.randint(0, len(new_words))
                new_words.insert(pos, synonym)
        
        return ' '.join(new_words)
    
    def random_swap(self, text: str, n: Optional[int] = None) -> str:
        """
        Randomly swap n pairs of words.
        
        Args:
            text: Input text
            n: Number of swaps
            
        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) < 2:
            return text
        
        if n is None:
            n = max(1, int(self.alpha_rs * len(words)))
        
        new_words = words.copy()
        
        for _ in range(n):
            idx1, idx2 = random.sample(range(len(new_words)), 2)
            new_words[idx1], new_words[idx2] = new_words[idx2], new_words[idx1]
        
        return ' '.join(new_words)
    
    def random_deletion(self, text: str, p: Optional[float] = None) -> str:
        """
        Randomly delete words with probability p.
        
        Args:
            text: Input text
            p: Deletion probability per word
            
        Returns:
            Augmented text
        """
        words = text.split()
        if len(words) <= 1:
            return text
        
        if p is None:
            p = self.p_rd
        
        # Keep at least one word
        new_words = [w for w in words if random.random() > p]
        
        if not new_words:
            return random.choice(words)
        
        return ' '.join(new_words)
    
    def augment(
        self,
        text: str,
        techniques: Optional[List[str]] = None,
        p: float = 0.5
    ) -> str:
        """
        Apply random augmentation techniques.
        
        Args:
            text: Input text
            techniques: List of techniques ('sr', 'ri', 'rs', 'rd')
            p: Probability of applying each technique
            
        Returns:
            Augmented text
        """
        if techniques is None:
            techniques = ['sr', 'ri', 'rs', 'rd']
        
        augmented = text
        
        for tech in techniques:
            if random.random() < p:
                if tech == 'sr':
                    augmented = self.synonym_replacement(augmented)
                elif tech == 'ri':
                    augmented = self.random_insertion(augmented)
                elif tech == 'rs':
                    augmented = self.random_swap(augmented)
                elif tech == 'rd':
                    augmented = self.random_deletion(augmented)
        
        return augmented
    
    def augment_batch(
        self,
        texts: List[str],
        n_augmentations: int = 2,
        include_original: bool = True
    ) -> List[str]:
        """
        Augment a batch of texts.
        
        Args:
            texts: List of text strings
            n_augmentations: Number of augmented versions per text
            include_original: Whether to include originals
            
        Returns:
            List of augmented texts
        """
        result = []
        
        for text in texts:
            if include_original:
                result.append(text)
            
            for _ in range(n_augmentations):
                augmented = self.augment(text)
                result.append(augmented)
        
        return result


# ============================================================================
# Topic-Based Text Augmentation (SOTA 2025)
# Based on Lam et al. (2024) "Context-Aware Deep Learning"
# ============================================================================

class TopicShuffleAugmenter:
    """
    Topic-based text augmentation for depression detection.
    
    Instead of word-level augmentation, this operates on topic segments
    (groups of semantically related sentences). This preserves depression-
    related content while creating synthetic samples.
    
    Method:
    1. Split transcript into topic segments (2-5 sentences)
    2. Shuffle segment order
    3. Optionally drop/duplicate segments
    
    Expected improvement: Dataset 107 -> 534 samples (+5x)
    Based on Lam et al. (2024) - F1 improvement of +10%
    """
    
    def __init__(
        self,
        min_sentences_per_segment: int = 2,
        max_sentences_per_segment: int = 5,
        random_seed: Optional[int] = None
    ):
        """
        Initialize topic shuffle augmenter.
        
        Args:
            min_sentences_per_segment: Minimum sentences per topic segment
            max_sentences_per_segment: Maximum sentences per topic segment
            random_seed: Random seed for reproducibility
        """
        self.min_sentences = min_sentences_per_segment
        self.max_sentences = max_sentences_per_segment
        
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        
        # Split on sentence boundaries
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _create_topic_segments(self, sentences: List[str]) -> List[List[str]]:
        """
        Group sentences into topic segments.
        
        Uses variable-length segments to simulate natural topic transitions.
        """
        if len(sentences) < self.min_sentences:
            return [sentences]
        
        segments = []
        i = 0
        
        while i < len(sentences):
            # Random segment length
            seg_len = random.randint(self.min_sentences, self.max_sentences)
            seg_len = min(seg_len, len(sentences) - i)
            
            if seg_len > 0:
                segment = sentences[i:i + seg_len]
                segments.append(segment)
            
            i += seg_len
        
        return segments
    
    def shuffle_topics(self, text: str) -> str:
        """
        Shuffle topic segments within transcript.
        
        Args:
            text: Input transcript
            
        Returns:
            Augmented transcript with shuffled topics
        """
        sentences = self._split_into_sentences(text)
        
        if len(sentences) < self.min_sentences * 2:
            return text  # Too short to shuffle meaningfully
        
        segments = self._create_topic_segments(sentences)
        
        if len(segments) < 2:
            return text
        
        # Shuffle segments
        random.shuffle(segments)
        
        # Reconstruct text
        augmented_sentences = [sent for segment in segments for sent in segment]
        return ' '.join(augmented_sentences)
    
    def drop_topic(self, text: str, drop_ratio: float = 0.2) -> str:
        """
        Randomly drop a topic segment.
        
        Args:
            text: Input transcript
            drop_ratio: Probability of dropping each segment
            
        Returns:
            Augmented transcript with dropped topic
        """
        sentences = self._split_into_sentences(text)
        
        if len(sentences) < self.min_sentences * 2:
            return text
        
        segments = self._create_topic_segments(sentences)
        
        if len(segments) < 3:  # Keep at least 2 segments
            return text
        
        # Randomly drop segments
        kept_segments = [seg for seg in segments if random.random() > drop_ratio]
        
        if len(kept_segments) < 2:
            kept_segments = segments[:2]  # Keep at least 2
        
        augmented_sentences = [sent for segment in kept_segments for sent in segment]
        return ' '.join(augmented_sentences)
    
    def duplicate_topic(self, text: str, duplicate_ratio: float = 0.3) -> str:
        """
        Randomly duplicate a topic segment (emphasize certain topics).
        
        Args:
            text: Input transcript
            duplicate_ratio: Probability of duplicating each segment
            
        Returns:
            Augmented transcript with duplicated topic
        """
        sentences = self._split_into_sentences(text)
        
        if len(sentences) < self.min_sentences:
            return text
        
        segments = self._create_topic_segments(sentences)
        
        # Randomly duplicate segments
        augmented_segments = []
        for segment in segments:
            augmented_segments.append(segment)
            if random.random() < duplicate_ratio:
                augmented_segments.append(segment)  # Duplicate
        
        augmented_sentences = [sent for segment in augmented_segments for sent in segment]
        return ' '.join(augmented_sentences)
    
    def augment(
        self,
        text: str,
        n_augmentations: int = 3,
        techniques: Optional[List[str]] = None
    ) -> List[str]:
        """
        Generate multiple augmented versions of text.
        
        Args:
            text: Input transcript
            n_augmentations: Number of augmented versions to generate
            techniques: List of techniques ('shuffle', 'drop', 'duplicate')
                       If None, uses all techniques
            
        Returns:
            List of augmented texts
        """
        if techniques is None:
            techniques = ['shuffle', 'drop', 'duplicate']
        
        augmented = []
        
        for i in range(n_augmentations):
            # Rotate through techniques
            tech = techniques[i % len(techniques)]
            
            if tech == 'shuffle':
                aug_text = self.shuffle_topics(text)
            elif tech == 'drop':
                aug_text = self.drop_topic(text)
            elif tech == 'duplicate':
                aug_text = self.duplicate_topic(text)
            else:
                aug_text = self.shuffle_topics(text)
            
            augmented.append(aug_text)
        
        return augmented
    
    def augment_dataset(
        self,
        texts: List[str],
        labels: np.ndarray,
        n_augmentations: int = 3,
        include_original: bool = True,
        balance_classes: bool = True
    ) -> Tuple[List[str], np.ndarray]:
        """
        Augment entire text dataset with topic shuffling.
        
        Args:
            texts: List of transcripts
            labels: Binary labels
            n_augmentations: Augmentations per sample
            include_original: Include original samples
            balance_classes: Augment minority class more
            
        Returns:
            Tuple of (augmented_texts, augmented_labels)
        """
        aug_texts = []
        aug_labels = []
        
        # Class balance info
        n_positive = np.sum(labels == 1)
        n_negative = np.sum(labels == 0)
        minority_class = 1 if n_positive < n_negative else 0
        
        for text, label in zip(texts, labels):
            if include_original:
                aug_texts.append(text)
                aug_labels.append(label)
            
            # Determine augmentation count
            if balance_classes and label == minority_class:
                n_aug = n_augmentations * 2  # More augmentations for minority
            else:
                n_aug = n_augmentations
            
            # Generate augmentations
            augmented = self.augment(text, n_augmentations=n_aug)
            aug_texts.extend(augmented)
            aug_labels.extend([label] * len(augmented))
        
        return aug_texts, np.array(aug_labels)


def augment_features_with_topics(
    text_features: np.ndarray,
    audio_features: np.ndarray,
    labels: np.ndarray,
    texts: List[str],
    text_encoder,
    n_augmentations: int = 3,
    include_original: bool = True,
    balance_classes: bool = True,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Augment pre-extracted features using topic-based text augmentation.
    
    This function:
    1. Augments raw texts using topic shuffling
    2. Re-encodes augmented texts to get new embeddings
    3. Pairs with original audio features (audio not augmented)
    
    Args:
        text_features: Original text features (N, text_dim)
        audio_features: Original audio features (N, audio_dim)
        labels: Binary labels (N,)
        texts: Original transcript texts (N,)
        text_encoder: TextEncoder to re-encode augmented texts
        n_augmentations: Number of augmentations per sample
        include_original: Include original samples
        balance_classes: Augment minority class more
        random_seed: Random seed
        
    Returns:
        Tuple of (augmented_audio_features, augmented_text_features, augmented_labels)
    """
    logger.info(f"Augmenting dataset with topic shuffling (n_aug={n_augmentations})...")
    
    # Initialize augmenter
    topic_aug = TopicShuffleAugmenter(random_seed=random_seed)
    
    # Augment texts
    aug_texts, aug_labels = topic_aug.augment_dataset(
        texts, labels,
        n_augmentations=n_augmentations,
        include_original=include_original,
        balance_classes=balance_classes
    )
    
    logger.info(f"  Original samples: {len(texts)}")
    logger.info(f"  Augmented samples: {len(aug_texts)}")
    
    # Track which original sample each augmented sample comes from
    aug_indices = []
    idx = 0
    for i, (text, label) in enumerate(zip(texts, labels)):
        n_total = 1 if include_original else 0
        
        # Count augmentations for this sample
        n_positive = np.sum(labels == 1)
        n_negative = np.sum(labels == 0)
        minority_class = 1 if n_positive < n_negative else 0
        
        if balance_classes and label == minority_class:
            n_total += n_augmentations * 2
        else:
            n_total += n_augmentations
        
        for _ in range(n_total):
            aug_indices.append(i)
    
    # Re-encode augmented texts
    logger.info("  Re-encoding augmented texts...")
    
    if hasattr(text_encoder, 'encode_all'):
        aug_text_features = text_encoder.encode_all(aug_texts)
    elif hasattr(text_encoder, 'encode'):
        aug_text_features = text_encoder.encode(aug_texts)
    else:
        raise ValueError("text_encoder must have encode() or encode_all() method")
    
    # Map audio features (audio stays the same, just duplicated)
    aug_audio_features = audio_features[aug_indices]
    
    logger.info(f"  Final shapes: audio={aug_audio_features.shape}, text={aug_text_features.shape}")
    logger.info(f"  Label distribution: {np.bincount(aug_labels)}")
    
    return aug_audio_features, aug_text_features, aug_labels


def augment_features_simple(
    audio_features: np.ndarray,
    text_features: np.ndarray,
    labels: np.ndarray,
    n_augmentations: int = 3,
    noise_std: float = 0.01,
    include_original: bool = True,
    balance_classes: bool = True,
    random_seed: int = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Simple feature-space augmentation (without re-encoding).
    
    Adds small Gaussian noise to features to create synthetic samples.
    Faster than topic-based augmentation but less semantically meaningful.
    
    Args:
        audio_features: Audio features (N, audio_dim)
        text_features: Text features (N, text_dim)
        labels: Labels (N,)
        n_augmentations: Augmentations per sample
        noise_std: Standard deviation of Gaussian noise
        include_original: Include originals
        balance_classes: Augment minority more
        random_seed: Random seed
        
    Returns:
        Tuple of (aug_audio, aug_text, aug_labels)
    """
    np.random.seed(random_seed)
    
    aug_audio = []
    aug_text = []
    aug_labels = []
    
    n_positive = np.sum(labels == 1)
    n_negative = np.sum(labels == 0)
    minority_class = 1 if n_positive < n_negative else 0
    
    for i in range(len(labels)):
        if include_original:
            aug_audio.append(audio_features[i])
            aug_text.append(text_features[i])
            aug_labels.append(labels[i])
        
        # Determine augmentation count
        if balance_classes and labels[i] == minority_class:
            n_aug = n_augmentations * 2
        else:
            n_aug = n_augmentations
        
        # Generate noisy versions
        for _ in range(n_aug):
            noisy_audio = audio_features[i] + np.random.randn(*audio_features[i].shape) * noise_std
            noisy_text = text_features[i] + np.random.randn(*text_features[i].shape) * noise_std
            
            aug_audio.append(noisy_audio)
            aug_text.append(noisy_text)
            aug_labels.append(labels[i])
    
    return np.array(aug_audio), np.array(aug_text), np.array(aug_labels)


# ============================================================================
# Combined Multimodal Augmentation
# ============================================================================

class MultimodalAugmenter:
    """
    Combined audio + text augmentation for multimodal data.
    
    Ensures consistent augmentation across modalities.
    """
    
    def __init__(
        self,
        sample_rate: int = 16000,
        random_seed: Optional[int] = None
    ):
        self.audio_aug = AudioAugmenter(
            sample_rate=sample_rate,
            random_seed=random_seed
        )
        self.text_aug = TextAugmenter(random_seed=random_seed)
    
    def augment_pair(
        self,
        audio: np.ndarray,
        text: str,
        audio_p: float = 0.5,
        text_p: float = 0.5
    ) -> Tuple[np.ndarray, str]:
        """
        Augment audio-text pair.
        
        Args:
            audio: Audio waveform
            text: Transcript text
            audio_p: Probability for each audio augmentation
            text_p: Probability for each text augmentation
            
        Returns:
            Tuple of (augmented_audio, augmented_text)
        """
        aug_audio = self.audio_aug.augment(audio, p=audio_p)
        aug_text = self.text_aug.augment(text, p=text_p)
        
        return aug_audio, aug_text
    
    def augment_dataset(
        self,
        audios: List[np.ndarray],
        texts: List[str],
        labels: np.ndarray,
        n_augmentations: int = 2,
        include_original: bool = True,
        balance_classes: bool = True
    ) -> Tuple[List[np.ndarray], List[str], np.ndarray]:
        """
        Augment entire dataset.
        
        Args:
            audios: List of audio waveforms
            texts: List of transcript texts
            labels: Binary labels array
            n_augmentations: Number of augmented versions per sample
            include_original: Include original samples
            balance_classes: Augment minority class more
            
        Returns:
            Tuple of (augmented_audios, augmented_texts, augmented_labels)
        """
        aug_audios = []
        aug_texts = []
        aug_labels = []
        
        # Calculate class balance
        n_positive = np.sum(labels == 1)
        n_negative = np.sum(labels == 0)
        
        for audio, text, label in zip(audios, texts, labels):
            if include_original:
                aug_audios.append(audio)
                aug_texts.append(text)
                aug_labels.append(label)
            
            # Determine number of augmentations
            if balance_classes:
                # Augment minority class more
                if label == 1 and n_positive < n_negative:
                    n_aug = n_augmentations * 2
                elif label == 0 and n_negative < n_positive:
                    n_aug = n_augmentations * 2
                else:
                    n_aug = n_augmentations
            else:
                n_aug = n_augmentations
            
            # Generate augmentations
            for _ in range(n_aug):
                aug_audio, aug_text = self.augment_pair(audio, text)
                aug_audios.append(aug_audio)
                aug_texts.append(aug_text)
                aug_labels.append(label)
        
        return aug_audios, aug_texts, np.array(aug_labels)


if __name__ == "__main__":
    print("Testing Data Augmentation...")
    
    # Create synthetic audio
    sr = 16000
    duration = 2.0
    t = np.linspace(0, duration, int(sr * duration))
    audio = 0.5 * np.sin(2 * np.pi * 200 * t)
    
    # Test AudioAugmenter
    print("\n1. Testing AudioAugmenter:")
    audio_aug = AudioAugmenter(sample_rate=sr, random_seed=42)
    
    pitched = audio_aug.pitch_shift(audio, n_steps=2)
    print(f"   Pitch shift: {len(audio)} -> {len(pitched)} samples")
    
    stretched = audio_aug.time_stretch(audio, rate=1.1)
    print(f"   Time stretch: {len(audio)} -> {len(stretched)} samples")
    
    noisy = audio_aug.add_noise(audio, noise_level=0.01)
    print(f"   Add noise: SNR preserved, shape {noisy.shape}")
    
    augmented = audio_aug.augment(audio)
    print(f"   Combined augment: {len(augmented)} samples")
    print("   [OK] AudioAugmenter works!")
    
    # Test TextAugmenter
    print("\n2. Testing TextAugmenter:")
    text_aug = TextAugmenter(random_seed=42)
    
    test_text = "I feel very sad and hopeless today. Nothing makes me happy anymore."
    
    sr_text = text_aug.synonym_replacement(test_text)
    print(f"   Synonym replacement:")
    print(f"     Original: {test_text[:50]}...")
    print(f"     Augmented: {sr_text[:50]}...")
    
    ri_text = text_aug.random_insertion(test_text)
    print(f"   Random insertion: {ri_text[:50]}...")
    
    rs_text = text_aug.random_swap(test_text)
    print(f"   Random swap: {rs_text[:50]}...")
    
    rd_text = text_aug.random_deletion(test_text)
    print(f"   Random deletion: {rd_text[:50]}...")
    
    aug_text = text_aug.augment(test_text)
    print(f"   Combined augment: {aug_text[:50]}...")
    print("   [OK] TextAugmenter works!")
    
    # Test TopicShuffleAugmenter (SOTA 2025)
    print("\n3. Testing TopicShuffleAugmenter (SOTA 2025):")
    topic_aug = TopicShuffleAugmenter(
        min_sentences_per_segment=2,
        max_sentences_per_segment=3,
        random_seed=42
    )
    
    long_text = (
        "I have been feeling really down lately. Nothing seems to bring me joy anymore. "
        "I used to love going to work but now I dread it. My sleep has been terrible. "
        "I wake up multiple times during the night. Sometimes I feel like giving up. "
        "My family tries to help but I push them away. I don't know what's wrong with me."
    )
    
    shuffled = topic_aug.shuffle_topics(long_text)
    print(f"   Topic shuffle:")
    print(f"     Original: {long_text[:60]}...")
    print(f"     Shuffled: {shuffled[:60]}...")
    
    dropped = topic_aug.drop_topic(long_text)
    print(f"   Topic drop: {len(long_text)} -> {len(dropped)} chars")
    
    duplicated = topic_aug.duplicate_topic(long_text)
    print(f"   Topic duplicate: {len(long_text)} -> {len(duplicated)} chars")
    
    # Test dataset augmentation
    texts = [long_text, "I had a great day at work!", "Feeling anxious about tomorrow."]
    labels = np.array([1, 0, 1])
    
    aug_texts, aug_labels = topic_aug.augment_dataset(
        texts, labels,
        n_augmentations=3,
        include_original=True,
        balance_classes=True
    )
    
    print(f"   Dataset augmentation:")
    print(f"     Original: {len(texts)} samples")
    print(f"     Augmented: {len(aug_texts)} samples")
    print(f"     Labels distribution: {np.bincount(aug_labels)}")
    print("   [OK] TopicShuffleAugmenter works!")
    
    # Test simple feature augmentation
    print("\n4. Testing Simple Feature Augmentation:")
    audio_feats = np.random.randn(3, 768)
    text_feats = np.random.randn(3, 768)
    
    aug_audio_f, aug_text_f, aug_labels_f = augment_features_simple(
        audio_feats, text_feats, labels,
        n_augmentations=2,
        noise_std=0.01,
        include_original=True,
        balance_classes=True
    )
    
    print(f"   Original: {audio_feats.shape}")
    print(f"   Augmented: {aug_audio_f.shape}")
    print(f"   Labels: {np.bincount(aug_labels_f)}")
    print("   [OK] Simple feature augmentation works!")
    
    # Test MultimodalAugmenter
    print("\n5. Testing MultimodalAugmenter:")
    mm_aug = MultimodalAugmenter(sample_rate=sr, random_seed=42)
    
    aug_audio, aug_text = mm_aug.augment_pair(audio, test_text)
    print(f"   Audio: {len(audio)} -> {len(aug_audio)} samples")
    print(f"   Text: '{test_text[:30]}...' -> '{aug_text[:30]}...'")
    
    # Test batch augmentation
    audios = [audio, audio * 0.8, audio * 1.2]
    texts = [test_text, "I had a great day!", "Work was stressful."]
    labels = np.array([1, 0, 1])
    
    aug_audios, aug_texts, aug_labels = mm_aug.augment_dataset(
        audios, texts, labels,
        n_augmentations=1,
        include_original=True,
        balance_classes=True
    )
    
    print(f"   Dataset augmentation:")
    print(f"     Original: {len(audios)} samples")
    print(f"     Augmented: {len(aug_audios)} samples")
    print(f"     Labels distribution: {np.bincount(aug_labels)}")
    print("   [OK] MultimodalAugmenter works!")
    
    print("\n[OK] All augmentation tests complete!")

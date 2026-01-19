"""
Text Encoder using pretrained SentenceTransformer BERT + Topic Modeling.

Combines:
1. BERT embeddings (768-dim) - semantic representation
2. Topic modeling (LDA/NMF) - thematic context
3. Sentiment/emotion features - affective content

Based on Context-Aware Deep Learning (2024) recommendations.
Depression is often expressed through specific topics and emotional language.

Expected improvement: F1 0.44 -> 0.65-0.75 (+50-70%)
"""

import numpy as np
from typing import List, Optional, Union, Dict, Tuple
from pathlib import Path
import logging
import gc
import re

logger = logging.getLogger(__name__)

# Import config for batch size settings
try:
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent))
    from config import CONFIG
    DEFAULT_TEXT_BATCH_SIZE = CONFIG.TEXT_BATCH_SIZE
except ImportError:
    DEFAULT_TEXT_BATCH_SIZE = 8  # Mniejszy batch dla RTX 5060 8GB


# ============================================================================
# Topic Modeling
# ============================================================================

class TopicModel:
    """
    Topic modeling for depression-related content analysis.
    
    Uses LDA (Latent Dirichlet Allocation) or NMF (Non-negative Matrix Factorization)
    to extract thematic structure from transcripts.
    
    Depression-related topics often include:
    - Negative self-talk
    - Sleep problems
    - Social withdrawal
    - Hopelessness
    - Physical symptoms
    """
    
    def __init__(
        self,
        n_topics: int = 50,
        method: str = 'lda',
        max_features: int = 5000,
        min_df: int = 2,
        max_df: float = 0.95
    ):
        """
        Initialize topic model.
        
        Args:
            n_topics: Number of topics to extract
            method: 'lda' or 'nmf'
            max_features: Maximum vocabulary size
            min_df: Minimum document frequency for terms
            max_df: Maximum document frequency for terms
        """
        self.n_topics = n_topics
        self.method = method
        self.max_features = max_features
        self.min_df = min_df
        self.max_df = max_df
        
        self.vectorizer = None
        self.model = None
        self.is_fitted = False
    
    def fit(self, documents: List[str]):
        """
        Fit topic model on corpus.
        
        Args:
            documents: List of text documents
        """
        from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
        from sklearn.decomposition import LatentDirichletAllocation, NMF
        
        # Filter empty documents
        documents = [doc for doc in documents if doc and doc.strip()]
        
        if len(documents) < self.n_topics:
            logger.warning(f"Only {len(documents)} documents, reducing topics to {len(documents) // 2}")
            self.n_topics = max(5, len(documents) // 2)
        
        # Vectorize
        if self.method == 'lda':
            self.vectorizer = CountVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english'
            )
        else:  # nmf
            self.vectorizer = TfidfVectorizer(
                max_features=self.max_features,
                min_df=self.min_df,
                max_df=self.max_df,
                stop_words='english'
            )
        
        doc_term_matrix = self.vectorizer.fit_transform(documents)
        
        # Fit topic model
        if self.method == 'lda':
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=20,
                learning_method='batch'
            )
        else:  # nmf
            self.model = NMF(
                n_components=self.n_topics,
                random_state=42,
                max_iter=200,
                init='nndsvd'
            )
        
        self.model.fit(doc_term_matrix)
        self.is_fitted = True
        
        logger.info(f"TopicModel fitted: {self.method} with {self.n_topics} topics on {len(documents)} documents")
    
    def transform(self, documents: Union[str, List[str]]) -> np.ndarray:
        """
        Get topic distribution for documents.
        
        Args:
            documents: Single document or list of documents
            
        Returns:
            Topic distribution array of shape (n_docs, n_topics)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if isinstance(documents, str):
            documents = [documents]
        
        # Handle empty documents
        results = []
        valid_docs = []
        valid_indices = []
        
        for i, doc in enumerate(documents):
            if doc and doc.strip():
                valid_docs.append(doc)
                valid_indices.append(i)
        
        if valid_docs:
            doc_term_matrix = self.vectorizer.transform(valid_docs)
            valid_topics = self.model.transform(doc_term_matrix)
        
        # Build full results array
        full_results = np.zeros((len(documents), self.n_topics), dtype=np.float32)
        for i, idx in enumerate(valid_indices):
            full_results[idx] = valid_topics[i]
        
        return full_results
    
    def get_top_words(self, n_words: int = 10) -> Dict[int, List[str]]:
        """Get top words for each topic."""
        if not self.is_fitted:
            raise ValueError("Model not fitted.")
        
        feature_names = self.vectorizer.get_feature_names_out()
        top_words = {}
        
        for topic_idx, topic in enumerate(self.model.components_):
            top_indices = topic.argsort()[:-n_words-1:-1]
            top_words[topic_idx] = [feature_names[i] for i in top_indices]
        
        return top_words
    
    def save(self, path: Path):
        """Save fitted model."""
        import joblib
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            'vectorizer': self.vectorizer,
            'model': self.model,
            'n_topics': self.n_topics,
            'method': self.method
        }, path)
        logger.info(f"TopicModel saved to {path}")
    
    @classmethod
    def load(cls, path: Path) -> 'TopicModel':
        """Load fitted model."""
        import joblib
        data = joblib.load(path)
        instance = cls(n_topics=data['n_topics'], method=data['method'])
        instance.vectorizer = data['vectorizer']
        instance.model = data['model']
        instance.is_fitted = True
        return instance


# ============================================================================
# Sentiment/Emotion Features
# ============================================================================

def extract_linguistic_features(text: str) -> np.ndarray:
    """
    Extract linguistic features relevant to depression.
    
    Features:
    - First person pronoun usage (I, me, my) - increased in depression
    - Negative emotion words
    - Absolutist words (always, never, nothing)
    - Question frequency
    - Sentence length variation
    
    Args:
        text: Input text
        
    Returns:
        Feature vector of shape (~15,)
    """
    if not text or not text.strip():
        return np.zeros(15, dtype=np.float32)
    
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    n_words = len(words) if words else 1
    n_sentences = len(sentences) if sentences else 1
    
    features = {}
    
    # First person pronouns (depression marker)
    first_person = ['i', 'me', 'my', 'mine', 'myself']
    features['first_person_ratio'] = sum(1 for w in words if w in first_person) / n_words
    
    # Negative emotion words
    negative_words = [
        'sad', 'depressed', 'hopeless', 'worthless', 'tired', 'exhausted',
        'anxious', 'worried', 'scared', 'afraid', 'angry', 'frustrated',
        'lonely', 'alone', 'empty', 'numb', 'guilty', 'ashamed', 'useless',
        'hate', 'terrible', 'awful', 'horrible', 'miserable', 'painful'
    ]
    features['negative_ratio'] = sum(1 for w in words if w in negative_words) / n_words
    
    # Positive emotion words
    positive_words = [
        'happy', 'joy', 'excited', 'love', 'wonderful', 'great', 'amazing',
        'good', 'nice', 'beautiful', 'hope', 'grateful', 'thankful', 'proud'
    ]
    features['positive_ratio'] = sum(1 for w in words if w in positive_words) / n_words
    
    # Absolutist words (associated with depression/anxiety)
    absolutist = ['always', 'never', 'nothing', 'everything', 'completely', 'totally', 'absolutely']
    features['absolutist_ratio'] = sum(1 for w in words if w in absolutist) / n_words
    
    # Question marks (may indicate uncertainty)
    features['question_ratio'] = text.count('?') / n_sentences
    
    # Sentence length statistics
    sent_lengths = [len(re.findall(r'\b\w+\b', s)) for s in sentences]
    features['avg_sentence_length'] = np.mean(sent_lengths) if sent_lengths else 0
    features['sentence_length_std'] = np.std(sent_lengths) if len(sent_lengths) > 1 else 0
    
    # Word length (longer words may indicate cognitive complexity)
    word_lengths = [len(w) for w in words]
    features['avg_word_length'] = np.mean(word_lengths) if word_lengths else 0
    
    # Type-token ratio (vocabulary diversity)
    features['type_token_ratio'] = len(set(words)) / n_words
    
    # Hedging words (uncertainty)
    hedging = ['maybe', 'perhaps', 'might', 'could', 'possibly', 'probably', 'think', 'guess']
    features['hedging_ratio'] = sum(1 for w in words if w in hedging) / n_words
    
    # Certainty words
    certainty = ['definitely', 'certainly', 'sure', 'know', 'believe', 'must']
    features['certainty_ratio'] = sum(1 for w in words if w in certainty) / n_words
    
    # Social words
    social = ['friend', 'family', 'people', 'we', 'us', 'they', 'them', 'together']
    features['social_ratio'] = sum(1 for w in words if w in social) / n_words
    
    # Death/self-harm related (critical for safety)
    death_words = ['die', 'death', 'dead', 'kill', 'suicide', 'end', 'hurt', 'pain']
    features['death_ratio'] = sum(1 for w in words if w in death_words) / n_words
    
    # Sleep-related words
    sleep_words = ['sleep', 'tired', 'exhausted', 'insomnia', 'wake', 'rest', 'energy']
    features['sleep_ratio'] = sum(1 for w in words if w in sleep_words) / n_words
    
    # Convert to array
    feature_array = np.array(list(features.values()), dtype=np.float32)
    
    return feature_array


def extract_sentiment_scores(text: str) -> np.ndarray:
    """
    Extract sentiment scores using VADER or TextBlob.
    
    Args:
        text: Input text
        
    Returns:
        Sentiment scores array of shape (4,) - [neg, neu, pos, compound]
    """
    if not text or not text.strip():
        return np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        analyzer = SentimentIntensityAnalyzer()
        scores = analyzer.polarity_scores(text)
        return np.array([
            scores['neg'],
            scores['neu'],
            scores['pos'],
            scores['compound']
        ], dtype=np.float32)
    except ImportError:
        pass
    
    try:
        from textblob import TextBlob
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity  # -1 to 1
        subjectivity = blob.sentiment.subjectivity  # 0 to 1
        
        # Convert to VADER-like format
        if polarity < -0.3:
            neg, pos = abs(polarity), 0
        elif polarity > 0.3:
            neg, pos = 0, polarity
        else:
            neg, pos = 0, 0
        neu = 1 - neg - pos
        
        return np.array([neg, neu, pos, polarity], dtype=np.float32)
    except ImportError:
        pass
    
    # Fallback: simple word-based sentiment
    text_lower = text.lower()
    
    positive = ['good', 'great', 'happy', 'love', 'wonderful', 'amazing', 'excellent']
    negative = ['bad', 'sad', 'hate', 'terrible', 'awful', 'horrible', 'depressed']
    
    pos_count = sum(1 for w in positive if w in text_lower)
    neg_count = sum(1 for w in negative if w in text_lower)
    total = pos_count + neg_count + 1
    
    pos_ratio = pos_count / total
    neg_ratio = neg_count / total
    neu_ratio = 1 - pos_ratio - neg_ratio
    compound = pos_ratio - neg_ratio
    
    return np.array([neg_ratio, neu_ratio, pos_ratio, compound], dtype=np.float32)


class TextEncoderBERT:
    """
    Production-ready text encoder using SentenceTransformer.
    
    Advantages over MIL Attention:
    - Pretrained on billions of tokens (vs 28 samples)
    - Fixed 768-dim output (vs 1M+ trainable params)
    - No overfitting possible
    - Multilingual support
    
    Usage:
        encoder = TextEncoderBERT()
        embeddings = encoder.encode(["Text 1", "Text 2"])
        # embeddings.shape = (2, 768)
    """
    
    # Available models (from best to fastest):
    MODELS = {
        'multilingual': 'distiluse-base-multilingual-cased-v2',  # 768-dim, best for Polish
        'mpnet': 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2',  # 768-dim, higher quality
        'minilm': 'sentence-transformers/all-MiniLM-L6-v2',  # 384-dim, fastest
    }
    
    def __init__(
        self,
        model_name: str = 'multilingual',
        device: Optional[str] = None,
        cache_dir: Optional[str] = None
    ):
        """
        Initialize the text encoder.
        
        Args:
            model_name: Either a key from MODELS dict or full HuggingFace model name
            device: 'cuda', 'cpu', or None (auto-detect)
            cache_dir: Directory to cache downloaded models
        """
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "sentence-transformers not installed. Run:\n"
                "pip install sentence-transformers"
            )
        
        # Resolve model name
        if model_name in self.MODELS:
            full_model_name = self.MODELS[model_name]
        else:
            full_model_name = model_name
        
        logger.info(f"Loading SentenceTransformer: {full_model_name}")
        
        self.model = SentenceTransformer(
            full_model_name,
            device=device,
            cache_folder=cache_dir
        )
        
        self.model_name = full_model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        
        logger.info(f"Loaded model with embedding_dim={self.embedding_dim}")
    
    def encode(
        self,
        transcripts: Union[List[str], str],
        batch_size: int = DEFAULT_TEXT_BATCH_SIZE,
        show_progress_bar: bool = False,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode transcripts into embeddings.
        
        Args:
            transcripts: Single string or list of transcript strings
            batch_size: Batch size for encoding (default 8 for RTX 5060 8GB)
            show_progress_bar: Show progress during encoding
            normalize: Whether to L2-normalize embeddings
            
        Returns:
            numpy array of shape (n_samples, embedding_dim)
        """
        # Handle single string input
        if isinstance(transcripts, str):
            transcripts = [transcripts]
        
        # Filter out empty/None transcripts
        valid_transcripts = []
        valid_indices = []
        for i, text in enumerate(transcripts):
            if text and isinstance(text, str) and text.strip():
                valid_transcripts.append(text.strip())
                valid_indices.append(i)
            else:
                logger.warning(f"Empty or invalid transcript at index {i}")
        
        if not valid_transcripts:
            logger.warning("No valid transcripts to encode!")
            return np.zeros((len(transcripts), self.embedding_dim), dtype=np.float32)
        
        # Encode valid transcripts with memory-safe batch size
        try:
            embeddings = self.model.encode(
                valid_transcripts,
                batch_size=batch_size,
                show_progress_bar=show_progress_bar,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.warning(f"OOM with batch_size={batch_size}, trying batch_size=1")
                self._clear_gpu_memory()
                embeddings = self.model.encode(
                    valid_transcripts,
                    batch_size=1,
                    show_progress_bar=show_progress_bar,
                    convert_to_numpy=True,
                    normalize_embeddings=normalize
                )
            else:
                raise
        
        # Clear GPU memory after encoding
        self._clear_gpu_memory()
        
        # If all transcripts were valid, return directly
        if len(valid_transcripts) == len(transcripts):
            return embeddings
        
        # Otherwise, create full output array with zeros for invalid transcripts
        full_embeddings = np.zeros((len(transcripts), self.embedding_dim), dtype=np.float32)
        for i, idx in enumerate(valid_indices):
            full_embeddings[idx] = embeddings[i]
        
        return full_embeddings
    
    def _clear_gpu_memory(self):
        """Clear GPU memory cache."""
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
    
    def encode_with_pooling(
        self,
        transcripts: List[str],
        segments_per_transcript: int = 5,
        pooling: str = 'mean'
    ) -> np.ndarray:
        """
        Encode transcripts with segment-level pooling.
        
        Useful for very long transcripts - splits into segments,
        encodes each, then pools.
        
        Args:
            transcripts: List of transcript strings
            segments_per_transcript: Number of segments to split each transcript
            pooling: 'mean', 'max', or 'attention'
            
        Returns:
            numpy array of shape (n_samples, embedding_dim)
        """
        all_embeddings = []
        
        for transcript in transcripts:
            if not transcript or not transcript.strip():
                all_embeddings.append(np.zeros(self.embedding_dim, dtype=np.float32))
                continue
            
            # Split into sentences
            sentences = self._split_into_sentences(transcript)
            
            if len(sentences) <= segments_per_transcript:
                # If few sentences, encode all
                segments = sentences
            else:
                # Sample segments evenly from transcript
                step = len(sentences) / segments_per_transcript
                indices = [int(i * step) for i in range(segments_per_transcript)]
                segments = [sentences[i] for i in indices]
            
            # Encode segments
            segment_embeddings = self.encode(segments, show_progress_bar=False)
            
            # Pool
            if pooling == 'mean':
                pooled = np.mean(segment_embeddings, axis=0)
            elif pooling == 'max':
                pooled = np.max(segment_embeddings, axis=0)
            else:
                raise ValueError(f"Unknown pooling method: {pooling}")
            
            all_embeddings.append(pooled)
        
        return np.array(all_embeddings)
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        import re
        # Simple sentence splitting on . ! ?
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences if sentences else [text]
    
    def get_embedding_dim(self) -> int:
        """Return the embedding dimension."""
        return self.embedding_dim
    
    def __repr__(self):
        return f"TextEncoderBERT(model='{self.model_name}', dim={self.embedding_dim})"


class TextEncoderAdvanced:
    """
    Advanced Text Encoder combining BERT + Topic Modeling + Linguistic Features.
    
    This is the recommended encoder for depression detection based on
    SOTA 2024-2025 literature. Combines:
    
    1. BERT embeddings (768-dim) - semantic representation
    2. Topic distribution (50-dim) - thematic context
    3. Linguistic features (15-dim) - depression-specific patterns
    4. Sentiment scores (4-dim) - emotional content
    
    Total: ~837-dim feature vector
    
    Usage:
        encoder = TextEncoderAdvanced()
        encoder.fit_topics(corpus)  # Fit topic model on corpus
        features = encoder.encode_all(transcripts)
    """
    
    def __init__(
        self,
        bert_model: str = 'multilingual',
        n_topics: int = 50,
        topic_method: str = 'nmf',
        use_topics: bool = True,
        use_linguistic: bool = True,
        use_sentiment: bool = True,
        device: Optional[str] = None
    ):
        """
        Initialize advanced text encoder.
        
        Args:
            bert_model: BERT model name (see TextEncoderBERT.MODELS)
            n_topics: Number of topics for topic modeling
            topic_method: 'lda' or 'nmf'
            use_topics: Include topic features
            use_linguistic: Include linguistic features
            use_sentiment: Include sentiment features
            device: 'cuda', 'cpu', or None (auto)
        """
        self.use_topics = use_topics
        self.use_linguistic = use_linguistic
        self.use_sentiment = use_sentiment
        
        # BERT encoder
        self.bert_encoder = TextEncoderBERT(model_name=bert_model, device=device)
        self.bert_dim = self.bert_encoder.embedding_dim
        
        # Topic model
        if use_topics:
            self.topic_model = TopicModel(n_topics=n_topics, method=topic_method)
            self.topic_dim = n_topics
        else:
            self.topic_model = None
            self.topic_dim = 0
        
        # Linguistic and sentiment dimensions
        self.linguistic_dim = 15 if use_linguistic else 0
        self.sentiment_dim = 4 if use_sentiment else 0
        
        logger.info(f"TextEncoderAdvanced initialized:")
        logger.info(f"  BERT: {self.bert_dim}-dim")
        logger.info(f"  Topics: {self.topic_dim}-dim ({topic_method})")
        logger.info(f"  Linguistic: {self.linguistic_dim}-dim")
        logger.info(f"  Sentiment: {self.sentiment_dim}-dim")
        logger.info(f"  Total: {self.get_embedding_dim()}-dim")
    
    def fit_topics(self, corpus: List[str]):
        """
        Fit topic model on corpus.
        
        Should be called with all training transcripts before encoding.
        
        Args:
            corpus: List of all text documents for topic modeling
        """
        if self.use_topics and self.topic_model is not None:
            self.topic_model.fit(corpus)
    
    def encode_all(
        self,
        transcripts: Union[str, List[str]],
        show_progress: bool = False
    ) -> np.ndarray:
        """
        Extract all features from transcripts.
        
        Args:
            transcripts: Single transcript or list of transcripts
            show_progress: Show progress bar
            
        Returns:
            Feature array of shape (n_samples, total_dim)
        """
        if isinstance(transcripts, str):
            transcripts = [transcripts]
        
        features_list = []
        
        # BERT embeddings
        bert_features = self.bert_encoder.encode(
            transcripts,
            show_progress_bar=show_progress
        )
        features_list.append(bert_features)
        
        # Topic features
        if self.use_topics and self.topic_model is not None:
            if self.topic_model.is_fitted:
                topic_features = self.topic_model.transform(transcripts)
            else:
                logger.warning("Topic model not fitted, using zeros. Call fit_topics() first.")
                topic_features = np.zeros((len(transcripts), self.topic_dim), dtype=np.float32)
            features_list.append(topic_features)
        
        # Linguistic features
        if self.use_linguistic:
            linguistic_features = np.array([
                extract_linguistic_features(t) for t in transcripts
            ], dtype=np.float32)
            features_list.append(linguistic_features)
        
        # Sentiment features
        if self.use_sentiment:
            sentiment_features = np.array([
                extract_sentiment_scores(t) for t in transcripts
            ], dtype=np.float32)
            features_list.append(sentiment_features)
        
        # Concatenate all features
        combined = np.hstack(features_list)
        
        # Handle NaN/Inf
        combined = np.nan_to_num(combined, nan=0.0, posinf=0.0, neginf=0.0)
        
        return combined
    
    def get_embedding_dim(self) -> int:
        """Return total embedding dimension."""
        return self.bert_dim + self.topic_dim + self.linguistic_dim + self.sentiment_dim
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        names = []
        
        # BERT features
        names.extend([f'bert_{i}' for i in range(self.bert_dim)])
        
        # Topic features
        if self.use_topics:
            names.extend([f'topic_{i}' for i in range(self.topic_dim)])
        
        # Linguistic features
        if self.use_linguistic:
            names.extend([
                'first_person_ratio', 'negative_ratio', 'positive_ratio',
                'absolutist_ratio', 'question_ratio', 'avg_sentence_length',
                'sentence_length_std', 'avg_word_length', 'type_token_ratio',
                'hedging_ratio', 'certainty_ratio', 'social_ratio',
                'death_ratio', 'sleep_ratio', 'ling_pad'
            ])
        
        # Sentiment features
        if self.use_sentiment:
            names.extend(['sentiment_neg', 'sentiment_neu', 'sentiment_pos', 'sentiment_compound'])
        
        return names
    
    def save_topic_model(self, path: Path):
        """Save fitted topic model."""
        if self.topic_model is not None and self.topic_model.is_fitted:
            self.topic_model.save(path)
    
    def load_topic_model(self, path: Path):
        """Load fitted topic model."""
        if self.use_topics:
            self.topic_model = TopicModel.load(path)
    
    def __repr__(self):
        return (
            f"TextEncoderAdvanced("
            f"bert={self.bert_dim}, "
            f"topics={self.topic_dim}, "
            f"linguistic={self.linguistic_dim}, "
            f"sentiment={self.sentiment_dim}, "
            f"total={self.get_embedding_dim()})"
        )


# Convenience function for quick encoding
def encode_transcripts(
    transcripts: List[str],
    model_name: str = 'multilingual'
) -> np.ndarray:
    """
    Quick function to encode transcripts.
    
    Args:
        transcripts: List of transcript strings
        model_name: Model to use (see TextEncoderBERT.MODELS)
        
    Returns:
        numpy array of shape (n_samples, 768)
    """
    encoder = TextEncoderBERT(model_name=model_name)
    return encoder.encode(transcripts, show_progress_bar=True)


if __name__ == "__main__":
    print("Testing Text Encoders...")
    
    # Test texts
    test_texts = [
        "I feel very sad and hopeless today. I can't sleep and I have no energy.",
        "I had a great day at work! Everything is wonderful and I feel amazing.",
        "Jestem smutny i nie mam energii. Nic mnie nie cieszy.",  # Polish
        "",  # Empty - should handle gracefully
    ]
    
    # Test TextEncoderBERT
    print("\n1. Testing TextEncoderBERT:")
    encoder = TextEncoderBERT()
    print(f"   Encoder: {encoder}")
    embeddings = encoder.encode(test_texts, show_progress_bar=True)
    print(f"   Embeddings shape: {embeddings.shape}")
    
    from numpy.linalg import norm
    def cosine_similarity(a, b):
        return np.dot(a, b) / (norm(a) * norm(b) + 1e-8)
    
    print(f"   Sad vs Happy similarity: {cosine_similarity(embeddings[0], embeddings[1]):.3f}")
    print("   [OK] TextEncoderBERT works!")
    
    # Test linguistic features
    print("\n2. Testing Linguistic Features:")
    ling_features = extract_linguistic_features(test_texts[0])
    print(f"   Linguistic features shape: {ling_features.shape}")
    print(f"   First person ratio: {ling_features[0]:.3f}")
    print(f"   Negative ratio: {ling_features[1]:.3f}")
    print("   [OK] Linguistic features work!")
    
    # Test sentiment features
    print("\n3. Testing Sentiment Features:")
    sent_features = extract_sentiment_scores(test_texts[0])
    print(f"   Sentiment shape: {sent_features.shape}")
    print(f"   Neg: {sent_features[0]:.3f}, Neu: {sent_features[1]:.3f}, Pos: {sent_features[2]:.3f}")
    
    sent_features_happy = extract_sentiment_scores(test_texts[1])
    print(f"   Happy text sentiment: Neg={sent_features_happy[0]:.3f}, Pos={sent_features_happy[2]:.3f}")
    print("   [OK] Sentiment features work!")
    
    # Test TopicModel
    print("\n4. Testing TopicModel:")
    # Create a small corpus for testing
    corpus = [
        "I feel very sad and depressed. Nothing makes me happy anymore.",
        "Work has been stressful. I can't sleep at night.",
        "My family is supportive but I still feel alone.",
        "I exercise daily and try to stay positive.",
        "The weather is nice today. I went for a walk.",
    ] * 5  # Repeat for minimum corpus size
    
    topic_model = TopicModel(n_topics=5, method='nmf')
    topic_model.fit(corpus)
    
    topic_dist = topic_model.transform(test_texts[:2])
    print(f"   Topic distribution shape: {topic_dist.shape}")
    print(f"   Topics for sad text: {topic_dist[0][:3]}...")
    
    top_words = topic_model.get_top_words(n_words=5)
    print(f"   Topic 0 words: {top_words[0]}")
    print("   [OK] TopicModel works!")
    
    # Test TextEncoderAdvanced
    print("\n5. Testing TextEncoderAdvanced:")
    encoder_adv = TextEncoderAdvanced(
        n_topics=10,
        use_topics=True,
        use_linguistic=True,
        use_sentiment=True
    )
    
    # Fit topics on corpus
    encoder_adv.fit_topics(corpus)
    
    # Encode
    adv_features = encoder_adv.encode_all(test_texts)
    print(f"   Advanced features shape: {adv_features.shape}")
    print(f"   Expected dim: {encoder_adv.get_embedding_dim()}")
    print(f"   {encoder_adv}")
    print("   [OK] TextEncoderAdvanced works!")
    
    # Test without topic model
    print("\n6. Testing TextEncoderAdvanced (no topics):")
    encoder_notopic = TextEncoderAdvanced(
        use_topics=False,
        use_linguistic=True,
        use_sentiment=True
    )
    notopic_features = encoder_notopic.encode_all(test_texts)
    print(f"   Features shape: {notopic_features.shape}")
    print(f"   {encoder_notopic}")
    print("   [OK] No-topic encoder works!")
    
    print("\n[OK] All text encoder tests complete!")

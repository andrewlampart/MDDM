"""Text feature extraction for depression detection"""

import numpy as np
from typing import Dict, Optional
import warnings

try:
    from transformers import AutoTokenizer, AutoModel
    import torch
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    warnings.warn("transformers not available. Text embeddings will be skipped.")


class TextFeatureExtractor:
    """Ekstrakcja cech tekstowych z transkrypcji"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", 
                 device: Optional[str] = None):
        """Initialize TextFeatureExtractor
        
        Args:
            model_name: Nazwa modelu transformers do embeddings
            device: Device dla modelu ('cuda', 'cpu', None=auto)
        """
        self.model_name = model_name
        self.device = device
        
        if TRANSFORMERS_AVAILABLE:
            try:
                if self.device is None:
                    self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
                
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model.eval()
                
                # Pobierz rozmiar embeddingu
                with torch.no_grad():
                    dummy_input = self.tokenizer("test", return_tensors='pt', 
                                                truncation=True, max_length=512)
                    dummy_input = {k: v.to(self.device) for k, v in dummy_input.items()}
                    dummy_output = self.model(**dummy_input)
                    self.embedding_dim = dummy_output.last_hidden_state.shape[-1]
            except Exception as e:
                warnings.warn(f"Failed to load transformer model: {e}")
                self.model = None
                self.tokenizer = None
                self.embedding_dim = 384  # Default dla MiniLM
        else:
            self.model = None
            self.tokenizer = None
            self.embedding_dim = 384
    
    def extract_linguistic_features(self, text: str) -> Optional[Dict]:
        """Ekstrakcja cech lingwistycznych
        
        Args:
            text: Tekst do analizy
            
        Returns:
            Dictionary z linguistic features lub None jeśli brak tekstu
        """
        if not text or len(text.strip()) == 0:
            return None
        
        words = text.split()
        sentences = [s.strip() for s in text.split('.') if s.strip()]
        
        features = {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0.0,
            'unique_words': len(set(words)),
            'vocabulary_richness': len(set(words)) / max(len(words), 1),
            'text_length': len(text),
            'avg_sentence_length': len(words) / max(len(sentences), 1) if sentences else 0.0,
        }
        
        # Sentiment-like markers (prosty heurystyk dla angielskiego)
        negative_words = ['not', 'never', 'no', 'nothing', 'nobody', 'sad', 'bad', 
                          'depressed', 'hopeless', 'worthless', 'tired', 'anxious']
        negative_count = sum(1 for w in words if w.lower() in negative_words)
        features['negative_word_ratio'] = negative_count / max(len(words), 1)
        
        # Pronoun usage patterns (mogą wskazywać na depresję)
        first_person = ['i', 'me', 'my', 'myself', 'mine']
        first_person_count = sum(1 for w in words if w.lower() in first_person)
        features['first_person_ratio'] = first_person_count / max(len(words), 1)
        
        # Question marks (może wskazywać na niepewność)
        features['question_count'] = text.count('?')
        
        # Exclamation marks
        features['exclamation_count'] = text.count('!')
        
        return features
    
    def get_embedding(self, text: str) -> np.ndarray:
        """Zwróć sentence embedding z transformera
        
        Args:
            text: Tekst do embeddingu
            
        Returns:
            Embedding vector (384-dim dla MiniLM)
        """
        if not TRANSFORMERS_AVAILABLE or self.model is None or not text or len(text.strip()) == 0:
            return np.zeros(self.embedding_dim)
        
        try:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, 
                                   max_length=512, padding=True)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Mean pooling
                embeddings = outputs.last_hidden_state.mean(dim=1)
            
            return embeddings[0].cpu().numpy()
        except Exception as e:
            warnings.warn(f"Embedding extraction failed: {e}")
            return np.zeros(self.embedding_dim)
    
    def process_text(self, text: Optional[str]) -> Optional[Dict]:
        """Pipeline: tekst → cechy lingwistyczne + embedding
        
        Args:
            text: Tekst do przetworzenia (może być None)
            
        Returns:
            Dictionary z linguistic_features i embedding, lub None jeśli brak tekstu
        """
        if not text or len(text.strip()) == 0:
            return None
        
        linguistic = self.extract_linguistic_features(text)
        embedding = self.get_embedding(text)
        
        return {
            'linguistic_features': linguistic,
            'embedding': embedding
        }
    
    def flatten_features(self, features: Optional[Dict], session_id: Optional[int] = None) -> Dict:
        """Spłaszcz features do formatu dla DataFrame
        
        Args:
            features: Output z process_text() lub None
            session_id: ID sesji (opcjonalne)
            
        Returns:
            Flattened dictionary z wszystkimi cechami
        """
        flat = {}
        
        if session_id is not None:
            flat['session_id'] = session_id
        
        if features is None:
            # Wypełnij zerami jeśli brak tekstu
            # Linguistic features
            flat['word_count'] = 0
            flat['sentence_count'] = 0
            flat['avg_word_length'] = 0.0
            flat['unique_words'] = 0
            flat['vocabulary_richness'] = 0.0
            flat['text_length'] = 0
            flat['avg_sentence_length'] = 0.0
            flat['negative_word_ratio'] = 0.0
            flat['first_person_ratio'] = 0.0
            flat['question_count'] = 0
            flat['exclamation_count'] = 0
            
            # Embedding
            for i in range(self.embedding_dim):
                flat[f'text_embedding_{i}'] = 0.0
        else:
            # Linguistic features
            if features['linguistic_features']:
                flat.update(features['linguistic_features'])
            else:
                # Fallback jeśli linguistic features None
                flat['word_count'] = 0
                flat['sentence_count'] = 0
                flat['avg_word_length'] = 0.0
                flat['unique_words'] = 0
                flat['vocabulary_richness'] = 0.0
                flat['text_length'] = 0
                flat['avg_sentence_length'] = 0.0
                flat['negative_word_ratio'] = 0.0
                flat['first_person_ratio'] = 0.0
                flat['question_count'] = 0
                flat['exclamation_count'] = 0
            
            # Embedding
            for i, val in enumerate(features['embedding']):
                flat[f'text_embedding_{i}'] = float(val)
        
        return flat

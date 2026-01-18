"""
MDDM Models - Deep learning models for multimodal depression detection

Modules:
- text_encoder_bert: BERT-based text encoding (SentenceTransformer)
- audio_encoder_advanced: Wav2Vec 2.0 + prosody features
- fusion_fixed: Fixed multimodal fusion with proper regularization
"""

from .text_encoder_bert import TextEncoderBERT, encode_transcripts
from .audio_encoder_advanced import (
    AudioEncoderHybrid,
    AudioEncoderProsodyOnly,
    extract_audio_features,
)
from .fusion_fixed import (
    LateFusionModelFixed,
    EarlyFusionModelFixed,
    FusionTrainer,
    create_fusion_model,
)

__all__ = [
    # Text
    'TextEncoderBERT',
    'encode_transcripts',
    # Audio
    'AudioEncoderHybrid',
    'AudioEncoderProsodyOnly',
    'extract_audio_features',
    # Fusion
    'LateFusionModelFixed',
    'EarlyFusionModelFixed',
    'FusionTrainer',
    'create_fusion_model',
]

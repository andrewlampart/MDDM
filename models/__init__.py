"""
MDDM Models - Deep learning models for multimodal depression detection

Modules:
- text_encoder_bert: BERT-based text encoding (SentenceTransformer)
- audio_encoder_advanced: Wav2Vec 2.0 + prosody features
- fusion_fixed: Fixed multimodal fusion with proper regularization
- attention_fusion: Attention-based fusion (SOTA 2024-2025)
- teacher_student: Teacher-Student Knowledge Distillation (SOTA 2025)
"""

from .text_encoder_bert import (
    TextEncoderBERT,
    TextEncoderAdvanced,
    TopicModel,
    encode_transcripts,
    extract_linguistic_features,
    extract_sentiment_scores,
)
from .audio_encoder_advanced import (
    VADFilter,
    AudioEncoderHybrid,
    AudioEncoderAdvanced,
    AudioEncoderProsodyOnly,
    extract_audio_features,
    extract_extended_mfcc,
    extract_glottal_features,
)
from .fusion_fixed import (
    ModalityNormalizer,
    NormalizedLateFusionModel,
    LateFusionModelFixed,
    EarlyFusionModelFixed,
    FusionTrainer,
    create_fusion_model,
)
from .attention_fusion import (
    MultiheadSelfAttention,
    CrossModalAttention,
    GatedFusion,
    AttentionFusionModel,
    AttentionFusionWithBiLSTM,
    AttentionFusionTrainer,
    create_attention_fusion_model,
)
from .teacher_student import (
    TeacherModel,
    StudentFusionModel,
    HybridKDLoss,
    TeacherStudentTrainer,
)

__all__ = [
    # Text
    'TextEncoderBERT',
    'TextEncoderAdvanced',
    'TopicModel',
    'encode_transcripts',
    'extract_linguistic_features',
    'extract_sentiment_scores',
    # Audio
    'VADFilter',
    'AudioEncoderHybrid',
    'AudioEncoderAdvanced',
    'AudioEncoderProsodyOnly',
    'extract_audio_features',
    'extract_extended_mfcc',
    'extract_glottal_features',
    # Fusion (basic)
    'ModalityNormalizer',
    'NormalizedLateFusionModel',
    'LateFusionModelFixed',
    'EarlyFusionModelFixed',
    'FusionTrainer',
    'create_fusion_model',
    # Fusion (attention-based)
    'MultiheadSelfAttention',
    'CrossModalAttention',
    'GatedFusion',
    'AttentionFusionModel',
    'AttentionFusionWithBiLSTM',
    'AttentionFusionTrainer',
    'create_attention_fusion_model',
    # Teacher-Student (SOTA 2025)
    'TeacherModel',
    'StudentFusionModel',
    'HybridKDLoss',
    'TeacherStudentTrainer',
]

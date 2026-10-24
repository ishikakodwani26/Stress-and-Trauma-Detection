"""Audio preprocessing, feature extraction wrappers, and pipeline orchestrator."""

from app.preprocessing.audio_converter import (
    AudioConversionError,
    convert_audio_to_wav_pcm16,
)
from app.preprocessing.acoustic_extractor import (
    AcousticFeatureExtractor,
    extract_acoustic_features,
)
from app.preprocessing.emotion_extractor import (
    XLSRFeatureExtractor,
    extract_emotion_embeddings,
)
from app.preprocessing.asr_extractor import (
    IndicConformerASR,
    transcribe_indic_conformer,
)
from app.preprocessing.nlp_extractor import (
    MuRILEmbeddingExtractor,
    extract_muril_embeddings,
)
from app.preprocessing.pipeline import (
    VoicePipelineFeatures,
    run_voice_feature_pipeline,
    stage_convert_audio,
    stage_extract_acoustic,
    stage_extract_emotion,
    stage_transcribe_audio,
    stage_extract_nlp,
    stage_extract_context,
    stage_concatenate_features,
)

__all__ = [
    "AudioConversionError",
    "convert_audio_to_wav_pcm16",
    "AcousticFeatureExtractor",
    "extract_acoustic_features",
    "XLSRFeatureExtractor",
    "extract_emotion_embeddings",
    "IndicConformerASR",
    "transcribe_indic_conformer",
    "MuRILEmbeddingExtractor",
    "extract_muril_embeddings",
    "VoicePipelineFeatures",
    "run_voice_feature_pipeline",
    "stage_convert_audio",
    "stage_extract_acoustic",
    "stage_extract_emotion",
    "stage_transcribe_audio",
    "stage_extract_nlp",
    "stage_extract_context",
    "stage_concatenate_features",
]

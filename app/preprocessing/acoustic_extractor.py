"""Acoustic feature extraction wrapper using openSMILE.

Extracts standard acoustic feature sets (e.g. eGeMAPSv02 or ComParE_2016)
from 16kHz mono audio.
"""

import io
import logging
import tempfile
from pathlib import Path
from typing import Optional, Union
import numpy as np

from app.config import settings

logger = logging.getLogger(__name__)


class AcousticFeatureExtractor:
    """openSMILE acoustic feature extractor wrapper."""

    def __init__(
        self,
        feature_set: Optional[str] = None,
        feature_level: Optional[str] = None,
    ):
        self.feature_set_name = feature_set or settings.OPENSMILE_FEATURE_SET
        self.feature_level_name = feature_level or settings.OPENSMILE_FEATURE_LEVEL
        self._smile = None

    def _get_smile(self):
        """Lazy load openSMILE instance."""
        if self._smile is None:
            try:
                import opensmile

                # Resolve feature set enum
                feat_set = getattr(
                    opensmile.FeatureSet,
                    self.feature_set_name,
                    opensmile.FeatureSet.eGeMAPSv02,
                )
                feat_lvl = getattr(
                    opensmile.FeatureLevel,
                    self.feature_level_name,
                    opensmile.FeatureLevel.Functionals,
                )
                self._smile = opensmile.Smile(
                    feature_set=feat_set,
                    feature_level=feat_lvl,
                )
                logger.info(
                    "Initialized openSMILE with feature_set=%s, level=%s",
                    self.feature_set_name,
                    self.feature_level_name,
                )
            except ImportError as err:
                raise ImportError(
                    "The 'opensmile' library is required for acoustic feature extraction. "
                    "Install it via `pip install opensmile`."
                ) from err
        return self._smile

    def extract_features(
        self,
        audio_input: Union[str, Path, bytes, np.ndarray],
        sample_rate: int = 16000,
    ) -> np.ndarray:
        """Extract openSMILE acoustic features as a 1D numpy array.

        Args:
            audio_input: File path, WAV binary bytes, or 1D numpy waveform array.
            sample_rate: Sampling rate in Hz (default: 16000).

        Returns:
            np.ndarray: 1D array of extracted acoustic features.
        """
        smile = self._get_smile()

        if isinstance(audio_input, np.ndarray):
            # Process directly from signal array
            signal = audio_input.astype(np.float32)
            df = smile.process_signal(signal, sample_rate)
            return df.to_numpy().flatten()

        elif isinstance(audio_input, (str, Path)):
            path = Path(audio_input)
            df = smile.process_file(str(path))
            return df.to_numpy().flatten()

        elif isinstance(audio_input, (bytes, bytearray)):
            # Process via a temporary WAV file for openSMILE file processing
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                tmp.write(audio_input)
                tmp_path = Path(tmp.name)
            try:
                df = smile.process_file(str(tmp_path))
                return df.to_numpy().flatten()
            finally:
                if tmp_path.exists():
                    tmp_path.unlink()

        else:
            raise ValueError(f"Unsupported audio input type: {type(audio_input)}")


# Standalone callable function for independent unit testing
_default_extractor = AcousticFeatureExtractor()


def extract_acoustic_features(
    audio_input: Union[str, Path, bytes, np.ndarray],
    sample_rate: int = 16000,
) -> np.ndarray:
    """Top-level callable function to extract openSMILE acoustic features."""
    return _default_extractor.extract_features(audio_input, sample_rate=sample_rate)

"""Audio conversion utilities.

Standardizes raw audio inputs (various formats) into:
- 16 kHz sample rate
- Mono (single channel)
- 16-bit PCM WAV
"""

import io
import logging
from pathlib import Path
from typing import BinaryIO, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class AudioConversionError(Exception):
    """Raised when audio conversion or normalization fails."""
    pass


def convert_audio_to_wav_pcm16(
    audio_input: Union[bytes, BinaryIO, str, Path],
    target_sr: int = 16000,
    target_channels: int = 1,
) -> Tuple[np.ndarray, int, bytes]:
    """Convert any incoming audio file/bytes to 16kHz mono 16-bit PCM WAV.

    Args:
        audio_input: Raw audio bytes, file-like object, or path to audio file.
        target_sr: Target sampling rate in Hz (default: 16000).
        target_channels: Number of target audio channels (default: 1 for mono).

    Returns:
        Tuple of:
            - waveform (np.ndarray): 1D float32 audio waveform normalized to [-1.0, 1.0].
            - sample_rate (int): Resulting sample rate (16000).
            - wav_bytes (bytes): Valid 16-bit PCM WAV file binary content.

    Raises:
        AudioConversionError: If input cannot be decoded or converted.
    """
    raw_bytes: bytes
    if isinstance(audio_input, (str, Path)):
        path = Path(audio_input)
        if not path.is_file():
            raise AudioConversionError(f"Audio file not found at: {path}")
        with open(path, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(audio_input, (bytes, bytearray)):
        raw_bytes = bytes(audio_input)
    elif hasattr(audio_input, "read"):
        raw_bytes = audio_input.read()
        if hasattr(audio_input, "seek"):
            audio_input.seek(0)
    else:
        raise AudioConversionError(f"Unsupported audio input type: {type(audio_input)}")

    if not raw_bytes:
        raise AudioConversionError("Audio input data is empty.")

    # Strategy 1: Attempt native decode using soundfile / librosa (no external ffmpeg required for wav/flac/ogg)
    try:
        import soundfile as sf

        with io.BytesIO(raw_bytes) as in_buf:
            data, sr = sf.read(in_buf, dtype="float32", always_2d=True)

        waveform = _resample_and_downmix(data, sr, target_sr, target_channels)
        wav_bytes = _encode_pcm16_wav(waveform, target_sr)
        return waveform, target_sr, wav_bytes

    except Exception as sf_err:
        logger.debug("soundfile failed to read audio: %s. Falling back to pydub...", sf_err)

    # Strategy 2: Attempt decode via pydub (handles mp3, m4a, webm, aac via ffmpeg)
    try:
        from pydub import AudioSegment

        segment = AudioSegment.from_file(io.BytesIO(raw_bytes))
        segment = segment.set_frame_rate(target_sr)
        segment = segment.set_channels(target_channels)
        segment = segment.set_sample_width(2)  # 2 bytes = 16-bit

        out_buf = io.BytesIO()
        segment.export(out_buf, format="wav", codec="pcm_s16le")
        wav_bytes = out_buf.getvalue()

        # Convert to float32 numpy array normalized to [-1.0, 1.0]
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        max_val = float(2 ** (16 - 1))  # 32768.0
        waveform = samples / max_val

        return waveform, target_sr, wav_bytes

    except Exception as pydub_err:
        logger.error("Audio decoding failed via both soundfile and pydub: %s", pydub_err)
        raise AudioConversionError(
            f"Failed to decode audio. Please ensure it is a valid audio format. "
            f"If using mp3/m4a/webm, ensure ffmpeg is installed on the system. Details: {pydub_err}"
        ) from pydub_err


def _resample_and_downmix(
    data: np.ndarray, current_sr: int, target_sr: int, target_channels: int
) -> np.ndarray:
    """Downmix to mono and resample to target_sr using librosa or scipy."""
    # Downmix multi-channel to mono (mean across channels)
    if data.ndim == 2:
        waveform = np.mean(data, axis=1)
    else:
        waveform = data.flatten()

    # Resample if sample rates differ
    if current_sr != target_sr:
        try:
            import librosa

            waveform = librosa.resample(waveform, orig_sr=current_sr, target_sr=target_sr)
        except ImportError:
            from scipy import signal

            num_samples = int(len(waveform) * float(target_sr) / current_sr)
            waveform = signal.resample(waveform, num_samples)

    # Ensure float32 representation in [-1.0, 1.0]
    waveform = np.clip(waveform, -1.0, 1.0).astype(np.float32)
    return waveform


def _encode_pcm16_wav(waveform: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 waveform into 16-bit PCM WAV bytes."""
    import soundfile as sf

    out_buf = io.BytesIO()
    sf.write(out_buf, waveform, sample_rate, format="WAV", subtype="PCM_16")
    return out_buf.getvalue()

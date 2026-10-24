"""Unit tests for the audio converter module."""

import io
import wave
import numpy as np
import pytest

from app.preprocessing.audio_converter import (
    AudioConversionError,
    convert_audio_to_wav_pcm16,
)


def _create_synthetic_wav_bytes(
    duration_sec: float = 1.0,
    sample_rate: int = 44100,
    channels: int = 2,
    frequency: float = 440.0,
) -> bytes:
    """Generate in-memory WAV audio bytes for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    # Generate stereo sine wave
    sine = (0.5 * np.sin(2 * np.pi * frequency * t) * 32767).astype(np.int16)

    if channels == 2:
        interleaved = np.empty((sine.size * 2,), dtype=np.int16)
        interleaved[0::2] = sine
        interleaved[1::2] = sine
        audio_data = interleaved.tobytes()
    else:
        audio_data = sine.tobytes()

    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data)

    return buf.getvalue()


def test_convert_synthetic_audio_to_16k_mono_pcm16():
    """Test converting 44.1kHz stereo WAV into 16kHz mono 16-bit PCM WAV."""
    raw_stereo_44k = _create_synthetic_wav_bytes(
        duration_sec=0.5,
        sample_rate=44100,
        channels=2,
        frequency=440.0,
    )

    waveform, sr, wav_bytes = convert_audio_to_wav_pcm16(raw_stereo_44k)

    assert sr == 16000
    assert isinstance(waveform, np.ndarray)
    assert waveform.ndim == 1
    assert waveform.dtype == np.float32

    # Verify output wav bytes structure with standard wave library
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1  # Mono
        assert wf.getsampwidth() == 2  # 16-bit PCM
        assert wf.getframerate() == 16000  # 16kHz
        n_frames = wf.getnframes()
        assert n_frames == pytest.approx(8000, abs=100)  # ~0.5 sec * 16000


def test_empty_audio_raises_error():
    """Test that empty bytes raise AudioConversionError."""
    with pytest.raises(AudioConversionError):
        convert_audio_to_wav_pcm16(b"")

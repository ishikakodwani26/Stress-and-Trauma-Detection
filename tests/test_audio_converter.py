"""Unit tests for the format-agnostic audio converter module."""

import io
from unittest.mock import MagicMock, patch
import wave
import numpy as np
import pytest
import soundfile as sf

from app.preprocessing.audio_converter import (
    AudioConversionError,
    convert_audio_to_wav_pcm16,
    detect_audio_format,
)
from app.preprocessing.pipeline import stage_convert_audio


def _create_synthetic_wav_bytes(
    duration_sec: float = 0.5,
    sample_rate: int = 44100,
    channels: int = 2,
    frequency: float = 440.0,
) -> bytes:
    """Generate in-memory WAV audio bytes for testing."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
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


def _create_synthetic_soundfile_bytes(
    fmt: str,
    subtype: str = None,
    duration_sec: float = 0.5,
    sample_rate: int = 48000,
    channels: int = 1,
) -> bytes:
    """Generate in-memory audio bytes using soundfile (OGG, MP3, FLAC, etc.)."""
    t = np.linspace(0, duration_sec, int(sample_rate * duration_sec), endpoint=False)
    waveform = (0.4 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)
    if channels == 2:
        waveform = np.column_stack([waveform, waveform])

    buf = io.BytesIO()
    kwargs = {"format": fmt}
    if subtype:
        kwargs["subtype"] = subtype
    sf.write(buf, waveform, sample_rate, **kwargs)
    return buf.getvalue()


def test_convert_synthetic_wav_to_16k_mono_pcm16():
    """Test converting 44.1kHz stereo WAV into 16kHz mono 16-bit PCM WAV."""
    raw_stereo_44k = _create_synthetic_wav_bytes(
        duration_sec=0.5,
        sample_rate=44100,
        channels=2,
        frequency=440.0,
    )

    waveform, sr, wav_bytes = stage_convert_audio(raw_stereo_44k)

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


def test_convert_ogg_vorbis_and_opus():
    """Test format-agnostic conversion of browser-compatible OGG Vorbis and Opus."""
    # 1. Test OGG Vorbis
    ogg_vorbis_bytes = _create_synthetic_soundfile_bytes("OGG", subtype="VORBIS", duration_sec=0.4)
    waveform_v, sr_v, wav_v = stage_convert_audio(ogg_vorbis_bytes, format_hint="ogg")
    assert sr_v == 16000
    assert waveform_v.ndim == 1
    with wave.open(io.BytesIO(wav_v), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000

    # 2. Test OGG Opus (standard browser MediaRecorder format)
    ogg_opus_bytes = _create_synthetic_soundfile_bytes("OGG", subtype="OPUS", duration_sec=0.4)
    waveform_o, sr_o, wav_o = stage_convert_audio(ogg_opus_bytes, format_hint="ogg")
    assert sr_o == 16000
    assert waveform_o.ndim == 1
    with wave.open(io.BytesIO(wav_o), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000


def test_convert_mp3_audio():
    """Test format-agnostic conversion of MP3 audio."""
    mp3_bytes = _create_synthetic_soundfile_bytes("MP3", duration_sec=0.4, sample_rate=44100)
    waveform, sr, wav_bytes = stage_convert_audio(mp3_bytes, format_hint="mp3")

    assert sr == 16000
    assert waveform.ndim == 1
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        assert wf.getnchannels() == 1
        assert wf.getsampwidth() == 2
        assert wf.getframerate() == 16000


def test_detect_audio_format_magic_bytes_and_mimes():
    """Test auto-detection of audio formats from binary signatures and MIME types."""
    # WebM EBML header
    webm_sig = b"\x1a\x45\xdf\xa3\x9f\x42\x86\x81\x01\x42\xf7\x81\x01"
    assert detect_audio_format(webm_sig) == "webm"

    # OGG container header
    ogg_sig = b"OggS\x00\x02\x00\x00\x00\x00\x00\x00\x00\x00"
    assert detect_audio_format(ogg_sig) == "ogg"

    # WAV header
    wav_sig = b"RIFF\x24\x00\x00\x00WAVEfmt \x10\x00\x00\x00"
    assert detect_audio_format(wav_sig) == "wav"

    # FLAC header
    flac_sig = b"fLaC\x00\x00\x00\x22\x10\x00\x10\x00"
    assert detect_audio_format(flac_sig) == "flac"

    # MP3 ID3 header
    mp3_id3 = b"ID3\x03\x00\x00\x00\x00\x0f\x76"
    assert detect_audio_format(mp3_id3) == "mp3"

    # MP3 sync frame
    mp3_sync = b"\xff\xfb\x90\x64\x00\x00\x00\x00"
    assert detect_audio_format(mp3_sync) == "mp3"

    # M4A / MP4 ftyp box
    m4a_sig = b"\x00\x00\x00\x20ftypM4A \x00\x00\x00\x00"
    assert detect_audio_format(m4a_sig) == "m4a"

    # MIME type detection
    assert detect_audio_format(b"", filename_or_mime="audio/webm;codecs=opus") == "webm"
    assert detect_audio_format(b"", filename_or_mime="audio/ogg; codecs=opus") == "ogg"
    assert detect_audio_format(b"", filename_or_mime="audio/mpeg") == "mp3"
    assert detect_audio_format(b"", filename_or_mime="audio/x-m4a") == "m4a"
    assert detect_audio_format(b"", filename_or_mime="recording.webm") == "webm"


def test_pydub_webm_conversion_fallback():
    """Verify pydub pipeline handles WebM conversion and standardizes to 16kHz mono PCM16."""
    fake_webm_bytes = b"\x1a\x45\xdf\xa3" + b"\x00" * 200

    # Mock pydub.AudioSegment.from_file to simulate decoding WebM stream
    mock_segment = MagicMock()
    # 0.5s of 16kHz mono audio = 8000 samples
    samples_array = np.zeros(8000, dtype=np.int16)
    mock_segment.get_array_of_samples.return_value = samples_array
    mock_segment.set_frame_rate.return_value = mock_segment
    mock_segment.set_channels.return_value = mock_segment
    mock_segment.set_sample_width.return_value = mock_segment

    raw_wav = _create_synthetic_wav_bytes(duration_sec=0.5, sample_rate=16000, channels=1)

    def fake_export(out_buf, format, codec):
        out_buf.write(raw_wav)

    mock_segment.export.side_effect = fake_export

    with patch("pydub.AudioSegment.from_file", return_value=mock_segment) as mock_from_file:
        waveform, sr, wav_bytes = convert_audio_to_wav_pcm16(fake_webm_bytes, format_hint="webm")

        assert sr == 16000
        assert waveform.ndim == 1
        assert len(waveform) == 8000
        assert mock_segment.set_frame_rate.called_with(16000)
        assert mock_segment.set_channels.called_with(1)
        assert mock_segment.set_sample_width.called_with(2)


def test_empty_audio_raises_error():
    """Test that empty bytes raise AudioConversionError."""
    with pytest.raises(AudioConversionError, match="empty"):
        stage_convert_audio(b"")


def test_corrupted_audio_raises_error():
    """Test that non-audio corrupted bytes raise AudioConversionError."""
    corrupted_data = b"NOT_A_VALID_AUDIO_CONTAINER_OR_CODEC_JUST_GARBAGE" * 20
    with pytest.raises(AudioConversionError, match="Failed to convert audio input"):
        stage_convert_audio(corrupted_data)

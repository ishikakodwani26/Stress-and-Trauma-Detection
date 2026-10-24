"""Audio conversion utilities.

Standardizes raw audio inputs (various formats: webm, ogg, mp3, wav, m4a, flac, etc.) into:
- 16 kHz sample rate
- Mono (single channel)
- 16-bit PCM WAV
"""

import io
import logging
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import BinaryIO, Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)


class AudioConversionError(Exception):
    """Raised when audio conversion or normalization fails."""
    pass


def detect_audio_format(
    data: bytes, filename_or_mime: Optional[str] = None
) -> Optional[str]:
    """Detect audio container or codec format.

    Examines filename/MIME hint first, then binary magic byte signatures.
    Supports common formats: webm, ogg, wav, mp3, m4a, flac, aac, aiff.

    Args:
        data: Raw audio binary content.
        filename_or_mime: Optional filename, extension, or MIME type string.

    Returns:
        Normalized format string (e.g. 'webm', 'ogg', 'mp3', 'wav', 'm4a') or None.
    """
    if filename_or_mime:
        val = filename_or_mime.lower().strip()
        # MIME type pattern checks (common browser MediaRecorder outputs)
        if "webm" in val:
            return "webm"
        if "ogg" in val or "opus" in val or "vorbis" in val:
            return "ogg"
        if "wav" in val or "wave" in val:
            return "wav"
        if "mp3" in val or "mpeg" in val:
            return "mp3"
        if "m4a" in val or "mp4" in val or "aac" in val:
            return "m4a"
        if "flac" in val:
            return "flac"

        # Check extension
        ext = Path(val).suffix.lstrip(".").lower()
        if ext in {"webm", "ogg", "wav", "mp3", "m4a", "flac", "aac", "aiff"}:
            return ext
        clean = val.lstrip(".").lower()
        if clean in {"webm", "ogg", "wav", "mp3", "m4a", "flac", "aac", "aiff"}:
            return clean

    # Binary magic byte signatures
    if len(data) >= 4:
        # WebM / Matroska EBML: 0x1A 0x45 0xDF 0xA3
        if data[:4] == b"\x1a\x45\xdf\xa3":
            return "webm"
        # Ogg container (Opus / Vorbis): 'OggS'
        if data[:4] == b"OggS":
            return "ogg"
        # RIFF ... WAVE
        if data[:4] == b"RIFF" and len(data) >= 12 and data[8:12] == b"WAVE":
            return "wav"
        # FLAC: 'fLaC'
        if data[:4] == b"fLaC":
            return "flac"
        # MP3: ID3v2 tag or MPEG audio sync frame (0xFF 0xE0-0xFF)
        if data[:3] == b"ID3" or (data[0] == 0xFF and (data[1] & 0xE0) == 0xE0):
            return "mp3"
        # MP4 / M4A container: 'ftyp' box
        if len(data) >= 12 and data[4:8] == b"ftyp":
            return "m4a"
        # AIFF: FORM ... AIFF
        if data[:4] == b"FORM" and len(data) >= 12 and data[8:12] == b"AIFF":
            return "aiff"

    return None


def _find_ffmpeg_executable() -> Optional[str]:
    """Locate ffmpeg binary in PATH or common system directories."""
    # Check standard PATH first
    found = shutil.which("ffmpeg")
    if found:
        return found

    # Check common Windows paths
    candidates = [
        Path(os.environ.get("LOCALAPPDATA", "")) / "Microsoft" / "WinGet" / "Links" / "ffmpeg.exe",
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "ffmpeg" / "bin" / "ffmpeg.exe",
        Path("C:/Program Files/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/ffmpeg/bin/ffmpeg.exe"),
        Path("C:/ProgramData/chocolatey/bin/ffmpeg.exe"),
        Path(os.environ.get("USERPROFILE", "")) / "scoop" / "shims" / "ffmpeg.exe",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)

    return None


def convert_audio_to_wav_pcm16(
    audio_input: Union[bytes, BinaryIO, str, Path],
    target_sr: int = 16000,
    target_channels: int = 1,
    format_hint: Optional[str] = None,
) -> Tuple[np.ndarray, int, bytes]:
    """Convert any incoming audio file/bytes to 16kHz mono 16-bit PCM WAV.

    Format-agnostic conversion accepting browser recordings (.webm, .ogg)
    as well as standard audio formats (.mp3, .wav, .m4a, .flac, .aac).

    Args:
        audio_input: Raw audio bytes, file-like object, or path to audio file.
        target_sr: Target sampling rate in Hz (default: 16000).
        target_channels: Number of target audio channels (default: 1 for mono).
        format_hint: Optional format hint or extension (e.g. 'webm', 'ogg', 'mp3').

    Returns:
        Tuple of:
            - waveform (np.ndarray): 1D float32 audio waveform normalized to [-1.0, 1.0].
            - sample_rate (int): Resulting sample rate (16000).
            - wav_bytes (bytes): Valid 16-bit PCM WAV file binary content.

    Raises:
        AudioConversionError: If input cannot be decoded or converted.
    """
    raw_bytes: bytes
    detected_format: Optional[str] = format_hint.lower().lstrip(".") if format_hint else None

    if isinstance(audio_input, (str, Path)):
        path = Path(audio_input)
        if not path.is_file():
            raise AudioConversionError(f"Audio file not found at: {path}")
        if not detected_format and path.suffix:
            detected_format = path.suffix.lower().lstrip(".")
        with open(path, "rb") as f:
            raw_bytes = f.read()
    elif isinstance(audio_input, (bytes, bytearray)):
        raw_bytes = bytes(audio_input)
    elif hasattr(audio_input, "read"):
        if not detected_format and hasattr(audio_input, "name") and audio_input.name:
            detected_format = Path(audio_input.name).suffix.lower().lstrip(".")
        raw_bytes = audio_input.read()
        if hasattr(audio_input, "seek"):
            audio_input.seek(0)
    else:
        raise AudioConversionError(f"Unsupported audio input type: {type(audio_input)}")

    if not raw_bytes:
        raise AudioConversionError("Audio input data is empty.")

    # Infer container/codec format if not already known
    if not detected_format:
        detected_format = detect_audio_format(raw_bytes)

    logger.debug(
        "Processing audio input: %d bytes, detected format: '%s'",
        len(raw_bytes),
        detected_format or "unknown",
    )

    conversion_errors = []

    # -------------------------------------------------------------------------
    # Strategy 1: soundfile (libsndfile native in-memory decode)
    # Extremely fast; natively handles WAV, OGG (Opus and Vorbis), FLAC, MP3
    # -------------------------------------------------------------------------
    try:
        import soundfile as sf

        with io.BytesIO(raw_bytes) as in_buf:
            data, sr = sf.read(in_buf, dtype="float32", always_2d=True)

        waveform = _resample_and_downmix(data, sr, target_sr, target_channels)
        wav_bytes = _encode_pcm16_wav(waveform, target_sr)
        return waveform, target_sr, wav_bytes

    except Exception as sf_err:
        conversion_errors.append(f"soundfile decode: {sf_err}")
        logger.debug("soundfile failed to read audio (%s). Trying pydub/ffmpeg...", sf_err)

    # -------------------------------------------------------------------------
    # Strategy 2: pydub + ffmpeg
    # Handles WebM (Chrome MediaRecorder), OGG, MP3, M4A, AAC, and any container
    # Uses a named temporary file so ffmpeg has seekable random access.
    # -------------------------------------------------------------------------
    suffix = f".{detected_format}" if detected_format else ".tmp"
    tmp_path: Optional[Path] = None

    try:
        from pydub import AudioSegment

        # Ensure ffmpeg binary is discoverable by pydub
        ffmpeg_bin = _find_ffmpeg_executable()
        if ffmpeg_bin:
            AudioSegment.converter = ffmpeg_bin
            ffmpeg_dir = str(Path(ffmpeg_bin).parent)
            if ffmpeg_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{ffmpeg_dir};{os.environ.get('PATH', '')}"

        # Write to named temp file for seekable container parsing (crucial for webm & m4a)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
            tmp_file.write(raw_bytes)
            tmp_path = Path(tmp_file.name)

        try:
            segment = AudioSegment.from_file(
                str(tmp_path),
                format=detected_format if detected_format in {"webm", "ogg", "mp3", "wav", "m4a", "flac", "aac"} else None,
            )
        except Exception:
            # Retry without explicit format hint (let ffmpeg auto-probe)
            segment = AudioSegment.from_file(str(tmp_path))

        segment = segment.set_frame_rate(target_sr)
        segment = segment.set_channels(target_channels)
        segment = segment.set_sample_width(2)  # 2 bytes = 16-bit PCM

        out_buf = io.BytesIO()
        segment.export(out_buf, format="wav", codec="pcm_s16le")
        wav_bytes = out_buf.getvalue()

        # Convert to float32 numpy array normalized to [-1.0, 1.0]
        samples = np.array(segment.get_array_of_samples(), dtype=np.float32)
        max_val = float(2 ** (16 - 1))  # 32768.0
        waveform = samples / max_val

        return waveform, target_sr, wav_bytes

    except Exception as pydub_err:
        conversion_errors.append(f"pydub/ffmpeg: {pydub_err}")
        logger.debug("pydub decode failed: %s. Trying direct ffmpeg / librosa...", pydub_err)

    finally:
        if tmp_path and tmp_path.exists():
            try:
                tmp_path.unlink()
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Strategy 3: Direct ffmpeg subprocess invocation (if pydub had wrapping issue)
    # -------------------------------------------------------------------------
    ffmpeg_bin = _find_ffmpeg_executable()
    if ffmpeg_bin:
        tmp_in: Optional[Path] = None
        tmp_out: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f_in:
                f_in.write(raw_bytes)
                tmp_in = Path(f_in.name)

            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f_out:
                tmp_out = Path(f_out.name)

            cmd = [
                ffmpeg_bin,
                "-y",
                "-i", str(tmp_in),
                "-ar", str(target_sr),
                "-ac", str(target_channels),
                "-c:a", "pcm_s16le",
                "-f", "wav",
                str(tmp_out),
            ]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            import soundfile as sf
            data, sr = sf.read(str(tmp_out), dtype="float32", always_2d=True)
            waveform = _resample_and_downmix(data, sr, target_sr, target_channels)
            wav_bytes = tmp_out.read_bytes()
            return waveform, target_sr, wav_bytes
        except Exception as ffmpeg_err:
            conversion_errors.append(f"direct ffmpeg: {ffmpeg_err}")
        finally:
            for p in (tmp_in, tmp_out):
                if p and p.exists():
                    try:
                        p.unlink()
                    except Exception:
                        pass

    # -------------------------------------------------------------------------
    # Strategy 4: librosa fallback
    # -------------------------------------------------------------------------
    tmp_librosa: Optional[Path] = None
    try:
        import librosa

        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f_librosa:
            f_librosa.write(raw_bytes)
            tmp_librosa = Path(f_librosa.name)

        data, sr = librosa.load(str(tmp_librosa), sr=target_sr, mono=(target_channels == 1))
        waveform = np.clip(data, -1.0, 1.0).astype(np.float32)
        wav_bytes = _encode_pcm16_wav(waveform, target_sr)
        return waveform, target_sr, wav_bytes

    except Exception as librosa_err:
        conversion_errors.append(f"librosa: {librosa_err}")

    finally:
        if tmp_librosa and tmp_librosa.exists():
            try:
                tmp_librosa.unlink()
            except Exception:
                pass

    # All conversion attempts failed
    err_summary = "; ".join(conversion_errors)
    logger.error("Audio conversion failed across all strategies: %s", err_summary)
    raise AudioConversionError(
        f"Failed to convert audio input (detected format: '{detected_format or 'unknown'}'). "
        f"Supported formats include webm, ogg, mp3, wav, m4a, flac. Details: {err_summary}"
    )


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


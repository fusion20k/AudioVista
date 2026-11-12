"""Decoding audio files into numpy sample arrays.

WAV files are read with the standard-library ``wave`` module; MP3 and FLAC
(and any WAV that ``wave`` cannot read) are decoded with ffmpeg, which is
bundled with the ``imageio-ffmpeg`` package.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ..config import SAMPLE_RATE
from ..errors import AudioDecodeError

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0  # CREATE_NO_WINDOW


def ffmpeg_executable() -> str:
    """Return a path to an ffmpeg binary or raise ``AudioDecodeError``."""
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        found = shutil.which("ffmpeg")
        if found:
            return found
    raise AudioDecodeError(
        "ffmpeg was not found. Install it with 'pip install imageio-ffmpeg'."
    )


@dataclass
class DecodedAudio:
    samples: np.ndarray  # float32, shape (n, 2), range [-1, 1]
    sample_rate: int
    tags: dict = field(default_factory=dict)

    @property
    def duration(self) -> float:
        return len(self.samples) / float(self.sample_rate)


def decode_audio(path: str | Path, sample_rate: int = SAMPLE_RATE) -> DecodedAudio:
    """Decode *path* into stereo float32 samples at *sample_rate*."""
    path = Path(path)
    if not path.is_file():
        raise AudioDecodeError(f"File not found: {path.name}")

    if path.suffix.lower() == ".wav":
        try:
            samples, rate = _read_wav(path)
            samples = _resample(samples, rate, sample_rate)
            return DecodedAudio(samples, sample_rate, {})
        except (wave.Error, EOFError, ValueError):
            pass  # e.g. 32-bit float WAV -> let ffmpeg handle it
    return _decode_with_ffmpeg(path, sample_rate)


def _read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as wf:
        channels = wf.getnchannels()
        width = wf.getsampwidth()
        rate = wf.getframerate()
        raw = wf.readframes(wf.getnframes())

    if width == 1:
        data = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif width == 2:
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    elif width == 3:
        b = np.frombuffer(raw, dtype=np.uint8).reshape(-1, 3).astype(np.int32)
        ints = b[:, 0] | (b[:, 1] << 8) | (b[:, 2] << 16)
        ints = np.where(ints & 0x800000, ints - 0x1000000, ints)
        data = ints.astype(np.float32) / 8388608.0
    elif width == 4:
        data = np.frombuffer(raw, dtype="<i4").astype(np.float32) / 2147483648.0
    else:
        raise ValueError(f"Unsupported sample width {width}")

    if channels < 1 or len(data) < channels:
        raise ValueError("WAV file has no audio data")
    data = data[: (len(data) // channels) * channels].reshape(-1, channels)
    return _to_stereo(data), rate


def _decode_with_ffmpeg(path: Path, sample_rate: int) -> DecodedAudio:
    exe = ffmpeg_executable()
    cmd = [exe, "-hide_banner", "-nostdin", "-i", str(path), "-vn",
           "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sample_rate), "-"]
    try:
        proc = subprocess.run(cmd, capture_output=True, creationflags=_NO_WINDOW, timeout=600)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise AudioDecodeError(f"Could not run ffmpeg: {exc}") from exc

    stderr = proc.stderr.decode("utf-8", errors="replace")
    if proc.returncode != 0 or not proc.stdout:
        raise AudioDecodeError(f"'{path.name}' is not a readable audio file.")

    data = np.frombuffer(proc.stdout, dtype="<f4")
    data = data[: (len(data) // 2) * 2].reshape(-1, 2).copy()
    if len(data) == 0:
        raise AudioDecodeError(f"'{path.name}' contains no audio.")
    np.clip(data, -1.0, 1.0, out=data)
    return DecodedAudio(data, sample_rate, parse_ffmpeg_tags(stderr))


_TAG_RE = re.compile(r"^\s{4,}(title|artist|album)\s*:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)


def parse_ffmpeg_tags(stderr: str) -> dict:
    """Pull title/artist/album tags out of ffmpeg's banner output."""
    tags: dict[str, str] = {}
    for key, value in _TAG_RE.findall(stderr):
        tags.setdefault(key.lower(), value)
    return tags


def _to_stereo(data: np.ndarray) -> np.ndarray:
    if data.shape[1] == 1:
        data = np.repeat(data, 2, axis=1)
    elif data.shape[1] > 2:
        data = data[:, :2]
    return np.ascontiguousarray(data, dtype=np.float32)


def _resample(samples: np.ndarray, src_rate: int, dst_rate: int) -> np.ndarray:
    if src_rate == dst_rate or len(samples) == 0:
        return samples
    n_out = max(1, int(round(len(samples) * dst_rate / src_rate)))
    x_old = np.arange(len(samples), dtype=np.float64)
    x_new = np.linspace(0, len(samples) - 1, n_out)
    out = np.empty((n_out, samples.shape[1]), dtype=np.float32)
    for ch in range(samples.shape[1]):
        out[:, ch] = np.interp(x_new, x_old, samples[:, ch])
    return out


def write_wav(path: str | Path, samples: np.ndarray, sample_rate: int = SAMPLE_RATE) -> None:
    """Write float samples (mono or stereo) to a 16-bit WAV file."""
    data = np.asarray(samples, dtype=np.float32)
    if data.ndim == 1:
        data = data[:, None]
    pcm = (np.clip(data, -1.0, 1.0) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(pcm.shape[1])
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())

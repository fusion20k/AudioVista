"""Domain classes from the AudioVista design class diagram (DCD).

Account, AudioFile, Visualization, VisualSettings and Session live here;
ExportFile is in :mod:`audiovista.export` and Database in
:mod:`audiovista.database`.
"""

from __future__ import annotations

import hashlib
import hmac
import re
import secrets
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Iterator

from .errors import NoAudioError, ValidationError

# --------------------------------------------------------------------- Account

_HASH_ALGO = "pbkdf2_sha256"
_HASH_ITERATIONS = 120_000


@dataclass
class Account:
    user_id: int | None
    username: str
    email: str
    password_hash: str
    created_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))

    @staticmethod
    def hash_password(password: str, salt: str | None = None,
                      iterations: int = _HASH_ITERATIONS) -> str:
        """Salted PBKDF2-SHA256 hash, stored as ``algo$iterations$salt$hash``."""
        salt = salt or secrets.token_hex(16)
        digest = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"),
                                     bytes.fromhex(salt), iterations)
        return f"{_HASH_ALGO}${iterations}${salt}${digest.hex()}"

    def verify_password(self, password: str) -> bool:
        try:
            algo, iterations, salt, _digest = self.password_hash.split("$")
            if algo != _HASH_ALGO:
                return False
            expected = Account.hash_password(password, salt, int(iterations))
        except (ValueError, AttributeError):
            return False
        return hmac.compare_digest(expected, self.password_hash)


# ------------------------------------------------------------------- AudioFile

class AudioFile:
    """An uploaded track. Samples and FFT analysis are loaded on demand."""

    def __init__(self, file_path: str | Path, audio_id: int | None = None,
                 file_name: str | None = None, title: str | None = None,
                 artist: str = "", duration: float = 0.0, file_size: int | None = None):
        self.file_path = str(file_path)
        p = Path(self.file_path)
        self.audio_id = audio_id
        self.file_name = file_name or p.name
        self.file_type = Path(self.file_name).suffix.lower().lstrip(".")
        self.file_size = file_size if file_size is not None else (p.stat().st_size if p.exists() else 0)
        self.title = title or Path(self.file_name).stem
        self.artist = artist
        self.duration = duration
        self.sample_rate = 0
        self.samples = None
        self._analyzer = None

    @property
    def is_loaded(self) -> bool:
        return self.samples is not None

    @property
    def display_name(self) -> str:
        return f"{self.artist} - {self.title}" if self.artist else self.title

    def extract_metadata(self) -> None:
        """Decode the file and fill in duration, sample rate, title and artist."""
        from .audio.decoder import decode_audio

        decoded = decode_audio(self.file_path)
        self.samples = decoded.samples
        self.sample_rate = decoded.sample_rate
        self.duration = decoded.duration
        p = Path(self.file_path)
        if p.exists():
            self.file_size = p.stat().st_size
        if decoded.tags.get("title"):
            self.title = decoded.tags["title"]
        if decoded.tags.get("artist"):
            self.artist = decoded.tags["artist"]

    def load(self) -> "AudioFile":
        """Make sure samples and analysis are ready (safe to call repeatedly)."""
        if not self.is_loaded:
            self.extract_metadata()
        self.get_analyzer()
        return self

    def get_analyzer(self):
        if self._analyzer is None:
            if not self.is_loaded:
                raise NoAudioError("Audio has not been loaded yet.")
            from .audio.analyzer import AudioAnalyzer

            self._analyzer = AudioAnalyzer(self.samples, self.sample_rate)
        return self._analyzer

    def to_record(self) -> dict:
        return {"audio_id": self.audio_id, "file_name": self.file_name,
                "file_type": self.file_type, "file_size": self.file_size,
                "file_path": self.file_path, "title": self.title,
                "artist": self.artist, "duration": self.duration}

    @classmethod
    def from_record(cls, rec: dict) -> "AudioFile":
        return cls(rec["file_path"], audio_id=rec["audio_id"], file_name=rec["file_name"],
                   title=rec["title"], artist=rec.get("artist") or "",
                   duration=rec.get("duration") or 0.0, file_size=rec.get("file_size"))

    def __repr__(self) -> str:
        return f"AudioFile(id={self.audio_id}, title={self.title!r}, {self.duration:.1f}s)"


# --------------------------------------------------------------- VisualSettings

# Style keys are stored in saved visualizations, so they stay stable even
# though the styles have been redesigned (they map to the current designs).
WAVE_TYPES = ("bars", "circular")
WAVE_TYPE_LABELS = {"bars": "Spectrum", "circular": "Radial"}
# Styles that were removed: saved visualizations that used them open as Radial
# (the old "particles" style becomes Radial with the particles option on).
LEGACY_WAVE_TYPES = {"meters": ("circular", False), "particles": ("circular", True),
                     "waveform": ("bars", False)}
# Backgrounds are skies at different times of day.
BACKGROUNDS = ("golden", "sunset", "dawn", "twilight", "night")
BACKGROUND_LABELS = {"golden": "Golden", "sunset": "Sunset", "dawn": "Dawn",
                     "twilight": "Twilight", "night": "Night"}
# Backgrounds from earlier versions of the app, mapped to the closest sky.
LEGACY_BACKGROUNDS = {"walnut": "golden", "ember": "sunset", "sepia": "dawn", "dusk": "twilight",
                      "charcoal": "night", "midnight": "night", "black": "night", "nebula": "twilight",
                      "ocean": "dawn"}
COLOR_PRESETS = ("#ffb35c", "#ff8a7a", "#ff7eb6", "#c3a6ff",
                 "#7fc8ff", "#6fe3cf", "#a8e27a", "#fff1d6")
_HEX_RE = re.compile(r"^#[0-9a-fA-F]{6}$")


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not _HEX_RE.match(value):
        raise ValidationError(f"'{value}' is not a colour like #12ab34.", "color")
    return tuple(int(value[i:i + 2], 16) for i in (1, 3, 5))


@dataclass
class VisualSettings:
    """How a visualization looks. Matches the VisualSettings class in the DCD."""

    settings_id: int | None = None
    color: str = "#ffb35c"
    secondary_color: str = "#7fc8ff"
    background: str = "sunset"
    wave_type: str = "bars"
    sensitivity: float = 1.0     # 0.5 (calm) .. 2.0 (very reactive)
    smoothing: float = 0.6       # 0 (jittery) .. 0.95 (very smooth)
    glow: bool = True
    show_title: bool = True
    atmosphere: bool = True      # "Haze": soft vignette and a warm glow from below
    particles: bool = False      # swirling particles around the Radial style

    def validate(self) -> "VisualSettings":
        hex_to_rgb(self.color)
        hex_to_rgb(self.secondary_color)
        if self.wave_type not in WAVE_TYPES:
            raise ValidationError(f"Unknown visualizer style '{self.wave_type}'.", "wave_type")
        if self.background not in BACKGROUNDS:
            raise ValidationError(f"Unknown background '{self.background}'.", "background")
        if not isinstance(self.sensitivity, (int, float)) or not 0.5 <= self.sensitivity <= 2.0:
            raise ValidationError("Sensitivity must be between 0.5 and 2.0.", "sensitivity")
        if not isinstance(self.smoothing, (int, float)) or not 0.0 <= self.smoothing <= 0.95:
            raise ValidationError("Smoothing must be between 0 and 0.95.", "smoothing")
        return self

    @property
    def primary_rgb(self) -> tuple[int, int, int]:
        return hex_to_rgb(self.color)

    @property
    def secondary_rgb(self) -> tuple[int, int, int]:
        return hex_to_rgb(self.secondary_color)

    def to_dict(self) -> dict:
        data = asdict(self)
        data.pop("settings_id")
        return data

    @classmethod
    def from_dict(cls, data: dict, settings_id: int | None = None) -> "VisualSettings":
        """Build validated settings; unknown keys are ignored, missing keys use defaults."""
        if not isinstance(data, dict):
            raise ValidationError("Settings must be a dictionary.")
        known = {f.name for f in fields(cls)} - {"settings_id"}
        if data.get("background") in LEGACY_BACKGROUNDS:  # saved by an earlier version
            data = {**data, "background": LEGACY_BACKGROUNDS[data["background"]]}
        if data.get("wave_type") in LEGACY_WAVE_TYPES:
            style, particles = LEGACY_WAVE_TYPES[data["wave_type"]]
            data = {**data, "wave_type": style, "particles": particles or bool(data.get("particles"))}
        if "vintage" in data and "atmosphere" not in data:
            data = {**data, "atmosphere": bool(data["vintage"])}
        settings = cls(settings_id=settings_id, **{k: v for k, v in data.items() if k in known})
        return settings.validate()

    def copy(self, **changes) -> "VisualSettings":
        data = asdict(self)
        data.update(changes)
        return VisualSettings(**data).validate()


# --------------------------------------------------------------- Visualization

class Visualization:
    """An audio file plus the settings used to visualise it."""

    def __init__(self, audio_file: AudioFile, settings: VisualSettings | None = None,
                 name: str | None = None, visualization_id: int | None = None,
                 date_created: str | None = None, save_status: bool = False):
        self.visualization_id = visualization_id
        self.audio_file = audio_file
        self.settings = (settings or VisualSettings()).validate()
        self.name = name or audio_file.title
        self.date_created = date_created or datetime.now().isoformat(timespec="seconds")
        self.save_status = save_status

    def set_title(self, title: str) -> None:
        title = (title or "").strip()
        if not title:
            raise ValidationError("Please give your visualization a name.", "title")
        if len(title) > 80:
            raise ValidationError("Names can be at most 80 characters.", "title")
        self.name = title

    def update_settings(self, **changes) -> None:
        new = self.settings.copy(**changes)
        if new != self.settings:
            self.settings = new
            self.save_status = False

    def get_audio(self) -> AudioFile:
        return self.audio_file

    def get_frames(self, resolution: tuple[int, int], fps: int = 30,
                   start: float = 0.0, duration: float | None = None) -> Iterator:
        """Yield rendered ``pygame.Surface`` frames for video export."""
        from .visuals.engine import VisualizerEngine

        self.audio_file.load()
        engine = VisualizerEngine(self.audio_file.get_analyzer(), self.settings,
                                  title=self.audio_file.display_name)
        return engine.frames(resolution, fps, start, duration)

    def frame_count(self, fps: int, start: float = 0.0, duration: float | None = None) -> int:
        total = self.audio_file.duration - start
        if duration is not None:
            total = min(total, duration)
        return max(1, int(total * fps))


# --------------------------------------------------------------------- Session

class PlaybackQueue:
    """'Up next' list of tracks (the sound queue from the requirements)."""

    def __init__(self):
        self._items: list[AudioFile] = []

    def add(self, audio: AudioFile) -> None:
        self._items.append(audio)

    def remove(self, index: int) -> AudioFile:
        if not 0 <= index < len(self._items):
            raise IndexError("queue index out of range")
        return self._items.pop(index)

    def pop_next(self) -> AudioFile | None:
        return self._items.pop(0) if self._items else None

    def clear(self) -> None:
        self._items.clear()

    @property
    def items(self) -> list[AudioFile]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


class Session:
    """State of a logged-in user: who they are and what they're working on."""

    def __init__(self, account: Account | None = None):
        self.account = account
        self.current_audio: AudioFile | None = None
        self.current_visualization: Visualization | None = None
        self.queue = PlaybackQueue()

    @property
    def is_authenticated(self) -> bool:
        return self.account is not None

    def get_account_id(self) -> int | None:
        return self.account.user_id if self.account else None

    def get_current_audio(self) -> AudioFile | None:
        return self.current_audio

    def set_current_audio(self, audio_file: AudioFile | None) -> None:
        self.current_audio = audio_file

    def get_current_visualization(self) -> Visualization | None:
        return self.current_visualization

    def set_current_visualization(self, visualization: Visualization | None) -> None:
        self.current_visualization = visualization

    def clear(self) -> None:
        self.account = None
        self.current_audio = None
        self.current_visualization = None
        self.queue.clear()

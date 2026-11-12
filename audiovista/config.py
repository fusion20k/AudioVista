"""Application-wide constants and file locations."""

from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "AudioVista"
APP_VERSION = "1.0.0"

# Audio
SAMPLE_RATE = 44100
SUPPORTED_FORMATS = (".mp3", ".wav", ".flac")
MAX_FILE_SIZE_MB = 200
MAX_FILE_SIZE = MAX_FILE_SIZE_MB * 1024 * 1024

# Accounts
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 5 * 60
MIN_PASSWORD_LENGTH = 8

# Export
EXPORT_RESOLUTIONS = {
    "480p": (854, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
    "Square": (1080, 1080),
    "Vertical": (1080, 1920),
}
EXPORT_FORMATS = ("mp4",)
EXPORT_FPS_OPTIONS = (30, 60)


def data_dir() -> Path:
    """Folder that holds the database and the users' audio libraries.

    Can be overridden with the ``AUDIOVISTA_DATA_DIR`` environment variable
    (the test-suite uses this to work in a temporary folder).
    """
    override = os.environ.get("AUDIOVISTA_DATA_DIR")
    if override:
        path = Path(override)
    elif sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        path = Path(base) / APP_NAME
    else:
        path = Path.home() / ".audiovista"
    path.mkdir(parents=True, exist_ok=True)
    return path


def database_path() -> Path:
    return data_dir() / "audiovista.db"


def library_dir() -> Path:
    path = data_dir() / "library"
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_export_dir() -> Path:
    videos = Path.home() / "Videos"
    base = videos if videos.exists() else Path.home()
    return base / APP_NAME

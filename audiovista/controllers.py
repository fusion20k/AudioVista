"""Controllers from the design class diagram. They hold the application
logic; the UI only calls these and displays the results."""

from __future__ import annotations

import re
import shutil
import time
import uuid
from pathlib import Path
from typing import Callable

from .config import (EXPORT_FORMATS, EXPORT_RESOLUTIONS, LOCKOUT_SECONDS, MAX_FILE_SIZE,
                     MAX_FILE_SIZE_MB, MAX_LOGIN_ATTEMPTS, MIN_PASSWORD_LENGTH,
                     SUPPORTED_FORMATS)
from .database import Database
from .errors import (AccountExistsError, AccountLockedError, AuthenticationError,
                     FileTooLargeError, NoAudioError, NotSavedError, UnsupportedFormatError,
                     ValidationError)
from .models import Account, AudioFile, Session, Visualization, VisualSettings

_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[A-Za-z]{2,}$")
_USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,24}$")


class AccountController:
    """Create Account and Log In use cases."""

    def __init__(self, database: Database, clock: Callable[[], float] = time.time):
        self.db = database
        self.clock = clock

    def validate_input(self, username: str, email: str, password: str) -> None:
        if not _USERNAME_RE.match(username or ""):
            raise ValidationError("Username must be 3-24 characters: letters, numbers, . _ -",
                                  "username")
        if not _EMAIL_RE.match(email or ""):
            raise ValidationError("Please enter a valid e-mail address.", "email")
        if len(password or "") < MIN_PASSWORD_LENGTH:
            raise ValidationError(f"Password must be at least {MIN_PASSWORD_LENGTH} characters.",
                                  "password")
        if not re.search(r"[A-Za-z]", password) or not re.search(r"\d", password):
            raise ValidationError("Password must contain at least one letter and one number.",
                                  "password")

    def create_account(self, username: str, email: str, password: str,
                       confirm_password: str | None = None) -> int:
        username, email = (username or "").strip(), (email or "").strip().lower()
        self.validate_input(username, email, password)
        if confirm_password is not None and confirm_password != password:
            raise ValidationError("Passwords do not match.", "confirm")
        if self.db.find_account_by_email(email):
            raise AccountExistsError("An account with this e-mail already exists. Please log in.",
                                     "email")
        if self.db.find_account_by_username(username):
            raise AccountExistsError("That username is taken.", "username")
        account = Account(None, username, email, Account.hash_password(password))
        return self.db.save_account(account)

    def login(self, email_or_username: str, password: str) -> Session:
        ident = (email_or_username or "").strip()
        if not ident or not password:
            raise AuthenticationError("Please enter your e-mail and password.")
        account = (self.db.find_account_by_email(ident.lower()) if "@" in ident
                   else self.db.find_account_by_username(ident))
        if account is None:
            raise AuthenticationError("No account was found with those details.")

        attempts, locked_until = self.db.get_login_state(account.user_id)
        now = self.clock()
        if locked_until > now:
            remaining = int(locked_until - now) + 1
            raise AccountLockedError(
                f"Too many failed attempts. Try again in {remaining // 60}m {remaining % 60}s.",
                remaining)

        if not account.verify_password(password):
            attempts += 1
            if attempts >= MAX_LOGIN_ATTEMPTS:
                self.db.set_login_state(account.user_id, 0, now + LOCKOUT_SECONDS)
                raise AccountLockedError(
                    f"Too many failed attempts. Account locked for {LOCKOUT_SECONDS // 60} minutes.",
                    LOCKOUT_SECONDS)
            self.db.set_login_state(account.user_id, attempts, 0)
            left = MAX_LOGIN_ATTEMPTS - attempts
            raise AuthenticationError(f"Incorrect password. {left} attempt{'s' * (left != 1)} left.")

        self.db.set_login_state(account.user_id, 0, 0)
        return Session(account)


class UploadController:
    """Upload Audio use case."""

    def __init__(self, database: Database, session: Session, library_root: str | Path):
        self.db = database
        self.session = session
        self.library_root = Path(library_root)

    @staticmethod
    def validate_format(file_path: str | Path) -> bool:
        """True if the file is an MP3/WAV/FLAC within the size limit, else raises."""
        p = Path(file_path)
        if p.suffix.lower() not in SUPPORTED_FORMATS:
            raise UnsupportedFormatError(
                f"'{p.name}' is not supported. Please choose an MP3, WAV or FLAC file.", "file")
        if not p.is_file():
            raise ValidationError(f"File not found: {p.name}", "file")
        size = p.stat().st_size
        if size == 0:
            raise ValidationError(f"'{p.name}' is empty.", "file")
        if size > MAX_FILE_SIZE:
            raise FileTooLargeError(f"'{p.name}' is larger than the {MAX_FILE_SIZE_MB} MB limit.",
                                    "file")
        return True

    def _require_account(self) -> int:
        account_id = self.session.get_account_id()
        if account_id is None:
            raise AuthenticationError("Please log in first.")
        return account_id

    def upload_audio(self, file_path: str | Path) -> AudioFile:
        """Validate, decode and store a file in the user's library; returns it loaded."""
        account_id = self._require_account()
        self.validate_format(file_path)
        src = Path(file_path)

        existing = self.db.find_audio(account_id, src.name, src.stat().st_size)
        if existing and Path(existing.file_path).is_file():
            existing.load()
            self.session.set_current_audio(existing)
            return existing

        audio = AudioFile(src)
        audio.extract_metadata()  # raises AudioDecodeError for corrupt files
        audio.get_analyzer()

        dest_dir = self.library_root / str(account_id)
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{uuid.uuid4().hex[:8]}_{src.name}"
        shutil.copy2(src, dest)
        audio.file_path = str(dest)

        self.db.save_audio(account_id, audio)
        self.session.set_current_audio(audio)
        return audio

    def list_library(self) -> list[AudioFile]:
        return self.db.list_audio(self._require_account())

    def load_from_library(self, audio_id: int) -> AudioFile:
        audio = self.db.get_audio(self._require_account(), audio_id)
        if audio is None:
            raise NoAudioError("That track is no longer in your library.")
        if not Path(audio.file_path).is_file():
            raise NoAudioError(f"The file for '{audio.title}' is missing. Please upload it again.")
        return audio.load()

    def remove_from_library(self, audio_id: int) -> None:
        account_id = self._require_account()
        audio = self.db.get_audio(account_id, audio_id)
        if audio is None:
            return
        self.db.delete_audio(account_id, audio_id)
        p = Path(audio.file_path)
        try:
            if self.library_root.resolve() in p.resolve().parents:
                p.unlink(missing_ok=True)
        except OSError:
            pass


class VisualizerController:
    """Generate Visuals use case."""

    def __init__(self, session: Session):
        self.session = session

    def generate_visuals(self, style: str | None = None,
                         settings: VisualSettings | None = None) -> Visualization:
        audio = self.session.get_current_audio()
        if audio is None:
            raise NoAudioError("Upload or choose a track first.")
        audio.load()
        settings = settings or VisualSettings()
        if style:
            settings = settings.copy(wave_type=style)
        visualization = Visualization(audio, settings)
        self.session.set_current_visualization(visualization)
        return visualization

    def update_settings(self, **changes) -> VisualSettings:
        vis = self.session.get_current_visualization()
        if vis is None:
            raise NoAudioError("Nothing to change yet - load a track first.")
        vis.update_settings(**changes)
        return vis.settings


class ResultController:
    """Save Result use case (and re-opening saved results)."""

    def __init__(self, database: Database, session: Session):
        self.db = database
        self.session = session

    def save_result(self, title: str) -> int:
        vis = self.session.get_current_visualization()
        if vis is None:
            raise NoAudioError("There is no visualization to save yet.")
        account_id = self.session.get_account_id()
        if account_id is None:
            raise AuthenticationError("Please log in first.")
        vis.set_title(title)
        result_id = self.db.save_visualization(account_id, vis)
        vis.save_status = True
        return result_id

    def list_results(self) -> list[dict]:
        return self.db.list_visualizations(self.session.get_account_id())

    def open_result(self, visualization_id: int) -> Visualization:
        vis = self.db.get_visualization(self.session.get_account_id(), visualization_id)
        if vis is None:
            raise NoAudioError("That saved visualization no longer exists.")
        if not Path(vis.audio_file.file_path).is_file():
            raise NoAudioError("The audio for this visualization is missing.")
        vis.audio_file.load()
        return vis

    def delete_result(self, visualization_id: int) -> None:
        self.db.delete_visualization(self.session.get_account_id(), visualization_id)


class ExportController:
    """Export Result use case."""

    def __init__(self, database: Database, session: Session):
        self.db = database
        self.session = session

    def export_result(self, file_format: str, resolution: str, output_path: str | Path,
                      fps: int = 30, duration: float | None = None,
                      progress: Callable[[float], None] | None = None,
                      cancel_event=None):
        from .export import ExportFile

        vis = self.session.get_current_visualization()
        if vis is None:
            raise NoAudioError("There is nothing to export yet.")
        if not vis.save_status or vis.visualization_id is None:
            raise NotSavedError("Please save your visualization before exporting it.")
        file_format = file_format.lower()
        if file_format not in EXPORT_FORMATS:
            raise ValidationError(f"Unsupported export format '{file_format}'.", "format")
        if resolution not in EXPORT_RESOLUTIONS:
            raise ValidationError(f"Unknown resolution '{resolution}'.", "resolution")

        size = EXPORT_RESOLUTIONS[resolution]
        frames = vis.get_frames(size, fps, 0.0, duration)
        export = ExportFile(frames, vis.get_audio(), file_format, output_path, size, fps,
                            frame_count=vis.frame_count(fps, 0.0, duration), duration=duration)
        export.encode(progress=progress, cancel_event=cancel_event)
        export.export_id = self.db.save_export(vis.visualization_id, export.file_name, file_format)
        return export

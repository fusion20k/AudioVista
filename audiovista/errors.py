"""Exceptions raised by AudioVista. Every message is safe to show to the user."""


class AudioVistaError(Exception):
    """Base class for all expected (user-facing) errors."""


class ValidationError(AudioVistaError):
    """User input did not pass validation.

    ``field`` names the offending input (e.g. ``"email"``) so the UI can
    highlight it; it is ``None`` when the error is not tied to one field.
    """

    def __init__(self, message: str, field: str | None = None):
        super().__init__(message)
        self.field = field


class AccountExistsError(ValidationError):
    """Registration used a username or e-mail that is already taken."""


class AuthenticationError(AudioVistaError):
    """Login credentials were wrong or the account does not exist."""


class AccountLockedError(AuthenticationError):
    """Too many failed logins; the account is temporarily locked."""

    def __init__(self, message: str, seconds_remaining: int):
        super().__init__(message)
        self.seconds_remaining = seconds_remaining


class UnsupportedFormatError(ValidationError):
    """The chosen file is not an MP3, WAV or FLAC file."""


class FileTooLargeError(ValidationError):
    """The chosen file is bigger than the allowed maximum."""


class AudioDecodeError(AudioVistaError):
    """The audio file could not be read or decoded."""


class NoAudioError(AudioVistaError):
    """An operation needs a loaded track but none is loaded."""


class NotSavedError(AudioVistaError):
    """Export was requested for a visualization that has not been saved."""


class ExportError(AudioVistaError):
    """Encoding the video failed."""


class ExportCancelled(ExportError):
    """The user cancelled an export in progress."""

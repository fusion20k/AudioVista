"""ExportFile: encodes rendered frames plus the original audio into an MP4."""

from __future__ import annotations

import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import pygame

from .audio.decoder import ffmpeg_executable
from .config import real_path
from .errors import ExportCancelled, ExportError
from .models import AudioFile

_NO_WINDOW = 0x08000000 if sys.platform == "win32" else 0


def _surface_bytes(surface: pygame.Surface) -> bytes:
    to_bytes = getattr(pygame.image, "tobytes", None) or pygame.image.tostring
    return to_bytes(surface, "RGB")


class ExportFile:
    def __init__(self, frames: Iterable[pygame.Surface], audio_file: AudioFile, file_format: str,
                 output_path: str | Path, resolution: tuple[int, int], fps: int = 30,
                 frame_count: int | None = None, duration: float | None = None):
        self.export_id: int | None = None
        self.frames = frames
        self.audio_file = audio_file
        self.file_format = file_format.lower()
        self.output_path = Path(output_path).with_suffix("." + self.file_format)
        self.file_name = self.output_path.name
        self.resolution = resolution
        self.fps = fps
        self.frame_count = frame_count
        self.duration = duration
        self.export_date: str | None = None

    def build_command(self, exe: str) -> list[str]:
        w, h = self.resolution
        cmd = [exe, "-y", "-hide_banner", "-loglevel", "error",
               "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", str(self.fps),
               "-i", "-"]
        if self.duration is not None:
            cmd += ["-t", f"{self.duration:.3f}"]
        cmd += ["-i", real_path(self.audio_file.file_path),
                "-map", "0:v:0", "-map", "1:a:0",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "20", "-pix_fmt", "yuv420p",
                "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart",
                str(self.output_path)]
        return cmd

    def encode(self, progress: Callable[[float], None] | None = None, cancel_event=None) -> Path:
        """Render all frames through ffmpeg. Returns the path of the finished video."""
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        exe = ffmpeg_executable()
        with tempfile.TemporaryFile() as errlog:
            try:
                proc = subprocess.Popen(self.build_command(exe), stdin=subprocess.PIPE,
                                        stdout=subprocess.DEVNULL, stderr=errlog,
                                        creationflags=_NO_WINDOW)
            except OSError as exc:
                raise ExportError(f"Could not start ffmpeg: {exc}") from exc

            cancelled = False
            try:
                for i, surface in enumerate(self.frames):
                    if cancel_event is not None and cancel_event.is_set():
                        cancelled = True
                        break
                    proc.stdin.write(_surface_bytes(surface))
                    if progress and self.frame_count:
                        progress(min(1.0, (i + 1) / self.frame_count))
            except (BrokenPipeError, OSError):
                pass  # ffmpeg exited early; its error is reported below
            finally:
                try:
                    proc.stdin.close()
                except OSError:
                    pass

            if cancelled:
                proc.kill()
                proc.wait()
                self.output_path.unlink(missing_ok=True)
                raise ExportCancelled("Export cancelled.")

            code = proc.wait()
            if code != 0:
                errlog.seek(0)
                detail = errlog.read().decode("utf-8", "replace").strip().splitlines()
                self.output_path.unlink(missing_ok=True)
                raise ExportError("Video encoding failed" + (f": {detail[-1]}" if detail else "."))

        self.export_date = datetime.now().isoformat(timespec="seconds")
        if progress:
            progress(1.0)
        return self.output_path

"""Modal dialogs (save, export, confirm) and native file pickers."""

from __future__ import annotations

import subprocess
import sys
import threading
from pathlib import Path

import pygame

from ..config import EXPORT_RESOLUTIONS, default_export_dir
from ..errors import AudioVistaError, ExportCancelled
from . import theme as T
from .theme import px
from .widgets import Button, ChoiceGroup, Modal, TextInput

# --------------------------------------------------------- native pickers
# Tk runs in a worker thread (via app.run_task) so the pygame window keeps
# drawing while the Windows file dialog is open.


def ask_open_audio() -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.askopenfilename(
            parent=root, title="Choose an audio file",
            filetypes=[("Audio files", "*.mp3 *.wav *.flac"), ("MP3", "*.mp3"), ("WAV", "*.wav"),
                       ("FLAC", "*.flac"), ("All files", "*.*")])
    finally:
        root.destroy()
    return path or None


def ask_save_video(initial: Path) -> str | None:
    import tkinter as tk
    from tkinter import filedialog

    initial.parent.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        path = filedialog.asksaveasfilename(
            parent=root, title="Export video", initialdir=str(initial.parent),
            initialfile=initial.name, defaultextension=".mp4", filetypes=[("MP4 video", "*.mp4")])
    finally:
        root.destroy()
    return path or None


def reveal_in_folder(path: Path) -> None:
    try:
        if sys.platform == "win32":
            subprocess.Popen(["explorer", "/select,", str(path)])
        elif sys.platform == "darwin":
            subprocess.Popen(["open", "-R", str(path)])
        else:
            subprocess.Popen(["xdg-open", str(path.parent)])
    except OSError:
        pass


def safe_filename(name: str) -> str:
    cleaned = "".join(c if c.isalnum() or c in " -_." else "_" for c in name).strip()
    return cleaned or "visualization"


# ----------------------------------------------------------------- dialogs

class ConfirmDialog(Modal):
    width, height = 440, 220

    def __init__(self, app, title, message, confirm_label, on_confirm, danger=True):
        super().__init__(app)
        self.title = title
        self.message = message
        self.on_confirm = on_confirm
        self.cancel = Button("Cancel", self.close, kind="secondary")
        self.ok = Button(confirm_label, self._confirm, kind="danger" if danger else "primary")
        self.widgets = [self.cancel, self.ok]

    def _confirm(self):
        self.close()
        self.on_confirm()

    def layout(self, size):
        super().layout(size)
        self.ok.rect = pygame.Rect(self.rect.right - px(28 + 140), self.rect.bottom - px(72), px(140), px(44))
        self.cancel.rect = pygame.Rect(self.ok.rect.x - px(12 + 110), self.rect.bottom - px(72), px(110), px(44))

    def draw_content(self, surf):
        for i, line in enumerate(T.wrap_text(self.message, 14, self.rect.w - px(56))):
            T.draw_text(surf, line, (self.rect.x + px(28), self.rect.y + px(66 + i * 21)), 14, T.TEXT_DIM)


class SaveDialog(Modal):
    """Save Result: asks for a title, then stores the visualization."""

    width, height = 460, 250
    title = "Save visualization"

    def __init__(self, app, studio, after_save=None):
        super().__init__(app)
        self.studio = studio
        self.after_save = after_save
        vis = app.session.get_current_visualization()
        self.input = TextInput("Name your visualization", max_length=80, on_submit=self.save,
                               text=vis.name if vis else "")
        self.input.focus()
        self.error = ""
        self.cancel = Button("Cancel", self.close, kind="secondary")
        self.ok = Button("Save", self.save, icon="save")
        self.widgets = [self.input, self.cancel, self.ok]

    def layout(self, size):
        super().layout(size)
        self.input.rect = pygame.Rect(self.rect.x + px(28), self.rect.y + px(84), self.rect.w - px(56), px(46))
        self.ok.rect = pygame.Rect(self.rect.right - px(28 + 120), self.rect.bottom - px(70), px(120), px(44))
        self.cancel.rect = pygame.Rect(self.ok.rect.x - px(12 + 110), self.rect.bottom - px(70), px(110), px(44))

    def save(self):
        try:
            self.app.click_save(self.input.text)
        except AudioVistaError as exc:
            self.error = str(exc)
            self.input.error = True
            return
        self.close()
        if self.after_save:
            self.after_save()

    def draw_content(self, surf):
        T.draw_text(surf, "NAME", (self.rect.x + px(30), self.rect.y + px(62)), 12, T.TEXT_DIM, bold=True)
        if self.error:
            T.draw_text(surf, self.error, (self.rect.x + px(30), self.rect.y + px(136)), 13, T.DANGER,
                        max_width=self.rect.w - px(60))


class ExportDialog(Modal):
    """Export Result: pick resolution / fps / length, then encode an MP4."""

    width, height = 520, 580
    title = "Export video"

    LENGTHS = [(None, "Full track"), (15.0, "15 s"), (30.0, "30 s"), (60.0, "60 s")]

    def __init__(self, app, studio):
        super().__init__(app)
        self.studio = studio
        vis = app.session.get_current_visualization()
        self.resolution = "1080p"
        self.fps = 30
        self.length = None
        self.output = default_export_dir() / (safe_filename(vis.name) + ".mp4")
        self.res_group = ChoiceGroup([(k, k if k not in ("Square", "Vertical") else
                                       f"{k} {EXPORT_RESOLUTIONS[k][0]}x{EXPORT_RESOLUTIONS[k][1]}")
                                      for k in EXPORT_RESOLUTIONS], self.resolution,
                                     lambda v: setattr(self, "resolution", v), columns=3, chip_height=34)
        self.fps_group = ChoiceGroup([(30, "30 fps"), (60, "60 fps")], self.fps,
                                     lambda v: setattr(self, "fps", v), columns=2, chip_height=34)
        self.len_group = ChoiceGroup(self.LENGTHS, None, lambda v: setattr(self, "length", v),
                                     columns=4, chip_height=34)
        self.change_btn = Button("Change…", self._choose_path, kind="secondary", size=13)
        self.cancel = Button("Cancel", self._cancel, kind="secondary")
        self.ok = Button("Export MP4", self._start, icon="download")
        self.open_btn = Button("Show in folder", lambda: reveal_in_folder(self.output), kind="secondary")
        self.open_btn.visible = False
        self.widgets = [self.res_group, self.fps_group, self.len_group, self.change_btn,
                        self.cancel, self.ok, self.open_btn]
        self.progress = 0.0
        self.state = "idle"  # idle | running | done | error
        self.message = ""
        self._cancel_event = threading.Event()

    def layout(self, size):
        super().layout(size)
        x, w = self.rect.x + px(28), self.rect.w - px(56)
        self.res_group.rect = pygame.Rect(x, self.rect.y + px(112), w, 0)
        self.res_group.rect.h = self.res_group.height
        self.fps_group.rect = pygame.Rect(x, self.res_group.rect.bottom + px(40), w, px(34))
        self.len_group.rect = pygame.Rect(x, self.fps_group.rect.bottom + px(40), w, px(34))
        self.path_y = self.len_group.rect.bottom + px(40)
        self.change_btn.rect = pygame.Rect(self.rect.right - px(28 + 96), self.path_y - px(4), px(96), px(34))
        self.ok.rect = pygame.Rect(self.rect.right - px(28 + 160), self.rect.bottom - px(70), px(160), px(44))
        self.cancel.rect = pygame.Rect(self.ok.rect.x - px(12 + 110), self.rect.bottom - px(70), px(110), px(44))
        self.open_btn.rect = self.ok.rect.copy()

    def _choose_path(self):
        def done(path):
            if path:
                self.output = Path(path).with_suffix(".mp4")
        self.app.run_task(lambda: ask_save_video(self.output), done)

    def _start(self):
        if self.state == "running":
            return
        self.state = "running"
        self.progress = 0.0
        self.message = ""
        self.closable = False
        self._cancel_event.clear()
        for wdg in (self.res_group, self.fps_group, self.len_group, self.change_btn, self.ok):
            wdg.enabled = False
        self.ok.visible = False

        def work():
            return self.app.click_export("mp4", self.resolution, self.output, self.fps, self.length,
                                         progress=lambda p: setattr(self, "progress", p),
                                         cancel_event=self._cancel_event)

        self.app.run_task(work, self._done, self._failed)

    def _done(self, export):
        self.state = "done"
        self.closable = True
        self.output = export.output_path
        self.message = f"Saved {export.file_name}"
        self.cancel.text = "Close"
        self.open_btn.visible = True
        self.app.toasts.show("Export finished - your video is ready.", "success")

    def _failed(self, exc):
        self.closable = True
        for wdg in (self.res_group, self.fps_group, self.len_group, self.change_btn, self.ok):
            wdg.enabled = True
        self.ok.visible = True
        if isinstance(exc, ExportCancelled):
            self.state = "idle"
            self.message = "Export cancelled."
        else:
            self.state = "error"
            self.message = str(exc)

    def _cancel(self):
        if self.state == "running":
            self._cancel_event.set()
        else:
            self.close()

    def handle_event(self, event):
        if self.state == "running" and event.type in (pygame.MOUSEBUTTONUP, pygame.MOUSEBUTTONDOWN):
            if self.cancel.handle_event(event):
                return True
            return True
        return super().handle_event(event)

    def draw_content(self, surf):
        x = self.rect.x + px(30)
        vis = self.app.session.get_current_visualization()
        T.draw_text(surf, vis.name if vis else "", (x, self.rect.y + px(56)), 13, T.TEXT_MUTED,
                    max_width=self.rect.w - px(60))
        for label, y in (("RESOLUTION", self.res_group.rect.y), ("FRAME RATE", self.fps_group.rect.y),
                         ("LENGTH", self.len_group.rect.y), ("SAVE TO", self.path_y)):
            T.draw_text(surf, label, (x, y - px(20)), 12, T.TEXT_DIM, bold=True)
        T.draw_text(surf, str(self.output), (x, self.path_y + px(4)), 13, T.TEXT,
                    max_width=self.rect.w - px(60 + 110))

        bar = pygame.Rect(x, self.path_y + px(48), self.rect.w - px(60), px(10))
        if self.state in ("running", "done"):
            T.rrect(surf, T.PANEL_3, bar, bar.h // 2)
            fill = bar.copy()
            fill.w = max(bar.h, int(bar.w * self.progress))
            T.rrect(surf, T.SUCCESS if self.state == "done" else T.ACCENT, fill, bar.h // 2)
            label = (f"Rendering… {int(self.progress * 100)}%" if self.state == "running"
                     else self.message)
            T.draw_text(surf, label, (x, bar.bottom + px(8)), 13, T.TEXT_DIM, max_width=bar.w)
        elif self.message:
            col = T.DANGER if self.state == "error" else T.TEXT_DIM
            for i, line in enumerate(T.wrap_text(self.message, 13, bar.w)[:2]):
                T.draw_text(surf, line, (x, bar.y + i * px(18)), 13, col)

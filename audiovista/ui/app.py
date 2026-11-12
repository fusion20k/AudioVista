"""AudioVistaUI - the window, main loop and the entry points used by the
sequence diagrams (submitRegistration, submitLogin, selectAudioFile,
requestGenerate, clickSave, clickExport, renderPreview)."""

from __future__ import annotations

import os
import queue
import sys
import threading
import traceback
from pathlib import Path
from typing import Any, Callable

import pygame

from .. import config
from ..audio.player import AudioPlayer
from ..controllers import (AccountController, ExportController, ResultController, UploadController,
                           VisualizerController)
from ..database import Database
from ..errors import AudioVistaError
from ..models import AudioFile, Session, Visualization, VisualSettings
from . import theme as T
from .theme import px
from .auth_screens import LoginScreen, RegisterScreen, Screen
from .studio import StudioScreen
from .widgets import Modal, Toasts


APP_USER_MODEL_ID = "AudioVista.Desktop"
ASSETS_DIR = Path(__file__).resolve().parents[2] / "assets"


def configure_windows_display() -> float:
    """Make the process DPI-aware and give it its own taskbar identity.

    Returns the display scale factor (1.0 = 100%, 2.0 = 200%). Without DPI
    awareness Windows renders the window at low resolution and stretches it,
    which looks pixelated on high-DPI screens. The AppUserModelID makes the
    taskbar show AudioVista's own icon instead of the Python icon.
    """
    override = os.environ.get("AUDIOVISTA_UI_SCALE")
    scale = 1.0
    if sys.platform == "win32":
        import ctypes

        try:
            ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_USER_MODEL_ID)
        except (AttributeError, OSError):
            pass
        if os.environ.get("SDL_VIDEODRIVER") != "dummy":
            try:
                ctypes.windll.shcore.SetProcessDpiAwareness(2)  # per-monitor aware
            except (AttributeError, OSError):
                try:
                    ctypes.windll.user32.SetProcessDPIAware()
                except (AttributeError, OSError):
                    pass
            try:
                scale = ctypes.windll.user32.GetDpiForSystem() / 96.0
            except (AttributeError, OSError):
                scale = 1.0
    if override:
        try:
            scale = float(override)
        except ValueError:
            pass
    return max(1.0, scale)


class AudioVistaUI:
    FPS = 60
    MIN_SIZE = (1000, 720)  # logical px

    def __init__(self, db_path: str | Path | None = None, size=(1360, 840)):
        os.environ.setdefault("SDL_HINT_IME_SHOW_UI", "1")
        T.set_scale(configure_windows_display())
        pygame.init()
        pygame.display.set_caption("AudioVista")
        pygame.display.set_icon(T.render_logo(128))
        info = pygame.display.Info()
        size = (px(size[0]), px(size[1]))
        if info.current_w > 0 and info.current_h > 0 and os.environ.get("SDL_VIDEODRIVER") != "dummy":
            size = (min(size[0], info.current_w - px(40)), min(size[1], info.current_h - px(90)))
        self.screen = pygame.display.set_mode(size, pygame.RESIZABLE)
        self._apply_taskbar_identity()
        pygame.key.set_repeat(400, 35)

        self.db = Database(db_path or config.database_path())
        self.session = Session()
        self.account_controller = AccountController(self.db)
        self._bind_controllers()
        self.player = AudioPlayer()
        self.toasts = Toasts()
        self.modals: list[Modal] = []
        self.clock = pygame.time.Clock()
        self.running = True
        self._results: queue.Queue = queue.Queue()
        self.current: Screen = LoginScreen(self)
        self.current.on_enter()

    @staticmethod
    def _apply_taskbar_identity():
        """Show the AudioVista logo (not Python's) on the taskbar and Alt+Tab."""
        if sys.platform != "win32" or os.environ.get("SDL_VIDEODRIVER") == "dummy":
            return
        from . import winshell

        try:
            hwnd = pygame.display.get_wm_info().get("window")
        except pygame.error:
            return
        ico = ASSETS_DIR / "audiovista.ico"
        if not ico.is_file():
            try:
                winshell.write_ico(ico, [T.render_logo(s) for s in (16, 24, 32, 48, 64, 128, 256)])
            except OSError:
                pass
        launcher = Path(sys.executable).with_name("pythonw.exe")
        exe = launcher if launcher.exists() else Path(sys.executable)
        winshell.set_window_icon(hwnd, ico)
        winshell.set_window_identity(hwnd, APP_USER_MODEL_ID, "AudioVista", f"{ico},0",
                                     f'"{exe}" "{ASSETS_DIR.parent / "main.py"}"')

    def _bind_controllers(self):
        self.upload_controller = UploadController(self.db, self.session, config.library_dir())
        self.visualizer_controller = VisualizerController(self.session)
        self.result_controller = ResultController(self.db, self.session)
        self.export_controller = ExportController(self.db, self.session)

    # ---------------------------------------------------------- navigation
    def set_screen(self, screen: Screen):
        self.current.on_exit()
        self.modals.clear()
        self.current = screen
        screen.on_enter()

    def show_login(self, email: str = "", info: str = ""):
        self.set_screen(LoginScreen(self, email, info))

    def show_register(self):
        self.set_screen(RegisterScreen(self))

    def open_modal(self, modal: Modal):
        modal.layout(self.screen.get_size())
        self.modals.append(modal)

    def close_modal(self, modal: Modal):
        if modal in self.modals:
            self.modals.remove(modal)

    # --------------------------------------------- sequence-diagram methods
    def submit_registration(self, username: str, email: str, password: str, confirm: str | None = None):
        self.account_controller.create_account(username, email, password, confirm)
        self.show_login(email.strip().lower(), "Account created - please log in.")

    def submit_login(self, email: str, password: str):
        session = self.account_controller.login(email, password)
        self.session = session
        self._bind_controllers()
        self.set_screen(StudioScreen(self))
        self.toasts.show(f"Welcome, {session.account.username}!", "success")

    def logout(self):
        self.player.unload()
        self.session.clear()
        self.show_login(info="You have been logged out.")

    def select_audio_file(self, file_path: str | Path) -> AudioFile:
        """Upload Audio (runs on a worker thread - decoding can take a moment)."""
        return self.upload_controller.upload_audio(file_path)

    def request_generate(self, settings: VisualSettings | None = None, style: str | None = None) -> Visualization:
        visualization = self.visualizer_controller.generate_visuals(style, settings)
        self.render_preview(visualization)
        return visualization

    def render_preview(self, visualization: Visualization):
        """Preview starts automatically: the studio renders the current
        visualization every frame in sync with the player."""
        return visualization

    def click_save(self, title: str) -> int:
        result_id = self.result_controller.save_result(title)
        self.toasts.show("Saved to My Library", "success")
        return result_id

    def click_export(self, file_format: str, resolution: str, output_path, fps: int = 30,
                     duration: float | None = None, progress=None, cancel_event=None):
        return self.export_controller.export_result(file_format, resolution, output_path, fps, duration,
                                                    progress, cancel_event)

    # --------------------------------------------------------- background
    def run_task(self, work: Callable[[], Any], on_done: Callable[[Any], Any] | None = None,
                 on_error: Callable[[Exception], Any] | None = None):
        """Run *work* on a thread; callbacks run later on the UI thread."""
        def runner():
            try:
                result = work()
                self._results.put((on_done, result, None))
            except Exception as exc:  # noqa: BLE001 - reported to the user
                if not isinstance(exc, AudioVistaError):
                    traceback.print_exc()
                self._results.put((on_error, None, exc))
        threading.Thread(target=runner, daemon=True).start()

    def _drain_results(self):
        while True:
            try:
                callback, result, exc = self._results.get_nowait()
            except queue.Empty:
                return
            try:
                if exc is not None:
                    if callback:
                        callback(exc)
                    else:
                        self.toasts.show(str(exc) if isinstance(exc, AudioVistaError)
                                         else f"Unexpected error: {exc}", "error", 5)
                elif callback:
                    callback(result)
            except AudioVistaError as err:
                self.toasts.show(str(err), "error", 5)

    # --------------------------------------------------------------- loop
    def handle_event(self, event):
        if event.type == pygame.QUIT:
            self.running = False
            return
        if event.type == pygame.VIDEORESIZE:
            w, h = max(px(self.MIN_SIZE[0]), event.w), max(px(self.MIN_SIZE[1]), event.h)
            if (w, h) != (event.w, event.h):
                self.screen = pygame.display.set_mode((w, h), pygame.RESIZABLE)
            self.current.layout(self.screen.get_size())
            for m in self.modals:
                m.layout(self.screen.get_size())
            return
        if event.type == pygame.WINDOWSIZECHANGED:
            self.current.layout(self.screen.get_size())
            for m in self.modals:
                m.layout(self.screen.get_size())
            return
        try:
            if self.modals:
                self.modals[-1].handle_event(event)
            else:
                self.current.handle_event(event)
        except AudioVistaError as exc:
            self.toasts.show(str(exc), "error", 5)

    def step(self, dt: float):
        for event in pygame.event.get():
            self.handle_event(event)
        self._drain_results()
        self.current.update(dt)
        for m in self.modals:
            m.update(dt)
        self.current.draw(self.screen)
        for m in self.modals:
            m.draw(self.screen)
        self.toasts.draw(self.screen)
        pygame.display.flip()

    def run(self):
        while self.running:
            dt = self.clock.tick(self.FPS) / 1000.0
            self.step(dt)
        self.player.unload()
        pygame.quit()


def main():
    AudioVistaUI().run()

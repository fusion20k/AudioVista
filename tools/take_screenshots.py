"""Drive the real AudioVista UI without a visible window and save screenshots
of every screen into docs/screenshots/. Useful for reports and as a smoke test.

    python tools/take_screenshots.py
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")
os.environ["AUDIOVISTA_DATA_DIR"] = tempfile.mkdtemp(prefix="audiovista_shots_")
os.environ.setdefault("AUDIOVISTA_UI_SCALE", "1.5")  # crisp, high-DPI screenshots

import pygame  # noqa: E402

import audiovista.ui.dialogs as dialogs  # noqa: E402
from audiovista.ui.app import AudioVistaUI  # noqa: E402

# Show a generic export folder instead of the real user profile path.
dialogs.default_export_dir = lambda: Path("C:/Users/You/Videos/AudioVista")
from audiovista.ui.auth_screens import LoginScreen, RegisterScreen  # noqa: E402
from audiovista.ui.dialogs import ExportDialog, SaveDialog  # noqa: E402

OUT = ROOT / "docs" / "screenshots"


def pump(app: AudioVistaUI, seconds: float = 0.3, fps: int = 30):
    end = time.time() + seconds
    while time.time() < end:
        app.step(1 / fps)
        time.sleep(1 / fps)


def wait_until(app, cond, timeout=20):
    end = time.time() + timeout
    while time.time() < end and not cond():
        pump(app, 0.1)
    if not cond():
        raise TimeoutError("condition not met")


def type_into(app, widget, text):
    widget.focus()
    widget.set_text(text)


def shot(app, name):
    OUT.mkdir(parents=True, exist_ok=True)
    pygame.image.save(app.screen, str(OUT / f"{name}.png"))
    print("saved", name)


def main():
    app = AudioVistaUI(size=(1360, 840))

    # 1. Register screen with a validation error
    app.show_register()
    reg: RegisterScreen = app.current
    type_into(app, reg.username, "demo_user")
    type_into(app, reg.email, "demo@example.com")
    type_into(app, reg.password, "short")
    type_into(app, reg.confirm, "short")
    reg.submit()
    pump(app)
    shot(app, "01_register_validation_error")

    reg.password.set_text("visualize123")
    reg.confirm.set_text("visualize123")
    reg.submit()
    pump(app)
    login: LoginScreen = app.current
    shot(app, "02_login_after_register")

    # 2. Wrong password, then correct login
    login.password.set_text("wrongpass1")
    login.submit()
    pump(app)
    shot(app, "03_login_wrong_password")
    login.password.set_text("visualize123")
    login.submit()
    pump(app, 0.5)
    shot(app, "04_studio_empty")

    # 3. Load the demo track and visualise it in each style
    studio = app.current
    studio.add_demo_track()
    wait_until(app, lambda: app.player.loaded and studio.loading is None)
    app.player.seek(14.0)
    pump(app, 1.0)
    shot(app, "05_studio_bars")
    for i, (style, bg, c1, c2, extra) in enumerate([("waveform", "twilight", "#ffb35c", "#7fc8ff", {}),
                                                     ("circular", "night", "#ffb35c", "#ff7eb6", {}),
                                                     ("circular", "sunset", "#ff8a7a", "#ffb35c",
                                                      {"particles": True})]):
        studio.change_setting(wave_type=style, background=bg, color=c1, secondary_color=c2, **extra)
        studio._apply_settings_to_widgets(studio.settings)
        pump(app, 1.0)
        style += "_particles" if extra else ""
        shot(app, f"{6 + i:02d}_studio_{style}")

    # 4. Queue tab, save dialog, saved tab, export dialog
    studio.enqueue(studio.library_list.items[0])
    studio.set_tab("queue")
    pump(app, 0.4)
    shot(app, "10_queue")
    app.open_modal(SaveDialog(app, studio, after_save=studio.refresh_saved))
    app.modals[-1].input.set_text("Sunset Radial")
    pump(app, 0.3)
    shot(app, "11_save_dialog")
    app.modals[-1].save()
    studio.set_tab("saved")
    pump(app, 0.4)
    shot(app, "12_saved_library")
    app.open_modal(ExportDialog(app, studio))
    pump(app, 0.3)
    shot(app, "13_export_dialog")
    app.modals.clear()

    # 5. Focus (full view) mode
    studio.toggle_focus()
    pump(app, 0.8)
    shot(app, "14_full_view")
    studio.toggle_focus()

    app.player.unload()
    pygame.quit()
    print("Screenshots written to", OUT)


if __name__ == "__main__":
    main()

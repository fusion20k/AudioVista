"""The main workspace: library, visualizer preview, settings and transport."""

from __future__ import annotations

import math
import time
from pathlib import Path

import pygame

from ..config import SUPPORTED_FORMATS, data_dir
from ..errors import AudioVistaError, NotSavedError
from ..models import (BACKGROUND_LABELS, BACKGROUNDS, COLOR_PRESETS, WAVE_TYPE_LABELS, WAVE_TYPES,
                      AudioFile, VisualSettings)
from ..visuals.engine import VisualizerEngine
from . import theme as T
from .auth_screens import Screen
from .dialogs import ConfirmDialog, ExportDialog, SaveDialog, ask_open_audio
from .widgets import Button, ChoiceGroup, ColorSwatches, ListView, SeekBar, Slider, Toggle

TOP_H, LEFT_W, RIGHT_W, BOTTOM_H = 60, 292, 304, 96


class StudioScreen(Screen):
    def __init__(self, app):
        super().__init__(app)
        self.settings = VisualSettings()
        self.engine: VisualizerEngine | None = None
        self.loading: str | None = None
        self.focus_mode = False
        self.tab = "library"
        self.history: list[AudioFile] = []
        self._audio_cache: dict[int, AudioFile] = {}
        self._last_render = time.perf_counter()
        self._canvas_surface: pygame.Surface | None = None
        self._mouse_moved_at = time.time()

        # --- top bar
        self.logout_btn = Button("Log out", app.logout, kind="ghost", icon="logout", size=13)

        # --- sidebar
        self.tab_buttons = {
            "library": Button("Library", lambda: self.set_tab("library"), kind="ghost", size=13),
            "queue": Button("Queue", lambda: self.set_tab("queue"), kind="ghost", size=13),
            "saved": Button("Saved", lambda: self.set_tab("saved"), kind="ghost", size=13),
        }
        self.upload_btn = Button("Upload audio", self.choose_file, icon="upload")
        self.demo_btn = Button("Add demo track", self.add_demo_track, kind="secondary", size=13)
        self.library_list = ListView(
            on_select=self.play_library_item, row_actions=[("queue", self.enqueue, "Add to queue"),
                                                           ("trash", self.confirm_remove_audio, "Remove")],
            title=lambda a: a.display_name,
            subtitle=lambda a: f"{T.format_time(a.duration)}  ·  {a.file_type.upper()}",
            empty_text="Your library is empty. Upload an MP3, WAV or FLAC file, or drag one onto the window.")
        self.library_list.key = lambda a: a.audio_id
        self.queue_list = ListView(
            on_select=self.play_from_queue, row_actions=[("close", self.dequeue, "Remove")],
            title=lambda a: a.display_name, subtitle=lambda a: T.format_time(a.duration),
            empty_text="The queue is empty. Hover a track in your library and press the queue icon.")
        self.saved_list = ListView(
            on_select=self.open_saved, row_actions=[("trash", self.confirm_delete_saved, "Delete")],
            title=lambda r: r["name"],
            subtitle=lambda r: f"{r['audio_title']}  ·  {self._style_of(r)}",
            empty_text="No saved visualizations yet. Press Save to keep one here.")
        self.saved_list.key = lambda r: r["visualization_id"]

        # --- settings panel
        self.style_group = ChoiceGroup([(s, WAVE_TYPE_LABELS[s]) for s in WAVE_TYPES], self.settings.wave_type,
                                       lambda v: self.change_setting(wave_type=v), columns=2)
        self.color_swatches = ColorSwatches(list(COLOR_PRESETS), self.settings.color,
                                            lambda v: self.change_setting(color=v))
        self.color2_swatches = ColorSwatches(list(COLOR_PRESETS), self.settings.secondary_color,
                                             lambda v: self.change_setting(secondary_color=v))
        self.bg_group = ChoiceGroup([(b, BACKGROUND_LABELS[b]) for b in BACKGROUNDS], self.settings.background,
                                    lambda v: self.change_setting(background=v), columns=3, chip_height=32)
        self.sens_slider = Slider(0.5, 2.0, self.settings.sensitivity,
                                  lambda v: self.change_setting(sensitivity=round(v, 2)))
        self.smooth_slider = Slider(0.0, 0.95, self.settings.smoothing,
                                    lambda v: self.change_setting(smoothing=round(v, 2)))
        self.glow_toggle = Toggle("Glow", self.settings.glow, lambda v: self.change_setting(glow=v))
        self.title_toggle = Toggle("Show title", self.settings.show_title,
                                   lambda v: self.change_setting(show_title=v))
        self.save_btn = Button("Save", self.click_save, kind="secondary", icon="save")
        self.export_btn = Button("Export", self.click_export, icon="download")

        # --- transport
        self.play_btn = Button("", self.toggle_play, icon="play", radius=26)
        self.prev_btn = Button("", self.previous_track, kind="ghost", icon="prev")
        self.next_btn = Button("", self.next_track, kind="ghost", icon="next")
        self.seek = SeekBar(self.seek_to)
        self.volume = Slider(0.0, 1.0, app.player.volume, app.player.set_volume)
        self.focus_btn = Button("", self.toggle_focus, kind="ghost", icon="fullscreen")

        # --- empty-state buttons on the canvas
        self.empty_upload = Button("Upload audio", self.choose_file, icon="upload")
        self.empty_demo = Button("Try the demo track", self.add_demo_track, kind="secondary")

        self.settings_widgets = [self.style_group, self.color_swatches, self.color2_swatches, self.bg_group,
                                 self.sens_slider, self.smooth_slider, self.glow_toggle, self.title_toggle,
                                 self.save_btn, self.export_btn]
        self.transport_widgets = [self.play_btn, self.prev_btn, self.next_btn, self.seek, self.volume,
                                  self.focus_btn]

    # ------------------------------------------------------------- lifecycle
    def on_enter(self):
        self.layout(self.app.screen.get_size())
        self.refresh_lists()

    def on_exit(self):
        self.app.player.unload()

    @property
    def lists(self):
        return {"library": self.library_list, "queue": self.queue_list, "saved": self.saved_list}

    def set_tab(self, tab):
        self.tab = tab
        if tab == "saved":
            self.refresh_saved()

    def refresh_lists(self):
        try:
            self.library_list.set_items(self.app.upload_controller.list_library())
        except AudioVistaError:
            pass
        self.queue_list.set_items(self.app.session.queue.items)
        self.refresh_saved()
        self._sync_selection()

    def refresh_saved(self):
        try:
            self.saved_list.set_items(self.app.result_controller.list_results())
        except AudioVistaError:
            pass

    def _sync_selection(self):
        audio = self.app.session.get_current_audio()
        vis = self.app.session.get_current_visualization()
        self.library_list.selected_key = audio.audio_id if audio else None
        self.saved_list.selected_key = vis.visualization_id if vis and vis.visualization_id else None

    @staticmethod
    def _style_of(rec):
        import json
        try:
            return WAVE_TYPE_LABELS.get(json.loads(rec["settings_json"]).get("wave_type"), "")
        except (ValueError, TypeError):
            return ""

    # ---------------------------------------------------------------- layout
    def layout(self, size):
        w, h = size
        self.size = size
        if self.focus_mode:
            self.canvas = pygame.Rect(0, 0, w, h)
        else:
            self.canvas = pygame.Rect(LEFT_W, TOP_H, max(50, w - LEFT_W - RIGHT_W), max(50, h - TOP_H - BOTTOM_H))
        self.topbar = pygame.Rect(0, 0, w, TOP_H)
        self.sidebar = pygame.Rect(0, TOP_H, LEFT_W, h - TOP_H - BOTTOM_H)
        self.panel = pygame.Rect(w - RIGHT_W, TOP_H, RIGHT_W, h - TOP_H - BOTTOM_H)
        self.transport = pygame.Rect(0, h - BOTTOM_H, w, BOTTOM_H)

        self.logout_btn.rect = pygame.Rect(w - 126, 12, 112, 36)

        # sidebar
        x, sw = 16, LEFT_W - 32
        self.upload_btn.rect = pygame.Rect(x, TOP_H + 16, sw, 44)
        tab_w = sw // 3
        for i, b in enumerate(self.tab_buttons.values()):
            b.rect = pygame.Rect(x + i * tab_w, TOP_H + 74, tab_w, 34)
        list_top = TOP_H + 120
        list_bottom = self.sidebar.bottom - 64
        for lv in self.lists.values():
            lv.rect = pygame.Rect(x - 6, list_top, sw + 12, max(40, list_bottom - list_top))
        self.demo_btn.rect = pygame.Rect(x, self.sidebar.bottom - 52, sw, 38)

        # settings panel
        px, pw = self.panel.x + 20, RIGHT_W - 40
        y = TOP_H + 48
        self.style_group.rect = pygame.Rect(px, y, pw, 0)
        self.style_group.rect.h = self.style_group.height
        y = self.style_group.rect.bottom + 34
        self.color_swatches.rect = pygame.Rect(px, y, pw, 0)
        self.color_swatches.rect.h = self.color_swatches.height
        y = self.color_swatches.rect.bottom + 34
        self.color2_swatches.rect = pygame.Rect(px, y, pw, 0)
        self.color2_swatches.rect.h = self.color2_swatches.height
        y = self.color2_swatches.rect.bottom + 34
        self.bg_group.rect = pygame.Rect(px, y, pw, 0)
        self.bg_group.rect.h = self.bg_group.height
        y = self.bg_group.rect.bottom + 38
        self.sens_slider.rect = pygame.Rect(px + 4, y, pw - 8, 16)
        y += 44
        self.smooth_slider.rect = pygame.Rect(px + 4, y, pw - 8, 16)
        y += 30
        half = (pw - 24) // 2
        self.glow_toggle.rect = pygame.Rect(px, y, half, 30)
        self.title_toggle.rect = pygame.Rect(px + half + 24, y, half, 30)
        bh = 44
        self.save_btn.rect = pygame.Rect(px, self.panel.bottom - bh - 16, (pw - 10) // 2, bh)
        self.export_btn.rect = pygame.Rect(self.save_btn.rect.right + 10, self.save_btn.rect.y,
                                           (pw - 10) // 2, bh)
        self.settings_clip = pygame.Rect(self.panel.x, self.panel.y, self.panel.w,
                                         self.save_btn.rect.y - 8 - self.panel.y)

        # transport
        ty = self.transport.y
        self.play_btn.rect = pygame.Rect(0, 0, 52, 52)
        self.play_btn.rect.center = (LEFT_W // 2, ty + BOTTOM_H // 2)
        self.prev_btn.rect = pygame.Rect(0, 0, 40, 40)
        self.prev_btn.rect.center = (self.play_btn.rect.centerx - 62, self.play_btn.rect.centery)
        self.next_btn.rect = pygame.Rect(0, 0, 40, 40)
        self.next_btn.rect.center = (self.play_btn.rect.centerx + 62, self.play_btn.rect.centery)
        seek_x = LEFT_W + 56
        seek_w = max(100, w - RIGHT_W - 56 - seek_x)
        self.seek.rect = pygame.Rect(seek_x, ty + 40, seek_w, 34)
        self.volume.rect = pygame.Rect(w - RIGHT_W + 56, ty + BOTTOM_H // 2 - 8, RIGHT_W - 130, 16)
        self.focus_btn.rect = pygame.Rect(w - 56, ty + BOTTOM_H // 2 - 18, 36, 36)
        self._update_overview()

        self.empty_upload.rect = pygame.Rect(0, 0, 220, 48)
        self.empty_upload.rect.center = (self.canvas.centerx, self.canvas.centery + 40)
        self.empty_demo.rect = pygame.Rect(0, 0, 220, 42)
        self.empty_demo.rect.center = (self.canvas.centerx, self.canvas.centery + 100)

    def _update_overview(self):
        audio = self.app.session.get_current_audio()
        if audio is not None and audio.is_loaded:
            self.seek.overview = audio.get_analyzer().overview(max(20, self.seek.rect.w // 4))
        else:
            self.seek.overview = None

    def toggle_focus(self):
        self.focus_mode = not self.focus_mode
        self.layout(self.app.screen.get_size())

    # ---------------------------------------------------------- audio loading
    def choose_file(self):
        self.app.run_task(ask_open_audio, lambda path: path and self.upload(path))

    def upload(self, path):
        self.loading = f"Analysing {Path(path).name}…"
        self.app.run_task(lambda: self.app.select_audio_file(path),
                          lambda audio: self._audio_ready(audio, toast=f"Added '{audio.title}' to your library"),
                          self._load_failed)

    def add_demo_track(self):
        from ..audio.demo import generate_demo_track

        path = data_dir() / "AudioVista Demo.wav"

        def work():
            if not path.exists():
                generate_demo_track(path)
            return self.app.select_audio_file(path)

        self.loading = "Creating demo track…"
        self.app.run_task(work, lambda a: self._audio_ready(a, toast="Demo track added to your library"),
                          self._load_failed)

    def _load_track(self, audio: AudioFile, autoplay=True, remember=True):
        cached = self._audio_cache.get(audio.audio_id)
        if cached is not None and cached.is_loaded:
            self._audio_ready(cached, autoplay=autoplay, remember=remember)
            return
        self.loading = f"Loading {audio.title}…"
        self.app.run_task(lambda: self.app.upload_controller.load_from_library(audio.audio_id),
                          lambda a: self._audio_ready(a, autoplay=autoplay, remember=remember),
                          self._load_failed)

    def _audio_ready(self, audio: AudioFile, autoplay=True, remember=True, visualization=None, toast=None):
        self.loading = None
        current = self.app.session.get_current_audio()
        if remember and current is not None and current is not audio:
            self.history.append(current)
            self.history = self.history[-30:]
        if audio.audio_id is not None:
            self._audio_cache[audio.audio_id] = audio
            while len(self._audio_cache) > 4:
                self._audio_cache.pop(next(iter(self._audio_cache)))
        self.app.session.set_current_audio(audio)
        self.app.player.load(audio.samples, audio.sample_rate)

        if visualization is not None:
            self.app.session.set_current_visualization(visualization)
            self._apply_settings_to_widgets(visualization.settings)
        else:
            visualization = self.app.request_generate(self.settings)
        self.engine = VisualizerEngine(audio.get_analyzer(), visualization.settings, title=audio.display_name)
        self._update_overview()
        if autoplay:
            self.app.player.play()
        self.refresh_lists()
        if toast:
            self.app.toasts.show(toast, "success")

    def _load_failed(self, exc):
        self.loading = None
        self.app.toasts.show(str(exc), "error", 5)

    # --------------------------------------------------------- library actions
    def play_library_item(self, audio):
        self._load_track(audio)

    def enqueue(self, audio):
        self.app.session.queue.add(audio)
        self.queue_list.set_items(self.app.session.queue.items)
        self.app.toasts.show(f"Queued '{audio.title}'  ({len(self.app.session.queue)} up next)")
        if self.app.session.get_current_audio() is None:
            self.next_track()

    def dequeue(self, audio):
        items = self.app.session.queue.items
        if audio in items:
            self.app.session.queue.remove(items.index(audio))
        self.queue_list.set_items(self.app.session.queue.items)

    def play_from_queue(self, audio):
        self.dequeue(audio)
        self._load_track(audio)

    def confirm_remove_audio(self, audio):
        def remove():
            self.app.upload_controller.remove_from_library(audio.audio_id)
            self._audio_cache.pop(audio.audio_id, None)
            current = self.app.session.get_current_audio()
            if current is not None and current.audio_id == audio.audio_id:
                self._clear_current()
            self.refresh_lists()
            self.app.toasts.show(f"Removed '{audio.title}'")
        self.app.open_modal(ConfirmDialog(
            self.app, "Remove track?",
            f"'{audio.title}' and any visualizations saved with it will be deleted from your library.",
            "Remove", remove))

    def _clear_current(self):
        self.app.player.unload()
        self.app.session.set_current_audio(None)
        self.app.session.set_current_visualization(None)
        self.engine = None
        self._update_overview()

    def open_saved(self, rec):
        self.loading = f"Opening {rec['name']}…"

        def done(vis):
            self._audio_ready(vis.audio_file, visualization=vis)
        self.app.run_task(lambda: self.app.result_controller.open_result(rec["visualization_id"]), done,
                          self._load_failed)

    def confirm_delete_saved(self, rec):
        def delete():
            self.app.result_controller.delete_result(rec["visualization_id"])
            vis = self.app.session.get_current_visualization()
            if vis is not None and vis.visualization_id == rec["visualization_id"]:
                vis.visualization_id = None
                vis.save_status = False
            self.refresh_saved()
            self.app.toasts.show(f"Deleted '{rec['name']}'")
        self.app.open_modal(ConfirmDialog(self.app, "Delete visualization?",
                                          f"'{rec['name']}' will be permanently deleted.", "Delete", delete))

    # -------------------------------------------------------------- settings
    def change_setting(self, **changes):
        self.settings = self.settings.copy(**changes)
        vis = self.app.session.get_current_visualization()
        if vis is not None:
            vis.update_settings(**changes)
            if self.engine:
                self.engine.set_settings(vis.settings)

    def _apply_settings_to_widgets(self, s: VisualSettings):
        self.settings = s.copy()
        self.style_group.value = s.wave_type
        self.color_swatches.value = s.color
        self.color2_swatches.value = s.secondary_color
        self.bg_group.value = s.background
        self.sens_slider.value = s.sensitivity
        self.smooth_slider.value = s.smoothing
        self.glow_toggle.value = s.glow
        self.title_toggle.value = s.show_title

    def click_save(self):
        if self.app.session.get_current_visualization() is None:
            self.app.toasts.show("Load a track before saving.", "warning")
            return
        self.app.open_modal(SaveDialog(self.app, self, after_save=self.refresh_saved))

    def click_export(self):
        vis = self.app.session.get_current_visualization()
        if vis is None:
            self.app.toasts.show("Load a track before exporting.", "warning")
            return
        if not vis.save_status:
            # Use case "Export Result", extension 3a: must save first.
            self.app.toasts.show(str(NotSavedError("Please save your visualization before exporting it.")),
                                 "warning")

            def then_export():
                self.refresh_saved()
                self.app.open_modal(ExportDialog(self.app, self))
            self.app.open_modal(SaveDialog(self.app, self, after_save=then_export))
            return
        self.app.open_modal(ExportDialog(self.app, self))

    # ------------------------------------------------------------- transport
    def toggle_play(self):
        if self.app.player.loaded:
            self.app.player.toggle()
        elif len(self.app.session.queue):
            self.next_track()

    def seek_to(self, t):
        self.app.player.seek(t)
        if self.engine:
            self.engine.reset()

    def next_track(self):
        nxt = self.app.session.queue.pop_next()
        self.queue_list.set_items(self.app.session.queue.items)
        if nxt is None:
            self.app.toasts.show("Queue is empty - add tracks from your library with the queue icon.")
            return
        self._load_track(nxt)

    def previous_track(self):
        if self.app.player.position > 3 or not self.history:
            self.seek_to(0)
            return
        prev = self.history.pop()
        self._load_track(prev, remember=False)

    # ---------------------------------------------------------------- events
    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEMOTION:
            self._mouse_moved_at = time.time()
        if event.type == pygame.DROPFILE:
            if Path(event.file).suffix.lower() in SUPPORTED_FORMATS:
                self.upload(event.file)
            else:
                self.app.toasts.show("Only MP3, WAV and FLAC files can be added.", "error")
            return True
        if event.type == pygame.KEYDOWN:
            ctrl = event.mod & pygame.KMOD_CTRL
            if event.key == pygame.K_SPACE:
                self.toggle_play()
                return True
            if event.key == pygame.K_f or (event.key == pygame.K_ESCAPE and self.focus_mode):
                self.toggle_focus()
                return True
            if event.key == pygame.K_RIGHT:
                self.seek_to(self.app.player.position + 5)
                return True
            if event.key == pygame.K_LEFT:
                self.seek_to(self.app.player.position - 5)
                return True
            if event.key == pygame.K_n:
                self.next_track()
                return True
            if ctrl and event.key == pygame.K_s:
                self.click_save()
                return True
            if ctrl and event.key == pygame.K_e:
                self.click_export()
                return True
            if ctrl and event.key == pygame.K_o:
                self.choose_file()
                return True
            if pygame.K_1 <= event.key <= pygame.K_4:
                style = WAVE_TYPES[event.key - pygame.K_1]
                self.style_group.value = style
                self.change_setting(wave_type=style)
                return True

        if self.focus_mode:
            if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                self.toggle_play()
            return True

        widgets = [self.logout_btn, self.upload_btn, *self.tab_buttons.values(), self.lists[self.tab],
                   self.demo_btn, *self.settings_widgets, *self.transport_widgets]
        if self.app.player.loaded is False and self.loading is None:
            widgets += [self.empty_upload, self.empty_demo]
        for w in widgets:
            if w.visible and w.handle_event(event):
                return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.canvas.collidepoint(event.pos):
            self.toggle_play()
            return True
        return False

    # ---------------------------------------------------------------- update
    def update(self, dt):
        player = self.app.player
        if player.finished:
            if len(self.app.session.queue):
                self.next_track()
            else:
                player.pause()
                player.seek(0)
        self.play_btn.icon = "pause" if player.is_playing else "play"
        self.seek.duration = player.duration
        self.seek.position = player.position
        self.tab_buttons[self.tab].active = True
        for k, b in self.tab_buttons.items():
            b.active = k == self.tab
        self.export_btn.enabled = self.save_btn.enabled = self.app.session.get_current_visualization() is not None

    # ------------------------------------------------------------------ draw
    def draw(self, surf):
        surf.fill(T.BG)
        self._draw_canvas(surf)
        if self.focus_mode:
            if time.time() - self._mouse_moved_at < 2.0:
                T.draw_text(surf, "Press F or Esc to exit full view  ·  Space to play/pause",
                            (surf.get_width() // 2, 16), 13, T.TEXT_DIM, anchor="midtop")
            return
        self._draw_topbar(surf)
        self._draw_sidebar(surf)
        self._draw_panel(surf)
        self._draw_transport(surf)

    def _draw_canvas(self, surf):
        c = self.canvas
        now = time.perf_counter()
        dt = now - self._last_render
        self._last_render = now
        if self.engine is not None and self.app.player.loaded:
            if self._canvas_surface is None or self._canvas_surface.get_size() != c.size:
                self._canvas_surface = pygame.Surface(c.size)
            # Freeze motion while paused so the frame doesn't decay to nothing.
            self.engine.render(self._canvas_surface, self.app.player.position,
                               dt if self.app.player.is_playing else 0.0001)
            surf.blit(self._canvas_surface, c)
        else:
            pygame.draw.rect(surf, (8, 9, 16), c)
            if self.loading is None:
                logo = pygame.Rect(0, 0, 70, 56)
                logo.center = (c.centerx, c.centery - 90)
                T.draw_icon(surf, "logo", logo, T.ACCENT)
                T.draw_text(surf, "Drop an audio file here", (c.centerx, c.centery - 32), 22, T.TEXT,
                            bold=True, anchor="center")
                T.draw_text(surf, "MP3, WAV or FLAC  ·  or pick a track from your library",
                            (c.centerx, c.centery - 4), 14, T.TEXT_DIM, anchor="center")
                self.empty_upload.draw(surf)
                self.empty_demo.draw(surf)
        if self.loading:
            shade = pygame.Surface(c.size, pygame.SRCALPHA)
            shade.fill((0, 0, 0, 150))
            surf.blit(shade, c)
            t = time.time()
            for i in range(8):
                a = t * 4 + i * 0.785
                alpha = (i + 1) / 8
                col = tuple(int(v * alpha) for v in T.ACCENT_HOVER)
                pygame.draw.circle(surf, col, (int(c.centerx + math.cos(a) * 22),
                                               int(c.centery - 20 + math.sin(a) * 22)), 4)
            T.draw_text(surf, self.loading, (c.centerx, c.centery + 24), 15, T.TEXT, anchor="midtop")

    def _draw_topbar(self, surf):
        pygame.draw.rect(surf, T.PANEL, self.topbar)
        pygame.draw.line(surf, T.BORDER, (0, TOP_H - 1), (self.topbar.w, TOP_H - 1))
        T.draw_icon(surf, "logo", pygame.Rect(18, 17, 28, 24), T.ACCENT)
        T.draw_text(surf, "AudioVista", (56, TOP_H // 2), 19, T.TEXT, bold=True, anchor="midleft")
        audio = self.app.session.get_current_audio()
        vis = self.app.session.get_current_visualization()
        if audio is not None:
            name = vis.name if vis and vis.save_status else audio.display_name
            T.draw_text(surf, name, (self.topbar.centerx, TOP_H // 2 - 8), 15, T.TEXT, bold=True,
                        anchor="center", max_width=self.topbar.w - 2 * LEFT_W - 40)
            bpm = audio.get_analyzer().estimate_tempo() if audio.is_loaded else None
            status = "Saved" if vis and vis.save_status else "Unsaved changes" if vis and vis.visualization_id else "Not saved"
            info = f"{T.format_time(audio.duration)}  ·  {status}" + (f"  ·  ~{bpm:.0f} BPM" if bpm else "")
            T.draw_text(surf, info, (self.topbar.centerx, TOP_H // 2 + 12), 12,
                        T.SUCCESS if vis and vis.save_status else T.TEXT_MUTED, anchor="center")
        acct = self.app.session.account
        if acct:
            r = pygame.Rect(0, 0, 34, 34)
            r.midright = (self.logout_btn.rect.x - 10 - T.text_width(acct.username, 14, True) - 10, TOP_H // 2)
            pygame.draw.circle(surf, T.ACCENT_DARK, r.center, 17)
            T.draw_text(surf, acct.username[:1].upper(), r.center, 15, T.TEXT, bold=True, anchor="center")
            T.draw_text(surf, acct.username, (r.right + 10, TOP_H // 2), 14, T.TEXT, bold=True, anchor="midleft")
        self.logout_btn.draw(surf)

    def _draw_sidebar(self, surf):
        pygame.draw.rect(surf, T.PANEL, self.sidebar)
        pygame.draw.line(surf, T.BORDER, (LEFT_W - 1, self.sidebar.y), (LEFT_W - 1, self.sidebar.bottom))
        self.upload_btn.draw(surf)
        for key, b in self.tab_buttons.items():
            b.draw(surf)
            count = {"library": len(self.library_list.items), "queue": len(self.app.session.queue),
                     "saved": len(self.saved_list.items)}[key]
            if count:
                T.draw_text(surf, str(count), (b.rect.right - 8, b.rect.y + 4), 10, T.TEXT_MUTED, anchor="topright")
        self.lists[self.tab].draw(surf)
        self.demo_btn.draw(surf)

    def _draw_panel(self, surf):
        pygame.draw.rect(surf, T.PANEL, self.panel)
        pygame.draw.line(surf, T.BORDER, (self.panel.x, self.panel.y), (self.panel.x, self.panel.bottom))
        px = self.panel.x + 20
        clip = surf.get_clip()
        surf.set_clip(self.settings_clip)
        T.draw_text(surf, "VISUALIZER STYLE", (px, self.style_group.rect.y - 24), 12, T.TEXT_DIM, bold=True)
        T.draw_text(surf, "PRIMARY COLOUR", (px, self.color_swatches.rect.y - 22), 12, T.TEXT_DIM, bold=True)
        T.draw_text(surf, "ACCENT COLOUR", (px, self.color2_swatches.rect.y - 22), 12, T.TEXT_DIM, bold=True)
        T.draw_text(surf, "BACKGROUND", (px, self.bg_group.rect.y - 22), 12, T.TEXT_DIM, bold=True)
        T.draw_text(surf, "SENSITIVITY", (px, self.sens_slider.rect.y - 22), 12, T.TEXT_DIM, bold=True)
        T.draw_text(surf, f"{self.settings.sensitivity:.2f}x", (self.panel.right - 20, self.sens_slider.rect.y - 22),
                    12, T.TEXT_MUTED, anchor="topright")
        T.draw_text(surf, "SMOOTHING", (px, self.smooth_slider.rect.y - 22), 12, T.TEXT_DIM, bold=True)
        T.draw_text(surf, f"{int(self.settings.smoothing * 100)}%", (self.panel.right - 20, self.smooth_slider.rect.y - 22),
                    12, T.TEXT_MUTED, anchor="topright")
        for w in self.settings_widgets[:-2]:
            w.draw(surf)
        surf.set_clip(clip)
        self.save_btn.draw(surf)
        self.export_btn.draw(surf)

    def _draw_transport(self, surf):
        pygame.draw.rect(surf, T.PANEL, self.transport)
        pygame.draw.line(surf, T.BORDER, (0, self.transport.y), (self.transport.w, self.transport.y))
        self.prev_btn.draw(surf)
        self.next_btn.draw(surf)
        pygame.draw.circle(surf, T.ACCENT_HOVER if self.play_btn.hovered else T.ACCENT,
                           self.play_btn.rect.center, 26)
        T.draw_icon(surf, self.play_btn.icon, self.play_btn.rect.inflate(-28, -28), (255, 255, 255))

        player = self.app.player
        audio = self.app.session.get_current_audio()
        title = audio.display_name if audio else "No track loaded"
        T.draw_text(surf, title, (self.seek.rect.x, self.transport.y + 14), 14, T.TEXT, bold=True,
                    max_width=self.seek.rect.w - 120)
        T.draw_text(surf, f"{T.format_time(player.position)} / {T.format_time(player.duration)}",
                    (self.seek.rect.right, self.transport.y + 14), 13, T.TEXT_DIM, anchor="topright")
        self.seek.draw(surf)
        vol_icon = pygame.Rect(self.volume.rect.x - 34, self.volume.rect.centery - 10, 20, 20)
        T.draw_icon(surf, "volume", vol_icon, T.TEXT_DIM)
        self.volume.draw(surf)
        self.focus_btn.draw(surf)
        if not player.audio_ok and player.loaded:
            T.draw_text(surf, "No audio device - visuals only", (self.volume.rect.x, self.transport.y + 12),
                        11, T.WARNING)

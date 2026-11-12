"""The "Audio Visualizer Generation System" from the use-case diagram.

Combines the FFT analysis of a track with a scene renderer, soft light (bloom,
sun haze, god rays) and an atmospheric vignette to produce finished frames -
both for the live preview and for video export.
"""

from __future__ import annotations

from typing import Iterator

import pygame

from ..audio.analyzer import AudioAnalyzer
from ..models import VisualSettings
from .renderers import create_renderer
from .scenery import vignette

_font_cache: dict = {}

LIGHT_SCALE = 4   # the light surface is 1/4 of the frame size


def _font(size: int) -> pygame.font.Font:
    if size not in _font_cache:
        if not pygame.font.get_init():
            pygame.font.init()
        _font_cache[size] = pygame.font.SysFont("segoeuisemibold,segoeui,arial", size)
    return _font_cache[size]


class VisualizerEngine:
    def __init__(self, analyzer: AudioAnalyzer, settings: VisualSettings, title: str = ""):
        self.analyzer = analyzer
        self.title = title
        self.settings = settings
        self.renderer = create_renderer(settings.wave_type)
        self.renderer.analyzer = analyzer
        self.renderer.title = title
        self._light: pygame.Surface | None = None
        self._half: pygame.Surface | None = None
        self._full: pygame.Surface | None = None
        self._title_img = None

    def set_settings(self, settings: VisualSettings) -> None:
        if settings.wave_type != self.settings.wave_type:
            self.renderer = create_renderer(settings.wave_type)
            self.renderer.analyzer = self.analyzer
            self.renderer.title = self.title
        self.settings = settings

    def reset(self) -> None:
        self.renderer.reset()

    def render(self, target: pygame.Surface, t: float, dt: float) -> None:
        """Draw the visualization for playback time *t* onto *target*."""
        w, h = target.get_size()
        if w < 8 or h < 8:
            return
        frame = self.analyzer.frame_at(t)
        s = self.settings
        dt = max(1e-4, min(dt, 0.1))

        light = None
        if s.glow or s.atmosphere:
            size = (max(2, w // LIGHT_SCALE), max(2, h // LIGHT_SCALE))
            if self._light is None or self._light.get_size() != size:
                self._light = pygame.Surface(size)
            light = self._light
            light.fill((0, 0, 0))
        k = light.get_width() / w if light is not None else 0.25

        self.renderer.draw(target, light, k, frame, s, dt)

        if light is not None:
            # Light was drawn at 1/4 size: a smooth 2x step blurs it, a cheap 2x
            # step brings it to full size (it is so soft the second step is
            # invisible), then it is added on top of the scene. The buffers are
            # reused - allocating big surfaces every frame is slower than scaling.
            half_size = (max(1, w // 2), max(1, h // 2))
            if self._half is None or self._half.get_size() != half_size:
                self._half = pygame.Surface(half_size)
                self._full = pygame.Surface((w, h))
            pygame.transform.smoothscale(light, half_size, self._half)
            pygame.transform.scale(self._half, (w, h), self._full)
            target.blit(self._full, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        if s.atmosphere:
            vignette(target)
        if s.show_title and self.title:
            self._draw_title(target)

    def _draw_title(self, target: pygame.Surface) -> None:
        w, h = target.get_size()
        size = max(12, int(min(w, h) * 0.034))
        key = (self.title, size)
        if self._title_img is None or self._title_img[0] != key:
            f = _font(size)
            text = f.render(self.title, True, (250, 246, 238))
            shadow = f.render(self.title, True, (10, 12, 30))
            shadow.set_alpha(120)
            self._title_img = (key, text, shadow)
        _, text, shadow = self._title_img
        x, y = int(w * 0.04), int(h - h * 0.055 - text.get_height())
        target.blit(shadow, (x + max(1, size // 18), y + max(1, size // 18)))
        target.blit(text, (x, y))

    def frames(self, resolution: tuple[int, int], fps: int = 30, start: float = 0.0,
               duration: float | None = None) -> Iterator[pygame.Surface]:
        """Render every frame between *start* and *start + duration* (export)."""
        end = self.analyzer.duration if duration is None else min(self.analyzer.duration,
                                                                  start + duration)
        total = max(1, int((end - start) * fps))
        surface = pygame.Surface(resolution)
        self.reset()
        dt = 1.0 / fps
        for i in range(total):
            self.render(surface, start + i * dt, dt)
            yield surface

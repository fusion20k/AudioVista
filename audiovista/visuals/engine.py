"""The "Audio Visualizer Generation System" from the use-case diagram.

Combines the FFT analysis of a track with a renderer style, a background
and optional glow to produce finished frames - both for the live preview and
for video export.
"""

from __future__ import annotations

from typing import Iterator

import pygame

from ..audio.analyzer import AudioAnalyzer
from ..models import VisualSettings
from . import retro
from .backgrounds import draw_background
from .renderers import create_renderer

_font_cache: dict = {}


def _font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    if key not in _font_cache:
        if not pygame.font.get_init():
            pygame.font.init()
        _font_cache[key] = pygame.font.SysFont("segoeuisemibold,segoeui,arial", size)
    return _font_cache[key]


class VisualizerEngine:
    def __init__(self, analyzer: AudioAnalyzer, settings: VisualSettings, title: str = ""):
        self.analyzer = analyzer
        self.title = title
        self.settings = settings
        self.renderer = create_renderer(settings.wave_type)
        self.renderer.analyzer = analyzer
        self.renderer.title = title
        self._layer: pygame.Surface | None = None
        self._blur: pygame.Surface | None = None

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
        size = target.get_size()
        if size[0] < 8 or size[1] < 8:
            return
        frame = self.analyzer.frame_at(t)
        s = self.settings
        draw_background(target, s.background, s.primary_rgb, frame.beat)
        self.renderer.draw_backdrop(target, frame, s, max(1e-4, min(dt, 0.1)))

        if self._layer is None or self._layer.get_size() != size:
            self._layer = pygame.Surface(size, pygame.SRCALPHA)
        layer = self._layer
        layer.fill((0, 0, 0, 0))
        self.renderer.draw(layer, frame, s, max(1e-3, min(dt, 0.1)))

        if s.glow:
            # Bloom: shrink the layer a lot (cheap), blur it by smooth-scaling the
            # tiny copy up to half size, then stretch that to full size. The
            # result is so soft that the final cheap stretch is invisible.
            # Small blur pyramid: shrink, then grow back in 2x steps so the glow
            # stays round and smooth (one big upscale looks blocky).
            w, h = size
            img = pygame.transform.scale(layer, (max(1, w // 4), max(1, h // 4)))
            for div in (8, 16):
                img = pygame.transform.smoothscale(img, (max(1, w // div), max(1, h // div)))
            for div in (8, 4, 2):
                img = pygame.transform.smoothscale(img, (max(1, w // div), max(1, h // div)))
            if self._blur is None or self._blur.get_size() != size:
                self._blur = pygame.Surface(size, pygame.SRCALPHA)
            blur = pygame.transform.scale(img, size, self._blur)   # reused buffer: no big allocation
            target.blit(blur, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        target.blit(layer, (0, 0))

        if s.atmosphere:
            retro.apply(target, t)
        if s.show_title and self.title:
            self._draw_title(target)

    def _draw_title(self, target: pygame.Surface) -> None:
        w, h = target.get_size()
        size = max(12, int(min(w, h) * 0.035))
        text = _font(size, bold=True).render(self.title, True, (250, 246, 240))
        shadow = _font(size, bold=True).render(self.title, True, (0, 0, 0))
        shadow.set_alpha(140)
        x, y = int(w * 0.04), int(h - h * 0.05 - text.get_height())
        target.blit(shadow, (x + 2, y + 2))
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

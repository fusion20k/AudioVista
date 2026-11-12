"""Visualizer styles. Each renderer draws one frame onto a transparent layer."""

from __future__ import annotations

import math

import numpy as np
import pygame

from ..audio.analyzer import AnalysisFrame
from ..models import VisualSettings


def lerp_color(a, b, t: float):
    t = min(max(t, 0.0), 1.0)
    return (int(a[0] + (b[0] - a[0]) * t), int(a[1] + (b[1] - a[1]) * t),
            int(a[2] + (b[2] - a[2]) * t))


def scale_color(c, k: float):
    return (min(255, int(c[0] * k)), min(255, int(c[1] * k)), min(255, int(c[2] * k)))


def resample(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) == n:
        return values
    return np.interp(np.linspace(0, len(values) - 1, n), np.arange(len(values)), values)


class BaseRenderer:
    style = "base"

    def __init__(self):
        self._levels: np.ndarray | None = None

    def reset(self) -> None:
        self._levels = None

    def levels(self, frame: AnalysisFrame, settings: VisualSettings, dt: float) -> np.ndarray:
        """Band levels after sensitivity and attack/decay smoothing."""
        target = np.clip(frame.bands, 0.0, 1.0) ** (1.0 / settings.sensitivity)
        if self._levels is None or len(self._levels) != len(target):
            self._levels = target.copy()
            return self._levels
        # Frame-rate independent smoothing: fast attack, configurable release.
        attack = 1.0 - 0.25 ** (dt * 60)
        release = 1.0 - settings.smoothing ** (dt * 60)
        rate = np.where(target > self._levels, attack, release)
        self._levels += (target - self._levels) * rate
        return self._levels

    def draw(self, layer: pygame.Surface, frame: AnalysisFrame, settings: VisualSettings,
             dt: float) -> None:
        raise NotImplementedError


class BarsRenderer(BaseRenderer):
    """Classic equalizer bars with falling peak caps and a reflection."""

    style = "bars"

    def __init__(self):
        super().__init__()
        self._peaks: np.ndarray | None = None

    def reset(self) -> None:
        super().reset()
        self._peaks = None

    def draw(self, layer, frame, settings, dt):
        w, h = layer.get_size()
        n = int(min(96, max(24, w // 20)))
        levels = resample(self.levels(frame, settings, dt), n)
        if self._peaks is None or len(self._peaks) != n:
            self._peaks = levels.copy()
        self._peaks = np.maximum(levels, self._peaks - dt * 0.45)

        margin = w * 0.06
        slot = (w - 2 * margin) / n
        bar_w = max(2, int(slot * 0.68))
        base_y = int(h * 0.68)
        max_h = h * 0.52
        c1, c2 = settings.primary_rgb, settings.secondary_rgb
        reflection = pygame.Surface((w, h - base_y), pygame.SRCALPHA)
        radius = min(bar_w // 2, 6)

        for i, v in enumerate(levels):
            x = int(margin + i * slot + (slot - bar_w) / 2)
            bh = max(3, int(v * max_h))
            col = lerp_color(c1, c2, i / max(1, n - 1))
            col = scale_color(col, 0.65 + 0.45 * v)
            pygame.draw.rect(layer, col, (x, base_y - bh, bar_w, bh), border_radius=radius)
            ref_h = int(bh * 0.35)
            if ref_h > 1:
                pygame.draw.rect(reflection, (*col, 55), (x, 4, bar_w, ref_h), border_radius=radius)
            py = int(base_y - self._peaks[i] * max_h) - 8
            pygame.draw.rect(layer, scale_color(col, 1.3), (x, py, bar_w, 3), border_radius=2)

        layer.blit(reflection, (0, base_y))


class WaveformRenderer(BaseRenderer):
    """Oscilloscope-style line showing the actual sound wave."""

    style = "waveform"

    def __init__(self):
        super().__init__()
        self._wave: np.ndarray | None = None

    def reset(self):
        super().reset()
        self._wave = None

    def draw(self, layer, frame, settings, dt):
        w, h = layer.get_size()
        levels = self.levels(frame, settings, dt)
        points = 360
        wave = resample(frame.waveform, points)
        # Normalise so quiet tracks are still visible, then smooth over time.
        peak = max(0.08, float(np.abs(wave).max()))
        wave = wave / peak * min(1.0, 0.35 + frame.volume) * settings.sensitivity
        if self._wave is None or len(self._wave) != points:
            self._wave = wave
        keep = settings.smoothing * 0.6
        self._wave = self._wave * keep + wave * (1 - keep)

        mid = h * 0.5
        amp = h * 0.32
        xs = np.linspace(w * 0.04, w * 0.96, points)
        taper = np.sin(np.linspace(0, math.pi, points)) ** 0.6
        ys = mid - self._wave * amp * taper
        pts = list(zip(xs.tolist(), ys.tolist()))
        c1, c2 = settings.primary_rgb, settings.secondary_rgb

        # Spectrum "mountain" behind the wave
        spec = resample(levels, points) * taper
        poly = [(w * 0.04, mid)] + [(x, mid - s * h * 0.18) for x, s in zip(xs, spec)] + [(w * 0.96, mid)]
        poly += [(x, mid + s * h * 0.18) for x, s in zip(xs[::-1], spec[::-1])]
        pygame.draw.polygon(layer, (*scale_color(c2, 0.55), 70), poly)

        pygame.draw.line(layer, (*c1, 40), (0, mid), (w, mid), 1)
        thick = max(2, int(h / 160))
        segs = 6
        step = len(pts) // segs
        for k in range(segs):
            seg = pts[k * step: (k + 1) * step + 1]
            col = lerp_color(c1, c2, k / (segs - 1))
            if len(seg) > 1:
                pygame.draw.lines(layer, col, False, seg, thick + 2)
        pygame.draw.aalines(layer, (255, 255, 255), False, pts)


class CircularRenderer(BaseRenderer):
    """Bars radiating from a pulsing circle that breathes with the beat."""

    style = "circular"

    def __init__(self):
        super().__init__()
        self._angle = 0.0

    def reset(self):
        super().reset()
        self._angle = 0.0

    def draw(self, layer, frame, settings, dt):
        w, h = layer.get_size()
        levels = self.levels(frame, settings, dt)
        half = resample(levels, 64)
        ring = np.concatenate([half, half[::-1]])  # mirror for symmetry
        n = len(ring)
        cx, cy = w / 2, h / 2
        size = min(w, h)
        radius = size * 0.17 * (1 + 0.10 * frame.beat + 0.05 * frame.bass)
        length = size * 0.24
        self._angle += dt * (0.15 + 0.6 * frame.volume)
        c1, c2 = settings.primary_rgb, settings.secondary_rgb
        width = max(2, int(size * 2 * math.pi * 0.17 / n * 0.55))

        for i, v in enumerate(ring):
            a = self._angle + 2 * math.pi * i / n - math.pi / 2
            ca, sa = math.cos(a), math.sin(a)
            r0 = radius + 4
            r1 = radius + 6 + v * length
            col = lerp_color(c1, c2, abs(i / n * 2 - 1))
            col = scale_color(col, 0.6 + 0.5 * v)
            pygame.draw.line(layer, col, (cx + ca * r0, cy + sa * r0), (cx + ca * r1, cy + sa * r1), width)

        inner = int(radius)
        pygame.draw.circle(layer, (*scale_color(c1, 0.25), 200), (int(cx), int(cy)), inner)
        pygame.draw.circle(layer, lerp_color(c1, (255, 255, 255), 0.3 + 0.5 * frame.beat),
                           (int(cx), int(cy)), inner, max(2, int(size / 180)))
        # Inner bass ripple
        rip = int(inner * (0.35 + 0.5 * frame.bass))
        if rip > 2:
            pygame.draw.circle(layer, (*c2, int(60 + 120 * frame.beat)), (int(cx), int(cy)), rip,
                               max(1, int(size / 300)))


class ParticlesRenderer(BaseRenderer):
    """Particles burst out on every beat, drifting around a spectrum ring."""

    style = "particles"
    MAX_PARTICLES = 1400

    def __init__(self, seed: int = 1234):
        super().__init__()
        self.seed = seed
        self.reset()

    def reset(self):
        super().reset()
        self._rng = np.random.default_rng(self.seed)
        self._pos = np.zeros((0, 2))
        self._vel = np.zeros((0, 2))
        self._life = np.zeros(0)
        self._max_life = np.zeros(0)
        self._size = np.zeros(0)
        self._hue = np.zeros(0)
        self._spin = 0.0

    def _emit(self, count: int, cx: float, cy: float, speed: float, scale: float):
        if count <= 0:
            return
        rng = self._rng
        ang = rng.uniform(0, 2 * math.pi, count)
        spd = rng.uniform(0.3, 1.0, count) * speed * scale
        r0 = rng.uniform(0.12, 0.16, count) * scale
        pos = np.stack([cx + np.cos(ang) * r0, cy + np.sin(ang) * r0], axis=1)
        vel = np.stack([np.cos(ang) * spd, np.sin(ang) * spd], axis=1)
        life = rng.uniform(1.2, 2.6, count)
        self._pos = np.concatenate([self._pos, pos])
        self._vel = np.concatenate([self._vel, vel])
        self._life = np.concatenate([self._life, life])
        self._max_life = np.concatenate([self._max_life, life])
        self._size = np.concatenate([self._size, rng.uniform(1.6, 5.0, count) * scale / 500])
        self._hue = np.concatenate([self._hue, rng.uniform(0, 1, count)])
        extra = len(self._life) - self.MAX_PARTICLES
        if extra > 0:
            for name in ("_pos", "_vel", "_life", "_max_life", "_size", "_hue"):
                setattr(self, name, getattr(self, name)[extra:])

    def draw(self, layer, frame, settings, dt):
        w, h = layer.get_size()
        cx, cy = w / 2, h / 2
        scale = min(w, h)
        levels = self.levels(frame, settings, dt)
        c1, c2 = settings.primary_rgb, settings.secondary_rgb

        steady = int(frame.volume * 90 * dt * settings.sensitivity)
        burst = int(70 * settings.sensitivity) if frame.is_beat else 0
        self._emit(steady + burst, cx, cy, 0.35 + 0.9 * frame.bass, scale)

        if len(self._life):
            # Swirl: rotate velocity slightly, accelerate with the music.
            self._spin = 0.4 + 1.6 * frame.volume
            rot = self._spin * dt * 0.5
            c, s = math.cos(rot), math.sin(rot)
            vx, vy = self._vel[:, 0].copy(), self._vel[:, 1].copy()
            self._vel[:, 0] = vx * c - vy * s
            self._vel[:, 1] = vx * s + vy * c
            self._pos += self._vel * dt * (1.0 + 1.5 * frame.beat)
            self._life -= dt
            alive = self._life > 0
            for name in ("_pos", "_vel", "_life", "_max_life", "_size", "_hue"):
                setattr(self, name, getattr(self, name)[alive])

        for (x, y), life, mlife, size, hue in zip(self._pos, self._life, self._max_life,
                                                  self._size, self._hue):
            k = life / mlife
            col = lerp_color(c1, c2, hue)
            r = max(1, int(size * (0.6 + 0.8 * frame.bass)))
            pygame.draw.circle(layer, (*col, int(255 * min(1.0, k * 1.5))), (int(x), int(y)), r)

        # Spectrum ring
        ring = resample(levels, 90)
        ring = np.concatenate([ring, ring[::-1]])
        base = scale * (0.12 + 0.02 * frame.beat)
        pts = []
        for i, v in enumerate(ring):
            a = 2 * math.pi * i / len(ring) - math.pi / 2
            r = base + v * scale * 0.08
            pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
        core = int(base * (0.55 + 0.25 * frame.bass))
        pygame.draw.circle(layer, (*scale_color(c2, 0.6), int(40 + 80 * frame.beat)), (int(cx), int(cy)), core)
        for (x, y), v, i in zip(pts, ring, range(len(ring))):
            a = 2 * math.pi * i / len(ring) - math.pi / 2
            r0 = base - 4
            pygame.draw.line(layer, lerp_color(c1, c2, v), (cx + math.cos(a) * r0, cy + math.sin(a) * r0),
                             (x, y), max(1, int(scale / 400)))
        pygame.draw.aalines(layer, lerp_color(c1, (255, 255, 255), 0.4 + 0.4 * frame.beat), True, pts)


RENDERERS = {cls.style: cls for cls in (BarsRenderer, WaveformRenderer,
                                        CircularRenderer, ParticlesRenderer)}


def create_renderer(style: str) -> BaseRenderer:
    try:
        return RENDERERS[style]()
    except KeyError:
        raise ValueError(f"Unknown visualizer style: {style}") from None

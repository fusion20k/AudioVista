"""Visualizer styles. Each renderer draws one frame onto a transparent layer.

Lines and circles use anti-aliased drawing (pygame.gfxdraw) so the visuals
stay smooth on high-resolution screens and in 1080p exports.
"""

from __future__ import annotations

import math

import numpy as np
import pygame
import pygame.gfxdraw as gfx

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


# --------------------------------------------------------- anti-aliasing

def aa_line(surf, color, a, b, width: float) -> None:
    """Anti-aliased line of any thickness."""
    if width <= 1.5:
        pygame.draw.aaline(surf, color, a, b)
        return
    dx, dy = b[0] - a[0], b[1] - a[1]
    length = math.hypot(dx, dy) or 1.0
    nx, ny = -dy / length * width / 2, dx / length * width / 2
    pts = [(a[0] + nx, a[1] + ny), (b[0] + nx, b[1] + ny), (b[0] - nx, b[1] - ny), (a[0] - nx, a[1] - ny)]
    gfx.aapolygon(surf, pts, color)
    gfx.filled_polygon(surf, pts, color)


def aa_polyline(surf, color, xs: np.ndarray, ys: np.ndarray, width: float) -> None:
    """Anti-aliased thick polyline built as one ribbon polygon."""
    if len(xs) < 2:
        return
    if width <= 1.5:
        pygame.draw.aalines(surf, color, False, list(zip(xs.tolist(), ys.tolist())))
        return
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    length = np.hypot(dx, dy)
    length[length == 0] = 1.0
    nx, ny = -dy / length * width / 2, dx / length * width / 2
    left = list(zip((xs + nx).tolist(), (ys + ny).tolist()))
    right = list(zip((xs - nx).tolist(), (ys - ny).tolist()))
    poly = left + right[::-1]
    gfx.filled_polygon(surf, poly, color)
    gfx.aapolygon(surf, poly, color)


def aa_gradient_polyline(surf, c1, c2, xs, ys, width: float, segments: int = 8, alpha: int = 255) -> None:
    """Thick polyline whose colour fades from c1 to c2 along its length."""
    n = len(xs)
    step = max(2, n // segments)
    for k, start in enumerate(range(0, n - 1, step)):
        stop = min(n, start + step + 1)
        col = lerp_color(c1, c2, k / max(1, segments - 1))
        aa_polyline(surf, (*col, alpha), xs[start:stop], ys[start:stop], width)


def aa_circle(surf, color, center, radius: float, width: int = 0) -> None:
    x, y, r = int(center[0]), int(center[1]), int(radius)
    if r < 1:
        return
    if width:
        for k in range(max(1, int(width))):
            gfx.aacircle(surf, x, y, max(1, r - k), color)
    else:
        gfx.filled_circle(surf, x, y, r, color)
        gfx.aacircle(surf, x, y, r, color)


# --------------------------------------------------------------- styles

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
        radius = min(bar_w // 2, max(2, int(h / 120)))
        cap = max(2, int(h / 240))

        for i, v in enumerate(levels):
            x = int(margin + i * slot + (slot - bar_w) / 2)
            bh = max(cap, int(v * max_h))
            col = lerp_color(c1, c2, i / max(1, n - 1))
            col = scale_color(col, 0.65 + 0.45 * v)
            pygame.draw.rect(layer, col, (x, base_y - bh, bar_w, bh), border_radius=radius)
            ref_h = int(bh * 0.35)
            if ref_h > 1:
                pygame.draw.rect(reflection, (*col, 55), (x, cap * 2, bar_w, ref_h), border_radius=radius)
            py = int(base_y - self._peaks[i] * max_h) - cap * 3
            pygame.draw.rect(layer, scale_color(col, 1.3), (x, py, bar_w, cap), border_radius=cap // 2)

        layer.blit(reflection, (0, base_y))


class WaveformRenderer(BaseRenderer):
    """Silk waves: layered ribbons that flow with the music.

    Each ribbon follows one slice of the spectrum (bass at the front, treble at
    the back) so you can see which frequencies are active, and the real sound
    wave - triggered like an oscilloscope so it stands still instead of
    jittering - runs through the middle as a bright line.
    """

    style = "waveform"
    LAYERS = 5
    POINTS = 240

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self._phase = np.linspace(0, 2 * math.pi, self.LAYERS, endpoint=False)
        self._drift = np.arange(self.LAYERS, dtype=float) * 1.7
        self._amps = np.zeros(self.LAYERS)
        self._trace: np.ndarray | None = None

    def _stable_trace(self, raw: np.ndarray, n: int) -> np.ndarray:
        """Oscilloscope trigger: start each frame on a rising zero crossing."""
        smooth = np.convolve(raw, np.hanning(19) / np.hanning(19).sum(), mode="same")
        half = len(smooth) // 2
        rising = np.where((smooth[: half - 1] <= 0) & (smooth[1:half] > 0))[0]
        start = int(rising[0]) if rising.size else 0
        seg = smooth[start: start + half]
        if len(seg) < 4:
            seg = smooth[:half]
        return resample(seg, n)

    def draw(self, layer, frame, settings, dt):
        w, h = layer.get_size()
        levels = self.levels(frame, settings, dt)
        n = self.POINTS
        mid = h * 0.5
        u = np.linspace(-1.0, 1.0, n)
        xs = (u + 1.0) * 0.5 * w
        edge = (1.0 - u ** 2) ** 2.2          # ribbons pinch to nothing at the edges
        c1, c2 = settings.primary_rgb, settings.secondary_rgb

        groups = np.array_split(levels, self.LAYERS)
        target = np.array([g.mean() for g in groups]) ** 0.7
        self._amps += (target - self._amps) * min(1.0, dt * 14)
        energy = 1.0 + 2.2 * frame.volume

        # Back (treble) to front (bass)
        for k in reversed(range(self.LAYERS)):
            direction = 1 if k % 2 == 0 else -1
            self._phase[k] += dt * (1.1 + 0.5 * k) * energy * direction
            self._drift[k] += dt * (0.25 + 0.08 * k)
            focus = 0.5 * math.sin(self._drift[k])          # where along the width this ribbon swells
            bump = np.exp(-((u - focus) / (0.55 - 0.05 * k)) ** 2)
            cycles = 1.3 + 0.7 * k
            amp = h * 0.42 * self._amps[k] * (1.0 + 0.3 * frame.beat) / (1 + 0.15 * k)
            y = np.abs(amp * edge * bump * np.sin(math.pi * cycles * u + self._phase[k]))
            if amp < 1.0:
                continue
            col = lerp_color(c1, c2, k / (self.LAYERS - 1))
            top, bottom = mid - y, mid + y
            poly = list(zip(xs.tolist(), top.tolist())) + list(zip(xs[::-1].tolist(), bottom[::-1].tolist()))
            gfx.filled_polygon(layer, poly, (*col, 64))
            core = 0.45
            inner = list(zip(xs.tolist(), (mid - y * core).tolist())) +                 list(zip(xs[::-1].tolist(), (mid + y[::-1] * core).tolist()))
            gfx.filled_polygon(layer, inner, (*lerp_color(col, (255, 255, 255), 0.35), 70))
            edge_col = (*lerp_color(col, (255, 255, 255), 0.25), 200)
            pygame.draw.aalines(layer, edge_col, False, poly[:n])
            pygame.draw.aalines(layer, edge_col, False, poly[n:])

        # Faint centre line, then the real (stabilised) waveform on top
        aa_line(layer, (*c1, 45), (0, mid), (w, mid), max(1.0, h / 700))
        trace = self._stable_trace(frame.waveform, n)
        peak = max(0.04, float(np.abs(trace).max()))
        trace = trace / peak * min(1.0, 0.15 + 1.1 * frame.volume)
        if self._trace is None or len(self._trace) != n:
            self._trace = trace
        self._trace = self._trace * 0.35 + trace * 0.65
        ys = mid - self._trace * h * 0.12 * (1.0 - u ** 2) ** 1.5
        width = max(2.0, h / 200) * (1.0 + 0.6 * frame.beat)
        white = (255, 255, 255)
        aa_gradient_polyline(layer, lerp_color(c1, white, 0.5), lerp_color(c2, white, 0.5), xs, ys, width)


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
        width = max(2.0, size * 2 * math.pi * 0.17 / n * 0.5)
        gap = size * 0.012

        for i, v in enumerate(ring):
            a = self._angle + 2 * math.pi * i / n - math.pi / 2
            ca, sa = math.cos(a), math.sin(a)
            r0 = radius + gap
            r1 = radius + gap * 1.5 + v * length
            col = lerp_color(c1, c2, abs(i / n * 2 - 1))
            col = scale_color(col, 0.6 + 0.5 * v)
            aa_line(layer, col, (cx + ca * r0, cy + sa * r0), (cx + ca * r1, cy + sa * r1), width)

        inner = radius
        aa_circle(layer, (*scale_color(c1, 0.25), 210), (cx, cy), inner)
        aa_circle(layer, lerp_color(c1, (255, 255, 255), 0.3 + 0.5 * frame.beat), (cx, cy), inner,
                  width=max(2, int(size / 180)))
        rip = inner * (0.35 + 0.5 * frame.bass)
        if rip > 2:
            aa_circle(layer, (*c2, int(60 + 120 * frame.beat)), (cx, cy), rip, width=max(1, int(size / 300)))


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
            r = max(1.0, size * (0.6 + 0.8 * frame.bass))
            aa_circle(layer, (*col, int(255 * min(1.0, k * 1.5))), (x, y), r)

        # Spectrum ring
        ring = resample(levels, 90)
        ring = np.concatenate([ring, ring[::-1]])
        base = scale * (0.12 + 0.02 * frame.beat)
        pts = []
        for i, v in enumerate(ring):
            a = 2 * math.pi * i / len(ring) - math.pi / 2
            r = base + v * scale * 0.08
            pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
        core = base * (0.55 + 0.25 * frame.bass)
        aa_circle(layer, (*scale_color(c2, 0.6), int(40 + 80 * frame.beat)), (cx, cy), core)
        spoke = max(1.0, scale / 420)
        r0 = base - scale * 0.006
        for (x, y), v, i in zip(pts, ring, range(len(ring))):
            a = 2 * math.pi * i / len(ring) - math.pi / 2
            aa_line(layer, lerp_color(c1, c2, v), (cx + math.cos(a) * r0, cy + math.sin(a) * r0), (x, y), spoke)
        pygame.draw.aalines(layer, lerp_color(c1, (255, 255, 255), 0.4 + 0.4 * frame.beat), True, pts)


RENDERERS = {cls.style: cls for cls in (BarsRenderer, WaveformRenderer,
                                        CircularRenderer, ParticlesRenderer)}


def create_renderer(style: str) -> BaseRenderer:
    try:
        return RENDERERS[style]()
    except KeyError:
        raise ValueError(f"Unknown visualizer style: {style}") from None

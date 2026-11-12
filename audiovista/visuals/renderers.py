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
    """Audiogram: the track's real waveform scrolling past a glowing playhead.

    Bars left of the playhead have already played (full colour); bars on the
    right are what's coming up (dimmed). Each bar is the peak level of a fixed
    slice of time, so the shape is the actual, readable sound wave of the song.
    """

    style = "waveform"
    BAR_SECONDS = 0.085

    def __init__(self):
        super().__init__()
        self.analyzer = None
        self._pulse = 0.0

    def reset(self):
        super().reset()
        self._pulse = 0.0

    def draw(self, layer, frame, settings, dt):
        if self.analyzer is None:
            return
        w, h = layer.get_size()
        c1, c2 = settings.primary_rgb, settings.secondary_rgb
        white = (255, 255, 255)
        mid = h * 0.5
        cx = w * 0.5
        spacing = max(4.0, w / 118.0)
        bar_w = max(2, int(spacing * 0.56))
        max_h = h * 0.27
        d = self.BAR_SECONDS
        t = frame.time

        half = int(cx / spacing) + 2
        first = int(math.floor(t / d)) - half
        count = 2 * half + 2
        env = self.analyzer.envelope_buckets(first, count, d)
        env = np.clip(env, 0, 1) ** (0.8 / settings.sensitivity)

        target = min(1.0, frame.volume * 0.6 + frame.beat * 0.6)
        keep = 0.5 + 0.45 * settings.smoothing
        self._pulse = self._pulse * keep + target * (1 - keep) if dt > 0.001 else self._pulse

        for i in range(count):
            j = first + i
            x = cx + (j * d - t) / d * spacing
            if x < -bar_w or x > w + bar_w:
                continue
            dist = (x - cx) / cx                       # -1 (left edge) .. 1 (right edge)
            fade = max(0.0, 1.0 - abs(dist) ** 2.2)    # fade out towards the edges
            near = math.exp(-(dist * 9) ** 2)          # bars right at the playhead
            v = env[i] * (1.0 + 0.35 * self._pulse * near)
            bh = max(bar_w, v * max_h * 2)
            played = (j + 1) * d <= t
            if played:
                col = lerp_color(c2, c1, 1.0 - min(1.0, -dist))
                col = lerp_color(col, white, 0.25 * near)
                alpha = int(255 * (0.35 + 0.65 * fade))
            else:
                col = lerp_color(c1, (90, 96, 130), 0.55)
                alpha = int(200 * fade)
            if alpha <= 4:
                continue
            rect = pygame.Rect(int(x - bar_w / 2), int(mid - bh / 2), bar_w, int(bh))
            pygame.draw.rect(layer, (*col, alpha), rect, border_radius=bar_w // 2)

        # Playhead
        glow_h = h * (0.40 + 0.06 * frame.beat)
        line_w = max(2.0, w / 700)
        aa_line(layer, (*lerp_color(c1, white, 0.6), 230), (cx, mid - glow_h), (cx, mid + glow_h), line_w)
        knob = max(4.0, h / 90) * (1.0 + 0.5 * frame.beat)
        aa_circle(layer, (*c1, 90), (cx, mid), knob * 2.2)
        aa_circle(layer, white, (cx, mid), knob)


def _circular_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian smoothing that wraps around (for closed shapes)."""
    k = max(1, int(sigma * 3))
    x = np.arange(-k, k + 1)
    kernel = np.exp(-(x / max(sigma, 1e-6)) ** 2 / 2)
    kernel /= kernel.sum()
    padded = np.concatenate([values[-k:], values, values[:k]])
    return np.convolve(padded, kernel, mode="valid")


_glow_masks: dict = {}


def _glow_mask(radius: int) -> pygame.Surface:
    """White radial glow (alpha 255 in the centre fading to 0 at *radius*), cached
    in 4 px size buckets."""
    radius = max(4, radius - radius % 4)
    surf = _glow_masks.get(radius)
    if surf is None:
        size = radius * 2
        yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
        d = np.sqrt((xx - radius + 0.5) ** 2 + (yy - radius + 0.5) ** 2) / radius
        alpha = (np.clip(1.0 - d, 0, 1) ** 1.6 * 255).astype(np.uint8)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        surf.fill((255, 255, 255, 0))
        pygame.surfarray.pixels_alpha(surf)[:] = alpha.T
        if len(_glow_masks) > 96:
            _glow_masks.clear()
        _glow_masks[radius] = surf
    return surf


class CircularRenderer(BaseRenderer):
    """Orb: a pulsing core inside two smooth spectrum shapes that rotate in
    opposite directions, a halo of dancing dots and shockwave rings on beats."""

    style = "circular"
    POINTS = 240

    def __init__(self):
        super().__init__()
        self.analyzer = None
        self._angle = 0.0

    def reset(self):
        super().reset()
        self._angle = 0.0

    def _shape(self, cx, cy, radius, amount, values, rotation):
        n = len(values)
        theta = rotation + np.linspace(0, 2 * math.pi, n, endpoint=False) - math.pi / 2
        r = radius * (1.0 + amount * values)
        return list(zip((cx + np.cos(theta) * r).tolist(), (cy + np.sin(theta) * r).tolist()))

    def draw(self, layer, frame, settings, dt):
        w, h = layer.get_size()
        cx, cy = w / 2, h / 2
        size = min(w, h)
        c1, c2 = settings.primary_rgb, settings.secondary_rgb
        white = (255, 255, 255)
        levels = self.levels(frame, settings, dt)
        self._angle += dt * (0.12 + 0.5 * frame.volume)
        R = size * 0.20 * (1.0 + 0.06 * frame.beat)

        # Front shape: spectrum mirrored four ways (bass on the diagonals, treble on
        # the axes) for a symmetric flower. Back shape: mirrored once and rotating.
        quarter = resample(levels, self.POINTS // 4)
        flower = _circular_smooth(np.concatenate([quarter, quarter[::-1]] * 2), 2.0)
        half = resample(levels, self.POINTS // 2)
        ring = np.concatenate([half, half[::-1]])
        back_vals = _circular_smooth(np.concatenate([half[::-1], half]), 3.0)

        # Beat shockwaves (deterministic: derived from the beat times)
        if self.analyzer is not None:
            for age in self.analyzer.recent_beats(frame.time, 1.1):
                k = age / 1.1
                rr = R * (1.05 + 1.5 * k)
                alpha = int(170 * (1 - k) ** 2)
                if alpha > 4:
                    aa_circle(layer, (*lerp_color(c1, c2, k), alpha), (cx, cy), rr,
                              width=max(1, int(size / 260 * (1 - k) + 1)))

        # Outer halo of dots
        n_dots = 96
        dot_vals = resample(ring, n_dots)
        for i, v in enumerate(dot_vals):
            a = -self._angle * 0.6 + 2 * math.pi * i / n_dots - math.pi / 2
            rr = R * (1.62 + 0.38 * v)
            col = lerp_color(c2, c1, v)
            aa_circle(layer, (*lerp_color(col, white, 0.2 * v), int(110 + 145 * v)),
                      (cx + math.cos(a) * rr, cy + math.sin(a) * rr), max(1.5, size / 330 * (0.6 + 1.6 * v)))

        # Layered wave rings: thin spectrum-shaped outlines, each slightly bigger,
        # dimmer and rotated, which reads as one shimmering, breathing shape.
        rings = 6
        line_w = max(1.0, size / 520)
        for k in reversed(range(rings)):
            vals = flower if k % 2 == 0 else back_vals
            rot = (math.pi / 4) + (self._angle * (0.5 + 0.25 * k)) * (1 if k % 2 == 0 else -1)
            pts = self._shape(cx, cy, R * (1.0 + 0.07 * k), 0.55 - 0.05 * k, vals, rot)
            col = lerp_color(c1, c2, k / (rings - 1))
            alpha = int(235 * (1.0 - k / rings) ** 1.2)
            if k == 0:
                gfx.filled_polygon(layer, pts, (*col, 40))
            # Thick lines via stacked 1px anti-aliased outlines (much faster than
            # filling a ribbon polygon around the whole circle).
            passes = max(1, int(round(line_w * (1.6 if k == 0 else 1.0))))
            for j in range(passes):
                grow = 1.0 + (j - (passes - 1) / 2) / max(1.0, R)
                ring_pts = pts if j == (passes - 1) // 2 and passes % 2 else                     [(cx + (x - cx) * grow, cy + (y - cy) * grow) for x, y in pts]
                pygame.draw.aalines(layer, (*col, alpha), True, ring_pts)

        # Core: a glowing orb (cached radial gradient, re-tinted with the music)
        core = R * (0.74 + 0.08 * frame.bass)
        dark = lerp_color(c1, (6, 6, 16), 0.8)
        hot = lerp_color(c1, white, 0.25 + 0.45 * frame.beat)
        aa_circle(layer, dark, (cx, cy), core)
        glow = _glow_mask(int(core)).copy()
        glow.fill((*hot, int(255 * (0.55 + 0.45 * frame.volume))), special_flags=pygame.BLEND_RGBA_MULT)
        layer.blit(glow, glow.get_rect(center=(int(cx), int(cy))))
        aa_circle(layer, lerp_color(c1, white, 0.45 + 0.4 * frame.beat), (cx, cy), core,
                  width=max(2, int(size / 220)))


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

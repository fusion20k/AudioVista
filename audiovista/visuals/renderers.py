"""Visualizer styles - clean, modern and constantly in motion.

* bars      - Spectrum: mirrored bars on springs (they overshoot and bounce),
              riding a floor that ripples with the music; peak caps fall with
              gravity and sparks fly off the bars on the beat.
* waveform  - Waveform: the live sound wave as a smooth glowing ribbon that
              changes every frame, trailed by fading echoes of the last few
              frames and a mirrored reflection.
* circular  - Radial: a breathing ring of spring-loaded spectrum bars around a
              wobbling spectrum "blob", orbiting dots and shockwaves on the
              beat; optionally particles stream out of the centre.

``draw_backdrop`` paints translucent parts straight onto the frame (they blend
correctly there); ``draw`` paints the bright parts onto a layer that the
engine blooms. Motion depends only on the analysis frame and ``dt``, so
exported videos match the live preview.
"""

from __future__ import annotations

import math
from collections import deque

import numpy as np
import pygame
import pygame.gfxdraw as gfx

from ..audio.analyzer import AnalysisFrame
from ..models import VisualSettings

TAU = 2 * math.pi
WHITE = (255, 250, 244)


# ------------------------------------------------------------------ helpers

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


def clamp01(v: float) -> float:
    return min(max(v, 0.0), 1.0)


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


def aa_circle(surf, color, center, radius: float) -> None:
    x, y, r = int(center[0]), int(center[1]), int(radius)
    if r >= 1:
        gfx.filled_circle(surf, x, y, r, color)
        gfx.aacircle(surf, x, y, r, color)


def aa_poly(surf, color, pts) -> None:
    gfx.filled_polygon(surf, pts, color)
    gfx.aapolygon(surf, pts, color)


def ribbon(surf, color, xs, ys, width: float) -> None:
    """A smooth thick curve through (xs, ys), anti-aliased, as one polygon."""
    dx = np.gradient(xs)
    dy = np.gradient(ys)
    length = np.hypot(dx, dy) + 1e-6
    nx, ny = -dy / length * width / 2, dx / length * width / 2
    top = list(zip((xs + nx).tolist(), (ys + ny).tolist()))
    bottom = list(zip((xs - nx).tolist(), (ys - ny).tolist()))
    aa_poly(surf, color, top + bottom[::-1])


class SpringBank:
    """One damped spring per value: things overshoot and bounce instead of snapping."""

    def __init__(self, stiffness: float = 260.0, damping: float = 13.0):
        self.k, self.c = stiffness, damping
        self.x: np.ndarray | None = None
        self.v: np.ndarray | None = None

    def update(self, target: np.ndarray, dt: float) -> np.ndarray:
        if self.x is None or len(self.x) != len(target):
            self.x, self.v = np.array(target, dtype=float), np.zeros(len(target))
            return self.x
        if dt <= 0.0015:
            return self.x
        steps = max(1, int(dt / 0.004))
        h = dt / steps
        for _ in range(steps):
            self.v += (self.k * (target - self.x) - self.c * self.v) * h
            self.x += self.v * h
        return self.x


class BaseRenderer:
    style = "base"

    def __init__(self):
        self._levels: np.ndarray | None = None
        self._levels_frame = None
        self.analyzer = None   # set by the engine
        self.title = ""

    def reset(self) -> None:
        self._levels = None
        self._levels_frame = None

    def draw_backdrop(self, target: pygame.Surface, frame: AnalysisFrame, settings: VisualSettings,
                      dt: float) -> None:
        """Optional: translucent parts drawn straight onto the frame, under the glow layer."""

    def levels(self, frame: AnalysisFrame, settings: VisualSettings, dt: float) -> np.ndarray:
        """Band levels after sensitivity and attack/decay smoothing (once per frame)."""
        if frame is self._levels_frame and self._levels is not None:
            return self._levels
        self._levels_frame = frame
        target = np.clip(frame.bands, 0.0, 1.0) ** (1.0 / settings.sensitivity)
        if self._levels is None or len(self._levels) != len(target):
            self._levels = target.copy()
            return self._levels
        if dt <= 0.0015:
            return self._levels
        attack = 1.0 - 0.25 ** (dt * 60)
        release = 1.0 - settings.smoothing ** (dt * 60)
        rate = np.where(target > self._levels, attack, release)
        self._levels += (target - self._levels) * rate
        return self._levels

    @staticmethod
    def bands3(levels: np.ndarray) -> tuple[float, float, float]:
        n = len(levels)
        return (float(levels[: n // 4].mean()), float(levels[n // 4: 5 * n // 8].mean()),
                float(levels[5 * n // 8:].mean()))

    def draw(self, layer: pygame.Surface, frame: AnalysisFrame, settings: VisualSettings,
             dt: float) -> None:
        raise NotImplementedError


# --------------------------------------------------------------- Spectrum

class SpectrumRenderer(BaseRenderer):
    style = "bars"
    PER_SIDE = 32

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self.springs = SpringBank(300.0, 11.0)
        self.heights: np.ndarray | None = None
        self.peaks: np.ndarray | None = None
        self.peak_v: np.ndarray | None = None
        self.t = 0.0
        self.sparks = np.zeros((0, 5))       # x, y, vx, vy, life
        self.rng = np.random.default_rng(7)
        self._frame = None

    def _update(self, frame, s, dt, w, h):
        """Advance the simulation once per frame (both draw passes call this)."""
        if frame is self._frame:
            return
        self._frame = frame
        lv = resample(self.levels(frame, s, dt), self.PER_SIDE)
        target = lv * (1 + 0.18 * frame.beat)
        self.heights = np.clip(self.springs.update(target, dt), 0, 1.25)
        if self.peaks is None:
            self.peaks, self.peak_v = self.heights.copy(), np.zeros_like(lv)
        if dt <= 0.0015:
            return
        self.t += dt * (0.8 + 1.6 * frame.volume)
        self.peak_v += dt * 1.8                         # caps fall back with gravity
        self.peaks = self.peaks - self.peak_v * dt
        hit = self.heights >= self.peaks
        self.peaks[hit], self.peak_v[hit] = self.heights[hit], -0.25 * self.heights[hit]   # little hop
        # sparks fly off the tallest bars on each beat
        if frame.is_beat or frame.beat > 0.97:
            xs, bw, floor, max_h = self._geometry(w, h)
            for i in np.argsort(self.heights)[-6:]:
                for side in (-1, 1):
                    x = w / 2 + side * (i + 0.5) * self._step(w)
                    y = self._floor_y(w, h, i, side) - max_h * self.heights[i]
                    n = 4
                    v = self.rng.normal(0, 1, (n, 2)) * [w * 0.06, h * 0.08] + [0, -h * 0.35]
                    new = np.column_stack([np.full(n, x), np.full(n, y), v, self.rng.uniform(0.6, 1.0, n)])
                    self.sparks = np.vstack([self.sparks, new])
        if len(self.sparks):
            self.sparks[:, 3] += h * 0.9 * dt                 # gravity
            self.sparks[:, 0] += self.sparks[:, 2] * dt
            self.sparks[:, 1] += self.sparks[:, 3] * dt
            self.sparks[:, 4] -= dt * 0.8
            self.sparks = self.sparks[self.sparks[:, 4] > 0][-300:]

    def _step(self, w):
        return w * 0.86 / (self.PER_SIDE * 2)

    def _geometry(self, w, h):
        step = self._step(w)
        bw = max(3, int(step * 0.62))
        return None, bw, h * 0.68, h * 0.50

    def _floor_y(self, w, h, i, side):
        """The floor ripples outward from the centre with the music."""
        return h * 0.68 + math.sin(self.t * 3.0 - i * 0.35) * h * 0.012 * (0.3 + self._vol)

    def draw_backdrop(self, target, frame, s, dt):
        w, h = target.get_size()
        self._vol = frame.volume
        self._update(frame, s, dt, w, h)
        _, bw, floor, max_h = self._geometry(w, h)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        step = self._step(w)
        for side in (-1, 1):            # soft reflection under each bar
            for i, v in enumerate(self.heights.tolist()):
                x = w / 2 + side * (i + 0.5) * step
                fy = self._floor_y(w, h, i, side)
                rh = int(max_h * (0.03 + v) * 0.35)
                if rh > 1:
                    gfx.box(target, (int(x - bw / 2), int(fy + 6), bw, rh),
                            (*lerp_color(c1, c2, i / (self.PER_SIDE - 1)), 45))

    def draw(self, layer, frame, s, dt):
        w, h = layer.get_size()
        u = min(w, h)
        self._vol = frame.volume
        self._update(frame, s, dt, w, h)
        _, bw, floor, max_h = self._geometry(w, h)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        step = self._step(w)
        r = bw // 2
        for side in (-1, 1):
            for i, v in enumerate(self.heights.tolist()):
                x = w / 2 + side * (i + 0.5) * step
                fy = self._floor_y(w, h, i, side)
                bh = max(bw, int(max_h * (0.03 + v)))
                col = lerp_color(lerp_color(c1, c2, (i / (self.PER_SIDE - 1)) ** 1.2), WHITE, 0.25 * frame.beat)
                pygame.draw.rect(layer, col, (int(x - bw / 2), int(fy - bh), bw, bh), border_radius=r)
                py = fy - max_h * (0.03 + max(float(self.peaks[i]), v)) - bw * 0.9
                pygame.draw.rect(layer, lerp_color(col, WHITE, 0.6), (int(x - bw / 2), int(py), bw, max(2, bw // 3)),
                                 border_radius=max(1, r // 2))
        for x, y, _vx, _vy, life in self.sparks.tolist():
            aa_circle(layer, lerp_color(c2, WHITE, life), (x, y), max(1.5, u * 0.004 * life))


# --------------------------------------------------------------- Waveform

class WaveformRenderer(BaseRenderer):
    style = "waveform"
    POINTS = 220
    ECHOES = 6

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self.history: deque = deque(maxlen=self.ECHOES)
        self.gain = 3.0
        self.t = 0.0
        self._frame = None
        self.curve = np.zeros(self.POINTS)

    def _update(self, frame, dt):
        if frame is self._frame:
            return
        self._frame = frame
        wave = np.asarray(frame.waveform, dtype=np.float64)
        # smooth out the hiss so the line flows, then shrink to POINTS samples
        wave = np.convolve(wave, np.hanning(17) / np.hanning(17).sum(), mode="same")
        curve = resample(wave, self.POINTS)
        # automatic gain: quiet passages still fill the screen, loud ones don't clip
        peak = max(0.02, float(np.abs(curve).max()))
        if dt > 0.0015:
            self.gain += (min(8.0, 0.9 / peak) - self.gain) * (1 - 0.2 ** dt)
            self.t += dt
            self.history.appendleft(self.curve.copy())
        # taper to the centre line at both ends
        self.curve = np.clip(curve * self.gain, -1.2, 1.2) * np.sin(np.linspace(0, math.pi, self.POINTS)) ** 0.6

    def _xs(self, w):
        return np.linspace(w * 0.04, w * 0.96, self.POINTS)

    def draw_backdrop(self, target, frame, s, dt):
        w, h = target.get_size()
        self._update(frame, dt)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        cy, amp = h * 0.48, h * 0.26 * (1 + 0.12 * frame.beat)
        xs = self._xs(w)
        # echoes of the last few frames drift outwards and fade
        for k, old in enumerate(self.history):
            f = (k + 1) / (self.ECHOES + 1)
            ys = cy - old * amp * (1 + 0.25 * f)
            ribbon(target, (*lerp_color(c1, c2, f), int(140 * (1 - f))), xs, ys, max(2.0, h * 0.006 * (1 - f)))
        # mirrored reflection
        ys = cy + self.curve * amp * 0.55
        gfx.filled_polygon(target, list(zip(xs.tolist(), ys.tolist())) + [(xs[-1], cy), (xs[0], cy)],
                           (*c2, 40))

    def draw(self, layer, frame, s, dt):
        w, h = layer.get_size()
        u = min(w, h)
        self._update(frame, dt)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        cy, amp = h * 0.48, h * 0.26 * (1 + 0.12 * frame.beat)
        xs = self._xs(w)
        ys = cy - self.curve * amp
        width = max(2.5, u * (0.007 + 0.006 * frame.volume))
        ribbon(layer, lerp_color(c1, c2, 0.5 + 0.5 * math.sin(self.t * 0.7)), xs, ys, width * 2.2)
        ribbon(layer, lerp_color(c1, WHITE, 0.35 + 0.3 * frame.beat), xs, ys, width)
        # a bright dot rides the wave from left to right with the beat
        j = int((self.t * 0.35 % 1.0) * (self.POINTS - 1))
        aa_circle(layer, WHITE, (xs[j], ys[j]), width * (1.2 + frame.beat))


# ------------------------------------------------------------------ Radial

class Emitter:
    """Particles streaming out of the centre: more when it's loud, a burst on each beat."""

    MAX = 1400

    def __init__(self, seed: int = 99):
        self.rng = np.random.default_rng(seed)
        self.p = np.zeros((0, 7))     # x, y (relative to centre, px), vx, vy, life, max_life, hue
        self.carry = 0.0

    def update(self, frame, u, bass, dt):
        if dt <= 0.0015:
            return
        rate = 25 + 260 * frame.volume                   # particles per second
        self.carry += rate * dt
        n = int(self.carry)
        self.carry -= n
        if frame.is_beat or frame.beat > 0.97:
            n += int(60 + 140 * bass)
        if n:
            a = self.rng.uniform(0, TAU, n)
            speed = u * self.rng.uniform(0.12, 0.45, n) * (0.6 + 0.9 * frame.volume + 0.8 * frame.beat)
            r0 = u * 0.05 * self.rng.random(n)
            life = self.rng.uniform(1.2, 2.6, n)
            new = np.column_stack([np.cos(a) * r0, np.sin(a) * r0, np.cos(a) * speed, np.sin(a) * speed,
                                   life, life, self.rng.random(n)])
            self.p = np.vstack([self.p, new])[-self.MAX:]
        p = self.p
        if len(p):
            # a gentle swirl and a little drag
            swirl = 0.35 + 0.8 * frame.volume
            vx, vy = p[:, 2].copy(), p[:, 3].copy()
            p[:, 2] += (-vy * swirl - vx * 0.25) * dt
            p[:, 3] += (vx * swirl - vy * 0.25) * dt
            p[:, 0] += p[:, 2] * dt
            p[:, 1] += p[:, 3] * dt
            p[:, 4] -= dt
            self.p = p[p[:, 4] > 0]

    def draw(self, target, cx, cy, c1, c2, u):
        for x, y, _vx, _vy, life, max_life, hue in self.p.tolist():
            f = life / max_life
            col = lerp_color(lerp_color(c1, c2, hue), WHITE, 0.3 * f)
            a = int(230 * min(1.0, f * 2.5))
            r = max(1.0, u * 0.0045 * (0.4 + f))
            gfx.filled_circle(target, int(cx + x), int(cy + y), int(r), (*col, a))


class RadialRenderer(BaseRenderer):
    style = "circular"
    BARS = 56           # per side; the ring is mirrored left/right

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self.springs = SpringBank(280.0, 10.0)
        self.breath = SpringBank(160.0, 8.0)
        self.orbit = [0.0, 2.1, 4.2]
        self.t = 0.0
        self.emitter = Emitter()
        self._frame = None
        self.bars = np.zeros(self.BARS)
        self.R = 0.0

    def _update(self, frame, s, dt, u):
        if frame is self._frame:
            return
        self._frame = frame
        lv = self.levels(frame, s, dt)
        bass, mid, treble = self.bands3(lv)
        self.bars = np.clip(self.springs.update(resample(lv, self.BARS), dt), 0, 1.3)
        breath = float(self.breath.update(np.array([bass + 0.6 * frame.beat]), dt)[0])
        self.R = u * 0.15 * (1 + 0.22 * breath)
        if dt > 0.0015:
            self.t += dt * (0.6 + 1.5 * frame.volume)
            for i, band in enumerate((bass, mid, treble)):
                self.orbit[i] += dt * (0.6 + 0.5 * i) * (0.4 + 3.0 * band) * (1 if i % 2 == 0 else -1)
        if s.particles:
            self.emitter.update(frame, u, bass, dt)

    def draw_backdrop(self, target, frame, s, dt):
        w, h = target.get_size()
        u = min(w, h)
        self._update(frame, s, dt, u)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        cx, cy = w / 2, h / 2
        if s.particles:
            self.emitter.draw(target, cx, cy, c1, c2, u)
        # shockwave rings spreading out after each beat
        if self.analyzer is not None:
            for age in self.analyzer.recent_beats(frame.time, 1.6).tolist():
                a = int(120 * (1 - age / 1.6))
                if a > 4:
                    for d in (0, 3):
                        gfx.aacircle(target, int(cx), int(cy), int(self.R * 1.1 + age * u * 0.35 + d),
                                     (*lerp_color(c1, WHITE, 0.4), a))
        # the wobbling spectrum blob inside the ring
        pts = self._blob(cx, cy, 0.78, 0.35, 0)
        gfx.filled_polygon(target, pts, (*lerp_color(c1, c2, 0.5), 70))

    def _blob(self, cx, cy, base, depth, phase):
        n = self.BARS
        vals = np.concatenate([self.bars, self.bars[::-1]])
        a = -math.pi / 2 + np.linspace(0, TAU, 2 * n, endpoint=False) + phase
        wob = 1 + 0.05 * np.sin(a * 5 + self.t * 2.0)
        r = self.R * (base + depth * vals) * wob
        return list(zip((cx + np.cos(a) * r).tolist(), (cy + np.sin(a) * r).tolist()))

    def draw(self, layer, frame, s, dt):
        w, h = layer.get_size()
        u = min(w, h)
        self._update(frame, s, dt, u)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        cx, cy = w / 2, h / 2
        R = self.R
        sway = 0.18 * math.sin(self.t * 0.5)                  # the ring rocks back and forth
        width = max(2.0, R * TAU / (self.BARS * 2) * 0.55)
        r0 = R * 1.08
        for side in (-1, 1):
            for i, v in enumerate(self.bars.tolist()):
                a = -math.pi / 2 + side * (i + 0.5) / self.BARS * math.pi + sway
                L = u * (0.012 + 0.22 * v)
                ca, sa = math.cos(a), math.sin(a)
                col = lerp_color(c1, c2, i / (self.BARS - 1))
                aa_line(layer, col, (cx + ca * r0, cy + sa * r0), (cx + ca * (r0 + L), cy + sa * (r0 + L)), width)
                if v > 0.55:                                  # bright tips on the loud bars
                    aa_circle(layer, lerp_color(col, WHITE, 0.6),
                              (cx + ca * (r0 + L + width), cy + sa * (r0 + L + width)), width * 0.7)
        gfx.aacircle(layer, int(cx), int(cy), int(R), (*lerp_color(c1, WHITE, 0.3), 255))
        pygame.draw.aalines(layer, lerp_color(c2, WHITE, 0.3), True, self._blob(cx, cy, 0.78, 0.35, 0))
        for i, ang in enumerate(self.orbit):
            rr = R * (0.45 + 0.12 * i)
            aa_circle(layer, (c1, lerp_color(c1, c2, 0.5), c2)[i],
                      (cx + math.cos(ang) * rr, cy + math.sin(ang) * rr), u * 0.008)
        aa_circle(layer, lerp_color(c1, WHITE, 0.6), (cx, cy), R * (0.12 + 0.12 * frame.beat))


RENDERERS = {cls.style: cls for cls in (SpectrumRenderer, WaveformRenderer, RadialRenderer)}


def create_renderer(style: str) -> BaseRenderer:
    try:
        return RENDERERS[style]()
    except KeyError:
        raise ValueError(f"Unknown visualizer style: {style}") from None

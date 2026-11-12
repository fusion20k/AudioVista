"""Visualizer styles - clean, modern and always moving.

* bars      - Spectrum: mirrored rounded bars (bass in the middle) with falling
              peak caps and a soft reflection.
* waveform  - Waveform: the real waveform scrolling past a playhead (played part
              bright, upcoming part faint) over flowing bass/mid/treble waves.
* circular  - Radial: a slowly turning ring of spectrum bars around a pulsing
              core, orbiting dots and shockwave rings on the beat.
* meters    - Rings: three arc gauges for bass, mids and treble with spinning
              dash rings whose speed follows each band.
* particles - Particles: hundreds of particles flowing through a swirling field;
              the music speeds them up and beats burst them outwards.

``draw_backdrop`` paints translucent parts straight onto the frame (cheap and
blends correctly); ``draw`` paints the bright parts onto a transparent layer
that the engine blooms. All motion depends only on the analysis frame and
``dt``, so exported videos match the live preview.
"""

from __future__ import annotations

import math

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


def arc_points(cx, cy, r, a0, a1, steps):
    return [(cx + math.cos(a) * r, cy + math.sin(a) * r) for a in np.linspace(a0, a1, max(2, steps))]


def thick_arc(surf, color, center, r, width, a0, a1) -> None:
    """Anti-aliased arc band from angle a0 to a1 (radians) with round ends."""
    if a1 - a0 < 1e-3:
        return
    steps = max(4, int((a1 - a0) * r / 6))
    outer = arc_points(center[0], center[1], r + width / 2, a0, a1, steps)
    inner = arc_points(center[0], center[1], r - width / 2, a1, a0, steps)
    aa_poly(surf, color, outer + inner)
    for a in (a0, a1):
        aa_circle(surf, color, (center[0] + math.cos(a) * r, center[1] + math.sin(a) * r), width / 2)


class Spring:
    """Damped spring so gauges sweep and settle smoothly instead of jumping."""

    def __init__(self, stiffness: float = 120.0, damping: float = 14.0):
        self.k, self.c = stiffness, damping
        self.x = 0.0
        self.v = 0.0

    def update(self, target: float, dt: float) -> float:
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
    PER_SIDE = 36

    def __init__(self):
        super().__init__()
        self.peaks: np.ndarray | None = None
        self.peak_v: np.ndarray | None = None

    def reset(self):
        super().reset()
        self.peaks = None

    def _bars(self, frame, s, dt):
        lv = resample(self.levels(frame, s, dt), self.PER_SIDE)
        if self.peaks is None:
            self.peaks = lv.copy()
            self.peak_v = np.zeros_like(lv)
        elif dt > 0.0015 and frame is not getattr(self, "_peak_frame", None):
            self._peak_frame = frame
            self.peak_v += dt * 1.6                       # peaks fall with gravity
            self.peaks = self.peaks - self.peak_v * dt
            hit = lv >= self.peaks
            self.peaks[hit] = lv[hit]
            self.peak_v[hit] = 0.0
        return lv

    def _geometry(self, w, h):
        n = self.PER_SIDE * 2
        span = w * 0.88
        step = span / n
        bw = max(2, int(step * 0.62))
        floor = h * 0.70
        max_h = h * 0.52
        xs = [w / 2 + (i + 0.5) * step * side for side in (-1, 1) for i in range(self.PER_SIDE)]
        return xs, bw, floor, max_h

    def draw_backdrop(self, target, frame, s, dt):
        w, h = target.get_size()
        lv = self._bars(frame, s, dt)
        xs, bw, floor, max_h = self._geometry(w, h)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        # soft reflection under the floor line
        for i, x in enumerate(xs):
            v = lv[i % self.PER_SIDE]
            rh = int(max_h * (0.02 + v) * 0.35)
            if rh > 1:
                gfx.box(target, (int(x - bw / 2), int(floor + 6), bw, rh),
                        (*lerp_color(c1, c2, v * 0.6), 46))
        gfx.hline(target, int(w * 0.05), int(w * 0.95), int(floor + 3), (*WHITE, 40))

    def draw(self, layer, frame, s, dt):
        w, h = layer.get_size()
        lv = self._bars(frame, s, dt)
        xs, bw, floor, max_h = self._geometry(w, h)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        r = bw // 2
        for i, x in enumerate(xs):
            j = i % self.PER_SIDE
            v = float(lv[j])
            bh = max(bw, int(max_h * (0.02 + v)))
            col = lerp_color(c1, c2, (j / (self.PER_SIDE - 1)) ** 1.3)
            pygame.draw.rect(layer, col, (int(x - bw / 2), int(floor - bh), bw, bh), border_radius=r)
            py = floor - max_h * (0.02 + float(self.peaks[j])) - bw * 1.2
            pygame.draw.rect(layer, lerp_color(col, WHITE, 0.6),
                             (int(x - bw / 2), int(py), bw, max(2, bw // 3)), border_radius=max(1, r // 2))


# --------------------------------------------------------------- Waveform

class WaveformRenderer(BaseRenderer):
    style = "waveform"
    BUCKETS = 120
    BUCKET_SECONDS = 0.06

    def __init__(self):
        super().__init__()
        self.phase = 0.0

    def reset(self):
        super().reset()
        self.phase = 0.0

    def draw_backdrop(self, target, frame, s, dt):
        """Three flowing translucent waves along the bottom, one per band group."""
        w, h = target.get_size()
        lv = self.levels(frame, s, dt)
        if dt > 0.0015:
            self.phase += dt * (0.6 + 1.2 * frame.volume)
        bass, mid, treble = self.bands3(lv)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        n = 90
        xs = np.linspace(0, w, n)
        u = xs / w
        for k, (band, col, base, freq, speed) in enumerate(
                ((treble, c2, 0.80, 3.1, 1.3), (mid, lerp_color(c1, c2, 0.5), 0.84, 2.2, -0.9),
                 (bass, c1, 0.88, 1.4, 0.6))):
            amp = h * (0.025 + 0.09 * band)
            ys = h * base - amp * (np.sin(TAU * freq * u + self.phase * speed + k)
                                   * (0.7 + 0.3 * np.sin(TAU * u * 0.7 - self.phase * 0.5)))
            pts = [(0, h)] + list(zip(xs.tolist(), ys.tolist())) + [(w, h)]
            gfx.filled_polygon(target, pts, (*col, 60))
            pygame.draw.aalines(target, lerp_color(col, WHITE, 0.2), False, pts[1:-1])

    def draw(self, layer, frame, s, dt):
        w, h = layer.get_size()
        u = min(w, h)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        n, bs = self.BUCKETS, self.BUCKET_SECONDS
        spacing = w / (n - 4)
        pos = frame.time / bs
        first = int(math.floor(pos)) - n // 2 - 1
        env = (self.analyzer.envelope_buckets(first, n + 2, bs) if self.analyzer is not None
               else np.zeros(n + 2, np.float32))
        cy = h * 0.42
        amp = h * 0.25
        bw = max(2, int(spacing * 0.55))
        cx = w / 2
        dim = scale_color(c2, 0.55)
        for j, e in enumerate(env.tolist()):
            x = cx + (first + j + 0.5 - pos) * spacing
            if x < -bw or x > w + bw:
                continue
            hh = max(bw, amp * e * (1 + 0.06 * frame.beat))
            played = x < cx
            col = lerp_color(c1, WHITE, 0.15 * frame.beat) if played else dim
            pygame.draw.rect(layer, col, (int(x - bw / 2), int(cy - hh), bw, int(hh * 2)), border_radius=bw // 2)
        # playhead
        aa_line(layer, WHITE, (cx, cy - amp * 1.12), (cx, cy + amp * 1.12), max(1.5, u / 400))
        aa_circle(layer, WHITE, (cx, cy - amp * 1.12), max(3, u * 0.008))
        # the live oscilloscope trace across the middle
        wave = resample(frame.waveform, 200)
        xs = np.linspace(w * 0.05, w * 0.95, len(wave))
        ys = cy + wave * amp * 0.35
        pygame.draw.aalines(layer, lerp_color(c1, WHITE, 0.5), False, list(zip(xs.tolist(), ys.tolist())))


# ------------------------------------------------------------------ Radial

class RadialRenderer(BaseRenderer):
    style = "circular"
    BARS = 64   # per side; the ring is mirrored left/right

    def __init__(self):
        super().__init__()
        self.spin = 0.0
        self.orbit = [0.0, 2.0, 4.0]

    def reset(self):
        super().reset()
        self.spin = 0.0
        self.orbit = [0.0, 2.0, 4.0]

    def draw_backdrop(self, target, frame, s, dt):
        w, h = target.get_size()
        u = min(w, h)
        cx, cy = w / 2, h / 2
        c1 = s.primary_rgb
        R = u * 0.17 * (1 + 0.08 * frame.bass)
        aa_circle(target, (*c1, 26), (cx, cy), R * 0.98)
        # shockwave rings spreading out after each beat
        if self.analyzer is not None:
            for age in self.analyzer.recent_beats(frame.time, 1.6).tolist():
                rr = R + age * u * 0.32
                a = int(110 * (1 - age / 1.6))
                if a > 4:
                    gfx.aacircle(target, int(cx), int(cy), int(rr), (*lerp_color(c1, WHITE, 0.4), a))

    def draw(self, layer, frame, s, dt):
        w, h = layer.get_size()
        u = min(w, h)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        lv = self.levels(frame, s, dt)
        bass, mid, treble = self.bands3(lv)
        if dt > 0.0015:
            self.spin += dt * (0.08 + 0.5 * frame.volume)
            for i, band in enumerate((bass, mid, treble)):
                self.orbit[i] += dt * (0.5 + 0.4 * i) * (0.4 + 2.5 * band) * (1 if i % 2 == 0 else -1)
        cx, cy = w / 2, h / 2
        R = u * 0.17 * (1 + 0.08 * bass)
        bars = resample(lv, self.BARS)
        width = max(2.0, R * TAU / (self.BARS * 2) * 0.55)
        for side in (-1, 1):
            for i, v in enumerate(bars.tolist()):
                a = -math.pi / 2 + side * (i + 0.5) / self.BARS * math.pi
                L = u * (0.012 + 0.22 * v)
                ca, sa = math.cos(a), math.sin(a)
                col = lerp_color(c1, c2, i / (self.BARS - 1))
                r0 = R * 1.08
                aa_line(layer, col, (cx + ca * r0, cy + sa * r0), (cx + ca * (r0 + L), cy + sa * (r0 + L)), width)
        # core
        gfx.aacircle(layer, int(cx), int(cy), int(R), (*lerp_color(c1, WHITE, 0.3), 255))
        for d in range(24):   # slowly turning dotted ring inside the bars
            a = self.spin + d * TAU / 24
            aa_circle(layer, lerp_color(c1, c2, 0.5), (cx + math.cos(a) * R * 0.86, cy + math.sin(a) * R * 0.86),
                      max(1.5, u * 0.003))
        aa_circle(layer, lerp_color(c1, c2, 0.5), (cx, cy), R * (0.30 + 0.25 * bass + 0.1 * frame.beat))
        aa_circle(layer, lerp_color(c1, WHITE, 0.6), (cx, cy), R * (0.12 + 0.1 * frame.beat))
        # orbiting dots
        for i, (ang, band) in enumerate(zip(self.orbit, (bass, mid, treble))):
            rr = R * (0.55 + 0.14 * i)
            x, y = cx + math.cos(ang) * rr, cy + math.sin(ang) * rr
            aa_circle(layer, (c1, lerp_color(c1, c2, 0.5), c2)[i], (x, y), u * (0.006 + 0.008 * band))


# ------------------------------------------------------------------- Rings

class RingsRenderer(BaseRenderer):
    style = "meters"
    START, SWEEP = math.radians(135), math.radians(270)

    def __init__(self):
        super().__init__()
        self.springs = [Spring(), Spring(), Spring()]
        self.spin = [0.0, 0.0, 0.0]

    def reset(self):
        super().reset()
        self.springs = [Spring(), Spring(), Spring()]
        self.spin = [0.0, 0.0, 0.0]

    @staticmethod
    def _geometry(w, h):
        u = min(w, h)
        return (w / 2, h / 2), [u * 0.40, u * 0.31, u * 0.22], max(4.0, u * 0.035)

    def draw_backdrop(self, target, frame, s, dt):
        w, h = target.get_size()
        c, radii, width = self._geometry(w, h)
        for r in radii:   # dim tracks behind the gauges
            thick_arc(target, (*WHITE, 22), c, r, width, self.START, self.START + self.SWEEP)

    def draw(self, layer, frame, s, dt):
        w, h = layer.get_size()
        u = min(w, h)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        lv = self.levels(frame, s, dt)
        bands = self.bands3(lv)
        (cx, cy), radii, width = self._geometry(w, h)
        for i, (r, band) in enumerate(zip(radii, bands)):
            col = lerp_color(c1, c2, i / 2)
            v = clamp01(self.springs[i].update(clamp01(band * 1.15), dt))
            thick_arc(layer, col, (cx, cy), r, width, self.START, self.START + self.SWEEP * max(0.01, v))
            tip = self.START + self.SWEEP * v
            aa_circle(layer, lerp_color(col, WHITE, 0.6), (cx + math.cos(tip) * r, cy + math.sin(tip) * r),
                      width * 0.32)
            # spinning dash ring just inside each gauge
            if dt > 0.0015:
                self.spin[i] += dt * (0.2 + 3.0 * band) * (1 if i % 2 == 0 else -1)
            rd = r - width * 1.15
            dashes = 36
            dcol = scale_color(col, 0.45 + 0.55 * band)
            for d in range(dashes):
                a = self.spin[i] + d * TAU / dashes
                a0, a1 = a, a + TAU / dashes * 0.45
                aa_line(layer, dcol, (cx + math.cos(a0) * rd, cy + math.sin(a0) * rd),
                        (cx + math.cos(a1) * rd, cy + math.sin(a1) * rd), max(1.5, width * 0.18))
        # centre: beat pulse
        pr = radii[-1] * (0.28 + 0.18 * frame.beat + 0.1 * frame.volume)
        aa_circle(layer, lerp_color(c1, c2, 0.5), (cx, cy), pr)
        aa_circle(layer, lerp_color(c1, WHITE, 0.7), (cx, cy), pr * 0.45)


# --------------------------------------------------------------- Particles

class ParticlesRenderer(BaseRenderer):
    style = "particles"
    COUNT = 420

    def __init__(self, seed: int = 1234):
        super().__init__()
        self.rng = np.random.default_rng(seed)
        self.reset()

    def reset(self):
        super().reset()
        rng = np.random.default_rng(1234)
        n = self.COUNT
        self.x = rng.random(n)
        self.y = rng.random(n)
        self.vx = np.zeros(n)
        self.vy = np.zeros(n)
        self.hue = rng.random(n)
        self.size = rng.random(n) ** 2
        self.t = 0.0

    def draw(self, layer, frame, s, dt):
        w, h = layer.get_size()
        u = min(w, h)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        lv = self.levels(frame, s, dt)
        bass, mid, treble = self.bands3(lv)
        aspect = w / h
        if dt > 0.0015:
            self.t += dt
            t = self.t
            # Velocity = curl of a slowly moving potential (worked out in screen-shaped
            # units): a swirl with no sinks, so the particles stay evenly spread.
            px, py = self.x * aspect * TAU + t * 0.15, self.y * TAU - t * 0.1
            fx = np.cos(px) * np.cos(py) + 0.5 * np.cos(py * 2 + t * 0.3)
            fy = np.sin(px) * np.sin(py) - 0.5 * np.cos(px * 2 - t * 0.2)
            speed = 0.03 + 0.16 * frame.volume + 0.08 * mid
            self.vx += (fx * speed / aspect - self.vx) * min(1.0, dt * 3)
            self.vy += (fy * speed - self.vy) * min(1.0, dt * 3)
            self.x = (self.x + self.vx * dt) % 1.0
            self.y = (self.y + self.vy * dt) % 1.0
        # beats make the whole field "breathe out" from the centre for a moment
        dx, dy = (self.x - 0.5) * aspect, self.y - 0.5
        push = 1 + 0.10 * frame.beat * (0.5 + bass)
        xs = (0.5 + dx * push / aspect) * w
        ys = (0.5 + dy * push) * h
        tail = 0.08 + 0.1 * frame.volume
        base_r = u * 0.0032
        bright = 0.55 + 0.45 * treble
        for x, y, vx, vy, hue, sz in zip(xs.tolist(), ys.tolist(), self.vx.tolist(), self.vy.tolist(),
                                          self.hue.tolist(), self.size.tolist()):
            col = scale_color(lerp_color(c1, c2, hue), bright + 0.3 * sz)
            r = base_r * (1 + 2.2 * sz) * (1 + 0.5 * bass)
            ex, ey = x - vx * w * tail, y - vy * h * tail
            if sz > 0.25 and abs(ex - x) + abs(ey - y) > 3 and abs(ex - x) < w / 2 and abs(ey - y) < h / 2:
                pygame.draw.aaline(layer, col, (ex, ey), (x, y))
            aa_circle(layer, col, (x, y), max(2.0, r))


RENDERERS = {cls.style: cls for cls in (SpectrumRenderer, WaveformRenderer, RadialRenderer,
                                        RingsRenderer, ParticlesRenderer)}


def create_renderer(style: str) -> BaseRenderer:
    try:
        return RENDERERS[style]()
    except KeyError:
        raise ValueError(f"Unknown visualizer style: {style}") from None

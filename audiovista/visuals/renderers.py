"""Visualizer styles - clean and modern, built like small machines with moving parts.

* bars      - Spectrum: mirrored piston bars (bass in the middle) riding in slots,
              sliding peak markers, and a gear train under the floor that the
              bass drives.
* waveform  - Waveform: the real waveform runs like a film strip between two
              sprocket gears, past a scanner head whose needle follows the
              signal; the sprocket holes and ruler ticks move with the music.
* circular  - Radial: a spectrum ring mounted on a planetary gearbox - sun gear,
              three planets and a ring gear that really mesh - with shockwave
              rings on the beat and optional swirling particles.

``draw_backdrop`` paints the machinery straight onto the frame (translucent
parts blend correctly there and stay out of the glow); ``draw`` paints the
bright, music-driven parts onto a layer that the engine blooms. All motion
depends only on the analysis frame and ``dt``, so exports match the preview.
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
STEEL = (210, 214, 236)


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


def gear_points(cx, cy, r, teeth: int, angle: float, depth: float = 0.16):
    """Outline of a spur gear with *teeth* trapezoid teeth, pitch radius *r*."""
    pts = []
    ro, ri = r * (1 + depth / 2), r * (1 - depth / 2)
    step = TAU / teeth
    for i in range(teeth):
        a = angle + i * step
        for f, rr in ((0.0, ri), (0.18, ro), (0.50, ro), (0.68, ri)):
            pts.append((cx + math.cos(a + f * step) * rr, cy + math.sin(a + f * step) * rr))
    return pts


def gear(surf, cx, cy, r, teeth, angle, body, hub, spokes: int = 5) -> None:
    """A gear with a hub and round lightening holes that turn with it."""
    aa_poly(surf, body, gear_points(cx, cy, r, teeth, angle))
    if r > 10 and spokes:
        for k in range(spokes):
            a = angle + k * TAU / spokes
            aa_circle(surf, hub, (cx + math.cos(a) * r * 0.55, cy + math.sin(a) * r * 0.55), r * 0.17)
    aa_circle(surf, hub, (cx, cy), r * 0.24)
    aa_circle(surf, body, (cx, cy), r * 0.11)


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
        """Optional: machinery drawn straight onto the frame, under the glow layer."""

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
        self.peaks: np.ndarray | None = None
        self.peak_v: np.ndarray | None = None
        self.drive = 0.0

    def reset(self):
        super().reset()
        self.peaks = None
        self.drive = 0.0

    def _geometry(self, w, h):
        n = self.PER_SIDE * 2
        step = w * 0.86 / n
        bw = max(3, int(step * 0.58))
        floor = h * 0.66
        max_h = h * 0.50
        xs = [w / 2 + (i + 0.5) * step * side for side in (-1, 1) for i in range(self.PER_SIDE)]
        return xs, bw, floor, max_h

    def draw_backdrop(self, target, frame, s, dt):
        w, h = target.get_size()
        u = min(w, h)
        lv = resample(self.levels(frame, s, dt), self.PER_SIDE)
        bass = float(lv[: self.PER_SIDE // 4].mean())
        if self.peaks is None:
            self.peaks, self.peak_v = lv.copy(), np.zeros_like(lv)
        elif dt > 0.0015:
            self.drive += dt * (0.4 + 5.0 * bass + 1.5 * frame.beat)
            self.peak_v += dt * 1.6                       # markers fall back with gravity
            self.peaks = self.peaks - self.peak_v * dt
            hit = lv >= self.peaks
            self.peaks[hit], self.peak_v[hit] = lv[hit], 0.0
        xs, bw, floor, max_h = self._geometry(w, h)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        # the slots the pistons ride in, with a sliding peak marker on each
        for i, x in enumerate(xs):
            j = i % self.PER_SIDE
            slot = pygame.Rect(int(x - bw / 2 - 2), int(floor - max_h - bw), bw + 4, int(max_h + bw))
            gfx.box(target, slot, (*WHITE, 16))
            py = floor - max_h * (0.03 + float(self.peaks[j])) - bw
            gfx.box(target, (slot.x - 2, int(py), slot.w + 4, max(2, bw // 3)), (*lerp_color(c2, WHITE, 0.4), 150))
        # deck plate and the gear train underneath, driven by the bass
        gfx.box(target, (int(w * 0.05), int(floor + 2), int(w * 0.90), max(3, int(u * 0.012))), (*STEEL, 70))
        gy = floor + u * 0.012 + (h - floor) * 0.42
        body, hub = (*lerp_color(c1, STEEL, 0.45), 120), (*scale_color(c1, 0.35), 160)
        pitch = u * 0.012                                  # tooth spacing shared by all gears
        x, prev_r, sign, phase = w * 0.06, None, 1, 0.0
        radii = [u * f for f in (0.075, 0.045, 0.075, 0.045, 0.060, 0.045, 0.075, 0.045, 0.075)]
        k = 0
        while x < w * 1.05:
            r = radii[k % len(radii)]
            teeth = max(8, int(TAU * r / pitch))
            if prev_r is not None:
                x += prev_r + r
            ang = sign * self.drive * (u * 0.06) / r + phase
            gear(target, x, gy, r, teeth, ang, body, hub)
            sign, prev_r, phase, k = -sign, r, phase + 0.5 * TAU / teeth, k + 1

    def draw(self, layer, frame, s, dt):
        w, h = layer.get_size()
        lv = resample(self.levels(frame, s, dt), self.PER_SIDE)
        xs, bw, floor, max_h = self._geometry(w, h)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        r = bw // 2
        rod = max(2, bw // 4)
        for i, x in enumerate(xs):
            j = i % self.PER_SIDE
            v = float(lv[j])
            top = floor - max_h * (0.03 + v)
            col = lerp_color(c1, c2, (j / (self.PER_SIDE - 1)) ** 1.3)
            head_h = max(bw, int(bw * 1.4))
            # piston rod from the deck up to the head, then the head itself
            pygame.draw.rect(layer, scale_color(col, 0.75), (int(x - rod / 2), int(top), rod, int(floor - top)))
            pygame.draw.rect(layer, col, (int(x - bw / 2), int(top), bw, int(min(head_h + (floor - top) * 0.55,
                                                                                 floor - top + 1))),
                             border_radius=r)
            pygame.draw.rect(layer, lerp_color(col, WHITE, 0.7), (int(x - bw / 2), int(top), bw, max(2, bw // 3)),
                             border_radius=max(1, r // 2))


# --------------------------------------------------------------- Waveform

class WaveformRenderer(BaseRenderer):
    style = "waveform"
    BUCKETS = 120
    BUCKET_SECONDS = 0.06

    def __init__(self):
        super().__init__()
        self.needle = 0.0

    def reset(self):
        super().reset()
        self.needle = 0.0

    def _geometry(self, w, h):
        u = min(w, h)
        cy = h * 0.46
        amp = h * 0.24
        spacing = w / (self.BUCKETS - 4)
        return u, cy, amp, spacing

    def draw_backdrop(self, target, frame, s, dt):
        """The film-strip transport: rails with sprocket holes, sprocket gears, ruler."""
        w, h = target.get_size()
        u, cy, amp, spacing = self._geometry(w, h)
        c1 = s.primary_rgb
        scroll = frame.time / self.BUCKET_SECONDS * spacing     # px the strip has moved
        rail_h = max(6, int(u * 0.035))
        steel = (*STEEL, 60)
        hole_col = lerp_color(c1, WHITE, 0.5)
        pitch = u * 0.05
        hole_w, hole_h = int(pitch * 0.45), max(3, int(rail_h * 0.45))
        for ry in (cy - amp * 1.18 - rail_h, cy + amp * 1.18):
            gfx.box(target, (0, int(ry), w, rail_h), steel)
            x = -(scroll % pitch)
            while x < w:
                gfx.box(target, (int(x), int(ry + (rail_h - hole_h) / 2), hole_w, hole_h), (*hole_col, 120))
                x += pitch
        # sprocket gears at both ends, turning exactly with the strip
        gr = amp * 1.18 + rail_h * 0.5
        teeth = max(10, int(TAU * gr / pitch))
        ang = scroll / gr
        body, hub = (*lerp_color(c1, STEEL, 0.5), 110), (*scale_color(c1, 0.35), 150)
        for gx in (0.0, float(w)):
            gear(target, gx, cy, gr * 0.9, teeth, -ang, body, hub, spokes=6)
        # ruler under the strip: one tick per bucket, a long one every second
        ry = cy + amp * 1.18 + rail_h + u * 0.03
        first = int((frame.time - w / 2 / spacing * self.BUCKET_SECONDS) / self.BUCKET_SECONDS) - 1
        per_sec = max(1, int(round(1 / self.BUCKET_SECONDS)))
        for j in range(first, first + self.BUCKETS + 4):
            x = w / 2 + (j * self.BUCKET_SECONDS - frame.time) / self.BUCKET_SECONDS * spacing
            if 0 <= x <= w and j >= 0:
                long = j % per_sec == 0
                gfx.vline(target, int(x), int(ry), int(ry + u * (0.03 if long else 0.012)),
                          (*STEEL, 110 if long else 50))

    def draw(self, layer, frame, s, dt):
        w, h = layer.get_size()
        u, cy, amp, spacing = self._geometry(w, h)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        n, bs = self.BUCKETS, self.BUCKET_SECONDS
        pos = frame.time / bs
        first = int(math.floor(pos)) - n // 2 - 1
        env = (self.analyzer.envelope_buckets(first, n + 2, bs) if self.analyzer is not None
               else np.zeros(n + 2, np.float32))
        bw = max(2, int(spacing * 0.55))
        cx = w / 2
        dim = scale_color(c2, 0.55)
        now = 0.0
        for j, e in enumerate(env.tolist()):
            x = cx + (first + j + 0.5 - pos) * spacing
            if x < -bw or x > w + bw:
                continue
            hh = max(bw, amp * e)
            if abs(x - cx) < spacing:
                now = max(now, e)
            col = c1 if x < cx else dim
            pygame.draw.rect(layer, col, (int(x - bw / 2), int(cy - hh), bw, int(hh * 2)), border_radius=bw // 2)
        # scanner head: a carriage on the top rail, a spinning drive gear and a
        # pair of needles that close in on the current level
        if dt > 0.0015:
            self.needle += (now - self.needle) * (1 - 0.05 ** dt)
        top = cy - amp * 1.18
        head_w = u * 0.05
        pygame.draw.rect(layer, STEEL, (int(cx - head_w / 2), int(top - u * 0.065), int(head_w), int(u * 0.045)),
                         border_radius=int(u * 0.008))
        gear(layer, cx, top - u * 0.088, u * 0.026, 10, frame.time * 4.0, lerp_color(c1, STEEL, 0.4),
             scale_color(c1, 0.4), spokes=0)
        aa_line(layer, WHITE, (cx, top - u * 0.02), (cx, cy + amp * 1.18), max(1.5, u / 420))
        ny = amp * max(0.02, self.needle)
        for sgn in (-1, 1):
            y = cy - sgn * ny
            aa_poly(layer, WHITE, [(cx - u * 0.016, y - sgn * u * 0.022), (cx + u * 0.016, y - sgn * u * 0.022),
                                   (cx, y)])


# ------------------------------------------------------------------ Radial

class ParticleField:
    """Particles swirling around the centre through a smooth flow field."""

    COUNT = 360

    def __init__(self):
        rng = np.random.default_rng(1234)
        n = self.COUNT
        self.x, self.y = rng.random(n), rng.random(n)
        self.vx, self.vy = np.zeros(n), np.zeros(n)
        self.hue = rng.random(n)
        self.size = rng.random(n) ** 2
        self.t = 0.0

    def update(self, frame, mid, aspect, dt):
        if dt <= 0.0015:
            return
        self.t += dt
        t = self.t
        # curl of a moving potential (no sinks, so the particles stay spread out)
        # plus a gentle orbit around the centre
        px, py = self.x * aspect * TAU + t * 0.15, self.y * TAU - t * 0.1
        fx = np.cos(px) * np.cos(py) + 0.5 * np.cos(py * 2 + t * 0.3)
        fy = np.sin(px) * np.sin(py) - 0.5 * np.cos(px * 2 - t * 0.2)
        dx, dy = (self.x - 0.5) * aspect, self.y - 0.5
        fx, fy = fx * 0.6 - dy * 1.6, fy * 0.6 + dx * 1.6
        speed = 0.03 + 0.14 * frame.volume + 0.06 * mid
        self.vx += (fx * speed / aspect - self.vx) * min(1.0, dt * 3)
        self.vy += (fy * speed - self.vy) * min(1.0, dt * 3)
        self.x = (self.x + self.vx * dt) % 1.0
        self.y = (self.y + self.vy * dt) % 1.0

    def draw(self, layer, frame, c1, c2, bass, treble, u):
        w, h = layer.get_size()
        aspect = w / h
        dx, dy = (self.x - 0.5) * aspect, self.y - 0.5
        push = 1 + 0.08 * frame.beat * (0.5 + bass)     # beats make the swarm breathe out
        xs = (0.5 + dx * push / aspect) * w
        ys = (0.5 + dy * push) * h
        lift = 0.1 + 0.3 * treble
        for x, y, hue, sz in zip(xs.tolist(), ys.tolist(), self.hue.tolist(), self.size.tolist()):
            col = lerp_color(lerp_color(c1, c2, hue), WHITE, lift + 0.3 * sz)   # never darker than the sky
            aa_circle(layer, col, (x, y), max(2.0, u * 0.0028 * (1 + 2 * sz) * (1 + 0.4 * bass)))


class RadialRenderer(BaseRenderer):
    style = "circular"
    BARS = 56           # per side; the ring is mirrored left/right
    # planetary gearbox: sun, planet and ring tooth counts (ring = sun + 2 * planet)
    Z_SUN, Z_PLANET, Z_RING = 14, 8, 30

    def __init__(self):
        super().__init__()
        self.sun = 0.0
        self.housing = 0.0
        self.particles = ParticleField()

    def reset(self):
        super().reset()
        self.sun = 0.0
        self.housing = 0.0
        self.particles = ParticleField()

    @staticmethod
    def _R(u):
        return u * 0.19

    def draw_backdrop(self, target, frame, s, dt):
        w, h = target.get_size()
        u = min(w, h)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        lv = self.levels(frame, s, dt)
        bass, mid, treble = self.bands3(lv)
        if dt > 0.0015:
            self.sun += dt * (0.5 + 6.0 * bass + 2.0 * frame.beat)
            self.housing -= dt * (0.05 + 0.25 * treble)
        cx, cy = w / 2, h / 2
        R = self._R(u)
        # shockwave rings spreading out after each beat
        if self.analyzer is not None:
            for age in self.analyzer.recent_beats(frame.time, 1.6).tolist():
                a = int(110 * (1 - age / 1.6))
                if a > 4:
                    gfx.aacircle(target, int(cx), int(cy), int(R * 1.1 + age * u * 0.3),
                                 (*lerp_color(c1, WHITE, 0.4), a))
        # Planetary gearbox. With the ring held, the carrier turns at
        # sun * Zs / (Zs + Zr) and each planet spins backwards against the ring.
        m = R * 0.80 / self.Z_RING                  # gear module (size per tooth)
        zs, zp, zr = self.Z_SUN, self.Z_PLANET, self.Z_RING
        g = self.housing                            # the whole gearbox turns slowly
        carrier = self.sun * zs / (zs + zr)
        planet = carrier - carrier * zr / zp
        body = (*lerp_color(c1, STEEL, 0.45), 130)
        hub = (*scale_color(c1, 0.3), 170)
        # ring gear: a band with teeth pointing inwards
        r_in, r_out = zr * m, R * 0.97
        for k in range(zr):
            a = g + k * TAU / zr
            pts = []
            for f, rr in ((-0.32, r_in * 1.01), (-0.16, r_in * 0.92), (0.16, r_in * 0.92), (0.32, r_in * 1.01)):
                aa = a + f * TAU / zr
                pts.append((cx + math.cos(aa) * rr, cy + math.sin(aa) * rr))
            aa_poly(target, body, pts)
        outer = [(cx + math.cos(a) * r_out, cy + math.sin(a) * r_out) for a in np.linspace(0, TAU, 96)]
        inner = [(cx + math.cos(a) * r_in, cy + math.sin(a) * r_in) for a in np.linspace(TAU, 0, 96)]
        aa_poly(target, body, outer + inner)
        orbit = (zs + zp) * m
        for i in range(3):
            a = g + carrier + i * TAU / 3
            gear(target, cx + math.cos(a) * orbit, cy + math.sin(a) * orbit, zp * m, zp, g + planet + i * 0.4,
                 body, hub, spokes=3)
        gear(target, cx, cy, zs * m, zs, g + self.sun, (*lerp_color(c2, STEEL, 0.4), 150), hub, spokes=4)

    def draw(self, layer, frame, s, dt):
        w, h = layer.get_size()
        u = min(w, h)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        lv = self.levels(frame, s, dt)
        bass, mid, treble = self.bands3(lv)
        cx, cy = w / 2, h / 2
        R = self._R(u)
        if s.particles:
            self.particles.update(frame, mid, w / h, dt)
            self.particles.draw(layer, frame, c1, c2, bass, treble, u)
        bars = resample(lv, self.BARS)
        width = max(2.0, R * TAU / (self.BARS * 2) * 0.55)
        r0 = R * 1.08
        for side in (-1, 1):
            for i, v in enumerate(bars.tolist()):
                a = -math.pi / 2 + side * (i + 0.5) / self.BARS * math.pi
                L = u * (0.012 + 0.21 * v)
                ca, sa = math.cos(a), math.sin(a)
                col = lerp_color(c1, c2, i / (self.BARS - 1))
                aa_line(layer, col, (cx + ca * r0, cy + sa * r0), (cx + ca * (r0 + L), cy + sa * (r0 + L)), width)
        gfx.aacircle(layer, int(cx), int(cy), int(R), (*lerp_color(c1, WHITE, 0.3), 255))
        # glowing cap on the sun gear's axle, pulsing with the beat
        aa_circle(layer, lerp_color(c1, WHITE, 0.6), (cx, cy), R * (0.07 + 0.05 * frame.beat))


RENDERERS = {cls.style: cls for cls in (SpectrumRenderer, WaveformRenderer, RadialRenderer)}


def create_renderer(style: str) -> BaseRenderer:
    try:
        return RENDERERS[style]()
    except KeyError:
        raise ValueError(f"Unknown visualizer style: {style}") from None

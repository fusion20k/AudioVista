"""Visualizer styles, designed like vintage hi-fi equipment with lots of moving parts.

* bars      - Equalizer: LED-ladder spectrum analyzer with peak-hold lights,
              signal lamps and rotary knobs.
* waveform  - Tape Deck: reel-to-reel machine; reels spin (and fill/empty as the
              song plays), the real waveform runs along the tape past the head,
              spring-loaded VU needles and a rolling tape counter.
* circular  - Turntable: spinning record and label, strobe dots, grooves that glow
              with the spectrum and a tonearm that tracks inward over the song.
* meters    - VU Meters: two big analog meters and bass/mid/treble gauges with
              springy needles, driven by meshing clockwork gears.
* particles - Embers: glowing sparks rising over spectrum-shaped dunes at sunset.

Every renderer draws onto a transparent layer; the engine adds the background,
glow and vintage film effects. All motion is driven by the analysis frame and
``dt`` only, so exported videos match the live preview.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import pygame
import pygame.gfxdraw as gfx

from ..audio.analyzer import AnalysisFrame
from ..models import VisualSettings

# Equipment colours
METAL = (184, 172, 152)
METAL_LIGHT = (226, 216, 196)
METAL_DARK = (96, 86, 74)
PLATE = (36, 28, 22)
PLATE_EDGE = (78, 62, 48)
CREAM = (242, 228, 200)
INK = (44, 32, 24)
RED = (206, 70, 48)
TAPE = (88, 56, 34)
WHITE = (255, 248, 232)


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


def aa_circle(surf, color, center, radius: float, width: int = 0) -> None:
    x, y, r = int(center[0]), int(center[1]), int(radius)
    if r < 1:
        return
    if width:
        for k in range(max(1, int(width))):
            if r - k >= 1:
                gfx.aacircle(surf, x, y, r - k, color)
    else:
        gfx.filled_circle(surf, x, y, r, color)
        gfx.aacircle(surf, x, y, r, color)


def aa_poly(surf, color, pts) -> None:
    gfx.filled_polygon(surf, pts, color)
    gfx.aapolygon(surf, pts, color)


def blend_poly(surf, color, pts) -> None:
    """Translucent polygon that blends correctly onto a per-pixel-alpha layer."""
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    x0, y0 = int(min(xs)) - 1, int(min(ys)) - 1
    tmp = pygame.Surface((int(max(xs)) - x0 + 2, int(max(ys)) - y0 + 2), pygame.SRCALPHA)
    aa_poly(tmp, color, [(x - x0, y - y0) for x, y in pts])
    surf.blit(tmp, (x0, y0))


_glows: dict = {}


def glow(surf, color, center, radius: float, strength: float = 1.0) -> None:
    """Soft round glow (cached sprite, blended properly)."""
    r = max(2, int(radius))
    key = (r, tuple(color[:3]))
    spr = _glows.get(key)
    if spr is None:
        size = 2 * r
        xx, yy = np.mgrid[0:size, 0:size].astype(np.float32)
        d = np.sqrt((xx - r + 0.5) ** 2 + (yy - r + 0.5) ** 2) / r
        spr = pygame.Surface((size, size), pygame.SRCALPHA)
        spr.fill((*color[:3], 0))
        pygame.surfarray.pixels_alpha(spr)[:] = (np.clip(1 - d, 0, 1) ** 2 * 255).astype(np.uint8)
        if len(_glows) > 200:
            _glows.clear()
        _glows[key] = spr
    spr.set_alpha(int(255 * clamp01(strength)))
    surf.blit(spr, (int(center[0]) - r, int(center[1]) - r))


def mix(base, color, alpha: float):
    """Opaque colour equal to *color* at *alpha* over *base* (avoids alpha holes)."""
    return lerp_color(base, color, alpha)


_fonts: dict = {}
_texts: OrderedDict = OrderedDict()


def text(surf, s: str, pos, size: int, color, anchor: str = "center", serif: bool = False) -> None:
    """Cached label text in an equipment-style typeface."""
    size = max(6, int(size))
    key = (s, size, tuple(color), serif)
    img = _texts.get(key)
    if img is None:
        fkey = (size, serif)
        if fkey not in _fonts:
            if not pygame.font.get_init():
                pygame.font.init()
            name = "georgia,segoeui,arial" if serif else "bahnschrift,segoeui,arial"
            _fonts[fkey] = pygame.font.SysFont(name, size, bold=serif)
        img = _fonts[fkey].render(s, True, color[:3])
        if len(color) == 4:
            img.set_alpha(color[3])
        _texts[key] = img
        if len(_texts) > 600:
            _texts.popitem(last=False)
    surf.blit(img, img.get_rect(**{anchor: (int(pos[0]), int(pos[1]))}))


class Spring:
    """Damped spring, used for needles so they swing and overshoot like real meters."""

    def __init__(self, stiffness: float = 170.0, damping: float = 13.0):
        self.k, self.c = stiffness, damping
        self.x = 0.0
        self.v = 0.0

    def update(self, target: float, dt: float) -> float:
        if dt <= 0.0015:          # paused: needles freeze
            return self.x
        steps = max(1, int(dt / 0.004))
        h = dt / steps
        for _ in range(steps):
            a = self.k * (target - self.x) - self.c * self.v
            self.v += a * h
            self.x += self.v * h
        return self.x


def faceplate(surf, rect, u: float, alpha: int = 235) -> None:
    r = pygame.Rect(rect)
    rad = max(4, int(u * 0.03))
    pygame.draw.rect(surf, (*PLATE, alpha), r, border_radius=rad)
    pygame.draw.rect(surf, (*PLATE_EDGE, 255), r, max(1, int(u / 320)), border_radius=rad)
    inset = int(u * 0.02)
    for x, y, a in ((r.x + inset, r.y + inset, 0.4), (r.right - inset, r.y + inset, 1.3),
                    (r.x + inset, r.bottom - inset, 2.1), (r.right - inset, r.bottom - inset, 0.9)):
        screw(surf, x, y, u * 0.011, a)


def screw(surf, x, y, r, angle) -> None:
    aa_circle(surf, METAL_DARK, (x, y), r)
    aa_circle(surf, METAL, (x, y), r * 0.82)
    dx, dy = math.cos(angle) * r * 0.7, math.sin(angle) * r * 0.7
    aa_line(surf, INK, (x - dx, y - dy), (x + dx, y + dy), max(1.0, r * 0.25))


def knob(surf, cx, cy, r, value: float, accent, label: str = "", u: float = 600) -> None:
    """Rotary knob with knurled edge, scale ticks and a pointer at *value* (0..1)."""
    for i in range(11):                                      # scale ticks
        a = math.radians(135 + 27 * i)
        aa_line(surf, mix(PLATE, CREAM, 0.6), (cx + math.cos(a) * r * 1.18, cy + math.sin(a) * r * 1.18),
                (cx + math.cos(a) * r * 1.32, cy + math.sin(a) * r * 1.32), max(1.0, r / 14))
    aa_circle(surf, (24, 19, 15), (cx, cy), r * 1.06)
    aa_circle(surf, (58, 48, 40), (cx, cy), r)
    angle = math.radians(135 + 270 * clamp01(value))
    for i in range(28):                                      # knurling turns with the knob
        a = angle + 2 * math.pi * i / 28
        aa_line(surf, (82, 70, 58), (cx + math.cos(a) * r * 0.78, cy + math.sin(a) * r * 0.78),
                (cx + math.cos(a) * r * 0.98, cy + math.sin(a) * r * 0.98), max(1.0, r / 18))
    aa_circle(surf, METAL, (cx, cy), r * 0.68)
    aa_circle(surf, METAL_LIGHT, (cx - r * 0.12, cy - r * 0.12), r * 0.42)
    aa_circle(surf, METAL, (cx, cy), r * 0.42)
    aa_line(surf, accent, (cx + math.cos(angle) * r * 0.2, cy + math.sin(angle) * r * 0.2),
            (cx + math.cos(angle) * r * 0.9, cy + math.sin(angle) * r * 0.9), max(2.0, r / 6))
    if label:
        text(surf, label, (cx, cy + r * 1.62), max(7, int(u * 0.024)), (*CREAM, 200))


def lamp(surf, cx, cy, r, on: float, color, label: str, u: float) -> None:
    aa_circle(surf, (20, 16, 13), (cx, cy), r * 1.25)
    if on > 0.05:
        glow(surf, color, (cx, cy), r * 3.2, 0.55 * on)
    bulb = lerp_color(scale_color(color, 0.25), color, on)
    aa_circle(surf, bulb, (cx, cy), r)
    aa_circle(surf, mix(bulb, WHITE, 0.55 * on), (cx - r * 0.3, cy - r * 0.3), r * 0.35)
    text(surf, label, (cx, cy + r * 2.4), max(7, int(u * 0.022)), (*CREAM, 190))


def gear_points(cx, cy, r, teeth: int, angle: float, depth: float = 0.13):
    pts = []
    step = 2 * math.pi / teeth
    for i in range(teeth):
        a0 = angle + i * step
        for frac, rad in ((0.0, r * (1 - depth)), (0.14, r), (0.38, r), (0.52, r * (1 - depth))):
            a = a0 + step * frac
            pts.append((cx + math.cos(a) * rad, cy + math.sin(a) * rad))
    return pts


def gear(surf, cx, cy, r, teeth, angle, color, alpha=255) -> None:
    aa_poly(surf, (*color, alpha), gear_points(cx, cy, r, teeth, angle))
    dark = scale_color(color, 0.45)
    for i in range(5):                                       # lightening holes
        a = angle + 2 * math.pi * i / 5
        aa_circle(surf, (*dark, alpha), (cx + math.cos(a) * r * 0.55, cy + math.sin(a) * r * 0.55), r * 0.17)
    aa_circle(surf, (*scale_color(color, 1.15), alpha), (cx, cy), r * 0.24)
    aa_circle(surf, (*dark, alpha), (cx, cy), r * 0.1)


# ------------------------------------------------------------ gauge faces

_faces: dict = {}


def _vu_curve(db: float) -> float:
    """Position (0..1) of a dB value on a classic VU scale (-20 .. +3)."""
    lo, hi = 10 ** (-20 / 20), 10 ** (3 / 20)
    return (10 ** (db / 20) - lo) / (hi - lo)


VU_SWING = math.radians(50)


def vu_face(w: int, h: int, label: str) -> pygame.Surface:
    """Cream, backlit VU meter face (cached)."""
    key = ("vu", w, h, label)
    face = _faces.get(key)
    if face is not None:
        return face
    xx, yy = np.mgrid[0:w, 0:h].astype(np.float32)
    d = np.sqrt(((xx - w / 2) / w) ** 2 + ((yy - h * 0.75) / h) ** 2)
    glow = np.clip(1.0 - d * 1.3, 0, 1)[..., None]
    rgb = np.array((206, 178, 128), np.float32) + glow * np.array((50, 58, 70), np.float32)
    face = pygame.Surface((w, h), pygame.SRCALPHA)
    face.blit(pygame.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8)), (0, 0))
    mask = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=max(3, int(h * 0.06)))
    face.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)

    px, py = w / 2, h * 1.12               # pivot below the face
    ra = h * 0.92
    lw = max(1.0, h / 120)

    def at(pos, rad):
        a = -VU_SWING + 2 * VU_SWING * pos - math.pi / 2
        return px + math.cos(a) * rad, py + math.sin(a) * rad

    for pos0, pos1, col, wid in ((0.0, _vu_curve(0), INK, lw), (_vu_curve(0), 1.0, RED, lw * 3)):
        pts = [at(pos0 + (pos1 - pos0) * k / 40, ra) for k in range(41)]
        for a, b in zip(pts, pts[1:]):
            aa_line(face, col, a, b, wid)
    fs = max(7, int(h * 0.085))
    for db in (-20, -10, -7, -5, -3, -2, -1, 0, 1, 2, 3):
        pos = _vu_curve(db)
        col = RED if db > 0 else INK
        aa_line(face, col, at(pos, ra), at(pos, ra + h * 0.07), max(1.0, h / 90))
        text(face, f"+{db}" if db > 0 else str(db), at(pos, ra + h * 0.15), fs, col)
    for k in range(21):                                    # minor ticks
        aa_line(face, (*INK, 150), at(k / 20, ra), at(k / 20, ra + h * 0.035), 1)
    text(face, "VU", (w / 2, h * 0.62), int(h * 0.17), INK, serif=True)
    text(face, label, (w * 0.08, h * 0.80), max(7, int(h * 0.065)), (*INK, 180), anchor="midleft")
    text(face, "AUDIOVISTA", (w * 0.92, h * 0.80), max(7, int(h * 0.055)), (*INK, 150), anchor="midright")
    blend_poly(face, (255, 255, 255, 22), [(0, 0), (w * 0.42, 0), (w * 0.22, h), (0, h)])   # glass glare
    _faces[key] = face
    return face


def draw_vu(surf, rect, position: float, label: str, peak: float) -> None:
    rect = pygame.Rect(rect)
    w, h = rect.size
    bezel = max(2, int(h * 0.05))
    pygame.draw.rect(surf, (18, 14, 11), rect.inflate(bezel * 2, bezel * 2), border_radius=int(h * 0.09))
    pygame.draw.rect(surf, PLATE_EDGE, rect.inflate(bezel * 2, bezel * 2), max(1, bezel // 3),
                     border_radius=int(h * 0.09))
    surf.blit(vu_face(w, h, label), rect)
    px, py = rect.x + w / 2, rect.y + h * 1.12
    a = -VU_SWING + 2 * VU_SWING * min(max(position, -0.04), 1.06) - math.pi / 2
    tip = (px + math.cos(a) * h * 1.02, py + math.sin(a) * h * 1.02)
    base = (px + math.cos(a) * h * 0.30, py + math.sin(a) * h * 0.30)
    clip = surf.get_clip()
    surf.set_clip(rect)
    aa_line(surf, (150, 128, 94), (base[0] + h * 0.03, base[1] + h * 0.02), (tip[0] + h * 0.03, tip[1] + h * 0.02),
            max(1.5, h / 70))   # needle shadow, pre-mixed with the face colour
    aa_line(surf, (30, 22, 18), base, tip, max(1.5, h / 70))
    surf.set_clip(clip)
    pygame.draw.rect(surf, (24, 19, 15), (rect.x, rect.bottom - h * 0.09, w, h * 0.09),
                     border_bottom_left_radius=int(h * 0.06), border_bottom_right_radius=int(h * 0.06))
    lamp_on = clamp01((peak - 0.85) / 0.15)
    aa_circle(surf, lerp_color((70, 22, 16), (255, 80, 50), lamp_on), (rect.right - h * 0.12, rect.y + h * 0.13),
              h * 0.045)


def gauge_face(r: int, label: str) -> pygame.Surface:
    key = ("gauge", r, label)
    face = _faces.get(key)
    if face is not None:
        return face
    size = 2 * r + 2
    face = pygame.Surface((size, size), pygame.SRCALPHA)
    c = (r + 1, r + 1)
    aa_circle(face, (20, 16, 13), c, r)
    aa_circle(face, METAL_DARK, c, r * 0.96)
    aa_circle(face, (232, 214, 178), c, r * 0.90)
    for i in range(51):
        a = math.radians(135 + 270 * i / 50)
        long = i % 5 == 0
        col = RED if i > 40 else INK
        r0 = r * (0.70 if long else 0.76)
        aa_line(face, col, (c[0] + math.cos(a) * r0, c[1] + math.sin(a) * r0),
                (c[0] + math.cos(a) * r * 0.82, c[1] + math.sin(a) * r * 0.82), max(1.0, r / (40 if long else 90)))
        if long:
            text(face, str(i // 5), (c[0] + math.cos(a) * r * 0.58, c[1] + math.sin(a) * r * 0.58),
                 max(6, int(r * 0.15)), col)
    text(face, label, (c[0], c[1] + r * 0.42), max(6, int(r * 0.16)), INK, serif=True)
    blend_poly(face, (255, 255, 255, 24), [(c[0] - r * 0.7, c[1] - r * 0.7), (c[0] + r * 0.1, c[1] - r * 0.88),
                                           (c[0] - r * 0.5, c[1] - r * 0.1)])
    _faces[key] = face
    return face


def draw_gauge(surf, cx, cy, r, value: float, label: str, accent) -> None:
    face = gauge_face(int(r), label)
    surf.blit(face, face.get_rect(center=(int(cx), int(cy))))
    a = math.radians(135 + 270 * min(max(value, -0.02), 1.03))
    tip = (cx + math.cos(a) * r * 0.80, cy + math.sin(a) * r * 0.80)
    tail = (cx - math.cos(a) * r * 0.18, cy - math.sin(a) * r * 0.18)
    aa_line(surf, (*scale_color(accent, 0.8), 255), tail, tip, max(1.5, r / 28))
    aa_circle(surf, METAL_DARK, (cx, cy), r * 0.09)
    aa_circle(surf, METAL, (cx, cy), r * 0.06)


# ------------------------------------------------------------ base class

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
        """Optional: draw opaque scenery straight onto the frame, under the layer
        (cheaper than the layer and left out of the glow pass)."""

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


# ------------------------------------------------------------- Equalizer

class BarsRenderer(BaseRenderer):
    """Graphic spectrum analyzer: LED ladders, falling peak lights, lamps and knobs."""

    style = "bars"
    SEGMENTS = 18

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self._peaks = None
        self._peak_vel = None
        self._knobs = [Spring(60, 11) for _ in range(3)]

    def draw(self, layer, frame, settings, dt):
        w, h = layer.get_size()
        u = min(w, h)
        c1, c2 = settings.primary_rgb, settings.secondary_rgb
        hot = lerp_color(RED, c1, 0.25)
        levels = self.levels(frame, settings, dt)

        plate = pygame.Rect(int(w * 0.04), int(h * 0.07), int(w * 0.92), int(h * 0.86))
        faceplate(layer, plate, u)
        text(layer, "AUDIOVISTA", (plate.x + u * 0.06, plate.y + u * 0.055), u * 0.034, c1,
             anchor="midleft", serif=True)
        text(layer, "GRAPHIC SPECTRUM ANALYZER", (plate.x + u * 0.06, plate.y + u * 0.095), u * 0.021,
             (*CREAM, 170), anchor="midleft")

        peak_now = float(levels.max())
        bass, mid, treble = self.bands3(levels)
        lr = u * 0.011
        for i, (lab, on, col) in enumerate((("BASS", clamp01((bass - 0.45) * 3), c1),
                                             ("BEAT", frame.beat, c2), ("PEAK", clamp01((peak_now - 0.9) * 10), RED))):
            lamp(layer, plate.right - u * (0.07 + 0.075 * (2 - i)), plate.y + u * 0.06, lr, on, col, lab, u)

        area = pygame.Rect(int(plate.x + plate.w * 0.09), int(plate.y + plate.h * 0.20),
                           int(plate.w * 0.84), int(plate.h * 0.52))
        pygame.draw.rect(layer, (14, 11, 9, 255), area.inflate(int(u * 0.02), int(u * 0.02)),
                         border_radius=int(u * 0.01))
        cols = int(min(40, max(14, area.w / (u * 0.034))))
        segs = self.SEGMENTS
        vals = resample(levels, cols)
        if self._peaks is None or len(self._peaks) != cols:
            self._peaks = vals.copy()
            self._peak_vel = np.zeros(cols)
        if dt > 0.0015:
            rising = vals >= self._peaks
            self._peak_vel = np.where(rising, 0.0, self._peak_vel + 1.4 * dt)
            self._peaks = np.where(rising, vals, np.maximum(vals, self._peaks - self._peak_vel * dt))

        col_w = area.w / cols
        seg_h = area.h / segs
        gx, gy = max(1, int(col_w * 0.14)), max(1, int(seg_h * 0.16))
        seg_cols = []
        for s in range(segs):
            k = s / (segs - 1)
            seg_cols.append(lerp_color(c2, c1, k / 0.72) if k < 0.72 else lerp_color(c1, hot, (k - 0.72) / 0.28))
        for c in range(cols):
            x = area.x + c * col_w
            lit = int(round(vals[c] * segs))
            pk = min(segs - 1, int(self._peaks[c] * segs))
            for s in range(segs):
                y = area.bottom - (s + 1) * seg_h
                col = seg_cols[s]
                if s < lit:
                    color = col
                elif s == pk and pk > 0:
                    color = lerp_color(col, WHITE, 0.35)
                else:
                    color = scale_color(col, 0.13)
                pygame.draw.rect(layer, color, (int(x + gx), int(y + gy), int(col_w - 2 * gx), int(seg_h - 2 * gy)))

        fs = max(7, int(u * 0.02))
        for db_i, lab in enumerate(("0", "-6", "-12", "-18", "-24", "-30")):
            y = area.y + area.h * db_i / 5
            text(layer, lab, (area.x - u * 0.02, y), fs, (*CREAM, 150), anchor="midright")
        if self.analyzer is not None:
            freqs = np.exp(resample(np.log(self.analyzer.band_frequencies()), cols))
            step = max(1, cols // 8)
            for c in range(0, cols, step):
                f = freqs[c]
                lab = f"{f / 1000:.1f}K".replace(".0K", "K") if f >= 1000 else f"{int(round(f, -1 if f > 100 else 0))}"
                text(layer, lab, (area.x + (c + 0.5) * col_w, area.bottom + u * 0.03), fs, (*CREAM, 150))
        text(layer, "Hz", (area.right + u * 0.012, area.bottom + u * 0.03), fs, (*CREAM, 120), anchor="midleft")

        values = (frame.volume, bass, treble)
        names = ("LEVEL", "BASS", "TREBLE")
        kr = u * 0.035
        ky = plate.bottom - plate.h * 0.115
        for i in range(3):
            v = self._knobs[i].update(values[i], dt)
            knob(layer, plate.x + plate.w * (0.30 + 0.2 * i), ky, kr, v, c1, names[i], u)


# ------------------------------------------------------------- Tape deck

class WaveformRenderer(BaseRenderer):
    """Reel-to-reel tape deck. The real waveform is printed on the moving tape."""

    style = "waveform"
    BAR_SECONDS = 0.06

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self._ang = [0.0, 0.0, 0.0]    # left reel, right reel, rollers
        self._vu = [Spring(), Spring()]

    def _reel(self, surf, cx, cy, R, pack, hub, angle, accent):
        aa_circle(surf, (20, 15, 12, 220), (cx, cy), R * 1.01)
        aa_circle(surf, TAPE, (cx, cy), pack)
        for k in (0.35, 0.6, 0.82):
            rr = hub + (pack - hub) * k
            if rr > hub + 2:
                aa_circle(surf, (106, 70, 44), (cx, cy), rr, width=1)
        aa_circle(surf, (120, 82, 52), (cx, cy), pack, width=1)
        for i in range(3):
            a = angle + i * 2 * math.pi / 3
            pts = [(cx + math.cos(a - 0.30) * hub * 1.2, cy + math.sin(a - 0.30) * hub * 1.2),
                   (cx + math.cos(a - 0.33) * R * 0.96, cy + math.sin(a - 0.33) * R * 0.96),
                   (cx + math.cos(a + 0.33) * R * 0.96, cy + math.sin(a + 0.33) * R * 0.96),
                   (cx + math.cos(a + 0.30) * hub * 1.2, cy + math.sin(a + 0.30) * hub * 1.2)]
            aa_poly(surf, (*METAL, 235), pts)
            hole = (cx + math.cos(a) * R * 0.62, cy + math.sin(a) * R * 0.62)
            aa_circle(surf, (*METAL_DARK, 235), hole, R * 0.07)
        aa_circle(surf, METAL_LIGHT, (cx, cy), R, width=max(2, int(R / 30)))
        aa_circle(surf, METAL, (cx, cy), hub * 1.25)
        aa_circle(surf, (40, 32, 26), (cx, cy), hub * 0.55)
        for i in range(3):
            a = angle + i * 2 * math.pi / 3 + math.pi / 3
            aa_line(surf, METAL, (cx + math.cos(a) * hub * 0.2, cy + math.sin(a) * hub * 0.2),
                    (cx + math.cos(a) * hub * 0.55, cy + math.sin(a) * hub * 0.55), max(2.0, hub * 0.18))
        aa_circle(surf, (*accent, 255), (cx, cy), hub * 0.15)

    def _roller(self, surf, cx, cy, r, angle, rubber=False):
        aa_circle(surf, (24, 20, 17) if rubber else METAL_DARK, (cx, cy), r)
        aa_circle(surf, (44, 38, 33) if rubber else METAL, (cx, cy), r * 0.78)
        for i in range(3):
            a = angle + i * 2 * math.pi / 3
            aa_line(surf, METAL_DARK, (cx, cy), (cx + math.cos(a) * r * 0.7, cy + math.sin(a) * r * 0.7),
                    max(1.0, r / 7))
        aa_circle(surf, METAL_LIGHT, (cx, cy), r * 0.2)

    def _counter(self, surf, x, y, digit_w, digit_h, value: float):
        digits = 4
        pygame.draw.rect(surf, (12, 10, 8), (x - digit_w * 0.2, y - digit_h * 0.2,
                                             digit_w * (digits + 0.4), digit_h * 1.4), border_radius=int(digit_h * 0.15))
        fs = int(digit_h * 0.82)
        clip = surf.get_clip()
        for i in range(digits):
            power = 10 ** (digits - 1 - i)
            v = value / power
            d = int(v) % 10
            if power == 1:
                roll = v - int(v)
            else:
                roll = clamp01((value % power) - (power - 1))   # rolls over during the carry
            cell = pygame.Rect(int(x + i * digit_w), int(y), int(digit_w * 0.9), int(digit_h))
            pygame.draw.rect(surf, (236, 224, 200), cell, border_radius=max(1, int(digit_h * 0.08)))
            surf.set_clip(cell.clip(clip) if clip else cell)
            off = roll * digit_h
            text(surf, str(d), (cell.centerx, cell.centery - off), fs, INK)
            text(surf, str((d + 1) % 10), (cell.centerx, cell.centery - off + digit_h), fs, INK)
            surf.set_clip(clip)
            pygame.draw.rect(surf, (0, 0, 0, 70), (cell.x, cell.y, cell.w, cell.h * 0.18))
            pygame.draw.rect(surf, (0, 0, 0, 70), (cell.x, cell.bottom - cell.h * 0.18, cell.w, cell.h * 0.18))

    def draw(self, layer, frame, settings, dt):
        w, h = layer.get_size()
        u = min(w, h)
        c1, c2 = settings.primary_rgb, settings.secondary_rgb
        playing = dt > 0.0015
        levels = self.levels(frame, settings, dt)
        dur = self.analyzer.duration if self.analyzer is not None else 1.0
        p = clamp01(frame.time / max(dur, 1e-6))

        plate = pygame.Rect(int(w * 0.03), int(h * 0.05), int(w * 0.94), int(h * 0.90))
        faceplate(layer, plate, u)

        R = min(w * 0.165, h * 0.25)
        cy = plate.y + R + h * 0.06
        lx, rx = w * 0.235, w * 0.765
        hub, full = R * 0.26, R * 0.93
        pack_l = math.sqrt(hub ** 2 + (full ** 2 - hub ** 2) * (1 - p))
        pack_r = math.sqrt(hub ** 2 + (full ** 2 - hub ** 2) * p)
        tape_speed = R * 2.2
        if playing:   # constant tape speed: the emptier reel spins faster
            self._ang[0] += tape_speed / pack_l * dt
            self._ang[1] += tape_speed / pack_r * dt

        strip = pygame.Rect(int(w * 0.10), int(plate.y + plate.h * 0.68), int(w * 0.80), int(h * 0.13))
        roller_r = strip.h * 0.42
        if playing:
            self._ang[2] += tape_speed / max(roller_r, 1) * dt

        tape_w = max(2.0, u / 120)
        aa_line(layer, TAPE, (lx - pack_l, cy), (strip.x - roller_r, strip.centery), tape_w)
        aa_line(layer, TAPE, (rx + pack_r, cy), (strip.right + roller_r, strip.centery), tape_w)

        self._reel(layer, lx, cy, R, pack_l, hub, self._ang[0], c1)
        self._reel(layer, rx, cy, R, pack_r, hub, self._ang[1], c1)

        gap_l, gap_r = lx + R + u * 0.03, rx - R - u * 0.03
        mw = max(20, gap_r - gap_l)
        mh = min(mw * 0.56, R * 0.85)
        bass, mid, treble = self.bands3(levels)
        targets = (clamp01(0.55 * bass + 0.45 * mid) * 1.05 * (0.5 + 0.5 * frame.volume) + 0.05,
                   clamp01(0.5 * mid + 0.5 * treble * 1.4) * 1.05 * (0.5 + 0.5 * frame.volume) + 0.05)
        below = strip.y - (cy + R) - u * 0.05           # free space under the reels
        side_by_side = below > mh * 1.15                # squarer canvases: meters go there, bigger
        if side_by_side:
            mh = min(below * 0.86, w * 0.36 * 0.56)
            mw = mh / 0.56
        for i, lab in enumerate(("LEFT", "RIGHT")):
            pos = self._vu[i].update(targets[i], dt)
            if side_by_side:
                x = w / 2 - mw - u * 0.03 if i == 0 else w / 2 + u * 0.03
                y = cy + R + (strip.y - (cy + R) - mh) / 2 - u * 0.02
                rect = pygame.Rect(int(x), int(y), int(mw), int(mh))
            else:
                rect = pygame.Rect(int(gap_l), int(cy - mh - u * 0.012 + i * (mh + u * 0.024)), int(mw), int(mh))
            draw_vu(layer, rect, pos, lab, targets[i])

        band = strip.inflate(0, -int(strip.h * 0.36))
        pygame.draw.rect(layer, TAPE, band)
        pygame.draw.line(layer, (126, 86, 54), band.topleft, band.topright, max(1, int(u / 400)))
        pygame.draw.line(layer, (52, 32, 20), band.bottomleft, band.bottomright, max(1, int(u / 400)))
        cx = strip.centerx
        if self.analyzer is not None:
            spacing = max(3.0, u / 95)
            bar_w = max(1, int(spacing * 0.5))
            d = self.BAR_SECONDS
            half = int((strip.w / 2) / spacing) + 2
            first = int(math.floor(frame.time / d)) - half
            env = self.analyzer.envelope_buckets(first, 2 * half + 2, d) ** (0.8 / settings.sensitivity)
            clip = layer.get_clip()
            layer.set_clip(band)
            for i, v in enumerate(env):
                j = first + i
                x = cx - (j * d - frame.time) / d * spacing     # tape runs left -> right
                bh = max(2, v * band.h * 0.92)
                played = (j + 1) * d <= frame.time
                col = lerp_color(c1, WHITE, 0.15) if played else scale_color(lerp_color(c2, TAPE, 0.35), 0.9)
                pygame.draw.rect(layer, col, (int(x - bar_w / 2), int(band.centery - bh / 2), bar_w, int(bh)))
            layer.set_clip(clip)
        for x in (strip.x, strip.right):
            self._roller(layer, x, strip.centery, roller_r, self._ang[2])

        head = pygame.Rect(0, 0, int(u * 0.06), int(strip.h * 1.05))
        head.center = (cx, strip.centery)
        pygame.draw.rect(layer, METAL, head, border_radius=int(u * 0.008))
        pygame.draw.rect(layer, METAL_LIGHT, head.inflate(-head.w * 0.5, -head.h * 0.3), border_radius=int(u * 0.005))
        gap_col = lerp_color((60, 40, 30), c1, 0.4 + 0.6 * frame.beat) if playing else (60, 40, 30)
        pygame.draw.rect(layer, gap_col, (cx - max(1, u / 600), head.y + head.h * 0.15, max(2, u / 300), head.h * 0.7))
        text(layer, "PLAYBACK", (cx, head.bottom + u * 0.022), u * 0.018, (*CREAM, 170))
        self._roller(layer, cx + u * 0.10, strip.centery, strip.h * 0.20, -self._ang[2] * 2.2)
        self._roller(layer, cx + u * 0.10, strip.y - strip.h * 0.18, strip.h * 0.24, self._ang[2] * 1.8, rubber=True)

        dh = u * 0.05
        self._counter(layer, strip.x, strip.bottom + u * 0.045, dh * 0.7, dh, frame.time)
        text(layer, "TAPE COUNTER", (strip.x, strip.bottom + u * 0.03), u * 0.017, (*CREAM, 150), anchor="midleft")
        kw, kh = u * 0.075, u * 0.05
        kx = strip.right - 4 * (kw + u * 0.012)
        ky = strip.bottom + u * 0.045
        for i, (name, active) in enumerate((("rew", False), ("play", playing), ("ff", False), ("stop", not playing))):
            down = u * 0.006 if active else 0
            key = pygame.Rect(int(kx + i * (kw + u * 0.012)), int(ky + down), int(kw), int(kh))
            pygame.draw.rect(layer, (16, 12, 10), key.move(0, int(u * 0.006 - down)), border_radius=int(u * 0.008))
            pygame.draw.rect(layer, lerp_color(METAL, c1, 0.55) if active else METAL, key, border_radius=int(u * 0.008))
            ic, s = key.center, kh * 0.22
            if name == "play":
                aa_poly(layer, INK, [(ic[0] - s, ic[1] - s), (ic[0] - s, ic[1] + s), (ic[0] + s, ic[1])])
            elif name == "stop":
                pygame.draw.rect(layer, INK, (ic[0] - s * 0.9, ic[1] - s * 0.9, s * 1.8, s * 1.8))
            else:
                dsign = 1 if name == "ff" else -1
                for off in (-0.55, 0.55):
                    ox = ic[0] + off * s
                    aa_poly(layer, INK, [(ox - dsign * s * 0.6, ic[1] - s * 0.8), (ox - dsign * s * 0.6, ic[1] + s * 0.8),
                                         (ox + dsign * s * 0.6, ic[1])])


# ------------------------------------------------------------- Turntable

_records: dict = {}
_wood: dict = {}


def wood_surface(size: tuple[int, int]) -> pygame.Surface:
    """Walnut wood-grain plinth texture (cached)."""
    surf = _wood.get(size)
    if surf is None:
        w, h = size
        xx, yy = np.mgrid[0:w, 0:h].astype(np.float32)
        grain = (np.sin(yy / h * 90 + np.sin(xx / w * 7) * 2.5 + np.sin(xx / w * 31) * 0.4) * 0.5 + 0.5)
        fine = np.sin(yy / h * 520 + xx / w * 3) * 0.5 + 0.5
        k = (0.65 * grain + 0.35 * fine)[..., None]
        dark = np.array((58, 34, 20), np.float32)
        light = np.array((96, 60, 36), np.float32)
        rgb = dark + (light - dark) * k
        surf = pygame.Surface(size, pygame.SRCALPHA)
        surf.blit(pygame.surfarray.make_surface(rgb.astype(np.uint8)), (0, 0))
        mask = pygame.Surface(size, pygame.SRCALPHA)
        pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=int(min(w, h) * 0.04))
        surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        if len(_wood) > 6:
            _wood.clear()
        _wood[size] = surf
    return surf


def record_surface(radius: int) -> pygame.Surface:
    """Black vinyl with grooves and a fixed light sheen (cached)."""
    surf = _records.get(radius)
    if surf is None:
        size = 2 * radius + 2
        xx, yy = np.mgrid[0:size, 0:size].astype(np.float32)
        dx, dy = xx - size / 2, yy - size / 2
        d = np.sqrt(dx ** 2 + dy ** 2) / radius
        theta = np.arctan2(dy, dx)
        grooves = (np.sin(d * radius * 1.4) * 0.5 + 0.5) * ((d > 0.36) & (d < 0.95))
        sheen = (np.abs(np.cos(theta - 0.8)) ** 14) * ((d > 0.36) & (d < 0.97)) * 52
        v = 16 + grooves * 9 + sheen
        rgb = np.stack([v + 2, v, v - 1], axis=-1)
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        surf.blit(pygame.surfarray.make_surface(np.clip(rgb, 0, 255).astype(np.uint8)), (0, 0))
        pygame.surfarray.pixels_alpha(surf)[:] = (np.clip((1.0 - d) * radius, 0, 1) * 255).astype(np.uint8)
        if len(_records) > 8:
            _records.clear()
        _records[radius] = surf
    return surf


def label_surface(radius: int, c1, c2, title: str) -> pygame.Surface:
    key = ("label", radius, c1, c2, title)
    surf = _records.get(key)
    if surf is None:
        size = 2 * radius + 2
        surf = pygame.Surface((size, size), pygame.SRCALPHA)
        c = (radius + 1, radius + 1)
        aa_circle(surf, scale_color(c1, 0.85), c, radius)
        aa_circle(surf, c1, c, radius * 0.94)
        aa_circle(surf, (*CREAM, 200), c, radius * 0.84, width=max(1, radius // 60))
        name = title if len(title) <= 26 else title[:25] + "…"
        text(surf, "AUDIOVISTA", (c[0], c[1] - radius * 0.58), radius * 0.13, INK, serif=True)
        text(surf, name, (c[0], c[1] - radius * 0.32), radius * 0.12, INK)
        text(surf, "33 1/3 RPM", (c[0], c[1] + radius * 0.38), radius * 0.11, INK)
        text(surf, "SIDE A  ·  STEREO", (c[0], c[1] + radius * 0.58), radius * 0.08, (*INK, 200))
        aa_circle(surf, (*c2, 255), (c[0] - radius * 0.45, c[1] + radius * 0.05), radius * 0.05)
        aa_circle(surf, (*c2, 255), (c[0] + radius * 0.45, c[1] + radius * 0.05), radius * 0.05)
        _records[key] = surf
    return surf


class CircularRenderer(BaseRenderer):
    """Turntable: record, label, strobe dots, glowing grooves and a tracking tonearm."""

    style = "circular"
    RPM = 33.333

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self._angle = 0.0
        self._fader = Spring(40, 9)
        self._bob = Spring(260, 9)

    @staticmethod
    def _geometry(w, h):
        plinth = pygame.Rect(int(w * 0.03), int(h * 0.05), int(w * 0.94), int(h * 0.90))
        P = min(w * 0.31, h * 0.355)
        return plinth, P, plinth.x + plinth.w * 0.42, plinth.centery

    def draw_backdrop(self, target, frame, settings, dt):
        w, h = target.get_size()
        u = min(w, h)
        plinth, P, cx, cy = self._geometry(w, h)
        target.blit(wood_surface(plinth.size), plinth)
        pygame.draw.rect(target, (30, 18, 10), plinth, max(1, int(u / 250)), border_radius=int(u * 0.04))
        aa_circle(target, (22, 18, 15), (cx, cy), P * 1.01)
        aa_circle(target, METAL, (cx, cy), P)
        aa_circle(target, METAL_DARK, (cx, cy), P * 0.955)
        rec = record_surface(int(P * 0.94))
        target.blit(rec, rec.get_rect(center=(int(cx), int(cy))))

    def draw(self, layer, frame, settings, dt):
        w, h = layer.get_size()
        u = min(w, h)
        c1, c2 = settings.primary_rgb, settings.secondary_rgb
        playing = dt > 0.0015
        levels = self.levels(frame, settings, dt)
        dur = self.analyzer.duration if self.analyzer is not None else 1.0
        p = clamp01(frame.time / max(dur, 1e-6))
        if playing:
            self._angle += dt * 2 * math.pi * self.RPM / 60
        plinth, P, cx, cy = self._geometry(w, h)

        # spectrum light ring around the platter
        n = 120
        half = resample(levels, n // 2)
        ring = np.concatenate([half, half[::-1]])
        for i, v in enumerate(ring):
            a = 2 * math.pi * i / n - math.pi / 2
            r0, r1 = P * 1.035, P * 1.035 + (0.006 + v * 0.07) * u
            col = mix((70, 42, 25), lerp_color(c2, c1, v), 0.5 + 0.5 * v)
            aa_line(layer, col, (cx + math.cos(a) * r0, cy + math.sin(a) * r0),
                    (cx + math.cos(a) * r1, cy + math.sin(a) * r1), max(1.5, P * 2 * math.pi / n * 0.45))

        # rotating strobe dots on the platter rim
        dots = 90
        for i in range(dots):
            a = self._angle + 2 * math.pi * i / dots
            col = METAL_LIGHT if i % 2 == 0 else (150, 140, 124)
            aa_circle(layer, col, (cx + math.cos(a) * P * 0.977, cy + math.sin(a) * P * 0.977), max(1.0, P * 0.009))

        rec_r = int(P * 0.94)

        # grooves light up with the spectrum: bass near the label, treble at the edge
        rings = 26
        vals = resample(levels, rings)
        for i, v in enumerate(vals):
            if v < 0.08:
                continue
            rr = rec_r * (0.40 + 0.53 * i / (rings - 1))
            col = lerp_color(c1, c2, i / (rings - 1))
            aa_circle(layer, mix((24, 21, 19), col, 0.8 * v), (cx, cy), rr, width=1)

        lab_r = int(P * 0.33)
        label = label_surface(lab_r, c1, c2, self.title or "Now Playing")
        rot = pygame.transform.rotozoom(label, -math.degrees(self._angle), 1.0)
        layer.blit(rot, rot.get_rect(center=(int(cx), int(cy))))
        aa_circle(layer, METAL_LIGHT, (cx, cy), P * 0.025)
        aa_circle(layer, METAL_DARK, (cx, cy), P * 0.012)

        # tonearm: needle radius moves inward as the song plays
        pivot = (cx + P * 1.22, cy - P * 0.80)
        bob = self._bob.update(frame.bass * 0.004 * P, dt)
        r_needle = P * (0.88 - 0.50 * p)
        arm_len = P * 1.15
        dx, dy = pivot[0] - cx, pivot[1] - cy
        dist = math.hypot(dx, dy)
        a = (r_needle ** 2 - arm_len ** 2 + dist ** 2) / (2 * dist)
        hh = math.sqrt(max(r_needle ** 2 - a ** 2, 0))
        bx, by = cx + a * dx / dist, cy + a * dy / dist
        cands = [(bx + hh * dy / dist, by - hh * dx / dist), (bx - hh * dy / dist, by + hh * dx / dist)]
        needle = max(cands, key=lambda q: q[1])
        needle = (needle[0], needle[1] + bob)
        ux, uy = (needle[0] - pivot[0]) / arm_len, (needle[1] - pivot[1]) / arm_len
        elbow = (pivot[0] + ux * arm_len * 0.82, pivot[1] + uy * arm_len * 0.82)
        tube = max(2.0, P * 0.028)
        blend_poly(layer, (0, 0, 0, 70), [(pivot[0] + P * 0.03, pivot[1] + P * 0.05),
                                          (needle[0] + P * 0.03, needle[1] + P * 0.05),
                                          (needle[0] + P * 0.05, needle[1] + P * 0.07),
                                          (pivot[0] + P * 0.05, pivot[1] + P * 0.07)])   # arm shadow
        cw = (pivot[0] - ux * P * 0.30, pivot[1] - uy * P * 0.30)
        aa_line(layer, METAL_DARK, pivot, cw, tube * 1.3)
        aa_circle(layer, (40, 34, 30), cw, P * 0.075)
        aa_circle(layer, METAL_DARK, cw, P * 0.075, width=max(1, int(P / 120)))
        aa_line(layer, METAL_LIGHT, pivot, elbow, tube)
        ha = math.atan2(uy, ux) + 0.35
        hs_end = (elbow[0] + math.cos(ha) * P * 0.21, elbow[1] + math.sin(ha) * P * 0.21)
        aa_line(layer, (48, 40, 34), elbow, hs_end, tube * 2.0)
        aa_circle(layer, c1, hs_end, tube * 0.6)
        aa_circle(layer, METAL_DARK, pivot, P * 0.11)
        aa_circle(layer, METAL, pivot, P * 0.085)
        aa_circle(layer, METAL_LIGHT, pivot, P * 0.04)
        rest = (pivot[0] - P * 0.02, pivot[1] + P * 0.42)
        aa_circle(layer, METAL_DARK, rest, P * 0.035)

        # controls: start/stop, 33/45, gain fader, strobe lamp
        bx, by = plinth.x + u * 0.08, plinth.bottom - u * 0.08
        aa_circle(layer, (16, 12, 10), (bx, by), u * 0.04)
        aa_circle(layer, METAL if not playing else lerp_color(METAL, c1, 0.5), (bx, by), u * 0.033)
        text(layer, "START·STOP", (bx, by + u * 0.065), u * 0.018, (*CREAM, 190))
        for i, (lab, on) in enumerate((("33", True), ("45", False))):
            x = bx + u * (0.09 + 0.065 * i)
            pygame.draw.rect(layer, (16, 12, 10), (x - u * 0.025, by - u * 0.017, u * 0.05, u * 0.034),
                             border_radius=int(u * 0.006))
            text(layer, lab, (x, by), u * 0.022, c1 if on else (*CREAM, 120))
        lamp(layer, plinth.x + u * 0.08, plinth.y + u * 0.08, u * 0.012, 0.35 + 0.65 * frame.beat, c1, "STROBE", u)

        fx = plinth.right - u * 0.07
        f_top, f_bot = plinth.bottom - u * 0.42, plinth.bottom - u * 0.10
        pygame.draw.rect(layer, (14, 10, 8), (fx - u * 0.006, f_top, u * 0.012, f_bot - f_top), border_radius=int(u * 0.006))
        for k in range(9):
            y = f_top + (f_bot - f_top) * k / 8
            pygame.draw.line(layer, mix((77, 47, 28), CREAM, 0.5), (fx - u * 0.025, y), (fx - u * 0.015, y), 1)
        level = self._fader.update(frame.volume, dt)
        ky = f_bot - (f_bot - f_top) * clamp01(level)
        pygame.draw.rect(layer, (16, 12, 10), (fx - u * 0.022, ky - u * 0.014, u * 0.044, u * 0.03),
                         border_radius=int(u * 0.005))
        pygame.draw.rect(layer, METAL, (fx - u * 0.02, ky - u * 0.016, u * 0.04, u * 0.028), border_radius=int(u * 0.005))
        pygame.draw.line(layer, c1, (fx - u * 0.016, ky - u * 0.002), (fx + u * 0.016, ky - u * 0.002), max(1, int(u / 300)))
        text(layer, "GAIN", (fx, f_bot + u * 0.03), u * 0.018, (*CREAM, 190))


# ------------------------------------------------------------- VU meters

class MetersRenderer(BaseRenderer):
    """Analog VU meters and gauges on a clockwork gear train."""

    style = "meters"

    def __init__(self):
        super().__init__()
        self.reset()

    def reset(self):
        super().reset()
        self._needles = [Spring(150, 11), Spring(150, 11), Spring(110, 9), Spring(110, 9), Spring(110, 9)]
        self._gear_angle = 0.0

    def draw(self, layer, frame, settings, dt):
        w, h = layer.get_size()
        u = min(w, h)
        c1, c2 = settings.primary_rgb, settings.secondary_rgb
        levels = self.levels(frame, settings, dt)
        bass, mid, treble = self.bands3(levels)
        if dt > 0.0015:
            self._gear_angle += dt * (0.25 + 2.4 * frame.volume + 1.5 * frame.beat)

        plate = pygame.Rect(int(w * 0.03), int(h * 0.05), int(w * 0.94), int(h * 0.90))
        faceplate(layer, plate, u)
        text(layer, "AUDIOVISTA", (plate.centerx, plate.y + u * 0.05), u * 0.034, c1, serif=True)
        text(layer, "STEREO LEVEL MONITOR", (plate.centerx, plate.y + u * 0.088), u * 0.019, (*CREAM, 160))

        # gear train behind the gauges (meshing gears: neighbours turn the other way)
        gr_y = plate.bottom - plate.h * 0.24
        g = min(w * 0.085, h * 0.12)
        spacing = min(w * 0.25, plate.w * 0.27)
        gear_r = spacing / 2 * 1.02
        teeth = 22
        for i, gx in enumerate((plate.centerx - spacing, plate.centerx, plate.centerx + spacing)):
            direction = 1 if i % 2 == 0 else -1
            phase = 0 if i % 2 == 0 else math.pi / teeth
            gear(layer, gx, gr_y, gear_r, teeth, direction * self._gear_angle + phase,
                 lerp_color((120, 96, 64), c1, 0.25), 255)
        small_r = gear_r * 0.45
        for sx, sign in ((plate.centerx - spacing / 2, -1), (plate.centerx + spacing / 2, 1)):
            sy = gr_y - gear_r - small_r * 0.82
            gear(layer, sx, sy, small_r, 10, sign * self._gear_angle * gear_r / small_r, (150, 124, 84), 255)

        # two big VU meters
        mw = min(plate.w * 0.42, (gr_y - gear_r - plate.y - u * 0.14) / 0.56)
        mh = mw * 0.56
        top = plate.y + u * 0.13
        targets = (clamp01(0.55 * bass + 0.45 * mid) * (0.45 + 0.6 * frame.volume) + 0.04,
                   clamp01(0.45 * mid + 0.55 * treble * 1.4) * (0.45 + 0.6 * frame.volume) + 0.04)
        for i, lab in enumerate(("LEFT", "RIGHT")):
            pos = self._needles[i].update(targets[i] * 1.05, dt)
            x = plate.centerx - mw - u * 0.02 if i == 0 else plate.centerx + u * 0.02
            draw_vu(layer, pygame.Rect(int(x), int(top), int(mw), int(mh)), pos, lab, targets[i])

        # gauges sit on the gears
        for i, (lab, v) in enumerate((("BASS", bass), ("MID", mid), ("TREBLE", treble))):
            val = self._needles[2 + i].update(clamp01(v * 1.15), dt)
            draw_gauge(layer, plate.centerx + (i - 1) * spacing, gr_y, g, val, lab, c1)

        ly = plate.y + u * 0.06
        for i, (lab, on, col) in enumerate((("BEAT", frame.beat, c2), ("PEAK", clamp01((frame.volume - 0.85) * 7), RED))):
            lamp(layer, plate.right - u * (0.15 - 0.075 * i), ly, u * 0.012, on, col, lab, u)


# ---------------------------------------------------------------- Embers

class ParticlesRenderer(BaseRenderer):
    """Warm embers rising over spectrum-shaped dunes in front of a striped sunset."""

    style = "particles"
    MAX = 900

    def __init__(self, seed: int = 1234):
        super().__init__()
        self.seed = seed
        self.reset()

    def reset(self):
        super().reset()
        self._rng = np.random.default_rng(self.seed)
        self._p = np.zeros((0, 7))   # x, y, vx, vy, life, max_life, size
        self._heat = np.zeros(0)
        self._phase = 0.0
        self._stripe = 0.0
        self._sparks = None

    def _emit(self, count, w, h, speed, u):
        if count <= 0:
            return
        rng = self._rng
        x = rng.uniform(0.05, 0.95, count) * w
        y = h * rng.uniform(0.82, 0.98, count)
        vx = rng.normal(0, 0.02, count) * u
        vy = -rng.uniform(0.06, 0.20, count) * u * speed
        life = rng.uniform(2.0, 4.5, count)
        size = rng.uniform(0.8, 2.6, count) * u / 600
        new = np.stack([x, y, vx, vy, life, life, size], axis=1)
        self._p = np.concatenate([self._p, new])[-self.MAX:]
        self._heat = np.concatenate([self._heat, rng.uniform(0, 1, count)])[-self.MAX:]

    def _sun(self, r, c1, c2, stripe_offset):
        surf = pygame.Surface((2 * r + 2, 2 * r + 2), pygame.SRCALPHA)
        steps = 24
        for i in range(steps):
            k = i / (steps - 1)
            band = pygame.Rect(0, int(k * 2 * r), 2 * r + 2, int(2 * r / steps) + 2)
            pygame.draw.rect(surf, lerp_color(c2, c1, k), band)
        mask = pygame.Surface(surf.get_size(), pygame.SRCALPHA)
        aa_circle(mask, (255, 255, 255, 255), (r + 1, r + 1), r)
        surf.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        gap = max(2, r // 9)
        for k in range(6):                             # retro stripes, drifting down
            y = r * 1.0 + (k + stripe_offset) * r * 0.16
            thick = max(1, int(gap * (0.25 + 0.15 * k)))
            pygame.draw.rect(surf, (0, 0, 0, 0), (0, int(y), 2 * r + 2, thick))
        return surf

    def draw_backdrop(self, target, frame, settings, dt):
        w, h = target.get_size()
        u = min(w, h)
        c1, c2 = settings.primary_rgb, settings.secondary_rgb
        levels = self.levels(frame, settings, dt)
        if dt > 0.0015:
            self._phase += dt * (0.15 + 0.6 * frame.volume)
            self._stripe = (self._stripe + dt * 0.35) % 1.0

        sr = int(u * 0.20 * (1.0 + 0.05 * frame.bass + 0.04 * frame.beat))
        sx, sy = w * 0.5, h * 0.60
        glow(target, c1, (sx, sy), sr * 1.9, 0.30 + 0.25 * frame.beat)
        sun = self._sun(sr, c1, c2, self._stripe)
        target.blit(sun, sun.get_rect(center=(int(sx), int(sy))))

        n = 120
        xs = np.linspace(0, w, n)
        for k in range(3):
            vals = resample(levels, n)
            shift = int(self._phase * (8 + 6 * k) + k * 17) % n
            vals = np.roll(np.convolve(np.concatenate([vals, vals[:8]]), np.ones(9) / 9, mode="same")[:n], shift)
            base = h * (0.74 + 0.07 * k)
            ys = base - vals * h * (0.16 - 0.035 * k) - np.sin(xs / w * (3 + k) * math.pi + k) * h * 0.02
            col = lerp_color(scale_color(c1, 0.28), (14, 9, 7), k / 2.4)
            pts = [(0, h)] + list(zip(xs.tolist(), ys.tolist())) + [(w, h)]
            aa_poly(target, col, pts)
            pygame.draw.aalines(target, lerp_color(col, c2, 0.5 - 0.15 * k), False, pts[1:-1])

    def draw(self, layer, frame, settings, dt):
        w, h = layer.get_size()
        u = min(w, h)
        c1, c2 = settings.primary_rgb, settings.secondary_rgb
        playing = dt > 0.0015

        burst = int(55 * settings.sensitivity) if frame.is_beat and playing else 0
        steady = int((8 + frame.volume * 70) * dt * settings.sensitivity) if playing else 0
        self._emit(steady + burst, w, h, 0.6 + 1.2 * frame.volume, u)
        if len(self._p) and playing:
            p = self._p
            sway = np.sin(p[:, 1] / u * 9 + self._phase * 3 + self._heat * 6) * u * 0.03
            p[:, 0] += (p[:, 2] + sway) * dt
            p[:, 1] += p[:, 3] * dt * (1 + 1.2 * frame.beat)
            p[:, 4] -= dt
            alive = p[:, 4] > 0
            self._p = p[alive]
            self._heat = self._heat[alive]
        sparks = layer   # the layer only holds sparks, so they can be drawn directly
        for (x, y, _vx, _vy, life, mlife, size), heat in zip(self._p, self._heat):
            k = life / mlife
            flick = 0.75 + 0.25 * math.sin(life * 23 + heat * 40)
            col = lerp_color(c1, WHITE if heat > 0.85 else c2, heat)
            alpha = int(255 * min(1.0, k * 1.6) * flick)
            r = max(1.0, size * (0.7 + 0.6 * k))
            aa_circle(sparks, (*col, alpha), (x, y), r)


RENDERERS = {cls.style: cls for cls in (BarsRenderer, WaveformRenderer, CircularRenderer,
                                        MetersRenderer, ParticlesRenderer)}


def create_renderer(style: str) -> BaseRenderer:
    try:
        return RENDERERS[style]()
    except KeyError:
        raise ValueError(f"Unknown visualizer style: {style}") from None

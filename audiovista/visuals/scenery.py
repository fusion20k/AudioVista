"""Building blocks for the landscape visualizers: skies, terrain, clouds, water and light.

The look is "shader-pack landscape at golden hour": soft gradient skies,
layered hills fading into haze, volumetric light rays, drifting clouds,
mirror-like water and fireflies.

Performance notes (everything is software-rendered with pygame):

* skies, clouds, fog bands and glow sprites are generated once and cached;
* terrain is drawn as plain polygons (fast scan-line fills) with anti-aliased
  top edges;
* all soft light (sun haze, god rays, bloom around bright things) is drawn on a
  quarter-resolution "light" surface that the engine blurs up and adds on top,
  so it costs a fraction of drawing it at full size.

Animations are pure functions of time, or state advanced by ``dt``, so
exported videos match the live preview.
"""

from __future__ import annotations

import math
from collections import OrderedDict
from dataclasses import dataclass, replace

import numpy as np
import pygame
import pygame.gfxdraw as gfx

TAU = 2 * math.pi


# ------------------------------------------------------------------ colour

def lerp(a, b, t: float):
    t = 0.0 if t < 0 else 1.0 if t > 1 else t
    return (int(a[0] + (b[0] - a[0]) * t + 0.5), int(a[1] + (b[1] - a[1]) * t + 0.5),
            int(a[2] + (b[2] - a[2]) * t + 0.5))


def scale(c, k: float):
    return (max(0, min(255, int(c[0] * k))), max(0, min(255, int(c[1] * k))),
            max(0, min(255, int(c[2] * k))))


def clamp01(v: float) -> float:
    return 0.0 if v < 0 else 1.0 if v > 1 else v


def smoothstep(e0: float, e1: float, x: float) -> float:
    t = clamp01((x - e0) / (e1 - e0)) if e1 != e0 else float(x >= e1)
    return t * t * (3 - 2 * t)


class LRU(OrderedDict):
    def __init__(self, limit: int):
        super().__init__()
        self.limit = limit

    def get_or(self, key, build):
        value = self.get(key)
        if value is None:
            value = build()
            self[key] = value
            if len(self) > self.limit:
                self.popitem(last=False)
        else:
            self.move_to_end(key)
        return value


def surface_from(arr: np.ndarray, alpha: np.ndarray | None = None) -> pygame.Surface:
    """Surface from a (w, h, 3) array in the default (display-matching) format."""
    w, h = arr.shape[0], arr.shape[1]
    if alpha is None:
        surf = pygame.Surface((w, h))
        pygame.surfarray.blit_array(surf, np.clip(arr, 0, 255).astype(np.uint8))
        return surf
    surf = pygame.Surface((w, h), pygame.SRCALPHA)
    pygame.surfarray.pixels3d(surf)[:] = np.clip(arr, 0, 255).astype(np.uint8)
    pygame.surfarray.pixels_alpha(surf)[:] = np.clip(alpha, 0, 255).astype(np.uint8)
    return surf


# ---------------------------------------------------------------- palettes

@dataclass(frozen=True)
class Palette:
    """Colours for one time of day."""

    name: str
    sky: tuple          # ((position, rgb), ...) from the top (0) to the horizon (1)
    sun: tuple          # sun / moon disc
    halo: tuple         # light around the sun, god rays
    haze: tuple         # atmospheric haze near the horizon (fog colour)
    far: tuple          # most distant hills (before haze)
    near: tuple         # closest silhouettes
    grass: tuple        # sun-lit grass / meadow tint
    water: tuple        # tint multiplied into reflections
    cloud_lit: tuple
    cloud_dark: tuple
    stars: float        # 0 (daytime) .. 1 (clear night)
    moon: bool          # draw a moon instead of the sun


PALETTES = {
    "golden": Palette(
        "golden", ((0.0, (78, 118, 190)), (0.42, (150, 166, 210)), (0.78, (244, 190, 150)), (1.0, (255, 220, 160))),
        sun=(255, 246, 220), halo=(255, 190, 120), haze=(246, 204, 164), far=(150, 140, 176), near=(30, 44, 40),
        grass=(150, 196, 96), water=(170, 184, 220), cloud_lit=(255, 236, 210), cloud_dark=(170, 158, 196),
        stars=0.0, moon=False),
    "sunset": Palette(
        "sunset", ((0.0, (30, 36, 88)), (0.40, (104, 66, 128)), (0.74, (232, 112, 104)), (1.0, (255, 172, 96))),
        sun=(255, 232, 180), halo=(255, 140, 80), haze=(240, 146, 112), far=(122, 78, 122), near=(24, 20, 38),
        grass=(214, 128, 88), water=(176, 128, 160), cloud_lit=(255, 180, 140), cloud_dark=(104, 66, 120),
        stars=0.18, moon=False),
    "dawn": Palette(
        "dawn", ((0.0, (104, 140, 202)), (0.48, (190, 182, 222)), (0.80, (250, 198, 198)), (1.0, (255, 228, 202))),
        sun=(255, 252, 240), halo=(255, 206, 178), haze=(244, 214, 214), far=(168, 160, 198), near=(48, 56, 82),
        grass=(176, 196, 150), water=(196, 196, 228), cloud_lit=(255, 242, 240), cloud_dark=(186, 176, 212),
        stars=0.0, moon=False),
    "twilight": Palette(
        "twilight", ((0.0, (10, 14, 40)), (0.44, (38, 38, 94)), (0.78, (116, 80, 140)), (1.0, (214, 128, 124))),
        sun=(236, 234, 252), halo=(210, 150, 196), haze=(160, 104, 140), far=(70, 54, 98), near=(12, 12, 28),
        grass=(126, 104, 168), water=(120, 108, 170), cloud_lit=(206, 144, 176), cloud_dark=(48, 42, 90),
        stars=0.7, moon=True),
    "night": Palette(
        "night", ((0.0, (4, 7, 20)), (0.50, (12, 22, 52)), (0.86, (26, 44, 88)), (1.0, (52, 74, 122))),
        sun=(238, 242, 255), halo=(126, 158, 232), haze=(46, 64, 108), far=(30, 40, 72), near=(6, 9, 20),
        grass=(62, 92, 126), water=(70, 92, 140), cloud_lit=(92, 106, 152), cloud_dark=(20, 26, 50),
        stars=1.0, moon=True),
}


def palette(name: str) -> Palette:
    return PALETTES.get(name, PALETTES["golden"])


def blend_palettes(a: Palette, b: Palette, t: float) -> Palette:
    """A palette part-way between *a* and *b* (used to darken day skies for night scenes)."""
    if t <= 0:
        return a
    sky = tuple((pa + (pb - pa) * t, lerp(ca, cb, t)) for (pa, ca), (pb, cb) in zip(a.sky, b.sky))
    return replace(
        a, name=f"{a.name}>{b.name}:{t:.2f}", sky=sky, sun=lerp(a.sun, b.sun, t), halo=lerp(a.halo, b.halo, t),
        haze=lerp(a.haze, b.haze, t), far=lerp(a.far, b.far, t), near=lerp(a.near, b.near, t),
        grass=lerp(a.grass, b.grass, t), water=lerp(a.water, b.water, t),
        cloud_lit=lerp(a.cloud_lit, b.cloud_lit, t), cloud_dark=lerp(a.cloud_dark, b.cloud_dark, t),
        stars=a.stars + (b.stars - a.stars) * t, moon=b.moon if t >= 0.5 else a.moon)


def night_version(p: Palette, amount: float = 0.6) -> Palette:
    """For night scenes: day palettes are pulled towards twilight so stars read."""
    if p.moon:
        return p
    return blend_palettes(p, PALETTES["twilight"], amount)


def land(p: Palette, depth: float):
    """Terrain colour at *depth* (0 = far away, hazy; 1 = right in front)."""
    base = lerp(p.far, p.near, depth ** 0.8)
    return lerp(base, p.haze, (1 - depth) ** 2 * 0.45)


def grassy(p: Palette, depth: float, amount: float = 0.35):
    """Terrain colour tinted towards the grass colour (rolling meadows)."""
    return lerp(land(p, depth), lerp(p.grass, p.near, depth ** 1.6), amount)


def sky_at(p: Palette, y: float):
    """Sky colour at height *y* (0 top .. 1 horizon)."""
    stops = p.sky
    for (p0, c0), (p1, c1) in zip(stops, stops[1:]):
        if y <= p1:
            return lerp(c0, c1, (y - p0) / max(1e-6, p1 - p0))
    return stops[-1][1]


# --------------------------------------------------------------------- sky

_skies = LRU(10)


def sky_surface(p: Palette, size: tuple[int, int], horizon: float) -> pygame.Surface:
    """Vertical multi-stop gradient with baked-in dithering (no colour banding)."""
    w, h = size
    hy = max(1, int(h * horizon))

    def build():
        ys = np.arange(h, dtype=np.float32) / hy
        pos = np.array([s[0] for s in p.sky], np.float32)
        cols = np.array([s[1] for s in p.sky], np.float32)
        col = np.stack([np.interp(ys, pos, cols[:, i]) for i in range(3)], axis=1)
        col[ys > 1] *= 0.9
        noise = np.random.default_rng(7).random((w, h), dtype=np.float32) - 0.5
        return surface_from(col[None, :, :] + noise[..., None] * 1.8)

    return _skies.get_or((p.name, w, h, hy), build)


_vignettes = LRU(6)


def vignette(target: pygame.Surface, strength: float = 1.0) -> None:
    """Soft, slightly cool darkening of the corners (multiply)."""
    w, h = target.get_size()

    def build():
        xx = np.linspace(-1, 1, w, dtype=np.float32)[:, None]
        yy = np.linspace(-1, 1, h, dtype=np.float32)[None, :]
        d = np.sqrt(xx ** 2 * 0.85 + yy ** 2) / 1.36
        f = np.clip(1 - 0.55 * strength * np.clip(d - 0.25, 0, 1) ** 1.7, 0, 1)[..., None]
        edge = np.array((176, 168, 206), np.float32)
        rgb = edge + (255 - edge) * f
        return surface_from(np.clip(rgb * (0.55 + 0.45 * f), 0, 255))

    target.blit(_vignettes.get_or((w, h, strength), build), (0, 0), special_flags=pygame.BLEND_RGB_MULT)


# ------------------------------------------------------------------- light

_glows = LRU(400)


def glow_sprite(radius: int, color, power: float = 2.0) -> pygame.Surface:
    """Radial glow on black, for additive blending."""
    r = max(1, int(radius))

    def build():
        size = 2 * r
        c = np.arange(size, dtype=np.float32) - r + 0.5
        d = np.sqrt(c[:, None] ** 2 + c[None, :] ** 2) / r
        if power <= 0:      # flat disc with a soft rim (out-of-focus "bokeh")
            f = np.clip((1 - d) * 6, 0, 1) * (0.75 + 0.25 * np.clip(d, 0, 1) ** 4)
        else:
            f = np.clip(1 - d, 0, 1) ** power
        return surface_from(f[..., None] * np.array(color, np.float32))

    return _glows.get_or((r, tuple(color), power), build)


def add_glow(surf: pygame.Surface, center, radius: float, color, strength: float = 1.0,
             power: float = 2.0) -> None:
    """Additive soft light. Strength is quantised so the sprite cache stays small."""
    q = int(strength * 16 + 0.5)
    if q <= 0 or radius < 1:
        return
    spr = glow_sprite(int(radius), scale(color, min(q, 32) / 16), power)
    r = spr.get_width() // 2
    surf.blit(spr, (int(center[0]) - r, int(center[1]) - r), special_flags=pygame.BLEND_RGB_ADD)


_ray_grids = LRU(4)
_ray_profiles: dict = {}


def god_rays(light: pygame.Surface, center, radius: float, color, strength: float, angle: float,
             floor: float | None = None, seed: int = 9) -> None:
    """Soft volumetric light shafts from *center*, drawn on the light surface.

    Computed at half the light surface's resolution with numpy, then smoothly
    scaled up. *floor* fades the rays out below that y (e.g. the horizon).
    """
    lw, lh = light.get_size()
    sw, sh = max(4, lw // 2), max(4, lh // 2)
    cx, cy, rad = center[0] / 2, center[1] / 2, max(4.0, radius / 2)
    key = (sw, sh, int(cx), int(cy), int(rad), None if floor is None else int(floor / 2))

    def build():
        xs = np.arange(sw, dtype=np.float32)[:, None] - cx
        ys = np.arange(sh, dtype=np.float32)[None, :] - cy
        ang = np.arctan2(ys, xs)
        dist = np.sqrt(xs ** 2 + ys ** 2) / rad
        fall = np.clip(1 - dist, 0, 1) ** 1.6 * np.clip(dist * 6, 0, 1)
        if floor is not None:
            fy = floor / 2
            fall = fall * np.clip(1 - (np.arange(sh, dtype=np.float32)[None, :] - fy) / (sh * 0.12), 0, 1)
        idx = ((ang / TAU + 0.5) * 1024).astype(np.int32) % 1024
        return idx, fall.astype(np.float32)

    idx, fall = _ray_grids.get_or(key, build)
    prof = _ray_profiles.get(seed)
    if prof is None:
        rng = np.random.default_rng(seed)
        a = np.linspace(0, TAU, 1024, endpoint=False)
        prof = np.zeros(1024, np.float32)
        for _ in range(16):
            c, wdt, amp = rng.uniform(0, TAU), rng.uniform(0.03, 0.10), rng.uniform(0.4, 1.0)
            d = np.angle(np.exp(1j * (a - c)))
            prof += amp * np.exp(-(d / wdt) ** 2)
        prof = np.clip(prof, 0, 1.2).astype(np.float32)
        _ray_profiles[seed] = prof
    shift = int(angle / TAU * 1024) % 1024
    inten = np.roll(prof, shift)[idx] * fall * strength
    arr = inten[..., None] * np.array(color, np.float32)
    add_scaled(light, arr)


_scratch = LRU(8)


def add_scaled(light: pygame.Surface, arr: np.ndarray) -> None:
    """Additively blend a small (w, h, 3) light array, smoothly scaled up to *light*."""
    sw, sh = arr.shape[0], arr.shape[1]
    lw, lh = light.get_size()
    small = _scratch.get_or(("s", sw, sh), lambda: pygame.Surface((sw, sh)))
    big = _scratch.get_or(("b", lw, lh), lambda: pygame.Surface((lw, lh)))
    pygame.surfarray.blit_array(small, np.clip(arr, 0, 255).astype(np.uint8))
    pygame.transform.smoothscale(small, (lw, lh), big)
    light.blit(big, (0, 0), special_flags=pygame.BLEND_RGB_ADD)


# ----------------------------------------------------------------- terrain

def ridge(xs: np.ndarray, seed: int, octaves: int = 4, sharp: float = 0.0) -> np.ndarray:
    """Smooth 1-D noise in 0..1 (sum of seeded sine waves). *sharp* makes peaks pointier."""
    rng = np.random.default_rng(seed)
    y = np.zeros_like(xs, dtype=np.float32)
    amp, freq, total = 1.0, 1.3, 0.0
    for _ in range(octaves):
        s = np.sin(TAU * freq * rng.uniform(0.85, 1.2) * xs + rng.uniform(0, TAU))
        if sharp:
            s = (1 - np.abs(s)) * 2 - 1
            s = s * (1 - sharp) + np.sign(s) * np.abs(s) ** 2 * sharp
        y += amp * s
        total += amp
        amp *= 0.5
        freq *= 2.1
    return (y / total + 1) * 0.5


def fill_profile(surf: pygame.Surface, color, xs, ys, bottom: float) -> None:
    """Fill the area under a profile line, with an anti-aliased top edge."""
    top = np.column_stack((xs, ys)).tolist()
    pygame.draw.polygon(surf, color, [(top[0][0], bottom)] + top + [(top[-1][0], bottom)])
    pygame.draw.aalines(surf, color, False, top)


_fogs = LRU(40)


def fog(surf: pygame.Surface, y_top: float, height: float, color, strength: float) -> None:
    """A soft horizontal band of haze that thickens downwards."""
    w = surf.get_width()
    hgt = max(2, int(height))
    a = int(clamp01(strength) * 32) * 8
    if a <= 0:
        return

    def build():
        f = np.linspace(0, 1, hgt, dtype=np.float32)
        rise = np.clip(f / 0.7, 0, 1)
        fall = np.clip((1 - f) / 0.3, 0, 1)
        alpha = (rise * rise * (3 - 2 * rise)) ** 1.3 * (fall * fall * (3 - 2 * fall)) * a
        rgb = np.broadcast_to(np.array(color, np.float32), (w, hgt, 3))
        return surface_from(rgb, np.broadcast_to(alpha[None, :], (w, hgt)))

    surf.blit(_fogs.get_or((w, hgt, tuple(color), a), build), (0, int(y_top)))


# ------------------------------------------------------------------ clouds

_clouds = LRU(48)


def cloud_sprite(width: int, seed: int, lit, dark) -> pygame.Surface:
    """A soft cumulus cloud, lit from above, as a smooth alpha sprite."""
    w = max(8, int(width))
    h = max(4, int(w * 0.38))

    def build():
        rng = np.random.default_rng(seed)
        gw, gh = 120, 46
        xx = np.arange(gw, dtype=np.float32)[:, None]
        yy = np.arange(gh, dtype=np.float32)[None, :]
        dens = np.zeros((gw, gh), np.float32)
        for _ in range(16):
            cx = rng.uniform(0.14, 0.86) * gw
            cy = rng.uniform(0.48, 0.72) * gh
            r = rng.uniform(0.07, 0.16) * gw * (1.25 - abs(cx / gw - 0.5))
            dens += np.exp(-(((xx - cx) / r) ** 2 + ((yy - cy) / (r * 0.8)) ** 2))
        dens *= np.clip((gh * 0.80 - yy) / (gh * 0.22), 0, 1)        # softly flattened base
        edge = np.clip(np.minimum(xx, gw - 1 - xx) / (gw * 0.08), 0, 1) * np.clip(yy / (gh * 0.12), 0, 1)
        dens *= edge
        alpha = np.clip((dens - 0.3) * 1.5, 0, 1) ** 0.9 * 235
        shade = np.clip(1.25 - yy / gh * 1.3 - np.clip(dens - 1.2, 0, 1) * 0.15, 0, 1)
        lit_a, dark_a = np.array(lit, np.float32), np.array(dark, np.float32)
        rgb = dark_a + (lit_a - dark_a) * shade[..., None]
        small = surface_from(rgb, alpha)
        return pygame.transform.smoothscale(small, (w, h))

    return _clouds.get_or((w, seed, tuple(lit), tuple(dark)), build)


class CloudLayer:
    """Clouds drifting across the sky; the wind picks up with the music."""

    def __init__(self, count: int = 6, seed: int = 21, y_range=(0.06, 0.40), size_range=(0.22, 0.42)):
        rng = np.random.default_rng(seed)
        self.items = [dict(seed=int(rng.integers(0, 10_000)), x=float(rng.random()),
                           y=float(rng.uniform(*y_range)), size=float(rng.uniform(*size_range)),
                           speed=float(rng.uniform(0.006, 0.016)))
                      for _ in range(count)]
        self.items.sort(key=lambda c: c["size"])          # small (far) clouds first

    def draw(self, surf: pygame.Surface, horizon_px: float, clock: float, p: Palette, alpha: float = 1.0) -> None:
        w = surf.get_width()
        for c in self.items:
            cw = int(w * c["size"])
            span = 1 + c["size"] * 2
            x = ((c["x"] + clock * c["speed"]) % span - c["size"]) * w
            y = horizon_px * c["y"]
            spr = cloud_sprite(cw, c["seed"], p.cloud_lit, p.cloud_dark)
            if x + cw < 0 or x > w:
                continue
            spr.set_alpha(int(255 * alpha))
            surf.blit(spr, (int(x), int(y)))


# ------------------------------------------------------------------- stars

class StarField:
    def __init__(self, count: int = 220, seed: int = 3):
        rng = np.random.default_rng(seed)
        self.x = rng.random(count).astype(np.float32)
        self.y = (rng.random(count) ** 1.3).astype(np.float32)
        self.mag = (rng.random(count) ** 3).astype(np.float32)
        self.phase = rng.uniform(0, TAU, count).astype(np.float32)
        self.freq = rng.uniform(0.5, 2.6, count).astype(np.float32)

    def positions(self, w: int, h: int, horizon_px: float, rotation: float = 0.0, pole=None):
        if pole is None:
            return self.x * w, self.y * horizon_px
        # a disc of stars turning slowly around the celestial pole
        px, py = pole
        reach = math.hypot(max(px, w - px), horizon_px - py)
        r = np.sqrt(self.x) * reach
        a = self.y * TAU * 3.7 + rotation
        return px + np.cos(a) * r, py + np.sin(a) * r

    def draw(self, surf, light, k, w, h, horizon_px, t, visibility, treble, u,
             rotation: float = 0.0, pole=None) -> None:
        if visibility <= 0.01:
            return
        xs, ys = self.positions(w, h, horizon_px, rotation, pole)
        tw = 0.65 + 0.35 * np.sin(t * self.freq + self.phase)
        b = np.clip(visibility * (0.3 + 0.7 * self.mag) * tw * (1 + 0.6 * treble), 0, 1)
        b = b * np.clip((horizon_px - ys) / (horizon_px * 0.25), 0, 1)
        keep = (b > 0.04) & (xs >= 0) & (xs < w) & (ys >= 0) & (ys < horizon_px)
        base_r = max(1.0, u / 900)
        for x, y, bb, m in zip(xs[keep].tolist(), ys[keep].tolist(), b[keep].tolist(), self.mag[keep].tolist()):
            r = int(base_r * (0.7 + 1.3 * m) + 0.5)
            col = (255, 248, 236, int(255 * bb))
            if r <= 1:
                gfx.pixel(surf, int(x), int(y), col)
            else:
                gfx.filled_circle(surf, int(x), int(y), r, col)
                gfx.aacircle(surf, int(x), int(y), r, col)
            if light is not None and m > 0.55:
                add_glow(light, (x * k, y * k), max(2, u * k * 0.012 * m), (200, 210, 255), bb * 0.8)


# ------------------------------------------------------------------- life

def draw_birds(surf, w, h, t, color, u, seed: int = 5) -> None:
    """Now and then a small flock glides across the sky."""
    period = 19.0
    k = int(t // period)
    phase = (t % period) / period
    rng = np.random.default_rng(seed * 1000 + k)
    if rng.random() < 0.3:
        return
    direction = 1 if rng.random() < 0.5 else -1
    y0 = h * rng.uniform(0.14, 0.34)
    n = int(rng.integers(3, 8))
    lead = -0.15 * w + phase * 1.3 * w
    if direction < 0:
        lead = w - lead
    width = max(1.0, u / 520)
    for i in range(n):
        row = (i + 1) // 2
        side = 1 if i % 2 else -1
        x = lead - row * u * 0.04 * direction
        y = y0 + row * u * 0.022 * (side if i else 0) + math.sin(t * 0.8 + i) * u * 0.004
        s = u * (0.011 + 0.002 * ((i * 7) % 3))
        flap = math.sin(t * 6.5 + i * 1.7)
        tip_y = y - s * 0.55 * flap
        mid = (x, y + s * 0.12)
        for sgn in (-1, 1):
            elbow = (x + sgn * s * 0.5, y - s * 0.18 - s * 0.25 * flap)
            aa_line(surf, color, mid, elbow, width)
            aa_line(surf, color, elbow, (x + sgn * s, tip_y), width)


class Motes:
    """Floating specks of dust / pollen catching the light."""

    def __init__(self, count: int = 60, seed: int = 11, region=(0.0, 1.0, 0.15, 0.9)):
        rng = np.random.default_rng(seed)
        x0, x1, y0, y1 = region
        self.x = rng.uniform(x0, x1, count)
        self.y = rng.uniform(y0, y1, count)
        self.vx = rng.uniform(-0.008, 0.012, count)
        self.vy = rng.uniform(-0.012, -0.002, count)
        self.ph = rng.uniform(0, TAU, count)
        self.size = rng.random(count) ** 2
        self.region = region

    def positions(self, w, h, clock):
        x0, x1, y0, y1 = self.region
        x = x0 + (self.x - x0 + self.vx * clock + 0.012 * np.sin(clock * 0.6 + self.ph)) % (x1 - x0)
        y = y0 + (self.y - y0 + self.vy * clock + 0.008 * np.sin(clock * 0.9 + self.ph * 1.3)) % (y1 - y0)
        return x * w, y * h

    def draw(self, surf, light, k, w, h, clock, t, color, u, strength: float = 1.0) -> None:
        xs, ys = self.positions(w, h, clock)
        tw = 0.55 + 0.45 * np.sin(t * 1.7 + self.ph * 3)
        for x, y, s, b in zip(xs.tolist(), ys.tolist(), self.size.tolist(), tw.tolist()):
            a = clamp01(b * strength)
            if a < 0.05:
                continue
            r = max(1, int(u * (0.0015 + 0.003 * s)))
            gfx.filled_circle(surf, int(x), int(y), r, (*color, int(150 * a)))
            if light is not None:
                add_glow(light, (x * k, y * k), max(2, u * k * (0.010 + 0.02 * s)), color, a * 0.45)


class Fireflies:
    """Fireflies wandering along smooth looping paths, blinking on their own rhythm."""

    def __init__(self, count: int = 80, seed: int = 17, region=(0.0, 1.0, 0.45, 0.95)):
        rng = np.random.default_rng(seed)
        self.n = count
        self.cx = rng.uniform(region[0], region[1], count)
        self.cy = rng.uniform(region[2], region[3], count)
        self.ax = rng.uniform(0.02, 0.07, count)
        self.ay = rng.uniform(0.015, 0.05, count)
        self.fx = rng.uniform(0.08, 0.25, count)
        self.fy = rng.uniform(0.10, 0.30, count)
        self.ph = rng.uniform(0, TAU, count)
        self.blink = rng.uniform(0.25, 0.7, count)
        self.order = rng.permutation(count)          # which ones light up first
        self.hue = np.round(rng.random(count) * 3) / 3      # 4 tints keep the glow cache small

    def draw(self, surf, light, k, w, h, clock, activity, flash, c1, c2, u, size: float = 1.0) -> None:
        lit = int(self.n * clamp01(0.25 + 0.75 * activity))
        idx = self.order[:lit]
        x = (self.cx[idx] + self.ax[idx] * np.sin(clock * self.fx[idx] * TAU + self.ph[idx])
             + 0.5 * self.ax[idx] * np.sin(clock * self.fx[idx] * 2.7 + self.ph[idx] * 2)) * w
        y = (self.cy[idx] + self.ay[idx] * np.sin(clock * self.fy[idx] * TAU + self.ph[idx] * 1.7)) * h
        pulse = np.clip(np.sin(clock * self.blink[idx] * TAU + self.ph[idx]) * 1.4 + 0.4, 0, 1)
        b = np.clip(pulse * (0.55 + 0.45 * activity) + flash * 0.8, 0, 1)
        for xx, yy, bb, hue in zip(x.tolist(), y.tolist(), b.tolist(), self.hue[idx].tolist()):
            if bb < 0.05:
                continue
            col = lerp(c1, c2, hue)
            r = max(1, int(u * 0.0035 * size * (0.7 + 0.5 * bb)))
            core = lerp(col, (255, 255, 240), 0.55)
            gfx.filled_circle(surf, int(xx), int(yy), r, (*core, int(255 * bb)))
            gfx.aacircle(surf, int(xx), int(yy), r, (*core, int(255 * bb)))
            if light is not None:
                add_glow(light, (xx * k, yy * k), max(2, u * k * 0.035 * size), col, bb)


# ------------------------------------------------------------------ shapes

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


def aa_poly(surf, color, pts) -> None:
    gfx.filled_polygon(surf, pts, color)
    gfx.aapolygon(surf, pts, color)


def aa_circle(surf, color, center, radius: float) -> None:
    x, y, r = int(center[0]), int(center[1]), int(radius)
    if r >= 1:
        gfx.filled_circle(surf, x, y, r, color)
        gfx.aacircle(surf, x, y, r, color)


def pine(surf, x: float, base: float, height: float, color, sway: float = 0.0) -> None:
    """A conifer silhouette; *sway* bends the tip sideways (in px)."""
    wdt = height * 0.36
    trunk_w = max(1.0, height * 0.05)
    pygame.draw.rect(surf, color, (int(x - trunk_w / 2), int(base - height * 0.15), max(1, int(trunk_w)),
                                   int(height * 0.15) + 1))
    for i in range(3):
        f0 = 0.12 + i * 0.25
        f1 = f0 + 0.55
        yb = base - height * f0
        yt = base - height * min(1.0, f1)
        bw = wdt * (1 - i * 0.24) / 2
        tip = x + sway * min(1.0, f1)
        side = sway * f0
        aa_poly(surf, color, [(x - bw + side, yb), (tip, yt), (x + bw + side, yb)])


def round_tree(surf, x: float, base: float, height: float, color, sway: float = 0.0, seed: int = 0) -> None:
    """A leafy, rounded tree silhouette made of overlapping circles."""
    rng = np.random.default_rng(seed)
    trunk_h = height * 0.42
    tw = max(1.5, height * 0.07)
    aa_poly(surf, color, [(x - tw, base), (x + sway * 0.3 - tw * 0.6, base - trunk_h),
                          (x + sway * 0.3 + tw * 0.6, base - trunk_h), (x + tw, base)])
    cr = height * 0.30
    cx, cy = x + sway * 0.6, base - trunk_h - cr * 0.55
    for _ in range(7):
        ox, oy = rng.uniform(-0.75, 0.75) * cr, rng.uniform(-0.6, 0.35) * cr
        r = cr * rng.uniform(0.5, 0.75)
        lean = sway * 0.4 * (1 - (oy / cr + 0.6))
        aa_circle(surf, color, (cx + ox + lean, cy + oy), r)


class Grass:
    """A field of grass blades that sway in travelling gusts of wind."""

    def __init__(self, count: int = 160, seed: int = 31, width: float = 1.0):
        rng = np.random.default_rng(seed)
        self.x = np.sort(rng.random(count))
        self.h = rng.uniform(0.5, 1.0, count)
        self.lean = rng.uniform(-0.25, 0.25, count)
        self.ph = rng.uniform(0, TAU, count)
        self.wd = rng.uniform(0.6, 1.2, count) * width
        self.shade = rng.random(count)

    def draw(self, surf, w, base_y, height, clock, wind, dark, lit, u, heights=None, rim=None,
             rim_levels=None) -> None:
        """Draw the blades; *rim* adds a thin back-lit edge (brighter with *rim_levels*)."""
        hs = self.h * height if heights is None else heights
        sway = (np.sin(clock * 1.7 - self.x * 9 + self.ph * 0.3) * 0.6
                + np.sin(clock * 0.7 - self.x * 4) * 0.4) * (0.10 + 0.32 * wind)
        bw = u * 0.0042 * self.wd
        bases = np.broadcast_to(np.asarray(base_y, dtype=np.float64), self.x.shape)
        rims = np.broadcast_to(np.asarray(0.5 if rim_levels is None else rim_levels, dtype=np.float64),
                               self.x.shape)
        for x, hh, ln, sw, bwi, sh, base_y, rl in zip((self.x * w).tolist(), np.asarray(hs).tolist(),
                                                      self.lean.tolist(), sway.tolist(), bw.tolist(),
                                                      self.shade.tolist(), bases.tolist(), rims.tolist()):
            if hh < 2:
                continue
            # Three tapering thick-line segments along a curve: ~10x cheaper than
            # filling a tall polygon, and the anti-aliased edge keeps it smooth.
            bend = (ln + sw) * hh
            p0 = (x, base_y + 2)
            p1 = (x + bend * 0.17, base_y - hh * 0.35)
            p2 = (x + bend * 0.52, base_y - hh * 0.68)
            p3 = (x + bend, base_y - hh)
            col = lerp(dark, lit, sh * 0.6)
            pygame.draw.line(surf, col, p0, p1, max(1, int(bwi * 2.0)))
            pygame.draw.line(surf, col, p1, p2, max(1, int(bwi * 1.4)))
            pygame.draw.line(surf, col, p2, p3, max(1, int(bwi * 0.8)))
            edge = col if rim is None else lerp(col, rim, 0.35 + 0.5 * rl)
            pygame.draw.aalines(surf, edge, False, (p1, p2, p3))


# ------------------------------------------------------------------- water

_water_tints = LRU(6)


def reflect(surf: pygame.Surface, horizon: int, clock: float, p: Palette, u: float,
            calm: float = 1.0) -> None:
    """Turn everything below *horizon* into a rippling mirror of what is above it."""
    w, h = surf.get_size()
    wh = min(h - horizon, horizon)
    if wh < 4:
        return
    src = pygame.transform.flip(surf.subsurface((0, horizon - wh, w, wh)), False, True)
    y = 0
    k = u / 1000.0
    amp = u * 0.006 * calm
    while y < wh:
        f = y / wh
        sh = max(2, int((2 + f * 9) * k))
        dx = amp * f * (math.sin(y / (8 * k + 1e-6) * 0.35 - clock * 1.9)
                        + 0.6 * math.sin(y / (8 * k + 1e-6) * 0.13 + clock * 1.1))
        surf.blit(src, (int(dx), horizon + y), (0, y, w, sh))
        y += sh

    def build():
        f = np.linspace(0, 1, wh, dtype=np.float32)[None, :, None]
        top = np.array(p.water, np.float32)
        rgb = top * (0.92 - 0.42 * f ** 0.8)
        return surface_from(np.broadcast_to(rgb, (w, wh, 3)).copy())

    surf.blit(_water_tints.get_or((w, wh, p.name), build), (0, horizon), special_flags=pygame.BLEND_RGB_MULT)


# ------------------------------------------------------------- set pieces

_treelines = LRU(12)


def treeline_surface(w: int, height: int, color, seed: int, density: float = 1.0,
                     hill: float = 0.35) -> pygame.Surface:
    """A cached strip of distant hills topped with trees (transparent above)."""
    height = max(4, int(height))

    def build():
        surf = pygame.Surface((w, height), pygame.SRCALPHA)
        surf.fill((*color, 0))
        xs = np.linspace(0, 1, 140)
        base = height - (ridge(xs, seed, 3) * hill + 0.08) * height
        fill_profile(surf, color, xs * w, base, height)
        rng = np.random.default_rng(seed)
        n = int(w / height * 6 * density)
        for _ in range(n):
            x = rng.uniform(-0.02, 1.02) * w
            by = float(np.interp(x / w, xs, base)) + 2
            th = height * rng.uniform(0.25, 0.55) * (1 - hill * 0.5)
            if rng.random() < 0.7:
                pine(surf, x, by, th, color)
            else:
                round_tree(surf, x, by, th * 0.9, color, 0.0, int(rng.integers(0, 999)))
        return surf

    return _treelines.get_or((w, height, tuple(color), seed, density, hill), build)


_moons = LRU(8)


def moon_sprite(radius: int, color) -> pygame.Surface:
    """A softly shaded full moon with faint craters."""
    r = max(2, int(radius))

    def build():
        size = 2 * r + 2
        c = np.arange(size, dtype=np.float32) - (r + 1) + 0.5
        X, Y = c[:, None], c[None, :]
        d = np.sqrt(X ** 2 + Y ** 2)
        alpha = np.clip(r - d + 0.5, 0, 1) * 255
        shade = 0.80 + 0.20 * np.clip(1 - d / r, 0, 1) ** 0.5
        rng = np.random.default_rng(4)
        for _ in range(10):
            cx, cy = rng.uniform(-0.6, 0.6, 2) * r
            cr = rng.uniform(0.08, 0.26) * r
            shade = shade - rng.uniform(0.05, 0.11) * np.exp(-((X - cx) ** 2 + (Y - cy) ** 2) / cr ** 2)
        rgb = np.array(color, np.float32)[None, None, :] * shade[..., None]
        return surface_from(rgb, alpha)

    return _moons.get_or((r, tuple(color)), build)

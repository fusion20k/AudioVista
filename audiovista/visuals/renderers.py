"""Visualizer styles: living landscapes that move with the music.

* bars      - Mountains: the front mountain range *is* the spectrum (bass on the
              left, treble on the right), with glowing rim light, falling
              peak-hold stars, parallax ranges, fog, clouds and god rays.
* waveform  - Lakeside: the real waveform drifts across the sky like an aurora
              ribbon (played part bright, upcoming part faint) and is mirrored
              in a rippling lake; beats send rings across the water.
* circular  - Starlight: a radial spectrum crown around the moon, an orrery of
              orbiting lights, aurora, a slowly turning star field and
              shooting stars on the beat.
* meters    - Windmills: rolling hills with three windmills whose sails are
              driven by the bass, mids and treble; the sun crosses the sky over
              the length of the song.
* particles - Fireflies: a dusk meadow where the grass height follows the
              spectrum and fireflies wake up with the treble and flash on beats.

``draw`` paints the scene onto the (opaque) frame. Soft light is drawn on the
quarter-resolution ``light`` surface (``k`` = its scale), which the engine
blurs and adds on top; ``light`` is ``None`` when glow and atmosphere are off.
All motion is driven by the analysis frame and ``dt``, so exported videos
match the live preview.
"""

from __future__ import annotations

import math

import numpy as np
import pygame
import pygame.gfxdraw as gfx

from ..audio.analyzer import AnalysisFrame
from ..models import VisualSettings
from .scenery import (TAU, CloudLayer, add_scaled, Fireflies, Grass, Motes, StarField, aa_circle, aa_line, aa_poly,
                      add_glow, clamp01, draw_birds, fill_profile, fog, god_rays, grassy, land, lerp,
                      moon_sprite, night_version, palette, pine, reflect, ridge, round_tree, scale,
                      sky_at, sky_surface, treeline_surface)

# Kept for code that imported the old helpers.
lerp_color = lerp
scale_color = scale


def resample(values: np.ndarray, n: int) -> np.ndarray:
    if len(values) == n:
        return values
    return np.interp(np.linspace(0, len(values) - 1, n), np.arange(len(values)), values)


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

    def draw(self, target: pygame.Surface, light: pygame.Surface | None, k: float,
             frame: AnalysisFrame, settings: VisualSettings, dt: float) -> None:
        raise NotImplementedError


class SceneRenderer(BaseRenderer):
    """Shared state for the landscape scenes."""

    horizon = 0.66

    def __init__(self):
        super().__init__()
        self.clock = 0.0          # "world" time: runs a little faster when the music is loud
        self.calm = 0.0           # slow-moving loudness

    def reset(self) -> None:
        super().reset()
        self.clock = 0.0
        self.calm = 0.0

    def tick(self, frame: AnalysisFrame, dt: float) -> bool:
        playing = dt > 0.0015
        if playing:
            self.clock += dt * (0.7 + 0.8 * frame.volume)
            self.calm += (frame.volume - self.calm) * (1 - 0.2 ** dt)
        return playing

    @staticmethod
    def sun(target, light, k, pos, r, pal, energy: float, u: float) -> None:
        """Sun (or moon, for night palettes) with its haze on the light surface."""
        x, y = pos
        if pal.moon:
            spr = moon_sprite(int(r * 0.8), pal.sun)
            target.blit(spr, spr.get_rect(center=(int(x), int(y))))
        else:
            aa_circle(target, lerp(pal.sun, pal.halo, 0.15), (x, y), r * 1.06)
            aa_circle(target, pal.sun, (x, y), r)
        if light is not None:
            soft = 0.45 if pal.moon else 1.0
            add_glow(light, (x * k, y * k), u * 0.62 * k, pal.halo, (0.30 + 0.30 * energy) * soft, power=2.2)
            add_glow(light, (x * k, y * k), r * 3.2 * k, pal.sun, (0.45 + 0.35 * energy) * soft, power=1.6)


# ------------------------------------------------------------- Mountains

class MountainsRenderer(SceneRenderer):
    style = "bars"
    horizon = 0.70

    def __init__(self):
        super().__init__()
        self.clouds = CloudLayer(6, seed=21, y_range=(0.06, 0.46))
        self.stars = StarField(200, seed=3)
        self.motes = Motes(32, seed=11, region=(0.0, 1.0, 0.25, 0.85))
        self.peaks: np.ndarray | None = None
        self.slow: np.ndarray | None = None
        rng = np.random.default_rng(55)
        self.trees = [(x, rng.uniform(0.08, 0.17)) for x in rng.uniform(0, 1, 16) if not 0.38 < x < 0.62]

    def reset(self):
        super().reset()
        self.peaks = None
        self.slow = None

    def draw(self, target, light, k, frame, s, dt):
        w, h = target.get_size()
        u = min(w, h)
        pal = palette(s.background)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        lv = self.levels(frame, s, dt)
        playing = self.tick(frame, dt)
        bass, mid, treble = self.bands3(lv)
        hy = h * self.horizon
        glow_on = light if s.glow else None

        if self.peaks is None or len(self.peaks) != len(lv):
            self.peaks = lv.copy()
            self.slow = lv.copy()
        if playing:
            self.peaks = np.maximum(self.peaks - dt * 0.6, lv)
            self.slow += (lv - self.slow) * (1 - 0.15 ** dt)

        target.blit(sky_surface(pal, (w, h), self.horizon), (0, 0))
        self.stars.draw(target, glow_on, k, w, h, hy, frame.time, pal.stars, treble, u)
        sx, sy = w * 0.5, hy - h * 0.24
        self.sun(target, light, k, (sx, sy), u * 0.065 * (1 + 0.05 * bass), pal, 0.6 * bass + 0.4 * frame.beat, u)
        if light is not None and s.atmosphere:
            god_rays(light, (sx * k, sy * k), u * 1.1 * k, pal.halo, 0.30 + 0.35 * bass + 0.2 * frame.beat,
                     self.clock * 0.04, floor=hy * k)
        self.clouds.draw(target, hy, self.clock, pal)
        draw_birds(target, w, h, frame.time, lerp(pal.near, sky_at(pal, 0.4), 0.25), u)

        n = 420          # dense sampling keeps the sharp ridges smooth
        xs = np.linspace(0, 1, n)
        X = xs * w
        # The ranges, back to front. Each is only filled down to where the next
        # one covers it (filling big areas is the expensive part).
        far_top = hy - h * (0.08 + 0.17 * ridge(xs + self.clock * 0.0015, 101, 5, sharp=0.6))
        bands_x = np.linspace(0.04, 0.96, len(lv))
        mid_spec = np.interp(xs, bands_x, self.slow)
        mid_r = ridge(xs + self.clock * 0.004, 202, 4, sharp=0.4)
        mid_top = hy - h * (0.02 + 0.09 * mid_r + 0.10 * mid_spec)
        # front range = the spectrum (bass left, treble right)
        sigma = 0.9 / len(lv)
        g = np.exp(-((xs[:, None] - bands_x[None, :]) / sigma) ** 2)
        spec = (g @ lv) / 1.55
        detail = ridge(xs, 303, 4, sharp=0.8)
        base_y = hy + h * 0.07
        top = base_y - h * (0.04 + 0.05 * detail + 0.36 * np.clip(spec, 0, 1.2) ** 1.2)
        fg_top = h * (0.93 - 0.06 * ridge(xs, 404, 3))

        fill_profile(target, land(pal, 0.15), X, far_top, mid_top.max() + 2)
        fog(target, hy - h * 0.16, h * 0.16, pal.haze, 0.55)
        fill_profile(target, land(pal, 0.42), X, mid_top, top.max() + 2)
        fog(target, hy - h * 0.07, h * 0.12, pal.haze, 0.45)
        front = land(pal, 0.72)
        fill_profile(target, front, X, top, fg_top.max() + 2)
        # rim light along the ridge in the user's colour
        rim = lerp(front, c1, 0.75)
        pts = np.column_stack((X, top)).tolist()
        pygame.draw.aalines(target, rim, False, pts)
        pygame.draw.aalines(target, rim, False, [(x, y + 1) for x, y in pts])
        if glow_on is not None:
            lw = max(1, int(u * k * 0.008))
            for i in range(0, n - 1, 5):
                j = min(n - 1, i + 5)
                e = clamp01(0.25 + spec[i] * 0.9)
                pygame.draw.line(light, scale(c1, 0.6 * e), (X[i] * k, top[i] * k), (X[j] * k, top[j] * k), lw)

        # peak-hold stars that fall back slowly, every other band
        peak_tops = np.interp(bands_x, xs, top)
        col = lerp(c2, (255, 255, 255), 0.35)
        for i in range(0, len(lv), 2):
            y = peak_tops[i] - (self.peaks[i] - lv[i]) * h * 0.36 - u * 0.018
            x = bands_x[i] * w
            b = clamp01(0.35 + self.peaks[i])
            aa_circle(target, lerp(sky_at(pal, min(1, y / hy)), col, b), (x, y), max(1.5, u * 0.0035))
            if glow_on is not None:
                add_glow(light, (x * k, y * k), u * k * 0.03, c2, b * 0.8)

        # foreground hill with swaying pines
        fog(target, h * 0.80, h * 0.14, pal.haze, 0.30)
        near = land(pal, 1.0)
        fill_profile(target, near, X, fg_top, h)
        for x, th in self.trees:
            th *= h
            by = float(np.interp(x, xs, fg_top)) + 3
            sway = math.sin(self.clock * 1.3 + x * 7) * th * (0.03 + 0.05 * self.calm)
            pine(target, x * w, by, th, near, sway)

        if s.atmosphere:
            self.motes.draw(target, light, k, w, h, self.clock, frame.time, lerp(pal.halo, (255, 255, 255), 0.4),
                            u, 0.5 + treble)


# --------------------------------------------------------------- Lakeside

class LakesideRenderer(SceneRenderer):
    style = "waveform"
    horizon = 0.58
    BUCKETS = 150
    BUCKET_SECONDS = 0.055

    def __init__(self):
        super().__init__()
        self.clouds = CloudLayer(5, seed=8, y_range=(0.05, 0.42), size_range=(0.18, 0.34))
        self.stars = StarField(180, seed=5)
        self.flies = Fireflies(36, seed=19, region=(0.0, 1.0, 0.70, 0.96))
        self.reeds = np.random.default_rng(12).random((26, 4))
        rng = np.random.default_rng(3)
        n = 70
        self.glitter_y = rng.random(n) ** 1.6
        self.glitter_x = rng.normal(0, 1, n) * (0.03 + 0.12 * self.glitter_y)
        self.glitter_ph = rng.uniform(0, TAU, n)
        self.glitter_f = rng.uniform(2, 6, n)

    def envelope(self, t: float, w: int):
        n, bs = self.BUCKETS, self.BUCKET_SECONDS
        spacing = w / (n - 4)
        pos = t / bs
        first = int(math.floor(pos)) - n // 2 - 1
        if self.analyzer is None:
            env = np.zeros(n + 2, np.float32)
        else:
            env = self.analyzer.envelope_buckets(first, n + 2, bs)
        env = np.convolve(np.pad(env, 1, mode="edge"), [0.25, 0.5, 0.25], mode="valid")
        xs = w / 2 + (np.arange(first, first + n + 2) + 0.5 - pos) * spacing
        return xs, env

    def draw(self, target, light, k, frame, s, dt):
        w, h = target.get_size()
        u = min(w, h)
        pal = palette(s.background)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        lv = self.levels(frame, s, dt)
        self.tick(frame, dt)
        bass, mid, treble = self.bands3(lv)
        hy = int(h * self.horizon)
        glow_on = light if s.glow else None

        target.blit(sky_surface(pal, (w, h), self.horizon), (0, 0))
        self.stars.draw(target, glow_on, k, w, h, hy, frame.time, pal.stars, treble, u)
        sx, sy = w * 0.5, hy - h * 0.05
        self.sun(target, light, k, (sx, sy), u * 0.07 * (1 + 0.04 * bass), pal, 0.5 * bass + 0.5 * frame.beat, u)
        if light is not None and s.atmosphere:
            god_rays(light, (sx * k, sy * k), u * 1.0 * k, pal.halo, 0.25 + 0.3 * bass, self.clock * 0.03,
                     floor=hy * k, seed=4)
        self.clouds.draw(target, hy, self.clock, pal)
        draw_birds(target, w, h, frame.time, lerp(pal.near, sky_at(pal, 0.4), 0.2), u, seed=9)

        # the waveform ribbon, drifting across the sky
        xs, env = self.envelope(frame.time, w)
        yc = hy * 0.47
        amp = h * 0.17 * env * (0.92 + 0.08 * frame.beat) + max(1.5, u * 0.002)
        sky_c = sky_at(pal, yc / hy)
        cx = w / 2
        split = int(np.searchsorted(xs, cx))
        a_now = float(np.interp(cx, xs, amp))
        for lo, hi, col, a, edge in ((0, split, c1, 205, lerp(c1, (255, 255, 255), 0.45)),
                                     (split, len(xs), c2, 95, lerp(sky_c, c2, 0.75))):
            seg_x = xs[lo:hi].tolist()
            seg_a = amp[lo:hi].tolist()
            if lo == 0:
                seg_x, seg_a = seg_x + [cx], seg_a + [a_now]
            else:
                seg_x, seg_a = [cx] + seg_x, [a_now] + seg_a
            if len(seg_x) < 2:
                continue
            upper = [(x, yc - a_) for x, a_ in zip(seg_x, seg_a)]
            lower = [(x, yc + a_) for x, a_ in zip(seg_x, seg_a)]
            gfx.filled_polygon(target, upper + lower[::-1], (*col, a))
            pygame.draw.aalines(target, edge, False, upper)
            pygame.draw.aalines(target, edge, False, lower)
        # playhead: a thin column of light joining the ribbon to the sun
        beam = lerp(sky_c, (255, 250, 236), 0.55)
        aa_line(target, beam, (cx, yc + a_now + u * 0.01), (cx, sy - u * 0.08), max(1.0, u / 700))
        for sgn in (-1, 1):
            aa_circle(target, (255, 252, 240), (cx, yc + sgn * a_now), max(2, u * 0.005))
        if glow_on is not None:
            step = max(1, len(xs) // 60)
            for i in range(0, len(xs), step):
                x = xs[i]
                if x < -u * 0.05 or x > w + u * 0.05:
                    continue
                played = x < cx
                add_glow(light, (x * k, yc * k), (amp[i] + u * 0.02) * k * 1.1, c1 if played else c2,
                         (0.30 if played else 0.12) * (0.4 + env[i]))
            add_glow(light, (cx * k, yc * k), (a_now + u * 0.05) * k, (255, 245, 225), 0.45 + 0.4 * frame.beat)

        # far shore with a tree line, then everything above is mirrored in the lake
        shore_h = int(h * 0.11)
        target.blit(treeline_surface(w, shore_h, land(pal, 0.5), 61, 1.1, 0.4), (0, hy - shore_h + 1))
        reflect(target, hy, self.clock, pal, u, 0.6 + 0.8 * self.calm)
        if light is not None:
            # the sun's long reflection on the water
            for j in range(6):
                fy = hy + (h - hy) * (0.05 + j * 0.16)
                add_glow(light, (sx * k, fy * k), u * k * (0.06 + j * 0.02), pal.halo,
                         (0.30 - j * 0.04) * (0.6 + 0.4 * bass))

        # glitter on the water under the sun
        flick = np.clip(np.sin(frame.time * self.glitter_f + self.glitter_ph) * 1.5 - 0.3 + treble, 0, 1)
        gc = lerp(pal.halo, pal.sun, 0.6)
        for x, y, f in zip((sx + self.glitter_x * w).tolist(),
                           (hy + 2 + self.glitter_y * (h - hy) * 0.9).tolist(), flick.tolist()):
            if f < 0.08:
                continue
            ln = u * (0.004 + 0.012 * f) * (0.4 + (y - hy) / (h - hy))
            gfx.hline(target, int(x - ln), int(x + ln), int(y), (*gc, int(120 * f)))
            if glow_on is not None and f > 0.5:
                add_glow(light, (x * k, y * k), u * k * 0.012, pal.halo, f * 0.5)

        # rings spreading across the water on each beat
        if self.analyzer is not None:
            for age in self.analyzer.recent_beats(frame.time, 3.0).tolist():
                bt = frame.time - age
                r_ = np.random.default_rng(int(bt * 1000) % 2_000_000)
                fx, fy = r_.uniform(0.15, 0.85), r_.uniform(0.1, 0.75)
                x, y = fx * w, hy + fy * (h - hy)
                persp = 0.16 + 0.3 * fy
                fade = 1 - age / 3.0
                for ring in (1.0, 0.62):
                    rx = age * u * 0.11 * ring * (0.5 + fy)
                    if rx < 2:
                        continue
                    col = (*lerp(pal.water, (255, 255, 255), 0.55), int(130 * fade * ring))
                    gfx.aaellipse(target, int(x), int(y), int(rx), max(1, int(rx * persp)), col)

        self.draw_reeds(target, w, h, u, lerp(pal.near, (0, 0, 0), 0.2))
        self.flies.draw(target, glow_on, k, w, h, self.clock, 0.3 + 0.7 * treble, frame.beat * 0.5, c1, c2, u, 0.8)
        if s.atmosphere:
            fog(target, hy - h * 0.025, h * 0.05, pal.haze, 0.35)

    def draw_reeds(self, target, w, h, u, color):
        width = max(1.5, u * 0.0035)
        for i, (fx, fh, fp, fk) in enumerate(self.reeds.tolist()):
            left = i % 2 == 0
            x = (fx * 0.16 if left else 1 - fx * 0.16) * w
            height = h * (0.12 + 0.16 * fh)
            sway = (math.sin(self.clock * 1.4 + fp * TAU) * 0.6 + math.sin(self.clock * 0.6 + i) * 0.4) \
                * height * (0.04 + 0.08 * self.calm)
            tip = (x + sway + (fk - 0.5) * height * 0.25, h - height)
            mid = (x + sway * 0.35 + (fk - 0.5) * height * 0.08, h - height * 0.5)
            aa_line(target, color, (x, h + 2), mid, width)
            aa_line(target, color, mid, tip, width * 0.8)
            if fk > 0.55:   # a cattail head
                hx, hy_ = mid[0] + (tip[0] - mid[0]) * 0.75, mid[1] + (tip[1] - mid[1]) * 0.75
                pygame.draw.ellipse(target, color, (int(hx - width * 1.6), int(hy_ - height * 0.06),
                                                    max(2, int(width * 3.2)), int(height * 0.12)))


# --------------------------------------------------------------- Starlight

class StarlightRenderer(SceneRenderer):
    style = "circular"
    horizon = 0.80

    def __init__(self):
        super().__init__()
        self.stars = StarField(1100, seed=7)
        self.flies = Fireflies(26, seed=23, region=(0.0, 1.0, 0.82, 0.98))
        self.orbit = [0.0, 2.1, 4.0]
        self.spin = 0.0

    def reset(self):
        super().reset()
        self.orbit = [0.0, 2.1, 4.0]
        self.spin = 0.0

    def aurora(self, light, k, hy, mids, c_low, c_high):
        lw, lh = light.get_size()
        sw, sh = max(8, lw // 2), max(8, lh // 2)
        xs = np.linspace(0, 1, sw, dtype=np.float32)[:, None]
        ys = np.arange(sh, dtype=np.float32)[None, :]
        t = self.clock
        base = hy * k / 2 * (0.42 + 0.07 * np.sin(xs * 5.1 + t * 0.25) + 0.04 * np.sin(xs * 11.3 - t * 0.4))
        curtain = (0.5 + 0.5 * np.sin(xs * 23 + t * 0.7 + 2 * np.sin(xs * 4 - t * 0.3))) ** 2
        curtain = curtain * (0.35 + 0.65 * (0.5 + 0.5 * np.sin(xs * 3.3 - t * 0.2)))
        above = base - ys
        up = np.exp(-np.clip(above, 0, None) / (sh * 0.22)) * (above >= 0)
        down = np.exp(-np.clip(-above, 0, None) / (sh * 0.012)) * (above < 0)
        v = (up + down) * curtain * (0.25 + 0.9 * mids)
        mixk = np.clip(above / (sh * 0.25), 0, 1)[..., None]
        lo, hi = np.array(c_low, np.float32), np.array(c_high, np.float32)
        rgb = (lo + (hi - lo) * mixk) * v[..., None] * 0.55
        add_scaled(light, rgb)

    def draw(self, target, light, k, frame, s, dt):
        w, h = target.get_size()
        u = min(w, h)
        pal = night_version(palette(s.background), 0.65)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        lv = self.levels(frame, s, dt)
        playing = self.tick(frame, dt)
        bass, mid, treble = self.bands3(lv)
        hy = h * self.horizon
        glow_on = light if s.glow else None
        if playing:
            self.spin += dt * (0.012 + 0.03 * frame.volume)
            for i, band in enumerate((bass, mid, treble)):
                self.orbit[i] += dt * (0.25 + 0.5 * i) * (0.35 + 2.2 * band)

        target.blit(sky_surface(pal, (w, h), self.horizon), (0, 0))
        self.stars.draw(target, glow_on, k, w, h, hy, frame.time, max(pal.stars, 0.6), treble, u,
                        rotation=self.spin, pole=(w * 0.5, -h * 0.12))
        if light is not None and s.atmosphere:
            self.aurora(light, k, hy, mid, c2, c1)
        self.shooting_stars(target, glow_on, k, w, h, u, frame.time)

        mx, my = w * 0.5, h * 0.40
        R = u * 0.085 * (1 + 0.04 * bass)
        orbits = [(R * 2.1, 0.30, c1, bass), (R * 2.9, 0.26, c2, mid), (R * 3.8, 0.22, (255, 246, 228), treble)]
        # orbit paths, and the planets that are behind the moon
        for (rx, sq, col, band), ang in zip(orbits, self.orbit):
            gfx.aaellipse(target, int(mx), int(my), int(rx), int(rx * sq), (*lerp(col, (255, 255, 255), 0.3), 50))
            if math.sin(ang) < 0:
                self.planet(target, glow_on, k, mx, my, rx, sq, ang, col, band, u)

        # radial spectrum crown (mirrored: bass at the top, treble at the bottom)
        n = len(lv)
        r0 = R * 1.32
        half_w = r0 * math.pi / (2 * n) * 0.8
        for side in (-1, 1):
            for i in range(n):
                a = -math.pi / 2 + side * (i + 0.5) / n * math.pi
                L = (0.015 + lv[i]) * u * 0.20
                ca, sa = math.cos(a), math.sin(a)
                col = lerp(c1, c2, i / (n - 1))
                b0 = (mx + ca * r0 - sa * half_w, my + sa * r0 + ca * half_w)
                b1 = (mx + ca * r0 + sa * half_w, my + sa * r0 - ca * half_w)
                tip = (mx + ca * (r0 + L), my + sa * (r0 + L))
                aa_poly(target, (*col, 215), [b0, tip, b1])
                if glow_on is not None and i % 2 == 0:
                    pygame.draw.line(light, scale(col, 0.15 + 0.4 * lv[i]), (b0[0] * k, b0[1] * k),
                                     (tip[0] * k, tip[1] * k), max(1, int(u * k * 0.006)))

        # halo rings
        for rr, a in ((R * 1.18, 120), (R * 1.75 + frame.beat * R * 0.25, 70)):
            gfx.aacircle(target, int(mx), int(my), int(rr), (*lerp(pal.halo, (255, 255, 255), 0.5),
                                                             int(a * (0.6 + 0.4 * frame.beat))))
        self.sun(target, light, k, (mx, my), R * 1.22, pal, 0.5 * bass + 0.5 * frame.beat, u)
        if glow_on is not None:
            add_glow(light, (mx * k, my * k), R * 4.5 * k, c1, 0.08 + 0.22 * frame.beat)

        for (rx, sq, col, band), ang in zip(orbits, self.orbit):
            if math.sin(ang) >= 0:
                self.planet(target, glow_on, k, mx, my, rx, sq, ang, col, band, u)

        # ground: hills, a lone tree and a cabin with a warm window
        xs = np.linspace(0, 1, 140)
        X = xs * w
        back = hy - h * (0.02 + 0.06 * ridge(xs, 77, 3))
        front = h * (0.90 - 0.07 * ridge(xs, 88, 3)) - h * 0.05 * np.exp(-((xs - 0.78) / 0.12) ** 2)
        fill_profile(target, land(pal, 0.55), X, back, front.max() + 2)
        if s.atmosphere:
            fog(target, hy - h * 0.06, h * 0.08, pal.haze, 0.4)
        near = land(pal, 1.0)
        fill_profile(target, near, X, front, h)
        tx = 0.80
        ty = float(np.interp(tx, xs, front)) + 3
        sway = math.sin(self.clock * 1.1) * h * 0.006 * (0.4 + self.calm)
        round_tree(target, tx * w, ty, h * 0.2, near, sway, seed=5)
        self.cabin(target, light, k, 0.20 * w, float(np.interp(0.20, xs, front)) + 2, u, near, frame)
        self.flies.draw(target, glow_on, k, w, h, self.clock, 0.4 + 0.6 * treble, frame.beat * 0.4, c1, c2, u, 0.7)

    def planet(self, target, light, k, mx, my, rx, sq, ang, col, band, u):
        x, y = mx + math.cos(ang) * rx, my + math.sin(ang) * rx * sq
        r = u * (0.006 + 0.008 * band)
        aa_circle(target, lerp(col, (255, 255, 255), 0.35), (x, y), r)
        if light is not None:
            add_glow(light, (x * k, y * k), (r * 4 + u * 0.02) * k, col, 0.5 + 0.5 * band)

    def cabin(self, target, light, k, x, base, u, color, frame):
        cw, ch = u * 0.075, u * 0.045
        pygame.draw.rect(target, color, (int(x - cw / 2), int(base - ch), int(cw), int(ch) + 2))
        aa_poly(target, color, [(x - cw * 0.62, base - ch + 1), (x, base - ch - cw * 0.42),
                                (x + cw * 0.62, base - ch + 1)])
        pygame.draw.rect(target, color, (int(x + cw * 0.18), int(base - ch - cw * 0.42), int(cw * 0.12),
                                         int(cw * 0.3)))
        flicker = 0.85 + 0.15 * math.sin(frame.time * 9.0) * math.sin(frame.time * 3.7)
        warm = scale((255, 196, 120), flicker)
        win = pygame.Rect(int(x - cw * 0.28), int(base - ch * 0.72), int(cw * 0.2), int(ch * 0.34))
        pygame.draw.rect(target, warm, win)
        pygame.draw.line(target, color, win.midtop, win.midbottom, max(1, int(u / 600)))
        if light is not None:
            add_glow(light, (win.centerx * k, win.centery * k), u * k * 0.09, (255, 170, 90), 0.5 * flicker)

    def shooting_stars(self, target, light, k, w, h, u, t):
        if self.analyzer is None:
            return
        for age in self.analyzer.recent_beats(t, 1.0).tolist():
            bt = t - age
            rng = np.random.default_rng(int(bt * 997) % 3_000_000)
            if rng.random() > 0.35:
                continue
            x0, y0 = rng.uniform(0.1, 0.9) * w, rng.uniform(0.04, 0.3) * h
            a = rng.uniform(2.55, 2.85) if x0 > w / 2 else rng.uniform(0.3, 0.6)
            prog = age / 0.85
            if prog >= 1:
                continue
            L = u * 0.45
            dx, dy = math.cos(a), math.sin(a)
            hx, hy_ = x0 + dx * L * prog, y0 + dy * L * prog
            tail = u * 0.13 * (1 - 0.4 * prog)
            fade = math.sin(prog * math.pi)
            for j in range(5):
                f0, f1 = j / 5, (j + 1) / 5
                aa_line(target, (255, 250, 240, int(230 * fade * (1 - f0))),
                        (hx - dx * tail * f0, hy_ - dy * tail * f0), (hx - dx * tail * f1, hy_ - dy * tail * f1),
                        max(1.0, u / 700 * (1.6 - f0)))
            if light is not None:
                add_glow(light, (hx * k, hy_ * k), u * k * 0.03, (220, 230, 255), fade)


# ---------------------------------------------------------------- Windmills

def windmill(surf, x, base, height, angle, body, sail, trim, u) -> None:
    """A tower windmill with four lattice sails at *angle*."""
    tb, tt = height * 0.15, height * 0.085
    top = base - height
    aa_poly(surf, body, [(x - tb, base + 2), (x - tt, top), (x + tt, top), (x + tb, base + 2)])
    aa_poly(surf, body, [(x - tt * 1.35, top + 1), (x, top - height * 0.12), (x + tt * 1.35, top + 1)])
    door_w = tb * 0.55
    pygame.draw.rect(surf, lerp(body, (0, 0, 0), 0.35),
                     (int(x - door_w / 2), int(base - height * 0.16), max(1, int(door_w)), int(height * 0.16)))
    hub = (x, top - height * 0.01)
    L = height * 0.78
    spar_w = max(1.0, height * 0.018)
    for i in range(4):
        a = angle + i * math.pi / 2
        ca, sa = math.cos(a), math.sin(a)

        def P(r, off):
            return (hub[0] + ca * r - sa * off, hub[1] + sa * r + ca * off)

        quad = [P(L * 0.16, 0), P(L, 0), P(L, L * 0.17), P(L * 0.16, L * 0.12)]
        gfx.filled_polygon(surf, quad, (*sail, 170))
        pygame.draw.aalines(surf, trim, True, quad)
        for f in (0.37, 0.58, 0.79):
            pygame.draw.aaline(surf, trim, P(L * f, 0), P(L * f, L * (0.12 + 0.05 * (f - 0.16) / 0.84)))
        aa_line(surf, trim, hub, P(L * 1.02, 0), spar_w)
    aa_circle(surf, trim, hub, max(2, height * 0.035))


class WindmillsRenderer(SceneRenderer):
    style = "meters"
    horizon = 0.60

    def __init__(self):
        super().__init__()
        self.clouds = CloudLayer(7, seed=33, y_range=(0.05, 0.5), size_range=(0.16, 0.34))
        self.stars = StarField(160, seed=13)
        self.motes = Motes(40, seed=15, region=(0.0, 1.0, 0.3, 0.9))
        self.grass = Grass(120, seed=41, width=1.25)
        self.angles = [0.0, 1.0, 2.0]
        self.flowers = np.random.default_rng(42).random((38, 3))

    def reset(self):
        super().reset()
        self.angles = [0.0, 1.0, 2.0]

    def draw(self, target, light, k, frame, s, dt):
        w, h = target.get_size()
        u = min(w, h)
        pal = palette(s.background)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        lv = self.levels(frame, s, dt)
        playing = self.tick(frame, dt)
        bass, mid, treble = self.bands3(lv)
        hy = h * self.horizon
        glow_on = light if s.glow else None
        if playing:
            for i, band in enumerate((treble, mid, bass)):
                self.angles[i] += dt * (0.25 + 5.5 * band ** 1.3) * (1.25 - 0.15 * i)

        target.blit(sky_surface(pal, (w, h), self.horizon), (0, 0))
        self.stars.draw(target, glow_on, k, w, h, hy, frame.time, pal.stars, treble, u)
        # the sun (or moon) crosses the sky over the course of the song
        dur = self.analyzer.duration if self.analyzer is not None else 1.0
        p = clamp01(frame.time / max(dur, 1e-6))
        arc = 0.06 + 0.88 * p
        sx, sy = w * arc, hy - math.sin(math.pi * arc) * hy * 0.66
        self.sun(target, light, k, (sx, sy), u * 0.055 * (1 + 0.05 * bass), pal, 0.5 * bass + 0.5 * frame.beat, u)
        if light is not None and s.atmosphere:
            god_rays(light, (sx * k, sy * k), u * 1.1 * k, pal.halo, 0.16 + 0.22 * bass + 0.12 * frame.beat,
                     self.clock * 0.05, floor=hy * k, seed=12)
        self.clouds.draw(target, hy, self.clock * (1 + 0.6 * self.calm), pal)
        draw_birds(target, w, h, frame.time, lerp(pal.near, sky_at(pal, 0.4), 0.25), u, seed=3)

        xs = np.linspace(0, 1, 360)
        X = xs * w
        mtn = hy - h * (0.04 + 0.16 * ridge(xs + self.clock * 0.001, 505, 5, sharp=0.5))
        hill1 = hy - h * (0.01 + 0.07 * ridge(xs + self.clock * 0.003, 606, 3))
        hill2 = h * (0.75 - 0.10 * ridge(xs, 707, 3))
        hill3 = h * (0.92 - 0.07 * ridge(xs, 808, 3)) - h * 0.10 * np.exp(-((xs - 0.84) / 0.16) ** 2)
        fill_profile(target, land(pal, 0.12), X, mtn, hill1.max() + 2)
        fog(target, hy - h * 0.12, h * 0.14, pal.haze, 0.55)

        # rolling hills, with the far windmill (treble)
        col1 = grassy(pal, 0.40, 0.35)
        fill_profile(target, col1, X, hill1, hill2.max() + 2)
        self.mill(target, 0.70, xs, hill1, w, h * 0.16, self.angles[0], col1, pal, c1, u, 0.40)
        fog(target, hy - h * 0.01, h * 0.10, pal.haze, 0.35)

        # middle hill, windmill driven by the mids
        col2 = grassy(pal, 0.68, 0.4)
        fill_profile(target, col2, X, hill2, hill3.max() + 2)
        self.mill(target, 0.26, xs, hill2, w, h * 0.27, self.angles[1], col2, pal, c1, u, 0.68)
        pygame.draw.aalines(target, lerp(col2, pal.halo, 0.35), False, np.column_stack((X, hill2)).tolist())

        # foreground meadow with the big windmill (bass), grass and flowers
        col3 = grassy(pal, 0.95, 0.35)
        fill_profile(target, col3, X, hill3, h)
        self.mill(target, 0.84, xs, hill3, w, h * 0.42, self.angles[2], col3, pal, c1, u, 0.95)
        base = np.interp(self.grass.x, xs, hill3)
        self.grass.draw(target, w, base, h * 0.07, self.clock, 0.3 + 0.7 * self.calm + 0.3 * frame.beat,
                        col3, lerp(col3, pal.grass, 0.5), u, rim=lerp(pal.grass, pal.halo, 0.5))
        for fx, fh, fc in self.flowers.tolist():
            x = fx * w
            y = float(np.interp(fx, xs, hill3)) - h * 0.03 * (0.4 + fh) \
                + math.sin(self.clock * 1.7 - fx * 9) * u * 0.002
            col = c1 if fc < 0.5 else c2
            aa_circle(target, lerp(col, (255, 255, 255), 0.25), (x, y), max(1.5, u * 0.0035))
            if glow_on is not None and fc > 0.7:
                add_glow(light, (x * k, y * k), u * k * 0.015, col, 0.3 + 0.5 * treble)

        if s.atmosphere:
            self.motes.draw(target, light, k, w, h, self.clock, frame.time, lerp(pal.halo, (255, 255, 255), 0.4),
                            u, 0.5 + treble)

    @staticmethod
    def mill(target, fx, xs, hill, w, height, angle, ground, pal, accent, u, depth):
        base = float(np.interp(fx, xs, hill)) + height * 0.02
        body = lerp(ground, lerp(pal.haze, (238, 226, 210), 0.5), 0.35 + 0.25 * depth)
        sail = lerp(body, (250, 244, 232), 0.5)
        trim = lerp(body, lerp(accent, (60, 40, 30), 0.3), 0.55)
        windmill(target, fx * w, base, height, angle, body, sail, trim, u)


# ---------------------------------------------------------------- Fireflies

class FirefliesRenderer(SceneRenderer):
    style = "particles"
    horizon = 0.66

    def __init__(self):
        super().__init__()
        self.stars = StarField(220, seed=29)
        self.flies = Fireflies(120, seed=17, region=(0.0, 1.0, 0.22, 0.95))
        self.grass = Grass(150, seed=37, width=1.5)
        self.bokeh = np.random.default_rng(91).random((16, 5))

    def draw(self, target, light, k, frame, s, dt):
        w, h = target.get_size()
        u = min(w, h)
        pal = night_version(palette(s.background), 0.45)
        c1, c2 = s.primary_rgb, s.secondary_rgb
        lv = self.levels(frame, s, dt)
        self.tick(frame, dt)
        bass, mid, treble = self.bands3(lv)
        hy = h * self.horizon
        glow_on = light if s.glow else None

        target.blit(sky_surface(pal, (w, h), self.horizon), (0, 0))
        self.stars.draw(target, glow_on, k, w, h, hy, frame.time, max(pal.stars, 0.35), treble, u)
        if pal.moon:
            self.sun(target, light, k, (w * 0.82, h * 0.16), u * 0.04, pal, 0.3 * frame.beat, u)
        elif light is not None:
            add_glow(light, (w * 0.5 * k, hy * k), u * 0.7 * k, pal.halo, 0.45 + 0.2 * bass, power=2.4)

        # out-of-focus lights drifting in the distance
        if light is not None and s.atmosphere:
            for i, (bx, by, br, bs, bc) in enumerate(self.bokeh.tolist()):
                x = ((bx + self.clock * (0.004 + 0.01 * bs)) % 1.2 - 0.1) * w
                y = (0.15 + 0.6 * by) * h + math.sin(self.clock * 0.5 + i) * u * 0.02
                col = (c1, c2, pal.halo)[int(bc * 3) % 3]
                add_glow(light, (x * k, y * k), u * k * (0.03 + 0.06 * br), col,
                         (0.10 + 0.16 * mid) * (0.6 + 0.4 * math.sin(self.clock * 0.8 + i * 1.7)), power=0)

        tl_h = int(h * 0.16)
        target.blit(treeline_surface(w, tl_h, land(pal, 0.45), 71, 1.4, 0.3), (0, int(hy - tl_h + 2)))
        fog(target, hy - h * 0.05, h * 0.09, pal.haze, 0.5)
        xs = np.linspace(0, 1, 140)
        meadow = hy + h * (0.06 - 0.05 * ridge(xs, 99, 3))
        near_meadow = h * (0.86 - 0.05 * ridge(xs + 0.3, 98, 3))
        fill_profile(target, land(pal, 0.62), xs * w, meadow, near_meadow.max() + 2)
        if s.atmosphere:
            fog(target, hy + h * 0.02, h * 0.10, pal.haze, 0.22)
        fill_profile(target, land(pal, 0.86), xs * w, near_meadow, h)

        # foreground grass: blade height follows the spectrum (bass left, treble right)
        band = np.clip((self.grass.x * len(lv)).astype(int), 0, len(lv) - 1)
        heights = h * (0.05 + 0.30 * lv[band] ** 1.1) * self.grass.h
        near = land(pal, 1.0)
        pygame.draw.rect(target, near, (0, int(h * 0.96), w, h - int(h * 0.96) + 1))
        self.grass.draw(target, w, h * 0.965, 0, self.clock, 0.3 + 0.7 * self.calm, near,
                        lerp(near, c1, 0.3), u, heights=heights, rim=c1, rim_levels=lv[band])

        activity = clamp01(0.25 * frame.volume + 0.85 * treble)
        self.flies.draw(target, glow_on, k, w, h, self.clock, activity, frame.beat, c1, c2, u)

        # a couple of big, blurry fireflies right in front of the lens
        if light is not None and s.atmosphere:
            for i in range(3):
                x = (0.15 + 0.35 * i + 0.06 * math.sin(self.clock * 0.23 + i * 2)) * w
                y = (0.55 + 0.25 * math.sin(self.clock * 0.17 + i * 1.3)) * h
                b = clamp01(math.sin(self.clock * 0.6 + i * 2.2) * 1.2 + 0.2) * (0.5 + 0.5 * activity)
                add_glow(light, (x * k, y * k), u * k * 0.07, c1 if i % 2 else c2, 0.22 * b, power=0)


RENDERERS = {cls.style: cls for cls in (MountainsRenderer, LakesideRenderer, StarlightRenderer,
                                        WindmillsRenderer, FirefliesRenderer)}


def create_renderer(style: str) -> BaseRenderer:
    try:
        return RENDERERS[style]()
    except KeyError:
        raise ValueError(f"Unknown visualizer style: {style}") from None

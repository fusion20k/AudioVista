"""Vintage film look: warm tint + vignette, animated film grain, dust and scratches.

Everything is derived from the playback time, so exported videos get exactly
the same grain as the preview.
"""

from __future__ import annotations

import numpy as np
import pygame

_GRAIN_TILE = 256
_grain: tuple[pygame.Surface, pygame.Surface] | None = None
_vignettes: dict = {}


def _grain_tiles() -> tuple[pygame.Surface, pygame.Surface]:
    """Two tiles of film grain: one brightens, one darkens (signed noise)."""
    global _grain
    if _grain is None:
        rng = np.random.default_rng(42)
        noise = rng.normal(0, 1, (_GRAIN_TILE, _GRAIN_TILE)).astype(np.float32)
        # slightly clumpy grain looks more like film than per-pixel noise
        noise = (noise + np.roll(noise, 1, 0) * 0.5 + np.roll(noise, 1, 1) * 0.5) / 1.6
        amp = 9.0
        up = np.clip(noise * amp, 0, 255).astype(np.uint8)
        down = np.clip(-noise * amp, 0, 255).astype(np.uint8)
        tiles = []
        for arr in (up, down):
            surf = pygame.Surface((_GRAIN_TILE, _GRAIN_TILE))
            pygame.surfarray.blit_array(surf, np.repeat(arr[..., None], 3, axis=2))
            tiles.append(surf)
        _grain = (tiles[0], tiles[1])
    return _grain


def _vignette(size: tuple[int, int]) -> pygame.Surface:
    """Multiply mask: warm cream in the centre fading to dark brown corners."""
    surf = _vignettes.get(size)
    if surf is None:
        w, h = size
        xx, yy = np.mgrid[0:w, 0:h].astype(np.float32)
        d = np.sqrt(((xx - w / 2) / (w / 2)) ** 2 + ((yy - h / 2) / (h / 2)) ** 2) / 1.414
        fall = np.clip(1.0 - 0.62 * d ** 2.2, 0.0, 1.0)[..., None]
        tint = np.array([255, 241, 222], np.float32)     # warm, slightly faded highlights
        rgb = tint * fall
        surf = pygame.surfarray.make_surface(rgb.astype(np.uint8))
        if len(_vignettes) > 6:
            _vignettes.clear()
        _vignettes[size] = surf
    return surf


def apply(target: pygame.Surface, t: float) -> None:
    w, h = target.get_size()
    target.blit(_vignette((w, h)), (0, 0), special_flags=pygame.BLEND_RGB_MULT)

    frame = int(t * 24)  # film runs at 24 fps
    rng = np.random.default_rng(frame * 7919 + 13)
    _up, down = _grain_tiles()
    ox, oy = rng.integers(0, _GRAIN_TILE, 2)
    # One subtractive pass (the vignette already lifts the mid-tones slightly)
    for y in range(-int(oy), h, _GRAIN_TILE):
        for x in range(-int(ox), w, _GRAIN_TILE):
            target.blit(down, (x, y), special_flags=pygame.BLEND_RGB_SUB)

    # Dust specks and the occasional scratch
    unit = max(1, min(w, h) // 400)
    for _ in range(int(rng.integers(0, 4))):
        x, y = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(1, 3)) * unit
        shade = int(rng.integers(150, 230))
        pygame.draw.circle(target, (shade, shade - 12, shade - 30), (x, y), r)
    if rng.random() < 0.12:
        x = int(rng.integers(int(w * 0.1), int(w * 0.9)))
        y0, y1 = sorted(int(v) for v in rng.integers(0, h, 2))
        pygame.draw.line(target, (205, 190, 165), (x, y0), (x + int(rng.integers(-3, 4)), y1), 1)

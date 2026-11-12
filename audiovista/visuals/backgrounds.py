"""Background presets drawn behind the visualizer: soft skies at different times of day."""

from __future__ import annotations

import numpy as np
import pygame

# name: colour stops from the top of the frame (0) to the bottom (1)
BACKGROUND_COLORS = {
    "golden": ((0.0, (64, 92, 158)), (0.55, (196, 150, 160)), (0.85, (246, 186, 140)), (1.0, (255, 210, 150))),
    "sunset": ((0.0, (26, 30, 74)), (0.45, (96, 60, 120)), (0.80, (214, 102, 104)), (1.0, (250, 156, 96))),
    "dawn": ((0.0, (88, 116, 176)), (0.55, (172, 160, 206)), (0.85, (236, 184, 192)), (1.0, (252, 214, 196))),
    "twilight": ((0.0, (10, 13, 36)), (0.5, (34, 34, 86)), (0.85, (104, 72, 128)), (1.0, (176, 106, 120))),
    "night": ((0.0, (4, 6, 18)), (0.55, (11, 19, 46)), (0.9, (24, 38, 78)), (1.0, (40, 58, 102))),
}

_cache: dict = {}
_flash: dict = {}


def _gradient(size: tuple[int, int], stops) -> pygame.Surface:
    w, h = size
    ys = np.linspace(0, 1, h, dtype=np.float32)
    pos = np.array([p for p, _ in stops], np.float32)
    cols = np.array([c for _, c in stops], np.float32)
    col = np.stack([np.interp(ys, pos, cols[:, i]) for i in range(3)], axis=1)   # (h, 3)
    # a little noise ("dithering") so the smooth gradient never shows bands
    noise = np.random.default_rng(7).random((w, h), dtype=np.float32) - 0.5
    rgb = np.clip(col[None, :, :] + noise[..., None] * 1.8, 0, 255).astype(np.uint8)
    surf = pygame.Surface(size)
    pygame.surfarray.blit_array(surf, rgb)
    return surf


def get_background(name: str, size: tuple[int, int]) -> pygame.Surface:
    key = (name, size)
    surf = _cache.get(key)
    if surf is None:
        surf = _gradient(size, BACKGROUND_COLORS.get(name, BACKGROUND_COLORS["sunset"]))
        if len(_cache) > 16:
            _cache.clear()
        _cache[key] = surf
    return surf


def draw_background(target: pygame.Surface, name: str, color, beat: float) -> None:
    target.blit(get_background(name, target.get_size()), (0, 0))
    if beat > 0.02:
        # A plain fill + additive blit is far faster than fill(special_flags=ADD).
        size = target.get_size()
        flash = _flash.get(size)
        if flash is None:
            _flash.clear()
            flash = _flash[size] = pygame.Surface(size)
        flash.fill(tuple(int(c * 0.16 * beat) for c in color))
        target.blit(flash, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

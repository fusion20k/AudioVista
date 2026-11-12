"""Background presets drawn behind the visualizer."""

from __future__ import annotations

import numpy as np
import pygame

BACKGROUND_COLORS = {
    # name: (top colour, bottom colour)
    "midnight": ((14, 18, 46), (2, 3, 10)),
    "black": ((0, 0, 0), (0, 0, 0)),
    "nebula": ((30, 10, 48), (4, 2, 14)),
    "sunset": ((58, 16, 52), (14, 6, 24)),
    "ocean": ((2, 40, 56), (1, 8, 16)),
}

_cache: dict = {}
_flash: dict = {}


def _gradient(size: tuple[int, int], top, bottom, radial: bool) -> pygame.Surface:
    w, h = size
    # pygame.surfarray uses (width, height, rgb) arrays.
    if radial:
        xx, yy = np.mgrid[0:w, 0:h].astype(np.float32)
        d = np.sqrt(((xx - w / 2) / (w * 0.6)) ** 2 + ((yy - h / 2) / (h * 0.6)) ** 2)
        t = np.clip(d, 0, 1)[..., None]
    else:
        t = np.broadcast_to(np.linspace(0, 1, h, dtype=np.float32)[None, :, None], (w, h, 1))
    top_a = np.array(top, np.float32)
    bot_a = np.array(bottom, np.float32)
    rgb = top_a + (bot_a - top_a) * t
    return pygame.surfarray.make_surface(rgb.astype(np.uint8))


def get_background(name: str, size: tuple[int, int]) -> pygame.Surface:
    key = (name, size)
    surf = _cache.get(key)
    if surf is None:
        top, bottom = BACKGROUND_COLORS.get(name, BACKGROUND_COLORS["midnight"])
        surf = _gradient(size, top, bottom, radial=name in ("nebula", "sunset"))
        if len(_cache) > 16:
            _cache.clear()
        _cache[key] = surf
    return surf


def draw_background(target: pygame.Surface, name: str, color, beat: float) -> None:
    target.blit(get_background(name, target.get_size()), (0, 0))
    if beat > 0.02 and name != "black":
        # A plain fill + additive blit is far faster than fill(special_flags=ADD).
        size = target.get_size()
        flash = _flash.get(size)
        if flash is None:
            _flash.clear()
            flash = _flash[size] = pygame.Surface(size)
        flash.fill(tuple(int(c * 0.22 * beat) for c in color))
        target.blit(flash, (0, 0), special_flags=pygame.BLEND_RGB_ADD)

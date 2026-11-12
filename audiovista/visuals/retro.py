"""The "Haze" look: a soft, slightly cool vignette and a faint warm glow rising
from the bottom of the frame, like the last light of the day. Both are cached,
so it costs two full-frame blits."""

from __future__ import annotations

import numpy as np
import pygame

_vignettes: dict = {}
_glows: dict = {}


def _vignette(size: tuple[int, int]) -> pygame.Surface:
    surf = _vignettes.get(size)
    if surf is None:
        w, h = size
        xx = np.linspace(-1, 1, w, dtype=np.float32)[:, None]
        yy = np.linspace(-1, 1, h, dtype=np.float32)[None, :]
        d = np.sqrt(xx ** 2 * 0.85 + yy ** 2) / 1.36
        f = np.clip(1 - 0.55 * np.clip(d - 0.25, 0, 1) ** 1.7, 0, 1)[..., None]
        edge = np.array((176, 168, 214), np.float32)
        rgb = np.clip((edge + (255 - edge) * f) * (0.55 + 0.45 * f), 0, 255).astype(np.uint8)
        surf = pygame.Surface(size)
        pygame.surfarray.blit_array(surf, rgb)
        if len(_vignettes) > 6:
            _vignettes.clear()
        _vignettes[size] = surf
    return surf


def _glow(size: tuple[int, int]) -> pygame.Surface:
    surf = _glows.get(size)
    if surf is None:
        w, h = size
        yy = np.linspace(0, 1, h, dtype=np.float32)[None, :, None]
        xx = np.linspace(-1, 1, w, dtype=np.float32)[:, None, None]
        f = np.clip((yy - 0.55) / 0.45, 0, 1) ** 2 * (1 - 0.5 * xx ** 2)
        rgb = np.clip(f * np.array((40, 22, 18), np.float32), 0, 255).astype(np.uint8)
        surf = pygame.Surface(size)
        pygame.surfarray.blit_array(surf, rgb)
        if len(_glows) > 6:
            _glows.clear()
        _glows[size] = surf
    return surf


def apply(target: pygame.Surface, t: float = 0.0) -> None:
    size = target.get_size()
    target.blit(_glow(size), (0, 0), special_flags=pygame.BLEND_RGB_ADD)
    target.blit(_vignette(size), (0, 0), special_flags=pygame.BLEND_RGB_MULT)

"""Colours, fonts, DPI scaling, anti-aliased drawing helpers, icons and the logo.

All layout code is written in *logical* pixels and converted with :func:`px`,
so the interface stays sharp on high-DPI displays (e.g. Windows at 150-200%).
Shapes and icons are drawn super-sampled and then smooth-scaled down, which
gives them clean anti-aliased edges; the results are cached.
"""

from __future__ import annotations

import math
from collections import OrderedDict

import numpy as np
import pygame

BG = (11, 13, 21)
PANEL = (18, 21, 33)
PANEL_2 = (26, 30, 46)
PANEL_3 = (34, 39, 58)
BORDER = (44, 50, 72)
TEXT = (233, 236, 246)
TEXT_DIM = (154, 160, 184)
TEXT_MUTED = (102, 108, 134)
ACCENT = (124, 92, 255)
ACCENT_HOVER = (148, 122, 255)
ACCENT_DARK = (70, 52, 150)
CYAN = (0, 212, 255)
DANGER = (255, 92, 122)
SUCCESS = (64, 214, 160)
WARNING = (255, 196, 76)

SUPERSAMPLE = 3

# ------------------------------------------------------------------ scaling

S = 1.0  # UI scale factor (1.0 = 96 DPI)


def set_scale(scale: float) -> None:
    global S
    S = max(1.0, min(4.0, float(scale)))
    _fonts.clear()
    _text_cache.clear()
    _shape_cache.clear()


def px(value: float) -> int:
    """Convert logical pixels to physical pixels."""
    return int(round(value * S))


# -------------------------------------------------------------------- text

_fonts: dict = {}
_text_cache: OrderedDict = OrderedDict()


def font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    f = _fonts.get(key)
    if f is None:
        f = pygame.font.SysFont("segoeui,helveticaneue,arial", px(size), bold=bold)
        _fonts[key] = f
    return f


def render_text(text: str, size: int = 14, color=TEXT, bold: bool = False) -> pygame.Surface:
    key = (text, size, tuple(color), bold)
    surf = _text_cache.get(key)
    if surf is None:
        surf = font(size, bold).render(text, True, color)
        _text_cache[key] = surf
        if len(_text_cache) > 1000:
            _text_cache.popitem(last=False)
    else:
        _text_cache.move_to_end(key)
    return surf


def draw_text(surf: pygame.Surface, text: str, pos, size: int = 14, color=TEXT,
              bold: bool = False, anchor: str = "topleft", max_width: int | None = None) -> pygame.Rect:
    if max_width is not None:
        text = ellipsize(text, size, max_width, bold)
    img = render_text(text, size, color, bold)
    rect = img.get_rect(**{anchor: pos})
    surf.blit(img, rect)
    return rect


def text_width(text: str, size: int = 14, bold: bool = False) -> int:
    return font(size, bold).size(text)[0]


def ellipsize(text: str, size: int, max_width: int, bold: bool = False) -> str:
    if text_width(text, size, bold) <= max_width:
        return text
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if text_width(text[:mid] + "…", size, bold) <= max_width:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo] + "…"


def wrap_text(text: str, size: int, max_width: int, bold: bool = False) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if text_width(trial, size, bold) <= max_width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return lines


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


# ------------------------------------------------------ anti-aliased shapes

_shape_cache: OrderedDict = OrderedDict()


def _cached(key, builder) -> pygame.Surface:
    img = _shape_cache.get(key)
    if img is None:
        img = builder()
        _shape_cache[key] = img
        if len(_shape_cache) > 700:
            _shape_cache.popitem(last=False)
    else:
        _shape_cache.move_to_end(key)
    return img


def _canvas(w: int, h: int, color) -> pygame.Surface:
    big = pygame.Surface((max(1, w), max(1, h)), pygame.SRCALPHA)
    big.fill((*color[:3], 0))  # transparent but same RGB -> no dark fringes
    return big


def rrect(surf: pygame.Surface, color, rect, radius: int = 0, width: int = 0) -> None:
    """Anti-aliased (optionally outlined) rounded rectangle. Sizes are physical px."""
    rect = pygame.Rect(rect)
    if rect.w <= 0 or rect.h <= 0:
        return
    color = tuple(color)
    if radius <= 0 and (len(color) == 3 or color[3] == 255):
        pygame.draw.rect(surf, color, rect, width)
        return
    radius = min(radius, rect.h // 2, rect.w // 2)

    def build():
        k = SUPERSAMPLE
        big = _canvas(rect.w * k, rect.h * k, color)
        pygame.draw.rect(big, color, big.get_rect(), width * k if width else 0, border_radius=radius * k)
        return pygame.transform.smoothscale(big, rect.size)

    surf.blit(_cached(("rr", rect.w, rect.h, radius, color, width), build), rect)


def circle(surf: pygame.Surface, color, center, radius: float, width: int = 0) -> None:
    """Anti-aliased filled circle or ring. Sizes are physical px."""
    if radius <= 0:
        return
    color = tuple(color)
    r = int(math.ceil(radius))
    size = 2 * r + 2

    def build():
        k = SUPERSAMPLE
        big = _canvas(size * k, size * k, color)
        pygame.draw.circle(big, color, ((r + 1) * k, (r + 1) * k), int(radius * k), width * k if width else 0)
        return pygame.transform.smoothscale(big, (size, size))

    img = _cached(("c", round(radius * 4), color, width), build)
    surf.blit(img, (int(center[0]) - r - 1, int(center[1]) - r - 1))


def draw_panel(surf, rect, color=PANEL, radius=12, border=BORDER):
    rrect(surf, color, rect, px(radius))
    if border:
        rrect(surf, border, rect, px(radius), max(1, px(1)))


def shade(surf: pygame.Surface, rect, alpha: int) -> None:
    layer = pygame.Surface(pygame.Rect(rect).size, pygame.SRCALPHA)
    layer.fill((0, 0, 0, alpha))
    surf.blit(layer, rect)


# --------------------------------------------------------------------- icons

def draw_icon(surf: pygame.Surface, name: str, rect, color) -> None:
    """Draw a vector icon (super-sampled for smooth edges)."""
    rect = pygame.Rect(rect)
    if rect.w <= 0 or rect.h <= 0:
        return
    color = tuple(color)

    def build():
        k = 4
        big = _canvas(rect.w * k, rect.h * k, color)
        width = max(2, round(min(rect.w, rect.h) / 9)) * k
        _icon_paths(big, name, big.get_rect(), color, width)
        return pygame.transform.smoothscale(big, rect.size)

    surf.blit(_cached(("icon", name, rect.w, rect.h, color), build), rect)


def _line(s, c, a, b, w):
    pygame.draw.line(s, c, a, b, w)
    pygame.draw.circle(s, c, a, w // 2)
    pygame.draw.circle(s, c, b, w // 2)


def _lines(s, c, pts, w, closed=False):
    pts = list(pts)
    pairs = list(zip(pts, pts[1:])) + ([(pts[-1], pts[0])] if closed else [])
    for a, b in pairs:
        _line(s, c, a, b, w)


def _icon_paths(surf, name, r, color, width):
    cx, cy = r.center
    s = min(r.w, r.h) / 2
    if name == "play":
        pts = [(cx - s * 0.42, cy - s * 0.62), (cx - s * 0.42, cy + s * 0.62), (cx + s * 0.68, cy)]
        pygame.draw.polygon(surf, color, pts)
        _lines(surf, color, pts, width // 2, closed=True)
    elif name == "pause":
        bw = s * 0.34
        for x in (cx - s * 0.48, cx + s * 0.48 - bw):
            pygame.draw.rect(surf, color, (x, cy - s * 0.58, bw, s * 1.16), border_radius=int(bw * 0.35))
    elif name in ("next", "prev"):
        d = 1 if name == "next" else -1
        pts = [(cx - d * s * 0.55, cy - s * 0.55), (cx - d * s * 0.55, cy + s * 0.55), (cx + d * s * 0.3, cy)]
        pygame.draw.polygon(surf, color, pts)
        _lines(surf, color, pts, width // 2, closed=True)
        bx = cx + d * s * 0.48
        _line(surf, color, (bx, cy - s * 0.52), (bx, cy + s * 0.52), width)
    elif name == "plus":
        _line(surf, color, (cx - s * 0.55, cy), (cx + s * 0.55, cy), width)
        _line(surf, color, (cx, cy - s * 0.55), (cx, cy + s * 0.55), width)
    elif name == "close":
        k = s * 0.45
        _line(surf, color, (cx - k, cy - k), (cx + k, cy + k), width)
        _line(surf, color, (cx - k, cy + k), (cx + k, cy - k), width)
    elif name == "upload":
        _line(surf, color, (cx, cy - s * 0.62), (cx, cy + s * 0.2), width)
        _lines(surf, color, [(cx - s * 0.36, cy - s * 0.26), (cx, cy - s * 0.62), (cx + s * 0.36, cy - s * 0.26)], width)
        _lines(surf, color, [(cx - s * 0.65, cy + s * 0.12), (cx - s * 0.65, cy + s * 0.62),
                             (cx + s * 0.65, cy + s * 0.62), (cx + s * 0.65, cy + s * 0.12)], width)
    elif name == "download":
        _line(surf, color, (cx, cy - s * 0.62), (cx, cy + s * 0.22), width)
        _lines(surf, color, [(cx - s * 0.36, cy - s * 0.14), (cx, cy + s * 0.22), (cx + s * 0.36, cy - s * 0.14)], width)
        _line(surf, color, (cx - s * 0.62, cy + s * 0.62), (cx + s * 0.62, cy + s * 0.62), width)
    elif name == "save":
        box = pygame.Rect(0, 0, s * 1.3, s * 1.3)
        box.center = (cx, cy)
        pygame.draw.rect(surf, color, box, width, border_radius=int(s * 0.22))
        pygame.draw.rect(surf, color, (box.x + box.w * 0.27, box.y, box.w * 0.46, box.h * 0.32), width)
        pygame.draw.circle(surf, color, (box.centerx, int(box.y + box.h * 0.66)), int(s * 0.16))
    elif name == "music":
        pygame.draw.circle(surf, color, (int(cx - s * 0.3), int(cy + s * 0.45)), int(s * 0.26))
        _line(surf, color, (cx - s * 0.06, cy + s * 0.45), (cx - s * 0.06, cy - s * 0.6), width)
        _line(surf, color, (cx - s * 0.06, cy - s * 0.6), (cx + s * 0.52, cy - s * 0.36), width)
    elif name == "volume":
        pts = [(cx - s * 0.75, cy - s * 0.25), (cx - s * 0.38, cy - s * 0.25), (cx - s * 0.02, cy - s * 0.62),
               (cx - s * 0.02, cy + s * 0.62), (cx - s * 0.38, cy + s * 0.25), (cx - s * 0.75, cy + s * 0.25)]
        pygame.draw.polygon(surf, color, pts)
        for k in (0.38, 0.7):
            arc = pygame.Rect(0, 0, s * k * 2, s * k * 2)
            arc.center = (cx, cy)
            pygame.draw.arc(surf, color, arc, -math.pi / 3.2, math.pi / 3.2, width)
    elif name == "fullscreen":
        k, m = s * 0.62, s * 0.34
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            x, y = cx + dx * k, cy + dy * k
            _lines(surf, color, [(x - dx * m, y), (x, y), (x, y - dy * m)], width)
    elif name == "trash":
        _line(surf, color, (cx - s * 0.6, cy - s * 0.45), (cx + s * 0.6, cy - s * 0.45), width)
        pygame.draw.rect(surf, color, (cx - s * 0.42, cy - s * 0.32, s * 0.84, s * 0.95), width,
                         border_radius=int(s * 0.12))
        _line(surf, color, (cx - s * 0.16, cy - s * 0.65), (cx + s * 0.16, cy - s * 0.65), width)
    elif name == "logout":
        _lines(surf, color, [(cx + s * 0.05, cy - s * 0.65), (cx - s * 0.62, cy - s * 0.65),
                             (cx - s * 0.62, cy + s * 0.65), (cx + s * 0.05, cy + s * 0.65)], width)
        _line(surf, color, (cx - s * 0.15, cy), (cx + s * 0.68, cy), width)
        _lines(surf, color, [(cx + s * 0.36, cy - s * 0.32), (cx + s * 0.68, cy), (cx + s * 0.36, cy + s * 0.32)], width)
    elif name == "queue":
        for k in (-0.5, 0.0, 0.5):
            _line(surf, color, (cx - s * 0.65, cy + s * k), (cx + (0.15 if k > 0 else 0.6) * s, cy + s * k), width)
        pygame.draw.polygon(surf, color, [(cx + s * 0.35, cy + s * 0.18), (cx + s * 0.35, cy + s * 0.82),
                                          (cx + s * 0.8, cy + s * 0.5)])


# ---------------------------------------------------------------------- logo

# The mark: a gradient app tile with a bold "V" (for Vista) flanked by sound
# waves. Coordinates are fractions of the tile size.
LOGO_V = [(0.355, 0.25), (0.5, 0.76), (0.645, 0.25)]
LOGO_WAVE_LEFT = [(0.13, 0.53), (0.19, 0.53), (0.225, 0.43), (0.265, 0.63), (0.30, 0.53)]
LOGO_WAVE_RIGHT = [(1 - x, y) for x, y in reversed(LOGO_WAVE_LEFT)]


def _logo_tile(size: int) -> pygame.Surface:
    """Gradient squircle background of the logo at *size* px (not cached)."""
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32) / max(1, size - 1)
    t = np.clip((xx * 0.55 + yy * 0.45), 0, 1)[..., None]
    c1 = np.array((132, 86, 255), np.float32)
    c2 = np.array((0, 206, 255), np.float32)
    rgb = c1 + (c2 - c1) * t
    # soft highlight in the top-left corner
    glow = np.clip(1.0 - np.sqrt((xx - 0.2) ** 2 + (yy - 0.15) ** 2) / 0.7, 0, 1)[..., None]
    rgb = np.clip(rgb + glow * 40, 0, 255)
    # mgrid gives (y, x) arrays; surfarray wants (x, y).
    tile = _to_alpha(pygame.surfarray.make_surface(rgb.transpose(1, 0, 2).astype(np.uint8)))
    mask = pygame.Surface((size, size), pygame.SRCALPHA)
    pygame.draw.rect(mask, (255, 255, 255, 255), mask.get_rect(), border_radius=int(size * 0.27))
    tile.blit(mask, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
    return tile


def _to_alpha(surface: pygame.Surface) -> pygame.Surface:
    out = pygame.Surface(surface.get_size(), pygame.SRCALPHA)
    out.blit(surface, (0, 0))
    return out


def render_logo(size: int, tile: bool = True) -> pygame.Surface:
    """The AudioVista logo at *size* x *size* px, anti-aliased.

    ``tile=False`` draws only the gradient wave mark (for use on dark backgrounds).
    """
    k = 4
    big_size = size * k
    if tile:
        big = _logo_tile(big_size)
        stroke_col = (255, 255, 255, 255)
    else:
        big = pygame.Surface((big_size, big_size), pygame.SRCALPHA)
        stroke_col = None
    def pts(path):
        return [(x * big_size, y * big_size) for x, y in path]

    stroke = pygame.Surface((big_size, big_size), pygame.SRCALPHA)
    white = (255, 255, 255, 255)
    _lines(stroke, white, pts(LOGO_V), max(2, int(big_size * 0.115)))
    for wave in (LOGO_WAVE_LEFT, LOGO_WAVE_RIGHT):
        _lines(stroke, white, pts(wave), max(2, int(big_size * 0.06)))
    if stroke_col is None:
        # colour the stroke with the brand gradient
        grad = _logo_tile(big_size)
        grad.blit(stroke, (0, 0), special_flags=pygame.BLEND_RGBA_MIN)
        big.blit(grad, (0, 0))
    else:
        shadow = stroke.copy()
        shadow.fill((40, 20, 110, 90), special_flags=pygame.BLEND_RGBA_MULT)
        big.blit(shadow, (0, int(big_size * 0.025)))
        big.blit(stroke, (0, 0))
    return pygame.transform.smoothscale(big, (size, size))


def draw_logo(surf: pygame.Surface, rect, tile: bool = True) -> None:
    rect = pygame.Rect(rect)
    size = min(rect.w, rect.h)
    img = _cached(("logo", size, tile), lambda: render_logo(size, tile))
    surf.blit(img, img.get_rect(center=rect.center))

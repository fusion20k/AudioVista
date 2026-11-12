"""Colours, fonts, text helpers and vector icons for the UI."""

from __future__ import annotations

import math
from collections import OrderedDict

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
DANGER = (255, 92, 122)
SUCCESS = (64, 214, 160)
WARNING = (255, 196, 76)

_fonts: dict = {}
_text_cache: OrderedDict = OrderedDict()


def font(size: int, bold: bool = False) -> pygame.font.Font:
    key = (size, bold)
    f = _fonts.get(key)
    if f is None:
        f = pygame.font.SysFont("segoeui,helveticaneue,arial", size, bold=bold)
        _fonts[key] = f
    return f


def render_text(text: str, size: int = 14, color=TEXT, bold: bool = False) -> pygame.Surface:
    key = (text, size, color, bold)
    surf = _text_cache.get(key)
    if surf is None:
        surf = font(size, bold).render(text, True, color)
        _text_cache[key] = surf
        if len(_text_cache) > 800:
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


def draw_panel(surf, rect, color=PANEL, radius=12, border=BORDER):
    pygame.draw.rect(surf, color, rect, border_radius=radius)
    if border:
        pygame.draw.rect(surf, border, rect, 1, border_radius=radius)


# ---------------------------------------------------------------------- icons

def draw_icon(surf: pygame.Surface, name: str, rect: pygame.Rect, color, width: int = 2) -> None:
    """Simple vector icons so the app needs no image assets."""
    r = pygame.Rect(rect)
    cx, cy = r.center
    s = min(r.w, r.h) / 2
    if name == "play":
        pts = [(cx - s * 0.45, cy - s * 0.6), (cx - s * 0.45, cy + s * 0.6), (cx + s * 0.65, cy)]
        pygame.draw.polygon(surf, color, pts)
    elif name == "pause":
        bw = max(2, int(s * 0.3))
        pygame.draw.rect(surf, color, (cx - s * 0.45, cy - s * 0.55, bw, s * 1.1), border_radius=2)
        pygame.draw.rect(surf, color, (cx + s * 0.45 - bw, cy - s * 0.55, bw, s * 1.1), border_radius=2)
    elif name in ("next", "prev"):
        d = 1 if name == "next" else -1
        pts = [(cx - d * s * 0.5, cy - s * 0.5), (cx - d * s * 0.5, cy + s * 0.5), (cx + d * s * 0.3, cy)]
        pygame.draw.polygon(surf, color, pts)
        pygame.draw.rect(surf, color, (cx + d * s * 0.35 - (2 if d > 0 else 0), cy - s * 0.5, 3, s))
    elif name == "plus":
        pygame.draw.line(surf, color, (cx - s * 0.5, cy), (cx + s * 0.5, cy), width)
        pygame.draw.line(surf, color, (cx, cy - s * 0.5), (cx, cy + s * 0.5), width)
    elif name == "close":
        k = s * 0.42
        pygame.draw.line(surf, color, (cx - k, cy - k), (cx + k, cy + k), width)
        pygame.draw.line(surf, color, (cx - k, cy + k), (cx + k, cy - k), width)
    elif name == "upload":
        pygame.draw.line(surf, color, (cx, cy - s * 0.6), (cx, cy + s * 0.2), width)
        pygame.draw.lines(surf, color, False, [(cx - s * 0.35, cy - s * 0.25), (cx, cy - s * 0.6),
                                               (cx + s * 0.35, cy - s * 0.25)], width)
        pygame.draw.lines(surf, color, False, [(cx - s * 0.6, cy + s * 0.15), (cx - s * 0.6, cy + s * 0.6),
                                               (cx + s * 0.6, cy + s * 0.6), (cx + s * 0.6, cy + s * 0.15)], width)
    elif name == "download":
        pygame.draw.line(surf, color, (cx, cy - s * 0.6), (cx, cy + s * 0.2), width)
        pygame.draw.lines(surf, color, False, [(cx - s * 0.35, cy - s * 0.15), (cx, cy + s * 0.2),
                                               (cx + s * 0.35, cy - s * 0.15)], width)
        pygame.draw.line(surf, color, (cx - s * 0.6, cy + s * 0.6), (cx + s * 0.6, cy + s * 0.6), width)
    elif name == "save":
        box = pygame.Rect(0, 0, s * 1.3, s * 1.3)
        box.center = (cx, cy)
        pygame.draw.rect(surf, color, box, width, border_radius=3)
        pygame.draw.rect(surf, color, (box.x + box.w * 0.25, box.y, box.w * 0.5, box.h * 0.3), width)
    elif name == "music":
        pygame.draw.circle(surf, color, (int(cx - s * 0.3), int(cy + s * 0.45)), max(2, int(s * 0.25)))
        pygame.draw.line(surf, color, (cx - s * 0.08, cy + s * 0.45), (cx - s * 0.08, cy - s * 0.6), width)
        pygame.draw.line(surf, color, (cx - s * 0.08, cy - s * 0.6), (cx + s * 0.5, cy - s * 0.35), width)
    elif name == "volume":
        pts = [(cx - s * 0.7, cy - s * 0.25), (cx - s * 0.35, cy - s * 0.25), (cx, cy - s * 0.6),
               (cx, cy + s * 0.6), (cx - s * 0.35, cy + s * 0.25), (cx - s * 0.7, cy + s * 0.25)]
        pygame.draw.polygon(surf, color, pts)
        for k in (0.35, 0.65):
            arc = pygame.Rect(0, 0, s * k * 2, s * k * 2)
            arc.center = (cx, cy)
            pygame.draw.arc(surf, color, arc, -math.pi / 3.5, math.pi / 3.5, width)
    elif name == "fullscreen":
        k, m = s * 0.6, s * 0.3
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1)):
            x, y = cx + dx * k, cy + dy * k
            pygame.draw.line(surf, color, (x, y), (x - dx * m, y), width)
            pygame.draw.line(surf, color, (x, y), (x, y - dy * m), width)
    elif name == "trash":
        pygame.draw.line(surf, color, (cx - s * 0.55, cy - s * 0.45), (cx + s * 0.55, cy - s * 0.45), width)
        pygame.draw.rect(surf, color, (cx - s * 0.4, cy - s * 0.35, s * 0.8, s * 0.95), width, border_radius=2)
        pygame.draw.line(surf, color, (cx - s * 0.15, cy - s * 0.6), (cx + s * 0.15, cy - s * 0.6), width)
    elif name == "logout":
        pygame.draw.lines(surf, color, False, [(cx, cy - s * 0.6), (cx - s * 0.6, cy - s * 0.6),
                                               (cx - s * 0.6, cy + s * 0.6), (cx, cy + s * 0.6)], width)
        pygame.draw.line(surf, color, (cx - s * 0.15, cy), (cx + s * 0.65, cy), width)
        pygame.draw.lines(surf, color, False, [(cx + s * 0.35, cy - s * 0.3), (cx + s * 0.65, cy),
                                               (cx + s * 0.35, cy + s * 0.3)], width)
    elif name == "queue":
        for k in (-0.45, 0.0, 0.45):
            pygame.draw.line(surf, color, (cx - s * 0.6, cy + s * k), (cx + s * 0.25, cy + s * k), width)
        pygame.draw.polygon(surf, color, [(cx + s * 0.4, cy + s * 0.15), (cx + s * 0.4, cy + s * 0.75),
                                          (cx + s * 0.8, cy + s * 0.45)])
    elif name == "logo":
        heights = (0.45, 0.8, 1.0, 0.65, 0.9, 0.5)
        bw = r.w / (len(heights) * 1.6)
        for i, hgt in enumerate(heights):
            bh = r.h * hgt
            x = r.x + i * bw * 1.6
            col = tuple(int(a + (b - a) * i / 5) for a, b in zip(ACCENT, (0, 229, 255)))
            pygame.draw.rect(surf, col, (x, r.bottom - bh, bw, bh), border_radius=int(bw / 2))

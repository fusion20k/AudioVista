"""A small set of pygame widgets: buttons, text inputs, sliders, lists...

Rects are in physical pixels (screens lay them out with ``theme.px``); the
widgets scale their own paddings, radii and fonts the same way.
"""

from __future__ import annotations

import time
from typing import Any, Callable

import pygame

from . import theme as T
from .theme import px


def mouse_pos():
    return pygame.mouse.get_pos()


class Widget:
    def __init__(self, rect=None):
        self.rect = pygame.Rect(rect or (0, 0, 0, 0))
        self.visible = True
        self.enabled = True

    @property
    def hovered(self) -> bool:
        return self.visible and self.rect.collidepoint(mouse_pos())

    def handle_event(self, event) -> bool:
        return False

    def update(self, dt: float) -> None:
        pass

    def draw(self, surf: pygame.Surface) -> None:
        pass


# ---------------------------------------------------------------- Button

class Button(Widget):
    KINDS = {
        # kind: (bg, hover bg, text colour, border)
        "primary": (T.ACCENT, T.ACCENT_HOVER, (255, 255, 255), None),
        "secondary": (T.PANEL_2, T.PANEL_3, T.TEXT, T.BORDER),
        "ghost": (None, T.PANEL_2, T.TEXT_DIM, None),
        "danger": (T.PANEL_2, (70, 30, 42), T.DANGER, T.BORDER),
        "link": (None, None, T.ACCENT_HOVER, None),
    }

    def __init__(self, text: str = "", on_click: Callable[[], Any] | None = None, kind: str = "primary",
                 icon: str | None = None, size: int = 14, rect=None, tooltip: str = "",
                 radius: int = 10):
        super().__init__(rect)
        self.text = text
        self.on_click = on_click
        self.kind = kind
        self.icon = icon
        self.size = size
        self.tooltip = tooltip
        self.radius = radius
        self.active = False  # for toggle-style buttons
        self._pressed = False

    def handle_event(self, event) -> bool:
        if not (self.visible and self.enabled):
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and self.rect.collidepoint(event.pos):
            self._pressed = True
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self._pressed:
            self._pressed = False
            if self.rect.collidepoint(event.pos) and self.on_click:
                self.on_click()
            return True
        return False

    def draw(self, surf):
        if not self.visible:
            return
        bg, bg_hover, fg, border = self.KINDS[self.kind]
        hovered = self.hovered and self.enabled
        if self.active:
            bg, bg_hover, fg, border = T.ACCENT_DARK, T.ACCENT_DARK, (255, 255, 255), T.ACCENT
        col = bg_hover if hovered and bg_hover else bg
        radius = px(self.radius)
        if col:
            T.rrect(surf, col, self.rect, radius)
        if border:
            T.rrect(surf, border, self.rect, radius, max(1, px(1)))
        if not self.enabled:
            fg = T.TEXT_MUTED
        elif self.kind == "link" and hovered:
            fg = (255, 255, 255)
        elif self.kind == "ghost" and hovered:
            fg = T.TEXT

        icon_size = min(self.rect.h - px(14), px(18))
        if self.icon and not self.text:
            ir = pygame.Rect(0, 0, icon_size, icon_size)
            ir.center = self.rect.center
            T.draw_icon(surf, self.icon, ir, fg)
            return
        tw = T.text_width(self.text, self.size, bold=True)
        gap = px(8)
        total = tw + (icon_size + gap if self.icon else 0)
        x = self.rect.centerx - total // 2
        if self.icon:
            ir = pygame.Rect(x, 0, icon_size, icon_size)
            ir.centery = self.rect.centery
            T.draw_icon(surf, self.icon, ir, fg)
            x += icon_size + gap
        T.draw_text(surf, self.text, (x, self.rect.centery), self.size, fg, bold=True, anchor="midleft")


# ------------------------------------------------------------- TextInput

class TextInput(Widget):
    def __init__(self, placeholder: str = "", password: bool = False, max_length: int = 120,
                 on_submit: Callable[[], Any] | None = None, rect=None, text: str = ""):
        super().__init__(rect)
        self.placeholder = placeholder
        self.password = password
        self.max_length = max_length
        self.on_submit = on_submit
        self.text = text
        self.cursor = len(text)
        self.focused = False
        self.error = False
        self._blink = 0.0

    def focus(self, on: bool = True):
        self.focused = on
        self._blink = 0.0
        if on:
            self.cursor = len(self.text)
            pygame.key.start_text_input()

    def set_text(self, text: str):
        self.text = text[: self.max_length]
        self.cursor = len(self.text)

    def _insert(self, s: str):
        s = s.replace("\r", "").replace("\n", "")
        room = self.max_length - len(self.text)
        if room <= 0 or not s:
            return
        s = s[:room]
        self.text = self.text[: self.cursor] + s + self.text[self.cursor:]
        self.cursor += len(s)
        self.error = False

    def handle_event(self, event) -> bool:
        if not self.visible:
            return False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            inside = self.rect.collidepoint(event.pos)
            if inside != self.focused:
                self.focus(inside)
            return inside
        if not self.focused:
            return False
        if event.type == pygame.TEXTINPUT:
            self._insert(event.text)
            self._blink = 0.0
            return True
        if event.type == pygame.KEYDOWN:
            self._blink = 0.0
            ctrl = event.mod & pygame.KMOD_CTRL
            if event.key == pygame.K_BACKSPACE:
                if ctrl:
                    cut = self.text[: self.cursor].rstrip()
                    cut = cut[: cut.rfind(" ") + 1] if " " in cut else ""
                    self.text = cut + self.text[self.cursor:]
                    self.cursor = len(cut)
                elif self.cursor > 0:
                    self.text = self.text[: self.cursor - 1] + self.text[self.cursor:]
                    self.cursor -= 1
            elif event.key == pygame.K_DELETE:
                self.text = self.text[: self.cursor] + self.text[self.cursor + 1:]
            elif event.key == pygame.K_LEFT:
                self.cursor = max(0, self.cursor - 1)
            elif event.key == pygame.K_RIGHT:
                self.cursor = min(len(self.text), self.cursor + 1)
            elif event.key == pygame.K_HOME:
                self.cursor = 0
            elif event.key == pygame.K_END:
                self.cursor = len(self.text)
            elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                if self.on_submit:
                    self.on_submit()
            elif ctrl and event.key == pygame.K_v:
                try:
                    clip = pygame.scrap.get_text() if hasattr(pygame.scrap, "get_text") else ""
                except pygame.error:
                    clip = ""
                self._insert(clip or "")
            elif ctrl and event.key == pygame.K_a:
                self.cursor = len(self.text)
            else:
                return event.key not in (pygame.K_TAB, pygame.K_ESCAPE)
            return True
        return False

    def update(self, dt):
        self._blink += dt

    def draw(self, surf):
        if not self.visible:
            return
        border = T.DANGER if self.error else (T.ACCENT if self.focused else T.BORDER)
        T.rrect(surf, T.PANEL_2, self.rect, px(10))
        T.rrect(surf, border, self.rect, px(10), px(2) if self.focused or self.error else max(1, px(1)))
        inner = self.rect.inflate(-2 * px(14), 0)
        shown = "•" * len(self.text) if self.password else self.text
        size = 15
        clip = surf.get_clip()
        surf.set_clip(inner.clip(clip) if clip else inner)
        before = shown[: self.cursor]
        cx = T.text_width(before, size)
        offset = max(0, cx - inner.w + px(4))
        if not self.text:
            T.draw_text(surf, self.placeholder, (inner.x, inner.centery), size, T.TEXT_MUTED, anchor="midleft")
        else:
            T.draw_text(surf, shown, (inner.x - offset, inner.centery), size, T.TEXT, anchor="midleft")
        if self.focused and (self._blink % 1.0) < 0.55:
            x = inner.x + cx - offset
            pygame.draw.rect(surf, T.TEXT, (x, inner.centery - px(10), max(1, px(1.5)), px(20)))
        surf.set_clip(clip)


# ---------------------------------------------------------------- Slider

class Slider(Widget):
    def __init__(self, minimum: float, maximum: float, value: float,
                 on_change: Callable[[float], Any] | None = None, rect=None,
                 on_release: Callable[[float], Any] | None = None):
        super().__init__(rect)
        self.min, self.max = minimum, maximum
        self.value = value
        self.on_change = on_change
        self.on_release = on_release
        self.dragging = False

    def _value_at(self, x: int) -> float:
        t = (x - self.rect.x) / max(1, self.rect.w)
        return self.min + (self.max - self.min) * min(max(t, 0.0), 1.0)

    def handle_event(self, event) -> bool:
        if not self.visible:
            return False
        hit = self.rect.inflate(px(12), px(16))
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1 and hit.collidepoint(event.pos):
            self.dragging = True
            self._set(self._value_at(event.pos[0]))
            return True
        if event.type == pygame.MOUSEMOTION and self.dragging:
            self._set(self._value_at(event.pos[0]))
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.dragging = False
            if self.on_release:
                self.on_release(self.value)
            return True
        return False

    def _set(self, v):
        if v != self.value:
            self.value = v
            if self.on_change:
                self.on_change(v)

    def draw(self, surf):
        if not self.visible:
            return
        th = px(6)
        track = pygame.Rect(self.rect.x, self.rect.centery - th // 2, self.rect.w, th)
        T.rrect(surf, T.PANEL_3, track, th // 2)
        t = (self.value - self.min) / max(1e-9, self.max - self.min)
        fill = track.copy()
        fill.w = max(th, int(track.w * t))
        T.rrect(surf, T.ACCENT, fill, th // 2)
        knob_r = px(9) if (self.dragging or self.hovered) else px(7.5)
        T.circle(surf, (255, 255, 255), (track.x + int(track.w * t), track.centery), knob_r)


# ---------------------------------------------------------------- Toggle

class Toggle(Widget):
    def __init__(self, label: str, value: bool, on_change: Callable[[bool], Any] | None = None, rect=None):
        super().__init__(rect)
        self.label = label
        self.value = value
        self.on_change = on_change

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.rect.collidepoint(event.pos):
            self.value = not self.value
            if self.on_change:
                self.on_change(self.value)
            return True
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

    def draw(self, surf):
        T.draw_text(surf, self.label, (self.rect.x, self.rect.centery), 14, T.TEXT_DIM, anchor="midleft")
        sw = pygame.Rect(0, 0, px(40), px(22))
        sw.midright = self.rect.midright
        T.rrect(surf, T.ACCENT if self.value else T.PANEL_3, sw, sw.h // 2)
        kx = sw.right - sw.h // 2 if self.value else sw.x + sw.h // 2
        T.circle(surf, (255, 255, 255), (kx, sw.centery), px(8))


# ---------------------------------------------------------------- Chips

class ChoiceGroup(Widget):
    """Grid of selectable chips (one active). Sizes are logical px."""

    def __init__(self, options: list[tuple[Any, str]], value: Any, on_change: Callable[[Any], Any],
                 columns: int = 2, chip_height: int = 36, gap: int = 8, rect=None):
        super().__init__(rect)
        self.options = options
        self.value = value
        self.on_change = on_change
        self.columns = columns
        self.chip_height = chip_height
        self.gap = gap

    @property
    def height(self) -> int:
        rows = (len(self.options) + self.columns - 1) // self.columns
        return rows * px(self.chip_height) + (rows - 1) * px(self.gap)

    def _chip_rects(self):
        gap, ch = px(self.gap), px(self.chip_height)
        cw = (self.rect.w - gap * (self.columns - 1)) / self.columns
        for i, opt in enumerate(self.options):
            r, c = divmod(i, self.columns)
            yield opt, pygame.Rect(int(self.rect.x + c * (cw + gap)), self.rect.y + r * (ch + gap), int(cw), ch)

    def handle_event(self, event) -> bool:
        if not self.enabled:
            return False
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            for (val, _label), r in self._chip_rects():
                if r.collidepoint(event.pos):
                    if val != self.value:
                        self.value = val
                        self.on_change(val)
                    return True
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

    def draw(self, surf):
        mp = mouse_pos()
        for (val, label), r in self._chip_rects():
            active = val == self.value
            bg = T.ACCENT_DARK if active else (T.PANEL_3 if r.collidepoint(mp) and self.enabled else T.PANEL_2)
            T.rrect(surf, bg, r, px(9))
            T.rrect(surf, T.ACCENT if active else T.BORDER, r, px(9), max(1, px(1)))
            col = (255, 255, 255) if active else (T.TEXT_DIM if self.enabled else T.TEXT_MUTED)
            T.draw_text(surf, label, r.center, 13, col, bold=active, anchor="center", max_width=r.w - px(8))


class ColorSwatches(Widget):
    def __init__(self, colors: list[str], value: str, on_change: Callable[[str], Any], rect=None,
                 size: int = 26, gap: int = 8):
        super().__init__(rect)
        self.colors = colors
        self.value = value
        self.on_change = on_change
        self.size = size
        self.gap = gap

    def _per_row(self):
        return max(1, (self.rect.w + px(self.gap)) // (px(self.size) + px(self.gap)))

    @property
    def height(self) -> int:
        rows = (len(self.colors) + self._per_row() - 1) // self._per_row()
        return rows * px(self.size) + (rows - 1) * px(self.gap)

    def _swatches(self):
        per_row = self._per_row()
        size, gap = px(self.size), px(self.gap)
        # spread evenly across the row
        step = (self.rect.w - size) / max(1, per_row - 1) if per_row > 1 else 0
        for i, c in enumerate(self.colors):
            r, col = divmod(i, per_row)
            yield c, pygame.Rect(int(self.rect.x + col * step), self.rect.y + r * (size + gap), size, size)

    def handle_event(self, event) -> bool:
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            for c, r in self._swatches():
                if r.collidepoint(event.pos):
                    self.value = c
                    self.on_change(c)
                    return True
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

    def draw(self, surf):
        mp = mouse_pos()
        for c, r in self._swatches():
            rgb = tuple(int(c[i:i + 2], 16) for i in (1, 3, 5))
            rad = r.w / 2 - (0 if r.collidepoint(mp) else px(2))
            if c.lower() == (self.value or "").lower():
                T.circle(surf, (255, 255, 255), r.center, r.w / 2 + px(3.5), max(1, px(2)))
            T.circle(surf, rgb, r.center, rad)


# -------------------------------------------------------------- ListView

class ListView(Widget):
    """Scrollable list. ``row_actions`` are (icon, callback(item), tooltip) shown on hover."""

    def __init__(self, on_select: Callable[[Any], Any] | None = None, row_height: int = 52,
                 row_actions: list | None = None, empty_text: str = "Nothing here yet",
                 subtitle: Callable[[Any], str] | None = None, title: Callable[[Any], str] | None = None,
                 icon: str = "music", rect=None):
        super().__init__(rect)
        self.items: list = []
        self.on_select = on_select
        self.row_height_logical = row_height
        self.row_actions = row_actions or []
        self.empty_text = empty_text
        self.subtitle = subtitle or (lambda item: "")
        self.title = title or str
        self.icon = icon
        self.scroll = 0.0
        self.selected_key = None
        self.key = lambda item: id(item)

    @property
    def row_height(self) -> int:
        return px(self.row_height_logical)

    def set_items(self, items: list):
        self.items = list(items)
        self._clamp()

    def _clamp(self):
        max_scroll = max(0, len(self.items) * self.row_height - self.rect.h)
        self.scroll = min(max(self.scroll, 0), max_scroll)

    def _row_rect(self, i: int) -> pygame.Rect:
        return pygame.Rect(self.rect.x, int(self.rect.y + i * self.row_height - self.scroll),
                           self.rect.w, self.row_height - px(4))

    def _action_rects(self, row: pygame.Rect):
        x = row.right - px(8)
        for action in reversed(self.row_actions):
            r = pygame.Rect(0, 0, px(28), px(28))
            r.midright = (x, row.centery)
            x -= px(32)
            yield action, r

    def handle_event(self, event) -> bool:
        if not self.visible:
            return False
        if event.type == pygame.MOUSEWHEEL and self.hovered:
            self.scroll -= event.y * px(40)
            self._clamp()
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.rect.collidepoint(event.pos):
            for i, item in enumerate(self.items):
                row = self._row_rect(i)
                if row.collidepoint(event.pos):
                    for (icon, cb, _tip), ar in self._action_rects(row):
                        if ar.collidepoint(event.pos):
                            cb(item)
                            return True
                    if self.on_select:
                        self.on_select(item)
                    return True
        return event.type == pygame.MOUSEBUTTONDOWN and self.rect.collidepoint(event.pos)

    def draw(self, surf):
        if not self.visible:
            return
        clip = surf.get_clip()
        surf.set_clip(self.rect)
        mp = mouse_pos()
        if not self.items:
            for k, line in enumerate(T.wrap_text(self.empty_text, 13, self.rect.w - px(24))):
                T.draw_text(surf, line, (self.rect.centerx, self.rect.y + px(24) + k * px(19)), 13,
                            T.TEXT_MUTED, anchor="midtop")
        for i, item in enumerate(self.items):
            row = self._row_rect(i)
            if row.bottom < self.rect.y or row.y > self.rect.bottom:
                continue
            hovered = row.collidepoint(mp) and self.rect.collidepoint(mp)
            selected = self.selected_key is not None and self.key(item) == self.selected_key
            if selected or hovered:
                T.rrect(surf, T.ACCENT_DARK if selected else T.PANEL_2, row, px(10))
            ic = pygame.Rect(row.x + px(8), row.centery - px(16), px(32), px(32))
            T.rrect(surf, T.PANEL_3, ic, px(8))
            T.draw_icon(surf, self.icon, ic.inflate(-px(14), -px(14)), T.ACCENT_HOVER if selected else T.TEXT_DIM)
            text_w = row.w - px(58) - (len(self.row_actions) * px(32) if hovered else 0)
            T.draw_text(surf, self.title(item), (row.x + px(50), row.y + px(7)), 14, T.TEXT, bold=selected,
                        max_width=text_w)
            T.draw_text(surf, self.subtitle(item), (row.x + px(50), row.y + px(27)), 12, T.TEXT_MUTED,
                        max_width=text_w)
            if hovered:
                for (icon, _cb, _tip), ar in self._action_rects(row):
                    if ar.collidepoint(mp):
                        T.rrect(surf, T.PANEL_3, ar, px(7))
                    T.draw_icon(surf, icon, ar.inflate(-px(12), -px(12)),
                                T.DANGER if icon == "trash" and ar.collidepoint(mp) else T.TEXT)
        surf.set_clip(clip)
        content = len(self.items) * self.row_height
        if content > self.rect.h:
            bar_h = max(px(30), self.rect.h * self.rect.h / content)
            y = self.rect.y + (self.rect.h - bar_h) * self.scroll / (content - self.rect.h)
            T.rrect(surf, T.PANEL_3, (self.rect.right - px(4), y, px(4), bar_h), px(2))


# --------------------------------------------------------------- SeekBar

class SeekBar(Widget):
    """Playback position bar drawn over the track's waveform overview."""

    def __init__(self, on_seek: Callable[[float], Any], rect=None):
        super().__init__(rect)
        self.on_seek = on_seek
        self.duration = 0.0
        self.position = 0.0
        self.overview = None
        self.dragging = False
        self._drag_pos = 0.0

    def _t_at(self, x):
        return min(max((x - self.rect.x) / max(1, self.rect.w), 0.0), 1.0) * self.duration

    def handle_event(self, event) -> bool:
        if self.duration <= 0:
            return False
        if (event.type == pygame.MOUSEBUTTONDOWN and event.button == 1
                and self.rect.inflate(0, px(10)).collidepoint(event.pos)):
            self.dragging = True
            self._drag_pos = self._t_at(event.pos[0])
            return True
        if event.type == pygame.MOUSEMOTION and self.dragging:
            self._drag_pos = self._t_at(event.pos[0])
            return True
        if event.type == pygame.MOUSEBUTTONUP and event.button == 1 and self.dragging:
            self.dragging = False
            self.on_seek(self._t_at(event.pos[0]))
            return True
        return False

    def draw(self, surf):
        r = self.rect
        pos = self._drag_pos if self.dragging else self.position
        t = pos / self.duration if self.duration > 0 else 0.0
        split = r.x + int(r.w * t)
        if self.overview is not None and len(self.overview):
            n = len(self.overview)
            bw = r.w / n
            bar = max(1, int(bw * 0.62))
            for i, v in enumerate(self.overview):
                x = r.x + i * bw
                hh = max(px(3), v * r.h)
                col = T.ACCENT_HOVER if x < split else T.PANEL_3
                T.rrect(surf, col, (int(x), int(r.centery - hh / 2), bar, int(hh)), bar // 2)
        else:
            th = px(4)
            T.rrect(surf, T.PANEL_3, (r.x, r.centery - th // 2, r.w, th), th // 2)
            if split > r.x:
                T.rrect(surf, T.ACCENT, (r.x, r.centery - th // 2, split - r.x, th), th // 2)
        if self.duration > 0:
            T.rrect(surf, (255, 255, 255), (split - px(1), r.y - px(3), px(2), r.h + px(6)), px(1))
        if self.hovered and self.duration > 0:
            hx = mouse_pos()[0]
            T.draw_text(surf, T.format_time(self._t_at(hx)), (hx, r.y - px(6)), 11, T.TEXT_DIM, anchor="midbottom")


# --------------------------------------------------------------- Toasts

class Toasts:
    def __init__(self):
        self.items: list[tuple[str, str, float]] = []

    def show(self, message: str, kind: str = "info", seconds: float = 3.5):
        self.items.append((message, kind, time.time() + seconds))
        self.items = self.items[-4:]

    def draw(self, surf):
        now = time.time()
        self.items = [t for t in self.items if t[2] > now]
        y = surf.get_height() - px(120)
        for message, kind, expires in reversed(self.items):
            col = {"error": T.DANGER, "success": T.SUCCESS, "warning": T.WARNING}.get(kind, T.ACCENT_HOVER)
            lines = T.wrap_text(message, 14, px(420))
            lh = px(22)
            h = lh * len(lines) + px(20)
            w = max(T.text_width(line, 14) for line in lines) + px(44)
            rect = pygame.Rect(0, 0, w, h)
            rect.midbottom = (surf.get_width() // 2, y)
            alpha = int(255 * min(1.0, (expires - now) / 0.3))
            box = pygame.Surface(rect.size, pygame.SRCALPHA)
            T.rrect(box, (*T.PANEL_3, 250), box.get_rect(), px(12))
            T.rrect(box, col, (px(9), px(10), px(4), rect.h - px(20)), px(2))
            for k, line in enumerate(lines):
                T.draw_text(box, line, (px(22), px(10) + k * lh), 14, T.TEXT)
            box.set_alpha(alpha)
            surf.blit(box, rect)
            y = rect.y - px(10)


# ----------------------------------------------------------------- Modal

class Modal:
    """Centered dialog. Subclasses add widgets in ``layout`` and draw content.

    ``width``/``height`` are logical px.
    """

    width, height = 460, 300
    title = ""

    def __init__(self, app):
        self.app = app
        self.rect = pygame.Rect(0, 0, px(self.width), px(self.height))
        self.widgets: list[Widget] = []
        self.closable = True

    def layout(self, screen_size):
        self.rect.size = (px(self.width), px(self.height))
        self.rect.center = (screen_size[0] // 2, screen_size[1] // 2)

    def close(self):
        self.app.close_modal(self)

    def handle_event(self, event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE and self.closable:
            self.close()
            return True
        for w in self.widgets:
            if w.visible and w.handle_event(event):
                return True
        if event.type == pygame.MOUSEBUTTONDOWN and not self.rect.collidepoint(event.pos) and self.closable:
            self.close()
        return True  # modal swallows everything

    def update(self, dt):
        for w in self.widgets:
            w.update(dt)

    def draw(self, surf):
        T.shade(surf, surf.get_rect(), 160)
        T.draw_panel(surf, self.rect, T.PANEL, radius=16)
        T.draw_text(surf, self.title, (self.rect.x + px(28), self.rect.y + px(24)), 20, T.TEXT, bold=True)
        self.draw_content(surf)
        for w in self.widgets:
            w.draw(surf)

    def draw_content(self, surf):
        pass

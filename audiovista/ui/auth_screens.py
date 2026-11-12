"""Log-in and Create-account screens."""

from __future__ import annotations

import time

import math

import pygame
import pygame.gfxdraw as gfx

from ..errors import AccountExistsError, AudioVistaError, ValidationError
from ..visuals.backgrounds import get_background
from . import theme as T
from .theme import px
from .widgets import Button, TextInput


class Screen:
    def __init__(self, app):
        self.app = app

    def layout(self, size): ...
    def handle_event(self, event) -> bool: return False
    def update(self, dt): ...
    def draw(self, surf): ...
    def on_enter(self): ...
    def on_exit(self): ...


class AmbientBackdrop:
    """A sunset-sky gradient with slow, glowing waves behind the account screens."""

    def __init__(self):
        self._surface: pygame.Surface | None = None
        self._scale = 2  # drawn at half resolution then smooth-scaled (soft look, cheap)

    def draw(self, surf: pygame.Surface, t: float) -> None:
        w, h = surf.get_size()
        sw, sh = max(1, w // self._scale), max(1, h // self._scale)
        if self._surface is None or self._surface.get_size() != (sw, sh):
            self._surface = pygame.Surface((sw, sh))
            self._layer = pygame.Surface((sw, sh), pygame.SRCALPHA)
        small = self._surface
        small.blit(get_background("sunset", (sw, sh)), (0, 0))
        n = 120
        xs = [i * sw / (n - 1) for i in range(n)]
        for k, (col, speed, freq, amp, base) in enumerate([
                (T.GOLD, 0.45, 1.6, 0.10, 0.70), (T.ACCENT, 0.30, 2.3, 0.08, 0.76),
                (T.RUST, 0.6, 1.1, 0.06, 0.82)]):
            pts = [(0, sh)]
            for i, x in enumerate(xs):
                u = i / (n - 1)
                y = (base + amp * math.sin(2 * math.pi * (u * freq + t * speed * 0.2) + k)
                     * (0.6 + 0.4 * math.sin(t * 0.7 + u * 3 + k))) * sh
                pts.append((x, y))
            pts.append((sw, sh))
            layer = self._layer
            layer.fill((0, 0, 0, 0))
            gfx.filled_polygon(layer, pts, (*T.BG, 70))
            gfx.aapolygon(layer, pts, (*T.BG, 70))
            pygame.draw.aalines(layer, (*col, 170), False, pts[1:-1])
            small.blit(layer, (0, 0))
        surf.blit(pygame.transform.smoothscale(small, (w, h)), (0, 0))


class AuthScreen(Screen):
    """Shared layout for the two account screens."""

    card_w = 440
    heading = ""
    subheading = ""

    def __init__(self, app):
        super().__init__(app)
        self.fields: list[tuple[str, str, TextInput]] = []  # (key, label, input)
        self.error = ""
        self.info = ""
        self.card = pygame.Rect(0, 0, px(self.card_w), 100)
        self.primary: Button | None = None
        self.switch_button: Button | None = None
        self.backdrop = AmbientBackdrop()
        self._t0 = time.time()

    def add_field(self, key, label, placeholder, password=False):
        inp = TextInput(placeholder, password=password, on_submit=self.submit)
        self.fields.append((key, label, inp))
        return inp

    def values(self) -> dict:
        return {k: inp.text for k, _l, inp in self.fields}

    def show_error(self, exc: Exception):
        self.info = ""
        self.error = str(exc)
        field = getattr(exc, "field", None)
        for k, _l, inp in self.fields:
            inp.error = (k == field)
            if k == field:
                self._focus(inp)

    def _focus(self, target):
        for _k, _l, inp in self.fields:
            inp.focus(inp is target)

    def layout(self, size):
        w, h = size
        field_h, step = px(46), px(76)
        cw = px(self.card_w)
        self.card.size = (cw, px(300) + len(self.fields) * step)
        self.card.center = (w // 2, h // 2)
        pad = px(36)
        y = self.card.y + px(140)
        for _k, _l, inp in self.fields:
            inp.rect = pygame.Rect(self.card.x + pad, y + px(24), cw - 2 * pad, field_h)
            y += step
        self.message_y = y + px(2)
        self.primary.rect = pygame.Rect(self.card.x + pad, y + px(30), cw - 2 * pad, px(48))
        self.switch_button.rect = pygame.Rect(self.card.x + pad, y + px(90), cw - 2 * pad, px(30))

    def on_enter(self):
        self.layout(self.app.screen.get_size())
        if self.fields:
            first_empty = next((inp for _k, _l, inp in self.fields if not inp.text), self.fields[0][2])
            self._focus(first_empty)

    def handle_event(self, event) -> bool:
        if event.type == pygame.KEYDOWN and event.key == pygame.K_TAB:
            inputs = [inp for _k, _l, inp in self.fields]
            idx = next((i for i, inp in enumerate(inputs) if inp.focused), -1)
            step = -1 if event.mod & pygame.KMOD_SHIFT else 1
            self._focus(inputs[(idx + step) % len(inputs)])
            return True
        for _k, _l, inp in self.fields:
            if inp.handle_event(event):
                if event.type in (pygame.TEXTINPUT, pygame.KEYDOWN):
                    self.error = ""
                return True
        for b in (self.primary, self.switch_button):
            if b.handle_event(event):
                return True
        return False

    def update(self, dt):
        for _k, _l, inp in self.fields:
            inp.update(dt)

    def draw(self, surf):
        self.backdrop.draw(surf, time.time() - self._t0)
        # frosted-glass card floating over the landscape
        T.rrect(surf, (6, 8, 20, 70), self.card.move(0, px(8)), px(24))
        T.rrect(surf, (*T.PANEL, 222), self.card, px(20))
        T.rrect(surf, (255, 255, 255, 34), self.card, px(20), max(1, px(1)))
        pad = px(36)
        logo = pygame.Rect(self.card.x + pad, self.card.y + px(28), px(38), px(38))
        T.draw_logo(surf, logo)
        T.draw_text(surf, "AudioVista", (logo.right + px(12), logo.centery), 20, T.TEXT, bold=True,
                    anchor="midleft")
        T.draw_text(surf, self.heading, (self.card.x + pad, self.card.y + px(78)), 24, T.TEXT, bold=True)
        T.draw_text(surf, self.subheading, (self.card.x + pad, self.card.y + px(110)), 14, T.TEXT_DIM)
        for _k, label, inp in self.fields:
            T.draw_text(surf, label, (inp.rect.x + px(2), inp.rect.y - px(22)), 12, T.TEXT_DIM, bold=True)
            inp.draw(surf)
        if self.error or self.info:
            msg, col = (self.error, T.DANGER) if self.error else (self.info, T.SUCCESS)
            T.draw_text(surf, msg, (self.card.x + pad, self.message_y), 13, col,
                        max_width=self.card.w - 2 * pad)
        self.primary.draw(surf)
        self.switch_button.draw(surf)
        T.draw_text(surf, "Tab to move between fields  ·  Enter to submit",
                    (self.card.centerx, self.card.bottom + px(18)), 12, (236, 226, 236), anchor="midtop")

    def submit(self): ...


class LoginScreen(AuthScreen):
    heading = "Welcome back"
    subheading = "Log in to visualise your music."

    def __init__(self, app, email: str = "", info: str = ""):
        super().__init__(app)
        self.email = self.add_field("email", "E-MAIL OR USERNAME", "you@example.com")
        self.email.set_text(email)
        self.password = self.add_field("password", "PASSWORD", "Your password", password=True)
        self.primary = Button("Log in", self.submit)
        self.switch_button = Button("New here?  Create an account", app.show_register, kind="link", size=13)
        self.info = info

    def submit(self):
        try:
            self.app.submit_login(self.email.text, self.password.text)
        except AudioVistaError as exc:
            self.password.set_text("")
            self.show_error(exc)
            self._focus(self.password)


class RegisterScreen(AuthScreen):
    heading = "Create your account"
    subheading = "Save your visualizations and come back to them later."

    def __init__(self, app):
        super().__init__(app)
        self.username = self.add_field("username", "USERNAME", "e.g. beatmaker42")
        self.email = self.add_field("email", "E-MAIL", "you@example.com")
        self.password = self.add_field("password", "PASSWORD", "8+ characters, a letter and a number",
                                       password=True)
        self.confirm = self.add_field("confirm", "CONFIRM PASSWORD", "Type it again", password=True)
        self.primary = Button("Create account", self.submit)
        self.switch_button = Button("Already have an account?  Log in", app.show_login, kind="link",
                                    size=13)

    def submit(self):
        try:
            self.app.submit_registration(self.username.text, self.email.text, self.password.text,
                                         self.confirm.text)
        except AccountExistsError as exc:
            self.show_error(exc)
            if exc.field == "email":
                self.switch_button.text = "Log in instead"
        except (ValidationError, AudioVistaError) as exc:
            self.show_error(exc)

"""Log-in and Create-account screens."""

from __future__ import annotations

import math
import time

import pygame

from ..errors import AccountExistsError, AudioVistaError, ValidationError
from . import theme as T
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


def draw_ambient(surf: pygame.Surface, t: float) -> None:
    """Decorative animated equalizer behind the auth card."""
    surf.fill(T.BG)
    w, h = surf.get_size()
    n = 64
    bw = w / n
    for i in range(n):
        x = i * bw
        v = (math.sin(t * 1.7 + i * 0.35) * 0.5 + 0.5) * (math.sin(t * 0.9 + i * 0.13) * 0.4 + 0.6)
        bh = 40 + v * h * 0.35
        k = i / n
        col = (int(T.ACCENT[0] * (1 - k) + 0 * k), int(T.ACCENT[1] * (1 - k) + 229 * k),
               int(T.ACCENT[2] * (1 - k) + 255 * k))
        col = tuple(int(c * 0.28) for c in col)
        pygame.draw.rect(surf, col, (x + 2, h - bh, bw - 4, bh), border_top_left_radius=4,
                         border_top_right_radius=4)


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
        self.card = pygame.Rect(0, 0, self.card_w, 100)
        self.buttons: list[Button] = []
        self.primary: Button | None = None
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
        field_h = 46
        body = len(self.fields) * (field_h + 30) + 150
        self.card.size = (self.card_w, 150 + body)
        self.card.center = (w // 2, h // 2)
        y = self.card.y + 140
        for _k, _l, inp in self.fields:
            inp.rect = pygame.Rect(self.card.x + 36, y + 24, self.card_w - 72, field_h)
            y += field_h + 30
        self.message_y = y + 2
        self.primary.rect = pygame.Rect(self.card.x + 36, y + 30, self.card_w - 72, 48)
        self.switch_button.rect = pygame.Rect(self.card.x + 36, y + 90, self.card_w - 72, 30)

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
        draw_ambient(surf, time.time() - self._t0)
        shadow = self.card.move(0, 8)
        pygame.draw.rect(surf, (5, 6, 10), shadow, border_radius=20)
        T.draw_panel(surf, self.card, T.PANEL, radius=20)
        logo = pygame.Rect(self.card.x + 36, self.card.y + 34, 30, 26)
        T.draw_icon(surf, "logo", logo, T.ACCENT)
        T.draw_text(surf, "AudioVista", (logo.right + 12, logo.centery), 20, T.TEXT, bold=True,
                    anchor="midleft")
        T.draw_text(surf, self.heading, (self.card.x + 36, self.card.y + 76), 24, T.TEXT, bold=True)
        T.draw_text(surf, self.subheading, (self.card.x + 36, self.card.y + 108), 14, T.TEXT_DIM)
        for _k, label, inp in self.fields:
            T.draw_text(surf, label, (inp.rect.x + 2, inp.rect.y - 22), 13, T.TEXT_DIM, bold=True)
            inp.draw(surf)
        if self.error or self.info:
            msg, col = (self.error, T.DANGER) if self.error else (self.info, T.SUCCESS)
            T.draw_text(surf, msg, (self.card.x + 36, self.message_y), 13, col,
                        max_width=self.card_w - 72)
        self.primary.draw(surf)
        self.switch_button.draw(surf)
        T.draw_text(surf, "Tab to move between fields  ·  Enter to submit",
                    (self.card.centerx, self.card.bottom + 16), 12, T.TEXT_MUTED, anchor="midtop")

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

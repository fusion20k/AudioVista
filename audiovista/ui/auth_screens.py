"""Log-in and Create-account screens."""

from __future__ import annotations

import time

import numpy as np
import pygame

from ..errors import AccountExistsError, AudioVistaError, ValidationError
from ..visuals import scenery as S
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
    """A calm sunset landscape behind the account screens: drifting clouds,
    light rays, a lake that mirrors the sky, fireflies.

    It is rendered at half the window size, at most 30 times a second, and
    smoothly scaled up - it is soft by nature, so this looks the same and keeps
    typing in the form responsive.
    """

    HORIZON = 0.63

    def __init__(self):
        self.pal = S.palette("sunset")
        self.clouds = S.CloudLayer(5, seed=4, y_range=(0.06, 0.42), size_range=(0.2, 0.36))
        self.stars = S.StarField(160, seed=8)
        self.motes = S.Motes(36, seed=2, region=(0.0, 1.0, 0.25, 0.8))
        self.flies = S.Fireflies(28, seed=6, region=(0.0, 1.0, 0.74, 0.97))
        self._scene: pygame.Surface | None = None
        self._big: pygame.Surface | None = None
        self._light: pygame.Surface | None = None
        self._light_big: pygame.Surface | None = None
        self._last = -1.0

    def draw(self, surf: pygame.Surface, t: float) -> None:
        w, h = surf.get_size()
        sw, sh = max(16, w // 2), max(16, h // 2)
        if self._scene is None or self._scene.get_size() != (sw, sh) or self._big.get_size() != (w, h):
            self._scene = pygame.Surface((sw, sh))
            self._big = pygame.Surface((w, h))
            self._light = pygame.Surface((max(4, sw // 4), max(4, sh // 4)))
            self._light_big = pygame.Surface((sw, sh))
            self._last = -1.0
        if self._last < 0 or t - self._last >= 1 / 30:
            self._last = t
            self._render(t)
            pygame.transform.smoothscale(self._scene, (w, h), self._big)
        surf.blit(self._big, (0, 0))

    def _render(self, t: float) -> None:
        s, light, pal = self._scene, self._light, self.pal
        w, h = s.get_size()
        u = min(w, h)
        k = light.get_width() / w
        hy = int(h * self.HORIZON)
        clock = t * 0.6
        light.fill((0, 0, 0))

        s.blit(S.sky_surface(pal, (w, h), self.HORIZON), (0, 0))
        self.stars.draw(s, light, k, w, h, hy, t, 0.35, 0.0, u)
        sx, sy = w * 0.70, hy - h * 0.07
        S.aa_circle(s, S.lerp(pal.sun, pal.halo, 0.2), (sx, sy), u * 0.062)
        S.aa_circle(s, pal.sun, (sx, sy), u * 0.058)
        S.add_glow(light, (sx * k, sy * k), u * 0.7 * k, pal.halo, 0.42, power=2.2)
        S.add_glow(light, (sx * k, sy * k), u * 0.2 * k, pal.sun, 0.5, power=1.6)
        S.god_rays(light, (sx * k, sy * k), u * 1.2 * k, pal.halo, 0.32, t * 0.02, floor=hy * k, seed=5)
        self.clouds.draw(s, hy, clock, pal)
        S.draw_birds(s, w, h, t, S.lerp(pal.near, S.sky_at(pal, 0.4), 0.25), u, seed=2)

        xs = np.linspace(0, 1, 360)
        X = xs * w
        far = hy - h * (0.07 + 0.15 * S.ridge(xs + clock * 0.002, 11, 5, sharp=0.6))
        mid = hy - h * (0.01 + 0.06 * S.ridge(xs + clock * 0.004, 12, 3))
        S.fill_profile(s, S.land(pal, 0.2), X, far, hy + 2)
        S.fog(s, hy - h * 0.12, h * 0.12, pal.haze, 0.5)
        S.fill_profile(s, S.land(pal, 0.5), X, mid, hy + 2)
        S.reflect(s, hy, clock, pal, u, 0.8)
        for j in range(5):
            fy = hy + (h - hy) * (0.06 + j * 0.18)
            S.add_glow(light, (sx * k, fy * k), u * k * (0.07 + j * 0.02), pal.halo, 0.26 - j * 0.04)
        self.motes.draw(s, light, k, w, h, clock, t, S.lerp(pal.halo, (255, 255, 255), 0.4), u, 0.8)
        self.flies.draw(s, light, k, w, h, clock, 0.6, 0.0, T.GOLD, T.SKY, u, 0.8)

        pygame.transform.smoothscale(light, (w, h), self._light_big)
        s.blit(self._light_big, (0, 0), special_flags=pygame.BLEND_RGB_ADD)
        S.vignette(s)


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

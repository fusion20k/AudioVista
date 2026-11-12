"""Audio playback with pygame.mixer and an accurate playback clock."""

from __future__ import annotations

import time

import numpy as np

try:
    import pygame
except ImportError:  # pragma: no cover - pygame is a hard dependency of the UI
    pygame = None


class AudioPlayer:
    """Plays a decoded track and reports the current position in seconds.

    Position is tracked with a high-resolution clock rather than by asking
    the mixer, because pygame cannot report the position of a ``Sound``.
    If no audio device is available the player still works silently, so the
    visualizer keeps running.
    """

    def __init__(self):
        self.samples: np.ndarray | None = None
        self.sample_rate = 44100
        self.duration = 0.0
        self.volume = 0.8
        self._offset = 0.0          # position (s) where the current Sound starts
        self._started_at = 0.0      # perf_counter when playback (re)started
        self._paused_pos = 0.0
        self._playing = False
        self._sound = None
        self._channel = None
        self.audio_ok = self._init_mixer()

    @staticmethod
    def _init_mixer() -> bool:
        if pygame is None:
            return False
        try:
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=44100, size=-16, channels=2, buffer=1024)
            return True
        except pygame.error:
            return False

    # ----------------------------------------------------------------- state
    @property
    def loaded(self) -> bool:
        return self.samples is not None

    @property
    def is_playing(self) -> bool:
        return self._playing

    @property
    def position(self) -> float:
        if self._playing:
            pos = self._offset + (time.perf_counter() - self._started_at)
        else:
            pos = self._paused_pos
        return min(max(pos, 0.0), self.duration)

    @property
    def finished(self) -> bool:
        return self.loaded and self._playing and self.position >= self.duration - 1e-3

    # -------------------------------------------------------------- controls
    def load(self, samples: np.ndarray, sample_rate: int) -> None:
        self.stop()
        self.samples = samples
        self.sample_rate = sample_rate
        self.duration = len(samples) / float(sample_rate)
        self._paused_pos = 0.0

    def unload(self) -> None:
        self.stop()
        self.samples = None
        self.duration = 0.0
        self._paused_pos = 0.0

    def play(self) -> None:
        if not self.loaded or self._playing:
            return
        if self._paused_pos >= self.duration - 0.05:
            self._paused_pos = 0.0
        self._start_from(self._paused_pos)

    def pause(self) -> None:
        if not self._playing:
            return
        self._paused_pos = self.position
        self._playing = False
        self._stop_sound()

    def toggle(self) -> None:
        self.pause() if self._playing else self.play()

    def stop(self) -> None:
        self._playing = False
        self._paused_pos = 0.0
        self._stop_sound()

    def seek(self, seconds: float) -> None:
        if not self.loaded:
            return
        seconds = min(max(seconds, 0.0), self.duration)
        if self._playing:
            self._stop_sound()
            self._start_from(seconds)
        else:
            self._paused_pos = seconds

    def set_volume(self, volume: float) -> None:
        self.volume = min(max(volume, 0.0), 1.0)
        if self._channel is not None:
            self._channel.set_volume(self.volume)

    # -------------------------------------------------------------- internals
    def _start_from(self, seconds: float) -> None:
        self._offset = seconds
        self._playing = True
        self._started_at = time.perf_counter()
        if not self.audio_ok:
            return
        try:
            freq, _size, channels = pygame.mixer.get_init()
            start = int(seconds * self.sample_rate)
            chunk = self.samples[start:]
            if freq != self.sample_rate:
                n_out = max(1, int(len(chunk) * freq / self.sample_rate))
                x = np.linspace(0, len(chunk) - 1, n_out)
                chunk = np.stack([np.interp(x, np.arange(len(chunk)), chunk[:, c])
                                  for c in range(chunk.shape[1])], axis=1)
            if channels == 1:
                chunk = chunk.mean(axis=1, keepdims=True)
            pcm = np.ascontiguousarray((np.clip(chunk, -1, 1) * 32767).astype(np.int16))
            if channels == 1:
                pcm = pcm[:, 0]
            if len(pcm) == 0:
                return
            self._sound = pygame.sndarray.make_sound(pcm)
            self._channel = self._sound.play()
            if self._channel is not None:
                self._channel.set_volume(self.volume)
            self._started_at = time.perf_counter()
        except pygame.error:
            self.audio_ok = False

    def _stop_sound(self) -> None:
        if self._channel is not None:
            try:
                self._channel.stop()
            except Exception:
                pass
        self._channel = None
        self._sound = None

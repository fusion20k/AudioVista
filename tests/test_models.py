"""Unit tests for VisualSettings, PlaybackQueue, renderers and the player clock."""

import os

import numpy as np
import pygame
import pytest

from audiovista.audio.analyzer import AudioAnalyzer
from audiovista.audio.player import AudioPlayer
from audiovista.errors import ValidationError
from audiovista.models import WAVE_TYPES, AudioFile, PlaybackQueue, VisualSettings
from audiovista.visuals.engine import VisualizerEngine
from audiovista.visuals.renderers import create_renderer


class TestVisualSettings:
    def test_defaults_are_valid(self):
        s = VisualSettings().validate()
        assert s.wave_type in WAVE_TYPES
        assert s.primary_rgb == (0xff, 0xb3, 0x5c)

    def test_round_trip_through_dict(self):
        s = VisualSettings(color="#00e5ff", wave_type="circular", particles=True, sensitivity=1.5, glow=False)
        again = VisualSettings.from_dict(s.to_dict())
        assert again == s

    def test_from_dict_ignores_unknown_keys_and_fills_defaults(self):
        s = VisualSettings.from_dict({"wave_type": "circular", "something_else": 1})
        assert s.wave_type == "circular" and s.background == "sunset"

    @pytest.mark.parametrize("bad", [
        {"color": "red"}, {"color": "#12345"}, {"secondary_color": "#GGGGGG"},
        {"wave_type": "spiral"}, {"background": "plaid"},
        {"sensitivity": 5}, {"sensitivity": "high"}, {"smoothing": -0.1}, {"smoothing": 1.0},
    ])
    def test_invalid_values_raise(self, bad):
        with pytest.raises(ValidationError):
            VisualSettings.from_dict(bad)

    def test_from_dict_requires_dict(self):
        with pytest.raises(ValidationError):
            VisualSettings.from_dict(["not", "a", "dict"])

    def test_copy_changes_only_given_fields(self):
        s = VisualSettings()
        c = s.copy(background="dawn")
        assert c.background == "dawn" and s.background == "sunset"
        assert c.color == s.color
        with pytest.raises(ValidationError):
            s.copy(background="nope")


class TestPlaybackQueue:
    def test_fifo_order(self):
        q = PlaybackQueue()
        a, b = AudioFile("a.wav"), AudioFile("b.wav")
        q.add(a)
        q.add(b)
        assert len(q) == 2
        assert q.pop_next() is a and q.pop_next() is b
        assert q.pop_next() is None

    def test_remove_and_clear(self):
        q = PlaybackQueue()
        for name in "abc":
            q.add(AudioFile(f"{name}.wav"))
        assert q.remove(1).title == "b"
        with pytest.raises(IndexError):
            q.remove(5)
        q.clear()
        assert q.items == []


@pytest.fixture(scope="module")
def analyzer():
    t = np.arange(44100 * 2) / 44100
    pulse = 0.55 + 0.45 * np.sin(2 * np.pi * 2 * t)          # a tone that swells twice a second
    return AudioAnalyzer((0.5 * pulse * np.sin(2 * np.pi * 220 * t)).astype(np.float32))


@pytest.mark.parametrize("style", WAVE_TYPES)
@pytest.mark.parametrize("effects", [False, True])
def test_every_style_renders_a_moving_scene(analyzer, style, effects):
    pygame.init()
    engine = VisualizerEngine(analyzer, VisualSettings(wave_type=style, background="night", show_title=False,
                                                       glow=effects, atmosphere=effects))
    surface = pygame.Surface((320, 180))
    engine.render(surface, 0.5, 1 / 30)
    first = pygame.surfarray.array3d(surface).copy()
    for i in range(1, 30):
        engine.render(surface, 0.5 + i / 30, 1 / 30)
    last = pygame.surfarray.array3d(surface)
    assert first.std() > 3, "the scene should not be a flat colour"
    assert (first != last).any(), "the scene should move while the music plays"


def test_unknown_style_raises():
    with pytest.raises(ValueError):
        create_renderer("hologram")


def test_engine_frames_count(analyzer):
    pygame.init()
    engine = VisualizerEngine(analyzer, VisualSettings())
    frames = list(engine.frames((64, 36), fps=10, start=0.0, duration=1.0))
    assert len(frames) == 10


def test_player_clock_without_audio_device(monkeypatch):
    player = AudioPlayer()
    player.audio_ok = False  # run silently
    player.load(np.zeros((44100 * 3, 2), np.float32), 44100)
    assert player.duration == pytest.approx(3.0)
    player.seek(1.5)
    assert player.position == pytest.approx(1.5)
    player.play()
    assert player.is_playing
    player.pause()
    paused = player.position
    assert 1.5 <= paused < 2.0
    player.seek(10)  # clamped to the end
    assert player.position == pytest.approx(3.0)
    player.stop()
    assert player.position == 0 and not player.is_playing


def test_data_dir_is_a_real_on_disk_path(tmp_path, monkeypatch):
    # Store-installed Python redirects AppData; data_dir must return the real path
    # so ffmpeg (a separate program) can open files saved there.
    from audiovista import config

    monkeypatch.setenv("AUDIOVISTA_DATA_DIR", str(tmp_path / "av"))
    path = config.data_dir()
    assert path.is_dir()
    assert str(path) == os.path.realpath(path)


@pytest.mark.parametrize("old_background, sky", [("nebula", "twilight"), ("walnut", "golden"),
                                                 ("charcoal", "night"), ("midnight", "night")])
def test_settings_saved_by_earlier_versions_still_load(old_background, sky):
    old = {"color": "#7c5cff", "secondary_color": "#00e5ff", "background": old_background,
           "wave_type": "circular", "sensitivity": 1.0, "smoothing": 0.6, "glow": True, "show_title": True,
           "vintage": False}
    s = VisualSettings.from_dict(old)
    assert s.background == sky
    assert s.atmosphere is False          # the old "retro" switch carries over


@pytest.mark.parametrize("old_style, style, particles", [("meters", "circular", False),
                                                          ("particles", "circular", True),
                                                          ("waveform", "bars", False)])
def test_removed_styles_still_open(old_style, style, particles):
    s = VisualSettings.from_dict({"wave_type": old_style})
    assert s.wave_type == style and s.particles is particles


def test_radial_particles_option_draws_extra_detail(analyzer):
    pygame.init()
    frames = []
    for particles in (False, True):
        engine = VisualizerEngine(analyzer, VisualSettings(wave_type="circular", particles=particles,
                                                           show_title=False, glow=False, atmosphere=False))
        surface = pygame.Surface((320, 180))
        for i in range(20):
            engine.render(surface, 0.5 + i / 30, 1 / 30)
        frames.append(pygame.surfarray.array3d(surface).copy())
    assert (frames[0] != frames[1]).any()

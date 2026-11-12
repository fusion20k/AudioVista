"""Unit tests for AudioAnalyzer (the FFT engine)."""

import numpy as np
import pytest

from audiovista.audio.analyzer import AudioAnalyzer
from audiovista.audio.decoder import decode_audio

SR = 44100


def sine(freq, seconds=2.0, amp=0.5):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * freq * t)).astype(np.float32)


@pytest.mark.parametrize("freq", [100.0, 440.0, 2000.0, 8000.0])
def test_loudest_band_matches_tone_frequency(freq):
    analyzer = AudioAnalyzer(sine(freq))
    frame = analyzer.frame_at(1.0)
    centres = analyzer.band_frequencies()
    peak_freq = centres[int(np.argmax(frame.bands))]
    # Bands are log spaced, so allow about one band of error.
    assert abs(np.log2(peak_freq / freq)) < 0.25


def test_band_frequencies_are_increasing_and_in_range():
    analyzer = AudioAnalyzer(sine(440), n_bands=32, fmin=40, fmax=12000)
    centres = analyzer.band_frequencies()
    assert len(centres) == 32
    assert np.all(np.diff(centres) > 0)
    assert 40 <= centres[0] and centres[-1] <= 12000


def test_frame_values_are_normalised():
    analyzer = AudioAnalyzer(sine(440))
    for t in (0.0, 0.5, 1.99):
        f = analyzer.frame_at(t)
        assert f.bands.shape == (analyzer.n_bands,)
        assert np.all((f.bands >= 0) & (f.bands <= 1))
        assert 0 <= f.volume <= 1 and 0 <= f.beat <= 1


def test_stereo_input_is_accepted():
    mono = sine(300)
    analyzer = AudioAnalyzer(np.stack([mono, mono], axis=1))
    assert analyzer.duration == pytest.approx(2.0, abs=1e-3)


def test_silence_produces_flat_output():
    analyzer = AudioAnalyzer(np.zeros(SR, dtype=np.float32))
    f = analyzer.frame_at(0.5)
    assert np.all(f.bands == 0) and f.volume == 0
    assert analyzer.beat_times.size == 0
    assert analyzer.estimate_tempo() is None


def test_out_of_range_time_is_clamped():
    analyzer = AudioAnalyzer(sine(440, seconds=1.0))
    assert analyzer.frame_at(-5).time == 0.0
    assert analyzer.frame_at(99).time == pytest.approx(analyzer.duration)


def test_detects_beats_and_tempo(beat_wav):
    decoded = decode_audio(beat_wav)
    analyzer = AudioAnalyzer(decoded.samples)
    assert 12 <= analyzer.beat_times.size <= 17       # 120 BPM for ~7.5 s
    assert analyzer.estimate_tempo() == pytest.approx(120, abs=4)
    first = analyzer.beat_times[0]
    assert analyzer.frame_at(first + 0.01).beat > 0.8   # pulse right after a beat
    assert analyzer.frame_at(first + 0.45).beat < 0.2   # decayed before the next one


def test_overview_is_normalised():
    analyzer = AudioAnalyzer(sine(440) * np.linspace(0, 1, 2 * SR, dtype=np.float32))
    ov = analyzer.overview(50)
    assert ov.shape == (50,)
    assert ov.max() == pytest.approx(1.0)
    assert ov[0] < ov[-1]  # fade-in is visible
    with pytest.raises(ValueError):
        analyzer.overview(0)


@pytest.mark.parametrize("kwargs", [
    {"sample_rate": 0}, {"n_bands": 0}, {"fmin": 0}, {"fmin": 5000, "fmax": 100},
])
def test_invalid_parameters_raise(kwargs):
    with pytest.raises(ValueError):
        AudioAnalyzer(sine(440), **kwargs)


def test_rejects_wrong_shape():
    with pytest.raises(ValueError):
        AudioAnalyzer(np.zeros((2, 2, 2)))

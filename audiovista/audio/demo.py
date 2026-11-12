"""Synthesises a short royalty-free demo track so the app can be tried
without having any music files on hand."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..config import SAMPLE_RATE
from .decoder import write_wav


def synth_demo(seconds: float = 32.0, bpm: float = 120.0, sample_rate: int = SAMPLE_RATE,
               seed: int = 7) -> np.ndarray:
    """Return stereo float32 samples of a simple electronic loop."""
    rng = np.random.default_rng(seed)
    n = int(seconds * sample_rate)
    out = np.zeros(n, dtype=np.float64)
    beat = 60.0 / bpm

    def place(sound: np.ndarray, at: float, gain: float = 1.0):
        i = int(at * sample_rate)
        if i >= n:
            return
        j = min(n, i + len(sound))
        out[i:j] += gain * sound[: j - i]

    # Kick drum: falling sine sweep
    kt = np.arange(int(0.35 * sample_rate)) / sample_rate
    kick = np.sin(2 * np.pi * (45 + 110 * np.exp(-kt * 25)) * kt) * np.exp(-kt * 9)
    # Snare / clap: filtered noise burst
    st = np.arange(int(0.2 * sample_rate)) / sample_rate
    snare = rng.standard_normal(len(st)) * np.exp(-st * 22) * 0.5
    # Hi-hat: bright noise tick
    ht = np.arange(int(0.05 * sample_rate)) / sample_rate
    hat = np.diff(rng.standard_normal(len(ht) + 1)) * np.exp(-ht * 90) * 0.18

    chords = [(220.00, 261.63, 329.63), (174.61, 220.00, 261.63),
              (196.00, 246.94, 293.66), (164.81, 207.65, 246.94)]
    bass_notes = [55.0, 43.65, 49.0, 41.2]

    total_beats = int(seconds / beat)
    for b in range(total_beats):
        at = b * beat
        bar = (b // 4) % 4
        intro = b < 8
        if not intro or b % 2 == 0:
            place(kick, at, 0.9)
        if not intro and b % 4 in (1, 3):
            place(snare, at, 0.55)
        for k in range(2):
            place(hat, at + k * beat / 2, 0.6 if k else 0.35)
        # Bass line on every beat after the intro
        if not intro:
            bt = np.arange(int(beat * 0.9 * sample_rate)) / sample_rate
            f = bass_notes[bar]
            bass = np.tanh(2.5 * np.sin(2 * np.pi * f * bt)) * np.exp(-bt * 3) * 0.35
            place(bass, at)

    # Pad chords swelling over each bar, plus a shimmering arpeggio
    for bar_start in np.arange(0, seconds, beat * 4):
        bar = int(bar_start / (beat * 4)) % 4
        pt = np.arange(int(beat * 4 * sample_rate)) / sample_rate
        env = np.minimum(1.0, pt / 0.4) * np.exp(-pt * 0.35)
        pad = sum(np.sin(2 * np.pi * f * pt) + 0.3 * np.sin(4 * np.pi * f * pt) for f in chords[bar])
        place(pad * env * 0.07, bar_start)
        for step in range(8):
            f = chords[bar][step % 3] * (2 if step % 2 else 4)
            at = bar_start + step * beat / 2
            arp_t = np.arange(int(0.25 * sample_rate)) / sample_rate
            place(np.sin(2 * np.pi * f * arp_t) * np.exp(-arp_t * 14) * 0.08, at)

    out /= max(1e-9, np.abs(out).max()) / 0.9
    fade = int(1.5 * sample_rate)
    out[-fade:] *= np.linspace(1, 0, fade)

    left = out + 0.02 * np.roll(out, 300)
    right = out + 0.02 * np.roll(out, 520)
    return np.stack([left, right], axis=1).astype(np.float32) * 0.95


def generate_demo_track(path: str | Path, seconds: float = 32.0) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_wav(path, synth_demo(seconds))
    return path

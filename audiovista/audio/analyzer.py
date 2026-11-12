"""Frequency / volume / beat analysis using the Fast Fourier Transform.

When a track is loaded the whole file is analysed once (short-time FFT with a
Hann window). Rendering then just looks up the pre-computed values for the
current playback time, which keeps the visuals perfectly in sync with the
audio and makes video export deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..config import SAMPLE_RATE


@dataclass
class AnalysisFrame:
    """Everything a renderer needs to draw one frame."""

    time: float
    bands: np.ndarray       # 0..1 loudness per log-spaced frequency band
    waveform: np.ndarray    # mono samples around ``time`` (-1..1)
    volume: float           # 0..1 overall loudness (RMS, normalised)
    bass: float             # 0..1 low-frequency energy
    beat: float             # 0..1 pulse that jumps to 1 on a beat then decays
    is_beat: bool           # True only on the frame a beat starts


class AudioAnalyzer:
    """Pre-computes a spectrogram, loudness envelope and beat times."""

    def __init__(self, samples: np.ndarray, sample_rate: int = SAMPLE_RATE,
                 n_bands: int = 64, n_fft: int = 4096, hop: int = 512,
                 fmin: float = 30.0, fmax: float = 16000.0):
        samples = np.asarray(samples, dtype=np.float32)
        if samples.ndim == 2:
            samples = samples.mean(axis=1)
        if samples.ndim != 1:
            raise ValueError("samples must be a 1-D (mono) or 2-D (n, channels) array")
        if sample_rate <= 0:
            raise ValueError("sample_rate must be positive")
        if n_bands < 1:
            raise ValueError("n_bands must be at least 1")
        if not 0 < fmin < fmax:
            raise ValueError("fmin must be positive and smaller than fmax")

        self.mono = samples
        self.sample_rate = int(sample_rate)
        self.n_bands = int(n_bands)
        self.n_fft = int(n_fft)
        self.hop = int(hop)
        self.fmin = float(fmin)
        self.fmax = float(min(fmax, sample_rate / 2.0))
        self.duration = len(samples) / float(sample_rate)
        self.frame_rate = sample_rate / float(hop)

        self._edges = self._band_edges()
        self._analyse()

    # ------------------------------------------------------------------ setup
    def band_frequencies(self) -> np.ndarray:
        """Centre frequency (Hz) of each band, low to high."""
        edges = np.geomspace(self.fmin, self.fmax, self.n_bands + 1)
        return np.sqrt(edges[:-1] * edges[1:])

    def _band_edges(self) -> tuple[np.ndarray, np.ndarray]:
        freqs = np.geomspace(self.fmin, self.fmax, self.n_bands + 1)
        bins = np.round(freqs * self.n_fft / self.sample_rate).astype(int)
        n_bins = self.n_fft // 2 + 1
        lo = np.clip(bins[:-1], 0, n_bins - 1)
        hi = np.clip(bins[1:], 0, n_bins)
        hi = np.maximum(hi, lo + 1)
        return lo, hi

    def _analyse(self) -> None:
        n_frames = max(1, int(np.ceil(len(self.mono) / self.hop)) + 1)
        pad = self.n_fft // 2
        padded = np.pad(self.mono, (pad, pad + self.n_fft))
        window = np.hanning(self.n_fft).astype(np.float32)
        lo, hi = self._edges

        band_db = np.empty((n_frames, self.n_bands), dtype=np.float32)
        rms = np.empty(n_frames, dtype=np.float32)
        chunk = 1024
        for start in range(0, n_frames, chunk):
            stop = min(n_frames, start + chunk)
            idx = (np.arange(start, stop)[:, None] * self.hop) + np.arange(self.n_fft)[None, :]
            frames = padded[idx]
            rms[start:stop] = np.sqrt(np.mean(frames ** 2, axis=1))
            mag = np.abs(np.fft.rfft(frames * window, axis=1)).astype(np.float32)
            mag /= (self.n_fft / 4.0)
            csum = np.concatenate([np.zeros((stop - start, 1), np.float32),
                                   np.cumsum(mag, axis=1)], axis=1)
            band_mag = (csum[:, hi] - csum[:, lo]) / (hi - lo)
            band_db[start:stop] = 20.0 * np.log10(band_mag + 1e-9)

        self.bands = self._normalise_bands(band_db)
        self._build_peak_envelope()

        peak_rms = float(np.percentile(rms, 99)) if rms.size else 0.0
        self.volume = np.clip(rms / peak_rms, 0, 1) if peak_rms > 1e-6 else np.zeros_like(rms)

        centres = self.band_frequencies()
        bass_idx = np.where(centres < 160.0)[0]
        if bass_idx.size == 0:
            bass_idx = np.array([0])
        bass_power = (10 ** (band_db[:, bass_idx] / 10.0)).mean(axis=1)
        self.bass = self.bands[:, bass_idx].mean(axis=1)
        self.beat_times = self._detect_beats(bass_power)

    def _build_peak_envelope(self) -> None:
        """Peak amplitude every 5 ms (normalised 0..1), used to draw the waveform."""
        block = max(1, self.sample_rate // 200)
        n = len(self.mono) // block
        if n == 0:
            self.peak_env = np.zeros(1, dtype=np.float32)
        else:
            peaks = np.abs(self.mono[: n * block]).reshape(n, block).max(axis=1)
            ref = float(np.percentile(peaks, 99.5))
            self.peak_env = (np.clip(peaks / ref, 0, 1) if ref > 1e-6 else np.zeros_like(peaks)).astype(np.float32)
        self.peak_rate = self.sample_rate / float(block)

    def envelope_buckets(self, first: int, count: int, bucket_seconds: float) -> np.ndarray:
        """Peak level of fixed time buckets ``[j*d, (j+1)*d)`` for j = first .. first+count-1.

        Buckets outside the track are 0. Using a fixed time grid means a bucket
        always has the same height while it scrolls across the screen.
        """
        if count < 1 or bucket_seconds <= 0:
            raise ValueError("count and bucket_seconds must be positive")
        edges = (np.arange(first, first + count + 1) * bucket_seconds * self.peak_rate).astype(int)
        edges = np.clip(edges, 0, len(self.peak_env))
        out = np.zeros(count, dtype=np.float32)
        for i in range(count):
            lo, hi = edges[i], edges[i + 1]
            if hi > lo:
                out[i] = self.peak_env[lo:hi].max()
        return out

    def recent_beats(self, t: float, window: float) -> np.ndarray:
        """Seconds since each beat that happened in the last *window* seconds."""
        if not self.beat_times.size:
            return np.zeros(0)
        lo = np.searchsorted(self.beat_times, t - window, side="left")
        hi = np.searchsorted(self.beat_times, t, side="right")
        return t - self.beat_times[lo:hi]

    def _normalise_bands(self, band_db: np.ndarray) -> np.ndarray:
        if band_db.size == 0:
            return band_db
        dynamic_range = 50.0
        global_ref = float(np.percentile(band_db, 99.5))
        if global_ref < -90.0:  # essentially silent file
            return np.zeros_like(band_db)
        per_band = np.percentile(band_db, 98, axis=0)
        ref = np.maximum(per_band, global_ref - 30.0)
        ref = 0.7 * ref + 0.3 * global_ref
        norm = (band_db - (ref - dynamic_range)) / dynamic_range
        return np.clip(norm, 0.0, 1.0) ** 1.4

    def _detect_beats(self, bass_power: np.ndarray) -> np.ndarray:
        """Return beat times (seconds) where bass energy spikes above its local average."""
        if bass_power.size < 3 or bass_power.max() <= 1e-12:
            return np.zeros(0)
        win = max(3, int(self.frame_rate * 0.6))
        kernel = np.ones(win) / win
        local_avg = np.convolve(bass_power, kernel, mode="same")
        threshold = 0.05 * float(np.percentile(bass_power, 99))
        is_peak = np.zeros_like(bass_power, dtype=bool)
        is_peak[1:-1] = (bass_power[1:-1] >= bass_power[:-2]) & (bass_power[1:-1] > bass_power[2:])
        candidates = np.where(is_peak & (bass_power > 1.4 * local_avg) & (bass_power > threshold))[0]

        min_gap = int(self.frame_rate * 0.25)
        beats: list[int] = []
        for i in candidates:
            if beats and i - beats[-1] < min_gap:
                if bass_power[i] > bass_power[beats[-1]]:
                    beats[-1] = i
                continue
            beats.append(int(i))
        return np.array(beats, dtype=np.float64) / self.frame_rate

    # ---------------------------------------------------------------- queries
    def frame_at(self, t: float, waveform_points: int = 1024) -> AnalysisFrame:
        """Analysis values at time *t* (seconds), interpolated between FFT frames."""
        t = float(min(max(t, 0.0), self.duration))
        pos = t * self.frame_rate
        i0 = int(pos)
        i1 = min(i0 + 1, len(self.bands) - 1)
        i0 = min(i0, len(self.bands) - 1)
        frac = pos - int(pos)

        bands = self.bands[i0] * (1 - frac) + self.bands[i1] * frac
        volume = float(self.volume[i0] * (1 - frac) + self.volume[i1] * frac)
        bass = float(self.bass[i0] * (1 - frac) + self.bass[i1] * frac)

        beat, is_beat = 0.0, False
        if self.beat_times.size:
            k = int(np.searchsorted(self.beat_times, t, side="right")) - 1
            if k >= 0:
                since = t - self.beat_times[k]
                beat = float(np.exp(-since / 0.18))
                is_beat = since < 1.0 / 30.0

        return AnalysisFrame(t, bands.astype(np.float32), self.waveform_at(t, waveform_points),
                             volume, bass, beat, is_beat)

    def waveform_at(self, t: float, points: int = 1024) -> np.ndarray:
        """Raw mono samples centred on *t* (zero-padded at the edges)."""
        centre = int(t * self.sample_rate)
        start = centre - points // 2
        out = np.zeros(points, dtype=np.float32)
        src_lo, src_hi = max(0, start), min(len(self.mono), start + points)
        if src_hi > src_lo:
            out[src_lo - start: src_hi - start] = self.mono[src_lo:src_hi]
        return out

    def overview(self, n_points: int) -> np.ndarray:
        """Peak amplitude of the whole track squeezed into *n_points* buckets (0..1)."""
        if n_points < 1:
            raise ValueError("n_points must be at least 1")
        if len(self.mono) == 0:
            return np.zeros(n_points, dtype=np.float32)
        edges = np.linspace(0, len(self.mono), n_points + 1).astype(int)
        absval = np.abs(self.mono)
        peaks = np.array([absval[a:max(b, a + 1)].max() if a < len(absval) else 0.0
                          for a, b in zip(edges[:-1], edges[1:])], dtype=np.float32)
        top = peaks.max()
        return peaks / top if top > 0 else peaks

    def estimate_tempo(self) -> float | None:
        """Rough tempo in BPM from the median gap between beats (None if unknown)."""
        if self.beat_times.size < 4:
            return None
        gaps = np.diff(self.beat_times)
        gaps = gaps[(gaps > 0.25) & (gaps < 2.0)]
        if gaps.size < 3:
            return None
        bpm = 60.0 / float(np.median(gaps))
        while bpm < 70:
            bpm *= 2
        while bpm > 180:
            bpm /= 2
        return round(bpm, 1)

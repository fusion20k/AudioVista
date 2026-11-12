import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

from audiovista.audio.decoder import write_wav  # noqa: E402
from audiovista.controllers import AccountController, UploadController  # noqa: E402
from audiovista.database import Database  # noqa: E402

SR = 44100


@pytest.fixture(autouse=True)
def isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("AUDIOVISTA_DATA_DIR", str(tmp_path / "data"))


@pytest.fixture
def db(tmp_path):
    return Database(tmp_path / "test.db")


@pytest.fixture
def clock():
    class Clock:
        now = 1_000_000.0

        def __call__(self):
            return self.now
    return Clock()


@pytest.fixture
def accounts(db, clock):
    return AccountController(db, clock=clock)


@pytest.fixture
def session(accounts):
    accounts.create_account("alice", "alice@example.com", "secret123")
    return accounts.login("alice@example.com", "secret123")


@pytest.fixture
def uploads(db, session, tmp_path):
    return UploadController(db, session, tmp_path / "library")


def make_tone(path, freq=440.0, seconds=2.0, amp=0.5, rate=SR):
    t = np.arange(int(seconds * rate)) / rate
    write_wav(path, amp * np.sin(2 * np.pi * freq * t), rate)
    return path


def make_clicks(path, bpm=120.0, seconds=8.0, rate=SR):
    """Kick-drum-like thumps on every beat."""
    out = np.zeros(int(seconds * rate))
    kt = np.arange(int(0.15 * rate)) / rate
    kick = np.sin(2 * np.pi * 60 * kt) * np.exp(-kt * 30)
    for b in np.arange(0.5, seconds - 0.2, 60.0 / bpm):
        i = int(b * rate)
        out[i:i + len(kick)] += kick[: len(out) - i]
    write_wav(path, out * 0.8, rate)
    return path


@pytest.fixture
def tone_wav(tmp_path):
    return make_tone(tmp_path / "tone.wav")


@pytest.fixture
def beat_wav(tmp_path):
    return make_clicks(tmp_path / "beats.wav")

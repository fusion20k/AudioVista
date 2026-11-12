<p align="center"><img src="assets/logo.png" width="120" alt="AudioVista logo"></p>

# AudioVista

**An interactive audio visualization application** — CSI 3370 group project.

AudioVista turns music into synchronized visuals. Upload an MP3, WAV or FLAC
file, pick a visualizer style, colours and background, preview it live while the
song plays, save it to your library and export it as an MP4 video for social
media or YouTube. No video-editing skills required.

The colours are meant to feel nostalgic in a calm, dreamy way: twilight-blue
panels, a warm sunset-coral accent and soft sunset skies behind the visualizers.

Team: Daniel Kraemer · Neb Horton · Theodore Kosarek · Alec Kubs

![Studio – Spectrum visualizer at sunset](docs/screenshots/05_studio_bars.png)

## Features

| Requirement (Phase 1–3) | Where it lives |
| --- | --- |
| Create account / log in (hashed passwords, lockout after 5 failed attempts) | `AccountController`, `Account`, `Database` |
| Upload audio (MP3, WAV, FLAC, 200 MB limit, drag & drop supported) | `UploadController`, `AudioFile` |
| Readable soundwave: the Waveform style scrolls the real waveform past a playhead (played part bright, upcoming part faint), plus a waveform overview on the seek bar | `WaveformRenderer`, `AudioAnalyzer.envelope_buckets` |
| Visuals that react to volume, beat and frequency (FFT) | `AudioAnalyzer`, `visuals/renderers.py` |
| Five modern styles: Spectrum (mirrored bars with falling peak caps and a reflection), Waveform (scrolling waveform over flowing bass/mid/treble waves), Radial (spectrum ring around a pulsing core, shockwaves on beats), Rings (bass/mid/treble arc gauges with spinning dash rings), Particles (hundreds of particles swirling with the music) | `visuals/renderers.py` |
| Settings: primary/accent colour, sky background (Golden, Sunset, Dawn, Twilight, Night), sensitivity, smoothing, glow, title, haze (soft vignette and warm glow) | `VisualSettings`, studio right panel |
| Play / pause, seek, volume, previous / next | `AudioPlayer`, studio transport bar |
| Sound queue ("up next") with auto-advance | `PlaybackQueue`, studio *Queue* tab |
| Save result to *My Library*, reopen and delete later | `ResultController`, studio *Saved* tab |
| Export result as MP4 (480p, 720p, 1080p, square, vertical; 30/60 fps) | `ExportController`, `ExportFile` |
| Sharp on high-DPI screens (125–200% scaling), anti-aliased UI, own taskbar icon | `ui/theme.py`, `ui/winshell.py` |

## Getting started (Windows)

Requires **Python 3.10+**.

```bash
python -m pip install -r requirements.txt
```

```bash
python main.py
```

Or double-click `run.bat`. ffmpeg is bundled through the `imageio-ffmpeg`
package, so nothing else needs installing.

First run: create an account, then press **Upload audio** (or drag a file onto
the window). Don't have a song handy? Press **Try the demo track** — AudioVista
synthesises a short royalty-free loop.

### Keyboard shortcuts

| Key | Action |
| --- | --- |
| Space | Play / pause |
| ← / → | Seek 5 seconds |
| 1 – 5 | Switch visualizer style |
| F / Esc | Full view on / off |
| N | Next track in queue |
| Ctrl+O / Ctrl+S / Ctrl+E | Upload / Save / Export |

## Running the tests

```bash
python -m pytest
```

The suite (71 tests) covers the classes chosen for Phase 4 — `AccountController`
and `AudioAnalyzer` — with normal and incorrect inputs, plus the upload, save and
export use cases, `VisualSettings`, `PlaybackQueue`, every renderer and the player
clock. Tests use a temporary data folder and never touch your real library.

To regenerate the screenshots in `docs/screenshots/` (drives the real UI with no
visible window):

```bash
python tools/take_screenshots.py
```

## Architecture

The code follows the Phase 3 design class diagram and sequence diagrams.

```
AudioVistaUI (ui/app.py)
 ├─ AccountController ── Account ── Database (SQLite)
 ├─ UploadController ─── AudioFile ── decoder (ffmpeg / wave) ── AudioAnalyzer (FFT)
 ├─ VisualizerController ─ Visualization ── VisualSettings
 │                          └─ VisualizerEngine ("Audio Visualizer Generation System")
 ├─ ResultController ─── Session ── Database
 └─ ExportController ─── ExportFile (ffmpeg H.264 + AAC)
```

```
audiovista/
  config.py          constants, data folder (%LOCALAPPDATA%\AudioVista)
  errors.py          user-facing exceptions
  models.py          Account, AudioFile, VisualSettings, Visualization, Session, PlaybackQueue
  database.py        Database (accounts, audio_files, visualizations, exports)
  controllers.py     Account/Upload/Visualizer/Result/Export controllers
  export.py          ExportFile – pipes rendered frames + audio into ffmpeg
  audio/
    decoder.py       MP3/WAV/FLAC -> numpy samples
    analyzer.py      AudioAnalyzer – STFT, log-spaced bands, loudness, beat detection, tempo
    player.py        AudioPlayer – pygame.mixer playback with an accurate clock
    demo.py          synthesised demo track
  visuals/
    renderers.py     Spectrum, Waveform, Radial, Rings, Particles
    backgrounds.py   sky gradients (dithered, no banding) + beat flash
    retro.py         "Haze": soft vignette and warm glow
    engine.py        VisualizerEngine – live preview and export frames, bloom
  ui/
    app.py           AudioVistaUI – window, main loop, background tasks
    auth_screens.py  Log in / Create account
    studio.py        main workspace
    dialogs.py       Save, Export, Confirm dialogs and file pickers
    widgets.py       buttons, inputs, sliders, lists, seek bar, toasts
    theme.py         colours, fonts, DPI scaling, anti-aliased shapes, icons, logo
    winshell.py      Windows taskbar icon / app identity (ctypes)
assets/            logo PNGs and the .ico used for the taskbar (tools/make_icons.py)
tests/               pytest suite
tools/               screenshot and icon scripts
```

### How the visualizer stays in sync

When a track loads, `AudioAnalyzer` runs a short-time Fast Fourier Transform over
the whole file (4096-sample Hann window, 512-sample hop ≈ 86 analyses per second)
and stores 64 log-spaced frequency bands, an RMS loudness envelope and detected
beat times (bass-energy onsets above the local average). Each frame, the renderer
asks for the analysis at the player's current position, so the visuals line up
with what you hear, and video export renders exactly the same frames.

### High-DPI displays

AudioVista tells Windows it is DPI-aware and scales its whole interface by the
display scale (e.g. 2x at 200%), so text and shapes are drawn at full
resolution instead of being stretched. Shapes and icons are super-sampled for
smooth edges. Set `AUDIOVISTA_UI_SCALE` (e.g. `1.25`) to override the scale.

### Where data is stored

Accounts, the library and saved visualizations are kept in
`%LOCALAPPDATA%\AudioVista` (SQLite database + copies of uploaded audio). Set the
`AUDIOVISTA_DATA_DIR` environment variable to use a different folder. Exported
videos go to `Videos\AudioVista` by default.

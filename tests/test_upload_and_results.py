"""Upload Audio, Generate Visuals, Save Result and Export Result use cases."""

import threading
from pathlib import Path

import pytest

from audiovista.controllers import (ExportController, ResultController, UploadController,
                                    VisualizerController)
from audiovista.errors import (AudioDecodeError, AuthenticationError, ExportCancelled,
                               FileTooLargeError, NoAudioError, NotSavedError,
                               UnsupportedFormatError, ValidationError)
from audiovista.models import Session

from .conftest import make_tone


# ---------------------------------------------------------------- upload

def test_validate_format_accepts_supported_files(tmp_path, tone_wav):
    assert UploadController.validate_format(tone_wav)
    for ext in (".mp3", ".flac", ".WAV"):
        f = tmp_path / f"song{ext}"
        f.write_bytes(b"x" * 10)
        assert UploadController.validate_format(f)


@pytest.mark.parametrize("name", ["notes.txt", "video.mp4", "song.ogg", "noextension"])
def test_validate_format_rejects_other_types(tmp_path, name):
    f = tmp_path / name
    f.write_bytes(b"data")
    with pytest.raises(UnsupportedFormatError):
        UploadController.validate_format(f)


def test_validate_format_rejects_missing_empty_and_huge(tmp_path, monkeypatch):
    with pytest.raises(ValidationError, match="not found"):
        UploadController.validate_format(tmp_path / "missing.mp3")
    empty = tmp_path / "empty.wav"
    empty.write_bytes(b"")
    with pytest.raises(ValidationError, match="empty"):
        UploadController.validate_format(empty)
    big = tmp_path / "big.wav"
    big.write_bytes(b"x" * 2048)
    monkeypatch.setattr("audiovista.controllers.MAX_FILE_SIZE", 1024)
    with pytest.raises(FileTooLargeError):
        UploadController.validate_format(big)


def test_upload_copies_into_library_and_sets_current(uploads, session, tone_wav, tmp_path):
    audio = uploads.upload_audio(tone_wav)
    assert audio.audio_id is not None
    assert audio.duration == pytest.approx(2.0, abs=0.01)
    assert Path(audio.file_path).is_file()
    assert Path(audio.file_path).parent == tmp_path / "library" / str(session.get_account_id())
    assert session.get_current_audio() is audio
    assert [a.audio_id for a in uploads.list_library()] == [audio.audio_id]


def test_uploading_same_file_twice_reuses_entry(uploads, tone_wav):
    first = uploads.upload_audio(tone_wav)
    second = uploads.upload_audio(tone_wav)
    assert first.audio_id == second.audio_id
    assert len(uploads.list_library()) == 1


def test_corrupt_audio_is_rejected(uploads, tmp_path):
    bad = tmp_path / "broken.mp3"
    bad.write_bytes(b"this is not really an mp3 file" * 20)
    with pytest.raises(AudioDecodeError):
        uploads.upload_audio(bad)
    assert uploads.list_library() == []


def test_upload_requires_login(db, tone_wav, tmp_path):
    controller = UploadController(db, Session(), tmp_path / "lib")
    with pytest.raises(AuthenticationError):
        controller.upload_audio(tone_wav)


def test_remove_from_library_deletes_file(uploads, tone_wav):
    audio = uploads.upload_audio(tone_wav)
    uploads.remove_from_library(audio.audio_id)
    assert uploads.list_library() == []
    assert not Path(audio.file_path).exists()
    with pytest.raises(NoAudioError):
        uploads.load_from_library(audio.audio_id)


# ------------------------------------------------- visuals, save, export

def test_generate_visuals_requires_audio(session):
    with pytest.raises(NoAudioError):
        VisualizerController(session).generate_visuals("bars")


def test_generate_save_and_reopen(db, session, uploads, tone_wav):
    uploads.upload_audio(tone_wav)
    vis = VisualizerController(session).generate_visuals("circular")
    assert vis.settings.wave_type == "circular" and not vis.save_status

    results = ResultController(db, session)
    with pytest.raises(ValidationError):
        results.save_result("   ")
    vid = results.save_result("My first visual")
    assert vis.save_status and vis.visualization_id == vid

    # Changing settings marks it unsaved; saving again updates the same row.
    VisualizerController(session).update_settings(color="#ff3cac")
    assert not vis.save_status
    assert results.save_result("Renamed") == vid
    listing = results.list_results()
    assert len(listing) == 1 and listing[0]["name"] == "Renamed"

    reopened = results.open_result(vid)
    assert reopened.settings.color == "#ff3cac"
    assert reopened.settings.wave_type == "circular"
    assert reopened.audio_file.is_loaded


def test_export_requires_saved_visualization(db, session, uploads, tone_wav, tmp_path):
    uploads.upload_audio(tone_wav)
    VisualizerController(session).generate_visuals("bars")
    exporter = ExportController(db, session)
    with pytest.raises(NotSavedError):
        exporter.export_result("mp4", "480p", tmp_path / "out.mp4")


def test_export_rejects_bad_options(db, session, uploads, tone_wav, tmp_path):
    uploads.upload_audio(tone_wav)
    VisualizerController(session).generate_visuals("bars")
    ResultController(db, session).save_result("x")
    exporter = ExportController(db, session)
    with pytest.raises(ValidationError):
        exporter.export_result("avi", "480p", tmp_path / "out.avi")
    with pytest.raises(ValidationError):
        exporter.export_result("mp4", "8K", tmp_path / "out.mp4")


def test_export_writes_mp4(db, session, uploads, tmp_path):
    uploads.upload_audio(make_tone(tmp_path / "short.wav", seconds=1.5))
    VisualizerController(session).generate_visuals("circular")
    vid = ResultController(db, session).save_result("Export me")
    progress = []
    export = ExportController(db, session).export_result(
        "mp4", "480p", tmp_path / "exports" / "video", fps=30, progress=progress.append)
    out = Path(export.output_path)
    assert out.suffix == ".mp4" and out.stat().st_size > 1000
    assert progress[-1] == 1.0
    assert db.list_exports(vid)[0]["file_name"] == "video.mp4"


def test_export_can_be_cancelled(db, session, uploads, tone_wav, tmp_path):
    uploads.upload_audio(tone_wav)
    VisualizerController(session).generate_visuals("bars")
    ResultController(db, session).save_result("cancel me")
    cancel = threading.Event()
    cancel.set()
    with pytest.raises(ExportCancelled):
        ExportController(db, session).export_result("mp4", "480p", tmp_path / "c.mp4",
                                                    cancel_event=cancel)
    assert not (tmp_path / "c.mp4").exists()

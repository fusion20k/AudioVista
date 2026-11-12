"""SQLite persistence for accounts, uploaded audio, saved visualizations and exports."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from .models import Account, AudioFile, Visualization, VisualSettings

_SCHEMA = """
CREATE TABLE IF NOT EXISTS accounts (
    user_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    username        TEXT NOT NULL UNIQUE COLLATE NOCASE,
    email           TEXT NOT NULL UNIQUE COLLATE NOCASE,
    password_hash   TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    failed_attempts INTEGER NOT NULL DEFAULT 0,
    locked_until    REAL NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS audio_files (
    audio_id    INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES accounts(user_id) ON DELETE CASCADE,
    file_name   TEXT NOT NULL,
    file_type   TEXT NOT NULL,
    file_size   INTEGER NOT NULL,
    file_path   TEXT NOT NULL,
    title       TEXT NOT NULL,
    artist      TEXT NOT NULL DEFAULT '',
    duration    REAL NOT NULL DEFAULT 0,
    uploaded_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS visualizations (
    visualization_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id          INTEGER NOT NULL REFERENCES accounts(user_id) ON DELETE CASCADE,
    audio_id         INTEGER NOT NULL REFERENCES audio_files(audio_id) ON DELETE CASCADE,
    name             TEXT NOT NULL,
    date_created     TEXT NOT NULL,
    date_modified    TEXT NOT NULL,
    settings_json    TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS exports (
    export_id        INTEGER PRIMARY KEY AUTOINCREMENT,
    visualization_id INTEGER NOT NULL REFERENCES visualizations(visualization_id) ON DELETE CASCADE,
    file_name        TEXT NOT NULL,
    file_format      TEXT NOT NULL,
    export_date      TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


class Database:
    """Thin data-access layer. A new connection is used per call so the
    database can safely be used from background loading threads."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        else:  # keep one shared connection alive for in-memory databases
            self._memory_conn = sqlite3.connect(":memory:", check_same_thread=False)
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    @contextmanager
    def _connect(self):
        if self.path == ":memory:":
            conn = self._memory_conn
        else:
            conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            if self.path != ":memory:":
                conn.close()

    # ------------------------------------------------------------- accounts
    @staticmethod
    def _account(row) -> Account | None:
        if row is None:
            return None
        return Account(row["user_id"], row["username"], row["email"],
                       row["password_hash"], row["created_at"])

    def find_account_by_email(self, email: str) -> Account | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM accounts WHERE email = ?", (email.strip(),)).fetchone()
        return self._account(row)

    def find_account_by_username(self, username: str) -> Account | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM accounts WHERE username = ?",
                               (username.strip(),)).fetchone()
        return self._account(row)

    def save_account(self, account: Account) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO accounts (username, email, password_hash, created_at) VALUES (?,?,?,?)",
                (account.username, account.email, account.password_hash, account.created_at))
            account.user_id = cur.lastrowid
        return account.user_id

    def get_login_state(self, user_id: int) -> tuple[int, float]:
        with self._connect() as conn:
            row = conn.execute("SELECT failed_attempts, locked_until FROM accounts WHERE user_id = ?",
                               (user_id,)).fetchone()
        return (row["failed_attempts"], row["locked_until"]) if row else (0, 0.0)

    def set_login_state(self, user_id: int, failed_attempts: int, locked_until: float) -> None:
        with self._connect() as conn:
            conn.execute("UPDATE accounts SET failed_attempts = ?, locked_until = ? WHERE user_id = ?",
                         (failed_attempts, locked_until, user_id))

    # ----------------------------------------------------------- audio files
    def save_audio(self, user_id: int, audio: AudioFile) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                """INSERT INTO audio_files (user_id, file_name, file_type, file_size, file_path,
                                            title, artist, duration, uploaded_at)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (user_id, audio.file_name, audio.file_type, audio.file_size, audio.file_path,
                 audio.title, audio.artist, audio.duration, _now()))
            audio.audio_id = cur.lastrowid
        return audio.audio_id

    def get_audio(self, user_id: int, audio_id: int) -> AudioFile | None:
        with self._connect() as conn:
            row = conn.execute("SELECT * FROM audio_files WHERE user_id = ? AND audio_id = ?",
                               (user_id, audio_id)).fetchone()
        return AudioFile.from_record(dict(row)) if row else None

    def find_audio(self, user_id: int, file_name: str, file_size: int) -> AudioFile | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM audio_files WHERE user_id = ? AND file_name = ? AND file_size = ?",
                (user_id, file_name, file_size)).fetchone()
        return AudioFile.from_record(dict(row)) if row else None

    def list_audio(self, user_id: int) -> list[AudioFile]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM audio_files WHERE user_id = ? ORDER BY audio_id DESC",
                                (user_id,)).fetchall()
        return [AudioFile.from_record(dict(r)) for r in rows]

    def delete_audio(self, user_id: int, audio_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM audio_files WHERE user_id = ? AND audio_id = ?",
                               (user_id, audio_id))
        return cur.rowcount > 0

    # -------------------------------------------------------- visualizations
    def save_visualization(self, user_id: int, visualization: Visualization) -> int:
        settings_json = json.dumps(visualization.settings.to_dict())
        audio_id = visualization.audio_file.audio_id
        with self._connect() as conn:
            if visualization.visualization_id is not None:
                cur = conn.execute(
                    """UPDATE visualizations SET name = ?, settings_json = ?, date_modified = ?,
                              audio_id = ? WHERE visualization_id = ? AND user_id = ?""",
                    (visualization.name, settings_json, _now(), audio_id,
                     visualization.visualization_id, user_id))
                if cur.rowcount:
                    return visualization.visualization_id
            cur = conn.execute(
                """INSERT INTO visualizations (user_id, audio_id, name, date_created,
                                               date_modified, settings_json)
                   VALUES (?,?,?,?,?,?)""",
                (user_id, audio_id, visualization.name, visualization.date_created, _now(),
                 settings_json))
            visualization.visualization_id = cur.lastrowid
        return visualization.visualization_id

    def list_visualizations(self, user_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute(
                """SELECT v.visualization_id, v.name, v.date_created, v.date_modified,
                          v.settings_json, a.title AS audio_title, a.audio_id
                   FROM visualizations v JOIN audio_files a ON a.audio_id = v.audio_id
                   WHERE v.user_id = ? ORDER BY v.date_modified DESC, v.visualization_id DESC""",
                (user_id,)).fetchall()
        return [dict(r) for r in rows]

    def get_visualization(self, user_id: int, visualization_id: int) -> Visualization | None:
        with self._connect() as conn:
            row = conn.execute(
                """SELECT v.*, a.* FROM visualizations v
                   JOIN audio_files a ON a.audio_id = v.audio_id
                   WHERE v.user_id = ? AND v.visualization_id = ?""",
                (user_id, visualization_id)).fetchone()
        if row is None:
            return None
        rec = dict(row)
        audio = AudioFile.from_record(rec)
        settings = VisualSettings.from_dict(json.loads(rec["settings_json"]))
        return Visualization(audio, settings, rec["name"], rec["visualization_id"],
                             rec["date_created"], save_status=True)

    def delete_visualization(self, user_id: int, visualization_id: int) -> bool:
        with self._connect() as conn:
            cur = conn.execute("DELETE FROM visualizations WHERE user_id = ? AND visualization_id = ?",
                               (user_id, visualization_id))
        return cur.rowcount > 0

    # ---------------------------------------------------------------- exports
    def save_export(self, visualization_id: int, file_name: str, file_format: str) -> int:
        with self._connect() as conn:
            cur = conn.execute(
                "INSERT INTO exports (visualization_id, file_name, file_format, export_date) "
                "VALUES (?,?,?,?)", (visualization_id, file_name, file_format, _now()))
        return cur.lastrowid

    def list_exports(self, visualization_id: int) -> list[dict]:
        with self._connect() as conn:
            rows = conn.execute("SELECT * FROM exports WHERE visualization_id = ? ORDER BY export_id",
                                (visualization_id,)).fetchall()
        return [dict(r) for r in rows]

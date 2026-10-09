"""Durable changed-person planning; publish a baseline only after a safe run."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

VERSION = 1
EXCLUDED = {"all", "_smart_albums", "_smart_albums_v2", "_smart_albums_simple_preview",
            "_duplicates", "_near_visual_review"}


def read_people_file(path: Path | None) -> set[str] | None:
    if path is None:
        return None
    names = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(names, list) or any(
        not isinstance(name, str) or not name or name.startswith((".", "_"))
        or Path(name).name != name or "/" in name or "\\" in name for name in names
    ):
        raise ValueError("Invalid changed-person scope; refusing to broaden it")
    return set(names)


def capture(root: Path) -> dict[str, tuple[str, str]]:
    if not root.is_dir():
        raise OSError(f"Organized destination unavailable: {root}")
    entries = {}
    def failed(error):
        raise error
    for person in sorted(root.iterdir()):
        if person.name.startswith((".", "_")) or not person.is_dir():
            continue
        for current, dirs, files in os.walk(person, onerror=failed, followlinks=False):
            dirs[:] = [name for name in dirs if name.casefold() not in EXCLUDED
                       and not name.startswith(".")]
            # Directory membership catches empty folders and external moves too.
            for path in [Path(current), *(Path(current) / name for name in files
                                          if not name.startswith("."))]:
                stat = path.lstat()
                signature = (stat.st_mode, stat.st_dev, stat.st_ino, stat.st_size,
                             stat.st_mtime_ns, stat.st_ctime_ns)
                entries[path.relative_to(root).as_posix()] = (
                    person.name, json.dumps(signature, separators=(",", ":")))
    return entries


def fingerprint(entries: dict) -> str:
    return hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class Inventory:
    def __init__(self, path: Path, root: Path, policy: str = "1"):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.root = str(root.resolve())
        self.policy = policy
        self.db = sqlite3.connect(path)
        try:
            self.db.execute("PRAGMA synchronous=FULL")
            self.db.executescript("""
            CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS baseline (path TEXT PRIMARY KEY, person TEXT NOT NULL, signature TEXT NOT NULL);
            CREATE TABLE IF NOT EXISTS runs (run_id TEXT PRIMARY KEY, changes TEXT NOT NULL, committed INTEGER NOT NULL);
            """)
        except Exception:
            self.db.close()
            raise

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.db.close()

    def changes(self, entries: dict) -> dict:
        meta = dict(self.db.execute("SELECT key, value FROM metadata"))
        if int(meta.get("version", "0")) > VERSION:
            raise ValueError("Daily inventory requires a newer version; refusing to downgrade")
        valid = (meta.get("version") == str(VERSION) and meta.get("root") == self.root
                 and meta.get("policy") == self.policy)
        baseline = {path: (person, sig) for path, person, sig in
                    self.db.execute("SELECT path, person, signature FROM baseline")} if valid else {}
        added = sorted(entries.keys() - baseline.keys())
        removed = sorted(baseline.keys() - entries.keys())
        changed = sorted(path for path in entries.keys() & baseline.keys() if entries[path] != baseline[path])
        people = {entries[path][0] for path in added + changed}
        people.update(baseline[path][0] for path in removed)
        return {"full": not valid, "reason": "first safe baseline" if not valid else "changed files and folders",
                "people": sorted(people), "added": added, "changed": changed, "removed": removed,
                "fingerprint": fingerprint(entries)}

    def checkpoint(self, run_id: str, changes: dict) -> None:
        with self.db:
            self.db.execute("INSERT INTO runs VALUES(?, ?, 0) ON CONFLICT(run_id) DO UPDATE SET changes=excluded.changes",
                            (run_id, json.dumps(changes)))

    def publish(self, run_id: str, entries: dict, people: set[str] | None) -> None:
        with self.db:
            if people is None:
                self.db.execute("DELETE FROM baseline")
            else:
                self.db.executemany("DELETE FROM baseline WHERE person=?", [(name,) for name in people])
            self.db.executemany("INSERT OR REPLACE INTO baseline VALUES(?, ?, ?)",
                [(path, person, sig) for path, (person, sig) in entries.items()
                 if people is None or person in people])
            self.db.executemany("INSERT OR REPLACE INTO metadata VALUES(?, ?)",
                                [("version", str(VERSION)), ("root", self.root), ("policy", self.policy)])
            self.db.execute("UPDATE runs SET committed=1 WHERE run_id=?", (run_id,))
            self.db.execute("DELETE FROM runs WHERE committed=1 AND run_id NOT IN "
                            "(SELECT run_id FROM runs WHERE committed=1 ORDER BY rowid DESC LIMIT 20)")


@contextmanager
def open_inventory(path: Path, root: Path, policy: str):
    try:
        store = Inventory(path, root, policy)
    except sqlite3.DatabaseError:
        backup = path.with_name(path.name + f".corrupt.{time.time_ns()}")
        path.replace(backup)
        print(f"Saved daily index was damaged; preserving {backup.name} and rebuilding a full safe baseline.", flush=True)
        store = Inventory(path, root, policy)
    with store:
        yield store

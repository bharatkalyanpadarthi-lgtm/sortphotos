"""Durable, content-addressed copy outcomes; legacy path checkpoints are hints only."""

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import content_identity


class CopyJournal:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.connection = sqlite3.connect(self.root / ".copy_operations.sqlite3")
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("PRAGMA synchronous=FULL")
        self.connection.execute(
            "CREATE TABLE IF NOT EXISTS copy_operations ("
            "operation_id TEXT PRIMARY KEY, source_sha256 TEXT NOT NULL, "
            "destination TEXT, state TEXT NOT NULL, updated_at REAL NOT NULL)"
        )
        self.connection.commit()

    @staticmethod
    def operation_id(person: str, digest: str, category: str) -> str:
        return hashlib.sha256(
            json.dumps([1, person.casefold(), digest, category]).encode()
        ).hexdigest()

    def verified_destination(self, operation_id: str, digest: str) -> Path | None:
        row = self.connection.execute(
            "SELECT destination FROM copy_operations "
            "WHERE operation_id=? AND source_sha256=? AND state='verified'",
            (operation_id, digest),
        ).fetchone()
        if row is None or not row[0]:
            return None
        destination = (self.root / row[0]).resolve()
        try:
            destination.relative_to(self.root)
            if content_identity.content_sha256(destination) == digest:
                return destination
        except (OSError, ValueError):
            pass
        return None

    def planned(self, operation_id: str, digest: str) -> None:
        self.connection.execute(
            "INSERT INTO copy_operations VALUES(?, ?, NULL, 'planned', ?) "
            "ON CONFLICT(operation_id) DO UPDATE SET state='planned', updated_at=excluded.updated_at",
            (operation_id, digest, time.time()),
        )
        self.connection.commit()

    def completed(self, operation_id: str, digest: str, destination: Path) -> None:
        relative = destination.resolve().relative_to(self.root).as_posix()
        if content_identity.content_sha256(destination) != digest:
            raise ValueError("Cannot commit unverified original copy")
        self.connection.execute(
            "INSERT INTO copy_operations VALUES(?, ?, ?, 'verified', ?) "
            "ON CONFLICT(operation_id) DO UPDATE SET destination=excluded.destination, "
            "source_sha256=excluded.source_sha256, state='verified', updated_at=excluded.updated_at",
            (operation_id, digest, relative, time.time()),
        )
        self.connection.commit()

    def close(self):
        self.connection.close()

"""Private local benchmark stage cache, independent of application modules."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from pathlib import Path


SCHEMA_VERSION = 1


def _validate_json(value) -> None:
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise TypeError("JSON object keys must be strings")
            _validate_json(item)
    elif type(value) is list:
        for item in value:
            _validate_json(item)
    elif type(value) is float:
        if not math.isfinite(value):
            raise ValueError("JSON numbers must be finite")
    elif value is not None and type(value) not in (str, int, bool):
        raise TypeError("Payload must contain only JSON types")


def _unique_object(pairs) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON object key")
        result[key] = value
    return result


def _checksum(signature: str, stage: str, payload: str) -> str:
    digest = hashlib.sha256(
        json.dumps([signature, stage], separators=(",", ":")).encode("ascii")
    )
    digest.update(b"\0")
    digest.update(payload.encode("utf-8"))
    return digest.hexdigest()


class BenchmarkCheckpoints:
    """Cache dict/list stages under an exact caller-supplied signature.

    Use inside a context manager. Each put commits independently, so completed
    stages survive later failures. Bad payload rows are misses; SQLite errors
    and unsupported schema versions propagate for the caller to fail closed.

    Opening records transactional recency without eviction, protecting other
    live runs. Call prune() with other cache users closed to retain the current
    namespace and the two most recently opened others. Checksums detect
    accidental corruption, not deliberate edits by someone controlling the DB.
    """

    def __init__(self, path: Path, signature: str):
        if type(signature) is not str:
            raise TypeError("signature must be a string")
        self._path = path.expanduser()
        self._signature = signature
        self._connection: sqlite3.Connection | None = None

    def __enter__(self) -> BenchmarkCheckpoints:
        if self._connection is not None:
            raise RuntimeError("Checkpoint cache is already open")
        connection = sqlite3.connect(self._path)
        try:
            connection.execute("PRAGMA synchronous=FULL")
            connection.execute("PRAGMA foreign_keys=ON")
            with connection:
                connection.execute("BEGIN IMMEDIATE")
                objects = connection.execute(
                    "SELECT name FROM sqlite_master WHERE name NOT LIKE 'sqlite_%'"
                ).fetchall()
                if not objects:
                    connection.execute(
                        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
                    )
                    connection.execute(
                        "CREATE TABLE namespaces (signature TEXT PRIMARY KEY, "
                        "opened_order INTEGER NOT NULL UNIQUE)"
                    )
                    connection.execute(
                        "CREATE TABLE checkpoints ("
                        "signature TEXT NOT NULL, stage TEXT NOT NULL, payload TEXT NOT NULL, "
                        "checksum TEXT NOT NULL, PRIMARY KEY (signature, stage), "
                        "FOREIGN KEY (signature) REFERENCES namespaces(signature) ON DELETE CASCADE)"
                    )
                    connection.execute(
                        "INSERT INTO metadata (key, value) VALUES ('schema_version', ?)",
                        (str(SCHEMA_VERSION),),
                    )
                versions = connection.execute(
                    "SELECT value FROM metadata WHERE key = 'schema_version'"
                ).fetchall()
                if versions != [(str(SCHEMA_VERSION),)]:
                    raise ValueError("Unsupported checkpoint schema version")
                connection.execute(
                    "SELECT signature, stage, payload, checksum FROM checkpoints LIMIT 0"
                )
                connection.execute("SELECT signature, opened_order FROM namespaces LIMIT 0")
                # BEGIN IMMEDIATE serializes this clock-free, indexed recency counter.
                connection.execute(
                    "INSERT INTO namespaces (signature, opened_order) "
                    "VALUES (?, (SELECT COALESCE(MAX(opened_order), 0) + 1 FROM namespaces)) "
                    "ON CONFLICT (signature) DO UPDATE SET opened_order = excluded.opened_order",
                    (self._signature,),
                )
        except BaseException:
            connection.close()
            raise
        self._connection = connection
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        connection = self._connection
        self._connection = None
        if connection is not None:
            connection.close()

    def _db(self) -> sqlite3.Connection:
        if self._connection is None:
            raise RuntimeError("Use checkpoint cache inside a context manager")
        return self._connection

    def get(self, stage: str) -> dict | list | None:
        connection = self._db()
        if type(stage) is not str:
            raise TypeError("stage must be a string")
        rows = connection.execute(
            "SELECT payload, checksum FROM checkpoints WHERE signature = ? AND stage = ?",
            (self._signature, stage),
        ).fetchall()
        if len(rows) != 1 or any(type(value) is not str for value in rows[0]):
            return None
        try:
            if rows[0][1] != _checksum(self._signature, stage, rows[0][0]):
                return None
            payload = json.loads(rows[0][0], object_pairs_hook=_unique_object)
            if type(payload) not in (dict, list):
                return None
            _validate_json(payload)
        except (TypeError, ValueError, RecursionError):
            return None
        return payload

    def put(self, stage: str, payload: dict | list) -> None:
        connection = self._db()
        if type(stage) is not str:
            raise TypeError("stage must be a string")
        if type(payload) not in (dict, list):
            raise TypeError("Checkpoint payload must be a JSON object or array")
        # Serialize before beginning a transaction, including circularity checks.
        serialized = json.dumps(payload, allow_nan=False, separators=(",", ":"))
        _validate_json(payload)
        checksum = _checksum(self._signature, stage, serialized)
        with connection:
            connection.execute(
                "INSERT INTO checkpoints (signature, stage, payload, checksum) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (signature, stage) DO UPDATE "
                "SET payload = excluded.payload, checksum = excluded.checksum",
                (self._signature, stage, serialized, checksum),
            )

    def prune(self) -> None:
        """Keep this namespace and the two newest others; call with other users closed.

        No live-process registry or leases are maintained. Explicit pruning can
        evict another open run, so it is never performed automatically. Freed
        SQLite pages are reusable; this does not VACUUM or shrink the DB file.
        """
        connection = self._db()
        with connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "DELETE FROM namespaces WHERE signature != ? AND signature NOT IN ("
                "SELECT signature FROM namespaces WHERE signature != ? "
                "ORDER BY opened_order DESC LIMIT 2)",
                (self._signature, self._signature),
            )

    def invalidate(self) -> None:
        """Discard this run's rows after input drift, not an ordinary interruption."""
        with self._db() as connection:
            connection.execute("DELETE FROM checkpoints WHERE signature = ?", (self._signature,))

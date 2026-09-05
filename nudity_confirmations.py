#!/usr/bin/env python3
"""Durable user-confirmed nudity overrides keyed by content hash and path."""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import pipeline_paths


DEFAULT_PATH = (
    pipeline_paths.SORTED_ROOT
    / "_source_review"
    / "nudity_audits"
    / "manual_nudity_confirmations.jsonl"
)
SCHEMA_VERSION = 1

_CACHE_PATH: Path | None = None
_CACHE_MTIME_NS = -1
_CACHE_HASHES: set[str] = set()
_CACHE_PATHS: set[str] = set()


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical(path: Path) -> str:
    return str(path.expanduser().resolve(strict=False))


def _load(path: Path = DEFAULT_PATH) -> tuple[set[str], set[str]]:
    global _CACHE_PATH, _CACHE_MTIME_NS, _CACHE_HASHES, _CACHE_PATHS

    path = path.expanduser()
    try:
        mtime_ns = path.stat().st_mtime_ns
    except OSError:
        mtime_ns = -1
    if _CACHE_PATH == path and _CACHE_MTIME_NS == mtime_ns:
        return _CACHE_HASHES, _CACHE_PATHS

    hashes: set[str] = set()
    paths: set[str] = set()
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                try:
                    record = json.loads(line)
                except (TypeError, ValueError):
                    continue
                sha256 = str(record.get("sha256") or "").strip().lower()
                if len(sha256) == 64:
                    hashes.add(sha256)
                for key in ("source_path", "destination_path"):
                    value = str(record.get(key) or "").strip()
                    if value:
                        paths.add(_canonical(Path(value)))

    _CACHE_PATH = path
    _CACHE_MTIME_NS = mtime_ns
    _CACHE_HASHES = hashes
    _CACHE_PATHS = paths
    return hashes, paths


def is_confirmed(
    path: Path,
    *,
    sha256: str | None = None,
    confirmations_path: Path = DEFAULT_PATH,
) -> bool:
    hashes, paths = _load(confirmations_path)
    if _canonical(path) in paths:
        return True
    normalized_hash = str(sha256 or "").strip().lower()
    return bool(normalized_hash and normalized_hash in hashes)


def record_confirmation(
    source: Path,
    destination: Path,
    *,
    person: str,
    reason: str,
    sha256: str | None = None,
    confirmations_path: Path = DEFAULT_PATH,
) -> str:
    """Append a durable confirmation before moving the source file."""
    global _CACHE_PATH, _CACHE_MTIME_NS, _CACHE_HASHES, _CACHE_PATHS

    confirmations_path = confirmations_path.expanduser()
    file_hash = str(sha256 or sha256_file(source)).strip().lower()
    if len(file_hash) != 64:
        raise ValueError(f"invalid SHA-256 for {source}")
    _hashes, confirmed_paths = _load(confirmations_path)
    source_key = _canonical(source)
    destination_key = _canonical(destination)
    if source_key in confirmed_paths and destination_key in confirmed_paths:
        return file_hash

    confirmations_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "schema_version": SCHEMA_VERSION,
        "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "person": person,
        "sha256": file_hash,
        "source_path": source_key,
        "destination_path": destination_key,
        "reason": reason,
    }
    with confirmations_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())

    # Keep the in-process index current so a large bulk confirmation remains
    # O(n) instead of reparsing the growing JSONL file after every append.
    _CACHE_PATH = confirmations_path
    try:
        _CACHE_MTIME_NS = confirmations_path.stat().st_mtime_ns
    except OSError:
        _CACHE_MTIME_NS = -1
    _CACHE_HASHES.add(file_hash)
    _CACHE_PATHS.update((source_key, destination_key))
    return file_hash

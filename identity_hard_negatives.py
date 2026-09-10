#!/usr/bin/env python3
"""Persistent user-confirmed evidence that a face is not a candidate person."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path

import numpy as np


SCHEMA_VERSION = 1


def default_path(cache_dir: Path) -> Path:
    return cache_dir / "identity_hard_negatives.json"


def load(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"version": SCHEMA_VERSION, "examples": []}
    examples = payload.get("examples", [])
    if not isinstance(examples, list):
        examples = []
    return {"version": SCHEMA_VERSION, "examples": examples}


def save(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(
                {"version": SCHEMA_VERSION, "examples": payload.get("examples", [])},
                handle,
                indent=2,
                sort_keys=True,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def encode_embedding(embedding: np.ndarray) -> tuple[str, int, str]:
    vector = np.asarray(embedding, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    vector = vector / max(norm, 1e-9)
    raw = vector.tobytes()
    return base64.b64encode(raw).decode("ascii"), int(vector.size), hashlib.sha256(raw).hexdigest()


def decode_embedding(item: dict) -> np.ndarray | None:
    try:
        raw = base64.b64decode(str(item.get("embedding_base64", "")), validate=True)
        dimension = int(item.get("embedding_dimension", 0))
        vector = np.frombuffer(raw, dtype=np.float32).copy()
        if dimension <= 0 or vector.size < dimension:
            return None
        vector = vector[:dimension]
        return vector / max(float(np.linalg.norm(vector)), 1e-9)
    except (ValueError, TypeError):
        return None


def record(
    path: Path,
    *,
    person: str,
    source_path: Path,
    face_index: int,
    embedding: np.ndarray,
    reason: str = "explicit_user_rejection",
) -> bool:
    encoded, dimension, embedding_hash = encode_embedding(embedding)
    payload = load(path)
    canonical_source = str(source_path.expanduser().resolve(strict=False))
    duplicate = any(
        str(item.get("person", "")).casefold() == person.casefold()
        and str(item.get("embedding_sha256", "")) == embedding_hash
        for item in payload["examples"]
    )
    if duplicate:
        return False
    payload["examples"].append({
        "person": person,
        "source_path": canonical_source,
        "face_index": int(face_index),
        "embedding_base64": encoded,
        "embedding_dimension": dimension,
        "embedding_sha256": embedding_hash,
        "reason": reason,
        "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    })
    payload["examples"].sort(key=lambda item: (
        str(item.get("person", "")).casefold(),
        str(item.get("confirmed_at", "")),
        str(item.get("embedding_sha256", "")),
    ))
    save(path, payload)
    return True


@lru_cache(maxsize=8)
def _cached_vectors(path_text: str, mtime_ns: int, byte_size: int) -> dict[str, list[np.ndarray]]:
    del mtime_ns, byte_size
    results: dict[str, list[np.ndarray]] = {}
    for item in load(Path(path_text))["examples"]:
        person = str(item.get("person", "")).strip()
        vector = decode_embedding(item)
        if person and vector is not None:
            results.setdefault(person, []).append(vector)
    return results


def vectors_by_person(path: Path) -> dict[str, list[np.ndarray]]:
    path = path.expanduser()
    try:
        stat = path.stat()
        signature = (int(stat.st_mtime_ns), int(stat.st_size))
    except OSError:
        signature = (0, 0)
    return _cached_vectors(str(path.resolve(strict=False)), *signature)


class ReferenceRejections:
    """Exclude exact rejected face evidence from positive reference training."""

    def __init__(self, path: Path):
        self.vectors: dict[str, list[np.ndarray]] = {}
        self.digests: dict[str, list[str]] = {}
        for item in load(path)["examples"]:
            name = str(item.get("person", "")).strip().casefold()
            value = decode_embedding(item)
            if not name or value is None or not np.isfinite(value).all():
                continue
            self.vectors.setdefault(name, []).append(value)
            self.digests.setdefault(name, []).append(hashlib.sha256(value.tobytes()).hexdigest())

    def signature(self, person: str) -> str:
        values = self.digests.get(person.casefold(), [])
        return hashlib.sha256("\n".join(sorted(set(values))).encode()).hexdigest() if values else ""

    def rejects(self, person: str, embedding: np.ndarray) -> bool:
        query = np.asarray(embedding, dtype=np.float32).reshape(-1)
        query = query / max(float(np.linalg.norm(query)), 1e-9)
        return any(value.shape == query.shape and np.allclose(value, query, atol=1e-6, rtol=0)
                   for value in self.vectors.get(person.casefold(), []))

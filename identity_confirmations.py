#!/usr/bin/env python3
"""Persistent, user-confirmed identity examples.

These records are deliberately separate from automatic labels.  They let the
identity builder retain a small number of difficult but trusted appearances
without widening global matching thresholds.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import content_identity


SCHEMA_VERSION = 2


def default_path(cache_dir: Path) -> Path:
    return cache_dir / "confirmed_identity_examples.json"


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
    normalized = {
        "version": SCHEMA_VERSION,
        "examples": payload.get("examples", []),
    }
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(normalized, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def record(
    path: Path,
    *,
    person: str,
    organized_path: Path,
    content_sha256: str,
    original_name: str,
    face=None,
    detector_version: str = "",
) -> None:
    payload = load(path)
    canonical_path = str(organized_path.expanduser().resolve())
    examples = [
        item for item in payload["examples"]
        if not (
            str(item.get("organized_path", "")) == canonical_path
            or (
                str(item.get("person", "")).casefold() == person.casefold()
                and str(item.get("content_sha256", "")) == content_sha256
            )
        )
    ]
    examples.append({
        "person": person,
        "organized_path": canonical_path,
        "content_sha256": content_sha256,
        "original_name": original_name,
        "confirmed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "face_id": content_identity.face_identity(face) if face is not None else "",
        "face_index": int(face.face_index) if face is not None else None,
        "detector_version": detector_version,
        "byte_size": organized_path.stat().st_size,
    })
    payload["examples"] = sorted(
        examples,
        key=lambda item: (
            str(item.get("person", "")).casefold(),
            str(item.get("organized_path", "")).casefold(),
        ),
    )
    save(path, payload)


def resolve_record(item: dict, candidates=()) -> Path | None:
    digest = str(item.get("content_sha256", ""))
    if len(digest) != 64:
        return None
    source = Path(str(item.get("organized_path", ""))).expanduser()

    def matches(candidate):
        try:
            candidate = Path(candidate).resolve()
            if item.get("byte_size") is not None and candidate.stat().st_size != int(item["byte_size"]):
                return False
            return content_identity.content_sha256(candidate) == digest
        except (OSError, ValueError):
            return False

    if matches(source):
        return source.resolve()
    person = str(item.get("person", "")).casefold()
    for candidate in candidates:
        candidate = Path(candidate)
        parts = candidate.parts
        if not any(part.casefold() == person and i + 1 < len(parts) and parts[i + 1] == "photos"
                   for i, part in enumerate(parts)):
            continue
        if matches(candidate):
            return candidate.resolve()
    return None


def selected_faces(item: dict, faces: list) -> list:
    """Legacy confirmations are usable only for unambiguous single-face files."""
    faces = [face for face in faces if not getattr(face, "content_sha256", "")
             or face.content_sha256 == item.get("content_sha256")]
    face_id = str(item.get("face_id", ""))
    if face_id:
        return [face for face in faces if content_identity.face_identity(face) == face_id]
    return list(faces) if len(faces) == 1 else []


def examples_for_person(path: Path, person: str, people_root: Path) -> dict[Path, dict]:
    root = people_root.expanduser().resolve()
    results: dict[Path, dict] = {}
    candidates = None
    for item in load(path)["examples"]:
        if str(item.get("person", "")).casefold() != person.casefold():
            continue
        resolved = resolve_record(item)
        if resolved is None:
            if candidates is None:
                person_dir = root / person / "photos"
                candidates = list(person_dir.rglob("*")) if person_dir.is_dir() else []
            resolved = resolve_record(item, candidates)
        if resolved is None:
            continue
        try:
            relative = resolved.relative_to(root)
        except (OSError, ValueError):
            continue
        if relative.parts and relative.parts[0].casefold() == person.casefold() and resolved.is_file():
            results[resolved] = item
    return results


def paths_for_person(path: Path, person: str, people_root: Path) -> list[Path]:
    return sorted(examples_for_person(path, person, people_root), key=lambda value: str(value).casefold())


def signature_for_person(path: Path, person: str, people_root: Path) -> str:
    digest = hashlib.sha256()
    for candidate, example in sorted(examples_for_person(path, person, people_root).items()):
        try:
            stat = candidate.stat()
        except OSError:
            continue
        digest.update(str(candidate).encode("utf-8", errors="surrogateescape"))
        digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
        digest.update(json.dumps({key: example.get(key) for key in
            ("content_sha256", "face_id", "detector_version")}, sort_keys=True).encode("utf-8"))
    return digest.hexdigest()

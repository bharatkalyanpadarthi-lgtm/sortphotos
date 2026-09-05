#!/usr/bin/env python3
"""Enroll explicit user identity confirmations into a protected evaluation set."""

from __future__ import annotations

import csv
import os
import sqlite3
import tempfile
from pathlib import Path

import pipeline_paths
import evaluation_dataset
import identity_confirmations


DEFAULT_PATH = (
    pipeline_paths.SOURCE_REVIEW
    / "identity_evaluation"
    / "confirmed_unknown_identity_set.csv"
)
FIELDS = evaluation_dataset.FIELDS


def _read(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    except OSError:
        return []


def _write(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def enroll(
    *,
    source: Path,
    person: str,
    content_sha256: str,
    pose_label: str = "unknown",
    quality: float = 1.0,
    path: Path = DEFAULT_PATH,
) -> None:
    case_types = {"known"}
    if "profile" in str(pose_label):
        case_types.add("side_profile")
    if float(quality) < 0.45:
        case_types.add("blurry_small")
    canonical = str(source.expanduser().resolve(strict=False))
    rows = [
        row for row in _read(path)
        if not (
            str(row.get("content_sha256", "")) == content_sha256
            and str(row.get("expected_person", "")).casefold() == person.casefold()
        )
    ]
    rows.append({
        "source": canonical,
        "expected_person": person,
        "case_types": "|".join(sorted(case_types)),
        "expected_face": "true",
        "expected_nudity": "unknown",
        "verified": "true",
        "notes": "Explicitly confirmed in the unknown identity review.",
        "content_sha256": content_sha256,
        "group_id": content_sha256,
        "expected_people": person,
        "expected_face_count": "1",
    })
    rows.sort(key=lambda row: (
        str(row.get("expected_person", "")).casefold(),
        str(row.get("source", "")).casefold(),
    ))
    _write(path, rows)


def backfill_confirmations(
    confirmation_payload: dict,
    cached_faces: list,
    *,
    path: Path = DEFAULT_PATH,
) -> int:
    """Enroll older explicit confirmations without re-detecting their files."""
    faces_by_path: dict[str, list] = {}
    for face in cached_faces:
        canonical = str(Path(face.src_str).expanduser().resolve(strict=False))
        faces_by_path.setdefault(canonical, []).append(face)
    enrolled = 0
    for item in confirmation_payload.get("examples", []):
        person = str(item.get("person", "")).strip()
        digest = str(item.get("content_sha256", "")).strip()
        source = identity_confirmations.resolve_record(item, faces_by_path)
        if not person or not digest or source is None:
            continue
        candidates = faces_by_path.get(str(source.resolve(strict=False)), [])
        selected = identity_confirmations.selected_faces(item, candidates)
        if len(candidates) != 1 or len(selected) != 1:
            continue
        face = selected[0]
        enroll(
            source=source,
            person=person,
            content_sha256=digest,
            pose_label=str(getattr(face, "pose_label", "unknown")) if face else "unknown",
            quality=float(getattr(face, "quality", 1.0)) if face else 1.0,
            path=path,
        )
        enrolled += 1
    return enrolled


def relink_missing_sources(
    *,
    analysis_index_path: Path,
    people_root: Path,
    path: Path = DEFAULT_PATH,
) -> dict[str, int]:
    """Repair moved benchmark paths using their immutable content hashes.

    A match is accepted only when the indexed file still exists underneath the
    expected person's folder. The benchmark remains manually verified; this
    function merely follows normal library renames and nudity-folder moves.
    """
    rows = _read(path)
    stats = {"rows": len(rows), "missing": 0, "relinked": 0, "unresolved": 0}
    if not rows or not analysis_index_path.expanduser().is_file():
        return stats

    root = people_root.expanduser().resolve(strict=False)
    connection = sqlite3.connect(analysis_index_path.expanduser())
    changed = False
    try:
        for row in rows:
            source = Path(str(row.get("source", ""))).expanduser()
            if source.is_file():
                continue
            stats["missing"] += 1
            digest = str(row.get("content_sha256", "")).strip().casefold()
            expected = str(row.get("expected_person", "")).strip().casefold()
            if not digest or not expected:
                stats["unresolved"] += 1
                continue
            candidates: list[Path] = []
            try:
                indexed_paths = connection.execute(
                    "SELECT path FROM assets WHERE sha256=? ORDER BY path",
                    (digest,),
                ).fetchall()
            except sqlite3.Error:
                stats["unresolved"] += 1
                continue
            for (candidate_text,) in indexed_paths:
                candidate = Path(str(candidate_text)).expanduser()
                if not candidate.is_file():
                    continue
                try:
                    relative = candidate.resolve().relative_to(root)
                except (OSError, ValueError):
                    continue
                if relative.parts and relative.parts[0].casefold() == expected:
                    candidates.append(candidate.resolve())
            if not candidates:
                stats["unresolved"] += 1
                continue
            replacement = min(candidates, key=lambda value: (len(value.parts), str(value).casefold()))
            row["source"] = str(replacement)
            stats["relinked"] += 1
            changed = True
    finally:
        connection.close()
    if changed:
        rows.sort(key=lambda row: (
            str(row.get("expected_person", "")).casefold(),
            str(row.get("source", "")).casefold(),
        ))
        _write(path, rows)
    return stats

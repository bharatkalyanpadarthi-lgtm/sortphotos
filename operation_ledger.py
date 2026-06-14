#!/usr/bin/env python3
"""Append-only ledger for file moves made by the photo pipeline.

The source guards tell us that something went wrong. This ledger tells us what
actually moved, where it went, and which run moved it.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any

DEFAULT_SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
DEFAULT_PEOPLE = DEFAULT_SORTED / "photos_by_person"
LEDGER_DIR_NAME = "operation_ledgers"
RUN_ID_ENV = "PHOTO_PIPELINE_RUN_ID"

_SESSION_RUN_ID = f"manual_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S%z")


def slugify(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return value.strip("_") or "photo_pipeline_run"


def current_run_id(run_id: str | None = None) -> str:
    return slugify(run_id or os.environ.get(RUN_ID_ENV) or _SESSION_RUN_ID)


def ledger_dir(sorted_root: Path = DEFAULT_SORTED) -> Path:
    return sorted_root / "_source_review" / LEDGER_DIR_NAME


def ledger_path(sorted_root: Path = DEFAULT_SORTED, run_id: str | None = None) -> Path:
    return ledger_dir(sorted_root) / f"{current_run_id(run_id)}_moves.jsonl"


def latest_run_path(sorted_root: Path = DEFAULT_SORTED) -> Path:
    return ledger_dir(sorted_root) / "latest_run_id.txt"


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def maybe_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except Exception:
        return ""


def infer_person(path: Path, people_dir: Path) -> str:
    rel = maybe_relative(path, people_dir)
    if not rel:
        return ""
    parts = Path(rel).parts
    return parts[0] if parts else ""


def metadata(path: Path, *, hash_file: bool = True, sha256: str | None = None) -> dict[str, Any]:
    st = path.stat()
    payload: dict[str, Any] = {
        "path": str(path),
        "exists": True,
        "size": int(st.st_size),
        "mtime_ns": int(st.st_mtime_ns),
        "is_file": path.is_file(),
    }
    if path.is_file() and hash_file:
        payload["sha256"] = sha256 or sha256_file(path)
    elif sha256:
        payload["sha256"] = sha256
    return payload


def build_event(*,
                operation: str,
                reason: str,
                status: str,
                source: Path,
                dest: Path,
                sorted_root: Path = DEFAULT_SORTED,
                run_id: str | None = None,
                source_meta: dict[str, Any] | None = None,
                dest_meta: dict[str, Any] | None = None,
                error: str = "",
                extra: dict[str, Any] | None = None) -> dict[str, Any]:
    people_dir = sorted_root / "photos_by_person"
    return {
        "version": 1,
        "run_id": current_run_id(run_id),
        "created_at": timestamp(),
        "operation": operation,
        "reason": reason,
        "status": status,
        "source_path": str(source),
        "dest_path": str(dest),
        "source_relative_to_sorted": maybe_relative(source, sorted_root),
        "dest_relative_to_sorted": maybe_relative(dest, sorted_root),
        "source_relative_to_people": maybe_relative(source, people_dir),
        "dest_relative_to_people": maybe_relative(dest, people_dir),
        "person": infer_person(source, people_dir) or infer_person(dest, people_dir),
        "source": source_meta or {},
        "dest": dest_meta or {},
        "error": error,
        "extra": extra or {},
    }


def append_event(event: dict[str, Any], *,
                 sorted_root: Path = DEFAULT_SORTED,
                 run_id: str | None = None) -> Path:
    rid = current_run_id(run_id or str(event.get("run_id") or ""))
    path = ledger_path(sorted_root, rid)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        json.dump(event, f, sort_keys=True)
        f.write("\n")
    latest_run_path(sorted_root).write_text(rid + "\n", encoding="utf-8")
    return path


def record_event(*,
                 operation: str,
                 reason: str,
                 status: str,
                 source: Path,
                 dest: Path,
                 sorted_root: Path = DEFAULT_SORTED,
                 run_id: str | None = None,
                 source_meta: dict[str, Any] | None = None,
                 dest_meta: dict[str, Any] | None = None,
                 error: str = "",
                 extra: dict[str, Any] | None = None) -> Path:
    event = build_event(
        operation=operation,
        reason=reason,
        status=status,
        source=source,
        dest=dest,
        sorted_root=sorted_root,
        run_id=run_id,
        source_meta=source_meta,
        dest_meta=dest_meta,
        error=error,
        extra=extra,
    )
    return append_event(event, sorted_root=sorted_root, run_id=event["run_id"])


def move_path(src: Path,
              dest: Path,
              *,
              sorted_root: Path = DEFAULT_SORTED,
              operation: str,
              reason: str,
              run_id: str | None = None,
              extra: dict[str, Any] | None = None,
              hash_file: bool = True) -> Path:
    """Move a path and record planned/completed/failed events.

    The source hash is captured before the move when the source is a file.
    """
    src = Path(src)
    dest = Path(dest)
    source_meta = metadata(src, hash_file=hash_file)
    record_event(
        operation=operation,
        reason=reason,
        status="planned",
        source=src,
        dest=dest,
        sorted_root=sorted_root,
        run_id=run_id,
        source_meta=source_meta,
        extra=extra,
    )
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        dest_meta = metadata(
            dest,
            hash_file=False,
            sha256=str(source_meta.get("sha256") or ""),
        )
        record_event(
            operation=operation,
            reason=reason,
            status="moved",
            source=src,
            dest=dest,
            sorted_root=sorted_root,
            run_id=run_id,
            source_meta=source_meta,
            dest_meta=dest_meta,
            extra=extra,
        )
        return dest
    except Exception as exc:
        record_event(
            operation=operation,
            reason=reason,
            status="failed",
            source=src,
            dest=dest,
            sorted_root=sorted_root,
            run_id=run_id,
            source_meta=source_meta,
            error=str(exc),
            extra=extra,
        )
        raise


def iter_events(sorted_root: Path = DEFAULT_SORTED) -> list[dict[str, Any]]:
    root = ledger_dir(sorted_root)
    events: list[dict[str, Any]] = []
    if not root.exists():
        return events
    for path in sorted(root.glob("*_moves.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    event = json.loads(line)
                    event["_ledger_path"] = str(path)
                    events.append(event)
                except json.JSONDecodeError:
                    continue
    return events

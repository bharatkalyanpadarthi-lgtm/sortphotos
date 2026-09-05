#!/usr/bin/env python3
"""Rebase persistent photo-pipeline cache paths after storage migration.

The command is a dry run unless ``--apply`` is supplied.  Every changed cache
file receives a timestamped backup before an atomic replacement.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import pipeline_paths
import sort_photos


CACHE_DIR = Path.home() / ".face_sort_cache"
OLD_SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
OLD_REFERENCES = Path.home() / "Pictures" / "Face References"
OLD_TO_PROCESS = Path.home() / "Pictures" / "To Process"


def replacements() -> tuple[tuple[str, str], ...]:
    return (
        (str(OLD_SORTED), str(pipeline_paths.SORTED_ROOT)),
        (str(OLD_REFERENCES), str(pipeline_paths.FACE_REFERENCES)),
        (str(OLD_TO_PROCESS), str(pipeline_paths.TO_PROCESS)),
    )


def rebase_string(value: str) -> str:
    for old, new in replacements():
        if value == old:
            return new
        prefix = old + os.sep
        if value.startswith(prefix):
            return new + value[len(old):]
    return value


def rebase_json(value: Any) -> tuple[Any, int]:
    if isinstance(value, str):
        updated = rebase_string(value)
        return updated, int(updated != value)
    if isinstance(value, list):
        changed = 0
        result = []
        for item in value:
            updated, count = rebase_json(item)
            result.append(updated)
            changed += count
        return result, changed
    if isinstance(value, dict):
        changed = 0
        result = {}
        for key, item in value.items():
            updated_key = rebase_string(key) if isinstance(key, str) else key
            updated_item, count = rebase_json(item)
            if updated_key in result and result[updated_key] != updated_item:
                raise ValueError(f"path rebasing would overwrite conflicting JSON key: {updated_key}")
            result[updated_key] = updated_item
            changed += int(updated_key != key) + count
        return result, changed
    return value, 0


def backup_and_replace(path: Path, payload: bytes, stamp: str) -> None:
    backup = path.with_name(f"{path.name}.bak.storage_migration_{stamp}")
    shutil.copy2(path, backup)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as handle:
        temporary = Path(handle.name)
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    print(f"  backup: {backup}")


def migrate_cache_pickle(path: Path, apply: bool, stamp: str) -> int:
    sort_photos.install_pickle_class_aliases()
    with path.open("rb") as handle:
        cache = pickle.load(handle)
    changed = 0
    signatures: dict[str, tuple[float, int]] = {}
    for source, signature in cache.file_signatures.items():
        updated = rebase_string(source)
        if updated in signatures and signatures[updated] != signature:
            raise ValueError(f"path rebasing would overwrite conflicting cache entry: {updated}")
        signatures[updated] = signature
        changed += int(updated != source)
    cache.file_signatures = signatures
    for face in cache.faces:
        updated = rebase_string(face.src_str)
        changed += int(updated != face.src_str)
        face.src_str = updated
    print(f"{path.name}: {changed:,} path value(s) to update")
    if apply and changed:
        backup_and_replace(path, pickle.dumps(cache, protocol=pickle.HIGHEST_PROTOCOL), stamp)
    return changed


def migrate_labeling_pickle(path: Path, apply: bool, stamp: str) -> int:
    sort_photos.install_pickle_class_aliases()
    with path.open("rb") as handle:
        state = pickle.load(handle)
    changed = 0
    for attribute in ("input_dir", "output_dir"):
        value = getattr(state, attribute, "")
        updated = rebase_string(value)
        changed += int(updated != value)
        setattr(state, attribute, updated)
    for face in state.faces:
        updated = rebase_string(face.src_str)
        changed += int(updated != face.src_str)
        face.src_str = updated
    print(f"{path.name}: {changed:,} path value(s) to update")
    if apply and changed:
        backup_and_replace(path, pickle.dumps(state, protocol=pickle.HIGHEST_PROTOCOL), stamp)
    return changed


def migrate_json_file(path: Path, apply: bool, stamp: str) -> int:
    data = json.loads(path.read_text(encoding="utf-8"))
    updated, changed = rebase_json(data)
    print(f"{path.name}: {changed:,} path value(s) to update")
    if apply and changed:
        payload = (json.dumps(updated, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
        backup_and_replace(path, payload, stamp)
    return changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true", help="Write rebased caches after making backups.")
    args = parser.parse_args()
    stamp = time.strftime("%Y%m%d_%H%M%S")
    total = 0
    cache_path = CACHE_DIR / "cache.pkl"
    if cache_path.exists():
        total += migrate_cache_pickle(cache_path, args.apply, stamp)
    labeling_path = CACHE_DIR / "labeling_state.pkl"
    if labeling_path.exists():
        total += migrate_labeling_pickle(labeling_path, args.apply, stamp)
    for path in sorted(CACHE_DIR.glob("*.json")):
        total += migrate_json_file(path, args.apply, stamp)
    print(f"Total path values: {total:,}")
    print("Applied." if args.apply else "Dry run only. Re-run with --apply after review.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Verified benchmark copies, independent of the movable intake queue.

CSV annotations and their baseline digest stay unchanged. Copies are never
training references, and neither this module nor its repair tool moves sources.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import shutil
import sqlite3
import tempfile
from contextlib import contextmanager
from pathlib import Path

import content_identity


def asset_path(dataset: Path, source: Path, digest: str) -> Path | None:
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        return None
    return dataset.with_suffix(".assets") / digest[:2] / (digest + source.suffix.lower())


def pin(dataset: Path, source: Path, digest: str, *, candidate: Path | None = None) -> Path:
    target = asset_path(dataset, source, digest)
    if target is None:
        raise ValueError("A recorded SHA-256 is required to protect a benchmark image")
    if target.exists():
        if content_identity.content_sha256(target) != digest:
            raise ValueError(f"Protected benchmark copy is corrupt: {target}")
        return target
    candidate = candidate or source
    version = content_identity.file_version(candidate)
    if content_identity.content_sha256(candidate) != digest:
        raise ValueError(f"Benchmark candidate does not match verified content: {candidate}")
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".incoming-", dir=target.parent)
    os.close(descriptor)
    temporary = Path(temporary)
    try:
        shutil.copy2(candidate, temporary)
        if (content_identity.file_version(candidate) != version
                or content_identity.content_sha256(temporary) != digest):
            raise ValueError(f"Benchmark image changed during copy: {candidate}")
        with temporary.open("rb") as handle:
            os.fsync(handle.fileno())
        # Publish without replacing an existing, potentially corrupt copy.
        try:
            os.link(temporary, target)
        except FileExistsError:
            if content_identity.content_sha256(target) != digest:
                raise ValueError(f"Protected benchmark copy is corrupt: {target}")
        with _directory_handle(target.parent) as handle:
            os.fsync(handle)
        return target
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def _directory_handle(path):
    descriptor = os.open(path, os.O_RDONLY)
    try:
        yield descriptor
    finally:
        os.close(descriptor)


def pin_dataset(dataset: Path, *, replacements=None) -> dict:
    """Pin verified intake examples only; never guess a new label or hash."""
    stats = {"protected": 0, "unresolved": []}
    if not dataset.is_file():
        return stats
    with dataset.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        source = Path(row.get("source") or "").expanduser()
        if ("unassigned_intake" not in source.parts
                or str(row.get("verified", "")).casefold() not in {"true", "1", "yes", "y", "verified"}):
            continue
        digest = row.get("content_sha256") or ""
        try:
            # A replaced source must be reverified, even if its old copy survives.
            if source.is_file() and content_identity.content_sha256(source) != digest:
                raise ValueError(f"Benchmark content changed since verification: {source}")
            pin(dataset, source, digest, candidate=(replacements or {}).get(digest))
            stats["protected"] += 1
        except (OSError, ValueError) as error:
            stats["unresolved"].append(str(error))
    return stats


def recover_candidates(dataset: Path, index_path: Path, roots: list[Path]) -> dict[str, Path]:
    """Use indexed hints first, then a size-filtered scan; verify every hit."""
    with dataset.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    wanted = {row.get("content_sha256", ""): Path(row["source"]) for row in rows
              if "unassigned_intake" in Path(row["source"]).parts
              and not Path(row["source"]).is_file()
              and re.fullmatch(r"[0-9a-f]{64}", row.get("content_sha256", ""))}
    found, sizes = {}, {}
    with sqlite3.connect(f"file:{index_path}?mode=ro", uri=True) as connection:
        for digest in wanted:
            sizes[digest] = {size for (size,) in connection.execute(
                "SELECT byte_size FROM assets WHERE sha256=?", (digest,))}
            for table in ("assets", "content_versions"):
                for (text,) in connection.execute(f"SELECT path FROM {table} WHERE sha256=?", (digest,)):
                    candidate = Path(text)
                    try:
                        if content_identity.content_sha256(candidate) == digest:
                            found[digest] = candidate
                            break
                    except OSError:
                        pass
                if digest in found:
                    break
    allowed_sizes = set().union(*(sizes[digest] for digest in wanted if digest not in found))
    for root in roots:
        for directory, subdirs, files in os.walk(root):
            subdirs[:] = [name for name in subdirs if not name.startswith(".") and not name.endswith(".assets")]
            for name in files:
                if len(found) == len(wanted):
                    return found
                candidate = Path(directory) / name
                try:
                    if allowed_sizes and candidate.stat().st_size not in allowed_sizes:
                        continue
                    digest = content_identity.content_sha256(candidate)
                    if digest in wanted and digest not in found:
                        found[digest] = candidate
                except OSError:
                    continue
    return found


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--index", type=Path)
    parser.add_argument("--recover-from", type=Path, action="append", default=[])
    args = parser.parse_args()
    replacements = (recover_candidates(args.dataset, args.index, args.recover_from)
                    if args.index else {})
    import json
    stats = pin_dataset(args.dataset, replacements=replacements)
    print(json.dumps(stats, indent=2))
    return int(bool(stats["unresolved"]))


if __name__ == "__main__":
    raise SystemExit(main())

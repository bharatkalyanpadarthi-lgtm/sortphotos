#!/usr/bin/env python3
"""Remove generated smart album folders after verifying originals are safe.

Smart albums are generated hardlink/copy views. This script verifies that each
photo-like image in _smart_albums, _smart_albums_v2, or
_smart_albums_simple_preview is already present in the same person's canonical
photos/ tree. If a smart-folder photo is unique, it is recovered into photos/
or photos/nude/ before generated smart folders are removed.

Default mode is a dry-run. Use --apply to recover unique images and remove the
generated smart album folders.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import operation_ledger
import source_manifest

DEFAULT_SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
SMART_DIR_NAMES = {"_smart_albums", "_smart_albums_v2", "_smart_albums_simple_preview"}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
SMART_STATE_FILES = [
    Path.home() / ".face_sort_cache" / "smart_album_person_state.json",
    Path.home() / ".face_sort_cache" / "smart_album_framing_cache.json",
]


def timestamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = "".join(dest.suffixes)
    for i in range(2, 100000):
        candidate = dest.with_name(f"{stem}__from_smart_{i}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"could not find unique destination for {dest}")


def person_dirs(people_dir: Path) -> list[Path]:
    if not people_dir.exists():
        return []
    return sorted(
        [p for p in people_dir.iterdir() if p.is_dir() and not p.name.startswith("_")],
        key=lambda p: p.name.lower(),
    )


def smart_dirs(person_dir: Path) -> list[Path]:
    found = [
        path
        for path in person_dir.rglob("*")
        if path.is_dir() and path.name in SMART_DIR_NAMES
    ]
    return sorted(found, key=lambda p: (len(p.parts), str(p).lower()), reverse=True)


def canonical_images(person_dir: Path) -> list[Path]:
    photos_dir = person_dir / "photos"
    if not photos_dir.exists():
        return []
    return sorted(
        [p for p in photos_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: str(p).lower(),
    )


def smart_image_candidates(smart_dir: Path) -> tuple[list[Path], int]:
    candidates: list[Path] = []
    generated_or_nonphoto = 0
    for path in sorted(smart_dir.rglob("*"), key=lambda p: str(p).lower()):
        if not path.is_file():
            continue
        rel_parts = path.relative_to(smart_dir).parts
        if "_data" in rel_parts or path.name == "_contact_sheet.jpg":
            generated_or_nonphoto += 1
            continue
        if path.suffix.lower() not in IMAGE_EXTS:
            generated_or_nonphoto += 1
            continue
        candidates.append(path)
    return candidates, generated_or_nonphoto


def nudity_status_for_path(path: Path) -> str:
    parts = [p.casefold() for p in path.parts]
    name = path.name.casefold()
    for i, part in enumerate(parts):
        next_part = parts[i + 1] if i + 1 < len(parts) else ""
        if part in {"photos", "all", "review"} and next_part in {"nude", "nudity_possible"}:
            return "possible"
        if part in {"photos_nude", "_possible_nudity", "nudity_possible", "nude"}:
            return "possible"
        if part in {"_uncertain_nudity", "uncertain_nudity"}:
            return "uncertain"
    if "nudity_possible" in name or "_nude" in name or "_nudity_" in name:
        return "possible"
    return "safe"


def destination_for_smart_image(person_dir: Path, smart_path: Path) -> Path:
    photos_dir = person_dir / "photos"
    if nudity_status_for_path(smart_path) == "possible":
        photos_dir = photos_dir / "nude"
    return unique_dest(photos_dir / smart_path.name)


def build_canonical_index(paths: list[Path]) -> tuple[set[tuple[int, int]], dict[tuple[int, str], list[Path]]]:
    inodes: set[tuple[int, int]] = set()
    by_hash: dict[tuple[int, str], list[Path]] = defaultdict(list)
    for path in paths:
        try:
            stat = path.stat()
            inodes.add((stat.st_dev, stat.st_ino))
            by_hash[(int(stat.st_size), sha256_file(path))].append(path)
        except OSError:
            continue
    return inodes, by_hash


def add_to_index(path: Path,
                 inodes: set[tuple[int, int]],
                 by_hash: dict[tuple[int, str], list[Path]],
                 digest: str | None = None) -> None:
    stat = path.stat()
    inodes.add((stat.st_dev, stat.st_ino))
    by_hash[(int(stat.st_size), digest or sha256_file(path))].append(path)


def count_canonical_originals(people_dir: Path) -> int:
    return sum(len(canonical_images(person_dir)) for person_dir in person_dirs(people_dir))


def remove_state_files(apply: bool, run_id: str, sorted_root: Path) -> list[Path]:
    removed: list[Path] = []
    for path in SMART_STATE_FILES:
        if not path.exists():
            continue
        removed.append(path)
        if apply:
            operation_ledger.record_event(
                operation="scrap_smart_albums.remove_state_file",
                reason="remove generated smart album cache state",
                status="planned",
                source=path,
                dest=path,
                sorted_root=sorted_root,
                run_id=run_id,
                extra={"kind": "smart_album_state"},
            )
            path.unlink()
            operation_ledger.record_event(
                operation="scrap_smart_albums.remove_state_file",
                reason="remove generated smart album cache state",
                status="removed",
                source=path,
                dest=path,
                sorted_root=sorted_root,
                run_id=run_id,
                extra={"kind": "smart_album_state"},
            )
    return removed


def write_report(report_path: Path, rows: list[dict[str, Any]]) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "person",
        "smart_dir",
        "smart_path",
        "status",
        "canonical_match",
        "recovered_to",
        "size_bytes",
        "sha256",
        "note",
    ]
    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(summary_path: Path, payload: dict[str, Any]) -> None:
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sorted-root", type=Path, default=DEFAULT_SORTED)
    parser.add_argument("--apply", action="store_true",
                        help="Recover unique smart-folder photos and remove generated smart folders.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sorted_root = args.sorted_root.expanduser().resolve()
    people_dir = sorted_root / "photos_by_person"
    run_id = f"scrap_smart_albums_{timestamp()}"
    report_dir = sorted_root / "_source_review" / "smart_album_scrap"
    report_path = report_dir / f"{run_id}.csv"
    summary_path = report_dir / f"{run_id}.json"

    validation = source_manifest.validate_current(
        label=f"{run_id}_before",
        people_dir=people_dir,
        manifest_path=sorted_root / "_source_review" / "source_manifest" / "last_known_good_originals.json",
        report_dir=sorted_root / "_source_review" / "source_manifest" / "reports",
    )
    if not validation.ok:
        source_manifest.print_validation(validation)
        print("ERROR: source manifest is not clean; refusing to scrap generated smart folders.")
        return source_manifest.SOURCE_GUARD_EXIT

    before_originals = count_canonical_originals(people_dir)
    rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    smart_dirs_to_remove: list[Path] = []

    for person_dir in person_dirs(people_dir):
        dirs = smart_dirs(person_dir)
        if not dirs:
            continue
        canonical = canonical_images(person_dir)
        canonical_inodes, canonical_hashes = build_canonical_index(canonical)
        for smart_dir in dirs:
            smart_dirs_to_remove.append(smart_dir)
            candidates, ignored = smart_image_candidates(smart_dir)
            counters["generated_or_nonphoto_files"] += ignored
            for smart_path in candidates:
                counters["smart_photo_candidates"] += 1
                row = {
                    "person": person_dir.name,
                    "smart_dir": smart_dir.name,
                    "smart_path": str(smart_path),
                    "status": "",
                    "canonical_match": "",
                    "recovered_to": "",
                    "size_bytes": "",
                    "sha256": "",
                    "note": "",
                }
                try:
                    stat = smart_path.stat()
                    row["size_bytes"] = int(stat.st_size)
                    inode = (stat.st_dev, stat.st_ino)
                    if inode in canonical_inodes:
                        row["status"] = "covered_by_inode"
                        row["canonical_match"] = "same inode as canonical photos file"
                        counters["covered_by_inode"] += 1
                        rows.append(row)
                        continue
                    digest = sha256_file(smart_path)
                    row["sha256"] = digest
                    matches = canonical_hashes.get((int(stat.st_size), digest), [])
                    if matches:
                        row["status"] = "covered_by_hash"
                        row["canonical_match"] = str(matches[0])
                        counters["covered_by_hash"] += 1
                        rows.append(row)
                        continue
                    dest = destination_for_smart_image(person_dir, smart_path)
                    row["recovered_to"] = str(dest)
                    if args.apply:
                        moved = operation_ledger.move_path(
                            smart_path,
                            dest,
                            sorted_root=sorted_root,
                            operation="scrap_smart_albums.recover_unique_photo",
                            reason="recover unique smart-folder photo before deleting generated smart albums",
                            run_id=run_id,
                            extra={
                                "person": person_dir.name,
                                "smart_dir": smart_dir.name,
                                "sha256": digest,
                            },
                        )
                        add_to_index(moved, canonical_inodes, canonical_hashes, digest)
                        row["status"] = "recovered"
                        counters["recovered"] += 1
                    else:
                        row["status"] = "would_recover"
                        counters["would_recover"] += 1
                    rows.append(row)
                except Exception as exc:  # noqa: BLE001
                    row["status"] = "error"
                    row["note"] = str(exc)
                    counters["errors"] += 1
                    rows.append(row)

    write_report(report_path, rows)

    removed_dirs = 0
    if args.apply and counters["errors"] == 0:
        for smart_dir in smart_dirs_to_remove:
            if not smart_dir.exists():
                continue
            operation_ledger.record_event(
                operation="scrap_smart_albums.remove_generated_dir",
                reason="remove generated smart album folder after original coverage audit",
                status="planned",
                source=smart_dir,
                dest=smart_dir,
                sorted_root=sorted_root,
                run_id=run_id,
                extra={"smart_dir_name": smart_dir.name},
            )
            shutil.rmtree(smart_dir)
            operation_ledger.record_event(
                operation="scrap_smart_albums.remove_generated_dir",
                reason="remove generated smart album folder after original coverage audit",
                status="removed",
                source=smart_dir,
                dest=smart_dir,
                sorted_root=sorted_root,
                run_id=run_id,
                extra={"smart_dir_name": smart_dir.name},
            )
            removed_dirs += 1

    state_files = remove_state_files(args.apply and counters["errors"] == 0, run_id, sorted_root)
    after_originals = count_canonical_originals(people_dir)

    summary = {
        "run_id": run_id,
        "mode": "apply" if args.apply else "dry-run",
        "sorted_root": str(sorted_root),
        "before_originals": before_originals,
        "after_originals": after_originals,
        "smart_dirs_found": len(smart_dirs_to_remove),
        "smart_dirs_removed": removed_dirs,
        "smart_state_files": [str(p) for p in state_files],
        "report_csv": str(report_path),
        "counts": dict(counters),
    }
    write_summary(summary_path, summary)

    print("Smart Album Scrap")
    print("=" * 60)
    print(f"Mode:                        {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"People folder:               {people_dir}")
    print(f"Smart dirs found:            {len(smart_dirs_to_remove)}")
    print(f"Photo candidates checked:    {counters['smart_photo_candidates']}")
    print(f"Covered by inode:            {counters['covered_by_inode']}")
    print(f"Covered by hash:             {counters['covered_by_hash']}")
    print(f"Would recover:               {counters['would_recover']}")
    print(f"Recovered:                   {counters['recovered']}")
    print(f"Generated/non-photo ignored: {counters['generated_or_nonphoto_files']}")
    print(f"Errors:                      {counters['errors']}")
    print(f"Smart dirs removed:          {removed_dirs}")
    print(f"Originals before/after:      {before_originals} -> {after_originals}")
    print(f"Report CSV:                  {report_path}")
    print(f"Summary JSON:                {summary_path}")
    if not args.apply:
        print()
        print("DRY-RUN only. Re-run with --apply to recover unique photos and remove smart folders.")
    elif counters["errors"]:
        print()
        print("ERROR: smart folders were not removed because one or more files failed audit.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

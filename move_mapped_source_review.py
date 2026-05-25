#!/usr/bin/env python3
"""
move_mapped_source_review.py — Move source-review images already mapped to people.

Scans:
  ~/Pictures/sorted_all_pictures/photos_by_person
  ~/Pictures/sorted_all_pictures/_source_review

Any image in _source_review whose exact content hash exists in photos_by_person
is moved to:
  ~/Pictures/sorted_all_pictures/_source_review/ready_to_delete

Default is dry-run. Use --apply to move files.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
              ".tif", ".tiff", ".heic", ".heif"}
DEFAULT_SORTED = Path.home() / "Pictures" / "sorted_all_pictures"


def iter_images(root: Path, skip_dir_names: set[str] | None = None) -> list[Path]:
    skip = skip_dir_names or set()
    out: list[Path] = []
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in skip]
        base = Path(dirpath)
        for filename in filenames:
            p = base / filename
            if p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
    return out


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = "".join(dest.suffixes)
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__dup{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sorted-root", type=Path, default=DEFAULT_SORTED)
    parser.add_argument("--apply", action="store_true",
                        help="Move mapped images. Default is dry-run.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print sample moved paths.")
    args = parser.parse_args()

    sorted_root = args.sorted_root.expanduser().resolve()
    people_dir = sorted_root / "photos_by_person"
    source_review = sorted_root / "_source_review"
    ready_dir = source_review / "ready_to_delete"

    if not people_dir.exists():
        print(f"ERROR: photos_by_person not found: {people_dir}")
        return 1
    if not source_review.exists():
        print(f"ERROR: _source_review not found: {source_review}")
        return 1

    print(f"Person folders: {people_dir}")
    print(f"Source review:  {source_review}")
    print(f"Ready folder:   {ready_dir}")
    print()

    person_images = iter_images(people_dir, skip_dir_names={"all", "_smart_albums"})
    by_size: dict[int, list[Path]] = defaultdict(list)
    for p in person_images:
        try:
            by_size[p.stat().st_size].append(p)
        except OSError:
            pass

    source_images = iter_images(source_review, skip_dir_names={"ready_to_delete"})
    mapped: list[tuple[Path, Path]] = []
    unmapped: list[Path] = []
    hash_cache: dict[Path, str] = {}
    people_hashes_by_size: dict[int, set[str]] = {}

    def people_hashes(size: int) -> set[str]:
        if size not in people_hashes_by_size:
            vals: set[str] = set()
            for p in by_size.get(size, []):
                try:
                    vals.add(sha1_file(p))
                except OSError:
                    pass
            people_hashes_by_size[size] = vals
        return people_hashes_by_size[size]

    for src in source_images:
        try:
            size = src.stat().st_size
        except OSError:
            continue
        candidates = people_hashes(size)
        if not candidates:
            unmapped.append(src)
            continue
        try:
            h = hash_cache[src] = sha1_file(src)
        except OSError:
            unmapped.append(src)
            continue
        if h in candidates:
            rel = src.relative_to(source_review)
            mapped.append((src, unique_dest(ready_dir / rel)))
        else:
            unmapped.append(src)

    print(f"photos_by_person images:       {len(person_images)}")
    print(f"_source_review images checked: {len(source_images)}")
    print(f"Mapped and ready to move:      {len(mapped)}")
    print(f"Unmapped left in place:        {len(unmapped)}")
    print()

    if not args.quiet:
        for src, dest in mapped[:80]:
            print(f"move: {src}")
            print(f"  -> {dest}")
        if len(mapped) > 80:
            print(f"... and {len(mapped) - 80} more")
        if unmapped:
            print()
            print("Unmapped samples:")
            for src in unmapped[:40]:
                print(src)
        print()

    if not args.apply:
        print("DRY-RUN — no files moved. Re-run with --apply to commit.")
        return 0

    moved = 0
    for src, dest in mapped:
        if not src.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        moved += 1

    print(f"Moved {moved} mapped image(s) to: {ready_dir}")
    print("Review that folder, then delete it when satisfied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

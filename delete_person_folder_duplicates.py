#!/usr/bin/env python3
"""
Remove exact duplicate images inside each person-specific folder.

This keeps one copy of each exact image per person folder. Duplicate files are
moved to:
  ~/Pictures/sorted_all_pictures/_source_review/ready_to_delete/person_folder_duplicates

Default is dry-run. Use --apply to move duplicates out of photos_by_person.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp",
              ".tif", ".tiff", ".heic", ".heif"}
DEFAULT_SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
EXCLUDED_DIRS = {"_smart_albums", "_near_visual_review"}


def iter_images(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in EXCLUDED_DIRS]
        base = Path(dirpath)
        for filename in filenames:
            p = base / filename
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
    return out


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def same_inode(a: Path, b: Path) -> bool:
    try:
        sa = a.stat()
        sb = b.stat()
    except OSError:
        return False
    return sa.st_dev == sb.st_dev and sa.st_ino == sb.st_ino


def choose_keeper(paths: list[Path]) -> Path:
    def score(p: Path) -> tuple[int, int, int, str]:
        parts = set(p.parts)
        duplicate_penalty = 1 if "_duplicates" in parts else 0
        blurred_penalty = 1 if "_blurred" in parts else 0
        inode_bonus = 0
        try:
            inode_bonus = -p.stat().st_nlink
        except OSError:
            pass
        return (duplicate_penalty, blurred_penalty, inode_bonus, str(p).lower())

    return sorted(paths, key=score)[0]


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


def person_dirs(people_root: Path) -> list[Path]:
    return sorted(
        [p for p in people_root.iterdir() if p.is_dir() and not p.name.startswith("_")],
        key=lambda p: p.name.lower(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sorted-root", type=Path, default=DEFAULT_SORTED)
    parser.add_argument("--apply", action="store_true",
                        help="Move duplicate files out of person folders. Default is dry-run.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print sample duplicate paths.")
    args = parser.parse_args()

    sorted_root = args.sorted_root.expanduser().resolve()
    people_root = sorted_root / "photos_by_person"
    delete_root = sorted_root / "_source_review" / "ready_to_delete" / "person_folder_duplicates"

    if not people_root.exists():
        print(f"ERROR: photos_by_person not found: {people_root}")
        return 1

    actions: list[tuple[Path, Path, Path]] = []
    total_images = 0
    duplicate_groups = 0
    person_count = 0
    already_same_inode = 0
    bytes_to_move = 0

    for person_dir in person_dirs(people_root):
        person_count += 1
        images = iter_images(person_dir)
        total_images += len(images)

        by_size: dict[int, list[Path]] = defaultdict(list)
        for p in images:
            try:
                by_size[p.stat().st_size].append(p)
            except OSError:
                continue

        by_hash: dict[str, list[Path]] = defaultdict(list)
        for same_size in by_size.values():
            if len(same_size) < 2:
                continue
            for p in same_size:
                try:
                    by_hash[sha1_file(p)].append(p)
                except OSError:
                    continue

        for paths in by_hash.values():
            if len(paths) < 2:
                continue
            duplicate_groups += 1
            keeper = choose_keeper(paths)
            for p in paths:
                if p == keeper:
                    continue
                if same_inode(p, keeper):
                    already_same_inode += 1
                try:
                    bytes_to_move += p.stat().st_size
                except OSError:
                    pass
                rel = p.relative_to(people_root)
                actions.append((p, keeper, unique_dest(delete_root / rel)))

    print(f"Person folders scanned:       {person_count}")
    print(f"Image entries scanned:        {total_images}")
    print(f"Exact duplicate groups:       {duplicate_groups}")
    print(f"Duplicate entries to remove:  {len(actions)}")
    print(f"Already hardlinked entries:   {already_same_inode}")
    print(f"Bytes leaving person folders: {bytes_to_move / (1024**3):.2f} GB")
    print(f"Destination:                  {delete_root}")
    print()

    if not args.quiet:
        for src, keeper, dest in actions[:80]:
            print(f"remove: {src}")
            print(f"  keep: {keeper}")
            print(f"  move: {dest}")
        if len(actions) > 80:
            print(f"... and {len(actions) - 80} more")
        print()

    if not args.apply:
        print("DRY-RUN — no files moved. Re-run with --apply to remove duplicates.")
        return 0

    moved = 0
    for src, _keeper, dest in actions:
        if not src.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        moved += 1

    print(f"Moved {moved} duplicate image entries out of person folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

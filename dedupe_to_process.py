#!/usr/bin/env python3
"""
dedupe_to_process.py — Move exact duplicate source images out of To Process.

This is for the common case where folders under ~/Pictures were copied/merged
into ~/Pictures/To Process, leaving redundant copies. The script hashes image
files by content, keeps one copy, and moves duplicate copies from To Process to
a review folder.

Default is dry-run. Use --apply to move files.

Usage:
    python dedupe_to_process.py
    python dedupe_to_process.py --apply
    python dedupe_to_process.py --all-sources --apply
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
DEFAULT_PICTURES = Path.home() / "Pictures"
DEFAULT_TO_PROCESS = DEFAULT_PICTURES / "To Process"
DEFAULT_REVIEW = DEFAULT_PICTURES / "duplicate_to_review"
EXCLUDE_DIRS = {
    ".photoslibrary",
    "sorted",
    "sorted_all_pictures",
    "face_clusters",
    "photos_by_person",
    "junk_to_review",
    "duplicate_to_review",
}


def is_under(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def iter_images(root: Path, exclude_dirs: set[str]) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d not in exclude_dirs
            and d.lower() not in exclude_dirs
            and not d.startswith(".")
        ]
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


def choose_keeper(paths: list[Path], to_process: Path) -> Path:
    outside = [p for p in paths if not is_under(p, to_process)]
    pool = outside if outside else paths
    return sorted(pool, key=lambda p: (len(p.parts), str(p).lower()))[0]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pictures-root", type=Path, default=DEFAULT_PICTURES)
    parser.add_argument("--to-process", type=Path, default=DEFAULT_TO_PROCESS)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--apply", action="store_true",
                        help="Move duplicates. Default is dry-run.")
    parser.add_argument("--all-sources", action="store_true",
                        help="Move duplicate copies anywhere under Pictures, not just To Process.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print each duplicate path.")
    args = parser.parse_args()

    pictures_root = args.pictures_root.expanduser().resolve()
    to_process = args.to_process.expanduser().resolve()
    review_dir = args.review_dir.expanduser().resolve()

    if not pictures_root.exists():
        print(f"ERROR: pictures root not found: {pictures_root}")
        return 1

    print(f"Scanning images under: {pictures_root}")
    print(f"To Process folder:    {to_process}")
    print(f"Review folder:        {review_dir}")
    print()

    images = iter_images(pictures_root, EXCLUDE_DIRS)
    by_size: dict[int, list[Path]] = defaultdict(list)
    for p in images:
        try:
            by_size[p.stat().st_size].append(p)
        except OSError:
            continue

    candidate_sizes = {size: paths for size, paths in by_size.items() if len(paths) > 1}
    print(f"Image files scanned:       {len(images)}")
    print(f"Same-size candidate files: {sum(len(v) for v in candidate_sizes.values())}")
    print(f"Same-size groups:          {len(candidate_sizes)}")
    print()

    by_hash: dict[str, list[Path]] = defaultdict(list)
    for paths in candidate_sizes.values():
        for p in paths:
            try:
                by_hash[sha1_file(p)].append(p)
            except OSError:
                continue

    moves: list[tuple[Path, Path, Path]] = []
    duplicate_groups = 0
    duplicate_files = 0
    for paths in by_hash.values():
        if len(paths) < 2:
            continue
        duplicate_groups += 1
        keeper = choose_keeper(paths, to_process)
        duplicate_files += len(paths) - 1
        for p in sorted(paths, key=lambda x: str(x).lower()):
            if p == keeper:
                continue
            if not args.all_sources and not is_under(p, to_process):
                continue
            try:
                rel = p.relative_to(pictures_root)
            except ValueError:
                rel = Path(p.name)
            dest = unique_dest(review_dir / rel)
            moves.append((p, dest, keeper))

    print(f"Exact duplicate groups:    {duplicate_groups}")
    print(f"Exact duplicate files:     {duplicate_files}")
    print(f"Duplicates selected move:  {len(moves)}")
    print()

    if not args.quiet:
        for src, dest, keeper in moves[:80]:
            print(f"move: {src}")
            print(f"keep: {keeper}")
            print(f"  -> {dest}")
        if len(moves) > 80:
            print(f"... and {len(moves) - 80} more")
        print()

    if not args.apply:
        print("DRY-RUN — no files moved. Re-run with --apply to commit.")
        return 0

    moved = 0
    for src, dest, _keeper in moves:
        if not src.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        moved += 1

    print(f"Moved {moved} duplicate file(s) to: {review_dir}")
    print("Review that folder in Finder, then delete it if everything looks right.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

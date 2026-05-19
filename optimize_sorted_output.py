#!/usr/bin/env python3
"""
optimize_sorted_output.py — Remove redundant storage from sorted person folders.

The sorted output intentionally copies a source photo into every person folder
that contains that person. That is useful for browsing, but it can create many
byte-for-byte duplicate files. This script scans photos_by_person, finds exact
content duplicates, and replaces duplicate copies with hardlinks to one keeper.

Result: every person folder still has its image entries, but duplicate file
data is stored once on disk.

Default is dry-run. Use --apply to rewrite duplicate copies as hardlinks.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import tempfile
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp",
              ".tif", ".tiff", ".heic", ".heif"}
DEFAULT_PHOTOS = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"


def iter_images(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(root):
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


def same_inode(a: Path, b: Path) -> bool:
    try:
        sa = a.stat()
        sb = b.stat()
    except OSError:
        return False
    return sa.st_dev == sb.st_dev and sa.st_ino == sb.st_ino


def choose_keeper(paths: list[Path]) -> Path:
    def score(p: Path) -> tuple[int, int, str]:
        parts = set(p.parts)
        duplicate_penalty = 1 if "_duplicates" in parts else 0
        blurred_penalty = 1 if "_blurred" in parts else 0
        return (duplicate_penalty, blurred_penalty, str(p).lower())

    return sorted(paths, key=score)[0]


def replace_with_hardlink(src: Path, keeper: Path) -> bool:
    if same_inode(src, keeper):
        return False
    tmp_fd, tmp_name = tempfile.mkstemp(prefix=".dedupe-hardlink-", dir=str(src.parent))
    os.close(tmp_fd)
    tmp = Path(tmp_name)
    try:
        tmp.unlink()
        os.link(str(keeper), str(tmp))
        tmp.replace(src)
        return True
    except Exception:
        try:
            if tmp.exists():
                tmp.unlink()
        except OSError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos_dir", nargs="?", default=str(DEFAULT_PHOTOS))
    parser.add_argument("--apply", action="store_true",
                        help="Replace duplicate copies with hardlinks. Default is dry-run.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print sample duplicate paths.")
    args = parser.parse_args()

    root = Path(args.photos_dir).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: photos_by_person folder not found: {root}")
        return 1

    print(f"Scanning sorted photos: {root}")
    images = iter_images(root)
    by_size: dict[int, list[Path]] = defaultdict(list)
    total_bytes = 0
    for p in images:
        try:
            size = p.stat().st_size
        except OSError:
            continue
        total_bytes += size
        by_size[size].append(p)

    candidate_groups = {size: paths for size, paths in by_size.items() if len(paths) > 1}
    print(f"Image files:                 {len(images)}")
    print(f"Apparent image bytes:        {total_bytes / (1024**3):.2f} GB")
    print(f"Same-size candidate groups:  {len(candidate_groups)}")
    print()

    by_hash: dict[str, list[Path]] = defaultdict(list)
    for paths in candidate_groups.values():
        for p in paths:
            try:
                by_hash[sha1_file(p)].append(p)
            except OSError:
                continue

    actions: list[tuple[Path, Path, int]] = []
    duplicate_groups = 0
    redundant_bytes = 0
    already_hardlinked = 0
    for paths in by_hash.values():
        if len(paths) < 2:
            continue
        duplicate_groups += 1
        keeper = choose_keeper(paths)
        try:
            size = keeper.stat().st_size
        except OSError:
            size = 0
        for p in paths:
            if p == keeper:
                continue
            if same_inode(p, keeper):
                already_hardlinked += 1
                continue
            actions.append((p, keeper, size))
            redundant_bytes += size

    print(f"Exact duplicate groups:      {duplicate_groups}")
    print(f"Duplicate files to optimize: {len(actions)}")
    print(f"Already hardlinked files:    {already_hardlinked}")
    print(f"Potential storage saved:     {redundant_bytes / (1024**3):.2f} GB")
    print()

    if not args.quiet:
        for src, keeper, _size in actions[:60]:
            print(f"link: {src}")
            print(f"  -> {keeper}")
        if len(actions) > 60:
            print(f"... and {len(actions) - 60} more")
        print()

    if not args.apply:
        print("DRY-RUN — no files changed. Re-run with --apply to hardlink duplicates.")
        return 0

    changed = 0
    for src, keeper, _size in actions:
        if src.exists() and keeper.exists():
            replace_with_hardlink(src, keeper)
            changed += 1

    print(f"Optimized {changed} duplicate file(s) with hardlinks.")
    print("Person folders are preserved; duplicate file data is no longer stored repeatedly.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Move truly empty person folders out of photos_by_person.

A folder is considered empty when it has no real source files after ignoring
generated views such as all/ and _smart_albums/. Folders are moved to
_source_review/ready_to_delete/empty_person_folders_<timestamp>, never deleted.
"""

from __future__ import annotations

import argparse
import os
import shutil
import time
from pathlib import Path

DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
DEFAULT_READY = Path.home() / "Pictures" / "sorted_all_pictures" / "_source_review" / "ready_to_delete"
SKIP_DIRS = {"all", "_smart_albums"}


def has_real_source_files(person_dir: Path) -> bool:
    for dirpath, dirnames, filenames in os.walk(person_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        visible = [name for name in filenames if name != ".DS_Store" and not name.startswith(".")]
        if visible:
            return True
    return False


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    parent = dest.parent
    stem = dest.name
    i = 2
    while True:
        candidate = parent / f"{stem}__{i}"
        if not candidate.exists():
            return candidate
        i += 1


def person_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")],
                  key=lambda p: p.name.lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-dir", type=Path, default=DEFAULT_PEOPLE)
    parser.add_argument("--ready-dir", type=Path, default=DEFAULT_READY)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    people_dir = args.people_dir.expanduser().resolve()
    ready_dir = args.ready_dir.expanduser().resolve()
    if not people_dir.exists():
        print(f"ERROR: people folder not found: {people_dir}")
        return 1

    empty = [p for p in person_dirs(people_dir) if not has_real_source_files(p)]
    archive_root = ready_dir / f"empty_person_folders_{time.strftime('%Y%m%d_%H%M%S')}"

    moved = 0
    for person_dir in empty:
        dest = unique_dest(archive_root / person_dir.name)
        if not args.quiet:
            print(f"empty: {person_dir}")
            print(f"  ->   {dest}")
        if args.apply:
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(person_dir), str(dest))
            moved += 1

    print()
    print(f"People folder:        {people_dir}")
    print(f"Empty folders found:  {len(empty)}")
    print(f"Moved:                {moved}")
    print(f"Archive root:         {archive_root}")
    if not args.apply:
        print("DRY-RUN - no folders moved. Re-run with --apply to commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

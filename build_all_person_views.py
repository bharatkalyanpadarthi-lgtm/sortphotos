#!/usr/bin/env python3
"""
Build a simple hardlinked "all" view inside each person folder.

For each person:
  photos_by_person/<person>/all/
      Person_0001_photo_portrait_q_high.jpg
      ...
      nude/
          Person_0042_nudity_possible_portrait_q_high.jpg

The files are hardlinks to the real organized files, so this does not duplicate
image data on disk. The view is rebuilt from scratch each run and exact content
duplicates are skipped within each view.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".gif"}
DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
VIEW_DIR = "all"
NUDE_DIR = "nude"
SKIP_DIRS = {VIEW_DIR, "_smart_albums"}
NUDE_SOURCE_DIRS = {"_possible_nudity", "_uncertain_nudity"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def iter_source_images(person_dir: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(person_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        base = Path(dirpath)
        for filename in filenames:
            path = base / filename
            if is_image(path):
                out.append(path)
    return sorted(out, key=lambda p: str(p.relative_to(person_dir)).lower())


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
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def category_rank(path: Path, person_dir: Path) -> tuple[int, str]:
    rel = path.relative_to(person_dir)
    if rel.parts and rel.parts[0] in NUDE_SOURCE_DIRS:
        return (1, str(rel).lower())
    if rel.parts and rel.parts[0].startswith("_"):
        return (2, str(rel).lower())
    return (0, str(rel).lower())


def is_nude_source(path: Path, person_dir: Path) -> bool:
    rel = path.relative_to(person_dir)
    return bool(rel.parts and rel.parts[0] in NUDE_SOURCE_DIRS)


def hardlink_or_symlink(src: Path, dest: Path, apply: bool) -> bool:
    if not apply:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(src), str(dest))
    except OSError:
        os.symlink(str(src), str(dest))
    return True


def build_for_person(person_dir: Path, apply: bool) -> tuple[int, int, int]:
    view_dir = person_dir / VIEW_DIR
    if apply and view_dir.exists():
        shutil.rmtree(view_dir)

    seen_all: set[str] = set()
    seen_nude: set[str] = set()
    linked_all = 0
    linked_nude = 0
    skipped_dupes = 0

    for src in sorted(iter_source_images(person_dir), key=lambda p: category_rank(p, person_dir)):
        try:
            digest = sha256_file(src)
        except OSError:
            continue
        if digest in seen_all:
            skipped_dupes += 1
        else:
            seen_all.add(digest)
            dest = unique_dest(view_dir / src.name)
            if hardlink_or_symlink(src, dest, apply):
                linked_all += 1
        if is_nude_source(src, person_dir):
            if digest in seen_nude:
                continue
            seen_nude.add(digest)
            dest = unique_dest(view_dir / NUDE_DIR / src.name)
            if hardlink_or_symlink(src, dest, apply):
                linked_nude += 1

    return linked_all, linked_nude, skipped_dupes


def person_dirs(root: Path, person: str | None) -> list[Path]:
    dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name != VIEW_DIR]
    if person:
        wanted = person.casefold()
        dirs = [p for p in dirs if p.name.casefold() == wanted]
    return sorted(dirs, key=lambda p: p.name.lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("people_dir", nargs="?", default=str(DEFAULT_PEOPLE))
    parser.add_argument("--person", default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    people_dir = Path(args.people_dir).expanduser().resolve()
    if not people_dir.exists():
        print(f"ERROR: people folder not found: {people_dir}")
        return 1

    dirs = person_dirs(people_dir, args.person)
    if args.person and not dirs:
        print(f"ERROR: person folder not found: {args.person}")
        return 1

    total_all = total_nude = total_dupes = 0
    for person_dir in dirs:
        linked_all, linked_nude, skipped_dupes = build_for_person(person_dir, args.apply)
        total_all += linked_all
        total_nude += linked_nude
        total_dupes += skipped_dupes
        if not args.quiet:
            print(f"{person_dir.name:<34} all={linked_all:<5} nude={linked_nude:<5} dupes_skipped={skipped_dupes}")

    print()
    print(f"People folder:        {people_dir}")
    print(f"Person folders:       {len(dirs)}")
    print(f"All-view links:       {total_all}")
    print(f"Nude-view links:      {total_nude}")
    print(f"Duplicates skipped:   {total_dupes}")
    if not args.apply:
        print("DRY-RUN - no all views created. Re-run with --apply to commit.")
    else:
        print("All views rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""
flatten_blurred.py — Merge _blurred/ subfolders back into the main person folder.

Use this if you ran sort_photos.py with the blur-split feature on, and now want
all photos in one folder per person (no _blurred/ subdir).

What it does:
  ~/Pictures/sorted_all_pictures/photos_by_person/Syamala/_blurred/photo.jpg
       → ~/Pictures/sorted_all_pictures/photos_by_person/Syamala/photo.jpg

  ~/Pictures/sorted_all_pictures/photos_by_person/Syamala/_blurred/_duplicates/photo.jpg
       → ~/Pictures/sorted_all_pictures/photos_by_person/Syamala/_duplicates/photo.jpg

  Then deletes the now-empty _blurred/ folders.

Filename collisions are handled by appending __2, __3, etc.
Safe to re-run — does nothing if no _blurred/ folders found.

Run:
    python flatten_blurred.py
    python flatten_blurred.py /path/to/photos_by_person
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

DEFAULT_DIR    = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
BLURRED_DIR    = "_blurred"
DUPLICATES_DIR = "_duplicates"


def unique_dest(dest_dir: Path, filename: str) -> Path:
    candidate = dest_dir / filename
    if not candidate.exists():
        return candidate
    stem, suffix = candidate.stem, candidate.suffix
    i = 2
    while True:
        candidate = dest_dir / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def move_files_up(src_dir: Path, dst_dir: Path) -> int:
    """Move files from src_dir into dst_dir. Returns number moved."""
    if not src_dir.exists() or not src_dir.is_dir():
        return 0
    dst_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in list(src_dir.iterdir()):
        if f.is_file():
            dest = unique_dest(dst_dir, f.name)
            shutil.move(str(f), str(dest))
            moved += 1
    return moved


def flatten_person(person_dir: Path) -> tuple[int, int]:
    """For one person folder, merge _blurred/* up. Returns (sharp_moved, dup_moved)."""
    blurred_dir = person_dir / BLURRED_DIR
    if not blurred_dir.exists():
        return (0, 0)

    # Move blurred duplicates first (deeper nesting)
    blurred_dups = blurred_dir / DUPLICATES_DIR
    main_dups   = person_dir / DUPLICATES_DIR
    dup_moved = move_files_up(blurred_dups, main_dups)
    if blurred_dups.exists():
        try:
            blurred_dups.rmdir()
        except OSError:
            pass

    # Move blurred sharp shots up to main
    sharp_moved = move_files_up(blurred_dir, person_dir)
    try:
        blurred_dir.rmdir()
    except OSError:
        pass  # still has something — leave it visible

    return (sharp_moved, dup_moved)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("photos_dir", nargs="?", default=str(DEFAULT_DIR))
    args = parser.parse_args()

    photos_dir = Path(args.photos_dir).expanduser()
    if not photos_dir.exists():
        print(f"ERROR: not found: {photos_dir}")
        return 1

    print(f"Flattening _blurred/ folders under: {photos_dir}\n")

    total_sharp = 0
    total_dup = 0
    affected = 0

    for person_dir in sorted(photos_dir.iterdir()):
        if not person_dir.is_dir():
            continue
        # Skip the top-level _blurred folder if any (legacy layout)
        if person_dir.name == BLURRED_DIR:
            continue
        sharp, dup = flatten_person(person_dir)
        if sharp or dup:
            affected += 1
            total_sharp += sharp
            total_dup += dup
            print(f"  {person_dir.name}: moved {sharp} photo(s), "
                  f"{dup} duplicate(s)")

    # Legacy: if there's a top-level _blurred/ from older script versions
    legacy = photos_dir / BLURRED_DIR
    if legacy.exists():
        print(f"\nFound legacy top-level _blurred/ — flattening it too:")
        for sub in sorted(legacy.iterdir()):
            if not sub.is_dir():
                continue
            target = photos_dir / sub.name
            sharp = move_files_up(sub, target)
            sub_dups = sub / DUPLICATES_DIR
            target_dups = target / DUPLICATES_DIR
            dup = move_files_up(sub_dups, target_dups)
            if sub_dups.exists():
                try: sub_dups.rmdir()
                except OSError: pass
            try: sub.rmdir()
            except OSError: pass
            if sharp or dup:
                affected += 1
                total_sharp += sharp
                total_dup += dup
                print(f"  {sub.name}: moved {sharp} photo(s), {dup} duplicate(s)")
        try:
            legacy.rmdir()
        except OSError:
            pass

    print(f"\nDone. {affected} person folder(s) updated. "
          f"Total: {total_sharp} photos, {total_dup} duplicates moved.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

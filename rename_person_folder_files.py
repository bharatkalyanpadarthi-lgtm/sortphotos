#!/usr/bin/env python3
"""
Rename images inside each person folder using the folder name plus a sequence.

Example:
  photos_by_person/Sonali Bendre/IMG_1234.jpg
  -> photos_by_person/Sonali Bendre/Sonali_Bendre_001.jpg

Default is dry-run. Use --apply to rename files.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp",
              ".tif", ".tiff", ".heic", ".heif"}
DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"


def iter_images(person_dir: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, _dirnames, filenames in os.walk(person_dir):
        base = Path(dirpath)
        for filename in filenames:
            p = base / filename
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
    return sorted(
        out,
        key=lambda p: (
            len(p.relative_to(person_dir).parts),
            str(p.relative_to(person_dir)).lower(),
        ),
    )


def clean_prefix(folder_name: str) -> str:
    name = folder_name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[/:\\]+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._- ")
    return name or "person"


def plan_for_person(person_dir: Path) -> list[tuple[Path, Path]]:
    images = iter_images(person_dir)
    if not images:
        return []

    width = max(3, len(str(len(images))))
    prefix = clean_prefix(person_dir.name)
    planned: list[tuple[Path, Path]] = []
    used: set[Path] = set()
    max_index = 0
    existing_pattern = re.compile(rf"^{re.escape(prefix)}_(\d+)$")

    for src in images:
        match = existing_pattern.fullmatch(src.stem)
        if match:
            max_index = max(max_index, int(match.group(1)))
            used.add(src)

    next_index = max_index + 1
    for src in images:
        if existing_pattern.fullmatch(src.stem):
            continue
        ext = src.suffix.lower()
        while True:
            dest = src.parent / f"{prefix}_{next_index:0{width}d}{ext}"
            next_index += 1
            if dest not in used and not dest.exists():
                break
        if dest in used:
            raise RuntimeError(f"duplicate planned path: {dest}")
        used.add(dest)
        planned.append((src, dest))
    return planned


def temp_path(src: Path, i: int) -> Path:
    return src.with_name(f".rename_tmp_{os.getpid()}_{i}{src.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("people_dir", nargs="?", default=str(DEFAULT_PEOPLE))
    parser.add_argument("--apply", action="store_true",
                        help="Rename files. Default is dry-run.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print sample rename paths.")
    args = parser.parse_args()

    people_dir = Path(args.people_dir).expanduser().resolve()
    if not people_dir.exists():
        print(f"ERROR: people folder not found: {people_dir}")
        return 1

    all_actions: list[tuple[Path, Path]] = []
    folder_count = 0
    image_count = 0
    for person_dir in sorted([p for p in people_dir.iterdir() if p.is_dir()],
                             key=lambda p: p.name.lower()):
        folder_count += 1
        images = iter_images(person_dir)
        image_count += len(images)
        all_actions.extend(plan_for_person(person_dir))

    print(f"People folder:      {people_dir}")
    print(f"Person folders:     {folder_count}")
    print(f"Image files:        {image_count}")
    print(f"Files to rename:    {len(all_actions)}")
    print()

    if not args.quiet:
        for src, dest in all_actions[:100]:
            print(f"rename: {src}")
            print(f"  ->    {dest}")
        if len(all_actions) > 100:
            print(f"... and {len(all_actions) - 100} more")
        print()

    if not args.apply:
        print("DRY-RUN — no files renamed. Re-run with --apply to commit.")
        return 0

    temp_actions: list[tuple[Path, Path, Path]] = []
    for i, (src, dest) in enumerate(all_actions, start=1):
        tmp = temp_path(src, i)
        if tmp.exists():
            raise RuntimeError(f"temporary path already exists: {tmp}")
        temp_actions.append((src, tmp, dest))

    for src, tmp, _dest in temp_actions:
        src.rename(tmp)

    try:
        for _src, tmp, dest in temp_actions:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise RuntimeError(f"destination unexpectedly exists: {dest}")
            tmp.rename(dest)
    except Exception:
        print("ERROR: rename failed after temporary step; some files may have .rename_tmp names.")
        raise

    print(f"Renamed {len(all_actions)} file(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

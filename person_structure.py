#!/usr/bin/env python3
"""
Audit and repair photos_by_person folder structure.

Canonical layout per person:

  Person/
    photos/              real non-nude person images
      nude/              real nudity-possible images
    review/
      duplicates/
      near_visual/
    all/                 generated hardlink view
    _smart_albums/       generated hardlink albums

Default is read-only. Use --apply to move/link files into the canonical layout.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import operation_ledger

IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif",
}
DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
DEFAULT_REVIEW = Path.home() / "Pictures" / "sorted_all_pictures" / "_source_review" / "structure_review"

PHOTOS_DIR = "photos"
NUDE_DIR = "nude"
REVIEW_DIR = "review"
GENERATED_DIRS = {"all", "_smart_albums", "_smart_albums_v2"}
CANONICAL_TOPS = {PHOTOS_DIR, REVIEW_DIR}
LEGACY_TOPS = {"_possible_nudity", "_uncertain_nudity", "_duplicates", "_near_visual_review"}


@dataclass
class Stats:
    people: int = 0
    suspicious_people: int = 0
    root_files: int = 0
    legacy_files: int = 0
    custom_files: int = 0
    generated_only: int = 0
    generated_only_unreadable: int = 0
    dot_files: int = 0
    empty_dirs: int = 0
    moved_files: int = 0
    linked_recovered: int = 0
    moved_suspicious: int = 0
    removed_dot_files: int = 0
    removed_empty_dirs: int = 0


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


@contextlib.contextmanager
def suppress_native_stderr(enabled: bool = True):
    if not enabled:
        yield
        return
    try:
        fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    saved_fd = os.dup(fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(saved_fd, fd)
        os.close(saved_fd)


def can_decode_image(path: Path) -> bool:
    with suppress_native_stderr():
        try:
            import cv2
            import numpy as np

            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is not None and getattr(img, "size", 0) > 0:
                return True
        except Exception:
            pass

        try:
            from PIL import Image, ImageFile
            import pillow_heif

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            if hasattr(pillow_heif, "register_heif_opener"):
                pillow_heif.register_heif_opener()
            with Image.open(path) as im:
                im.load()
                return im.size[0] > 0 and im.size[1] > 0
        except Exception:
            return False


def clean_prefix(name: str) -> str:
    name = re.sub(r"\s+", "_", name.strip())
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._- ")
    return name or "person"


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    suffix = "".join(dest.suffixes)
    stem = dest.name[:-len(suffix)] if suffix else dest.name
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def sorted_root_for_path(path: Path) -> Path:
    for parent in path.parents:
        if parent.name == "photos_by_person":
            return parent.parent
    return DEFAULT_PEOPLE.parent


def person_dirs(people_dir: Path) -> list[Path]:
    if not people_dir.exists():
        return []
    return sorted([p for p in people_dir.iterdir() if p.is_dir()], key=lambda p: p.name.casefold())


def image_files(person_dir: Path) -> list[Path]:
    return sorted(
        [p for p in person_dir.rglob("*") if is_image(p)],
        key=lambda p: str(p.relative_to(person_dir)).casefold(),
    )


def dot_files(person_dir: Path) -> list[Path]:
    return sorted(
        [p for p in person_dir.rglob(".*") if p.is_file()],
        key=lambda p: str(p.relative_to(person_dir)).casefold(),
    )


def is_generated_path(path: Path, person_dir: Path) -> bool:
    try:
        rel = path.relative_to(person_dir)
    except ValueError:
        return False
    return bool(rel.parts and rel.parts[0] in GENERATED_DIRS)


def canonical_target(person_dir: Path, src: Path) -> Path | None:
    rel = src.relative_to(person_dir)
    if not rel.parts:
        return None
    top = rel.parts[0]
    name_lower = src.name.casefold()

    if top in GENERATED_DIRS:
        return None
    if top == PHOTOS_DIR or top == REVIEW_DIR:
        return None
    if top == "photos_nude":
        return person_dir / PHOTOS_DIR / NUDE_DIR / Path(*rel.parts[1:])
    if top == "_possible_nudity":
        return person_dir / PHOTOS_DIR / NUDE_DIR / Path(*rel.parts[1:])
    if top == "_uncertain_nudity":
        return person_dir / PHOTOS_DIR / NUDE_DIR / Path(*rel.parts[1:])
    if top == "_duplicates":
        return person_dir / REVIEW_DIR / "duplicates" / Path(*rel.parts[1:])
    if top == "_near_visual_review":
        return person_dir / REVIEW_DIR / "near_visual" / Path(*rel.parts[1:])
    if len(rel.parts) == 1:
        if "nudity_possible" in name_lower or "nude" in name_lower:
            return person_dir / PHOTOS_DIR / NUDE_DIR / src.name
        if "nudity_uncertain" in name_lower:
            return person_dir / PHOTOS_DIR / NUDE_DIR / src.name
        return person_dir / PHOTOS_DIR / src.name
    if top.startswith("_"):
        return person_dir / REVIEW_DIR / "other" / Path(*rel.parts)
    return person_dir / PHOTOS_DIR / Path(*rel.parts)


def move_file(src: Path, dest: Path, apply: bool) -> bool:
    dest = unique_dest(dest)
    if not apply:
        return True
    operation_ledger.move_path(
        src,
        dest,
        sorted_root=sorted_root_for_path(src),
        operation="person_structure.move_file",
        reason="normalize person folder structure",
    )
    return True


def hardlink_recovery(src: Path, dest: Path, apply: bool) -> bool:
    dest = unique_dest(dest)
    if not apply:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(src), str(dest))
    except OSError:
        shutil.copy2(str(src), str(dest))
    return True


def generated_only_images(person_dir: Path) -> list[Path]:
    by_inode: dict[tuple[int, int], list[Path]] = defaultdict(list)
    for path in image_files(person_dir):
        try:
            st = path.stat()
        except OSError:
            continue
        by_inode[(st.st_dev, st.st_ino)].append(path)

    recover: list[Path] = []
    for paths in by_inode.values():
        if paths and all(is_generated_path(path, person_dir) for path in paths):
            recover.append(sorted(paths, key=lambda p: str(p).casefold())[0])
    return recover


def prune_empty_dirs(root: Path, apply: bool) -> int:
    empty: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        path = Path(dirpath)
        if path == root:
            continue
        if dirnames or filenames:
            continue
        empty.append(path)
    if apply:
        for path in empty:
            try:
                path.rmdir()
            except OSError:
                pass
    return len(empty)


def move_suspicious_person(person_dir: Path, review_root: Path, apply: bool) -> bool:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest = unique_dest(review_root / "unknown_person" / f"{person_dir.name}_{stamp}")
    if apply:
        operation_ledger.move_path(
            person_dir,
            dest,
            sorted_root=person_dir.parent.parent,
            operation="person_structure.move_suspicious_person",
            reason="move suspicious person folder to structure review",
            hash_file=False,
        )
    return True


def audit_or_repair(people_dir: Path, review_root: Path, apply: bool, quiet: bool) -> Stats:
    stats = Stats()
    for person_dir in person_dirs(people_dir):
        if person_dir.name.startswith("_"):
            stats.suspicious_people += 1
            stats.people += 1
            if not quiet:
                print(f"suspicious person folder: {person_dir.name}")
            if apply:
                move_suspicious_person(person_dir, review_root, apply=True)
                stats.moved_suspicious += 1
            continue

        stats.people += 1

        recover_candidates = generated_only_images(person_dir)
        recover = [src for src in recover_candidates if can_decode_image(src)]
        stats.generated_only += len(recover)
        stats.generated_only_unreadable += len(recover_candidates) - len(recover)
        for i, src in enumerate(recover, start=1):
            dest = person_dir / PHOTOS_DIR / f"{clean_prefix(person_dir.name)}_recovered_{i:04d}{src.suffix.lower()}"
            if hardlink_recovery(src, dest, apply):
                stats.linked_recovered += 1 if apply else 0

        for src in image_files(person_dir):
            if is_generated_path(src, person_dir):
                continue
            target = canonical_target(person_dir, src)
            if target is None:
                continue
            rel = src.relative_to(person_dir)
            if len(rel.parts) == 1:
                stats.root_files += 1
            elif rel.parts[0] in LEGACY_TOPS:
                stats.legacy_files += 1
            else:
                stats.custom_files += 1
            if move_file(src, target, apply):
                stats.moved_files += 1 if apply else 0

        dots = dot_files(person_dir)
        stats.dot_files += len(dots)
        if apply:
            for path in dots:
                try:
                    path.unlink()
                    stats.removed_dot_files += 1
                except OSError:
                    pass

        stats.empty_dirs += prune_empty_dirs(person_dir, apply=False)
        if apply:
            stats.removed_empty_dirs += prune_empty_dirs(person_dir, apply=True)

    return stats


def print_stats(stats: Stats, people_dir: Path, review_root: Path, apply: bool) -> None:
    print("Person Folder Structure")
    print("=" * 60)
    print(f"People folder:               {people_dir}")
    print(f"Unknown review folder:       {review_root / 'unknown_person'}")
    print(f"Mode:                        {'APPLY' if apply else 'DRY-RUN'}")
    print()
    print(f"Person folders scanned:      {stats.people}")
    print(f"Suspicious person folders:   {stats.suspicious_people}")
    print(f"Root image files to move:    {stats.root_files}")
    print(f"Legacy folder files to move: {stats.legacy_files}")
    print(f"Custom folder files to move: {stats.custom_files}")
    print(f"Generated-only images:       {stats.generated_only}")
    print(f"Generated-only unreadable:   {stats.generated_only_unreadable}")
    print(f"Dot/metadata files:          {stats.dot_files}")
    print(f"Empty dirs found:            {stats.empty_dirs}")
    if apply:
        print()
        print(f"Files moved:                 {stats.moved_files}")
        print(f"Generated-only recovered:    {stats.linked_recovered}")
        print(f"Suspicious folders moved:    {stats.moved_suspicious}")
        print(f"Dot files removed:           {stats.removed_dot_files}")
        print(f"Empty dirs removed:          {stats.removed_empty_dirs}")
    else:
        print()
        print("DRY-RUN - no files changed. Re-run with --apply to commit.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-dir", type=Path, default=DEFAULT_PEOPLE)
    parser.add_argument("--review-root", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    people_dir = args.people_dir.expanduser().resolve()
    review_root = args.review_root.expanduser().resolve()
    if not people_dir.exists():
        print(f"ERROR: people folder not found: {people_dir}")
        return 1

    stats = audit_or_repair(people_dir, review_root, args.apply, args.quiet)
    print_stats(stats, people_dir, review_root, args.apply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

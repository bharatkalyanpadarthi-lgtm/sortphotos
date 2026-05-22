#!/usr/bin/env python3
"""
Rename images inside each person folder using a smart canonical name.

Example:
  photos_by_person/Sonali Bendre/IMG_1234.jpg
  -> photos_by_person/Sonali Bendre/Sonali_Bendre_0001_photo_portrait_q_high.jpg

Default is dry-run. Use --apply to rename files.
"""

from __future__ import annotations

import argparse
import contextlib
import os
import re
import sys
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp",
              ".tif", ".tiff", ".heic", ".heif", ".gif"}
DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
SKIP_DIRS = {"all", "_smart_albums"}
CATEGORY_RANK = {
    "photo": 0,
    "nudity_possible": 1,
    "nudity_uncertain": 2,
    "near_visual_review": 3,
    "duplicate_review": 4,
    "review": 5,
}


class ImageMeta:
    def __init__(self, width: int = 0, height: int = 0, sharpness: float = 0.0) -> None:
        self.width = width
        self.height = height
        self.sharpness = sharpness


@contextlib.contextmanager
def suppress_native_stderr(enabled: bool = True):
    """Hide noisy libjpeg/libpng warnings emitted while reading recovered files."""
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


def iter_images(person_dir: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(person_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
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


def read_image_meta(path: Path) -> ImageMeta:
    with suppress_native_stderr():
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                return ImageMeta(w, h, sharp)
        except Exception:
            pass
        try:
            from PIL import Image, ImageFile

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(path) as im:
                w, h = im.size
                frame = im.convert("RGB")
                arr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                return ImageMeta(w, h, sharp)
        except Exception:
            return ImageMeta()


def orientation_label(meta: ImageMeta) -> str:
    if meta.width <= 0 or meta.height <= 0:
        return "unknown"
    ratio = meta.width / max(1, meta.height)
    if ratio > 1.2:
        return "landscape"
    if ratio < 0.82:
        return "portrait"
    return "square"


def quality_label(meta: ImageMeta) -> str:
    if meta.width <= 0 or meta.height <= 0:
        return "q_unknown"
    megapixels = (meta.width * meta.height) / 1_000_000.0
    if megapixels >= 1.0 and meta.sharpness >= 90.0:
        return "q_high"
    if megapixels >= 0.45 and meta.sharpness >= 25.0:
        return "q_good"
    return "q_review"


def category_label(rel: Path) -> str:
    if not rel.parts:
        return "photo"
    first = rel.parts[0]
    if len(rel.parts) >= 2 and first == "photos" and rel.parts[1] == "nude":
        return "nudity_possible"
    if first in {"photos_nude", "_possible_nudity"}:
        return "nudity_possible"
    if first == "_uncertain_nudity":
        return "nudity_uncertain"
    if first == "_near_visual_review":
        return "near_visual_review"
    if first == "_duplicates":
        return "duplicate_review"
    if first == "review" and len(rel.parts) > 1:
        if rel.parts[1] == "duplicates":
            return "duplicate_review"
        if rel.parts[1] == "near_visual":
            return "near_visual_review"
        if rel.parts[1] == "nudity_possible":
            return "nudity_possible"
        if rel.parts[1] == "uncertain_nudity":
            return "nudity_uncertain"
        return "review"
    if first == "photos":
        return "photo"
    if first.startswith("_"):
        return "review"
    return "photo"


def canonical_pattern(prefix: str) -> re.Pattern:
    return re.compile(
        rf"^{re.escape(prefix)}_(\d+)(?:_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)?$",
        re.IGNORECASE,
    )


def parsed_index(path: Path, prefix: str) -> int | None:
    match = canonical_pattern(prefix).fullmatch(path.stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def sort_key_for_image(path: Path, person_dir: Path, prefix: str) -> tuple:
    rel = path.relative_to(person_dir)
    category = category_label(rel)
    index = parsed_index(path, prefix)
    return (
        CATEGORY_RANK.get(category, 9),
        0 if index is not None else 1,
        index if index is not None else 999_999,
        len(rel.parts),
        str(rel).lower(),
    )


def smart_filename(prefix: str, index: int, meta: ImageMeta, category: str,
                   width: int, suffix: str) -> str:
    return (
        f"{prefix}_{index:0{width}d}_"
        f"{category}_{orientation_label(meta)}_{quality_label(meta)}"
        f"{suffix.lower()}"
    )


def plan_for_person(person_dir: Path, compact: bool = False) -> list[tuple[Path, Path]]:
    images = iter_images(person_dir)
    if not images:
        return []

    prefix = clean_prefix(person_dir.name)
    width = max(4, len(str(len(images))))
    planned: list[tuple[Path, Path]] = []
    used_paths: set[Path] = set()
    used_indices: set[int] = set()
    next_index = 1

    sorted_images = sorted(images, key=lambda p: sort_key_for_image(p, person_dir, prefix))
    if not compact:
        existing_indices = [i for i in (parsed_index(src, prefix) for src in sorted_images) if i is not None]
        next_index = max(existing_indices, default=0) + 1

    for ordinal, src in enumerate(sorted_images, start=1):
        rel = src.relative_to(person_dir)
        category = category_label(rel)
        meta = read_image_meta(src)
        index = ordinal if compact else parsed_index(src, prefix)
        if index is None or index in used_indices:
            while next_index in used_indices:
                next_index += 1
            index = next_index
            used_indices.add(index)
            next_index += 1
        else:
            used_indices.add(index)
        dest = src.parent / smart_filename(prefix, index, meta, category, width, src.suffix)
        if dest in used_paths:
            raise RuntimeError(f"duplicate planned path: {dest}")
        used_paths.add(dest)
        if src != dest:
            planned.append((src, dest))
    return planned


def temp_path(src: Path, i: int) -> Path:
    return src.with_name(f".rename_tmp_{os.getpid()}_{i}{src.suffix}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("people_dir", nargs="?", default=str(DEFAULT_PEOPLE))
    parser.add_argument("--person", default=None,
                        help="Rename one person folder only.")
    parser.add_argument("--compact", action="store_true",
                        help="Renumber each person from 0001. Default preserves existing numbers.")
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
    person_dirs = [p for p in people_dir.iterdir() if p.is_dir() and not p.name.startswith("_")]
    if args.person:
        wanted = args.person.casefold()
        person_dirs = [p for p in person_dirs if p.name.casefold() == wanted]
        if not person_dirs:
            print(f"ERROR: person folder not found: {args.person}")
            return 1

    for person_dir in sorted(person_dirs, key=lambda p: p.name.lower()):
        folder_count += 1
        images = iter_images(person_dir)
        image_count += len(images)
        all_actions.extend(plan_for_person(person_dir, compact=args.compact))

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

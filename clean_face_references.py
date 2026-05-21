#!/usr/bin/env python3
"""
Clean and compact ~/Pictures/Face References.

For each person folder, this script:
  - moves unreadable images to _reference_review/bad_images/<person>
  - moves exact/pixel duplicates to _reference_review/duplicates/<person>
  - keeps the best N images by size/sharpness/face-crop friendliness
  - moves extras to _reference_review/extras/<person>
  - renames kept images as Person_001.jpg, Person_002.jpg, ...

Default is dry-run. Use --apply to move/rename files.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

DEFAULT_REF_DIR = Path.home() / "Pictures" / "Face References"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
REVIEW_DIR_NAME = "_reference_review"


@dataclass
class RefImage:
    path: Path
    width: int
    height: int
    sharpness: float
    file_sha: str
    pixel_sha: str
    score: float


def imread(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        from PIL import Image
        with Image.open(path) as im:
            return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def pixel_sha256(img: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(img.shape).encode("ascii"))
    h.update(np.ascontiguousarray(img).tobytes())
    return h.hexdigest()


def sharpness(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def clean_prefix(folder_name: str) -> str:
    name = folder_name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[/:\\]+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._- ")
    return name or "person"


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = "".join(dest.suffixes)
    i = 2
    while True:
        candidate = dest.with_name(f"{stem}__{i}{suffix}")
        if not candidate.exists():
            return candidate
        i += 1


def move_file(src: Path, dest: Path, apply: bool) -> None:
    if not apply:
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(unique_dest(dest)))


def reference_score(img: np.ndarray, sharp: float) -> float:
    h, w = img.shape[:2]
    min_dim = min(w, h)
    max_dim = max(w, h)
    ratio = w / max(1, h)
    square_bonus = 1.0 if 0.75 <= ratio <= 1.33 and max_dim <= 768 else 0.0
    size_score = min(min_dim, 512) / 512.0
    sharp_score = min(np.log1p(sharp) / np.log1p(500.0), 1.0)
    return float(size_score * 2.0 + sharp_score * 2.0 + square_bonus)


def person_dirs(ref_dir: Path) -> list[Path]:
    if not ref_dir.exists():
        return []
    return sorted(
        [p for p in ref_dir.iterdir() if p.is_dir() and not p.name.startswith("_")],
        key=lambda p: p.name.lower(),
    )


def iter_person_images(person_dir: Path) -> list[Path]:
    return sorted(
        [p for p in person_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
        key=lambda p: p.name.lower(),
    )


def plan_person(person_dir: Path, review_root: Path, max_keep: int) -> tuple[list[RefImage], list[tuple[Path, Path, str]]]:
    good: list[RefImage] = []
    moves: list[tuple[Path, Path, str]] = []
    seen_file: dict[str, Path] = {}
    seen_pixel: dict[str, Path] = {}

    for path in iter_person_images(person_dir):
        img = imread(path)
        if img is None:
            moves.append((path, review_root / "bad_images" / person_dir.name / path.name, "bad"))
            continue
        try:
            file_hash = sha256_file(path)
            pixel_hash = pixel_sha256(img)
        except OSError:
            moves.append((path, review_root / "bad_images" / person_dir.name / path.name, "bad"))
            continue
        if file_hash in seen_file or pixel_hash in seen_pixel:
            moves.append((path, review_root / "duplicates" / person_dir.name / path.name, "duplicate"))
            continue
        seen_file[file_hash] = path
        seen_pixel[pixel_hash] = path
        sharp = sharpness(img)
        h, w = img.shape[:2]
        good.append(RefImage(
            path=path,
            width=w,
            height=h,
            sharpness=sharp,
            file_sha=file_hash,
            pixel_sha=pixel_hash,
            score=reference_score(img, sharp),
        ))

    good.sort(key=lambda x: (-x.score, x.path.name.lower()))
    keep = good[:max_keep]
    extras = good[max_keep:]
    for item in extras:
        moves.append((item.path, review_root / "extras" / person_dir.name / item.path.name, "extra"))
    return keep, moves


def rename_kept(person_dir: Path, keep_paths: list[Path], apply: bool) -> int:
    prefix = clean_prefix(person_dir.name)
    width = max(3, len(str(len(keep_paths))))
    targets: list[tuple[Path, Path]] = []
    for i, src in enumerate(sorted(keep_paths, key=lambda p: p.name.lower()), start=1):
        ext = src.suffix.lower() or ".jpg"
        dest = person_dir / f"{prefix}_{i:0{width}d}{ext}"
        if src != dest:
            targets.append((src, dest))
    if not apply:
        return len(targets)

    temp_moves: list[tuple[Path, Path, Path]] = []
    for i, (src, dest) in enumerate(targets, start=1):
        tmp = src.with_name(f".ref_rename_tmp_{os.getpid()}_{i}{src.suffix}")
        temp_moves.append((src, tmp, dest))
    for src, tmp, _dest in temp_moves:
        if src.exists():
            src.rename(tmp)
    for _src, tmp, dest in temp_moves:
        if tmp.exists():
            if dest.exists():
                dest = unique_dest(dest)
            tmp.rename(dest)
    return len(targets)


def rebuild_refs(max_per_person: int) -> int:
    script = Path(__file__).resolve().parent / "build_celeb_centroids.py"
    cmd = [
        sys.executable,
        str(script),
        str(DEFAULT_REF_DIR),
        str(Path.home() / ".face_sort_cache" / "reference_centroids.pkl"),
        "--max-per-person",
        str(max_per_person),
    ]
    return subprocess.run(cmd, check=False).returncode


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--max-keep", type=int, default=20,
                        help="Best images to keep per person. Default 20.")
    parser.add_argument("--apply", action="store_true",
                        help="Move/rename files. Default is dry-run.")
    parser.add_argument("--rebuild", action="store_true",
                        help="Rebuild reference centroids after cleaning.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    ref_dir = args.ref_dir.expanduser().resolve()
    review_root = ref_dir / REVIEW_DIR_NAME
    dirs = person_dirs(ref_dir)
    if not dirs:
        print(f"No person folders found in {ref_dir}")
        return 1

    totals = {"folders": 0, "kept": 0, "bad": 0, "duplicate": 0, "extra": 0, "renamed": 0}
    planned_moves: list[tuple[Path, Path, str]] = []

    for person_dir in dirs:
        keep, moves = plan_person(person_dir, review_root, max(1, int(args.max_keep)))
        keep_paths = [item.path for item in keep]
        renamed = rename_kept(person_dir, keep_paths, args.apply)
        for src, dest, reason in moves:
            move_file(src, dest, args.apply)
            totals[reason] += 1
        totals["folders"] += 1
        totals["kept"] += len(keep)
        totals["renamed"] += renamed
        planned_moves.extend(moves)
        if not args.quiet:
            print(f"{person_dir.name:<32} keep={len(keep):<3} move={len(moves):<3} rename={renamed}")

    print()
    print(f"Reference folder: {ref_dir}")
    print(f"Person folders:   {totals['folders']}")
    print(f"Kept images:      {totals['kept']}")
    print(f"Moved bad:        {totals['bad']}")
    print(f"Moved duplicates: {totals['duplicate']}")
    print(f"Moved extras:     {totals['extra']}")
    print(f"Renamed kept:     {totals['renamed']}")
    print(f"Review folder:    {review_root}")

    if not args.apply:
        print()
        print("DRY-RUN - no files moved or renamed. Re-run with --apply to commit.")
        return 0

    if args.rebuild:
        print()
        print("Rebuilding Face References DB...")
        return rebuild_refs(max(1, int(args.max_keep)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

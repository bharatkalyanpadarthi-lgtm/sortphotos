#!/usr/bin/env python3
"""
Clean and compact ~/Pictures/Face References.

For each person folder, this script:
  - moves unreadable images to _reference_review/bad_images/<person>
  - moves exact/pixel duplicates to _reference_review/duplicates/<person>
  - moves avoidable no-face/group/tiny-face/blurry refs to review
  - keeps the best N images by face size, sharpness, resolution, and framing
  - moves extras to _reference_review/extras/<person>
  - renames kept images as Person_001.jpg, Person_002.jpg, ...

Default is dry-run. Use --apply to move/rename files.
"""

from __future__ import annotations

import argparse
import csv
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
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
REVIEW_DIR_NAME = "_reference_review"
REPORT_NAME = "face_reference_quality_report.csv"
_FACE_CASCADE = None


@dataclass
class RefImage:
    path: Path
    width: int
    height: int
    sharpness: float
    face_count: int
    largest_face_ratio: float
    face_center_x: float
    face_center_y: float
    file_sha: str
    pixel_sha: str
    score: float
    quality_reason: str = "keep"


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


def face_cascade() -> cv2.CascadeClassifier | None:
    global _FACE_CASCADE
    if _FACE_CASCADE is not None:
        return _FACE_CASCADE
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    if cascade.empty():
        return None
    _FACE_CASCADE = cascade
    return cascade


def face_metrics(img: np.ndarray) -> tuple[int, float, float, float]:
    cascade = face_cascade()
    if cascade is None:
        return 0, 0.0, 0.0, 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    min_side = max(40, min(w, h) // 16)
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=5,
        minSize=(min_side, min_side),
    )
    if len(faces) == 0:
        return 0, 0.0, 0.0, 0.0
    largest = max(faces, key=lambda box: int(box[2]) * int(box[3]))
    x, y, fw, fh = [int(v) for v in largest]
    ratio = (fw * fh) / max(1, w * h)
    center_x = (x + fw / 2) / max(1, w)
    center_y = (y + fh / 2) / max(1, h)
    return int(len(faces)), float(ratio), float(center_x), float(center_y)


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


def quality_reason(face_count: int, face_ratio: float, sharp: float) -> str:
    if face_count <= 0:
        return "no_face"
    if face_count > 1:
        return "multiple_faces"
    if face_ratio < 0.018:
        return "tiny_face"
    if sharp < 18.0:
        return "blurry"
    return "keep"


def reference_score(img: np.ndarray, sharp: float,
                    face_count: int, face_ratio: float,
                    center_x: float, center_y: float) -> float:
    h, w = img.shape[:2]
    min_dim = min(w, h)
    max_dim = max(w, h)
    ratio = w / max(1, h)
    square_bonus = 1.0 if 0.70 <= ratio <= 1.45 and max_dim <= 1200 else 0.0
    size_score = min(min_dim, 512) / 512.0
    sharp_score = min(np.log1p(sharp) / np.log1p(500.0), 1.0)
    face_score = min(face_ratio / 0.16, 1.0) * 2.0 if face_count == 1 else -1.0
    center_bonus = 1.0 if 0.30 <= center_x <= 0.70 and 0.22 <= center_y <= 0.62 else 0.0
    return float(size_score * 1.4 + sharp_score * 2.0 + face_score + square_bonus + center_bonus)


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


def plan_person(person_dir: Path, review_root: Path, min_keep: int, max_keep: int) -> tuple[list[RefImage], list[tuple[Path, Path, str]], list[RefImage]]:
    good: list[RefImage] = []
    moves: list[tuple[Path, Path, str]] = []
    all_scored: list[RefImage] = []
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
        face_count, face_ratio, center_x, center_y = face_metrics(img)
        reason = quality_reason(face_count, face_ratio, sharp)
        item = RefImage(
            path=path,
            width=w,
            height=h,
            sharpness=sharp,
            face_count=face_count,
            largest_face_ratio=face_ratio,
            face_center_x=center_x,
            face_center_y=center_y,
            file_sha=file_hash,
            pixel_sha=pixel_hash,
            score=reference_score(img, sharp, face_count, face_ratio, center_x, center_y),
            quality_reason=reason,
        )
        all_scored.append(item)
        good.append(item)

    good.sort(key=lambda x: (-x.score, x.path.name.lower()))
    qualified = [item for item in good if item.quality_reason == "keep"]
    if len(qualified) >= min_keep:
        keep_pool = qualified
        for item in good:
            if item.quality_reason != "keep":
                moves.append((item.path, review_root / item.quality_reason / person_dir.name / item.path.name, item.quality_reason))
    else:
        keep_pool = good
    keep = keep_pool[:max_keep]
    keep_paths = {item.path for item in keep}
    extras = [item for item in keep_pool[max_keep:] if item.path not in keep_paths]
    for item in extras:
        moves.append((item.path, review_root / "extras" / person_dir.name / item.path.name, "extra"))
    return keep, moves, all_scored


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


def write_report(report: Path, rows: list[RefImage], actions: dict[Path, str]) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "person", "file_path", "action", "quality_reason", "score", "width", "height",
            "sharpness", "face_count", "largest_face_ratio", "face_center_x", "face_center_y",
        ])
        for item in sorted(rows, key=lambda r: (r.path.parent.name.lower(), -r.score, r.path.name.lower())):
            writer.writerow([
                item.path.parent.name,
                str(item.path),
                actions.get(item.path, "keep"),
                item.quality_reason,
                f"{item.score:.4f}",
                item.width,
                item.height,
                f"{item.sharpness:.1f}",
                item.face_count,
                f"{item.largest_face_ratio:.4f}",
                f"{item.face_center_x:.3f}",
                f"{item.face_center_y:.3f}",
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ref-dir", type=Path, default=DEFAULT_REF_DIR)
    parser.add_argument("--max-keep", type=int, default=20,
                        help="Best images to keep per person. Default 20.")
    parser.add_argument("--min-keep", type=int, default=15,
                        help="Do not remove lower-quality readable refs unless at least this many strong refs remain. Default 15.")
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
    scored_rows: list[RefImage] = []
    actions: dict[Path, str] = {}

    for person_dir in dirs:
        keep, moves, scored = plan_person(
            person_dir,
            review_root,
            max(1, int(args.min_keep)),
            max(1, int(args.max_keep)),
        )
        keep_paths = [item.path for item in keep]
        renamed = rename_kept(person_dir, keep_paths, args.apply)
        for src, dest, reason in moves:
            move_file(src, dest, args.apply)
            totals[reason] = totals.get(reason, 0) + 1
            actions[src] = f"move_{reason}"
        totals["folders"] += 1
        totals["kept"] += len(keep)
        totals["renamed"] += renamed
        planned_moves.extend(moves)
        scored_rows.extend(scored)
        if not args.quiet:
            print(f"{person_dir.name:<32} keep={len(keep):<3} move={len(moves):<3} rename={renamed}")
    write_report(review_root / REPORT_NAME, scored_rows, actions)

    print()
    print(f"Reference folder: {ref_dir}")
    print(f"Person folders:   {totals['folders']}")
    print(f"Kept images:      {totals['kept']}")
    print(f"Moved bad:        {totals['bad']}")
    print(f"Moved duplicates: {totals['duplicate']}")
    print(f"Moved no face:    {totals.get('no_face', 0)}")
    print(f"Moved multi face: {totals.get('multiple_faces', 0)}")
    print(f"Moved tiny face:  {totals.get('tiny_face', 0)}")
    print(f"Moved blurry:     {totals.get('blurry', 0)}")
    print(f"Moved extras:     {totals['extra']}")
    print(f"Renamed kept:     {totals['renamed']}")
    print(f"Review folder:    {review_root}")
    print(f"Quality report:   {review_root / REPORT_NAME}")

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

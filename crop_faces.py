"""
Batch face detector + cropper.
- Walks INPUT_DIR (recursively), finds images, detects faces with InsightFace (RetinaFace),
  and saves square, padded crops to OUTPUT_DIR — but only when faces are 'clearly visible'.

Tested target: Apple Silicon (M4) with CoreML / CPU ONNX runtime.

Install (one-time):
    pip install insightface onnxruntime opencv-python numpy tqdm
    # Apple Silicon GPU (optional, faster):
    # pip install onnxruntime-silicon

Run:
    python crop_faces.py
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from tqdm import tqdm

# =============================================================================
# CONFIG — tweak here
# =============================================================================

#  Resolves to /Users/<your-mac-username>/Pictures/...
#  Edit the folder names below if yours differ.
INPUT_DIR  = Path.home() / "Pictures" / "To Process"     # 500+ source images
OUTPUT_DIR = Path.home() / "Pictures" / "face_crops"     # crops will be written here

# "Clearly visible" thresholds
MIN_DET_SCORE   = 0.65   # RetinaFace confidence (0..1). Higher = stricter.
MIN_FACE_PX     = 80     # min(width, height) of the face bbox in pixels
MIN_SHARPNESS   = 60.0   # Laplacian variance on the face crop. Higher = sharper.

# Crop geometry
PADDING_RATIO   = 0.30   # 30% padding around the bbox before squaring
SQUARE_CROP     = True   # output square crops (recommended for downstream ML)
OUTPUT_SIZE     = None   # int (e.g., 512) to resize, or None to keep native size
JPEG_QUALITY    = 92

# Detection
DET_SIZE        = (1024, 1024)  # bigger = catches smaller / further faces, slower
PROVIDERS       = ["CPUExecutionProvider"]
# CPU-only: avoids a known CoreML shape-inference bug with RetinaFace on Apple Silicon.
# Plenty fast on an M-series Mac.

# Filtering / behavior
SAVE_ALL_FACES  = True   # False = save only the largest qualifying face per image
SKIP_EXISTING   = True   # don't re-process if any crop already exists for an image
WRITE_REPORT    = True   # write a CSV manifest of what happened

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}

# =============================================================================

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("crop_faces")


@dataclass
class CropResult:
    src: Path
    status: str              # "ok" | "no_face" | "filtered" | "error" | "skipped"
    n_saved: int = 0
    n_detected: int = 0
    note: str = ""


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def imread_unicode(path: Path) -> np.ndarray | None:
    """cv2.imread chokes on some unicode paths and HEIC; use a byte-buffer fallback."""
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    # HEIC fallback via pillow-heif if available
    try:
        from PIL import Image
        import pillow_heif  # noqa: F401  # registers HEIF opener
        with Image.open(path) as im:
            return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def sharpness(gray_or_bgr: np.ndarray) -> float:
    g = gray_or_bgr if gray_or_bgr.ndim == 2 else cv2.cvtColor(gray_or_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def square_pad_bbox(x1: float, y1: float, x2: float, y2: float,
                    img_w: int, img_h: int,
                    pad_ratio: float, square: bool) -> tuple[int, int, int, int]:
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    side = max(w, h) if square else None

    if square:
        half = side / 2 * (1 + pad_ratio)
        nx1, ny1, nx2, ny2 = cx - half, cy - half, cx + half, cy + half
    else:
        nx1 = x1 - w * pad_ratio
        ny1 = y1 - h * pad_ratio
        nx2 = x2 + w * pad_ratio
        ny2 = y2 + h * pad_ratio

    nx1 = max(0, int(round(nx1)))
    ny1 = max(0, int(round(ny1)))
    nx2 = min(img_w, int(round(nx2)))
    ny2 = min(img_h, int(round(ny2)))
    return nx1, ny1, nx2, ny2


def build_detector():
    from insightface.app import FaceAnalysis
    log.info("Loading InsightFace (buffalo_l)…")
    app = FaceAnalysis(
        name="buffalo_l",
        allowed_modules=["detection"],   # detection-only is faster; no recognition needed
        providers=PROVIDERS,
    )
    app.prepare(ctx_id=0, det_size=DET_SIZE, det_thresh=MIN_DET_SCORE * 0.8)
    # We pass a slightly looser threshold to the detector and re-filter ourselves,
    # so we can reason about borderline cases consistently.
    return app


def process_one(src: Path, app, out_root: Path, in_root: Path) -> CropResult:
    rel = src.relative_to(in_root).with_suffix("")
    out_dir = out_root / rel.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    if SKIP_EXISTING and any(out_dir.glob(f"{src.stem}__face*")):
        return CropResult(src, "skipped", note="crops already exist")

    img = imread_unicode(src)
    if img is None:
        return CropResult(src, "error", note="could not read image")

    H, W = img.shape[:2]
    faces = app.get(img)
    if not faces:
        return CropResult(src, "no_face", n_detected=0)

    # Sort largest-first by bbox area
    def area(f):
        x1, y1, x2, y2 = f.bbox
        return (x2 - x1) * (y2 - y1)
    faces = sorted(faces, key=area, reverse=True)
    if not SAVE_ALL_FACES:
        faces = faces[:1]

    saved = 0
    for i, f in enumerate(faces):
        score = float(getattr(f, "det_score", 0.0))
        if score < MIN_DET_SCORE:
            continue

        x1, y1, x2, y2 = [float(v) for v in f.bbox]
        bw, bh = x2 - x1, y2 - y1
        if min(bw, bh) < MIN_FACE_PX:
            continue

        nx1, ny1, nx2, ny2 = square_pad_bbox(x1, y1, x2, y2, W, H, PADDING_RATIO, SQUARE_CROP)
        crop = img[ny1:ny2, nx1:nx2]
        if crop.size == 0:
            continue

        if sharpness(crop) < MIN_SHARPNESS:
            continue

        if OUTPUT_SIZE:
            crop = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)

        out_name = f"{src.stem}__face{i}_{int(score * 100)}.jpg"
        out_path = out_dir / out_name
        cv2.imwrite(str(out_path), crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
        saved += 1

    status = "ok" if saved else "filtered"
    return CropResult(src, status, n_saved=saved, n_detected=len(faces))


def main() -> int:
    if not INPUT_DIR.exists():
        log.error("INPUT_DIR does not exist: %s", INPUT_DIR)
        return 2
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(INPUT_DIR))
    log.info("Found %d images under %s", len(images), INPUT_DIR)
    if not images:
        return 0

    app = build_detector()

    results: list[CropResult] = []
    for src in tqdm(images, desc="Processing", unit="img"):
        try:
            results.append(process_one(src, app, OUTPUT_DIR, INPUT_DIR))
        except Exception as e:  # noqa: BLE001
            log.exception("Failed on %s", src)
            results.append(CropResult(src, "error", note=str(e)))

    # Summary
    by_status = {}
    for r in results:
        by_status[r.status] = by_status.get(r.status, 0) + 1
    total_saved = sum(r.n_saved for r in results)

    log.info("Done. Crops written: %d", total_saved)
    for k, v in sorted(by_status.items()):
        log.info("  %-9s %d", k, v)

    if WRITE_REPORT:
        report_path = OUTPUT_DIR / "_report.csv"
        with report_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["source", "status", "detected", "saved", "note"])
            for r in results:
                w.writerow([str(r.src), r.status, r.n_detected, r.n_saved, r.note])
        log.info("Report: %s", report_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())

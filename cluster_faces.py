"""
Face clustering: groups faces from INPUT_DIR into one folder per person.

Pipeline:
1. Detect faces with RetinaFace (same filters as crop_faces.py)
2. Extract 512-d ArcFace embeddings
3. Cluster with DBSCAN (cosine distance)
4. Save crops into person_001/, person_002/, ... + unknown/ for singletons
5. Generate a montage per cluster + a CSV manifest

Run:
    python cluster_faces.py
"""

from __future__ import annotations

import csv
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================

INPUT_DIR  = Path.home() / "Pictures" / "To Process"
OUTPUT_DIR = Path.home() / "Pictures" / "face_clusters"

# "Clearly visible" thresholds (same as crop_faces.py)
MIN_DET_SCORE = 0.65
MIN_FACE_PX   = 80
MIN_SHARPNESS = 60.0

# Crop geometry for saved face images
PADDING_RATIO = 0.30
SQUARE_CROP   = True
CROP_SIZE     = 256          # resize crops to this size (px)
JPEG_QUALITY  = 92

# Detection
DET_SIZE  = (1024, 1024)
PROVIDERS = ["CPUExecutionProvider"]   # CPU-only: avoids CoreML bug on Apple Silicon

# Clustering
# DBSCAN eps = max cosine distance for two faces to be considered the same person.
# Lower (0.3-0.4) = stricter, may split one person into multiple clusters
# Higher (0.5-0.6) = looser, may merge similar-looking different people
CLUSTER_EPS         = 0.40
CLUSTER_MIN_SAMPLES = 2      # need at least 2 faces to form a cluster; 1 = "unknown"

# Output
MAKE_MONTAGES   = True
MONTAGE_COLS    = 6
MONTAGE_TILE_PX = 160

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}

# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cluster_faces")


@dataclass
class FaceRecord:
    src: Path
    face_index: int          # which face within the source image
    score: float
    embedding: np.ndarray    # 512-d normalised vector
    crop: np.ndarray         # BGR image, ready to save
    cluster_id: int = -1     # filled in after clustering


def iter_images(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            yield p


def imread_unicode(path: Path) -> np.ndarray | None:
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is not None:
            return img
    except Exception:
        pass
    try:
        from PIL import Image
        import pillow_heif  # noqa: F401
        with Image.open(path) as im:
            return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
    except Exception:
        return None


def sharpness(bgr: np.ndarray) -> float:
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def square_pad_bbox(x1, y1, x2, y2, img_w, img_h, pad_ratio, square):
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    if square:
        side = max(w, h)
        half = side / 2 * (1 + pad_ratio)
        nx1, ny1, nx2, ny2 = cx - half, cy - half, cx + half, cy + half
    else:
        nx1 = x1 - w * pad_ratio
        ny1 = y1 - h * pad_ratio
        nx2 = x2 + w * pad_ratio
        ny2 = y2 + h * pad_ratio
    return (max(0, int(round(nx1))), max(0, int(round(ny1))),
            min(img_w, int(round(nx2))), min(img_h, int(round(ny2))))


def build_app():
    from insightface.app import FaceAnalysis
    log.info("Loading InsightFace (buffalo_l, detection + recognition)…")
    # allowed_modules=None means load everything, including the recognition (ArcFace) model
    app = FaceAnalysis(name="buffalo_l", providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=DET_SIZE, det_thresh=MIN_DET_SCORE * 0.8)
    return app


def extract_faces(images: list[Path], app) -> list[FaceRecord]:
    records: list[FaceRecord] = []
    for src in tqdm(images, desc="Detecting & embedding", unit="img"):
        try:
            img = imread_unicode(src)
            if img is None:
                continue
            H, W = img.shape[:2]
            faces = app.get(img)
            if not faces:
                continue

            for i, f in enumerate(faces):
                score = float(getattr(f, "det_score", 0.0))
                if score < MIN_DET_SCORE:
                    continue
                x1, y1, x2, y2 = [float(v) for v in f.bbox]
                if min(x2 - x1, y2 - y1) < MIN_FACE_PX:
                    continue

                nx1, ny1, nx2, ny2 = square_pad_bbox(x1, y1, x2, y2, W, H,
                                                     PADDING_RATIO, SQUARE_CROP)
                crop = img[ny1:ny2, nx1:nx2]
                if crop.size == 0 or sharpness(crop) < MIN_SHARPNESS:
                    continue

                # ArcFace embedding (512-d). InsightFace returns it on the face object.
                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    continue

                if CROP_SIZE:
                    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                                      interpolation=cv2.INTER_AREA)

                records.append(FaceRecord(
                    src=src, face_index=i, score=score,
                    embedding=np.asarray(emb, dtype=np.float32),
                    crop=crop,
                ))
        except Exception as e:  # noqa: BLE001
            log.warning("Skipped %s: %s", src.name, e)
    return records


def cluster_embeddings(records: list[FaceRecord]) -> int:
    """Cluster in-place; returns number of distinct clusters (excluding -1/noise)."""
    if not records:
        return 0
    embs = np.stack([r.embedding for r in records])
    log.info("Clustering %d face embeddings (DBSCAN eps=%.2f)…", len(embs), CLUSTER_EPS)
    db = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES, metric="cosine")
    labels = db.fit_predict(embs)
    for r, lbl in zip(records, labels):
        r.cluster_id = int(lbl)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    return n_clusters


def make_montage(crops: list[np.ndarray], cols: int, tile: int) -> np.ndarray:
    if not crops:
        return np.zeros((tile, tile, 3), dtype=np.uint8)
    rows = (len(crops) + cols - 1) // cols
    canvas = np.full((rows * tile, cols * tile, 3), 32, dtype=np.uint8)
    for i, c in enumerate(crops):
        r, k = divmod(i, cols)
        thumb = cv2.resize(c, (tile, tile), interpolation=cv2.INTER_AREA)
        canvas[r * tile:(r + 1) * tile, k * tile:(k + 1) * tile] = thumb
    return canvas


def write_outputs(records: list[FaceRecord]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Group by cluster_id
    by_cluster: dict[int, list[FaceRecord]] = {}
    for r in records:
        by_cluster.setdefault(r.cluster_id, []).append(r)

    # Sort: -1 last, then largest clusters first
    real_clusters = sorted(
        (cid for cid in by_cluster if cid != -1),
        key=lambda c: -len(by_cluster[c]),
    )

    # Map cluster_id -> person_NNN
    name_map: dict[int, str] = {}
    for new_idx, cid in enumerate(real_clusters, start=1):
        name_map[cid] = f"person_{new_idx:03d}"
    if -1 in by_cluster:
        name_map[-1] = "unknown"

    # Write crops
    for cid, group in by_cluster.items():
        folder_name = name_map[cid]
        out_dir = OUTPUT_DIR / folder_name
        out_dir.mkdir(parents=True, exist_ok=True)
        # Sort by detection score within each group (best face first)
        group.sort(key=lambda r: -r.score)
        for r in group:
            fname = f"{r.src.stem}__face{r.face_index}_{int(r.score * 100)}.jpg"
            cv2.imwrite(str(out_dir / fname), r.crop,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        # Montage (skip for "unknown" — it's just unrelated singletons)
        if MAKE_MONTAGES and cid != -1:
            montage = make_montage([r.crop for r in group],
                                   cols=MONTAGE_COLS, tile=MONTAGE_TILE_PX)
            cv2.imwrite(str(OUTPUT_DIR / f"{folder_name}_montage.jpg"), montage,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    # CSV manifest
    csv_path = OUTPUT_DIR / "_clusters.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "face_index", "score", "person"])
        for r in sorted(records, key=lambda r: (name_map[r.cluster_id], str(r.src))):
            w.writerow([str(r.src), r.face_index, f"{r.score:.3f}",
                        name_map[r.cluster_id]])
    log.info("Manifest: %s", csv_path)


def main() -> int:
    if not INPUT_DIR.exists():
        log.error("INPUT_DIR does not exist: %s", INPUT_DIR)
        return 2

    images = list(iter_images(INPUT_DIR))
    log.info("Found %d images under %s", len(images), INPUT_DIR)
    if not images:
        return 0

    app = build_app()
    records = extract_faces(images, app)
    log.info("Extracted %d qualifying faces from %d images.", len(records), len(images))
    if not records:
        log.warning("No faces qualified — nothing to cluster.")
        return 0

    n_clusters = cluster_embeddings(records)
    n_unknown = sum(1 for r in records if r.cluster_id == -1)
    log.info("Found %d people. %d faces did not cluster (unknown).",
             n_clusters, n_unknown)

    write_outputs(records)
    log.info("Done. Output: %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

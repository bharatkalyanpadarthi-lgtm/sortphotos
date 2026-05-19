"""
Face clustering v2: substantially better recall than v1.

Key improvements over cluster_faces.py:
  1. Stronger model: antelopev2 (glint360k ResNet-100) instead of buffalo_l
  2. Quality-weighted: blurry/small/profile faces contribute less to centroids
  3. Two-stage clustering:
       Stage A: strict DBSCAN finds confident "core" clusters
       Stage B: every unknown face is reassigned to its nearest centroid if close enough
  4. Centroid-based merging: clusters whose centroids are very close get merged
     (catches the "same person split into two folders" problem)
  5. Multi-pass: after merging, recompute centroids and reassign once more
  6. Per-cluster confidence scoring in the manifest

Run:
    python cluster_faces_v2.py
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
from sklearn.cluster import DBSCAN
from tqdm import tqdm

# =============================================================================
# CONFIG
# =============================================================================

INPUT_DIR  = Path.home() / "Pictures" / "To Process"
OUTPUT_DIR = Path.home() / "Pictures" / "face_clusters_v2"

# Detection / quality filters
MIN_DET_SCORE = 0.55          # slightly looser than v1 — we'll quality-weight later
MIN_FACE_PX   = 70
MIN_SHARPNESS = 40.0

# Crop geometry
PADDING_RATIO = 0.30
SQUARE_CROP   = True
CROP_SIZE     = 256
JPEG_QUALITY  = 92

# Detection
DET_SIZE  = (1024, 1024)
PROVIDERS = ["CPUExecutionProvider"]

# Model: "antelopev2" is stronger than "buffalo_l".
# First run will auto-download (~350 MB) into ~/.insightface/models/
MODEL_NAME = "antelopev2"

# --- Two-stage clustering ---------------------------------------------------
# Stage A (strict): only confident matches form initial clusters.
STAGE_A_EPS         = 0.32
STAGE_A_MIN_SAMPLES = 2

# Stage B (recovery): unknown faces join their nearest cluster centroid
# if cosine distance is below this. Looser than stage A.
STAGE_B_MAX_DIST = 0.50

# --- Post-hoc cluster merging ----------------------------------------------
# If two cluster centroids are closer than this, they're merged into one person.
MERGE_CENTROID_DIST = 0.38

# Output
MAKE_MONTAGES   = True
MONTAGE_COLS    = 6
MONTAGE_TILE_PX = 160

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}

# =============================================================================

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("cluster_faces_v2")


@dataclass
class FaceRecord:
    src: Path
    face_index: int
    det_score: float
    bbox_size: float          # min(w,h) of detected box
    sharpness: float
    yaw_proxy: float          # rough frontalness 0..1 (1 = head-on)
    embedding: np.ndarray     # 512-d L2-normalised
    crop: np.ndarray
    quality: float = 0.0      # combined weight 0..1
    cluster_id: int = -1


# ---------- IO helpers ------------------------------------------------------

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
        nx1, ny1 = x1 - w * pad_ratio, y1 - h * pad_ratio
        nx2, ny2 = x2 + w * pad_ratio, y2 + h * pad_ratio
    return (max(0, int(round(nx1))), max(0, int(round(ny1))),
            min(img_w, int(round(nx2))), min(img_h, int(round(ny2))))


def yaw_proxy_from_kps(kps: np.ndarray | None, bbox: np.ndarray) -> float:
    """
    Rough frontalness in [0..1]. 1.0 = perfectly head-on, 0 = full profile.
    Uses the offset of the nose tip from the horizontal midpoint between eyes,
    normalised by face width. Cheap and surprisingly effective.
    """
    if kps is None or len(kps) < 3:
        return 0.5
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    face_w = max(1.0, bbox[2] - bbox[0])
    offset = abs(nose[0] - eye_mid_x) / face_w
    # offset ~0 = frontal, ~0.25+ = strong profile
    return float(np.clip(1.0 - offset * 4.0, 0.0, 1.0))


def quality_score(rec: FaceRecord) -> float:
    """Combine signals into a single 0..1 weight used for centroid averaging."""
    s_det   = np.clip((rec.det_score - 0.4) / 0.55, 0.0, 1.0)         # 0.4..0.95
    s_size  = np.clip((rec.bbox_size - 60.0) / 240.0, 0.0, 1.0)       # 60..300 px
    s_sharp = np.clip(np.log1p(rec.sharpness) / np.log1p(400.0), 0.0, 1.0)
    s_front = rec.yaw_proxy
    # Geometric mean rewards balanced quality across all dims.
    parts = np.array([s_det, s_size, s_sharp, s_front]) + 1e-3
    return float(np.exp(np.log(parts).mean()))


# ---------- Detection + embedding ------------------------------------------

def build_app():
    from insightface.app import FaceAnalysis
    log.info("Loading InsightFace model: %s …", MODEL_NAME)
    app = FaceAnalysis(name=MODEL_NAME, providers=PROVIDERS)
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
                bbox = np.asarray(f.bbox, dtype=np.float32)
                x1, y1, x2, y2 = bbox.tolist()
                bw, bh = x2 - x1, y2 - y1
                if min(bw, bh) < MIN_FACE_PX:
                    continue

                nx1, ny1, nx2, ny2 = square_pad_bbox(x1, y1, x2, y2, W, H,
                                                     PADDING_RATIO, SQUARE_CROP)
                crop = img[ny1:ny2, nx1:nx2]
                if crop.size == 0:
                    continue
                sharp = sharpness(crop)
                if sharp < MIN_SHARPNESS:
                    continue

                emb = getattr(f, "normed_embedding", None)
                if emb is None:
                    continue

                kps = getattr(f, "kps", None)
                yaw = yaw_proxy_from_kps(kps, bbox)

                if CROP_SIZE:
                    crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                                      interpolation=cv2.INTER_AREA)

                rec = FaceRecord(
                    src=src, face_index=i, det_score=score,
                    bbox_size=float(min(bw, bh)), sharpness=sharp, yaw_proxy=yaw,
                    embedding=np.asarray(emb, dtype=np.float32), crop=crop,
                )
                rec.quality = quality_score(rec)
                records.append(rec)
        except Exception as e:  # noqa: BLE001
            log.warning("Skipped %s: %s", src.name, e)
    return records


# ---------- Clustering -----------------------------------------------------

def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def stage_a_dbscan(records: list[FaceRecord]) -> int:
    """Strict DBSCAN. Mutates records in place. Returns number of clusters."""
    embs = np.stack([r.embedding for r in records])
    db = DBSCAN(eps=STAGE_A_EPS, min_samples=STAGE_A_MIN_SAMPLES, metric="cosine")
    labels = db.fit_predict(embs)
    for r, lbl in zip(records, labels):
        r.cluster_id = int(lbl)
    return len(set(labels)) - (1 if -1 in labels else 0)


def compute_centroids(records: list[FaceRecord]) -> dict[int, np.ndarray]:
    """Quality-weighted, L2-normalised centroid per cluster (excl. -1)."""
    centroids: dict[int, np.ndarray] = {}
    by_id: dict[int, list[FaceRecord]] = {}
    for r in records:
        if r.cluster_id == -1:
            continue
        by_id.setdefault(r.cluster_id, []).append(r)
    for cid, group in by_id.items():
        embs = np.stack([r.embedding for r in group])
        weights = np.array([r.quality for r in group], dtype=np.float32)
        weights = weights / max(weights.sum(), 1e-9)
        c = (embs * weights[:, None]).sum(axis=0)
        centroids[cid] = _l2norm(c[None, :])[0]
    return centroids


def merge_close_clusters(records: list[FaceRecord]) -> int:
    """Merge clusters whose centroids are within MERGE_CENTROID_DIST. Iterative."""
    merged_total = 0
    while True:
        centroids = compute_centroids(records)
        ids = sorted(centroids.keys())
        if len(ids) < 2:
            break
        # Build distance matrix between centroids
        C = np.stack([centroids[i] for i in ids])
        sim = C @ C.T
        dist = 1.0 - sim
        np.fill_diagonal(dist, np.inf)
        i_min, j_min = np.unravel_index(np.argmin(dist), dist.shape)
        if dist[i_min, j_min] >= MERGE_CENTROID_DIST:
            break
        # Merge cluster ids[j_min] into ids[i_min]
        keep, drop = ids[i_min], ids[j_min]
        for r in records:
            if r.cluster_id == drop:
                r.cluster_id = keep
        merged_total += 1
    return merged_total


def stage_b_reassign(records: list[FaceRecord]) -> int:
    """Reassign 'unknown' faces to nearest centroid if within STAGE_B_MAX_DIST."""
    centroids = compute_centroids(records)
    if not centroids:
        return 0
    ids = sorted(centroids.keys())
    C = np.stack([centroids[i] for i in ids])
    reassigned = 0
    for r in records:
        if r.cluster_id != -1:
            continue
        sims = C @ r.embedding
        dists = 1.0 - sims
        j = int(np.argmin(dists))
        if dists[j] <= STAGE_B_MAX_DIST:
            r.cluster_id = ids[j]
            reassigned += 1
    return reassigned


# ---------- Output ---------------------------------------------------------

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


def write_outputs(records: list[FaceRecord], centroids: dict[int, np.ndarray]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    by_cluster: dict[int, list[FaceRecord]] = {}
    for r in records:
        by_cluster.setdefault(r.cluster_id, []).append(r)

    real_clusters = sorted(
        (cid for cid in by_cluster if cid != -1),
        key=lambda c: -len(by_cluster[c]),
    )
    name_map: dict[int, str] = {cid: f"person_{i:03d}"
                                for i, cid in enumerate(real_clusters, start=1)}
    if -1 in by_cluster:
        name_map[-1] = "unknown"

    # Write crops + per-face confidence (= 1 - distance to its centroid)
    for cid, group in by_cluster.items():
        out_dir = OUTPUT_DIR / name_map[cid]
        out_dir.mkdir(parents=True, exist_ok=True)
        group.sort(key=lambda r: -r.quality)
        for r in group:
            conf = 0.0
            if cid != -1 and cid in centroids:
                conf = float(centroids[cid] @ r.embedding)
            tag = f"q{int(r.quality*100):02d}_c{int(max(conf, 0)*100):02d}"
            fname = f"{r.src.stem}__face{r.face_index}_{tag}.jpg"
            cv2.imwrite(str(out_dir / fname), r.crop,
                        [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

        if MAKE_MONTAGES and cid != -1:
            montage = make_montage([r.crop for r in group],
                                   cols=MONTAGE_COLS, tile=MONTAGE_TILE_PX)
            cv2.imwrite(str(OUTPUT_DIR / f"{name_map[cid]}_montage.jpg"),
                        montage, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    # Manifest
    csv_path = OUTPUT_DIR / "_clusters.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "face_index", "det_score", "quality",
                    "person", "centroid_similarity"])
        for r in sorted(records, key=lambda r: (name_map[r.cluster_id], str(r.src))):
            sim = ""
            if r.cluster_id != -1 and r.cluster_id in centroids:
                sim = f"{float(centroids[r.cluster_id] @ r.embedding):.3f}"
            w.writerow([str(r.src), r.face_index, f"{r.det_score:.3f}",
                        f"{r.quality:.3f}", name_map[r.cluster_id], sim])
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
    log.info("Extracted %d qualifying faces from %d images.",
             len(records), len(images))
    if not records:
        return 0

    # Stage A
    n_a = stage_a_dbscan(records)
    n_unk_a = sum(1 for r in records if r.cluster_id == -1)
    log.info("Stage A (strict DBSCAN): %d clusters, %d unknown.", n_a, n_unk_a)

    # Merge near-duplicate clusters
    n_merged = merge_close_clusters(records)
    if n_merged:
        log.info("Merged %d near-duplicate cluster(s).", n_merged)

    # Stage B
    n_reassigned = stage_b_reassign(records)
    log.info("Stage B (centroid reassignment): recovered %d unknown faces.",
             n_reassigned)

    # One more merge pass — reassignment can bring previously-distant clusters closer
    n_merged2 = merge_close_clusters(records)
    if n_merged2:
        log.info("Final merge pass: combined %d more cluster(s).", n_merged2)

    centroids = compute_centroids(records)
    n_final = len(centroids)
    n_unk_final = sum(1 for r in records if r.cluster_id == -1)
    log.info("Final: %d people, %d unknown faces.", n_final, n_unk_final)

    write_outputs(records, centroids)
    log.info("Done. Output: %s", OUTPUT_DIR)
    return 0


if __name__ == "__main__":
    sys.exit(main())

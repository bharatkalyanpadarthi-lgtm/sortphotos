"""
sort_photos.py — Unified face-sorting pipeline with persistent memory.

Hardened version with:
  • Atomic originals copy with resume checkpoint
  • Detailed Claude API error reporting and user prompt on repeated failures
  • Resumable labeling sessions (saves every 10 labels)
  • Companion validate_cache.py for cache health audits

Auto-name suggestions via Claude Sonnet 4.6:
   For each unlabeled cluster, the script asks Claude to identify the person
   from the best face crop. The suggestion is pre-filled in the labeling
   prompt — press Enter to accept, or type a different name to override.

Detection runs in a SUBPROCESS PER BATCH so ONNX/native memory leaks can't
kill long runs. Default 500 images per batch.

Set up:
   export ANTHROPIC_API_KEY=sk-ant-...
   pip install anthropic

Run:
   python sort_photos.py             # normal run; auto-detects saved labeling
   python sort_photos.py --resume-label  # skip directly to resuming labeling
   python sort_photos.py --no-label
   python sort_photos.py --no-ai
   python sort_photos.py --no-dedup
   python sort_photos.py --no-review
   python sort_photos.py --reset-cache
   python sort_photos.py --batch-size 250
"""

from __future__ import annotations

import argparse
import base64
import csv
import gc
import json
import logging
import os
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_INPUT  = Path.home() / "Pictures"
DEFAULT_OUTPUT = Path.home() / "Pictures" / "sorted_all_pictures"
CACHE_DIR      = Path.home() / ".face_sort_cache"
CACHE_FILE     = CACHE_DIR / "cache.pkl"
AI_CACHE_FILE  = CACHE_DIR / "ai_suggestions.json"
LABEL_STATE_FILE = CACHE_DIR / "labeling_state.pkl"
IDENTITY_DB_FILE = CACHE_DIR / "person_identity_db.pkl"
REFERENCE_CENTROIDS_FILE = CACHE_DIR / "reference_centroids.pkl"
CACHE_VERSION  = 2
LABEL_STATE_VERSION = 1
IDENTITY_DB_VERSION = 1

BATCH_SIZE = 50
DETECT_WORKERS = 1
LABEL_SAVE_EVERY = 10   # save labeling state every N labels applied

MODEL_NAME = "antelopev2"
PROVIDERS  = ["CPUExecutionProvider"]
DET_SIZE   = (1024, 1024)

MIN_DET_SCORE = 0.55
MIN_FACE_PX   = 70
MIN_SHARPNESS = 40.0

CROP_SIZE     = 256
JPEG_QUALITY  = 92
PADDING_RATIO = 0.30

STAGE_A_EPS               = 0.32
STAGE_A_MIN_SAMPLES       = 2
STAGE_B_MAX_DIST          = 0.50
MERGE_CENTROID_DIST       = 0.42   # raised from 0.38 — auto-merges more pairs
                                    # without asking. Tuned for celebrity
                                    # collections where false merges are rare.
                                    # For family photos, lower back to 0.38.
ANCHOR_MAX_DIST           = 0.55
ANCHOR_CLUSTER_MERGE_DIST = 0.42
REVIEW_CLOSE_PAIRS_DIST   = 0.46   # lowered from 0.50 — fewer review prompts.

SHARPNESS_BLUR_THRESHOLD = 0.0

PHASH_THRESHOLD = 8
DUPLICATES_DIR  = "_duplicates"
BLURRED_DIR     = "_blurred"

INTERACTIVE_LABELING = True
DEDUP_DUPLICATES     = True
REVIEW_CLOSE_PAIRS   = True
USE_AI_SUGGESTIONS   = True
NUDITY_SORT_ENABLED  = True

MAKE_MONTAGES   = True
MONTAGE_COLS    = 6
MONTAGE_TILE_PX = 160
INCLUDE_UNKNOWN = True

NUDITY_THRESHOLD = 0.35
NUDITY_UNCERTAIN_THRESHOLD = 0.20
NUDITY_POSSIBLE_DIR = "_possible_nudity"
NUDITY_UNCERTAIN_DIR = "_uncertain_nudity"
NUDITY_EXPLICIT_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}
_NUDITY_DETECTOR = None
_NUDITY_IMPORT_WARNED = False

AUTO_PERSON_MATCH_ENABLED = True
AUTO_PERSON_MATCH_DIST = 0.40
AUTO_PERSON_MATCH_MARGIN = 0.04
IDENTITY_MAX_IMAGES_PER_PERSON = 10
POST_PROCESS_OUTPUT = True
USE_HARDLINKS = True
ARCHIVE_ORGANIZED_SOURCES = False
ARCHIVE_SCANNED_SOURCES = False
UNATTENDED_FINISH_KNOWN = False
SOURCE_ARCHIVE_DIR_NAME = "organized_sources"
SCANNED_SOURCE_ARCHIVE_DIR_NAME = "ready_to_delete/scanned_sources"

GOOGLE_LENS_URL = "https://lens.google.com/"
LENS_SEARCH_CROP_SIZE = 512

AI_MODEL = "claude-sonnet-4-6"
AI_MAX_TOKENS = 50
AI_TIMEOUT_SECONDS = 30
AI_MAX_RETRIES = 2
AI_FAILURE_PAUSE_THRESHOLD = 3
AI_PROMPT = (
    "This is a cropped face from a photograph. Look carefully at the face.\n"
    "\n"
    "If you recognize this person as a well-known public figure (actor, "
    "actress, politician, musician, athlete, or other widely-known celebrity), "
    "respond with ONLY their full name on a single line. Use the spelling most "
    "commonly used in English-language press.\n"
    "\n"
    "If you don't recognize the person, or if you're not highly confident, "
    "respond with exactly: UNKNOWN\n"
    "\n"
    "Respond with only the name or UNKNOWN. No other text, no qualifications, "
    "no 'I think', no parentheses, no notes."
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp",
              ".tif", ".tiff", ".heic", ".heif"}
INVALID_NAME_CHARS = '/\\:*?"<>|'
DEFAULT_EXCLUDED_SCAN_DIRS = {
    "_nudity_review",
    "_source_review",
    "duplicate_to_review",
    "Face References",
    "face_clusters",
    "junk_to_review",
    "photos_by_person",
    "ready_to_delete",
    "sorted_all_pictures",
    "videos",
}
ALWAYS_EXCLUDED_SCAN_DIRS = {
    "videos",
}

# ============================================================================

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("sort_photos")


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class FaceRecord:
    src: Path
    face_index: int
    det_score: float
    bbox_size: float
    sharpness: float
    yaw_proxy: float
    embedding: np.ndarray
    crop_jpeg: bytes = b""
    image_phash: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=bool))
    quality: float = 0.0
    cluster_id: int = -1
    prior_label: str | None = None
    _crop_array: np.ndarray | None = None

    def crop(self) -> np.ndarray:
        if self._crop_array is not None:
            return self._crop_array
        if not self.crop_jpeg:
            return np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        arr = cv2.imdecode(np.frombuffer(self.crop_jpeg, dtype=np.uint8),
                           cv2.IMREAD_COLOR)
        if arr is None:
            return np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        return arr

    def drop_decoded(self) -> None:
        self._crop_array = None


@dataclass
class CachedFace:
    src_str: str
    face_index: int
    det_score: float
    bbox_size: float
    sharpness: float
    yaw_proxy: float
    quality: float
    embedding: np.ndarray
    image_phash: np.ndarray
    crop_jpeg: bytes
    label: str | None = None


@dataclass
class CacheState:
    version: int = CACHE_VERSION
    config_fingerprint: str = ""
    file_signatures: dict[str, tuple[float, int]] = field(default_factory=dict)
    faces: list[CachedFace] = field(default_factory=list)


@dataclass
class LabelingState:
    """Snapshot of the post-clustering pipeline state, saved before interactive
    labeling begins and updated every N labels. Lets the user quit labeling
    and pick up exactly where they stopped."""
    version: int = LABEL_STATE_VERSION
    output_dir: str = ""
    input_dir: str = ""
    config_fingerprint: str = ""
    # All face records (each face = one cached entry + its current cluster_id).
    # We persist the cluster assignment so we don't need to re-cluster on resume.
    faces: list[CachedFace] = field(default_factory=list)
    cluster_ids: list[int] = field(default_factory=list)   # parallel to faces
    name_map: dict[int, str] = field(default_factory=dict)
    completed: bool = False


@dataclass
class IdentityDB:
    version: int = IDENTITY_DB_VERSION
    config_fingerprint: str = ""
    identities: dict[str, np.ndarray] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)


# ============================================================================
# IO + IMAGE HELPERS
# ============================================================================

def iter_images(root: Path,
                excluded_dir_names: set[str] | None = None) -> Iterable[Path]:
    excluded = DEFAULT_EXCLUDED_SCAN_DIRS if excluded_dir_names is None else excluded_dir_names
    excluded_casefold = {d.casefold() for d in excluded} | ALWAYS_EXCLUDED_SCAN_DIRS
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d.casefold() not in excluded_casefold and not d.startswith(".")
        ]
        base = Path(dirpath)
        for filename in filenames:
            p = base / filename
            if p.suffix.lower() in IMAGE_EXTS:
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


def square_pad_bbox(x1, y1, x2, y2, img_w, img_h, pad_ratio):
    w, h = x2 - x1, y2 - y1
    cx, cy = x1 + w / 2, y1 + h / 2
    side = max(w, h)
    half = side / 2 * (1 + pad_ratio)
    nx1, ny1, nx2, ny2 = cx - half, cy - half, cx + half, cy + half
    return (max(0, int(round(nx1))), max(0, int(round(ny1))),
            min(img_w, int(round(nx2))), min(img_h, int(round(ny2))))


def yaw_proxy_from_kps(kps, bbox: np.ndarray) -> float:
    if kps is None or len(kps) < 3:
        return 0.5
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_mid_x = (left_eye[0] + right_eye[0]) / 2.0
    face_w = max(1.0, bbox[2] - bbox[0])
    offset = abs(nose[0] - eye_mid_x) / face_w
    return float(np.clip(1.0 - offset * 4.0, 0.0, 1.0))


def quality_score_from_parts(det_score: float, bbox_size: float,
                              sharp: float, yaw: float) -> float:
    s_det   = np.clip((det_score - 0.4) / 0.55, 0.0, 1.0)
    s_size  = np.clip((bbox_size - 60.0) / 240.0, 0.0, 1.0)
    s_sharp = np.clip(np.log1p(sharp) / np.log1p(400.0), 0.0, 1.0)
    parts = np.array([s_det, s_size, s_sharp, yaw]) + 1e-3
    return float(np.exp(np.log(parts).mean()))


def perceptual_hash(bgr: np.ndarray, hash_size: int = 8) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY) if bgr.ndim == 3 else bgr
    resized = cv2.resize(gray, (hash_size * 4, hash_size * 4),
                         interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(resized)
    dct_low = dct[:hash_size, :hash_size].flatten()
    median = float(np.median(dct_low[1:]))
    return dct_low > median


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def sanitize_name(name: str) -> str:
    name = name.strip()
    for ch in INVALID_NAME_CHARS:
        name = name.replace(ch, "_")
    return name


def is_real_person_label(name: str | None) -> bool:
    return bool(name
                and name != "unknown"
                and name != "__junk__"
                and not name.startswith("person_"))


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


def unique_path(dest: Path) -> Path:
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


def filename_prefix_from_person(person: str) -> str:
    name = sanitize_name(person).strip()
    name = "_".join(name.split())
    name = "".join(ch if ch.isalnum() or ch in "._- " else "_" for ch in name)
    while "__" in name:
        name = name.replace("__", "_")
    return name.strip("._- ") or "person"


def next_numbered_dest(base_dir: Path,
                       person: str,
                       src: Path,
                       next_indexes: dict[Path, int]) -> Path:
    prefix = filename_prefix_from_person(person)
    if base_dir not in next_indexes:
        max_index = 0
        pattern_prefix = f"{prefix}_"
        if base_dir.exists():
            for existing in base_dir.iterdir():
                if not existing.is_file():
                    continue
                if existing.suffix.lower() not in IMAGE_EXTS:
                    continue
                stem = existing.stem
                if not stem.startswith(pattern_prefix):
                    continue
                suffix = stem[len(pattern_prefix):]
                if suffix.isdigit():
                    max_index = max(max_index, int(suffix))
        next_indexes[base_dir] = max_index + 1

    ext = src.suffix.lower() or ".jpg"
    while True:
        i = next_indexes[base_dir]
        next_indexes[base_dir] = i + 1
        candidate = base_dir / f"{prefix}_{i:03d}{ext}"
        if not candidate.exists():
            return candidate


def _get_nudity_detector():
    global _NUDITY_DETECTOR, _NUDITY_IMPORT_WARNED
    if not NUDITY_SORT_ENABLED:
        return None
    if _NUDITY_DETECTOR is not None:
        return _NUDITY_DETECTOR
    try:
        from nudenet import NudeDetector
    except ImportError:
        if not _NUDITY_IMPORT_WARNED:
            log.warning("NudeNet is not installed; nudity subfolder sorting skipped. "
                        "Install with: pip install --upgrade nudenet")
            _NUDITY_IMPORT_WARNED = True
        return None
    _NUDITY_DETECTOR = NudeDetector()
    return _NUDITY_DETECTOR


def _nudity_category(detections: list[dict]) -> tuple[str | None, str, float]:
    explicit = [d for d in detections if d.get("class") in NUDITY_EXPLICIT_CLASSES]
    if not explicit:
        return None, "", 0.0
    best = max(explicit, key=lambda d: float(d.get("score", 0.0)))
    best_class = str(best.get("class", ""))
    best_score = float(best.get("score", 0.0))
    if best_score >= NUDITY_THRESHOLD:
        return NUDITY_POSSIBLE_DIR, best_class, best_score
    if best_score >= NUDITY_UNCERTAIN_THRESHOLD:
        return NUDITY_UNCERTAIN_DIR, best_class, best_score
    return None, best_class, best_score


def maybe_move_to_nudity_subfolder(path: Path, person_dir: Path) -> tuple[Path, str | None]:
    detector = _get_nudity_detector()
    if detector is None or not path.exists():
        return path, None
    if NUDITY_POSSIBLE_DIR in path.parts or NUDITY_UNCERTAIN_DIR in path.parts:
        return path, None
    try:
        detections = detector.detect(str(path))
    except Exception as e:  # noqa: BLE001
        log.warning("Nudity check failed for %s: %s", path.name, e)
        return path, "error"
    subdir, _best_class, _best_score = _nudity_category(detections)
    if not subdir:
        return path, None
    dest = unique_dest(person_dir / subdir, path.name)
    dest.parent.mkdir(parents=True, exist_ok=True)
    path.rename(dest)
    return dest, subdir


# ============================================================================
# CACHE
# ============================================================================

def config_fingerprint() -> str:
    parts = [MODEL_NAME, str(DET_SIZE), str(MIN_DET_SCORE),
             str(MIN_FACE_PX), str(MIN_SHARPNESS), str(CROP_SIZE), str(PADDING_RATIO)]
    return "|".join(parts)


def file_signature(path: Path) -> tuple[float, int]:
    st = path.stat()
    return (float(st.st_mtime), int(st.st_size))


def load_cache() -> CacheState:
    if not CACHE_FILE.exists():
        return CacheState(config_fingerprint=config_fingerprint())
    try:
        with CACHE_FILE.open("rb") as f:
            data: CacheState = pickle.load(f)
        if data.version != CACHE_VERSION:
            log.warning("Cache version changed. Discarding cache.")
            return CacheState(config_fingerprint=config_fingerprint())
        if data.config_fingerprint != config_fingerprint():
            log.warning("Detection config changed. Discarding face cache, keeping labels.")
            preserved = [c for c in data.faces if c.label]
            data = CacheState(config_fingerprint=config_fingerprint(),
                              file_signatures={}, faces=preserved)
        return data
    except Exception as e:  # noqa: BLE001
        log.warning("Cache load failed (%s). Starting fresh.", e)
        return CacheState(config_fingerprint=config_fingerprint())


def load_identity_db() -> IdentityDB | None:
    if not IDENTITY_DB_FILE.exists():
        return None
    try:
        with IDENTITY_DB_FILE.open("rb") as f:
            db: IdentityDB = pickle.load(f)
        if db.version != IDENTITY_DB_VERSION:
            log.warning("Identity DB version changed. Rebuild it.")
            return None
        if db.config_fingerprint != config_fingerprint():
            log.warning("Identity DB model config changed. Rebuild it.")
            return None
        return db
    except Exception as e:  # noqa: BLE001
        log.warning("Could not load identity DB: %s", e)
        return None


def load_reference_centroids(path: Path) -> IdentityDB | None:
    path = path.expanduser()
    if not path.exists():
        return None
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
        if payload.get("model") != MODEL_NAME:
            log.warning("Reference DB model mismatch in %s. Rebuild references.", path)
            return None
        names = list(payload.get("names", []))
        centroids = np.asarray(payload.get("centroids"), dtype=np.float32)
        counts = list(payload.get("counts", [0] * len(names)))
        if len(names) == 0 or centroids.ndim != 2 or len(names) != len(centroids):
            log.warning("Reference DB is malformed: %s", path)
            return None
        db = IdentityDB(config_fingerprint=config_fingerprint())
        for i, name in enumerate(names):
            clean = str(name).strip()
            if not clean:
                continue
            db.identities[clean] = _l2norm(centroids[i:i + 1])[0]
            db.source_counts[clean] = int(counts[i]) if i < len(counts) else 0
        log.info("Loaded reference identity DB: %s (%d people)",
                 path, len(db.identities))
        return db
    except Exception as e:  # noqa: BLE001
        log.warning("Could not load reference DB %s: %s", path, e)
        return None


def merge_identity_dbs(primary: IdentityDB | None,
                       extra: IdentityDB | None) -> IdentityDB | None:
    if primary is None:
        return extra
    if extra is None:
        return primary
    merged = IdentityDB(config_fingerprint=config_fingerprint())
    merged.identities.update(primary.identities)
    merged.source_counts.update(primary.source_counts)
    added = 0
    for name, centroid in extra.identities.items():
        if name in merged.identities:
            continue
        merged.identities[name] = centroid
        merged.source_counts[name] = extra.source_counts.get(name, 0)
        added += 1
    if added:
        log.info("Added %d people from reference DB to matcher.", added)
    return merged


def save_identity_db(db: IdentityDB) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = IDENTITY_DB_FILE.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(IDENTITY_DB_FILE)


def build_identity_db_from_person_folders(people_dir: Path) -> IdentityDB:
    from tqdm import tqdm

    people_dir = people_dir.expanduser().resolve()
    db = load_identity_db()
    if db is None:
        db = IdentityDB(config_fingerprint=config_fingerprint())
    if not people_dir.exists():
        log.warning("Identity DB source folder does not exist: %s", people_dir)
        return db

    app = _build_app()
    person_dirs = sorted([p for p in people_dir.iterdir() if p.is_dir()],
                         key=lambda p: p.name.lower())
    for person_dir in tqdm(person_dirs, desc="Building identity DB", unit="person"):
        if person_dir.name.startswith("_") or not is_real_person_label(person_dir.name):
            continue
        if person_dir.name in db.identities:
            continue
        embs: list[np.ndarray] = []
        images = [
            p for p in iter_images(person_dir, excluded_dir_names=set())
            if not any(part in {DUPLICATES_DIR, BLURRED_DIR} for part in p.relative_to(person_dir).parts[:-1])
        ]
        images = sorted(images, key=lambda p: (len(p.relative_to(person_dir).parts), str(p).lower()))
        if IDENTITY_MAX_IMAGES_PER_PERSON > 0:
            images = images[:IDENTITY_MAX_IMAGES_PER_PERSON]
        for img in images:
            faces = _detect_one_image(img, app)
            if not faces:
                continue
            best = max(faces, key=lambda f: f.quality)
            embs.append(best.embedding)
        if embs:
            c = np.mean(np.stack(embs), axis=0)
            db.identities[person_dir.name] = _l2norm(c[None, :])[0]
            db.source_counts[person_dir.name] = len(embs)
            if len(db.identities) % 5 == 0:
                save_identity_db(db)

    save_identity_db(db)
    log.info("Identity DB saved: %s (%d people)", IDENTITY_DB_FILE, len(db.identities))
    return db


def save_cache(cache: CacheState) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(CACHE_FILE)


def cached_to_record(c: CachedFace) -> FaceRecord:
    return FaceRecord(
        src=Path(c.src_str), face_index=c.face_index, det_score=c.det_score,
        bbox_size=c.bbox_size, sharpness=c.sharpness, yaw_proxy=c.yaw_proxy,
        embedding=c.embedding, crop_jpeg=c.crop_jpeg, image_phash=c.image_phash,
        quality=c.quality, prior_label=c.label,
    )


def record_to_cached(rec: FaceRecord, label: str | None) -> CachedFace:
    return CachedFace(
        src_str=str(rec.src), face_index=rec.face_index, det_score=rec.det_score,
        bbox_size=rec.bbox_size, sharpness=rec.sharpness, yaw_proxy=rec.yaw_proxy,
        quality=rec.quality, embedding=rec.embedding, image_phash=rec.image_phash,
        crop_jpeg=rec.crop_jpeg, label=label,
    )


# ============================================================================
# LABELING STATE (resumable session)
# ============================================================================

def save_labeling_state(records: list[FaceRecord],
                        name_map: dict[int, str],
                        output_dir: Path,
                        input_dir: Path,
                        completed: bool = False) -> None:
    """Snapshot the current cluster→name mapping plus all face records, so
    a future invocation can resume labeling without re-running detection or
    clustering."""
    state = LabelingState(
        version=LABEL_STATE_VERSION,
        output_dir=str(output_dir),
        input_dir=str(input_dir),
        config_fingerprint=config_fingerprint(),
        faces=[record_to_cached(r, label=None) for r in records],
        cluster_ids=[r.cluster_id for r in records],
        name_map=dict(name_map),
        completed=completed,
    )
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LABEL_STATE_FILE.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(state, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(LABEL_STATE_FILE)


def load_labeling_state() -> LabelingState | None:
    if not LABEL_STATE_FILE.exists():
        return None
    try:
        with LABEL_STATE_FILE.open("rb") as f:
            state: LabelingState = pickle.load(f)
        if state.version != LABEL_STATE_VERSION:
            return None
        return state
    except Exception as e:  # noqa: BLE001
        log.warning("Labeling state load failed (%s).", e)
        return None


def clear_labeling_state() -> None:
    if LABEL_STATE_FILE.exists():
        try:
            LABEL_STATE_FILE.unlink()
        except OSError:
            pass


def backup_labeling_state(reason: str) -> Path | None:
    """Keep a copy before discarding an old interactive session."""
    if not LABEL_STATE_FILE.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    safe_reason = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_"
                          for ch in reason).strip("_") or "backup"
    backup = LABEL_STATE_FILE.with_name(
        f"{LABEL_STATE_FILE.stem}.{safe_reason}.{stamp}{LABEL_STATE_FILE.suffix}")
    try:
        shutil.copy2(LABEL_STATE_FILE, backup)
        return backup
    except OSError as exc:
        log.warning("Could not back up labeling state before discard: %s", exc)
        return None


def labeling_state_summary(state: LabelingState) -> dict:
    """How many clusters, how many already labeled, how many still need work."""
    cid_set = set(state.cluster_ids)
    cid_set.discard(-1)
    n_clusters = len(cid_set)
    n_labeled = sum(1 for cid, n in state.name_map.items()
                     if cid != -1 and not n.startswith("person_"))
    n_remaining = sum(1 for cid, n in state.name_map.items()
                       if cid != -1 and n.startswith("person_"))
    return {
        "n_clusters": n_clusters,
        "n_labeled": n_labeled,
        "n_remaining": n_remaining,
        "total_faces": len(state.faces),
        "input_dir": state.input_dir,
        "output_dir": state.output_dir,
        "completed": state.completed,
    }


# ============================================================================
# AI SUGGESTION CACHE
# ============================================================================

def load_ai_cache() -> dict[str, str]:
    if not AI_CACHE_FILE.exists():
        return {}
    try:
        with AI_CACHE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:  # noqa: BLE001
        return {}


def save_ai_cache(cache: dict[str, str]) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = AI_CACHE_FILE.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2, sort_keys=True)
    tmp.replace(AI_CACHE_FILE)


def crop_content_hash(crop_jpeg: bytes) -> str:
    import hashlib
    return hashlib.sha1(crop_jpeg).hexdigest()


# ============================================================================
# CLAUDE AI SUGGESTION
# ============================================================================

class AIError(Exception):
    REASONS = {
        "no_key": "ANTHROPIC_API_KEY environment variable is empty.",
        "no_package": "Python package 'anthropic' is not installed (pip install anthropic).",
        "auth": "API key is invalid or has been revoked.",
        "rate_limit": "API rate limit exceeded. Wait a minute and try again.",
        "credits": "Insufficient API credits. Add credits at console.anthropic.com.",
        "timeout": "API call timed out.",
        "network": "Network error reaching Anthropic API.",
        "bad_response": "API returned an unexpected response.",
        "unknown": "Unexpected API error.",
    }

    def __init__(self, reason: str, detail: str = ""):
        self.reason = reason
        self.detail = detail
        msg = self.REASONS.get(reason, reason)
        if detail:
            msg = f"{msg}  ({detail})"
        super().__init__(msg)


def _classify_anthropic_error(exc: Exception) -> str:
    name = type(exc).__name__
    msg = str(exc).lower()
    if "authentication" in name.lower() or "auth" in msg or "401" in msg:
        return "auth"
    if "ratelimit" in name.lower() or "rate_limit" in msg or "429" in msg:
        return "rate_limit"
    if "credit" in msg or "balance" in msg or "402" in msg or "insufficient" in msg:
        return "credits"
    if "timeout" in name.lower() or "timeout" in msg:
        return "timeout"
    if "connection" in msg or "network" in msg or "dns" in msg:
        return "network"
    return "unknown"


def claude_suggest_name(crop_jpeg: bytes) -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise AIError("no_key")
    try:
        import anthropic  # type: ignore
    except ImportError:
        raise AIError("no_package")

    last_exc: Exception | None = None
    for attempt in range(AI_MAX_RETRIES + 1):
        try:
            client = anthropic.Anthropic(api_key=api_key, timeout=AI_TIMEOUT_SECONDS)
            b64 = base64.standard_b64encode(crop_jpeg).decode("ascii")
            message = client.messages.create(
                model=AI_MODEL,
                max_tokens=AI_MAX_TOKENS,
                messages=[{
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64,
                            },
                        },
                        {"type": "text", "text": AI_PROMPT},
                    ],
                }],
            )
            if not message.content:
                raise AIError("bad_response", "empty content")
            parts = [b.text for b in message.content if getattr(b, "type", "") == "text"]
            text = " ".join(parts).strip()
            if not text:
                raise AIError("bad_response", "no text in response")
            text = text.splitlines()[0].strip().strip('".\'')
            if text.upper() == "UNKNOWN":
                return None
            if len(text) > 80:
                raise AIError("bad_response", f"name too long: {text[:40]}…")
            return text
        except AIError:
            raise
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            reason = _classify_anthropic_error(exc)
            if reason in ("auth", "credits", "no_key", "no_package"):
                raise AIError(reason, str(exc)[:200])
            if attempt < AI_MAX_RETRIES:
                time.sleep(1.5 * (attempt + 1))
                continue
            raise AIError(reason, str(exc)[:200])

    raise AIError("unknown", str(last_exc) if last_exc else "")


# ============================================================================
# DETECTION WORKER (subprocess)
# ============================================================================

def _build_app():
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name=MODEL_NAME, providers=PROVIDERS)
    app.prepare(ctx_id=0, det_size=DET_SIZE, det_thresh=MIN_DET_SCORE * 0.8)
    return app


def _detect_one_image(src: Path, app) -> list[CachedFace]:
    out: list[CachedFace] = []
    try:
        img = imread_unicode(src)
        if img is None:
            return out
        H, W = img.shape[:2]
        img_phash = perceptual_hash(img)
        faces = app.get(img)
        if not faces:
            return out
        for i, f in enumerate(faces):
            score = float(getattr(f, "det_score", 0.0))
            if score < MIN_DET_SCORE:
                continue
            bbox = np.asarray(f.bbox, dtype=np.float32)
            x1, y1, x2, y2 = bbox.tolist()
            bw, bh = x2 - x1, y2 - y1
            if min(bw, bh) < MIN_FACE_PX:
                continue
            nx1, ny1, nx2, ny2 = square_pad_bbox(x1, y1, x2, y2, W, H, PADDING_RATIO)
            crop = img[ny1:ny2, nx1:nx2]
            if crop.size == 0:
                continue
            sharp = sharpness(crop)
            if sharp < MIN_SHARPNESS:
                continue
            emb = getattr(f, "normed_embedding", None)
            if emb is None:
                continue
            yaw = yaw_proxy_from_kps(getattr(f, "kps", None), bbox)
            if CROP_SIZE:
                crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE),
                                  interpolation=cv2.INTER_AREA)
            ok, jpeg = cv2.imencode(".jpg", crop,
                                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            if not ok:
                continue
            q = quality_score_from_parts(score, float(min(bw, bh)), sharp, yaw)
            out.append(CachedFace(
                src_str=str(src), face_index=i, det_score=score,
                bbox_size=float(min(bw, bh)), sharpness=sharp, yaw_proxy=yaw,
                quality=q, embedding=np.asarray(emb, dtype=np.float32),
                image_phash=img_phash, crop_jpeg=jpeg.tobytes(), label=None,
            ))
    except Exception as e:  # noqa: BLE001
        sys.stderr.write(f"Skipped {src.name}: {e}\n")
    return out


def run_detection_worker(job_path: Path) -> int:
    global DET_SIZE
    with job_path.open("rb") as f:
        job = pickle.load(f)
    input_paths: list[str] = job["input_paths"]
    output_path: Path = Path(job["output_path"])
    if "det_size" in job:
        det = int(job["det_size"])
        DET_SIZE = (det, det)

    from tqdm import tqdm
    app = _build_app()
    all_faces: list[CachedFace] = []
    for s in tqdm(input_paths, desc="Worker", unit="img"):
        faces = _detect_one_image(Path(s), app)
        all_faces.extend(faces)

    with output_path.open("wb") as f:
        pickle.dump(all_faces, f, protocol=pickle.HIGHEST_PROTOCOL)
    return 0


def detect_in_batches_subprocess(new_images: list[Path],
                                 cache: CacheState,
                                 batch_size: int,
                                 workers: int = 1) -> list[FaceRecord]:
    all_new_records: list[FaceRecord] = []
    total = len(new_images)
    if total == 0:
        return all_new_records

    n_batches = (total + batch_size - 1) // batch_size
    workers = max(1, int(workers))
    log.info("Detection plan: %d image(s) in %d subprocess batch(es) of up to %d "
             "(workers=%d).", total, n_batches, batch_size, workers)

    tmp_dir = Path(tempfile.mkdtemp(prefix="sort_photos_"))
    log.info("Worker scratch dir: %s", tmp_dir)

    try:
        script_path = Path(__file__).resolve()
        jobs: list[tuple[int, int, int, Path, Path, list[Path]]] = []
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = min(start + batch_size, total)
            batch = new_images[start:end]
            job_path = tmp_dir / f"job_{batch_idx:04d}.pkl"
            out_path = tmp_dir / f"out_{batch_idx:04d}.pkl"
            with job_path.open("wb") as f:
                pickle.dump({
                    "input_paths": [str(p) for p in batch],
                    "output_path": str(out_path),
                    "det_size": DET_SIZE[0],
                }, f, protocol=pickle.HIGHEST_PROTOCOL)
            jobs.append((batch_idx, start, end, job_path, out_path, batch))

        if workers == 1:
            for batch_idx, start, end, job_path, out_path, batch in jobs:
                log.info("--- Batch %d/%d: images %d–%d (subprocess) ---",
                         batch_idx + 1, n_batches, start + 1, end)

                cmd = [sys.executable, str(script_path),
                       "--detect-batch", str(job_path)]

                try:
                    proc = subprocess.run(cmd, check=False)
                except KeyboardInterrupt:
                    log.warning("Interrupted. Cache up to last completed batch is saved.")
                    return all_new_records

                if proc.returncode != 0:
                    log.error("Worker for batch %d exited with code %d.",
                              batch_idx + 1, proc.returncode)
                    return all_new_records

                if not out_path.exists():
                    log.error("Worker for batch %d produced no output.", batch_idx + 1)
                    return all_new_records

                with out_path.open("rb") as f:
                    batch_faces: list[CachedFace] = pickle.load(f)

                for src in batch:
                    try:
                        cache.file_signatures[str(src)] = file_signature(src)
                    except OSError:
                        pass
                cache.faces.extend(batch_faces)
                save_cache(cache)

                log.info("Batch %d/%d: %d new faces. Cache saved (%d files, %d faces).",
                         batch_idx + 1, n_batches, len(batch_faces),
                         len(cache.file_signatures), len(cache.faces))

                for f in batch_faces:
                    all_new_records.append(cached_to_record(f))

                try:
                    job_path.unlink()
                    out_path.unlink()
                except OSError:
                    pass

                gc.collect()
        else:
            active: dict[subprocess.Popen, tuple[int, int, int, Path, Path, list[Path]]] = {}
            pending = list(jobs)
            completed = 0
            try:
                while pending or active:
                    while pending and len(active) < workers:
                        job_info = pending.pop(0)
                        batch_idx, start, end, job_path, _out_path, _batch = job_info
                        log.info("--- Batch %d/%d: images %d–%d (subprocess) ---",
                                 batch_idx + 1, n_batches, start + 1, end)
                        cmd = [sys.executable, str(script_path),
                               "--detect-batch", str(job_path)]
                        active[subprocess.Popen(cmd)] = job_info

                    time.sleep(0.5)
                    for proc, job_info in list(active.items()):
                        if proc.poll() is None:
                            continue
                        active.pop(proc)
                        batch_idx, _start, _end, job_path, out_path, batch = job_info
                        if proc.returncode != 0:
                            log.error("Worker for batch %d exited with code %d.",
                                      batch_idx + 1, proc.returncode)
                            for p in active:
                                p.terminate()
                            return all_new_records
                        if not out_path.exists():
                            log.error("Worker for batch %d produced no output.",
                                      batch_idx + 1)
                            for p in active:
                                p.terminate()
                            return all_new_records

                        with out_path.open("rb") as f:
                            batch_faces: list[CachedFace] = pickle.load(f)

                        for src in batch:
                            try:
                                cache.file_signatures[str(src)] = file_signature(src)
                            except OSError:
                                pass
                        cache.faces.extend(batch_faces)
                        save_cache(cache)

                        completed += 1
                        log.info("Batch %d/%d complete (%d/%d finished): %d new faces. "
                                 "Cache saved (%d files, %d faces).",
                                 batch_idx + 1, n_batches, completed, n_batches,
                                 len(batch_faces), len(cache.file_signatures),
                                 len(cache.faces))

                        for f in batch_faces:
                            all_new_records.append(cached_to_record(f))

                        try:
                            job_path.unlink()
                            out_path.unlink()
                        except OSError:
                            pass
                        gc.collect()
            except KeyboardInterrupt:
                log.warning("Interrupted. Terminating active workers; cache up to last "
                            "completed batch is saved.")
                for proc in active:
                    proc.terminate()
                return all_new_records
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            pass

    return all_new_records


# ============================================================================
# GOOGLE LENS LOOKUP
# ============================================================================

def copy_image_to_clipboard_macos(image_path: Path) -> bool:
    if sys.platform != "darwin":
        return False
    if not image_path.exists():
        return False
    script = (
        f'set the clipboard to (read (POSIX file "{image_path}") as JPEG picture)'
    )
    try:
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=10,
        )
        return result.returncode == 0
    except Exception:  # noqa: BLE001
        return False


def lens_lookup(records_for_cluster: list[FaceRecord]) -> None:
    if not records_for_cluster:
        print("  No face data available for this cluster.")
        return
    best = max(records_for_cluster, key=lambda r: r.quality)
    crop = best.crop()
    if crop.shape[0] < LENS_SEARCH_CROP_SIZE:
        crop = cv2.resize(crop, (LENS_SEARCH_CROP_SIZE, LENS_SEARCH_CROP_SIZE),
                          interpolation=cv2.INTER_CUBIC)
    tmp_dir = Path(tempfile.gettempdir()) / "sort_photos_lens"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    tmp_path = tmp_dir / "lens_lookup.jpg"
    cv2.imwrite(str(tmp_path), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
    best.drop_decoded()
    copied = copy_image_to_clipboard_macos(tmp_path)
    webbrowser.open(GOOGLE_LENS_URL)
    if copied:
        print("  → Google Lens opened. Click the upload area and press Cmd+V to paste.")
    else:
        print(f"  → Google Lens opened. Drag this file in: {tmp_path}")


# ============================================================================
# CLUSTERING
# ============================================================================

def _l2norm(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / np.maximum(n, 1e-9)


def stage_a_dbscan(records: list[FaceRecord]) -> int:
    from sklearn.cluster import DBSCAN
    embs = np.stack([r.embedding for r in records])
    db = DBSCAN(eps=STAGE_A_EPS, min_samples=STAGE_A_MIN_SAMPLES, metric="cosine")
    labels = db.fit_predict(embs)
    for r, lbl in zip(records, labels):
        r.cluster_id = int(lbl)
    return len(set(labels)) - (1 if -1 in labels else 0)


def auto_merge_by_prior_labels(records: list[FaceRecord]) -> int:
    label_to_clusters: dict[str, set[int]] = defaultdict(set)
    for r in records:
        if r.prior_label and r.cluster_id != -1:
            label_to_clusters[r.prior_label].add(r.cluster_id)
    n_merged = 0
    for cids in label_to_clusters.values():
        if len(cids) <= 1:
            continue
        cids_sorted = sorted(cids)
        keep = cids_sorted[0]
        for drop in cids_sorted[1:]:
            for r in records:
                if r.cluster_id == drop:
                    r.cluster_id = keep
            n_merged += 1
    return n_merged


def compute_centroids(records: list[FaceRecord]) -> dict[int, np.ndarray]:
    by_id: dict[int, list[FaceRecord]] = defaultdict(list)
    for r in records:
        if r.cluster_id != -1:
            by_id[r.cluster_id].append(r)
    centroids: dict[int, np.ndarray] = {}
    for cid, group in by_id.items():
        embs = np.stack([r.embedding for r in group])
        weights = np.array([r.quality for r in group], dtype=np.float32)
        weights = weights / max(weights.sum(), 1e-9)
        c = (embs * weights[:, None]).sum(axis=0)
        centroids[cid] = _l2norm(c[None, :])[0]
    return centroids


def merge_close_clusters(records: list[FaceRecord]) -> int:
    merged_total = 0
    while True:
        centroids = compute_centroids(records)
        ids = sorted(centroids.keys())
        if len(ids) < 2:
            break
        C = np.stack([centroids[i] for i in ids])
        dist = 1.0 - C @ C.T
        np.fill_diagonal(dist, np.inf)
        i_min, j_min = np.unravel_index(np.argmin(dist), dist.shape)
        if dist[i_min, j_min] >= MERGE_CENTROID_DIST:
            break
        keep, drop = ids[i_min], ids[j_min]
        for r in records:
            if r.cluster_id == drop:
                r.cluster_id = keep
        merged_total += 1
    return merged_total


def stage_b_reassign(records: list[FaceRecord]) -> int:
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
        j = int(np.argmin(1.0 - sims))
        if (1.0 - sims[j]) <= STAGE_B_MAX_DIST:
            r.cluster_id = ids[j]
            reassigned += 1
    return reassigned


def anchor_pass(records: list[FaceRecord]) -> int:
    anchored = [(r, r.prior_label) for r in records if r.prior_label and r.cluster_id != -1]
    if not anchored:
        return 0
    anchor_embs = np.stack([r.embedding for r, _ in anchored])
    anchor_cids = [r.cluster_id for r, _ in anchored]
    n_reassigned = 0
    for r in records:
        if r.cluster_id != -1:
            continue
        sims = anchor_embs @ r.embedding
        j = int(np.argmax(sims))
        if (1.0 - sims[j]) <= ANCHOR_MAX_DIST:
            r.cluster_id = anchor_cids[j]
            n_reassigned += 1
    return n_reassigned


def make_initial_name_map(records: list[FaceRecord]) -> dict[int, str]:
    by_id: dict[int, list[FaceRecord]] = defaultdict(list)
    for r in records:
        by_id[r.cluster_id].append(r)
    name_map: dict[int, str] = {}
    used_names: set[str] = set()
    for cid, group in by_id.items():
        if cid == -1:
            continue
        labels = [r.prior_label for r in group if r.prior_label]
        if labels:
            top = Counter(labels).most_common(1)[0][0]
            if top not in used_names:
                name_map[cid] = top
                used_names.add(top)
    remaining = [cid for cid in by_id if cid not in name_map and cid != -1]
    remaining.sort(key=lambda c: -len(by_id[c]))
    counter = 1
    for cid in remaining:
        while f"person_{counter:03d}" in used_names:
            counter += 1
        name = f"person_{counter:03d}"
        name_map[cid] = name
        used_names.add(name)
        counter += 1
    if -1 in by_id:
        name_map[-1] = "unknown"
    return name_map


def apply_identity_db_labels(records: list[FaceRecord],
                             name_map: dict[int, str],
                             identity_db: IdentityDB | None) -> int:
    if not AUTO_PERSON_MATCH_ENABLED or identity_db is None or not identity_db.identities:
        return 0
    centroids = compute_centroids(records)
    names = sorted(identity_db.identities.keys())
    identity_C = np.stack([identity_db.identities[n] for n in names])
    assigned = 0
    used_real = {
        n for n in name_map.values()
        if is_real_person_label(n) and not n.startswith("person_")
    }
    for cid, current_name in sorted(name_map.items(), key=lambda kv: kv[0]):
        if cid == -1 or not current_name.startswith("person_") or cid not in centroids:
            continue
        sims = identity_C @ centroids[cid]
        order = np.argsort(-sims)
        best_idx = int(order[0])
        best_dist = 1.0 - float(sims[best_idx])
        second_dist = 1.0 - float(sims[int(order[1])]) if len(order) > 1 else 1.0
        matched_name = names[best_idx]
        if matched_name in used_real:
            continue
        if best_dist <= AUTO_PERSON_MATCH_DIST and (second_dist - best_dist) >= AUTO_PERSON_MATCH_MARGIN:
            log.info("Existing-person match: %s -> %s (dist %.3f, margin %.3f)",
                     current_name, matched_name, best_dist, second_dist - best_dist)
            name_map[cid] = matched_name
            used_real.add(matched_name)
            assigned += 1
    return assigned


# ============================================================================
# CLUSTER MONTAGES
# ============================================================================

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


def write_cluster_crops(records: list[FaceRecord],
                        name_map: dict[int, str],
                        clusters_dir: Path,
                        centroids: dict[int, np.ndarray]) -> None:
    clusters_dir.mkdir(parents=True, exist_ok=True)
    by_cluster: dict[int, list[FaceRecord]] = defaultdict(list)
    for r in records:
        by_cluster[r.cluster_id].append(r)
    for cid, group in by_cluster.items():
        out_dir = clusters_dir / name_map[cid]
        out_dir.mkdir(parents=True, exist_ok=True)
        group.sort(key=lambda r: -r.quality)
        for r in group:
            conf = float(centroids[cid] @ r.embedding) if cid in centroids else 0.0
            tag = f"q{int(r.quality * 100):02d}_c{int(max(conf, 0) * 100):02d}"
            fname = f"{r.src.stem}__face{r.face_index}_{tag}.jpg"
            (out_dir / fname).write_bytes(r.crop_jpeg)
        if MAKE_MONTAGES and cid != -1:
            crops = []
            for r in group:
                arr = r.crop()
                thumb = cv2.resize(arr, (MONTAGE_TILE_PX, MONTAGE_TILE_PX),
                                   interpolation=cv2.INTER_AREA)
                crops.append(thumb)
                r.drop_decoded()
            montage = make_montage(crops, cols=MONTAGE_COLS, tile=MONTAGE_TILE_PX)
            cv2.imwrite(str(clusters_dir / f"{name_map[cid]}_montage.jpg"),
                        montage, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            del crops, montage
        gc.collect()


# ============================================================================
# DISK MERGE
# ============================================================================

def merge_clusters_on_disk(records: list[FaceRecord],
                            name_map: dict[int, str],
                            clusters_dir: Path,
                            keep_cid: int,
                            drop_cid: int) -> None:
    keep_name = name_map[keep_cid]
    drop_name = name_map[drop_cid]
    src_folder = clusters_dir / drop_name
    dst_folder = clusters_dir / keep_name
    if src_folder.exists():
        dst_folder.mkdir(parents=True, exist_ok=True)
        for f in src_folder.iterdir():
            if f.is_file():
                dest = dst_folder / f.name
                i = 2
                while dest.exists():
                    dest = dst_folder / f"{f.stem}__{i}{f.suffix}"
                    i += 1
                shutil.move(str(f), str(dest))
        try:
            src_folder.rmdir()
        except OSError:
            pass
    drop_montage = clusters_dir / f"{drop_name}_montage.jpg"
    if drop_montage.exists():
        drop_montage.unlink()
    for r in records:
        if r.cluster_id == drop_cid:
            r.cluster_id = keep_cid
    if drop_cid in name_map:
        del name_map[drop_cid]


# ============================================================================
# INTERACTIVE LABELING (resumable, saves every LABEL_SAVE_EVERY labels)
# ============================================================================

def _print_ai_error_help(err: AIError) -> None:
    print()
    print("  ⚠️  Claude AI suggestion failed:")
    print(f"     {err}")
    print()
    print("  Hints by error type:")
    if err.reason == "auth":
        print("     → Check that ANTHROPIC_API_KEY is set to a valid key")
        print("     → Get a new key at: https://console.anthropic.com/account/keys")
    elif err.reason == "credits":
        print("     → Add credits at: https://console.anthropic.com/billing")
    elif err.reason == "rate_limit":
        print("     → Wait 60 seconds and try again")
    elif err.reason == "no_key":
        print("     → Run: export ANTHROPIC_API_KEY=sk-ant-...")
    elif err.reason == "no_package":
        print("     → Run: pip install anthropic")
    elif err.reason in ("network", "timeout"):
        print("     → Check your internet connection")
    print()


def _ai_failure_dialog(err: AIError, n_failures: int) -> str:
    print()
    print(f"  ⚠️  {n_failures} consecutive Claude failures. Most recent:")
    print(f"     {err}")
    print()
    print("  Options:")
    print("    [r] Retry once more (maybe transient)")
    print("    [d] Disable AI suggestions for the rest of this session")
    print("    [q] Quit labeling now (state saved, can resume later)")
    print()
    while True:
        try:
            ans = input("  Choose [r/d/q]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "quit"
        if ans in ("r", ""):
            return "continue"
        if ans == "d":
            return "disable"
        if ans == "q":
            return "quit"


def interactive_label(records: list[FaceRecord],
                      name_map: dict[int, str],
                      clusters_dir: Path,
                      use_ai: bool,
                      output_dir: Path,
                      input_dir: Path,
                      min_cluster_size: int = 1) -> bool:
    """Interactive labeling loop. Saves labeling state every LABEL_SAVE_EVERY
    labels so 'q' or a crash never loses more than 10 labels of work.

    Returns True if all current clusters were visited, or False if the user
    quit/interrupted and the saved labeling state should be preserved.
    """
    by_id: dict[int, list[FaceRecord]] = defaultdict(list)
    for r in records:
        by_id[r.cluster_id].append(r)

    needs_label_all = [cid for cid, n in name_map.items()
                       if cid != -1 and n.startswith("person_")]
    needs_label = [cid for cid in needs_label_all
                   if len(by_id[cid]) >= min_cluster_size]
    needs_label.sort(key=lambda c: -len(by_id[c]))
    auto_labeled = sum(1 for cid, n in name_map.items()
                       if cid != -1 and not n.startswith("person_"))
    total = len(needs_label)
    skipped_small = len(needs_label_all) - len(needs_label)

    if auto_labeled:
        log.info("Auto-recognized %d cluster(s) from previous labels.", auto_labeled)
    if skipped_small:
        log.info("Skipping %d small unlabeled cluster(s) below --min-label-cluster-size=%d.",
                 skipped_small, min_cluster_size)
    if total == 0:
        if needs_label_all:
            log.info("No unlabeled clusters meet the minimum size. "
                     "Finalizing labeled folders and preserving resume state.")
            save_labeling_state(records, name_map, output_dir, input_dir)
            return False
        log.info("Nothing new to label.")
        return True

    ai_cache = load_ai_cache() if use_ai else {}
    ai_active = use_ai
    consecutive_failures = 0

    print(f"\n=== Labeling {total} new cluster(s) ===")
    if skipped_small:
        print(f"  Skipping {skipped_small} small cluster(s) below "
              f"{min_cluster_size} faces.")
    if ai_active:
        try:
            import anthropic  # noqa: F401
            api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
            if api_key:
                print(f"  AI suggestions: ON (Claude Sonnet 4.6)")
            else:
                print(f"  AI suggestions: OFF (no ANTHROPIC_API_KEY set)")
                ai_active = False
        except ImportError:
            print(f"  AI suggestions: OFF (anthropic not installed: pip install anthropic)")
            ai_active = False
    else:
        print(f"  AI suggestions: OFF")
    print(f"For each cluster, the montage opens in Preview.")
    print(f"  Press Enter to accept the suggestion (if any)")
    print(f"  Type a name and press Enter to use that name instead")
    print(f"  Type '?' to look up the face on Google Lens")
    print(f"  Type 'u' for UNKNOWN (keep person_NNN, won't be remembered)")
    print(f"  Type 'j' to mark as JUNK (accumulates in __junk__/; clean up via cleanup_junk.py)")
    print(f"  'q' = stop labeling now (finalizes labeled folders, preserves resume state)")
    print(f"  Progress is saved automatically every {LABEL_SAVE_EVERY} labels.\n")

    used_names: dict[str, int] = {n: cid for cid, n in name_map.items()
                                   if cid != -1 and not n.startswith("person_")}

    labels_since_save = 0

    def maybe_save_state() -> None:
        nonlocal labels_since_save
        if labels_since_save >= LABEL_SAVE_EVERY:
            save_labeling_state(records, name_map, output_dir, input_dir)
            labels_since_save = 0
            log.info("Labeling state checkpointed.")

    for idx, cid in enumerate(needs_label, start=1):
        if cid not in name_map:
            continue
        old_name = name_map[cid]
        montage = clusters_dir / f"{old_name}_montage.jpg"
        n_faces = len(by_id[cid])
        if montage.exists():
            subprocess.run(["open", str(montage)], check=False)

        suggestion: str | None = None
        if ai_active:
            best = max(by_id[cid], key=lambda r: r.quality)
            chash = crop_content_hash(best.crop_jpeg)
            if chash in ai_cache:
                cached_val = ai_cache[chash]
                suggestion = cached_val if cached_val else None
                consecutive_failures = 0
            else:
                print(f"[{idx}/{total}] {old_name}  ({n_faces} faces)  [asking Claude…]",
                      end="", flush=True)
                try:
                    suggestion = claude_suggest_name(best.crop_jpeg)
                    ai_cache[chash] = suggestion or ""
                    save_ai_cache(ai_cache)
                    consecutive_failures = 0
                    print("\r" + " " * 80 + "\r", end="", flush=True)
                except AIError as err:
                    print("\r" + " " * 80 + "\r", end="", flush=True)
                    consecutive_failures += 1
                    if err.reason in ("no_key", "no_package", "auth", "credits"):
                        _print_ai_error_help(err)
                        print("  → Disabling AI suggestions for the rest of this session.\n")
                        ai_active = False
                        suggestion = None
                    elif consecutive_failures >= AI_FAILURE_PAUSE_THRESHOLD:
                        choice = _ai_failure_dialog(err, consecutive_failures)
                        if choice == "quit":
                            save_labeling_state(records, name_map, output_dir, input_dir)
                            print("Stopping labeling. State saved.\n")
                            return False
                        if choice == "disable":
                            ai_active = False
                        consecutive_failures = 0
                        suggestion = None
                    else:
                        log.warning("Claude failed for cluster %d (%d/%d): %s",
                                    cid, consecutive_failures, AI_FAILURE_PAUSE_THRESHOLD, err)
                        suggestion = None

        header = f"[{idx}/{total}] {old_name}  ({n_faces} faces)"
        if suggestion:
            print(f"{header}  →  Claude suggests: \033[1;32m{suggestion}\033[0m")
            prompt_str = f"  [Enter=accept '{suggestion}', or type new name, '?'=Lens, 'u'=unknown, 'j'=junk, 'q'=quit]: "
        else:
            print(header)
            prompt_str = "  Name (or '?'=Lens, 'u'=unknown, 'j'=junk, 'q'=quit): "

        while True:
            try:
                raw = input(prompt_str).strip()
            except (EOFError, KeyboardInterrupt):
                save_labeling_state(records, name_map, output_dir, input_dir)
                print("\nInterrupted — state saved. Resume with: "
                      "python sort_photos.py --resume-label\n")
                return False

            if raw == "?":
                lens_lookup(by_id[cid])
                continue
            if raw.lower() == "q":
                save_labeling_state(records, name_map, output_dir, input_dir)
                print("Stopping labeling. State saved.")
                print("Resume with: python sort_photos.py --resume-label\n")
                return False
            if not raw:
                if suggestion:
                    chosen = suggestion
                else:
                    print(f"  → kept as '{old_name}'\n")
                    break
            elif raw.lower() == "u":
                print(f"  → marked unknown, kept as '{old_name}'\n")
                break
            elif raw.lower() == "j":
                # Mark this cluster as junk. Falls through to the regular
                # rename/merge path with the special label "__junk__", so
                # repeated 'j' presses accumulate into one __junk__ folder.
                print(f"  → marked as JUNK")
                chosen = "__junk__"
            else:
                chosen = raw

            new_name = sanitize_name(chosen)
            if not new_name:
                print(f"  Invalid name, kept as '{old_name}'\n")
                break

            if new_name in used_names:
                target_cid = used_names[new_name]
                merge_clusters_on_disk(records, name_map, clusters_dir,
                                        keep_cid=target_cid, drop_cid=cid)
                print(f"  → merged into '{new_name}'\n")
            else:
                src_folder = clusters_dir / old_name
                dst_folder = clusters_dir / new_name
                if src_folder.exists() and src_folder != dst_folder:
                    if dst_folder.exists():
                        # Destination folder already exists on disk (e.g. from a
                        # prior session). POSIX rename() refuses to clobber a
                        # non-empty dir, so move children in one-by-one with
                        # collision-avoidance suffixes, then drop the empty src.
                        dst_folder.mkdir(parents=True, exist_ok=True)
                        for child in src_folder.iterdir():
                            target = dst_folder / child.name
                            if target.exists():
                                stem = target.stem
                                suffix = "".join(target.suffixes)
                                k = 1
                                while True:
                                    candidate = dst_folder / f"{stem}__dup{k}{suffix}"
                                    if not candidate.exists():
                                        target = candidate
                                        break
                                    k += 1
                            child.rename(target)
                        try:
                            src_folder.rmdir()
                        except OSError:
                            pass
                    else:
                        src_folder.rename(dst_folder)
                if montage.exists():
                    montage.rename(clusters_dir / f"{new_name}_montage.jpg")
                name_map[cid] = new_name
                used_names[new_name] = cid
                print(f"  → '{new_name}'\n")

            labels_since_save += 1
            maybe_save_state()
            break

    # Loop completed naturally — save final state
    save_labeling_state(records, name_map, output_dir, input_dir)
    return skipped_small == 0


# ============================================================================
# ANCHOR-CLUSTER MERGE + REVIEW
# ============================================================================

def anchor_cluster_merge(records: list[FaceRecord],
                         name_map: dict[int, str],
                         clusters_dir: Path) -> int:
    centroids = compute_centroids(records)
    labeled_cids = [cid for cid, n in name_map.items()
                    if cid != -1 and not n.startswith("person_") and cid in centroids]
    if not labeled_cids:
        return 0
    labeled_C = np.stack([centroids[c] for c in labeled_cids])
    unlabeled_cids = [cid for cid, n in name_map.items()
                       if cid != -1 and n.startswith("person_") and cid in centroids]
    n_merged = 0
    for ucid in unlabeled_cids:
        if ucid not in name_map:
            continue
        c = centroids.get(ucid)
        if c is None:
            continue
        sims = labeled_C @ c
        j = int(np.argmax(sims))
        d = 1.0 - float(sims[j])
        if d <= ANCHOR_CLUSTER_MERGE_DIST:
            target_cid = labeled_cids[j]
            target_name = name_map[target_cid]
            log.info("Auto-merge: %s → %s (centroid distance %.3f)",
                     name_map[ucid], target_name, d)
            merge_clusters_on_disk(records, name_map, clusters_dir,
                                    keep_cid=target_cid, drop_cid=ucid)
            n_merged += 1
    return n_merged


def review_close_pairs(records: list[FaceRecord],
                       name_map: dict[int, str],
                       clusters_dir: Path) -> tuple[int, bool]:
    """Returns (n_merged, quit_early). quit_early is True if user pressed 'q'
    or sent SIGINT before all pairs were reviewed."""
    centroids = compute_centroids(records)
    cids = [c for c in centroids if c != -1 and c in name_map]
    if len(cids) < 2:
        return 0, False
    pairs: list[tuple[int, int, float]] = []
    for i in range(len(cids)):
        for j in range(i + 1, len(cids)):
            c1, c2 = cids[i], cids[j]
            d = 1.0 - float(centroids[c1] @ centroids[c2])
            if d <= REVIEW_CLOSE_PAIRS_DIST and d > MERGE_CENTROID_DIST:
                pairs.append((c1, c2, d))
    pairs.sort(key=lambda x: x[2])
    if not pairs:
        return 0, False
    print(f"\n=== Reviewing {len(pairs)} similar cluster pair(s) ===")
    print(f"Both montages open in Preview.")
    print(f"  'y' = same person → merge them")
    print(f"  Enter or 'n' = different people, leave separate")
    print(f"  'q' = stop reviewing now (resume later — state will be saved)\n")
    n_merged = 0
    quit_early = False
    for c1, c2, d in pairs:
        if c1 not in name_map or c2 not in name_map:
            continue
        n1, n2 = name_map[c1], name_map[c2]
        m1 = clusters_dir / f"{n1}_montage.jpg"
        m2 = clusters_dir / f"{n2}_montage.jpg"
        if m1.exists():
            subprocess.run(["open", str(m1)], check=False)
        if m2.exists():
            subprocess.run(["open", str(m2)], check=False)
        print(f"distance={d:.3f}  |  '{n1}'  vs  '{n2}'")
        try:
            ans = input("  Same person? (y/n/q): ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            quit_early = True
            break
        if ans == "q":
            quit_early = True
            break
        if ans == "y":
            n1_labeled = not n1.startswith("person_")
            n2_labeled = not n2.startswith("person_")
            if n1_labeled and not n2_labeled:
                keep, drop = c1, c2
            elif n2_labeled and not n1_labeled:
                keep, drop = c2, c1
            else:
                size1 = sum(1 for r in records if r.cluster_id == c1)
                size2 = sum(1 for r in records if r.cluster_id == c2)
                keep, drop = (c1, c2) if size1 >= size2 else (c2, c1)
            kept_name = name_map[keep]
            dropped_name = name_map[drop]
            merge_clusters_on_disk(records, name_map, clusters_dir,
                                    keep_cid=keep, drop_cid=drop)
            n_merged += 1
            print(f"  → merged '{dropped_name}' into '{kept_name}'\n")
        else:
            print(f"  → kept separate\n")
    return n_merged, quit_early


# ============================================================================
# DUPLICATE DETECTION
# ============================================================================

def dedup_within_bucket(items: list[tuple[Path, float, np.ndarray]],
                        threshold: int) -> tuple[list[Path], dict[Path, Path]]:
    items_sorted = sorted(items, key=lambda x: -x[1])
    keepers: list[Path] = []
    keeper_hashes: list[np.ndarray] = []
    dup_to_winner: dict[Path, Path] = {}
    for src, _q, h in items_sorted:
        if h.size == 0:
            keepers.append(src); keeper_hashes.append(h); continue
        best_match: Path | None = None
        best_dist = threshold + 1
        for kept_src, kept_h in zip(keepers, keeper_hashes):
            if kept_h.size == 0:
                continue
            d = hamming(h, kept_h)
            if d <= threshold and d < best_dist:
                best_dist = d
                best_match = kept_src
        if best_match is not None:
            dup_to_winner[src] = best_match
        else:
            keepers.append(src); keeper_hashes.append(h)
    return keepers, dup_to_winner


# ============================================================================
# ATOMIC ORIGINALS COPY WITH RESUME
# ============================================================================

def _checkpoint_path(originals_dir: Path) -> Path:
    return originals_dir / ".copy_checkpoint.json"


def _load_checkpoint(originals_dir: Path) -> set[str]:
    p = _checkpoint_path(originals_dir)
    if not p.exists():
        return set()
    try:
        with p.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return set(data.get("completed", []))
    except Exception:  # noqa: BLE001
        return set()


def _save_checkpoint(originals_dir: Path, completed: set[str]) -> None:
    p = _checkpoint_path(originals_dir)
    tmp = p.with_suffix(".json.tmp")
    try:
        with tmp.open("w", encoding="utf-8") as f:
            json.dump({"completed": sorted(completed)}, f)
        tmp.replace(p)
    except OSError:
        pass


def _atomic_copy(src: Path, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if USE_HARDLINKS:
        try:
            os.link(str(src), str(dest))
            return
        except OSError:
            pass
    tmp = dest.parent / (dest.name + ".part")
    try:
        shutil.copy2(str(src), str(tmp))
        tmp.replace(dest)
    except Exception:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass
        raise


def archive_organized_sources(sources: set[Path],
                              input_dir: Path,
                              output_dir: Path) -> int:
    if not sources:
        return 0
    archive_root = output_dir / "_source_review" / SOURCE_ARCHIVE_DIR_NAME
    moved = 0
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    for src in sorted(sources, key=lambda p: str(p).lower()):
        if not src.exists():
            continue
        try:
            resolved = src.resolve()
        except OSError:
            continue
        try:
            resolved.relative_to(output_dir)
            continue
        except ValueError:
            pass
        try:
            rel = resolved.relative_to(input_dir)
        except ValueError:
            rel = Path(resolved.name)
        dest = unique_path(archive_root / rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(resolved), str(dest))
            moved += 1
        except Exception as e:  # noqa: BLE001
            log.warning("Could not archive source %s: %s", resolved, e)
    return moved


def archive_scanned_sources(sources: Iterable[Path],
                            input_dir: Path,
                            output_dir: Path) -> int:
    archive_root = output_dir / "_source_review" / SCANNED_SOURCE_ARCHIVE_DIR_NAME
    moved = 0
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    for src in sorted(set(sources), key=lambda p: str(p).lower()):
        if not src.exists():
            continue
        try:
            resolved = src.resolve()
        except OSError:
            continue
        try:
            resolved.relative_to(output_dir)
            continue
        except ValueError:
            pass
        try:
            rel = resolved.relative_to(input_dir)
        except ValueError:
            rel = Path(resolved.name)
        dest = unique_path(archive_root / rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.move(str(resolved), str(dest))
            moved += 1
        except Exception as e:  # noqa: BLE001
            log.warning("Could not archive scanned source %s: %s", resolved, e)
    return moved


def organize_originals(records: list[FaceRecord],
                       name_map: dict[int, str],
                       originals_dir: Path,
                       input_dir: Path | None = None,
                       output_dir: Path | None = None) -> None:
    from tqdm import tqdm

    best_per_pair: dict[tuple[str, Path], FaceRecord] = {}
    for r in records:
        if r.cluster_id == -1 and not INCLUDE_UNKNOWN:
            continue
        if r.cluster_id not in name_map:
            continue
        person = name_map[r.cluster_id]
        if not is_real_person_label(person):
            continue
        key = (person, r.src)
        if key not in best_per_pair or r.quality > best_per_pair[key].quality:
            best_per_pair[key] = r

    buckets: dict[tuple[str, str], list[tuple[Path, float, np.ndarray]]] = defaultdict(list)
    for (person, src), rec in best_per_pair.items():
        if not src.exists():
            continue
        is_blurred = rec.sharpness < SHARPNESS_BLUR_THRESHOLD
        kind = "blurred" if is_blurred else "sharp"
        buckets[(person, kind)].append((src, rec.sharpness, rec.image_phash))

    originals_dir.mkdir(parents=True, exist_ok=True)
    completed = _load_checkpoint(originals_dir)
    if completed:
        log.info("Resume: %d copy task(s) already completed in previous run.",
                 len(completed))

    counts = {"sharp_keep": 0, "sharp_dup": 0,
              "blurred_keep": 0, "blurred_dup": 0,
              "missing": 0, "skipped_existing": 0,
              "nudity_possible": 0, "nudity_uncertain": 0,
              "nudity_errors": 0}
    per_person: dict[str, dict[str, int]] = defaultdict(
        lambda: {"sharp_keep": 0, "sharp_dup": 0,
                 "blurred_keep": 0, "blurred_dup": 0})

    save_every = 50
    pending_writes = 0
    next_indexes: dict[Path, int] = {}
    organized_sources: set[Path] = set()

    try:
        for (person, kind), items in tqdm(sorted(buckets.items()),
                                           desc="Copying originals", unit="bucket"):
            if kind == "sharp":
                base_dir = originals_dir / person
            else:
                base_dir = originals_dir / person / BLURRED_DIR
            person_dir = originals_dir / person
            base_dir.mkdir(parents=True, exist_ok=True)

            if DEDUP_DUPLICATES:
                keepers, dup_map = dedup_within_bucket(items, threshold=PHASH_THRESHOLD)
            else:
                keepers = [it[0] for it in items]; dup_map = {}

            for src in keepers:
                key = f"{person}||{src}||{kind}||main"
                if key in completed:
                    counts["skipped_existing"] += 1
                    continue
                if not src.exists():
                    counts["missing"] += 1
                    completed.add(key)
                    continue
                dest = next_numbered_dest(base_dir, person, src, next_indexes)
                try:
                    _atomic_copy(src, dest)
                    dest, nudity_status = maybe_move_to_nudity_subfolder(dest, person_dir)
                except Exception as e:  # noqa: BLE001
                    log.error("Copy failed: %s → %s: %s", src.name, dest.name, e)
                    continue
                if nudity_status == NUDITY_POSSIBLE_DIR:
                    counts["nudity_possible"] += 1
                elif nudity_status == NUDITY_UNCERTAIN_DIR:
                    counts["nudity_uncertain"] += 1
                elif nudity_status == "error":
                    counts["nudity_errors"] += 1
                organized_sources.add(src)
                completed.add(key)
                counts[f"{kind}_keep"] += 1
                per_person[person][f"{kind}_keep"] += 1
                pending_writes += 1
                if pending_writes >= save_every:
                    _save_checkpoint(originals_dir, completed)
                    pending_writes = 0

            if dup_map:
                dup_dir = base_dir / DUPLICATES_DIR
                dup_dir.mkdir(parents=True, exist_ok=True)
                for src in dup_map:
                    key = f"{person}||{src}||{kind}||dup"
                    if key in completed:
                        counts["skipped_existing"] += 1
                        continue
                    if not src.exists():
                        counts["missing"] += 1
                        completed.add(key)
                        continue
                    dest = next_numbered_dest(dup_dir, person, src, next_indexes)
                    try:
                        _atomic_copy(src, dest)
                        dest, nudity_status = maybe_move_to_nudity_subfolder(dest, person_dir)
                    except Exception as e:  # noqa: BLE001
                        log.error("Copy failed: %s → %s: %s", src.name, dest.name, e)
                        continue
                    if nudity_status == NUDITY_POSSIBLE_DIR:
                        counts["nudity_possible"] += 1
                    elif nudity_status == NUDITY_UNCERTAIN_DIR:
                        counts["nudity_uncertain"] += 1
                    elif nudity_status == "error":
                        counts["nudity_errors"] += 1
                    organized_sources.add(src)
                    completed.add(key)
                    counts[f"{kind}_dup"] += 1
                    per_person[person][f"{kind}_dup"] += 1
                    pending_writes += 1
                    if pending_writes >= save_every:
                        _save_checkpoint(originals_dir, completed)
                        pending_writes = 0
    finally:
        _save_checkpoint(originals_dir, completed)

    cp = _checkpoint_path(originals_dir)
    if cp.exists():
        try:
            cp.unlink()
        except OSError:
            pass

    log.info("---- Originals summary ----")
    for person in sorted(per_person.keys()):
        c = per_person[person]
        log.info("%-20s  sharp=%4d (+%d dup)   blurred=%4d (+%d dup)",
                 person, c["sharp_keep"], c["sharp_dup"],
                 c["blurred_keep"], c["blurred_dup"])
    log.info("TOTAL: sharp=%d (+%d dup), blurred=%d (+%d dup), missing=%d, "
             "skipped (already copied)=%d",
             counts["sharp_keep"], counts["sharp_dup"],
             counts["blurred_keep"], counts["blurred_dup"],
             counts["missing"], counts["skipped_existing"])
    if NUDITY_SORT_ENABLED:
        log.info("Nudity subfolder sort: possible=%d, uncertain=%d, errors=%d",
                 counts["nudity_possible"], counts["nudity_uncertain"],
                 counts["nudity_errors"])
    if ARCHIVE_ORGANIZED_SOURCES and input_dir is not None and output_dir is not None:
        moved = archive_organized_sources(organized_sources, input_dir, output_dir)
        log.info("Archived %d organized source image(s) to %s",
                 moved, output_dir / "_source_review" / SOURCE_ARCHIVE_DIR_NAME)


# ============================================================================
# MANIFEST
# ============================================================================

def write_manifest(records: list[FaceRecord],
                   name_map: dict[int, str],
                   centroids: dict[int, np.ndarray],
                   csv_path: Path) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "face_index", "det_score", "sharpness", "quality",
                    "person", "centroid_similarity", "from_cache"])
        for r in sorted(records, key=lambda r: (name_map.get(r.cluster_id, "?"), str(r.src))):
            sim = ""
            if r.cluster_id != -1 and r.cluster_id in centroids:
                sim = f"{float(centroids[r.cluster_id] @ r.embedding):.3f}"
            w.writerow([str(r.src), r.face_index, f"{r.det_score:.3f}",
                        f"{r.sharpness:.1f}", f"{r.quality:.3f}",
                        name_map.get(r.cluster_id, "?"), sim,
                        "yes" if r.prior_label else "no"])
    log.info("Manifest: %s", csv_path)


def run_post_process(output_dir: Path) -> None:
    if not POST_PROCESS_OUTPUT:
        return
    script_dir = Path(__file__).resolve().parent
    steps = [
        [sys.executable, str(script_dir / "delete_person_folder_duplicates.py"),
         "--sorted-root", str(output_dir), "--apply", "--quiet"],
        [sys.executable, str(script_dir / "optimize_sorted_output.py"),
         str(output_dir / "photos_by_person"), "--apply", "--quiet"],
        [sys.executable, str(script_dir / "rename_person_folder_files.py"),
         str(output_dir / "photos_by_person"), "--apply", "--quiet"],
        [sys.executable, str(script_dir / "advanced_duplicate_matching.py"),
         str(output_dir / "photos_by_person"), "--apply", "--quarantine-bad", "--quiet"],
    ]
    log.info("Running automatic output cleanup.")
    for cmd in steps:
        script_name = Path(cmd[1]).name
        if not Path(cmd[1]).exists():
            log.warning("Post-process helper missing, skipped: %s", cmd[1])
            continue
        result = subprocess.run(cmd, check=False)
        if result.returncode != 0:
            log.warning("Post-process step failed (%s), exit code %d.",
                        script_name, result.returncode)


# ============================================================================
# POST-LABELING PIPELINE (shared between fresh and resume)
# ============================================================================

def finish_pipeline(all_records: list[FaceRecord],
                    name_map: dict[int, str],
                    output_dir: Path,
                    input_dir: Path,
                    do_review: bool,
                    interactive_was_run: bool,
                    preserve_labeling_state: bool = False,
                    scanned_sources: Iterable[Path] | None = None) -> None:
    """Anchor-merge → close-pair review → originals copy → manifest → final
    cache save. Called whether labeling was fresh or resumed.

    If the user quits review with 'q', re-saves the labeling state so the next
    run can resume directly into review (no re-labeling needed)."""
    clusters_dir  = output_dir / "face_clusters"
    originals_dir = output_dir / "photos_by_person"
    csv_path      = output_dir / "_clusters.csv"

    n_acm = anchor_cluster_merge(all_records, name_map, clusters_dir)
    if n_acm:
        log.info("Anchor-cluster-merge: folded %d unlabeled cluster(s) into labels.",
                 n_acm)

    review_quit = False
    if do_review and interactive_was_run and not preserve_labeling_state:
        n_rev, review_quit = review_close_pairs(all_records, name_map, clusters_dir)
        if n_rev:
            log.info("Close-pair review: merged %d additional pair(s).", n_rev)
        if review_quit:
            # Save state so user can resume review later. Skip the rest of the
            # pipeline (originals copy, manifest) — we'll do those once review
            # is fully complete.
            save_labeling_state(all_records, name_map, output_dir, input_dir)
            print()
            log.info("Review interrupted — state saved.")
            log.info("Resume review (and finish the pipeline) with:  "
                     "python sort_photos.py --resume-label")
            return

    log.info("Copying originals to: %s", originals_dir)
    organize_originals(all_records, name_map, originals_dir,
                       input_dir=input_dir, output_dir=output_dir)

    centroids = compute_centroids(all_records)
    write_manifest(all_records, name_map, centroids, csv_path)
    run_post_process(output_dir)

    if ARCHIVE_SCANNED_SOURCES and scanned_sources is not None:
        moved = archive_scanned_sources(scanned_sources, input_dir, output_dir)
        log.info("Archived %d scanned source image(s) to %s",
                 moved, output_dir / "_source_review" / SCANNED_SOURCE_ARCHIVE_DIR_NAME)

    new_cache = CacheState(version=CACHE_VERSION,
                           config_fingerprint=config_fingerprint())
    for r in all_records:
        s = str(r.src)
        if s not in new_cache.file_signatures:
            try:
                new_cache.file_signatures[s] = file_signature(r.src)
            except OSError:
                continue
        final_name = name_map.get(r.cluster_id)
        is_real_label = is_real_person_label(final_name) or final_name == "__junk__"
        new_cache.faces.append(record_to_cached(
            r, label=final_name if is_real_label else None))

    save_cache(new_cache)
    log.info("Cache saved: %d files, %d faces (%d labeled).",
             len(new_cache.file_signatures), len(new_cache.faces),
             sum(1 for c in new_cache.faces if c.label))

    if preserve_labeling_state:
        save_labeling_state(all_records, name_map, output_dir, input_dir)
        log.info("Partial finalize complete. Labeled folders were written; "
                 "labeling state preserved for the remaining clusters.")
        log.info("Resume remaining labels with: python sort_photos.py --resume-label")
    else:
        # Pipeline fully completed — clear the labeling state
        clear_labeling_state()
    log.info("All done. Output: %s", output_dir)


# ============================================================================
# MAIN
# ============================================================================

def confirm_overwrite(output_dir: Path) -> bool:
    """Returns True if the pipeline should proceed. Optionally wipes the
    output folder if the user explicitly chooses to.

    Three choices:
      [Enter] / m / merge  – keep existing folders, merge new photos in (default)
      w / wipe / y         – delete output folder, start fresh
      c / cancel / n       – abort
    """
    if not output_dir.exists():
        return True
    print(f"\nOutput folder already exists: {output_dir}")
    print(f"  [Enter] or 'm'  →  keep existing labeled folders, merge new photos in (default)")
    print(f"  'w'             →  wipe and start completely fresh")
    print(f"  'c'             →  cancel")
    try:
        ans = input("Choose: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    if ans in ("c", "cancel", "n", "no"):
        return False
    if ans in ("w", "wipe", "y", "yes"):
        shutil.rmtree(output_dir)
        log.info("Output folder wiped: %s", output_dir)
        return True
    # Empty (Enter), 'm', 'merge', or anything unrecognized -> merge
    log.info("Merging into existing output folder: %s", output_dir)
    return True


def offer_resume(state: LabelingState) -> str:
    """Show summary of saved session and ask user what to do.
    Returns 'resume', 'fresh', or 'cancel'."""
    s = labeling_state_summary(state)
    print()
    print("=" * 60)
    print("FOUND SAVED LABELING SESSION")
    print("=" * 60)
    print(f"  Input folder:    {s['input_dir']}")
    print(f"  Output folder:   {s['output_dir']}")
    print(f"  Total clusters:  {s['n_clusters']}  ({s['total_faces']} faces total)")
    print(f"  Already labeled: {s['n_labeled']}")
    print(f"  Still to label:  {s['n_remaining']}")
    print()
    print("  [r] Resume — continue labeling where you left off")
    print("  [f] Start fresh — discard saved session and re-detect everything")
    print("  [c] Cancel — exit without doing anything")
    print()
    while True:
        try:
            ans = input("  Choose [r/f/c]: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return "cancel"
        if ans in ("r", ""):
            return "resume"
        if ans == "f":
            return "fresh"
        if ans == "c":
            return "cancel"


def do_resume(state: LabelingState, do_review: bool, use_ai: bool,
              min_cluster_size: int = 1, finish_labeled: bool = False) -> int:
    """Resume from saved labeling state. Skips detection + clustering."""
    output_dir = Path(state.output_dir)
    input_dir = Path(state.input_dir)
    clusters_dir = output_dir / "face_clusters"

    if not clusters_dir.exists():
        log.error("Saved labeling state references a missing clusters folder: %s",
                  clusters_dir)
        log.error("The output folder may have been deleted. Start fresh instead.")
        return 1

    log.info("Resuming labeling session from %s", LABEL_STATE_FILE)
    s = labeling_state_summary(state)
    log.info("  %d clusters, %d already labeled, %d remaining.",
             s["n_clusters"], s["n_labeled"], s["n_remaining"])

    # Reconstruct face records with their cluster IDs from the saved state
    all_records: list[FaceRecord] = []
    for cf, cid in zip(state.faces, state.cluster_ids):
        rec = cached_to_record(cf)
        rec.cluster_id = cid
        all_records.append(rec)
    name_map = dict(state.name_map)

    if finish_labeled:
        log.info("Finishing already-labeled clusters only; remaining labels preserved.")
        finish_pipeline(all_records, name_map, output_dir, input_dir=input_dir,
                        do_review=False, interactive_was_run=False,
                        preserve_labeling_state=True)
        return 0

    labeling_complete = interactive_label(all_records, name_map, clusters_dir,
                                          use_ai=use_ai, output_dir=output_dir,
                                          input_dir=input_dir,
                                          min_cluster_size=min_cluster_size)

    finish_pipeline(all_records, name_map, output_dir, input_dir=input_dir,
                    do_review=do_review, interactive_was_run=True,
                    preserve_labeling_state=not labeling_complete)
    return 0


def mark_small_clusters_junk(state: LabelingState, threshold: int,
                             finish_labeled: bool) -> int:
    """Mark saved person_NNN clusters below threshold as __junk__."""
    output_dir = Path(state.output_dir)
    input_dir = Path(state.input_dir)

    all_records: list[FaceRecord] = []
    by_cid: dict[int, int] = defaultdict(int)
    for cf, cid in zip(state.faces, state.cluster_ids):
        rec = cached_to_record(cf)
        rec.cluster_id = cid
        all_records.append(rec)
        by_cid[cid] += 1

    name_map = dict(state.name_map)
    targets = [
        cid for cid, name in name_map.items()
        if cid != -1 and name.startswith("person_") and by_cid.get(cid, 0) < threshold
    ]
    targets.sort(key=lambda cid: (by_cid.get(cid, 0), cid))

    if not targets:
        log.info("No unlabeled clusters smaller than %d face(s).", threshold)
    else:
        log.info("Marking %d small unlabeled cluster(s) below %d face(s) as junk.",
                 len(targets), threshold)
        log.info("This covers %d face(s).", sum(by_cid.get(cid, 0) for cid in targets))
        for cid in targets:
            name_map[cid] = "__junk__"

    save_labeling_state(all_records, name_map, output_dir, input_dir)

    if finish_labeled:
        finish_pipeline(all_records, name_map, output_dir, input_dir=input_dir,
                        do_review=False, interactive_was_run=False,
                        preserve_labeling_state=True)
    else:
        log.info("Saved labeling state updated. Run --finish-labeled to write cache/output.")

    return 0


def main() -> int:
    global INTERACTIVE_LABELING, DEDUP_DUPLICATES, REVIEW_CLOSE_PAIRS
    global USE_AI_SUGGESTIONS, NUDITY_SORT_ENABLED, AUTO_PERSON_MATCH_ENABLED
    global AUTO_PERSON_MATCH_DIST, IDENTITY_MAX_IMAGES_PER_PERSON
    global POST_PROCESS_OUTPUT, USE_HARDLINKS, ARCHIVE_ORGANIZED_SOURCES
    global ARCHIVE_SCANNED_SOURCES, SOURCE_ARCHIVE_DIR_NAME
    global UNATTENDED_FINISH_KNOWN, DET_SIZE

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument("output", nargs="?", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--no-label", action="store_true")
    parser.add_argument("--no-dedup", action="store_true")
    parser.add_argument("--no-review", action="store_true")
    parser.add_argument("--no-ai", action="store_true",
                        help="Disable Claude auto-name suggestions")
    parser.add_argument("--no-nudity-sort", action="store_true",
                        help="Do not auto-place flagged images into _possible_nudity/_uncertain_nudity subfolders.")
    parser.add_argument("--no-person-match", action="store_true",
                        help="Do not auto-label new clusters from the existing person identity DB.")
    parser.add_argument("--no-reference-match", action="store_true",
                        help="Do not auto-label from the optional Face References DB.")
    parser.add_argument("--external-centroids", type=Path,
                        default=REFERENCE_CENTROIDS_FILE,
                        help="Optional reference centroid DB built by build_celeb_centroids.py.")
    parser.add_argument("--rebuild-identity-db", action="store_true",
                        help="Rebuild known-person identity DB from output/photos_by_person before running.")
    parser.add_argument("--identity-db-only", action="store_true",
                        help="Only rebuild the known-person identity DB, then exit.")
    parser.add_argument("--person-match-dist", type=float, default=AUTO_PERSON_MATCH_DIST,
                        help="Distance threshold for existing-person auto-labels.")
    parser.add_argument("--identity-max-images", type=int, default=IDENTITY_MAX_IMAGES_PER_PERSON,
                        help="Max images per person used when rebuilding the identity DB.")
    parser.add_argument("--no-post-process", action="store_true",
                        help="Skip automatic output cleanup after finishing.")
    parser.add_argument("--copy-output", action="store_true",
                        help="Copy sorted images instead of hardlinking when possible.")
    parser.add_argument("--archive-organized-sources", action="store_true",
                        help="After successful organization, move source images to _source_review/organized_sources (or ready_to_delete/organized_sources with --archive-sources-to-ready-delete).")
    parser.add_argument("--archive-sources-to-ready-delete", action="store_true",
                        help="With --archive-organized-sources, move organized source images under _source_review/ready_to_delete/organized_sources.")
    parser.add_argument("--archive-scanned-sources", action="store_true",
                        help="After finishing, move every scanned input image out of the input folder to _source_review/ready_to_delete/scanned_sources. Best for a To Process inbox.")
    parser.add_argument("--unattended", action="store_true",
                        help="Do not ask for labels; organize known/auto-matched people, preserve unknown clusters for later.")
    parser.add_argument("--reset-cache", action="store_true")
    parser.add_argument("--resume-label", action="store_true",
                        help="Skip directly to resuming a saved labeling session")
    parser.add_argument("--finish-labeled", action="store_true",
                        help="Finalize/copy already-labeled clusters from the saved session without asking for more labels.")
    parser.add_argument("--min-label-cluster-size", type=int, default=1,
                        help="Only ask to manually label unlabeled clusters with at least N faces. Smaller clusters are preserved for later.")
    parser.add_argument("--junk-small-clusters", type=int, default=0,
                        help="From the saved labeling session, mark unlabeled clusters smaller than N faces as __junk__.")
    parser.add_argument("--fast", action="store_true",
                        help="Fast resume mode: no AI, no close-pair review, and only show clusters with at least 5 faces.")
    parser.add_argument("--scan-all-dirs", action="store_true",
                        help="Do not skip folders named sorted/photos_by_person/face_clusters. Use only for importing old outputs.")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE,
                        help="Images per detection subprocess. Default 50 keeps macOS memory stable.")
    parser.add_argument("--detect-workers", type=int, default=DETECT_WORKERS,
                        help="Run this many detection subprocess batches in parallel. Use 1 for safest memory use; 2 can be faster on large CPU-only scans.")
    parser.add_argument("--det-size", type=int, default=DET_SIZE[0],
                        help="Face detector input size. Default 1024 for accuracy; 640 is faster but may miss small faces and invalidates the cache.")
    parser.add_argument("--detect-batch", type=str, default=None,
                        help="Internal: run detection worker.")
    args = parser.parse_args()

    if args.detect_batch:
        return run_detection_worker(Path(args.detect_batch))

    if args.fast:
        args.no_ai = True
        args.no_review = True
        if args.min_label_cluster_size == 1:
            args.min_label_cluster_size = 5
    if args.no_nudity_sort:
        NUDITY_SORT_ENABLED = False
    if args.no_person_match:
        AUTO_PERSON_MATCH_ENABLED = False
    AUTO_PERSON_MATCH_DIST = float(args.person_match_dist)
    IDENTITY_MAX_IMAGES_PER_PERSON = max(0, int(args.identity_max_images))
    if args.no_post_process:
        POST_PROCESS_OUTPUT = False
    if args.copy_output:
        USE_HARDLINKS = False
    if args.archive_organized_sources:
        ARCHIVE_ORGANIZED_SOURCES = True
    if args.archive_sources_to_ready_delete:
        ARCHIVE_ORGANIZED_SOURCES = True
        SOURCE_ARCHIVE_DIR_NAME = str(Path("ready_to_delete") / "organized_sources")
    if args.archive_scanned_sources:
        ARCHIVE_SCANNED_SOURCES = True
    if args.unattended:
        UNATTENDED_FINISH_KNOWN = True
        args.no_ai = True
        args.no_review = True
    det_size = max(320, int(args.det_size))
    DET_SIZE = (det_size, det_size)

    use_ai = not args.no_ai
    do_review = not args.no_review
    min_label_cluster_size = max(1, int(args.min_label_cluster_size))
    requested_input_dir = Path(args.input).expanduser()
    requested_output_dir = Path(args.output).expanduser()

    if args.identity_db_only or args.rebuild_identity_db:
        build_identity_db_from_person_folders(requested_output_dir / "photos_by_person")
        if args.identity_db_only:
            return 0

    # Saved-session-only operations: skip detection/clustering.
    if args.resume_label or args.finish_labeled or args.junk_small_clusters:
        state = load_labeling_state()
        if state is None:
            log.error("No saved labeling session found at %s", LABEL_STATE_FILE)
            return 1
        if args.junk_small_clusters:
            return mark_small_clusters_junk(
                state,
                threshold=max(1, int(args.junk_small_clusters)),
                finish_labeled=args.finish_labeled,
            )
        return do_resume(state, do_review=do_review, use_ai=use_ai,
                         min_cluster_size=min_label_cluster_size,
                         finish_labeled=args.finish_labeled)

    # Normal flow: check for saved session first and offer to resume
    saved = load_labeling_state()
    if saved is not None and not args.reset_cache:
        if UNATTENDED_FINISH_KNOWN:
            backup = backup_labeling_state("unattended_discard")
            clear_labeling_state()
            if backup:
                log.info("Unattended mode: archived old saved labeling session at %s",
                         backup)
            log.info("Unattended mode: starting a new scan without manual resume prompt.")
        else:
            choice = offer_resume(saved)
            if choice == "cancel":
                log.info("Cancelled.")
                return 1
            if choice == "resume":
                return do_resume(saved, do_review=do_review, use_ai=use_ai,
                                 min_cluster_size=min_label_cluster_size)
            # 'fresh' falls through and clears the state
            backup = backup_labeling_state("fresh_discard")
            clear_labeling_state()
            if backup:
                log.info("Backed up discarded saved session at %s", backup)
            log.info("Discarded saved session. Starting fresh.")

    # Fresh run path
    input_dir = requested_input_dir
    output_dir = requested_output_dir
    if not input_dir.exists():
        log.error("Input folder does not exist: %s", input_dir)
        return 2

    if args.reset_cache:
        if CACHE_FILE.exists():
            CACHE_FILE.unlink()
            log.info("Cache wiped: %s", CACHE_FILE)
        clear_labeling_state()

    if not confirm_overwrite(output_dir):
        log.info("Cancelled.")
        return 1

    if args.no_label:
        INTERACTIVE_LABELING = False
    if args.no_dedup:
        DEDUP_DUPLICATES = False
    if args.no_review:
        REVIEW_CLOSE_PAIRS = False
    if args.no_ai:
        USE_AI_SUGGESTIONS = False

    batch_size = max(1, int(args.batch_size))

    clusters_dir  = output_dir / "face_clusters"

    cache = load_cache()
    log.info("Loaded cache: %d known files, %d cached faces, %d labeled.",
             len(cache.file_signatures), len(cache.faces),
             sum(1 for c in cache.faces if c.label))

    excluded_scan_dirs = set() if args.scan_all_dirs else None
    images = list(iter_images(input_dir, excluded_dir_names=excluded_scan_dirs))
    log.info("Found %d images under %s", len(images), input_dir)
    if not images:
        return 0

    cached_records: list[FaceRecord] = []
    new_images: list[Path] = []
    cache_face_index: dict[str, list[CachedFace]] = defaultdict(list)
    for c in cache.faces:
        cache_face_index[c.src_str].append(c)

    for img in images:
        try:
            sig = file_signature(img)
        except OSError:
            continue
        s = str(img)
        if cache.file_signatures.get(s) == sig:
            for c in cache_face_index.get(s, []):
                cached_records.append(cached_to_record(c))
        else:
            new_images.append(img)
            cache.file_signatures.pop(s, None)
            if s in cache_face_index:
                cache_face_index.pop(s)
                cache.faces = [c for c in cache.faces if c.src_str != s]

    del cache_face_index
    gc.collect()

    log.info("Cache hit: %d images (%d faces). New / changed: %d images.",
             len(images) - len(new_images), len(cached_records), len(new_images))

    new_records: list[FaceRecord] = []
    if new_images:
        new_records = detect_in_batches_subprocess(
            new_images, cache, batch_size=batch_size,
            workers=max(1, int(args.detect_workers)))
        log.info("Total new faces extracted across all batches: %d.", len(new_records))

    all_records = cached_records + new_records
    if not all_records:
        log.warning("No faces to process. Exiting.")
        return 0

    n_a = stage_a_dbscan(all_records)
    log.info("Stage A: %d clusters, %d unknown.",
             n_a, sum(1 for r in all_records if r.cluster_id == -1))
    n_pl = auto_merge_by_prior_labels(all_records)
    if n_pl:
        log.info("Auto-merged %d cluster(s) sharing the same prior label.", n_pl)
    n_merged = merge_close_clusters(all_records)
    if n_merged:
        log.info("Merged %d near-duplicate cluster(s).", n_merged)
    n_anchor = anchor_pass(all_records)
    if n_anchor:
        log.info("Anchor pass: pulled %d unknown faces into prior-labeled clusters.",
                 n_anchor)
    n_b = stage_b_reassign(all_records)
    log.info("Stage B: recovered %d unknown faces.", n_b)
    n_merged2 = merge_close_clusters(all_records)
    if n_merged2:
        log.info("Final merge: combined %d more cluster(s).", n_merged2)

    centroids = compute_centroids(all_records)
    log.info("Final clustering: %d people, %d unknown faces.",
             len(centroids), sum(1 for r in all_records if r.cluster_id == -1))

    name_map = make_initial_name_map(all_records)
    if AUTO_PERSON_MATCH_ENABLED:
        identity_db = load_identity_db()
        if identity_db is None and (output_dir / "photos_by_person").exists():
            log.info("No usable identity DB found; building it from existing person folders.")
            identity_db = build_identity_db_from_person_folders(output_dir / "photos_by_person")
        if not args.no_reference_match:
            reference_db = load_reference_centroids(args.external_centroids)
            identity_db = merge_identity_dbs(identity_db, reference_db)
        n_identity = apply_identity_db_labels(all_records, name_map, identity_db)
        if n_identity:
            log.info("Existing-person identity DB auto-labeled %d cluster(s).", n_identity)
    write_cluster_crops(all_records, name_map, clusters_dir, centroids)
    log.info("Wrote face crops to: %s", clusters_dir)

    # Save labeling state BEFORE entering interactive labeling
    save_labeling_state(all_records, name_map, output_dir, input_dir)
    log.info("Labeling state saved. You can quit anytime with 'q' and resume "
             "with: python sort_photos.py --resume-label")

    if UNATTENDED_FINISH_KNOWN:
        log.info("Unattended mode: skipping manual labeling and preserving remaining "
                 "unknown clusters for later.")
        labeling_complete = False
    elif INTERACTIVE_LABELING:
        labeling_complete = interactive_label(all_records, name_map, clusters_dir,
                                              use_ai=USE_AI_SUGGESTIONS,
                                              output_dir=output_dir,
                                              input_dir=input_dir,
                                              min_cluster_size=min_label_cluster_size)
    else:
        log.info("Skipping interactive labeling.")
        labeling_complete = True

    finish_pipeline(all_records, name_map, output_dir, input_dir=input_dir,
                    do_review=REVIEW_CLOSE_PAIRS,
                    interactive_was_run=INTERACTIVE_LABELING,
                    preserve_labeling_state=not labeling_complete,
                    scanned_sources=images)
    return 0


if __name__ == "__main__":
    sys.exit(main())

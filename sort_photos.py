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
import contextlib
import csv
import gc
import hashlib
import json
import logging
import os
import pickle
import re
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import cv2
import numpy as np

import operation_ledger
import generated_artifacts
import pipeline_paths
import nudity_confirmations
import identity_profiles
import identity_confirmations
import identity_hard_negatives
import analysis_index
import copy_journal
import content_identity
import asset_processing
import appearance_profiles
import face_detection
import file_operations
import routing_policy
import secondary_identity_matcher
import source_batch_consensus

warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\..*")
warnings.filterwarnings("ignore", message=r".*`estimate` is deprecated.*", category=FutureWarning)
warnings.filterwarnings(
    "ignore",
    message=r"resource_tracker: There appear to be .* leaked semaphore objects.*",
    category=UserWarning,
)

SUBPROCESS_FORK_RETRIES = 8
SUBPROCESS_FORK_RETRY_SECONDS = 8
DETECTION_WORKER_ENV_LIMITS = {
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "VECLIB_MAXIMUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
}

# ============================================================================
# CONFIG
# ============================================================================

DEFAULT_INPUT  = Path.home() / "Pictures"
DEFAULT_OUTPUT = pipeline_paths.SORTED_ROOT
CACHE_DIR      = Path.home() / ".face_sort_cache"
CACHE_FILE     = CACHE_DIR / "cache.pkl"
AI_CACHE_FILE  = CACHE_DIR / "ai_suggestions.json"
LABEL_STATE_FILE = CACHE_DIR / "labeling_state.pkl"
IDENTITY_DB_FILE = CACHE_DIR / "person_identity_db.pkl"
IDENTITY_DB_BUILD_FILE = CACHE_DIR / "person_identity_db.building.pkl"
IDENTITY_CONFIRMATIONS_FILE = identity_confirmations.default_path(CACHE_DIR)
IDENTITY_HARD_NEGATIVES_FILE = identity_hard_negatives.default_path(CACHE_DIR)
REFERENCE_CENTROIDS_FILE = CACHE_DIR / "reference_centroids.pkl"
DEFAULT_FINGERPRINT_CACHE_FILE = CACHE_DIR / "advanced_duplicate_fingerprints.json"
FINGERPRINT_CACHE_FILE = DEFAULT_FINGERPRINT_CACHE_FILE
CACHE_VERSION  = 2
LABEL_STATE_VERSION = 1
IDENTITY_DB_VERSION = 2
IDENTITY_CALIBRATION_VERSION = 7
FINGERPRINT_CACHE_VERSION = 1

BATCH_SIZE = 50
DETECT_WORKERS = 1
LABEL_SAVE_EVERY = 10   # save labeling state every N labels applied

MODEL_NAME = "antelopev2"
PROVIDERS  = ["CPUExecutionProvider"]
DET_SIZE   = (1024, 1024)

MIN_DET_SCORE = 0.55
MIN_FACE_PX   = 70
MIN_SHARPNESS = 40.0

# The strict thresholds above select good reference/training faces. They must
# not be used to decide whether an image contains a face at all. A second,
# conservative recovery tier keeps soft, small, or compressed faces available
# for clustering and identity review instead of incorrectly filing them as
# "no usable face".
RECOVERY_MIN_DET_SCORE = 0.45
RECOVERY_MIN_FACE_PX = 24
RECOVERY_MIN_SHARPNESS = 1.0
RECOVERY_EMBEDDING_DEDUP_SIMILARITY = 0.94
FACE_PRESENCE_MIN_DET_SCORE = 0.20
FALLBACK_MAX_IMAGE_DIMENSION = 1600

CROP_SIZE     = 256
JPEG_QUALITY  = 92
PADDING_RATIO = 0.30

STAGE_A_EPS               = 0.32
STAGE_A_MIN_SAMPLES       = 2
STAGE_B_MAX_DIST          = 0.50
STAGE_B_MIN_MARGIN        = 0.04
MERGE_CENTROID_DIST       = 0.42   # raised from 0.38 — auto-merges more pairs
                                    # without asking. Tuned for celebrity
                                    # collections where false merges are rare.
                                    # For family photos, lower back to 0.38.
ANCHOR_MAX_DIST           = 0.55
ANCHOR_MIN_MARGIN         = 0.05
ANCHOR_CLUSTER_MERGE_DIST = 0.42
ANCHOR_CLUSTER_MERGE_MIN_MARGIN = 0.08
REVIEW_CLOSE_PAIRS_DIST   = 0.46   # lowered from 0.50 — fewer review prompts.

SHARPNESS_BLUR_THRESHOLD = 0.0

PHASH_THRESHOLD = 8
INTAKE_DUP_PHASH_THRESHOLD = 5
DUPLICATES_DIR  = "review/duplicates"
LEGACY_DUPLICATES_DIR = "_duplicates"
BLURRED_DIR     = "_blurred"
PERSON_PHOTOS_DIR = "photos"
PERSON_NUDE_DIR = "nude"
PERSON_REVIEW_DIR = "review"

INTERACTIVE_LABELING = True
DEDUP_DUPLICATES     = True
REVIEW_CLOSE_PAIRS   = True
USE_AI_SUGGESTIONS   = True
# New scans place only high-confidence, non-conflicting explicit detections in
# each person's photos/nude folder. Borderline model output remains reviewable
# instead of being mislabeled as confirmed nudity.
NUDITY_SORT_ENABLED  = True

MAKE_MONTAGES   = True
MONTAGE_COLS    = 6
MONTAGE_TILE_PX = 160
INCLUDE_UNKNOWN = True

NUDITY_THRESHOLD = 0.80
NUDITY_UNCERTAIN_THRESHOLD = 0.45
NUDITY_POSSIBLE_DIR = f"{PERSON_PHOTOS_DIR}/{PERSON_NUDE_DIR}"
NUDITY_UNCERTAIN_DIR = f"{PERSON_REVIEW_DIR}/uncertain_nudity"
# This remains conservative by default. A user who has explicitly chosen to
# treat every uncertain result as nude can persist that routing preference in
# the per-Mac face pipeline config without changing global detector thresholds.
ROUTE_UNCERTAIN_NUDITY_TO_NUDE = pipeline_paths.configured_bool(
    "route_uncertain_nudity_to_nude",
    False,
    "FACE_ROUTE_UNCERTAIN_NUDITY_TO_NUDE",
)
NUDITY_CLASS_THRESHOLDS = dict(routing_policy.DEFAULT_CLASS_THRESHOLDS)
NUDITY_EXPLICIT_CLASSES = set(routing_policy.EXPLICIT_CLASSES)
NUDITY_COVERED_EQUIVALENTS = dict(routing_policy.COVERED_EQUIVALENTS)
NUDITY_COVERED_CLASSES = set(routing_policy.COVERED_CLASSES)
NUDITY_EXPOSED_OVER_COVERED_MARGIN = 0.15
_NUDITY_DETECTOR = None
_NUDITY_IMPORT_WARNED = False
_NUDITY_STATUS_CACHE: dict[str, str] = {}
NUDITY_ANALYSIS_VERSION = "nudenet-policy-v3-spatial-conflicts"

AUTO_PERSON_MATCH_ENABLED = True
AUTO_PERSON_MATCH_DIST = 0.40
AUTO_PERSON_MATCH_MARGIN = 0.04
AUTO_PERSON_MATCH_MIN_CLUSTER_FACES = 2
AUTO_PERSON_MATCH_MIN_CLUSTER_SOURCES = 2
AUTO_PERSON_MATCH_MIN_AGREEMENT = 0.80
AUTO_PERSON_MATCH_MIN_REFERENCE_FACES = 3
IDENTITY_MAX_IMAGES_PER_PERSON = 80
IDENTITY_MAX_PROTOTYPES_PER_PERSON = 12
IDENTITY_MAX_TRUSTED_PROTOTYPES_PER_PERSON = 4
IDENTITY_CANDIDATE_POOL_MIN = 40
AUTO_PERSON_SINGLE_MATCH_DIST = 0.27
# Calibrated against the labeled local cache. A single image has no independent
# corroborating source, so require a clearly separated best identity. Multi-
# image clusters continue to use the consensus lane below.
AUTO_PERSON_SINGLE_MATCH_MARGIN = 0.42
AUTO_PERSON_SINGLE_MIN_QUALITY = 0.58
POST_PROCESS_OUTPUT = True
USE_HARDLINKS = True
ARCHIVE_ORGANIZED_SOURCES = False
ARCHIVE_SCANNED_SOURCES = False
UNATTENDED_FINISH_KNOWN = False
ASSUME_MERGE_EXISTING_OUTPUT = False
SOURCE_ARCHIVE_DIR_NAME = "organized_sources"
SCANNED_SOURCE_ARCHIVE_DIR_NAME = "ready_to_delete/scanned_sources"
INTAKE_DUPLICATE_ARCHIVE_DIR_NAME = "ready_to_delete/intake_duplicates"
INTAKE_NEAR_VISUAL_REVIEW_DIR_NAME = f"{PERSON_REVIEW_DIR}/near_visual"
UNASSIGNED_INTAKE_DIR_NAME = "unassigned_intake"
UNASSIGNED_INTAKE_REPORT_DIR_NAME = "unassigned_intake/reports"

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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
              ".tif", ".tiff", ".heic", ".heif"}
VIDEO_EXTS = {
    ".3g2", ".3gp", ".avi", ".m4v", ".mkv", ".mov", ".mp4",
    ".mpeg", ".mpg", ".mts", ".m2ts", ".webm", ".wmv",
}
TO_PROCESS_DIR = pipeline_paths.TO_PROCESS
VIDEOS_DIR = Path.home() / "Pictures" / "videos"
INVALID_NAME_CHARS = '/\\:*?"<>|'
DEFAULT_EXCLUDED_SCAN_DIRS = {
    "Celebrities",
    "_nudity_review",
    "_smart_albums",
    "_smart_albums_v2",
    "_smart_albums_simple_preview",
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
    "all",
    "_duplicates",
    "_near_visual_review",
    "_smart_albums",
    "_smart_albums_v2",
    "_smart_albums_simple_preview",
    "_blurred",
    "review",
    "videos",
}

# ============================================================================

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("sort_photos")


@contextlib.contextmanager
def suppress_native_stderr(enabled: bool = True):
    """Hide noisy native decoder warnings while still returning read failures."""
    if not enabled:
        yield
        return
    try:
        sys.stderr.flush()
        fd = sys.stderr.fileno()
        saved = os.dup(fd)
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), fd)
            try:
                yield
            finally:
                sys.stderr.flush()
                os.dup2(saved, fd)
                os.close(saved)
    except Exception:
        yield


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
    pose_label: str = "unknown"
    identity_review_reason: str = ""
    recovery_assigned: bool = False
    bbox: tuple[float, ...] = ()
    keypoints: tuple[tuple[float, ...], ...] = ()
    content_sha256: str = ""

    def crop(self) -> np.ndarray:
        if self._crop_array is not None:
            return self._crop_array
        if not self.crop_jpeg:
            return np.zeros((CROP_SIZE, CROP_SIZE, 3), dtype=np.uint8)
        with suppress_native_stderr():
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
    pose_label: str = "unknown"
    bbox: tuple[float, ...] = ()
    keypoints: tuple[tuple[float, ...], ...] = ()
    content_sha256: str = ""


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
    recovery_metadata: list[tuple[str, bool]] = field(default_factory=list)


@dataclass
class IdentityDB:
    version: int = IDENTITY_DB_VERSION
    config_fingerprint: str = ""
    identities: dict[str, np.ndarray] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    prototypes: dict[str, list[np.ndarray]] = field(default_factory=dict)
    intrinsic_match_thresholds: dict[str, float] = field(default_factory=dict)
    intrinsic_strict_thresholds: dict[str, float] = field(default_factory=dict)
    match_thresholds: dict[str, float] = field(default_factory=dict)
    strict_thresholds: dict[str, float] = field(default_factory=dict)
    nearest_impostor_distances: dict[str, float] = field(default_factory=dict)
    source_signatures: dict[str, str] = field(default_factory=dict)
    prototype_sources: dict[str, list[str]] = field(default_factory=dict)
    pose_prototypes: dict[str, dict[str, list[np.ndarray]]] = field(default_factory=dict)
    pose_prototype_sources: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    appearance_prototypes: dict[str, dict[str, list[np.ndarray]]] = field(default_factory=dict)
    appearance_prototype_sources: dict[str, dict[str, list[str]]] = field(default_factory=dict)
    appearance_era_cutoffs: dict[str, float] = field(default_factory=dict)
    calibration_version: int = IDENTITY_CALIBRATION_VERSION


def install_pickle_class_aliases() -> None:
    main_mod = sys.modules.get("__main__")
    if main_mod is None:
        return
    for name in ("CachedFace", "CacheState", "LabelingState", "IdentityDB", "FaceRecord"):
        if hasattr(main_mod, name):
            continue
        setattr(main_mod, name, globals()[name])


# ============================================================================
# IO + IMAGE HELPERS
# ============================================================================

def iter_images(root: Path,
                excluded_dir_names: set[str] | None = None,
                always_excluded_dir_names: set[str] | None = None) -> Iterable[Path]:
    excluded = DEFAULT_EXCLUDED_SCAN_DIRS if excluded_dir_names is None else excluded_dir_names
    always_excluded = (
        ALWAYS_EXCLUDED_SCAN_DIRS
        if always_excluded_dir_names is None
        else always_excluded_dir_names
    )
    excluded_casefold = {d.casefold() for d in excluded} | {
        d.casefold() for d in always_excluded
    }
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if d.casefold() not in excluded_casefold and not d.startswith(".")
        ]
        base = Path(dirpath)
        for filename in filenames:
            p = base / filename
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                yield p


def bounded_input_images(images: Iterable[Path], limit: int = 0) -> list[Path]:
    """Return a deterministic input slice so interrupted intake can resume."""
    ordered = sorted(images, key=lambda path: str(path).casefold())
    return ordered[:limit] if limit > 0 else ordered


def cache_path_key(path: str | os.PathLike[str]) -> str:
    """Return a stable path key without touching every path component on disk.

    ``realpath`` performs several filesystem lookups per path. Building cache
    indexes with it made large external libraries appear frozen for minutes.
    A lexical absolute path is sufficient here; a symlink alias can only cause
    a safe cache miss and re-analysis, never an incorrect cache hit.
    """
    return os.path.normcase(os.path.abspath(os.path.expanduser(os.fspath(path))))


def iter_person_original_images(people_dir: Path) -> Iterable[Path]:
    """Yield only canonical person-library originals, never sibling staging folders."""
    if not people_dir.exists():
        return
    for person_dir in sorted(people_dir.iterdir(), key=lambda p: p.name.casefold()):
        if not person_dir.is_dir() or person_dir.name.startswith((".", "_")):
            continue
        photos_dir = person_dir / PERSON_PHOTOS_DIR
        scan_root = photos_dir if photos_dir.exists() else person_dir
        yield from iter_images(scan_root, excluded_dir_names=set())


def iter_videos(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".DS_Store" and not d.startswith(".")]
        base = Path(dirpath)
        for filename in filenames:
            p = base / filename
            if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
                yield p


def prune_empty_dirs(root: Path) -> int:
    removed = 0
    if not root.exists():
        return removed
    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        path = Path(dirpath)
        if path == root:
            continue
        try:
            if not dirnames and not filenames:
                path.rmdir()
                removed += 1
        except OSError:
            pass
    return removed


def move_to_process_videos(input_dir: Path, videos_dir: Path = VIDEOS_DIR) -> int:
    try:
        resolved_input = input_dir.expanduser().resolve()
        resolved_to_process = TO_PROCESS_DIR.resolve()
    except OSError:
        return 0
    if resolved_input != resolved_to_process:
        return 0

    videos = list(iter_videos(resolved_input))
    if not videos:
        return 0
    log.warning(
        "Leaving %d video file(s) in To Process. Face Terminal classifies videos "
        "with sort_videos.py before the image scan.",
        len(videos),
    )
    return 0


def imread_unicode(path: Path) -> np.ndarray | None:
    with suppress_native_stderr():
        decoded = asset_processing.decode_image(path)
    return decoded.bgr if decoded is not None else None


def sharpness(bgr: np.ndarray) -> float:
    return face_detection.sharpness(bgr)


def square_pad_bbox(x1, y1, x2, y2, img_w, img_h, pad_ratio):
    return face_detection.square_pad_bbox(
        x1, y1, x2, y2, img_w, img_h, pad_ratio,
    )


def yaw_proxy_from_kps(kps, bbox: np.ndarray) -> float:
    return face_detection.yaw_proxy_from_keypoints(kps, bbox)


def quality_score_from_parts(det_score: float, bbox_size: float,
                              sharp: float, yaw: float) -> float:
    return face_detection.quality_score(det_score, bbox_size, sharp, yaw)


def perceptual_hash(bgr: np.ndarray, hash_size: int = 8) -> np.ndarray:
    return asset_processing.perceptual_hash(bgr, hash_size)


def phash_to_int(bits: np.ndarray) -> int:
    return asset_processing.phash_to_int(bits)


def sha256_file(path: Path) -> str:
    return file_operations.sha256_file(path)


def decoded_pixel_sha256(img: np.ndarray) -> str:
    return asset_processing.decoded_pixel_sha256(img)


def hamming(a: np.ndarray, b: np.ndarray) -> int:
    return int(np.count_nonzero(a != b))


def hamming_int(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def duplicate_nudity_status_for_path(path: Path) -> str:
    return routing_policy.duplicate_nudity_status_for_path(path)


def classified_nudity_status(
    path: Path,
    file_hash: str | None = None,
    asset_index: analysis_index.AnalysisIndex | None = None,
) -> str:
    """Return a duplicate category without merging normal and nude variants.

    Explicit folder/name markers take precedence. Otherwise NudeNet is used and
    the decision is cached by content hash so the copied destination does not
    need to be analysed a second time. Detection failures are deliberately
    returned as ``unknown`` so an intake file is preserved instead of being
    discarded as a normal duplicate.
    """
    try:
        current_hash = content_identity.content_sha256(path)
    except OSError:
        return "unknown"
    if file_hash and file_hash != current_hash:
        return "unknown"
    file_hash = current_hash
    if nudity_confirmations.is_confirmed(path, sha256=file_hash):
        if asset_index is not None:
            asset_index.record_nudity(
                path, model="manual-confirmation-v1", status="possible", detections=[], expected_sha256=file_hash)
        if file_hash:
            _NUDITY_STATUS_CACHE[file_hash] = "possible"
        return "possible"

    path_status = duplicate_nudity_status_for_path(path)
    if path_status != "safe":
        if asset_index is not None:
            asset_index.record_nudity(
                path, model="path-policy-v1", status=path_status, detections=[], expected_sha256=file_hash)
        return path_status

    if file_hash and file_hash in _NUDITY_STATUS_CACHE:
        return _NUDITY_STATUS_CACHE[file_hash]
    if asset_index is not None:
        cached = asset_index.cached_nudity(path, NUDITY_ANALYSIS_VERSION)
        if cached is not None:
            status, _detections = cached
            if file_hash:
                _NUDITY_STATUS_CACHE[file_hash] = status
            return status

    detector = _get_nudity_detector()
    if detector is None or not path.exists():
        return "unknown"
    try:
        detections = _detect_nudity_with_fallback(detector, path)
    except Exception as exc:  # noqa: BLE001
        log.debug("Nudity classification unavailable for %s: %s", path.name, exc)
        return "unknown"

    subdir, _best_class, _best_score = _nudity_category(detections)
    try:
        unchanged = content_identity.content_sha256(path) == file_hash
    except OSError:
        unchanged = False
    if not unchanged:
        return "unknown"
    if subdir == NUDITY_POSSIBLE_DIR:
        status = "possible"
    elif subdir == NUDITY_UNCERTAIN_DIR:
        status = "uncertain"
    else:
        status = "safe"
    if asset_index is not None:
        asset_index.record_nudity(
            path,
            model=NUDITY_ANALYSIS_VERSION,
            status=status,
            detections=detections,
            expected_sha256=file_hash,
        )
    if file_hash:
        _NUDITY_STATUS_CACHE[file_hash] = status
    return status


def sanitize_name(name: str) -> str:
    name = name.strip()
    for ch in INVALID_NAME_CHARS:
        name = name.replace(ch, "_")
    return name


def is_real_person_label(name: str | None) -> bool:
    normalized = str(name or "").strip().casefold()
    return bool(
        normalized
        and normalized not in {"unknown", "__junk__", "junk"}
        and not normalized.startswith("person_")
    )


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
                       person_dir: Path,
                       person: str,
                       src: Path,
                       next_indexes: dict[Path, int]) -> Path:
    prefix = filename_prefix_from_person(person)
    if person_dir not in next_indexes:
        max_index = 0
        pattern = re.compile(
            rf"^{re.escape(prefix)}_(\d+)(?:_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)?$",
            re.IGNORECASE,
        )
        photos_root = person_dir / PERSON_PHOTOS_DIR
        if photos_root.exists():
            for existing in photos_root.rglob("*"):
                if not existing.is_file():
                    continue
                if existing.suffix.lower() not in IMAGE_EXTS:
                    continue
                relative_parts = existing.relative_to(photos_root).parts
                if any(part in {
                    "all",
                    "_smart_albums",
                    "_smart_albums_v2",
                    "_smart_albums_simple_preview",
                    "review",
                    "_duplicates",
                    "_near_visual_review",
                } for part in relative_parts[:-1]):
                    continue
                match = pattern.fullmatch(existing.stem)
                if match:
                    max_index = max(max_index, int(match.group(1)))
        next_indexes[person_dir] = max_index + 1

    ext = src.suffix.lower() or ".jpg"
    while True:
        i = next_indexes[person_dir]
        next_indexes[person_dir] = i + 1
        candidate = base_dir / f"{prefix}_{i:05d}{ext}"
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


def nudity_decision(detections: list[dict]) -> tuple[str, str, float, str]:
    """Return confirmed_nude, likely_safe, or needs_review.

    NudeNet is a useful candidate detector, not a final semantic judge. In
    particular, swimsuits can trigger BUTTOCKS_EXPOSED and sheer/skin-toned
    clothing can trigger breast classes. Only strong, non-conflicting evidence
    is allowed to auto-file an image as confirmed nudity.
    """
    return routing_policy.nudity_decision(
        detections,
        class_thresholds=NUDITY_CLASS_THRESHOLDS,
        default_threshold=NUDITY_THRESHOLD,
        uncertain_threshold=NUDITY_UNCERTAIN_THRESHOLD,
        exposed_over_covered_margin=NUDITY_EXPOSED_OVER_COVERED_MARGIN,
    )


def _nudity_category(detections: list[dict]) -> tuple[str | None, str, float]:
    decision, best_class, best_score, _reason = nudity_decision(detections)
    if decision == "confirmed_nude":
        return NUDITY_POSSIBLE_DIR, best_class, best_score
    if decision == "needs_review" and best_class:
        return NUDITY_UNCERTAIN_DIR, best_class, best_score
    return None, best_class, best_score


def _detect_nudity_with_fallback(detector, path: Path) -> list[dict]:
    try:
        return detector.detect(str(path))
    except Exception as first_error:  # noqa: BLE001
        img = imread_unicode(path)
        if img is None:
            raise first_error

        tmp_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                prefix="nudity_input_",
                suffix=".jpg",
                delete=False,
            ) as tmp:
                tmp_path = Path(tmp.name)
            ok, encoded = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 94])
            if not ok:
                raise first_error
            tmp_path.write_bytes(encoded.tobytes())
            return detector.detect(str(tmp_path))
        except Exception as fallback_error:  # noqa: BLE001
            raise fallback_error from first_error
        finally:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except OSError:
                    pass


def maybe_move_to_nudity_subfolder(path: Path,
                                   person_dir: Path,
                                   file_hash: str | None = None,
                                   preclassified_status: str | None = None,
                                   ) -> tuple[Path, str | None]:
    if not path.exists():
        return path, None
    try:
        rel_parts = path.relative_to(person_dir).parts
    except ValueError:
        rel_parts = path.parts
    if (
        PERSON_NUDE_DIR in rel_parts
        or (len(rel_parts) >= 2 and rel_parts[0] == PERSON_PHOTOS_DIR and rel_parts[1] == PERSON_NUDE_DIR)
        or rel_parts[0] == "photos_nude"
        or (len(rel_parts) >= 2 and rel_parts[0] == PERSON_REVIEW_DIR and rel_parts[1] == "nudity_possible")
        or (len(rel_parts) >= 2 and rel_parts[0] == PERSON_REVIEW_DIR and rel_parts[1] == "uncertain_nudity")
        or "_possible_nudity" in rel_parts
        or "_uncertain_nudity" in rel_parts
    ):
        return path, None
    status = preclassified_status or classified_nudity_status(path, file_hash=file_hash)
    if status == "unknown":
        return path, "error"
    if status == "possible":
        target_subdir = NUDITY_POSSIBLE_DIR
    elif status == "uncertain":
        target_subdir = (
            NUDITY_POSSIBLE_DIR
            if ROUTE_UNCERTAIN_NUDITY_TO_NUDE
            else NUDITY_UNCERTAIN_DIR
        )
    else:
        return path, None
    dest = unique_dest(person_dir / target_subdir, path.name)
    sorted_root = person_dir.parent.parent if person_dir.parent.name == "photos_by_person" else DEFAULT_OUTPUT
    operation_ledger.move_path(
        path,
        dest,
        sorted_root=sorted_root,
        operation="sort_photos.nudity_subfolder",
        reason=(
            "move confirmed nudity into per-person photos/nude folder"
            if status == "possible"
            else "move user-routed uncertain nudity into per-person photos/nude folder"
            if target_subdir == NUDITY_POSSIBLE_DIR
            else "move ambiguous explicit nudity evidence into per-person review folder"
        ),
        extra={"nudity_subdir": target_subdir, "nudity_status": status},
    )
    return dest, target_subdir


# ============================================================================
# CACHE
# ============================================================================

def config_fingerprint() -> str:
    parts = [MODEL_NAME, str(DET_SIZE), str(MIN_DET_SCORE),
             str(MIN_FACE_PX), str(MIN_SHARPNESS), str(CROP_SIZE), str(PADDING_RATIO)]
    parts.extend(["recovery=" + str(face_detection.RECOVERY_VERSION),
                  "quality=" + str(face_detection.QUALITY_VERSION),
                  str(RECOVERY_MIN_DET_SCORE), str(RECOVERY_MIN_FACE_PX),
                  str(RECOVERY_MIN_SHARPNESS), str(FACE_PRESENCE_MIN_DET_SCORE),
                  str(FALLBACK_MAX_IMAGE_DIMENSION), model_fingerprint()])
    return "|".join(parts)


def model_fingerprint() -> str:
    root = Path.home() / ".insightface" / "models" / MODEL_NAME
    parts = [MODEL_NAME]
    for model in sorted(root.glob("*.onnx")):
        if model.name in {"genderage.onnx", "1k3d68.onnx", "2d106det.onnx"}:
            continue
        try:
            parts.append(f"{model.name}:{content_identity.content_sha256(model)}")
        except OSError:
            parts.append(f"{model.name}:unavailable")
    return "model=" + hashlib.sha256("|".join(parts).encode()).hexdigest()


def file_signature(path: Path) -> tuple[float, int]:
    st = path.stat()
    return (float(st.st_mtime), int(st.st_size))


def load_cache() -> CacheState:
    if not CACHE_FILE.exists():
        return CacheState(config_fingerprint=config_fingerprint())
    try:
        with CACHE_FILE.open("rb") as f:
            install_pickle_class_aliases()
            data: CacheState = pickle.load(f)
        if data.version != CACHE_VERSION:
            log.warning("Cache version changed. Discarding cache.")
            return CacheState(config_fingerprint=config_fingerprint())
        if data.config_fingerprint != config_fingerprint():
            log.warning("Detection config changed. Invalidating negative results, preserving labeled evidence.")
            preserved = [c for c in data.faces if c.label]
            old_model = str(data.config_fingerprint).split("|")[-1]
            compatible = (str(data.config_fingerprint).split("|")[0] == MODEL_NAME
                          and (not old_model.startswith("model=") or old_model == model_fingerprint()))
            preserved_sources = {c.src_str for c in preserved} if compatible else set()
            data = CacheState(config_fingerprint=config_fingerprint(),
                              file_signatures={p: sig for p, sig in data.file_signatures.items()
                                               if p in preserved_sources}, faces=preserved)
        for face in data.faces:
            if not getattr(face, "pose_label", "") or face.pose_label == "unknown":
                face.pose_label = face_detection.pose_label_from_yaw_proxy(face.yaw_proxy)
        return data
    except Exception as e:  # noqa: BLE001
        log.warning("Cache load failed (%s). Starting fresh.", e)
        return CacheState(config_fingerprint=config_fingerprint())


def normalize_identity_db(db: IdentityDB) -> IdentityDB:
    """Upgrade older single-centroid pickles in memory without data loss."""
    if not hasattr(db, "prototypes"):
        db.prototypes = {}
    if not hasattr(db, "match_thresholds"):
        db.match_thresholds = {}
    if not hasattr(db, "strict_thresholds"):
        db.strict_thresholds = {}
    if not hasattr(db, "intrinsic_match_thresholds"):
        db.intrinsic_match_thresholds = dict(db.match_thresholds)
    if not hasattr(db, "intrinsic_strict_thresholds"):
        db.intrinsic_strict_thresholds = dict(db.strict_thresholds)
    if not hasattr(db, "nearest_impostor_distances"):
        db.nearest_impostor_distances = {}
    if not hasattr(db, "source_signatures"):
        db.source_signatures = {}
    if not hasattr(db, "prototype_sources"):
        db.prototype_sources = {}
    if not hasattr(db, "pose_prototypes"):
        db.pose_prototypes = {}
    if not hasattr(db, "pose_prototype_sources"):
        db.pose_prototype_sources = {}
    if not hasattr(db, "appearance_prototypes"):
        db.appearance_prototypes = {}
    if not hasattr(db, "appearance_prototype_sources"):
        db.appearance_prototype_sources = {}
    if not hasattr(db, "appearance_era_cutoffs"):
        db.appearance_era_cutoffs = {}
    if not hasattr(db, "calibration_version"):
        db.calibration_version = 1
    for name, centroid in db.identities.items():
        db.prototypes.setdefault(name, [np.asarray(centroid, dtype=np.float32)])
        db.pose_prototypes.setdefault(name, {})
        db.pose_prototype_sources.setdefault(name, {})
        db.appearance_prototypes.setdefault(name, {})
        db.appearance_prototype_sources.setdefault(name, {})
        db.appearance_era_cutoffs.setdefault(name, 0.0)
        db.intrinsic_match_thresholds.setdefault(
            name, db.match_thresholds.get(name, AUTO_PERSON_MATCH_DIST))
        db.intrinsic_strict_thresholds.setdefault(
            name, db.strict_thresholds.get(name, AUTO_PERSON_SINGLE_MATCH_DIST))
        db.match_thresholds.setdefault(name, AUTO_PERSON_MATCH_DIST)
        db.strict_thresholds.setdefault(name, AUTO_PERSON_SINGLE_MATCH_DIST)
    db.version = IDENTITY_DB_VERSION
    return db


def load_identity_db() -> IdentityDB | None:
    if not IDENTITY_DB_FILE.exists():
        return None
    try:
        with IDENTITY_DB_FILE.open("rb") as f:
            install_pickle_class_aliases()
            db: IdentityDB = pickle.load(f)
        version = int(getattr(db, "version", 1))
        if version not in {1, IDENTITY_DB_VERSION}:
            log.warning("Identity DB version changed. Rebuild it.")
            return None
        if db.config_fingerprint != config_fingerprint():
            old_model = str(db.config_fingerprint).split("|")[-1]
            if (str(db.config_fingerprint).split("|")[0] != MODEL_NAME
                    or (old_model.startswith("model=") and old_model != model_fingerprint())):
                log.warning("Identity embedding model changed. Rebuild required.")
                return None
            db.config_fingerprint = config_fingerprint()
        return normalize_identity_db(db)
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
        payload_prototypes = payload.get("prototypes", {})
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
            raw_prototypes = payload_prototypes.get(clean, []) if isinstance(payload_prototypes, dict) else []
            db.prototypes[clean] = (
                [_l2norm(np.asarray(value, dtype=np.float32)[None, :])[0] for value in raw_prototypes]
                or [db.identities[clean]]
            )
            db.intrinsic_match_thresholds[clean] = AUTO_PERSON_MATCH_DIST
            db.intrinsic_strict_thresholds[clean] = AUTO_PERSON_SINGLE_MATCH_DIST
            db.match_thresholds[clean] = AUTO_PERSON_MATCH_DIST
            db.strict_thresholds[clean] = AUTO_PERSON_SINGLE_MATCH_DIST
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
    merged.prototypes.update(primary.prototypes)
    merged.intrinsic_match_thresholds.update(primary.intrinsic_match_thresholds)
    merged.intrinsic_strict_thresholds.update(primary.intrinsic_strict_thresholds)
    merged.match_thresholds.update(primary.match_thresholds)
    merged.strict_thresholds.update(primary.strict_thresholds)
    merged.nearest_impostor_distances.update(primary.nearest_impostor_distances)
    merged.source_signatures.update(primary.source_signatures)
    merged.prototype_sources.update(primary.prototype_sources)
    merged.pose_prototypes.update(primary.pose_prototypes)
    merged.pose_prototype_sources.update(primary.pose_prototype_sources)
    merged.appearance_prototypes.update(primary.appearance_prototypes)
    merged.appearance_prototype_sources.update(primary.appearance_prototype_sources)
    merged.appearance_era_cutoffs.update(primary.appearance_era_cutoffs)
    added = 0
    for name, centroid in extra.identities.items():
        if name in merged.identities:
            continue
        merged.identities[name] = centroid
        merged.source_counts[name] = extra.source_counts.get(name, 0)
        merged.prototypes[name] = extra.prototypes.get(name, [centroid])
        merged.intrinsic_match_thresholds[name] = extra.intrinsic_match_thresholds.get(
            name, extra.match_thresholds.get(name, AUTO_PERSON_MATCH_DIST))
        merged.intrinsic_strict_thresholds[name] = extra.intrinsic_strict_thresholds.get(
            name, extra.strict_thresholds.get(name, AUTO_PERSON_SINGLE_MATCH_DIST))
        merged.match_thresholds[name] = extra.match_thresholds.get(name, AUTO_PERSON_MATCH_DIST)
        merged.strict_thresholds[name] = extra.strict_thresholds.get(name, AUTO_PERSON_SINGLE_MATCH_DIST)
        if name in extra.nearest_impostor_distances:
            merged.nearest_impostor_distances[name] = extra.nearest_impostor_distances[name]
        if name in extra.source_signatures:
            merged.source_signatures[name] = extra.source_signatures[name]
        if name in extra.prototype_sources:
            merged.prototype_sources[name] = list(extra.prototype_sources[name])
        if name in extra.pose_prototypes:
            merged.pose_prototypes[name] = {
                pose: list(values)
                for pose, values in extra.pose_prototypes[name].items()
            }
        if name in extra.pose_prototype_sources:
            merged.pose_prototype_sources[name] = {
                pose: list(values)
                for pose, values in extra.pose_prototype_sources[name].items()
            }
        if name in extra.appearance_prototypes:
            merged.appearance_prototypes[name] = {
                label: list(values)
                for label, values in extra.appearance_prototypes[name].items()
            }
        if name in extra.appearance_prototype_sources:
            merged.appearance_prototype_sources[name] = {
                label: list(values)
                for label, values in extra.appearance_prototype_sources[name].items()
            }
        if name in extra.appearance_era_cutoffs:
            merged.appearance_era_cutoffs[name] = float(extra.appearance_era_cutoffs[name])
        added += 1
    if added:
        log.info("Added %d people from reference DB to matcher.", added)
    return merged


def write_identity_db(db: IdentityDB, destination: Path) -> None:
    db = normalize_identity_db(db)
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(db, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(destination)


def save_identity_db(db: IdentityDB) -> None:
    """Atomically promote a complete identity DB and retain the prior version."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if IDENTITY_DB_FILE.exists():
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = IDENTITY_DB_FILE.with_name(
            f"{IDENTITY_DB_FILE.name}.bak.profile_refresh_{stamp}"
        )
        shutil.copy2(IDENTITY_DB_FILE, backup)
    write_identity_db(db, IDENTITY_DB_FILE)


def identity_source_images(person_dir: Path) -> list[Path]:
    photos_dir = person_dir / PERSON_PHOTOS_DIR
    scan_root = photos_dir if photos_dir.exists() else person_dir
    images = [
        path for path in iter_images(scan_root, excluded_dir_names=set())
        if not any(
            part in {
                LEGACY_DUPLICATES_DIR,
                BLURRED_DIR,
                PERSON_REVIEW_DIR,
                "all",
                "_smart_albums",
                "_smart_albums_v2",
                "_smart_albums_simple_preview",
            }
            for part in path.relative_to(person_dir).parts[:-1]
        )
    ]
    return sorted(images, key=lambda path: (
        len(path.relative_to(person_dir).parts),
        str(path).casefold(),
    ))


def identity_source_signature(person_dir: Path, images: list[Path]) -> str:
    digest = hashlib.sha256()
    for image in images:
        try:
            stat = image.stat()
            relative = image.relative_to(person_dir).as_posix()
            digest.update(relative.encode("utf-8", errors="surrogateescape"))
            digest.update(f"\0{stat.st_size}\0{stat.st_mtime_ns}\n".encode("ascii"))
        except (OSError, ValueError):
            continue
    return digest.hexdigest()


def evenly_sample_paths(paths: list[Path], limit: int) -> list[Path]:
    if limit <= 0 or len(paths) <= limit:
        return paths
    indexes = np.linspace(0, len(paths) - 1, num=limit, dtype=int)
    return [paths[int(index)] for index in sorted(set(indexes.tolist()))]


def copy_identity_profile(source: IdentityDB, destination: IdentityDB, name: str) -> None:
    destination.identities[name] = source.identities[name]
    destination.source_counts[name] = source.source_counts.get(name, 0)
    destination.prototypes[name] = list(source.prototypes.get(name, [source.identities[name]]))
    destination.intrinsic_match_thresholds[name] = source.intrinsic_match_thresholds.get(
        name, source.match_thresholds.get(name, AUTO_PERSON_MATCH_DIST))
    destination.intrinsic_strict_thresholds[name] = source.intrinsic_strict_thresholds.get(
        name, source.strict_thresholds.get(name, AUTO_PERSON_SINGLE_MATCH_DIST))
    destination.match_thresholds[name] = source.match_thresholds.get(name, AUTO_PERSON_MATCH_DIST)
    destination.strict_thresholds[name] = source.strict_thresholds.get(
        name, AUTO_PERSON_SINGLE_MATCH_DIST)
    if name in source.nearest_impostor_distances:
        destination.nearest_impostor_distances[name] = source.nearest_impostor_distances[name]
    if name in source.source_signatures:
        destination.source_signatures[name] = source.source_signatures[name]
    if name in source.prototype_sources:
        destination.prototype_sources[name] = list(source.prototype_sources[name])
    if name in source.pose_prototypes:
        destination.pose_prototypes[name] = {
            pose: list(values)
            for pose, values in source.pose_prototypes[name].items()
        }
    if name in source.pose_prototype_sources:
        destination.pose_prototype_sources[name] = {
            pose: list(values)
            for pose, values in source.pose_prototype_sources[name].items()
        }
    if name in source.appearance_prototypes:
        destination.appearance_prototypes[name] = {
            label: list(values)
            for label, values in source.appearance_prototypes[name].items()
        }
    if name in source.appearance_prototype_sources:
        destination.appearance_prototype_sources[name] = {
            label: list(values)
            for label, values in source.appearance_prototype_sources[name].items()
        }
    if name in source.appearance_era_cutoffs:
        destination.appearance_era_cutoffs[name] = float(source.appearance_era_cutoffs[name])


def calibrate_identity_db_against_impostors(db: IdentityDB) -> IdentityDB:
    db = normalize_identity_db(db)
    for name in db.identities:
        consensus, strict, nearest = identity_profiles.impostor_aware_thresholds(
            name,
            db.identities,
            db.prototypes,
            base_consensus_distance=db.intrinsic_match_thresholds.get(
                name, AUTO_PERSON_MATCH_DIST),
            base_strict_distance=db.intrinsic_strict_thresholds.get(
                name, AUTO_PERSON_SINGLE_MATCH_DIST),
        )
        db.match_thresholds[name] = consensus
        db.strict_thresholds[name] = strict
        db.nearest_impostor_distances[name] = nearest
    db.calibration_version = IDENTITY_CALIBRATION_VERSION
    return db


def build_identity_db_from_person_folders(people_dir: Path,
                                          force_rebuild: bool = False) -> IdentityDB:
    from tqdm import tqdm

    people_dir = people_dir.expanduser().resolve()
    if not people_dir.exists():
        log.warning("Identity DB source folder does not exist: %s", people_dir)
        return IdentityDB(config_fingerprint=config_fingerprint())

    person_dirs = sorted([p for p in people_dir.iterdir() if p.is_dir()],
                         key=lambda p: p.name.lower())
    existing = None if force_rebuild else load_identity_db()
    partial: IdentityDB | None = None
    if not force_rebuild and IDENTITY_DB_BUILD_FILE.is_file():
        try:
            with IDENTITY_DB_BUILD_FILE.open("rb") as handle:
                install_pickle_class_aliases()
                candidate_partial = normalize_identity_db(pickle.load(handle))
            if candidate_partial.config_fingerprint == config_fingerprint():
                partial = candidate_partial
                log.info(
                    "Resuming identity profile checkpoint: %d completed people.",
                    len(partial.identities),
                )
        except Exception as exc:  # noqa: BLE001
            log.warning("Ignoring unreadable identity build checkpoint: %s", exc)
    db = IdentityDB(config_fingerprint=config_fingerprint())
    work: list[tuple[Path, list[Path], str, list[Path]]] = []
    rejected_references = identity_hard_negatives.ReferenceRejections(IDENTITY_HARD_NEGATIVES_FILE)
    for person_dir in person_dirs:
        name = person_dir.name
        if name.startswith(("_", ".")) or not is_real_person_label(name):
            continue
        images = identity_source_images(person_dir)
        confirmed_paths = identity_confirmations.paths_for_person(
            IDENTITY_CONFIRMATIONS_FILE, name, people_dir
        )
        source_signature = identity_source_signature(person_dir, images)
        confirmation_signature = identity_confirmations.signature_for_person(
            IDENTITY_CONFIRMATIONS_FILE, name, people_dir
        )
        signature = (
            hashlib.sha256(
                f"{source_signature}:{confirmation_signature}".encode("ascii")
            ).hexdigest()
            if confirmed_paths
            else source_signature
        )
        rejection_signature = rejected_references.signature(name)
        if rejection_signature:
            signature = hashlib.sha256(
                f"{signature}:reference-rejections-v1:{rejection_signature}".encode("ascii")
            ).hexdigest()
        if (
            partial is not None
            and name in partial.identities
            and partial.source_signatures.get(name) == signature
        ):
            copy_identity_profile(partial, db, name)
            continue
        if (
            existing is not None
            and existing.calibration_version == IDENTITY_CALIBRATION_VERSION
            and name in existing.identities
            and existing.source_signatures.get(name) == signature
        ):
            copy_identity_profile(existing, db, name)
            continue
        work.append((person_dir, images, signature, confirmed_paths))

    if not work:
        if len(db.nearest_impostor_distances) != len(db.identities):
            calibrate_identity_db_against_impostors(db)
            save_identity_db(db)
            log.info(
                "Identity DB calibration evidence upgraded: %d people.",
                len(db.identities),
            )
            return db
        log.info("Identity DB already current: %d people.", len(db.identities))
        return db

    cache = load_cache()
    cached_signatures_by_path = {
        os.path.realpath(path): signature
        for path, signature in cache.file_signatures.items()
    }
    cached_faces_by_path: dict[str, list[CachedFace]] = defaultdict(list)
    for cached_face in cache.faces:
        cached_faces_by_path[os.path.realpath(cached_face.src_str)].append(cached_face)

    app = None
    rebuilt = 0
    for person_dir, images, signature, confirmed_paths in tqdm(
        work, desc="Refreshing identity profiles", unit="person"
    ):
        name = person_dir.name
        pool_limit = (
            max(IDENTITY_CANDIDATE_POOL_MIN, IDENTITY_MAX_IMAGES_PER_PERSON)
            if IDENTITY_MAX_IMAGES_PER_PERSON > 0 else 0
        )
        candidate_images = evenly_sample_paths(images, pool_limit)
        candidate_keys = {os.path.realpath(str(path)) for path in candidate_images}
        for confirmed_path in confirmed_paths:
            confirmed_key = os.path.realpath(str(confirmed_path))
            if confirmed_key not in candidate_keys:
                candidate_images.append(confirmed_path)
                candidate_keys.add(confirmed_key)
        confirmed_keys = {os.path.realpath(str(path)) for path in confirmed_paths}
        confirmed_examples = identity_confirmations.examples_for_person(
            IDENTITY_CONFIRMATIONS_FILE, name, people_dir)
        samples: list[identity_profiles.ReferenceSample] = []
        trusted_samples: list[identity_profiles.ReferenceSample] = []
        for image in candidate_images:
            image_key = str(image)
            canonical_image_key = os.path.realpath(image_key)
            try:
                current_signature = file_signature(image)
            except OSError:
                continue
            cache_is_current = cached_signatures_by_path.get(canonical_image_key) == current_signature
            cached_candidates = (
                cached_faces_by_path.get(canonical_image_key, []) if cache_is_current else []
            )
            if cache_is_current:
                matching_labels = [
                    face for face in cached_candidates
                    if face.label and face.label.casefold() == name.casefold()
                ]
                usable_faces = matching_labels or cached_candidates
            elif existing is not None and name in existing.identities:
                # A changed source that has not reached the detector cache yet
                # must not force a long native-model session during profile
                # maintenance. The existing profile remains active until the
                # normal bounded detector worker analyzes that source.
                continue
            else:
                if app is None:
                    app = _build_app()
                usable_faces = _detect_one_image(image, app)
            example = confirmed_examples.get(Path(canonical_image_key))
            if example is not None:
                # An explicit selection overrides old folder/cache labels, but
                # never authorizes the other faces in a confirmed group photo.
                usable_faces = identity_confirmations.selected_faces(
                    example, cached_candidates if cache_is_current else usable_faces)
            for face in usable_faces:
                if rejected_references.rejects(name, face.embedding):
                    continue
                sample = identity_profiles.ReferenceSample(
                    source=image_key,
                    embedding=np.asarray(face.embedding, dtype=np.float32),
                    quality=float(face.quality),
                    pose_label=str(getattr(face, "pose_label", "unknown") or "unknown"),
                    lighting_label=appearance_profiles.lighting_label(
                        bytes(getattr(face, "crop_jpeg", b"") or b"")
                    ),
                    capture_timestamp=appearance_profiles.capture_timestamp(image),
                )
                samples.append(sample)
                if example is not None and any(
                    selected is face for selected in identity_confirmations.selected_faces(example, usable_faces)
                ):
                    trusted_samples.append(sample)

        dominant = identity_profiles.dominant_identity_samples(samples)
        centroid_samples = identity_profiles.select_diverse_samples(
            dominant,
            limit=max(1, IDENTITY_MAX_PROTOTYPES_PER_PERSON),
        )
        selected = identity_profiles.select_profile_prototypes(
            dominant,
            trusted_samples,
            limit=max(1, IDENTITY_MAX_PROTOTYPES_PER_PERSON),
            trusted_limit=min(
                IDENTITY_MAX_TRUSTED_PROTOTYPES_PER_PERSON,
                max(1, IDENTITY_MAX_PROTOTYPES_PER_PERSON // 2),
            ),
        )
        if not selected:
            log.warning("No usable identity references for %s; keeping it out of auto-match.", name)
            if (existing is not None and name in existing.identities
                    and not rejected_references.signature(name)):
                copy_identity_profile(existing, db, name)
            continue
        # Keep the centroid tied to the dominant, repeatedly observed identity.
        # Explicit confirmations extend pose coverage only through prototypes.
        centroid = identity_profiles.weighted_centroid(centroid_samples or selected)
        prototypes = [identity_profiles.normalize_vector(sample.embedding) for sample in selected]
        # Old cached detections retain a safe ``profile_unknown`` bucket.
        # Newly analyzed faces carry signed left/right pose labels. Avoiding a
        # mass redetection here keeps profile refresh memory-bounded.
        pose_evidence = list(selected)
        consensus_threshold, strict_threshold = identity_profiles.calibrated_thresholds(
            dominant,
            centroid,
            prototypes,
            maximum_consensus_distance=AUTO_PERSON_MATCH_DIST,
        )
        db.identities[name] = centroid
        db.prototypes[name] = prototypes
        db.prototype_sources[name] = [sample.source for sample in selected]
        pose_prototypes: dict[str, list[np.ndarray]] = {}
        pose_prototype_sources: dict[str, list[str]] = {}
        for pose in ("frontal", "left_profile", "right_profile", "profile_unknown"):
            pose_samples = [sample for sample in pose_evidence if sample.pose_label == pose]
            pose_selected = identity_profiles.select_diverse_samples(pose_samples, limit=2)
            if pose_selected:
                pose_prototypes[pose] = [
                    identity_profiles.normalize_vector(sample.embedding)
                    for sample in pose_selected
                ]
                pose_prototype_sources[pose] = [sample.source for sample in pose_selected]
        db.pose_prototypes[name] = pose_prototypes
        db.pose_prototype_sources[name] = pose_prototype_sources
        person_era_cutoff = appearance_profiles.era_cutoff(
            [sample.capture_timestamp for sample in dominant]
        )
        appearance_values: dict[str, list[np.ndarray]] = {}
        appearance_sources: dict[str, list[str]] = {}
        for label in ("low_light", "normal_light", "era_older", "era_newer"):
            matching = [
                sample for sample in pose_evidence
                if label in appearance_profiles.labels(
                    light=sample.lighting_label,
                    timestamp=sample.capture_timestamp,
                    person_era_cutoff=person_era_cutoff,
                )
            ]
            appearance_selected = identity_profiles.select_diverse_samples(
                matching, limit=2
            )
            if appearance_selected:
                appearance_values[label] = [
                    identity_profiles.normalize_vector(sample.embedding)
                    for sample in appearance_selected
                ]
                appearance_sources[label] = [sample.source for sample in appearance_selected]
        db.appearance_prototypes[name] = appearance_values
        db.appearance_prototype_sources[name] = appearance_sources
        db.appearance_era_cutoffs[name] = person_era_cutoff
        db.source_counts[name] = identity_profiles.source_count(dominant)
        db.intrinsic_match_thresholds[name] = consensus_threshold
        db.intrinsic_strict_thresholds[name] = strict_threshold
        db.match_thresholds[name] = consensus_threshold
        db.strict_thresholds[name] = strict_threshold
        db.source_signatures[name] = signature
        rebuilt += 1
        if rebuilt % 5 == 0:
            # Checkpoints are deliberately not made active. If the refresh is
            # interrupted, normal sorting continues with the previous complete
            # database rather than a partially rebuilt one.
            write_identity_db(db, IDENTITY_DB_BUILD_FILE)

    calibrate_identity_db_against_impostors(db)
    if existing is not None and existing.identities:
        import evaluation_enrollment
        import identity_evaluation

        evaluation_enrollment.backfill_confirmations(
            identity_confirmations.load(IDENTITY_CONFIRMATIONS_FILE),
            cache.faces,
            path=evaluation_enrollment.DEFAULT_PATH,
        )
        allowed, gate = identity_evaluation.activation_gate(
            db,
            existing,
            cache,
            confirmed_set=evaluation_enrollment.DEFAULT_PATH,
        )
        gate_path = (
            pipeline_paths.SOURCE_REVIEW
            / "identity_evaluation"
            / "latest_profile_activation_gate.json"
        )
        gate_path.parent.mkdir(parents=True, exist_ok=True)
        gate_path.write_text(json.dumps(gate, indent=2, default=str) + "\n", encoding="utf-8")
        if not allowed:
            IDENTITY_DB_BUILD_FILE.unlink(missing_ok=True)
            log.error(
                "Identity profile refresh blocked by activation gate: %s. Report: %s",
                "; ".join(str(value) for value in gate["failures"]),
                gate_path,
            )
            return existing
        log.info("Identity activation gate passed: %s", gate_path)
    save_identity_db(db)
    IDENTITY_DB_BUILD_FILE.unlink(missing_ok=True)
    log.info(
        "Identity DB saved: %s (%d people, %d refreshed, %d reused)",
        IDENTITY_DB_FILE,
        len(db.identities),
        rebuilt,
        len(db.identities) - rebuilt,
    )
    return db


def save_cache(cache: CacheState) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = CACHE_FILE.with_suffix(".pkl.tmp")
    with tmp.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(CACHE_FILE)


def analysis_index_file() -> Path:
    if FINGERPRINT_CACHE_FILE != DEFAULT_FINGERPRINT_CACHE_FILE:
        return FINGERPRINT_CACHE_FILE.parent / "analysis_index.sqlite3"
    return pipeline_paths.ANALYSIS_INDEX


def cached_face_to_index_record(face: CachedFace) -> analysis_index.DetectionRecord:
    embedding = np.asarray(face.embedding, dtype=np.float32).reshape(-1)
    phash = np.asarray(face.image_phash, dtype=np.uint8).reshape(-1)
    return analysis_index.DetectionRecord(
        face_index=int(face.face_index),
        det_score=float(face.det_score),
        bbox_size=float(face.bbox_size),
        sharpness=float(face.sharpness),
        yaw_proxy=float(face.yaw_proxy),
        quality=float(face.quality),
        embedding=embedding.tobytes(),
        embedding_dimension=int(embedding.size),
        image_phash=np.packbits(phash).tobytes(),
        image_phash_bits=int(phash.size),
        crop_jpeg=bytes(face.crop_jpeg),
        label=face.label,
        pose_label=str(getattr(face, "pose_label", "unknown") or "unknown"),
        bbox=tuple(getattr(face, "bbox", ())),
        keypoints=tuple(getattr(face, "keypoints", ())),
    )


def index_record_to_cached_face(
    path: Path,
    record: analysis_index.DetectionRecord,
) -> CachedFace:
    embedding = np.frombuffer(record.embedding, dtype=np.float32).copy()
    if record.embedding_dimension > 0:
        embedding = embedding[:record.embedding_dimension]
    packed_phash = np.frombuffer(record.image_phash, dtype=np.uint8)
    image_phash = np.unpackbits(packed_phash)[:record.image_phash_bits].astype(bool)
    return CachedFace(
        src_str=str(path),
        face_index=int(record.face_index),
        det_score=float(record.det_score),
        bbox_size=float(record.bbox_size),
        sharpness=float(record.sharpness),
        yaw_proxy=float(record.yaw_proxy),
        quality=float(record.quality),
        embedding=embedding,
        image_phash=image_phash,
        crop_jpeg=bytes(record.crop_jpeg),
        label=record.label,
        pose_label=(
            face_detection.pose_label_from_yaw_proxy(record.yaw_proxy)
            if not record.pose_label or record.pose_label == "unknown"
            else str(record.pose_label)
        ),
        bbox=record.bbox,
        keypoints=record.keypoints,
    )


def persist_detection_batch(
    paths: Iterable[Path],
    faces: Iterable[CachedFace],
    diagnostics: dict[str, str],
    index_path: Path | None,
    fingerprints: dict[str, dict[str, int | str]] | None = None,
) -> None:
    if index_path is None:
        return
    by_source: dict[str, list[CachedFace]] = defaultdict(list)
    for face in faces:
        by_source[os.path.realpath(face.src_str)].append(face)
    try:
        with analysis_index.AnalysisIndex(index_path) as index:
            for path in paths:
                canonical = os.path.realpath(str(path))
                source_faces = by_source.get(canonical, [])
                fingerprint_data = (fingerprints or {}).get(str(path))
                if fingerprint_data is None:
                    fingerprint_data = (fingerprints or {}).get(canonical)
                if fingerprint_data is not None:
                    index.upsert_fingerprint(
                        path,
                        analysis_index.AssetFingerprint(
                            sha256=str(fingerprint_data["sha256"]),
                            pixel_sha256=str(fingerprint_data["pixel_sha256"]),
                            phash=int(fingerprint_data["phash"]),
                            width=int(fingerprint_data["width"]),
                            height=int(fingerprint_data["height"]),
                        ),
                    )
                status = diagnostics.get(
                    str(path),
                    "accepted_face" if source_faces else "no_usable_face",
                )
                index.replace_detections(
                    path,
                    config_fingerprint(),
                    status,
                    [cached_face_to_index_record(face) for face in source_faces],
                    expected_sha256=str(fingerprint_data["sha256"]) if fingerprint_data else None,
                )
    except Exception as exc:  # noqa: BLE001
        log.warning("Could not mirror detection batch into SQLite: %s", exc)


def cached_to_record(c: CachedFace) -> FaceRecord:
    return FaceRecord(
        src=Path(c.src_str), face_index=c.face_index, det_score=c.det_score,
        bbox_size=c.bbox_size, sharpness=c.sharpness, yaw_proxy=c.yaw_proxy,
        embedding=c.embedding, crop_jpeg=c.crop_jpeg, image_phash=c.image_phash,
        quality=c.quality, prior_label=c.label,
        pose_label=str(getattr(c, "pose_label", "unknown") or "unknown"),
        bbox=tuple(getattr(c, "bbox", ())),
        keypoints=tuple(getattr(c, "keypoints", ())),
        content_sha256=str(getattr(c, "content_sha256", "")),
    )


def record_to_cached(rec: FaceRecord, label: str | None) -> CachedFace:
    return CachedFace(
        src_str=str(rec.src), face_index=rec.face_index, det_score=rec.det_score,
        bbox_size=rec.bbox_size, sharpness=rec.sharpness, yaw_proxy=rec.yaw_proxy,
        quality=rec.quality, embedding=rec.embedding, image_phash=rec.image_phash,
        crop_jpeg=rec.crop_jpeg, label=label,
        bbox=tuple(getattr(rec, "bbox", ())),
        keypoints=tuple(getattr(rec, "keypoints", ())),
        content_sha256=str(getattr(rec, "content_sha256", "")),
        pose_label=str(getattr(rec, "pose_label", "unknown") or "unknown"),
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
        recovery_metadata=[(getattr(r, "identity_review_reason", ""),
                            getattr(r, "recovery_assigned", False)) for r in records],
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
            install_pickle_class_aliases()
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


def restore_recovery_state(record: FaceRecord, state: LabelingState, index: int) -> None:
    metadata = getattr(state, "recovery_metadata", [])
    if index < len(metadata):
        record.identity_review_reason, record.recovery_assigned = metadata[index]


def save_remaining_labeling_state(records: list[FaceRecord],
                                  name_map: dict[int, str],
                                  output_dir: Path,
                                  input_dir: Path) -> int:
    """Preserve only still-unlabeled clusters after a partial finalize.

    Already-entered labels have been written to person folders by this point.
    Keeping them in the saved session makes repeated Finish/Review actions copy
    the same sources again.
    """
    remaining_cids = {
        cid for cid, name in name_map.items()
        if cid != -1 and name.startswith("person_")
    }
    if not remaining_cids:
        clear_labeling_state()
        return 0

    remaining_records = [r for r in records if r.cluster_id in remaining_cids]
    remaining_name_map = {
        cid: name for cid, name in name_map.items()
        if cid in remaining_cids
    }
    save_labeling_state(remaining_records, remaining_name_map, output_dir, input_dir)
    return len(remaining_name_map)


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

@contextlib.contextmanager
def quiet_model_startup():
    """Hide noisy third-party model startup logs while keeping progress output."""
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            yield


def _build_app(det_size: tuple[int, int] | None = None):
    from insightface.app import FaceAnalysis
    with quiet_model_startup():
        # Sorting needs only face boxes/keypoints and identity embeddings.
        # Loading age/gender and extra landmark models makes every image slower
        # without contributing to classification.
        app = FaceAnalysis(
            name=MODEL_NAME,
            allowed_modules=["detection", "recognition"],
            providers=PROVIDERS,
        )
        app.prepare(
            ctx_id=0,
            det_size=det_size or DET_SIZE,
            det_thresh=MIN_DET_SCORE * 0.8,
        )
    return app


def _fallback_detection_views(img: np.ndarray) -> Iterable[np.ndarray]:
    """Yield alternate views only after the normal detector pass fails.

    Overlapping tiles make a small face occupy more of the detector input
    without upscaling or modifying the original file. Rotation is reserved for
    smaller files where regional detection cannot help and orientation metadata
    is more likely to be absent.
    """
    yield from face_detection.fallback_detection_views(
        img, max_dimension=FALLBACK_MAX_IMAGE_DIMENSION,
    )


def _cached_face_from_detection(src: Path,
                                img: np.ndarray,
                                detection,
                                image_phash: np.ndarray,
                                face_index: int,
                                *,
                                min_score: float,
                                min_face_px: float,
                                min_sharpness: float,
                                ) -> tuple[CachedFace | None, str]:
    score = float(getattr(detection, "det_score", 0.0))
    if score < min_score:
        return None, "face_below_detection_confidence"
    bbox = np.asarray(detection.bbox, dtype=np.float32)
    x1, y1, x2, y2 = bbox.tolist()
    bbox_size = float(min(x2 - x1, y2 - y1))
    if bbox_size < min_face_px:
        return None, "face_too_small"
    height, width = img.shape[:2]
    nx1, ny1, nx2, ny2 = square_pad_bbox(
        x1, y1, x2, y2, width, height, PADDING_RATIO
    )
    crop = img[ny1:ny2, nx1:nx2]
    if crop.size == 0:
        return None, "invalid_face_crop"
    face_sharpness = sharpness(crop)
    if face_sharpness < min_sharpness:
        return None, "face_too_blurry"
    embedding = getattr(detection, "normed_embedding", None)
    if embedding is None:
        return None, "missing_face_embedding"
    keypoints = getattr(detection, "kps", None)
    yaw = yaw_proxy_from_kps(keypoints, bbox)
    pose_label = face_detection.pose_label_from_keypoints(keypoints, bbox)
    if CROP_SIZE:
        crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)
    ok, jpeg = cv2.imencode(
        ".jpg", crop, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY]
    )
    if not ok:
        return None, "face_crop_encode_failed"
    return CachedFace(
        src_str=str(src),
        face_index=face_index,
        det_score=score,
        bbox_size=bbox_size,
        sharpness=face_sharpness,
        yaw_proxy=yaw,
        quality=quality_score_from_parts(score, bbox_size, face_sharpness, yaw),
        embedding=np.asarray(embedding, dtype=np.float32),
        image_phash=image_phash,
        crop_jpeg=jpeg.tobytes(),
        label=None,
        pose_label=pose_label,
        bbox=tuple(map(float, bbox)),
        keypoints=tuple(tuple(map(float, p)) for p in keypoints) if keypoints is not None else (),
    ), "accepted"


def _detect_one_image(src: Path,
                      app,
                      diagnostics: dict[str, str] | None = None,
                      fallback_app=None,
                      fingerprints: dict[str, dict[str, int | str]] | None = None,
                      ) -> list[CachedFace]:
    out: list[CachedFace] = []
    strict_rejected: Counter[str] = Counter()
    recovery_rejected: Counter[str] = Counter()

    def record_status(status: str) -> None:
        if diagnostics is not None:
            diagnostics[str(src)] = status

    try:
        with suppress_native_stderr():
            decoded = asset_processing.load_decoded_asset(src)
        if decoded is None:
            record_status("unreadable_image")
            return out
        img = decoded.bgr
        img_phash = decoded.phash_bits
        if fingerprints is not None:
            fingerprints[str(src)] = {
                "sha256": decoded.sha256,
                "pixel_sha256": decoded.pixel_sha256,
                "phash": phash_to_int(decoded.phash_bits),
                "width": decoded.width,
                "height": decoded.height,
            }
        faces = app.get(img)
        for i, f in enumerate(faces):
            cached, reason = _cached_face_from_detection(
                src, img, f, img_phash, i,
                min_score=MIN_DET_SCORE,
                min_face_px=MIN_FACE_PX,
                min_sharpness=MIN_SHARPNESS,
            )
            if cached is None:
                strict_rejected[reason] += 1
            else:
                cached.content_sha256 = decoded.sha256
                out.append(cached)
        if out and not strict_rejected:
            record_status("accepted_face")
            return out

        # A soft/small face is still a face. Keep its embedding at a lower
        # quality score so downstream identity consensus remains conservative.
        saw_raw_face = bool(faces)
        recovery_index = 0

        def recover_from_view(view: np.ndarray, view_faces: list, transform=None) -> None:
            nonlocal recovery_index
            for f in view_faces:
                cached, reason = _cached_face_from_detection(
                    src, view, f, img_phash, recovery_index,
                    min_score=RECOVERY_MIN_DET_SCORE,
                    min_face_px=RECOVERY_MIN_FACE_PX,
                    min_sharpness=RECOVERY_MIN_SHARPNESS,
                )
                recovery_index += 1
                if cached is None:
                    recovery_rejected[reason] += 1
                    continue
                cached.bbox, cached.keypoints = face_detection.original_geometry(
                    cached.bbox, cached.keypoints, np.eye(3) if transform is None else transform)
                if any(face_detection.same_detection(existing.bbox, cached.bbox) for existing in out):
                    continue
                cached.face_index = max((face.face_index for face in out), default=-1) + 1
                cached.content_sha256 = decoded.sha256
                out.append(cached)

        recover_from_view(img, faces)
        if not out:
            alternate_app = fallback_app or app
            alternate_detector = getattr(alternate_app, "det_model", None)
            original_threshold = getattr(alternate_detector, "det_thresh", None)
            if original_threshold is not None:
                alternate_detector.det_thresh = min(
                    float(original_threshold), FACE_PRESENCE_MIN_DET_SCORE
                )
            try:
                successful_kind = None
                for frame in face_detection.fallback_detection_frames(img, max_dimension=FALLBACK_MAX_IMAGE_DIMENSION):
                    if successful_kind is not None and frame.kind != successful_kind:
                        break
                    view_faces = alternate_app.get(frame.image)
                    saw_raw_face = saw_raw_face or bool(view_faces)
                    recover_from_view(frame.image, view_faces, frame.to_original)
                    if out:
                        successful_kind = frame.kind
                        if frame.kind != "tile":
                            break
            finally:
                if original_threshold is not None:
                    alternate_detector.det_thresh = original_threshold

        if out:
            record_status("accepted_face_recovery")
        elif saw_raw_face:
            reason_counts = recovery_rejected or strict_rejected
            reason = (
                reason_counts.most_common(1)[0][0]
                if reason_counts else "face_failed_recovery_thresholds"
            )
            record_status(f"face_quality_review:{reason}")
        else:
            record_status("no_face_detected")
    except Exception as e:  # noqa: BLE001
        record_status(f"detector_error:{type(e).__name__}")
        sys.stderr.write(f"Skipped {src.name}: {e}\n")
    return out


def validate_detection_batch(paths, faces, fingerprints, diagnostics):
    """Do not attach worker results to a source that changed during inference."""
    accepted, keys = [], set()
    for path in paths:
        expected = fingerprints.get(str(path), {}).get("sha256")
        try:
            if expected and content_identity.content_sha256(path) != expected:
                raise OSError("source changed during detection")
            if not path.is_file():
                raise OSError("source disappeared during detection")
        except OSError:
            diagnostics[str(path)] = "processing_failed:source_changed"
            continue
        accepted.append(path)
        keys.add(os.path.realpath(str(path)))
    return accepted, [face for face in faces if os.path.realpath(face.src_str) in keys]


def run_detection_worker(job_path: Path) -> int:
    global DET_SIZE
    warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\..*")
    warnings.filterwarnings("ignore", message=r".*`estimate` is deprecated.*", category=FutureWarning)
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
    diagnostics: dict[str, str] = {}
    fingerprints: dict[str, dict[str, int | str]] = {}
    for s in tqdm(input_paths, desc="Worker", unit="img"):
        faces = _detect_one_image(
            Path(s), app, diagnostics=diagnostics, fingerprints=fingerprints,
        )
        all_faces.extend(faces)

    with output_path.open("wb") as f:
        pickle.dump(
            {
                "faces": all_faces,
                "diagnostics": diagnostics,
                "fingerprints": fingerprints,
            },
            f,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return 0


def detect_in_batches_subprocess(new_images: list[Path],
                                 cache: CacheState,
                                 batch_size: int,
                                 workers: int = 1,
                                 index_path: Path | None = None,
                                 ) -> tuple[list[FaceRecord], set[Path], dict[str, str]]:
    all_new_records: list[FaceRecord] = []
    processed_sources: set[Path] = set()
    diagnostics: dict[str, str] = {}
    total = len(new_images)
    if total == 0:
        return all_new_records, processed_sources, diagnostics

    n_batches = (total + batch_size - 1) // batch_size
    workers = max(1, int(workers))
    log.info("Detection plan: %d image(s) in %d subprocess batch(es) of up to %d "
             "(workers=%d).", total, n_batches, batch_size, workers)

    tmp_dir = Path(tempfile.mkdtemp(prefix="sort_photos_"))
    log.info("Worker scratch dir: %s", tmp_dir)

    try:
        script_path = Path(__file__).resolve()
        worker_env = os.environ.copy()
        for key, value in DETECTION_WORKER_ENV_LIMITS.items():
            worker_env.setdefault(key, value)

        def is_transient_fork_error(exc: OSError) -> bool:
            return isinstance(exc, BlockingIOError) or getattr(exc, "errno", None) in {11, 35}

        def run_worker_with_retry(cmd: list[str], batch_number: int) -> subprocess.CompletedProcess[str] | None:
            for attempt in range(1, SUBPROCESS_FORK_RETRIES + 1):
                try:
                    return subprocess.run(
                        cmd,
                        check=False,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        env=worker_env,
                    )
                except KeyboardInterrupt:
                    raise
                except OSError as exc:
                    if not is_transient_fork_error(exc):
                        raise
                    if attempt >= SUBPROCESS_FORK_RETRIES:
                        log.error(
                            "Could not start worker for batch %d after %d retry attempt(s): %s",
                            batch_number,
                            attempt,
                            exc,
                        )
                        return None
                    wait_seconds = SUBPROCESS_FORK_RETRY_SECONDS * attempt
                    log.warning(
                        "macOS temporarily refused to start worker for batch %d (%s). "
                        "Retrying in %d second(s), attempt %d/%d.",
                        batch_number,
                        exc,
                        wait_seconds,
                        attempt + 1,
                        SUBPROCESS_FORK_RETRIES,
                    )
                    time.sleep(wait_seconds)
                    gc.collect()

        def popen_worker_with_retry(cmd: list[str], batch_number: int) -> subprocess.Popen | None:
            for attempt in range(1, SUBPROCESS_FORK_RETRIES + 1):
                try:
                    return subprocess.Popen(
                        cmd,
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.STDOUT,
                        env=worker_env,
                    )
                except OSError as exc:
                    if not is_transient_fork_error(exc):
                        raise
                    if attempt >= SUBPROCESS_FORK_RETRIES:
                        log.error(
                            "Could not start worker for batch %d after %d retry attempt(s): %s",
                            batch_number,
                            attempt,
                            exc,
                        )
                        return None
                    wait_seconds = SUBPROCESS_FORK_RETRY_SECONDS * attempt
                    log.warning(
                        "macOS temporarily refused to start worker for batch %d (%s). "
                        "Retrying in %d second(s), attempt %d/%d.",
                        batch_number,
                        exc,
                        wait_seconds,
                        attempt + 1,
                        SUBPROCESS_FORK_RETRIES,
                    )
                    time.sleep(wait_seconds)
                    gc.collect()

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
                    proc = run_worker_with_retry(cmd, batch_idx + 1)
                except KeyboardInterrupt:
                    log.warning("Interrupted. Cache up to last completed batch is saved.")
                    return all_new_records, processed_sources, diagnostics

                if proc is None:
                    return all_new_records, processed_sources, diagnostics

                if proc.returncode != 0:
                    log.error("Worker for batch %d exited with code %d.",
                              batch_idx + 1, proc.returncode)
                    if proc.stdout:
                        log.error("Worker output:\n%s", proc.stdout[-4000:])
                    return all_new_records, processed_sources, diagnostics

                if not out_path.exists():
                    log.error("Worker for batch %d produced no output.", batch_idx + 1)
                    return all_new_records, processed_sources, diagnostics

                with out_path.open("rb") as f:
                    payload = pickle.load(f)
                if isinstance(payload, dict):
                    batch_faces = list(payload.get("faces", []))
                    diagnostics.update(payload.get("diagnostics", {}))
                    batch_fingerprints = dict(payload.get("fingerprints", {}))
                else:
                    batch_faces = list(payload)
                    batch_fingerprints = {}

                batch, batch_faces = validate_detection_batch(batch, batch_faces, batch_fingerprints, diagnostics)
                for src in batch:
                    try:
                        cache.file_signatures[str(src)] = file_signature(src)
                    except OSError:
                        pass
                    processed_sources.add(src)
                cache.faces.extend(batch_faces)
                save_cache(cache)
                persist_detection_batch(
                    batch, batch_faces, diagnostics, index_path, batch_fingerprints,
                )

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
                        proc = popen_worker_with_retry(cmd, batch_idx + 1)
                        if proc is None:
                            for active_proc in active:
                                active_proc.terminate()
                            return all_new_records, processed_sources, diagnostics
                        active[proc] = job_info

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
                            return all_new_records, processed_sources, diagnostics
                        if not out_path.exists():
                            log.error("Worker for batch %d produced no output.",
                                      batch_idx + 1)
                            for p in active:
                                p.terminate()
                            return all_new_records, processed_sources, diagnostics

                        with out_path.open("rb") as f:
                            payload = pickle.load(f)
                        if isinstance(payload, dict):
                            batch_faces = list(payload.get("faces", []))
                            diagnostics.update(payload.get("diagnostics", {}))
                            batch_fingerprints = dict(payload.get("fingerprints", {}))
                        else:
                            batch_faces = list(payload)
                            batch_fingerprints = {}

                        batch, batch_faces = validate_detection_batch(batch, batch_faces, batch_fingerprints, diagnostics)
                        for src in batch:
                            try:
                                cache.file_signatures[str(src)] = file_signature(src)
                            except OSError:
                                pass
                            processed_sources.add(src)
                        cache.faces.extend(batch_faces)
                        save_cache(cache)
                        persist_detection_batch(
                            batch, batch_faces, diagnostics, index_path,
                            batch_fingerprints,
                        )

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
                return all_new_records, processed_sources, diagnostics
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            pass

    return all_new_records, processed_sources, diagnostics


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
        labels: dict[int, set[str]] = defaultdict(set)
        protected = set()
        for record in records:
            if record.prior_label:
                labels[record.cluster_id].add(record.prior_label.casefold())
            if record.identity_review_reason:
                protected.add(record.cluster_id)
        for left, left_id in enumerate(ids):
            for right in range(left + 1, len(ids)):
                right_id = ids[right]
                if (left_id in protected or right_id in protected
                        or len(labels[left_id] | labels[right_id]) > 1):
                    dist[left, right] = dist[right, left] = np.inf
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
        order = np.argsort(-sims)
        j = int(order[0])
        best_distance = 1.0 - float(sims[j])
        second_distance = (
            1.0 - float(sims[int(order[1])]) if len(order) > 1 else 1.0
        )
        if (
            best_distance <= STAGE_B_MAX_DIST
            and second_distance - best_distance >= STAGE_B_MIN_MARGIN
        ):
            r.cluster_id = ids[j]
            reassigned += 1
    return reassigned


def anchor_pass(records: list[FaceRecord]) -> int:
    anchored_cids = sorted({r.cluster_id for r in records if r.prior_label and r.cluster_id != -1})
    if not anchored_cids:
        return 0
    centroids = compute_centroids(records)
    anchor_cids = [cid for cid in anchored_cids if cid in centroids]
    anchor_embs = np.stack([centroids[cid] for cid in anchor_cids])
    n_reassigned = 0
    for r in records:
        if r.cluster_id != -1:
            continue
        sims = anchor_embs @ r.embedding
        order = np.argsort(-sims)
        j = int(order[0])
        best_distance = 1.0 - float(sims[j])
        second_distance = (
            1.0 - float(sims[int(order[1])]) if len(order) > 1 else 1.0
        )
        if (
            best_distance <= ANCHOR_MAX_DIST
            and second_distance - best_distance >= ANCHOR_MIN_MARGIN
        ):
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
        if labels and len(labels) == len(group) and len(set(labels)) == 1:
            top = Counter(labels).most_common(1)[0][0]
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
                             identity_db: IdentityDB | None,
                             decision_recorder=None,
                             source_batch_root: Path | None = None,
                             use_secondary_verifier: bool = True) -> int:
    import identity_assignment
    return identity_assignment.assign_identity_labels(
        records, name_map, identity_db, decision_recorder, source_batch_root,
        use_secondary_verifier, pipeline=sys.modules[__name__])


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
    recovered_cids = {record.cluster_id for record in records
                      if getattr(record, "recovery_assigned", False)}
    labeled_cids = [cid for cid, n in name_map.items()
                    if cid != -1 and not n.startswith("person_") and cid in centroids
                    and cid not in recovered_cids]
    if not labeled_cids:
        return 0
    unlabeled_cids = [cid for cid, n in name_map.items()
                       if cid != -1 and n.startswith("person_") and cid in centroids]
    held_cids = {record.cluster_id for record in records
                 if getattr(record, "identity_review_reason", "")}
    n_merged = 0
    for ucid in unlabeled_cids:
        # A later centroid merge must not undo an individual recovery rejection.
        if ucid not in name_map or ucid in held_cids:
            continue
        c = centroids.get(ucid)
        if c is None:
            continue
        # Compare unique identities, not labeled cluster IDs. Separate pose
        # clusters already carrying the same label are supporting evidence for
        # one person and must not consume the second-place margin.
        candidates: list[tuple[float, str, int]] = []
        for target_name in sorted({name_map[cid] for cid in labeled_cids}, key=str.casefold):
            same_name = [cid for cid in labeled_cids if name_map[cid] == target_name]
            target_cid = max(same_name, key=lambda cid: float(centroids[cid] @ c))
            distance = 1.0 - float(centroids[target_cid] @ c)
            candidates.append((distance, target_name, target_cid))
        candidates.sort(key=lambda item: (item[0], item[1].casefold()))
        best_distance, target_name, target_cid = candidates[0]
        second_distance = candidates[1][0] if len(candidates) > 1 else 1.0
        margin = second_distance - best_distance
        if (
            best_distance <= ANCHOR_CLUSTER_MERGE_DIST
            and margin >= ANCHOR_CLUSTER_MERGE_MIN_MARGIN
        ):
            log.info("Auto-merge: %s → %s (centroid distance %.3f, margin %.3f)",
                     name_map[ucid], target_name, best_distance, margin)
            merge_clusters_on_disk(records, name_map, clusters_dir,
                                    keep_cid=target_cid, drop_cid=ucid)
            n_merged += 1
        elif best_distance <= ANCHOR_CLUSTER_MERGE_DIST:
            log.info(
                "Anchor merge held for review: %s → %s "
                "(centroid distance %.3f, margin %.3f).",
                name_map[ucid], target_name, best_distance, margin,
            )
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
                        threshold: int,
                        category_for_path: Callable[[Path], str] | None = None,
                        ) -> tuple[list[Path], dict[Path, Path]]:
    items_sorted = sorted(items, key=lambda x: -x[1])
    keepers: list[Path] = []
    keeper_hashes: list[tuple[str, np.ndarray]] = []
    dup_to_winner: dict[Path, Path] = {}
    for src, _q, h in items_sorted:
        category = category_for_path(src) if category_for_path is not None else "all"
        if h.size == 0:
            keepers.append(src); keeper_hashes.append((category, h)); continue
        best_match: Path | None = None
        best_dist = threshold + 1
        for kept_src, (kept_category, kept_h) in zip(keepers, keeper_hashes):
            if category != kept_category:
                continue
            if kept_h.size == 0:
                continue
            d = hamming(h, kept_h)
            if d <= threshold and d < best_dist:
                best_dist = d
                best_match = kept_src
        if best_match is not None:
            dup_to_winner[src] = best_match
        else:
            keepers.append(src); keeper_hashes.append((category, h))
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
    file_operations.atomic_copy(src, dest, use_hardlinks=USE_HARDLINKS)


def verify_original_copy(src: Path, dest: Path, expected_sha256: str = "") -> str:
    return file_operations.verify_original_copy(src, dest, expected_sha256)


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
        try:
            operation_ledger.move_path(
                resolved,
                dest,
                sorted_root=output_dir,
                operation="sort_photos.archive_organized_sources",
                reason="archive organized source image to ready_to_delete",
                extra={"relative_path": rel.as_posix()},
            )
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
        try:
            operation_ledger.move_path(
                resolved,
                dest,
                sorted_root=output_dir,
                operation="sort_photos.archive_scanned_sources",
                reason="archive scanned inbox source image to ready_to_delete",
                extra={"relative_path": rel.as_posix()},
            )
            moved += 1
        except Exception as e:  # noqa: BLE001
            log.warning("Could not archive scanned source %s: %s", resolved, e)
    return moved


def archive_unassigned_sources(sources: Iterable[Path],
                               records: list[FaceRecord],
                               name_map: dict[int, str],
                               organized_sources: set[Path],
                               input_dir: Path,
                               output_dir: Path,
                               processed_sources: set[Path] | None = None,
                               detection_outcomes: dict[str, str] | None = None,
                               ) -> tuple[dict[str, int], Path | None]:
    """Move unresolved inbox images into visible, reason-specific review queues.

    A scanned image is not considered handled merely because face detection ran.
    It is handled only after an original was copied (or confirmed already present)
    in a named person's library. Everything else remains recoverable for review.
    """
    input_dir = input_dir.resolve()
    output_dir = output_dir.resolve()
    review_root = output_dir / "_source_review" / UNASSIGNED_INTAKE_DIR_NAME
    records_by_source: dict[Path, list[FaceRecord]] = defaultdict(list)
    for record in records:
        try:
            records_by_source[record.src.resolve()].append(record)
        except OSError:
            records_by_source[record.src].append(record)

    resolved_organized: set[Path] = set()
    for source in organized_sources:
        try:
            resolved_organized.add(source.resolve())
        except OSError:
            resolved_organized.add(source)
    resolved_processed: set[Path] | None = None
    if processed_sources is not None:
        resolved_processed = set()
        for source in processed_sources:
            try:
                resolved_processed.add(source.resolve())
            except OSError:
                resolved_processed.add(source)

    counts: Counter[str] = Counter()
    report_rows: list[dict[str, str | int]] = []
    for src in sorted(set(sources), key=lambda p: str(p).lower()):
        try:
            resolved = src.resolve()
        except OSError:
            resolved = src
        if resolved in resolved_organized:
            continue

        source_records = records_by_source.get(resolved, [])
        assigned_people = sorted({
            name_map.get(record.cluster_id, "")
            for record in source_records
            if is_real_person_label(name_map.get(record.cluster_id))
        })
        detector_status = (detection_outcomes or {}).get(str(resolved), "")
        if not src.exists():
            if not assigned_people:
                continue
            reason = "copy_failed"
            detail = "source_missing_before_review"
        elif not source_records and resolved_processed is not None and resolved not in resolved_processed:
            reason = "processing_failed"
            detail = "detector_batch_did_not_complete"
        elif not source_records and detector_status.startswith("detector_error:"):
            reason = "processing_failed"
            detail = detector_status
        elif not source_records and detector_status == "unreadable_image":
            reason = "unreadable_image"
            detail = "image_decoder_failed"
        elif not source_records and detector_status.startswith("face_quality_review:"):
            reason = "face_quality_review"
            detail = detector_status.split(":", 1)[1]
        elif not source_records and not detector_status and imread_unicode(src) is None:
            reason = "unreadable_image"
            detail = "image_decoder_failed"
        elif not source_records:
            reason = "no_usable_face"
            detail = detector_status or "no_face_passed_detection_and_quality_thresholds"
        elif not assigned_people:
            reason = "unknown_identity"
            details = {getattr(record, "identity_review_reason", "")
                       for record in source_records}
            detail = ";".join(sorted(details - {""})) or "usable_face_not_confidently_assigned"
        else:
            reason = "copy_failed"
            detail = "assigned_source_not_confirmed_in_person_library"

        if not src.exists():
            counts[reason] += 1
            report_rows.append({
                "outcome": reason,
                "reason_detail": detail,
                "source_path": str(resolved),
                "review_path": "",
                "detected_faces": len(source_records),
                "assigned_people": " | ".join(assigned_people),
                "error": "source file is missing",
            })
            continue

        try:
            rel = resolved.relative_to(input_dir)
        except ValueError:
            rel = Path(resolved.name)
        dest = unique_path(review_root / reason / rel)
        try:
            operation_ledger.move_path(
                resolved,
                dest,
                sorted_root=output_dir,
                operation="sort_photos.archive_unassigned_sources",
                reason=f"preserve unresolved intake image: {reason}",
                extra={
                    "relative_path": rel.as_posix(),
                    "outcome": reason,
                    "detected_faces": len(source_records),
                    "assigned_people": assigned_people,
                },
            )
        except Exception as exc:  # noqa: BLE001
            log.warning("Could not preserve unresolved source %s: %s", resolved, exc)
            counts["move_failed"] += 1
            report_rows.append({
                "outcome": "move_failed",
                "reason_detail": detail,
                "source_path": str(resolved),
                "review_path": "",
                "detected_faces": len(source_records),
                "assigned_people": " | ".join(assigned_people),
                "error": str(exc),
            })
            continue

        # Unknown-cluster review state is saved after this function returns.
        # Point those records at the preserved file so later review can still
        # render the source instead of retaining a stale To Process path.
        for record in source_records:
            record.src = dest
        counts[reason] += 1
        report_rows.append({
            "outcome": reason,
            "reason_detail": detail,
            "source_path": str(resolved),
            "review_path": str(dest),
            "detected_faces": len(source_records),
            "assigned_people": " | ".join(assigned_people),
            "error": "",
        })

    if not report_rows:
        return dict(counts), None

    run_id = os.environ.get("PHOTO_PIPELINE_RUN_ID") or time.strftime("intake_%Y%m%d_%H%M%S")
    safe_run_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", run_id)
    report_path = (
        output_dir
        / "_source_review"
        / UNASSIGNED_INTAKE_REPORT_DIR_NAME
        / f"{safe_run_id}.csv"
    )
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "outcome", "reason_detail", "source_path", "review_path",
            "detected_faces", "assigned_people", "error",
        ])
        writer.writeheader()
        writer.writerows(report_rows)
    return dict(counts), report_path


def archive_intake_duplicates(sources: Iterable[Path],
                              input_dir: Path,
                              output_dir: Path) -> int:
    archive_root = output_dir / "_source_review" / INTAKE_DUPLICATE_ARCHIVE_DIR_NAME
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
        try:
            operation_ledger.move_path(
                resolved,
                dest,
                sorted_root=output_dir,
                operation="sort_photos.archive_intake_duplicates",
                reason="archive duplicate inbox source image to ready_to_delete",
                extra={"relative_path": rel.as_posix()},
            )
            moved += 1
        except Exception as e:  # noqa: BLE001
            log.warning("Could not archive intake duplicate %s: %s", resolved, e)
    return moved


def archive_intake_near_visual_review(matches: Iterable[tuple[Path, Path]],
                                      output_dir: Path) -> int:
    moved = 0
    output_dir = output_dir.resolve()
    people_dir = (output_dir / "photos_by_person").resolve()
    seen: set[Path] = set()
    for src, matched_existing in sorted(matches, key=lambda pair: str(pair[0]).lower()):
        if src in seen:
            continue
        seen.add(src)
        if not src.exists():
            continue
        try:
            resolved = src.resolve()
            matched_resolved = matched_existing.resolve()
        except OSError:
            continue
        try:
            resolved.relative_to(output_dir)
            continue
        except ValueError:
            pass
        try:
            matched_rel = matched_resolved.relative_to(people_dir)
        except ValueError:
            log.warning("Near-visual match is not inside person folders: %s",
                        matched_existing)
            continue
        if len(matched_rel.parts) < 2:
            continue
        person_dir = people_dir / matched_rel.parts[0]
        dest = unique_path(person_dir / INTAKE_NEAR_VISUAL_REVIEW_DIR_NAME / resolved.name)
        try:
            operation_ledger.move_path(
                resolved,
                dest,
                sorted_root=output_dir,
                operation="sort_photos.archive_intake_near_visual_review",
                reason="move near-visual intake candidate into matched person review folder",
                extra={
                    "matched_existing": str(matched_existing),
                    "matched_relative_to_people": matched_rel.as_posix(),
                },
            )
            moved += 1
        except Exception as e:  # noqa: BLE001
            log.warning("Could not archive near-visual intake candidate %s: %s",
                        resolved, e)
    return moved


def _fingerprint_signature(path: Path) -> dict[str, int | float | str]:
    st = path.stat()
    return {
        "mtime": float(st.st_mtime),
        "size": int(st.st_size),
        "path": str(path),
    }


def load_fingerprint_cache() -> dict:
    if not FINGERPRINT_CACHE_FILE.exists():
        return {"version": FINGERPRINT_CACHE_VERSION, "entries": {}}
    try:
        with FINGERPRINT_CACHE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != FINGERPRINT_CACHE_VERSION:
            return {"version": FINGERPRINT_CACHE_VERSION, "entries": {}}
        data.setdefault("entries", {})
        return data
    except Exception:
        return {"version": FINGERPRINT_CACHE_VERSION, "entries": {}}


def save_fingerprint_cache(data: dict) -> None:
    FINGERPRINT_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = FINGERPRINT_CACHE_FILE.with_suffix(
        FINGERPRINT_CACHE_FILE.suffix + f".{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(FINGERPRINT_CACHE_FILE)


def image_duplicate_fingerprint(
    path: Path,
    cache_entries: dict | None = None,
    stats: Counter | None = None,
    asset_index: analysis_index.AnalysisIndex | None = None,
) -> tuple[str, str, int, float] | None:
    try:
        sig = _fingerprint_signature(path)
        path_key = str(path)
        if asset_index is not None:
            indexed = asset_index.fingerprint(path)
            if indexed is not None:
                if stats is not None:
                    stats["sqlite_hits"] += 1
                return (
                    indexed.sha256,
                    indexed.pixel_sha256,
                    indexed.phash,
                    indexed.width / max(1, indexed.height),
                )
        if cache_entries is not None:
            cached = cache_entries.get(path_key)
            if cached and cached.get("signature") == sig:
                try:
                    file_hash = str(cached["sha256"])
                    pixel_hash = str(cached["pixel_sha256"])
                    phash = int(str(cached["phash"]), 16)
                    width = int(cached["width"])
                    height = int(cached["height"])
                    if asset_index is not None:
                        asset_index.upsert_fingerprint(path, analysis_index.AssetFingerprint(
                            sha256=file_hash,
                            pixel_sha256=pixel_hash,
                            phash=phash,
                            width=width,
                            height=height,
                        ))
                    if stats is not None:
                        stats["json_cache_hits"] += 1
                    return file_hash, pixel_hash, phash, width / max(1, height)
                except Exception:
                    pass
        if stats is not None:
            stats["cache_misses"] += 1
        with suppress_native_stderr():
            decoded = asset_processing.load_decoded_asset(path)
        if decoded is None:
            if stats is not None:
                stats["decode_errors"] += 1
            return None
        file_hash = decoded.sha256
        h, w = decoded.height, decoded.width
        ratio = w / max(1, h)
        pixel_hash = decoded.pixel_sha256
        phash = phash_to_int(decoded.phash_bits)
        if asset_index is not None:
            asset_index.upsert_fingerprint(path, analysis_index.AssetFingerprint(
                sha256=file_hash,
                pixel_sha256=pixel_hash,
                phash=phash,
                width=w,
                height=h,
            ))
        if cache_entries is not None:
            cache_entries[path_key] = {
                "signature": sig,
                "size_bytes": int(sig["size"]),
                "width": w,
                "height": h,
                "sha256": file_hash,
                "pixel_sha256": pixel_hash,
                "phash": f"{phash:016x}",
            }
        return file_hash, pixel_hash, phash, ratio
    except Exception as exc:  # noqa: BLE001
        if stats is not None:
            stats["errors"] += 1
        log.debug("Could not fingerprint %s: %s", path, exc)
        return None


def filter_existing_person_duplicates(images: list[Path],
                                      output_dir: Path) -> tuple[list[Path], list[Path], list[tuple[Path, Path]]]:
    people_dir = output_dir / "photos_by_person"
    if not images or not people_dir.exists():
        return images, [], []

    person_images = list(iter_person_original_images(people_dir))
    if not person_images:
        return images, [], []

    log.info("Intake duplicate check: indexing %d existing person-folder image(s).",
             len(person_images))
    fp_cache = load_fingerprint_cache()
    entries = fp_cache.setdefault("entries", {})
    stats: Counter = Counter()
    exact: dict[tuple[str, str], Path] = {}
    exact_statuses_by_hash: dict[str, set[str]] = defaultdict(set)
    incoming_first_by_hash: dict[str, tuple[Path, str]] = {}
    kept: list[Path] = []
    safe_duplicates: list[Path] = []
    near_visual_review: list[tuple[Path, Path]] = []
    counts = {"exact_file": 0, "same_pixels": 0, "near_visual": 0, "errors": 0}
    index_path = analysis_index_file()
    with analysis_index.AnalysisIndex(index_path) as asset_index:
        for i, path in enumerate(person_images, start=1):
            fp = image_duplicate_fingerprint(path, entries, stats, asset_index)
            if fp is None:
                continue
            file_hash, _pixel_hash, _phash, _ratio = fp
            nudity_status = duplicate_nudity_status_for_path(path)
            exact.setdefault((nudity_status, file_hash), path)
            exact_statuses_by_hash[file_hash].add(nudity_status)
            if i % 1000 == 0:
                asset_index.commit()
                log.info("Intake duplicate check: indexed %d/%d existing image(s).",
                         i, len(person_images))

        log.info("Intake duplicate check: checking %d incoming image(s).", len(images))
        for i, path in enumerate(images, start=1):
            fp = image_duplicate_fingerprint(path, entries, stats, asset_index)
            if fp is None:
                counts["errors"] += 1
                kept.append(path)
                continue
            file_hash, _pixel_hash, _phash, _ratio = fp
            nudity_status = duplicate_nudity_status_for_path(path)
            if file_hash in exact_statuses_by_hash and nudity_status == "safe":
                nudity_status = classified_nudity_status(
                    path, file_hash=file_hash, asset_index=asset_index)

            first_incoming = incoming_first_by_hash.get(file_hash)
            if first_incoming is not None:
                first_path, first_status = first_incoming
                # Resolve provisional path-only "safe" categories when the
                # same bytes occur again in this intake. Content classification
                # is cached by hash, so both copies receive one stable result.
                if first_status == "safe":
                    resolved_first_status = classified_nudity_status(
                        first_path, file_hash=file_hash, asset_index=asset_index)
                    if resolved_first_status != first_status:
                        if exact.get((first_status, file_hash)) == first_path:
                            exact.pop((first_status, file_hash), None)
                            exact_statuses_by_hash[file_hash].discard(first_status)
                        exact.setdefault((resolved_first_status, file_hash), first_path)
                        exact_statuses_by_hash[file_hash].add(resolved_first_status)
                        incoming_first_by_hash[file_hash] = (
                            first_path, resolved_first_status)
                        first_status = resolved_first_status
                if nudity_status == "safe":
                    nudity_status = classified_nudity_status(
                        path, file_hash=file_hash, asset_index=asset_index)
            asset_index.update_classification(path, nudity_status=nudity_status)
            duplicate_kind: str | None = None
            if (nudity_status, file_hash) in exact:
                duplicate_kind = "exact_file"

            if duplicate_kind is None:
                kept.append(path)
                # Treat the first occurrence in this intake batch as part of
                # the baseline immediately. Otherwise identical files from
                # separate intake subfolders can reach clustering twice and
                # be routed to different people in the same run.
                exact.setdefault((nudity_status, file_hash), path)
                exact_statuses_by_hash[file_hash].add(nudity_status)
                incoming_first_by_hash.setdefault(file_hash, (path, nudity_status))
            else:
                safe_duplicates.append(path)
                counts[duplicate_kind] += 1
            if i % 500 == 0 or i == len(images):
                asset_index.commit()
                log.info("Intake duplicate check: checked %d/%d incoming image(s).",
                         i, len(images))

    save_fingerprint_cache(fp_cache)
    log.info("Intake duplicate fingerprint cache: sqlite=%d, json=%d, misses=%d, "
             "decode_errors=%d, errors=%d.",
             stats["sqlite_hits"], stats["json_cache_hits"], stats["cache_misses"],
             stats["decode_errors"], stats["errors"])

    if safe_duplicates or near_visual_review:
        log.info("Intake duplicate check found %d safe duplicate(s) and %d "
                 "near-visual review candidate(s): exact=%d, same-pixel=%d, "
                 "near-visual=%d, errors=%d.",
                 len(safe_duplicates), len(near_visual_review),
                 counts["exact_file"], counts["same_pixels"], counts["near_visual"],
                 counts["errors"])
    else:
        log.info("Intake duplicate check: no incoming images already exist in person folders.")
    return kept, safe_duplicates, near_visual_review


def organize_originals(records: list[FaceRecord],
                       name_map: dict[int, str],
                       originals_dir: Path,
                       input_dir: Path | None = None,
                       output_dir: Path | None = None) -> set[Path]:
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
        expected_hash = str(getattr(rec, "content_sha256", ""))
        if expected_hash:
            try:
                if content_identity.content_sha256(src) != expected_hash:
                    log.warning("Keeping changed source unfiled; its recognition is stale: %s", src)
                    continue
            except OSError:
                continue
        is_blurred = rec.sharpness < SHARPNESS_BLUR_THRESHOLD
        kind = "blurred" if is_blurred else "sharp"
        buckets[(person, kind)].append((src, rec.sharpness, rec.image_phash))

    originals_dir.mkdir(parents=True, exist_ok=True)
    completed = _load_checkpoint(originals_dir)
    if completed:
        log.info("Resume: verifying %d legacy copy hint(s) against original content.",
                 len(completed))
    journal = copy_journal.CopyJournal(originals_dir)

    counts = {"sharp_keep": 0, "sharp_dup": 0,
              "blurred_keep": 0, "blurred_dup": 0,
              "missing": 0, "skipped_existing": 0,
              "nudity_possible": 0, "nudity_review": 0, "nudity_errors": 0}
    per_person: dict[str, dict[str, int]] = defaultdict(
        lambda: {"sharp_keep": 0, "sharp_dup": 0,
                 "blurred_keep": 0, "blurred_dup": 0})

    save_every = 50
    pending_writes = 0
    next_indexes: dict[Path, int] = {}
    organized_sources: set[Path] = set()
    existing_hashes_by_person: dict[str, dict[str, dict[str, Path]]] = {}
    source_hashes: dict[Path, str] = {}
    source_nudity_statuses: dict[Path, str] = {}
    asset_index = analysis_index.AnalysisIndex(analysis_index_file())

    def source_hash_and_nudity_status(src: Path) -> tuple[str, str]:
        if src in source_hashes:
            return source_hashes[src], source_nudity_statuses[src]
        indexed = asset_index.fingerprint(src)
        if indexed is not None:
            file_hash = indexed.sha256
        else:
            try:
                file_hash = sha256_file(src)
            except OSError:
                file_hash = ""
        status = classified_nudity_status(
            src,
            file_hash=file_hash or None,
            asset_index=asset_index,
        )
        # Nudity caching may begin a SQLite write transaction. Release that
        # transaction before operation_ledger opens its short-lived SQLite
        # mirror while moving a possible-nudity copy. The JSONL ledger remains
        # authoritative, but leaving this transaction open caused a 30-second
        # self-lock for every move.
        if asset_index.connection.in_transaction:
            asset_index.commit()
        source_hashes[src] = file_hash
        source_nudity_statuses[src] = status
        return file_hash, status

    def existing_original_hashes(person: str, person_dir: Path) -> dict[str, dict[str, Path]]:
        if person in existing_hashes_by_person:
            return existing_hashes_by_person[person]
        hashes: dict[str, dict[str, Path]] = defaultdict(dict)
        root = person_dir / PERSON_PHOTOS_DIR
        if root.exists():
            for existing in root.rglob("*"):
                if not existing.is_file() or existing.suffix.lower() not in IMAGE_EXTS:
                    continue
                try:
                    file_hash = asset_index.content_sha256(existing)
                except OSError:
                    continue
                status = duplicate_nudity_status_for_path(existing)
                if file_hash:
                    hashes[status][file_hash] = existing
        asset_index.commit()
        existing_hashes_by_person[person] = hashes
        return hashes

    try:
        for (person, kind), items in tqdm(sorted(buckets.items()),
                                           desc="Copying originals", unit="bucket"):
            if kind == "sharp":
                base_dir = originals_dir / person / PERSON_PHOTOS_DIR
            else:
                base_dir = originals_dir / person / PERSON_PHOTOS_DIR / BLURRED_DIR
            person_dir = originals_dir / person
            base_dir.mkdir(parents=True, exist_ok=True)

            if DEDUP_DUPLICATES:
                keepers, dup_map = dedup_within_bucket(
                    items,
                    threshold=PHASH_THRESHOLD,
                    category_for_path=lambda src: source_hash_and_nudity_status(src)[1],
                )
            else:
                keepers = [it[0] for it in items]; dup_map = {}

            for src in keepers:
                key = f"{person}||{src}||{kind}||main"
                if not src.exists():
                    counts["missing"] += 1
                    completed.add(key)
                    continue
                src_hash, src_nudity_status = source_hash_and_nudity_status(src)
                operation_id = journal.operation_id(person, src_hash, f"{src_nudity_status}:{kind}:main")
                if src_hash and journal.verified_destination(operation_id, src_hash) is not None:
                    counts["skipped_existing"] += 1
                    organized_sources.add(src)
                    continue
                if (src_hash
                        and src_hash in existing_original_hashes(person, person_dir)[src_nudity_status]):
                    journal.completed(operation_id, src_hash,
                        existing_original_hashes(person, person_dir)[src_nudity_status][src_hash])
                    completed.add(key)
                    counts["skipped_existing"] += 1
                    organized_sources.add(src)
                    continue
                dest = next_numbered_dest(base_dir, person_dir, person, src, next_indexes)
                try:
                    journal.planned(operation_id, src_hash)
                    _atomic_copy(src, dest)
                    dest, nudity_status = maybe_move_to_nudity_subfolder(
                        dest,
                        person_dir,
                        file_hash=src_hash or None,
                        preclassified_status=src_nudity_status,
                    )
                    src_hash = verify_original_copy(src, dest, src_hash)
                    journal.completed(operation_id, src_hash, dest)
                except Exception as e:  # noqa: BLE001
                    log.error("Copy failed: %s → %s: %s", src.name, dest.name, e)
                    continue
                if nudity_status == NUDITY_POSSIBLE_DIR:
                    counts["nudity_possible"] += 1
                elif nudity_status == NUDITY_UNCERTAIN_DIR:
                    counts["nudity_review"] += 1
                elif nudity_status == "error":
                    counts["nudity_errors"] += 1
                if src_hash:
                    final_status = (
                        "possible" if nudity_status == NUDITY_POSSIBLE_DIR
                        else "uncertain" if nudity_status == NUDITY_UNCERTAIN_DIR
                        else src_nudity_status
                    )
                    existing_original_hashes(person, person_dir)[final_status][src_hash] = dest
                organized_sources.add(src)
                completed.add(key)
                counts[f"{kind}_keep"] += 1
                per_person[person][f"{kind}_keep"] += 1
                pending_writes += 1
                if pending_writes >= save_every:
                    _save_checkpoint(originals_dir, completed)
                    pending_writes = 0

            if dup_map:
                dup_dir = person_dir / DUPLICATES_DIR
                dup_dir.mkdir(parents=True, exist_ok=True)
                for src in dup_map:
                    key = f"{person}||{src}||{kind}||dup"
                    if not src.exists():
                        counts["missing"] += 1
                        completed.add(key)
                        continue
                    src_hash, src_nudity_status = source_hash_and_nudity_status(src)
                    operation_id = journal.operation_id(person, src_hash, f"{src_nudity_status}:{kind}:dup")
                    if src_hash and journal.verified_destination(operation_id, src_hash) is not None:
                        counts["skipped_existing"] += 1
                        organized_sources.add(src)
                        continue
                    if (src_hash
                            and src_hash in existing_original_hashes(person, person_dir)[src_nudity_status]):
                        completed.add(key)
                        counts["skipped_existing"] += 1
                        organized_sources.add(src)
                        continue
                    dest = next_numbered_dest(dup_dir, person_dir, person, src, next_indexes)
                    try:
                        journal.planned(operation_id, src_hash)
                        _atomic_copy(src, dest)
                        dest, nudity_status = maybe_move_to_nudity_subfolder(
                            dest,
                            person_dir,
                            file_hash=src_hash or None,
                            preclassified_status=src_nudity_status,
                        )
                        src_hash = verify_original_copy(src, dest, src_hash)
                        journal.completed(operation_id, src_hash, dest)
                    except Exception as e:  # noqa: BLE001
                        log.error("Copy failed: %s → %s: %s", src.name, dest.name, e)
                        continue
                    if nudity_status == NUDITY_POSSIBLE_DIR:
                        counts["nudity_possible"] += 1
                    elif nudity_status == NUDITY_UNCERTAIN_DIR:
                        counts["nudity_review"] += 1
                    elif nudity_status == "error":
                        counts["nudity_errors"] += 1
                    if src_hash:
                        final_status = (
                            "possible" if nudity_status == NUDITY_POSSIBLE_DIR
                            else "uncertain" if nudity_status == NUDITY_UNCERTAIN_DIR
                            else src_nudity_status
                        )
                        existing_original_hashes(person, person_dir)[final_status][src_hash] = dest
                    organized_sources.add(src)
                    completed.add(key)
                    counts[f"{kind}_dup"] += 1
                    per_person[person][f"{kind}_dup"] += 1
                    pending_writes += 1
                    if pending_writes >= save_every:
                        _save_checkpoint(originals_dir, completed)
                        pending_writes = 0
    finally:
        asset_index.close()
        journal.close()
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
        log.info("Nudity subfolder sort: confirmed=%d, review=%d, errors=%d",
                 counts["nudity_possible"], counts["nudity_review"],
                 counts["nudity_errors"])
    if ARCHIVE_ORGANIZED_SOURCES and input_dir is not None and output_dir is not None:
        moved = archive_organized_sources(organized_sources, input_dir, output_dir)
        log.info("Archived %d organized source image(s) to %s",
                 moved, output_dir / "_source_review" / SOURCE_ARCHIVE_DIR_NAME)
    return organized_sources


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


def records_are_from_library(records: list[FaceRecord], originals_dir: Path) -> bool:
    if not records:
        return False
    try:
        root = originals_dir.resolve()
    except OSError:
        return False
    for rec in records:
        try:
            rec.src.resolve().relative_to(root)
        except (OSError, ValueError):
            return False
    return True


def post_process_steps(output_dir: Path) -> list[list[str]]:
    script_dir = Path(__file__).resolve().parent
    return [
        [sys.executable, str(script_dir / "delete_person_folder_duplicates.py"),
         "--sorted-root", str(output_dir), "--quiet"],
        [sys.executable, str(script_dir / "optimize_sorted_output.py"),
         str(output_dir / "photos_by_person"), "--apply", "--quiet"],
        [sys.executable, str(script_dir / "rename_person_folder_files.py"),
         str(output_dir / "photos_by_person"), "--simple", "--apply", "--quiet"],
        [sys.executable, str(script_dir / "advanced_duplicate_matching.py"),
         str(output_dir / "photos_by_person"), "--quiet"],
    ]


def run_post_process(output_dir: Path) -> None:
    if not POST_PROCESS_OUTPUT:
        return
    steps = post_process_steps(output_dir)
    log.info("Running automatic output cleanup. Duplicate cleanup is report-only; originals are not moved automatically.")
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
                    scanned_sources: Iterable[Path] | None = None,
                    processed_sources: set[Path] | None = None,
                    detection_outcomes: dict[str, str] | None = None) -> None:
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
    organized_sources = organize_originals(
        all_records,
        name_map,
        originals_dir,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    centroids = compute_centroids(all_records)
    write_manifest(all_records, name_map, centroids, csv_path)
    run_post_process(output_dir)

    # In archive mode the source files are about to be moved out of the inbox.
    # Replacing the organized-photo cache here with soon-stale inbox paths makes
    # the next daily run expensive. Keep the batch cache intact and let the
    # daily cache-rehydrate step reconcile only real person-folder originals.
    if ARCHIVE_SCANNED_SOURCES and scanned_sources is not None:
        live_cache = load_cache()
        log.info("Preserved existing face cache before source archive: %d files, "
                 "%d faces (%d labeled). Cache refresh will reconcile organized "
                 "person-folder originals.",
                 len(live_cache.file_signatures), len(live_cache.faces),
                 sum(1 for c in live_cache.faces if c.label))
    elif not records_are_from_library(all_records, originals_dir):
        live_cache = load_cache()
        log.info("Preserved existing library face cache: %d files, %d faces "
                 "(%d labeled). Finish/resume records came from temporary "
                 "source paths; cache_tools rehydrate/relink owns the "
                 "photos_by_person cache.",
                 len(live_cache.file_signatures), len(live_cache.faces),
                 sum(1 for c in live_cache.faces if c.label))
    else:
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

    if ARCHIVE_SCANNED_SOURCES and scanned_sources is not None:
        outcome_counts, report_path = archive_unassigned_sources(
            scanned_sources,
            all_records,
            name_map,
            organized_sources,
            input_dir,
            output_dir,
            processed_sources=processed_sources,
            detection_outcomes=detection_outcomes,
        )
        unresolved = sum(
            outcome_counts.get(reason, 0)
            for reason in (
                "no_usable_face", "unknown_identity", "copy_failed",
                "processing_failed", "unreadable_image",
            )
        )
        log.info(
            "Preserved %d unresolved intake image(s) for review: "
            "no usable face=%d, unknown identity=%d, copy failed=%d, "
            "processing failed=%d, unreadable=%d, move failed=%d.",
            unresolved,
            outcome_counts.get("no_usable_face", 0),
            outcome_counts.get("unknown_identity", 0),
            outcome_counts.get("copy_failed", 0),
            outcome_counts.get("processing_failed", 0),
            outcome_counts.get("unreadable_image", 0),
            outcome_counts.get("move_failed", 0),
        )
        if report_path is not None:
            log.info("Unassigned intake report: %s", report_path)

    if preserve_labeling_state:
        remaining = save_remaining_labeling_state(
            all_records, name_map, output_dir, input_dir
        )
        log.info("Partial finalize complete. Labeled folders were written; "
                 "%d remaining unlabeled cluster(s) preserved.", remaining)
        if remaining:
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
    if ASSUME_MERGE_EXISTING_OUTPUT:
        log.info("Merging into existing output folder: %s", output_dir)
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
    for index, (cf, cid) in enumerate(zip(state.faces, state.cluster_ids)):
        rec = cached_to_record(cf)
        rec.cluster_id = cid
        restore_recovery_state(rec, state, index)
        all_records.append(rec)
    name_map = dict(state.name_map)

    if not finish_labeled and min_cluster_size > 1:
        by_id: dict[int, list[FaceRecord]] = defaultdict(list)
        for rec in all_records:
            by_id[rec.cluster_id].append(rec)
        needs_label_all = [
            cid for cid, name in name_map.items()
            if cid != -1 and name.startswith("person_")
        ]
        needs_label = [
            cid for cid in needs_label_all
            if len(by_id[cid]) >= min_cluster_size
        ]
        if needs_label_all and not needs_label:
            skipped_small = len(needs_label_all)
            log.info("Skipping %d small unlabeled cluster(s) below "
                     "--min-label-cluster-size=%d.",
                     skipped_small, min_cluster_size)
            log.info("No unlabeled clusters meet the minimum size.")
            log.info("Nothing to review in this run; not finalizing already-entered "
                     "labels again.")
            log.info("To label every remaining small cluster, run: "
                     "python sort_photos.py --resume-label --min-label-cluster-size 1")
            log.info("To write already-entered labels once, use Finish Entered Labels.")
            return 0

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
    for index, (cf, cid) in enumerate(zip(state.faces, state.cluster_ids)):
        rec = cached_to_record(cf)
        rec.cluster_id = cid
        restore_recovery_state(rec, state, index)
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
    global IDENTITY_MAX_PROTOTYPES_PER_PERSON
    global POST_PROCESS_OUTPUT, USE_HARDLINKS, ARCHIVE_ORGANIZED_SOURCES
    global ARCHIVE_SCANNED_SOURCES, SOURCE_ARCHIVE_DIR_NAME
    global UNATTENDED_FINISH_KNOWN, ASSUME_MERGE_EXISTING_OUTPUT, DET_SIZE

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
                        help="Do not auto-place flagged images into photos/nude subfolders.")
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
                        help="Max candidate images scanned per person when rebuilding the identity DB.")
    parser.add_argument("--identity-prototypes", type=int,
                        default=IDENTITY_MAX_PROTOTYPES_PER_PERSON,
                        help="Max diverse face prototypes retained per person.")
    parser.add_argument("--no-post-process", action="store_true",
                        help="Skip automatic output cleanup after finishing.")
    parser.add_argument("--skip-output-cleanup", action="store_true",
                        help="Alias for --no-post-process; used by the guarded daily pipeline.")
    parser.add_argument("--copy-output", action="store_true",
                        help="Copy sorted images instead of hardlinking when possible.")
    parser.add_argument("--archive-organized-sources", action="store_true",
                        help="After successful organization, move source images to _source_review/organized_sources (or ready_to_delete/organized_sources with --archive-sources-to-ready-delete).")
    parser.add_argument("--archive-sources-to-ready-delete", action="store_true",
                        help="With --archive-organized-sources, move organized source images under _source_review/ready_to_delete/organized_sources.")
    parser.add_argument("--archive-scanned-sources", action="store_true",
                        help="After finishing, move every scanned input image out of the input folder to _source_review/ready_to_delete/scanned_sources. Best for a To Process inbox.")
    parser.add_argument("--merge-existing-output", action="store_true",
                        help="Do not prompt when output exists; always merge into existing output.")
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
    parser.add_argument("--max-input-images", type=int, default=0,
                        help="Process at most N input images, in deterministic path order. "
                             "Used by the memory-bounded daily supervisor; 0 means all images.")
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
    IDENTITY_MAX_PROTOTYPES_PER_PERSON = max(1, int(args.identity_prototypes))
    if args.no_post_process or args.skip_output_cleanup:
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
    if args.merge_existing_output:
        ASSUME_MERGE_EXISTING_OUTPUT = True
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
        build_identity_db_from_person_folders(
            requested_output_dir / "photos_by_person",
            force_rebuild=bool(args.rebuild_identity_db),
        )
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

    move_to_process_videos(input_dir)

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

    try:
        is_to_process = input_dir.resolve() == TO_PROCESS_DIR.expanduser().resolve()
    except OSError:
        is_to_process = False
    if is_to_process:
        # To Process is a real inbox: scan every user-dropped subfolder, even
        # when a source folder happens to be named like a generated view.
        excluded_scan_dirs = set()
        always_excluded_scan_dirs = set()
    else:
        excluded_scan_dirs = set() if args.scan_all_dirs else None
        always_excluded_scan_dirs = None
    discovered_images = list(iter_images(
        input_dir,
        excluded_dir_names=excluded_scan_dirs,
        always_excluded_dir_names=always_excluded_scan_dirs,
    ))
    total_input_images = len(discovered_images)
    max_input_images = max(0, int(args.max_input_images))
    images = bounded_input_images(discovered_images, max_input_images)
    del discovered_images
    log.info("Found %d images under %s", total_input_images, input_dir)
    if max_input_images and total_input_images > len(images):
        log.info(
            "Memory-bounded intake: processing %d/%d images in this worker; "
            "the supervisor will resume the remainder.",
            len(images), total_input_images,
        )
    images, generated_matches = generated_artifacts.partition_intake_artifacts(
        images,
        index_path=analysis_index_file(),
    )
    if generated_matches:
        moved_generated = generated_artifacts.archive_intake_matches(
            generated_matches,
            input_dir=input_dir,
            sorted_root=output_dir,
        )
        log.warning(
            "Diverted %d/%d generated contact/review sheet(s) before face sorting. "
            "Recoverable files: %s",
            moved_generated,
            len(generated_matches),
            output_dir / "_source_review" / "ready_to_delete" / "generated_contact_sheets",
        )
    if not images:
        return 0

    cached_records: list[FaceRecord] = []
    new_images: list[Path] = []
    processed_sources: set[Path] = set()
    detection_outcomes: dict[str, str] = {}
    log.info(
        "Preparing cache lookup for %d file signature(s) and %d cached face(s).",
        len(cache.file_signatures),
        len(cache.faces),
    )
    cache_face_index: dict[str, list[CachedFace]] = defaultdict(list)
    for index, cached_face in enumerate(cache.faces, start=1):
        cache_face_index[cache_path_key(cached_face.src_str)].append(cached_face)
        if index % 10_000 == 0 or index == len(cache.faces):
            log.info(
                "Cache lookup: indexed %d/%d cached face(s).",
                index,
                len(cache.faces),
            )
    cache_signature_index: dict[str, tuple[float, int]] = {}
    signature_total = len(cache.file_signatures)
    for index, (path, signature) in enumerate(cache.file_signatures.items(), start=1):
        cache_signature_index[cache_path_key(path)] = signature
        if index % 10_000 == 0 or index == signature_total:
            log.info(
                "Cache lookup: indexed %d/%d file signature(s).",
                index,
                signature_total,
            )
    stale_cache_keys: set[str] = set()
    sqlite_hydrated = 0
    hydrated_signatures: dict[str, tuple[float, int]] = {}
    hydrated_records: list[CachedFace] = []
    hydrated_keys: set[str] = set()
    detection_config = config_fingerprint()
    with analysis_index.AnalysisIndex(analysis_index_file()) as asset_index:
        for image_index, img in enumerate(images, start=1):
            try:
                sig = file_signature(img)
            except OSError:
                continue
            s = str(img)
            canonical = cache_path_key(s)
            if image_index % 500 == 0 or image_index == len(images):
                log.info(
                    "Cache lookup: checking intake image %d/%d (%d new or changed so far).",
                    image_index,
                    len(images),
                    len(new_images),
                )
            if cache_signature_index.get(canonical) == sig:
                processed_sources.add(img)
                cached_faces = cache_face_index.get(canonical, [])
                detection_outcomes[s] = (
                    "accepted_face_cached" if cached_faces else "no_usable_face_cached"
                )
                for c in cached_faces:
                    cached_records.append(cached_to_record(c))
                continue

            indexed = asset_index.cached_detections(img, detection_config)
            if indexed is not None:
                hydrated_faces = [
                    index_record_to_cached_face(img, record)
                    for record in indexed.detections
                ]
                hydrated_keys.add(canonical)
                hydrated_signatures[s] = sig
                hydrated_records.extend(hydrated_faces)
                cache_signature_index[canonical] = sig
                cache_face_index[canonical] = hydrated_faces
                processed_sources.add(img)
                detection_outcomes[s] = indexed.status
                cached_records.extend(cached_to_record(face) for face in hydrated_faces)
                sqlite_hydrated += 1
                continue

            new_images.append(img)
            stale_cache_keys.add(canonical)

    replaced_keys = stale_cache_keys | hydrated_keys
    if replaced_keys:
        cache.file_signatures = {
            path: signature for path, signature in cache.file_signatures.items()
            if cache_path_key(path) not in replaced_keys
        }
        cache.faces = [
            face for face in cache.faces
            if cache_path_key(face.src_str) not in replaced_keys
        ]
    cache.file_signatures.update(hydrated_signatures)
    cache.faces.extend(hydrated_records)
    if sqlite_hydrated or stale_cache_keys:
        save_cache(cache)
    if sqlite_hydrated:
        log.info("Recovered %d current image analysis result(s) from SQLite.", sqlite_hydrated)

    del cache_face_index
    gc.collect()

    log.info("Cache hit: %d images (%d faces). New / changed: %d images.",
             len(images) - len(new_images), len(cached_records), len(new_images))

    intake_duplicates: list[Path] = []
    intake_near_visual: list[tuple[Path, Path]] = []
    if new_images:
        new_images, intake_duplicates, intake_near_visual = filter_existing_person_duplicates(
            new_images, output_dir)
        if intake_duplicates and ARCHIVE_SCANNED_SOURCES:
            moved = archive_intake_duplicates(intake_duplicates, input_dir, output_dir)
            log.info("Archived %d intake duplicate source image(s) to %s",
                     moved,
                     output_dir / "_source_review" / INTAKE_DUPLICATE_ARCHIVE_DIR_NAME)
        elif intake_duplicates:
            log.info("Detected %d intake duplicate image(s); leaving sources in place "
                     "because --archive-scanned-sources is not enabled.",
                     len(intake_duplicates))
        if intake_near_visual:
            moved = archive_intake_near_visual_review(
                intake_near_visual, output_dir)
            log.info("Moved %d near-visual intake candidate(s) into matched "
                     "person-folder review subfolders named %s.",
                     moved,
                     INTAKE_NEAR_VISUAL_REVIEW_DIR_NAME)
        if intake_duplicates or intake_near_visual:
            log.info("New / changed images after intake duplicate check: %d.",
                     len(new_images))

    new_records: list[FaceRecord] = []
    if new_images:
        new_records, newly_processed_sources, new_detection_outcomes = detect_in_batches_subprocess(
            new_images, cache, batch_size=batch_size,
            workers=max(1, int(args.detect_workers)),
            index_path=analysis_index_file())
        processed_sources.update(newly_processed_sources)
        detection_outcomes.update(new_detection_outcomes)
        log.info("Total new faces extracted across all batches: %d.", len(new_records))

    all_records = cached_records + new_records
    if not all_records:
        log.warning("No assignable faces were available; preserving scanned images by outcome for review.")
        if ARCHIVE_SCANNED_SOURCES:
            outcome_counts, report_path = archive_unassigned_sources(
                images, [], {}, set(), input_dir, output_dir,
                processed_sources=processed_sources,
                detection_outcomes=detection_outcomes,
            )
            log.info(
                "Preserved unresolved intake for review at %s: no usable face=%d, "
                "processing failed=%d, unreadable=%d.",
                output_dir / "_source_review" / UNASSIGNED_INTAKE_DIR_NAME,
                outcome_counts.get("no_usable_face", 0),
                outcome_counts.get("processing_failed", 0),
                outcome_counts.get("unreadable_image", 0),
            )
            if report_path is not None:
                log.info("Unassigned intake report: %s", report_path)
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
    import daily_identity_recovery
    manual_count = daily_identity_recovery.apply_manual_overrides(
        all_records, name_map, None, people_root=output_dir / "photos_by_person")
    if manual_count:
        log.info("Preserved %d manual content decision(s) before automatic matching.", manual_count)
    if AUTO_PERSON_MATCH_ENABLED:
        identity_db = load_identity_db()
        if identity_db is None and (output_dir / "photos_by_person").exists():
            log.info("No usable identity DB found; building it from existing person folders.")
            identity_db = build_identity_db_from_person_folders(output_dir / "photos_by_person")
        # Quick Review and its independent verifier use the canonical database,
        # not the optional legacy-reference merge used by cluster matching.
        recovery_identity_db = identity_db
        if not args.no_reference_match:
            reference_db = load_reference_centroids(args.external_centroids)
            identity_db = merge_identity_dbs(identity_db, reference_db)
        if identity_db is not None:
            identity_db = calibrate_identity_db_against_impostors(identity_db)
        with analysis_index.AnalysisIndex(analysis_index_file()) as identity_index:
            def record_identity_decision(
                record: FaceRecord,
                identity_name: str,
                distance: float,
                margin: float,
                lane: str,
            ) -> None:
                identity_index.record_identity(
                    record.src,
                    face_index=record.face_index,
                    identity_name=identity_name,
                    distance=distance,
                    margin=margin,
                    lane=lane,
                    run_id=operation_ledger.current_run_id(),
                )

            n_identity = apply_identity_db_labels(
                all_records,
                name_map,
                identity_db,
                decision_recorder=record_identity_decision,
                source_batch_root=input_dir,
                use_secondary_verifier=False,
            )
        if n_identity:
            log.info("Existing-person identity DB auto-labeled %d cluster(s).", n_identity)
        import daily_identity_recovery
        recovery_plan = daily_identity_recovery.plan_recovery(
            all_records, name_map, recovery_identity_db, cache,
            people_root=output_dir / "photos_by_person",
            progress=log.info,
        )
        recovery_report = (
            output_dir / "_source_review" / UNASSIGNED_INTAKE_DIR_NAME / "reports"
            / f"{operation_ledger.current_run_id()}_identity_recovery.json"
        )
        daily_identity_recovery.write_report(recovery_plan, recovery_report)
        with analysis_index.AnalysisIndex(analysis_index_file()) as identity_index:
            recovered = daily_identity_recovery.apply_plan(
                recovery_plan, all_records, name_map, record_identity_decision
            )
        daily_identity_recovery.write_report(recovery_plan, recovery_report)
        log.info("Individual recovery: %d image(s) accepted; reasons: %s. Report: %s",
                 recovered, recovery_plan.counts, recovery_report)
        centroids = compute_centroids(all_records)
    write_cluster_crops(all_records, name_map, clusters_dir, centroids)
    log.info("Wrote face crops to: %s", clusters_dir)

    # Interactive runs need a resumable labeling snapshot. Unattended intake
    # already routes unresolved sources into review folders, so retaining the
    # same records here only creates large, obsolete backup files per slice.
    if not UNATTENDED_FINISH_KNOWN:
        save_labeling_state(all_records, name_map, output_dir, input_dir)
        log.info("Labeling state saved. You can quit anytime with 'q' and resume "
                 "with: python sort_photos.py --resume-label")

    if UNATTENDED_FINISH_KNOWN:
        log.info("Unattended mode: skipping manual labeling; unresolved sources "
                 "will be preserved in review folders.")
        labeling_complete = True
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
                    scanned_sources=images,
                    processed_sources=processed_sources,
                    detection_outcomes=detection_outcomes)
    return 0


if __name__ == "__main__":
    # Shared recovery must see this process's CLI settings and compatibility
    # classes, rather than importing a second sorter with default settings.
    sys.modules["sort_photos"] = sys.modules[__name__]
    sys.exit(main())

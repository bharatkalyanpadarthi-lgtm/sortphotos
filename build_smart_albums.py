#!/usr/bin/env python3
"""
Build non-destructive smart albums inside photos_by_person folders.

The script keeps original images where they are and creates hardlinked views:

  photos_by_person/Anushka/_smart_albums/
      00_best/top_020_technical_quality/
      01_quality/large_high_score/
      02_format/portrait/
      03_face_framing/closeup_face/
      04_visual_similar/high_confidence/001_12_photos_portrait_from_Anushka_023/
          _nudity_possible/
      05_same_scene/high_confidence/001_18_photos_balanced_light_green_portrait_upper_body_from_Anushka_041/
      07_review_needed/small/
      08_outfit/01_likely_saree_or_draped_ethnic/

Hardlinks do not duplicate file contents on disk. If a hardlink cannot be
created, the script falls back to a symlink.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

import source_manifest

warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\..*")
warnings.filterwarnings("ignore", message=r".*`estimate` is deprecated.*", category=FutureWarning)

DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
DEFAULT_NUDITY_CACHE = Path.home() / ".face_sort_cache" / "smart_album_nudity_cache.json"
DEFAULT_NUDITY_OVERRIDES = Path.home() / ".face_sort_cache" / "smart_album_nudity_overrides.json"
DEFAULT_FRAMING_CACHE = Path.home() / ".face_sort_cache" / "smart_album_framing_cache.json"
DEFAULT_SMART_STATE = Path.home() / ".face_sort_cache" / "smart_album_person_state.json"
NUDITY_CACHE_VERSION = 2
FRAMING_CACHE_VERSION = 2
SMART_ALBUM_LOGIC_VERSION = 19
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
SMART_DIR = "_smart_albums"
SMART_V2_DIR = "_smart_albums_v2"
EXCLUDED_DIRS = {
    SMART_DIR,
    SMART_V2_DIR,
    "_smart_albums_simple_preview",
    "_duplicates",
    "_near_visual_review",
    "_blurred",
}
NUDITY_NESTED_DIRS = {
    "photos/nude": "_nudity_possible",
    "photos_nude": "_nudity_possible",
    "_possible_nudity": "_nudity_possible",
    "_uncertain_nudity": "_nudity_uncertain",
}
AI_VIDEO_REFERENCE_FOLDERS = [
    "03_face_framing/06_ai_video_reference_set/00_best_single_starting_image",
    "03_face_framing/06_ai_video_reference_set/01_front_facing_portrait_chest_up",
    "03_face_framing/06_ai_video_reference_set/02_three_quarter_left_waist_up",
    "03_face_framing/06_ai_video_reference_set/03_three_quarter_right_waist_up",
    "03_face_framing/06_ai_video_reference_set/04_full_body_front_view",
    "03_face_framing/06_ai_video_reference_set/05_full_body_three_quarter",
    "03_face_framing/06_ai_video_reference_set/06_side_profile_chest_up",
    "03_face_framing/06_ai_video_reference_set/07_talking_head_refs",
    "03_face_framing/06_ai_video_reference_set/08_cinematic_portrait_refs",
    "03_face_framing/06_ai_video_reference_set/09_walking_action_refs",
    "03_face_framing/06_ai_video_reference_set/10_orbit_head_turn_refs",
]
SMART_V2_STRUCTURE_DIRS = [
    "00_START_HERE/01_best_ai_video_inputs",
    "00_START_HERE/02_best_saree_inputs",
    "00_START_HERE/03_best_same_scene_sets",
    "00_START_HERE/04_best_image_edit_inputs",
    "01_ATTIRE_AND_FRAMING/saree/01_close_up_face",
    "01_ATTIRE_AND_FRAMING/saree/02_chest_up",
    "01_ATTIRE_AND_FRAMING/saree/03_half_body_or_waist_up",
    "01_ATTIRE_AND_FRAMING/saree/04_full_body",
    "01_ATTIRE_AND_FRAMING/saree/05_three_quarter_view",
    "01_ATTIRE_AND_FRAMING/saree/06_front_straight_face",
    "01_ATTIRE_AND_FRAMING/saree/07_clear_face_visible",
    "01_ATTIRE_AND_FRAMING/western_or_modern/01_close_up_face",
    "01_ATTIRE_AND_FRAMING/western_or_modern/02_half_body_or_waist_up",
    "01_ATTIRE_AND_FRAMING/western_or_modern/03_full_body",
    "01_ATTIRE_AND_FRAMING/uncertain_draped_ethnic",
    "02_FACE_REFERENCES/01_clear_straight_face",
    "02_FACE_REFERENCES/02_front_face",
    "02_FACE_REFERENCES/03_left_three_quarter",
    "02_FACE_REFERENCES/04_right_three_quarter",
    "02_FACE_REFERENCES/05_side_profile",
    "02_FACE_REFERENCES/06_face_visible_but_not_perfect",
    "02_FACE_REFERENCES/90_face_uncertain_or_bad",
    "03_BODY_POSE/01_standing",
    "03_BODY_POSE/02_sitting",
    "03_BODY_POSE/03_laying_down",
    "03_BODY_POSE/04_laying_on_bed",
    "03_BODY_POSE/05_on_bed_sitting_or_reclining",
    "03_BODY_POSE/06_walking_or_action",
    "03_BODY_POSE/90_pose_uncertain",
    "04_BACKGROUND_SCENE/01_bedroom",
    "04_BACKGROUND_SCENE/02_bed_visible_or_on_bed",
    "04_BACKGROUND_SCENE/03_indoor_home",
    "04_BACKGROUND_SCENE/04_studio_or_plain_background",
    "04_BACKGROUND_SCENE/05_stage_or_event",
    "04_BACKGROUND_SCENE/06_outdoor",
    "04_BACKGROUND_SCENE/07_vehicle_or_travel",
    "04_BACKGROUND_SCENE/90_background_uncertain",
    "05_SAME_SCENE_SETS/high_confidence",
    "05_SAME_SCENE_SETS/medium_confidence",
    "05_SAME_SCENE_SETS/review",
    "06_AI_VIDEO_PACKS/talking_head",
    "06_AI_VIDEO_PACKS/orbit_head_turn",
    "06_AI_VIDEO_PACKS/cinematic_waist_up",
    "06_AI_VIDEO_PACKS/walking_or_full_body",
    "06_AI_VIDEO_PACKS/_shared_identity_refs",
    "06_AI_VIDEO_PACKS/saree_scene_packs",
    "80_FACETS_OPTIONAL/quality",
    "80_FACETS_OPTIONAL/format",
    "80_FACETS_OPTIONAL/color",
    "80_FACETS_OPTIONAL/raw_face_angle",
    "80_FACETS_OPTIONAL/raw_outfit_tags",
    "90_REVIEW/low_quality",
    "90_REVIEW/face_uncertain",
    "90_REVIEW/outfit_uncertain",
    "90_REVIEW/pose_uncertain",
    "90_REVIEW/background_uncertain",
    "90_REVIEW/nudity_possible",
    "_data",
]
SIMPLE_SMART_STRUCTURE_DIRS = [
    "01_quality/00_high_quality",
    "01_quality/01_good_quality",
    "01_quality/90_low_resolution",
    "02_format/landscape",
    "02_format/portrait",
    "02_format/square",
    "02_format/gif",
    "03_face_framing/00_clear_front_face",
    "03_face_framing/01_three_quarter_face",
    "03_face_framing/02_close_up_face",
    "03_face_framing/03_upper_body",
    "03_face_framing/04_half_or_three_quarter_body",
    "03_face_framing/05_full_body",
    "04_visual_similar/high_confidence",
    "04_visual_similar/review",
    "05_same_scene/high_confidence",
    "05_same_scene/medium_confidence",
    "06_saree_clear_views/00_saree_front_face",
    "06_saree_clear_views/01_saree_three_quarter",
    "06_saree_clear_views/02_saree_medium_or_full_body",
    "_data",
]
VALIDATED_SAREE_THREE_QUARTER_STEMS = {
    "soundarya_00376_photo_portrait_q_good",
}
EXPLICIT_NUDITY_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}
NUDITY_THRESHOLD = 0.70
NUDITY_UNCERTAIN_THRESHOLD = 0.45
NUDITY_CLASS_THRESHOLDS = {
    "FEMALE_BREAST_EXPOSED": 0.72,
    "BUTTOCKS_EXPOSED": 0.72,
    "FEMALE_GENITALIA_EXPOSED": 0.55,
    "MALE_GENITALIA_EXPOSED": 0.55,
    "ANUS_EXPOSED": 0.55,
}


@dataclass
class ImageInfo:
    path: Path
    rel: Path
    width: int
    height: int
    phash: int
    color: np.ndarray
    center_color: np.ndarray
    scene: np.ndarray
    context_hue: float
    context_sat: float
    clothing_hue: float
    clothing_sat: float
    clothing_value: float
    clothing_skin_ratio: float
    sharpness: float
    brightness: float
    contrast: float
    face_count: int
    largest_face_ratio: float
    quality: float
    face_source: str = "haar"
    face_angle: str = ""
    face_roll: float = 0.0
    face_center_x: float = 0.5
    face_center_y: float = 0.5
    face_width_ratio: float = 0.0
    face_height_ratio: float = 0.0
    nudity_status: str = ""
    nudity_class: str = ""
    nudity_score: float = 0.0


@dataclass
class SmartGroup:
    items: list[ImageInfo]
    confidence: str
    label: str


@dataclass
class SemanticTag:
    source: Path
    category: str
    label: str
    confidence: float
    method: str


def load_face_cascade() -> cv2.CascadeClassifier | None:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    return None if cascade.empty() else cascade


def load_nudity_cache(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") == NUDITY_CACHE_VERSION and isinstance(data.get("items"), dict):
            return data
    except Exception:
        pass
    return {"version": NUDITY_CACHE_VERSION, "items": {}}


def load_framing_cache(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") == FRAMING_CACHE_VERSION and isinstance(data.get("items"), dict):
            return data
    except Exception:
        pass
    return {"version": FRAMING_CACHE_VERSION, "items": {}}


def load_nudity_overrides(path: Path) -> dict[str, str]:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != 1:
            return {}
        items = data.get("items", {})
        if not isinstance(items, dict):
            return {}
        return {
            str(Path(k).expanduser().resolve()): str(v)
            for k, v in items.items()
            if str(v) in {"safe", "possible", "uncertain"}
        }
    except Exception:
        return {}


def save_nudity_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(path)


def save_framing_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(path)


def file_signature(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def load_smart_state(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") == 1 and isinstance(data.get("people"), dict):
            return data
    except Exception:
        pass
    return {"version": 1, "people": {}}


def save_smart_state(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def path_nudity_status(rel: Path) -> str:
    if not rel.parts:
        return ""
    if len(rel.parts) >= 2 and rel.parts[0] in {"all", "photos"} and rel.parts[1] == "nude":
        return "possible"
    if rel.parts[0] in {"photos_nude", "_possible_nudity"}:
        return "possible"
    if rel.parts[0] == "_uncertain_nudity":
        return "uncertain"
    if len(rel.parts) >= 2 and rel.parts[0] == "review" and rel.parts[1] == "nudity_possible":
        return "possible"
    if len(rel.parts) >= 2 and rel.parts[0] == "review" and rel.parts[1] == "uncertain_nudity":
        return "uncertain"
    return ""


def classify_nudity_detections(detections: list[dict]) -> tuple[str, str, float]:
    explicit = [d for d in detections if d.get("class") in EXPLICIT_NUDITY_CLASSES]
    if not explicit:
        return "", "", 0.0
    best = max(explicit, key=lambda d: float(d.get("score", 0.0)))
    best_class = str(best.get("class", ""))
    best_score = float(best.get("score", 0.0))
    class_threshold = max(NUDITY_THRESHOLD, NUDITY_CLASS_THRESHOLDS.get(best_class, NUDITY_THRESHOLD))
    if best_score >= class_threshold:
        return "possible", best_class, best_score
    if best_score >= NUDITY_UNCERTAIN_THRESHOLD:
        return "uncertain", best_class, best_score
    return "", best_class, best_score


def load_nudity_detector():
    try:
        from nudenet import NudeDetector
    except ImportError:
        return None
    return NudeDetector()


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


@contextlib.contextmanager
def quiet_model_startup():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull), suppress_native_stderr():
            yield


def load_insightface_detector(det_size: int):
    try:
        from insightface.app import FaceAnalysis
        import sort_photos
    except Exception:
        return None
    try:
        with quiet_model_startup():
            app = FaceAnalysis(
                name=sort_photos.MODEL_NAME,
                allowed_modules=["detection"],
                providers=sort_photos.PROVIDERS,
            )
            app.prepare(
                ctx_id=0,
                det_size=(int(det_size), int(det_size)),
                det_thresh=0.35,
            )
        return app
    except Exception:
        return None


def imread(path: Path) -> np.ndarray | None:
    def valid_bgr(img: np.ndarray | None) -> np.ndarray | None:
        if img is None or getattr(img, "size", 0) <= 0:
            return None
        if img.ndim != 3 or img.shape[0] <= 0 or img.shape[1] <= 0 or img.shape[2] < 3:
            return None
        return img

    with suppress_native_stderr():
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            valid = valid_bgr(img)
            if valid is not None:
                return valid
        except Exception:
            pass

        try:
            from PIL import Image, ImageFile
            import pillow_heif

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            if hasattr(pillow_heif, "register_heif_opener"):
                pillow_heif.register_heif_opener()
            with Image.open(path) as im:
                im.seek(0)
                im.load()
                img = cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
                return valid_bgr(img)
        except Exception:
            return None


def phash64(img: np.ndarray) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    vals = dct[:8, :8].flatten()
    median = float(np.median(vals[1:]))
    value = 0
    for bit in vals > median:
        value = (value << 1) | int(bool(bit))
    return value


def _masked_hsv_hist(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


def color_context(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    # Border-weighted color descriptor leans toward background/context.
    border = max(8, min(h, w) // 8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:border, :] = 255
    mask[-border:, :] = 255
    mask[:, :border] = 255
    mask[:, -border:] = 255
    return _masked_hsv_hist(img, mask)


def center_color_context(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, x2 = int(w * 0.20), int(w * 0.80)
    y1, y2 = int(h * 0.18), int(h * 0.88)
    mask[y1:y2, x1:x2] = 255
    return _masked_hsv_hist(img, mask)


def clothing_context_hsv(img: np.ndarray) -> tuple[float, float, float, float]:
    """Approximate outfit color from the lower/central body region.

    This is intentionally heuristic. It gives useful "likely outfit" folders
    without adding a large vision-language model dependency.
    """
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, x2 = int(w * 0.18), int(w * 0.82)
    y1, y2 = int(h * 0.34), int(h * 0.94)
    mask[y1:y2, x1:x2] = 255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    selected = hsv[mask > 0]
    if selected.size == 0:
        return 0.0, 0.0, 0.0, 0.0
    hue = float(np.median(selected[:, 0]))
    sat = float(np.median(selected[:, 1]))
    val = float(np.median(selected[:, 2]))
    skin_like = (
        (selected[:, 0] >= 0) & (selected[:, 0] <= 28)
        & (selected[:, 1] >= 35) & (selected[:, 1] <= 180)
        & (selected[:, 2] >= 55)
    )
    skin_ratio = float(np.mean(skin_like)) if selected.size else 0.0
    return hue, sat, val, skin_ratio


def layout_context(img: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    block = dct[:8, :8].flatten()[1:].astype(np.float32)
    norm = float(np.linalg.norm(block))
    if norm > 0:
        block = block / norm
    return block


def context_hsv(img: np.ndarray) -> tuple[float, float]:
    h, w = img.shape[:2]
    border = max(8, min(h, w) // 8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:border, :] = 255
    mask[-border:, :] = 255
    mask[:, :border] = 255
    mask[:, -border:] = 255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    selected = hsv[mask > 0]
    if selected.size == 0:
        return 0.0, 0.0
    return float(np.median(selected[:, 0])), float(np.median(selected[:, 1]))


def scene_context(border: np.ndarray,
                  center: np.ndarray,
                  layout: np.ndarray,
                  face_ratio: float,
                  face_x: float,
                  face_y: float,
                  width: int,
                  height: int) -> np.ndarray:
    aspect = width / max(1, height)
    geometry = np.array([
        min(max(aspect / 2.0, 0.0), 1.0),
        min(max(face_ratio * 8.0, 0.0), 1.0),
        min(max(face_x, 0.0), 1.0),
        min(max(face_y, 0.0), 1.0),
    ], dtype=np.float32)
    vec = np.concatenate([
        border.astype(np.float32) * 0.48,
        center.astype(np.float32) * 0.28,
        layout.astype(np.float32) * 0.18,
        geometry * 0.06,
    ])
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


def sharpness_score(img: np.ndarray) -> float:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def exposure_stats(img: np.ndarray) -> tuple[float, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(np.mean(gray)), float(np.std(gray))


def image_quality(img: np.ndarray, sharp: float, brightness: float, contrast: float) -> float:
    h, w = img.shape[:2]
    resolution = float(np.sqrt(max(1, w * h)))
    res_score = min(resolution / 1800.0, 1.0)
    sharp_score = min(np.log1p(max(sharp, 0.0)) / np.log1p(900.0), 1.0)
    contrast_score = min(max(contrast, 0.0) / 75.0, 1.0)
    exposure_score = 1.0 - min(abs(brightness - 128.0) / 128.0, 1.0)
    return float(
        0.40 * sharp_score
        + 0.30 * res_score
        + 0.20 * contrast_score
        + 0.10 * exposure_score
    )


def iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    union = aw * ah + bw * bh - inter
    return float(inter / union) if union > 0 else 0.0


def dedupe_boxes(boxes: list[tuple[int, int, int, int]]) -> list[tuple[int, int, int, int]]:
    out: list[tuple[int, int, int, int]] = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        if all(iou(box, existing) < 0.35 for existing in out):
            out.append(box)
    return out


def detect_face_stats(img: np.ndarray,
                      cascade: cv2.CascadeClassifier | None) -> tuple[int, float]:
    if cascade is None:
        return 0, 0.0
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return 0, 0.0
    max_dim = max(h, w)
    scale = min(1.0, 900.0 / max(1, max_dim))
    work = img
    if scale < 1.0:
        target = (max(1, int(w * scale)), max(1, int(h * scale)))
        work = cv2.resize(img, target, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(work, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    min_side = max(24, int(min(gray.shape[:2]) * 0.035))
    raw = cascade.detectMultiScale(
        gray,
        scaleFactor=1.08,
        minNeighbors=6,
        flags=cv2.CASCADE_SCALE_IMAGE,
        minSize=(min_side, min_side),
    )
    boxes: list[tuple[int, int, int, int]] = []
    inv = 1.0 / scale
    for x, y, bw, bh in raw:
        boxes.append((int(x * inv), int(y * inv), int(bw * inv), int(bh * inv)))
    boxes = dedupe_boxes(boxes)
    image_area = max(1, w * h)
    largest = max((bw * bh for _x, _y, bw, bh in boxes), default=0)
    significant = [
        box for box in boxes
        if box[2] * box[3] >= max(largest * 0.30, image_area * 0.004)
    ]
    return len(significant), float(largest / image_area)


def face_angle_from_keypoints(kps: np.ndarray | None) -> tuple[str, float]:
    if kps is None or len(kps) < 3:
        return "", 0.0
    left_eye, right_eye, nose = kps[0], kps[1], kps[2]
    eye_dx = float(right_eye[0] - left_eye[0])
    eye_dy = float(right_eye[1] - left_eye[1])
    eye_dist = float(np.hypot(eye_dx, eye_dy))
    if eye_dist <= 1.0:
        return "", 0.0
    roll = float(np.degrees(np.arctan2(eye_dy, eye_dx)))
    nose_offset = float((nose[0] - ((left_eye[0] + right_eye[0]) / 2.0)) / eye_dist)
    if nose_offset <= -0.34:
        angle = "side_angle_left"
    elif nose_offset >= 0.34:
        angle = "side_angle_right"
    elif nose_offset <= -0.18:
        angle = "turned_left"
    elif nose_offset >= 0.18:
        angle = "turned_right"
    else:
        angle = "front_facing"
    return angle, roll


def detect_face_stats_insightface(img: np.ndarray, app) -> tuple[int, float, str, float, float, float, float, float]:
    if app is None:
        return 0, 0.0, "", 0.0, 0.5, 0.5, 0.0, 0.0
    h, w = img.shape[:2]
    image_area = max(1, w * h)
    try:
        with suppress_native_stderr():
            faces = app.get(img)
    except Exception:
        return 0, 0.0, "", 0.0, 0.5, 0.5, 0.0, 0.0

    boxes: list[tuple[int, int, int, int, float, str, float]] = []
    for face in faces:
        score = float(getattr(face, "det_score", 0.0) or 0.0)
        if score < 0.35:
            continue
        bbox = getattr(face, "bbox", None)
        if bbox is None or len(bbox) < 4:
            continue
        x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
        x1 = max(0.0, min(float(w), x1))
        x2 = max(0.0, min(float(w), x2))
        y1 = max(0.0, min(float(h), y1))
        y2 = max(0.0, min(float(h), y2))
        bw = max(0, int(round(x2 - x1)))
        bh = max(0, int(round(y2 - y1)))
        if bw <= 0 or bh <= 0:
            continue
        angle, roll = face_angle_from_keypoints(getattr(face, "kps", None))
        boxes.append((int(round(x1)), int(round(y1)), bw, bh, score, angle, roll))

    if not boxes:
        return 0, 0.0, "", 0.0, 0.5, 0.5, 0.0, 0.0

    deduped: list[tuple[int, int, int, int, float, str, float]] = []
    for box in sorted(boxes, key=lambda b: b[2] * b[3], reverse=True):
        rect = box[:4]
        if all(iou(rect, existing[:4]) < 0.35 for existing in deduped):
            deduped.append(box)
    largest_box = max(deduped, key=lambda b: b[2] * b[3])
    largest_area = largest_box[2] * largest_box[3]
    significant = [
        box for box in deduped
        if box[2] * box[3] >= max(largest_area * 0.25, image_area * 0.001)
    ]
    cx = float((largest_box[0] + largest_box[2] / 2.0) / max(1, w))
    cy = float((largest_box[1] + largest_box[3] / 2.0) / max(1, h))
    fw = float(largest_box[2] / max(1, w))
    fh = float(largest_box[3] / max(1, h))
    return len(significant), float(largest_area / image_area), largest_box[5], largest_box[6], cx, cy, fw, fh


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def iter_images(person_dir: Path) -> list[Path]:
    out: list[Path] = []
    photos_dir = person_dir / "photos"
    has_photos_sources = False
    if photos_dir.is_dir():
        for _, _, filenames in os.walk(photos_dir):
            if any(Path(name).suffix.lower() in IMAGE_EXTS for name in filenames):
                has_photos_sources = True
                break
    for dirpath, dirnames, filenames in os.walk(person_dir):
        base = Path(dirpath)
        try:
            rel = base.relative_to(person_dir)
        except ValueError:
            rel = Path()
        if rel.parts == ("all",):
            if has_photos_sources:
                dirnames[:] = []
                continue
            dirnames[:] = [
                d for d in dirnames
                if d == "nude" and not d.startswith(".")
            ]
        elif len(rel.parts) >= 1 and rel.parts[0] == "all":
            if has_photos_sources or rel.parts[:2] != ("all", "nude"):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        elif rel.parts == ("review",):
            dirnames[:] = [
                d for d in dirnames
                if d == "nudity_possible" and not d.startswith(".")
            ]
        elif len(rel.parts) >= 1 and rel.parts[0] == "review":
            if rel.parts[:2] != ("review", "nudity_possible"):
                dirnames[:] = []
                continue
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        else:
            dirnames[:] = [
                d for d in dirnames
                if d not in EXCLUDED_DIRS and not d.startswith(".") and not d.startswith("_smart_albums")
            ]
        for filename in filenames:
            p = base / filename
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
    return sorted(out, key=lambda p: str(p.relative_to(person_dir)).lower())


def evenly_limited_paths(paths: list[Path], max_count: int) -> list[Path]:
    if max_count <= 0 or len(paths) <= max_count:
        return paths
    if max_count == 1:
        return [paths[0]]
    last = len(paths) - 1
    indexes = {
        int(round(i * last / max(1, max_count - 1)))
        for i in range(max_count)
    }
    return [paths[i] for i in sorted(indexes)]


def person_content_signature(person_dir: Path) -> dict[str, int | str]:
    h = hashlib.sha256()
    count = 0
    total_size = 0
    newest_mtime_ns = 0
    for path in iter_images(person_dir):
        try:
            st = path.stat()
        except OSError:
            continue
        rel = str(path.relative_to(person_dir))
        h.update(rel.encode("utf-8", errors="surrogateescape"))
        h.update(b"\0")
        h.update(str(int(st.st_size)).encode("ascii"))
        h.update(b"\0")
        h.update(str(int(st.st_mtime_ns)).encode("ascii"))
        h.update(b"\n")
        count += 1
        total_size += int(st.st_size)
        newest_mtime_ns = max(newest_mtime_ns, int(st.st_mtime_ns))
    return {
        "hash": h.hexdigest(),
        "count": count,
        "total_size": total_size,
        "newest_mtime_ns": newest_mtime_ns,
        "logic_version": SMART_ALBUM_LOGIC_VERSION,
    }


def load_infos(person_dir: Path, max_images: int = 0, trust_source_nudity: bool = True) -> list[ImageInfo]:
    infos: list[ImageInfo] = []
    face_cascade = load_face_cascade()
    for path in evenly_limited_paths(iter_images(person_dir), max_images):
        img = imread(path)
        if img is None:
            continue
        h, w = img.shape[:2]
        sharp = sharpness_score(img)
        brightness, contrast = exposure_stats(img)
        face_count, largest_face_ratio = detect_face_stats(img, face_cascade)
        border_color = color_context(img)
        center_color = center_color_context(img)
        layout = layout_context(img)
        hue, sat = context_hsv(img)
        clothing_hue, clothing_sat, clothing_value, clothing_skin_ratio = clothing_context_hsv(img)
        infos.append(ImageInfo(
            path=path,
            rel=path.relative_to(person_dir),
            width=w,
            height=h,
            phash=phash64(img),
            color=border_color,
            center_color=center_color,
            scene=scene_context(border_color, center_color, layout, largest_face_ratio, 0.5, 0.5, w, h),
            context_hue=hue,
            context_sat=sat,
            clothing_hue=clothing_hue,
            clothing_sat=clothing_sat,
            clothing_value=clothing_value,
            clothing_skin_ratio=clothing_skin_ratio,
            sharpness=sharp,
            brightness=brightness,
            contrast=contrast,
            face_count=face_count,
            largest_face_ratio=largest_face_ratio,
            quality=image_quality(img, sharp, brightness, contrast),
            nudity_status=path_nudity_status(path.relative_to(person_dir)) if trust_source_nudity else "",
        ))
    return infos


def annotate_framing(infos: list[ImageInfo],
                     app,
                     cache: dict,
                     cache_path: Path | None,
                     quiet: bool,
                     max_new_checks: int = 0) -> tuple[int, int]:
    if app is None:
        return 0, 0
    items = cache.setdefault("items", {})
    pending: list[tuple[ImageInfo, str, dict[str, int]]] = []
    cache_hits = 0
    changed = 0
    for info in infos:
        key = str(info.path)
        try:
            sig = file_signature(info.path)
        except OSError:
            continue
        cached = items.get(key)
        if cached and cached.get("sig") == sig:
            cached_count = int(cached.get("face_count", 0) or 0)
            if cached_count > 0 or info.face_count == 0:
                info.face_count = cached_count
                info.largest_face_ratio = float(cached.get("largest_face_ratio", 0.0) or 0.0)
                info.face_angle = str(cached.get("face_angle", "") or "")
                info.face_roll = float(cached.get("face_roll", 0.0) or 0.0)
                info.face_center_x = float(cached.get("face_center_x", 0.5) or 0.5)
                info.face_center_y = float(cached.get("face_center_y", 0.5) or 0.5)
                info.face_width_ratio = float(cached.get("face_width_ratio", 0.0) or 0.0)
                info.face_height_ratio = float(cached.get("face_height_ratio", 0.0) or 0.0)
                info.face_source = str(cached.get("source", "insightface") or "insightface")
            cache_hits += 1
            continue
        pending.append((info, key, sig))

    deferred = 0
    if max_new_checks > 0 and len(pending) > max_new_checks:
        deferred = len(pending) - max_new_checks
        pending = pending[:max_new_checks]

    if (pending or deferred) and not quiet:
        if deferred:
            print(
                f"  framing: {cache_hits} cached, {len(pending)} new InsightFace checks "
                f"this run, {deferred} deferred...",
                flush=True,
            )
        else:
            print(f"  framing: {cache_hits} cached, {len(pending)} new InsightFace checks...", flush=True)

    for idx, (info, key, sig) in enumerate(pending, start=1):
        if info.path.suffix.lower() == ".gif":
            items[key] = {
                "sig": sig,
                "face_count": info.face_count,
                "largest_face_ratio": round(info.largest_face_ratio, 8),
                "face_angle": info.face_angle,
                "face_roll": round(info.face_roll, 4),
                "face_center_x": round(info.face_center_x, 6),
                "face_center_y": round(info.face_center_y, 6),
                "face_width_ratio": round(info.face_width_ratio, 6),
                "face_height_ratio": round(info.face_height_ratio, 6),
                "source": info.face_source or "opencv_gif",
            }
            changed += 1
            if not quiet and (idx % 250 == 0 or idx == len(pending)):
                print(f"  framing: checked {idx}/{len(pending)}", flush=True)
            if cache_path is not None and (idx % 250 == 0 or idx == len(pending)):
                save_framing_cache(cache_path, cache)
            continue
        img = imread(info.path)
        if img is None:
            continue
        (
            face_count,
            largest_face_ratio,
            face_angle,
            face_roll,
            face_center_x,
            face_center_y,
            face_width_ratio,
            face_height_ratio,
        ) = detect_face_stats_insightface(img, app)
        del img
        if face_count > 0 or info.face_count == 0:
            info.face_count = face_count
            info.largest_face_ratio = largest_face_ratio
            info.face_angle = face_angle
            info.face_roll = face_roll
            info.face_center_x = face_center_x
            info.face_center_y = face_center_y
            info.face_width_ratio = face_width_ratio
            info.face_height_ratio = face_height_ratio
            info.face_source = "insightface"
        items[key] = {
            "sig": sig,
            "face_count": face_count,
            "largest_face_ratio": round(largest_face_ratio, 8),
            "face_angle": face_angle,
            "face_roll": round(face_roll, 4),
            "face_center_x": round(face_center_x, 6),
            "face_center_y": round(face_center_y, 6),
            "face_width_ratio": round(face_width_ratio, 6),
            "face_height_ratio": round(face_height_ratio, 6),
            "source": "insightface",
        }
        changed += 1
        if not quiet and (idx % 250 == 0 or idx == len(pending)):
            print(f"  framing: checked {idx}/{len(pending)}", flush=True)
        if cache_path is not None and (idx % 250 == 0 or idx == len(pending)):
            save_framing_cache(cache_path, cache)
        if idx % 50 == 0:
            gc.collect()
    return changed, deferred


def annotate_nudity(infos: list[ImageInfo],
                    detector,
                    cache: dict,
                    overrides: dict[str, str],
                    batch_size: int) -> int:
    if detector is None:
        return 0
    items = cache.setdefault("items", {})
    pending: list[ImageInfo] = []
    changed = 0

    for info in infos:
        override = overrides.get(str(info.path.resolve()))
        if override == "safe":
            info.nudity_status = ""
            info.nudity_class = "manual_safe"
            info.nudity_score = 0.0
            continue
        if override in {"possible", "uncertain"}:
            info.nudity_status = override
            info.nudity_class = "manual_override"
            info.nudity_score = 1.0
            continue
        if info.nudity_status:
            continue
        key = str(info.path)
        try:
            sig = file_signature(info.path)
        except OSError:
            continue
        cached = items.get(key)
        if cached and cached.get("sig") == sig:
            info.nudity_status = str(cached.get("status", ""))
            info.nudity_class = str(cached.get("class", ""))
            info.nudity_score = float(cached.get("score", 0.0))
            continue
        pending.append(info)

    for start in range(0, len(pending), max(1, batch_size)):
        batch = pending[start:start + max(1, batch_size)]
        paths = [str(i.path) for i in batch]
        try:
            with suppress_native_stderr():
                results = detector.detect_batch(paths, batch_size=len(paths))
        except Exception:
            results = []
            for info in batch:
                try:
                    with suppress_native_stderr():
                        results.append(detector.detect(str(info.path)))
                except Exception:
                    results.append([])

        for info, detections in zip(batch, results):
            if not isinstance(detections, list):
                detections = []
            status, best_class, best_score = classify_nudity_detections(detections)
            info.nudity_status = status
            info.nudity_class = best_class
            info.nudity_score = best_score
            try:
                sig = file_signature(info.path)
            except OSError:
                continue
            items[str(info.path)] = {
                "sig": sig,
                "status": status,
                "class": best_class,
                "score": round(best_score, 6),
            }
            changed += 1
    return changed


def safe_name(path: Path) -> str:
    parts = list(path.parts)
    stem = "__".join(parts)
    for ch in '/\\:*?"<>|':
        stem = stem.replace(ch, "_")
    while "__." in stem:
        stem = stem.replace("__.", ".")
    return stem


def nudity_nested_dir(info: ImageInfo) -> str | None:
    if info.nudity_status == "possible":
        return "_nudity_possible"
    rel_status = path_nudity_status(info.rel)
    if rel_status == "possible":
        return "_nudity_possible"
    if rel_status == "uncertain":
        return "_nudity_uncertain"
    if info.rel.parts and info.rel.parts[0] in NUDITY_NESTED_DIRS:
        return NUDITY_NESTED_DIRS[info.rel.parts[0]]
    return None


def visible_rel_name(rel: Path) -> Path:
    if len(rel.parts) > 2 and rel.parts[:2] == ("all", "nude"):
        return Path(*rel.parts[2:])
    if rel.parts and rel.parts[0] == "all" and len(rel.parts) > 1:
        return Path(*rel.parts[1:])
    if len(rel.parts) > 2 and rel.parts[:2] == ("photos", "nude"):
        return Path(*rel.parts[2:])
    if len(rel.parts) > 2 and rel.parts[:2] == ("review", "nudity_possible"):
        return Path(*rel.parts[2:])
    if rel.parts and rel.parts[0] in NUDITY_NESTED_DIRS and len(rel.parts) > 1:
        return Path(*rel.parts[1:])
    if rel.parts and rel.parts[0] == "photos" and len(rel.parts) > 1:
        return Path(*rel.parts[1:])
    return rel


def safe_component(value: str, max_len: int = 64) -> str:
    value = value.strip()
    value = re.sub(r"\s+", "_", value)
    value = re.sub(r"[/:\\*?\"<>|]+", "_", value)
    value = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("._-")
    if not value:
        return "image"
    return value[:max_len].rstrip("._-") or "image"


def orientation_name(info: ImageInfo) -> str:
    ratio = info.width / max(1, info.height)
    if ratio > 1.2:
        return "landscape"
    if ratio < 0.82:
        return "portrait"
    return "square"


def dominant_orientation(group: list[ImageInfo]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for info in group:
        counts[orientation_name(info)] += 1
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def dominant_face_angle(group: list[ImageInfo]) -> str:
    counts: dict[str, int] = defaultdict(int)
    for info in group:
        if info.face_angle:
            counts[info.face_angle] += 1
    if not counts:
        return ""
    return max(counts.items(), key=lambda item: (item[1], item[0]))[0]


def framing_label(group: list[ImageInfo]) -> str:
    ratio = float(np.median([i.largest_face_ratio for i in group])) if group else 0.0
    if ratio >= 0.22:
        return "closeup_face"
    if ratio >= 0.10:
        return "head_and_shoulders"
    if ratio >= 0.045:
        return "portrait_upper_body"
    if ratio >= 0.015:
        return "wide_full_body"
    return "wide_scene"


def light_label(group: list[ImageInfo]) -> str:
    brightness = float(np.median([i.brightness for i in group])) if group else 128.0
    if brightness < 78:
        return "low_light"
    if brightness > 172:
        return "bright"
    return "balanced_light"


def color_label(group: list[ImageInfo]) -> str:
    if not group:
        return "neutral"
    sat = float(np.median([i.context_sat for i in group]))
    if sat < 38:
        return "neutral"
    hue = float(np.median([i.context_hue for i in group]))
    if hue < 10 or hue >= 168:
        return "red_warm"
    if hue < 24:
        return "orange_warm"
    if hue < 38:
        return "yellow_warm"
    if hue < 80:
        return "green"
    if hue < 112:
        return "blue_cool"
    if hue < 145:
        return "purple_cool"
    return "pink_warm"


def scene_group_label(group: list[ImageInfo]) -> str:
    parts = [light_label(group), color_label(group), framing_label(group)]
    angle = dominant_face_angle(group)
    if angle in {"front", "three_quarter_left", "three_quarter_right", "side_profile"}:
        parts.append(angle)
    return safe_component("_".join(parts), max_len=56)


def representative_info(group: list[ImageInfo]) -> ImageInfo:
    return sorted(
        group,
        key=lambda i: (
            len(i.rel.parts),
            str(i.rel).lower(),
            -i.width * i.height,
        ),
    )[0]


def group_folder_name(group_id: int, group: list[ImageInfo], kind: str) -> str:
    rep = representative_info(group)
    source_stem = safe_component(rep.path.stem, max_len=42)
    orient = dominant_orientation(group)
    count = len(group)
    label = scene_group_label(group) if kind == "scene" else orient
    return f"{group_id:03d}_{count:02d}_photos_{label}_from_{source_stem}"


def clear_smart_albums(person_dir: Path) -> None:
    smart = person_dir / SMART_DIR
    if smart.exists():
        shutil.rmtree(smart)


def clear_smart_albums_v2(person_dir: Path) -> None:
    smart = person_dir / SMART_V2_DIR
    if smart.exists():
        shutil.rmtree(smart)


def clear_named_smart_album_dir(person_dir: Path, album_dir_name: str) -> Path:
    if not album_dir_name or "/" in album_dir_name or "\\" in album_dir_name:
        raise ValueError(f"unsafe smart album folder name: {album_dir_name!r}")
    album_root = (person_dir / album_dir_name).resolve()
    person_root = person_dir.resolve()
    if album_root.parent != person_root or not album_root.name.startswith("_smart_albums"):
        raise ValueError(f"refusing to clear non-smart-album folder: {album_root}")
    if album_root.exists():
        shutil.rmtree(album_root)
    return album_root


def ensure_smart_v2_structure(album_root: Path, apply: bool) -> None:
    if not apply:
        return
    for rel in SMART_V2_STRUCTURE_DIRS:
        (album_root / rel).mkdir(parents=True, exist_ok=True)


def ensure_simple_smart_structure(album_root: Path, apply: bool) -> None:
    if not apply:
        return
    for rel in SIMPLE_SMART_STRUCTURE_DIRS:
        (album_root / rel).mkdir(parents=True, exist_ok=True)


def link_image(info: ImageInfo, album_dir: Path, apply: bool) -> Path:
    nested = nudity_nested_dir(info)
    if nested:
        album_dir = album_dir / nested
    dest = album_dir / safe_name(visible_rel_name(info.rel))
    if not apply:
        return dest
    album_dir.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(str(info.path), str(dest))
    except OSError:
        os.symlink(str(info.path), str(dest))
    return dest


def link_image_plain(info: ImageInfo,
                     album_dir: Path,
                     apply: bool,
                     prefix: str = "") -> Path:
    dest = album_dir / f"{prefix}{safe_name(visible_rel_name(info.rel))}"
    if not apply:
        return dest
    album_dir.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(str(info.path), str(dest))
    except OSError:
        os.symlink(str(info.path), str(dest))
    return dest


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0:
        return 1.0
    return float(1.0 - float(np.dot(a, b)) / denom)


def aspect_ratio(info: ImageInfo) -> float:
    return info.width / max(1, info.height)


def current_scene_vector(info: ImageInfo) -> np.ndarray:
    geometry = np.array([
        min(max(aspect_ratio(info) / 2.0, 0.0), 1.0),
        min(max(info.largest_face_ratio * 8.0, 0.0), 1.0),
        min(max(info.face_center_x, 0.0), 1.0),
        min(max(info.face_center_y, 0.0), 1.0),
        min(max(info.face_width_ratio * 3.0, 0.0), 1.0),
        min(max(info.face_height_ratio * 3.0, 0.0), 1.0),
    ], dtype=np.float32)
    vec = np.concatenate([info.scene * 0.92, geometry * 0.08])
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec = vec / norm
    return vec.astype(np.float32)


def visual_pair_ok(a: ImageInfo, b: ImageInfo, threshold: int) -> bool:
    phash_dist = hamming(a.phash, b.phash)
    if phash_dist > threshold:
        return False
    if abs(aspect_ratio(a) - aspect_ratio(b)) > 0.055:
        return False
    border_dist = cosine_distance(a.color, b.color)
    center_dist = cosine_distance(a.center_color, b.center_color)
    if phash_dist <= max(2, threshold - 3):
        if max(border_dist, center_dist) > 0.34:
            return False
    elif max(border_dist, center_dist) > 0.22:
        return False
    if a.face_count and b.face_count:
        center_delta = abs(a.face_center_x - b.face_center_x) + abs(a.face_center_y - b.face_center_y)
        if center_delta > 0.22:
            return False
        ratio_delta = abs(a.largest_face_ratio - b.largest_face_ratio)
        if ratio_delta > max(0.035, min(a.largest_face_ratio, b.largest_face_ratio) * 0.65):
            return False
    return True


def group_pair_stats(group: list[ImageInfo]) -> tuple[float, float, float]:
    if len(group) < 2:
        return 0.0, 0.0, 0.0
    scene_dists: list[float] = []
    phashes: list[int] = []
    color_dists: list[float] = []
    vectors = [current_scene_vector(i) for i in group]
    for i, a in enumerate(group):
        for j in range(i + 1, len(group)):
            b = group[j]
            scene_dists.append(cosine_distance(vectors[i], vectors[j]))
            phashes.append(hamming(a.phash, b.phash))
            color_dists.append(max(cosine_distance(a.color, b.color), cosine_distance(a.center_color, b.center_color)))
    return (
        float(np.mean(scene_dists)) if scene_dists else 0.0,
        float(np.mean(phashes)) if phashes else 0.0,
        float(np.mean(color_dists)) if color_dists else 0.0,
    )


def visual_confidence(group: list[ImageInfo], threshold: int) -> str:
    scene_dist, phash_dist, color_dist = group_pair_stats(group)
    if phash_dist <= max(2.5, threshold - 2) and color_dist <= 0.16 and scene_dist <= 0.055:
        return "high_confidence"
    return "review"


def split_visual_component(component: list[ImageInfo], threshold: int, min_group: int) -> list[list[ImageInfo]]:
    groups: list[list[ImageInfo]] = []
    for info in sorted(component, key=lambda i: (str(i.rel).lower(), -i.quality)):
        placed = False
        for group in groups:
            matches = sum(1 for existing in group if visual_pair_ok(info, existing, threshold))
            if matches == len(group):
                group.append(info)
                placed = True
                break
        if not placed:
            groups.append([info])
    return [g for g in groups if len(g) >= min_group]


def group_visual_similar(infos: list[ImageInfo], threshold: int, min_group: int) -> list[SmartGroup]:
    if len(infos) < min_group:
        return []
    parent = {i: i for i in range(len(infos))}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i, a in enumerate(infos):
        for j in range(i + 1, len(infos)):
            if visual_pair_ok(a, infos[j], threshold):
                union(i, j)

    components: dict[int, list[ImageInfo]] = defaultdict(list)
    for i, info in enumerate(infos):
        components[find(i)].append(info)

    groups: list[SmartGroup] = []
    for component in components.values():
        if len(component) < min_group:
            continue
        for group in split_visual_component(component, threshold, min_group):
            groups.append(SmartGroup(
                items=group,
                confidence=visual_confidence(group, threshold),
                label=dominant_orientation(group),
            ))
    return sorted(groups, key=lambda g: (-len(g.items), g.confidence, str(g.items[0].rel).lower()))


def scene_confidence(group: list[ImageInfo]) -> str:
    scene_dist, phash_dist, color_dist = group_pair_stats(group)
    orientations = defaultdict(int)
    angles = defaultdict(int)
    for info in group:
        orientations[orientation_name(info)] += 1
        if info.face_angle:
            angles[info.face_angle] += 1
    orient_ratio = max(orientations.values()) / max(1, len(group))
    angle_ratio = max(angles.values()) / max(1, len(group)) if angles else 1.0
    if scene_dist <= 0.045 and color_dist <= 0.16 and orient_ratio >= 0.70 and angle_ratio >= 0.55:
        return "high_confidence"
    if scene_dist <= 0.075 and color_dist <= 0.24 and orient_ratio >= 0.55:
        return "medium_confidence"
    return "review"


def chunk_large_group(group: list[ImageInfo], max_group: int, min_group: int) -> list[list[ImageInfo]]:
    ordered = sorted(
        group,
        key=lambda i: (
            round(i.context_hue / 8.0),
            round(i.brightness / 16.0),
            round(i.largest_face_ratio, 3),
            str(i.rel).lower(),
        ),
    )
    chunks = [ordered[i:i + max_group] for i in range(0, len(ordered), max_group)]
    return [chunk for chunk in chunks if len(chunk) >= min_group]


def split_large_scene_group(group: list[ImageInfo],
                            eps: float,
                            min_group: int,
                            max_group: int,
                            depth: int = 0) -> list[list[ImageInfo]]:
    if len(group) <= max_group:
        return [group]
    if depth >= 8 or eps <= 0.0125:
        return chunk_large_group(group, max_group, min_group)
    vectors = np.stack([current_scene_vector(i) for i in group])
    sub_labels = DBSCAN(
        eps=max(eps * 0.62, 0.012),
        min_samples=min_group,
        metric="cosine",
    ).fit_predict(vectors)
    subgroups: dict[int, list[ImageInfo]] = defaultdict(list)
    for info, label in zip(group, sub_labels):
        if label >= 0:
            subgroups[int(label)].append(info)
    if not subgroups:
        return chunk_large_group(group, max_group, min_group)
    if len(subgroups) == 1 and len(next(iter(subgroups.values()))) == len(group):
        return chunk_large_group(group, max_group, min_group)
    out: list[list[ImageInfo]] = []
    for subgroup in subgroups.values():
        if len(subgroup) < min_group:
            continue
        if len(subgroup) <= max_group:
            out.append(subgroup)
        else:
            out.extend(g for g in split_large_scene_group(subgroup, eps * 0.62, min_group, max_group, depth + 1)
                       if len(g) <= max_group)
    return out or chunk_large_group(group, max_group, min_group)


def group_same_scene(infos: list[ImageInfo],
                     eps: float,
                     min_group: int,
                     max_group: int) -> list[SmartGroup]:
    if len(infos) < min_group:
        return []
    vectors = np.stack([current_scene_vector(i) for i in infos])
    labels = DBSCAN(eps=eps, min_samples=min_group, metric="cosine").fit_predict(vectors)
    groups: dict[int, list[ImageInfo]] = defaultdict(list)
    for info, label in zip(infos, labels):
        if label >= 0:
            groups[int(label)].append(info)
    out: list[SmartGroup] = []
    for group in groups.values():
        if len(group) < min_group:
            continue
        for subgroup in split_large_scene_group(group, eps, min_group, max_group):
            confidence = scene_confidence(subgroup)
            out.append(SmartGroup(
                items=subgroup,
                confidence=confidence,
                label=scene_group_label(subgroup),
            ))
    confidence_order = {"high_confidence": 0, "medium_confidence": 1, "review": 2}
    return sorted(
        out,
        key=lambda g: (confidence_order.get(g.confidence, 9), -len(g.items), str(g.items[0].rel).lower()),
    )


def same_scene_merge_key(label: str) -> tuple[str, str] | None:
    portrait_suffixes = (
        "_portrait_upper_body",
        "_head_and_shoulders",
    )
    for suffix in portrait_suffixes:
        if label.endswith(suffix):
            return (label[:-len(suffix)], "portrait_person")
    return None


def broad_portrait_scene_merge_candidate(info: ImageInfo) -> bool:
    """Allow cross-group scene pooling only when enough body/context is visible."""
    return (
        likely_social_screenshot(info)
        and outfit_visibility(info) in {"outfit_visible", "partial_outfit_visible"}
        and info.largest_face_ratio <= 0.115
        and info.face_height_ratio <= 0.320
        and info.face_count <= 2
    )


def merge_related_same_scene_groups(groups: list[SmartGroup],
                                    max_group: int) -> list[SmartGroup]:
    keyed: dict[tuple[str, str], list[SmartGroup]] = defaultdict(list)
    passthrough: list[SmartGroup] = []
    for group in groups:
        key = same_scene_merge_key(group.label)
        if key is None:
            passthrough.append(group)
        else:
            keyed[key].append(group)

    merged: list[SmartGroup] = list(passthrough)
    confidence_order = {"high_confidence": 0, "medium_confidence": 1, "review": 2}
    for key, scene_groups in keyed.items():
        if len(scene_groups) == 1:
            merged.extend(scene_groups)
            continue
        items_by_path: dict[str, ImageInfo] = {}
        remainder_by_label: dict[str, list[ImageInfo]] = defaultdict(list)
        for group in scene_groups:
            for item in group.items:
                if key[1] == "portrait_person" and not broad_portrait_scene_merge_candidate(item):
                    remainder_by_label[group.label].append(item)
                else:
                    items_by_path[str(item.path)] = item
        for label, remainder_items in remainder_by_label.items():
            if len(remainder_items) < 3:
                continue
            confidence = scene_confidence(remainder_items)
            if confidence not in {"high_confidence", "medium_confidence"}:
                continue
            merged.append(SmartGroup(
                items=sorted(remainder_items, key=lambda i: str(i.rel).lower()),
                confidence=confidence,
                label=scene_group_label(remainder_items) or label,
            ))
        items = sorted(items_by_path.values(), key=lambda i: str(i.rel).lower())
        if len(items) < 3:
            continue
        chunks = [items[i:i + max_group] for i in range(0, len(items), max_group)]
        worst_confidence = max(
            scene_groups,
            key=lambda g: confidence_order.get(g.confidence, 9),
        ).confidence
        confidence = "medium_confidence" if worst_confidence == "review" else worst_confidence
        for chunk in chunks:
            if len(chunk) < 3:
                continue
            merged.append(SmartGroup(
                items=chunk,
                confidence=confidence,
                label=f"{key[0]}_{key[1]}",
            ))

    return sorted(
        merged,
        key=lambda g: (confidence_order.get(g.confidence, 9), -len(g.items), str(g.items[0].rel).lower()),
    )


def context_name(info: ImageInfo) -> str:
    return f"02_format/{orientation_name(info)}"


def quality_album_names(info: ImageInfo) -> list[str]:
    names: list[str] = []
    pixels = info.width * info.height
    min_dim = min(info.width, info.height)
    if info.quality >= 0.68 and pixels >= 800_000:
        names.append("01_quality/large_high_score")
    if info.sharpness < 25.0:
        names.append("01_quality/low_sharpness_score")
    if min_dim < 450 or pixels < 350_000:
        names.append("01_quality/low_resolution")
        names.append("07_review_needed/low_resolution")
    if info.brightness < 45.0:
        names.append("01_quality/low_brightness_score")
    elif info.brightness > 220.0:
        names.append("01_quality/high_brightness_score")
    return names


def simple_quality_album_names(info: ImageInfo) -> list[str]:
    pixels = info.width * info.height
    min_dim = min(info.width, info.height)
    if min_dim < 450 or pixels < 350_000:
        return ["01_quality/90_low_resolution"]
    if (
        info.quality >= 0.68
        and pixels >= 900_000
        and info.sharpness >= 35.0
        and 45.0 <= info.brightness <= 220.0
    ):
        return ["01_quality/00_high_quality"]
    if is_reference_quality(info, min_score=0.46):
        return ["01_quality/01_good_quality"]
    return []


def simple_face_visible(info: ImageInfo, min_score: float = 0.48) -> bool:
    return (
        info.face_count == 1
        and info.face_source == "insightface"
        and info.largest_face_ratio >= 0.014
        and abs(info.face_roll) <= 20.0
        and 0.18 <= info.face_center_y <= 0.64
        and is_reference_quality(info, min_score=min_score)
    )


def simple_eye_level(info: ImageInfo) -> bool:
    eye_line = estimated_eye_line_y(info)
    return 0.20 <= eye_line <= 0.46


@lru_cache(maxsize=20000)
def plain_margin_card_score_for_path(path: str) -> float:
    img = imread(Path(path))
    if img is None:
        return 0.0
    return plain_margin_card_score_for_image(img)


def likely_plain_margin_capture(info: ImageInfo) -> bool:
    return plain_margin_card_score_for_path(str(info.path)) >= 0.58


@lru_cache(maxsize=20000)
def letterboxed_video_frame_score_for_path(path: str) -> float:
    img = imread(Path(path))
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0
    aspect = w / max(1, h)
    if aspect < 1.35:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    top = gray[:max(1, int(h * 0.16)), :]
    bottom = gray[int(h * 0.84):, :]
    left = gray[:, :max(1, int(w * 0.08))]
    right = gray[:, int(w * 0.92):]
    dark_top = float(np.mean(top < 35)) if top.size else 0.0
    dark_bottom = float(np.mean(bottom < 42)) if bottom.size else 0.0
    dark_side = max(
        float(np.mean(left < 42)) if left.size else 0.0,
        float(np.mean(right < 42)) if right.size else 0.0,
    )
    return max(min(dark_top, dark_bottom), min(max(dark_top, dark_bottom), dark_side))


def likely_letterboxed_video_frame(info: ImageInfo) -> bool:
    return letterboxed_video_frame_score_for_path(str(info.path)) >= 0.62


def simple_visual_artifact_capture(info: ImageInfo) -> bool:
    return (
        likely_plain_margin_capture(info)
        or likely_letterboxed_video_frame(info)
        or likely_social_screenshot(info)
    )


def simple_close_up_face_candidate(info: ImageInfo) -> bool:
    face_bottom = info.face_center_y + (info.face_height_ratio * 0.5)
    eye_line = estimated_eye_line_y(info)
    very_large_face_crop = (
        info.face_height_ratio >= 0.48
        and info.face_width_ratio >= 0.26
        and info.largest_face_ratio >= 0.16
        and 0.16 <= info.face_center_y <= 0.50
        and face_bottom <= 0.72
    )
    front_or_large_close = (
        info.face_height_ratio >= 0.34
        and info.face_width_ratio >= 0.18
        and info.largest_face_ratio >= 0.080
        and 0.18 <= info.face_center_y <= 0.49
        and face_bottom <= 0.66
        and 0.19 <= eye_line <= 0.45
    )
    turned_close = (
        info.face_angle in {"turned_left", "turned_right"}
        and info.face_width_ratio >= 0.16
        and (info.face_height_ratio >= 0.25 or info.largest_face_ratio >= 0.050)
        and 0.16 <= info.face_center_y <= 0.43
        and face_bottom <= 0.66
        and 0.15 <= eye_line <= 0.45
    )
    return very_large_face_crop or front_or_large_close or turned_close


def simple_full_body_pose_geometry(info: ImageInfo) -> bool:
    aspect = info.width / max(1, info.height)
    classic_wide_full_body = (
        aspect >= 1.18
        and info.face_center_y <= 0.32
        and info.largest_face_ratio <= 0.026
        and info.face_height_ratio <= 0.22
        and 0.22 <= info.face_center_x <= 0.78
    )
    distant_upright_full_body = (
        info.face_height_ratio <= 0.105
        and info.largest_face_ratio <= 0.020
        and info.face_center_y <= 0.34
        and 0.28 <= info.face_center_x <= 0.72
    )
    return classic_wide_full_body or distant_upright_full_body


def simple_body_framing(info: ImageInfo) -> str:
    ratio = info.largest_face_ratio
    face_height = info.face_height_ratio
    if simple_close_up_face_candidate(info):
        return "close_up_face"
    if ratio >= 0.055 or face_height >= 0.26:
        return "chest_or_waist_up"
    if ratio >= 0.045 and face_height >= 0.17:
        return "chest_or_waist_up"
    if simple_full_body_pose_geometry(info):
        return "full_body"
    if ratio >= 0.018 or face_height >= 0.105:
        return "three_quarter_body"
    return "full_body"


def simple_full_body_candidate(info: ImageInfo) -> bool:
    return simple_body_framing(info) == "full_body" and simple_full_body_pose_geometry(info)


def simple_face_framing_album_names(info: ImageInfo) -> list[str]:
    if not simple_face_visible(info):
        return []
    if simple_visual_artifact_capture(info):
        return []

    names: list[str] = []
    framing = simple_body_framing(info)
    centered = 0.34 <= info.face_center_x <= 0.66
    wide_centered = 0.26 <= info.face_center_x <= 0.74
    front = info.face_angle == "front_facing"
    three_quarter = info.face_angle in {"turned_left", "turned_right"}

    if (
        front
        and centered
        and simple_eye_level(info)
        and abs(info.face_roll) <= 8.0
        and info.largest_face_ratio >= 0.026
        and info.quality >= 0.56
        and info.sharpness >= 32.0
    ):
        names.append("03_face_framing/00_clear_front_face")

    true_three_quarter_face = (
        three_quarter
        and not simple_close_up_face_candidate(info)
        and wide_centered
        and abs(info.face_roll) <= 14.0
        and 0.030 <= info.largest_face_ratio <= 0.048
        and info.face_height_ratio <= 0.245
        and info.face_center_y <= 0.34
    )
    if true_three_quarter_face:
        names.append("03_face_framing/01_three_quarter_face")

    clean_body_geometry = (
        wide_centered
        and 0.16 <= info.face_center_y <= 0.46
        and abs(info.face_roll) <= 14.0
    )
    if clean_body_geometry:
        if framing == "close_up_face":
            names.append("03_face_framing/02_close_up_face")
        elif framing == "chest_or_waist_up":
            names.append("03_face_framing/03_upper_body")
        elif framing == "three_quarter_body":
            names.append("03_face_framing/04_half_or_three_quarter_body")
        elif simple_full_body_candidate(info):
            names.append("03_face_framing/05_full_body")

    return sorted(set(names))


def simple_saree_album_names(info: ImageInfo,
                             scene_map: dict[str, list[ImageInfo]],
                             validated_candidate: bool = False) -> list[str]:
    visibility = outfit_visibility(info)
    framing = shot_framing(info)
    drape = diagonal_drape_score(info)
    pallu = central_pallu_span_score(info)
    midriff = midriff_skin_score(info)
    high_precision = v2_high_precision_saree_confidence(info, scene_map)
    relaxed_score = simple_saree_validation_score(info, scene_map)

    if visibility not in {"outfit_visible", "partial_outfit_visible"}:
        return []
    if not (framing in {"chest_up", "waist_up", "full_body"} or info.face_angle in {"turned_left", "turned_right"}):
        return []
    if not simple_saree_face_visible(info, min_score=0.38):
        return []
    if abs(info.face_roll) > 24.0:
        return []
    if not validated_candidate and not (likely_saree_or_draped_ethnic(info) or v2_saree_confidence(info, scene_map) >= 0.72):
        return []
    if not validated_candidate and high_precision < 0.74 and relaxed_score < 0.96:
        return []
    if not validated_candidate and drape < 0.055 and pallu < 0.08 and midriff < 0.22:
        return []

    names: list[str] = []
    if (
        info.face_angle == "front_facing"
        and simple_eye_level(info)
        and 0.30 <= info.face_center_x <= 0.70
        and abs(info.face_roll) <= 18.0
    ):
        names.append("06_saree_clear_views/00_saree_front_face")
    if info.face_angle in {"turned_left", "turned_right"} or info.path.stem in VALIDATED_SAREE_THREE_QUARTER_STEMS:
        names.append("06_saree_clear_views/01_saree_three_quarter")
    if framing in {"chest_up", "waist_up", "full_body"} or validated_candidate:
        names.append("06_saree_clear_views/02_saree_medium_or_full_body")
    return sorted(set(names))


def simple_saree_face_visible(info: ImageInfo, min_score: float = 0.38) -> bool:
    return (
        info.face_count == 1
        and info.face_source == "insightface"
        and info.largest_face_ratio >= 0.014
        and abs(info.face_roll) <= 24.0
        and 0.18 <= info.face_center_y <= 0.66
        and is_reference_quality(info, min_score=min_score)
    )


def simple_saree_validation_score(info: ImageInfo,
                                  scene_map: dict[str, list[ImageInfo]]) -> float:
    if info.face_count > 2:
        return 0.0
    visibility = outfit_visibility(info)
    if visibility not in {"outfit_visible", "partial_outfit_visible"}:
        return 0.0
    if not (likely_saree_or_draped_ethnic(info) or v2_saree_confidence(info, scene_map) >= 0.70):
        return 0.0
    if not simple_saree_face_visible(info, min_score=0.34):
        return 0.0

    framing = shot_framing(info)
    drape = diagonal_drape_score(info)
    pallu = central_pallu_span_score(info)
    midriff = midriff_skin_score(info)
    ready = v2_saree_ready_confidence(info, scene_map)
    high_precision = v2_high_precision_saree_confidence(info, scene_map)
    score = max(ready, high_precision)
    score += min(0.20, drape * 0.16)
    score += min(0.16, pallu * 0.20)
    score += min(0.14, midriff * 0.12)
    score += min(0.10, max(0.0, info.quality - 0.45) * 0.25)
    if visibility == "outfit_visible":
        score += 0.10
    if framing in {"waist_up", "full_body"}:
        score += 0.12
    elif framing == "chest_up":
        score += 0.04
    if info.face_angle in {"front_facing", "turned_left", "turned_right"}:
        score += 0.06
    if bool(clear_saree_view_album_names(info)):
        score += 0.18
    return score


def simple_saree_validation_rows(infos: list[ImageInfo],
                                 scene_map: dict[str, list[ImageInfo]],
                                 limit: int = 100) -> list[dict[str, str]]:
    ranked = [
        (simple_saree_validation_score(info, scene_map), info)
        for info in infos
    ]
    chosen = [
        info for score, info in sorted(
            ranked,
            key=lambda item: (
                -item[0],
                str(item[1].rel).lower(),
            ),
        )
        if score > 0.0
    ][:max(1, limit)]
    rows: list[dict[str, str]] = []
    for info in chosen:
        row = manifest_row(
            info,
            "saree_candidates_for_validation/top_likely_saree_candidates",
            info.path,
        )
        row["saree_validation_score"] = f"{simple_saree_validation_score(info, scene_map):.3f}"
        row["saree_ready_score"] = f"{v2_saree_ready_confidence(info, scene_map):.3f}"
        row["saree_high_precision_score"] = f"{v2_high_precision_saree_confidence(info, scene_map):.3f}"
        row["drape_score"] = f"{diagonal_drape_score(info):.3f}"
        row["pallu_span_score"] = f"{central_pallu_span_score(info):.3f}"
        row["midriff_skin_score"] = f"{midriff_skin_score(info):.3f}"
        rows.append(row)
    return rows


def simple_nudity_album_names(info: ImageInfo) -> list[str]:
    return []


def simple_source_nudity_hint(info: ImageInfo) -> bool:
    rel_status = path_nudity_status(info.rel)
    name = info.path.name.lower()
    return rel_status == "possible" or "nudity_possible" in name


def simple_visual_nudity_hint(info: ImageInfo) -> bool:
    return (
        outfit_visibility(info) in {"outfit_visible", "partial_outfit_visible"}
        and info.clothing_skin_ratio >= 0.48
        and info.face_count <= 2
    )


def simple_nudity_candidate(info: ImageInfo) -> bool:
    return (
        info.nudity_status == "possible"
        or simple_source_nudity_hint(info)
        or simple_visual_nudity_hint(info)
    )


def color_bucket_from_hsv(hue: float, sat: float, value: float) -> str:
    if value < 55:
        return "black_or_very_dark"
    if sat < 28 and value > 188:
        return "white_or_light"
    if sat < 36:
        return "neutral_gray_beige"
    if hue < 10 or hue >= 168:
        return "red"
    if hue < 24:
        return "orange"
    if hue < 38:
        return "yellow_gold"
    if hue < 80:
        return "green"
    if hue < 112:
        return "blue"
    if hue < 145:
        return "purple"
    return "pink_magenta"


def outfit_visibility(info: ImageInfo) -> str:
    framing = shot_framing(info)
    if framing in {"full_body", "waist_up"}:
        return "outfit_visible"
    if framing == "chest_up":
        return "partial_outfit_visible"
    return "outfit_unclear_closeup"


def likely_saree_or_draped_ethnic(info: ImageInfo) -> bool:
    """Broad saree/draped-ethnic hint using local pixel/face features."""
    framing = shot_framing(info)
    if framing not in {"full_body", "waist_up", "chest_up"}:
        return False
    if info.face_count > 3:
        return False

    body_visible = info.largest_face_ratio <= 0.10 or info.face_height_ratio <= 0.36
    if not body_visible:
        return False

    drape_score = diagonal_drape_score(info)
    colorful_fabric = info.clothing_sat >= 58 and info.clothing_value >= 62
    warm_or_rich_color = (
        info.clothing_hue < 38
        or 70 <= info.clothing_hue < 112
        or info.clothing_hue >= 145
    )
    original_color_hint = (
        colorful_fabric
        and warm_or_rich_color
        and info.clothing_skin_ratio >= 0.035
    )
    draped_fabric_hint = (
        drape_score >= 0.08
        and info.clothing_sat >= 18.0
        and info.clothing_value >= 45.0
        and info.clothing_skin_ratio >= 0.02
    )
    pale_or_neutral_drape = (
        drape_score >= 0.30
        and info.clothing_value >= 70.0
        and info.clothing_skin_ratio >= 0.04
    )
    skin_plus_plain_drape = (
        drape_score >= 0.08
        and info.clothing_skin_ratio >= 0.12
        and info.clothing_sat >= 35.0
    )

    return original_color_hint or draped_fabric_hint or pale_or_neutral_drape or skin_plus_plain_drape


def possible_draped_ethnic(info: ImageInfo) -> bool:
    """Conservative guard for saree-like frames that are not clear enough to label."""
    framing = shot_framing(info)
    if framing not in {"full_body", "waist_up", "chest_up"}:
        return False
    if info.face_count > 3:
        return False

    body_visible = info.largest_face_ratio <= 0.11 or info.face_height_ratio <= 0.38
    if not body_visible:
        return False

    drape_score = diagonal_drape_score(info)
    has_outfit_color = info.clothing_value >= 42.0 and info.clothing_sat >= 12.0
    diagonal_hint = (
        drape_score >= 0.045
        and has_outfit_color
        and info.clothing_skin_ratio >= 0.012
    )
    shoulder_or_midriff_hint = (
        info.clothing_skin_ratio >= 0.09
        and info.clothing_sat >= 24.0
        and info.clothing_value >= 55.0
    )
    pale_drape_hint = (
        drape_score >= 0.18
        and info.clothing_value >= 82.0
        and info.clothing_skin_ratio >= 0.025
    )
    visible_neutral_traditional_hint = (
        outfit_visibility(info) == "outfit_visible"
        and info.clothing_skin_ratio >= 0.035
        and 18.0 <= info.clothing_sat < 58.0
        and info.clothing_value >= 70.0
    )

    return (
        diagonal_hint
        or shoulder_or_midriff_hint
        or pale_drape_hint
        or visible_neutral_traditional_hint
    )


def likely_western_or_modern(info: ImageInfo) -> bool:
    if outfit_visibility(info) == "outfit_unclear_closeup":
        return False
    if likely_saree_or_draped_ethnic(info):
        return False
    if possible_draped_ethnic(info):
        return False
    if info.face_count == 0 or info.face_count > 2:
        return False

    drape_score = diagonal_drape_score(info)
    if drape_score >= 0.045:
        return False
    if info.clothing_skin_ratio >= 0.12 and outfit_visibility(info) == "outfit_visible":
        return False

    low_skin_modern = (
        info.clothing_skin_ratio < 0.024
        and info.clothing_sat >= 70.0
        and info.clothing_value >= 45.0
    )
    plain_modern = (
        info.clothing_skin_ratio < 0.025
        and info.clothing_sat < 26.0
        and info.clothing_value >= 55.0
    )
    dark_modern = (
        info.clothing_skin_ratio < 0.025
        and info.clothing_value < 55.0
        and info.clothing_sat < 85.0
    )
    return low_skin_modern or plain_modern or dark_modern


@lru_cache(maxsize=20000)
def diagonal_drape_score_for_path(path: str) -> float:
    try:
        img = imread(Path(path))
        if img is None:
            return 0.0
        h, w = img.shape[:2]
        crop = img[int(h * 0.22):int(h * 0.86), int(w * 0.08):int(w * 0.92)]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        edges = cv2.Canny(gray, 60, 140)
        min_len = max(35, min(crop.shape[:2]) // 5)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                                minLineLength=min_len, maxLineGap=12)
        if lines is None:
            return 0.0
        total = 0.0
        diagonal = 0.0
        for x1, y1, x2, y2 in lines[:, 0]:
            dx = float(x2 - x1)
            dy = float(y2 - y1)
            length = float(np.hypot(dx, dy))
            if length <= 0.0:
                continue
            angle = abs(float(np.degrees(np.arctan2(dy, dx))))
            if angle > 90.0:
                angle = 180.0 - angle
            total += length
            if 20.0 <= angle <= 70.0:
                diagonal += length
        return diagonal / total if total > 0.0 else 0.0
    except Exception:
        return 0.0


def diagonal_drape_score(info: ImageInfo) -> float:
    """Detect a broad diagonal garment edge, common in saree/pallu views."""
    return diagonal_drape_score_for_path(str(info.path))


@lru_cache(maxsize=20000)
def skin_ratio_region_for_path(path: str, x1f: float, y1f: float, x2f: float, y2f: float) -> float:
    img = imread(Path(path))
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    x1 = int(w * x1f)
    y1 = int(h * y1f)
    x2 = int(w * x2f)
    y2 = int(h * y2f)
    crop = img[max(0, y1):min(h, y2), max(0, x1):min(w, x2)]
    if crop.size == 0:
        return 0.0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    ycrcb = cv2.cvtColor(crop, cv2.COLOR_BGR2YCrCb)
    hch, sch, vch = cv2.split(hsv)
    _, cr, cb = cv2.split(ycrcb)
    skin = (
        ((hch < 25) | (hch > 160))
        & (sch >= 25)
        & (sch <= 180)
        & (vch >= 55)
        & (cr >= 135)
        & (cr <= 180)
        & (cb >= 75)
        & (cb <= 140)
    )
    return float(np.mean(skin))


def midriff_skin_score(info: ImageInfo) -> float:
    path = str(info.path)
    return max(
        skin_ratio_region_for_path(path, 0.28, 0.48, 0.72, 0.78),
        skin_ratio_region_for_path(path, 0.25, 0.58, 0.75, 0.88),
    )


@lru_cache(maxsize=20000)
def central_pallu_span_score_for_path(path: str) -> float:
    img = imread(Path(path))
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    crop = img[int(h * 0.22):int(h * 0.86), int(w * 0.08):int(w * 0.92)]
    if crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(gray, 60, 140)
    min_len = max(35, min(crop.shape[:2]) // 5)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=40,
                            minLineLength=min_len, maxLineGap=12)
    if lines is None:
        return 0.0
    total = 0.0
    central_span = 0.0
    ch, cw = crop.shape[:2]
    for x1, y1, x2, y2 in lines[:, 0]:
        dx = float(x2 - x1)
        dy = float(y2 - y1)
        length = float(np.hypot(dx, dy))
        if length <= 0.0:
            continue
        total += length
        angle = abs(float(np.degrees(np.arctan2(dy, dx))))
        if angle > 90.0:
            angle = 180.0 - angle
        span_x = abs(dx) / max(1, cw)
        span_y = abs(dy) / max(1, ch)
        mid_x = (x1 + x2) / max(1, 2 * cw)
        mid_y = (y1 + y2) / max(1, 2 * ch)
        if (
            20.0 <= angle <= 70.0
            and span_x >= 0.22
            and span_y >= 0.16
            and 0.20 <= mid_x <= 0.80
            and 0.25 <= mid_y <= 0.65
        ):
            central_span += length
    return central_span / total if total > 0.0 else 0.0


def central_pallu_span_score(info: ImageInfo) -> float:
    return central_pallu_span_score_for_path(str(info.path))


def clear_saree_view_album_names(info: ImageInfo) -> list[str]:
    """Combined saree + framing views for reference-image browsing.

    The broad saree album intentionally catches many draped/ethnic possibilities.
    This view is stricter: it keeps images like a clear waist-up saree frame and
    filters out many swimwear/crop-top false positives from the color heuristic.
    """
    if not likely_saree_or_draped_ethnic(info):
        return []
    if outfit_visibility(info) != "outfit_visible":
        return []
    if info.face_count > 2:
        return []
    if info.face_angle in {"side_angle_left", "side_angle_right"}:
        return []
    if abs(info.face_roll) > 16.0:
        return []
    if not (0.24 <= info.face_center_x <= 0.76):
        return []
    if not is_reference_quality(info, min_score=0.50):
        return []

    framing = shot_framing(info)
    three_quarter = info.face_angle in {"turned_left", "turned_right"}
    medium_or_full = framing in {"waist_up", "full_body"}
    if not (three_quarter or medium_or_full):
        return []

    high_skin_saturated = info.clothing_skin_ratio >= 0.45 and info.clothing_sat >= 78.0
    if high_skin_saturated:
        return []

    drape_score = diagonal_drape_score(info)
    pale_saree_like = (
        info.clothing_skin_ratio >= 0.45
        and 45.0 <= info.clothing_sat < 78.0
        and info.clothing_value >= 115.0
    )
    clear_three_quarter_cloth = three_quarter and info.clothing_skin_ratio < 0.35
    if drape_score < 0.08 and not pale_saree_like and not clear_three_quarter_cloth:
        return []

    pallu_span = central_pallu_span_score(info)
    midriff_skin = midriff_skin_score(info)
    has_central_pallu = pallu_span >= 0.12
    has_saree_waist_signal = midriff_skin >= 0.24 and drape_score >= 0.10
    if not (has_central_pallu or has_saree_waist_signal):
        return []

    names = ["08_outfit/02_saree_clear_views/00_three_quarter_or_medium_full_body"]
    if three_quarter:
        names.append("08_outfit/02_saree_clear_views/01_three_quarter_view")
    if medium_or_full:
        names.append("08_outfit/02_saree_clear_views/02_medium_or_full_body")
    return names


def outfit_album_names(info: ImageInfo) -> list[str]:
    names: list[str] = []
    visibility = outfit_visibility(info)
    color = color_bucket_from_hsv(info.clothing_hue, info.clothing_sat, info.clothing_value)

    names.append(f"08_outfit/00_visibility/{visibility}")
    names.append(f"08_outfit/10_by_outfit_color/{color}")

    if likely_saree_or_draped_ethnic(info):
        names.append("08_outfit/01_likely_saree_or_draped_ethnic")
        names.extend(clear_saree_view_album_names(info))
    elif likely_western_or_modern(info):
        names.append("08_outfit/03_likely_western_or_modern")
    else:
        names.append("08_outfit/90_outfit_uncertain")
        if possible_draped_ethnic(info):
            names.append("08_outfit/90_outfit_uncertain/possible_draped_ethnic")

    if info.clothing_sat >= 78 and info.clothing_value >= 70:
        names.append("08_outfit/11_colorful_outfits")
    elif info.clothing_sat <= 36:
        names.append("08_outfit/12_neutral_or_plain_outfits")

    return sorted(set(names))


def estimated_eye_line_y(info: ImageInfo) -> float:
    return float(info.face_center_y - (info.face_height_ratio * 0.16))


def is_video_quality_candidate(info: ImageInfo) -> bool:
    pixels = info.width * info.height
    return (
        info.face_count == 1
        and pixels >= 800_000
        and min(info.width, info.height) >= 700
        and info.quality >= 0.58
        and info.sharpness >= 35.0
        and 45.0 <= info.brightness <= 220.0
    )


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def video_candidate_score(info: ImageInfo) -> float:
    """Rank images for single-image AI video workflows."""
    if info.face_count <= 0:
        return 0.0
    pixels = info.width * info.height
    score = 0.0
    if info.face_count == 1:
        score += 0.16
    elif info.face_count == 2:
        score += 0.04
    else:
        score -= 0.18

    if info.face_angle == "front_facing":
        score += 0.20
    elif info.face_angle in {"turned_left", "turned_right"}:
        score += 0.17
    elif info.face_angle in {"side_angle_left", "side_angle_right"}:
        score -= 0.18

    roll = abs(info.face_roll)
    if roll <= 8:
        score += 0.10
    elif roll <= 16:
        score += 0.04
    elif roll >= 32:
        score -= 0.12

    ratio = info.largest_face_ratio
    if 0.035 <= ratio <= 0.18:
        score += 0.11
    elif 0.020 <= ratio <= 0.30:
        score += 0.05
    else:
        score -= 0.04

    eye_line = estimated_eye_line_y(info)
    if 0.22 <= eye_line <= 0.44:
        score += 0.10
    elif 0.16 <= eye_line <= 0.52:
        score += 0.04

    if 0.36 <= info.face_center_x <= 0.64:
        score += 0.08
    elif 0.28 <= info.face_center_x <= 0.72:
        score += 0.03
    else:
        score -= 0.05

    min_dim = min(info.width, info.height)
    if pixels >= 1_400_000 and min_dim >= 900:
        score += 0.08
    elif pixels >= 800_000 and min_dim >= 700:
        score += 0.05
    elif min_dim < 450:
        score -= 0.08

    score += 0.14 * clamp(info.quality, 0.0, 1.0)
    if info.sharpness >= 45.0:
        score += 0.06
    elif info.sharpness >= 30.0:
        score += 0.03
    else:
        score -= 0.08

    if 55.0 <= info.brightness <= 205.0:
        score += 0.05
    elif info.brightness < 35.0 or info.brightness > 235.0:
        score -= 0.07

    return clamp(score, 0.0, 1.0)


def is_eye_level_reference(info: ImageInfo) -> bool:
    if info.face_count > 2:
        return False
    eye_line = estimated_eye_line_y(info)
    return abs(info.face_roll) <= 14.0 and 0.18 <= eye_line <= 0.50


def is_reference_quality(info: ImageInfo, min_score: float = 0.46) -> bool:
    pixels = info.width * info.height
    return (
        info.face_count >= 1
        and pixels >= 350_000
        and min(info.width, info.height) >= 450
        and info.quality >= min_score
        and info.sharpness >= 25.0
        and 40.0 <= info.brightness <= 225.0
    )


def shot_framing(info: ImageInfo) -> str:
    """Approximate person framing from face size when body keypoints are unavailable."""
    ratio = info.largest_face_ratio
    face_height = info.face_height_ratio
    if ratio >= 0.090 or face_height >= 0.36:
        return "portrait"
    if ratio >= 0.055 or face_height >= 0.26:
        return "chest_up"
    if ratio >= 0.028 or face_height >= 0.16:
        return "waist_up"
    return "full_body"


def ai_video_reference_album_names(info: ImageInfo) -> list[str]:
    """Shot-list folders for building a small identity reference set."""
    if not is_reference_quality(info):
        return []

    names: list[str] = []
    angle = info.face_angle
    framing = shot_framing(info)
    side_profile = angle in {"side_angle_left", "side_angle_right"}
    front = angle == "front_facing" and abs(info.face_roll) <= 12.0
    three_quarter_left = angle == "turned_left" and abs(info.face_roll) <= 18.0
    three_quarter_right = angle == "turned_right" and abs(info.face_roll) <= 18.0
    three_quarter = three_quarter_left or three_quarter_right
    eye_level = is_eye_level_reference(info)
    centered = 0.30 <= info.face_center_x <= 0.70
    portrait_or_chest = framing in {"portrait", "chest_up"}
    medium_or_waist = framing in {"chest_up", "waist_up"}
    full_body = (
        framing == "full_body"
        or (info.largest_face_ratio <= 0.065 and info.face_height_ratio <= 0.30)
    )

    base = "03_face_framing/06_ai_video_reference_set"
    if front and eye_level and portrait_or_chest:
        names.append(f"{base}/01_front_facing_portrait_chest_up")
        names.append(f"{base}/07_talking_head_refs")
        names.append(f"{base}/10_orbit_head_turn_refs")
    if three_quarter_left and eye_level and medium_or_waist:
        names.append(f"{base}/02_three_quarter_left_waist_up")
        names.append(f"{base}/08_cinematic_portrait_refs")
        names.append(f"{base}/10_orbit_head_turn_refs")
    if three_quarter_right and eye_level and medium_or_waist:
        names.append(f"{base}/03_three_quarter_right_waist_up")
        names.append(f"{base}/08_cinematic_portrait_refs")
        names.append(f"{base}/10_orbit_head_turn_refs")
    if front and eye_level and full_body and centered:
        names.append(f"{base}/04_full_body_front_view")
        names.append(f"{base}/09_walking_action_refs")
    if three_quarter and eye_level and full_body and centered:
        names.append(f"{base}/05_full_body_three_quarter")
        names.append(f"{base}/09_walking_action_refs")
    if side_profile and portrait_or_chest and abs(info.face_roll) <= 18.0:
        names.append(f"{base}/06_side_profile_chest_up")
        names.append(f"{base}/10_orbit_head_turn_refs")

    best_single = (
        three_quarter
        and eye_level
        and medium_or_waist
        and centered
        and video_candidate_score(info) >= 0.58
    )
    if best_single:
        names.append(f"{base}/00_best_single_starting_image")
        names.append("03_face_framing/00_best_ai_video_candidates")

    return sorted(set(names))


def face_framing_album_names(info: ImageInfo) -> list[str]:
    aspect = max(info.width, info.height) / max(1, min(info.width, info.height))
    if (
        aspect >= 2.8
        and (info.face_count == 0 or info.face_count >= 3 or info.largest_face_ratio < 0.025)
    ):
        return ["03_face_framing/90_not_ideal_for_ai_video/contact_sheet_or_collage"]
    if info.face_count == 0:
        return ["03_face_framing/90_not_ideal_for_ai_video/face_detection_uncertain"]

    names: list[str] = []
    if info.face_count >= 8 and info.largest_face_ratio < 0.025:
        names.append("03_face_framing/90_not_ideal_for_ai_video/contact_sheet_or_collage")
        return names
    if info.face_count >= 3:
        names.append("03_face_framing/90_not_ideal_for_ai_video/multi_face_or_group_photo")

    ratio = info.largest_face_ratio
    face_height = info.face_height_ratio

    front = info.face_angle == "front_facing" and abs(info.face_roll) <= 10.0
    three_quarter = info.face_angle in {"turned_left", "turned_right"} and abs(info.face_roll) <= 16.0
    side_profile = info.face_angle in {"side_angle_left", "side_angle_right"}
    centered = 0.34 <= info.face_center_x <= 0.66
    eye_line = estimated_eye_line_y(info)
    eye_level = (
        info.face_count <= 2
        and not side_profile
        and abs(info.face_roll) <= 12.0
        and 0.20 <= eye_line <= 0.46
    )
    medium_or_full_body = (
        info.face_count <= 2
        and ratio < 0.055
        and face_height < 0.26
        and centered
        and abs(info.face_roll) <= 12.0
        and info.face_center_y <= 0.44
        and aspect < 2.8
    )
    slight_low_angle_chest_up = (
        info.face_count <= 2
        and not side_profile
        and 0.045 <= ratio <= 0.16
        and 0.18 <= face_height <= 0.42
        and 0.24 <= info.face_center_y <= 0.54
        and abs(info.face_roll) <= 14.0
    )

    if front and ratio >= 0.035:
        names.append("03_face_framing/01_front_facing_straight_on")
        if centered and eye_level and is_reference_quality(info, min_score=0.52):
            names.append("03_face_framing/00_clear_straight_face_best")
    if three_quarter and ratio >= 0.025:
        names.append("03_face_framing/02_three_quarter_view")
    if eye_level and ratio >= 0.025:
        names.append("03_face_framing/03_eye_level_camera")
    if medium_or_full_body:
        names.append("03_face_framing/04_medium_or_full_body")
    if slight_low_angle_chest_up:
        names.append("03_face_framing/05_slight_low_angle_chest_up")

    names.extend(ai_video_reference_album_names(info))

    if is_video_quality_candidate(info) and (
        "03_face_framing/01_front_facing_straight_on" in names
        or "03_face_framing/02_three_quarter_view" in names
    ) and (
        "03_face_framing/03_eye_level_camera" in names
        or "03_face_framing/04_medium_or_full_body" in names
    ):
        names.append("03_face_framing/00_best_ai_video_candidates")

    if side_profile:
        names.append("03_face_framing/90_not_ideal_for_ai_video/side_profile_or_extreme_angle")
    if info.face_count <= 2 and ratio >= 0.025 and not centered:
        names.append("03_face_framing/90_not_ideal_for_ai_video/off_center_subject")
    if abs(info.face_roll) >= 32.0:
        names.append("03_face_framing/90_not_ideal_for_ai_video/strongly_tilted_or_rotated")
    elif abs(info.face_roll) >= 14.0:
        names.append("03_face_framing/90_not_ideal_for_ai_video/tilted_face")
    if not names:
        names.append("03_face_framing/90_not_ideal_for_ai_video/needs_manual_angle_review")
    return names


def write_group(album_root: Path,
                group_path: str,
                group_id: int,
                group: list[ImageInfo],
                kind: str,
                apply: bool,
                rows: list[dict[str, str]]) -> int:
    album_dir = album_root / group_path / group_folder_name(group_id, group, kind)
    for info in group:
        dest = link_image(info, album_dir, apply)
        rows.append(manifest_row(info, album_name_for_dest(album_root, dest), dest))
    return len(group)


def write_smart_group(album_root: Path,
                      base_path: str,
                      group_id: int,
                      smart_group: SmartGroup,
                      kind: str,
                      apply: bool,
                      rows: list[dict[str, str]]) -> int:
    group_path = f"{base_path}/{smart_group.confidence}"
    return write_group(album_root, group_path, group_id, smart_group.items, kind, apply, rows)


def confidence_numbered(groups: list[SmartGroup]):
    counters: dict[str, int] = defaultdict(int)
    for group in groups:
        counters[group.confidence] += 1
        yield counters[group.confidence], group


def album_name_for_dest(album_root: Path, dest: Path) -> str:
    try:
        return str(dest.parent.relative_to(album_root))
    except ValueError:
        return str(dest.parent)


def manifest_row(info: ImageInfo, album: str, link: Path) -> dict[str, str]:
    return {
        "album": album,
        "source": str(info.path),
        "link": str(link),
        "width": str(info.width),
        "height": str(info.height),
        "quality": f"{info.quality:.4f}",
        "sharpness": f"{info.sharpness:.1f}",
        "brightness": f"{info.brightness:.1f}",
        "contrast": f"{info.contrast:.1f}",
        "face_count": str(info.face_count),
        "largest_face_ratio": f"{info.largest_face_ratio:.4f}",
        "face_source": info.face_source,
        "face_angle": info.face_angle,
        "face_roll": f"{info.face_roll:.1f}",
        "face_center_x": f"{info.face_center_x:.3f}",
        "face_center_y": f"{info.face_center_y:.3f}",
        "face_width_ratio": f"{info.face_width_ratio:.4f}",
        "face_height_ratio": f"{info.face_height_ratio:.4f}",
        "ai_video_score": f"{video_candidate_score(info):.3f}",
        "nudity_status": info.nudity_status,
        "nudity_class": info.nudity_class,
        "nudity_score": f"{info.nudity_score:.3f}",
        "outfit_color": color_bucket_from_hsv(info.clothing_hue, info.clothing_sat, info.clothing_value),
        "outfit_visibility": outfit_visibility(info),
        "clothing_hue": f"{info.clothing_hue:.1f}",
        "clothing_sat": f"{info.clothing_sat:.1f}",
        "clothing_value": f"{info.clothing_value:.1f}",
        "clothing_skin_ratio": f"{info.clothing_skin_ratio:.3f}",
    }


def link_single(info: ImageInfo,
                album_root: Path,
                album_path: str,
                apply: bool,
                rows: list[dict[str, str]]) -> int:
    dest = link_image(info, album_root / album_path, apply)
    rows.append(manifest_row(info, album_name_for_dest(album_root, dest), dest))
    return 1


def link_collection(infos: list[ImageInfo],
                    album_root: Path,
                    album_path: str,
                    apply: bool,
                    rows: list[dict[str, str]]) -> int:
    count = 0
    for info in infos:
        count += link_single(info, album_root, album_path, apply, rows)
    return count


def link_single_simple(info: ImageInfo,
                       album_root: Path,
                       album_path: str,
                       apply: bool,
                       rows: list[dict[str, str]],
                       nest_nudity: bool = False) -> int:
    if nest_nudity:
        return link_single(info, album_root, album_path, apply, rows)
    dest = link_image_plain(info, album_root / album_path, apply)
    rows.append(manifest_row(info, album_name_for_dest(album_root, dest), dest))
    return 1


def write_smart_group_simple(album_root: Path,
                             base_path: str,
                             group_id: int,
                             smart_group: SmartGroup,
                             kind: str,
                             apply: bool,
                             rows: list[dict[str, str]],
                             nest_nudity: bool = False) -> int:
    group_path = f"{base_path}/{smart_group.confidence}"
    album_path = f"{group_path}/{group_folder_name(group_id, smart_group.items, kind)}"
    count = 0
    for info in smart_group.items:
        count += link_single_simple(info, album_root, album_path, apply, rows, nest_nudity=nest_nudity)
    return count


def v2_safe_for_ready(info: ImageInfo) -> bool:
    return (
        not info.nudity_status
        and info.face_count == 1
        and is_reference_quality(info, min_score=0.50)
    )


def v2_face_visible(info: ImageInfo) -> bool:
    return (
        info.face_count == 1
        and info.largest_face_ratio >= 0.018
        and abs(info.face_roll) <= 22.0
        and is_reference_quality(info, min_score=0.46)
    )


def v2_clear_face(info: ImageInfo) -> bool:
    return (
        v2_face_visible(info)
        and info.largest_face_ratio >= 0.025
        and abs(info.face_roll) <= 16.0
        and 0.20 <= info.face_center_y <= 0.58
    )


def v2_front_straight(info: ImageInfo) -> bool:
    return (
        v2_clear_face(info)
        and info.face_angle == "front_facing"
        and abs(info.face_roll) <= 10.0
        and 0.32 <= info.face_center_x <= 0.68
    )


def v2_saree_base_confidence(info: ImageInfo) -> float:
    if clear_saree_view_album_names(info):
        return 0.92
    if likely_saree_or_draped_ethnic(info):
        return 0.84
    if possible_draped_ethnic(info):
        return 0.62
    return 0.0


def scene_members_by_path(scene_groups: list[SmartGroup]) -> dict[str, list[ImageInfo]]:
    out: dict[str, list[ImageInfo]] = {}
    for group in scene_groups:
        for info in group.items:
            out[str(info.path)] = group.items
    return out


def v2_saree_confidence(info: ImageInfo, scene_map: dict[str, list[ImageInfo]]) -> float:
    base = v2_saree_base_confidence(info)
    if base >= 0.78:
        return base

    members = scene_map.get(str(info.path), [])
    if not members or shot_framing(info) not in {"portrait", "chest_up"} or not v2_clear_face(info):
        return base

    member_scores = [v2_saree_base_confidence(member) for member in members if member.path != info.path]
    strong = [score for score in member_scores if score >= 0.84]
    if len(strong) >= 2:
        return max(base, 0.80)
    if strong and (sum(1 for score in member_scores if score >= 0.62) / max(1, len(member_scores))) >= 0.45:
        return max(base, 0.78)
    return base


def v2_saree_ready_confidence(info: ImageInfo, scene_map: dict[str, list[ImageInfo]]) -> float:
    confidence = v2_saree_confidence(info, scene_map)
    if confidence < 0.78:
        return 0.0
    if info.face_count > 2:
        return 0.0
    clear_view = bool(clear_saree_view_album_names(info))
    drape_score = diagonal_drape_score(info)
    if not clear_view and drape_score < 0.08:
        return 0.0
    if info.clothing_skin_ratio >= 0.34 and not clear_view:
        return 0.0
    if info.clothing_skin_ratio >= 0.48 and info.clothing_sat >= 70.0 and not clear_view:
        return 0.0
    if info.clothing_skin_ratio >= 0.30 and info.clothing_sat >= 70.0 and drape_score < 0.15 and not clear_view:
        return 0.0
    if info.clothing_skin_ratio >= 0.55 and drape_score < 0.08:
        return 0.0
    return confidence


def v2_high_precision_saree_confidence(info: ImageInfo,
                                       scene_map: dict[str, list[ImageInfo]]) -> float:
    confidence = v2_saree_ready_confidence(info, scene_map)
    if confidence < 0.78 or not clear_saree_view_album_names(info):
        return 0.0
    pallu_span = central_pallu_span_score(info)
    midriff_skin = midriff_skin_score(info)
    drape_score = diagonal_drape_score(info)
    skin = info.clothing_skin_ratio

    strong_pallu = pallu_span >= 0.25 and skin >= 0.22
    pallu_with_waist = pallu_span >= 0.12 and midriff_skin >= 0.30 and skin >= 0.20
    colorful_waist_wrap = (
        midriff_skin >= 0.34
        and 0.20 <= skin <= 0.50
        and drape_score >= 0.18
        and not (info.clothing_value >= 215.0 and pallu_span < 0.10)
    )
    bridal_or_bare_waist = skin >= 0.55 and drape_score >= 0.20 and pallu_span >= 0.12
    if strong_pallu or pallu_with_waist or colorful_waist_wrap or bridal_or_bare_waist:
        return confidence
    return 0.0


def v2_clear_saree_ready(info: ImageInfo, scene_map: dict[str, list[ImageInfo]]) -> bool:
    return v2_high_precision_saree_confidence(info, scene_map) >= 0.78


def v2_framing_folder(info: ImageInfo) -> str:
    framing = shot_framing(info)
    if framing == "portrait":
        return "01_close_up_face"
    if framing == "chest_up":
        return "02_chest_up"
    if framing == "waist_up":
        return "03_half_body_or_waist_up"
    return "04_full_body"


def v2_western_framing_folder(info: ImageInfo) -> str:
    framing = shot_framing(info)
    if framing == "portrait":
        return "01_close_up_face"
    if framing in {"chest_up", "waist_up"}:
        return "02_half_body_or_waist_up"
    return "03_full_body"


@lru_cache(maxsize=20000)
def social_screenshot_score_for_path(path: str) -> float:
    img = imread(Path(path))
    if img is None:
        return 0.0
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0
    aspect = w / max(1, h)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    top = gray[:max(1, int(h * 0.10)), :]
    bottom = gray[int(h * 0.84):, :]
    side_w = max(1, int(w * 0.11))
    right = gray[:, w - side_w:]
    right_lower = gray[int(h * 0.35):int(h * 0.96), w - side_w:]
    right_lower_edges = cv2.Canny(right_lower, 70, 150) if right_lower.size else right_lower
    dark_top = float(np.mean(top < 34)) if top.size else 0.0
    dark_bottom = float(np.mean(bottom < 42)) if bottom.size else 0.0
    bright_right = float(np.mean(right > 215)) if right.size else 0.0
    bright_right_lower = float(np.mean(right_lower > 205)) if right_lower.size else 0.0
    edge_right_lower = float(np.mean(right_lower_edges > 0)) if right_lower_edges.size else 0.0
    square_overlay = (
        0.78 <= aspect <= 1.05
        and bright_right_lower >= 0.020
        and edge_right_lower >= 0.025
    )
    portrait_score = 0.0
    if aspect < 0.72:
        portrait_score = max(dark_top, dark_bottom, bright_right * 0.80)
    overlay_score = 0.34 if square_overlay else 0.0
    return max(portrait_score, overlay_score, plain_margin_card_score_for_image(img))


def plain_margin_card_score_for_image(img: np.ndarray) -> float:
    h, w = img.shape[:2]
    if h <= 0 or w <= 0:
        return 0.0
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 70, 150)
    large_regions = [
        gray[:max(1, int(h * 0.25)), :],
        gray[int(h * 0.78):, :],
        gray[:, :max(1, int(w * 0.08))],
        gray[:, int(w * 0.92):],
    ]
    large_edges = [
        edges[:max(1, int(h * 0.25)), :],
        edges[int(h * 0.78):, :],
        edges[:, :max(1, int(w * 0.08))],
        edges[:, int(w * 0.92):],
    ]
    quiet_regions = [
        float(np.std(region)) <= 30.0 and float(np.mean(edge > 0)) <= 0.030
        for region, edge in zip(large_regions, large_edges)
        if region.size and edge.size
    ]
    center = gray[int(h * 0.24):int(h * 0.78), int(w * 0.08):int(w * 0.92)]
    if sum(1 for ok in quiet_regions if ok) >= 3 and center.size and float(np.std(center)) >= 34.0:
        return 0.82

    border = max(12, min(h, w) // 9)
    if border * 2 >= min(h, w):
        return 0.0
    bands = [
        gray[:border, :],
        gray[-border:, :],
        gray[:, :border],
        gray[:, -border:],
    ]
    uniform = [float(np.std(band)) <= 18.0 for band in bands if band.size]
    if not uniform:
        return 0.0
    uniform_ratio = sum(1 for ok in uniform if ok) / len(uniform)
    if uniform_ratio < 0.50:
        return 0.0
    center = gray[border:h - border, border:w - border]
    if center.size == 0 or float(np.std(center)) < 28.0:
        return 0.0
    top_h = max(1, int(h * 0.18))
    bottom_h = max(1, int(h * 0.18))
    side_w = max(1, int(w * 0.10))
    margin_area = (top_h + bottom_h) * w + 2 * side_w * max(1, h - top_h - bottom_h)
    margin_fraction = margin_area / max(1, h * w)
    if margin_fraction < 0.28:
        return 0.0
    return min(0.95, 0.45 + uniform_ratio * 0.35 + margin_fraction * 0.35)


def likely_social_screenshot(info: ImageInfo) -> bool:
    return social_screenshot_score_for_path(str(info.path)) >= 0.28


def v2_clean_ready_source(info: ImageInfo) -> bool:
    name = info.path.name.lower()
    rel = str(info.rel).lower()
    if "_q_review" in name or "_q_low" in name or "/review/" in rel:
        return False
    return not likely_social_screenshot(info)


def v2_saree_reference_score(info: ImageInfo, scene_map: dict[str, list[ImageInfo]]) -> float:
    framing = shot_framing(info)
    score = (
        0.34 * v2_saree_ready_confidence(info, scene_map)
        + 0.34 * video_candidate_score(info)
        + 0.18 * clamp(info.quality, 0.0, 1.0)
    )
    if framing in {"waist_up", "full_body"}:
        score += 0.16
    elif framing == "chest_up":
        score += 0.08
    else:
        score -= 0.18
    if v2_front_straight(info):
        score += 0.08
    elif v2_clear_face(info):
        score += 0.05
    if info.face_angle in {"turned_left", "turned_right", "front_facing"}:
        score += 0.04
    score += min(0.05, diagonal_drape_score(info) * 0.08)
    if info.clothing_skin_ratio >= 0.58:
        score -= 0.18
    elif info.clothing_skin_ratio >= 0.42:
        score -= 0.07
    if not v2_clean_ready_source(info):
        score -= 1.0
    return score


@lru_cache(maxsize=20000)
def background_region_stats_for_path(path: str) -> dict[str, float]:
    img = imread(Path(path))
    if img is None:
        return {"hue": 0.0, "sat": 0.0, "value": 0.0, "gray_std": 0.0, "edge_density": 0.0}
    h, w = img.shape[:2]
    border = max(8, min(h, w) // 7)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:border, :] = 255
    mask[-border:, :] = 255
    mask[:, :border] = 255
    mask[:, -border:] = 255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    selected = hsv[mask > 0]
    if selected.size == 0:
        return {"hue": 0.0, "sat": 0.0, "value": 0.0, "gray_std": 0.0, "edge_density": 0.0}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edge = cv2.Canny(gray, 70, 150)
    border_gray = gray[mask > 0]
    border_edge = edge[mask > 0]
    return {
        "hue": float(np.median(selected[:, 0])),
        "sat": float(np.median(selected[:, 1])),
        "value": float(np.median(selected[:, 2])),
        "gray_std": float(np.std(border_gray)) if border_gray.size else 0.0,
        "edge_density": float(np.mean(border_edge > 0)) if border_edge.size else 0.0,
    }


def path_keyword_flags(info: ImageInfo) -> set[str]:
    text = str(info.rel).lower().replace("_", " ").replace("-", " ")
    flags: set[str] = set()
    keywords = {
        "bedroom": ["bedroom"],
        "bed": [" bed ", "on bed", "bed visible", "bedsheet", "bed sheet"],
        "standing": ["standing", "stand "],
        "sitting": ["sitting", "seated", "sit "],
        "laying": ["laying", "lying", "reclining"],
        "outdoor": ["outdoor", "outside", "garden", "beach", "street"],
        "vehicle": ["car", "bus", "train", "vehicle"],
        "stage": ["stage", "event", "award", "function"],
        "studio": ["studio", "plain background", "white background"],
    }
    padded = f" {text} "
    for label, words in keywords.items():
        if any(word in padded for word in words):
            flags.add(label)
    return flags


def v2_background_tags(info: ImageInfo) -> list[SemanticTag]:
    tags: list[SemanticTag] = []
    flags = path_keyword_flags(info)
    if "bedroom" in flags:
        tags.append(SemanticTag(info.path, "background", "bedroom", 0.95, "path_keyword"))
    if "bed" in flags:
        tags.append(SemanticTag(info.path, "background", "bed_visible_or_on_bed", 0.92, "path_keyword"))
    if "vehicle" in flags:
        tags.append(SemanticTag(info.path, "background", "vehicle_or_travel", 0.90, "path_keyword"))
    if "stage" in flags:
        tags.append(SemanticTag(info.path, "background", "stage_or_event", 0.88, "path_keyword"))
    if "studio" in flags:
        tags.append(SemanticTag(info.path, "background", "studio_or_plain_background", 0.90, "path_keyword"))
    if "outdoor" in flags:
        tags.append(SemanticTag(info.path, "background", "outdoor", 0.90, "path_keyword"))

    stats = background_region_stats_for_path(str(info.path))
    if stats["sat"] <= 26.0 and stats["gray_std"] <= 34.0 and stats["edge_density"] <= 0.060:
        tags.append(SemanticTag(info.path, "background", "studio_or_plain_background", 0.80, "plain_background_heuristic"))
    if (
        ((38.0 <= stats["hue"] <= 92.0) or (95.0 <= stats["hue"] <= 122.0))
        and stats["sat"] >= 58.0
        and stats["value"] >= 92.0
        and info.brightness >= 82.0
    ):
        tags.append(SemanticTag(info.path, "background", "outdoor", 0.74, "green_blue_background_heuristic"))
    if info.brightness < 95.0 and info.context_sat >= 72.0 and info.contrast >= 62.0:
        tags.append(SemanticTag(info.path, "background", "stage_or_event", 0.70, "dark_colorful_scene_heuristic"))
    if not tags:
        tags.append(SemanticTag(info.path, "background", "background_uncertain", 0.35, "no_high_confidence_signal"))
    return tags


def v2_pose_tags(info: ImageInfo) -> list[SemanticTag]:
    tags: list[SemanticTag] = []
    flags = path_keyword_flags(info)
    if "standing" in flags:
        tags.append(SemanticTag(info.path, "pose", "standing", 0.92, "path_keyword"))
    if "sitting" in flags:
        tags.append(SemanticTag(info.path, "pose", "sitting", 0.92, "path_keyword"))
    if "laying" in flags:
        tags.append(SemanticTag(info.path, "pose", "laying_down", 0.92, "path_keyword"))
    if "bed" in flags and ("sitting" in flags or "laying" in flags):
        tags.append(SemanticTag(info.path, "pose", "on_bed_sitting_or_reclining", 0.90, "path_keyword"))
    if "bed" in flags and "laying" in flags:
        tags.append(SemanticTag(info.path, "pose", "laying_on_bed", 0.92, "path_keyword"))

    framing = shot_framing(info)
    if (
        framing == "full_body"
        and info.face_count == 1
        and abs(info.face_roll) <= 14.0
        and 0.18 <= info.face_center_y <= 0.44
        and info.face_height_ratio <= 0.22
    ):
        tags.append(SemanticTag(info.path, "pose", "standing", 0.72, "upright_full_body_heuristic"))
    if (
        framing == "full_body"
        and info.face_count == 1
        and info.face_angle in {"front_facing", "turned_left", "turned_right"}
        and video_candidate_score(info) >= 0.55
    ):
        tags.append(SemanticTag(info.path, "pose", "walking_or_action", 0.70, "full_body_video_candidate_heuristic"))
    if not tags:
        tags.append(SemanticTag(info.path, "pose", "pose_uncertain", 0.35, "no_high_confidence_signal"))
    return tags


def v2_semantic_tags_for_info(info: ImageInfo,
                              scene_map: dict[str, list[ImageInfo]]) -> list[SemanticTag]:
    tags: list[SemanticTag] = []
    framing = shot_framing(info)
    tags.append(SemanticTag(info.path, "framing", framing, 0.90, "face_geometry"))
    if v2_front_straight(info):
        tags.append(SemanticTag(info.path, "face", "clear_straight_face", 0.92, "insightface_geometry"))
    elif v2_clear_face(info):
        tags.append(SemanticTag(info.path, "face", "clear_face_visible", 0.84, "insightface_geometry"))
    elif v2_face_visible(info):
        tags.append(SemanticTag(info.path, "face", "face_visible_but_not_perfect", 0.70, "insightface_geometry"))
    else:
        tags.append(SemanticTag(info.path, "face", "face_uncertain_or_bad", 0.45, "insightface_geometry"))

    saree_conf = v2_high_precision_saree_confidence(info, scene_map)
    if saree_conf >= 0.78:
        tags.append(SemanticTag(info.path, "attire", "saree", saree_conf, "drape_color_scene_heuristic"))
    elif possible_draped_ethnic(info) or v2_saree_ready_confidence(info, scene_map) >= 0.78:
        tags.append(SemanticTag(info.path, "attire", "uncertain_draped_ethnic", max(0.55, saree_conf), "drape_guard"))
    elif likely_western_or_modern(info):
        tags.append(SemanticTag(info.path, "attire", "western_or_modern", 0.76, "conservative_outfit_heuristic"))
    else:
        tags.append(SemanticTag(info.path, "attire", "outfit_uncertain", 0.40, "no_high_confidence_signal"))

    tags.append(SemanticTag(
        info.path,
        "outfit_color",
        color_bucket_from_hsv(info.clothing_hue, info.clothing_sat, info.clothing_value),
        0.70,
        "lower_body_hsv",
    ))
    tags.extend(v2_pose_tags(info))
    tags.extend(v2_background_tags(info))
    return tags


def semantic_best(tags: list[SemanticTag], category: str) -> SemanticTag | None:
    matches = [tag for tag in tags if tag.category == category]
    if not matches:
        return None
    return max(matches, key=lambda tag: (tag.confidence, tag.label))


def semantic_ready_labels(tags: list[SemanticTag], category: str, min_confidence: float = 0.70) -> list[str]:
    return sorted({
        tag.label
        for tag in tags
        if tag.category == category and tag.confidence >= min_confidence and not tag.label.endswith("_uncertain")
    })


def write_contact_sheet(infos: list[ImageInfo],
                        output_path: Path,
                        title: str,
                        apply: bool,
                        max_images: int = 40) -> None:
    if not apply or not infos:
        return
    try:
        from PIL import Image, ImageDraw, ImageOps
    except Exception:
        return
    thumb_w, thumb_h = 180, 180
    label_h = 34
    margin = 12
    title_h = 42
    cols = min(5, max(1, len(infos)))
    selected = infos[:max_images]
    rows = int(np.ceil(len(selected) / cols))
    sheet_w = margin * 2 + cols * thumb_w
    sheet_h = margin * 2 + title_h + rows * (thumb_h + label_h)
    sheet = Image.new("RGB", (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((margin, margin), title[:120], fill=(20, 20, 20))
    for idx, info in enumerate(selected):
        try:
            with Image.open(info.path) as im:
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((thumb_w, thumb_h))
                x = margin + (idx % cols) * thumb_w
                y = margin + title_h + (idx // cols) * (thumb_h + label_h)
                tile = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
                tile.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
                sheet.paste(tile, (x, y))
                draw.text((x + 4, y + thumb_h + 3), info.path.stem[:28], fill=(30, 30, 30))
        except Exception:
            continue
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=88)


def v2_manifest_row(info: ImageInfo, album: str, link: Path) -> dict[str, str]:
    row = manifest_row(info, album, link)
    row["v2_album"] = album
    row["v2_link"] = str(link)
    row["framing"] = shot_framing(info)
    return row


def link_v2(info: ImageInfo,
            album_root: Path,
            album_path: str,
            apply: bool,
            rows: list[dict[str, str]],
            prefix: str = "") -> int:
    dest = link_image_plain(info, album_root / album_path, apply, prefix=prefix)
    rows.append(v2_manifest_row(info, album_name_for_dest(album_root, dest), dest))
    return 1


def link_v2_ranked(infos: list[ImageInfo],
                   album_root: Path,
                   album_path: str,
                   apply: bool,
                   rows: list[dict[str, str]],
                   limit: int = 50) -> int:
    count = 0
    for idx, info in enumerate(infos[:limit], start=1):
        count += link_v2(info, album_root, album_path, apply, rows, prefix=f"{idx:03d}__")
    write_contact_sheet(infos[:limit], album_root / album_path / "_contact_sheet.jpg", album_path, apply)
    return count


def v2_write_csv(path: Path, rows: list[dict[str, str]], apply: bool) -> None:
    if not apply:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def v2_write_json(path: Path, data: dict, apply: bool) -> None:
    if not apply:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)


def ensure_dir_symlink(target: Path, link: Path, apply: bool) -> None:
    if not apply:
        return
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.exists() or link.is_symlink():
        if link.is_symlink() or link.is_file():
            link.unlink()
        elif link.is_dir():
            shutil.rmtree(link)
    try:
        os.symlink(str(target), str(link), target_is_directory=True)
    except TypeError:
        os.symlink(str(target), str(link))
    except OSError:
        link.mkdir(parents=True, exist_ok=True)


def v2_image_index_row(info: ImageInfo,
                       tags: list[SemanticTag],
                       scene_map: dict[str, list[ImageInfo]]) -> dict[str, str]:
    attire = semantic_best(tags, "attire")
    pose = semantic_best(tags, "pose")
    background = semantic_best(tags, "background")
    face = semantic_best(tags, "face")
    return {
        "source": str(info.path),
        "rel": str(info.rel),
        "width": str(info.width),
        "height": str(info.height),
        "quality": f"{info.quality:.4f}",
        "ai_video_score": f"{video_candidate_score(info):.3f}",
        "framing": shot_framing(info),
        "face_label": face.label if face else "",
        "face_confidence": f"{face.confidence:.3f}" if face else "",
        "attire_label": attire.label if attire else "",
        "attire_confidence": f"{attire.confidence:.3f}" if attire else "",
        "pose_label": pose.label if pose else "",
        "pose_confidence": f"{pose.confidence:.3f}" if pose else "",
        "background_label": background.label if background else "",
        "background_confidence": f"{background.confidence:.3f}" if background else "",
        "outfit_color": color_bucket_from_hsv(info.clothing_hue, info.clothing_sat, info.clothing_value),
        "nudity_status": info.nudity_status,
        "same_scene_size": str(len(scene_map.get(str(info.path), []))),
    }


def v2_score_row(info: ImageInfo, tags: list[SemanticTag]) -> dict[str, str]:
    attire = semantic_best(tags, "attire")
    face = semantic_best(tags, "face")
    return {
        "source": str(info.path),
        "ai_video_score": f"{video_candidate_score(info):.3f}",
        "quality": f"{info.quality:.4f}",
        "safe_for_ready": "1" if v2_safe_for_ready(info) else "0",
        "clear_face": "1" if v2_clear_face(info) else "0",
        "front_straight": "1" if v2_front_straight(info) else "0",
        "framing": shot_framing(info),
        "attire": attire.label if attire else "",
        "attire_confidence": f"{attire.confidence:.3f}" if attire else "",
        "face": face.label if face else "",
        "face_confidence": f"{face.confidence:.3f}" if face else "",
    }


def v2_scene_folder_name(idx: int, group: SmartGroup) -> str:
    return group_folder_name(idx, group.items, "scene")


def v2_pack_slug(start: ImageInfo, tags: list[SemanticTag]) -> str:
    color = color_bucket_from_hsv(start.clothing_hue, start.clothing_sat, start.clothing_value)
    background = semantic_best(tags, "background")
    bg = background.label if background and background.confidence >= 0.70 else "background_uncertain"
    return safe_component(f"{color}_saree_{shot_framing(start)}_{bg}", max_len=70)


def build_smart_v2(person_dir: Path,
                   infos: list[ImageInfo],
                   scene_groups: list[SmartGroup],
                   apply: bool,
                   quiet: bool) -> dict[str, int]:
    stats = {
        "v2_links": 0,
        "v2_packs": 0,
        "v2_scene_groups": 0,
    }
    album_root = person_dir / SMART_V2_DIR
    if apply:
        clear_smart_albums_v2(person_dir)
        ensure_smart_v2_structure(album_root, apply)
    if not infos:
        return stats

    scene_map = scene_members_by_path(scene_groups)
    semantic_map: dict[str, list[SemanticTag]] = {
        str(info.path): v2_semantic_tags_for_info(info, scene_map)
        for info in infos
    }
    rows: list[dict[str, str]] = []
    image_rows = [v2_image_index_row(info, semantic_map[str(info.path)], scene_map) for info in infos]
    score_rows = [v2_score_row(info, semantic_map[str(info.path)]) for info in infos]
    semantic_rows = [
        {
            "source": str(tag.source),
            "category": tag.category,
            "label": tag.label,
            "confidence": f"{tag.confidence:.3f}",
            "method": tag.method,
        }
        for tags in semantic_map.values()
        for tag in tags
    ]
    scene_rows: list[dict[str, str]] = []
    pack_rows: list[dict[str, str]] = []

    safe_infos = [info for info in infos if not info.nudity_status]
    ready_infos = [info for info in safe_infos if v2_safe_for_ready(info)]
    clean_ready_infos = [info for info in ready_infos if v2_clean_ready_source(info)]
    best_ai = sorted(
        [info for info in clean_ready_infos if video_candidate_score(info) >= 0.58 and v2_face_visible(info)],
        key=lambda i: (-video_candidate_score(i), -i.quality, str(i.rel).lower()),
    )
    best_saree = sorted(
        [
            info for info in clean_ready_infos
            if (
                v2_clear_saree_ready(info, scene_map)
                and v2_face_visible(info)
                and shot_framing(info) in {"chest_up", "waist_up", "full_body"}
                and v2_saree_reference_score(info, scene_map) >= 0.82
            )
        ],
        key=lambda i: (-v2_saree_reference_score(i, scene_map), -video_candidate_score(i), -i.quality, str(i.rel).lower()),
    )
    image_edit = sorted(
        [info for info in clean_ready_infos if v2_clear_face(info) and info.quality >= 0.58],
        key=lambda i: (-i.quality, -video_candidate_score(i), str(i.rel).lower()),
    )

    stats["v2_links"] += link_v2_ranked(best_ai, album_root, "00_START_HERE/01_best_ai_video_inputs", apply, rows, 50)
    stats["v2_links"] += link_v2_ranked(best_saree, album_root, "00_START_HERE/02_best_saree_inputs", apply, rows, 50)
    stats["v2_links"] += link_v2_ranked(image_edit, album_root, "00_START_HERE/04_best_image_edit_inputs", apply, rows, 50)

    for info in infos:
        tags = semantic_map[str(info.path)]
        if info.nudity_status:
            stats["v2_links"] += link_v2(info, album_root, "90_REVIEW/nudity_possible", apply, rows)
            continue

        quality_bucket = "large_high_score" if info.quality >= 0.68 and info.width * info.height >= 800_000 else "usable"
        if not is_reference_quality(info, min_score=0.46):
            quality_bucket = "low_quality"
            stats["v2_links"] += link_v2(info, album_root, "90_REVIEW/low_quality", apply, rows)
        stats["v2_links"] += link_v2(info, album_root, f"80_FACETS_OPTIONAL/quality/{quality_bucket}", apply, rows)
        stats["v2_links"] += link_v2(info, album_root, f"80_FACETS_OPTIONAL/format/{orientation_name(info)}", apply, rows)
        stats["v2_links"] += link_v2(
            info,
            album_root,
            f"80_FACETS_OPTIONAL/color/{color_bucket_from_hsv(info.clothing_hue, info.clothing_sat, info.clothing_value)}",
            apply,
            rows,
        )
        stats["v2_links"] += link_v2(info, album_root, f"80_FACETS_OPTIONAL/raw_face_angle/{info.face_angle or 'unknown'}", apply, rows)

        attire = semantic_best(tags, "attire")
        attire_label = attire.label if attire else "outfit_uncertain"
        stats["v2_links"] += link_v2(info, album_root, f"80_FACETS_OPTIONAL/raw_outfit_tags/{attire_label}", apply, rows)

        if v2_clear_saree_ready(info, scene_map) and v2_face_visible(info):
            stats["v2_links"] += link_v2(info, album_root, f"01_ATTIRE_AND_FRAMING/saree/{v2_framing_folder(info)}", apply, rows)
            if info.face_angle in {"turned_left", "turned_right"} and abs(info.face_roll) <= 18.0:
                stats["v2_links"] += link_v2(info, album_root, "01_ATTIRE_AND_FRAMING/saree/05_three_quarter_view", apply, rows)
            if v2_front_straight(info):
                stats["v2_links"] += link_v2(info, album_root, "01_ATTIRE_AND_FRAMING/saree/06_front_straight_face", apply, rows)
            if v2_clear_face(info):
                stats["v2_links"] += link_v2(info, album_root, "01_ATTIRE_AND_FRAMING/saree/07_clear_face_visible", apply, rows)
        elif attire_label == "uncertain_draped_ethnic":
            stats["v2_links"] += link_v2(info, album_root, "01_ATTIRE_AND_FRAMING/uncertain_draped_ethnic", apply, rows)
            stats["v2_links"] += link_v2(info, album_root, "90_REVIEW/outfit_uncertain", apply, rows)
        elif attire_label == "western_or_modern" and v2_face_visible(info):
            stats["v2_links"] += link_v2(info, album_root, f"01_ATTIRE_AND_FRAMING/western_or_modern/{v2_western_framing_folder(info)}", apply, rows)
        else:
            stats["v2_links"] += link_v2(info, album_root, "90_REVIEW/outfit_uncertain", apply, rows)

        if v2_front_straight(info):
            stats["v2_links"] += link_v2(info, album_root, "02_FACE_REFERENCES/01_clear_straight_face", apply, rows)
            stats["v2_links"] += link_v2(info, album_root, "02_FACE_REFERENCES/02_front_face", apply, rows)
        elif info.face_angle == "front_facing" and v2_face_visible(info):
            stats["v2_links"] += link_v2(info, album_root, "02_FACE_REFERENCES/02_front_face", apply, rows)
        elif info.face_angle == "turned_left" and v2_face_visible(info):
            stats["v2_links"] += link_v2(info, album_root, "02_FACE_REFERENCES/03_left_three_quarter", apply, rows)
        elif info.face_angle == "turned_right" and v2_face_visible(info):
            stats["v2_links"] += link_v2(info, album_root, "02_FACE_REFERENCES/04_right_three_quarter", apply, rows)
        elif info.face_angle in {"side_angle_left", "side_angle_right"} and v2_face_visible(info):
            stats["v2_links"] += link_v2(info, album_root, "02_FACE_REFERENCES/05_side_profile", apply, rows)
        elif v2_face_visible(info):
            stats["v2_links"] += link_v2(info, album_root, "02_FACE_REFERENCES/06_face_visible_but_not_perfect", apply, rows)
        else:
            stats["v2_links"] += link_v2(info, album_root, "02_FACE_REFERENCES/90_face_uncertain_or_bad", apply, rows)
            stats["v2_links"] += link_v2(info, album_root, "90_REVIEW/face_uncertain", apply, rows)

        pose_labels = semantic_ready_labels(tags, "pose")
        if pose_labels:
            for label in pose_labels:
                pose_album = {
                    "standing": "03_BODY_POSE/01_standing",
                    "sitting": "03_BODY_POSE/02_sitting",
                    "laying_down": "03_BODY_POSE/03_laying_down",
                    "laying_on_bed": "03_BODY_POSE/04_laying_on_bed",
                    "on_bed_sitting_or_reclining": "03_BODY_POSE/05_on_bed_sitting_or_reclining",
                    "walking_or_action": "03_BODY_POSE/06_walking_or_action",
                }.get(label)
                if pose_album:
                    stats["v2_links"] += link_v2(info, album_root, pose_album, apply, rows)
        else:
            stats["v2_links"] += link_v2(info, album_root, "03_BODY_POSE/90_pose_uncertain", apply, rows)
            stats["v2_links"] += link_v2(info, album_root, "90_REVIEW/pose_uncertain", apply, rows)

        background_labels = semantic_ready_labels(tags, "background")
        if background_labels:
            for label in background_labels:
                background_album = {
                    "bedroom": "04_BACKGROUND_SCENE/01_bedroom",
                    "bed_visible_or_on_bed": "04_BACKGROUND_SCENE/02_bed_visible_or_on_bed",
                    "indoor_home": "04_BACKGROUND_SCENE/03_indoor_home",
                    "studio_or_plain_background": "04_BACKGROUND_SCENE/04_studio_or_plain_background",
                    "stage_or_event": "04_BACKGROUND_SCENE/05_stage_or_event",
                    "outdoor": "04_BACKGROUND_SCENE/06_outdoor",
                    "vehicle_or_travel": "04_BACKGROUND_SCENE/07_vehicle_or_travel",
                }.get(label)
                if background_album:
                    stats["v2_links"] += link_v2(info, album_root, background_album, apply, rows)
        else:
            stats["v2_links"] += link_v2(info, album_root, "04_BACKGROUND_SCENE/90_background_uncertain", apply, rows)
            stats["v2_links"] += link_v2(info, album_root, "90_REVIEW/background_uncertain", apply, rows)

    for idx, group in confidence_numbered(scene_groups):
        safe_group = [info for info in group.items if not info.nudity_status]
        if not safe_group:
            continue
        folder = f"05_SAME_SCENE_SETS/{group.confidence}/{v2_scene_folder_name(idx, group)}"
        for info in safe_group:
            stats["v2_links"] += link_v2(info, album_root, folder, apply, rows)
        write_contact_sheet(safe_group, album_root / folder / "_contact_sheet.jpg", folder, apply)
        rep = max(safe_group, key=lambda i: (video_candidate_score(i), i.quality, i.width * i.height))
        saree_count = sum(1 for info in safe_group if v2_clear_saree_ready(info, scene_map))
        scene_rows.append({
            "group": folder,
            "confidence": group.confidence,
            "count": str(len(safe_group)),
            "representative": str(rep.path),
            "saree_count": str(saree_count),
            "best_ai_video_score": f"{video_candidate_score(rep):.3f}",
        })
    stats["v2_scene_groups"] = len(scene_rows)

    scene_reps = [
        max([info for info in group.items if not info.nudity_status],
            key=lambda i: (video_candidate_score(i), i.quality, i.width * i.height))
        for group in scene_groups
        if any(not info.nudity_status for info in group.items)
    ]
    scene_reps = sorted(scene_reps, key=lambda i: (-video_candidate_score(i), -i.quality, str(i.rel).lower()))
    stats["v2_links"] += link_v2_ranked(scene_reps, album_root, "00_START_HERE/03_best_same_scene_sets", apply, rows, 50)

    pack_sources = {
        "talking_head": [
            info for info in best_ai
            if shot_framing(info) in {"portrait", "chest_up"} and info.face_angle == "front_facing"
        ],
        "orbit_head_turn": [
            info for info in best_ai
            if info.face_angle in {"turned_left", "turned_right", "side_angle_left", "side_angle_right"}
        ],
        "cinematic_waist_up": [
            info for info in best_ai
            if shot_framing(info) in {"chest_up", "waist_up"} and info.face_angle in {"turned_left", "turned_right", "front_facing"}
        ],
        "walking_or_full_body": [
            info for info in best_ai
            if shot_framing(info) == "full_body"
        ],
    }
    for pack_name, pack_infos in pack_sources.items():
        stats["v2_links"] += link_v2_ranked(pack_infos, album_root, f"06_AI_VIDEO_PACKS/{pack_name}", apply, rows, 20)

    identity_refs = sorted(
        [info for info in clean_ready_infos if v2_clear_face(info)],
        key=lambda i: (-video_candidate_score(i), -i.quality, str(i.rel).lower()),
    )
    shared_identity_refs = identity_refs[:8]
    stats["v2_links"] += link_v2_ranked(
        shared_identity_refs,
        album_root,
        "06_AI_VIDEO_PACKS/_shared_identity_refs",
        apply,
        rows,
        8,
    )
    saree_pack_candidates: list[tuple[SmartGroup, list[ImageInfo]]] = []
    for group in scene_groups:
        safe_group = [info for info in group.items if not info.nudity_status]
        saree_group = [
            info for info in safe_group
            if (
                v2_clear_saree_ready(info, scene_map)
                and v2_face_visible(info)
                and v2_clean_ready_source(info)
                and v2_saree_reference_score(info, scene_map) >= 0.82
            )
        ]
        if saree_group and group.confidence in {"high_confidence", "medium_confidence"}:
            saree_pack_candidates.append((group, saree_group))
    saree_pack_candidates = sorted(
        saree_pack_candidates,
        key=lambda item: (-len(item[1]), item[0].confidence, str(item[1][0].rel).lower()),
    )[:20]
    for pack_idx, (group, saree_group) in enumerate(saree_pack_candidates, start=1):
        start = max(saree_group, key=lambda i: (video_candidate_score(i), i.quality, i.width * i.height))
        start_tags = semantic_map[str(start.path)]
        slug = v2_pack_slug(start, start_tags)
        pack_folder = f"06_AI_VIDEO_PACKS/saree_scene_packs/{pack_idx:03d}_{slug}"
        stats["v2_links"] += link_v2(start, album_root, f"{pack_folder}/00_start_frame", apply, rows, prefix="001__")
        refs = [info for info in shared_identity_refs if info.path != start.path]
        ensure_dir_symlink(album_root / "06_AI_VIDEO_PACKS" / "_shared_identity_refs",
                           album_root / pack_folder / "01_identity_refs",
                           apply)
        safe_scene_group = [info for info in group.items if not info.nudity_status]
        scene_alts = sorted(safe_scene_group, key=lambda i: (-video_candidate_score(i), -i.quality, str(i.rel).lower()))[:12]
        stats["v2_links"] += link_v2_ranked(scene_alts, album_root, f"{pack_folder}/02_same_scene_alternates", apply, rows, 12)
        start_color = color_bucket_from_hsv(start.clothing_hue, start.clothing_sat, start.clothing_value)
        attire_alts = [
            info for info in best_saree
            if info.path not in {member.path for member in scene_alts}
            and color_bucket_from_hsv(info.clothing_hue, info.clothing_sat, info.clothing_value) == start_color
        ][:12]
        stats["v2_links"] += link_v2_ranked(attire_alts, album_root, f"{pack_folder}/03_same_attire_alternates", apply, rows, 12)
        contact_infos = [start] + refs[:8] + scene_alts[:12]
        write_contact_sheet(contact_infos, album_root / pack_folder / "_contact_sheet.jpg", pack_folder, apply)
        pack_info = {
            "pack": pack_folder,
            "start_frame": str(start.path),
            "scene_confidence": group.confidence,
            "same_scene_count": len(scene_alts),
            "identity_ref_count": len(refs),
            "identity_refs": str(album_root / "06_AI_VIDEO_PACKS" / "_shared_identity_refs"),
            "same_attire_count": len(attire_alts),
            "attire": "saree",
            "outfit_color": start_color,
            "framing": shot_framing(start),
            "ai_video_score": round(video_candidate_score(start), 3),
            "tags": [
                {
                    "category": tag.category,
                    "label": tag.label,
                    "confidence": round(tag.confidence, 3),
                    "method": tag.method,
                }
                for tag in start_tags
            ],
        }
        v2_write_json(album_root / pack_folder / "_pack_info.json", pack_info, apply)
        pack_rows.append({
            "pack": pack_folder,
            "start_frame": str(start.path),
            "scene_confidence": group.confidence,
            "same_scene_count": str(len(scene_alts)),
            "identity_ref_count": str(len(refs)),
            "same_attire_count": str(len(attire_alts)),
            "outfit_color": start_color,
            "framing": shot_framing(start),
            "ai_video_score": f"{video_candidate_score(start):.3f}",
        })
    stats["v2_packs"] = len(pack_rows)

    start_here_infos = (best_ai[:20] + best_saree[:20] + scene_reps[:20] + image_edit[:20])
    seen: set[str] = set()
    deduped_start_here: list[ImageInfo] = []
    for info in start_here_infos:
        key = str(info.path)
        if key not in seen:
            seen.add(key)
            deduped_start_here.append(info)
    write_contact_sheet(deduped_start_here, album_root / "00_START_HERE" / "_contact_sheet.jpg", "00_START_HERE", apply)
    pack_contact = [
        Path(row["start_frame"])
        for row in pack_rows
        if row.get("start_frame")
    ]
    pack_contact_infos = [info for info in infos if info.path in set(pack_contact)]
    write_contact_sheet(pack_contact_infos, album_root / "06_AI_VIDEO_PACKS" / "_contact_sheet.jpg", "06_AI_VIDEO_PACKS", apply)

    v2_write_csv(album_root / "_data" / "image_index.csv", image_rows, apply)
    v2_write_csv(album_root / "_data" / "semantic_tags.csv", semantic_rows, apply)
    v2_write_csv(album_root / "_data" / "scene_groups.csv", scene_rows, apply)
    v2_write_csv(album_root / "_data" / "ai_video_scores.csv", score_rows, apply)
    v2_write_csv(album_root / "_data" / "pack_index.csv", pack_rows, apply)
    v2_write_csv(album_root / "_data" / "link_index.csv", rows, apply)

    if not quiet:
        print(f"{'':<32} smart_v2_links={stats['v2_links']:<5} "
              f"smart_v2_scene_groups={stats['v2_scene_groups']:<4} "
              f"smart_v2_packs={stats['v2_packs']}", flush=True)
    return stats


def write_smart_album_indexes(album_root: Path, rows: list[dict[str, str]], apply: bool) -> None:
    if not apply:
        return
    fieldnames = [
        "album",
        "source",
        "link",
        "width",
        "height",
        "quality",
        "sharpness",
        "brightness",
        "contrast",
        "face_count",
        "largest_face_ratio",
        "face_source",
        "face_angle",
        "face_roll",
        "face_center_x",
        "face_center_y",
        "face_width_ratio",
        "face_height_ratio",
        "ai_video_score",
        "nudity_status",
        "nudity_class",
        "nudity_score",
        "outfit_color",
        "outfit_visibility",
        "clothing_hue",
        "clothing_sat",
        "clothing_value",
        "clothing_skin_ratio",
    ]
    for manifest in [album_root / "_smart_album_index.csv", album_root / "_data" / "smart_album_index.csv"]:
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


def contact_sheet_filename(album: str, page: int) -> str:
    stem = safe_component(album.replace("/", "__"), max_len=110)
    return f"{stem}__page_{page:03d}.jpg"


def contact_sheet_category(album: str) -> str:
    parts = album.split("/")
    if not parts:
        return "other"
    if parts[0] == "saree_candidates_for_validation":
        return "saree/saree_candidates_for_validation"
    if parts[0] == "03_face_framing":
        return "face_framing"
    if parts[0] == "06_saree_clear_views":
        return "saree"
    if parts[0] == "04_visual_similar":
        return "visual_similar"
    if parts[0] == "05_same_scene":
        return "same_scene"
    return "other"


def validation_contact_sheet_album(album: str) -> str | None:
    parts = album.split("/")
    if not parts:
        return None
    if parts[0] in {"03_face_framing", "06_saree_clear_views"} and len(parts) >= 2:
        if len(parts) >= 3 and parts[2] in {"_nudity_possible", "_nudity_uncertain"}:
            return "/".join(parts[:3])
        return "/".join(parts[:2])
    if parts[0] in {"04_visual_similar", "05_same_scene"} and len(parts) >= 3:
        return album
    return None


def write_validation_contact_sheets(album_root: Path,
                                    rows: list[dict[str, str]],
                                    apply: bool,
                                    extra_rows: list[dict[str, str]] | None = None,
                                    per_page: int = 30) -> None:
    if not apply:
        return
    try:
        from PIL import Image, ImageDraw, ImageFont, ImageOps
    except Exception:
        return

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    seen: dict[str, set[str]] = defaultdict(set)
    for row in [*rows, *(extra_rows or [])]:
        album = validation_contact_sheet_album(str(row.get("album", "")))
        if not album and extra_rows and row in extra_rows:
            album = str(row.get("album", ""))
        source = str(row.get("source", ""))
        if not album or not source or source in seen[album]:
            continue
        seen[album].add(source)
        grouped[album].append(row)

    out_dir = album_root / "_data" / "contact_sheets"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    index_rows: list[dict[str, str]] = []
    thumb_w, thumb_h = 190, 220
    label_h = 46
    title_h = 42
    margin = 12
    cols = 5
    font = ImageFont.load_default()
    for album in sorted(grouped):
        album_rows = sorted(grouped[album], key=lambda r: str(r.get("source", "")).lower())
        for page_num, start in enumerate(range(0, len(album_rows), max(1, per_page)), start=1):
            chunk = album_rows[start:start + max(1, per_page)]
            rows_count = int(np.ceil(len(chunk) / cols))
            sheet = Image.new(
                "RGB",
                (margin * 2 + cols * thumb_w, margin * 2 + title_h + rows_count * (thumb_h + label_h)),
                "white",
            )
            draw = ImageDraw.Draw(sheet)
            title = f"{album}  page {page_num}  images {start + 1}-{start + len(chunk)} of {len(album_rows)}"
            draw.text((margin, margin), title[:150], fill=(20, 20, 20), font=font)
            category = contact_sheet_category(album)
            category_dir = out_dir / category
            category_dir.mkdir(parents=True, exist_ok=True)
            filename = contact_sheet_filename(album, page_num)
            contact_sheet_rel = f"{category}/{filename}"
            for local_idx, row in enumerate(chunk, start=1):
                global_idx = start + local_idx
                source = Path(str(row.get("source", "")))
                r, c = divmod(local_idx - 1, cols)
                x = margin + c * thumb_w
                y = margin + title_h + r * (thumb_h + label_h)
                tile = Image.new("RGB", (thumb_w, thumb_h), (245, 245, 245))
                try:
                    with Image.open(source) as im:
                        im = ImageOps.exif_transpose(im).convert("RGB")
                        im.thumbnail((thumb_w, thumb_h), Image.Resampling.LANCZOS)
                        tile.paste(im, ((thumb_w - im.width) // 2, (thumb_h - im.height) // 2))
                except Exception:
                    draw.text((x + 8, y + 8), "ERR", fill=(180, 0, 0), font=font)
                sheet.paste(tile, (x, y))
                badge = f"{global_idx}"
                draw.rectangle((x + 4, y + 4, x + 42, y + 26), fill=(255, 255, 255), outline=(0, 0, 0))
                draw.text((x + 8, y + 8), badge, fill=(0, 0, 0), font=font)
                draw.text((x + 4, y + thumb_h + 4), f"{global_idx}: {source.stem[:24]}", fill=(20, 20, 20), font=font)
                index_rows.append({
                    "contact_sheet": contact_sheet_rel,
                    "category": category,
                    "album": album,
                    "page": str(page_num),
                    "number": str(global_idx),
                    "source": str(source),
                    "link": str(row.get("link", "")),
                })
            sheet.save(category_dir / filename, quality=90)

    index_path = out_dir / "contact_sheet_index.csv"
    with index_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["contact_sheet", "category", "album", "page", "number", "source", "link"])
        writer.writeheader()
        writer.writerows(index_rows)


def build_for_person_simple(person_dir: Path,
                            apply: bool,
                            visual_threshold: int,
                            scene_eps: float,
                            min_group: int,
                            max_scene_group: int,
                            detector,
                            nudity_cache: dict,
                            nudity_overrides: dict[str, str],
                            nudity_batch_size: int,
                            framing_app,
                            framing_cache: dict,
                            framing_cache_path: Path | None,
                            quiet: bool,
                            max_framing_checks: int = 0,
                            max_images: int = 0,
                            album_dir_name: str = SMART_DIR,
                            remove_smart_v2: bool = False,
                            nest_nudity: bool = False,
                            trust_source_nudity: bool = True) -> dict[str, int]:
    infos = load_infos(person_dir, max_images=max_images, trust_source_nudity=trust_source_nudity)
    if apply:
        album_root = clear_named_smart_album_dir(person_dir, album_dir_name)
        ensure_simple_smart_structure(album_root, apply)
        if remove_smart_v2 and album_dir_name == SMART_DIR:
            clear_smart_albums_v2(person_dir)
    else:
        album_root = person_dir / album_dir_name

    framing_scanned, framing_deferred = annotate_framing(
        infos,
        framing_app,
        framing_cache,
        framing_cache_path,
        quiet,
        max_new_checks=max_framing_checks,
    )
    nudity_scanned = annotate_nudity(
        infos,
        detector,
        nudity_cache,
        nudity_overrides,
        nudity_batch_size,
    )
    stats = {
        "images": len(infos),
        "links": 0,
        "visual_groups": 0,
        "scene_groups": 0,
        "best_links": 0,
        "review_links": 0,
        "framing_scanned": framing_scanned,
        "framing_deferred": framing_deferred,
        "face_uncertain": 0,
        "nudity_scanned": nudity_scanned,
        "nudity_images": sum(1 for i in infos if i.nudity_status),
        "outfit_links": 0,
        "v2_links": 0,
        "v2_scene_groups": 0,
        "v2_packs": 0,
    }
    if not infos:
        return stats

    rows: list[dict[str, str]] = []
    for info in infos:
        stats["links"] += link_single_simple(info, album_root, context_name(info), apply, rows, nest_nudity=nest_nudity)
        for album_name in simple_quality_album_names(info):
            stats["links"] += link_single_simple(info, album_root, album_name, apply, rows, nest_nudity=nest_nudity)

    visual_groups = group_visual_similar(infos, visual_threshold, min_group)
    for idx, group in confidence_numbered(visual_groups):
        stats["links"] += write_smart_group_simple(
            album_root, "04_visual_similar", idx, group, "visual", apply, rows, nest_nudity=nest_nudity
        )
    stats["visual_groups"] = len(visual_groups)

    scene_groups = [
        group for group in group_same_scene(infos, scene_eps, min_group, max_scene_group)
        if group.confidence in {"high_confidence", "medium_confidence"}
    ]
    scene_groups = merge_related_same_scene_groups(scene_groups, max(max_scene_group, 40))
    scene_map = scene_members_by_path(scene_groups)
    for idx, group in confidence_numbered(scene_groups):
        stats["links"] += write_smart_group_simple(
            album_root, "05_same_scene", idx, group, "scene", apply, rows, nest_nudity=nest_nudity
        )
    stats["scene_groups"] = len(scene_groups)

    saree_validation_rows = simple_saree_validation_rows(infos, scene_map, limit=100)
    saree_validation_sheet_rows = sorted(
        saree_validation_rows,
        key=lambda row: str(row.get("source", "")).lower(),
    )
    validated_saree_sources = {
        str(row.get("source", ""))
        for row in saree_validation_sheet_rows[:90]
        if row.get("source")
    }

    for info in infos:
        for album_name in simple_face_framing_album_names(info):
            stats["links"] += link_single_simple(info, album_root, album_name, apply, rows, nest_nudity=nest_nudity)
    for info in infos:
        for album_name in simple_saree_album_names(
            info,
            scene_map,
            validated_candidate=str(info.path) in validated_saree_sources,
        ):
            stats["links"] += link_single_simple(info, album_root, album_name, apply, rows, nest_nudity=nest_nudity)
            stats["outfit_links"] += 1

    write_smart_album_indexes(album_root, rows, apply)
    write_validation_contact_sheets(album_root, rows, apply, extra_rows=saree_validation_rows)

    if not quiet:
        subset_note = f" subset={len(infos)}" if max_images and len(infos) <= max_images else ""
        print(f"{person_dir.name:<32} simple{subset_note} images={stats['images']:<5} "
              f"visual_groups={stats['visual_groups']:<4} scene_groups={stats['scene_groups']:<4} "
              f"framing_scan={stats['framing_scanned']:<4} framing_deferred={stats['framing_deferred']:<4} "
              f"nudity={stats['nudity_images']:<4} saree_links={stats['outfit_links']:<4} "
              f"links={stats['links']}", flush=True)
    return stats


def build_for_person(person_dir: Path,
                     apply: bool,
                     visual_threshold: int,
                     scene_eps: float,
                     min_group: int,
                     max_scene_group: int,
                     detector,
                     nudity_cache: dict,
                     nudity_overrides: dict[str, str],
                     nudity_batch_size: int,
                     framing_app,
                     framing_cache: dict,
                     framing_cache_path: Path | None,
                     quiet: bool,
                     max_framing_checks: int = 0) -> dict[str, int]:
    infos = load_infos(person_dir)
    if apply:
        clear_smart_albums(person_dir)
    framing_scanned, framing_deferred = annotate_framing(
        infos,
        framing_app,
        framing_cache,
        framing_cache_path,
        quiet,
        max_new_checks=max_framing_checks,
    )
    nudity_scanned = annotate_nudity(
        infos,
        detector,
        nudity_cache,
        nudity_overrides,
        nudity_batch_size,
    )
    stats = {
        "images": len(infos),
        "links": 0,
        "visual_groups": 0,
        "scene_groups": 0,
        "best_links": 0,
        "review_links": 0,
        "framing_scanned": framing_scanned,
        "framing_deferred": framing_deferred,
        "face_uncertain": 0,
        "nudity_scanned": nudity_scanned,
        "nudity_images": sum(1 for i in infos if i.nudity_status),
        "outfit_links": 0,
        "v2_links": 0,
        "v2_scene_groups": 0,
        "v2_packs": 0,
    }
    if not infos:
        return stats
    album_root = person_dir / SMART_DIR
    rows: list[dict[str, str]] = []

    best = sorted(infos, key=lambda i: (-i.quality, -i.width * i.height, str(i.rel).lower()))
    top_n = min(20, len(best))
    if top_n:
        linked = link_collection(
            best[:top_n],
            album_root,
            f"00_best/top_{top_n:03d}_technical_quality",
            apply,
            rows,
        )
        stats["links"] += linked
        stats["best_links"] += linked

    video_ranked = sorted(
        [i for i in infos if video_candidate_score(i) >= 0.62],
        key=lambda i: (-video_candidate_score(i), -i.quality, -i.width * i.height, str(i.rel).lower()),
    )[:50]
    if video_ranked:
        stats["links"] += link_collection(
            video_ranked,
            album_root,
            "03_face_framing/00_best_ai_video_candidates/top_050_ranked",
            apply,
            rows,
        )

    for info in infos:
        stats["links"] += 1
        dest = link_image(info, album_root / context_name(info), apply)
        rows.append(manifest_row(info, album_name_for_dest(album_root, dest), dest))
        for album_name in quality_album_names(info):
            stats["links"] += link_single(info, album_root, album_name, apply, rows)
            if album_name.startswith("07_review_needed/"):
                stats["review_links"] += 1
        for album_name in face_framing_album_names(info):
            stats["links"] += link_single(info, album_root, album_name, apply, rows)
            if album_name == "03_face_framing/90_not_ideal_for_ai_video/face_detection_uncertain":
                stats["face_uncertain"] += 1
            if album_name.startswith("07_review_needed/"):
                stats["review_links"] += 1
        for album_name in outfit_album_names(info):
            stats["links"] += link_single(info, album_root, album_name, apply, rows)
            stats["outfit_links"] += 1

    visual_groups = group_visual_similar(infos, visual_threshold, min_group)
    for idx, group in confidence_numbered(visual_groups):
        stats["links"] += write_smart_group(album_root, "04_visual_similar", idx, group, "visual", apply, rows)
    stats["visual_groups"] = len(visual_groups)

    scene_groups = group_same_scene(infos, scene_eps, min_group, max_scene_group)
    for idx, group in confidence_numbered(scene_groups):
        stats["links"] += write_smart_group(album_root, "05_same_scene", idx, group, "scene", apply, rows)
    stats["scene_groups"] = len(scene_groups)

    v2_stats = build_smart_v2(person_dir, infos, scene_groups, apply, quiet)
    stats.update(v2_stats)

    if apply:
        manifest = album_root / "_smart_album_index.csv"
        manifest.parent.mkdir(parents=True, exist_ok=True)
        with manifest.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "album",
                "source",
                "link",
                "width",
                "height",
                "quality",
                "sharpness",
                "brightness",
                "contrast",
                "face_count",
                "largest_face_ratio",
                "face_source",
                "face_angle",
                "face_roll",
                "face_center_x",
                "face_center_y",
                "face_width_ratio",
                "face_height_ratio",
                "ai_video_score",
                "nudity_status",
                "nudity_class",
                "nudity_score",
                "outfit_color",
                "outfit_visibility",
                "clothing_hue",
                "clothing_sat",
                "clothing_value",
                "clothing_skin_ratio",
            ])
            writer.writeheader()
            writer.writerows(rows)

    if not quiet:
        print(f"{person_dir.name:<32} images={stats['images']:<5} "
              f"visual_groups={stats['visual_groups']:<4} scene_groups={stats['scene_groups']:<4} "
              f"best={stats['best_links']:<3} review={stats['review_links']:<4} "
              f"face_uncertain={stats['face_uncertain']:<4} framing_scan={stats['framing_scanned']:<4} "
              f"framing_deferred={stats['framing_deferred']:<4} "
              f"nudity={stats['nudity_images']:<4} nudity_scan={stats['nudity_scanned']:<4} "
              f"outfit_links={stats['outfit_links']:<4} links={stats['links']}", flush=True)
    return stats


def person_dirs(root: Path, only: str | None) -> list[Path]:
    if only:
        target = root / only
        if target.is_dir():
            return [target]
        matches = [p for p in root.iterdir() if p.is_dir() and p.name.lower() == only.lower()]
        return matches[:1]
    return sorted(
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")],
        key=lambda p: p.name.lower(),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("people_dir", nargs="?", default=str(DEFAULT_PEOPLE))
    parser.add_argument("--person", default=None,
                        help="Build albums for one person folder only, e.g. anushka.")
    parser.add_argument("--apply", action="store_true",
                        help="Create hardlink smart albums. Default is dry-run.")
    parser.add_argument("--visual-threshold", type=int, default=5,
                        help="pHash threshold for visual-similar groups. Default 5.")
    parser.add_argument("--scene-eps", type=float, default=0.075,
                        help="DBSCAN cosine eps for multi-feature same-scene groups. Default 0.075.")
    parser.add_argument("--min-group", type=int, default=3,
                        help="Minimum images per smart group. Default 3.")
    parser.add_argument("--max-scene-group", type=int, default=12,
                        help="Skip/split broad same-scene groups above this size. Default 12.")
    parser.add_argument("--no-detect-nudity", action="store_true",
                        help="Only use existing _possible_nudity/_uncertain_nudity folders; do not run NudeNet.")
    parser.add_argument("--nudity-batch-size", type=int, default=8,
                        help="Batch size for cached NudeNet smart-album classification. Default 8.")
    parser.add_argument("--nudity-cache", type=Path, default=DEFAULT_NUDITY_CACHE,
                        help=f"Nudity classification cache. Default: {DEFAULT_NUDITY_CACHE}")
    parser.add_argument("--nudity-overrides", type=Path, default=DEFAULT_NUDITY_OVERRIDES,
                        help=f"Manual nudity overrides JSON. Default: {DEFAULT_NUDITY_OVERRIDES}")
    parser.add_argument("--no-insightface-framing", action="store_true",
                        help="Use only the faster OpenCV Haar fallback for face-framing albums.")
    parser.add_argument("--framing-cache", type=Path, default=DEFAULT_FRAMING_CACHE,
                        help=f"InsightFace framing cache. Default: {DEFAULT_FRAMING_CACHE}")
    parser.add_argument("--framing-det-size", type=int, default=1024,
                        help="InsightFace detector size for smart-album framing. Default 1024 for higher quality.")
    parser.add_argument("--max-framing-checks-per-person", type=int, default=0,
                        help="Limit fresh InsightFace framing checks per person for this run. "
                             "Deferred checks keep the person marked changed for the next incremental run. "
                             "Default 0 means no limit.")
    parser.add_argument("--incremental", action="store_true",
                        help="Skip person folders whose source images and smart-album logic have not changed.")
    parser.add_argument("--max-people-per-run", type=int, default=0,
                        help="With --incremental, rebuild at most this many changed people in one run. "
                             "Default 0 means no limit.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild selected person folders even if incremental state says they are current.")
    parser.add_argument("--smart-state", type=Path, default=DEFAULT_SMART_STATE,
                        help=f"Incremental smart-album state file. Default: {DEFAULT_SMART_STATE}")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--simple", action="store_true",
                        help="Build the consolidated/simple smart album structure only; no V2 or AI-video folders.")
    parser.add_argument("--album-dir-name", default=SMART_DIR,
                        help="Output smart album folder name. Must start with _smart_albums. "
                             "Default: _smart_albums.")
    parser.add_argument("--max-images-per-person", type=int, default=0,
                        help="Use an evenly sampled subset per person. Default 0 scans all images.")
    parser.add_argument("--remove-smart-v2", action="store_true",
                        help="With --simple and --album-dir-name _smart_albums, remove generated _smart_albums_v2.")
    parser.add_argument("--simple-nest-nudity", action="store_true",
                        help="With --simple, keep _nudity_possible subfolders inside each smart album. "
                             "Default for simple albums.")
    parser.add_argument("--simple-flat-nudity", action="store_false", dest="simple_nest_nudity",
                        help="With --simple, place nudity-possible links flat in each matching smart album.")
    parser.set_defaults(simple_nest_nudity=True)
    parser.add_argument("--simple-refresh-nudity", action="store_true",
                        help="With --simple, ignore existing photos/nude labels and refresh nudity metadata "
                             "for filtering and candidate scoring.")
    args = parser.parse_args()

    root = Path(args.people_dir).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: people folder not found: {root}")
        return 1
    if args.simple:
        try:
            if args.album_dir_name != Path(args.album_dir_name).name:
                raise ValueError
            if not str(args.album_dir_name).startswith("_smart_albums"):
                raise ValueError
        except Exception:
            print(f"ERROR: unsafe smart album output folder: {args.album_dir_name!r}")
            return 1

    if args.apply:
        manifest_check = source_manifest.validate_current(
            label="build_smart_albums_before",
            people_dir=root,
        )
        source_manifest.print_validation(manifest_check)
        if not manifest_check.ok:
            print("ERROR: smart-album rebuild blocked because protected originals are missing or changed.")
            return source_manifest.SOURCE_GUARD_EXIT

    dirs = person_dirs(root, args.person)
    if not dirs:
        print(f"ERROR: no matching person folders under {root}")
        return 1

    total = {
        "images": 0,
        "links": 0,
        "visual_groups": 0,
        "scene_groups": 0,
        "best_links": 0,
        "review_links": 0,
        "framing_scanned": 0,
        "framing_deferred": 0,
        "face_uncertain": 0,
        "nudity_scanned": 0,
        "nudity_images": 0,
        "outfit_links": 0,
        "v2_links": 0,
        "v2_scene_groups": 0,
        "v2_packs": 0,
    }
    smart_state = load_smart_state(args.smart_state.expanduser()) if args.incremental else None
    signatures: dict[str, dict[str, int | str]] = {}
    build_dirs = dirs
    skipped_count = 0
    if smart_state is not None and not args.force:
        build_dirs = []
        people_state = smart_state.setdefault("people", {})
        current_keys = {str(person_dir.resolve()) for person_dir in dirs}
        if args.person is None:
            for stale_key in [key for key in people_state if key not in current_keys]:
                del people_state[stale_key]
        for person_dir in dirs:
            key = str(person_dir.resolve())
            sig = person_content_signature(person_dir)
            signatures[key] = sig
            existing = people_state.get(key, {})
            manifest = person_dir / str(args.album_dir_name) / "_smart_album_index.csv"
            v2_manifest = person_dir / SMART_V2_DIR / "_data" / "image_index.csv"
            source_count = int(sig.get("count", 0) or 0)
            output_exists = manifest.exists() if args.simple else (manifest.exists() and v2_manifest.exists())
            if existing.get("signature") == sig and (output_exists or source_count == 0):
                skipped_count += 1
            else:
                build_dirs.append(person_dir)
        if not args.quiet:
            print(f"Incremental smart albums: {len(build_dirs)} changed, {skipped_count} unchanged skipped.", flush=True)
    elif smart_state is not None:
        for person_dir in dirs:
            signatures[str(person_dir.resolve())] = person_content_signature(person_dir)

    deferred_people = 0
    if args.incremental and args.max_people_per_run > 0 and len(build_dirs) > args.max_people_per_run:
        deferred_people = len(build_dirs) - args.max_people_per_run
        build_dirs = build_dirs[:args.max_people_per_run]
        if not args.quiet:
            print(
                f"Incremental smart albums: limiting this run to {len(build_dirs)} changed people; "
                f"{deferred_people} deferred.",
                flush=True,
            )

    if not build_dirs:
        print()
        print(f"People folder:       {root}")
        print(f"Person folders:      {len(dirs)}")
        print(f"Skipped unchanged:   {skipped_count}")
        print("No smart albums needed rebuilding.")
        return 0

    detector = None
    nudity_cache = {"version": NUDITY_CACHE_VERSION, "items": {}}
    nudity_overrides = load_nudity_overrides(args.nudity_overrides.expanduser())
    framing_cache_path = args.framing_cache.expanduser()
    if not args.no_detect_nudity:
        detector = load_nudity_detector()
        if detector is None:
            print("WARNING: NudeNet is not installed; smart albums will only use existing nudity subfolders.")
        else:
            nudity_cache = load_nudity_cache(args.nudity_cache.expanduser())
    framing_app = None
    framing_cache = {"version": FRAMING_CACHE_VERSION, "items": {}}
    if not args.no_insightface_framing:
        framing_app = load_insightface_detector(max(320, int(args.framing_det_size)))
        if framing_app is None:
            print("WARNING: InsightFace framing detector unavailable; using OpenCV fallback only.")
        else:
            framing_cache = load_framing_cache(args.framing_cache.expanduser())

    for person_dir in build_dirs:
        build_kwargs = {
            "person_dir": person_dir,
            "apply": args.apply,
            "visual_threshold": max(0, int(args.visual_threshold)),
            "scene_eps": float(args.scene_eps),
            "min_group": max(2, int(args.min_group)),
            "max_scene_group": max(3, int(args.max_scene_group)),
            "detector": detector,
            "nudity_cache": nudity_cache,
            "nudity_overrides": nudity_overrides,
            "nudity_batch_size": max(1, int(args.nudity_batch_size)),
            "framing_app": framing_app,
            "framing_cache": framing_cache,
            "framing_cache_path": framing_cache_path if framing_app is not None else None,
            "quiet": args.quiet,
            "max_framing_checks": max(0, int(args.max_framing_checks_per_person)),
        }
        if args.simple:
            stats = build_for_person_simple(
                **build_kwargs,
                max_images=max(0, int(args.max_images_per_person)),
                album_dir_name=str(args.album_dir_name),
                remove_smart_v2=bool(args.remove_smart_v2),
                nest_nudity=bool(args.simple_nest_nudity),
                trust_source_nudity=not bool(args.simple_refresh_nudity),
            )
        else:
            stats = build_for_person(**build_kwargs)
        for key in total:
            total[key] += stats[key]
        if detector is not None:
            save_nudity_cache(args.nudity_cache.expanduser(), nudity_cache)
        if framing_app is not None:
            save_framing_cache(args.framing_cache.expanduser(), framing_cache)
        if smart_state is not None and args.apply:
            key = str(person_dir.resolve())
            if stats.get("framing_deferred", 0):
                smart_state.setdefault("people", {}).pop(key, None)
            else:
                sig = signatures.get(key) or person_content_signature(person_dir)
                smart_state.setdefault("people", {})[key] = {
                    "signature": sig,
                    "updated_at": int(time.time()),
                    "person": person_dir.name,
                }
                save_smart_state(args.smart_state.expanduser(), smart_state)

    if detector is not None:
        save_nudity_cache(args.nudity_cache.expanduser(), nudity_cache)
    if framing_app is not None:
        save_framing_cache(args.framing_cache.expanduser(), framing_cache)
    if smart_state is not None and args.apply:
        save_smart_state(args.smart_state.expanduser(), smart_state)

    print()
    print(f"People folder:       {root}")
    print(f"Person folders:      {len(dirs)}")
    if args.incremental:
        print(f"Rebuilt people:      {len(build_dirs)}")
        print(f"Skipped unchanged:   {skipped_count}")
        if deferred_people:
            print(f"Deferred people:     {deferred_people}")
    print(f"Images scanned:      {total['images']}")
    if args.simple:
        print(f"Output folder:       {args.album_dir_name}")
    print(f"Best-quality links:  {total['best_links']}")
    print(f"Review-needed links: {total['review_links']}")
    print(f"Face uncertain:      {total['face_uncertain']}")
    print(f"Framing newly scanned: {total['framing_scanned']}")
    print(f"Framing deferred:    {total['framing_deferred']}")
    print(f"Nudity images:       {total['nudity_images']}")
    print(f"Nudity newly scanned:{total['nudity_scanned']}")
    print(f"{'Saree clear links' if args.simple else 'Outfit album links'}:  {total['outfit_links']}")
    if not args.simple:
        print(f"Smart V2 links:      {total['v2_links']}")
        print(f"Smart V2 scene groups:{total['v2_scene_groups']}")
        print(f"Smart V2 packs:      {total['v2_packs']}")
    print(f"Visual groups:       {total['visual_groups']}")
    print(f"Same-scene groups:   {total['scene_groups']}")
    print(f"Smart album links:   {total['links']}")
    print()
    if not args.apply:
        print("DRY-RUN - no smart albums created. Re-run with --apply to commit.")
    else:
        if args.simple:
            print(f"Smart albums created under each person folder's {args.album_dir_name} directory.")
        else:
            print("Smart albums created under each person folder's _smart_albums and _smart_albums_v2 directories.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

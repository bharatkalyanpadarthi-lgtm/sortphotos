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
      06_nudity/possible/visual_similar/high_confidence/001_05_photos_portrait_from_Anushka_087/
      07_review_needed/small/

Hardlinks do not duplicate file contents on disk. If a hardlink cannot be
created, the script falls back to a symlink.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import re
import shutil
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface\..*")
warnings.filterwarnings("ignore", message=r".*`estimate` is deprecated.*", category=FutureWarning)

DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
DEFAULT_NUDITY_CACHE = Path.home() / ".face_sort_cache" / "smart_album_nudity_cache.json"
DEFAULT_NUDITY_OVERRIDES = Path.home() / ".face_sort_cache" / "smart_album_nudity_overrides.json"
DEFAULT_FRAMING_CACHE = Path.home() / ".face_sort_cache" / "smart_album_framing_cache.json"
DEFAULT_SMART_STATE = Path.home() / ".face_sort_cache" / "smart_album_person_state.json"
FRAMING_CACHE_VERSION = 2
SMART_ALBUM_LOGIC_VERSION = 8
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
SMART_DIR = "_smart_albums"
EXCLUDED_DIRS = {
    "all",
    SMART_DIR,
    "_duplicates",
    "_near_visual_review",
    "_blurred",
    "review",
}
NUDITY_DIRS = {
    "photos_nude": "nudity_possible",
    "_possible_nudity": "nudity_possible",
    "_uncertain_nudity": "nudity_uncertain",
}
NUDITY_NESTED_DIRS = {
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
EXPLICIT_NUDITY_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}
NUDITY_THRESHOLD = 0.35
NUDITY_UNCERTAIN_THRESHOLD = 0.20


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


def load_face_cascade() -> cv2.CascadeClassifier | None:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    return None if cascade.empty() else cascade


def load_nudity_cache(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") == 1 and isinstance(data.get("items"), dict):
            return data
    except Exception:
        pass
    return {"version": 1, "items": {}}


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
    if rel.parts[0] in {"photos_nude", "_possible_nudity"}:
        return "possible"
    if rel.parts[0] == "_uncertain_nudity":
        return "uncertain"
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
    if best_score >= NUDITY_THRESHOLD:
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
def quiet_model_startup():
    with open(os.devnull, "w", encoding="utf-8") as devnull:
        with contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
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
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


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
    max_dim = max(h, w)
    scale = min(1.0, 900.0 / max(1, max_dim))
    work = img
    if scale < 1.0:
        work = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
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
    for dirpath, dirnames, filenames in os.walk(person_dir):
        dirnames[:] = [
            d for d in dirnames
            if d not in EXCLUDED_DIRS and not d.startswith(".")
        ]
        base = Path(dirpath)
        for filename in filenames:
            p = base / filename
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
    return sorted(out, key=lambda p: str(p.relative_to(person_dir)).lower())


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


def load_infos(person_dir: Path) -> list[ImageInfo]:
    infos: list[ImageInfo] = []
    face_cascade = load_face_cascade()
    for path in iter_images(person_dir):
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
            nudity_status=path_nudity_status(path.relative_to(person_dir)),
        ))
    return infos


def annotate_framing(infos: list[ImageInfo],
                     app,
                     cache: dict,
                     cache_path: Path | None,
                     quiet: bool) -> int:
    if app is None:
        return 0
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

    if pending and not quiet:
        print(f"  framing: {cache_hits} cached, {len(pending)} new InsightFace checks...", flush=True)

    for idx, (info, key, sig) in enumerate(pending, start=1):
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
    return changed


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
            results = detector.detect_batch(paths, batch_size=len(paths))
        except Exception:
            results = []
            for info in batch:
                try:
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
    if info.nudity_status == "uncertain":
        return "_nudity_uncertain"
    if info.rel.parts and info.rel.parts[0] in NUDITY_NESTED_DIRS:
        return NUDITY_NESTED_DIRS[info.rel.parts[0]]
    return None


def visible_rel_name(rel: Path) -> Path:
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


def link_image(info: ImageInfo, album_dir: Path, apply: bool) -> Path:
    nested = nudity_nested_dir(info)
    if nested and "06_nudity" not in album_dir.parts:
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


def split_large_scene_group(group: list[ImageInfo], eps: float, min_group: int, max_group: int) -> list[list[ImageInfo]]:
    if len(group) <= max_group:
        return [group]
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
    out: list[list[ImageInfo]] = []
    for subgroup in subgroups.values():
        if len(subgroup) < min_group:
            continue
        if len(subgroup) <= max_group:
            out.append(subgroup)
        else:
            out.extend(g for g in split_large_scene_group(subgroup, eps * 0.62, min_group, max_group)
                       if len(g) <= max_group)
    return out


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
    """Broad saree/draped-ethnic hint using only local pixel/face features."""
    framing = shot_framing(info)
    if framing not in {"full_body", "waist_up", "chest_up"}:
        return False
    if info.face_count > 3:
        return False
    colorful_fabric = info.clothing_sat >= 58 and info.clothing_value >= 62
    warm_or_rich_color = (
        info.clothing_hue < 38
        or 70 <= info.clothing_hue < 112
        or info.clothing_hue >= 145
    )
    body_visible = info.largest_face_ratio <= 0.10 or info.face_height_ratio <= 0.36
    # Saree photos often show a large draped garment region plus some skin/arm
    # area. Keep this broad and put it under "likely" rather than exact.
    return colorful_fabric and warm_or_rich_color and body_visible and info.clothing_skin_ratio >= 0.035


def likely_western_or_modern(info: ImageInfo) -> bool:
    if outfit_visibility(info) == "outfit_unclear_closeup":
        return False
    if likely_saree_or_draped_ethnic(info):
        return False
    return info.clothing_sat < 58 or info.clothing_skin_ratio < 0.035


def outfit_album_names(info: ImageInfo) -> list[str]:
    names: list[str] = []
    visibility = outfit_visibility(info)
    color = color_bucket_from_hsv(info.clothing_hue, info.clothing_sat, info.clothing_value)

    names.append(f"08_outfit/00_visibility/{visibility}")
    names.append(f"08_outfit/10_by_outfit_color/{color}")

    if likely_saree_or_draped_ethnic(info):
        names.append("08_outfit/01_likely_saree_or_draped_ethnic")
    elif likely_western_or_modern(info):
        names.append("08_outfit/03_likely_western_or_modern")
    else:
        names.append("08_outfit/90_outfit_uncertain")

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


def subset_for_nudity(infos: list[ImageInfo], folder_name: str) -> list[ImageInfo]:
    return [i for i in infos if i.rel.parts and i.rel.parts[0] == folder_name]


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
                     quiet: bool) -> dict[str, int]:
    infos = load_infos(person_dir)
    if apply:
        clear_smart_albums(person_dir)
    framing_scanned = annotate_framing(infos, framing_app, framing_cache, framing_cache_path, quiet)
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
        "face_uncertain": 0,
        "nudity_scanned": nudity_scanned,
        "nudity_images": sum(1 for i in infos if i.nudity_status),
        "outfit_links": 0,
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

    for folder_name, album_name in NUDITY_DIRS.items():
        subset = subset_for_nudity(infos, folder_name)
        if not subset:
            continue
        category = "possible" if "possible" in album_name else "uncertain"
        stats["links"] += link_collection(subset, album_root, f"06_nudity/{category}/all", apply, rows)
        visual = group_visual_similar(subset, visual_threshold, max(2, min_group))
        for idx, group in confidence_numbered(visual):
            stats["links"] += write_smart_group(
                album_root, f"06_nudity/{category}/visual_similar", idx, group, "visual", apply, rows)
        scene = group_same_scene(subset, scene_eps, max(2, min_group), max_scene_group)
        for idx, group in confidence_numbered(scene):
            stats["links"] += write_smart_group(
                album_root, f"06_nudity/{category}/same_scene", idx, group, "scene", apply, rows)

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
        [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")],
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
    parser.add_argument("--incremental", action="store_true",
                        help="Skip person folders whose source images and smart-album logic have not changed.")
    parser.add_argument("--force", action="store_true",
                        help="Rebuild selected person folders even if incremental state says they are current.")
    parser.add_argument("--smart-state", type=Path, default=DEFAULT_SMART_STATE,
                        help=f"Incremental smart-album state file. Default: {DEFAULT_SMART_STATE}")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = Path(args.people_dir).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: people folder not found: {root}")
        return 1

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
        "face_uncertain": 0,
        "nudity_scanned": 0,
        "nudity_images": 0,
        "outfit_links": 0,
    }
    smart_state = load_smart_state(args.smart_state.expanduser()) if args.incremental and args.apply else None
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
            manifest = person_dir / SMART_DIR / "_smart_album_index.csv"
            source_count = int(sig.get("count", 0) or 0)
            if existing.get("signature") == sig and (manifest.exists() or source_count == 0):
                skipped_count += 1
            else:
                build_dirs.append(person_dir)
        if not args.quiet:
            print(f"Incremental smart albums: {len(build_dirs)} changed, {skipped_count} unchanged skipped.", flush=True)
    elif smart_state is not None:
        for person_dir in dirs:
            signatures[str(person_dir.resolve())] = person_content_signature(person_dir)

    if not build_dirs:
        print()
        print(f"People folder:       {root}")
        print(f"Person folders:      {len(dirs)}")
        print(f"Skipped unchanged:   {skipped_count}")
        print("No smart albums needed rebuilding.")
        return 0

    detector = None
    nudity_cache = {"version": 1, "items": {}}
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
        stats = build_for_person(
            person_dir,
            apply=args.apply,
            visual_threshold=max(0, int(args.visual_threshold)),
            scene_eps=float(args.scene_eps),
            min_group=max(2, int(args.min_group)),
            max_scene_group=max(3, int(args.max_scene_group)),
            detector=detector,
            nudity_cache=nudity_cache,
            nudity_overrides=nudity_overrides,
            nudity_batch_size=max(1, int(args.nudity_batch_size)),
            framing_app=framing_app,
            framing_cache=framing_cache,
            framing_cache_path=framing_cache_path if framing_app is not None else None,
            quiet=args.quiet,
        )
        for key in total:
            total[key] += stats[key]
        if detector is not None:
            save_nudity_cache(args.nudity_cache.expanduser(), nudity_cache)
        if framing_app is not None:
            save_framing_cache(args.framing_cache.expanduser(), framing_cache)
        if smart_state is not None and args.apply:
            key = str(person_dir.resolve())
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
    if args.incremental and args.apply:
        print(f"Rebuilt people:      {len(build_dirs)}")
        print(f"Skipped unchanged:   {skipped_count}")
    print(f"Images scanned:      {total['images']}")
    print(f"Best-quality links:  {total['best_links']}")
    print(f"Review-needed links: {total['review_links']}")
    print(f"Face uncertain:      {total['face_uncertain']}")
    print(f"Framing newly scanned: {total['framing_scanned']}")
    print(f"Nudity images:       {total['nudity_images']}")
    print(f"Nudity newly scanned:{total['nudity_scanned']}")
    print(f"Outfit album links:  {total['outfit_links']}")
    print(f"Visual groups:       {total['visual_groups']}")
    print(f"Same-scene groups:   {total['scene_groups']}")
    print(f"Smart album links:   {total['links']}")
    print()
    if not args.apply:
        print("DRY-RUN - no smart albums created. Re-run with --apply to commit.")
    else:
        print("Smart albums created under each person folder's _smart_albums directory.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

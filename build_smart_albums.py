#!/usr/bin/env python3
"""
Build non-destructive smart albums inside photos_by_person folders.

The script keeps original images where they are and creates hardlinked views:

  photos_by_person/Anushka/_smart_albums/
      00_best/top_020_technical_quality/
      01_quality/large_high_score/
      02_format/portrait/
      03_face_framing/closeup_face/
      04_visual_similar/001_12_photos_portrait_from_Anushka_023/
          _nudity_possible/
      05_same_scene/001_18_photos_scene_from_Anushka_041/
      06_nudity/possible/visual_similar/001_05_photos_portrait_from_Anushka_087/
      07_review_needed/small/

Hardlinks do not duplicate file contents on disk. If a hardlink cannot be
created, the script falls back to a symlink.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import DBSCAN

DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
DEFAULT_NUDITY_CACHE = Path.home() / ".face_sort_cache" / "smart_album_nudity_cache.json"
DEFAULT_NUDITY_OVERRIDES = Path.home() / ".face_sort_cache" / "smart_album_nudity_overrides.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}
SMART_DIR = "_smart_albums"
EXCLUDED_DIRS = {
    SMART_DIR,
    "_duplicates",
    "_blurred",
}
NUDITY_DIRS = {
    "_possible_nudity": "nudity_possible",
    "_uncertain_nudity": "nudity_uncertain",
}
NUDITY_NESTED_DIRS = {
    "_possible_nudity": "_nudity_possible",
    "_uncertain_nudity": "_nudity_uncertain",
}
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
    sharpness: float
    brightness: float
    contrast: float
    face_count: int
    largest_face_ratio: float
    quality: float
    nudity_status: str = ""
    nudity_class: str = ""
    nudity_score: float = 0.0


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


def file_signature(path: Path) -> dict[str, int]:
    st = path.stat()
    return {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}


def path_nudity_status(rel: Path) -> str:
    if not rel.parts:
        return ""
    if rel.parts[0] == "_possible_nudity":
        return "possible"
    if rel.parts[0] == "_uncertain_nudity":
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


def color_context(img: np.ndarray) -> np.ndarray:
    h, w = img.shape[:2]
    # Border-weighted color descriptor leans toward background/context.
    border = max(8, min(h, w) // 8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:border, :] = 255
    mask[-border:, :] = 255
    mask[:, :border] = 255
    mask[:, -border:] = 255
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], mask, [8, 4, 4], [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten().astype(np.float32)
    return hist


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
        infos.append(ImageInfo(
            path=path,
            rel=path.relative_to(person_dir),
            width=w,
            height=h,
            phash=phash64(img),
            color=color_context(img),
            sharpness=sharp,
            brightness=brightness,
            contrast=contrast,
            face_count=face_count,
            largest_face_ratio=largest_face_ratio,
            quality=image_quality(img, sharp, brightness, contrast),
            nudity_status=path_nudity_status(path.relative_to(person_dir)),
        ))
    return infos


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
    label = "scene" if kind == "scene" else orient
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


def group_visual_similar(infos: list[ImageInfo], threshold: int, min_group: int) -> list[list[ImageInfo]]:
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
        ratio_a = a.width / max(1, a.height)
        for j in range(i + 1, len(infos)):
            b = infos[j]
            ratio_b = b.width / max(1, b.height)
            if abs(ratio_a - ratio_b) > 0.08:
                continue
            if hamming(a.phash, b.phash) <= threshold:
                union(i, j)

    clusters: dict[int, list[ImageInfo]] = defaultdict(list)
    for i, info in enumerate(infos):
        clusters[find(i)].append(info)
    groups = [g for g in clusters.values() if len(g) >= min_group]
    return sorted(groups, key=lambda g: (-len(g), str(g[0].rel).lower()))


def group_same_scene(infos: list[ImageInfo],
                     eps: float,
                     min_group: int,
                     max_group: int) -> list[list[ImageInfo]]:
    if len(infos) < min_group:
        return []
    colors = np.stack([i.color for i in infos])
    labels = DBSCAN(eps=eps, min_samples=min_group, metric="cosine").fit_predict(colors)
    groups: dict[int, list[ImageInfo]] = defaultdict(list)
    for info, label in zip(infos, labels):
        if label >= 0:
            groups[int(label)].append(info)
    out: list[list[ImageInfo]] = []
    for group in groups.values():
        if len(group) < min_group:
            continue
        if len(group) <= max_group:
            out.append(group)
            continue
        # Large color clusters are often broad palette matches, not true scenes.
        # Try one stricter split; if still too broad, omit for accuracy.
        sub_colors = np.stack([i.color for i in group])
        sub_labels = DBSCAN(
            eps=max(eps * 0.60, 0.006),
            min_samples=min_group,
            metric="cosine",
        ).fit_predict(sub_colors)
        subgroups: dict[int, list[ImageInfo]] = defaultdict(list)
        for info, label in zip(group, sub_labels):
            if label >= 0:
                subgroups[int(label)].append(info)
        for subgroup in subgroups.values():
            if min_group <= len(subgroup) <= max_group:
                out.append(subgroup)
    return sorted(out, key=lambda g: (-len(g), str(g[0].rel).lower()))


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


def face_framing_album_names(info: ImageInfo) -> list[str]:
    if info.face_count == 0:
        return ["03_face_framing/framing_unknown"]
    names: list[str] = []
    ratio = info.largest_face_ratio
    if ratio >= 0.16:
        names.append("03_face_framing/closeup_face")
    elif ratio >= 0.045:
        names.append("03_face_framing/portrait_or_upper_body")
    else:
        names.append("03_face_framing/wide_or_full_body")
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
        "nudity_status": info.nudity_status,
        "nudity_class": info.nudity_class,
        "nudity_score": f"{info.nudity_score:.3f}",
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
                     quiet: bool) -> dict[str, int]:
    infos = load_infos(person_dir)
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
        "nudity_scanned": nudity_scanned,
        "nudity_images": sum(1 for i in infos if i.nudity_status),
    }
    if not infos:
        return stats
    if apply:
        clear_smart_albums(person_dir)
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
            if album_name.startswith("07_review_needed/"):
                stats["review_links"] += 1

    visual_groups = group_visual_similar(infos, visual_threshold, min_group)
    for idx, group in enumerate(visual_groups, start=1):
        stats["links"] += write_group(album_root, "04_visual_similar", idx, group, "visual", apply, rows)
    stats["visual_groups"] = len(visual_groups)

    scene_groups = group_same_scene(infos, scene_eps, min_group, max_scene_group)
    for idx, group in enumerate(scene_groups, start=1):
        stats["links"] += write_group(album_root, "05_same_scene", idx, group, "scene", apply, rows)
    stats["scene_groups"] = len(scene_groups)

    for folder_name, album_name in NUDITY_DIRS.items():
        subset = subset_for_nudity(infos, folder_name)
        if not subset:
            continue
        category = "possible" if "possible" in album_name else "uncertain"
        stats["links"] += link_collection(subset, album_root, f"06_nudity/{category}/all", apply, rows)
        visual = group_visual_similar(subset, visual_threshold, max(2, min_group))
        for idx, group in enumerate(visual, start=1):
            stats["links"] += write_group(
                album_root, f"06_nudity/{category}/visual_similar", idx, group, "visual", apply, rows)
        scene = group_same_scene(subset, scene_eps, max(2, min_group), max_scene_group)
        for idx, group in enumerate(scene, start=1):
            stats["links"] += write_group(
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
                "nudity_status",
                "nudity_class",
                "nudity_score",
            ])
            writer.writeheader()
            writer.writerows(rows)

    if not quiet:
        print(f"{person_dir.name:<32} images={stats['images']:<5} "
              f"visual_groups={stats['visual_groups']:<4} scene_groups={stats['scene_groups']:<4} "
              f"best={stats['best_links']:<3} review={stats['review_links']:<4} "
              f"nudity={stats['nudity_images']:<4} nudity_scan={stats['nudity_scanned']:<4} "
              f"links={stats['links']}")
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
    parser.add_argument("--scene-eps", type=float, default=0.02,
                        help="Strict DBSCAN cosine eps for same-scene context groups. Default 0.02.")
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
        "nudity_scanned": 0,
        "nudity_images": 0,
    }
    detector = None
    nudity_cache = {"version": 1, "items": {}}
    nudity_overrides = load_nudity_overrides(args.nudity_overrides.expanduser())
    if not args.no_detect_nudity:
        detector = load_nudity_detector()
        if detector is None:
            print("WARNING: NudeNet is not installed; smart albums will only use existing nudity subfolders.")
        else:
            nudity_cache = load_nudity_cache(args.nudity_cache.expanduser())

    for person_dir in dirs:
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
            quiet=args.quiet,
        )
        for key in total:
            total[key] += stats[key]

    if detector is not None:
        save_nudity_cache(args.nudity_cache.expanduser(), nudity_cache)

    print()
    print(f"People folder:       {root}")
    print(f"Person folders:      {len(dirs)}")
    print(f"Images scanned:      {total['images']}")
    print(f"Best-quality links:  {total['best_links']}")
    print(f"Review-needed links: {total['review_links']}")
    print(f"Nudity images:       {total['nudity_images']}")
    print(f"Nudity newly scanned:{total['nudity_scanned']}")
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

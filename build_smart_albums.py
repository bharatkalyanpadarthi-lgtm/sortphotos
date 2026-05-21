#!/usr/bin/env python3
"""
Build non-destructive smart albums inside photos_by_person folders.

The script keeps original images where they are and creates hardlinked views:

  photos_by_person/Anushka/_smart_albums/
      00_best/top_020_quality/
      01_quality/sharp_large/
      02_format/portrait/
      03_face_framing/closeup_face/
      04_visual_similar/001_12_photos_portrait_from_Anushka_023/
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


def load_face_cascade() -> cv2.CascadeClassifier | None:
    cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(str(cascade_path))
    return None if cascade.empty() else cascade


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
        ))
    return infos


def safe_name(path: Path) -> str:
    parts = list(path.parts)
    stem = "__".join(parts)
    for ch in '/\\:*?"<>|':
        stem = stem.replace(ch, "_")
    while "__." in stem:
        stem = stem.replace("__.", ".")
    return stem


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


def link_image(src: Path, album_dir: Path, rel: Path, apply: bool) -> Path:
    dest = album_dir / safe_name(rel)
    if not apply:
        return dest
    album_dir.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    try:
        os.link(str(src), str(dest))
    except OSError:
        os.symlink(str(src), str(dest))
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


def group_same_scene(infos: list[ImageInfo], eps: float, min_group: int) -> list[list[ImageInfo]]:
    if len(infos) < min_group:
        return []
    colors = np.stack([i.color for i in infos])
    labels = DBSCAN(eps=eps, min_samples=min_group, metric="cosine").fit_predict(colors)
    groups: dict[int, list[ImageInfo]] = defaultdict(list)
    for info, label in zip(infos, labels):
        if label >= 0:
            groups[int(label)].append(info)
    out = [g for g in groups.values() if len(g) >= min_group]
    return sorted(out, key=lambda g: (-len(g), str(g[0].rel).lower()))


def context_name(info: ImageInfo) -> str:
    return f"02_format/{orientation_name(info)}"


def quality_album_names(info: ImageInfo) -> list[str]:
    names: list[str] = []
    pixels = info.width * info.height
    min_dim = min(info.width, info.height)
    if info.quality >= 0.68 and pixels >= 800_000:
        names.append("01_quality/sharp_large")
    if info.sharpness < 25.0:
        names.append("01_quality/low_sharpness_score")
    if min_dim < 450 or pixels < 350_000:
        names.append("01_quality/small")
        names.append("07_review_needed/small")
    if info.brightness < 45.0:
        names.append("01_quality/dark")
        names.append("07_review_needed/dark")
    elif info.brightness > 220.0:
        names.append("01_quality/overexposed")
        names.append("07_review_needed/overexposed")
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
        dest = link_image(info.path, album_dir, info.rel, apply)
        rows.append(manifest_row(info, str(album_dir.relative_to(album_root.parent)), dest))
    return len(group)


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
    }


def link_single(info: ImageInfo,
                album_root: Path,
                album_path: str,
                apply: bool,
                rows: list[dict[str, str]]) -> int:
    dest = link_image(info.path, album_root / album_path, info.rel, apply)
    rows.append(manifest_row(info, album_path, dest))
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
                     quiet: bool) -> dict[str, int]:
    infos = load_infos(person_dir)
    stats = {
        "images": len(infos),
        "links": 0,
        "visual_groups": 0,
        "scene_groups": 0,
        "best_links": 0,
        "review_links": 0,
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
        linked = link_collection(best[:top_n], album_root, f"00_best/top_{top_n:03d}_quality", apply, rows)
        stats["links"] += linked
        stats["best_links"] += linked

    for info in infos:
        stats["links"] += 1
        dest = link_image(info.path, album_root / context_name(info), info.rel, apply)
        rows.append(manifest_row(info, context_name(info), dest))
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

    scene_groups = group_same_scene(infos, scene_eps, min_group)
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
        scene = group_same_scene(subset, scene_eps, max(2, min_group))
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
            ])
            writer.writeheader()
            writer.writerows(rows)

    if not quiet:
        print(f"{person_dir.name:<32} images={stats['images']:<5} "
              f"visual_groups={stats['visual_groups']:<4} scene_groups={stats['scene_groups']:<4} "
              f"best={stats['best_links']:<3} review={stats['review_links']:<4} "
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
    parser.add_argument("--scene-eps", type=float, default=0.10,
                        help="DBSCAN cosine eps for focused same-scene color/context groups. Default 0.10.")
    parser.add_argument("--min-group", type=int, default=3,
                        help="Minimum images per smart group. Default 3.")
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
    }
    for person_dir in dirs:
        stats = build_for_person(
            person_dir,
            apply=args.apply,
            visual_threshold=max(0, int(args.visual_threshold)),
            scene_eps=float(args.scene_eps),
            min_group=max(2, int(args.min_group)),
            quiet=args.quiet,
        )
        for key in total:
            total[key] += stats[key]

    print()
    print(f"People folder:       {root}")
    print(f"Person folders:      {len(dirs)}")
    print(f"Images scanned:      {total['images']}")
    print(f"Best-quality links:  {total['best_links']}")
    print(f"Review-needed links: {total['review_links']}")
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

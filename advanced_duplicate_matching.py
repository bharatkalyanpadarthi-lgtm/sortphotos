#!/usr/bin/env python3
"""
advanced_duplicate_matching.py — layered duplicate detection for sorted photos.

It uses three levels:
  1. SHA-256 of file bytes: exact same file.
  2. SHA-256 of decoded pixels: same visible image, different metadata/container.
  3. OpenCV pHash distance: visually similar images, e.g. resized/recompressed.

Default is a dry-run report. With --apply, only exact file and same-pixel
duplicates are moved to ready_to_delete. Near-duplicates are reported in the
CSV but not moved unless --move-near is explicitly set.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import shutil
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp",
              ".tif", ".tiff", ".heic", ".heif"}
DEFAULT_SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
DEFAULT_PHOTOS = DEFAULT_SORTED / "photos_by_person"
DEFAULT_REVIEW = DEFAULT_SORTED / "_source_review" / "ready_to_delete" / "advanced_duplicates"
DEFAULT_REPORT = DEFAULT_SORTED / "_source_review" / "duplicate_reports" / "advanced_duplicates.csv"
DEFAULT_CACHE = Path.home() / ".face_sort_cache" / "advanced_duplicate_fingerprints.json"
CACHE_VERSION = 1
ALWAYS_EXCLUDED_DIRS = {"all", "videos", "_duplicates", "_near_visual_review", "_smart_albums", "review"}


@contextlib.contextmanager
def suppress_native_stderr(enabled: bool = True):
    """Hide noisy native decoder warnings while still counting failed reads."""
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


@dataclass(frozen=True)
class ImageInfo:
    path: Path
    scope: str
    size_bytes: int
    width: int
    height: int
    sha256: str
    pixel_sha256: str | None
    phash: int | None


@dataclass(frozen=True)
class DuplicateMember:
    group_id: int
    kind: str
    confidence: int
    action: str
    keeper: Path
    path: Path
    distance: int | None = None


def iter_images(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d.casefold() not in ALWAYS_EXCLUDED_DIRS
        ]
        base = Path(dirpath)
        for filename in filenames:
            p = base / filename
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
    return out


def file_signature(path: Path) -> dict[str, int | float | str]:
    st = path.stat()
    return {
        "mtime": float(st.st_mtime),
        "size": int(st.st_size),
        "path": str(path),
    }


def load_fingerprint_cache(path: Path) -> dict:
    if not path.exists():
        return {"version": CACHE_VERSION, "entries": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") != CACHE_VERSION:
            return {"version": CACHE_VERSION, "entries": {}}
        data.setdefault("entries", {})
        return data
    except Exception:
        return {"version": CACHE_VERSION, "entries": {}}


def save_fingerprint_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f)
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def imread(path: Path) -> np.ndarray | None:
    with suppress_native_stderr():
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None and getattr(img, "size", 0) > 0:
                return img
        except Exception:
            pass

        try:
            from PIL import Image, ImageFile
            import pillow_heif  # noqa: F401

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(path) as im:
                return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            return None


def pixel_sha256(img: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(img.shape).encode("ascii"))
    h.update(np.ascontiguousarray(img).tobytes())
    return h.hexdigest()


def phash64(img: np.ndarray) -> int:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA)
    dct = cv2.dct(np.float32(small))
    block = dct[:8, :8].copy()
    vals = block.flatten()
    median = float(np.median(vals[1:]))
    bits = vals > median
    value = 0
    for bit in bits:
        value = (value << 1) | int(bool(bit))
    return value


def hamming(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def scope_for(path: Path, root: Path, scope: str) -> str:
    if scope == "global":
        return "__global__"
    try:
        rel = path.relative_to(root)
    except ValueError:
        return path.parent.name
    return rel.parts[0] if rel.parts else path.parent.name


def info_from_cache(path: Path, scope_name: str, entry: dict) -> ImageInfo | None:
    try:
        return ImageInfo(
            path=path,
            scope=scope_name,
            size_bytes=int(entry["size_bytes"]),
            width=int(entry["width"]),
            height=int(entry["height"]),
            sha256=str(entry["sha256"]),
            pixel_sha256=entry.get("pixel_sha256"),
            phash=int(str(entry["phash"]), 16) if entry.get("phash") else None,
        )
    except Exception:
        return None


def collect(root: Path, scope: str, cache_path: Path | None = DEFAULT_CACHE,
            quarantine_errors: bool = False,
            bad_dir: Path | None = None) -> tuple[list[ImageInfo], list[tuple[Path, str]], dict[str, int]]:
    infos: list[ImageInfo] = []
    errors: list[tuple[Path, str]] = []
    stats = {"cache_hits": 0, "cache_misses": 0, "cache_pruned": 0, "bad_moved": 0}
    cache = load_fingerprint_cache(cache_path) if cache_path else {"entries": {}}
    entries = cache.setdefault("entries", {})
    live_paths: set[str] = set()

    for path in iter_images(root):
        path_key = str(path)
        live_paths.add(path_key)
        try:
            sig = file_signature(path)
            scope_name = scope_for(path, root, scope)
            cached = entries.get(path_key)
            if cached and cached.get("signature") == sig:
                info = info_from_cache(path, scope_name, cached)
                if info is not None:
                    infos.append(info)
                    stats["cache_hits"] += 1
                    continue
            stats["cache_misses"] += 1
            size = int(sig["size"])
            file_hash = sha256_file(path)
            img = imread(path)
            if img is None:
                errors.append((path, "decode_failed"))
                if quarantine_errors and bad_dir is not None:
                    try:
                        dest = unique_dest(bad_dir / path.relative_to(root))
                    except ValueError:
                        dest = unique_dest(bad_dir / path.name)
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        shutil.move(str(path), str(dest))
                        stats["bad_moved"] += 1
                    except OSError as exc:
                        errors.append((path, f"quarantine_failed: {exc}"))
                continue
            height, width = img.shape[:2]
            pix_hash = pixel_sha256(img)
            p_hash = phash64(img)
            info = ImageInfo(
                path=path,
                scope=scope_name,
                size_bytes=size,
                width=width,
                height=height,
                sha256=file_hash,
                pixel_sha256=pix_hash,
                phash=p_hash,
            )
            infos.append(info)
            entries[path_key] = {
                "signature": sig,
                "size_bytes": size,
                "width": width,
                "height": height,
                "sha256": file_hash,
                "pixel_sha256": pix_hash,
                "phash": f"{p_hash:016x}",
            }
        except Exception as exc:  # noqa: BLE001
            errors.append((path, str(exc)[:160]))
            if quarantine_errors and bad_dir is not None:
                try:
                    dest = unique_dest(bad_dir / path.relative_to(root))
                except ValueError:
                    dest = unique_dest(bad_dir / path.name)
                try:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(path), str(dest))
                    stats["bad_moved"] += 1
                except OSError:
                    pass

    if cache_path:
        stale = [k for k in entries if k not in live_paths]
        for k in stale:
            entries.pop(k, None)
        stats["cache_pruned"] = len(stale)
        save_fingerprint_cache(cache_path, cache)
    return infos, errors, stats


def same_inode(a: Path, b: Path) -> bool:
    try:
        sa = a.stat()
        sb = b.stat()
    except OSError:
        return False
    return sa.st_dev == sb.st_dev and sa.st_ino == sb.st_ino


def choose_keeper(items: list[ImageInfo]) -> ImageInfo:
    def score(info: ImageInfo) -> tuple[int, int, int, int, str]:
        parts = set(info.path.parts)
        duplicate_penalty = 1 if "_duplicates" in parts else 0
        blurred_penalty = 1 if "_blurred" in parts else 0
        review_penalty = 1 if "_source_review" in parts else 0
        pixels = -(info.width * info.height)
        return (review_penalty, duplicate_penalty, blurred_penalty, pixels,
                str(info.path).lower())

    return sorted(items, key=score)[0]


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = "".join(dest.suffixes)
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__dup{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def add_hash_groups(groups: list[list[ImageInfo]], items: list[ImageInfo],
                    key_name: str) -> set[Path]:
    seen: set[Path] = set()
    by_key: dict[tuple[str, str | int], list[ImageInfo]] = defaultdict(list)
    for info in items:
        value = getattr(info, key_name)
        if value is not None:
            by_key[(info.scope, value)].append(info)
    for members in by_key.values():
        unique = [m for m in members if m.path not in seen]
        if len(unique) > 1:
            groups.append(unique)
            seen.update(m.path for m in unique)
    return seen


def find_near_groups(items: list[ImageInfo], threshold: int,
                     already_grouped: set[Path]) -> list[tuple[list[ImageInfo], int]]:
    out: list[tuple[list[ImageInfo], int]] = []
    by_scope: dict[str, list[ImageInfo]] = defaultdict(list)
    for info in items:
        if info.phash is not None and info.path not in already_grouped:
            by_scope[info.scope].append(info)

    for scoped in by_scope.values():
        parent = {i: i for i in range(len(scoped))}
        best_dist: dict[tuple[int, int], int] = {}

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int, dist: int) -> None:
            ra, rb = find(a), find(b)
            if ra == rb:
                return
            parent[rb] = ra
            best_dist[(min(ra, rb), max(ra, rb))] = dist

        for i in range(len(scoped)):
            a = scoped[i]
            for j in range(i + 1, len(scoped)):
                b = scoped[j]
                if a.phash is None or b.phash is None:
                    continue
                # Avoid comparing wildly different images; pHash can collide.
                ratio_a = a.width / max(1, a.height)
                ratio_b = b.width / max(1, b.height)
                if abs(ratio_a - ratio_b) > 0.08:
                    continue
                dist = hamming(a.phash, b.phash)
                if dist <= threshold:
                    union(i, j, dist)

        clusters: dict[int, list[ImageInfo]] = defaultdict(list)
        for i, info in enumerate(scoped):
            clusters[find(i)].append(info)
        for members in clusters.values():
            if len(members) > 1:
                dists = [
                    hamming(a.phash, b.phash)
                    for idx, a in enumerate(members)
                    for b in members[idx + 1:]
                    if a.phash is not None and b.phash is not None
                ]
                out.append((members, min(dists) if dists else threshold))
    return out


def build_duplicates(items: list[ImageInfo], near_threshold: int,
                     move_near: bool) -> list[DuplicateMember]:
    groups: list[DuplicateMember] = []
    exact_groups: list[list[ImageInfo]] = []
    grouped = add_hash_groups(exact_groups, items, "sha256")

    pixel_candidates = [i for i in items if i.path not in grouped]
    pixel_groups: list[list[ImageInfo]] = []
    grouped |= add_hash_groups(pixel_groups, pixel_candidates, "pixel_sha256")

    group_id = 1
    for kind, confidence, hash_groups in (
        ("exact_file", 100, exact_groups),
        ("same_pixels", 100, pixel_groups),
    ):
        for members in hash_groups:
            keeper = choose_keeper(members).path
            for info in sorted(members, key=lambda x: str(x.path).lower()):
                action = "keep" if info.path == keeper else "move"
                if action == "move" and same_inode(info.path, keeper):
                    action = "already_hardlinked"
                groups.append(DuplicateMember(group_id, kind, confidence, action,
                                              keeper, info.path))
            group_id += 1

    for members, dist in find_near_groups(items, near_threshold, grouped):
        keeper = choose_keeper(members).path
        confidence = max(70, int(98 - (dist * 4)))
        for info in sorted(members, key=lambda x: str(x.path).lower()):
            action = "keep" if info.path == keeper else ("move" if move_near else "review")
            groups.append(DuplicateMember(group_id, "visually_similar", confidence,
                                          action, keeper, info.path, dist))
        group_id += 1

    return groups


def write_report(report: Path, infos: dict[Path, ImageInfo],
                 duplicates: list[DuplicateMember]) -> None:
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "group_id", "type", "confidence", "action", "distance",
            "file_path", "keeper_path", "scope", "width", "height",
            "size_bytes", "sha256", "pixel_sha256", "phash",
        ])
        for dup in duplicates:
            info = infos[dup.path]
            w.writerow([
                dup.group_id, dup.kind, dup.confidence, dup.action,
                "" if dup.distance is None else dup.distance,
                str(dup.path), str(dup.keeper), info.scope,
                info.width, info.height, info.size_bytes, info.sha256,
                info.pixel_sha256 or "",
                "" if info.phash is None else f"{info.phash:016x}",
            ])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("root", nargs="?", default=str(DEFAULT_PHOTOS),
                        help="Folder to scan. Default: sorted_all_pictures/photos_by_person.")
    parser.add_argument("--scope", choices=["per-folder", "global"], default="per-folder",
                        help="per-folder compares only within each person folder. global compares all images together.")
    parser.add_argument("--near-threshold", type=int, default=5,
                        help="pHash Hamming distance for visually similar candidates. Default 5.")
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--fingerprint-cache", type=Path, default=DEFAULT_CACHE,
                        help="JSON cache for SHA/pixel/pHash fingerprints. Use --no-cache to disable.")
    parser.add_argument("--no-cache", action="store_true",
                        help="Recompute all duplicate fingerprints and do not update the cache.")
    parser.add_argument("--quarantine-bad", action="store_true",
                        help="Move unreadable/corrupt images to ready_to_delete/bad_images.")
    parser.add_argument("--bad-dir", type=Path,
                        default=DEFAULT_SORTED / "_source_review" / "ready_to_delete" / "bad_images")
    parser.add_argument("--apply", action="store_true",
                        help="Move exact_file and same_pixels duplicates to review-dir.")
    parser.add_argument("--move-near", action="store_true",
                        help="Also move visually_similar candidates. Riskier; default only reports them.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    if not root.exists():
        print(f"ERROR: root not found: {root}")
        return 1

    print(f"Scanning: {root}")
    print(f"Scope:    {args.scope}")
    cache_path = None if args.no_cache else args.fingerprint_cache.expanduser()
    infos, errors, cache_stats = collect(
        root, args.scope, cache_path=cache_path,
        quarantine_errors=args.quarantine_bad,
        bad_dir=args.bad_dir.expanduser())
    info_by_path = {i.path: i for i in infos}
    duplicates = build_duplicates(infos, args.near_threshold, args.move_near)
    write_report(args.report.expanduser(), info_by_path, duplicates)

    groups = {d.group_id for d in duplicates}
    to_move = [
        d for d in duplicates
        if d.action == "move"
        and (d.kind in {"exact_file", "same_pixels"} or args.move_near)
    ]
    by_kind = defaultdict(int)
    for d in duplicates:
        if d.action != "keep":
            by_kind[d.kind] += 1

    bytes_to_move = 0
    for d in to_move:
        try:
            bytes_to_move += d.path.stat().st_size
        except OSError:
            pass

    print(f"Images scanned:          {len(infos)}")
    print(f"Decode/read errors:      {len(errors)}")
    print(f"Fingerprint cache hits:  {cache_stats['cache_hits']}")
    print(f"Fingerprint cache misses:{cache_stats['cache_misses']:>7}")
    print(f"Fingerprint cache pruned:{cache_stats['cache_pruned']:>7}")
    if args.quarantine_bad:
        print(f"Bad images quarantined:  {cache_stats['bad_moved']}")
    print(f"Duplicate groups:        {len(groups)}")
    print(f"Exact-file candidates:   {by_kind['exact_file']}")
    print(f"Same-pixel candidates:   {by_kind['same_pixels']}")
    print(f"Near-visual candidates:  {by_kind['visually_similar']}")
    print(f"Selected to move:        {len(to_move)}")
    print(f"Bytes selected to move:  {bytes_to_move / (1024**3):.2f} GB")
    print(f"CSV report:              {args.report.expanduser()}")
    print(f"Review destination:      {args.review_dir.expanduser()}")
    print()

    if not args.quiet:
        for d in to_move[:60]:
            print(f"move [{d.kind}]: {d.path}")
            print(f"  keep: {d.keeper}")
        review_only = [d for d in duplicates if d.action == "review"]
        for d in review_only[:20]:
            print(f"review [{d.kind}, dist={d.distance}]: {d.path}")
            print(f"  possible keeper: {d.keeper}")
        if len(to_move) > 60 or len(review_only) > 20:
            print("... more in CSV report")
        print()

    if not args.apply:
        print("DRY-RUN — no files moved.")
        print("Use --apply to move exact-file/same-pixel duplicates to review.")
        return 0

    moved = 0
    review_dir = args.review_dir.expanduser().resolve()
    for d in to_move:
        if not d.path.exists():
            continue
        try:
            rel = d.path.relative_to(root)
        except ValueError:
            rel = Path(d.path.name)
        dest = unique_dest(review_dir / d.kind / rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(d.path), str(dest))
        moved += 1

    print(f"Moved {moved} duplicate file(s) to: {review_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

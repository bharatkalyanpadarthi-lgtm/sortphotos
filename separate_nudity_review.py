#!/usr/bin/env python3
"""Classify normal person images with the shared high-precision policy.

The normal full-library workflow writes a report without duplicating images.
The follow-up placer moves confirmed images into photos/nude and ambiguous
explicit detections into each person's review/uncertain_nudity folder.
Default is dry-run; use --apply to write the report.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import os
import shutil
import sys
import time
from pathlib import Path

import pipeline_paths
from typing import Any

import analysis_index
import sort_photos

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
              ".tif", ".tiff", ".heic", ".heif"}
DEFAULT_INPUT = pipeline_paths.PEOPLE_ROOT
DEFAULT_OUTPUT = pipeline_paths.SORTED_ROOT / "_nudity_review"
POLICY_VERSION = "7-spatial-conflicts"
EXCLUDED_DIRS = {
    "all",
    "photos/nude",
    "photos_nude",
    "_possible_nudity",
    "_smart_albums",
    "_smart_albums_v2",
    "_smart_albums_simple_preview",
    "_duplicates",
    "_near_visual_review",
    "_uncertain_nudity",
    "review",
}
NUDITY_THRESHOLD = sort_photos.NUDITY_THRESHOLD
NUDITY_UNCERTAIN_THRESHOLD = sort_photos.NUDITY_UNCERTAIN_THRESHOLD


def iter_images(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        try:
            rel = base.relative_to(root)
        except ValueError:
            rel = Path()
        if (
            len(rel.parts) >= 2 and rel.parts[:2] == ("photos", "nude")
        ) or (
            len(rel.parts) >= 3 and rel.parts[1:3] == ("photos", "nude")
        ):
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d.casefold() not in EXCLUDED_DIRS
        ]
        for filename in filenames:
            p = base / filename
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
    return sorted(out, key=lambda p: str(p).lower())


def chunks(items: list[Path], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


@contextlib.contextmanager
def suppress_native_stderr(enabled: bool = True):
    """Hide noisy libjpeg/libpng warnings emitted below Python's warnings layer."""
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


def _cv2_to_rgba(mat: Any) -> Any | None:
    import cv2

    if mat is None or getattr(mat, "size", 0) == 0:
        return None
    if mat.ndim == 2:
        return cv2.cvtColor(mat, cv2.COLOR_GRAY2RGBA)
    if mat.ndim != 3:
        return None
    channels = mat.shape[2]
    if channels == 4:
        return cv2.cvtColor(mat, cv2.COLOR_BGRA2RGBA)
    if channels == 3:
        return cv2.cvtColor(mat, cv2.COLOR_BGR2RGBA)
    if channels == 1:
        return cv2.cvtColor(mat, cv2.COLOR_GRAY2RGBA)
    return None


def decode_for_detector(path: Path, suppress_warnings: bool = True) -> tuple[Any | None, str]:
    """Decode an image to RGBA for NudeNet, falling back to Pillow when needed."""
    errors: list[str] = []

    with suppress_native_stderr(suppress_warnings):
        try:
            import cv2
            import numpy as np

            data = np.fromfile(str(path), dtype=np.uint8)
            mat = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            rgba = _cv2_to_rgba(mat)
            if rgba is not None:
                return rgba, ""
            errors.append("opencv_decoder_returned_no_pixels")
        except Exception as e:  # noqa: BLE001
            errors.append(f"opencv:{e}")

        try:
            import numpy as np
            from PIL import Image, ImageFile, ImageOps
            import pillow_heif

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            if hasattr(pillow_heif, "register_heif_opener"):
                pillow_heif.register_heif_opener()
            with Image.open(path) as im:
                im = ImageOps.exif_transpose(im)
                im.load()
                return np.array(im.convert("RGBA")), ""
        except Exception as e:  # noqa: BLE001
            errors.append(f"pillow:{e}")

    return None, "; ".join(errors) if errors else "unreadable_image"


def detect_batch_safely(detector: Any,
                        batch: list[Path],
                        batch_size: int,
                        suppress_warnings: bool = True) -> list[Any]:
    prepared: list[Any] = []
    prepared_paths: list[Path] = []
    results: list[Any] = [None] * len(batch)

    for i, path in enumerate(batch):
        image, error = decode_for_detector(path, suppress_warnings=suppress_warnings)
        if image is None:
            results[i] = {"__error__": error}
            continue
        prepared.append(image)
        prepared_paths.append(path)

    if not prepared:
        return results

    try:
        with suppress_native_stderr(suppress_warnings):
            detected = detector.detect_batch(prepared, batch_size=batch_size)
    except Exception as batch_error:  # noqa: BLE001
        detected = []
        for image in prepared:
            try:
                with suppress_native_stderr(suppress_warnings):
                    detected.append(detector.detect(image))
            except Exception as e:  # noqa: BLE001
                detected.append({"__error__": f"detector:{e}; batch:{batch_error}"})

    detected_iter = iter(detected)
    by_path = {path: next(detected_iter, {"__error__": "missing_detector_result"})
               for path in prepared_paths}
    for i, path in enumerate(batch):
        if results[i] is None:
            results[i] = by_path.get(path, {"__error__": "missing_detector_result"})
    return results


def unique_dest(dest: Path) -> Path:
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


def classify_detections(detections: list[dict],
                        threshold: float,
                        uncertain_threshold: float) -> tuple[str, str, float]:
    # Kept in the signature for CLI compatibility. Classification thresholds
    # are centralized in sort_photos so manual and daily runs cannot diverge.
    del threshold, uncertain_threshold
    decision, best_class, best_score, _reason = sort_photos.nudity_decision(detections)
    # On a broad scan, no anatomy detection is ordinary safe content. The
    # stricter audit behavior is reserved for files already inside nude.
    if decision == "needs_review" and not best_class:
        decision = "likely_safe"
    return decision, best_class, best_score


def copy_or_move(src: Path, dest: Path, move: bool) -> None:
    if move:
        raise ValueError("moving protected originals out of photos_by_person is disabled")
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dest))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apply", action="store_true",
                        help="Copy flagged images to review folders. Default is dry-run.")
    parser.add_argument("--move", action="store_true",
                        help="Deprecated and disabled: protected originals are never moved to review output.")
    parser.add_argument("--threshold", type=float, default=NUDITY_THRESHOLD,
                        help="Deprecated compatibility option; the shared policy is always used.")
    parser.add_argument("--uncertain-threshold", type=float, default=NUDITY_UNCERTAIN_THRESHOLD,
                        help="Deprecated compatibility option; the shared policy is always used.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only scan first N images, useful for testing.")
    parser.add_argument("--copy-safe", action="store_true",
                        help="Also copy safe images to _nudity_review/safe.")
    parser.add_argument(
        "--no-export-copies",
        action="store_true",
        help="Write the complete report without duplicating images into _nudity_review.",
    )
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show-codec-warnings", action="store_true",
                        help="Show low-level JPEG/PNG decoder warnings.")
    args = parser.parse_args()

    if args.move:
        print("ERROR: --move is disabled. Review export may only copy protected originals.")
        return 1
    if args.uncertain_threshold > args.threshold:
        print("ERROR: --uncertain-threshold cannot be above --threshold.")
        return 1

    input_dir = Path(args.input_dir).expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not input_dir.exists():
        print(f"ERROR: input folder not found: {input_dir}")
        return 1

    try:
        from nudenet import NudeDetector
    except ImportError:
        print("ERROR: NudeNet is not installed. Run: pip install --upgrade nudenet")
        return 1

    images = iter_images(input_dir)
    if args.limit > 0:
        images = images[:args.limit]

    report_path = output_dir / f"nudity_review_report_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    if args.apply:
        output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Input:              {input_dir}")
    print(f"Output:             {output_dir}")
    print(f"Images to scan:     {len(images)}")
    print(f"Mode:               COPY {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Policy:             {POLICY_VERSION} (shared with daily ingest)")
    print()

    detector = None
    counts = {
        "confirmed_nude": 0,
        "needs_review": 0,
        "likely_safe": 0,
        "error": 0,
        "cache_hits": 0,
        "model_scans": 0,
    }
    rows: list[dict[str, str]] = []
    actions: list[tuple[str, Path, Path]] = []
    asset_index = analysis_index.AnalysisIndex(sort_photos.analysis_index_file())

    scanned = 0
    try:
        for batch in chunks(images, max(1, args.batch_size)):
            results: list[list[dict] | dict | None] = [None] * len(batch)
            uncached_paths: list[Path] = []
            uncached_indexes: list[int] = []
            for index, src in enumerate(batch):
                cached = asset_index.cached_nudity(
                    src,
                    sort_photos.NUDITY_ANALYSIS_VERSION,
                )
                detections = cached[1] if cached is not None else None
                if detections is None:
                    fingerprint = asset_index.fingerprint(src)
                    sha256 = (
                        fingerprint.sha256
                        if fingerprint is not None
                        else asset_index.file_sha256(src)
                    )
                    if sha256:
                        detections = asset_index.cached_nudity_detections_by_sha256(sha256)
                if detections is None:
                    uncached_paths.append(src)
                    uncached_indexes.append(index)
                else:
                    results[index] = detections
                    counts["cache_hits"] += 1

            if uncached_paths:
                if detector is None:
                    detector = NudeDetector()
                fresh = detect_batch_safely(
                    detector,
                    uncached_paths,
                    batch_size=len(uncached_paths),
                    suppress_warnings=not args.show_codec_warnings,
                )
                for index, detections in zip(uncached_indexes, fresh):
                    results[index] = detections
                    counts["model_scans"] += 1

            for src, detections in zip(batch, results):
                scanned += 1
                if detections is None:
                    detections = {"__error__": "missing_detector_result"}
                if isinstance(detections, dict) and "__error__" in detections:
                    category, best_class, best_score = "error", "ERROR", 0.0
                    detail = detections["__error__"]
                else:
                    category, best_class, best_score = classify_detections(
                        detections, args.threshold, args.uncertain_threshold)
                    detail_parts: list[str] = []
                    for detection in detections:
                        try:
                            score = float(detection.get("score", 0.0))
                        except (TypeError, ValueError):
                            score = 0.0
                        detail_parts.append(f"{detection.get('class')}:{score:.3f}")
                    detail = ";".join(detail_parts)
                    status = {
                        "confirmed_nude": "possible",
                        "needs_review": "uncertain",
                        "likely_safe": "safe",
                    }[category]
                    asset_index.record_nudity(
                        src,
                        model=sort_photos.NUDITY_ANALYSIS_VERSION,
                        status=status,
                        detections=detections,
                    )

                counts[category] += 1
                rel = src.relative_to(input_dir)
                rows.append({
                    "policy_version": POLICY_VERSION,
                    "category": category,
                    "best_class": best_class,
                    "best_score": f"{best_score:.3f}",
                    "threshold": f"{args.threshold:.3f}",
                    "uncertain_threshold": f"{args.uncertain_threshold:.3f}",
                    "source": str(src),
                    "relative_path": str(rel),
                    "detections": detail,
                })

                should_export = category in {"confirmed_nude", "needs_review"} or (
                    category == "likely_safe" and args.copy_safe)
                if should_export and not args.no_export_copies:
                    dest = unique_dest(output_dir / category / rel)
                    actions.append((category, src, dest))

            asset_index.commit()
            if not args.quiet and (scanned == len(images) or scanned % 250 == 0):
                print(f"Scanned {scanned}/{len(images)} "
                      f"(confirmed={counts['confirmed_nude']}, "
                      f"review={counts['needs_review']}, "
                      f"safe={counts['likely_safe']}, "
                      f"errors={counts['error']}, "
                      f"cache={counts['cache_hits']}, "
                      f"model={counts['model_scans']})", flush=True)
    finally:
        asset_index.close()

    if args.apply:
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["policy_version", "category", "best_class", "best_score",
                            "threshold", "uncertain_threshold", "source",
                            "relative_path", "detections"],
            )
            writer.writeheader()
            writer.writerows(rows)

    print()
    print("---- Results ----")
    print(f"Confirmed nude:  {counts['confirmed_nude']}")
    print(f"Needs review:    {counts['needs_review']}")
    print(f"Likely safe:     {counts['likely_safe']}")
    print(f"Errors:          {counts['error']}")
    print(f"Cache reused:    {counts['cache_hits']}")
    print(f"Model scans:     {counts['model_scans']}")
    print(f"Files to export: {len(actions)}")
    print()

    if not args.apply:
        print("DRY-RUN — no files copied/moved and no report written.")
        print("Re-run with --apply to create the review folders.")
        return 0

    exported = 0
    for _category, src, dest in actions:
        if not src.exists():
            continue
        copy_or_move(src, dest, args.move)
        exported += 1

    print(f"Exported {exported} file(s).")
    print(f"Report: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

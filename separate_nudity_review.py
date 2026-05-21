#!/usr/bin/env python3
"""
Separate possible nudity images into review folders using NudeNet.

The script scans person folders and copies flagged images to:
  ~/Pictures/sorted_all_pictures/_nudity_review/possible_nudity
  ~/Pictures/sorted_all_pictures/_nudity_review/uncertain

It leaves photos_by_person untouched unless --move is used. A CSV report is
always written. Default is dry-run; use --apply to copy/move files.
"""

from __future__ import annotations

import argparse
import csv
import os
import shutil
import time
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp",
              ".tif", ".tiff", ".heic", ".heif"}
DEFAULT_INPUT = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
DEFAULT_OUTPUT = Path.home() / "Pictures" / "sorted_all_pictures" / "_nudity_review"
EXCLUDED_DIRS = {
    "all",
    "photos_nude",
    "_possible_nudity",
    "_smart_albums",
    "_uncertain_nudity",
    "review",
}

EXPLICIT_CLASSES = {
    "FEMALE_BREAST_EXPOSED",
    "FEMALE_GENITALIA_EXPOSED",
    "MALE_GENITALIA_EXPOSED",
    "BUTTOCKS_EXPOSED",
    "ANUS_EXPOSED",
}


def iter_images(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [
            d for d in dirnames
            if not d.startswith(".") and d.casefold() not in EXCLUDED_DIRS
        ]
        base = Path(dirpath)
        for filename in filenames:
            p = base / filename
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
    return sorted(out, key=lambda p: str(p).lower())


def chunks(items: list[Path], size: int):
    for i in range(0, len(items), size):
        yield items[i:i + size]


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
    explicit = [
        d for d in detections
        if d.get("class") in EXPLICIT_CLASSES
    ]
    if not explicit:
        return "safe", "", 0.0

    best = max(explicit, key=lambda d: float(d.get("score", 0.0)))
    best_class = str(best.get("class", ""))
    best_score = float(best.get("score", 0.0))

    if best_score >= threshold:
        return "possible_nudity", best_class, best_score
    if best_score >= uncertain_threshold:
        return "uncertain", best_class, best_score
    return "safe", best_class, best_score


def copy_or_move(src: Path, dest: Path, move: bool) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if move:
        shutil.move(str(src), str(dest))
    else:
        shutil.copy2(str(src), str(dest))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input_dir", nargs="?", default=str(DEFAULT_INPUT))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--apply", action="store_true",
                        help="Copy flagged images to review folders. Default is dry-run.")
    parser.add_argument("--move", action="store_true",
                        help="Move flagged images instead of copying them. Use carefully.")
    parser.add_argument("--threshold", type=float, default=0.35,
                        help="Score needed for possible_nudity. Default: 0.35")
    parser.add_argument("--uncertain-threshold", type=float, default=0.20,
                        help="Lower score sent to uncertain. Default: 0.20")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0,
                        help="Only scan first N images, useful for testing.")
    parser.add_argument("--copy-safe", action="store_true",
                        help="Also copy safe images to _nudity_review/safe.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    if args.move and not args.apply:
        print("ERROR: --move requires --apply.")
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
    print(f"Mode:               {'MOVE' if args.move else 'COPY'} {'APPLY' if args.apply else 'DRY-RUN'}")
    print(f"Thresholds:         possible={args.threshold:.2f}, uncertain={args.uncertain_threshold:.2f}")
    print()

    detector = NudeDetector()
    counts = {"possible_nudity": 0, "uncertain": 0, "safe": 0, "error": 0}
    rows: list[dict[str, str]] = []
    actions: list[tuple[str, Path, Path]] = []

    scanned = 0
    for batch in chunks(images, max(1, args.batch_size)):
        try:
            results = detector.detect_batch([str(p) for p in batch], batch_size=len(batch))
        except Exception:
            results = []
            for p in batch:
                try:
                    results.append(detector.detect(str(p)))
                except Exception as e:  # noqa: BLE001
                    results.append({"__error__": str(e)})

        for src, detections in zip(batch, results):
            scanned += 1
            if isinstance(detections, dict) and "__error__" in detections:
                category, best_class, best_score = "error", "ERROR", 0.0
                detail = detections["__error__"]
            else:
                category, best_class, best_score = classify_detections(
                    detections, args.threshold, args.uncertain_threshold)
                detail = ";".join(
                    f"{d.get('class')}:{float(d.get('score', 0.0)):.3f}"
                    for d in detections
                )

            counts[category] += 1
            rel = src.relative_to(input_dir)
            rows.append({
                "category": category,
                "best_class": best_class,
                "best_score": f"{best_score:.3f}",
                "source": str(src),
                "relative_path": str(rel),
                "detections": detail,
            })

            should_export = category in {"possible_nudity", "uncertain"} or (
                category == "safe" and args.copy_safe)
            if should_export:
                dest = unique_dest(output_dir / category / rel)
                actions.append((category, src, dest))

        if not args.quiet and (scanned == len(images) or scanned % 250 == 0):
            print(f"Scanned {scanned}/{len(images)} "
                  f"(possible={counts['possible_nudity']}, "
                  f"uncertain={counts['uncertain']}, safe={counts['safe']}, "
                  f"errors={counts['error']})")

    if args.apply:
        with report_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["category", "best_class", "best_score",
                            "source", "relative_path", "detections"],
            )
            writer.writeheader()
            writer.writerows(rows)

    print()
    print("---- Results ----")
    print(f"Possible nudity: {counts['possible_nudity']}")
    print(f"Uncertain:       {counts['uncertain']}")
    print(f"Safe:            {counts['safe']}")
    print(f"Errors:          {counts['error']}")
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

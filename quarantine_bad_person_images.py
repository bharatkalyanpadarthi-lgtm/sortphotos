#!/usr/bin/env python3
"""
Move unreadable image-extension files out of photos_by_person.

This catches recovery artifacts that have names like .jpg/.png but do not
decode as images. Files are moved to:
  sorted_all_pictures/_source_review/ready_to_delete/bad_person_images

Default is dry-run. Use --apply to move files.
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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp",
              ".tif", ".tiff", ".heic", ".heif", ".gif"}
DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
DEFAULT_REVIEW = (
    Path.home()
    / "Pictures"
    / "sorted_all_pictures"
    / "_source_review"
    / "ready_to_delete"
    / "bad_person_images"
)
DEFAULT_REPORT_DIR = (
    Path.home()
    / "Pictures"
    / "sorted_all_pictures"
    / "_source_review"
    / "repair_logs"
)
SKIP_DIRS = {"all", "_smart_albums"}


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


def iter_images(people_dir: Path, include_review: bool = False) -> list[Path]:
    out: list[Path] = []
    if not people_dir.exists():
        return out
    for person_dir in sorted(
        [p for p in people_dir.iterdir() if p.is_dir() and not p.name.startswith("_")],
        key=lambda p: p.name.casefold(),
    ):
        for dirpath, dirnames, filenames in os.walk(person_dir):
            dirnames[:] = [
                d for d in dirnames
                if d not in SKIP_DIRS
                and not d.startswith(".")
                and (include_review or d != "review")
            ]
            base = Path(dirpath)
            for filename in filenames:
                path = base / filename
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    out.append(path)
    return sorted(out, key=lambda p: str(p).casefold())


def can_decode_image(path: Path, show_codec_warnings: bool = False) -> tuple[bool, str]:
    errors: list[str] = []
    with suppress_native_stderr(not show_codec_warnings):
        try:
            import cv2
            import numpy as np

            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is not None and getattr(img, "size", 0) > 0:
                return True, "opencv"
            errors.append("opencv_decoder_returned_no_pixels")
        except Exception as e:  # noqa: BLE001
            errors.append(f"opencv:{e}")

        try:
            from PIL import Image, ImageFile

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(path) as im:
                im.load()
                if im.size[0] > 0 and im.size[1] > 0:
                    return True, "pillow"
                errors.append("pillow_decoder_returned_no_pixels")
        except Exception as e:  # noqa: BLE001
            errors.append(f"pillow:{e}")

    return False, "; ".join(errors) if errors else "unreadable_image"


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


def move_bad(src: Path, people_dir: Path, review_dir: Path) -> Path:
    rel = src.relative_to(people_dir)
    dest = unique_dest(review_dir / rel)
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(src), str(dest))
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("people_dir", nargs="?", default=str(DEFAULT_PEOPLE))
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--include-review", action="store_true",
                        help="Also scan per-person review folders.")
    parser.add_argument("--apply", action="store_true",
                        help="Move unreadable files. Default is dry-run.")
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--show-codec-warnings", action="store_true",
                        help="Show low-level JPEG/PNG decoder warnings.")
    args = parser.parse_args()

    people_dir = Path(args.people_dir).expanduser().resolve()
    review_dir = args.review_dir.expanduser().resolve()
    report_dir = args.report_dir.expanduser().resolve()
    if not people_dir.exists():
        print(f"ERROR: people folder not found: {people_dir}")
        return 1

    images = iter_images(people_dir, include_review=args.include_review)
    report = report_dir / f"bad_person_images_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    print(f"People folder:      {people_dir}")
    print(f"Bad-image review:   {review_dir}")
    print(f"Images to check:    {len(images)}")
    print(f"Mode:               {'APPLY' if args.apply else 'DRY-RUN'}")
    print()

    rows: list[dict[str, str]] = []
    checked = good = bad = moved = move_errors = 0
    for path in images:
        checked += 1
        ok, detail = can_decode_image(path, show_codec_warnings=args.show_codec_warnings)
        if ok:
            good += 1
            continue

        bad += 1
        row = {
            "source": str(path),
            "relative_path": str(path.relative_to(people_dir)),
            "status": "bad",
            "detail": detail,
            "dest": "",
        }
        if args.apply:
            try:
                dest = move_bad(path, people_dir, review_dir)
                row["status"] = "moved"
                row["dest"] = str(dest)
                moved += 1
            except Exception as e:  # noqa: BLE001
                row["status"] = f"move_error:{type(e).__name__}"
                row["detail"] = f"{detail}; move:{e}"
                move_errors += 1
        rows.append(row)

        if not args.quiet and (checked == len(images) or checked % 500 == 0):
            print(f"Checked {checked}/{len(images)} "
                  f"(good={good}, bad={bad}, moved={moved}, move_errors={move_errors})")

    report_dir.mkdir(parents=True, exist_ok=True)
    with report.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["source", "relative_path", "status", "detail", "dest"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("---- Results ----")
    print(f"Good images:        {good}")
    print(f"Unreadable/bad:     {bad}")
    print(f"Moved:              {moved}")
    print(f"Move errors:        {move_errors}")
    print(f"Report:             {report}")
    if not args.apply:
        print()
        print("DRY-RUN only. Re-run with --apply to move bad files.")
    return 1 if move_errors else 0


if __name__ == "__main__":
    raise SystemExit(main())

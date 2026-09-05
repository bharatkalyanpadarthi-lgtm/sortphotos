#!/usr/bin/env python3
"""Run image intake in restartable subprocesses with bounded memory use."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import pipeline_paths


SCRIPT_DIR = Path(__file__).resolve().parent
IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
    ".tif", ".tiff", ".heic", ".heif",
}
DEFAULT_MAX_IMAGES = max(
    100,
    int(os.environ.get("FACE_INTAKE_BATCH_IMAGES", "5000")),
)


def count_images(root: Path) -> int:
    """Count the same visible image files that the To Process scan can see."""
    total = 0
    for _dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [name for name in dirnames if not name.startswith(".")]
        total += sum(
            1 for filename in filenames
            if Path(filename).suffix.lower() in IMAGE_EXTS
        )
    return total


def build_sort_command(
    input_dir: Path,
    output_dir: Path,
    *,
    max_images: int,
    detection_batch_size: int,
    detect_workers: int,
) -> list[str]:
    return [
        sys.executable,
        str(SCRIPT_DIR / "sort_photos.py"),
        str(input_dir),
        str(output_dir),
        "--unattended",
        "--archive-organized-sources",
        "--archive-sources-to-ready-delete",
        "--archive-scanned-sources",
        "--merge-existing-output",
        "--skip-output-cleanup",
        "--max-input-images", str(max(1, max_images)),
        "--batch-size", str(max(1, detection_batch_size)),
        "--detect-workers", str(max(1, detect_workers)),
    ]


def run_batches(
    input_dir: Path,
    output_dir: Path,
    *,
    max_images: int,
    detection_batch_size: int,
    detect_workers: int,
    max_batches: int = 0,
) -> int:
    completed = 0
    while True:
        before = count_images(input_dir)
        if before == 0:
            print("Image intake complete: no images remain in To Process.", flush=True)
            return 0
        if max_batches > 0 and completed >= max_batches:
            print(
                f"Stopped after {completed} requested worker batch(es); "
                f"{before:,} image(s) remain.",
                flush=True,
            )
            return 0

        batch_number = completed + 1
        this_batch = min(max_images, before)
        print(flush=True)
        print("Image Intake Supervisor", flush=True)
        print("=" * 60, flush=True)
        print(f"Worker batch: {batch_number}", flush=True)
        print(f"Remaining:    {before:,} image(s)", flush=True)
        print(f"This worker:  up to {this_batch:,} image(s)", flush=True)
        print("Memory safety: worker exits after this slice", flush=True)

        command = build_sort_command(
            input_dir,
            output_dir,
            max_images=max_images,
            detection_batch_size=detection_batch_size,
            detect_workers=detect_workers,
        )
        result = subprocess.run(command, check=False)
        if result.returncode != 0:
            signal_hint = (
                " The worker was killed by macOS; rerunning will resume this slice."
                if result.returncode < 0 else ""
            )
            print(
                f"Image worker batch {batch_number} stopped with exit "
                f"{result.returncode}.{signal_hint}",
                flush=True,
            )
            print("Completed earlier batches remain preserved.", flush=True)
            return result.returncode

        after = count_images(input_dir)
        completed += 1
        processed = max(0, before - after)
        print(
            f"Worker batch {batch_number} complete: {processed:,} image(s) "
            f"left the inbox; {after:,} remain.",
            flush=True,
        )
        if after >= before:
            print(
                "ERROR: the worker completed but made no inbox progress. "
                "Stopping to avoid an endless retry loop.",
                flush=True,
            )
            return 5


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=pipeline_paths.TO_PROCESS)
    parser.add_argument("output", nargs="?", type=Path, default=pipeline_paths.SORTED_ROOT)
    parser.add_argument("--max-images-per-process", type=int, default=DEFAULT_MAX_IMAGES)
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--detect-workers", type=int, default=1)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=0,
        help="Stop after N workers; 0 continues until the inbox is empty.",
    )
    parser.add_argument(
        "--skip-output-cleanup",
        action="store_true",
        help="Compatibility flag; cleanup is always deferred to daily_runner.",
    )
    args = parser.parse_args()
    return run_batches(
        args.input.expanduser().resolve(),
        args.output.expanduser().resolve(),
        max_images=max(100, int(args.max_images_per_process)),
        detection_batch_size=max(1, int(args.batch_size)),
        detect_workers=max(1, int(args.detect_workers)),
        max_batches=max(0, int(args.max_batches)),
    )


if __name__ == "__main__":
    raise SystemExit(main())

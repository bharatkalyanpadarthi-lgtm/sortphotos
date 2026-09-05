#!/usr/bin/env python3
"""Run video identity analysis in restartable, memory-bounded worker chunks."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import pipeline_paths
import sort_photos


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SORTED = pipeline_paths.SORTED_ROOT
LEGACY_VIDEO_INBOX = Path.home() / "Pictures" / "videos"
DEFAULT_CHUNK_SIZE = 6


def chunks(items: list[Path], size: int) -> list[list[Path]]:
    size = max(1, size)
    return [items[index:index + size] for index in range(0, len(items), size)]


def discover_videos(
    input_dir: Path,
    sorted_root: Path,
    *,
    review_only: bool,
    include_legacy: bool,
) -> list[Path]:
    if review_only:
        review_root = sorted_root / "_source_review" / "unassigned_intake" / "videos"
        roots = [review_root / "unknown_identity", review_root / "no_usable_face"]
    else:
        roots = [input_dir]
        legacy = LEGACY_VIDEO_INBOX.expanduser().resolve()
        if include_legacy and legacy.exists() and legacy != input_dir:
            roots.append(legacy)

    by_path: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for video in sort_photos.iter_videos(root):
            resolved = video.resolve()
            by_path[str(resolved)] = resolved
    return sorted(by_path.values(), key=lambda path: str(path).casefold())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", nargs="?", type=Path, default=pipeline_paths.TO_PROCESS)
    parser.add_argument("output", nargs="?", type=Path, default=DEFAULT_SORTED)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--no-legacy-inbox", action="store_true")
    parser.add_argument("--max-videos", type=int, default=0)
    parser.add_argument("--max-samples", type=int, default=48)
    parser.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)
    args = parser.parse_args()

    input_dir = args.input.expanduser().resolve()
    sorted_root = args.output.expanduser().resolve()
    videos = discover_videos(
        input_dir,
        sorted_root,
        review_only=args.review_only,
        include_legacy=not args.no_legacy_inbox,
    )
    if args.max_videos > 0:
        videos = videos[:args.max_videos]
    if not videos:
        print("No videos need analysis.")
        return 0

    work = chunks(videos, args.chunk_size)
    print("Video Analysis Supervisor")
    print("=" * 60)
    print(f"Videos:       {len(videos)}")
    print(f"Worker chunks:{len(work):>5} (up to {max(1, args.chunk_size)} videos each)")
    print("Memory safety: detector restarts after every chunk")

    for chunk_index, paths in enumerate(work, start=1):
        print()
        print(f"Worker chunk {chunk_index}/{len(work)}: {len(paths)} video(s)", flush=True)
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", prefix="face_video_worker_", delete=False,
            encoding="utf-8",
        ) as handle:
            json.dump([str(path) for path in paths], handle)
            file_list = Path(handle.name)
        command = [
            sys.executable,
            str(SCRIPT_DIR / "sort_videos.py"),
            str(input_dir),
            str(sorted_root),
            "--file-list", str(file_list),
            "--max-samples", str(max(8, int(args.max_samples))),
            "--no-legacy-inbox",
        ]
        if args.dry_run:
            command.append("--dry-run")
        if args.review_only:
            command.append("--review-only")
        try:
            result = subprocess.run(command, check=False)
        finally:
            file_list.unlink(missing_ok=True)
        if result.returncode != 0:
            print(f"Worker chunk {chunk_index} stopped with exit {result.returncode}.")
            print("Completed earlier chunks are preserved. Run the same command to resume remaining videos.")
            return result.returncode

    print()
    print(f"Video analysis complete: {len(videos)} video(s) across {len(work)} bounded workers.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Back up sorted_all_pictures/_source_review to the external photo backup drive.

Default behavior copies/syncs and verifies file counts. It does not delete local
files unless --delete-local is supplied.
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

DEFAULT_SOURCE = Path.home() / "Pictures" / "sorted_all_pictures" / "_source_review"
DEFAULT_DEST_ROOT = Path("/Volumes/Photos & Videos  Backup/photo_source_review_backup")


def count_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())


def size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            total += p.stat().st_size
        except OSError:
            pass
    return total


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def checksum_verify(source: Path, dest: Path) -> list[str]:
    cmd = [
        "rsync",
        "-ahcn",
        "--delete",
        f"{source}/",
        f"{dest}/",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"rsync verify failed: {result.returncode}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--checksum-verify", action="store_true",
                        help="Run rsync checksum dry-run after copy. Slower but stronger.")
    parser.add_argument("--delete-local", action="store_true",
                        help="Delete local _source_review only after checksum verification passes.")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    dest_root = args.dest_root.expanduser().resolve()
    dest = dest_root / source.name

    if not source.exists():
        print(f"ERROR: source folder not found: {source}")
        return 1
    if not dest_root.parent.exists():
        print(f"ERROR: external drive path not found: {dest_root.parent}")
        return 1

    dest.mkdir(parents=True, exist_ok=True)

    print(f"Source:      {source}")
    print(f"Destination: {dest}")
    print()

    cmd = [
        "rsync",
        "-ah",
        "--progress",
        f"{source}/",
        f"{dest}/",
    ]
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: rsync failed with exit code {result.returncode}")
        return result.returncode

    source_count = count_files(source)
    dest_count = count_files(dest)
    source_size = size_bytes(source)
    dest_size = size_bytes(dest)

    print()
    print("Backup verification")
    print(f"  Source files:      {source_count}")
    print(f"  Destination files: {dest_count}")
    print(f"  Source size:       {human_size(source_size)}")
    print(f"  Destination size:  {human_size(dest_size)}")

    if source_count != dest_count:
        print("ERROR: file counts do not match. Do not delete local source.")
        return 2

    verify_lines: list[str] = []
    if args.checksum_verify or args.delete_local:
        print()
        print("Running checksum dry-run verification...")
        verify_lines = checksum_verify(source, dest)
        if verify_lines:
            print("ERROR: checksum verification found differences. Do not delete local source.")
            for line in verify_lines[:80]:
                print(line)
            if len(verify_lines) > 80:
                print(f"... and {len(verify_lines) - 80} more")
            return 3
        print("Checksum verification passed.")

    if args.delete_local:
        print()
        print(f"Deleting local source: {source}")
        shutil.rmtree(source)
        source.mkdir(parents=True, exist_ok=True)
        print("Local _source_review recreated empty for future pipeline runs.")

    print()
    print("Backup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

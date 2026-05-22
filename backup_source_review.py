#!/usr/bin/env python3
"""
Back up sorted_all_pictures/_source_review to the external photo backup drive.

Default behavior copies/syncs and verifies file counts. The guided cleanup mode
also runs checksum verification and asks you to type "yes" before deleting the
local _source_review folder.
"""

from __future__ import annotations

import argparse
import select
import shutil
import subprocess
import time
from pathlib import Path

DEFAULT_SOURCE = Path.home() / "Pictures" / "sorted_all_pictures" / "_source_review"
DEFAULT_DEST_ROOT = Path("/Volumes/Photos & Videos  Backup/photo_source_review_backup")


def scan_tree(root: Path, label: str) -> tuple[int, int]:
    if not root.exists():
        return 0, 0
    print(f"Scanning {label} for verification counts...")
    started = time.monotonic()
    last_print = started
    count = 0
    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        try:
            count += 1
            total += p.stat().st_size
        except OSError:
            pass
        now = time.monotonic()
        if count % 10000 == 0 or now - last_print >= 10:
            print(f"  {label}: {count:,} files, {human_size(total)} scanned...")
            last_print = now
    print(f"  {label}: {count:,} files, {human_size(total)} total.")
    return count, total


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
        "-ahcni",
        "--delete",
        f"{source}/",
        f"{dest}/",
    ]
    print("  Verifying checksums with rsync dry-run. This can take a while for large folders.")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    started = time.monotonic()
    last_print = started
    while True:
        ready, _w, _x = select.select([proc.stdout], [], [], 1.0)
        if ready:
            line = proc.stdout.readline()
            if line:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
                    if len(lines) <= 20:
                        print(f"  Difference candidate: {stripped}")
            elif proc.poll() is not None:
                break
        if proc.poll() is not None:
            for line in proc.stdout:
                stripped = line.strip()
                if stripped:
                    lines.append(stripped)
                    if len(lines) <= 20:
                        print(f"  Difference candidate: {stripped}")
            break
        now = time.monotonic()
        if now - last_print >= 10:
            elapsed = int(now - started)
            print(f"  Checksum verification still running... {elapsed}s elapsed")
            last_print = now
    returncode = proc.wait()
    if returncode != 0:
        raise RuntimeError(f"rsync verify failed: {returncode}")
    return lines


def is_dangerous_path(source: Path, dest: Path) -> bool:
    source = source.resolve()
    dest = dest.resolve()
    if source == dest:
        return True
    try:
        dest.relative_to(source)
        return True
    except ValueError:
        return False


def confirm_delete_local(source: Path, dest: Path) -> bool:
    print()
    print("Backup is verified.")
    print(f"Local folder to delete: {source}")
    print(f"Verified backup folder: {dest}")
    print()
    ans = input("Type yes to delete the local _source_review now: ").strip()
    return ans == "yes"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest-root", type=Path, default=DEFAULT_DEST_ROOT)
    parser.add_argument("--snapshot", action="store_true",
                        help="Write to a timestamped backup folder instead of reusing _source_review.")
    parser.add_argument("--mirror-destination", action="store_true",
                        help="Use rsync --delete so destination exactly mirrors local _source_review.")
    parser.add_argument("--checksum-verify", action="store_true",
                        help="Run rsync checksum dry-run after copy. Slower but stronger.")
    parser.add_argument("--ask-delete-local", action="store_true",
                        help="After successful verification, ask whether to delete local _source_review.")
    parser.add_argument("--delete-local", action="store_true",
                        help="Delete local _source_review after verification. Requires typing yes unless --yes is supplied.")
    parser.add_argument("--yes", action="store_true",
                        help="With --delete-local, skip the interactive yes prompt.")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    dest_root = args.dest_root.expanduser().resolve()
    if args.snapshot:
        dest = dest_root / f"{source.name}_{time.strftime('%Y%m%d_%H%M%S')}"
    else:
        dest = dest_root / source.name

    if not source.exists():
        print(f"ERROR: source folder not found: {source}")
        return 1
    if not dest_root.parent.exists():
        print(f"ERROR: external drive path not found: {dest_root.parent}")
        return 1
    if is_dangerous_path(source, dest):
        print("ERROR: destination must not be the same as, or inside, the local source folder.")
        return 1

    dest.mkdir(parents=True, exist_ok=True)

    print(f"Source:      {source}", flush=True)
    print(f"Destination: {dest}", flush=True)
    if args.mirror_destination:
        print()
        print("WARNING: --mirror-destination uses rsync --delete and can remove files")
        print("from the destination that are not currently present in the local source.")
        if not args.yes:
            ans = input("Type mirror-delete to continue with destination deletion: ").strip()
            if ans != "mirror-delete":
                print("Mirror backup cancelled before rsync.")
                return 1
    print(flush=True)

    print("Starting rsync backup. Per-file progress will be shown below.", flush=True)
    cmd = [
        "rsync",
        "-ah",
        "--progress",
        "--stats",
    ]
    if args.mirror_destination:
        cmd.append("--delete")
    cmd.extend([f"{source}/", f"{dest}/"])
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"ERROR: rsync failed with exit code {result.returncode}")
        return result.returncode

    print()
    source_count, source_size = scan_tree(source, "source")
    dest_count, dest_size = scan_tree(dest, "destination")

    print()
    print("Backup verification")
    print(f"  Source files:      {source_count}")
    print(f"  Destination files: {dest_count}")
    print(f"  Source size:       {human_size(source_size)}")
    print(f"  Destination size:  {human_size(dest_size)}")

    if source_count != dest_count:
        print("ERROR: file counts do not match. Do not delete local source.")
        return 2
    if source_size != dest_size:
        print("ERROR: folder sizes do not match. Do not delete local source.")
        return 2

    verify_lines: list[str] = []
    need_checksum = args.checksum_verify or args.ask_delete_local or args.delete_local
    if need_checksum:
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

    should_delete = False
    if args.ask_delete_local or args.delete_local:
        if args.yes and args.delete_local:
            should_delete = True
        else:
            should_delete = confirm_delete_local(source, dest)

    if should_delete:
        print()
        print(f"Deleting local source: {source}")
        shutil.rmtree(source)
        source.mkdir(parents=True, exist_ok=True)
        print("Local _source_review recreated empty for future pipeline runs.")
    elif args.ask_delete_local or args.delete_local:
        print()
        print("Local _source_review was kept.")

    print()
    print("Backup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

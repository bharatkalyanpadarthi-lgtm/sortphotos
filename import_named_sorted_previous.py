#!/usr/bin/env python3
"""
Import already named folders from old sorted output into photos_by_person.

This is for archived folders like:
  sorted_all_pictures/_source_review/sorted_previous/photos_by_person/Anushka

It intentionally skips generic old cluster names such as person_039, junk, and
unknown folders. Imported images are exact-hash deduped per target person; when
the same image content already exists elsewhere in photos_by_person, the new
entry is made as a hardlink to avoid redundant storage.

Default is dry-run. Use --apply to create links/copies.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif",
              ".tif", ".tiff", ".heic", ".heif"}
DEFAULT_SOURCE = (
    Path.home()
    / "Pictures"
    / "sorted_all_pictures"
    / "_source_review"
    / "sorted_previous"
    / "photos_by_person"
)
DEFAULT_DEST = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
SKIP_NAMES = {"__junk__", "junk", "unknown", "_unknown", "_duplicates", "_blurred"}


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def sha1_file(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def iter_images(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in {".DS_Store"}]
        base = Path(dirpath)
        for filename in filenames:
            p = base / filename
            if is_image(p):
                out.append(p)
    return out


def safe_folder_name(name: str) -> str:
    clean = name.strip().replace("/", "_").replace(":", "_")
    return clean or "unnamed"


def should_import_folder(name: str) -> bool:
    lowered = name.strip().lower()
    if lowered in SKIP_NAMES:
        return False
    if lowered.startswith("_"):
        return False
    if re.fullmatch(r"person_\d+", lowered):
        return False
    return True


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = "".join(dest.suffixes)
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__import{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def build_dest_index(dest_root: Path) -> dict[int, list[Path]]:
    by_size: dict[int, list[Path]] = defaultdict(list)
    for p in iter_images(dest_root):
        try:
            by_size[p.stat().st_size].append(p)
        except OSError:
            continue
    return by_size


def existing_hashes_for_person(person_dir: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for p in iter_images(person_dir):
        try:
            out.setdefault(sha1_file(p), p)
        except OSError:
            continue
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--apply", action="store_true",
                        help="Import named folders. Default is dry-run.")
    parser.add_argument("--global-hardlink", action="store_true",
                        help="Find matching content anywhere in photos_by_person and hardlink it. Slower.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print sample actions.")
    args = parser.parse_args()

    source = args.source.expanduser().resolve()
    dest_root = args.dest.expanduser().resolve()
    if not source.exists():
        print(f"ERROR: source folder not found: {source}")
        return 1
    if not dest_root.exists():
        print(f"ERROR: destination folder not found: {dest_root}")
        return 1

    named_dirs: list[Path] = []
    skipped_dirs: list[Path] = []
    for child in sorted(source.iterdir(), key=lambda p: p.name.lower()):
        if not child.is_dir():
            continue
        if should_import_folder(child.name):
            named_dirs.append(child)
        else:
            skipped_dirs.append(child)

    by_size = build_dest_index(dest_root) if args.global_hardlink else {}
    global_by_hash: dict[str, Path] = {}
    hashed_sizes: set[int] = set()

    def global_match(src: Path, src_hash: str) -> Path | None:
        try:
            size = src.stat().st_size
        except OSError:
            return None
        if size not in hashed_sizes:
            for p in by_size.get(size, []):
                try:
                    global_by_hash.setdefault(sha1_file(p), p)
                except OSError:
                    continue
            hashed_sizes.add(size)
        return global_by_hash.get(src_hash)

    actions: list[tuple[str, Path, Path, Path | None]] = []
    skipped_duplicate = 0
    source_images = 0
    for src_dir in named_dirs:
        person = safe_folder_name(src_dir.name)
        person_dir = dest_root / person
        existing_for_person = existing_hashes_for_person(person_dir)
        seen_source_hashes: set[str] = set()
        for src in iter_images(src_dir):
            source_images += 1
            try:
                h = sha1_file(src)
            except OSError:
                continue
            if h in existing_for_person or h in seen_source_hashes:
                skipped_duplicate += 1
                continue
            seen_source_hashes.add(h)
            link_from = global_match(src, h) if args.global_hardlink else None
            dest = unique_dest(person_dir / src.name)
            action = "hardlink" if link_from is not None else "copy"
            actions.append((action, src, dest, link_from))
            if link_from is None:
                global_by_hash[h] = dest
            existing_for_person[h] = dest

    print(f"Source:                 {source}")
    print(f"Destination:            {dest_root}")
    print(f"Named folders imported: {len(named_dirs)}")
    print(f"Skipped folders:        {len(skipped_dirs)}")
    print(f"Source images checked:  {source_images}")
    print(f"Already in person dir:  {skipped_duplicate}")
    print(f"New entries to create:  {len(actions)}")
    print(f"  hardlinks:            {sum(1 for a, *_ in actions if a == 'hardlink')}")
    print(f"  copies:               {sum(1 for a, *_ in actions if a == 'copy')}")
    print()

    if not args.quiet:
        print("Named folders:")
        for d in named_dirs[:80]:
            print(f"  {d.name}")
        if len(named_dirs) > 80:
            print(f"  ... and {len(named_dirs) - 80} more")
        print()
        for action, src, dest, link_from in actions[:80]:
            if action == "hardlink":
                print(f"hardlink: {dest}")
                print(f"  from existing: {link_from}")
                print(f"  source ref:    {src}")
            else:
                print(f"copy: {src}")
                print(f"  -> {dest}")
        if len(actions) > 80:
            print(f"... and {len(actions) - 80} more")
        print()

    if not args.apply:
        print("DRY-RUN — no files changed. Re-run with --apply to import.")
        return 0

    created = 0
    for action, src, dest, link_from in actions:
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        if action == "hardlink" and link_from is not None and link_from.exists():
            os.link(str(link_from), str(dest))
        else:
            shutil.copy2(src, dest)
        created += 1

    print(f"Imported {created} image entries into named person folders.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

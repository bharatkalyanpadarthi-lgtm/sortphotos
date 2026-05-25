#!/usr/bin/env python3
"""
Apply reusable merge/rename/remove rules to photos_by_person.

Default is a dry-run. Use --apply only after reviewing the printed summary.
Removed folders and duplicate merge losers are moved to ready_to_delete, never
deleted directly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
DEFAULT_CLEANUP_ROOT = (
    Path.home() / "Pictures" / "sorted_all_pictures" / "_source_review" / "ready_to_delete"
)
DEFAULT_RULES = Path(__file__).resolve().with_name("person_folder_rules.json")
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
SKIP_DIRS = {"all", "_smart_albums"}


def load_rules(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    return {
        "merge": data.get("merge", {}),
        "rename": data.get("rename", {}),
        "remove": data.get("remove", []),
    }


def clean_name(name: str) -> str:
    return " ".join(name.strip().split()).lower()


def folder_map(people_dir: Path) -> dict[str, Path]:
    return {clean_name(p.name): p for p in people_dir.iterdir() if p.is_dir()}


def iter_files(root: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        base = Path(dirpath)
        for filename in filenames:
            path = base / filename
            if path.is_file():
                out.append(path)
    return sorted(out, key=lambda p: str(p.relative_to(root)).lower())


def file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def move_path(src: Path, dest: Path, apply: bool) -> None:
    if apply:
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))


def merge_folder(src: Path,
                 dest: Path,
                 archive_root: Path,
                 apply: bool,
                 rows: list[dict[str, str]]) -> tuple[int, int]:
    moved = 0
    dupes = 0
    dest_hashes: dict[str, Path] = {}
    if dest.exists():
        for existing in iter_files(dest):
            if existing.suffix.lower() in IMAGE_EXTS:
                try:
                    dest_hashes[file_sha(existing)] = existing
                except OSError:
                    pass

    for src_file in iter_files(src):
        rel = src_file.relative_to(src)
        target = dest / rel
        try:
            sha = file_sha(src_file) if src_file.suffix.lower() in IMAGE_EXTS else ""
        except OSError:
            sha = ""

        if sha and sha in dest_hashes:
            archive_dest = unique_dest(archive_root / "duplicate_merge_losers" / src.name / rel)
            rows.append({
                "action": "duplicate_archived",
                "source": str(src_file),
                "dest": str(dest_hashes[sha]),
                "archive": str(archive_dest),
            })
            move_path(src_file, archive_dest, apply)
            dupes += 1
            continue

        target = unique_dest(target)
        rows.append({
            "action": "merged",
            "source": str(src_file),
            "dest": str(target),
            "archive": "",
        })
        move_path(src_file, target, apply)
        moved += 1

    archive_dest = unique_dest(archive_root / "merged_source_folders" / src.name)
    rows.append({"action": "source_folder_archived", "source": str(src), "dest": "", "archive": str(archive_dest)})
    if apply:
        if src.exists():
            archive_dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(archive_dest))
    return moved, dupes


def rename_folder(src: Path,
                  dest: Path,
                  archive_root: Path,
                  apply: bool,
                  rows: list[dict[str, str]]) -> tuple[int, int]:
    if src.resolve() == dest.resolve():
        return 0, 0
    if dest.exists():
        return merge_folder(src, dest, archive_root, apply, rows)
    rows.append({"action": "renamed_folder", "source": str(src), "dest": str(dest), "archive": ""})
    if apply:
        src.rename(dest)
    return 0, 0


def archive_removed(src: Path, archive_root: Path, apply: bool, rows: list[dict[str, str]]) -> int:
    archive_dest = unique_dest(archive_root / "removed_person_folders" / src.name)
    rows.append({"action": "removed_folder_archived", "source": str(src), "dest": "", "archive": str(archive_dest)})
    count = len([p for p in iter_files(src) if p.suffix.lower() in IMAGE_EXTS])
    if apply:
        archive_dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(archive_dest))
    return count


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["action", "source", "dest", "archive"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-dir", type=Path, default=DEFAULT_PEOPLE)
    parser.add_argument("--rules", type=Path, default=DEFAULT_RULES)
    parser.add_argument("--cleanup-root", type=Path, default=DEFAULT_CLEANUP_ROOT)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    people_dir = args.people_dir.expanduser().resolve()
    if not people_dir.exists():
        print(f"ERROR: people folder not found: {people_dir}")
        return 1
    rules = load_rules(args.rules.expanduser().resolve())
    archive_root = args.cleanup_root.expanduser().resolve() / f"person_folder_cleanup_{time.strftime('%Y%m%d_%H%M%S')}"

    rows: list[dict[str, str]] = []
    moved = dupes = removed_images = renamed = missing = 0
    folders = folder_map(people_dir)

    for target_name, sources in rules["merge"].items():
        target = folders.get(clean_name(target_name)) or people_dir / target_name
        if args.apply:
            target.mkdir(parents=True, exist_ok=True)
        for source_name in sources:
            src = folders.get(clean_name(source_name))
            if src is None:
                missing += 1
                continue
            if src.resolve() == target.resolve():
                continue
            m, d = merge_folder(src, target, archive_root, args.apply, rows)
            moved += m
            dupes += d
            folders = folder_map(people_dir)

    for old_name, new_name in rules["rename"].items():
        src = folders.get(clean_name(old_name))
        if src is None:
            missing += 1
            continue
        dest = folders.get(clean_name(new_name)) or people_dir / new_name
        m, d = rename_folder(src, dest, archive_root, args.apply, rows)
        moved += m
        dupes += d
        renamed += 1
        folders = folder_map(people_dir)

    for remove_name in rules["remove"]:
        src = folders.get(clean_name(remove_name))
        if src is None:
            missing += 1
            continue
        removed_images += archive_removed(src, archive_root, args.apply, rows)
        folders = folder_map(people_dir)

    report = archive_root / "person_folder_cleanup_report.csv"
    if rows and args.apply:
        write_report(report, rows)

    print(f"People folder:         {people_dir}")
    print(f"Rules file:            {args.rules.expanduser().resolve()}")
    print(f"Folders renamed:       {renamed}")
    print(f"Files merged:          {moved}")
    print(f"Duplicates archived:   {dupes}")
    print(f"Removed-folder images: {removed_images}")
    print(f"Missing old folders:   {missing}")
    print(f"Report:                {report if rows and args.apply else 'apply run only'}")
    if not args.apply:
        print()
        print("DRY-RUN - no folders changed. Re-run with --apply to commit.")
    elif not args.quiet:
        print()
        print(f"Archived removed/duplicate content under: {archive_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

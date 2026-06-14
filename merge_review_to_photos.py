#!/usr/bin/env python3
"""Move reviewed person-folder images back into canonical photos folders.

This only scans real per-person review folders:

    photos_by_person/<person>/review/...

Generated smart-album review views are ignored. Files are moved with the
operation ledger so source-manifest recovery still has a trail.
"""

from __future__ import annotations

import argparse
import re
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import operation_ledger
import source_manifest

SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
PEOPLE = SORTED / "photos_by_person"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}


@dataclass
class MovePlan:
    source: Path
    dest: Path
    person: str
    bucket: str


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def cleaned_name(path: Path) -> str:
    name = path.name
    replacements = {
        "_near_visual_review_": "_photo_",
        "_duplicate_review_": "_photo_",
    }
    for old, new in replacements.items():
        name = name.replace(old, new)
    name = re.sub(r"_+", "_", name)
    return name.strip("._- ") or path.name


def is_nudity_review(path: Path, review_dir: Path) -> bool:
    rel_parts = [part.casefold() for part in path.relative_to(review_dir).parts]
    name = path.name.casefold()
    return (
        "nude" in rel_parts
        or "nudity" in rel_parts
        or "possible_nudity" in rel_parts
        or "nudity_possible" in name
    )


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    suffix = "".join(dest.suffixes)
    stem = dest.name[:-len(suffix)] if suffix else dest.name
    for idx in range(2, 100000):
        candidate = dest.with_name(f"{stem}__reviewmerge{idx}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find unique destination for {dest}")


def iter_review_images(people_dir: Path) -> list[tuple[Path, Path, Path]]:
    items: list[tuple[Path, Path, Path]] = []
    if not people_dir.exists():
        return items
    for person_dir in sorted(people_dir.iterdir(), key=lambda p: p.name.casefold()):
        if not person_dir.is_dir() or person_dir.name.startswith("_") or person_dir.name.startswith("."):
            continue
        review_dir = person_dir / "review"
        if not review_dir.is_dir():
            continue
        for path in sorted(review_dir.rglob("*"), key=lambda p: str(p).casefold()):
            if is_image(path):
                items.append((person_dir, review_dir, path))
    return items


def build_plan(people_dir: Path) -> list[MovePlan]:
    plan: list[MovePlan] = []
    reserved: set[Path] = set()
    for person_dir, review_dir, source in iter_review_images(people_dir):
        bucket = "photos/nude" if is_nudity_review(source, review_dir) else "photos"
        dest_dir = person_dir / "photos" / "nude" if bucket == "photos/nude" else person_dir / "photos"
        dest = dest_dir / cleaned_name(source)
        while dest in reserved:
            dest = unique_dest(dest)
        dest = unique_dest(dest)
        reserved.add(dest)
        plan.append(MovePlan(source=source, dest=dest, person=person_dir.name, bucket=bucket))
    return plan


def remove_empty_dirs(root: Path, *, apply: bool) -> int:
    if not root.exists():
        return 0
    removed = 0
    for path in sorted([p for p in root.rglob("*") if p.is_dir()], key=lambda p: len(p.parts), reverse=True):
        try:
            next(path.iterdir())
        except StopIteration:
            if apply:
                path.rmdir()
            removed += 1
    try:
        next(root.iterdir())
    except StopIteration:
        if apply:
            root.rmdir()
        removed += 1
    except FileNotFoundError:
        pass
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-dir", type=Path, default=PEOPLE)
    parser.add_argument("--apply", action="store_true", help="Move files. Default is dry-run.")
    parser.add_argument("--promote-manifest", action="store_true",
                        help="After a successful apply, promote current photos originals into the manifest.")
    parser.add_argument("--keep-empty-review-dirs", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    people_dir = args.people_dir.expanduser().resolve()
    sorted_root = people_dir.parent

    validation = source_manifest.validate_current(
        label="merge_review_to_photos_before",
        people_dir=people_dir,
    )
    source_manifest.print_validation(validation)
    if not validation.ok:
        return source_manifest.SOURCE_GUARD_EXIT

    plan = build_plan(people_dir)
    by_bucket = Counter(item.bucket for item in plan)
    by_person = Counter(item.person for item in plan)
    collisions = sum(1 for item in plan if "__reviewmerge" in item.dest.name)

    print("Review Merge To Photos")
    print("=" * 52)
    print(f"People folder:       {people_dir}")
    print(f"Files to move:       {len(plan)}")
    print(f"To photos:           {by_bucket.get('photos', 0)}")
    print(f"To photos/nude:      {by_bucket.get('photos/nude', 0)}")
    print(f"Filename collisions: {collisions}")
    print(f"People affected:     {len(by_person)}")
    print()
    for person, count in by_person.most_common(20):
        print(f"  {person}: {count}")
    if len(by_person) > 20:
        print(f"  ... {len(by_person) - 20} more")
    print()

    if not args.quiet:
        for item in plan[:80]:
            print(f"move: {item.source}")
            print(f"  ->  {item.dest}")
        if len(plan) > 80:
            print(f"... and {len(plan) - 80} more")
        print()

    if not args.apply:
        print("DRY-RUN - no files moved. Re-run with --apply to commit.")
        return 0

    moved = 0
    for item in plan:
        if not item.source.exists():
            continue
        operation_ledger.move_path(
            item.source,
            item.dest,
            sorted_root=sorted_root,
            operation="merge_review_to_photos.move_review_image",
            reason="merge manually reviewed person-folder image back into photos",
            extra={"bucket": item.bucket},
        )
        moved += 1

    empty_removed = 0
    if not args.keep_empty_review_dirs:
        for person_dir in sorted(people_dir.iterdir(), key=lambda p: p.name.casefold()):
            review_dir = person_dir / "review"
            if review_dir.is_dir():
                empty_removed += remove_empty_dirs(review_dir, apply=True)

    after = source_manifest.validate_current(
        label="merge_review_to_photos_after",
        people_dir=people_dir,
    )
    source_manifest.print_validation(after)
    if not after.ok:
        print("ERROR: merge completed but manifest validation failed; not promoting manifest.")
        return source_manifest.SOURCE_GUARD_EXIT

    manifest_path = None
    if args.promote_manifest:
        manifest_path = source_manifest.promote_current(
            label="merge_review_to_photos_promote",
            reason=f"merged {moved} reviewed images back into photos",
            people_dir=people_dir,
        )

    print()
    print(f"Moved files:          {moved}")
    print(f"Empty dirs removed:   {empty_removed}")
    if manifest_path:
        print(f"Manifest promoted:    {manifest_path}")
    else:
        print("Manifest promoted:    no")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

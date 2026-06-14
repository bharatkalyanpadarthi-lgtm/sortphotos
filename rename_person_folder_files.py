#!/usr/bin/env python3
"""
Rename images inside each person folder using a canonical name.

Example:
  photos_by_person/Sonali Bendre/IMG_1234.jpg
  -> photos_by_person/Sonali Bendre/Sonali_Bendre_0001_photo_portrait_q_high.jpg

Simple mode:
  photos_by_person/Sonali Bendre/photos/Sonali_Bendre_0001_photo_portrait_q_high.jpg
  -> photos_by_person/Sonali Bendre/photos/Sonali_Bendre_00001.jpg

Default is dry-run. Use --apply to rename files.

Generated hardlink views and review holding areas are intentionally skipped.
The renamer should only touch canonical originals under photos/; generated
albums are rebuilt from those sources and can contain tens of thousands of
hardlinks.
"""

from __future__ import annotations

import argparse
import csv
import contextlib
import os
import re
import sys
import time
from pathlib import Path

import source_guard
import source_manifest

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp",
              ".tif", ".tiff", ".heic", ".heif", ".gif"}
DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
SKIP_DIRS = {
    "all",
    "_smart_albums",
    "_smart_albums_v2",
    "_smart_albums_simple_preview",
}
SIMPLE_WIDTH = 5
SOURCE_GUARD_EXIT = 3
SOURCE_TOP_DIRS = {
    "photos",
    "photos_nude",
    "_possible_nudity",
    "_uncertain_nudity",
}
CATEGORY_RANK = {
    "photo": 0,
    "nudity_possible": 1,
    "nudity_uncertain": 2,
    "near_visual_review": 3,
    "duplicate_review": 4,
    "review": 5,
}


class ImageMeta:
    def __init__(self, width: int = 0, height: int = 0, sharpness: float = 0.0) -> None:
        self.width = width
        self.height = height
        self.sharpness = sharpness


@contextlib.contextmanager
def suppress_native_stderr(enabled: bool = True):
    """Hide noisy libjpeg/libpng warnings emitted while reading recovered files."""
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


def iter_images(person_dir: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(person_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        base = Path(dirpath)
        rel = base.relative_to(person_dir)
        if rel.parts:
            top = rel.parts[0]
            if top in SKIP_DIRS or top not in SOURCE_TOP_DIRS:
                dirnames[:] = []
                continue
        else:
            dirnames[:] = [
                d for d in dirnames
                if d in SOURCE_TOP_DIRS and d not in SKIP_DIRS and not d.startswith(".")
            ]
        for filename in filenames:
            p = base / filename
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                out.append(p)
    return sorted(
        out,
        key=lambda p: (
            len(p.relative_to(person_dir).parts),
            str(p.relative_to(person_dir)).lower(),
        ),
    )


def clean_prefix(folder_name: str) -> str:
    name = folder_name.strip()
    name = re.sub(r"\s+", "_", name)
    name = re.sub(r"[/:\\]+", "_", name)
    name = re.sub(r"[^A-Za-z0-9._ -]+", "_", name)
    name = re.sub(r"_+", "_", name).strip("._- ")
    return name or "person"


def read_image_meta(path: Path) -> ImageMeta:
    with suppress_native_stderr():
        try:
            import cv2
            import numpy as np

            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None:
                h, w = img.shape[:2]
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
                sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                return ImageMeta(w, h, sharp)
        except Exception:
            pass
        try:
            import cv2
            import numpy as np
            from PIL import Image, ImageFile

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(path) as im:
                w, h = im.size
                frame = im.convert("RGB")
                arr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)
                gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
                sharp = float(cv2.Laplacian(gray, cv2.CV_64F).var())
                return ImageMeta(w, h, sharp)
        except Exception:
            return ImageMeta()


def orientation_label(meta: ImageMeta) -> str:
    if meta.width <= 0 or meta.height <= 0:
        return "unknown"
    ratio = meta.width / max(1, meta.height)
    if ratio > 1.2:
        return "landscape"
    if ratio < 0.82:
        return "portrait"
    return "square"


def quality_label(meta: ImageMeta) -> str:
    if meta.width <= 0 or meta.height <= 0:
        return "q_unknown"
    megapixels = (meta.width * meta.height) / 1_000_000.0
    if megapixels >= 1.0 and meta.sharpness >= 90.0:
        return "q_high"
    if megapixels >= 0.45 and meta.sharpness >= 25.0:
        return "q_good"
    return "q_review"


def category_label(rel: Path) -> str:
    if not rel.parts:
        return "photo"
    first = rel.parts[0]
    if len(rel.parts) >= 2 and first == "photos" and rel.parts[1] == "nude":
        return "nudity_possible"
    if first in {"photos_nude", "_possible_nudity"}:
        return "nudity_possible"
    if first == "_uncertain_nudity":
        return "nudity_uncertain"
    if first == "_near_visual_review":
        return "near_visual_review"
    if first == "_duplicates":
        return "duplicate_review"
    if first == "review" and len(rel.parts) > 1:
        if rel.parts[1] == "duplicates":
            return "duplicate_review"
        if rel.parts[1] == "near_visual":
            return "near_visual_review"
        if rel.parts[1] == "nudity_possible":
            return "nudity_possible"
        if rel.parts[1] == "uncertain_nudity":
            return "nudity_uncertain"
        return "review"
    if first == "photos":
        return "photo"
    if first.startswith("_"):
        return "review"
    return "photo"


def canonical_pattern(prefix: str) -> re.Pattern:
    return re.compile(
        rf"^{re.escape(prefix)}_(\d+)(?:_[A-Za-z0-9]+(?:_[A-Za-z0-9]+)*)?$",
        re.IGNORECASE,
    )


def stable_canonical_pattern(prefix: str, category: str) -> re.Pattern:
    return re.compile(
        rf"^{re.escape(prefix)}_(\d+)_"
        rf"{re.escape(category)}_"
        r"(?:landscape|portrait|square|unknown)_"
        r"(?:q_high|q_good|q_review|q_unknown)$",
        re.IGNORECASE,
    )


def parsed_index(path: Path, prefix: str) -> int | None:
    match = canonical_pattern(prefix).fullmatch(path.stem)
    if not match:
        return None
    try:
        return int(match.group(1))
    except ValueError:
        return None


def is_stable_canonical(path: Path, prefix: str, category: str) -> bool:
    return stable_canonical_pattern(prefix, category).fullmatch(path.stem) is not None


def sort_key_for_image(path: Path, person_dir: Path, prefix: str) -> tuple:
    rel = path.relative_to(person_dir)
    category = category_label(rel)
    index = parsed_index(path, prefix)
    return (
        CATEGORY_RANK.get(category, 9),
        0 if index is not None else 1,
        index if index is not None else 999_999,
        len(rel.parts),
        str(rel).lower(),
    )


def smart_filename(prefix: str, index: int, meta: ImageMeta, category: str,
                   width: int, suffix: str) -> str:
    return (
        f"{prefix}_{index:0{width}d}_"
        f"{category}_{orientation_label(meta)}_{quality_label(meta)}"
        f"{suffix.lower()}"
    )


def simple_filename(prefix: str, index: int, suffix: str, copy_number: int | None = None) -> str:
    copy_part = f"_copy{copy_number}" if copy_number is not None else ""
    return f"{prefix}_{index:0{SIMPLE_WIDTH}d}{copy_part}{suffix}"


def iter_simple_images(person_dir: Path) -> list[Path]:
    images: list[Path] = []
    for path in iter_images(person_dir):
        rel = path.relative_to(person_dir)
        if rel.parts and rel.parts[0] == "photos":
            images.append(path)
    return images


def simple_bucket_dir(person_dir: Path, src: Path) -> Path:
    rel = src.relative_to(person_dir)
    if len(rel.parts) >= 2 and rel.parts[0] == "photos" and rel.parts[1] == "nude":
        return person_dir / "photos" / "nude"
    return person_dir / "photos"


def path_key(path: Path) -> str:
    return path.as_posix().casefold()


def same_existing_path(left: Path, right: Path) -> bool:
    if left == right:
        return True
    try:
        return left.exists() and right.exists() and os.path.samefile(left, right)
    except OSError:
        return False


def classify_simple_action(src: Path, dest: Path) -> str:
    if src.parent == dest.parent:
        return "rename"
    if src.name == dest.name:
        return "move"
    return "move_rename"


def next_unassigned_index(used_indices: set[int], start: int) -> int:
    candidate = max(1, start)
    while candidate in used_indices:
        candidate += 1
    used_indices.add(candidate)
    return candidate


def plan_for_person_simple(person_dir: Path) -> list[dict[str, object]]:
    images = iter_simple_images(person_dir)
    if not images:
        return []

    prefix = clean_prefix(person_dir.name)
    sorted_images = sorted(images, key=lambda p: sort_key_for_image(p, person_dir, prefix))
    parsed_indices = [i for i in (parsed_index(src, prefix) for src in sorted_images) if i is not None]
    used_indices: set[int] = set(parsed_indices)
    next_index = max(parsed_indices, default=0) + 1
    used_target_keys: set[str] = set()
    actions: list[dict[str, object]] = []

    for src in sorted_images:
        rel = src.relative_to(person_dir)
        existing_index = parsed_index(src, prefix)
        assigned_index = existing_index
        assigned_reason = "preserved"
        if assigned_index is None:
            assigned_index = next_unassigned_index(used_indices, next_index)
            next_index = assigned_index + 1
            assigned_reason = "assigned"

        target_dir = simple_bucket_dir(person_dir, src)
        copy_number: int | None = None
        dest = target_dir / simple_filename(prefix, assigned_index, src.suffix)
        dest_key = path_key(dest)
        if dest_key in used_target_keys and not same_existing_path(src, dest):
            copy_number = 2
            while True:
                candidate = target_dir / simple_filename(prefix, assigned_index, src.suffix, copy_number)
                candidate_key = path_key(candidate)
                if candidate_key not in used_target_keys or same_existing_path(src, candidate):
                    dest = candidate
                    dest_key = candidate_key
                    break
                copy_number += 1
        used_target_keys.add(dest_key)

        if same_existing_path(src, dest):
            continue

        actions.append({
            "person": person_dir.name,
            "source": src,
            "destination": dest,
            "source_relative": rel.as_posix(),
            "destination_relative": dest.relative_to(person_dir).as_posix(),
            "index": int(assigned_index),
            "index_source": assigned_reason,
            "copy_number": copy_number or "",
            "action": classify_simple_action(src, dest),
        })
    return actions


def plan_for_person(person_dir: Path, compact: bool = False) -> list[tuple[Path, Path]]:
    images = iter_images(person_dir)
    if not images:
        return []

    prefix = clean_prefix(person_dir.name)
    width = max(4, len(str(len(images))))
    planned: list[tuple[Path, Path]] = []
    used_paths: set[Path] = set()
    used_indices: set[int] = set()
    next_index = 1

    sorted_images = sorted(images, key=lambda p: sort_key_for_image(p, person_dir, prefix))
    if not compact:
        existing_indices = [i for i in (parsed_index(src, prefix) for src in sorted_images) if i is not None]
        next_index = max(existing_indices, default=0) + 1

    for ordinal, src in enumerate(sorted_images, start=1):
        rel = src.relative_to(person_dir)
        category = category_label(rel)
        index = ordinal if compact else parsed_index(src, prefix)
        if (
            not compact
            and index is not None
            and index not in used_indices
            and is_stable_canonical(src, prefix, category)
        ):
            used_indices.add(index)
            used_paths.add(src)
            continue
        meta = read_image_meta(src)
        if index is None or index in used_indices:
            while next_index in used_indices:
                next_index += 1
            index = next_index
            used_indices.add(index)
            next_index += 1
        else:
            used_indices.add(index)
        dest = src.parent / smart_filename(prefix, index, meta, category, width, src.suffix)
        if dest in used_paths:
            raise RuntimeError(f"duplicate planned path: {dest}")
        used_paths.add(dest)
        if src != dest:
            planned.append((src, dest))
    return planned


def temp_path(src: Path, i: int) -> Path:
    return src.with_name(f".rename_tmp_{os.getpid()}_{i}{src.suffix}")


def report_dir_for_people(people_dir: Path) -> Path:
    if people_dir.name == "photos_by_person":
        return people_dir.parent / "_source_review" / "rename_reports"
    return people_dir / "_rename_reports"


def default_report_csv(people_dir: Path, mode: str) -> Path:
    stamp = time.strftime("%Y%m%d_%H%M%S")
    return report_dir_for_people(people_dir) / f"{mode}_person_filename_plan_{stamp}.csv"


def write_simple_plan_csv(path: Path, actions: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "person",
        "action",
        "index",
        "index_source",
        "copy_number",
        "source",
        "destination",
        "source_relative",
        "destination_relative",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for action in actions:
            row = dict(action)
            row["source"] = str(row["source"])
            row["destination"] = str(row["destination"])
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def write_legacy_plan_csv(path: Path, actions: list[tuple[Path, Path]], people_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["person", "action", "source", "destination", "source_relative", "destination_relative"]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for src, dest in actions:
            person = ""
            source_relative = ""
            destination_relative = ""
            try:
                source_parts = src.relative_to(people_dir).parts
                if source_parts:
                    person = source_parts[0]
                source_relative = src.relative_to(people_dir).as_posix()
                destination_relative = dest.relative_to(people_dir).as_posix()
            except ValueError:
                pass
            writer.writerow({
                "person": person,
                "action": classify_simple_action(src, dest),
                "source": str(src),
                "destination": str(dest),
                "source_relative": source_relative,
                "destination_relative": destination_relative,
            })


def source_counts_ok(before: dict[str, int], people_dir: Path) -> bool:
    after = source_guard.person_original_counts(people_dir)
    violations = source_guard.source_count_violations(before, after)
    if violations:
        source_guard.print_violations(violations)
        return False
    source_guard.print_ok(before, after)
    return True


def apply_actions(actions: list[tuple[Path, Path]]) -> None:
    temp_actions: list[tuple[Path, Path, Path]] = []
    for i, (src, dest) in enumerate(actions, start=1):
        tmp = temp_path(src, i)
        if tmp.exists():
            raise RuntimeError(f"temporary path already exists: {tmp}")
        temp_actions.append((src, tmp, dest))

    for src, tmp, _dest in temp_actions:
        src.rename(tmp)

    try:
        for _src, tmp, dest in temp_actions:
            dest.parent.mkdir(parents=True, exist_ok=True)
            if dest.exists():
                raise RuntimeError(f"destination unexpectedly exists: {dest}")
            tmp.rename(dest)
    except Exception:
        print("ERROR: rename failed after temporary step; some files may have .rename_tmp names.")
        raise


def prune_empty_simple_dirs(person_dirs: list[Path]) -> int:
    removed = 0
    for person_dir in person_dirs:
        photos_dir = person_dir / "photos"
        if not photos_dir.exists():
            continue
        for dirpath, dirnames, _filenames in os.walk(photos_dir, topdown=False):
            current = Path(dirpath)
            if current == photos_dir or current == photos_dir / "nude":
                continue
            try:
                current.rmdir()
                removed += 1
            except OSError:
                continue
    return removed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("people_dir", nargs="?", default=str(DEFAULT_PEOPLE))
    parser.add_argument("--person", default=None,
                        help="Rename one person folder only.")
    parser.add_argument("--compact", action="store_true",
                        help="Renumber each person from 0001. Default preserves existing numbers.")
    parser.add_argument("--simple", action="store_true",
                        help="Use simple Person_00001.ext names and flatten nested photos/ originals.")
    parser.add_argument("--apply", action="store_true",
                        help="Rename files. Default is dry-run.")
    parser.add_argument("--report-csv", type=Path, default=None,
                        help="Write the rename plan CSV to this path. Default: _source_review/rename_reports.")
    parser.add_argument("--skip-manifest-promote", action="store_true",
                        help="With --apply, do not promote the source manifest after a clean rename.")
    parser.add_argument("--quiet", action="store_true",
                        help="Do not print sample rename paths.")
    args = parser.parse_args()

    people_dir = Path(args.people_dir).expanduser().resolve()
    if not people_dir.exists():
        print(f"ERROR: people folder not found: {people_dir}")
        return 1

    all_actions: list[tuple[Path, Path]] = []
    simple_actions: list[dict[str, object]] = []
    folder_count = 0
    image_count = 0
    person_dirs = [
        p for p in people_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")
    ]
    if args.person:
        wanted = args.person.casefold()
        person_dirs = [p for p in person_dirs if p.name.casefold() == wanted]
        if not person_dirs:
            print(f"ERROR: person folder not found: {args.person}")
            return 1

    for person_dir in sorted(person_dirs, key=lambda p: p.name.lower()):
        folder_count += 1
        images = iter_simple_images(person_dir) if args.simple else iter_images(person_dir)
        image_count += len(images)
        if args.simple:
            simple_actions.extend(plan_for_person_simple(person_dir))
        else:
            all_actions.extend(plan_for_person(person_dir, compact=args.compact))

    if args.simple:
        all_actions = [
            (action["source"], action["destination"])
            for action in simple_actions
            if isinstance(action["source"], Path) and isinstance(action["destination"], Path)
        ]

    report_csv = (args.report_csv.expanduser() if args.report_csv else
                  default_report_csv(people_dir, "simple" if args.simple else "smart"))
    if args.simple:
        write_simple_plan_csv(report_csv, simple_actions)
    else:
        write_legacy_plan_csv(report_csv, all_actions, people_dir)

    print(f"People folder:      {people_dir}")
    print(f"Person folders:     {folder_count}")
    print(f"Image files:        {image_count}")
    print(f"Files to rename:    {len(all_actions)}")
    print(f"Plan CSV:           {report_csv}")
    print()

    if not args.quiet:
        for src, dest in all_actions[:100]:
            print(f"rename: {src}")
            print(f"  ->    {dest}")
        if len(all_actions) > 100:
            print(f"... and {len(all_actions) - 100} more")
        print()

    if not args.apply:
        print("DRY-RUN — no files renamed. Re-run with --apply to commit.")
        return 0

    manifest_check = source_manifest.validate_current(
        label="rename_person_folder_files_start",
        people_dir=people_dir,
    )
    source_manifest.print_validation(manifest_check)
    if not manifest_check.ok:
        return SOURCE_GUARD_EXIT

    before_counts = source_guard.person_original_counts(people_dir)
    apply_actions(all_actions)
    removed_empty_dirs = prune_empty_simple_dirs(person_dirs) if args.simple else 0

    print(f"Renamed {len(all_actions)} file(s).")
    if removed_empty_dirs:
        print(f"Removed {removed_empty_dirs} empty flattened folder(s).")
    if not source_counts_ok(before_counts, people_dir):
        return SOURCE_GUARD_EXIT

    post_manifest_check = source_manifest.validate_current(
        label="rename_person_folder_files_after",
        people_dir=people_dir,
    )
    source_manifest.print_validation(post_manifest_check)
    if not post_manifest_check.ok:
        return SOURCE_GUARD_EXIT
    if not args.skip_manifest_promote:
        manifest_path = source_manifest.promote_current(
            label="rename_person_folder_files_completed",
            reason="successful person filename normalization",
            people_dir=people_dir,
        )
        print(f"Source manifest promoted: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

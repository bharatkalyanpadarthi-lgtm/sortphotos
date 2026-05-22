#!/usr/bin/env python3
"""
Recover still-available labeled originals from an older face cache.

This is intentionally conservative:
  - read-only by default
  - matches old source images by exact cached (mtime, size) signature
  - restores into current/canonical person folders
  - hardlinks when possible, otherwise copies
  - never deletes or overwrites any file
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import sort_photos

for _name in ("CacheState", "CachedFace", "FaceRecord", "LabelingState"):
    if hasattr(sort_photos, _name):
        setattr(sys.modules["__main__"], _name, getattr(sort_photos, _name))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic"}
CACHE_DIR = Path.home() / ".face_sort_cache"
DEFAULT_CACHE = CACHE_DIR / "cache.pkl.bak.mark_small_junk"
SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
PEOPLE_ROOT = SORTED / "photos_by_person"
RULES_FILE = Path(__file__).with_name("person_folder_rules.json")
REPORT_DIR = SORTED / "_source_review" / "recovery_reports"
DEFAULT_RECOVERED_BACKUP = Path("/Volumes/ssd 1/Photos Recovered/Photos & Videos  Backup/photo_source_review_backup")
DEFAULT_EXTERNAL_BACKUP = Path("/Volumes/Photos & Videos  Backup/photo_source_review_backup")


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


def can_decode_image(path: Path) -> bool:
    with suppress_native_stderr():
        try:
            import cv2
            import numpy as np

            data = np.fromfile(str(path), dtype=np.uint8)
            img = cv2.imdecode(data, cv2.IMREAD_UNCHANGED)
            if img is not None and getattr(img, "size", 0) > 0:
                return True
        except Exception:
            pass
        try:
            from PIL import Image, ImageFile

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(path) as im:
                im.load()
                return im.size[0] > 0 and im.size[1] > 0
        except Exception:
            return False


def load_rules() -> tuple[dict[str, str], set[str]]:
    if not RULES_FILE.exists():
        return {}, set()
    data = json.loads(RULES_FILE.read_text(encoding="utf-8"))
    aliases: dict[str, str] = {}
    for dest, sources in data.get("merge", {}).items():
        aliases[dest.casefold()] = dest
        for src in sources:
            aliases[src.casefold()] = dest
    for src, dest in data.get("rename", {}).items():
        aliases[src.casefold()] = dest
    removed = {x.casefold() for x in data.get("remove", [])}
    return aliases, removed


def current_people() -> dict[str, str]:
    if not PEOPLE_ROOT.exists():
        return {}
    return {p.name.casefold(): p.name for p in PEOPLE_ROOT.iterdir() if p.is_dir()}


def canonical_label(
    label: str,
    aliases: dict[str, str],
    removed: set[str],
    people: dict[str, str],
    create_missing_people: bool,
) -> str | None:
    key = label.casefold()
    if key in removed:
        return None
    mapped = aliases.get(key, label)
    if mapped.casefold() in removed:
        return None
    current = people.get(mapped.casefold())
    if current:
        return current
    if create_missing_people:
        return mapped
    return None


def image_files(root: Path):
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if not d.endswith(".photoslibrary")]
        for filename in filenames:
            path = Path(dirpath) / filename
            if path.suffix.lower() in IMAGE_EXTS:
                yield path


def index_candidates(
    roots: list[tuple[str, Path]],
    target_sigs: set[tuple[int, int]],
) -> tuple[dict[tuple[int, int], list[tuple[str, Path]]], dict[str, set[tuple[int, int]]], Counter, Counter, Counter]:
    by_sig: dict[tuple[int, int], list[tuple[str, Path]]] = defaultdict(list)
    existing_by_person: dict[str, set[tuple[int, int]]] = defaultdict(set)
    scanned_counts: Counter = Counter()
    matched_counts: Counter = Counter()
    rejected_counts: Counter = Counter()
    for root_name, root in roots:
        if not root.exists():
            continue
        print(f"  scanning {root_name}: {root}")
        for path in image_files(root):
            try:
                st = path.stat()
            except OSError:
                continue
            scanned_counts[root_name] += 1
            sig = (int(st.st_mtime), int(st.st_size))
            if sig not in target_sigs:
                continue
            if not can_decode_image(path):
                rejected_counts[root_name] += 1
                continue
            by_sig[sig].append((root_name, path))
            matched_counts[root_name] += 1
            if root_name == "people":
                try:
                    person = path.relative_to(PEOPLE_ROOT).parts[0]
                except (ValueError, IndexError):
                    continue
                existing_by_person[person].add(sig)
    return by_sig, existing_by_person, scanned_counts, matched_counts, rejected_counts


def load_cache(path: Path):
    with path.open("rb") as fh:
        return pickle.load(fh)


def choose_source(hits: list[tuple[str, Path]], target_person: str) -> tuple[str, Path] | None:
    ranked = {
        "recovered_backup": 0,
        "junk_to_review": 1,
        "source_review": 2,
        "external_backup": 3,
        "to_process": 4,
    }
    usable: list[tuple[int, str, Path]] = []
    for root_name, path in hits:
        if root_name == "people":
            continue
        usable.append((ranked.get(root_name, 99), root_name, path))
    if not usable:
        return None
    usable.sort(key=lambda x: (x[0], len(str(x[2]))))
    return usable[0][1], usable[0][2]


def unique_dest(folder: Path, filename: str) -> Path:
    stem = Path(filename).stem
    suffix = Path(filename).suffix
    dest = folder / filename
    i = 1
    while dest.exists():
        dest = folder / f"{stem}_{i:03d}{suffix}"
        i += 1
    return dest


def restore_file(src: Path, dest: Path, apply: bool) -> str:
    if not apply:
        return "dry_run"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
        return "hardlinked"
    except OSError:
        shutil.copy2(src, dest)
        return "copied"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--extra-root", action="append", type=Path, default=[],
                        help="Extra recovered/backup root to search. Can be passed more than once.")
    parser.add_argument("--no-default-recovered-backup", action="store_true",
                        help="Do not automatically search the recovered SSD backup path.")
    parser.add_argument("--existing-people-only", action="store_true",
                        help="Only restore labels that already have a current person folder.")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--report", type=Path, default=None)
    args = parser.parse_args()

    cache_path = args.cache.expanduser()
    if not cache_path.exists():
        print(f"ERROR: cache not found: {cache_path}")
        return 1

    aliases, removed = load_rules()
    people = current_people()
    if not people:
        print(f"ERROR: no current person folders found: {PEOPLE_ROOT}")
        return 1

    print(f"Loading old cache: {cache_path}")
    cache = load_cache(cache_path)

    labels_by_src: dict[str, set[str]] = defaultdict(set)
    skipped_label = Counter()
    for face in cache.faces:
        label = getattr(face, "label", None)
        if not label or label == "__junk__":
            continue
        canonical = canonical_label(
            label,
            aliases,
            removed,
            people,
            create_missing_people=not args.existing_people_only,
        )
        if not canonical:
            skipped_label[label] += 1
            continue
        labels_by_src[face.src_str].add(canonical)

    target_sigs = {
        (int(sig[0]), int(sig[1]))
        for old_src in labels_by_src
        if (sig := cache.file_signatures.get(old_src))
    }

    roots = [
        ("to_process", Path.home() / "Pictures" / "To Process"),
        ("people", PEOPLE_ROOT),
        ("source_review", SORTED / "_source_review"),
        ("junk_to_review", SORTED / "junk_to_review"),
        ("external_backup", DEFAULT_EXTERNAL_BACKUP),
    ]
    if not args.no_default_recovered_backup and DEFAULT_RECOVERED_BACKUP.exists():
        roots.insert(0, ("recovered_backup", DEFAULT_RECOVERED_BACKUP))
    for i, root in enumerate(args.extra_root, start=1):
        roots.insert(0, (f"extra_root_{i}", root.expanduser()))

    print("Indexing currently available images for old labeled signatures...")
    print(f"  target signatures: {len(target_sigs):,}")
    by_sig, existing_by_person, scanned_counts, matched_counts, rejected_counts = index_candidates(roots, target_sigs)
    for root_name, _root in roots:
        scanned = scanned_counts[root_name]
        matched = matched_counts[root_name]
        rejected = rejected_counts[root_name]
        if scanned or matched or rejected:
            print(f"  {root_name:<18} scanned={scanned:>8,} matched={matched:>8,} "
                  f"rejected_bad={rejected:>8,}")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = args.report or REPORT_DIR / "recover_labeled_sources_from_old_cache.csv"
    rows: list[dict[str, str]] = []
    totals = Counter()
    by_person = Counter()

    for old_src, labels in sorted(labels_by_src.items()):
        sig_raw = cache.file_signatures.get(old_src)
        if not sig_raw:
            totals["missing_signature"] += len(labels)
            continue
        sig = (int(sig_raw[0]), int(sig_raw[1]))
        hits = by_sig.get(sig, [])
        for person in sorted(labels):
            row = {
                "person": person,
                "old_source": old_src,
                "mtime": str(sig[0]),
                "size": str(sig[1]),
                "status": "",
                "source_found": "",
                "dest": "",
                "source_root": "",
            }
            if sig in existing_by_person.get(person, set()):
                totals["already_present"] += 1
                row["status"] = "already_present"
                rows.append(row)
                continue
            source_choice = choose_source(hits, person)
            if not source_choice:
                totals["missing"] += 1
                row["status"] = "missing"
                rows.append(row)
                continue
            source_root, source_path = source_choice
            photos_dir = PEOPLE_ROOT / person / "photos"
            dest = unique_dest(photos_dir, f"recovered__{source_path.name}")
            try:
                status = restore_file(source_path, dest, args.apply)
            except Exception as exc:
                totals["restore_error"] += 1
                row["status"] = f"error:{type(exc).__name__}:{exc}"
                row["source_found"] = str(source_path)
                row["source_root"] = source_root
                row["dest"] = str(dest)
                rows.append(row)
                continue
            totals[status] += 1
            by_person[person] += 1
            row["status"] = status
            row["source_found"] = str(source_path)
            row["source_root"] = source_root
            row["dest"] = str(dest)
            rows.append(row)

    with report.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=[
            "person", "status", "source_root", "source_found", "dest", "old_source", "mtime", "size",
        ])
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Recovery audit")
    print(f"  Old labeled source images: {len(labels_by_src):,}")
    print(f"  Already present:           {totals['already_present']:,}")
    print(f"  Restored / restorable:     {totals['hardlinked'] + totals['copied'] + totals['dry_run']:,}")
    print(f"  Missing:                   {totals['missing']:,}")
    print(f"  Restore errors:            {totals['restore_error']:,}")
    print(f"  Skipped removed/unknown labels: {sum(skipped_label.values()):,} face labels")
    print(f"  Report: {report}")
    if by_person:
        print()
        print("Top restored/restorable people")
        for person, count in by_person.most_common(20):
            print(f"  {person:<28} {count:>5}")
    if skipped_label:
        print()
        print("Top skipped labels")
        for label, count in skipped_label.most_common(20):
            print(f"  {label:<28} {count:>5}")
    if not args.apply:
        print()
        print("DRY-RUN only. Re-run with --apply to restore available files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

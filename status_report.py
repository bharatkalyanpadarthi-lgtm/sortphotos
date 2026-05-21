#!/usr/bin/env python3
"""
status_report.py — compact dashboard for the sorted photo pipeline.
"""

from __future__ import annotations

import csv
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sort_photos  # noqa: E402

for _name in ("CacheState", "CachedFace", "FaceRecord", "LabelingState", "IdentityDB"):
    if hasattr(sort_photos, _name):
        setattr(sys.modules["__main__"], _name, getattr(sort_photos, _name))

SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
PEOPLE = SORTED / "photos_by_person"
SOURCE_REVIEW = SORTED / "_source_review"
READY = SOURCE_REVIEW / "ready_to_delete"
ADV_REPORT = SOURCE_REVIEW / "duplicate_reports" / "advanced_duplicates.csv"
FINGERPRINT_CACHE = Path.home() / ".face_sort_cache" / "advanced_duplicate_fingerprints.json"
IDENTITY_DB = Path.home() / ".face_sort_cache" / "person_identity_db.pkl"
REFERENCE_DB = Path.home() / ".face_sort_cache" / "reference_centroids.pkl"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp",
              ".tif", ".tiff", ".heic", ".heif"}
EXCLUDED_DIRS = {"_duplicates", "_near_visual_review", "_smart_albums"}


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        if any(part in EXCLUDED_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            total += 1
    return total


def count_dirs(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.iterdir() if p.is_dir())


def size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        if any(part in EXCLUDED_DIRS for part in p.parts):
            continue
        if p.is_file():
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


def identity_count() -> int:
    if not IDENTITY_DB.exists():
        return 0
    try:
        with IDENTITY_DB.open("rb") as f:
            db = pickle.load(f)
        return len(getattr(db, "identities", {}))
    except Exception:
        return 0


def reference_count() -> int:
    if not REFERENCE_DB.exists():
        return 0
    try:
        with REFERENCE_DB.open("rb") as f:
            payload = pickle.load(f)
        return len(payload.get("names", []))
    except Exception:
        return 0


def labeling_summary() -> dict[str, int]:
    state = sort_photos.load_labeling_state()
    if state is None:
        return {
            "clusters": 0, "labeled": 0, "remaining": 0,
            "remaining_20": 0, "remaining_50": 0, "remaining_faces": 0,
        }
    sizes: dict[int, int] = defaultdict(int)
    for cid in state.cluster_ids:
        sizes[cid] += 1
    labeled = 0
    remaining: list[int] = []
    for cid, n in sizes.items():
        if cid == -1:
            continue
        label = state.name_map.get(cid, "")
        if label.startswith("person_"):
            remaining.append(n)
        else:
            labeled += 1
    return {
        "clusters": len([cid for cid in sizes if cid != -1]),
        "labeled": labeled,
        "remaining": len(remaining),
        "remaining_20": sum(1 for n in remaining if n >= 20),
        "remaining_50": sum(1 for n in remaining if n >= 50),
        "remaining_faces": sum(remaining),
    }


def advanced_report_summary() -> dict[str, int]:
    counts = {"exact_file": 0, "same_pixels": 0, "visually_similar": 0}
    if not ADV_REPORT.exists():
        return counts
    try:
        with ADV_REPORT.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                action = row.get("action")
                kind = row.get("type")
                if action in {"move", "review"} and kind in counts:
                    counts[kind] += 1
    except Exception:
        pass
    return counts


def fingerprint_count() -> int:
    if not FINGERPRINT_CACHE.exists():
        return 0
    try:
        with FINGERPRINT_CACHE.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return len(data.get("entries", {}))
    except Exception:
        return 0


def main() -> int:
    labels = labeling_summary()
    adv = advanced_report_summary()

    print("Photo Pipeline Status")
    print("=" * 60)
    print(f"Sorted folder:          {SORTED}")
    print(f"Person folders:         {count_dirs(PEOPLE)}")
    print(f"Organized images:       {count_images(PEOPLE)}")
    print(f"People data size:       {human_size(size_bytes(PEOPLE))}")
    print(f"Known identities DB:    {identity_count()} people")
    print(f"Reference identities:   {reference_count()} people")
    print()
    print("Pending labeling")
    print(f"  Remaining clusters:   {labels['remaining']} ({labels['remaining_faces']} faces)")
    print(f"  Big clusters >=20:    {labels['remaining_20']}")
    print(f"  Big clusters >=50:    {labels['remaining_50']}")
    print()
    print("Duplicates")
    print(f"  Exact-file pending:   {adv['exact_file']}")
    print(f"  Same-pixel pending:   {adv['same_pixels']}")
    print(f"  Near-visual review:   {adv['visually_similar']}")
    print(f"  Fingerprints cached:  {fingerprint_count()}")
    print(f"  Report:               {ADV_REPORT}")
    print()
    print("Review / delete holding areas")
    print(f"  ready_to_delete size: {human_size(size_bytes(READY))}")
    print(f"  _source_review size:  {human_size(size_bytes(SOURCE_REVIEW))}")
    print(f"  ready_to_delete path: {READY}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

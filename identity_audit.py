#!/usr/bin/env python3
"""
Audit the person identity DB against current photos_by_person folders.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sort_photos  # noqa: E402

for _name in ("IdentityDB",):
    if hasattr(sort_photos, _name):
        setattr(sys.modules["__main__"], _name, getattr(sort_photos, _name))

DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
DEFAULT_DB = Path.home() / ".face_sort_cache" / "person_identity_db.pkl"
DEFAULT_REPORT = (
    Path.home() / "Pictures" / "sorted_all_pictures" / "_source_review" / "identity_audit.csv"
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


def norm(name: str) -> str:
    return " ".join(name.strip().split()).lower()


def person_dirs(root: Path) -> list[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_")],
                  key=lambda p: p.name.lower())


def image_count(root: Path) -> int:
    total = 0
    for p in root.rglob("*"):
        if "_smart_albums" in p.parts:
            continue
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            total += 1
    return total


def load_identity_db(path: Path):
    if not path.exists():
        return None
    with path.open("rb") as f:
        return pickle.load(f)


def write_report(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["kind", "name", "detail", "count"])
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-dir", type=Path, default=DEFAULT_PEOPLE)
    parser.add_argument("--identity-db", type=Path, default=DEFAULT_DB)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--low-count", type=int, default=3)
    parser.add_argument("--fail-on-issues", action="store_true")
    args = parser.parse_args()

    people_dir = args.people_dir.expanduser().resolve()
    folders = person_dirs(people_dir)
    folder_by_norm = {norm(p.name): p for p in folders}

    db = load_identity_db(args.identity_db.expanduser().resolve())
    identities = getattr(db, "identities", {}) if db is not None else {}
    identity_by_norm = {norm(name): name for name in identities}

    stale = sorted([name for key, name in identity_by_norm.items() if key not in folder_by_norm],
                   key=str.lower)
    missing = sorted([p.name for key, p in folder_by_norm.items() if key not in identity_by_norm],
                     key=str.lower)
    low_counts = []
    for p in folders:
        count = image_count(p)
        if count < args.low_count:
            low_counts.append((p.name, count))

    rows: list[dict[str, str]] = []
    for name in stale:
        rows.append({"kind": "stale_identity", "name": name, "detail": "identity not in current folders", "count": ""})
    for name in missing:
        rows.append({"kind": "missing_identity", "name": name, "detail": "folder not in identity DB", "count": ""})
    for name, count in low_counts:
        rows.append({"kind": "low_source_count", "name": name, "detail": f"fewer than {args.low_count} images", "count": str(count)})
    write_report(args.report.expanduser().resolve(), rows)

    print("Identity Audit")
    print("=" * 60)
    print(f"People folder:          {people_dir}")
    print(f"Person folders:         {len(folders)}")
    print(f"Identity DB people:     {len(identities)}")
    print(f"Stale identities:       {len(stale)}")
    print(f"Folders missing in DB:  {len(missing)}")
    print(f"Low-source folders:     {len(low_counts)}")
    print(f"Report:                 {args.report.expanduser().resolve()}")
    if stale or missing:
        print()
        print("Tip: run `python face.py rebuild-id` after intentional folder renames/merges.")
    if args.fail_on_issues and (stale or missing):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Move NudeNet-flagged originals into subfolders inside each person folder.

Reads a nudity review CSV produced by separate_nudity_review.py and moves:
  photos_by_person/<person>/<file>
to:
  photos_by_person/<person>/photos/nude/<file>

Default is dry-run. Use --apply to move files.
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

DEFAULT_SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
SUPPORTED_POLICY_VERSIONS = {"2", "3", ""}


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = "".join(dest.suffixes)
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def latest_report(review_dir: Path) -> Path | None:
    reports = sorted(review_dir.glob("nudity_review_report_*.csv"),
                     key=lambda p: p.stat().st_mtime,
                     reverse=True)
    return reports[0] if reports else None


def target_subdir(category: str, confirm_possible: bool) -> str | None:
    if category in {"possible_nudity", "uncertain"}:
        return "photos/nude"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sorted-root", type=Path, default=DEFAULT_SORTED)
    parser.add_argument("--report", type=Path, default=None,
                        help="Nudity report CSV. Defaults to latest report in _nudity_review.")
    parser.add_argument("--apply", action="store_true",
                        help="Move flagged originals. Default is dry-run.")
    parser.add_argument("--remove-review-copies", action="store_true",
                        help="After moving originals, remove copied _nudity_review image folders.")
    parser.add_argument("--confirm-possible", action="store_true",
                        help="Deprecated; flagged rows always move to photos/nude.")
    parser.add_argument("--allow-legacy-report", action="store_true",
                        help="Accepted for old workflows; legacy reports are allowed by default.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    sorted_root = args.sorted_root.expanduser().resolve()
    people_root = sorted_root / "photos_by_person"
    review_dir = sorted_root / "_nudity_review"
    report = args.report.expanduser().resolve() if args.report else latest_report(review_dir)

    if not people_root.exists():
        print(f"ERROR: photos_by_person not found: {people_root}")
        return 1
    if not report or not report.exists():
        print(f"ERROR: nudity report CSV not found in: {review_dir}")
        return 1

    actions: list[tuple[Path, Path, str]] = []
    missing = 0
    skipped = 0
    skipped_legacy = 0

    with report.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            category = row.get("category", "")
            subdir = target_subdir(category, args.confirm_possible)
            if not subdir:
                skipped += 1
                continue
            policy_version = row.get("policy_version", "")
            if policy_version not in SUPPORTED_POLICY_VERSIONS and not args.allow_legacy_report:
                skipped += 1
                skipped_legacy += 1
                continue
            src = Path(row.get("source", "")).expanduser()
            if not src.exists():
                missing += 1
                continue
            try:
                rel = src.resolve().relative_to(people_root)
            except ValueError:
                skipped += 1
                continue
            parts = rel.parts
            if len(parts) < 2:
                skipped += 1
                continue
            if parts[1] in {"photos_nude", "_possible_nudity", "_uncertain_nudity"}:
                skipped += 1
                continue
            if len(parts) >= 3 and parts[1] == "photos" and parts[2] == "nude":
                skipped += 1
                continue
            if len(parts) >= 3 and parts[1] == "review" and parts[2] in {"nudity_possible", "uncertain_nudity"}:
                skipped += 1
                continue
            person_dir = people_root / parts[0]
            dest = unique_dest(person_dir / subdir / src.name)
            actions.append((src, dest, category))

    possible = sum(1 for _src, _dest, cat in actions if cat == "possible_nudity")
    uncertain = sum(1 for _src, _dest, cat in actions if cat == "uncertain")

    print(f"Report:                 {report}")
    print(f"Person folders:         {people_root}")
    print(f"Files to move:          {len(actions)}")
    print(f"  possible_nudity:      {possible}")
    print(f"  lower-confidence:     {uncertain}")
    print("Possible target:        photos/nude")
    print(f"Missing source files:   {missing}")
    print(f"Skipped report rows:    {skipped}")
    print(f"Legacy/old-policy rows: {skipped_legacy}")
    print()

    if skipped_legacy:
        print("Report rows with an unsupported policy version were ignored.")
        print()

    if not args.quiet:
        for src, dest, _category in actions[:80]:
            print(f"move: {src}")
            print(f"  ->  {dest}")
        if len(actions) > 80:
            print(f"... and {len(actions) - 80} more")
        print()

    if not args.apply:
        print("DRY-RUN — no files moved. Re-run with --apply to commit.")
        return 0

    moved = 0
    for src, dest, _category in actions:
        if not src.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
        moved += 1

    print(f"Moved {moved} file(s) into person-folder nudity subfolders.")

    if args.remove_review_copies:
        for child_name in ("possible_nudity", "uncertain"):
            child = review_dir / child_name
            if child.exists():
                shutil.rmtree(child)
                print(f"Removed copied review folder: {child}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

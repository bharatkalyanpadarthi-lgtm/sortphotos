"""One cached exact-duplicate report for daily ingest; never moves originals."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import advanced_duplicate_matching as matching
import analysis_index
import daily_inventory
import pipeline_paths
from file_operations import sync_directory


def refresh(root: Path, report: Path, people: set[str] | None, database: Path) -> dict:
    old_rows = []
    if people is not None and report.exists():
        with report.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not {"scope", "group_id", "file_path", "action", "type", "keeper_path", "size_bytes"}.issubset(reader.fieldnames or []):
                people = None
            else:
                old_rows = [row for row in reader if row["scope"] not in people]
                if any(not row.get("keeper_path") or not str(row.get("size_bytes", "")).isdigit()
                       or None in row for row in old_rows):
                    people, old_rows = None, []
    elif not report.exists():
        people = None
    if not root.is_dir():
        raise OSError(f"Organized destination unavailable: {root}")
    roots = [root / name for name in sorted(people) if (root / name).is_dir()] if people is not None else [root]
    infos = []
    with analysis_index.AnalysisIndex(database) as index:
        for scan_root in roots:
            for path in matching.iter_images(scan_root):
                digest = index.content_sha256(path)
                if not digest:
                    raise OSError(f"Cannot verify duplicate candidate: {path}")
                infos.append(matching.ImageInfo(path, path.relative_to(root).parts[0],
                    matching.nudity_status_for_path(path), path.stat().st_size, 0, 0, digest, None, None))
        hash_hits, hash_misses = index.hash_hits, index.hash_misses
    duplicates = matching.build_duplicates(infos, 5, False)
    report.parent.mkdir(parents=True, exist_ok=True)
    temporary = report.with_suffix(report.suffix + ".tmp")
    try:
        matching.write_report(temporary, {info.path: info for info in infos}, duplicates)
        with temporary.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            fields = reader.fieldnames
            rows = old_rows + list(reader)
        # Group numbers are report-local, including preserved unaffected people.
        groups = {}
        for row in rows:
            key = (row["scope"], row["group_id"], row["keeper_path"], row["type"])
            row["group_id"] = str(groups.setdefault(key, len(groups) + 1))
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            import os
            os.fsync(handle.fileno())
        temporary.replace(report)
        sync_directory(report.parent)
    finally:
        temporary.unlink(missing_ok=True)
    hardlinked = sum(row["action"] == "already_hardlinked" for row in rows)
    actionable = [row for row in rows if row["action"] == "move"]
    result = {"scanned": len(infos), "groups": len(groups), "already_hardlinked": hardlinked,
              "exact_candidates": len(actionable), "reclaimable_bytes": sum(int(row["size_bytes"]) for row in actionable),
              "hash_hits": hash_hits, "hash_misses": hash_misses}
    print(f"Images checked for exact duplicates: {len(infos)}")
    print(f"Indexed content hashes: {hash_hits} reused; {hash_misses} newly hashed")
    print(f"Already hardlinked entries: {hardlinked}")
    print(f"Exact-file candidates: {len(actionable)} (report only; no files moved)")
    summary = report.with_suffix(".summary.json")
    import source_manifest
    source_manifest.write_json_atomic(summary, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-dir", type=Path, default=pipeline_paths.PEOPLE_ROOT)
    parser.add_argument("--report", type=Path, default=matching.DEFAULT_REPORT)
    parser.add_argument("--people-file", type=Path)
    parser.add_argument("--database", type=Path, default=pipeline_paths.ANALYSIS_INDEX)
    args = parser.parse_args()
    refresh(args.people_dir, args.report, daily_inventory.read_people_file(args.people_file), args.database)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Maintain and validate the protected manifest of original person photos.

This is stronger than a per-run count guard. It records every canonical original
inside photos_by_person/<person>/photos/ and blocks later operations if a known
original disappears before cache or smart-album indexes are refreshed.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import operation_ledger

SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
PEOPLE = SORTED / "photos_by_person"
SOURCE_REVIEW = SORTED / "_source_review"
MANIFEST_DIR = SOURCE_REVIEW / "source_manifest"
MANIFEST_PATH = MANIFEST_DIR / "last_known_good_originals.json"
REPORT_DIR = MANIFEST_DIR / "reports"
RECOVERY_REPORT_DIR = MANIFEST_DIR / "recovery_reports"
READY_TO_DELETE = SOURCE_REVIEW / "ready_to_delete"
RECOVERY_CONFLICT_DIR = READY_TO_DELETE / "source_manifest_recovery_conflicts"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
SOURCE_GUARD_EXIT = 3


@dataclass
class ManifestValidation:
    ok: bool
    manifest_path: Path
    report_prefix: Path
    expected_total: int
    current_total: int
    expected_people: int
    current_people: int
    missing: list[dict[str, Any]]
    size_changed: list[dict[str, Any]]
    renamed: list[dict[str, Any]]
    extra: list[dict[str, Any]]
    missing_csv: Path
    changed_csv: Path
    renamed_csv: Path
    extra_csv: Path


def is_person_dir_name(name: str) -> bool:
    return bool(name) and not name.startswith("_") and not name.startswith(".")


def is_manifest_original_entry(entry: dict[str, Any]) -> bool:
    person = str(entry.get("person", ""))
    rel = str(entry.get("relative_path", ""))
    first = Path(rel).parts[0] if rel else person
    return is_person_dir_name(person) and is_person_dir_name(first)


def run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("_") or "source_manifest"


def image_files(people_dir: Path = PEOPLE) -> list[Path]:
    if not people_dir.exists():
        return []
    files: list[Path] = []
    for person_dir in people_dir.iterdir():
        if not person_dir.is_dir() or not is_person_dir_name(person_dir.name):
            continue
        photos_dir = person_dir / "photos"
        if not photos_dir.exists():
            continue
        for path in photos_dir.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                files.append(path)
    return sorted(files, key=lambda p: str(p).lower())


def entry_for_path(path: Path, people_dir: Path = PEOPLE) -> dict[str, Any]:
    stat = path.stat()
    rel = path.relative_to(people_dir).as_posix()
    parts = Path(rel).parts
    person = parts[0] if parts else ""
    return {
        "relative_path": rel,
        "person": person,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def collect_entries(people_dir: Path = PEOPLE) -> list[dict[str, Any]]:
    return [entry_for_path(path, people_dir) for path in image_files(people_dir)]


def person_names(people_dir: Path = PEOPLE) -> list[str]:
    if not people_dir.exists():
        return []
    return sorted(
        [p.name for p in people_dir.iterdir() if p.is_dir() and is_person_dir_name(p.name)],
        key=str.lower,
    )


def person_counts(entries: list[dict[str, Any]], people_dir: Path | None = None) -> dict[str, int]:
    counts: dict[str, int] = {name: 0 for name in person_names(people_dir)} if people_dir is not None else {}
    for entry in entries:
        person = str(entry["person"])
        counts[person] = counts.get(person, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: item[0].lower()))


def build_manifest(people_dir: Path = PEOPLE, *, reason: str = "") -> dict[str, Any]:
    entries = collect_entries(people_dir)
    counts = person_counts(entries, people_dir)
    now = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    return {
        "version": 1,
        "created_at": now,
        "updated_at": now,
        "reason": reason,
        "people_dir": str(people_dir),
        "image_exts": sorted(IMAGE_EXTS),
        "total": len(entries),
        "person_count": len(counts),
        "person_counts": counts,
        "files": entries,
    }


def write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    tmp.replace(path)


def load_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_manifest(manifest: dict[str, Any], path: Path = MANIFEST_PATH) -> None:
    existing = load_manifest(path)
    if existing and existing.get("created_at"):
        manifest["created_at"] = existing["created_at"]
    manifest["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    manifest["total"] = len(manifest.get("files", []))
    people_dir = Path(str(manifest.get("people_dir") or PEOPLE)).expanduser()
    manifest["person_counts"] = person_counts(list(manifest.get("files", [])), people_dir)
    manifest["person_count"] = len(manifest["person_counts"])
    write_json_atomic(path, manifest)


def signature(entry: dict[str, Any]) -> tuple[str, int, int]:
    return (str(entry.get("person", "")), int(entry.get("size", 0)), int(entry.get("mtime_ns", 0)))


def compare_manifest(manifest: dict[str, Any],
                     current_entries: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    expected_entries = [
        entry for entry in list(manifest.get("files", []))
        if is_manifest_original_entry(entry)
    ]
    current_by_path = {str(entry["relative_path"]): entry for entry in current_entries}
    matched_current_paths: set[str] = set()
    missing_candidates: list[dict[str, Any]] = []
    size_changed: list[dict[str, Any]] = []

    for expected in expected_entries:
        rel = str(expected["relative_path"])
        current = current_by_path.get(rel)
        if current is None:
            missing_candidates.append(expected)
            continue
        if int(current.get("size", 0)) != int(expected.get("size", 0)):
            size_changed.append({
                "person": expected.get("person", ""),
                "relative_path": rel,
                "expected_size": int(expected.get("size", 0)),
                "current_size": int(current.get("size", 0)),
                "expected_mtime_ns": int(expected.get("mtime_ns", 0)),
                "current_mtime_ns": int(current.get("mtime_ns", 0)),
            })
            continue
        matched_current_paths.add(rel)

    current_by_signature: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
    for current in current_entries:
        rel = str(current["relative_path"])
        if rel in matched_current_paths:
            continue
        current_by_signature.setdefault(signature(current), []).append(current)

    missing: list[dict[str, Any]] = []
    renamed: list[dict[str, Any]] = []
    for expected in missing_candidates:
        candidates = current_by_signature.get(signature(expected), [])
        if candidates:
            current = candidates.pop(0)
            matched_current_paths.add(str(current["relative_path"]))
            renamed.append({
                "person": expected.get("person", ""),
                "old_relative_path": expected.get("relative_path", ""),
                "current_relative_path": current.get("relative_path", ""),
                "size": int(expected.get("size", 0)),
                "mtime_ns": int(expected.get("mtime_ns", 0)),
            })
        else:
            missing.append({
                "person": expected.get("person", ""),
                "relative_path": expected.get("relative_path", ""),
                "size": int(expected.get("size", 0)),
                "mtime_ns": int(expected.get("mtime_ns", 0)),
            })

    extra = [
        {
            "person": entry.get("person", ""),
            "relative_path": entry.get("relative_path", ""),
            "size": int(entry.get("size", 0)),
            "mtime_ns": int(entry.get("mtime_ns", 0)),
        }
        for entry in current_entries
        if str(entry["relative_path"]) not in matched_current_paths
    ]

    return {
        "missing": missing,
        "size_changed": size_changed,
        "renamed": renamed,
        "extra": extra,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["status"])
        writer.writeheader()
        writer.writerows(rows)


def validate_current(*,
                     label: str = "source_manifest_validate",
                     people_dir: Path = PEOPLE,
                     manifest_path: Path = MANIFEST_PATH,
                     report_dir: Path = REPORT_DIR) -> ManifestValidation:
    manifest = load_manifest(manifest_path)
    rid = run_id()
    report_prefix = report_dir / f"{slugify(label)}_{rid}"
    missing_csv = report_prefix.with_name(report_prefix.name + "_missing_originals.csv")
    changed_csv = report_prefix.with_name(report_prefix.name + "_changed_originals.csv")
    renamed_csv = report_prefix.with_name(report_prefix.name + "_renamed_originals.csv")
    extra_csv = report_prefix.with_name(report_prefix.name + "_extra_originals.csv")

    current_entries = collect_entries(people_dir)
    if manifest is None:
        missing = [{
            "error": "source manifest is missing",
            "manifest_path": str(manifest_path),
            "current_total": len(current_entries),
        }]
        write_csv(missing_csv, missing)
        return ManifestValidation(
            ok=False,
            manifest_path=manifest_path,
            report_prefix=report_prefix,
            expected_total=0,
            current_total=len(current_entries),
            expected_people=0,
            current_people=len(person_counts(current_entries, people_dir)),
            missing=missing,
            size_changed=[],
            renamed=[],
            extra=[],
            missing_csv=missing_csv,
            changed_csv=changed_csv,
            renamed_csv=renamed_csv,
            extra_csv=extra_csv,
        )

    comparison = compare_manifest(manifest, current_entries)
    expected_entries = [
        entry for entry in list(manifest.get("files", []))
        if is_manifest_original_entry(entry)
    ]
    write_csv(missing_csv, comparison["missing"])
    write_csv(changed_csv, comparison["size_changed"])
    write_csv(renamed_csv, comparison["renamed"])
    write_csv(extra_csv, comparison["extra"])
    ok = not comparison["missing"] and not comparison["size_changed"]
    expected_people = len(person_counts(expected_entries, None))
    return ManifestValidation(
        ok=ok,
        manifest_path=manifest_path,
        report_prefix=report_prefix,
        expected_total=len(expected_entries),
        current_total=len(current_entries),
        expected_people=expected_people,
        current_people=len(person_counts(current_entries, people_dir)),
        missing=comparison["missing"],
        size_changed=comparison["size_changed"],
        renamed=comparison["renamed"],
        extra=comparison["extra"],
        missing_csv=missing_csv,
        changed_csv=changed_csv,
        renamed_csv=renamed_csv,
        extra_csv=extra_csv,
    )


def print_validation(result: ManifestValidation) -> None:
    if result.ok:
        print(
            "Source manifest OK: "
            f"{result.expected_total} protected originals, {result.current_total} current originals, "
            f"{len(result.extra)} new, {len(result.renamed)} renamed."
        )
        return
    print("ERROR: protected source manifest check failed.", file=sys.stderr)
    print(f"Manifest: {result.manifest_path}", file=sys.stderr)
    print(f"Missing originals: {len(result.missing)}", file=sys.stderr)
    print(f"Changed originals: {len(result.size_changed)}", file=sys.stderr)
    print(f"Missing report: {result.missing_csv}", file=sys.stderr)
    print(f"Changed report: {result.changed_csv}", file=sys.stderr)
    for row in result.missing[:10]:
        print(f"  MISSING {row.get('relative_path', row)}", file=sys.stderr)
    for row in result.size_changed[:10]:
        print(
            f"  CHANGED {row.get('relative_path')}: "
            f"{row.get('expected_size')} -> {row.get('current_size')}",
            file=sys.stderr,
        )


def promote_current(*,
                    label: str = "source_manifest_promote",
                    reason: str = "",
                    people_dir: Path = PEOPLE,
                    manifest_path: Path = MANIFEST_PATH) -> Path:
    manifest = build_manifest(people_dir, reason=reason or label)
    save_manifest(manifest, manifest_path)
    return manifest_path


def sorted_root_for_people(people_dir: Path) -> Path:
    people_dir = people_dir.expanduser().resolve()
    if people_dir.name == "photos_by_person":
        return people_dir.parent
    return SORTED


def unique_path(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = "".join(dest.suffixes)
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__restore{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def latest_failed_run_report(sorted_root: Path) -> Path | None:
    roots = [
        sorted_root / "_source_review" / "daily_run_summaries",
        sorted_root / "_source_review" / "source_count_guards",
    ]
    candidates: list[Path] = []
    for root in roots:
        if root.exists():
            candidates.extend(root.glob("*_source_count_violations.csv"))
    non_empty: list[Path] = []
    for path in candidates:
        try:
            with path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            if rows:
                non_empty.append(path)
        except Exception:
            continue
    if not non_empty:
        return None
    return max(non_empty, key=lambda p: p.stat().st_mtime)


def iter_candidate_files(search_roots: list[Path]) -> list[Path]:
    out: list[Path] = []
    for root in search_roots:
        root = root.expanduser()
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                out.append(path)
    return out


def build_candidate_index(search_roots: list[Path]) -> dict[tuple[str, int], list[Path]]:
    index: dict[tuple[str, int], list[Path]] = {}
    for path in iter_candidate_files(search_roots):
        try:
            size = path.stat().st_size
        except OSError:
            continue
        index.setdefault((path.name.casefold(), int(size)), []).append(path)
    for paths in index.values():
        paths.sort(key=lambda p: str(p).casefold())
    return index


def completed_ledger_destinations(sorted_root: Path) -> dict[str, list[Path]]:
    by_relative: dict[str, list[Path]] = {}
    for event in operation_ledger.iter_events(sorted_root):
        if event.get("status") != "moved":
            continue
        rel = str(event.get("source_relative_to_people") or "")
        dest = Path(str(event.get("dest_path") or ""))
        if rel and dest.exists() and dest.is_file():
            by_relative.setdefault(rel, []).append(dest)
    return by_relative


def candidate_score(candidate: Path, entry: dict[str, Any], rel: str) -> int | None:
    try:
        stat = candidate.stat()
    except OSError:
        return None
    if int(stat.st_size) != int(entry.get("size", -1)):
        return None
    score = 1000
    candidate_posix = candidate.as_posix()
    if candidate_posix.endswith("/" + rel):
        score -= 500
    if candidate.name.casefold() == Path(rel).name.casefold():
        score -= 200
    if int(stat.st_mtime_ns) == int(entry.get("mtime_ns", -1)):
        score -= 100
    if "/ready_to_delete/" in candidate_posix:
        score -= 50
    if "/person_folder_duplicates/" in candidate_posix:
        score -= 40
    return score


def find_restore_candidate(entry: dict[str, Any],
                           *,
                           candidate_index: dict[tuple[str, int], list[Path]],
                           ledger_by_relative: dict[str, list[Path]]) -> tuple[Path | None, str]:
    rel = str(entry.get("relative_path") or "")
    size = int(entry.get("size", -1))
    name = Path(rel).name
    candidates: list[tuple[int, Path, str]] = []
    for path in ledger_by_relative.get(rel, []):
        score = candidate_score(path, entry, rel)
        if score is not None:
            candidates.append((score - 300, path, "operation_ledger"))
    for path in candidate_index.get((name.casefold(), size), []):
        score = candidate_score(path, entry, rel)
        if score is not None:
            candidates.append((score, path, "ready_to_delete_search"))
    if not candidates:
        return None, ""
    candidates.sort(key=lambda item: (item[0], str(item[1]).casefold()))
    return candidates[0][1], candidates[0][2]


def restore_file(candidate: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(candidate), str(target))
        return "hardlink"
    except OSError:
        shutil.copy2(str(candidate), str(target))
        return "copy2"


def write_restore_report(path: Path, rows: list[dict[str, Any]]) -> None:
    write_csv(path, rows)


def restore_from_manifest(*,
                          people_dir: Path,
                          manifest_path: Path,
                          search_roots: list[Path],
                          conflict_dir: Path,
                          report_dir: Path,
                          label: str,
                          apply: bool,
                          restore_changed: bool = True,
                          last_failed_run: bool = False) -> tuple[bool, Path, list[dict[str, Any]]]:
    sorted_root = sorted_root_for_people(people_dir)
    result = validate_current(
        label=f"{label}_before_restore",
        people_dir=people_dir,
        manifest_path=manifest_path,
        report_dir=report_dir,
    )
    targets: list[dict[str, Any]] = []
    for row in result.missing:
        item = dict(row)
        item["restore_kind"] = "missing"
        targets.append(item)
    if restore_changed:
        for row in result.size_changed:
            item = {
                "person": row.get("person", ""),
                "relative_path": row.get("relative_path", ""),
                "size": int(row.get("expected_size", 0)),
                "mtime_ns": int(row.get("expected_mtime_ns", 0)),
                "current_size": int(row.get("current_size", 0)),
                "current_mtime_ns": int(row.get("current_mtime_ns", 0)),
                "restore_kind": "changed",
            }
            targets.append(item)

    rid = run_id()
    report_path = report_dir / f"{slugify(label)}_{rid}_restore_report.csv"
    rows: list[dict[str, Any]] = []
    failed_report = latest_failed_run_report(sorted_root) if last_failed_run else None

    if not targets:
        rows.append({
            "status": "nothing_to_restore",
            "detail": "manifest has no missing or changed protected originals",
            "last_failed_run_report": str(failed_report or ""),
        })
        write_restore_report(report_path, rows)
        return True, report_path, rows

    candidate_index = build_candidate_index(search_roots)
    ledger_by_relative = completed_ledger_destinations(sorted_root)
    conflict_root = conflict_dir / rid

    for entry in targets:
        rel = str(entry.get("relative_path") or "")
        target = people_dir / rel
        candidate, source = find_restore_candidate(
            entry,
            candidate_index=candidate_index,
            ledger_by_relative=ledger_by_relative,
        )
        row: dict[str, Any] = {
            "status": "planned",
            "restore_kind": entry.get("restore_kind", ""),
            "person": entry.get("person", ""),
            "relative_path": rel,
            "target_path": str(target),
            "candidate_path": str(candidate or ""),
            "candidate_source": source,
            "expected_size": int(entry.get("size", 0)),
            "last_failed_run_report": str(failed_report or ""),
            "conflict_path": "",
            "restore_method": "",
        }
        if candidate is None:
            row["status"] = "missing_candidate"
            rows.append(row)
            continue

        try:
            candidate_size = candidate.stat().st_size
        except OSError as exc:
            row["status"] = "candidate_unreadable"
            row["error"] = str(exc)
            rows.append(row)
            continue
        if int(candidate_size) != int(entry.get("size", -1)):
            row["status"] = "candidate_size_mismatch"
            row["candidate_size"] = int(candidate_size)
            rows.append(row)
            continue

        if not apply:
            rows.append(row)
            continue

        try:
            if target.exists():
                conflict = unique_path(conflict_root / rel)
                operation_ledger.move_path(
                    target,
                    conflict,
                    sorted_root=sorted_root,
                    operation="source_manifest.restore_conflict",
                    reason="move changed/current target aside before restoring manifest original",
                    extra={"relative_path": rel},
                )
                row["conflict_path"] = str(conflict)
            method = restore_file(candidate, target)
            row["restore_method"] = method
            row["status"] = "restored"
            operation_ledger.record_event(
                operation="source_manifest.restore",
                reason="restore protected original from ready_to_delete",
                status="restored",
                source=candidate,
                dest=target,
                sorted_root=sorted_root,
                source_meta=operation_ledger.metadata(candidate, hash_file=True),
                dest_meta=operation_ledger.metadata(target, hash_file=False),
                extra={"relative_path": rel, "restore_method": method},
            )
        except Exception as exc:  # noqa: BLE001
            row["status"] = "restore_failed"
            row["error"] = str(exc)
        rows.append(row)

    write_restore_report(report_path, rows)
    ok = not any(row["status"] in {
        "missing_candidate",
        "candidate_unreadable",
        "candidate_size_mismatch",
        "restore_failed",
    } for row in rows)
    if apply:
        post = validate_current(
            label=f"{label}_after_restore",
            people_dir=people_dir,
            manifest_path=manifest_path,
            report_dir=report_dir,
        )
        ok = ok and post.ok
    return ok, report_path, rows


def cmd_init(args: argparse.Namespace) -> int:
    manifest_path = args.manifest_path.expanduser()
    people_dir = args.people_dir.expanduser()
    if manifest_path.exists() and not args.replace:
        print(f"ERROR: manifest already exists: {manifest_path}", file=sys.stderr)
        print("Use promote after a clean validation, or init --replace only after manual review.", file=sys.stderr)
        return 2
    manifest = build_manifest(people_dir, reason=args.reason or "initial source manifest")
    print(f"Current originals: {manifest['total']} across {manifest['person_count']} person folders")
    if not args.apply:
        print("DRY-RUN - no manifest written. Re-run with --apply to write.")
        return 0
    save_manifest(manifest, manifest_path)
    print(f"Wrote source manifest: {manifest_path}")
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    result = validate_current(
        label=args.label,
        people_dir=args.people_dir.expanduser(),
        manifest_path=args.manifest_path.expanduser(),
        report_dir=args.report_dir.expanduser(),
    )
    print_validation(result)
    return 0 if result.ok else SOURCE_GUARD_EXIT


def cmd_promote(args: argparse.Namespace) -> int:
    manifest_path = args.manifest_path.expanduser()
    people_dir = args.people_dir.expanduser()
    if manifest_path.exists():
        result = validate_current(
            label=f"{args.label}_before_promote",
            people_dir=people_dir,
            manifest_path=manifest_path,
            report_dir=args.report_dir.expanduser(),
        )
        print_validation(result)
        if not result.ok and not args.force:
            print("Not promoting because protected originals are missing or changed.", file=sys.stderr)
            return SOURCE_GUARD_EXIT
    if not args.apply:
        print("DRY-RUN - no manifest written. Re-run with --apply to promote current state.")
        return 0
    path = promote_current(
        label=args.label,
        reason=args.reason or args.label,
        people_dir=people_dir,
        manifest_path=manifest_path,
    )
    print(f"Promoted current originals to source manifest: {path}")
    return 0


def cmd_restore(args: argparse.Namespace) -> int:
    people_dir = args.people_dir.expanduser()
    sorted_root = sorted_root_for_people(people_dir)
    search_roots = [p.expanduser() for p in (args.search_root or [READY_TO_DELETE])]
    report_dir = (
        RECOVERY_REPORT_DIR
        if args.report_dir.expanduser() == REPORT_DIR
        else args.report_dir.expanduser()
    )
    ok, report_path, rows = restore_from_manifest(
        people_dir=people_dir,
        manifest_path=args.manifest_path.expanduser(),
        search_roots=search_roots,
        conflict_dir=args.conflict_dir.expanduser(),
        report_dir=report_dir,
        label=args.label,
        apply=args.apply,
        restore_changed=not args.skip_changed,
        last_failed_run=args.last_failed_run,
    )
    restored = sum(1 for row in rows if row.get("status") == "restored")
    planned = sum(1 for row in rows if row.get("status") == "planned")
    missing = sum(1 for row in rows if row.get("status") == "missing_candidate")
    failed = sum(1 for row in rows if str(row.get("status", "")).endswith("failed"))
    nothing = any(row.get("status") == "nothing_to_restore" for row in rows)
    print("Source Manifest Restore")
    print("=" * 60)
    print(f"People folder:       {people_dir}")
    print(f"Search root(s):      {', '.join(str(p) for p in search_roots)}")
    print(f"Operation ledgers:   {operation_ledger.ledger_dir(sorted_root)}")
    print(f"Report CSV:          {report_path}")
    if nothing:
        print("Nothing to restore: protected originals are present and unchanged.")
    elif args.apply:
        print(f"Restored:            {restored}")
        print(f"Missing candidates:  {missing}")
        print(f"Failures:            {failed}")
    else:
        print(f"Planned restores:    {planned}")
        print(f"Missing candidates:  {missing}")
        print("DRY-RUN - no files restored. Re-run with --apply to commit.")
    return 0 if ok else SOURCE_GUARD_EXIT


def cmd_status(args: argparse.Namespace) -> int:
    manifest = load_manifest(args.manifest_path.expanduser())
    current_entries = collect_entries(args.people_dir.expanduser())
    print(f"Manifest:         {args.manifest_path.expanduser()}")
    if manifest is None:
        print("Manifest status:  missing")
    else:
        print(f"Manifest total:   {manifest.get('total', len(manifest.get('files', [])))}")
        print(f"Manifest updated: {manifest.get('updated_at', '')}")
    print(f"Current total:    {len(current_entries)}")
    if args.validate:
        result = validate_current(
            label=args.label,
            people_dir=args.people_dir.expanduser(),
            manifest_path=args.manifest_path.expanduser(),
            report_dir=args.report_dir.expanduser(),
        )
        print_validation(result)
        return 0 if result.ok else SOURCE_GUARD_EXIT
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate protected original person-photo manifest.")
    parser.add_argument("--people-dir", type=Path, default=PEOPLE, help=f"People folder. Default: {PEOPLE}")
    parser.add_argument("--manifest-path", type=Path, default=MANIFEST_PATH,
                        help=f"Manifest path. Default: {MANIFEST_PATH}")
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR,
                        help=f"Report directory. Default: {REPORT_DIR}")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init", help="Create the initial last-known-good manifest.")
    init.add_argument("--apply", action="store_true", help="Write the manifest.")
    init.add_argument("--replace", action="store_true", help="Replace an existing manifest.")
    init.add_argument("--reason", default="initial source manifest")
    init.set_defaults(func=cmd_init)

    validate = sub.add_parser("validate", help="Validate current originals against the manifest.")
    validate.add_argument("--label", default="source_manifest_validate")
    validate.set_defaults(func=cmd_validate)

    promote = sub.add_parser("promote", help="Promote current originals after a successful safe operation.")
    promote.add_argument("--apply", action="store_true", help="Write the manifest.")
    promote.add_argument("--force", action="store_true", help="Promote even if validation fails. Use only after manual recovery/review.")
    promote.add_argument("--label", default="source_manifest_promote")
    promote.add_argument("--reason", default="")
    promote.set_defaults(func=cmd_promote)

    restore = sub.add_parser("restore", help="Restore missing protected originals from ready_to_delete.")
    restore.add_argument("--last-failed-run", action="store_true",
                         help="Include the latest source-count violation report in the restore report.")
    restore.add_argument("--apply", action="store_true", help="Restore files. Default is dry-run.")
    restore.add_argument("--skip-changed", action="store_true",
                         help="Only restore missing originals; do not move changed files aside.")
    restore.add_argument("--search-root", type=Path, action="append", default=None,
                         help=f"Holding folder to search. Can be repeated. Default: {READY_TO_DELETE}")
    restore.add_argument("--conflict-dir", type=Path, default=RECOVERY_CONFLICT_DIR,
                         help=f"Where changed target files are moved before restore. Default: {RECOVERY_CONFLICT_DIR}")
    restore.add_argument("--label", default="source_manifest_restore")
    restore.set_defaults(func=cmd_restore)

    status = sub.add_parser("status", help="Print manifest and current totals.")
    status.add_argument("--validate", action="store_true", help="Also run validation.")
    status.add_argument("--label", default="source_manifest_status")
    status.set_defaults(func=cmd_status)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Guard canonical per-person original photo counts before and after operations.

The guard counts only original images inside:

  ~/Pictures/sorted_all_pictures/photos_by_person/<person>/photos/

That recursive count includes photos/nude/ and ignores generated smart albums,
review folders, and old all/ views by construction.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import source_manifest

SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
PEOPLE = SORTED / "photos_by_person"
SOURCE_REVIEW = SORTED / "_source_review"
GUARD_DIR = SOURCE_REVIEW / "source_count_guards"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
SOURCE_GUARD_EXIT = 3


def run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def slugify(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^a-z0-9._-]+", "_", value)
    return value.strip("_") or "codex_operation"


def person_original_counts(people_dir: Path = PEOPLE) -> dict[str, int]:
    """Count canonical original images per person under photos/, including photos/nude/."""
    counts: dict[str, int] = {}
    if not people_dir.exists():
        return counts
    person_dirs = [
        p for p in people_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")
    ]
    for person_dir in sorted(person_dirs, key=lambda p: p.name.lower()):
        photos_dir = person_dir / "photos"
        total = 0
        if photos_dir.exists():
            total = sum(
                1 for p in photos_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
        counts[person_dir.name] = total
    return counts


def total_count(counts: dict[str, int]) -> int:
    return sum(int(value) for value in counts.values())


def guard_paths(label: str, rid: str, guard_dir: Path = GUARD_DIR) -> dict[str, Path]:
    prefix = guard_dir / f"{slugify(label)}_{rid}"
    return {
        "before_csv": prefix.with_name(prefix.name + "_source_counts_before.csv"),
        "before_json": prefix.with_name(prefix.name + "_source_counts_before.json"),
        "after_csv": prefix.with_name(prefix.name + "_source_counts_after.csv"),
        "after_json": prefix.with_name(prefix.name + "_source_counts_after.json"),
        "violations_csv": prefix.with_name(prefix.name + "_source_count_violations.csv"),
    }


def write_counts_csv(path: Path, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["person", "original_photos_count"])
        writer.writeheader()
        for person, count in sorted(counts.items(), key=lambda item: item[0].lower()):
            writer.writerow({"person": person, "original_photos_count": int(count)})


def write_snapshot_json(path: Path,
                        *,
                        label: str,
                        rid: str,
                        phase: str,
                        people_dir: Path,
                        counts: dict[str, int],
                        before_json: Path | None = None,
                        command: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "label": slugify(label),
        "run_id": rid,
        "phase": phase,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "people_dir": str(people_dir),
        "person_count": len(counts),
        "total": total_count(counts),
        "counts": {str(k): int(v) for k, v in sorted(counts.items(), key=lambda item: item[0].lower())},
    }
    if before_json is not None:
        payload["baseline_json"] = str(before_json)
    if command is not None:
        payload["command"] = command
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def write_violations_csv(path: Path, violations: list[dict[str, int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["person", "before", "after", "delta"])
        writer.writeheader()
        writer.writerows(violations)


def source_count_violations(before: dict[str, int],
                            after: dict[str, int]) -> list[dict[str, int | str]]:
    violations: list[dict[str, int | str]] = []
    for person, before_count in sorted(before.items(), key=lambda item: item[0].lower()):
        before_count = int(before_count)
        after_count = int(after.get(person, 0))
        if after_count < before_count:
            violations.append({
                "person": person,
                "before": before_count,
                "after": after_count,
                "delta": after_count - before_count,
            })
    return violations


def read_counts_csv(path: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            person = str(row.get("person") or "").strip()
            if not person:
                continue
            counts[person] = int(row.get("original_photos_count") or 0)
    return counts


def load_baseline(path: Path) -> tuple[str, str, Path, dict[str, int]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        label = slugify(str(payload.get("label") or "codex_operation"))
        rid = str(payload.get("run_id") or run_id())
        people_dir = Path(str(payload.get("people_dir") or PEOPLE)).expanduser()
        counts = {str(k): int(v) for k, v in dict(payload.get("counts") or {}).items()}
        return label, rid, people_dir, counts
    label = slugify(path.stem.replace("_source_counts_before", ""))
    return label, run_id(), PEOPLE, read_counts_csv(path)


def print_ok(before: dict[str, int], after: dict[str, int]) -> None:
    print(
        "Source guard OK: "
        f"{total_count(before)} -> {total_count(after)} originals "
        f"across {len(after)} person folders"
    )


def print_violations(violations: list[dict[str, int | str]]) -> None:
    print(f"Source guard FAILED: {len(violations)} person folder(s) lost originals", file=sys.stderr)
    for row in violations[:10]:
        print(
            f"  {row['person']}: {row['before']} -> {row['after']} ({row['delta']})",
            file=sys.stderr,
        )


def make_snapshot(label: str,
                  *,
                  people_dir: Path = PEOPLE,
                  guard_dir: Path = GUARD_DIR,
                  command: list[str] | None = None) -> tuple[Path, dict[str, int]]:
    rid = run_id()
    paths = guard_paths(label, rid, guard_dir)
    counts = person_original_counts(people_dir)
    write_counts_csv(paths["before_csv"], counts)
    write_snapshot_json(
        paths["before_json"],
        label=label,
        rid=rid,
        phase="before",
        people_dir=people_dir,
        counts=counts,
        command=command,
    )
    print(paths["before_json"], flush=True)
    print(f"Snapshot: {total_count(counts)} originals across {len(counts)} person folders", flush=True)
    return paths["before_json"], counts


def validate_counts(baseline_json: Path,
                    before: dict[str, int],
                    *,
                    label: str,
                    rid: str,
                    people_dir: Path,
                    guard_dir: Path = GUARD_DIR,
                    command: list[str] | None = None) -> bool:
    paths = guard_paths(label, rid, guard_dir)
    after = person_original_counts(people_dir)
    violations = source_count_violations(before, after)
    write_counts_csv(paths["after_csv"], after)
    write_snapshot_json(
        paths["after_json"],
        label=label,
        rid=rid,
        phase="after",
        people_dir=people_dir,
        counts=after,
        before_json=baseline_json,
        command=command,
    )
    write_violations_csv(paths["violations_csv"], violations)
    if violations:
        print_violations(violations)
        print(f"After snapshot: {paths['after_json']}", file=sys.stderr)
        print(f"Violations CSV: {paths['violations_csv']}", file=sys.stderr)
        return False
    print_ok(before, after)
    print(f"After snapshot: {paths['after_json']}", flush=True)
    return True


def cmd_snapshot(args: argparse.Namespace) -> int:
    manifest_check = source_manifest.validate_current(
        label=f"{args.label}_snapshot",
        people_dir=args.people_dir.expanduser(),
    )
    source_manifest.print_validation(manifest_check)
    if not manifest_check.ok:
        return SOURCE_GUARD_EXIT
    make_snapshot(
        args.label,
        people_dir=args.people_dir.expanduser(),
        guard_dir=args.guard_dir.expanduser(),
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    baseline = args.baseline.expanduser()
    base_label, base_rid, people_dir, before = load_baseline(baseline)
    label = slugify(args.label or base_label)
    rid = args.run_id or base_rid or run_id()
    ok = validate_counts(
        baseline,
        before,
        label=label,
        rid=rid,
        people_dir=args.people_dir.expanduser() if args.people_dir else people_dir,
        guard_dir=args.guard_dir.expanduser(),
    )
    return 0 if ok else SOURCE_GUARD_EXIT


def cmd_run(args: argparse.Namespace) -> int:
    command = list(args.command or [])
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print("source_guard.py run requires a command after --", file=sys.stderr)
        return 2

    label = slugify(args.label)
    people_dir = args.people_dir.expanduser()
    guard_dir = args.guard_dir.expanduser()
    manifest_check = source_manifest.validate_current(
        label=f"{label}_before",
        people_dir=people_dir,
    )
    source_manifest.print_validation(manifest_check)
    if not manifest_check.ok:
        return SOURCE_GUARD_EXIT

    baseline_json, before = make_snapshot(
        label,
        people_dir=people_dir,
        guard_dir=guard_dir,
        command=command,
    )
    _, rid, _, _ = load_baseline(baseline_json)

    command_returncode = 0
    try:
        env = os.environ.copy()
        env["PHOTO_PIPELINE_RUN_ID"] = rid
        completed = subprocess.run(command, check=False, env=env)
        command_returncode = int(completed.returncode)
    except KeyboardInterrupt:
        command_returncode = 130
    except FileNotFoundError as exc:
        print(f"Command not found: {exc.filename}", file=sys.stderr)
        command_returncode = 127
    finally:
        ok = validate_counts(
            baseline_json,
            before,
            label=label,
            rid=rid,
            people_dir=people_dir,
            guard_dir=guard_dir,
            command=command,
        )

    if not ok:
        return SOURCE_GUARD_EXIT
    if command_returncode == 0:
        post_manifest_check = source_manifest.validate_current(
            label=f"{label}_before_promote",
            people_dir=people_dir,
        )
        source_manifest.print_validation(post_manifest_check)
        if not post_manifest_check.ok:
            return SOURCE_GUARD_EXIT
        manifest_path = source_manifest.promote_current(
            label=f"{label}_completed",
            reason=f"successful guarded command: {label}",
            people_dir=people_dir,
        )
        print(f"Source manifest promoted: {manifest_path}", flush=True)
    return command_returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Snapshot and validate original per-person photo counts.",
    )
    parser.add_argument(
        "--guard-dir",
        type=Path,
        default=GUARD_DIR,
        help=f"Where guard reports are written. Default: {GUARD_DIR}",
    )
    subparsers = parser.add_subparsers(dest="command_name", required=True)

    snapshot = subparsers.add_parser("snapshot", help="Write a before-count snapshot.")
    snapshot.add_argument("--label", default="codex_operation", help="Short report label.")
    snapshot.add_argument("--people-dir", type=Path, default=PEOPLE, help=f"People folder. Default: {PEOPLE}")
    snapshot.set_defaults(func=cmd_snapshot)

    validate = subparsers.add_parser("validate", help="Validate current counts against a snapshot.")
    validate.add_argument("--baseline", type=Path, required=True, help="Before snapshot JSON or CSV.")
    validate.add_argument("--label", default=None, help="Optional label for after reports.")
    validate.add_argument("--run-id", default=None, help="Optional run id for after reports.")
    validate.add_argument("--people-dir", type=Path, default=None, help=f"People folder. Default: from baseline or {PEOPLE}")
    validate.set_defaults(func=cmd_validate)

    run = subparsers.add_parser("run", help="Snapshot, run a command, then validate.")
    run.add_argument("--label", default="codex_operation", help="Short report label.")
    run.add_argument("--people-dir", type=Path, default=PEOPLE, help=f"People folder. Default: {PEOPLE}")
    run.add_argument("command", nargs=argparse.REMAINDER, help="Command to run after --")
    run.set_defaults(func=cmd_run)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())

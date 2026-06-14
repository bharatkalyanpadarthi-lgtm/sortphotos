#!/usr/bin/env python3
"""
Run the daily photo pipeline with progress state, memory checks, and a summary.

Use this through:
  python face.py daily

If a step fails or the run is interrupted, resume with:
  python face.py daily --resume

Preview without moving files:
  python face.py daily --dry-run
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import source_manifest

SCRIPT_DIR = Path(__file__).resolve().parent
SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
PEOPLE = SORTED / "photos_by_person"
SOURCE_REVIEW = SORTED / "_source_review"
READY = SOURCE_REVIEW / "ready_to_delete"
TO_PROCESS = Path.home() / "Pictures" / "To Process"
STATE_FILE = Path.home() / ".face_sort_cache" / "daily_run_state.json"
SUMMARY_DIR = SOURCE_REVIEW / "daily_run_summaries"
ADV_REPORT = SOURCE_REVIEW / "duplicate_reports" / "advanced_duplicates.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
VIDEO_EXTS = {
    ".3g2", ".3gp", ".avi", ".m4v", ".mkv", ".mov", ".mp4",
    ".mpeg", ".mpg", ".mts", ".m2ts", ".webm", ".wmv",
}
SMART_DIRS = {"all", "_smart_albums", "_smart_albums_v2", "_duplicates", "_near_visual_review", "review"}
SOURCE_GUARD_EXIT = 3


def run_id() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def count_images(root: Path, exclude_generated_dirs: bool = True) -> int:
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        if exclude_generated_dirs and any(part in SMART_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
            total += 1
    return total


def count_videos(root: Path, exclude_generated_dirs: bool = True) -> int:
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        if exclude_generated_dirs and any(part in SMART_DIRS for part in p.parts):
            continue
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            total += 1
    return total


def count_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())


def original_person_counts() -> dict[str, int]:
    """Count canonical original images per person under photos/, including photos/nude/."""
    counts: dict[str, int] = {}
    if not PEOPLE.exists():
        return counts
    for person_dir in sorted(
        [p for p in PEOPLE.iterdir() if p.is_dir() and not p.name.startswith("_")],
        key=lambda p: p.name.lower(),
    ):
        photos_dir = person_dir / "photos"
        total = 0
        if photos_dir.exists():
            for p in photos_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                    total += 1
        counts[person_dir.name] = total
    return counts


def original_person_total(counts: dict[str, int]) -> int:
    return sum(int(value) for value in counts.values())


def source_guard_paths(run_id_value: str) -> dict[str, Path]:
    prefix = SUMMARY_DIR / f"daily_run_{run_id_value}"
    return {
        "before": prefix.with_name(prefix.name + "_source_counts_before.csv"),
        "after": prefix.with_name(prefix.name + "_source_counts_after.csv"),
        "violations": prefix.with_name(prefix.name + "_source_count_violations.csv"),
    }


def write_source_counts_csv(path: Path, counts: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["person", "original_photos_count"])
        writer.writeheader()
        for person, count in sorted(counts.items(), key=lambda item: item[0].lower()):
            writer.writerow({"person": person, "original_photos_count": int(count)})


def source_count_violations(before: dict[str, int],
                            after: dict[str, int]) -> list[dict[str, int | str]]:
    violations: list[dict[str, int | str]] = []
    for person, before_count in sorted(before.items(), key=lambda item: item[0].lower()):
        after_count = int(after.get(person, 0))
        before_count = int(before_count)
        if after_count < before_count:
            violations.append({
                "person": person,
                "before": before_count,
                "after": after_count,
                "delta": after_count - before_count,
            })
    return violations


def write_source_violations_csv(path: Path,
                                violations: list[dict[str, int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["person", "before", "after", "delta"])
        writer.writeheader()
        writer.writerows(violations)


def ensure_source_guard_baseline(state: dict) -> dict[str, int]:
    guard = state.setdefault("source_guard", {})
    before = guard.get("before")
    if not isinstance(before, dict):
        before = original_person_counts()
        guard["before"] = before
        guard["started_at"] = int(time.time())
    if not isinstance(guard.get("floor"), dict):
        guard["floor"] = {str(k): int(v) for k, v in before.items()}
    paths = source_guard_paths(str(state["run_id"]))
    guard["before_csv"] = str(paths["before"])
    guard["after_csv"] = str(paths["after"])
    guard["violations_csv"] = str(paths["violations"])
    write_source_counts_csv(paths["before"], {str(k): int(v) for k, v in before.items()})
    return {str(k): int(v) for k, v in before.items()}


def check_source_guard(state: dict, stage: str) -> tuple[bool, dict[str, int], list[dict[str, int | str]]]:
    before = ensure_source_guard_baseline(state)
    guard = state.setdefault("source_guard", {})
    floor = {str(k): int(v) for k, v in guard.get("floor", before).items()}
    after = original_person_counts()
    paths = source_guard_paths(str(state["run_id"]))
    violations = source_count_violations(floor, after)
    write_source_counts_csv(paths["after"], after)
    if violations:
        write_source_violations_csv(paths["violations"], violations)
    else:
        next_floor = dict(floor)
        for person, count in after.items():
            next_floor[person] = max(int(next_floor.get(person, 0)), int(count))
        guard["floor"] = next_floor
    state.setdefault("source_guard", {}).update({
        "last_checked_stage": stage,
        "last_checked_at": int(time.time()),
        "before_total": original_person_total(before),
        "after_total": original_person_total(after),
        "floor_total": original_person_total(floor),
        "violation_count": len(violations),
    })
    return len(violations) == 0, after, violations


def check_source_manifest(state: dict, stage: str) -> source_manifest.ManifestValidation:
    result = source_manifest.validate_current(
        label=f"daily_run_{state['run_id']}_{stage}",
        people_dir=PEOPLE,
    )
    source_manifest.print_validation(result)
    state["source_manifest"] = {
        "last_checked_stage": stage,
        "last_checked_at": int(time.time()),
        "ok": result.ok,
        "manifest_path": str(result.manifest_path),
        "expected_total": result.expected_total,
        "current_total": result.current_total,
        "missing": len(result.missing),
        "size_changed": len(result.size_changed),
        "renamed": len(result.renamed),
        "extra": len(result.extra),
        "missing_csv": str(result.missing_csv),
        "changed_csv": str(result.changed_csv),
        "renamed_csv": str(result.renamed_csv),
        "extra_csv": str(result.extra_csv),
    }
    return result


def size_bytes(root: Path) -> int:
    if not root.exists():
        return 0
    total = 0
    for p in root.rglob("*"):
        if not p.is_file():
            continue
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


def nudity_count() -> int:
    total = 0
    if not PEOPLE.exists():
        return 0
    review_parts = {
        ("photos", "nude"),
        ("review", "nudity_possible"),
        ("review", "uncertain_nudity"),
        ("photos_nude",),
        ("_possible_nudity",),
        ("_uncertain_nudity",),
    }
    for p in PEOPLE.rglob("*"):
        if not p.is_file() or p.suffix.lower() not in IMAGE_EXTS:
            continue
        try:
            rel = p.relative_to(PEOPLE)
        except ValueError:
            continue
        parts = rel.parts[1:]
        if any(tuple(parts[:len(prefix)]) == prefix for prefix in review_parts):
            total += 1
    return total


def duplicate_counts() -> dict[str, int]:
    counts = {"exact_file": 0, "same_pixels": 0, "visually_similar": 0}
    if not ADV_REPORT.exists():
        return counts
    try:
        with ADV_REPORT.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("action") in {"move", "review"} and row.get("type") in counts:
                    counts[row["type"]] += 1
    except Exception:
        pass
    return counts


def labeling_remaining() -> dict[str, int]:
    try:
        import sort_photos
        for name in ("CacheState", "CachedFace", "FaceRecord", "LabelingState", "IdentityDB"):
            if hasattr(sort_photos, name):
                setattr(sys.modules["__main__"], name, getattr(sort_photos, name))
        state = sort_photos.load_labeling_state()
    except Exception:
        state = None
    if state is None:
        return {"clusters": 0, "faces": 0}
    sizes: dict[int, int] = {}
    for cid in state.cluster_ids:
        sizes[cid] = sizes.get(cid, 0) + 1
    clusters = 0
    faces = 0
    for cid, count in sizes.items():
        if cid == -1:
            continue
        label = state.name_map.get(cid, "")
        if label.startswith("person_"):
            clusters += 1
            faces += count
    return {"clusters": clusters, "faces": faces}


def snapshot() -> dict:
    dups = duplicate_counts()
    labels = labeling_remaining()
    original_counts = original_person_counts()
    return {
        "to_process_images": count_images(TO_PROCESS, exclude_generated_dirs=False),
        "to_process_videos": count_videos(TO_PROCESS, exclude_generated_dirs=False),
        "organized_images": count_images(PEOPLE),
        "person_original_images": original_person_total(original_counts),
        "person_folders": len(original_counts),
        "nudity_images": nudity_count(),
        "ready_to_delete_files": count_files(READY),
        "ready_to_delete_size": size_bytes(READY),
        "organized_sources_files": count_files(READY / "organized_sources"),
        "scanned_sources_files": count_files(READY / "scanned_sources"),
        "intake_duplicates_files": count_files(READY / "intake_duplicates"),
        "unknown_clusters": labels["clusters"],
        "unknown_faces": labels["faces"],
        "near_visual_review": dups["visually_similar"],
        "exact_file_duplicates": dups["exact_file"],
        "same_pixel_duplicates": dups["same_pixels"],
    }


def delta(after: dict, before: dict, key: str) -> int:
    return int(after.get(key, 0)) - int(before.get(key, 0))


def load_state() -> dict | None:
    if not STATE_FILE.exists():
        return None
    try:
        with STATE_FILE.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
    tmp.replace(STATE_FILE)


def clear_state() -> None:
    try:
        STATE_FILE.unlink()
    except FileNotFoundError:
        pass


def available_memory_mb() -> int | None:
    try:
        page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True).strip())
        output = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return None
    pages = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        raw = raw.strip().strip(".").replace(".", "")
        try:
            pages[name] = int(raw)
        except ValueError:
            continue
    freeish = (
        pages.get("Pages free", 0)
        + pages.get("Pages inactive", 0)
        + pages.get("Pages speculative", 0)
    )
    return int((freeish * page_size) / (1024 * 1024))


def memory_profile() -> dict:
    mb = available_memory_mb()
    if mb is None:
        return {"available_mb": None, "batch_size": 50, "ok": True, "message": "memory unavailable"}
    if mb < 1200:
        return {"available_mb": mb, "batch_size": 15, "ok": False,
                "message": "available memory is critically low"}
    if mb < 3500:
        return {"available_mb": mb, "batch_size": 25, "ok": True,
                "message": "low-memory mode"}
    return {"available_mb": mb, "batch_size": 50, "ok": True, "message": "normal"}


def empty_inbox_skippable_step_names() -> set[str]:
    """Steps that have nothing useful to do when the intake folder is empty."""
    return {"process"}


def cleanup_holding_count() -> int:
    return count_files(SOURCE_REVIEW)


def step_list(batch_size: int) -> list[dict]:
    py = sys.executable
    return [
        {
            "name": "preflight",
            "desc": "Preflight folders, memory, disk, and process safety",
            "cmd": [py, str(SCRIPT_DIR / "preflight_check.py")],
        },
        {
            "name": "process",
            "desc": "Process new inbox images",
            "cmd": [
                py, str(SCRIPT_DIR / "sort_photos.py"),
                str(TO_PROCESS), str(SORTED),
                "--unattended",
                "--archive-organized-sources",
                "--archive-sources-to-ready-delete",
                "--archive-scanned-sources",
                "--merge-existing-output",
                "--skip-output-cleanup",
                "--batch-size", str(batch_size),
                "--detect-workers", "1",
            ],
            "heavy": True,
        },
        {"name": "structure", "desc": "Normalize person folder structure",
         "cmd": [py, str(SCRIPT_DIR / "person_structure.py"), "--apply", "--quiet"]},
        {"name": "rename", "desc": "Normalize person filenames",
         "cmd": [py, str(SCRIPT_DIR / "rename_person_folder_files.py"), "--apply", "--quiet"]},
        {"name": "exact-dedupe", "desc": "Report exact person-folder duplicates without moving originals",
         "cmd": [py, str(SCRIPT_DIR / "delete_person_folder_duplicates.py"), "--quiet"]},
        {"name": "advanced-dedupe", "desc": "Refresh advanced duplicate report without moving originals",
         "cmd": [py, str(SCRIPT_DIR / "advanced_duplicate_matching.py"), "--quiet"], "heavy": True},
        {"name": "cleanup-empty", "desc": "Move empty person folders to ready_to_delete",
         "cmd": [py, str(SCRIPT_DIR / "cleanup_empty_person_folders.py"), "--apply", "--quiet"]},
        {"name": "cache-rehydrate", "desc": "Refresh face cache after all file-moving cleanup",
         "cmd": [py, str(SCRIPT_DIR / "cache_tools.py"), "rehydrate", "--apply", "--batch-size", str(batch_size)], "heavy": True},
        {"name": "smart-albums", "desc": "Refresh changed smart albums",
         "cmd": [
             py,
             str(SCRIPT_DIR / "build_smart_albums.py"),
             "--apply",
             "--incremental",
             "--no-detect-nudity",
             "--framing-det-size",
             "640",
             "--max-framing-checks-per-person",
             "300",
             "--max-people-per-run",
             "25",
         ], "heavy": True},
        {"name": "unknown-triage", "desc": "Write unknown-cluster triage report",
         "cmd": [py, str(SCRIPT_DIR / "unknown_triage.py"), "--quiet"]},
        {"name": "integration-audit", "desc": "Verify final cross-script invariants",
         "cmd": [py, str(SCRIPT_DIR / "integration_audit.py")]},
        {"name": "status", "desc": "Print final dashboard",
         "cmd": [py, str(SCRIPT_DIR / "status_report.py")]},
    ]


def run_command(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    run_id_value = ""
    if "daily_run_" in log_path.stem:
        run_id_value = log_path.stem.replace("daily_run_", "", 1)
    if run_id_value:
        env["PHOTO_PIPELINE_RUN_ID"] = f"daily_run_{run_id_value}"
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n$ {' '.join(cmd)}\n")
        log.flush()
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, bufsize=1, env=env)
        assert proc.stdout is not None
        for line in proc.stdout:
            print(line, end="")
            log.write(line)
        return proc.wait()


def write_summary(path: Path, state: dict, before: dict, after: dict, status: str) -> None:
    summary = {
        "run_id": state["run_id"],
        "status": status,
        "started_at": state["started_at"],
        "finished_at": int(time.time()),
        "before": before,
        "after": after,
        "delta": {key: delta(after, before, key) for key in after},
        "steps": state["steps"],
        "memory": state.get("memory", {}),
        "source_guard": state.get("source_guard", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def print_summary(before: dict, after: dict, summary_path: Path) -> None:
    print()
    print("Daily Run Summary")
    print("=" * 60)
    print(f"New organized images:       {delta(after, before, 'organized_images')}")
    print(f"Original photos before/after:{before.get('person_original_images', 0)} -> {after.get('person_original_images', 0)}")
    print(f"To Process before/after:    {before['to_process_images']} -> {after['to_process_images']}")
    print(f"To Process videos moved:    {before.get('to_process_videos', 0)} -> {after.get('to_process_videos', 0)}")
    print(f"Nudity-folder image change: {delta(after, before, 'nudity_images')}")
    print(f"Archived organized sources: +{delta(after, before, 'organized_sources_files')}")
    print(f"Archived scanned sources:   +{delta(after, before, 'scanned_sources_files')}")
    print(f"Archived intake duplicates: +{delta(after, before, 'intake_duplicates_files')}")
    print(f"Unknown clusters/faces:     {after['unknown_clusters']} / {after['unknown_faces']}")
    print(f"Near-visual review items:   {after['near_visual_review']}")
    print(f"ready_to_delete size:       {human_size(after['ready_to_delete_size'])}")
    print(f"Summary JSON:               {summary_path}")


def print_dry_run(steps: list[dict], before: dict, profile: dict,
                  full_maintenance: bool) -> None:
    empty_inbox = (
        int(before.get("to_process_images", 0)) == 0
        and int(before.get("to_process_videos", 0)) == 0
    )
    print("Daily Dry Run")
    print("=" * 60)
    print(f"Sorted folder:              {SORTED}")
    print(f"Input folder:               {TO_PROCESS}")
    print(f"To Process images:          {before['to_process_images']}")
    print(f"To Process videos:          {before.get('to_process_videos', 0)}")
    print(f"Organized images:           {before['organized_images']}")
    print(f"Original person photos:     {before.get('person_original_images', 0)}")
    print(f"_source_review files:       {cleanup_holding_count():,}")
    if profile.get("available_mb") is not None:
        print(f"Memory mode:                {profile['message']} ({profile['available_mb']} MB), batch {profile['batch_size']}")
    else:
        print(f"Memory mode:                {profile['message']}, batch {profile['batch_size']}")
    print()
    print("Steps that would run")
    skip_when_empty = empty_inbox_skippable_step_names()
    for index, step in enumerate(steps, start=1):
        would_skip = empty_inbox and not full_maintenance and step["name"] in skip_when_empty
        status = "skip: empty inbox" if would_skip else "run"
        print(f"[{index}/{len(steps)}] {status:18} {step['desc']}")
        print(f"    {' '.join(step['cmd'])}")
    print()
    print("DRY-RUN only. No files were moved, renamed, or deleted.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true",
                        help="Resume the previous incomplete daily run.")
    parser.add_argument("--restart", action="store_true",
                        help="Discard previous daily run state and start from step 1.")
    parser.add_argument("--ignore-low-memory", action="store_true",
                        help="Run even if the memory safety check says memory is critically low.")
    parser.add_argument("--full-maintenance", action="store_true",
                        help="Run maintenance steps even when To Process has no images.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what daily would do without running any step.")
    args = parser.parse_args()

    if args.dry_run:
        profile = memory_profile()
        before = snapshot()
        print_dry_run(
            step_list(int(profile.get("batch_size") or 50)),
            before,
            profile,
            args.full_maintenance,
        )
        return 0

    if args.restart:
        clear_state()

    state = load_state() if args.resume else None
    if state is None:
        profile = memory_profile()
        if not profile["ok"] and not args.ignore_low_memory:
            print(f"ERROR: {profile['message']} ({profile['available_mb']} MB available).")
            print("Close other apps, then re-run. Override with --ignore-low-memory.")
            return 2
        rid = run_id()
        before_snapshot = snapshot()
        state = {
            "run_id": rid,
            "started_at": int(time.time()),
            "before": before_snapshot,
            "steps": {},
            "memory": profile,
            "source_guard": {
                "before": original_person_counts(),
                "started_at": int(time.time()),
            },
        }
        ensure_source_guard_baseline(state)
        save_state(state)
    else:
        print(f"Resuming daily run: {state.get('run_id')}")
        ensure_source_guard_baseline(state)
        save_state(state)

    before = state["before"]
    profile = state.get("memory") or memory_profile()
    batch_size = int(profile.get("batch_size") or 50)
    if profile.get("available_mb") is not None:
        print(f"Memory mode: {profile['message']} ({profile['available_mb']} MB available), batch size {batch_size}.")
    else:
        print(f"Memory mode: {profile['message']}, batch size {batch_size}.")
    guard = state.get("source_guard", {})
    print(f"Source guard baseline: {guard.get('before_total', original_person_total(guard.get('before', {})))} original photos "
          f"across {len(guard.get('before', {}))} person folders.")
    if guard.get("before_csv"):
        print(f"Source guard before CSV: {guard['before_csv']}")

    ok, _, violations = check_source_guard(state, "start")
    save_state(state)
    if not ok:
        paths = source_guard_paths(str(state["run_id"]))
        print("ERROR: source guard failed before running steps.")
        print(f"Violation report: {paths['violations']}")
        for row in violations[:10]:
            print(f"  {row['person']}: {row['before']} -> {row['after']} ({row['delta']})")
        return SOURCE_GUARD_EXIT

    manifest_result = check_source_manifest(state, "start")
    save_state(state)
    if not manifest_result.ok:
        print("ERROR: protected source manifest failed before running steps.")
        print("Fix or recover the missing originals before cache/smart-album refresh can run.")
        return SOURCE_GUARD_EXIT

    log_path = SUMMARY_DIR / f"daily_run_{state['run_id']}.log"
    summary_path = SUMMARY_DIR / f"daily_run_{state['run_id']}.json"
    steps = step_list(batch_size)
    empty_inbox = (
        int(before.get("to_process_images", 0)) == 0
        and int(before.get("to_process_videos", 0)) == 0
    )
    skip_when_empty = empty_inbox_skippable_step_names()
    for index, step in enumerate(steps, start=1):
        if state["steps"].get(step["name"], {}).get("status") == "completed":
            print(f"[{index}/{len(steps)}] Skipping completed step: {step['desc']}")
            continue
        if empty_inbox and not args.full_maintenance and step["name"] in skip_when_empty:
            print(f"[{index}/{len(steps)}] Skipping empty-inbox step: {step['desc']}")
            state["steps"][step["name"]] = {
                "status": "skipped_empty_inbox",
                "finished_at": int(time.time()),
            }
            save_state(state)
            continue
        if step.get("heavy"):
            profile_now = memory_profile()
            if not profile_now["ok"] and not args.ignore_low_memory:
                state["steps"][step["name"]] = {
                    "status": "blocked_low_memory",
                    "finished_at": int(time.time()),
                    "memory": profile_now,
                }
                save_state(state)
                print(f"ERROR: low memory before {step['name']}: {profile_now['available_mb']} MB available.")
                print("Close other apps, then run: python face.py daily --resume")
                return 2
        print()
        print(f"[{index}/{len(steps)}] {step['desc']}")
        state["steps"][step["name"]] = {"status": "running", "started_at": int(time.time())}
        save_state(state)
        rc = run_command(step["cmd"], log_path)
        if rc != 0:
            state["steps"][step["name"]] = {
                "status": "failed",
                "returncode": rc,
                "finished_at": int(time.time()),
            }
            save_state(state)
            after = snapshot()
            write_summary(summary_path, state, before, after, "failed")
            print(f"ERROR: step failed: {step['name']} (exit {rc})")
            print(f"Resume with: python face.py daily --resume")
            return rc
        ok, guarded_after, violations = check_source_guard(state, step["name"])
        if not ok:
            state["steps"][step["name"]] = {
                "status": "failed_source_count_guard",
                "returncode": SOURCE_GUARD_EXIT,
                "finished_at": int(time.time()),
            }
            save_state(state)
            after = snapshot()
            write_summary(summary_path, state, before, after, "failed_source_count_guard")
            paths = source_guard_paths(str(state["run_id"]))
            print()
            print("ERROR: source guard stopped the daily run.")
            print("One or more person folders now have fewer original images under photos/ than at run start.")
            print(f"Before count CSV:     {paths['before']}")
            print(f"After count CSV:      {paths['after']}")
            print(f"Violation report CSV: {paths['violations']}")
            for row in violations[:10]:
                print(f"  {row['person']}: {row['before']} -> {row['after']} ({row['delta']})")
            if len(violations) > 10:
                print(f"  ... {len(violations) - 10} more")
            print(f"Resume after fixing counts with: python face.py daily --resume")
            return SOURCE_GUARD_EXIT
        manifest_result = check_source_manifest(state, step["name"])
        save_state(state)
        if not manifest_result.ok:
            state["steps"][step["name"]] = {
                "status": "failed_source_manifest_guard",
                "returncode": SOURCE_GUARD_EXIT,
                "finished_at": int(time.time()),
            }
            save_state(state)
            after = snapshot()
            write_summary(summary_path, state, before, after, "failed_source_manifest_guard")
            print()
            print("ERROR: protected source manifest stopped the daily run.")
            print("A known original image is missing or changed, so derived cache/index refresh is blocked.")
            print(f"Missing report CSV: {manifest_result.missing_csv}")
            print(f"Changed report CSV: {manifest_result.changed_csv}")
            print(f"Resume after fixing originals with: python face.py daily --resume")
            return SOURCE_GUARD_EXIT
        state["steps"][step["name"]] = {"status": "completed", "finished_at": int(time.time())}
        save_state(state)

    after = snapshot()
    ok, _, violations = check_source_guard(state, "completed")
    if not ok:
        save_state(state)
        write_summary(summary_path, state, before, after, "failed_source_count_guard")
        paths = source_guard_paths(str(state["run_id"]))
        print("ERROR: source guard failed at final validation.")
        print(f"Violation report CSV: {paths['violations']}")
        for row in violations[:10]:
            print(f"  {row['person']}: {row['before']} -> {row['after']} ({row['delta']})")
        return SOURCE_GUARD_EXIT
    manifest_result = check_source_manifest(state, "completed_before_promote")
    save_state(state)
    if not manifest_result.ok:
        write_summary(summary_path, state, before, after, "failed_source_manifest_guard")
        print("ERROR: protected source manifest failed at final validation.")
        print(f"Missing report CSV: {manifest_result.missing_csv}")
        print(f"Changed report CSV: {manifest_result.changed_csv}")
        return SOURCE_GUARD_EXIT
    manifest_path = source_manifest.promote_current(
        label=f"daily_run_{state['run_id']}_completed",
        reason=f"completed daily run {state['run_id']}",
        people_dir=PEOPLE,
    )
    state.setdefault("source_manifest", {})["promoted_manifest_path"] = str(manifest_path)
    save_state(state)
    write_summary(summary_path, state, before, after, "completed")
    print_summary(before, after, summary_path)
    clear_state()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

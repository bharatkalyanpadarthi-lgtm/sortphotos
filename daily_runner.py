#!/usr/bin/env python3
"""
Run the daily photo pipeline with progress state, memory checks, and a summary.

Use this through:
  python face.py daily

If a step fails or the run is interrupted, resume with:
  python face.py daily --resume

Preview without moving files:
  python face.py daily --dry-run

Smart albums are intentionally not part of the daily ingest and are disabled
from the launcher for now.
"""

from __future__ import annotations

import argparse
import codecs
import csv
import json
import os
import selectors
import signal
import subprocess
import sys
import time
import hashlib
import re
import stat as stat_types
from pathlib import Path

import source_manifest
import pipeline_paths
import daily_inventory
from daily_progress import CommandProgress, OutputLines, STEP_LABELS, duration

SCRIPT_DIR = Path(__file__).resolve().parent
SORTED = pipeline_paths.SORTED_ROOT
PEOPLE = SORTED / "photos_by_person"
SOURCE_REVIEW = SORTED / "_source_review"
READY = SOURCE_REVIEW / "ready_to_delete"
TO_PROCESS = pipeline_paths.TO_PROCESS
LEGACY_VIDEO_INBOX = Path.home() / "Pictures" / "videos"
STATE_FILE = Path.home() / ".face_sort_cache" / "daily_run_state.json"
SUMMARY_DIR = SOURCE_REVIEW / "daily_run_summaries"
ADV_REPORT = SOURCE_REVIEW / "duplicate_reports" / "advanced_duplicates.csv"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
VIDEO_EXTS = {
    ".3g2", ".3gp", ".avi", ".m4v", ".mkv", ".mov", ".mp4",
    ".mpeg", ".mpg", ".mts", ".m2ts", ".webm", ".wmv",
}
SMART_DIRS = {
    "all",
    "_smart_albums",
    "_smart_albums_v2",
    "_smart_albums_simple_preview",
    "_duplicates",
    "_near_visual_review",
    "review",
}
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


def tree_contains_media(root: Path, extensions: set[str]) -> bool:
    """Return after finding the first supported file instead of counting a tree."""
    if not root.exists():
        return False
    try:
        for current_root, _dirs, files in os.walk(root):
            if any(Path(name).suffix.lower() in extensions for name in files):
                return True
    except OSError:
        return False
    return False


def intake_has_media(
    to_process: Path = TO_PROCESS,
    legacy_video_inbox: Path = LEGACY_VIDEO_INBOX,
) -> bool:
    """Cheap gate used before the expensive full-library daily snapshot."""
    return (
        tree_contains_media(to_process, IMAGE_EXTS | VIDEO_EXTS)
        or tree_contains_media(legacy_video_inbox, VIDEO_EXTS)
    )


def original_person_counts() -> dict[str, int]:
    """Count canonical original images per person under photos/, including photos/nude/."""
    counts: dict[str, int] = {}
    if not PEOPLE.exists():
        return counts
    for person_dir in sorted(
        [p for p in PEOPLE.iterdir() if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")],
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


def check_source_guard(state: dict, stage: str, *, counts: dict | None = None) -> tuple[bool, dict[str, int], list[dict[str, int | str]]]:
    before = ensure_source_guard_baseline(state)
    guard = state.setdefault("source_guard", {})
    floor = {str(k): int(v) for k, v in guard.get("floor", before).items()}
    after = original_person_counts() if counts is None else counts
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


def check_source_manifest(state: dict, stage: str, *, verbose: bool = True) -> source_manifest.ManifestValidation:
    result = source_manifest.validate_current(
        label=f"daily_run_{state['run_id']}_{stage}",
        people_dir=PEOPLE,
    )
    if verbose or not result.ok:
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
        "app_trashed": len(result.app_trashed),
        "missing_csv": str(result.missing_csv),
        "changed_csv": str(result.changed_csv),
        "renamed_csv": str(result.renamed_csv),
        "extra_csv": str(result.extra_csv),
        "app_trashed_csv": str(result.app_trashed_csv),
    }
    return result


def verified_original_counts(result) -> dict | None:
    entries = getattr(result, "current_entries", None)
    if entries is None:
        return None
    counts = {name: 0 for name in source_manifest.person_names(PEOPLE)}
    for entry in entries:
        parts = Path(entry["relative_path"]).parts
        if len(parts) >= 3 and parts[1] == "photos":
            counts[parts[0]] = counts.get(parts[0], 0) + 1
    return counts


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
    holding = {"files": 0, "size": 0, "organized_sources": 0, "scanned_sources": 0, "intake_duplicates": 0}
    if READY.exists():
        for current, _dirs, files in os.walk(READY):
            for name in files:
                path = Path(current) / name
                try:
                    stat = path.stat()
                except FileNotFoundError:
                    continue
                if not stat_types.S_ISREG(stat.st_mode):
                    continue
                holding["files"] += 1
                holding["size"] += stat.st_size
                category = path.relative_to(READY).parts[0]
                if category in {"organized_sources", "scanned_sources", "intake_duplicates"}:
                    holding[category] += 1
    return {
        "to_process_images": count_images(TO_PROCESS, exclude_generated_dirs=False),
        "to_process_videos": count_videos(TO_PROCESS, exclude_generated_dirs=False),
        "legacy_videos": count_videos(LEGACY_VIDEO_INBOX, exclude_generated_dirs=False),
        "organized_images": original_person_total(original_counts),
        "person_original_images": original_person_total(original_counts),
        "person_folders": len(original_counts),
        "person_counts": original_counts,
        "person_videos": count_videos(PEOPLE),
        "nudity_images": nudity_count(),
        "ready_to_delete_files": holding["files"],
        "ready_to_delete_size": holding["size"],
        "organized_sources_files": holding["organized_sources"],
        "scanned_sources_files": holding["scanned_sources"],
        "intake_duplicates_files": holding["intake_duplicates"],
        "unassigned_no_face": count_images(
            SOURCE_REVIEW / "unassigned_intake" / "no_usable_face",
            exclude_generated_dirs=False,
        ),
        "unassigned_face_quality": count_images(
            SOURCE_REVIEW / "unassigned_intake" / "face_quality_review",
            exclude_generated_dirs=False,
        ),
        "unassigned_unknown_identity": count_images(
            SOURCE_REVIEW / "unassigned_intake" / "unknown_identity",
            exclude_generated_dirs=False,
        ),
        "unassigned_multi_face_review": count_images(
            SOURCE_REVIEW / "unassigned_intake" / "multi_face_review",
            exclude_generated_dirs=False,
        ),
        "unassigned_copy_failed": count_images(
            SOURCE_REVIEW / "unassigned_intake" / "copy_failed",
            exclude_generated_dirs=False,
        ),
        "unassigned_processing_failed": count_images(
            SOURCE_REVIEW / "unassigned_intake" / "processing_failed",
            exclude_generated_dirs=False,
        ),
        "unassigned_unreadable": count_files(
            SOURCE_REVIEW / "unassigned_intake" / "unreadable_image"
        ),
        "video_review_multiple_people": count_videos(
            SOURCE_REVIEW / "unassigned_intake" / "videos" / "multiple_people",
            exclude_generated_dirs=False,
        ),
        "video_review_unknown_identity": count_videos(
            SOURCE_REVIEW / "unassigned_intake" / "videos" / "unknown_identity",
            exclude_generated_dirs=False,
        ),
        "video_review_no_usable_face": count_videos(
            SOURCE_REVIEW / "unassigned_intake" / "videos" / "no_usable_face",
            exclude_generated_dirs=False,
        ),
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
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Saved daily checkpoint is unreadable; preserve it and repair before restarting: {STATE_FILE}") from error


def save_state(state: dict) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE_FILE.with_suffix(STATE_FILE.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2, sort_keys=True)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(STATE_FILE)
    from file_operations import sync_directory
    sync_directory(STATE_FILE.parent)


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
    return {"video-process", "process"}


def cleanup_holding_count() -> int:
    return count_files(SOURCE_REVIEW)


def step_list(batch_size: int) -> list[dict]:
    py = sys.executable
    return [
        {
            "name": "preflight",
            "desc": "Preflight folders, memory, disk, and process safety",
            "cmd": [py, str(SCRIPT_DIR / "preflight_check.py")],
            "mutates": False,
        },
        {
            "name": "video-process",
            "desc": "Identify people in videos and file after one recognized identity frame",
            "cmd": [
                py, str(SCRIPT_DIR / "video_batch_runner.py"),
                str(TO_PROCESS), str(SORTED),
            ],
            "heavy": True,
        },
        {
            "name": "process",
            "desc": "Process new inbox images in memory-bounded slices",
            "cmd": [
                py, str(SCRIPT_DIR / "image_batch_runner.py"),
                str(TO_PROCESS), str(SORTED),
                "--skip-output-cleanup",
                "--max-images-per-process", "5000",
                "--batch-size", str(batch_size),
                "--detect-workers", "1",
            ],
            "heavy": True,
        },
        {"name": "structure", "desc": "Normalize person folder structure",
         "cmd": [py, str(SCRIPT_DIR / "person_structure.py"), "--apply", "--quiet"]},
        {"name": "rename", "desc": "Normalize person filenames with simple stable names",
         "cmd": [py, str(SCRIPT_DIR / "rename_person_folder_files.py"), "--simple", "--apply", "--quiet", "--skip-manifest-promote"]},
        {"name": "advanced-dedupe", "desc": "Update one cached exact-duplicate and hardlink report",
         "cmd": [py, str(SCRIPT_DIR / "daily_duplicates.py")], "mutates": False},
        {"name": "cleanup-empty", "desc": "Move empty person folders to ready_to_delete",
         "cmd": [py, str(SCRIPT_DIR / "cleanup_empty_person_folders.py"), "--apply", "--quiet"]},
        {"name": "cache-rehydrate", "desc": "Refresh face cache after all file-moving cleanup",
         "cmd": [py, str(SCRIPT_DIR / "cache_tools.py"), "rehydrate", "--apply", "--batch-size", str(batch_size)], "heavy": True, "mutates": False},
        {"name": "unknown-triage", "desc": "Write unknown-cluster triage report",
         "cmd": [py, str(SCRIPT_DIR / "unknown_triage.py"), "--quiet"], "mutates": False},
        {"name": "integration-audit", "desc": "Verify final cross-script invariants",
         "cmd": [py, str(SCRIPT_DIR / "integration_audit.py")], "mutates": False},
        {"name": "status", "desc": "Print final dashboard",
         "cmd": [py, str(SCRIPT_DIR / "status_report.py")], "mutates": False},
    ]


SCOPED_STEPS = {"structure", "rename", "advanced-dedupe", "cleanup-empty", "cache-rehydrate"}


def inventory_policy() -> str:
    import sort_photos
    digest = hashlib.sha256(sort_photos.config_fingerprint().encode())
    for name in ("daily_inventory.py", "person_structure.py", "rename_person_folder_files.py",
                 "cleanup_empty_person_folders.py", "daily_duplicates.py",
                 "advanced_duplicate_matching.py", "analysis_index.py", "cache_tools.py"):
        digest.update((SCRIPT_DIR / name).read_bytes())
    return digest.hexdigest()


def prepare_inventory(state: dict, *, full: bool = False) -> dict:
    entries = daily_inventory.capture(PEOPLE)
    with daily_inventory.open_inventory(STATE_FILE.with_name("daily_inventory.sqlite3"), PEOPLE, inventory_policy()) as inventory:
        changes = inventory.changes(entries)
        previous = state.get("incremental", {})
        changes["full"] = bool(full or previous.get("full") or changes["full"])
        changes["people"] = sorted(set(changes["people"]) | set(previous.get("people", [])))
        inventory.checkpoint(state["run_id"], changes)
    state["incremental"] = {key: changes[key] for key in ("full", "reason", "people", "fingerprint")}
    state["incremental"].update({key + "_count": len(changes[key]) for key in ("added", "changed", "removed")})
    scope_path = SUMMARY_DIR / f"daily_run_{state['run_id']}_people.json"
    source_manifest.write_json_atomic(scope_path, changes["people"])
    state["incremental"]["scope_path"] = str(scope_path)
    save_state(state)
    return entries


def skip_reason(step: dict, state: dict, *, full: bool = False) -> str | None:
    name = step["name"]
    if name == "process" and not tree_contains_media(TO_PROCESS, IMAGE_EXTS):
        return "no new photos"
    if name == "video-process" and not (tree_contains_media(TO_PROCESS, VIDEO_EXTS)
                                       or tree_contains_media(LEGACY_VIDEO_INBOX, VIDEO_EXTS)):
        return "no new videos"
    plan = state.get("incremental", {})
    if name in SCOPED_STEPS and not plan.get("full", True) and not plan.get("people"):
        if name != "advanced-dedupe" or ADV_REPORT.exists():
            return "no changed people or folders"
    if name == "unknown-triage":
        import unknown_triage
        if not unknown_triage.DEFAULT_STATE.is_file():
            return "no legacy review state; Quick Review remains available"
        if not full and unknown_triage.report_current(unknown_triage.DEFAULT_STATE, unknown_triage.DEFAULT_OUTPUT_DIR):
            return "legacy review report is already current"
    if name == "status" and not full:
        return "included in the final summary"
    return None


def scoped_command(step: dict, state: dict) -> list[str]:
    command = list(step["cmd"])
    plan = state.get("incremental", {})
    if step["name"] in SCOPED_STEPS and plan and not plan["full"]:
        command.extend(["--people-file", plan["scope_path"]])
    return command


def worker_diagnostics(log_path: Path, offset: int) -> dict:
    if not log_path.exists():
        return {}
    counters = {}
    benchmark = []
    patterns = {"cached_files": r"Already cached candidate files:\s*(\d+)",
                "missing_files": r"Remaining candidate files:\s*(\d+)",
                "reused_detections": r"Verified detections reused:\s*(\d+)",
                "fingerprint_hits": r"Fingerprint cache hits:\s*(\d+)",
                "fingerprint_misses": r"Fingerprint cache misses:\s*(\d+)",
                "hash_hits": r"Indexed content hashes:\s*(\d+) reused",
                "hash_misses": r"Indexed content hashes:\s*\d+ reused; (\d+) newly hashed"}
    with log_path.open("rb") as handle:
        handle.seek(offset)
        for raw in handle:
            line = raw.decode("utf-8", errors="replace")
            for name, pattern in patterns.items():
                match = re.search(pattern, line)
                if match:
                    counters[name] = int(match.group(1))
            if "Safety benchmark reason:" in line or "Safety benchmark: reusing unchanged cached result" in line:
                benchmark.append(line.strip())
    return {"counters": counters, "benchmark": benchmark[-10:]}


def run_command(cmd: list[str], log_path: Path, *, verbose: bool = True, step_name: str = "") -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    run_id_value = ""
    if "daily_run_" in log_path.stem:
        run_id_value = log_path.stem.replace("daily_run_", "", 1)
    if run_id_value:
        env["PHOTO_PIPELINE_RUN_ID"] = f"daily_run_{run_id_value}"
    with log_path.open("ab") as log:
        log.write(f"\n$ {' '.join(cmd)}\n".encode("utf-8"))
        log.flush()
        from pipeline_writer import child_process_options
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, **child_process_options(),
                                bufsize=0, env=env)
        assert proc.stdout is not None
        selector = selectors.DefaultSelector()
        selector.register(proc.stdout, selectors.EVENT_READ)
        started_at = time.monotonic()
        last_flush = started_at
        command_name = Path(cmd[1] if len(cmd) > 1 else cmd[0]).name
        progress = CommandProgress(STEP_LABELS.get(step_name, command_name))
        lines = OutputLines(progress.consume)
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        try:
            while True:
                events = selector.select(timeout=1.0)
                if events:
                    chunk = os.read(proc.stdout.fileno(), 65536)
                    if not chunk:
                        break
                    log.write(chunk)
                    text = decoder.decode(chunk)
                    if verbose:
                        print(text, end="", flush=True)
                    else:
                        lines.feed(text)
                elif proc.poll() is not None:
                    break
                now = time.monotonic()
                if now - last_flush >= 1.0:
                    log.flush()
                    last_flush = now
                if not verbose:
                    progress.tick()
            tail = decoder.decode(b"", final=True)
            if verbose:
                print(tail, end="", flush=True)
            else:
                lines.feed(tail)
                lines.finish()
            log.flush()
            returncode = proc.wait()
            if not verbose:
                progress.finish(returncode)
            return returncode
        except KeyboardInterrupt:
            print("\nStopping this step safely; completed work remains saved.", flush=True)
            # Keep workers in the terminal's existing process group so Ctrl+C
            # continues to reach nested workers as well as the supervisor.
            for sig, timeout in ((signal.SIGINT, 10), (signal.SIGTERM, 5), (signal.SIGKILL, 5)):
                try:
                    proc.send_signal(sig)
                except ProcessLookupError:
                    break
                try:
                    proc.wait(timeout=timeout)
                    break
                except subprocess.TimeoutExpired:
                    continue
            return 130
        finally:
            selector.close()
            proc.stdout.close()
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()


def write_summary(path: Path, state: dict, before: dict, after: dict, status: str) -> None:
    summary = {
        "run_id": state["run_id"],
        "status": status,
        "started_at": state["started_at"],
        "finished_at": int(time.time()),
        "before": before,
        "after": after,
        "delta": {key: delta(after, before, key) for key, value in after.items() if isinstance(value, (int, float))},
        "steps": state["steps"],
        "memory": state.get("memory", {}),
        "source_guard": state.get("source_guard", {}),
        "timings_seconds": state.get("timings_seconds", {}),
        "safety_timings_seconds": state.get("safety_timings_seconds", {}),
        "incremental": state.get("incremental", {}),
        "worker_diagnostics": state.get("worker_diagnostics", {}),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)


def print_summary(before: dict, after: dict, summary_path: Path, *, verbose: bool = True) -> None:
    if not verbose:
        print("\nDaily run complete - original-file safety checks passed.")
        print(f"  Photo inbox: {before.get('to_process_images', 0):,} -> {after.get('to_process_images', 0):,} remaining")
        video_before = int(before.get("to_process_videos", 0)) + int(before.get("legacy_videos", 0))
        video_after = int(after.get("to_process_videos", 0)) + int(after.get("legacy_videos", 0))
        if video_before or video_after:
            print(f"  Video inbox: {video_before:,} -> {video_after:,} remaining")
        print(f"  Library photo entries: {before.get('person_original_images', 0):,} -> {after.get('person_original_images', 0):,}")
        unknown = int(after.get("unassigned_unknown_identity", 0))
        review_keys = ("unassigned_no_face", "unassigned_face_quality", "unassigned_multi_face_review")
        other_review = sum(int(after.get(key, 0)) for key in review_keys)
        problems = sum(int(after.get(key, 0)) for key in (
            "unassigned_copy_failed", "unassigned_processing_failed", "unassigned_unreadable"))
        print(f"  Waiting for review (including earlier runs): {unknown:,} unknown identity; {other_review:,} other image reviews")
        if problems:
            print(f"  Files needing technical attention: {problems:,} (copy, read or processing problems)")
        video_review = sum(int(after.get(key, 0)) for key in (
            "video_review_multiple_people", "video_review_unknown_identity", "video_review_no_usable_face"))
        if video_review:
            print(f"  Videos waiting for review (including earlier runs): {video_review:,}")
        if unknown:
            print("  Review unknown faces when ready: face unknown-review")
        if after.get("to_process_images", 0) or video_after:
            print("  Files remain in the inbox; check the log before starting another run.")
        print(f"  Full log: {summary_path.with_suffix('.log')}")
        print(f"  Saved summary: {summary_path}")
        return
    print()
    print("Daily Run Summary")
    print("=" * 60)
    print(f"New organized images:       {delta(after, before, 'organized_images')}")
    print(f"Original photos before/after:{before.get('person_original_images', 0)} -> {after.get('person_original_images', 0)}")
    print(f"To Process before/after:    {before['to_process_images']} -> {after['to_process_images']}")
    print(f"To Process videos moved:    {before.get('to_process_videos', 0)} -> {after.get('to_process_videos', 0)}")
    print(f"Legacy videos before/after: {before.get('legacy_videos', 0)} -> {after.get('legacy_videos', 0)}")
    print(f"New person videos:          {delta(after, before, 'person_videos')}")
    review_before = sum(
        int(before.get(key, 0)) for key in (
            "video_review_multiple_people",
            "video_review_unknown_identity",
            "video_review_no_usable_face",
        )
    )
    review_after = sum(
        int(after.get(key, 0)) for key in (
            "video_review_multiple_people",
            "video_review_unknown_identity",
            "video_review_no_usable_face",
        )
    )
    print(f"New videos needing review:  {review_after - review_before}")
    print(f"Nudity-folder image change: {delta(after, before, 'nudity_images')}")
    print(f"Archived organized sources: +{delta(after, before, 'organized_sources_files')}")
    print(f"Archived scanned sources:   +{delta(after, before, 'scanned_sources_files')}")
    print(f"Archived intake duplicates: +{delta(after, before, 'intake_duplicates_files')}")
    print(
        "Needs intake review:       "
        f"+{delta(after, before, 'unassigned_no_face') + delta(after, before, 'unassigned_face_quality') + delta(after, before, 'unassigned_unknown_identity') + delta(after, before, 'unassigned_multi_face_review') + delta(after, before, 'unassigned_copy_failed') + delta(after, before, 'unassigned_processing_failed') + delta(after, before, 'unassigned_unreadable')} "
        f"(no face +{delta(after, before, 'unassigned_no_face')}, "
        f"quality review +{delta(after, before, 'unassigned_face_quality')}, "
        f"unknown +{delta(after, before, 'unassigned_unknown_identity')}, "
        f"multi-face +{delta(after, before, 'unassigned_multi_face_review')}, "
        f"copy failed +{delta(after, before, 'unassigned_copy_failed')}, "
        f"processing failed +{delta(after, before, 'unassigned_processing_failed')}, "
        f"unreadable +{delta(after, before, 'unassigned_unreadable')})"
    )
    print(f"Unknown clusters/faces:     {after['unknown_clusters']} / {after['unknown_faces']}")
    print(f"Near-visual review items:   {after['near_visual_review']}")
    print(f"ready_to_delete size:       {human_size(after['ready_to_delete_size'])}")
    print(f"Summary JSON:               {summary_path}")


def print_dry_run(steps: list[dict], before: dict, profile: dict,
                  full_maintenance: bool) -> None:
    empty_inbox = (
        int(before.get("to_process_images", 0)) == 0
        and int(before.get("to_process_videos", 0)) == 0
        and int(before.get("legacy_videos", 0)) == 0
    )
    print("Daily Dry Run")
    print("=" * 60)
    print(f"Sorted folder:              {SORTED}")
    print(f"Input folder:               {TO_PROCESS}")
    print(f"To Process images:          {before['to_process_images']}")
    print(f"To Process videos:          {before.get('to_process_videos', 0)}")
    print(f"Legacy videos pending:      {before.get('legacy_videos', 0)}")
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
        would_skip = empty_inbox and (
            not full_maintenance or step["name"] in skip_when_empty
        )
        status = "skip: empty inbox" if would_skip else "run"
        if not would_skip:
            if step["name"] == "process" and not int(before.get("to_process_images", 0)):
                status = "skip: no new photos"
            elif step["name"] == "video-process" and not (int(before.get("to_process_videos", 0))
                                                        + int(before.get("legacy_videos", 0))):
                status = "skip: no new videos"
            elif step["name"] in SCOPED_STEPS and not full_maintenance:
                status = "changed folders only (full baseline on first run)"
            elif step["name"] == "unknown-triage":
                status = "only if saved legacy review labels changed"
            elif step["name"] == "status" and not full_maintenance:
                status = "skip: included in final summary"
        print(f"[{index}/{len(steps)}] {status:18} {step['desc']}")
        print(f"    {' '.join(step['cmd'])}")
    print()
    if empty_inbox and not full_maintenance:
        print("Daily would finish immediately because no new images or videos are waiting.")
        print("Use --full-maintenance to run library maintenance explicitly.")
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
    parser.add_argument("--verbose", action="store_true",
                        help="Show full worker diagnostics instead of compact progress; full logs are always saved.")
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

    state = load_state()
    resumed = state is not None
    if state is None and not args.full_maintenance and not intake_has_media():
        print("Daily Ingest")
        print("=" * 60)
        print("Already current: no new images or videos are waiting in To Process.")
        print("No library scan or maintenance was run.")
        print("Use --full-maintenance when you intentionally want a full maintenance pass.")
        return 0

    if state is None:
        profile = memory_profile()
        if not profile["ok"] and not args.ignore_low_memory:
            print(f"ERROR: {profile['message']} ({profile['available_mb']} MB available).")
            print("Close other apps, then re-run. Override with --ignore-low-memory.")
            return 2
        rid = run_id()
        print("Preparing library counts and checking existing originals...", flush=True)
        before_snapshot = snapshot()
        state = {
            "run_id": rid,
            "started_at": int(time.time()),
            "before": before_snapshot,
            "steps": {},
            "memory": profile,
            "source_guard": {
                "before": before_snapshot.get("person_counts") or original_person_counts(),
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
    if args.verbose:
        print(f"Memory mode: {profile['message']} ({profile.get('available_mb', 'unknown')} MB available), batch size {batch_size}.")
    elif batch_size < 50:
        print("Using smaller batches because available memory is low.", flush=True)
    guard = state.get("source_guard", {})
    print(f"Inbox at run start: {before.get('to_process_images', 0):,} photos, "
          f"{int(before.get('to_process_videos', 0)) + int(before.get('legacy_videos', 0)):,} videos.", flush=True)
    if args.verbose:
        print(f"Source guard baseline: {guard.get('before_total', original_person_total(guard.get('before', {})))} original photos "
              f"across {len(guard.get('before', {}))} person folders.")
    if args.verbose and guard.get("before_csv"):
        print(f"Source guard before CSV: {guard['before_csv']}")

    manifest_result = check_source_manifest(state, "start", verbose=args.verbose)
    ok, _, violations = check_source_guard(state, "start", counts=verified_original_counts(manifest_result))
    save_state(state)
    if not ok:
        paths = source_guard_paths(str(state["run_id"]))
        print("ERROR: source guard failed before running steps.")
        print(f"Violation report: {paths['violations']}")
        for row in violations[:10]:
            print(f"  {row['person']}: {row['before']} -> {row['after']} ({row['delta']})")
        return SOURCE_GUARD_EXIT

    if not manifest_result.ok:
        print("ERROR: protected source manifest failed before running steps.")
        print("Fix or recover the missing originals before cache/smart-album refresh can run.")
        return SOURCE_GUARD_EXIT

    # Reconcile external moves as well as this run's writes. Never publish this
    # inventory as a successful baseline until final original verification.
    inventory_entries = prepare_inventory(state, full=args.full_maintenance)
    if resumed:
        for name in SCOPED_STEPS:
            if state["steps"].get(name, {}).get("status") == "completed":
                state["steps"][name]["status"] = "needs_revalidation"
        save_state(state)

    log_path = SUMMARY_DIR / f"daily_run_{state['run_id']}.log"
    summary_path = SUMMARY_DIR / f"daily_run_{state['run_id']}.json"
    print(f"Full diagnostics are saved to: {log_path}", flush=True)
    steps = step_list(batch_size)
    for index, step in enumerate(steps, start=1):
        label = step["desc"] if args.verbose else STEP_LABELS[step["name"]]
        if step["name"] == "structure":
            inventory_entries = prepare_inventory(state, full=args.full_maintenance)
            plan = state["incremental"]
            print("  Maintenance: " + ("full safe baseline" if plan["full"] else
                  f"{len(plan['people'])} changed person folder(s) only"), flush=True)
        if step["name"] != "preflight" and state["steps"].get(step["name"], {}).get("status") == "completed":
            print(f"[{index}/{len(steps)}] {label} - already completed", flush=True)
            continue
        reason = skip_reason(step, state, full=args.full_maintenance)
        if reason:
            print(f"[{index}/{len(steps)}] {label} - skipped ({reason})", flush=True)
            state["steps"][step["name"]] = {
                "status": "skipped",
                "reason": reason,
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
        print(f"[{index}/{len(steps)}] {label}", flush=True)
        started_at = int(time.time())
        state["steps"][step["name"]] = {"status": "running", "started_at": started_at}
        save_state(state)
        step_started = time.perf_counter()
        log_offset = log_path.stat().st_size if log_path.exists() else 0
        rc = run_command(scoped_command(step, state), log_path, verbose=args.verbose, step_name=step["name"])
        elapsed = time.perf_counter() - step_started
        state.setdefault("worker_diagnostics", {})[step["name"]] = worker_diagnostics(log_path, log_offset)
        state.setdefault("timings_seconds", {})[step["name"]] = round(elapsed, 3)
        if args.verbose:
            print(f"Step timing: {step['name']} {elapsed:.1f}s (exit {rc})", flush=True)
        if rc != 0:
            state["steps"][step["name"]] = {
                "status": "failed",
                "started_at": started_at,
                "worker_seconds": round(elapsed, 3),
                "returncode": rc,
                "finished_at": int(time.time()),
            }
            save_state(state)
            after = snapshot()
            write_summary(summary_path, state, before, after, "failed")
            print(f"Stopped: {label} (exit {rc}). Completed steps remain saved.")
            print("Resume with: face daily --resume")
            print(f"Full error details: {log_path}")
            return rc
        safety_started = time.perf_counter()
        mutates = step.get("mutates", True)
        if not args.verbose and mutates:
            print("  Checking that originals are preserved...", flush=True)
        manifest_result = (check_source_manifest(state, step["name"], verbose=args.verbose) if mutates else None)
        ok, guarded_after, violations = (check_source_guard(state, step["name"],
            counts=verified_original_counts(manifest_result)) if mutates else (True, {}, []))
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
        save_state(state)
        if manifest_result is not None and not manifest_result.ok:
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
        safety_seconds = time.perf_counter() - safety_started
        state.setdefault("safety_timings_seconds", {})[step["name"]] = round(safety_seconds, 3)
        state["steps"][step["name"]] = {"status": "completed", "started_at": started_at,
            "finished_at": int(time.time()), "worker_seconds": round(elapsed, 3),
            "safety_seconds": round(safety_seconds, 3)}
        if step["name"] == "cache-rehydrate":
            inventory_entries = daily_inventory.capture(PEOPLE)
        save_state(state)
        if not args.verbose:
            print(f"  Done ({duration(time.perf_counter() - step_started)}, including safety checks).", flush=True)

    print("Finalizing library counts and safety checks...", flush=True)
    after = snapshot()
    manifest_result = check_source_manifest(state, "completed_before_promote", verbose=args.verbose)
    ok, _, violations = check_source_guard(state, "completed", counts=verified_original_counts(manifest_result))
    if not ok:
        save_state(state)
        write_summary(summary_path, state, before, after, "failed_source_count_guard")
        paths = source_guard_paths(str(state["run_id"]))
        print("ERROR: source guard failed at final validation.")
        print(f"Violation report CSV: {paths['violations']}")
        for row in violations[:10]:
            print(f"  {row['person']}: {row['before']} -> {row['after']} ({row['delta']})")
        return SOURCE_GUARD_EXIT
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
    with daily_inventory.open_inventory(STATE_FILE.with_name("daily_inventory.sqlite3"), PEOPLE, inventory_policy()) as inventory:
        inventory.publish(state["run_id"], inventory_entries,
                          None if state["incremental"]["full"] else set(state["incremental"]["people"]))
    save_state(state)
    write_summary(summary_path, state, before, after, "completed")
    print_summary(before, after, summary_path, verbose=args.verbose)
    clear_state()
    return 0


if __name__ == "__main__":
    import pipeline_writer
    main = pipeline_writer.serialized(main)
    raise SystemExit(main())

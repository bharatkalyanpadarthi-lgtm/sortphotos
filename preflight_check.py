#!/usr/bin/env python3
"""
Preflight health checks for the photo sorting pipeline.

This is intentionally read-only. It checks the folders, cache files, free disk
space, duplicate face.py/sort_photos.py processes, and memory readiness.
"""

from __future__ import annotations

import os
import pickle
import shlex
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pipeline_paths

SORTED = pipeline_paths.SORTED_ROOT
PEOPLE = SORTED / "photos_by_person"
SOURCE_REVIEW = SORTED / "_source_review"
TO_PROCESS = pipeline_paths.TO_PROCESS
FACE_REFS = pipeline_paths.FACE_REFERENCES
CACHE_DIR = Path.home() / ".face_sort_cache"
CACHE_FILES = {
    "face cache": CACHE_DIR / "cache.pkl",
    "identity DB": CACHE_DIR / "person_identity_db.pkl",
    "fingerprint cache": CACHE_DIR / "advanced_duplicate_fingerprints.json",
}
MIN_FREE_GB = 20

CONFLICTING_PROCESS_LABELS = {
    "face.py": "Face command",
    "daily_runner.py": "Daily Ingest",
    "sort_photos.py": "photo sorting worker",
    "review_unknown_identities.py": "Unknown Identity Quick Review",
}

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import sort_photos  # noqa: E402
    for _name in ("CacheState", "CachedFace", "FaceRecord", "LabelingState", "IdentityDB"):
        if hasattr(sort_photos, _name):
            setattr(sys.modules["__main__"], _name, getattr(sort_photos, _name))
except Exception:
    sort_photos = None


@dataclass
class Check:
    level: str
    name: str
    detail: str


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def available_memory_mb() -> int | None:
    try:
        page_size = int(subprocess.check_output(["sysctl", "-n", "hw.pagesize"], text=True).strip())
        output = subprocess.check_output(["vm_stat"], text=True)
    except Exception:
        return None
    pages: dict[str, int] = {}
    for line in output.splitlines():
        if ":" not in line:
            continue
        name, raw = line.split(":", 1)
        try:
            pages[name] = int(raw.strip().strip(".").replace(".", ""))
        except ValueError:
            continue
    freeish = (
        pages.get("Pages free", 0)
        + pages.get("Pages inactive", 0)
        + pages.get("Pages speculative", 0)
    )
    return int((freeish * page_size) / (1024 * 1024))


def folder_check(path: Path, name: str, must_exist: bool = True) -> Check:
    if path.exists():
        return Check("OK", name, str(path))
    level = "FAIL" if must_exist else "WARN"
    return Check(level, name, f"missing: {path}")


def source_review_check(path: Path) -> Check:
    if path.is_symlink():
        target = path.resolve(strict=False)
        if not path.exists():
            return Check("FAIL", "_source_review", f"external target unavailable: {target}")
        if not os.access(path, os.R_OK | os.W_OK):
            return Check("FAIL", "_source_review", f"external target is not readable/writable: {target}")
        return Check("OK", "_source_review", f"external storage: {target}")
    return folder_check(path, "_source_review")


def free_space_check(path: Path) -> Check:
    root = path if path.exists() else path.parent
    while not root.exists() and root != root.parent:
        root = root.parent
    try:
        usage = shutil.disk_usage(root)
    except OSError as exc:
        return Check("WARN", "free disk space", f"could not check {root}: {exc}")
    free_gb = usage.free / (1024 ** 3)
    level = "OK" if free_gb >= MIN_FREE_GB else "WARN"
    return Check(level, "free disk space", f"{free_gb:.1f} GB free on {root}")


def cache_check(name: str, path: Path) -> Check:
    if not path.exists():
        return Check("WARN", name, f"missing: {path}")
    try:
        if path.suffix == ".pkl":
            with path.open("rb") as f:
                pickle.load(f)
        else:
            with path.open("rb") as f:
                f.read(128)
    except Exception as exc:
        return Check("FAIL", name, f"not readable: {path} ({exc})")
    return Check("OK", name, str(path))


def _process_script(command: str) -> str:
    try:
        arguments = shlex.split(command)
    except ValueError:
        arguments = command.split()
    if arguments:
        direct_name = Path(arguments[0]).name
        if direct_name in CONFLICTING_PROCESS_LABELS:
            return direct_name
    executable_index = 0
    if arguments and Path(arguments[0]).name == "env":
        executable_index = next(
            (index for index, value in enumerate(arguments[1:], 1) if not value.startswith("-")),
            len(arguments),
        )
    if executable_index >= len(arguments):
        return ""
    if not Path(arguments[executable_index]).name.startswith("python"):
        return ""
    for argument in arguments[executable_index + 1:]:
        name = Path(argument).name
        if name in CONFLICTING_PROCESS_LABELS:
            return name
    return ""


def process_conflicts(process_output: str, *, current_pid: int) -> list[str]:
    rows: dict[int, tuple[int, str, str]] = {}
    for line in process_output.splitlines():
        parts = line.strip().split(None, 2)
        if len(parts) < 3:
            continue
        try:
            pid = int(parts[0])
            ppid = int(parts[1])
        except ValueError:
            continue
        command = parts[2]
        rows[pid] = (ppid, command, _process_script(command))

    parent_by_pid: dict[int, int] = {}
    for pid, (ppid, _command, _script) in rows.items():
        parent_by_pid[pid] = ppid

    ancestors = {current_pid}
    pid = parent_by_pid.get(current_pid, 0)
    while pid and pid not in ancestors:
        ancestors.add(pid)
        pid = parent_by_pid.get(pid, 0)

    candidates = {
        pid: (ppid, command, script)
        for pid, (ppid, command, script) in rows.items()
        if pid not in ancestors and script
    }

    def has_active_child(parent_pid: int) -> bool:
        for candidate_pid in candidates:
            pid = parent_by_pid.get(candidate_pid, 0)
            seen: set[int] = set()
            while pid and pid not in seen:
                if pid == parent_pid:
                    return True
                seen.add(pid)
                pid = parent_by_pid.get(pid, 0)
        return False

    matches: list[str] = []
    for pid, (_ppid, command, script) in sorted(candidates.items()):
        if script == "face.py" and has_active_child(pid):
            continue
        label = CONFLICTING_PROCESS_LABELS[script]
        matches.append(f"{pid}: {label} ({command})")
    return matches


def process_check() -> Check:
    try:
        output = subprocess.check_output(
            ["ps", "-axo", "pid=,ppid=,command="], text=True
        )
    except Exception as exc:
        return Check("WARN", "duplicate running process", f"could not inspect processes: {exc}")
    matches = process_conflicts(output, current_pid=os.getpid())
    if matches:
        return Check("FAIL", "duplicate running process", "; ".join(matches[:3]))
    return Check("OK", "duplicate running process", "no other face pipeline process found")


def memory_check() -> Check:
    mb = available_memory_mb()
    if mb is None:
        return Check("WARN", "available memory", "could not read vm_stat")
    if mb < 1200:
        return Check("FAIL", "available memory", f"{mb} MB available; close apps before heavy scans")
    if mb < 3500:
        return Check("WARN", "available memory", f"{mb} MB available; daily will use low-memory batch size")
    return Check("OK", "available memory", f"{mb} MB available")


def print_checks(checks: list[Check]) -> None:
    width = max(len(c.name) for c in checks)
    for c in checks:
        print(f"[{c.level:4}] {c.name:<{width}}  {c.detail}")


def main() -> int:
    checks = [
        folder_check(SORTED, "sorted_all_pictures"),
        folder_check(PEOPLE, "photos_by_person"),
        source_review_check(SOURCE_REVIEW),
        folder_check(TO_PROCESS, "To Process", must_exist=False),
        folder_check(FACE_REFS, "Face References", must_exist=False),
        free_space_check(SORTED),
        memory_check(),
        process_check(),
    ]
    checks.extend(cache_check(name, path) for name, path in CACHE_FILES.items())
    checks.append(Check("OK", "smart album state", "disabled / not required"))

    print("Photo Pipeline Preflight")
    print("=" * 60)
    print_checks(checks)
    failures = [c for c in checks if c.level == "FAIL"]
    warnings = [c for c in checks if c.level == "WARN"]
    print()
    print(f"Result: {len(failures)} failure(s), {len(warnings)} warning(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

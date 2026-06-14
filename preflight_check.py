#!/usr/bin/env python3
"""
Preflight health checks for the photo sorting pipeline.

This is intentionally read-only. It checks the folders, cache files, free disk
space, duplicate face.py/sort_photos.py processes, and memory readiness.
"""

from __future__ import annotations

import os
import pickle
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
PEOPLE = SORTED / "photos_by_person"
SOURCE_REVIEW = SORTED / "_source_review"
TO_PROCESS = Path.home() / "Pictures" / "To Process"
FACE_REFS = Path.home() / "Pictures" / "Face References"
CACHE_DIR = Path.home() / ".face_sort_cache"
CACHE_FILES = {
    "face cache": CACHE_DIR / "cache.pkl",
    "identity DB": CACHE_DIR / "person_identity_db.pkl",
    "fingerprint cache": CACHE_DIR / "advanced_duplicate_fingerprints.json",
    "smart album state": CACHE_DIR / "smart_album_person_state.json",
}
MIN_FREE_GB = 20

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


def process_check() -> Check:
    try:
        output = subprocess.check_output(["ps", "-axo", "pid=,command="], text=True)
    except Exception as exc:
        return Check("WARN", "duplicate running process", f"could not inspect processes: {exc}")
    current = os.getpid()
    parent_by_pid: dict[int, int] = {}
    try:
        parent_output = subprocess.check_output(["ps", "-axo", "pid=,ppid=,command="], text=True)
        for line in parent_output.splitlines():
            parts = line.strip().split(None, 2)
            if len(parts) < 3:
                continue
            pid = int(parts[0])
            parent_by_pid[pid] = int(parts[1])
    except Exception:
        parent_by_pid = {}

    ancestors = {current}
    pid = os.getppid()
    while pid and pid not in ancestors:
        ancestors.add(pid)
        pid = parent_by_pid.get(pid, 0)

    matches = []
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        pid_raw, _, command = line.partition(" ")
        try:
            pid = int(pid_raw)
        except ValueError:
            continue
        if pid in ancestors:
            continue
        if "face.py" in command or "sort_photos.py" in command or "daily_runner.py" in command:
            if "preflight_check.py" not in command:
                matches.append(f"{pid}: {command}")
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
        folder_check(SOURCE_REVIEW, "_source_review"),
        folder_check(TO_PROCESS, "To Process", must_exist=False),
        folder_check(FACE_REFS, "Face References", must_exist=False),
        free_space_check(SORTED),
        memory_check(),
        process_check(),
    ]
    checks.extend(cache_check(name, path) for name, path in CACHE_FILES.items())

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

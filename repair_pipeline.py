#!/usr/bin/env python3
"""
Run a read-only repair/audit pass for the photo sorting pipeline.

Default mode does not move or delete files. With --apply, it performs only
safe rebuild actions: rebuild identity DB, normalize names, and refresh review
reports. Destructive cleanup still stays in dry-run.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOG_DIR = Path.home() / "Pictures" / "sorted_all_pictures" / "_source_review" / "repair_logs"


def run_step(name: str, cmd: list[str], log) -> int:
    print()
    print(f"{name}")
    print("-" * 60)
    print(" ".join(cmd))
    log.write(f"\n## {name}\n$ {' '.join(cmd)}\n")
    log.flush()
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        print(line, end="")
        log.write(line)
    rc = proc.wait()
    log.write(f"\nexit={rc}\n")
    log.flush()
    return rc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--apply", action="store_true",
                        help="Run safe rebuild/refresh steps. Destructive cleanup remains dry-run.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Run all steps even if one fails.")
    args = parser.parse_args()

    py = sys.executable
    if args.apply:
        steps = [
            ("Preflight", [py, str(SCRIPT_DIR / "preflight_check.py")]),
            ("Integration audit", [py, str(SCRIPT_DIR / "integration_audit.py")]),
            ("Cache validation", [py, str(SCRIPT_DIR / "validate_cache.py")]),
            ("Face cache status", [py, str(SCRIPT_DIR / "cache_tools.py"), "status"]),
            ("Person structure repair", [py, str(SCRIPT_DIR / "person_structure.py"), "--apply", "--quiet"]),
            ("Rebuild identity DB", [py, str(SCRIPT_DIR / "sort_photos.py"), "--identity-db-only", "--identity-max-images", "80"]),
            ("Identity audit", [py, str(SCRIPT_DIR / "identity_audit.py")]),
            ("Normalize person filenames", [py, str(SCRIPT_DIR / "rename_person_folder_files.py"), "--apply", "--quiet"]),
            ("Exact duplicate dry-run", [py, str(SCRIPT_DIR / "delete_person_folder_duplicates.py"), "--quiet"]),
            ("Advanced duplicate report", [py, str(SCRIPT_DIR / "advanced_duplicate_matching.py"), "--quiet"]),
            ("Unknown triage", [py, str(SCRIPT_DIR / "unknown_triage.py"), "--quiet"]),
            ("Review dashboard", [py, str(SCRIPT_DIR / "review_dashboard.py"), "--no-refresh"]),
            ("Final status", [py, str(SCRIPT_DIR / "status_report.py")]),
        ]
    else:
        steps = [
            ("Preflight", [py, str(SCRIPT_DIR / "preflight_check.py")]),
            ("Integration audit", [py, str(SCRIPT_DIR / "integration_audit.py")]),
            ("Cache validation", [py, str(SCRIPT_DIR / "validate_cache.py")]),
            ("Face cache status", [py, str(SCRIPT_DIR / "cache_tools.py"), "status"]),
            ("Person structure audit", [py, str(SCRIPT_DIR / "person_structure.py"), "--quiet"]),
            ("Identity audit", [py, str(SCRIPT_DIR / "identity_audit.py")]),
            ("Exact duplicate dry-run", [py, str(SCRIPT_DIR / "delete_person_folder_duplicates.py"), "--quiet"]),
            ("Advanced duplicate report dry-run", [py, str(SCRIPT_DIR / "advanced_duplicate_matching.py"), "--quiet"]),
            ("Unknown triage", [py, str(SCRIPT_DIR / "unknown_triage.py"), "--quiet"]),
            ("Review dashboard", [py, str(SCRIPT_DIR / "review_dashboard.py"), "--no-refresh"]),
            ("Final status", [py, str(SCRIPT_DIR / "status_report.py")]),
        ]

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / f"repair_{time.strftime('%Y%m%d_%H%M%S')}.log"
    failures: list[tuple[str, int]] = []
    with log_path.open("w", encoding="utf-8") as log:
        for name, cmd in steps:
            rc = run_step(name, cmd, log)
            if rc != 0:
                failures.append((name, rc))
                if not args.continue_on_error:
                    break

    print()
    print("Repair Summary")
    print("=" * 60)
    print(f"Mode: {'apply safe rebuilds' if args.apply else 'read-only audit'}")
    print(f"Log:  {log_path}")
    if failures:
        for name, rc in failures:
            print(f"FAIL: {name} exited {rc}")
        return failures[0][1]
    print("All repair/audit steps completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

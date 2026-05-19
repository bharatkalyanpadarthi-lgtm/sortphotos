"""
face.py — Simple launcher for the photo sorting pipeline.

Use the first option for normal day-to-day work: dump new images into
~/Pictures/To Process, then run `python face.py process`.

Run:
    python face.py             # show compact menu
    python face.py process     # recommended inbox-only unattended workflow
    python face.py review      # optional: label only larger unknown clusters
    python face.py finish      # finalize labels you already entered
    python face.py status      # quick dashboard
    python face.py refs        # rebuild optional Face References DB
    python face.py health      # validate cache and duplicate status
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ACTIONS = [
    {
        "key": "process",
        "aliases": ["process-new", "process-move", "sort"],
        "label": "Process New Photos",
        "desc": "Fast daily run: scan only ~/Pictures/To Process, organize known people, then empty scanned inbox files to ready_to_delete",
        "script": "sort_photos.py",
        "args": [
            str(Path.home() / "Pictures" / "To Process"),
            str(Path.home() / "Pictures" / "sorted_all_pictures"),
            "--unattended",
            "--archive-organized-sources",
            "--archive-sources-to-ready-delete",
            "--archive-scanned-sources",
            "--detect-workers", "2",
        ],
    },
    {
        "key": "process-all",
        "aliases": ["scan-all-pictures"],
        "label": "Process All Pictures",
        "desc": "Full scan of ~/Pictures. Slower; use only when old source folders must be swept again",
        "script": "sort_photos.py",
        "args": [
            "--unattended",
            "--archive-organized-sources",
            "--archive-sources-to-ready-delete",
            "--detect-workers", "2",
        ],
    },
    {
        "key": "review",
        "aliases": ["fast", "resume"],
        "label": "Review Important Unknowns",
        "desc": "Optional: manually label only larger unknown clusters; press q anytime, then run Finish",
        "script": "sort_photos.py",
        "args": ["--resume-label", "--fast", "--min-label-cluster-size", "20"],
    },
    {
        "key": "finish",
        "label": "Finish Entered Labels",
        "desc": "Finalize labels already entered without asking for more manual labeling",
        "script": "sort_photos.py",
        "args": ["--finish-labeled"],
    },
    {
        "key": "fix",
        "label": "Fix Mistakes",
        "desc": "Rename, merge, or split person folders only when a person folder is wrong",
        "script": "fix_clusters.py",
    },
    {
        "key": "status",
        "label": "Status Dashboard",
        "desc": "Quick counts for organized photos, pending labels, duplicates, and ready-to-delete",
        "script": "status_report.py",
    },
    {
        "key": "rebuild-id",
        "aliases": ["build-id"],
        "label": "Rebuild Identity DB",
        "desc": "Maintenance: relearn known people from photos_by_person after many manual folder edits",
        "script": "sort_photos.py",
        "args": ["--identity-db-only"],
        "hidden": True,
    },
    {
        "key": "refs",
        "aliases": ["references", "build-refs"],
        "label": "Build Face References",
        "desc": "Maintenance: build optional AI/reference matching DB from ~/Pictures/Face References",
        "script": "build_celeb_centroids.py",
        "hidden": True,
    },
    {
        "key": "health",
        "aliases": ["validate", "dedupe", "optimize", "cleanup"],
        "label": "Health Check",
        "desc": "Validate cache and check duplicate status without moving files",
        "steps": [
            {"script": "validate_cache.py"},
            {"script": "delete_person_folder_duplicates.py"},
            {"script": "advanced_duplicate_matching.py", "args": ["--quiet"]},
        ],
    },
]


def show_menu() -> dict | None:
    print()
    print("=" * 60)
    print("  Photo Sorting Pipeline")
    print("=" * 60)
    print()
    visible_actions = [a for a in ACTIONS if not a.get("hidden")]
    for i, action in enumerate(visible_actions, start=1):
        print(f"  [{i}] {action['label']}")
        print(f"      {action['desc']}")
        print()
    print(f"  [q] Quit")
    print()
    while True:
        try:
            ans = input("  Choose: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if ans in ("q", "quit", "exit"):
            return None
        if not ans:
            continue
        # Numeric
        if ans.isdigit():
            idx = int(ans) - 1
            if 0 <= idx < len(visible_actions):
                return visible_actions[idx]
        # Keyword
        for a in ACTIONS:
            if a["key"] == ans:
                return a
        print(f"  Unknown choice: {ans}")


def find_action_by_key(key: str) -> dict | None:
    for a in ACTIONS:
        if a["key"] == key or key in a.get("aliases", []):
            return a
    return None


def run_action(action: dict, extra_args: list[str] | None = None) -> int:
    steps = action.get("steps") or [action]
    try:
        for i, step in enumerate(steps, start=1):
            script_path = SCRIPT_DIR / step["script"]
            if not script_path.exists():
                print(f"ERROR: missing script {script_path}")
                return 1
            cmd = [sys.executable, str(script_path)] + step.get("args", [])
            if extra_args and len(steps) == 1:
                cmd.extend(extra_args)
            prefix = f"[{i}/{len(steps)}] " if len(steps) > 1 else ""
            print(f"\n→ {prefix}Running: {' '.join(cmd)}\n")
            result = subprocess.run(cmd)
            if result.returncode != 0:
                return result.returncode
        return 0
    except KeyboardInterrupt:
        return 130


def main() -> int:
    # Shortcut form: `python face.py sort` etc.
    if len(sys.argv) > 1:
        key = sys.argv[1].lower()
        if key in ("-h", "--help", "help"):
            print(__doc__)
            return 0
        action = find_action_by_key(key)
        if action is None:
            print(f"Unknown action: {key}")
            print(f"Available: {', '.join(a['key'] for a in ACTIONS)}")
            return 1
        return run_action(action, sys.argv[2:])

    # Interactive menu
    action = show_menu()
    if action is None:
        return 0
    return run_action(action)


if __name__ == "__main__":
    sys.exit(main())

"""
face.py — Simple launcher for the photo sorting pipeline.

Use the first option for normal day-to-day work: dump new images into
~/Pictures/To Process, then run `python face.py daily`.

Run:
    python face.py             # show compact menu
    python face.py daily       # full daily end-to-end workflow
    python face.py process     # recommended inbox-only unattended workflow
    python face.py review      # optional: label only larger unknown clusters
    python face.py finish      # finalize labels you already entered
    python face.py status      # quick dashboard
    python face.py refs        # rebuild optional Face References DB
    python face.py clean-refs  # clean/compact Face References then rebuild
    python face.py nudity      # scan sorted people folders for nudity
    python face.py rename      # name/number files inside person folders
    python face.py smart-albums # create hardlinked smart album views
    python face.py people-cleanup # apply reusable person-folder merge/rename/remove rules
    python face.py identity-audit # compare identity DB to current person folders
    python face.py backup-review # back up _source_review to external drive
    python face.py health      # validate cache and duplicate status
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent

ACTIONS = [
    {
        "key": "daily",
        "aliases": ["run", "go", "end-to-end"],
        "label": "Daily End-to-End Run",
        "desc": "Process new inbox images, place nudity, rename, dedupe, rebuild changed smart albums, then show status",
        "steps": [
            {
                "script": "sort_photos.py",
                "args": [
                    str(Path.home() / "Pictures" / "To Process"),
                    str(Path.home() / "Pictures" / "sorted_all_pictures"),
                    "--unattended",
                    "--archive-organized-sources",
                    "--archive-sources-to-ready-delete",
                    "--archive-scanned-sources",
                    "--batch-size", "50",
                    "--detect-workers", "1",
                ],
            },
            {"script": "separate_nudity_review.py", "args": ["--apply", "--quiet"]},
            {"script": "place_nudity_inside_person_folders.py", "args": ["--apply", "--remove-review-copies", "--quiet"]},
            {"script": "rename_person_folder_files.py", "args": ["--apply", "--quiet"]},
            {"script": "delete_person_folder_duplicates.py", "args": ["--apply", "--quiet"]},
            {"script": "advanced_duplicate_matching.py", "args": ["--apply", "--quiet"]},
            {"script": "build_smart_albums.py", "args": ["--apply", "--incremental"]},
            {"script": "status_report.py"},
        ],
    },
    {
        "key": "process",
        "aliases": ["process-new", "process-move", "sort"],
        "label": "Process New Photos",
        "desc": "Fast daily run: scan ~/Pictures/To Process, organize known people, run nudity placement, then empty scanned inbox files",
        "script": "sort_photos.py",
        "args": [
            str(Path.home() / "Pictures" / "To Process"),
            str(Path.home() / "Pictures" / "sorted_all_pictures"),
            "--unattended",
            "--archive-organized-sources",
            "--archive-sources-to-ready-delete",
            "--archive-scanned-sources",
            "--batch-size", "50",
            "--detect-workers", "1",
        ],
    },
    {
        "key": "process-all",
        "aliases": ["scan-all-pictures"],
        "label": "Process All Pictures",
        "desc": "Full scan of ~/Pictures with nudity placement. Slower; use only when old source folders must be swept again",
        "script": "sort_photos.py",
        "args": [
            "--unattended",
            "--archive-organized-sources",
            "--archive-sources-to-ready-delete",
            "--batch-size", "50",
            "--detect-workers", "1",
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
        "key": "nudity",
        "aliases": ["nudity-check", "scan-nudity"],
        "label": "Run Nudity Check",
        "desc": "Scan all sorted person folders and move flagged images into each person's nudity subfolders",
        "steps": [
            {
                "script": "separate_nudity_review.py",
                "args": ["--apply", "--quiet"],
            },
            {
                "script": "place_nudity_inside_person_folders.py",
                "args": ["--apply", "--remove-review-copies", "--quiet"],
            },
        ],
    },
    {
        "key": "rename",
        "aliases": ["number", "number-files", "rename-files"],
        "label": "Rename Person Files",
        "desc": "Name and number images inside sorted person folders as Person_001, Person_002, etc.",
        "script": "rename_person_folder_files.py",
        "args": ["--apply", "--quiet"],
    },
    {
        "key": "smart-albums",
        "aliases": ["albums", "smart", "organize-smart"],
        "label": "Build Smart Albums",
        "desc": "Create hardlinked smart views for best, quality, framing, format, same-scene, visual-similar, nudity, and review folders",
        "script": "build_smart_albums.py",
        "args": ["--apply", "--incremental"],
    },
    {
        "key": "people-cleanup",
        "aliases": ["folder-cleanup", "person-cleanup"],
        "label": "Person Folder Cleanup",
        "desc": "Apply reusable merge/rename/remove rules for photos_by_person. Add --apply after dry-run review",
        "script": "person_folder_cleanup.py",
    },
    {
        "key": "identity-audit",
        "aliases": ["audit-id", "audit-identities"],
        "label": "Identity Audit",
        "desc": "Check whether the identity DB matches current person folders after cleanup or renames",
        "script": "identity_audit.py",
    },
    {
        "key": "backup-review",
        "aliases": ["backup", "backup-source-review"],
        "label": "Backup Source Review",
        "desc": "Sync _source_review to /Volumes/Photos & Videos  Backup and verify counts",
        "script": "backup_source_review.py",
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
        "args": ["--max-per-person", "20"],
        "hidden": True,
    },
    {
        "key": "clean-refs",
        "aliases": ["clean-references", "optimize-refs"],
        "label": "Clean Face References",
        "desc": "Keep best reference images per person, move duplicates/extras to review, then rebuild references",
        "script": "clean_face_references.py",
        "args": ["--apply", "--rebuild", "--quiet"],
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

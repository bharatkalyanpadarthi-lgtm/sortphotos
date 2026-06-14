"""
face.py — Simple launcher for the photo sorting pipeline.

Use the first option for normal day-to-day work: dump new images into
~/Pictures/To Process, then run `python face.py daily`.

Daily:
    python face.py daily       # safe intake/cache workflow for ~/Pictures/To Process
    python face.py dry-run     # preview the daily workflow
    python face.py status      # quick dashboard
    python face.py health      # read-only safety checks
    python face.py scrap-smart # remove generated smart album views safely

Useful manual tools:
    python face.py review-dashboard
    python face.py review
    python face.py nudity
    python face.py repair

Advanced commands remain available by name. Run `python face.py` and type `?`
to see every command keyword.
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import time
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import daily_runner  # noqa: E402
import source_manifest  # noqa: E402

SOURCE_GUARD_EXIT = 3

MENU_GROUPS = [
    (
        "Daily",
        [
            "daily",
            "dry-run",
            "status",
            "health",
        ],
    ),
    (
        "Review",
        [
            "review-dashboard",
            "review",
            "finish",
            "duplicate-review",
        ],
    ),
    (
        "Maintenance",
        [
            "nudity",
            "scrap-smart-albums",
            "repair",
            "integration-audit",
        ],
    ),
]

ADVANCED_MENU_KEYS = {
    "all-views",
    "bad-images",
    "cache-rehydrate",
    "cache-relink",
    "cache-status",
    "cleanup-empty",
    "clean-refs",
    "fix",
    "identity-audit",
    "nudity-confirm",
    "people-cleanup",
    "process",
    "process-all",
    "recover-bad-images",
    "recover-old-cache",
    "refs",
    "rename",
    "rebuild-id",
    "structure",
    "synthetic-tests",
    "unknown-triage",
}

ACTIONS = [
    {
        "key": "daily",
        "aliases": ["run", "go", "end-to-end"],
        "label": "Daily Ingest / Cache Run",
        "desc": "Memory-safe resumable daily ingest with cleanup, cache refresh, audit, and summary",
        "script": "daily_runner.py",
    },
    {
        "key": "dry-run",
        "aliases": ["preview", "daily-dry-run"],
        "label": "Preview Daily Run",
        "desc": "Show exactly what daily would scan, skip, and move without changing files",
        "script": "daily_runner.py",
        "args": ["--dry-run"],
    },
    {
        "key": "process",
        "aliases": ["process-new", "process-move", "sort"],
        "label": "Process New Photos",
        "desc": "Alias for Daily Ingest / Cache Run so new-photo processing uses the safe workflow",
        "script": "daily_runner.py",
    },
    {
        "key": "process-all",
        "aliases": ["scan-all-pictures"],
        "label": "Process All Pictures",
        "desc": "Full scan of ~/Pictures without automatic nudity moves. Slower; use only when old source folders must be swept again",
        "script": "sort_photos.py",
        "args": [
            "--unattended",
            "--archive-organized-sources",
            "--archive-sources-to-ready-delete",
            "--no-nudity-sort",
            "--skip-output-cleanup",
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
        "desc": "Optional manual command: scan people folders and move detector hits into per-person photos/nude folders",
        "steps": [
            {
                "script": "quarantine_bad_person_images.py",
                "args": ["--apply", "--quiet"],
            },
            {
                "script": "separate_nudity_review.py",
                "args": ["--apply", "--quiet"],
            },
            {
                "script": "place_nudity_inside_person_folders.py",
                "args": ["--apply", "--remove-review-copies", "--quiet"],
            },
            {
                "script": "rename_person_folder_files.py",
                "args": ["--apply", "--quiet"],
            },
        ],
    },
    {
        "key": "bad-images",
        "aliases": ["quarantine-bad", "bad-person-images", "clean-bad-images"],
        "label": "Quarantine Bad Image Files",
        "desc": "Move unreadable .jpg/.png recovery artifacts out of person folders",
        "steps": [
            {
                "script": "quarantine_bad_person_images.py",
                "args": ["--apply", "--quiet"],
            },
        ],
    },
    {
        "key": "recover-bad-images",
        "aliases": ["recover-bad", "repair-bad-images"],
        "label": "Recover Bad Image Files",
        "desc": "Dry-run recovery of quarantined bad person images from valid source folders",
        "script": "recover_bad_person_images.py",
        "args": ["--phash-threshold", "0"],
    },
    {
        "key": "nudity-confirm",
        "aliases": ["confirm-nudity", "promote-nudity"],
        "label": "Place Latest Nudity Report",
        "desc": "Dry-run by default: move latest possible_nudity report rows into photos/nude when you add --apply",
        "script": "place_nudity_inside_person_folders.py",
    },
    {
        "key": "rename",
        "aliases": ["number", "number-files", "rename-files"],
        "label": "Rename Person Files",
        "desc": "Smart-name person images as Person_0001_category_orientation_quality.ext",
        "script": "rename_person_folder_files.py",
        "args": ["--apply", "--quiet"],
    },
    {
        "key": "all-views",
        "aliases": ["all", "person-all", "all-photos"],
        "label": "Build Legacy All Person Views",
        "desc": "Legacy/manual only: create hardlinked all/ and all/nude views inside each person folder",
        "script": "build_all_person_views.py",
        "args": ["--apply", "--quiet"],
    },
    {
        "key": "structure",
        "aliases": ["structure-audit", "structure-repair", "layout", "organize-structure"],
        "label": "Person Folder Structure",
        "desc": "Audit person folders; add --apply to migrate photos/photos_nude/review layout",
        "script": "person_structure.py",
    },
    {
        "key": "cleanup-empty",
        "aliases": ["empty-folders", "remove-empty"],
        "label": "Cleanup Empty Person Folders",
        "desc": "Move person folders with no real source files to ready_to_delete",
        "script": "cleanup_empty_person_folders.py",
    },
    {
        "key": "scrap-smart-albums",
        "aliases": ["scrap-smart", "remove-smart", "delete-smart", "smart"],
        "label": "Remove Smart Albums",
        "desc": "Verify smart-folder images are in photos/photos/nude, recover unique files, then remove generated smart folders",
        "script": "scrap_smart_albums.py",
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
        "key": "review-dashboard",
        "aliases": ["dashboard", "review-ui"],
        "label": "Review Dashboard",
        "desc": "Create one local HTML dashboard linking unknowns, duplicates, references, nudity, and ready-to-delete",
        "script": "review_dashboard.py",
        "args": ["--open"],
    },
    {
        "key": "duplicate-review",
        "aliases": ["review-duplicates", "near-visual-review", "visual-review"],
        "label": "Review Duplicates / Near-Visuals",
        "desc": "Open a local browser review page grouped by person with batch keep/move controls",
        "script": "near_visual_review.py",
        "allow_original_count_decrease": True,
    },
    {
        "key": "unknown-triage",
        "aliases": ["unknowns", "triage-unknowns"],
        "label": "Unknown Face Triage",
        "desc": "Write HTML/CSV samples for unlabeled clusters so manual naming is faster",
        "script": "unknown_triage.py",
    },
    {
        "key": "cache-status",
        "aliases": ["cache", "detector-cache"],
        "label": "Face Cache Status",
        "desc": "Show whether the detector cache is useful or points at moved source files",
        "script": "cache_tools.py",
        "args": ["status"],
    },
    {
        "key": "recover-old-cache",
        "aliases": ["recover-missing", "recover-labeled"],
        "label": "Recover Labeled Sources",
        "desc": "Audit or restore still-available labeled originals from an older cache backup",
        "script": "recover_labeled_sources_from_cache.py",
    },
    {
        "key": "cache-relink",
        "aliases": ["relink-cache", "fast-cache"],
        "label": "Fast Cache Relink",
        "desc": "Rebuild cache and identity DB from old cached embeddings matched to current files",
        "script": "relink_cache_from_old_cache.py",
    },
    {
        "key": "cache-rehydrate",
        "aliases": ["rehydrate-cache", "rebuild-cache"],
        "label": "Rehydrate Face Cache",
        "desc": "Rebuild detector cache from current photos_by_person files. Add --apply to write it",
        "script": "cache_tools.py",
        "args": ["rehydrate"],
    },
    {
        "key": "repair",
        "aliases": ["audit-repair", "repair-all"],
        "label": "Audit / Repair",
        "desc": "Run preflight, cache validation, identity audit, duplicate audits, smart album check, and reports",
        "script": "repair_pipeline.py",
    },
    {
        "key": "rebuild-id",
        "aliases": ["build-id"],
        "label": "Rebuild Identity DB",
        "desc": "Maintenance: relearn known people from photos_by_person after many manual folder edits",
        "script": "sort_photos.py",
        "args": ["--identity-db-only", "--identity-max-images", "80"],
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
        "key": "integration-audit",
        "aliases": ["audit-flow", "flow-audit"],
        "label": "Integration Audit",
        "desc": "Read-only cross-script safety checks for daily ordering, cache, duplicates, and inbox visibility",
        "script": "integration_audit.py",
    },
    {
        "key": "synthetic-tests",
        "aliases": ["test-flow", "flow-tests"],
        "label": "Synthetic Integration Tests",
        "desc": "Run temporary-workspace edge-case tests without touching real photos",
        "script": "synthetic_integration_tests.py",
        "hidden": True,
    },
    {
        "key": "health",
        "aliases": ["validate", "dedupe", "optimize", "cleanup"],
        "label": "Health Check",
        "desc": "Preflight folders/cache/memory, synthetic tests, cache validation, and duplicate status",
        "steps": [
            {"script": "preflight_check.py"},
            {"script": "integration_audit.py"},
            {"script": "synthetic_integration_tests.py"},
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
    visible_actions: list[dict] = []
    action_by_key = {a["key"]: a for a in ACTIONS}
    for heading, keys in MENU_GROUPS:
        print(f"  {heading}")
        for key in keys:
            action = action_by_key.get(key)
            if action is None:
                continue
            visible_actions.append(action)
            print(f"    [{len(visible_actions)}] {action['label']}")
            print(f"        {action['desc']}")
        print()
    print("  Type a command name directly for advanced tools.")
    print("  Type ? to list every command keyword.")
    print(f"  [q] Quit")
    print()
    while True:
        try:
            ans = input("  Choose: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            return None
        if ans in ("q", "quit", "exit"):
            return None
        if ans in ("?", "list", "commands"):
            print_all_commands()
            continue
        if not ans:
            continue
        # Numeric
        if ans.isdigit():
            idx = int(ans) - 1
            if 0 <= idx < len(visible_actions):
                return visible_actions[idx]
        # Keyword
        action = find_action_by_key(ans)
        if action is not None:
            return action
        print(f"  Unknown choice: {ans}")


def print_all_commands() -> None:
    visible = {key for _heading, keys in MENU_GROUPS for key in keys}
    regular = [
        a for a in ACTIONS
        if not a.get("hidden") and a["key"] not in visible and a["key"] not in ADVANCED_MENU_KEYS
    ]
    advanced = [
        a for a in ACTIONS
        if a["key"] in ADVANCED_MENU_KEYS or a.get("hidden")
    ]

    def line(action: dict) -> str:
        aliases = action.get("aliases") or []
        alias_text = f" ({', '.join(aliases)})" if aliases else ""
        return f"    {action['key']}{alias_text}: {action['label']}"

    print()
    print("  Command keywords")
    print("  " + "-" * 56)
    print("  Main menu:")
    for _heading, keys in MENU_GROUPS:
        for key in keys:
            action = find_action_by_key(key)
            if action:
                print(line(action))
    if regular:
        print("  Other:")
        for action in regular:
            print(line(action))
    if advanced:
        print("  Advanced / recovery:")
        for action in advanced:
            print(line(action))
    print()


def find_action_by_key(key: str) -> dict | None:
    for a in ACTIONS:
        if a["key"] == key or key in a.get("aliases", []):
            return a
    return None


def source_guard_run_id(action: dict) -> str:
    safe_key = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in action["key"])
    return f"face_{safe_key}_{time.strftime('%Y%m%d_%H%M%S')}"


def write_guard_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames or ["person"])
        writer.writeheader()
        writer.writerows(rows)


def count_rows(counts: dict[str, int]) -> list[dict[str, int | str]]:
    return [
        {"person": person, "original_photos_count": int(count)}
        for person, count in sorted(counts.items(), key=lambda item: item[0].lower())
    ]


def source_guard_start(action: dict) -> dict:
    rid = source_guard_run_id(action)
    before = daily_runner.original_person_counts()
    prefix = daily_runner.SUMMARY_DIR / rid
    paths = {
        "before": prefix.with_name(prefix.name + "_source_counts_before.csv"),
        "after": prefix.with_name(prefix.name + "_source_counts_after.csv"),
        "violations": prefix.with_name(prefix.name + "_source_count_violations.csv"),
    }
    write_guard_csv(paths["before"], count_rows(before))
    print(
        f"Source guard: {daily_runner.original_person_total(before)} original photos "
        f"across {len(before)} person folders."
    )
    print(f"Source guard before CSV: {paths['before']}")
    return {"run_id": rid, "before": before, "paths": paths}


def source_guard_finish(guard: dict, *, allow_decrease: bool = False) -> int:
    before = {str(k): int(v) for k, v in guard["before"].items()}
    after = daily_runner.original_person_counts()
    paths = guard["paths"]
    violations = daily_runner.source_count_violations(before, after)
    write_guard_csv(paths["after"], count_rows(after))
    if not violations:
        print(
            f"Source guard OK: {daily_runner.original_person_total(before)} -> "
            f"{daily_runner.original_person_total(after)} original photos."
        )
        return 0
    write_guard_csv(paths["violations"], violations)
    if allow_decrease:
        print()
        print("Source guard: original-count decreases allowed for this manual review action.")
        print(f"Before count CSV:     {paths['before']}")
        print(f"After count CSV:      {paths['after']}")
        print(f"Change report CSV:    {paths['violations']}")
        for row in violations[:10]:
            print(f"  {row['person']}: {row['before']} -> {row['after']} ({row['delta']})")
        if len(violations) > 10:
            print(f"  ... {len(violations) - 10} more")
        return 0
    print()
    print("ERROR: source guard blocked this operation.")
    print("One or more person folders now have fewer original images under photos/ than before the command.")
    print(f"Before count CSV:     {paths['before']}")
    print(f"After count CSV:      {paths['after']}")
    print(f"Violation report CSV: {paths['violations']}")
    for row in violations[:10]:
        print(f"  {row['person']}: {row['before']} -> {row['after']} ({row['delta']})")
    if len(violations) > 10:
        print(f"  ... {len(violations) - 10} more")
    return SOURCE_GUARD_EXIT


def run_action(action: dict, extra_args: list[str] | None = None) -> int:
    allow_original_count_decrease = bool(action.get("allow_original_count_decrease"))
    manifest_check = source_manifest.validate_current(
        label=f"face_{action['key']}_start",
        people_dir=daily_runner.PEOPLE,
    )
    source_manifest.print_validation(manifest_check)
    if not manifest_check.ok:
        print("ERROR: protected source manifest blocked this command.")
        print("Fix or recover missing originals before running commands that may refresh cache/indexes.")
        return SOURCE_GUARD_EXIT

    guard = source_guard_start(action)
    command_rc = 0
    steps = action.get("steps") or [action]
    try:
        for i, step in enumerate(steps, start=1):
            script_path = SCRIPT_DIR / step["script"]
            if not script_path.exists():
                print(f"ERROR: missing script {script_path}")
                command_rc = 1
                break
            cmd = [sys.executable, str(script_path)] + step.get("args", [])
            if extra_args and len(steps) == 1:
                cmd.extend(extra_args)
            prefix = f"[{i}/{len(steps)}] " if len(steps) > 1 else ""
            print(f"\n→ {prefix}Running: {' '.join(cmd)}\n", flush=True)
            env = os.environ.copy()
            env["PHOTO_PIPELINE_RUN_ID"] = str(guard["run_id"])
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                command_rc = int(result.returncode)
                break
    except KeyboardInterrupt:
        command_rc = 130

    guard_rc = source_guard_finish(guard, allow_decrease=allow_original_count_decrease)
    if guard_rc != 0:
        return guard_rc
    if allow_original_count_decrease and command_rc in {130, -2}:
        print("Manual review server stopped; continuing final manifest update.")
        command_rc = 0
    if command_rc != 0:
        return command_rc

    post_manifest_check = source_manifest.validate_current(
        label=f"face_{action['key']}_before_promote",
        people_dir=daily_runner.PEOPLE,
    )
    if post_manifest_check.ok:
        source_manifest.print_validation(post_manifest_check)
    elif not allow_original_count_decrease:
        source_manifest.print_validation(post_manifest_check)
        return SOURCE_GUARD_EXIT
    else:
        print("Source manifest changed after allowed manual duplicate review; promoting current originals.")
    manifest_path = source_manifest.promote_current(
        label=f"face_{action['key']}_completed",
        reason=f"successful face.py action: {action['key']}",
        people_dir=daily_runner.PEOPLE,
    )
    print(f"Source manifest promoted: {manifest_path}")
    return 0


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

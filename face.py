"""
face.py — Simple launcher for the photo sorting pipeline.

Use the first option for normal day-to-day work: add new images to the
configured To Process inbox, then run `python face.py daily`.

Daily:
    python face.py daily       # safe intake/cache workflow for the configured inbox
    python face.py dry-run     # preview the daily workflow
    python face.py status      # quick dashboard
    python face.py health      # read-only safety checks

Useful manual tools:
    python face.py review-dashboard
    python face.py cross-person-audit
    python face.py nudity

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
import cache_tools  # noqa: E402
import source_manifest  # noqa: E402

SOURCE_GUARD_EXIT = 3
CACHE_GUARD_EXIT = 4
CACHE_GUARD_MIN_COVERAGE = 0.50
CACHE_GUARD_MIN_LIBRARY_SIZE = 1000
CACHE_GUARD_BYPASS_KEYS = {
    "analysis-migrate",
    "appearance-refresh",
    "cache-rehydrate",
    "cache-relink",
    "cache-status",
    "health",
    "identity-audit",
    "cross-person-audit",
    "identity-eval",
    "benchmark-review",
    "secondary-id",
    "pose-refresh",
    "integration-audit",
    "rebuild-id",
    "refs",
    "repair",
    "status",
    "synthetic-tests",
}


def source_review_storage_check() -> int:
    source_review = daily_runner.SOURCE_REVIEW
    if source_review.is_symlink():
        target = source_review.resolve(strict=False)
        if not source_review.exists():
            print()
            print("ERROR: external _source_review storage is unavailable.")
            print(f"Expected target: {target}")
            print("Connect and mount the external SSD before running Face Terminal actions.")
            return SOURCE_GUARD_EXIT
        if not os.access(source_review, os.R_OK | os.W_OK):
            print()
            print("ERROR: external _source_review storage is not readable and writable.")
            print(f"Target: {target}")
            return SOURCE_GUARD_EXIT
    elif not source_review.exists():
        print()
        print(f"ERROR: _source_review is missing: {source_review}")
        return SOURCE_GUARD_EXIT

    to_process = daily_runner.TO_PROCESS
    if not to_process.exists():
        print()
        print("ERROR: configured To Process inbox is unavailable.")
        print(f"Expected inbox: {to_process}")
        print("Connect and mount the external SSD before running Face Terminal actions.")
        return SOURCE_GUARD_EXIT
    if not os.access(to_process, os.R_OK | os.W_OK):
        print()
        print("ERROR: configured To Process inbox is not readable and writable.")
        print(f"Inbox: {to_process}")
        return SOURCE_GUARD_EXIT
    return 0

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
            "unknown-review",
            "benchmark-review",
            "cross-person-audit",
            "confirm-unknown",
        ],
    ),
    (
        "Maintenance",
        [
            "recover-unknown",
            "recover-no-face",
            "recover-videos",
            "nudity",
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
    "generated-artifacts",
    "identity-audit",
    "identity-eval",
    "secondary-id",
    "pose-refresh",
    "people-cleanup",
    "process",
    "process-all",
    "review",
    "finish",
    "duplicate-review",
    "nudity-audit",
    "scrap-smart-albums",
    "repair",
    "integration-audit",
    "recover-bad-images",
    "recover-old-cache",
    "refs",
    "rename",
    "rebuild-id",
    "structure",
    "synthetic-tests",
    "unknown-triage",
    "uncertain-nudity",
}


def menu_action_is_relevant(action: dict) -> bool:
    """Hide recovery actions when their queues contain no supported media."""
    key = action["key"]
    if key in {"recover-no-face", "recover-unknown", "unknown-review"}:
        queue_name = "unknown_identity" if key == "recover-unknown" else "no_usable_face"
        if key == "unknown-review":
            queue_name = "unknown_identity"
        queue_root = daily_runner.SOURCE_REVIEW / "unassigned_intake" / queue_name
        return daily_runner.tree_contains_media(queue_root, daily_runner.IMAGE_EXTS)
    if key == "recover-videos":
        queue_root = daily_runner.SOURCE_REVIEW / "unassigned_intake" / "videos"
        return daily_runner.tree_contains_media(queue_root, daily_runner.VIDEO_EXTS)
    return True

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
        "read_only": True,
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
        "key": "generated-artifacts",
        "aliases": ["contact-sheet-cleanup", "smart-sheet-cleanup"],
        "label": "Audit Generated Contact Sheets",
        "desc": "Report old smart-album contact/review sheets; add --apply to move them recoverably",
        "script": "generated_artifacts.py",
        "read_only": True,
    },
    {
        "key": "recover-unknown",
        "aliases": ["recover-known-faces", "recheck-unknown", "known-face-recovery"],
        "label": "Recover Known Faces",
        "desc": "Recheck unknown_identity with cached detection, multi-prototype matching, and safe calibrated decisions",
        "script": "recover_no_usable_faces.py",
        "args": [
            "--input", str(daily_runner.SOURCE_REVIEW / "unassigned_intake" / "unknown_identity"),
            "--source-kind", "unknown_identity",
        ],
    },
    {
        "key": "confirm-unknown",
        "aliases": ["confirm-known-face", "teach-identity"],
        "label": "Confirm Known Unknown Face",
        "desc": "Safely file user-confirmed unknowns and teach difficult poses without weakening match thresholds",
        "script": "confirm_unknown_identity.py",
    },
    {
        "key": "unknown-review",
        "aliases": ["review-unknown-faces", "unknown-dashboard", "active-learning"],
        "label": "Quick Review Unknown Faces",
        "desc": "Auto-files independently verified matches, then shows only ambiguous clusters for quick review",
        "script": "review_unknown_identities.py",
        "args": ["--serve", "--open", "--auto-safe"],
    },
    {
        "key": "recover-no-face",
        "aliases": ["recheck-no-face", "recover-faces"],
        "label": "Recover Missed Faces",
        "desc": "Recheck no_usable_face with soft/small/rotated-face recovery and safely file only strong identity matches",
        "script": "recover_no_usable_faces.py",
    },
    {
        "key": "recover-videos",
        "aliases": ["recheck-videos", "video-recovery", "recover-missed-videos"],
        "label": "Recover Missed Videos",
        "desc": "Recheck unknown/no-face videos and move after one recognized identity frame",
        "script": "video_batch_runner.py",
        "args": ["--review-only"],
    },
    {
        "key": "review",
        "aliases": ["fast", "resume"],
        "label": "Review Important Unknowns",
        "desc": "Optional: manually label only larger unknown clusters; press q anytime, then run Finish",
        "script": "sort_photos.py",
        "args": [
            str(daily_runner.SOURCE_REVIEW / "unassigned_intake" / "unknown_identity"),
            str(daily_runner.SORTED),
            "--merge-existing-output",
            "--skip-output-cleanup",
            "--scan-all-dirs",
            "--archive-organized-sources",
            "--fast",
            "--min-label-cluster-size", "20",
        ],
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
        "read_only": True,
    },
    {
        "key": "nudity",
        "aliases": ["nudity-check", "scan-nudity"],
        "label": "Run High-Precision Nudity Check",
        "desc": "Scan normal and nude folders: confirmed images stay/go nude, ambiguous images go to review, and safe images stay/return normal",
        "steps": [
            {
                "script": "separate_nudity_review.py",
                "args": ["--apply", "--no-export-copies"],
            },
            {
                "script": "place_nudity_inside_person_folders.py",
                "args": ["--apply", "--remove-review-copies", "--quiet"],
            },
            {
                "script": "audit_nude_folders.py",
                "args": ["--apply"],
            },
            {
                "script": "rename_person_folder_files.py",
                "args": ["--simple", "--apply", "--quiet"],
            },
        ],
    },
    {
        "key": "nudity-audit",
        "aliases": ["audit-nude-folders", "recheck-nude"],
        "label": "Audit Existing Nude Folders",
        "desc": "Report-only recheck of existing nude folders; add --apply explicitly to move safe/review cases",
        "script": "audit_nude_folders.py",
    },
    {
        "key": "uncertain-nudity",
        "aliases": ["review-uncertain-nudity", "confirm-nude-review"],
        "label": "Recheck Uncertain Nudity",
        "desc": "Recheck uncertain reviews; add --move-all --apply only to confirm and move every current image",
        "script": "audit_nude_folders.py",
        "args": ["--source", "uncertain"],
        # Moving a protected original from review/uncertain_nudity to
        # photos/nude changes its manifest path but not its content or owner.
        # The command is explicitly user-authorized and fully ledgered.
        "allow_original_count_decrease": True,
    },
    {
        "key": "bad-images",
        "aliases": ["quarantine-bad", "bad-person-images", "clean-bad-images"],
        "label": "Audit Bad Image Files",
        "desc": "Report unreadable image files without moving protected originals",
        "steps": [
            {
                "script": "quarantine_bad_person_images.py",
                "args": ["--quiet"],
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
        "key": "rename",
        "aliases": ["number", "number-files", "rename-files"],
        "label": "Rename Person Files",
        "desc": "Simplify person image names as Person_00001.ext and flatten photos subfolders",
        "script": "rename_person_folder_files.py",
        "args": ["--simple", "--apply", "--quiet"],
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
        "key": "cross-person-audit",
        "aliases": ["audit-cross-person", "wrong-person-audit", "identity-conflicts"],
        "label": "Cross-Person Identity Audit",
        "desc": "Review conflicting identities with keep, correct, ignore, and recoverable discard actions",
        "script": "audit_cross_person_identity.py",
        "args": ["--serve", "--open"],
        "allow_original_count_decrease": True,
    },
    {
        "key": "identity-eval",
        "aliases": ["evaluate-id", "benchmark-id"],
        "label": "Identity Accuracy Evaluation",
        "desc": "Read-only precision/recall benchmark against labeled cached faces",
        "script": "identity_evaluation.py",
        "read_only": True,
    },
    {
        "key": "benchmark-review",
        "aliases": ["protected-benchmark", "golden-review"],
        "label": "Review Protected Benchmark",
        "desc": "Manually verify identity, lookalike, face-detection, and nudity regression cases",
        "script": "review_identity_benchmark.py",
        "args": ["--open"],
        "read_only": True,
    },
    {
        "key": "secondary-id",
        "aliases": ["build-secondary-id", "borderline-verifier"],
        "label": "Build Secondary Identity Verifier",
        "desc": "Build the independent buffalo_l verifier on the external SSD for borderline matches",
        "script": "secondary_identity_matcher.py",
        "args": ["--build"],
    },
    {
        "key": "pose-refresh",
        "aliases": ["refresh-pose-profiles", "split-profile-poses"],
        "label": "Refresh Pose Profiles",
        "desc": "Safely classify selected legacy profile prototypes as left/right in bounded workers",
        "script": "refresh_pose_profiles.py",
        "hidden": True,
    },
    {
        "key": "appearance-refresh",
        "aliases": ["refresh-appearance-profiles", "low-light-profiles"],
        "label": "Refresh Appearance Profiles",
        "desc": "Safely build low-light and capture-era prototypes from cached identity faces",
        "script": "refresh_appearance_profiles.py",
        "hidden": True,
    },
    {
        "key": "review-dashboard",
        "aliases": ["dashboard", "review-ui"],
        "label": "Review Dashboard",
        "desc": "Create one local HTML dashboard linking unknowns, duplicates, references, nudity, and ready-to-delete",
        "script": "review_dashboard.py",
        "args": ["--open"],
        "read_only": True,
    },
    {
        "key": "duplicate-review",
        "aliases": ["review-duplicates", "near-visual-review", "visual-review"],
        "label": "Duplicate Review",
        "desc": "Refresh exact duplicates, then review revalidated byte-identical files in the browser",
        "steps": [
            {"script": "advanced_duplicate_matching.py", "args": ["--quiet"]},
            {"script": "near_visual_review.py"},
        ],
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
        "key": "analysis-migrate",
        "aliases": ["migrate-analysis", "sqlite-cache"],
        "label": "Migrate Analysis Cache",
        "desc": "Maintenance: copy current detector cache into the incremental SQLite index",
        "script": "cache_tools.py",
        "args": ["migrate-sqlite"],
        "hidden": True,
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
        "desc": "Maintenance: build optional AI/reference matching DB from the configured Face References folder",
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
        "read_only": True,
    },
    {
        "key": "synthetic-tests",
        "aliases": ["test-flow", "flow-tests"],
        "label": "Synthetic Integration Tests",
        "desc": "Run temporary-workspace edge-case tests without touching real photos",
        "script": "synthetic_integration_tests.py",
        "hidden": True,
        "read_only": True,
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
        "read_only": True,
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
            if action is None or not menu_action_is_relevant(action):
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


def cache_guard_check(action: dict) -> int:
    if action["key"] in CACHE_GUARD_BYPASS_KEYS:
        return 0
    coverage = cache_tools.coverage_summary(daily_runner.PEOPLE)
    total = int(coverage["total"])
    if total < CACHE_GUARD_MIN_LIBRARY_SIZE:
        return 0
    cached = int(coverage["cached"])
    missing = int(coverage["missing"])
    ratio = float(coverage["coverage"])
    if ratio >= CACHE_GUARD_MIN_COVERAGE:
        return 0
    print()
    print("ERROR: face cache coverage is too low for this action.")
    print(f"Current originals:       {total}")
    print(f"Cached current files:    {cached}")
    print(f"Missing cache files:     {missing}")
    print(f"Coverage:                {ratio * 100:.1f}%")
    print()
    print("Run one of these repair commands first:")
    print("  python face.py cache-relink --old-cache /path/to/full/cache.pkl.bak --apply")
    print("  python face.py cache-rehydrate --apply --max-missing 1000")
    print("  python face.py cache-status")
    return CACHE_GUARD_EXIT


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


def run_steps(action: dict, extra_args: list[str], *, run_id_value: str = "") -> int:
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
            print(f"\n→ {prefix}Running: {' '.join(cmd)}\n", flush=True)
            env = os.environ.copy()
            if run_id_value:
                env["PHOTO_PIPELINE_RUN_ID"] = run_id_value
            result = subprocess.run(cmd, env=env)
            if result.returncode != 0:
                return int(result.returncode)
    except KeyboardInterrupt:
        return 130
    return 0


def run_action(action: dict, extra_args: list[str] | None = None) -> int:
    storage_rc = source_review_storage_check()
    if storage_rc != 0:
        return storage_rc
    extra_args = extra_args or []
    daily_control_args = {"--resume", "--restart", "--full-maintenance", "--dry-run"}
    if (
        action["key"] == "daily"
        and not daily_control_args.intersection(extra_args)
        and not daily_runner.intake_has_media()
    ):
        print()
        print("Daily Ingest")
        print("=" * 60)
        print("Already current: no new images or videos are waiting in To Process.")
        print("No full-library scan was needed.")
        print("Run `face daily --full-maintenance` only when you intentionally need maintenance.")
        return 0
    if action.get("read_only"):
        return run_steps(action, extra_args)
    allow_original_count_decrease = bool(action.get("allow_original_count_decrease"))
    cache_rc = cache_guard_check(action)
    if cache_rc != 0:
        return cache_rc

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
    command_rc = run_steps(action, extra_args, run_id_value=str(guard["run_id"]))

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
        print("Source manifest changed after an approved, ledgered review relocation; promoting current originals.")
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

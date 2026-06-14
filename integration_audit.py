#!/usr/bin/env python3
"""
Read-only integration audit for the photo sorting pipeline.

This catches cross-script bugs that individual scripts can miss, for example:
  - daily steps ordered so file-moving cleanup happens before cache refresh
  - face.py actions pointing at missing scripts
  - cache entries pointing at moved/deleted files
  - exact/same-pixel duplicate report still asking for moves after cleanup
"""

from __future__ import annotations

import csv
import sys
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
PEOPLE = SORTED / "photos_by_person"
ADV_REPORT = SORTED / "_source_review" / "duplicate_reports" / "advanced_duplicates.csv"
REQUIRED_GENERATED_EXCLUSIONS = {
    "all",
    "_smart_albums",
    "_smart_albums_v2",
    "_smart_albums_simple_preview",
    "review",
    "_duplicates",
    "_near_visual_review",
}

sys.path.insert(0, str(SCRIPT_DIR))
import advanced_duplicate_matching  # noqa: E402
import cache_tools  # noqa: E402
import cleanup_empty_person_folders  # noqa: E402
import daily_runner  # noqa: E402
import delete_person_folder_duplicates  # noqa: E402
import face  # noqa: E402
import operation_ledger  # noqa: E402
import source_manifest  # noqa: E402
import sort_photos  # noqa: E402

for _name in ("CacheState", "CachedFace", "FaceRecord", "LabelingState", "IdentityDB"):
    if hasattr(sort_photos, _name):
        setattr(sys.modules["__main__"], _name, getattr(sort_photos, _name))


@dataclass
class Finding:
    level: str
    name: str
    detail: str


def add(findings: list[Finding], level: str, name: str, detail: str) -> None:
    findings.append(Finding(level, name, detail))


def check_scripts_exist(findings: list[Finding]) -> None:
    missing: list[str] = []
    for action in face.ACTIONS:
        steps = action.get("steps") or [action]
        for step in steps:
            script = step.get("script")
            if script and not (SCRIPT_DIR / script).exists():
                missing.append(f"{action['key']} -> {script}")
    for step in daily_runner.step_list(50):
        cmd = step.get("cmd", [])
        if len(cmd) >= 2 and cmd[1].endswith(".py") and not Path(cmd[1]).exists():
            missing.append(f"daily:{step['name']} -> {cmd[1]}")
    if missing:
        add(findings, "FAIL", "missing scripts", "; ".join(missing[:10]))
    else:
        add(findings, "OK", "launcher scripts", "all face.py and daily steps point at existing scripts")


def check_daily_order(findings: list[Finding]) -> None:
    names = [step["name"] for step in daily_runner.step_list(50)]
    index = {name: i for i, name in enumerate(names)}
    required = [
        ("structure", "cache-rehydrate"),
        ("rename", "cache-rehydrate"),
        ("exact-dedupe", "cache-rehydrate"),
        ("advanced-dedupe", "cache-rehydrate"),
        ("cleanup-empty", "cache-rehydrate"),
        ("cache-rehydrate", "integration-audit"),
        ("integration-audit", "status"),
    ]
    bad = [
        f"{before} should run before {after}"
        for before, after in required
        if before not in index or after not in index or index[before] >= index[after]
    ]
    if bad:
        add(findings, "FAIL", "daily step order", "; ".join(bad))
    else:
        add(findings, "OK", "daily step order", "file-moving cleanup happens before cache refresh")


def check_daily_destructive_commands(findings: list[Finding]) -> None:
    bad: list[str] = []
    for step in daily_runner.step_list(50):
        cmd = [str(part) for part in step.get("cmd", [])]
        script = Path(cmd[1]).name if len(cmd) > 1 and cmd[1].endswith(".py") else ""
        if step["name"] == "process" and "--skip-output-cleanup" not in cmd:
            bad.append("daily process must pass --skip-output-cleanup")
        if script in {"delete_person_folder_duplicates.py", "advanced_duplicate_matching.py"}:
            if "--apply" in cmd:
                bad.append(f"daily:{step['name']} must not pass --apply to {script}")
            if "--quarantine-bad" in cmd:
                bad.append(f"daily:{step['name']} must not pass --quarantine-bad to {script}")

    for cmd in sort_photos.post_process_steps(SORTED):
        cmd = [str(part) for part in cmd]
        script = Path(cmd[1]).name if len(cmd) > 1 and cmd[1].endswith(".py") else ""
        if script in {"delete_person_folder_duplicates.py", "advanced_duplicate_matching.py"}:
            if "--apply" in cmd:
                bad.append(f"sort_photos.run_post_process must not pass --apply to {script}")
            if "--quarantine-bad" in cmd:
                bad.append(f"sort_photos.run_post_process must not pass --quarantine-bad to {script}")

    if bad:
        add(findings, "FAIL", "destructive daily commands", "; ".join(bad))
    else:
        add(findings, "OK", "destructive daily commands", "duplicate cleanup is report-only and process skips auto cleanup")


def check_scanner_scope(findings: list[Finding]) -> None:
    scopes = {
        "sort_photos.ALWAYS_EXCLUDED_SCAN_DIRS": sort_photos.ALWAYS_EXCLUDED_SCAN_DIRS,
        "advanced_duplicate_matching.ALWAYS_EXCLUDED_DIRS": advanced_duplicate_matching.ALWAYS_EXCLUDED_DIRS,
        "delete_person_folder_duplicates.EXCLUDED_DIRS": delete_person_folder_duplicates.EXCLUDED_DIRS,
        "cleanup_empty_person_folders.SKIP_DIRS": cleanup_empty_person_folders.SKIP_DIRS,
        "cache_tools.CACHE_SCAN_EXCLUDED_DIRS": cache_tools.CACHE_SCAN_EXCLUDED_DIRS,
        "daily_runner.SMART_DIRS": daily_runner.SMART_DIRS,
    }
    bad: list[str] = []
    for name, values in scopes.items():
        missing = sorted(REQUIRED_GENERATED_EXCLUSIONS - set(values))
        if missing:
            bad.append(f"{name} missing {', '.join(missing)}")
    if bad:
        add(findings, "FAIL", "scanner generated-folder exclusions", "; ".join(bad))
    else:
        add(findings, "OK", "scanner generated-folder exclusions", "all scanner exclusion sets include generated view folders")


def check_rollback_tools(findings: list[Finding]) -> None:
    bad: list[str] = []
    if not hasattr(source_manifest, "restore_from_manifest"):
        bad.append("source_manifest.restore_from_manifest missing")
    if not hasattr(operation_ledger, "move_path"):
        bad.append("operation_ledger.move_path missing")
    if not hasattr(operation_ledger, "iter_events"):
        bad.append("operation_ledger.iter_events missing")
    parser = source_manifest.build_parser()
    subcommands = [
        action
        for action in parser._actions  # noqa: SLF001
        if getattr(action, "choices", None)
    ]
    choices = set(subcommands[0].choices) if subcommands else set()
    if "restore" not in choices:
        bad.append("source_manifest.py restore command missing")
    if bad:
        add(findings, "FAIL", "rollback tooling", "; ".join(bad))
    else:
        add(
            findings,
            "OK",
            "rollback tooling",
            f"move ledger and source_manifest restore command available under {operation_ledger.LEDGER_DIR_NAME}",
        )


def check_cache(findings: list[Finding]) -> None:
    cache = cache_tools.cache_summary()
    candidates = cache_tools.person_folder_images(PEOPLE)
    missing = int(cache["missing_files"])
    config_ok = bool(cache["config_ok"])
    cache_files = int(cache["file_signatures"])
    candidate_count = len(candidates)
    if not config_ok:
        add(findings, "FAIL", "face cache config", "cache fingerprint does not match current detector settings")
    elif missing:
        add(findings, "FAIL", "face cache paths", f"{missing} cached file(s) no longer exist")
    elif cache_files != candidate_count:
        add(
            findings,
            "WARN",
            "face cache coverage",
            f"cache has {cache_files} file(s), current person photos have {candidate_count}",
        )
    else:
        add(findings, "OK", "face cache", f"{cache_files} current organized files, no stale paths")


def check_duplicate_report(findings: list[Finding]) -> None:
    counts = {"exact_file": 0, "same_pixels": 0, "visually_similar": 0}
    if not ADV_REPORT.exists():
        add(findings, "WARN", "duplicate report", f"missing report: {ADV_REPORT}")
        return
    try:
        with ADV_REPORT.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                if row.get("action") in {"move", "review"} and row.get("type") in counts:
                    counts[row["type"]] += 1
    except Exception as exc:  # noqa: BLE001
        add(findings, "FAIL", "duplicate report", f"could not read {ADV_REPORT}: {exc}")
        return
    if counts["exact_file"] or counts["same_pixels"]:
        add(
            findings,
            "WARN",
            "duplicate candidates",
            f"exact={counts['exact_file']}, same_pixels={counts['same_pixels']} review-only; daily does not move originals",
        )
    else:
        add(
            findings,
            "OK",
            "duplicate candidates",
            f"none pending; near-visual review-only candidates={counts['visually_similar']}",
        )


def check_to_process_visibility(findings: list[Finding]) -> None:
    if not daily_runner.TO_PROCESS.exists():
        add(findings, "WARN", "To Process", f"missing optional inbox: {daily_runner.TO_PROCESS}")
        return
    visible = daily_runner.count_images(daily_runner.TO_PROCESS, exclude_generated_dirs=False)
    hidden_by_generated_names = (
        daily_runner.count_images(daily_runner.TO_PROCESS, exclude_generated_dirs=False)
        - daily_runner.count_images(daily_runner.TO_PROCESS, exclude_generated_dirs=True)
    )
    if hidden_by_generated_names:
        add(
            findings,
            "OK",
            "To Process scan visibility",
            f"{hidden_by_generated_names} image(s) are inside generated-like folder names and will still be scanned",
        )
    else:
        add(findings, "OK", "To Process scan visibility", f"{visible} inbox image(s) visible")


def check_source_manifest(findings: list[Finding]) -> None:
    result = source_manifest.validate_current(label="integration_audit_source_manifest", people_dir=PEOPLE)
    if result.ok:
        add(
            findings,
            "OK",
            "source manifest",
            f"{result.expected_total} protected originals, {result.current_total} current, "
            f"{len(result.extra)} new, {len(result.renamed)} renamed",
        )
        return
    add(
        findings,
        "FAIL",
        "source manifest",
        f"missing={len(result.missing)}, changed={len(result.size_changed)}; "
        f"missing report={result.missing_csv}",
    )


def print_findings(findings: list[Finding]) -> None:
    print("Photo Pipeline Integration Audit")
    print("=" * 60)
    width = max(len(f.name) for f in findings) if findings else 1
    for finding in findings:
        print(f"[{finding.level:4}] {finding.name:<{width}}  {finding.detail}")
    print()
    failures = sum(1 for f in findings if f.level == "FAIL")
    warnings = sum(1 for f in findings if f.level == "WARN")
    print(f"Result: {failures} failure(s), {warnings} warning(s)")


def main() -> int:
    findings: list[Finding] = []
    check_scripts_exist(findings)
    check_daily_order(findings)
    check_daily_destructive_commands(findings)
    check_scanner_scope(findings)
    check_rollback_tools(findings)
    check_cache(findings)
    check_source_manifest(findings)
    check_duplicate_report(findings)
    check_to_process_visibility(findings)
    print_findings(findings)
    return 1 if any(f.level == "FAIL" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())

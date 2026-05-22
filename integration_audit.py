#!/usr/bin/env python3
"""
Read-only integration audit for the photo sorting pipeline.

This catches cross-script bugs that individual scripts can miss, for example:
  - daily steps ordered so a cleanup moves files after cache refresh
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

sys.path.insert(0, str(SCRIPT_DIR))
import cache_tools  # noqa: E402
import daily_runner  # noqa: E402
import face  # noqa: E402
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
        ("cache-rehydrate", "all-views"),
        ("cache-rehydrate", "smart-albums"),
        ("smart-albums", "integration-audit"),
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
            "FAIL",
            "safe duplicates pending",
            f"exact={counts['exact_file']}, same_pixels={counts['same_pixels']}",
        )
    else:
        add(
            findings,
            "OK",
            "safe duplicates",
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
    check_cache(findings)
    check_duplicate_report(findings)
    check_to_process_visibility(findings)
    print_findings(findings)
    return 1 if any(f.level == "FAIL" for f in findings) else 0


if __name__ == "__main__":
    raise SystemExit(main())

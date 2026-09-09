#!/usr/bin/env python3
"""
Build one local HTML dashboard for the review queues.

It links saved Quick Review and duplicate reports without starting recognition
or rescanning photo folders. --refresh explicitly refreshes the duplicate
preview and folder counts. No photo files are moved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import subprocess
import sys
import webbrowser
from datetime import datetime
from pathlib import Path

import pipeline_paths

SORTED = pipeline_paths.SORTED_ROOT
SOURCE_REVIEW = SORTED / "_source_review"
PEOPLE = SORTED / "photos_by_person"
UNKNOWN_REPORT_DIR = SOURCE_REVIEW / "identity_audits" / "unknown_review"
UNKNOWN_HTML = UNKNOWN_REPORT_DIR / "unknown_identity_review.html"
UNKNOWN_SUMMARY = UNKNOWN_REPORT_DIR / "latest_unknown_identity_review.json"
DUPLICATE_REVIEW_HTML = SOURCE_REVIEW / "duplicate_reports" / "near_visual_review.html"
ADV_REPORT = SOURCE_REVIEW / "duplicate_reports" / "advanced_duplicates.csv"
REF_REPORT = pipeline_paths.FACE_REFERENCES / "_reference_review" / "face_reference_quality_report.csv"
DASHBOARD = SOURCE_REVIEW / "review_dashboard.html"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}


def count_files(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file())


def count_images(root: Path) -> int:
    if not root.exists():
        return 0
    return sum(1 for p in root.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def csv_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            return max(0, sum(1 for _ in csv.reader(f)) - 1)
    except Exception:
        return 0


def duplicate_summary() -> dict[str, int]:
    counts = {"exact_file": 0, "same_pixels": 0, "visually_similar": 0}
    if not ADV_REPORT.exists():
        return counts
    try:
        with ADV_REPORT.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                kind = row.get("type")
                action = row.get("action")
                if kind in counts and action in {"move", "review"}:
                    counts[kind] += 1
    except Exception:
        pass
    return counts


def file_link(path: Path, label: str | None = None) -> str:
    if not path.exists():
        return "<span class='missing'>not generated yet</span>"
    return f"<a href='{path.resolve().as_uri()}'>{html.escape(label or str(path))}</a>"


def ensure_reports() -> None:
    py = sys.executable
    subprocess.run([py, str(Path(__file__).with_name("near_visual_review.py")), "--html-only", "--quiet"], check=True)


def saved_unknown_count() -> int | None:
    try:
        payload = json.loads(UNKNOWN_SUMMARY.read_text(encoding="utf-8"))
        count = payload["progress"]["pending"]
        return count if type(count) is int and count >= 0 else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def dashboard_signature() -> str:
    versions = []
    for source in (Path(__file__), UNKNOWN_HTML, UNKNOWN_SUMMARY,
                   DUPLICATE_REVIEW_HTML, ADV_REPORT, REF_REPORT):
        try:
            stat = source.stat()
            version = (stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns, stat.st_ino)
        except OSError:
            version = None
        versions.append((str(source), version))
    for folder in (SOURCE_REVIEW / "unassigned_intake", PEOPLE, SOURCE_REVIEW / "ready_to_delete"):
        versions.append((str(folder), folder.exists()))
    return hashlib.sha256(json.dumps(versions).encode("utf-8")).hexdigest()


def dashboard_is_current(path: Path) -> bool:
    try:
        return path.read_text(encoding="utf-8").rstrip().endswith(
            f"</html>\n<!-- face-review-dashboard: {dashboard_signature()} -->"
        )
    except (OSError, UnicodeError):
        return False


def write_dashboard(path: Path, *, scan_counts: bool = False) -> None:
    signature = dashboard_signature()
    dups = duplicate_summary()
    review_rows = [
        {
            "title": "Unassigned intake",
            "count": count_images(SOURCE_REVIEW / "unassigned_intake") if scan_counts else None,
            "detail": "Scanned images that need review because no usable face or confident identity was available.",
            "link": file_link(SOURCE_REVIEW / "unassigned_intake", "Open unassigned intake"),
        },
        {
            "title": "Unknown Identity Quick Review",
            "count": saved_unknown_count(),
            "detail": "Pending files in the last saved Quick Review report.",
            "link": file_link(UNKNOWN_HTML, "Open saved report (read-only)"),
        },
        {
            "title": "Duplicate review",
            "count": sum(dups.values()) if ADV_REPORT.exists() else None,
            "detail": "Saved duplicate candidates. Live actions revalidate exact file contents.",
            "link": file_link(DUPLICATE_REVIEW_HTML, "Open duplicate review preview"),
        },
        {
            "title": "Exact/same-pixel duplicates",
            "count": dups["exact_file"] + dups["same_pixels"] if ADV_REPORT.exists() else None,
            "detail": "Saved candidates, not evidence that originals were removed.",
            "link": file_link(ADV_REPORT, "Open duplicate CSV"),
        },
        {
            "title": "Face References quality",
            "count": csv_count(REF_REPORT) if REF_REPORT.exists() else None,
            "detail": "Reference images scored, moved, or kept by quality cleanup.",
            "link": file_link(REF_REPORT, "Open reference CSV"),
        },
        {
            "title": "Person Nude Folders",
            "count": sum(
                count_images(p / "photos" / "nude")
                + count_images(p / "photos_nude")
                + count_images(p / "_possible_nudity")
                for p in PEOPLE.iterdir()
                if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")
            ) if scan_counts and PEOPLE.exists() else None,
            "detail": "Images currently inside person photos/nude folders.",
            "link": file_link(PEOPLE, "Open photos_by_person"),
        },
        {
            "title": "ready_to_delete",
            "count": count_files(SOURCE_REVIEW / "ready_to_delete") if scan_counts else None,
            "detail": "Files staged for deletion or external backup.",
            "link": file_link(SOURCE_REVIEW / "ready_to_delete", "Open ready_to_delete"),
        },
    ]
    cards = "\n".join(f"""
      <article>
        {f'<div class="count">{row["count"]:,}</div>' if row['count'] is not None else ''}
        <h2>{html.escape(row['title'])}</h2>
        <p>{html.escape(row['detail'])}</p>
        <div>{row['link']}</div>
      </article>
    """ for row in review_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Photo Review Dashboard</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: #f6f6f3; color: #23231f; }}
    header {{ padding: 22px 28px; border-bottom: 1px solid #d8d6cc; background: white; }}
    h1 {{ margin: 0; font-size: 24px; }}
    main {{ padding: 22px 28px 36px; display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 14px; }}
    article {{ background: white; border: 1px solid #dedbd2; border-radius: 8px; padding: 16px; }}
    .count {{ font-size: 30px; font-weight: 700; }}
    h2 {{ font-size: 17px; margin: 6px 0; }}
    p {{ color: #66645d; line-height: 1.4; min-height: 40px; }}
    a {{ color: #155cb0; text-decoration: none; font-weight: 600; }}
    .missing {{ color: #9a6615; }}
  </style>
</head>
<body>
  <header>
    <h1>Photo Review Dashboard</h1>
    <p>Saved report snapshot: {datetime.now().astimezone().strftime('%Y-%m-%d %H:%M %Z')}</p>
  </header>
  <main>{cards}</main>
</body>
</html>
<!-- face-review-dashboard: {signature} -->
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DASHBOARD)
    refresh = parser.add_mutually_exclusive_group()
    refresh.add_argument("--refresh", action="store_true",
                         help="Refresh the duplicate preview and folder counts; no recognition or photo moves.")
    refresh.add_argument("--no-refresh", action="store_true",
                         help="Compatibility flag: saved reports are reused by default.")
    parser.add_argument("--open", action="store_true",
                        help="Open the dashboard in the default browser.")
    args = parser.parse_args()

    if args.refresh:
        try:
            ensure_reports()
        except subprocess.CalledProcessError as error:
            print(f"ERROR: duplicate preview refresh failed (exit {error.returncode}); saved dashboard retained.")
            return error.returncode or 1
    output = args.output.expanduser().resolve()
    if args.refresh or not dashboard_is_current(output):
        write_dashboard(output, scan_counts=args.refresh)
    else:
        print("Reusing saved dashboard; no folder scan or report rebuild.")
    print(f"Review dashboard: {output}")
    print("For live confirmations: face unknown-review")
    print("For updated folder counts and duplicate preview: face review-dashboard --refresh")
    if args.open:
        webbrowser.open(output.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

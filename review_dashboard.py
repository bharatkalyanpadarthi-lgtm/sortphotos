#!/usr/bin/env python3
"""
Build one local HTML dashboard for the review queues.

It links together the existing near-visual duplicate review, unknown triage,
Face References quality report, duplicate report, and nudity review folders.
This script is read-only except for writing the dashboard HTML.
"""

from __future__ import annotations

import argparse
import csv
import html
import subprocess
import sys
import webbrowser
from pathlib import Path

SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
SOURCE_REVIEW = SORTED / "_source_review"
PEOPLE = SORTED / "photos_by_person"
UNKNOWN_HTML = SOURCE_REVIEW / "unknown_triage" / "unknown_triage.html"
UNKNOWN_CSV = SOURCE_REVIEW / "unknown_triage" / "unknown_triage.csv"
NEAR_VISUAL_HTML = SOURCE_REVIEW / "near_visual_review" / "near_visual_review.html"
ADV_REPORT = SOURCE_REVIEW / "duplicate_reports" / "advanced_duplicates.csv"
REF_REPORT = Path.home() / "Pictures" / "Face References" / "_reference_review" / "face_reference_quality_report.csv"
DASHBOARD = SOURCE_REVIEW / "review_dashboard.html"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


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
    subprocess.run([py, str(Path(__file__).with_name("unknown_triage.py")), "--quiet"], check=False)
    subprocess.run([py, str(Path(__file__).with_name("near_visual_review.py")), "--no-open"], check=False)


def write_dashboard(path: Path) -> None:
    dups = duplicate_summary()
    review_rows = [
        {
            "title": "Unknown faces",
            "count": csv_count(UNKNOWN_CSV),
            "detail": "Clusters that still need a person name.",
            "link": file_link(UNKNOWN_HTML, "Open unknown triage"),
        },
        {
            "title": "Near-visual duplicates",
            "count": dups["visually_similar"],
            "detail": "Similar-looking files. Review before deleting.",
            "link": file_link(NEAR_VISUAL_HTML, "Open duplicate review"),
        },
        {
            "title": "Exact/same-pixel duplicates",
            "count": dups["exact_file"] + dups["same_pixels"],
            "detail": "Safe duplicate classes already handled by cleanup commands.",
            "link": file_link(ADV_REPORT, "Open duplicate CSV"),
        },
        {
            "title": "Face References quality",
            "count": csv_count(REF_REPORT),
            "detail": "Reference images scored, moved, or kept by quality cleanup.",
            "link": file_link(REF_REPORT, "Open reference CSV"),
        },
        {
            "title": "Nudity possible",
            "count": sum(
                count_images(p / "photos_nude") + count_images(p / "_possible_nudity")
                for p in PEOPLE.iterdir()
                if p.is_dir() and not p.name.startswith("_")
            ) if PEOPLE.exists() else 0,
            "detail": "Images currently inside person nudity review folders.",
            "link": file_link(PEOPLE, "Open photos_by_person"),
        },
        {
            "title": "ready_to_delete",
            "count": count_files(SOURCE_REVIEW / "ready_to_delete"),
            "detail": "Files staged for deletion or external backup.",
            "link": file_link(SOURCE_REVIEW / "ready_to_delete", "Open ready_to_delete"),
        },
    ]
    cards = "\n".join(f"""
      <article>
        <div class="count">{row['count']:,}</div>
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
  </header>
  <main>{cards}</main>
</body>
</html>
""", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DASHBOARD)
    parser.add_argument("--no-refresh", action="store_true",
                        help="Do not regenerate unknown/near-visual reports first.")
    parser.add_argument("--open", action="store_true",
                        help="Open the dashboard in the default browser.")
    args = parser.parse_args()

    if not args.no_refresh:
        ensure_reports()
    output = args.output.expanduser().resolve()
    write_dashboard(output)
    print(f"Review dashboard: {output}")
    if args.open:
        webbrowser.open(output.as_uri())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

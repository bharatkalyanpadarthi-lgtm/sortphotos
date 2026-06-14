#!/usr/bin/env python3
"""
Build a simple hardlinked "all" view inside each person folder.

For each person:
  photos_by_person/<person>/all/
      Person_0001_photo_portrait_q_high.jpg
      ...
      best/
      by_quality/high/
      by_quality/good/
      by_quality/review/
      nude/
          Person_0042_nudity_possible_portrait_q_high.jpg
      index.html

The files are hardlinks to the real organized files, so this does not duplicate
image data on disk. The view is rebuilt from scratch each run and exact content
duplicates are skipped within each view.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import os
import shutil
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".gif"}
DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
VIEW_DIR = "all"
NUDE_DIR = "nude"
BEST_DIR = "best"
QUALITY_DIR = "by_quality"
SKIP_DIRS = {
    VIEW_DIR,
    "_smart_albums",
    "_smart_albums_v2",
    "_smart_albums_simple_preview",
    "_duplicates",
    "_near_visual_review",
}
NUDE_SOURCE_PARTS: set[tuple[str, ...]] = {
    ("photos", "nude"),
    ("photos_nude",),
    ("_possible_nudity",),
    ("review", "nudity_possible"),
    ("review", "uncertain_nudity"),
}
QUALITY_ORDER = {"q_high": 0, "q_good": 1, "q_review": 2, "q_unknown": 3}


class ViewStats:
    def __init__(self) -> None:
        self.all_links = 0
        self.nude_links = 0
        self.best_links = 0
        self.quality_links = 0
        self.duplicates_skipped = 0


def is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_EXTS


def iter_source_images(person_dir: Path) -> list[Path]:
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(person_dir):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS and not d.startswith(".")]
        base = Path(dirpath)
        for filename in filenames:
            path = base / filename
            if not is_image(path):
                continue
            rel = path.relative_to(person_dir)
            if rel.parts and rel.parts[0] == "review" and not any(
                tuple(rel.parts[:len(prefix)]) == prefix for prefix in NUDE_SOURCE_PARTS
            ):
                continue
            if is_image(path):
                out.append(path)
    return sorted(out, key=lambda p: str(p.relative_to(person_dir)).lower())


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = "".join(dest.suffixes)
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def quality_label(path: Path) -> str:
    stem = path.stem.casefold()
    for label in ("q_high", "q_good", "q_review", "q_unknown"):
        if label in stem:
            return label
    return "q_unknown"


def quality_dir_name(label: str) -> str:
    return {
        "q_high": "high",
        "q_good": "good",
        "q_review": "review",
        "q_unknown": "unknown",
    }.get(label, "unknown")


def best_rank(path: Path, person_dir: Path) -> tuple:
    rel = path.relative_to(person_dir)
    q = quality_label(path)
    nude_penalty = 1 if is_nude_source(path, person_dir) else 0
    review_penalty = 1 if any(part.startswith("_") for part in rel.parts) else 0
    return (
        QUALITY_ORDER.get(q, 9),
        nude_penalty,
        review_penalty,
        str(rel).lower(),
    )


def category_rank(path: Path, person_dir: Path) -> tuple[int, str]:
    rel = path.relative_to(person_dir)
    if is_nude_source(path, person_dir):
        return (1, str(rel).lower())
    if rel.parts and rel.parts[0].startswith("_"):
        return (2, str(rel).lower())
    return (0, str(rel).lower())


def is_nude_source(path: Path, person_dir: Path) -> bool:
    rel = path.relative_to(person_dir)
    if not rel.parts:
        return False
    return any(tuple(rel.parts[:len(prefix)]) == prefix for prefix in NUDE_SOURCE_PARTS)


def hardlink_or_symlink(src: Path, dest: Path, apply: bool) -> bool:
    if not apply:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(str(src), str(dest))
    except OSError:
        os.symlink(str(src), str(dest))
    return True


def relative_href(path: Path, base: Path) -> str:
    try:
        rel = path.relative_to(base)
    except ValueError:
        rel = path
    return html.escape(str(rel).replace(os.sep, "/"), quote=True)


def write_person_dashboard(person_dir: Path, apply: bool) -> None:
    if not apply:
        return
    view_dir = person_dir / VIEW_DIR
    all_files = sorted([p for p in view_dir.iterdir() if is_image(p)], key=lambda p: p.name.lower()) if view_dir.exists() else []
    best_files = sorted([p for p in (view_dir / BEST_DIR).glob("*") if is_image(p)], key=lambda p: p.name.lower())
    nude_files = sorted([p for p in (view_dir / NUDE_DIR).glob("*") if is_image(p)], key=lambda p: p.name.lower())

    def figures(files: list[Path], limit: int = 80) -> str:
        if not files:
            return "<p class='empty'>No files in this view.</p>"
        items = []
        for path in files[:limit]:
            href = relative_href(path, view_dir)
            label = html.escape(path.name)
            items.append(f"<figure><a href='{href}'><img src='{href}' alt='{label}' loading='lazy'></a><figcaption>{label}</figcaption></figure>")
        if len(files) > limit:
            items.append(f"<p class='more'>Showing first {limit} of {len(files)} files. Open the folder for the full set.</p>")
        return "<div class='grid'>" + "\n".join(items) + "</div>"

    view_dir.mkdir(parents=True, exist_ok=True)
    (view_dir / "index.html").write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(person_dir.name)} Photo Dashboard</title>
  <style>
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: #242420; background: #f7f7f4; }}
    header {{ padding: 18px 22px; background: white; border-bottom: 1px solid #ddd9cf; position: sticky; top: 0; z-index: 1; }}
    h1 {{ margin: 0 0 6px; font-size: 24px; }}
    nav a {{ margin-right: 14px; color: #155cb0; text-decoration: none; font-weight: 600; }}
    section {{ padding: 18px 22px 26px; border-bottom: 1px solid #ddd9cf; }}
    h2 {{ margin: 0 0 10px; font-size: 18px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(120px, 1fr)); gap: 10px; }}
    figure {{ margin: 0; background: white; border: 1px solid #dedbd2; border-radius: 8px; overflow: hidden; }}
    img {{ width: 100%; height: 150px; object-fit: cover; display: block; background: #eeeae1; }}
    figcaption {{ padding: 6px; font-size: 11px; color: #5f5d56; overflow-wrap: anywhere; }}
    .empty, .more {{ color: #66645d; }}
  </style>
</head>
<body>
  <header>
    <h1>{html.escape(person_dir.name)}</h1>
    <nav>
      <a href=".">All ({len(all_files)})</a>
      <a href="best/">Best ({len(best_files)})</a>
      <a href="nude/">Nude ({len(nude_files)})</a>
      <a href="by_quality/high/">High quality</a>
    </nav>
  </header>
  <section><h2>Best</h2>{figures(best_files, 50)}</section>
  <section><h2>Nude</h2>{figures(nude_files, 50)}</section>
  <section><h2>All Photos</h2>{figures(all_files, 80)}</section>
</body>
</html>
""", encoding="utf-8")


def build_for_person(person_dir: Path, apply: bool, best_count: int) -> ViewStats:
    stats = ViewStats()
    view_dir = person_dir / VIEW_DIR
    if apply and view_dir.exists():
        shutil.rmtree(view_dir)

    seen_all: set[str] = set()
    seen_nude: set[str] = set()
    seen_best: set[str] = set()
    seen_quality: set[tuple[str, str]] = set()
    unique_sources: list[tuple[Path, str]] = []

    for src in sorted(iter_source_images(person_dir), key=lambda p: category_rank(p, person_dir)):
        try:
            digest = sha256_file(src)
        except OSError:
            continue
        nude = is_nude_source(src, person_dir)
        if digest in seen_all:
            stats.duplicates_skipped += 1
        else:
            seen_all.add(digest)
            unique_sources.append((src, digest))
            if not nude:
                dest = unique_dest(view_dir / src.name)
                if hardlink_or_symlink(src, dest, apply):
                    stats.all_links += 1
        if nude:
            if digest in seen_nude:
                continue
            seen_nude.add(digest)
            dest = unique_dest(view_dir / NUDE_DIR / src.name)
            if hardlink_or_symlink(src, dest, apply):
                stats.nude_links += 1

    safe_sources = [
        (src, digest) for src, digest in unique_sources
        if not is_nude_source(src, person_dir)
    ]

    for src, digest in sorted(safe_sources, key=lambda item: best_rank(item[0], person_dir))[:best_count]:
        if digest in seen_best:
            continue
        seen_best.add(digest)
        dest = unique_dest(view_dir / BEST_DIR / src.name)
        if hardlink_or_symlink(src, dest, apply):
            stats.best_links += 1

    for src, digest in safe_sources:
        label = quality_dir_name(quality_label(src))
        key = (label, digest)
        if key in seen_quality:
            continue
        seen_quality.add(key)
        dest = unique_dest(view_dir / QUALITY_DIR / label / src.name)
        if hardlink_or_symlink(src, dest, apply):
            stats.quality_links += 1

    write_person_dashboard(person_dir, apply)
    return stats


def write_root_dashboard(people_dir: Path, dirs: list[Path], apply: bool) -> None:
    if not apply:
        return
    rows = []
    for person_dir in dirs:
        view_dir = person_dir / VIEW_DIR
        all_count = len([p for p in view_dir.iterdir() if is_image(p)]) if view_dir.exists() else 0
        nude_count = len([p for p in (view_dir / NUDE_DIR).glob("*") if is_image(p)])
        best_count = len([p for p in (view_dir / BEST_DIR).glob("*") if is_image(p)])
        href = html.escape(f"{person_dir.name}/all/index.html", quote=True)
        rows.append((person_dir.name.lower(), f"<tr><td><a href='{href}'>{html.escape(person_dir.name)}</a></td><td>{all_count}</td><td>{best_count}</td><td>{nude_count}</td></tr>"))
    body = "\n".join(row for _key, row in sorted(rows))
    (people_dir / "_browse_people.html").write_text(f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>People Photo Dashboard</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 24px; background: #f7f7f4; color: #242420; }}
    table {{ border-collapse: collapse; width: 100%; background: white; }}
    th, td {{ text-align: left; padding: 9px 11px; border-bottom: 1px solid #dedbd2; }}
    th {{ background: #eeeae1; }}
    a {{ color: #155cb0; text-decoration: none; font-weight: 600; }}
  </style>
</head>
<body>
  <h1>People Photo Dashboard</h1>
  <table>
    <thead><tr><th>Person</th><th>All</th><th>Best</th><th>Nude</th></tr></thead>
    <tbody>{body}</tbody>
  </table>
</body>
</html>
""", encoding="utf-8")


def person_dirs(root: Path, person: str | None) -> list[Path]:
    dirs = [p for p in root.iterdir() if p.is_dir() and not p.name.startswith("_") and p.name != VIEW_DIR]
    if person:
        wanted = person.casefold()
        dirs = [p for p in dirs if p.name.casefold() == wanted]
    return sorted(dirs, key=lambda p: p.name.lower())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("people_dir", nargs="?", default=str(DEFAULT_PEOPLE))
    parser.add_argument("--person", default=None)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--best-count", type=int, default=50,
                        help="Number of best images to link into all/best. Default 50.")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    people_dir = Path(args.people_dir).expanduser().resolve()
    if not people_dir.exists():
        print(f"ERROR: people folder not found: {people_dir}")
        return 1

    dirs = person_dirs(people_dir, args.person)
    if args.person and not dirs:
        print(f"ERROR: person folder not found: {args.person}")
        return 1

    total_all = total_nude = total_best = total_quality = total_dupes = 0
    for person_dir in dirs:
        stats = build_for_person(person_dir, args.apply, max(1, int(args.best_count)))
        total_all += stats.all_links
        total_nude += stats.nude_links
        total_best += stats.best_links
        total_quality += stats.quality_links
        total_dupes += stats.duplicates_skipped
        if not args.quiet:
            print(
                f"{person_dir.name:<34} all={stats.all_links:<5} "
                f"best={stats.best_links:<4} nude={stats.nude_links:<5} "
                f"dupes_skipped={stats.duplicates_skipped}"
            )

    write_root_dashboard(people_dir, dirs, args.apply)

    print()
    print(f"People folder:        {people_dir}")
    print(f"Person folders:       {len(dirs)}")
    print(f"All-view links:       {total_all}")
    print(f"Best-view links:      {total_best}")
    print(f"Quality-view links:   {total_quality}")
    print(f"Nude-view links:      {total_nude}")
    print(f"Duplicates skipped:   {total_dupes}")
    if not args.apply:
        print("DRY-RUN - no all views created. Re-run with --apply to commit.")
    else:
        print("All views rebuilt.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

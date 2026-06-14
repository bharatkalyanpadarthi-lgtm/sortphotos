#!/usr/bin/env python3
"""
Review duplicate and near-visual candidates in a local browser.

The tool reads the CSV produced by advanced_duplicate_matching.py and serves a
local page grouped by person. Nothing is moved automatically: each candidate
needs an explicit Move or Keep click, either one at a time or via selected
checkboxes.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import mimetypes
import os
import shutil
import time
import webbrowser
from dataclasses import dataclass, field
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import operation_ledger

DEFAULT_SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
DEFAULT_PEOPLE = DEFAULT_SORTED / "photos_by_person"
DEFAULT_REPORT = DEFAULT_SORTED / "_source_review" / "duplicate_reports" / "advanced_duplicates.csv"
DEFAULT_REVIEW_DIR = DEFAULT_SORTED / "_source_review" / "ready_to_delete" / "manual_duplicate_review"
DEFAULT_THUMB_DIR = DEFAULT_SORTED / "_source_review" / "duplicate_reports" / "near_visual_thumbnails"
DEFAULT_DECISIONS = Path.home() / ".face_sort_cache" / "near_visual_review_decisions.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif", ".tiff", ".heic", ".heif"}
THUMB_SIZE = 520
THUMB_EXT = ".jpg"
PENDING_ACTIONS = {"move", "review"}
TYPE_LABELS = {
    "exact_file": "Exact duplicate",
    "same_pixels": "Same pixels",
    "visually_similar": "Near visual",
}
TYPE_ORDER = {"exact_file": 0, "same_pixels": 1, "visually_similar": 2}


@dataclass
class Candidate:
    path: Path
    kind: str
    action: str
    confidence: int
    distance: int | None
    width: int
    height: int
    size_bytes: int


@dataclass
class ReviewGroup:
    group_id: str
    kind: str
    person: str
    scope: str
    keeper: Path | None = None
    confidence: int = 0
    distance: int | None = None
    candidates: list[Candidate] = field(default_factory=list)


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


def safe_mtime_ns(path: Path) -> int:
    try:
        return int(path.stat().st_mtime_ns)
    except OSError:
        return 0


def thumb_name(path: Path) -> str:
    stat = path.stat()
    key = f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}".encode("utf-8", "surrogateescape")
    return hashlib.sha1(key).hexdigest() + THUMB_EXT


def thumb_path_for(path: Path, thumb_dir: Path) -> Path:
    return thumb_dir / thumb_name(path)


def decode_image_for_thumb(path: Path):
    try:
        from PIL import Image, ImageFile, ImageOps
        import pillow_heif

        ImageFile.LOAD_TRUNCATED_IMAGES = True
        if hasattr(pillow_heif, "register_heif_opener"):
            pillow_heif.register_heif_opener()
        with Image.open(path) as im:
            im.seek(0)
            im = ImageOps.exif_transpose(im)
            im.load()
            return im.convert("RGB")
    except Exception:
        pass

    try:
        import cv2
        import numpy as np
        from PIL import Image

        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        img = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if img is None or getattr(img, "size", 0) == 0:
            return None
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return Image.fromarray(rgb)
    except Exception:
        return None


def ensure_thumbnail(path: Path, thumb_dir: Path) -> Path | None:
    try:
        dest = thumb_path_for(path, thumb_dir)
    except OSError:
        return None
    if dest.exists() and safe_mtime_ns(dest) >= safe_mtime_ns(path):
        return dest
    image = decode_image_for_thumb(path)
    if image is None:
        return None
    thumb_dir.mkdir(parents=True, exist_ok=True)
    image.thumbnail((THUMB_SIZE, THUMB_SIZE))
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    try:
        image.save(tmp, format="JPEG", quality=86, optimize=True)
        tmp.replace(dest)
        return dest
    except Exception:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        return None


def iter_card_paths(groups: list[ReviewGroup]) -> list[Path]:
    paths: list[Path] = []
    seen: set[str] = set()
    for group in groups:
        if group.keeper is not None:
            key = decision_key(group.keeper)
            if key not in seen:
                seen.add(key)
                paths.append(group.keeper)
        for candidate in group.candidates:
            if candidate.action not in PENDING_ACTIONS:
                continue
            key = decision_key(candidate.path)
            if key in seen:
                continue
            seen.add(key)
            paths.append(candidate.path)
    return paths


def prepare_thumbnails(groups: list[ReviewGroup], thumb_dir: Path, *, quiet: bool = False) -> dict[str, Path]:
    thumbnails: dict[str, Path] = {}
    paths = iter_card_paths(groups)
    failures = 0
    for i, path in enumerate(paths, start=1):
        thumb = ensure_thumbnail(path, thumb_dir)
        if thumb is None:
            failures += 1
            continue
        thumbnails[decision_key(path)] = thumb
        if not quiet and i % 250 == 0:
            print(f"Prepared thumbnails: {i}/{len(paths)}")
    if not quiet:
        print(f"Thumbnails ready:    {len(thumbnails)}/{len(paths)} ({failures} failed)")
        print(f"Thumbnail cache:     {thumb_dir}")
    return thumbnails


def candidate_kinds_by_path(groups: list[ReviewGroup]) -> dict[str, str]:
    out: dict[str, str] = {}
    for group in groups:
        if group.keeper is not None:
            out[str(group.keeper.resolve())] = group.kind
        for candidate in group.candidates:
            out[str(candidate.path.resolve())] = candidate.kind
    return out


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = dest.suffix
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}_{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def load_decisions(path: Path) -> dict:
    if not path.exists():
        return {"version": 1, "items": {}}
    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if data.get("version") == 1 and isinstance(data.get("items"), dict):
            return data
    except Exception:
        pass
    return {"version": 1, "items": {}}


def save_decisions(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, sort_keys=True)
    tmp.replace(path)


def decision_key(path: Path) -> str:
    return str(path.resolve())


def int_or_zero(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def int_or_none(value: str | None) -> int | None:
    try:
        if value in {None, ""}:
            return None
        return int(value)
    except ValueError:
        return None


def type_label(kind: str) -> str:
    return TYPE_LABELS.get(kind, kind.replace("_", " ").title())


def person_for_path(path: Path, root: Path) -> str:
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return "_outside"
    return rel.parts[0] if rel.parts else "_unknown"


def is_inside_root(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def load_groups(report: Path, decisions: dict, root: Path, limit: int | None = None) -> list[ReviewGroup]:
    grouped: dict[tuple[str, str], ReviewGroup] = {}
    if not report.exists():
        return []
    decided = decisions.setdefault("items", {})
    root = root.resolve()
    with report.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            kind = row.get("type", "")
            if kind not in TYPE_LABELS:
                continue
            path = Path(row.get("file_path", ""))
            if not path.exists() or not path.suffix.lower() in IMAGE_EXTS:
                continue
            if not is_inside_root(path, root):
                continue
            key = decision_key(path)
            action = row.get("action", "")
            previous_decision = decided.get(key)
            if (
                action in PENDING_ACTIONS
                and isinstance(previous_decision, dict)
                and previous_decision.get("action") == "keep"
            ):
                continue
            group_id = row.get("group_id", "")
            person = person_for_path(path, root)
            group_key = (kind, group_id)
            group = grouped.setdefault(
                group_key,
                ReviewGroup(
                    group_id=group_id,
                    kind=kind,
                    person=person,
                    scope=row.get("scope", ""),
                    confidence=int_or_zero(row.get("confidence")),
                    distance=int_or_none(row.get("distance")),
                ),
            )
            keeper = Path(row.get("keeper_path", ""))
            if keeper.exists() and is_inside_root(keeper, root):
                group.keeper = keeper
                group.person = person_for_path(keeper, root)
            candidate = Candidate(
                path=path,
                kind=kind,
                action=action,
                confidence=int_or_zero(row.get("confidence")),
                distance=int_or_none(row.get("distance")),
                width=int_or_zero(row.get("width")),
                height=int_or_zero(row.get("height")),
                size_bytes=int_or_zero(row.get("size_bytes")),
            )
            group.candidates.append(candidate)

    out: list[ReviewGroup] = []
    for group in grouped.values():
        review_candidates = [c for c in group.candidates if c.action in PENDING_ACTIONS]
        keepers = [c for c in group.candidates if c.action == "keep"]
        if group.keeper is None and keepers:
            group.keeper = keepers[0].path
        if not review_candidates or group.keeper is None:
            continue
        if not is_inside_root(group.keeper, root):
            continue
        group.person = person_for_path(group.keeper, root)
        group.candidates = sorted(
            group.candidates,
            key=lambda c: (
                c.action != "keep",
                c.action not in PENDING_ACTIONS,
                str(c.path).lower(),
            ),
        )
        out.append(group)
    out.sort(
        key=lambda g: (
            g.person.casefold(),
            TYPE_ORDER.get(g.kind, 9),
            -g.confidence,
            g.distance if g.distance is not None else 99,
            int_or_zero(g.group_id),
        )
    )
    if limit:
        return out[:limit]
    return out


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def image_url(path: Path, *, static_images: bool = False) -> str:
    if static_images:
        return path.resolve().as_uri()
    return "/image?path=" + quote(str(path), safe="")


def thumbnail_url(path: Path, thumbnails: dict[str, Path], *, static_images: bool = False) -> str:
    if static_images:
        thumb = thumbnails.get(decision_key(path))
        if thumb and thumb.exists():
            return thumb.resolve().as_uri()
        return image_url(path, static_images=True)
    return "/thumb?path=" + quote(str(path), safe="")


def render_card(candidate: Candidate, root: Path, is_keeper: bool, thumbnails: dict[str, Path],
                *, static_images: bool = False) -> str:
    path = candidate.path
    label = "KEEPER" if is_keeper else ("DUPLICATE" if candidate.action == "move" else "NEAR VISUAL")
    badge = "keeper" if is_keeper else ("duplicate" if candidate.action == "move" else "review")
    buttons = ""
    escaped_path = html.escape(str(path), quote=True)
    original_url = image_url(path, static_images=static_images)
    preview_url = thumbnail_url(path, thumbnails, static_images=static_images)
    if not is_keeper:
        buttons = f"""
          <div class="actions">
            <label class="select-row"><input type="checkbox" class="candidate-check" data-path="{escaped_path}"> Select</label>
            <button class="move" data-path="{escaped_path}">Move to Ready To Delete</button>
            <button class="keep" data-path="{escaped_path}">Keep</button>
          </div>
        """
    return f"""
      <article class="card {badge}" data-path="{escaped_path}" data-kind="{html.escape(candidate.kind, quote=True)}">
        <div class="thumb-wrap">
          <a href="{original_url}" target="_blank" rel="noopener">
            <img loading="lazy" src="{preview_url}" alt="{html.escape(path.name)}">
          </a>
        </div>
        <div class="meta">
          <span class="badge">{label}</span>
          <strong>{html.escape(path.name)}</strong>
          <span class="detail-main">{html.escape(type_label(candidate.kind))} | {candidate.width} x {candidate.height} | {human_size(candidate.size_bytes)}</span>
          <span class="detail-extra">{html.escape(safe_rel(path, root))}</span>
        </div>
        {buttons}
      </article>
    """


def render_html(groups: list[ReviewGroup], root: Path, report: Path, review_dir: Path,
                decisions: dict, *, thumbnails: dict[str, Path] | None = None,
                static_images: bool = False) -> str:
    thumbnails = thumbnails or {}
    people = sorted({g.person for g in groups}, key=str.casefold)
    type_counts = {kind: 0 for kind in TYPE_LABELS}
    person_counts = {person: 0 for person in people}
    total_candidates = 0
    for group in groups:
        pending = sum(1 for c in group.candidates if c.action in PENDING_ACTIONS)
        total_candidates += pending
        type_counts[group.kind] = type_counts.get(group.kind, 0) + pending
        person_counts[group.person] = person_counts.get(group.person, 0) + pending

    person_options = "\n".join(
        f'<option value="{html.escape(person, quote=True)}">'
        f'{html.escape(person)} ({person_counts.get(person, 0)})</option>'
        for person in people
    )
    type_buttons = "\n".join(
        f'<button class="filter-type" data-type="{html.escape(kind, quote=True)}">'
        f'{html.escape(label)} <span>{type_counts.get(kind, 0)}</span></button>'
        for kind, label in TYPE_LABELS.items()
    )

    sections = []
    for person in people:
        person_groups = [g for g in groups if g.person == person]
        group_cards = []
        for group in person_groups:
            keeper_card = ""
            review_cards = []
            for candidate in group.candidates:
                is_keeper = group.keeper is not None and candidate.path.resolve() == group.keeper.resolve()
                if is_keeper:
                    keeper_card = render_card(candidate, root, True, thumbnails, static_images=static_images)
                elif candidate.action in PENDING_ACTIONS:
                    review_cards.append(render_card(candidate, root, False, thumbnails, static_images=static_images))
            if not keeper_card and group.keeper:
                keeper_card = render_card(
                    Candidate(group.keeper, group.kind, "keep", group.confidence, group.distance, 0, 0, 0),
                    root,
                    True,
                    thumbnails,
                    static_images=static_images,
                )
            pending_count = len(review_cards)
            group_cards.append(f"""
              <section class="group" data-person="{html.escape(person, quote=True)}" data-kind="{html.escape(group.kind, quote=True)}">
                <header>
                  <div>
                    <h3>{html.escape(type_label(group.kind))} group {html.escape(group.group_id)}</h3>
                    <p>{pending_count} candidate(s) | Confidence {group.confidence}% | Distance {group.distance if group.distance is not None else "exact"} | {html.escape(group.scope)}</p>
                  </div>
                  <div class="group-actions">
                    <button class="select-group">Select group</button>
                    <button class="skip-group">Hide group</button>
                  </div>
                </header>
                <div class="grid">
                  {keeper_card}
                  {"".join(review_cards)}
                </div>
              </section>
            """)
        sections.append(f"""
          <section class="person-section" data-person="{html.escape(person, quote=True)}">
            <div class="person-heading">
              <h2>{html.escape(person)}</h2>
              <span>{person_counts.get(person, 0)} candidate(s)</span>
            </div>
            {"".join(group_cards)}
          </section>
        """)

    decided_count = len(decisions.get("items", {}))
    mode_note = (
        "Static preview: filters and image links work here; open via face option 8 for keep/move actions."
        if static_images
        else "Live review mode: keep/move actions are enabled."
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Duplicate Review</title>
  <style>
    :root {{ color-scheme: light; --bg:#f7f7f4; --ink:#222; --muted:#6a6a65; --line:#d9d7cf; --good:#1f7a4d; --warn:#a45b14; --bad:#a63434; --soft:#fff; }}
    * {{ box-sizing: border-box; }}
    body {{ --card-min: 170px; --thumb-h: 185px; --grid-gap: 8px; margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
    body[data-density="comfortable"] {{ --card-min: 225px; --thumb-h: 260px; --grid-gap: 12px; }}
    body[data-density="large"] {{ --card-min: 300px; --thumb-h: 340px; --grid-gap: 14px; }}
    .top {{ position: sticky; top: 0; z-index: 5; background: rgba(247,247,244,.96); border-bottom: 1px solid var(--line); backdrop-filter: blur(12px); padding: 14px 22px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 8px; color: var(--muted); font-size: 13px; }}
    .controls {{ display: grid; grid-template-columns: minmax(180px, 260px) minmax(260px, 1fr) auto; gap: 12px; align-items: center; margin-top: 12px; }}
    select {{ width: 100%; border: 1px solid var(--line); border-radius: 6px; background: white; padding: 8px 10px; font: inherit; }}
    .quick-actions {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-top: 10px; }}
    .quick-actions .hint {{ color: var(--muted); font-size: 13px; margin-right: 4px; }}
    .type-filters, .batch-actions, .density-controls, .group-actions, .actions {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .type-filters button.active, .density-controls button.active {{ background: #222; color: white; border-color: #222; }}
    .type-filters span {{ color: inherit; opacity: .7; }}
    .selected-count, .visible-stats {{ color: var(--muted); font-size: 13px; }}
    .selected-count {{ min-width: 92px; }}
    main {{ padding: 18px 22px 40px; }}
    .person-section {{ border-top: 1px solid var(--line); padding: 20px 0 30px; }}
    .person-heading {{ display: flex; align-items: baseline; justify-content: space-between; gap: 12px; margin-bottom: 14px; }}
    .person-heading h2 {{ margin: 0; font-size: 22px; }}
    .person-heading span {{ color: var(--muted); }}
    .group {{ border: 1px solid var(--line); background: rgba(255,255,255,.52); border-radius: 8px; padding: 14px; margin-bottom: 14px; }}
    .group header {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 12px; }}
    h3 {{ margin: 0; font-size: 17px; }}
    p {{ margin: 4px 0 0; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(var(--card-min), 1fr)); gap: var(--grid-gap); align-items: start; }}
    .card {{ background: white; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .card.keeper {{ border-color: rgba(31,122,77,.55); }}
    .card.duplicate {{ border-color: rgba(166,52,52,.34); }}
    .thumb-wrap {{ background: #eceae3; height: var(--thumb-h); display: flex; align-items: center; justify-content: center; }}
    .thumb-wrap.broken::after {{ content: "Preview unavailable - click to open"; color: var(--muted); font-size: 13px; padding: 12px; text-align: center; }}
    .thumb-wrap a {{ width: 100%; height: 100%; display: flex; align-items: center; justify-content: center; }}
    img {{ max-width: 100%; max-height: var(--thumb-h); object-fit: contain; display: block; }}
    img.broken {{ display: none; }}
    .meta {{ display: grid; gap: 5px; padding: 10px 12px; font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }}
    .meta strong {{ color: var(--ink); font-size: 13px; }}
    body[data-density="compact"] .meta strong,
    body[data-density="compact"] .meta .detail-main,
    body[data-density="compact"] .meta .detail-extra {{ display: none; }}
    body[data-density="compact"] .meta {{ padding: 6px 8px; }}
    body[data-density="compact"] .actions {{ padding: 0 8px 8px; }}
    body[data-density="compact"] .actions button {{ display: none; }}
    .badge {{ width: fit-content; padding: 2px 6px; border-radius: 4px; background: #eee; color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: .02em; }}
    .keeper .badge {{ background: rgba(31,122,77,.12); color: var(--good); }}
    .duplicate .badge {{ background: rgba(166,52,52,.10); color: var(--bad); }}
    .actions {{ padding: 0 12px 12px; }}
    .select-row {{ display: inline-flex; align-items: center; gap: 6px; margin-right: auto; font-size: 13px; color: var(--muted); }}
    input[type="checkbox"] {{ width: 16px; height: 16px; }}
    button {{ border: 1px solid var(--line); background: white; color: var(--ink); border-radius: 6px; padding: 8px 10px; cursor: pointer; font-weight: 600; }}
    button:hover {{ background: #f0f0ec; }}
    button.move {{ color: var(--bad); border-color: rgba(166,52,52,.35); }}
    button.keep {{ color: var(--good); border-color: rgba(31,122,77,.35); }}
    button.warn {{ color: var(--bad); border-color: rgba(166,52,52,.35); }}
    button.primary {{ background: #222; color: white; border-color: #222; }}
    .empty {{ padding: 44px 0; color: var(--muted); }}
    .mode-note {{ color: #744f10; background: #fff7dc; border: 1px solid #ead89e; border-radius: 6px; padding: 8px 10px; margin-top: 10px; font-size: 13px; }}
    .toast {{ position: fixed; right: 18px; bottom: 18px; background: #222; color: white; border-radius: 8px; padding: 10px 12px; opacity: 0; transform: translateY(8px); transition: .18s; }}
    .toast.show {{ opacity: 1; transform: translateY(0); }}
    .hidden, .filter-hidden, .collapsed {{ display: none !important; }}
    @media (max-width: 820px) {{
      .controls {{ grid-template-columns: 1fr; }}
      .group header {{ align-items: flex-start; flex-direction: column; }}
    }}
  </style>
</head>
<body data-static-mode="{"1" if static_images else "0"}" data-density="compact">
  <div class="top">
    <h1>Duplicate Review</h1>
    <div class="summary">
      <span>{len(groups)} groups pending</span>
      <span>{total_candidates} candidates pending</span>
      <span>{decided_count} decisions saved</span>
      <span>Report: {html.escape(str(report))}</span>
      <span>Moved files go to: {html.escape(str(review_dir))}</span>
    </div>
    <div class="mode-note">{html.escape(mode_note)}</div>
    <div class="controls">
      <select id="personFilter">
        <option value="all">All people ({total_candidates})</option>
        {person_options}
      </select>
      <div class="type-filters">
        <button class="filter-type active" data-type="all">All <span>{total_candidates}</span></button>
        {type_buttons}
      </div>
      <div class="batch-actions">
        <span class="selected-count" id="selectedCount">0 selected</span>
        <button id="selectVisible">Select all filtered</button>
        <button id="clearSelected">Clear</button>
        <button class="keep" id="keepSelected">Keep selected</button>
        <button class="move primary" id="moveSelected">Move selected</button>
      </div>
    </div>
    <div class="quick-actions">
      <span class="hint" id="visibleStats">0 visible</span>
      <button id="showExactForPerson">Show exact for selected person</button>
      <button id="selectExactVisible">Select filtered exact duplicates</button>
      <button class="warn" id="moveExactVisible">Move filtered exact duplicates</button>
      <div class="density-controls" aria-label="View density">
        <button class="density active" data-density="compact">Compact</button>
        <button class="density" data-density="comfortable">Comfort</button>
        <button class="density" data-density="large">Large</button>
      </div>
    </div>
  </div>
  <main>
    {"".join(sections) if sections else '<div class="empty">No pending duplicate or near-visual groups found.</div>'}
    <div class="empty hidden" id="emptyFiltered">No candidates match the current filters.</div>
  </main>
  <div class="toast" id="toast"></div>
  <script>
    const toast = document.getElementById('toast');
    const personFilter = document.getElementById('personFilter');
    const selectedCount = document.getElementById('selectedCount');
    const visibleStats = document.getElementById('visibleStats');
    const emptyFiltered = document.getElementById('emptyFiltered');
    const canDecide = document.body.dataset.staticMode !== '1';
    let activeType = 'all';

    function showToast(text) {{
      toast.textContent = text;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 1800);
    }}

    function isGroupVisible(group) {{
      return !group.classList.contains('filter-hidden') && !group.classList.contains('collapsed');
    }}

    function visibleGroups() {{
      return Array.from(document.querySelectorAll('.group')).filter(isGroupVisible);
    }}

    function visibleCandidateChecks(kind = null) {{
      return Array.from(document.querySelectorAll('.candidate-check')).filter(cb => {{
        const group = cb.closest('.group');
        return group && isGroupVisible(group) && (!kind || group.dataset.kind === kind);
      }});
    }}

    function checkedVisibleCandidateChecks() {{
      return Array.from(document.querySelectorAll('.candidate-check:checked'))
        .filter(cb => isGroupVisible(cb.closest('.group')));
    }}

    function pathsFromChecks(checks) {{
      return Array.from(new Set(checks.map(cb => cb.dataset.path)));
    }}

    function updateSelected() {{
      const count = checkedVisibleCandidateChecks().length;
      const visibleCount = visibleCandidateChecks().length;
      const exactCount = visibleCandidateChecks('exact_file').length;
      selectedCount.textContent = `${{count}} selected`;
      if (visibleStats) {{
        visibleStats.textContent = `${{visibleCount}} visible candidates | ${{exactCount}} visible exact`;
      }}
    }}

    function setActiveType(type) {{
      activeType = type;
      document.querySelectorAll('.filter-type').forEach(button => {{
        button.classList.toggle('active', button.dataset.type === type);
      }});
      applyFilters();
    }}

    function setDensity(density) {{
      document.body.dataset.density = density;
      try {{ localStorage.setItem('duplicateReviewDensity', density); }} catch (error) {{}}
      document.querySelectorAll('.density').forEach(button => {{
        button.classList.toggle('active', button.dataset.density === density);
      }});
    }}

    function cleanupEmpty() {{
      document.querySelectorAll('.group').forEach(group => {{
        if (!group.querySelector('.card:not(.keeper)')) group.remove();
      }});
      document.querySelectorAll('.person-section').forEach(section => {{
        if (!section.querySelector('.group')) section.remove();
      }});
      updateSelected();
    }}

    function applyFilters() {{
      const person = personFilter.value;
      document.querySelectorAll('.group').forEach(group => {{
        const personOk = person === 'all' || group.dataset.person === person;
        const typeOk = activeType === 'all' || group.dataset.kind === activeType;
        group.classList.toggle('filter-hidden', !(personOk && typeOk));
      }});
      document.querySelectorAll('.person-section').forEach(section => {{
        const hasVisible = Array.from(section.querySelectorAll('.group')).some(isGroupVisible);
        section.classList.toggle('filter-hidden', !hasVisible);
      }});
      if (emptyFiltered) emptyFiltered.classList.toggle('hidden', visibleGroups().length > 0);
      updateSelected();
    }}

    async function decide(path, action) {{
      await decidePaths([path], action, null);
    }}

    async function decidePaths(paths, action, confirmText) {{
      if (!canDecide) {{
        showToast('Open with face option 8 to keep or move files');
        return;
      }}
      if (!paths.length) {{
        showToast('No visible candidates selected');
        return;
      }}
      if (confirmText && !confirm(confirmText)) return;
      const body = paths.length === 1
        ? new URLSearchParams({{path: paths[0], action}})
        : new URLSearchParams({{action, paths: JSON.stringify(paths)}});
      const endpoint = paths.length === 1 ? '/decide' : '/decide_many';
      let res;
      try {{
        res = await fetch(endpoint, {{method:'POST', body}});
      }} catch (error) {{
        showToast('Review server is not running');
        return;
      }}
      const data = await res.json();
      if (!res.ok) {{
        showToast(data.error || 'Action failed');
        return;
      }}
      const decidedPaths = data.paths || paths;
      for (const decidedPath of decidedPaths) {{
        document.querySelectorAll('.card[data-path]').forEach(card => {{
          if (card.dataset.path === decidedPath) card.remove();
        }});
      }}
      cleanupEmpty();
      showToast(data.message);
    }}

    async function decideMany(action) {{
      const paths = pathsFromChecks(checkedVisibleCandidateChecks());
      const verb = action === 'move' ? 'move to ready_to_delete' : 'keep';
      await decidePaths(paths, action, `${{verb}} ${{paths.length}} selected candidate(s)?`);
    }}

    document.addEventListener('click', (event) => {{
      const button = event.target.closest('button');
      if (!button) return;
      if (button.dataset.path) {{
        decide(button.dataset.path, button.classList.contains('move') ? 'move' : 'keep');
      }}
      if (button.classList.contains('skip-group')) {{
        const section = button.closest('.group');
        if (section) section.classList.add('collapsed');
        updateSelected();
      }}
      if (button.classList.contains('select-group')) {{
        button.closest('.group').querySelectorAll('.candidate-check').forEach(cb => cb.checked = true);
        updateSelected();
      }}
    }});

    document.querySelectorAll('.filter-type').forEach(button => {{
      button.addEventListener('click', () => {{
        setActiveType(button.dataset.type);
      }});
    }});
    personFilter.addEventListener('change', applyFilters);
    document.addEventListener('change', event => {{
      if (event.target.classList && event.target.classList.contains('candidate-check')) updateSelected();
    }});
    document.getElementById('selectVisible').addEventListener('click', () => {{
      const filtered = visibleCandidateChecks();
      filtered.forEach(cb => cb.checked = true);
      updateSelected();
      showToast(`Selected ${{filtered.length}} filtered candidate(s)`);
    }});
    document.getElementById('showExactForPerson').addEventListener('click', () => setActiveType('exact_file'));
    document.getElementById('selectExactVisible').addEventListener('click', () => {{
      const exact = visibleCandidateChecks('exact_file');
      exact.forEach(cb => cb.checked = true);
      updateSelected();
      showToast(`Selected ${{exact.length}} visible exact duplicate(s)`);
    }});
    document.getElementById('moveExactVisible').addEventListener('click', () => {{
      const paths = pathsFromChecks(visibleCandidateChecks('exact_file'));
      decidePaths(paths, 'move', `move ${{paths.length}} visible exact duplicate candidate(s) to ready_to_delete?`);
    }});
    document.getElementById('clearSelected').addEventListener('click', () => {{
      document.querySelectorAll('.candidate-check').forEach(cb => cb.checked = false);
      updateSelected();
    }});
    document.getElementById('keepSelected').addEventListener('click', () => decideMany('keep'));
    document.getElementById('moveSelected').addEventListener('click', () => decideMany('move'));
    document.querySelectorAll('.density').forEach(button => {{
      button.addEventListener('click', () => setDensity(button.dataset.density));
    }});
    document.addEventListener('error', event => {{
      if (event.target && event.target.tagName === 'IMG') {{
        event.target.classList.add('broken');
        const wrap = event.target.closest('.thumb-wrap');
        if (wrap) wrap.classList.add('broken');
      }}
    }}, true);
    try {{
      setDensity(localStorage.getItem('duplicateReviewDensity') || 'compact');
    }} catch (error) {{
      setDensity('compact');
    }}
    applyFilters();
  </script>
</body>
</html>
"""


class ReviewServer(ThreadingHTTPServer):
    root: Path
    report: Path
    review_dir: Path
    decisions_path: Path
    limit: int | None


def make_handler(server_state: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            if not server_state["quiet"]:
                super().log_message(fmt, *args)

        def send_text(self, body: str, content_type: str = "text/html", status: int = 200) -> None:
            data = body.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", f"{content_type}; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/image":
                self.serve_image(parsed.query)
                return
            if parsed.path == "/thumb":
                self.serve_thumb(parsed.query)
                return
            decisions = load_decisions(server_state["decisions"])
            groups = load_groups(server_state["report"], decisions, server_state["root"], server_state["limit"])
            self.send_text(render_html(
                groups,
                server_state["root"],
                server_state["report"],
                server_state["review_dir"],
                decisions,
                thumbnails=server_state.get("thumbnails") or {},
            ))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path not in {"/decide", "/decide_many"}:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            params = parse_qs(body)
            action = params.get("action", [""])[0]
            try:
                if parsed.path == "/decide_many":
                    raw_paths = params.get("paths", ["[]"])[0]
                    paths = [Path(p) for p in json.loads(raw_paths)]
                    result = apply_decisions(paths, action, server_state)
                    self.send_text(json.dumps(result), "application/json")
                    return
                path = Path(params.get("path", [""])[0])
                message = apply_decision(path, action, server_state)
            except Exception as exc:
                self.send_text(json.dumps({"error": str(exc)}), "application/json", 400)
                return
            self.send_text(json.dumps({"message": message}), "application/json")

        def serve_image(self, query: str) -> None:
            params = parse_qs(query)
            raw = params.get("path", [""])[0]
            path = Path(unquote(raw)).resolve()
            try:
                path.relative_to(server_state["root"].resolve())
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
            try:
                data = path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def serve_thumb(self, query: str) -> None:
            params = parse_qs(query)
            raw = params.get("path", [""])[0]
            path = Path(unquote(raw)).resolve()
            try:
                path.relative_to(server_state["root"].resolve())
            except ValueError:
                self.send_error(HTTPStatus.FORBIDDEN)
                return
            if not path.exists() or not path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            thumb = ensure_thumbnail(path, server_state["thumb_dir"])
            if thumb is None or not thumb.exists():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            server_state.setdefault("thumbnails", {})[decision_key(path)] = thumb
            try:
                data = thumb.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "image/jpeg")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    return Handler


def apply_decision(path: Path, action: str, state: dict) -> str:
    root = state["root"].resolve()
    path = path.resolve()
    try:
        rel = path.relative_to(root)
    except ValueError as exc:
        raise ValueError("path is outside photos_by_person") from exc
    if action not in {"keep", "move"}:
        raise ValueError("unknown action")
    if not path.exists():
        raise ValueError("file no longer exists")
    kind = state.get("candidate_kinds", {}).get(str(path))
    if action == "move" and kind != "exact_file":
        raise ValueError("only exact-file duplicates can be moved")

    decisions = load_decisions(state["decisions"])
    items = decisions.setdefault("items", {})
    record = {
        "action": action,
        "path": str(path),
        "decided_at": int(time.time()),
    }
    if action == "move":
        dest = unique_dest(state["review_dir"] / rel)
        operation_ledger.move_path(
            path,
            dest,
            sorted_root=DEFAULT_SORTED,
            operation="duplicate_review.move_candidate",
            reason="manual duplicate/near-visual review move",
            extra={"relative_path": rel.as_posix()},
        )
        record["moved_to"] = str(dest)
        message = f"Moved {path.name}"
    else:
        message = f"Kept {path.name}"
    items[decision_key(path)] = record
    save_decisions(state["decisions"], decisions)
    return message


def apply_decisions(paths: list[Path], action: str, state: dict) -> dict:
    if action not in {"keep", "move"}:
        raise ValueError("unknown action")
    moved_or_kept: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    for path in paths:
        key = str(path)
        if key in seen:
            continue
        seen.add(key)
        try:
            apply_decision(path, action, state)
            moved_or_kept.append(str(path.resolve()))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{path}: {exc}")
    if errors and not moved_or_kept:
        raise ValueError("; ".join(errors[:3]))
    noun = "Moved" if action == "move" else "Kept"
    message = f"{noun} {len(moved_or_kept)} file(s)"
    if errors:
        message += f"; {len(errors)} failed"
    return {
        "message": message,
        "paths": moved_or_kept,
        "errors": errors,
    }


def write_static_html(output: Path, groups: list[ReviewGroup], root: Path,
                      report: Path, review_dir: Path, decisions: dict,
                      thumbnails: dict[str, Path]) -> None:
    html_text = render_html(
        groups,
        root,
        report,
        review_dir,
        decisions,
        thumbnails=thumbnails,
        static_images=True,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--root", type=Path, default=DEFAULT_PEOPLE)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--thumb-dir", type=Path, default=DEFAULT_THUMB_DIR,
                        help="Cache folder for browser-safe JPEG thumbnails.")
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-open", action="store_true")
    parser.add_argument("--html-only", action="store_true",
                        help="Write a static preview HTML file and exit. Buttons require the server mode.")
    parser.add_argument("--output", type=Path,
                        default=DEFAULT_SORTED / "_source_review" / "duplicate_reports" / "near_visual_review.html")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = args.root.expanduser().resolve()
    report = args.report.expanduser().resolve()
    review_dir = args.review_dir.expanduser().resolve()
    thumb_dir = args.thumb_dir.expanduser().resolve()
    decisions_path = args.decisions.expanduser().resolve()
    if not root.exists():
        print(f"ERROR: photos folder not found: {root}")
        return 1
    if not report.exists():
        print(f"ERROR: duplicate report not found: {report}")
        print("Run: python face.py health")
        return 1

    decisions = load_decisions(decisions_path)
    groups = load_groups(report, decisions, root, args.limit)
    print(f"Report:             {report}")
    print(f"Pending groups:     {len(groups)}")
    print(f"Decisions file:     {decisions_path}")
    print(f"Move destination:   {review_dir}")
    thumbnails = prepare_thumbnails(groups, thumb_dir, quiet=args.quiet)

    if args.html_only:
        output = args.output.expanduser().resolve()
        write_static_html(output, groups, root, report, review_dir, decisions, thumbnails)
        print(f"HTML preview:       {output}")
        return 0

    state = {
        "root": root,
        "report": report,
        "review_dir": review_dir,
        "thumb_dir": thumb_dir,
        "thumbnails": thumbnails,
        "candidate_kinds": candidate_kinds_by_path(groups),
        "decisions": decisions_path,
        "limit": args.limit,
        "quiet": args.quiet,
    }
    server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(state))
    url = f"http://{args.host}:{args.port}/"
    print(f"Review URL:         {url}")
    print("Press Ctrl+C to stop the review server.")
    if not args.no_open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped near-visual review server.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

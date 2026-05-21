#!/usr/bin/env python3
"""
Review near-visual duplicate candidates in a local browser.

The tool reads the CSV produced by advanced_duplicate_matching.py and serves a
small local page. Nothing is moved automatically: each candidate needs an
explicit Move or Keep click.
"""

from __future__ import annotations

import argparse
import csv
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

DEFAULT_SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
DEFAULT_PEOPLE = DEFAULT_SORTED / "photos_by_person"
DEFAULT_REPORT = DEFAULT_SORTED / "_source_review" / "duplicate_reports" / "advanced_duplicates.csv"
DEFAULT_REVIEW_DIR = DEFAULT_SORTED / "_source_review" / "ready_to_delete" / "near_visual_duplicates"
DEFAULT_DECISIONS = Path.home() / ".face_sort_cache" / "near_visual_review_decisions.json"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif"}


@dataclass
class Candidate:
    path: Path
    action: str
    confidence: int
    distance: int | None
    width: int
    height: int
    size_bytes: int


@dataclass
class ReviewGroup:
    group_id: str
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


def load_groups(report: Path, decisions: dict, root: Path, limit: int | None = None) -> list[ReviewGroup]:
    grouped: dict[str, ReviewGroup] = {}
    if not report.exists():
        return []
    decided = decisions.setdefault("items", {})
    with report.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("type") != "visually_similar":
                continue
            path = Path(row.get("file_path", ""))
            if not path.exists() or not path.suffix.lower() in IMAGE_EXTS:
                continue
            key = decision_key(path)
            if key in decided:
                continue
            group_id = row.get("group_id", "")
            group = grouped.setdefault(
                group_id,
                ReviewGroup(
                    group_id=group_id,
                    scope=row.get("scope", ""),
                    confidence=int_or_zero(row.get("confidence")),
                    distance=int_or_none(row.get("distance")),
                ),
            )
            keeper = Path(row.get("keeper_path", ""))
            if keeper.exists():
                group.keeper = keeper
            candidate = Candidate(
                path=path,
                action=row.get("action", ""),
                confidence=int_or_zero(row.get("confidence")),
                distance=int_or_none(row.get("distance")),
                width=int_or_zero(row.get("width")),
                height=int_or_zero(row.get("height")),
                size_bytes=int_or_zero(row.get("size_bytes")),
            )
            group.candidates.append(candidate)

    out: list[ReviewGroup] = []
    root = root.resolve()
    for group in grouped.values():
        review_candidates = [c for c in group.candidates if c.action == "review"]
        keepers = [c for c in group.candidates if c.action == "keep"]
        if group.keeper is None and keepers:
            group.keeper = keepers[0].path
        if not review_candidates or group.keeper is None:
            continue
        try:
            group.keeper.resolve().relative_to(root)
        except ValueError:
            continue
        group.candidates = sorted(group.candidates, key=lambda c: (c.action != "keep", str(c.path).lower()))
        out.append(group)
    out.sort(key=lambda g: (-g.confidence, g.distance if g.distance is not None else 99, g.scope.lower(), int_or_zero(g.group_id)))
    if limit:
        return out[:limit]
    return out


def safe_rel(path: Path, root: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return path.name


def image_url(path: Path) -> str:
    return "/image?path=" + quote(str(path), safe="")


def render_card(candidate: Candidate, root: Path, is_keeper: bool) -> str:
    path = candidate.path
    label = "KEEPER" if is_keeper else "REVIEW"
    badge = "keeper" if is_keeper else "review"
    buttons = ""
    escaped_path = html.escape(str(path), quote=True)
    if not is_keeper:
        buttons = f"""
          <div class="actions">
            <button class="move" data-path="{escaped_path}">Move to Ready To Delete</button>
            <button class="keep" data-path="{escaped_path}">Keep</button>
          </div>
        """
    return f"""
      <article class="card {badge}">
        <div class="thumb-wrap">
          <img loading="lazy" src="{image_url(path)}" alt="{html.escape(path.name)}">
        </div>
        <div class="meta">
          <span class="badge">{label}</span>
          <strong>{html.escape(path.name)}</strong>
          <span>{candidate.width} x {candidate.height} | {human_size(candidate.size_bytes)}</span>
          <span>{html.escape(safe_rel(path, root))}</span>
        </div>
        {buttons}
      </article>
    """


def render_html(groups: list[ReviewGroup], root: Path, report: Path, review_dir: Path, decisions: dict) -> str:
    cards = []
    for group in groups:
        keeper_card = ""
        review_cards = []
        for candidate in group.candidates:
            is_keeper = group.keeper is not None and candidate.path.resolve() == group.keeper.resolve()
            if is_keeper:
                keeper_card = render_card(candidate, root, True)
            elif candidate.action == "review":
                review_cards.append(render_card(candidate, root, False))
        if not keeper_card and group.keeper:
            keeper_card = render_card(
                Candidate(group.keeper, "keep", group.confidence, group.distance, 0, 0, 0),
                root,
                True,
            )
        cards.append(f"""
          <section class="group" id="group-{html.escape(group.group_id)}">
            <header>
              <div>
                <h2>Group {html.escape(group.group_id)} - {html.escape(group.scope)}</h2>
                <p>Confidence {group.confidence}% | pHash distance {group.distance if group.distance is not None else "?"}</p>
              </div>
              <button class="skip-group" data-group="{html.escape(group.group_id)}">Skip group</button>
            </header>
            <div class="grid">
              {keeper_card}
              {"".join(review_cards)}
            </div>
          </section>
        """)
    decided_count = len(decisions.get("items", {}))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Near Visual Duplicate Review</title>
  <style>
    :root {{ color-scheme: light; --bg:#f7f7f4; --ink:#222; --muted:#6a6a65; --line:#d9d7cf; --good:#1f7a4d; --warn:#a45b14; --bad:#a63434; }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; background: var(--bg); color: var(--ink); }}
    .top {{ position: sticky; top: 0; z-index: 5; background: rgba(247,247,244,.94); border-bottom: 1px solid var(--line); backdrop-filter: blur(12px); padding: 14px 22px; }}
    h1 {{ margin: 0; font-size: 22px; }}
    .summary {{ display: flex; flex-wrap: wrap; gap: 14px; margin-top: 8px; color: var(--muted); font-size: 13px; }}
    main {{ padding: 18px 22px 40px; }}
    .group {{ border-top: 1px solid var(--line); padding: 20px 0 26px; }}
    .group header {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; margin-bottom: 12px; }}
    h2 {{ margin: 0; font-size: 18px; }}
    p {{ margin: 4px 0 0; color: var(--muted); }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(240px, 1fr)); gap: 14px; align-items: start; }}
    .card {{ background: white; border: 1px solid var(--line); border-radius: 8px; overflow: hidden; }}
    .card.keeper {{ border-color: rgba(31,122,77,.55); }}
    .thumb-wrap {{ background: #eceae3; height: 320px; display: flex; align-items: center; justify-content: center; }}
    img {{ max-width: 100%; max-height: 320px; object-fit: contain; display: block; }}
    .meta {{ display: grid; gap: 5px; padding: 10px 12px; font-size: 12px; color: var(--muted); overflow-wrap: anywhere; }}
    .meta strong {{ color: var(--ink); font-size: 13px; }}
    .badge {{ width: fit-content; padding: 2px 6px; border-radius: 4px; background: #eee; color: var(--muted); font-size: 11px; font-weight: 700; letter-spacing: .02em; }}
    .keeper .badge {{ background: rgba(31,122,77,.12); color: var(--good); }}
    .actions {{ display: flex; gap: 8px; padding: 0 12px 12px; }}
    button {{ border: 1px solid var(--line); background: white; color: var(--ink); border-radius: 6px; padding: 8px 10px; cursor: pointer; font-weight: 600; }}
    button:hover {{ background: #f0f0ec; }}
    button.move {{ color: var(--bad); border-color: rgba(166,52,52,.35); }}
    button.keep {{ color: var(--good); border-color: rgba(31,122,77,.35); }}
    .empty {{ padding: 44px 0; color: var(--muted); }}
    .toast {{ position: fixed; right: 18px; bottom: 18px; background: #222; color: white; border-radius: 8px; padding: 10px 12px; opacity: 0; transform: translateY(8px); transition: .18s; }}
    .toast.show {{ opacity: 1; transform: translateY(0); }}
  </style>
</head>
<body>
  <div class="top">
    <h1>Near Visual Duplicate Review</h1>
    <div class="summary">
      <span>{len(groups)} groups pending</span>
      <span>{decided_count} decisions saved</span>
      <span>Report: {html.escape(str(report))}</span>
      <span>Moved files go to: {html.escape(str(review_dir))}</span>
    </div>
  </div>
  <main>
    {"".join(cards) if cards else '<div class="empty">No pending near-visual duplicate groups found.</div>'}
  </main>
  <div class="toast" id="toast"></div>
  <script>
    const toast = document.getElementById('toast');
    function showToast(text) {{
      toast.textContent = text;
      toast.classList.add('show');
      setTimeout(() => toast.classList.remove('show'), 1800);
    }}
    async function decide(path, action) {{
      const body = new URLSearchParams({{path, action}});
      const res = await fetch('/decide', {{method:'POST', body}});
      const data = await res.json();
      if (!res.ok) {{
        showToast(data.error || 'Action failed');
        return;
      }}
      const button = document.querySelector(`button[data-path="${{CSS.escape(path)}}"]`);
      const card = button ? button.closest('.card') : null;
      if (card) card.remove();
      showToast(data.message);
    }}
    document.addEventListener('click', (event) => {{
      const button = event.target.closest('button');
      if (!button) return;
      if (button.dataset.path) {{
        decide(button.dataset.path, button.classList.contains('move') ? 'move' : 'keep');
      }}
      if (button.classList.contains('skip-group')) {{
        const section = document.getElementById('group-' + button.dataset.group);
        if (section) section.remove();
      }}
    }});
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
            decisions = load_decisions(server_state["decisions"])
            groups = load_groups(server_state["report"], decisions, server_state["root"], server_state["limit"])
            self.send_text(render_html(groups, server_state["root"], server_state["report"], server_state["review_dir"], decisions))

        def do_POST(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path != "/decide":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length).decode("utf-8")
            params = parse_qs(body)
            path = Path(params.get("path", [""])[0])
            action = params.get("action", [""])[0]
            try:
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

    decisions = load_decisions(state["decisions"])
    items = decisions.setdefault("items", {})
    record = {
        "action": action,
        "path": str(path),
        "decided_at": int(time.time()),
    }
    if action == "move":
        dest = unique_dest(state["review_dir"] / rel)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(dest))
        record["moved_to"] = str(dest)
        message = f"Moved {path.name}"
    else:
        message = f"Kept {path.name}"
    items[decision_key(path)] = record
    save_decisions(state["decisions"], decisions)
    return message


def write_static_html(output: Path, groups: list[ReviewGroup], root: Path,
                      report: Path, review_dir: Path, decisions: dict) -> None:
    html_text = render_html(groups, root, report, review_dir, decisions)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_text, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--root", type=Path, default=DEFAULT_PEOPLE)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
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

    if args.html_only:
        output = args.output.expanduser().resolve()
        write_static_html(output, groups, root, report, review_dir, decisions)
        print(f"HTML preview:       {output}")
        return 0

    state = {
        "root": root,
        "report": report,
        "review_dir": review_dir,
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

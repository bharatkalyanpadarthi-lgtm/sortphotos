#!/usr/bin/env python3
"""Review exact media stored under conflicting person folders.

The scanner is read-only. Explicit browser actions may keep or ignore a
membership, or move a selected membership to recoverable review storage. Every
move is ledgered; this tool never permanently deletes an original.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import mimetypes
import os
import sqlite3
import threading
import time
import webbrowser
from collections import Counter, defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterable
from urllib.parse import parse_qs, unquote, urlencode, urlparse

import appearance_profiles
import identity_profiles
import identity_hard_negatives
import near_visual_review
import operation_ledger
import pipeline_paths
import sort_photos


DEFAULT_OUTPUT_DIR = pipeline_paths.SOURCE_REVIEW / "identity_audits"
DEFAULT_DECISIONS = DEFAULT_OUTPUT_DIR / "cross_person_identity_decisions.json"
DEFAULT_REVIEW_DIR = (
    pipeline_paths.SOURCE_REVIEW / "ready_to_delete" / "cross_person_identity"
)
DEFAULT_THUMB_DIR = DEFAULT_OUTPUT_DIR / "cross_person_thumbnails"


@dataclass(frozen=True)
class Membership:
    person: str
    path: Path


@dataclass(frozen=True)
class StrictMatch:
    person: str
    distance: float
    margin: float
    quality: float


@dataclass(frozen=True)
class AuditGroup:
    content_key: str
    memberships: tuple[Membership, ...]
    representative: Path
    detected_faces: int
    strict_matches: tuple[StrictMatch, ...]
    status: str
    predicted_person: str

    @property
    def people(self) -> tuple[str, ...]:
        return tuple(sorted({item.person for item in self.memberships}, key=str.casefold))

    @property
    def wrong_people(self) -> tuple[str, ...]:
        if self.status != "high_confidence_conflict":
            return ()
        return tuple(person for person in self.people if person != self.predicted_person)


def membership_key(group: AuditGroup, membership: Membership, people_dir: Path) -> str:
    try:
        relative = membership.path.resolve(strict=False).relative_to(
            people_dir.resolve(strict=False)
        ).as_posix()
    except ValueError:
        relative = str(membership.path.resolve(strict=False))
    return f"{group.content_key}|{membership.person.casefold()}|{relative}"


def load_decisions(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"version": 1, "items": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), dict):
        return {"version": 1, "items": {}}
    payload.setdefault("version", 1)
    return payload


def save_decisions(path: Path, decisions: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(decisions, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def current_fingerprints(index_path: Path) -> dict[str, tuple[int, int, str]]:
    if not index_path.is_file():
        return {}
    uri = index_path.resolve().as_uri() + "?mode=ro"
    try:
        connection = sqlite3.connect(uri, uri=True, timeout=10.0)
        rows = connection.execute(
            "SELECT path, byte_size, mtime_ns, sha256 FROM assets WHERE sha256 != ''"
        ).fetchall()
        connection.close()
    except sqlite3.Error:
        return {}
    return {
        str(path): (int(byte_size), int(mtime_ns), str(sha256))
        for path, byte_size, mtime_ns, sha256 in rows
    }


def content_groups(
    people_dir: Path,
    fingerprints: dict[str, tuple[int, int, str]],
) -> tuple[dict[str, list[Membership]], Counter]:
    groups: dict[str, list[Membership]] = defaultdict(list)
    stats: Counter = Counter()
    for path in sort_photos.iter_person_original_images(people_dir):
        try:
            stat = path.stat()
            person = path.relative_to(people_dir).parts[0]
        except (OSError, ValueError, IndexError):
            stats["read_errors"] += 1
            continue
        indexed = fingerprints.get(str(path.resolve(strict=False)))
        if indexed is not None and indexed[:2] == (int(stat.st_size), int(stat.st_mtime_ns)):
            key = f"sha256:{indexed[2]}"
            stats["indexed_files"] += 1
        else:
            key = f"inode:{stat.st_dev}:{stat.st_ino}:{stat.st_size}"
            stats["inode_fallback_files"] += 1
        groups[key].append(Membership(person=person, path=path))
        stats["files"] += 1
    return groups, stats


def strict_match(face: sort_photos.CachedFace, identity_db: sort_photos.IdentityDB) -> StrictMatch | None:
    lighting, captured_at = appearance_profiles.query_attributes(
        face.crop_jpeg, face.src_str
    )
    candidates = identity_profiles.rank_candidates(
        face.embedding,
        identity_db.identities,
        identity_db.prototypes,
        pose_label=str(getattr(face, "pose_label", "unknown") or "unknown"),
        pose_prototypes=identity_db.pose_prototypes,
        lighting_label=lighting,
        capture_timestamp=captured_at,
        appearance_prototypes=identity_db.appearance_prototypes,
        appearance_era_cutoffs=identity_db.appearance_era_cutoffs,
        hard_negatives=identity_hard_negatives.vectors_by_person(
            sort_photos.IDENTITY_HARD_NEGATIVES_FILE
        ),
    )
    if not candidates:
        return None
    best = candidates[0]
    margin = identity_profiles.candidate_margin(candidates)
    threshold = min(
        sort_photos.AUTO_PERSON_SINGLE_MATCH_DIST,
        identity_db.strict_thresholds.get(
            best.name, sort_photos.AUTO_PERSON_SINGLE_MATCH_DIST),
    )
    if (
        best.distance > threshold
        or margin < sort_photos.AUTO_PERSON_SINGLE_MATCH_MARGIN
        or face.quality < sort_photos.AUTO_PERSON_SINGLE_MIN_QUALITY
    ):
        return None
    return StrictMatch(
        person=best.name,
        distance=float(best.distance),
        margin=float(margin),
        quality=float(face.quality),
    )


def analyze_groups(
    groups: dict[str, list[Membership]],
    faces_by_path: dict[str, list[sort_photos.CachedFace]],
    identity_db: sort_photos.IdentityDB,
) -> list[AuditGroup]:
    results: list[AuditGroup] = []
    for content_key, memberships in groups.items():
        people = {item.person for item in memberships}
        if len(people) < 2:
            continue
        representative = max(
            (item.path for item in memberships),
            key=lambda path: len(faces_by_path.get(str(path), ())),
        )
        faces = faces_by_path.get(str(representative), [])
        matches = tuple(
            match
            for face in faces
            if (match := strict_match(face, identity_db)) is not None
        )
        identities = {match.person for match in matches}
        predicted = ""
        if len(faces) == 1 and len(identities) == 1:
            predicted = next(iter(identities))
            if predicted in people and any(person != predicted for person in people):
                status = "high_confidence_conflict"
            elif predicted not in people:
                status = "predicted_outside_memberships"
            else:
                status = "single_face_consistent"
        elif not faces:
            status = "no_cached_face"
        elif len(faces) > 1 and people.issubset(identities):
            status = "multi_person_consistent"
        elif len(faces) > 1:
            status = "multi_face_review"
        else:
            status = "ambiguous_single_face"
        results.append(AuditGroup(
            content_key=content_key,
            memberships=tuple(sorted(
                memberships,
                key=lambda item: (item.person.casefold(), str(item.path).casefold()),
            )),
            representative=representative,
            detected_faces=len(faces),
            strict_matches=matches,
            status=status,
            predicted_person=predicted,
        ))
    return sorted(results, key=lambda group: (
        group.status != "high_confidence_conflict",
        group.predicted_person.casefold(),
        str(group.representative).casefold(),
    ))


def write_csv(path: Path, groups: Iterable[AuditGroup], people_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "status", "content_key", "predicted_person", "membership_person",
            "candidate_action", "distance", "margin", "quality",
            "detected_faces", "relative_path", "absolute_path",
        ])
        for group in groups:
            best = group.strict_matches[0] if group.strict_matches else None
            for membership in group.memberships:
                action = "review"
                if group.status == "high_confidence_conflict":
                    action = "keep" if membership.person == group.predicted_person else "review_wrong_membership"
                writer.writerow([
                    group.status,
                    group.content_key,
                    group.predicted_person,
                    membership.person,
                    action,
                    f"{best.distance:.6f}" if best else "",
                    f"{best.margin:.6f}" if best else "",
                    f"{best.quality:.6f}" if best else "",
                    group.detected_faces,
                    membership.path.relative_to(people_dir).as_posix(),
                    str(membership.path),
                ])


def decision_for(
    decisions: dict,
    group: AuditGroup,
    membership: Membership,
    people_dir: Path,
) -> dict:
    value = decisions.get("items", {}).get(membership_key(group, membership, people_dir), {})
    return value if isinstance(value, dict) else {}


def content_still_matches(group: AuditGroup, path: Path) -> bool:
    if not path.is_file():
        return False
    if group.content_key.startswith("sha256:"):
        expected = group.content_key.split(":", 1)[1]
        try:
            return operation_ledger.sha256_file(path) == expected
        except OSError:
            return False
    if group.content_key.startswith("inode:"):
        parts = group.content_key.split(":")
        if len(parts) != 4:
            return False
        try:
            stat = path.stat()
            return (
                int(stat.st_dev) == int(parts[1])
                and int(stat.st_ino) == int(parts[2])
                and int(stat.st_size) == int(parts[3])
            )
        except (OSError, ValueError):
            return False
    return False


def remove_cache_source(state: dict, source: Path) -> None:
    cache = state.get("cache")
    if cache is None:
        return
    canonical = os.path.realpath(str(source))
    stale_keys = [
        key for key in cache.file_signatures
        if os.path.realpath(str(key)) == canonical
    ]
    for key in stale_keys:
        cache.file_signatures.pop(key, None)
    before = len(cache.faces)
    cache.faces = [
        face for face in cache.faces
        if os.path.realpath(face.src_str) != canonical
    ]
    if stale_keys or len(cache.faces) != before:
        state["cache_dirty"] = True


def find_membership(group: AuditGroup, person: str, path_text: str) -> Membership:
    requested = Path(path_text).expanduser().resolve(strict=False)
    for membership in group.memberships:
        if (
            membership.person == person
            and membership.path.resolve(strict=False) == requested
        ):
            return membership
    raise ValueError("membership is not part of this audit group")


def record_decision(
    decisions: dict,
    group: AuditGroup,
    membership: Membership,
    people_dir: Path,
    action: str,
    **extra: object,
) -> None:
    decisions.setdefault("items", {})[
        membership_key(group, membership, people_dir)
    ] = {
        "action": action,
        "person": membership.person,
        "path": str(membership.path),
        "content_key": group.content_key,
        "decided_at": int(time.time()),
        **extra,
    }


def move_membership_to_review(
    state: dict,
    group: AuditGroup,
    membership: Membership,
    reason: str,
) -> Path:
    people_dir = state["people_dir"].resolve()
    source = membership.path.resolve()
    try:
        relative = source.relative_to(people_dir)
    except ValueError as exc:
        raise ValueError("membership is outside photos_by_person") from exc
    if not content_still_matches(group, source):
        raise ValueError("file changed after the audit; run the audit again")
    destination = sort_photos.unique_path(state["review_dir"] / reason / relative)
    operation_ledger.move_path(
        source,
        destination,
        sorted_root=state["sorted_root"],
        operation=f"cross_person_identity.{reason}",
        reason="manual cross-person identity review",
        extra={
            "content_key": group.content_key,
            "membership_person": membership.person,
            "predicted_person": group.predicted_person,
            "relative_path": relative.as_posix(),
        },
    )
    remove_cache_source(state, source)
    state.setdefault("affected_people", set()).add(membership.person)
    return destination


def apply_decision(
    state: dict,
    *,
    content_key: str,
    person: str,
    path_text: str,
    action: str,
) -> str:
    if action not in {"keep", "correct", "ignore", "discard"}:
        raise ValueError("unknown review action")
    group = state["groups_by_key"].get(content_key)
    if group is None or group.status != "high_confidence_conflict":
        raise ValueError("audit group is no longer available")
    membership = find_membership(group, person, path_text)
    decisions = load_decisions(state["decisions_path"])
    people_dir = state["people_dir"]

    if action in {"keep", "ignore"}:
        if not membership.path.is_file():
            raise ValueError("file no longer exists")
        record_decision(decisions, group, membership, people_dir, action)
        save_decisions(state["decisions_path"], decisions)
        return (
            f"Kept {membership.path.name} under {membership.person}"
            if action == "keep"
            else f"Ignored {membership.path.name} for this audit"
        )

    if action == "discard":
        destination = move_membership_to_review(state, group, membership, "discarded")
        record_decision(
            decisions,
            group,
            membership,
            people_dir,
            "discarded",
            moved_to=str(destination),
        )
        save_decisions(state["decisions_path"], decisions)
        return f"Moved {membership.path.name} to recoverable review storage"

    target_person = membership.person
    target_memberships = [item for item in group.memberships if item.person == target_person]
    if not any(content_still_matches(group, item.path) for item in target_memberships):
        raise ValueError("the selected correct-person copy is no longer available")
    moved = 0
    for item in group.memberships:
        if item.person == target_person:
            record_decision(
                decisions,
                group,
                item,
                people_dir,
                "correct_keep",
                correct_person=target_person,
            )
            continue
        if not item.path.exists():
            continue
        destination = move_membership_to_review(state, group, item, "corrected")
        record_decision(
            decisions,
            group,
            item,
            people_dir,
            "corrected",
            correct_person=target_person,
            moved_to=str(destination),
        )
        moved += 1
        save_decisions(state["decisions_path"], decisions)
    save_decisions(state["decisions_path"], decisions)
    return f"Set {target_person} as correct; moved {moved} conflicting membership(s)"


def render_html(
    groups: list[AuditGroup],
    summary: dict[str, object],
    decisions: dict,
    people_dir: Path,
    *,
    interactive: bool,
    thumb_dir: Path,
) -> str:
    suspects = [group for group in groups if group.status == "high_confidence_conflict"]
    cards: list[str] = []
    pending_groups = 0
    pending_memberships = 0
    action_labels = {
        "keep": "Kept here",
        "ignore": "Ignored",
        "discarded": "Discarded to recovery",
        "correct_keep": "Correct person",
        "corrected": "Corrected to another person",
    }
    for group in suspects:
        best = group.strict_matches[0]
        rows: list[str] = []
        group_pending = 0
        for membership in group.memberships:
            decision = decision_for(decisions, group, membership, people_dir)
            decided_action = str(decision.get("action") or "")
            exists = membership.path.is_file()
            if not decided_action and exists:
                group_pending += 1
                pending_memberships += 1
            model_badge = (
                "<span class='badge suggested'>AI suggested</span>"
                if membership.person == group.predicted_person
                else "<span class='badge conflict'>Possible wrong folder</span>"
            )
            status = action_labels.get(decided_action, "Pending review" if exists else "File moved")
            encoded_path = html.escape(str(membership.path), quote=True)
            disabled = "" if interactive and exists else " disabled"
            buttons = (
                f"<button class='keep' data-action='keep'{disabled}>Keep Here</button>"
                f"<button class='correct' data-action='correct'{disabled}>Correct Person</button>"
                f"<button class='ignore' data-action='ignore'{disabled}>Ignore</button>"
                f"<button class='discard' data-action='discard'{disabled}>Discard Copy</button>"
            )
            rows.append(
                "<section class='membership{}' data-content='{}' data-person='{}' data-path='{}'>"
                "<div class='membership-main'><strong>{}</strong>{}<span class='decision'>{}</span>"
                "<code>{}</code></div><div class='actions'>{}</div></section>".format(
                    " resolved" if decided_action or not exists else "",
                    html.escape(group.content_key, quote=True),
                    html.escape(membership.person, quote=True),
                    encoded_path,
                    html.escape(membership.person),
                    model_badge,
                    html.escape(status),
                    html.escape(str(membership.path)),
                    buttons,
                )
            )
        if group_pending:
            pending_groups += 1
        if interactive:
            image_url = "/image?" + urlencode({"content_key": group.content_key})
        else:
            source = (
                group.representative
                if group.representative.is_file()
                else next(
                    (item.path for item in group.memberships if item.path.is_file()),
                    group.representative,
                )
            )
            thumb = near_visual_review.ensure_thumbnail(source, thumb_dir)
            image_path = thumb if thumb is not None else source
            image_url = image_path.resolve().as_uri()
        cards.append(
            "<article class='{}'><img loading='lazy' src='{}' alt='Candidate image'>"
            "<div><div class='title-row'><h2>{}</h2><span class='model'>AI predicts {}</span></div>"
            "<p class='metrics'>distance {:.3f} | margin {:.3f} | quality {:.3f}</p>{}</div></article>".format(
                "resolved-group" if not group_pending else "",
                html.escape(image_url, quote=True),
                html.escape(" / ".join(group.people)),
                html.escape(group.predicted_person),
                best.distance,
                best.margin,
                best.quality,
                "".join(rows),
            )
        )
    mode_note = (
        "Actions are live. Discard and Correct move files only to recoverable review storage."
        if interactive
        else "Static report. Run Face Terminal > Cross-Person Identity Audit to use actions."
    )
    document = """<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Cross-Person Identity Review</title>
<style>
:root{{color-scheme:dark;--bg:#0b0c0e;--panel:#15171a;--line:#30343a;--text:#f5f5f7;--muted:#a8adb5;--blue:#4da3ff;--green:#5bd46d;--red:#ff6961;--amber:#ffc857}}
*{{box-sizing:border-box}}body{{font:15px -apple-system,BlinkMacSystemFont,sans-serif;margin:0;background:var(--bg);color:var(--text);letter-spacing:0}}
header{{position:sticky;top:0;z-index:5;padding:18px 24px;background:rgba(11,12,14,.96);border-bottom:1px solid var(--line)}}
.header-inner,main{{max-width:1180px;margin:auto}}h1{{font-size:28px;margin:0 0 6px}}.summary,.note,.metrics{{color:var(--muted)}}
.toolbar{{display:flex;gap:14px;align-items:center;margin-top:12px}}main{{padding:8px 24px 40px}}
article{{display:grid;grid-template-columns:260px minmax(0,1fr);gap:20px;padding:22px 0;border-bottom:1px solid var(--line)}}
article.resolved-group{{display:none}}body.show-resolved article.resolved-group{{display:grid}}
img{{width:260px;height:260px;object-fit:contain;background:#000;border:1px solid var(--line);border-radius:6px}}
.title-row{{display:flex;align-items:center;justify-content:space-between;gap:12px}}h2{{font-size:18px;margin:0}}.model{{color:var(--blue);font-weight:600}}
.membership{{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:16px;padding:13px 0;border-top:1px solid var(--line)}}
.membership-main{{display:grid;grid-template-columns:auto auto 1fr;align-items:center;gap:8px;min-width:0}}code{{grid-column:1/-1;font-size:11px;color:var(--muted);overflow-wrap:anywhere}}
.badge,.decision{{font-size:12px;border-radius:4px;padding:3px 6px}}.suggested{{color:var(--green);background:#17351e}}.conflict{{color:var(--amber);background:#382f16}}.decision{{justify-self:end;color:var(--muted)}}
.actions{{display:flex;flex-wrap:wrap;justify-content:flex-end;gap:6px}}button{{min-height:34px;border:1px solid var(--line);border-radius:6px;padding:6px 10px;background:#202328;color:var(--text);font:inherit;cursor:pointer}}
button:disabled{{opacity:.35;cursor:not-allowed}}button.keep{{color:var(--green)}}button.correct{{color:var(--blue)}}button.ignore{{color:var(--muted)}}button.discard{{color:var(--red)}}
#toast{{position:fixed;right:20px;bottom:20px;max-width:420px;padding:12px 15px;border:1px solid var(--line);border-radius:6px;background:#202328;display:none}}
@media(max-width:780px){{article{{grid-template-columns:1fr}}img{{width:100%;height:min(72vw,380px)}}.membership{{grid-template-columns:1fr}}.actions{{justify-content:flex-start}}}}
</style></head><body><header><div class="header-inner"><h1>Cross-Person Identity Review</h1>
<div class="summary">{summary}</div><div class="note">{mode_note}</div>
<div class="toolbar"><label><input id="showResolved" type="checkbox"> Show resolved groups</label></div></div></header>
<main>{cards}</main><div id="toast"></div>
<script>
const interactive={interactive};
const toast=document.getElementById('toast');
function notify(message,error=false){{toast.textContent=message;toast.style.display='block';toast.style.borderColor=error?'var(--red)':'var(--line)';setTimeout(()=>toast.style.display='none',4500);}}
document.getElementById('showResolved').addEventListener('change',event=>document.body.classList.toggle('show-resolved',event.target.checked));
document.querySelectorAll('button[data-action]').forEach(button=>button.addEventListener('click',async()=>{{
  if(!interactive){{notify('Launch this review from the Face Terminal menu to use actions',true);return;}}
  const row=button.closest('.membership');const action=button.dataset.action;const person=row.dataset.person;
  if(action==='correct'&&!confirm(`Set ${{person}} as the correct person and move all conflicting memberships to recoverable storage?`))return;
  if(action==='discard'&&!confirm(`Move this copy under ${{person}} to recoverable storage?`))return;
  row.querySelectorAll('button').forEach(item=>item.disabled=true);
  const body=new URLSearchParams({{action,content_key:row.dataset.content,person,path:row.dataset.path}});
  try{{const response=await fetch('/decide',{{method:'POST',headers:{{'Content-Type':'application/x-www-form-urlencoded'}},body}});const result=await response.json();if(!response.ok)throw new Error(result.error||'Action failed');notify(result.message);setTimeout(()=>location.reload(),350);}}
  catch(error){{notify(error.message,true);row.querySelectorAll('button').forEach(item=>item.disabled=false);}}
}}));
</script></body></html>""".format(
        summary=html.escape(
            f"{summary['current_files']} files scanned | "
            f"{pending_groups} pending groups | {pending_memberships} pending memberships"
        ),
        mode_note=html.escape(mode_note),
        cards="".join(cards) or "<p>No high-confidence conflicts found.</p>",
        interactive="true" if interactive else "false",
    )
    return document


def write_html(
    path: Path,
    groups: list[AuditGroup],
    summary: dict[str, object],
    decisions: dict,
    people_dir: Path,
    thumb_dir: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        render_html(
            groups,
            summary,
            decisions,
            people_dir,
            interactive=False,
            thumb_dir=thumb_dir,
        ),
        encoding="utf-8",
    )


def make_handler(state: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            if not state.get("quiet"):
                super().log_message(fmt, *args)

        def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(data)

        def send_json(self, payload: dict, status: int = 200) -> None:
            self.send_bytes(
                json.dumps(payload).encode("utf-8"),
                "application/json; charset=utf-8",
                status,
            )

        def do_GET(self) -> None:
            parsed = urlparse(self.path)
            if parsed.path == "/image":
                self.serve_image(parsed.query)
                return
            if parsed.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            decisions = load_decisions(state["decisions_path"])
            page = render_html(
                state["groups"],
                state["summary"],
                decisions,
                state["people_dir"],
                interactive=True,
                thumb_dir=state["thumb_dir"],
            )
            self.send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")

        def serve_image(self, query: str) -> None:
            content_key = parse_qs(query).get("content_key", [""])[0]
            group = state["groups_by_key"].get(content_key)
            if group is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            source = next(
                (item.path for item in group.memberships if item.path.is_file()),
                None,
            )
            if source is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            thumbnail = near_visual_review.ensure_thumbnail(source, state["thumb_dir"])
            image_path = thumbnail if thumbnail is not None else source
            try:
                data = image_path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            self.send_bytes(data, content_type)

        def do_POST(self) -> None:
            if urlparse(self.path).path != "/decide":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 64 * 1024)
                params = parse_qs(self.rfile.read(length).decode("utf-8"))
                with state["lock"]:
                    message = apply_decision(
                        state,
                        content_key=params.get("content_key", [""])[0],
                        person=params.get("person", [""])[0],
                        path_text=unquote(params.get("path", [""])[0]),
                        action=params.get("action", [""])[0],
                    )
            except Exception as exc:  # noqa: BLE001
                self.send_json({"error": str(exc)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"message": message})

    return Handler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Review exact files linked or copied into conflicting person folders.")
    parser.add_argument("--people-dir", type=Path, default=pipeline_paths.PEOPLE_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--review-dir", type=Path, default=DEFAULT_REVIEW_DIR)
    parser.add_argument("--thumb-dir", type=Path, default=DEFAULT_THUMB_DIR)
    parser.add_argument("--serve", action="store_true",
                        help="Run the local action server after generating the audit.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8771)
    parser.add_argument("--open", action="store_true", help="Open the visual review in the browser.")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    people_dir = args.people_dir.expanduser().resolve()
    if not people_dir.is_dir():
        print(f"ERROR: people folder is unavailable: {people_dir}")
        return 2
    identity_db = sort_photos.load_identity_db()
    if identity_db is None or not identity_db.identities:
        print("ERROR: identity database is unavailable. Run face rebuild-id first.")
        return 2
    identity_db = sort_photos.normalize_identity_db(identity_db)
    cache = sort_photos.load_cache()
    faces_by_path: dict[str, list[sort_photos.CachedFace]] = defaultdict(list)
    for face in cache.faces:
        faces_by_path[face.src_str].append(face)

    fingerprints = current_fingerprints(pipeline_paths.ANALYSIS_INDEX)
    groups, scan_stats = content_groups(people_dir, fingerprints)
    results = analyze_groups(groups, faces_by_path, identity_db)
    counts = Counter(group.status for group in results)
    suspects = [group for group in results if group.status == "high_confidence_conflict"]
    stamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir.expanduser().resolve()
    decisions_path = args.decisions.expanduser().resolve()
    review_dir = args.review_dir.expanduser().resolve()
    thumb_dir = args.thumb_dir.expanduser().resolve()
    csv_path = output_dir / f"cross_person_identity_audit_{stamp}.csv"
    html_path = output_dir / f"cross_person_identity_audit_{stamp}.html"
    json_path = output_dir / "latest_cross_person_identity_audit.json"
    summary: dict[str, object] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "people_dir": str(people_dir),
        "current_files": int(scan_stats["files"]),
        "indexed_files": int(scan_stats["indexed_files"]),
        "inode_fallback_files": int(scan_stats["inode_fallback_files"]),
        "cross_person_groups": len(results),
        "high_confidence_groups": len(suspects),
        "wrong_memberships": sum(len(group.wrong_people) for group in suspects),
        "status_counts": dict(sorted(counts.items())),
        "csv_report": str(csv_path),
        "html_report": str(html_path),
        "decisions": str(decisions_path),
        "review_storage": str(review_dir),
    }
    decisions = load_decisions(decisions_path)
    write_csv(csv_path, results, people_dir)
    write_html(html_path, results, summary, decisions, people_dir, thumb_dir)
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("Cross-Person Identity Audit")
    print("=" * 60)
    print(f"Current original files:        {scan_stats['files']}")
    print(f"SQLite fingerprints reused:    {scan_stats['indexed_files']}")
    print(f"Inode fallback files:          {scan_stats['inode_fallback_files']}")
    print(f"Cross-person exact groups:     {len(results)}")
    print(f"High-confidence suspect groups:{len(suspects):>6}")
    print(f"Wrong memberships to review:  {summary['wrong_memberships']:>6}")
    print(f"CSV report:                    {csv_path}")
    print(f"Visual report:                 {html_path}")
    print(f"Saved decisions:               {decisions_path}")
    print(f"Recoverable move destination:  {review_dir}")
    print()
    if not args.serve:
        print("STATIC REPORT - no photos were moved or deleted.")
        if args.open:
            webbrowser.open(html_path.resolve().as_uri())
        return 0

    state = {
        "people_dir": people_dir,
        "sorted_root": people_dir.parent,
        "review_dir": review_dir,
        "thumb_dir": thumb_dir,
        "decisions_path": decisions_path,
        "groups": results,
        "groups_by_key": {group.content_key: group for group in results},
        "summary": summary,
        "cache": cache,
        "cache_dirty": False,
        "affected_people": set(),
        "lock": threading.Lock(),
        "quiet": args.quiet,
    }
    try:
        server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(state))
    except OSError:
        server = ThreadingHTTPServer((args.host, 0), make_handler(state))
    port = int(server.server_address[1])
    url = f"http://{args.host}:{port}/"
    print(f"Live review URL:               {url}")
    print("Actions: Keep Here | Correct Person | Ignore | Discard Copy")
    print("Correct and Discard are recoverable moves, never permanent deletion.")
    print("Press Ctrl+C in this terminal when review is finished.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping cross-person identity review...")
    finally:
        server.server_close()
        if state["cache_dirty"]:
            print("Saving updated face cache...")
            sort_photos.save_cache(state["cache"])
            print("Refreshing changed identity profiles...")
            sort_photos.build_identity_db_from_person_folders(people_dir)
        final_decisions = load_decisions(decisions_path)
        write_html(html_path, results, summary, final_decisions, people_dir, thumb_dir)
    print("Cross-person identity review closed safely.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

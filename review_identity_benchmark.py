#!/usr/bin/env python3
"""Local review UI for the protected identity, face, and nudity benchmark."""

from __future__ import annotations

import argparse
import csv
import errno
import html
import json
import mimetypes
import os
import subprocess
import sys
import tempfile
import threading
import webbrowser
from collections import deque
from io import BytesIO
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from itertools import islice
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse
from urllib.request import urlopen

from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except (ImportError, AttributeError):
    pass

import evaluation_dataset
import evaluation_enrollment
import identity_evaluation
import identity_hard_negatives
import pipeline_paths
import sort_photos


FIELDS = evaluation_dataset.FIELDS
VALID_NUDITY = {"safe", "possible", "unknown"}
THUMBNAIL_SIZE = (560, 560)
THUMBNAIL_CACHE_LIMIT = 512


def read_rows(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            return [{field: str(row.get(field, "")) for field in FIELDS}
                    for row in csv.DictReader(handle)]
    except OSError:
        return []


def write_rows(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows({field: row.get(field, "") for field in FIELDS} for row in rows)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def upsert_row(path: Path, row: dict[str, str]) -> None:
    source = str(Path(row["source"]).expanduser().resolve(strict=False))
    normalized = {field: str(row.get(field, "")).strip() for field in FIELDS}
    normalized["source"] = source
    rows = [item for item in read_rows(path) if item.get("source") != source]
    rows.append(normalized)
    rows.sort(key=lambda item: item["source"].casefold())
    write_rows(path, rows)


def _normalized_row(row: dict[str, str]) -> dict[str, str]:
    normalized = {field: str(row.get(field, "")).strip() for field in FIELDS}
    normalized["source"] = str(
        Path(normalized["source"]).expanduser().resolve(strict=False)
    )
    types = evaluation_dataset.parse_case_types(normalized["case_types"])
    normalized["case_types"] = "|".join(sorted(types))
    normalized["expected_face"] = str(
        evaluation_dataset.parse_bool(normalized["expected_face"], default=True)
    ).lower()
    normalized["expected_nudity"] = (
        normalized["expected_nudity"].casefold() or "unknown"
    )
    normalized["verified"] = str(
        evaluation_dataset.parse_bool(normalized["verified"])
    ).lower()
    return normalized


def _verification_errors(row: dict[str, str]) -> list[str]:
    errors: list[str] = []
    source = Path(row["source"])
    case_types = evaluation_dataset.parse_case_types(row["case_types"])
    expected_face = evaluation_dataset.parse_bool(row["expected_face"], default=True)
    if not source.is_file():
        errors.append("source file is missing")
    if not case_types:
        errors.append("case type is required")
    unknown_types = case_types - evaluation_dataset.REQUIRED_CASE_TYPES
    if unknown_types:
        errors.append("unknown case type: " + ", ".join(sorted(unknown_types)))
    if row["expected_nudity"] not in VALID_NUDITY:
        errors.append("nudity must be safe, possible, or unknown")
    if expected_face and "unknown" not in case_types and not (row["expected_person"] or row.get("expected_people")):
        errors.append("person is required for a known face")
    if "group" in case_types:
        try:
            count = int(row.get("expected_face_count") or "")
            people = [name for name in row.get("expected_people", "").split("|") if name.strip()]
            if count < 2 or not people or count < len(people):
                raise ValueError()
        except ValueError:
            errors.append("group requires every known person (separated by |) and total face count")
    try:
        total_faces = int(row.get("expected_face_count") or "")
    except ValueError:
        total_faces = None
    errors.extend(evaluation_dataset.identity_scope_errors(
        row.get("identity_face_id", ""), case_types, expected_face, row["expected_person"],
        tuple(name.strip() for name in row.get("expected_people", "").split("|") if name.strip()),
        total_faces))
    if not errors:
        row["content_sha256"] = sort_photos.content_identity.content_sha256(source)
        row["group_id"] = row.get("group_id") or row["content_sha256"]
    return errors


def bulk_update_rows(
    path: Path,
    sources: list[str],
    updates: dict[str, str],
    *,
    verified: bool | None = None,
) -> list[dict[str, str]]:
    """Apply one validated benchmark update atomically to several source rows."""
    selected = {
        str(Path(source).expanduser().resolve(strict=False))
        for source in sources if str(source).strip()
    }
    if not selected:
        raise ValueError("Select at least one benchmark image.")
    allowed_updates = {
        "expected_person", "expected_people", "expected_face_count", "group_id",
        "case_types", "expected_face", "expected_nudity", "notes"
    }
    invalid_fields = set(updates) - allowed_updates
    if invalid_fields:
        raise ValueError("Unsupported batch field: " + ", ".join(sorted(invalid_fields)))

    rows = read_rows(path)
    found: set[str] = set()
    changed: list[dict[str, str]] = []
    output: list[dict[str, str]] = []
    validation_errors: list[str] = []
    for original in rows:
        source = str(Path(original["source"]).expanduser().resolve(strict=False))
        if source not in selected:
            output.append(original)
            continue
        found.add(source)
        candidate = dict(original)
        candidate.update(updates)
        if verified is not None:
            candidate["verified"] = str(verified).lower()
        candidate = _normalized_row(candidate)
        if evaluation_dataset.parse_bool(candidate["verified"]):
            row_errors = _verification_errors(candidate)
            if row_errors:
                validation_errors.append(
                    f"{Path(source).name}: " + "; ".join(row_errors)
                )
        output.append(candidate)
        changed.append(candidate)

    missing = selected - found
    if missing:
        validation_errors.append(
            f"{len(missing)} selected image(s) are no longer in the benchmark dataset"
        )
    if validation_errors:
        preview = validation_errors[:6]
        if len(validation_errors) > len(preview):
            preview.append(f"and {len(validation_errors) - len(preview)} more")
        raise ValueError("Cannot apply batch update. " + " | ".join(preview))
    output.sort(key=lambda item: item["source"].casefold())
    write_rows(path, output)
    return changed


def _seed_row(
    source: Path,
    *,
    person: str = "",
    case_types: str,
    expected_face: bool,
    nudity: str = "unknown",
    verified: bool = False,
    notes: str = "Suggested candidate; verify manually.",
) -> dict[str, str]:
    return {
        "source": str(source.expanduser().resolve(strict=False)),
        "expected_person": person,
        "case_types": case_types,
        "expected_face": str(expected_face).lower(),
        "expected_nudity": nudity,
        "verified": str(verified).lower(),
        "notes": notes,
    }


def seed_dataset(path: Path) -> int:
    original_rows = read_rows(path)
    existing = {row["source"]: row for row in original_rows}
    db = sort_photos.load_identity_db()
    cache = sort_photos.load_cache()
    if db is not None:
        best_faces: dict[str, sort_photos.CachedFace] = {}
        for face in cache.faces:
            if not face.label:
                continue
            current = best_faces.get(face.label)
            if current is None or face.quality > current.quality:
                best_faces[face.label] = face
        for person, face in best_faces.items():
            source = Path(face.src_str)
            types = {"known"}
            if "profile" in str(face.pose_label):
                types.add("side_profile")
            if face.quality < 0.45:
                types.add("blurry_small")
            try:
                relative = source.resolve(strict=False).relative_to(
                    pipeline_paths.PEOPLE_ROOT.resolve(strict=False)
                )
                if "nude" in {part.casefold() for part in relative.parts}:
                    types.add("nude")
            except ValueError:
                pass
            key = str(source.resolve(strict=False))
            existing.setdefault(key, _seed_row(
                source,
                person=person,
                case_types="|".join(sorted(types)),
                expected_face=True,
                nudity="possible" if "nude" in types else "unknown",
            ))

    review = pipeline_paths.SOURCE_REVIEW / "unassigned_intake"
    for folder, case_type, expected_face in (
        (review / "unknown_identity", "unknown", True),
        (review / "no_usable_face", "no_face", False),
    ):
        if not folder.exists():
            continue
        for source in islice(sort_photos.iter_images(
            folder, excluded_dir_names=set(), always_excluded_dir_names=set()
        ), 8):
            key = str(source.resolve(strict=False))
            existing.setdefault(key, _seed_row(
                source, case_types=case_type, expected_face=expected_face
            ))

    for item in identity_hard_negatives.load(
        sort_photos.IDENTITY_HARD_NEGATIVES_FILE
    ).get("examples", []):
        source = Path(str(item.get("source_path", "")))
        if not source.is_file():
            continue
        key = str(source.resolve(strict=False))
        existing.setdefault(key, _seed_row(
            source,
            case_types="lookalike|unknown",
            expected_face=True,
            notes=f"Explicit lookalike rejection for {item.get('person', '')}; verify manually.",
        ))

    # Enrollment adds candidates; existing benchmark annotations remain authoritative.
    for row in evaluation_enrollment._read(evaluation_enrollment.DEFAULT_PATH):
        source = Path(str(row.get("source", "")))
        if not source.is_file():
            continue
        key = str(source.resolve(strict=False))
        existing.setdefault(key, {field: str(row.get(field, "")) for field in FIELDS})

    rows = list(existing.values())
    if rows != original_rows:
        write_rows(path, rows)
    return len(existing)


class BenchmarkServer(ThreadingHTTPServer):
    dataset: Path
    baseline: Path
    last_gate_output: str = ""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dataset_lock = threading.Lock()
        self.gate_lock = threading.Lock()
        self.thumbnail_lock = threading.Lock()
        self.thumbnail_cache: dict[str, tuple[tuple[int, int], bytes]] = {}

    def run_gate(self) -> str:
        if not self.gate_lock.acquire(blocking=False):
            return "Activation test is already running"
        try:
            validation = evaluation_dataset.load_dataset(self.dataset)
            if not validation.activation_ready:
                missing = sorted(evaluation_dataset.REQUIRED_CASE_TYPES - validation.covered_types)
                self.last_gate_output = (
                    "Benchmark is not ready. Missing verified types: " + ", ".join(missing)
                    + ("\n" + "\n".join(validation.errors) if validation.errors else ""))
                return "Activation blocked"
            command = [sys.executable, "-u", str(Path(identity_evaluation.__file__).resolve()),
                       "--golden-set", str(self.dataset)]
            if self.baseline.is_file():
                command += ["--baseline", str(self.baseline)]
            else:
                command += ["--write-baseline", str(self.baseline), "--fresh-detection"]
            lines = deque(maxlen=200)
            self.last_gate_output = "Activation test running..."
            with subprocess.Popen(command, text=True, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, bufsize=1) as process:
                for line in process.stdout:
                    lines.append(line)
                    self.last_gate_output = "Activation test running...\n" + "".join(lines)
                    print(line, end="", flush=True)
                code = process.wait()
            if code == 0:
                message = "Activation test passed"
            elif code < 0:
                message = f"Activation test interrupted by signal {-code}; no successful result"
            else:
                message = f"Activation test failed (exit {code}); no successful result"
            self.last_gate_output = message + "\n" + "".join(lines)
            return message
        except Exception as error:
            self.last_gate_output = f"Activation test failed: {error}"
            return "Activation test failed"
        finally:
            self.gate_lock.release()


class Handler(BaseHTTPRequestHandler):
    server: BenchmarkServer

    def _redirect(self, message: str = "") -> None:
        location = "/" + ("?" + urlencode({"message": message}) if message else "")
        self.send_response(HTTPStatus.SEE_OTHER)
        self.send_header("Location", location)
        self.end_headers()

    def _json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _thumbnail(self, source: Path) -> bytes:
        stat = source.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        key = str(source.resolve(strict=False))
        with self.server.thumbnail_lock:
            cached = self.server.thumbnail_cache.get(key)
            if cached and cached[0] == signature:
                return cached[1]
        with Image.open(source) as opened:
            opened.seek(0)
            image = ImageOps.exif_transpose(opened).convert("RGB")
            image.thumbnail(THUMBNAIL_SIZE, Image.Resampling.LANCZOS)
            output = BytesIO()
            image.save(output, format="JPEG", quality=84, optimize=True)
            payload = output.getvalue()
        with self.server.thumbnail_lock:
            if len(self.server.thumbnail_cache) >= THUMBNAIL_CACHE_LIMIT:
                oldest = next(iter(self.server.thumbnail_cache), None)
                if oldest is not None:
                    self.server.thumbnail_cache.pop(oldest, None)
            self.server.thumbnail_cache[key] = (signature, payload)
        return payload

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path == "/thumbnail":
            source = Path(parse_qs(parsed.query).get("path", [""])[0]).expanduser()
            if not source.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            try:
                payload = self._thumbnail(source)
            except (OSError, ValueError):
                payload = source.read_bytes()
                content_type = mimetypes.guess_type(source.name)[0] or "application/octet-stream"
            else:
                content_type = "image/jpeg"
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "private, max-age=300")
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/media":
            source = Path(parse_qs(parsed.query).get("path", [""])[0]).expanduser()
            if not source.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            payload = source.read_bytes()
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", mimetypes.guess_type(source.name)[0] or "application/octet-stream")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "private, max-age=300")
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/gate":
            self._redirect(self.server.run_gate())
            return
        self._render(parse_qs(parsed.query).get("message", [""])[0])

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0") or 0)
        values = parse_qs(
            self.rfile.read(length).decode("utf-8"), keep_blank_values=True
        )
        if self.path == "/save":
            row = {field: values.get(field, [""])[0] for field in FIELDS}
            row["verified"] = "true" if values.get("verified") else "false"
            if not row["source"]:
                self._redirect("Source path is required")
                return
            if "identity_face_id" not in values:
                canonical = str(Path(row["source"]).expanduser().resolve(strict=False))
                with self.server.dataset_lock:
                    previous = next((item for item in read_rows(self.server.dataset)
                                     if item["source"] == canonical), {})
                row["identity_face_id"] = previous.get("identity_face_id", "")
            normalized = _normalized_row(row)
            if evaluation_dataset.parse_bool(normalized["verified"]):
                errors = _verification_errors(normalized)
                if errors:
                    self._redirect("Cannot verify: " + "; ".join(errors))
                    return
            with self.server.dataset_lock:
                upsert_row(self.server.dataset, normalized)
            self._redirect("Case saved")
            return
        if self.path == "/bulk-save":
            updates: dict[str, str] = {}
            for field in (
                "expected_person", "case_types", "expected_face",
                "expected_nudity", "notes",
            ):
                if values.get(f"apply_{field}", [""])[0] == "true":
                    updates[field] = values.get(field, [""])[0]
            verified_raw = values.get("verified", [""])[0]
            verified = None if verified_raw == "" else evaluation_dataset.parse_bool(verified_raw)
            try:
                with self.server.dataset_lock:
                    changed = bulk_update_rows(
                        self.server.dataset,
                        values.get("source", []),
                        updates,
                        verified=verified,
                    )
            except ValueError as exc:
                self._json(HTTPStatus.BAD_REQUEST, {"ok": False, "error": str(exc)})
                return
            self._json(HTTPStatus.OK, {
                "ok": True,
                "count": len(changed),
                "rows": changed,
            })
            return
        self.send_error(HTTPStatus.NOT_FOUND)

    def _render(self, message: str) -> None:
        with self.server.dataset_lock:
            rows = read_rows(self.server.dataset)
        validation = evaluation_dataset.load_dataset(self.server.dataset)
        missing = sorted(evaluation_dataset.REQUIRED_CASE_TYPES - validation.covered_types)
        cards: list[str] = []
        for row in rows:
            source = row["source"]
            media = "/media?" + urlencode({"path": source})
            thumbnail = "/thumbnail?" + urlencode({"path": source})
            verified = evaluation_dataset.parse_bool(row["verified"])
            checked = " checked" if verified else ""
            expected_face = evaluation_dataset.parse_bool(
                row["expected_face"], default=True
            )
            status = "verified" if verified else "pending"
            case_types = sorted(evaluation_dataset.parse_case_types(row["case_types"]))
            case_chips = "".join(
                f'<span class="case-chip">{html.escape(case_type)}</span>'
                for case_type in case_types
            ) or '<span class="case-chip muted">unclassified</span>'
            person_summary = row["expected_person"] or (
                "Unknown person" if expected_face else "No face expected"
            )
            search_value = " ".join((
                Path(source).name,
                source,
                row["expected_person"],
                row["case_types"],
                row["notes"],
            )).casefold()
            nudity_options = "".join(
                f'<option value="{value}"'
                f'{" selected" if (row["expected_nudity"] or "unknown") == value else ""}'
                f'>{value.title()}</option>'
                for value in ("unknown", "safe", "possible")
            )
            selected_face = row.get("identity_face_id", "")
            identity_options = '<option value="">All detected faces</option>'
            if selected_face:
                identity_options += (f'<option value="{html.escape(selected_face)}" selected>'
                                     'Confirmed face only</option>')
            cards.append(f"""
            <article class="case-card {status}" data-status="{status}"
                     data-types="{html.escape('|'.join(case_types))}"
                     data-search="{html.escape(search_value)}">
              <div class="image-area">
                <button class="preview-button" type="button"
                        data-full="{html.escape(media)}"
                        data-name="{html.escape(Path(source).name)}"
                        aria-label="Open {html.escape(Path(source).name)} full size">
                  <img src="{html.escape(thumbnail)}" loading="lazy" decoding="async"
                       alt="Benchmark candidate {html.escape(Path(source).name)}">
                </button>
                <label class="select-control" title="Select this image">
                  <input class="row-select" type="checkbox" value="{html.escape(source)}">
                  <span></span>
                </label>
                <span class="status-badge">{"Verified" if verified else "Pending"}</span>
              </div>
              <div class="card-summary">
                <strong class="file-name" title="{html.escape(source)}">{html.escape(Path(source).name)}</strong>
                <div class="person-summary">{html.escape(person_summary)}</div>
                <div class="case-chips">{case_chips}</div>
                <div class="case-meta">{"Face expected" if expected_face else "No face"} &middot; {html.escape((row['expected_nudity'] or 'unknown').title())}</div>
              </div>
              <details class="case-editor">
                <summary>Edit details</summary>
                <form method="post" action="/save">
                <input type="hidden" name="source" value="{html.escape(source)}">
                <label>Person<input name="expected_person" value="{html.escape(row['expected_person'])}"></label>
                <label>All known people<input name="expected_people" value="{html.escape(row.get('expected_people', ''))}" placeholder="Alice | Bob"></label>
                <label>Face count<input type="number" min="0" name="expected_face_count" value="{html.escape(row.get('expected_face_count', ''))}"></label>
                <label>Identity evaluation<select name="identity_face_id">{identity_options}</select></label>
                <label>Source group<input name="group_id" value="{html.escape(row.get('group_id', ''))}"></label>
                <label>Case types<input name="case_types" value="{html.escape(row['case_types'])}"></label>
                <label>Face<select name="expected_face"><option value="true"{' selected' if expected_face else ''}>Expected</option><option value="false"{' selected' if not expected_face else ''}>No face</option></select></label>
                <label>Nudity<select name="expected_nudity">{nudity_options}</select></label>
                <label>Notes<input name="notes" value="{html.escape(row['notes'])}"></label>
                <label class="verify"><input type="checkbox" name="verified" value="true"{checked}> Manually verified</label>
                <button class="primary-button">Save case</button>
                </form>
              </details>
            </article>""")
        verified_count = sum(
            evaluation_dataset.parse_bool(row["verified"]) for row in rows
        )
        pending_count = len(rows) - verified_count
        output = html.escape(self.server.last_gate_output)
        styles = """
        :root { color-scheme: dark; --bg:#0c0c0e; --panel:#18181c; --panel2:#222228;
          --line:#34343b; --text:#f6f6f7; --muted:#a7a7b0; --blue:#0a84ff;
          --green:#30d158; --amber:#ffd60a; --danger:#ff453a; }
        * { box-sizing:border-box; }
        body { margin:0; font:14px -apple-system,BlinkMacSystemFont,"SF Pro Text",sans-serif;
          background:var(--bg); color:var(--text); }
        button,input,select { font:inherit; }
        button { cursor:pointer; }
        .workspace-header { position:sticky; top:0; z-index:20; padding:16px 20px 12px;
          background:rgba(12,12,14,.94); border-bottom:1px solid var(--line);
          backdrop-filter:blur(18px); }
        .title-row,.toolbar,.batch-row,.batch-fields,.metric-row { display:flex; align-items:center; }
        .title-row { justify-content:space-between; gap:18px; }
        h1 { margin:0; font-size:24px; letter-spacing:0; }
        .subtitle { margin:4px 0 0; color:var(--muted); }
        .metric-row { gap:8px; margin-top:12px; flex-wrap:wrap; }
        .metric { padding:5px 9px; border-radius:6px; background:var(--panel2); color:var(--muted); }
        .metric strong { color:var(--text); }
        .toolbar { gap:8px; margin-top:12px; flex-wrap:wrap; }
        .search { min-width:230px; flex:1; }
        input,select { min-height:36px; padding:7px 9px; color:var(--text); background:var(--panel2);
          border:1px solid var(--line); border-radius:6px; }
        input:focus,select:focus { outline:2px solid var(--blue); outline-offset:-1px; }
        .toolbar select { width:auto; }
        .secondary-button,.primary-button,.danger-button,a.primary-button { min-height:36px;
          display:inline-flex; align-items:center; justify-content:center; gap:6px; padding:7px 11px;
          border:1px solid var(--line); border-radius:6px; color:var(--text); background:var(--panel2);
          text-decoration:none; white-space:nowrap; }
        .primary-button,a.primary-button { border-color:var(--blue); background:var(--blue); color:white; }
        .danger-button { border-color:#6e2926; background:#361c1b; color:#ff9b96; }
        .density-switch { display:flex; border:1px solid var(--line); border-radius:6px; overflow:hidden; }
        .density-switch button { min-height:34px; border:0; border-right:1px solid var(--line);
          padding:6px 9px; color:var(--muted); background:var(--panel); }
        .density-switch button:last-child { border-right:0; }
        .density-switch button.active { color:white; background:#3b3b44; }
        .message { color:#9dd0ff; }
        .batch-panel { display:none; margin:12px 0 0; padding:10px; background:#15263a;
          border:1px solid #285987; border-radius:8px; }
        .batch-panel.active { display:block; }
        .batch-row { justify-content:space-between; gap:10px; }
        .batch-fields { gap:8px; margin-top:9px; flex-wrap:wrap; }
        .batch-field { display:flex; align-items:center; gap:6px; min-width:190px; flex:1; }
        .batch-field input[type=checkbox] { min-height:0; width:17px; height:17px; flex:none; }
        .batch-field input[type=text] { width:100%; }
        .batch-field select { width:100%; }
        .quick-presets { display:flex; gap:6px; flex-wrap:wrap; margin-top:8px; }
        .page-content { padding:18px 20px 48px; }
        .gate-output,.add-candidate { margin:0 0 14px; padding:10px 12px; background:var(--panel);
          border:1px solid var(--line); border-radius:7px; }
        .gate-output pre { max-height:220px; overflow:auto; white-space:pre-wrap; color:#d6d6dc; }
        .add-form { display:grid; grid-template-columns:2fr 1fr 2fr auto; gap:8px; margin-top:10px; }
        .grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(190px,1fr)); gap:10px; }
        .grid.comfortable { grid-template-columns:repeat(auto-fill,minmax(270px,1fr)); gap:14px; }
        .case-card { min-width:0; overflow:hidden; background:var(--panel); border:1px solid var(--line);
          border-radius:8px; transition:border-color .12s,box-shadow .12s; }
        .case-card.hidden { display:none; }
        .case-card.selected { border-color:var(--blue); box-shadow:0 0 0 2px rgba(10,132,255,.3); }
        .image-area { position:relative; aspect-ratio:1/1; background:#050506; }
        .preview-button { width:100%; height:100%; padding:0; border:0; background:#050506; }
        .preview-button img { display:block; width:100%; height:100%; object-fit:contain; }
        .select-control { position:absolute; left:8px; top:8px; width:30px; height:30px; padding:6px;
          border-radius:50%; background:rgba(0,0,0,.68); }
        .select-control input { position:absolute; opacity:0; pointer-events:none; }
        .select-control span { display:block; width:18px; height:18px; border:2px solid white;
          border-radius:50%; background:rgba(0,0,0,.2); }
        .select-control input:checked + span { border-color:var(--blue); background:var(--blue); }
        .select-control input:checked + span:after { content:""; display:block; width:8px; height:4px;
          margin:4px 0 0 3px; border-left:2px solid white; border-bottom:2px solid white;
          transform:rotate(-45deg); }
        .status-badge { position:absolute; right:7px; top:7px; padding:4px 7px; border-radius:5px;
          color:#ffe680; background:rgba(83,65,0,.88); font-size:11px; font-weight:700; }
        .verified .status-badge { color:#a8f7b9; background:rgba(19,79,34,.9); }
        .card-summary { padding:10px; }
        .file-name { display:block; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .person-summary { margin-top:5px; color:#ddd; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .case-chips { display:flex; gap:4px; flex-wrap:wrap; min-height:22px; margin-top:7px; }
        .case-chip { padding:3px 5px; border-radius:4px; color:#c7e4ff; background:#16344d; font-size:11px; }
        .case-chip.muted { color:var(--muted); background:#2b2b31; }
        .case-meta { margin-top:7px; color:var(--muted); font-size:12px; }
        .case-editor { border-top:1px solid var(--line); }
        .case-editor summary { padding:9px 10px; color:#b9d9ff; cursor:pointer; }
        .case-editor form { padding:0 10px 10px; }
        .case-editor label { display:block; margin:8px 0; color:var(--muted); }
        .case-editor label input,.case-editor label select { display:block; width:100%; margin-top:4px; }
        .case-editor .verify { display:flex; align-items:center; gap:7px; }
        .case-editor .verify input { display:inline-block; width:17px; height:17px; min-height:0; margin:0; }
        dialog { width:min(92vw,1200px); height:min(92vh,920px); padding:0; border:1px solid #555;
          border-radius:8px; background:#080809; color:white; }
        dialog::backdrop { background:rgba(0,0,0,.82); }
        .viewer-bar { height:48px; display:flex; align-items:center; justify-content:space-between;
          gap:12px; padding:0 12px; border-bottom:1px solid var(--line); }
        .viewer-name { overflow:hidden; text-overflow:ellipsis; white-space:nowrap; }
        .viewer-image { width:100%; height:calc(100% - 48px); object-fit:contain; background:#000; }
        @media (max-width:760px) {
          .workspace-header { padding:12px; }
          .page-content { padding:12px 12px 36px; }
          .grid { grid-template-columns:repeat(auto-fill,minmax(150px,1fr)); gap:7px; }
          .grid.comfortable { grid-template-columns:repeat(auto-fill,minmax(220px,1fr)); }
          .add-form { grid-template-columns:1fr; }
          .title-row { align-items:flex-start; }
          .batch-row { align-items:flex-start; flex-direction:column; }
        }
        """
        script = """
        const grid = document.getElementById('caseGrid');
        const cards = () => [...document.querySelectorAll('.case-card')];
        const checks = () => [...document.querySelectorAll('.row-select')];
        const selected = () => checks().filter(input => input.checked);
        const search = document.getElementById('search');
        const statusFilter = document.getElementById('statusFilter');
        const typeFilter = document.getElementById('typeFilter');
        const batchPanel = document.getElementById('batchPanel');
        const selectedCount = document.getElementById('selectedCount');
        const visibleCount = document.getElementById('visibleCount');
        const message = document.getElementById('message');
        let lastChecked = null;

        function showMessage(text, isError=false) {
          message.textContent = text;
          message.style.color = isError ? '#ff9b96' : '#9dd0ff';
        }
        function updateSelection() {
          cards().forEach(card => card.classList.toggle('selected', card.querySelector('.row-select').checked));
          const count = selected().length;
          selectedCount.textContent = count;
          batchPanel.classList.toggle('active', count > 0);
        }
        function applyFilters() {
          const query = search.value.trim().toLocaleLowerCase();
          const status = statusFilter.value;
          const type = typeFilter.value;
          let visible = 0;
          cards().forEach(card => {
            const matches = (!query || card.dataset.search.includes(query)) &&
              (status === 'all' || card.dataset.status === status) &&
              (type === 'all' || card.dataset.types.split('|').includes(type));
            card.classList.toggle('hidden', !matches);
            if (matches) visible += 1;
          });
          visibleCount.textContent = visible;
          localStorage.setItem('benchmarkSearch', search.value);
          localStorage.setItem('benchmarkStatus', status);
          localStorage.setItem('benchmarkType', type);
        }
        checks().forEach(input => input.addEventListener('click', event => {
          if (event.shiftKey && lastChecked) {
            const visibleChecks = checks().filter(item => !item.closest('.case-card').classList.contains('hidden'));
            const start = visibleChecks.indexOf(lastChecked);
            const end = visibleChecks.indexOf(input);
            if (start >= 0 && end >= 0) {
              visibleChecks.slice(Math.min(start,end), Math.max(start,end)+1)
                .forEach(item => { item.checked = input.checked; });
            }
          }
          lastChecked = input;
          updateSelection();
        }));
        [search,statusFilter,typeFilter].forEach(control => control.addEventListener('input', applyFilters));
        document.getElementById('selectVisible').addEventListener('click', () => {
          cards().filter(card => !card.classList.contains('hidden'))
            .forEach(card => { card.querySelector('.row-select').checked = true; });
          updateSelection();
        });
        document.getElementById('clearSelection').addEventListener('click', () => {
          checks().forEach(input => { input.checked = false; });
          updateSelection();
        });

        document.querySelectorAll('[data-density]').forEach(button => button.addEventListener('click', () => {
          const density = button.dataset.density;
          grid.classList.toggle('comfortable', density === 'comfortable');
          document.querySelectorAll('[data-density]').forEach(item => item.classList.toggle('active', item === button));
          localStorage.setItem('benchmarkDensity', density);
        }));

        const viewer = document.getElementById('viewer');
        document.querySelectorAll('.preview-button').forEach(button => button.addEventListener('click', () => {
          document.getElementById('viewerImage').src = button.dataset.full;
          document.getElementById('viewerName').textContent = button.dataset.name;
          viewer.showModal();
        }));
        document.getElementById('closeViewer').addEventListener('click', () => viewer.close());

        function applyPreset(name) {
          const personApply = document.getElementById('applyPerson');
          const person = document.getElementById('bulkPerson');
          const casesApply = document.getElementById('applyCases');
          const cases = document.getElementById('bulkCases');
          const face = document.getElementById('bulkFace');
          if (name === 'known') {
            personApply.checked = true; casesApply.checked = true; cases.value = 'known'; face.value = 'true'; person.focus();
          } else if (name === 'unknown') {
            personApply.checked = true; person.value = ''; casesApply.checked = true; cases.value = 'unknown'; face.value = 'true';
          } else if (name === 'no_face') {
            personApply.checked = true; person.value = ''; casesApply.checked = true; cases.value = 'no_face'; face.value = 'false';
          }
        }
        document.querySelectorAll('[data-preset]').forEach(button => button.addEventListener('click', () => applyPreset(button.dataset.preset)));
        document.getElementById('bulkPerson').addEventListener('input', () => { document.getElementById('applyPerson').checked = true; });
        document.getElementById('bulkCases').addEventListener('input', () => { document.getElementById('applyCases').checked = true; });

        async function submitBatch(verified, applyFields) {
          const chosen = selected();
          if (!chosen.length) return;
          const params = new URLSearchParams();
          chosen.forEach(input => params.append('source', input.value));
          params.set('verified', verified);
          if (applyFields) {
            if (document.getElementById('applyPerson').checked) {
              params.set('apply_expected_person','true');
              params.set('expected_person',document.getElementById('bulkPerson').value);
            }
            if (document.getElementById('applyCases').checked) {
              params.set('apply_case_types','true');
              params.set('case_types',document.getElementById('bulkCases').value);
            }
            const face = document.getElementById('bulkFace').value;
            if (face !== 'preserve') { params.set('apply_expected_face','true'); params.set('expected_face',face); }
            const nudity = document.getElementById('bulkNudity').value;
            if (nudity !== 'preserve') { params.set('apply_expected_nudity','true'); params.set('expected_nudity',nudity); }
          }
          showMessage(`Saving ${chosen.length} selected cases...`);
          try {
            const response = await fetch('/bulk-save', {method:'POST', headers:{'Content-Type':'application/x-www-form-urlencoded'}, body:params});
            const result = await response.json();
            if (!response.ok || !result.ok) throw new Error(result.error || 'Batch update failed');
            sessionStorage.setItem('benchmarkMessage', `${result.count} cases updated successfully.`);
            sessionStorage.setItem('benchmarkScroll', String(window.scrollY));
            window.location.reload();
          } catch (error) { showMessage(error.message, true); }
        }
        document.getElementById('verifySelected').addEventListener('click', () => submitBatch('true', false));
        document.getElementById('applySelected').addEventListener('click', () => submitBatch('true', true));
        document.getElementById('pendingSelected').addEventListener('click', () => submitBatch('false', false));

        search.value = localStorage.getItem('benchmarkSearch') || '';
        statusFilter.value = localStorage.getItem('benchmarkStatus') || 'all';
        typeFilter.value = localStorage.getItem('benchmarkType') || 'all';
        const density = localStorage.getItem('benchmarkDensity') || 'compact';
        document.querySelector(`[data-density="${density}"]`).click();
        applyFilters(); updateSelection();
        const savedMessage = sessionStorage.getItem('benchmarkMessage');
        const savedScroll = Number(sessionStorage.getItem('benchmarkScroll') || 0);
        if (savedMessage) { showMessage(savedMessage); sessionStorage.removeItem('benchmarkMessage'); }
        if (savedScroll) { requestAnimationFrame(() => window.scrollTo(0,savedScroll)); sessionStorage.removeItem('benchmarkScroll'); }
        """
        type_options = "".join(
            f'<option value="{case_type}">{case_type.replace("_", " ").title()}</option>'
            for case_type in sorted(evaluation_dataset.REQUIRED_CASE_TYPES)
        )
        page = f"""<!doctype html><html><head><meta charset="utf-8">
        <meta name="viewport" content="width=device-width,initial-scale=1">
        <title>Protected Face Benchmark</title><style>{styles}</style></head><body>
        <header class="workspace-header">
          <div class="title-row"><div><h1>Protected Face Benchmark</h1>
            <p class="subtitle">Review many images together, then confirm only labels you have visually checked.</p></div>
            <a class="primary-button" href="/gate">Run activation gate</a></div>
          <div class="metric-row">
            <span class="metric"><strong>{len(rows)}</strong> total</span>
            <span class="metric"><strong>{verified_count}</strong> verified</span>
            <span class="metric"><strong>{pending_count}</strong> pending</span>
            <span class="metric"><strong id="visibleCount">{len(rows)}</strong> visible</span>
            <span class="metric">Covered <strong>{len(validation.covered_types)}/{len(evaluation_dataset.REQUIRED_CASE_TYPES)}</strong></span>
            <span class="metric">Missing <strong>{html.escape(', '.join(missing) or 'none')}</strong></span>
            <span id="message" class="message">{html.escape(message)}</span>
          </div>
          <div class="toolbar">
            <input id="search" class="search" type="search" placeholder="Search person, filename, type, or notes">
            <select id="statusFilter" aria-label="Filter by verification status"><option value="all">All status</option><option value="pending">Pending only</option><option value="verified">Verified only</option></select>
            <select id="typeFilter" aria-label="Filter by case type"><option value="all">All case types</option>{type_options}</select>
            <button id="selectVisible" class="secondary-button" type="button">Select visible</button>
            <button id="clearSelection" class="secondary-button" type="button">Clear</button>
            <div class="density-switch" aria-label="Image density"><button class="active" type="button" data-density="compact">Compact</button><button type="button" data-density="comfortable">Large</button></div>
          </div>
          <section id="batchPanel" class="batch-panel" aria-label="Batch confirmation tools">
            <div class="batch-row"><strong><span id="selectedCount">0</span> images selected</strong>
              <div><button id="pendingSelected" class="secondary-button" type="button">Mark pending</button>
              <button id="verifySelected" class="primary-button" type="button">Confirm existing labels</button>
              <button id="applySelected" class="primary-button" type="button">Apply fields and confirm</button></div></div>
            <div class="batch-fields">
              <label class="batch-field"><input id="applyPerson" type="checkbox"><input id="bulkPerson" type="text" placeholder="Person (check to apply; blank clears)"></label>
              <label class="batch-field"><input id="applyCases" type="checkbox"><input id="bulkCases" type="text" placeholder="Case types, e.g. known|side_profile"></label>
              <label class="batch-field"><span>Face</span><select id="bulkFace"><option value="preserve">Preserve</option><option value="true">Expected</option><option value="false">No face</option></select></label>
              <label class="batch-field"><span>Nudity</span><select id="bulkNudity"><option value="preserve">Preserve</option><option value="safe">Safe</option><option value="possible">Possible</option><option value="unknown">Unknown</option></select></label>
            </div>
            <div class="quick-presets"><span class="subtitle">Quick setup:</span><button class="secondary-button" type="button" data-preset="known">Known person</button><button class="secondary-button" type="button" data-preset="unknown">Unknown person</button><button class="secondary-button" type="button" data-preset="no_face">No face</button></div>
          </section>
        </header>
        <main class="page-content">
          <details class="add-candidate"><summary>Add a local benchmark image</summary>
            <form class="add-form" method="post" action="/save"><input name="source" placeholder="/absolute/path/to/image.jpg"><input name="expected_person" placeholder="Expected person"><input name="case_types" placeholder="known|lookalike|side_profile|blurry_small|group|unknown|no_face|normal|nude|swimwear"><input type="hidden" name="expected_face" value="true"><input type="hidden" name="expected_nudity" value="unknown"><input type="hidden" name="notes" value="Added manually"><button class="primary-button">Add candidate</button></form>
          </details>
          {f'<details class="gate-output" open><summary>Latest activation result</summary><pre>{output}</pre></details>' if output else ''}
          <div id="caseGrid" class="grid">{''.join(cards)}</div>
        </main>
        <dialog id="viewer"><div class="viewer-bar"><strong id="viewerName" class="viewer-name"></strong><button id="closeViewer" class="secondary-button" type="button">Close</button></div><img id="viewerImage" class="viewer-image" alt="Full benchmark image"></dialog>
        <script>{script}</script></body></html>"""
        payload = page.encode("utf-8")
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, _format: str, *_args) -> None:
        return


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=identity_evaluation.DEFAULT_PROTECTED_SET)
    parser.add_argument("--baseline", type=Path, default=identity_evaluation.DEFAULT_PROTECTED_BASELINE)
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--open", action="store_true")
    parser.add_argument("--seed-only", action="store_true")
    args = parser.parse_args()
    dataset = args.dataset.expanduser().resolve(strict=False)
    if args.seed_only:
        count = seed_dataset(dataset)
        print(f"Protected benchmark candidates: {count}")
        print(f"Dataset: {dataset}")
        return 0
    url = f"http://127.0.0.1:{args.port}/"
    try:
        server = BenchmarkServer(("127.0.0.1", args.port), Handler)
    except OSError as exc:
        if exc.errno != errno.EADDRINUSE:
            raise
        existing_dashboard = False
        try:
            with urlopen(url, timeout=1.0) as response:
                preview = response.read(8_192)
                existing_dashboard = (
                    response.status == HTTPStatus.OK
                    and b"Protected Face Benchmark" in preview
                )
        except OSError:
            pass
        if not existing_dashboard:
            print(f"ERROR: port {args.port} is in use by another application.")
            return 2
        print(f"Protected benchmark dashboard is already running: {url}")
        if args.open:
            webbrowser.open(url)
        return 0
    try:
        count = seed_dataset(dataset)
        print(f"Protected benchmark candidates: {count}")
        print(f"Dataset: {dataset}")
        server.dataset = dataset
        server.baseline = args.baseline.expanduser().resolve(strict=False)
        print(f"Review dashboard: {url}")
        if args.open:
            threading.Timer(0.4, lambda: webbrowser.open(url)).start()
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Detect and recoverably remove generated contact/review sheets.

Old smart-album contact sheets were occasionally ingested as normal person
photos after their filenames were normalized.  This tool combines historical
operation-ledger provenance with the exact layouts used by both contact-sheet
generators.  Report-only is the default; ``--apply`` moves confirmed artifacts
to dated FaceFolders app trash so they remain recoverable.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

import analysis_index
import asset_processing
import operation_ledger
import pipeline_paths


IMAGE_EXTS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".tif",
    ".tiff", ".heic", ".heif",
}
HISTORICAL_NAME_MARKERS = (
    "contact_sheet",
    "contact sheet",
    "__page_",
    "review_sheet",
    "review sheet",
)
VALIDATION_WIDTH = 974
VALIDATION_HEIGHTS = frozenset({332, 598, 864, 1130, 1396, 1662})
LEGACY_WIDTHS = frozenset({204, 384, 564, 744, 924})
REPORT_COLUMNS = (
    "person", "relative_path", "path", "sha256", "byte_size", "width",
    "height", "detection_lane", "score", "reasons", "status",
    "destination",
)


@dataclass(frozen=True)
class GeneratedArtifactMatch:
    path: str
    person: str
    relative_path: str
    sha256: str
    byte_size: int
    width: int
    height: int
    detection_lane: str
    score: float
    reasons: str
    status: str = "confirmed"
    destination: str = ""


def obvious_generated_name(path: Path) -> bool:
    text = path.as_posix().casefold()
    return any(marker in text for marker in HISTORICAL_NAME_MARKERS)


def layout_kind(width: int, height: int) -> str | None:
    if width == VALIDATION_WIDTH and height in VALIDATION_HEIGHTS:
        return "validation_layout"
    if (
        width in LEGACY_WIDTHS
        and height >= 280
        and (height - 66) % 214 == 0
        and 1 <= (height - 66) // 214 <= 8
    ):
        return "legacy_layout"
    return None


def historical_generated_hashes(sorted_root: Path) -> set[str]:
    hashes: set[str] = set()
    root = sorted_root / "_source_review" / operation_ledger.LEDGER_DIR_NAME
    if not root.exists():
        return hashes
    for ledger in root.glob("*_moves.jsonl"):
        try:
            lines = ledger.open("r", encoding="utf-8", errors="ignore")
        except OSError:
            continue
        with lines:
            for line in lines:
                try:
                    event = json.loads(line)
                except (json.JSONDecodeError, TypeError):
                    continue
                source_text = " ".join(
                    str(event.get(key, ""))
                    for key in (
                        "source_path",
                        "source_relative_to_sorted",
                        "source_relative_to_people",
                    )
                ).casefold()
                if not any(marker in source_text for marker in HISTORICAL_NAME_MARKERS):
                    continue
                digest = str((event.get("source") or {}).get("sha256") or "").casefold()
                if digest:
                    hashes.add(digest)
    return hashes


def _layout_metrics(bgr: np.ndarray, kind: str) -> dict[str, float]:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    title = gray[:54, :]
    metrics = {
        "title_light": float(np.mean(title >= 238)),
        "title_dark": float(np.mean(title <= 80)),
        "canvas_light": float(np.mean(gray >= 245)),
    }
    if kind == "validation_layout":
        badge = gray[58:80, 16:54]
        metrics["badge_light"] = float(np.mean(badge >= 230))
        metrics["badge_dark"] = float(np.mean(badge <= 80))
    return metrics


def visually_matches_layout(bgr: np.ndarray, kind: str) -> tuple[bool, float, str]:
    metrics = _layout_metrics(bgr, kind)
    if kind == "validation_layout":
        checks = {
            "light title strip": metrics["title_light"] >= 0.94,
            "title text": metrics["title_dark"] >= 0.005,
            "light sheet canvas": metrics["canvas_light"] >= 0.20,
            "number badge fill": metrics["badge_light"] >= 0.82,
            "number badge border/text": metrics["badge_dark"] >= 0.04,
        }
    else:
        checks = {
            "light title strip": metrics["title_light"] >= 0.92,
            "title text": metrics["title_dark"] >= 0.002,
            "light sheet canvas": metrics["canvas_light"] >= 0.20,
        }
    score = sum(checks.values()) / len(checks)
    detail = "; ".join(
        [name for name, passed in checks.items() if passed]
        + [f"{key}={value:.3f}" for key, value in metrics.items()]
    )
    return all(checks.values()), score, detail


def canonical_person_images(people_dir: Path) -> Iterable[Path]:
    if not people_dir.exists():
        return
    for person_dir in sorted(people_dir.iterdir(), key=lambda p: p.name.casefold()):
        if not person_dir.is_dir() or person_dir.name.startswith((".", "_")):
            continue
        photos = person_dir / "photos"
        if not photos.exists():
            continue
        for path in photos.rglob("*"):
            if path.is_file() and path.suffix.casefold() in IMAGE_EXTS:
                yield path


def _match_for_path(
    path: Path,
    *,
    people_dir: Path,
    fingerprint: analysis_index.AssetFingerprint | None,
    known_hashes: set[str],
) -> GeneratedArtifactMatch | None:
    decoded = None
    digest = fingerprint.sha256 if fingerprint is not None else ""
    width = fingerprint.width if fingerprint is not None else 0
    height = fingerprint.height if fingerprint is not None else 0

    if digest and digest.casefold() in known_hashes:
        lane = "historical_ledger_hash"
        score = 1.0
        reasons = "SHA-256 matches a historical generated contact/review sheet"
    elif obvious_generated_name(path):
        lane = "generated_filename"
        score = 1.0
        reasons = "path contains a generated contact/review-sheet marker"
    else:
        kind = layout_kind(width, height)
        if kind is None:
            return None
        decoded = asset_processing.decode_image(path)
        if decoded is None:
            return None
        matched, score, reasons = visually_matches_layout(decoded.bgr, kind)
        if not matched:
            return None
        digest = digest or decoded.sha256
        width = decoded.width
        height = decoded.height
        lane = kind

    if not digest or not width or not height:
        decoded = decoded or asset_processing.decode_image(path)
        if decoded is None:
            return None
        digest = digest or decoded.sha256
        width = width or decoded.width
        height = height or decoded.height
    try:
        relative = path.resolve().relative_to(people_dir.resolve())
    except ValueError:
        relative = Path(path.name)
    person = relative.parts[0] if relative.parts else ""
    try:
        byte_size = path.stat().st_size
    except OSError:
        return None
    return GeneratedArtifactMatch(
        path=str(path),
        person=person,
        relative_path=relative.as_posix(),
        sha256=digest,
        byte_size=int(byte_size),
        width=int(width),
        height=int(height),
        detection_lane=lane,
        score=float(score),
        reasons=reasons,
    )


def detect_paths(
    paths: Iterable[Path],
    *,
    people_dir: Path,
    index_path: Path,
    known_hashes: set[str] | None = None,
    progress_every: int = 1000,
) -> list[GeneratedArtifactMatch]:
    matches: list[GeneratedArtifactMatch] = []
    hashes = known_hashes or set()
    ordered = list(paths)
    with analysis_index.AnalysisIndex(index_path) as index:
        for number, path in enumerate(ordered, start=1):
            fingerprint = index.fingerprint(path)
            needs_decode = obvious_generated_name(path)
            if fingerprint is None and hashes and not needs_decode:
                try:
                    needs_decode = operation_ledger.sha256_file(path).casefold() in hashes
                except OSError:
                    needs_decode = False
            if fingerprint is None and not needs_decode:
                try:
                    from PIL import Image

                    with Image.open(path) as image:
                        needs_decode = layout_kind(*image.size) is not None
                except Exception:
                    needs_decode = False
            if fingerprint is None and needs_decode:
                decoded = asset_processing.load_decoded_asset(path)
                if decoded is not None:
                    fingerprint = analysis_index.AssetFingerprint(
                        sha256=decoded.sha256,
                        pixel_sha256=decoded.pixel_sha256,
                        phash=asset_processing.phash_to_int(decoded.phash_bits),
                        width=decoded.width,
                        height=decoded.height,
                    )
                    index.upsert_fingerprint(path, fingerprint)
            match = _match_for_path(
                path,
                people_dir=people_dir,
                fingerprint=fingerprint,
                known_hashes=hashes,
            )
            if match is not None:
                matches.append(match)
            if progress_every and (number % progress_every == 0 or number == len(ordered)):
                print(
                    f"Generated-sheet audit: checked {number:,}/{len(ordered):,}; "
                    f"confirmed {len(matches):,}.",
                    flush=True,
                )
        index.commit()
    return matches


def partition_intake_artifacts(
    paths: Iterable[Path],
    *,
    index_path: Path,
) -> tuple[list[Path], list[GeneratedArtifactMatch]]:
    """Split high-confidence generated sheets from an intake batch."""
    paths = list(paths)
    matches = detect_paths(
        paths,
        people_dir=Path.cwd(),
        index_path=index_path,
        progress_every=0,
    )
    matched = {str(Path(row.path)) for row in matches}
    return [path for path in paths if str(path) not in matched], matches


def unique_destination(path: Path) -> Path:
    if not path.exists():
        return path
    suffix = "".join(path.suffixes)
    stem = path.name[:-len(suffix)] if suffix else path.name
    counter = 2
    while True:
        candidate = path.with_name(f"{stem}__artifact{counter}{suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def apply_matches(
    matches: list[GeneratedArtifactMatch],
    *,
    people_dir: Path,
    sorted_root: Path,
    day: str | None = None,
) -> list[GeneratedArtifactMatch]:
    day = day or time.strftime("%Y-%m-%d")
    trash_root = people_dir / ".photo_app_trash" / day
    applied: list[GeneratedArtifactMatch] = []
    run_id = operation_ledger.current_run_id(
        f"generated_contact_sheet_cleanup_{time.strftime('%Y%m%d_%H%M%S')}"
    )
    for number, row in enumerate(matches, start=1):
        source = Path(row.path)
        if not source.exists():
            applied.append(GeneratedArtifactMatch(**{
                **asdict(row),
                "status": "source_missing",
            }))
            continue
        destination = unique_destination(trash_root / row.relative_path)
        try:
            operation_ledger.move_path(
                source,
                destination,
                sorted_root=sorted_root,
                operation="generated_artifacts.remove_contact_sheet",
                reason="remove generated smart-album contact/review sheet from canonical person photos",
                run_id=run_id,
                extra={
                    "detection_lane": row.detection_lane,
                    "score": row.score,
                    "width": row.width,
                    "height": row.height,
                },
                mirror_sqlite=False,
            )
            status = "moved_to_recoverable_app_trash"
        except Exception as exc:  # noqa: BLE001
            status = f"failed: {exc}"
        applied.append(GeneratedArtifactMatch(**{
            **asdict(row),
            "status": status,
            "destination": str(destination),
        }))
        if number % 250 == 0 or number == len(matches):
            moved = sum(item.status == "moved_to_recoverable_app_trash" for item in applied)
            print(
                f"Generated-sheet cleanup: processed {number:,}/{len(matches):,}; "
                f"moved {moved:,}.",
                flush=True,
            )
    try:
        if sorted_root.expanduser().resolve() == pipeline_paths.SORTED_ROOT.expanduser().resolve():
            ledger = operation_ledger.ledger_path(sorted_root, run_id)
            with analysis_index.AnalysisIndex(pipeline_paths.ANALYSIS_INDEX) as index:
                with ledger.open("r", encoding="utf-8") as handle:
                    for line in handle:
                        try:
                            index.record_operation(json.loads(line))
                        except (json.JSONDecodeError, TypeError):
                            continue
                index.commit()
    except Exception as exc:  # noqa: BLE001
        print(
            "WARNING: JSON operation ledger is complete, but its SQLite mirror "
            f"could not be refreshed: {exc}",
            flush=True,
        )
    return applied


def archive_intake_matches(
    matches: list[GeneratedArtifactMatch],
    *,
    input_dir: Path,
    sorted_root: Path,
) -> int:
    archive_root = sorted_root / "_source_review" / "ready_to_delete" / "generated_contact_sheets"
    moved = 0
    for row in matches:
        source = Path(row.path)
        if not source.exists():
            continue
        try:
            relative = source.resolve().relative_to(input_dir.resolve())
        except ValueError:
            relative = Path(source.name)
        destination = unique_destination(archive_root / relative)
        operation_ledger.move_path(
            source,
            destination,
            sorted_root=sorted_root,
            operation="sort_photos.archive_generated_contact_sheet",
            reason="divert generated contact/review sheet before face sorting",
            extra={
                "detection_lane": row.detection_lane,
                "score": row.score,
                "width": row.width,
                "height": row.height,
            },
        )
        moved += 1
    return moved


def write_report(
    rows: list[GeneratedArtifactMatch],
    *,
    report_dir: Path,
    mode: str,
) -> tuple[Path, Path]:
    report_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    csv_path = report_dir / f"generated_contact_sheets_{mode}_{stamp}.csv"
    json_path = report_dir / f"generated_contact_sheets_{mode}_{stamp}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPORT_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))
    counts = Counter(row.detection_lane for row in rows)
    payload = {
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "mode": mode,
        "confirmed_count": len(rows),
        "total_bytes": sum(row.byte_size for row in rows),
        "people_count": len({row.person for row in rows}),
        "detection_lanes": dict(sorted(counts.items())),
        "csv": str(csv_path),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return csv_path, json_path


def human_size(size: int) -> str:
    value = float(size)
    for unit in ("B", "KB", "MB", "GB"):
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-dir", type=Path, default=pipeline_paths.PEOPLE_ROOT)
    parser.add_argument("--index", type=Path, default=pipeline_paths.ANALYSIS_INDEX)
    parser.add_argument("--report-dir", type=Path, default=None)
    parser.add_argument("--apply", action="store_true", help="Move confirmed sheets to recoverable app trash")
    args = parser.parse_args()

    people_dir = args.people_dir.expanduser().resolve()
    sorted_root = people_dir.parent
    report_dir = args.report_dir or (
        sorted_root / "_source_review" / "generated_contact_sheet_audits"
    )
    paths = list(canonical_person_images(people_dir))
    known_hashes = historical_generated_hashes(sorted_root)
    print(f"Scanning {len(paths):,} canonical person images.")
    print(f"Historical generated hashes: {len(known_hashes):,}.")
    matches = detect_paths(
        paths,
        people_dir=people_dir,
        index_path=args.index,
        known_hashes=known_hashes,
    )
    rows = apply_matches(
        matches,
        people_dir=people_dir,
        sorted_root=sorted_root,
    ) if args.apply else matches
    mode = "apply" if args.apply else "audit"
    csv_path, json_path = write_report(rows, report_dir=report_dir, mode=mode)
    moved = sum(row.status == "moved_to_recoverable_app_trash" for row in rows)
    failed = sum(row.status.startswith("failed") for row in rows)
    print()
    print(f"Confirmed generated sheets: {len(matches):,} ({human_size(sum(row.byte_size for row in matches))})")
    print(f"Affected people: {len({row.person for row in matches}):,}")
    if args.apply:
        print(f"Moved recoverably: {moved:,}")
        print(f"Failed: {failed:,}")
    else:
        print("Report-only: no files were moved.")
    print(f"CSV report: {csv_path}")
    print(f"Summary: {json_path}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

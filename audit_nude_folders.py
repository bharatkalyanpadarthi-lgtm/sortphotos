#!/usr/bin/env python3
"""Audit existing nude or uncertain-nudity folders with the shared policy.

Confirmed explicit images stay in place. Likely-safe images return to normal
photos, while borderline explicit images move to the person's recoverable
review/uncertain_nudity folder. This keeps every original inside the protected
person tree. All moves use the operation ledger and the default mode is
report-only.

With ``--source uncertain``, only newly confirmed nude images move from
``review/uncertain_nudity`` to ``photos/nude``. Safe and still-ambiguous files
remain in review until the user chooses a separate manual disposition.

``--source uncertain --move-all --apply`` is the explicit manual override: it
records every current review image as user-confirmed and moves all of them.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import pipeline_paths

import cv2

import analysis_index
import nudity_confirmations
import operation_ledger
import sort_photos


DEFAULT_SORTED = pipeline_paths.SORTED_ROOT
DEFAULT_PEOPLE = DEFAULT_SORTED / "photos_by_person"
AUDIT_DIR = DEFAULT_SORTED / "_source_review" / "nudity_audits"
CACHE_PATH = AUDIT_DIR / "nudity_audit_cache.json"
POLICY_VERSION = "7-spatial-conflicts"
DETECTION_CACHE_VERSION = 1


def iter_nude_images(people_root: Path, person_filter: str = "") -> list[Path]:
    images: list[Path] = []
    for person_dir in sorted(people_root.iterdir(), key=lambda path: path.name.casefold()):
        if not person_dir.is_dir() or person_dir.name.startswith((".", "_")):
            continue
        if person_filter and person_dir.name.casefold() != person_filter.casefold():
            continue
        nude_root = person_dir / sort_photos.PERSON_PHOTOS_DIR / sort_photos.PERSON_NUDE_DIR
        if not nude_root.exists():
            continue
        images.extend(
            path for path in nude_root.rglob("*")
            if path.is_file() and path.suffix.lower() in sort_photos.IMAGE_EXTS
        )
    return sorted(images, key=lambda path: str(path).casefold())


def iter_uncertain_images(people_root: Path, person_filter: str = "") -> list[Path]:
    images: list[Path] = []
    for person_dir in sorted(people_root.iterdir(), key=lambda path: path.name.casefold()):
        if not person_dir.is_dir() or person_dir.name.startswith((".", "_")):
            continue
        if person_filter and person_dir.name.casefold() != person_filter.casefold():
            continue
        review_root = person_dir / sort_photos.PERSON_REVIEW_DIR / "uncertain_nudity"
        if not review_root.exists():
            continue
        images.extend(
            path for path in review_root.rglob("*")
            if path.is_file() and path.suffix.lower() in sort_photos.IMAGE_EXTS
        )
    return sorted(images, key=lambda path: str(path).casefold())


def person_and_nude_root(path: Path, people_root: Path) -> tuple[Path, Path]:
    relative = path.relative_to(people_root)
    person_dir = people_root / relative.parts[0]
    return person_dir, person_dir / sort_photos.PERSON_PHOTOS_DIR / sort_photos.PERSON_NUDE_DIR


def person_and_uncertain_root(path: Path, people_root: Path) -> tuple[Path, Path]:
    relative = path.relative_to(people_root)
    person_dir = people_root / relative.parts[0]
    return person_dir, person_dir / sort_photos.PERSON_REVIEW_DIR / "uncertain_nudity"


def destination_for(path: Path,
                    people_root: Path,
                    decision: str,
                    source_kind: str = "nude") -> Path | None:
    if source_kind == "uncertain":
        if decision != "confirmed_nude":
            return None
        person_dir, uncertain_root = person_and_uncertain_root(path, people_root)
        relative = path.relative_to(uncertain_root)
        return sort_photos.unique_path(
            person_dir / sort_photos.PERSON_PHOTOS_DIR /
            sort_photos.PERSON_NUDE_DIR / relative
        )
    if decision == "confirmed_nude":
        return None
    person_dir, nude_root = person_and_nude_root(path, people_root)
    relative = path.relative_to(nude_root)
    if decision == "likely_safe":
        root = person_dir / sort_photos.PERSON_PHOTOS_DIR
    elif decision == "needs_review":
        root = person_dir / sort_photos.PERSON_REVIEW_DIR / "uncertain_nudity"
    else:
        return None
    return sort_photos.unique_path(root / relative)


def decode_for_detector(path: Path):
    image = sort_photos.imread_unicode(path)
    if image is None or image.size == 0:
        raise ValueError("image decoder returned no pixels")
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    if image.ndim == 3 and image.shape[2] == 3:
        return image
    raise ValueError(f"unsupported image shape: {image.shape}")


def detect_batch_safely(detector, paths: list[Path]) -> list[list[dict] | Exception]:
    images = []
    image_indexes = []
    results: list[list[dict] | Exception | None] = [None] * len(paths)
    for index, path in enumerate(paths):
        try:
            images.append(decode_for_detector(path))
            image_indexes.append(index)
        except Exception as error:  # noqa: BLE001
            results[index] = error
    if images:
        try:
            detected = detector.detect_batch(images, batch_size=len(images))
        except Exception:  # noqa: BLE001
            detected = []
            for image in images:
                try:
                    detected.append(detector.detect(image))
                except Exception as error:  # noqa: BLE001
                    detected.append(error)
        for index, value in zip(image_indexes, detected):
            results[index] = value
    return [value if value is not None else RuntimeError("missing detector result") for value in results]


def load_cache(path: Path) -> dict[str, dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        # Cache stores raw detector output, so policy changes can safely reuse
        # it and immediately reclassify without rerunning the model.
        if isinstance(payload.get("entries"), dict):
            return dict(payload.get("entries", {}))
    except (OSError, ValueError, TypeError):
        pass
    return {}


def save_cache(path: Path, entries: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps({
            "detector_cache_version": DETECTION_CACHE_VERSION,
            "policy_version": POLICY_VERSION,
            "entries": entries,
        }, sort_keys=True),
        encoding="utf-8",
    )
    temporary.replace(path)


def signature(path: Path) -> str:
    metadata = path.stat()
    return f"{metadata.st_size}:{metadata.st_mtime_ns}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--people-root", type=Path, default=DEFAULT_PEOPLE)
    parser.add_argument("--person", default="", help="Audit one exact person folder name.")
    parser.add_argument(
        "--source",
        choices=("nude", "uncertain"),
        default="nude",
        help="Audit photos/nude or review/uncertain_nudity. Default: nude.",
    )
    parser.add_argument("--apply", action="store_true", help="Apply ledgered safe/review moves.")
    parser.add_argument(
        "--move-all",
        action="store_true",
        help=(
            "Treat every image in review/uncertain_nudity as user-confirmed and move it "
            "to photos/nude. Requires --source uncertain; use --apply to commit."
        ),
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--no-cache", action="store_true")
    args = parser.parse_args()

    if args.move_all and args.source != "uncertain":
        parser.error("--move-all requires --source uncertain")

    people_root = args.people_root.expanduser().resolve()
    if not people_root.exists():
        print(f"People root does not exist: {people_root}")
        return 2
    paths = (
        iter_uncertain_images(people_root, args.person)
        if args.source == "uncertain"
        else iter_nude_images(people_root, args.person)
    )
    if args.limit > 0:
        paths = paths[:args.limit]

    from nudenet import NudeDetector

    cache = {} if args.no_cache else load_cache(CACHE_PATH)
    report_id = time.strftime("%Y%m%d_%H%M%S")
    AUDIT_DIR.mkdir(parents=True, exist_ok=True)
    report_stem = "uncertain_nudity_audit" if args.source == "uncertain" else "nude_folder_audit"
    report_path = AUDIT_DIR / f"{report_stem}_{report_id}.jsonl"
    detector = None
    counts: Counter[str] = Counter()
    asset_index = analysis_index.AnalysisIndex(sort_photos.analysis_index_file())

    print("Person Uncertain-Nudity Review" if args.source == "uncertain" else "Person Nude-Folder Audit")
    print("=" * 60)
    print(f"People root:       {people_root}")
    print(f"Person filter:     {args.person or 'all'}")
    print(f"Source:            {args.source}")
    print(f"Images:            {len(paths)}")
    print(f"Mode:              {'APPLY' if args.apply else 'REPORT ONLY'}")
    if args.move_all:
        print("Decision:          USER-CONFIRMED ALL")
    print(f"Policy:            {POLICY_VERSION}")

    try:
        report = report_path.open("w", encoding="utf-8")
        for start in range(0, len(paths), max(1, args.batch_size)):
            batch = paths[start:start + max(1, args.batch_size)]
            batch_results: list[list[dict] | Exception | None] = [None] * len(batch)
            uncached_paths: list[Path] = []
            uncached_indexes: list[int] = []
            if args.move_all:
                batch_results = [[] for _path in batch]
            else:
                for index, path in enumerate(batch):
                    cached = cache.get(str(path))
                    current_signature = signature(path)
                    if cached and cached.get("signature") == current_signature:
                        batch_results[index] = list(cached.get("detections", []))
                        counts["cache_hits"] += 1
                    else:
                        current = asset_index.cached_nudity(
                            path,
                            sort_photos.NUDITY_ANALYSIS_VERSION,
                        )
                        detections = current[1] if current is not None else None
                        if detections is None:
                            fingerprint = asset_index.fingerprint(path)
                            sha256 = (
                                fingerprint.sha256
                                if fingerprint is not None
                                else asset_index.file_sha256(path)
                            )
                            if sha256:
                                detections = asset_index.cached_nudity_detections_by_sha256(sha256)
                        if detections is None:
                            uncached_paths.append(path)
                            uncached_indexes.append(index)
                        else:
                            batch_results[index] = detections
                            counts["cache_hits"] += 1
            if uncached_paths and not args.move_all:
                if detector is None:
                    detector = NudeDetector()
                fresh = detect_batch_safely(detector, uncached_paths)
                for index, path, value in zip(uncached_indexes, uncached_paths, fresh):
                    batch_results[index] = value
                    counts["model_scans"] += 1
                    if not isinstance(value, Exception):
                        cache[str(path)] = {
                            "signature": signature(path),
                            "detections": value,
                        }

            for path, detections in zip(batch, batch_results):
                record: dict = {"source": str(path), "applied": bool(args.apply)}
                if args.move_all:
                    decision = "confirmed_nude"
                    best_class = "USER_CONFIRMED"
                    best_score = 1.0
                    reason = "user_confirmed_all_uncertain"
                    record.update({
                        "decision": decision,
                        "best_class": best_class,
                        "best_score": best_score,
                        "reason": reason,
                        "detections": [],
                    })
                    counts[decision] += 1
                    destination = destination_for(path, people_root, decision, args.source)
                    if destination is not None:
                        record["destination"] = str(destination)
                        if args.apply:
                            person = path.relative_to(people_root).parts[0]
                            fingerprint = asset_index.fingerprint(path)
                            file_hash = (
                                fingerprint.sha256
                                if fingerprint is not None
                                else nudity_confirmations.sha256_file(path)
                            )
                            nudity_confirmations.record_confirmation(
                                path,
                                destination,
                                person=person,
                                reason=reason,
                                sha256=file_hash,
                            )
                            moved = operation_ledger.move_path(
                                path,
                                destination,
                                sorted_root=DEFAULT_SORTED,
                                operation="audit_nude_folders.confirm_all_uncertain",
                                reason="user confirmed all uncertain-nudity review images",
                                source_sha256=file_hash,
                                mirror_sqlite=False,
                                extra={
                                    "manual_confirmation": True,
                                    "policy_version": POLICY_VERSION,
                                },
                            )
                            asset_index.record_nudity(
                                moved,
                                model="manual-confirmation-v1",
                                status="possible",
                                detections=[],
                            )
                            if fingerprint is not None:
                                asset_index.upsert_fingerprint(moved, fingerprint)
                            record["destination"] = str(moved)
                            counts["moved"] += 1
                elif isinstance(detections, Exception) or detections is None:
                    record.update({"decision": "error", "reason": str(detections)})
                    counts["error"] += 1
                else:
                    fingerprint = asset_index.fingerprint(path)
                    manually_confirmed = nudity_confirmations.is_confirmed(
                        path,
                        sha256=fingerprint.sha256 if fingerprint is not None else None,
                    )
                    if manually_confirmed:
                        decision, best_class, best_score, reason = (
                            "confirmed_nude", "USER_CONFIRMED", 1.0,
                            "durable_user_confirmation",
                        )
                    else:
                        decision, best_class, best_score, reason = sort_photos.nudity_decision(detections)
                    status = {
                        "confirmed_nude": "possible",
                        "needs_review": "uncertain",
                        "likely_safe": "safe",
                    }[decision]
                    asset_index.record_nudity(
                        path,
                        model=(
                            "manual-confirmation-v1"
                            if manually_confirmed
                            else sort_photos.NUDITY_ANALYSIS_VERSION
                        ),
                        status=status,
                        detections=[] if manually_confirmed else detections,
                    )
                    record.update({
                        "decision": decision,
                        "best_class": best_class,
                        "best_score": round(best_score, 6),
                        "reason": reason,
                        "detections": detections,
                    })
                    counts[decision] += 1
                    destination = destination_for(path, people_root, decision, args.source)
                    if destination is not None:
                        record["destination"] = str(destination)
                        if args.apply:
                            moved = operation_ledger.move_path(
                                path,
                                destination,
                                sorted_root=DEFAULT_SORTED,
                                operation=(
                                    "audit_nude_folders.confirm_uncertain_nude"
                                    if args.source == "uncertain"
                                    else "audit_nude_folders.reclassify"
                                ),
                                reason=f"nudity policy {POLICY_VERSION}: {decision} ({reason})",
                                extra={
                                    "best_class": best_class,
                                    "best_score": round(best_score, 6),
                                    "policy_version": POLICY_VERSION,
                                },
                            )
                            record["destination"] = str(moved)
                report.write(json.dumps(record, sort_keys=True) + "\n")
                report.flush()

            asset_index.commit()
            completed = min(start + len(batch), len(paths))
            if (
                not args.no_cache
                and (completed == len(paths) or completed % 500 < len(batch))
            ):
                save_cache(CACHE_PATH, cache)
            if completed == len(paths) or completed % 100 == 0:
                print(
                    f"[{completed}/{len(paths)}] confirmed={counts['confirmed_nude']} "
                    f"safe={counts['likely_safe']} review={counts['needs_review']} "
                    f"errors={counts['error']} cache={counts['cache_hits']}",
                    flush=True,
                )
    finally:
        asset_index.close()
        try:
            report.close()
        except (NameError, OSError):
            pass

    print("\nAudit Summary")
    print("=" * 60)
    for key in (
        "confirmed_nude", "likely_safe", "needs_review", "error",
        "moved", "cache_hits", "model_scans",
    ):
        print(f"{key:20s} {counts[key]}")
    print(f"Report: {report_path}")
    if args.apply and args.move_all:
        mirrored, malformed = operation_ledger.mirror_run_to_sqlite(
            sorted_root=DEFAULT_SORTED,
            run_id=operation_ledger.current_run_id(),
        )
        print(f"Ledger mirrored:      {mirrored} event(s)")
        print(f"Malformed events:     {malformed}")
    return 1 if counts["error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

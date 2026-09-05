#!/usr/bin/env python3
"""Recheck false-negative no-face images and recover safe person matches.

This command is deliberately stricter for identity assignment than the normal
cluster workflow. It can recover a single image only when its face embedding is
an unambiguous match to a well-supported existing identity. Detected faces that
do not meet that bar move to unknown_identity for review; genuine detector
misses remain in no_usable_face.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pipeline_paths

import numpy as np

import analysis_index
import appearance_profiles
import identity_profiles
import identity_hard_negatives
import operation_ledger
import sort_photos
import source_batch_consensus
import secondary_identity_matcher


DEFAULT_SORTED = pipeline_paths.SORTED_ROOT
DEFAULT_PEOPLE = DEFAULT_SORTED / "photos_by_person"
DEFAULT_REVIEW = DEFAULT_SORTED / "_source_review" / "unassigned_intake"
DEFAULT_INPUT = DEFAULT_REVIEW / "no_usable_face"
REPORT_DIR = DEFAULT_SORTED / "_source_review" / "recovery_reports"

MATCH_MAX_DISTANCE = 0.32
MATCH_MIN_MARGIN = 0.08
MATCH_MIN_QUALITY = 0.45
MATCH_EXCEPTIONAL_DISTANCE = 0.18
MATCH_EXCEPTIONAL_MARGIN = 0.20
MIN_IDENTITY_REFERENCE_FACES = 3
MIN_INTERNAL_FREE_BYTES = 20 * 1024 * 1024 * 1024


class RecoveryPaused(RuntimeError):
    """Raised when a resumable safety condition requires the run to stop."""


def load_existing_hashes(people_root: Path) -> dict[str, set[str]]:
    hashes: dict[str, set[str]] = defaultdict(set)
    cache_path = sort_photos.FINGERPRINT_CACHE_FILE
    try:
        payload = json.loads(cache_path.read_text(encoding="utf-8"))
        entries = payload.get("entries", {})
    except (OSError, ValueError, TypeError):
        entries = {}
    for path_text, metadata in entries.items():
        digest = str(metadata.get("sha256", "")).strip()
        if not digest:
            continue
        try:
            relative = Path(path_text).resolve().relative_to(people_root.resolve())
        except (OSError, ValueError):
            continue
        if len(relative.parts) >= 2:
            hashes[relative.parts[0]].add(digest)
    return hashes


def rank_face_candidates(embedding: np.ndarray,
               identity_names: list[str],
               identity_matrix: np.ndarray,
               identity_db: sort_photos.IdentityDB,
               pose_label: str = "unknown",
               lighting_label: str = "unknown",
               capture_timestamp: float = 0.0,
               hard_negatives: dict[str, list[np.ndarray]] | None = None,
               ) -> list[identity_profiles.IdentityCandidate]:
    del identity_matrix  # Retained in the signature for older callers.
    identity_db = sort_photos.normalize_identity_db(identity_db)
    eligible = {
        name: identity_db.identities[name]
        for name in identity_names
        if name in identity_db.identities
        and identity_db.source_counts.get(name, 0) >= MIN_IDENTITY_REFERENCE_FACES
    }
    if not eligible:
        return []
    prototypes = {
        name: identity_db.prototypes.get(name, [centroid])
        for name, centroid in eligible.items()
    }
    return identity_profiles.rank_candidates(
        embedding,
        eligible,
        prototypes,
        pose_label=pose_label,
        pose_prototypes=identity_db.pose_prototypes,
        lighting_label=lighting_label,
        capture_timestamp=capture_timestamp,
        appearance_prototypes=identity_db.appearance_prototypes,
        appearance_era_cutoffs=identity_db.appearance_era_cutoffs,
        hard_negatives=hard_negatives,
    )


def match_face(embedding: np.ndarray,
               identity_names: list[str],
               identity_matrix: np.ndarray,
               identity_db: sort_photos.IdentityDB,
               quality: float = 1.0,
               pose_label: str = "unknown",
               lighting_label: str = "unknown",
               capture_timestamp: float = 0.0,
               hard_negatives: dict[str, list[np.ndarray]] | None = None,
               ) -> tuple[str, float, float] | None:
    candidates = rank_face_candidates(
        embedding,
        identity_names,
        identity_matrix,
        identity_db,
        pose_label=pose_label,
        lighting_label=lighting_label,
        capture_timestamp=capture_timestamp,
        hard_negatives=hard_negatives,
    )
    if not candidates:
        return None
    best = candidates[0]
    distance = float(best.distance)
    margin = identity_profiles.candidate_margin(candidates)
    threshold = min(
        MATCH_MAX_DISTANCE,
        float(identity_db.match_thresholds.get(best.name, MATCH_MAX_DISTANCE)),
    )
    quality_ok = (
        quality >= MATCH_MIN_QUALITY
        or (distance <= MATCH_EXCEPTIONAL_DISTANCE and margin >= MATCH_EXCEPTIONAL_MARGIN)
    )
    if distance > threshold or margin < MATCH_MIN_MARGIN or not quality_ok:
        return None
    return best.name, distance, margin


def build_source_batch_consensus(
    files: list[Path],
    input_root: Path,
    identity_names: list[str],
    identity_matrix: np.ndarray,
    identity_db: sort_photos.IdentityDB,
    hard_negatives: dict[str, list[np.ndarray]],
    *,
    workers: int,
    primary_det_size: int,
    fallback_det_size: int,
    stats: Counter[str],
) -> dict[tuple[str, int], source_batch_consensus.ConsensusDecision]:
    evidence: list[source_batch_consensus.BatchEvidence] = []
    with analysis_index.AnalysisIndex(sort_photos.analysis_index_file()) as index:
        contexts = {
            str(path): (
                source_batch_consensus.direct_batch_key(path, input_root)
                or index.source_batch_context(path)
            )
            for path in files
        }
    results = iter_detection_results(
        files,
        workers,
        primary_det_size,
        fallback_det_size,
        index_path=sort_photos.analysis_index_file(),
        cache_stats=stats,
    )
    for _task_index, source, _status, faces, _error in results:
        batch_key = contexts.get(str(source), "")
        if not batch_key:
            continue
        for face in faces:
            lighting, captured_at = appearance_profiles.query_attributes(
                face.crop_jpeg, source
            )
            candidates = rank_face_candidates(
                face.embedding,
                identity_names,
                identity_matrix,
                identity_db,
                pose_label=face.pose_label,
                lighting_label=lighting,
                capture_timestamp=captured_at,
                hard_negatives=hard_negatives,
            )
            if not candidates:
                continue
            best = candidates[0]
            threshold = min(
                MATCH_MAX_DISTANCE,
                float(identity_db.match_thresholds.get(best.name, MATCH_MAX_DISTANCE)),
            )
            evidence.append(source_batch_consensus.BatchEvidence(
                source=str(source),
                face_index=int(face.face_index),
                batch_key=batch_key,
                embedding=np.asarray(face.embedding, dtype=np.float32),
                candidate_name=best.name,
                distance=float(best.distance),
                margin=identity_profiles.candidate_margin(candidates),
                quality=float(face.quality),
                threshold=threshold,
            ))
    return source_batch_consensus.consensus_decisions(evidence)


def borderline_secondary_match(
    face: sort_photos.CachedFace,
    identity_names: list[str],
    identity_matrix: np.ndarray,
    identity_db: sort_photos.IdentityDB,
    hard_negatives: dict[str, list[np.ndarray]],
    matcher: secondary_identity_matcher.SecondaryMatcher | None,
) -> tuple[str, float, float] | None:
    if matcher is None or not face.crop_jpeg or face.quality < 0.32:
        return None
    candidates = rank_face_candidates(
        face.embedding,
        identity_names,
        identity_matrix,
        identity_db,
        pose_label=face.pose_label,
        lighting_label=appearance_profiles.lighting_label(face.crop_jpeg),
        capture_timestamp=appearance_profiles.capture_timestamp(face.src_str),
        hard_negatives=hard_negatives,
    )
    if not candidates:
        return None
    best = candidates[0]
    margin = identity_profiles.candidate_margin(candidates)
    threshold = min(
        MATCH_MAX_DISTANCE,
        float(identity_db.match_thresholds.get(best.name, MATCH_MAX_DISTANCE)),
    )
    if best.distance > threshold + 0.08 or margin < 0.03:
        return None
    verification = matcher.verify(face.crop_jpeg, best.name)
    if not verification.accepted:
        return None
    return best.name, float(best.distance), float(margin)


def move_review_file(source: Path,
                     destination_root: Path,
                     *,
                     status: str,
                     extra: dict,
                     dry_run: bool,
                     ) -> Path:
    destination = sort_photos.unique_path(destination_root / source.name)
    if dry_run:
        return destination
    return operation_ledger.move_path(
        source,
        destination,
        sorted_root=DEFAULT_SORTED,
        operation="recover_no_usable_faces.reclassify",
        reason=f"recover queued face result: {status}",
        extra=extra,
    )


def copy_to_person(source: Path,
                   person: str,
                   source_hash: str,
                   existing_hashes: dict[str, set[str]],
                   next_indexes: dict[Path, int],
                   *,
                   dry_run: bool,
                   ) -> tuple[Path | None, bool]:
    person_dir = DEFAULT_PEOPLE / person
    if source_hash in existing_hashes[person]:
        return None, True
    if not dry_run:
        free_bytes = shutil.disk_usage(DEFAULT_SORTED).free
        if free_bytes - source.stat().st_size < MIN_INTERNAL_FREE_BYTES:
            raise RecoveryPaused(
                "Mac free-space safety reserve would be crossed "
                f"({free_bytes / (1024 ** 3):.1f} GB currently free)"
            )
    base_dir = person_dir / sort_photos.PERSON_PHOTOS_DIR
    destination = sort_photos.next_numbered_dest(
        base_dir, person_dir, person, source, next_indexes
    )
    if dry_run:
        return destination, False
    sort_photos._atomic_copy(source, destination)
    try:
        destination, _nudity_status = sort_photos.maybe_move_to_nudity_subfolder(
            destination, person_dir
        )
        sort_photos.verify_original_copy(source, destination, source_hash)
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    existing_hashes[person].add(source_hash)
    return destination, False


def merge_recovered_faces_into_cache(
    cache: sort_photos.CacheState,
    entries: list[tuple[Path, sort_photos.CachedFace, str]],
) -> list[sort_photos.CachedFace]:
    merged: list[sort_photos.CachedFace] = []
    destination_paths = {os.path.realpath(str(path)) for path, _face, _person in entries}
    cache.file_signatures = {
        path: signature for path, signature in cache.file_signatures.items()
        if os.path.realpath(path) not in destination_paths
    }
    cache.faces = [
        face for face in cache.faces
        if os.path.realpath(face.src_str) not in destination_paths
    ]
    merged_keys: set[tuple[str, int]] = set()
    for destination, face, person in entries:
        if not destination.is_file():
            continue
        destination_text = str(destination.resolve())
        key = (destination_text, int(face.face_index))
        if key in merged_keys:
            continue
        merged_keys.add(key)
        cache.file_signatures[destination_text] = sort_photos.file_signature(destination)
        recovered = replace(face, src_str=destination_text, label=person)
        cache.faces.append(recovered)
        merged.append(recovered)
    return merged


def persist_recovered_faces(
    entries: list[tuple[Path, sort_photos.CachedFace, str]],
    *,
    rebuild_identity: bool = True,
) -> int:
    if not entries:
        return 0
    cache = sort_photos.load_cache()
    recovered = merge_recovered_faces_into_cache(cache, entries)
    if not recovered:
        return 0
    sort_photos.save_cache(cache)
    destinations = [Path(face.src_str) for face in recovered]
    diagnostics = {str(path): "accepted_face_recovery" for path in destinations}
    sort_photos.persist_detection_batch(
        destinations,
        recovered,
        diagnostics,
        sort_photos.analysis_index_file(),
    )
    if rebuild_identity:
        sort_photos.build_identity_db_from_person_folders(DEFAULT_PEOPLE)
    return len(recovered)


def iter_images(root: Path) -> list[Path]:
    return sorted(
        (
            path for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in sort_photos.IMAGE_EXTS
        ),
        key=lambda path: str(path.relative_to(root)).casefold(),
    )


def group_exact_duplicates(files: list[Path]) -> list[tuple[str, list[Path]]]:
    groups: dict[str, list[Path]] = defaultdict(list)
    for index, path in enumerate(files, 1):
        try:
            digest = sort_photos.sha256_file(path)
        except OSError:
            digest = f"unreadable:{path}"
        groups[digest].append(path)
        if index % 1000 == 0 or index == len(files):
            print(f"Hashing review files: {index}/{len(files)}")
    return sorted(groups.items(), key=lambda item: item[1][0].name.casefold())


def _detection_worker(task_queue, result_queue,
                      primary_det_size: int,
                      fallback_det_size: int) -> None:
    try:
        app = sort_photos._build_app((primary_det_size, primary_det_size))
        fallback_app = sort_photos._build_app((fallback_det_size, fallback_det_size))
    except Exception as error:  # noqa: BLE001
        result_queue.put(("worker_start_failed", str(error)))
        return
    while True:
        task = task_queue.get()
        if task is None:
            return
        task_index, source_text = task
        source = Path(source_text)
        try:
            diagnostics: dict[str, str] = {}
            faces = sort_photos._detect_one_image(
                source, app, diagnostics=diagnostics, fallback_app=fallback_app
            )
            result_queue.put((
                "result",
                task_index,
                source_text,
                diagnostics.get(str(source), ""),
                faces,
            ))
        except Exception as error:  # noqa: BLE001
            result_queue.put(("result", task_index, source_text,
                              f"detector_error:{type(error).__name__}", [], str(error)))


def iter_detection_results(files: list[Path], workers: int,
                           primary_det_size: int,
                           fallback_det_size: int,
                           *, index_path: Path | None = None,
                           cache_stats: Counter[str] | None = None):
    stats = cache_stats if cache_stats is not None else Counter()
    database = analysis_index.AnalysisIndex(
        index_path or sort_photos.analysis_index_file()
    )
    pending: list[Path] = []
    try:
        for task_index, source in enumerate(files):
            cached = database.cached_detections(source, sort_photos.config_fingerprint())
            if cached is None:
                pending.append(source)
                continue
            faces = [
                sort_photos.index_record_to_cached_face(source, record)
                for record in cached.detections
            ]
            stats["sqlite_hits"] += 1
            yield task_index, source, cached.status, faces, ""

        if not pending:
            return

        def persist(source: Path, status: str, faces: list[sort_photos.CachedFace]) -> None:
            database.replace_detections(
                source,
                sort_photos.config_fingerprint(),
                status,
                [sort_photos.cached_face_to_index_record(face) for face in faces],
            )
            stats["detected"] += 1
            if stats["detected"] % 25 == 0:
                database.commit()

        if workers <= 1:
            app = sort_photos._build_app((primary_det_size, primary_det_size))
            fallback_app = sort_photos._build_app((fallback_det_size, fallback_det_size))
            for task_index, source in enumerate(pending):
                diagnostics: dict[str, str] = {}
                faces = sort_photos._detect_one_image(
                    source, app, diagnostics=diagnostics, fallback_app=fallback_app
                )
                status = diagnostics.get(str(source), "")
                persist(source, status, faces)
                yield task_index, source, status, faces, ""
            return

        for key, value in sort_photos.DETECTION_WORKER_ENV_LIMITS.items():
            os.environ.setdefault(key, value)
        context = mp.get_context("spawn")
        task_queue = context.Queue()
        result_queue = context.Queue()
        processes = [
            context.Process(
                target=_detection_worker,
                args=(task_queue, result_queue, primary_det_size, fallback_det_size),
            )
            for _ in range(workers)
        ]
        for process in processes:
            process.start()
        for task_index, source in enumerate(pending):
            task_queue.put((task_index, str(source)))
        for _ in processes:
            task_queue.put(None)

        completed = 0
        try:
            while completed < len(pending):
                try:
                    payload = result_queue.get(timeout=30)
                except queue.Empty:
                    failed = [process for process in processes if not process.is_alive() and process.exitcode]
                    if failed:
                        raise RuntimeError(
                            f"detection worker exited unexpectedly: {failed[0].exitcode}"
                        )
                    continue
                if payload[0] == "worker_start_failed":
                    raise RuntimeError(f"detection worker could not start: {payload[1]}")
                _kind, task_index, source_text, detector_status, faces, *error = payload
                completed += 1
                source = Path(source_text)
                persist(source, detector_status, faces)
                yield task_index, source, detector_status, faces, (error[0] if error else "")
        finally:
            for process in processes:
                if process.is_alive():
                    process.terminate()
                process.join(timeout=5)
            for work_queue in (task_queue, result_queue):
                work_queue.cancel_join_thread()
                work_queue.close()
    finally:
        database.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--source-kind",
        choices=("no_usable_face", "unknown_identity"),
        default=None,
        help="How unresolved files should be handled. Inferred from the input folder by default.",
    )
    parser.add_argument("--max-images", type=int, default=0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--workers", type=int, default=1,
                        help="Persistent detector workers. Use 2 for a faster one-time recovery.")
    parser.add_argument("--primary-det-size", type=int, default=1024)
    parser.add_argument("--fallback-det-size", type=int, default=384)
    args = parser.parse_args()

    input_root = args.input.expanduser().resolve()
    source_kind = args.source_kind or (
        "unknown_identity" if input_root.name == "unknown_identity" else "no_usable_face"
    )
    if not input_root.exists():
        print(f"Input folder does not exist: {input_root}")
        return 2
    identity_db = sort_photos.load_identity_db()
    if identity_db is None or not identity_db.identities:
        print("Known-person identity DB is unavailable. Rebuild it before recovery.")
        return 2

    files = iter_images(input_root)
    if args.max_images > 0:
        files = files[:args.max_images]
    print("Known-Face Recovery" if source_kind == "unknown_identity" else "No-Usable-Face Recovery")
    print("=" * 60)
    print(f"Input:             {input_root}")
    print(f"Images to recheck: {len(files)}")
    print(f"Known identities:  {len(identity_db.identities)}")
    print(f"Detector workers:  {max(1, args.workers)}")
    print(f"Detector sizes:    {max(320, args.primary_det_size)} primary / "
          f"{max(320, args.fallback_det_size)} fallback")
    print(f"Source queue:      {source_kind}")
    print(f"Mode:              {'DRY RUN' if args.dry_run else 'SAFE RECOVERY'}")
    if not files:
        return 0

    duplicate_groups = group_exact_duplicates(files)
    representatives = [paths[0] for _digest, paths in duplicate_groups]
    group_by_representative = {
        str(paths[0]): (digest, paths) for digest, paths in duplicate_groups
    }
    print(f"Unique file bytes: {len(representatives)} ({len(files) - len(representatives)} exact duplicates reused)")

    names = sorted(identity_db.identities)
    identity_matrix = np.stack([identity_db.identities[name] for name in names])
    existing_hashes = load_existing_hashes(DEFAULT_PEOPLE)
    next_indexes: dict[Path, int] = {}
    counts: Counter[str] = Counter()
    paused = False
    report_id = time.strftime("%Y%m%d_%H%M%S")
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / f"{source_kind}_recovery_{report_id}.jsonl"
    detection_cache_stats: Counter[str] = Counter()
    recovered_cache_entries: list[tuple[Path, sort_photos.CachedFace, str]] = []
    hard_negatives = identity_hard_negatives.vectors_by_person(
        sort_photos.IDENTITY_HARD_NEGATIVES_FILE
    )
    print(f"Hard-negative guards:{sum(map(len, hard_negatives.values())):>5}")
    consensus_stats: Counter[str] = Counter()
    batch_decisions = (
        build_source_batch_consensus(
            representatives,
            input_root,
            names,
            identity_matrix,
            identity_db,
            hard_negatives,
            workers=max(1, args.workers),
            primary_det_size=max(320, args.primary_det_size),
            fallback_det_size=max(320, args.fallback_det_size),
            stats=consensus_stats,
        )
        if source_kind == "unknown_identity"
        else {}
    )
    print(f"Batch consensus:    {len(batch_decisions):>5} corroborated face(s)")
    secondary_db = secondary_identity_matcher.load()
    secondary_matcher = (
        secondary_identity_matcher.SecondaryMatcher(secondary_db)
        if secondary_db is not None
        and secondary_db.primary_signature == secondary_identity_matcher.primary_signature(identity_db)
        else None
    )
    print(f"Secondary verifier: {'ready' if secondary_matcher else 'not built (safe skip)'}")

    with report_path.open("a", encoding="utf-8") as report:
        results = iter_detection_results(
            representatives,
            max(1, args.workers),
            max(320, args.primary_det_size),
            max(320, args.fallback_det_size),
            index_path=sort_photos.analysis_index_file(),
            cache_stats=detection_cache_stats,
        )
        for processed_index, (_task_index, source, detector_status, faces, detector_error) in enumerate(results, 1):
            source_hash, source_group = group_by_representative[str(source)]
            group_size = len(source_group)
            matches: dict[str, tuple[float, float, float, sort_photos.CachedFace]] = {}
            for face in faces:
                lighting, captured_at = appearance_profiles.query_attributes(
                    face.crop_jpeg, source
                )
                matched = match_face(
                    np.asarray(face.embedding, dtype=np.float32),
                    names,
                    identity_matrix,
                    identity_db,
                    quality=float(face.quality),
                    pose_label=str(getattr(face, "pose_label", "unknown") or "unknown"),
                    lighting_label=lighting,
                    capture_timestamp=captured_at,
                    hard_negatives=hard_negatives,
                )
                if matched is None:
                    batch_match = batch_decisions.get((str(source), int(face.face_index)))
                    if batch_match is not None:
                        person, distance, margin = (
                            batch_match.person, batch_match.distance, batch_match.margin
                        )
                        counts["source_batch_consensus"] += 1
                    else:
                        secondary_match = borderline_secondary_match(
                            face,
                            names,
                            identity_matrix,
                            identity_db,
                            hard_negatives,
                            secondary_matcher,
                        )
                        if secondary_match is None:
                            continue
                        person, distance, margin = secondary_match
                        counts["secondary_agreement"] += 1
                else:
                    person, distance, margin = matched
                prior = matches.get(person)
                if prior is None or distance < prior[0]:
                    matches[person] = (distance, margin, float(face.quality), face)

            record: dict = {
                "source": str(source),
                "sources": [str(path) for path in source_group],
                "exact_duplicate_count": group_size - 1,
                "detector_status": detector_status,
                "detected_faces": len(faces),
                "matches": {
                    person: {
                        "distance": round(values[0], 6),
                        "margin": round(values[1], 6),
                        "quality": round(values[2], 6),
                    }
                    for person, values in sorted(matches.items())
                },
                "dry_run": bool(args.dry_run),
            }
            if detector_error:
                record["detector_error"] = detector_error
            should_stop = False
            try:
                if matches:
                    destinations: list[str] = []
                    reused = 0
                    for person in sorted(matches):
                        destination, already_present = copy_to_person(
                            source, person, source_hash, existing_hashes, next_indexes,
                            dry_run=args.dry_run,
                        )
                        if already_present:
                            reused += 1
                        elif destination is not None:
                            destinations.append(str(destination))
                            if not args.dry_run:
                                recovered_cache_entries.append(
                                    (destination, matches[person][3], person)
                                )
                    archived = [
                        str(move_review_file(
                            group_source,
                            DEFAULT_SORTED / "_source_review" / "ready_to_delete" / f"recovered_{source_kind}",
                            status="matched",
                            extra={"source_kind": source_kind, "people": sorted(matches), "destinations": destinations,
                                   "content_sha256": source_hash},
                            dry_run=args.dry_run,
                        ))
                        for group_source in source_group
                    ]
                    record.update({
                        "outcome": "matched",
                        "destinations": destinations,
                        "already_present": reused,
                        "review_archive": archived,
                    })
                    counts["matched"] += group_size
                elif source_kind == "unknown_identity":
                    record.update({
                        "outcome": "still_unknown",
                        "retained_paths": [str(path) for path in source_group],
                    })
                    if faces:
                        counts["still_unknown_with_face"] += group_size
                    elif detector_status.startswith("face_quality_review:"):
                        counts["still_unknown_quality"] += group_size
                    else:
                        counts["still_unknown_no_face"] += group_size
                elif faces:
                    destinations = [
                        str(move_review_file(
                            group_source,
                            DEFAULT_REVIEW / "unknown_identity",
                            status="unknown_identity",
                            extra={"detected_faces": len(faces),
                                   "detector_status": detector_status,
                                   "content_sha256": source_hash},
                            dry_run=args.dry_run,
                        ))
                        for group_source in source_group
                    ]
                    record.update({"outcome": "unknown_identity", "destination": destinations})
                    counts["unknown_identity"] += group_size
                elif detector_status.startswith("face_quality_review:"):
                    destinations = [
                        str(move_review_file(
                            group_source,
                            DEFAULT_REVIEW / "face_quality_review",
                            status="face_quality_review",
                            extra={"detector_status": detector_status,
                                   "content_sha256": source_hash},
                            dry_run=args.dry_run,
                        ))
                        for group_source in source_group
                    ]
                    record.update({"outcome": "face_quality_review", "destination": destinations})
                    counts["face_quality_review"] += group_size
                else:
                    record["outcome"] = "no_face_detected"
                    counts["no_face_detected"] += group_size
            except RecoveryPaused as error:
                record.update({"outcome": "paused_insufficient_space", "error": str(error)})
                counts["paused_insufficient_space"] += group_size
                paused = True
                should_stop = True
            except Exception as error:  # noqa: BLE001
                record.update({"outcome": "failed", "error": str(error)})
                counts["failed"] += group_size

            report.write(json.dumps(record, sort_keys=True) + "\n")
            report.flush()
            if should_stop:
                results.close()
                print(f"Recovery paused safely: {record['error']}")
                break
            if processed_index == 1 or processed_index % 25 == 0 or processed_index == len(representatives):
                unresolved = (
                    counts["still_unknown_with_face"]
                    + counts["still_unknown_quality"]
                    + counts["still_unknown_no_face"]
                    + counts["unknown_identity"]
                )
                print(
                    f"[{processed_index}/{len(representatives)} unique] matched={counts['matched']} "
                    f"unresolved={unresolved} "
                    f"quality-review={counts['face_quality_review']} "
                    f"no-face={counts['no_face_detected']} failed={counts['failed']}"
                )

    cached_recoveries = (
        0 if args.dry_run else persist_recovered_faces(recovered_cache_entries)
    )
    print("\nRecovery Summary")
    print("=" * 60)
    for key in (
        "matched", "still_unknown_with_face", "still_unknown_quality",
        "still_unknown_no_face", "unknown_identity", "face_quality_review",
        "no_face_detected", "paused_insufficient_space", "failed",
        "source_batch_consensus",
        "secondary_agreement",
    ):
        print(f"{key:22s} {counts[key]}")
    print(f"SQLite detections: {detection_cache_stats['sqlite_hits']} reused / "
          f"{detection_cache_stats['detected']} newly analyzed")
    if consensus_stats:
        print(f"Consensus prepass:  {consensus_stats['sqlite_hits']} SQLite reused / "
              f"{consensus_stats['detected']} newly analyzed")
    if secondary_matcher is not None:
        secondary_matcher.flush()
    print(f"Recovered cache entries: {cached_recoveries}")
    print(f"Report: {report_path}")
    if paused:
        return 3
    return 1 if counts["failed"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

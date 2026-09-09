#!/usr/bin/env python3
"""Continuous Quick Review for known faces held in unknown_identity.

The dashboard first applies a benchmark-gated safe automatic lane, then presents
only ambiguous visual clusters for manual review. Automatic matches require a
strict single-image decision, independent-model agreement, or corroborated
multi-image consensus. Manual confirmations alone become trusted prototypes.
"""

from __future__ import annotations

import argparse
import datetime as dt
import gc
import hashlib
import html
import json
import mimetypes
import os
import pickle
import queue
import shutil
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse

import numpy as np
from sklearn.cluster import DBSCAN

import confirm_unknown_identity
import appearance_profiles
import identity_confirmations
import identity_evaluation
import evaluation_runtime
from evaluation_checkpoints import BenchmarkCheckpoints
import identity_hard_negatives
import identity_profiles
import evaluation_enrollment
import pipeline_paths
import recover_no_usable_faces
import sort_photos
import secondary_identity_matcher
from pipeline_progress import StageProgress, terminal_progress


DEFAULT_UNKNOWN_ROOT = (
    pipeline_paths.SOURCE_REVIEW / "unassigned_intake" / "unknown_identity"
)
DEFAULT_OUTPUT_DIR = pipeline_paths.SOURCE_REVIEW / "identity_audits" / "unknown_review"
DEFAULT_DECISIONS = DEFAULT_OUTPUT_DIR / "unknown_identity_review_decisions.json"
DEFAULT_SESSION = DEFAULT_OUTPUT_DIR / "unknown_identity_review_session.json"
DECISIONS_VERSION = 3
REVIEW_POLICY_VERSION = 7
MAX_CLUSTER_ACTION_ITEMS = 500
MAX_BATCH_ACTION_ITEMS = 50
AUTO_CLUSTER_MIN_MARGIN = 0.08
AUTO_CLUSTER_MIN_AGREEMENT = 0.80
AUTO_CLUSTER_MAX_COHESION = 0.45
AUTO_JOINT_MAX_PRIMARY_DISTANCE = 0.38
AUTO_JOINT_MAX_SECONDARY_DISTANCE = 0.42
AUTO_JOINT_MIN_PRIMARY_MARGIN = 0.08
AUTO_JOINT_MIN_SECONDARY_MARGIN = 0.08
AUTO_JOINT_MIN_QUALITY = 0.40
AUTO_RESCUE_MAX_PRIMARY_DISTANCE = 0.58
AUTO_RESCUE_MIN_PRIMARY_MARGIN = 0.04
AUTO_RESCUE_MAX_SECONDARY_DISTANCE = 0.24
AUTO_RESCUE_MIN_SECONDARY_MARGIN = 0.14
AUTO_RESCUE_MIN_QUALITY = 0.45
AUTO_GATE_MIN_CONFIRMED_CASES = 100
AUTO_GATE_CACHE_VERSION = 7
AUTO_SWEEP_STATE_VERSION = 3
TRUSTED_REVIEW_PROTOTYPES_PER_PERSON = 24
AUTO_SWEEP_BATCH_SIZE = 500
RESOLVED_ACTIONS = {
    "confirmed",
    "keep_unknown",
    "ignored",
    "moved_to_junk",
    "routed_multi_face",
    "routed_no_usable_face",
    "routed_face_quality_review",
    "routed_technical_review",
}
MOVING_ACTIONS = {
    "confirmed",
    "moved_to_junk",
    "routed_multi_face",
    "routed_no_usable_face",
    "routed_face_quality_review",
    "routed_technical_review",
}
RETAINED_ACTIONS = {"keep_unknown", "ignored"}
ACTION_OUTCOME_STATES = {
    "confirmed": "verified_organized",
    "keep_unknown": "retained_unknown",
    "ignored": "retained_ignored",
    "moved_to_junk": "recoverable_junk",
    "routed_multi_face": "routed_specialist",
    "routed_no_usable_face": "routed_specialist",
    "routed_face_quality_review": "routed_specialist",
    "routed_technical_review": "routed_specialist",
}

_ITEM_CONTENT_CACHE: dict[str, tuple[tuple[int, ...], str]] = {}


class ReviewActionConflict(ValueError):
    """Raised when an item already belongs to another active review job."""


@dataclass(frozen=True)
class UnknownItem:
    key: str
    path: Path
    face: sort_photos.CachedFace
    candidates: tuple[identity_profiles.IdentityCandidate, ...]
    secondary: secondary_identity_matcher.SecondaryVerification | None = None


@dataclass(frozen=True)
class UnknownCluster:
    key: str
    items: tuple[UnknownItem, ...]
    candidates: tuple[identity_profiles.IdentityCandidate, ...]
    cohesion: float


@dataclass(frozen=True)
class UnsupportedUnknown:
    key: str
    path: Path
    queue_kind: str
    detector_status: str
    error: str = ""


from recognition_policy import AutomaticMatch


def prepare_secondary_verifier(
    identity_db: sort_photos.IdentityDB,
    cache: sort_photos.CacheState,
    *,
    requested: bool,
    reference_index=None,
    progress=terminal_progress,
) -> tuple[secondary_identity_matcher.SecondaryMatcher | None, str]:
    """Load the verifier and refresh it when identity profiles have changed.

    Safe automatic review intentionally requires an independent model. Identity
    confirmations rebuild the primary database, so treating a stale secondary
    signature as permanently unavailable causes an avoidable manual-review
    backlog. Refresh only when safe auto-review was explicitly requested.
    """
    expected_signature = secondary_identity_matcher.primary_signature(identity_db)
    secondary_db = secondary_identity_matcher.load()
    if (
        secondary_db is not None
        and secondary_db.primary_signature == expected_signature
        and secondary_identity_matcher.trusted_snapshot_is_current(
            secondary_db, sort_photos.IDENTITY_CONFIRMATIONS_FILE
        )
        and secondary_db.identities
    ):
        return secondary_identity_matcher.SecondaryMatcher(secondary_db), "ready"
    if not requested:
        return None, "not requested"

    reason = "missing" if secondary_db is None else "out of date"
    progress(
        f"Independent verifier is {reason}; refreshing it for the current "
        f"{len(identity_db.identities)}-person identity library..."
    )
    try:
        secondary_db = secondary_identity_matcher.build_database(
            identity_db, cache, confirmations_path=sort_photos.IDENTITY_CONFIRMATIONS_FILE,
            reference_index=reference_index, progress=progress,
        )
    except Exception as error:  # noqa: BLE001
        return None, f"refresh failed: {error}"
    if (
        secondary_db is None
        or not secondary_db.identities
        or secondary_db.primary_signature != expected_signature
        or not secondary_identity_matcher.trusted_snapshot_is_current(
            secondary_db, sort_photos.IDENTITY_CONFIRMATIONS_FILE
        )
    ):
        return None, "refresh did not produce a current verifier database"
    return secondary_identity_matcher.SecondaryMatcher(secondary_db), "refreshed"


def _automatic_item_evidence(item, identity_db):
    import recognition_policy
    return recognition_policy._automatic_item_evidence(item, identity_db, settings=sys.modules[__name__])


def _rank_face_candidates(
    face: sort_photos.CachedFace,
    path: Path,
    identity_db: sort_photos.IdentityDB,
    hard_negatives: dict[str, list[np.ndarray]],
    review_prototypes: dict[str, list[np.ndarray]] | None = None,
    compiled: identity_profiles.CompiledProfiles | None = None,
) -> tuple[identity_profiles.IdentityCandidate, ...]:
    lighting, captured_at = appearance_profiles.query_attributes(face.crop_jpeg, path)
    return tuple(identity_profiles.rank_candidates(
        face.embedding,
        identity_db.identities,
        review_prototypes or identity_db.prototypes,
        pose_label=str(getattr(face, "pose_label", "unknown") or "unknown"),
        pose_prototypes=identity_db.pose_prototypes,
        lighting_label=lighting,
        capture_timestamp=captured_at,
        appearance_prototypes=identity_db.appearance_prototypes,
        appearance_era_cutoffs=identity_db.appearance_era_cutoffs,
        hard_negatives=hard_negatives,
        compiled=compiled,
    )[:3])


def automatic_review_item(
    key: str,
    path: Path,
    face: sort_photos.CachedFace,
    identity_db: sort_photos.IdentityDB,
    hard_negatives: dict[str, list[np.ndarray]],
    review_prototypes: dict[str, list[np.ndarray]] | None = None,
    secondary_matcher: secondary_identity_matcher.SecondaryMatcher | None = None,
    compiled: identity_profiles.CompiledProfiles | None = None,
) -> UnknownItem:
    """Share candidate selection and bounded verifier work with daily ingest."""
    candidates = _rank_face_candidates(
        face, path, identity_db, hard_negatives, review_prototypes, compiled
    )
    secondary = None
    if candidates and secondary_matcher is not None:
        best = candidates[0]
        margin = identity_profiles.candidate_margin(candidates)
        threshold = min(
            recover_no_usable_faces.MATCH_MAX_DISTANCE,
            identity_db.match_thresholds.get(
                best.name, recover_no_usable_faces.MATCH_MAX_DISTANCE
            ),
        )
        quality = float(face.quality)
        exceptional = (
            best.distance <= recover_no_usable_faces.MATCH_EXCEPTIONAL_DISTANCE
            and margin >= recover_no_usable_faces.MATCH_EXCEPTIONAL_MARGIN
        )
        joint_candidate = (
            best.distance <= min(AUTO_JOINT_MAX_PRIMARY_DISTANCE, threshold + 0.08)
            and margin >= AUTO_JOINT_MIN_PRIMARY_MARGIN
            and quality >= AUTO_JOINT_MIN_QUALITY
        )
        consensus_candidate = (
            best.distance <= threshold
            and margin >= recover_no_usable_faces.MATCH_MIN_MARGIN
            and (quality >= recover_no_usable_faces.MATCH_MIN_QUALITY or exceptional)
        )
        rescue_candidate = (
            best.distance <= AUTO_RESCUE_MAX_PRIMARY_DISTANCE
            and margin >= AUTO_RESCUE_MIN_PRIMARY_MARGIN
            and quality >= AUTO_RESCUE_MIN_QUALITY
        )
        if face.crop_jpeg and (joint_candidate or consensus_candidate or rescue_candidate):
            secondary = secondary_matcher.verify(face.crop_jpeg, best.name)
    return UnknownItem(key, path, face, candidates, secondary)


def _without_exact_query_prototype(
    profiles: dict[str, list[np.ndarray]] | None,
    embedding: np.ndarray,
) -> dict[str, list[np.ndarray]] | None:
    """Remove the benchmark image itself from prototype retrieval."""
    if profiles is None:
        return None
    query = identity_profiles.normalize_vector(embedding)
    filtered: dict[str, list[np.ndarray]] = {}
    for person, values in profiles.items():
        kept = [
            value for value in values
            if 1.0 - float(identity_profiles.normalize_vector(value) @ query) > 1e-6
        ]
        filtered[person] = kept
    return filtered


def build_trusted_review_prototypes(
    identity_db: sort_photos.IdentityDB,
    cache: sort_photos.CacheState,
    *,
    confirmations_path: Path = sort_photos.IDENTITY_CONFIRMATIONS_FILE,
    limit_per_person: int = TRUSTED_REVIEW_PROTOTYPES_PER_PERSON,
    reference_index=None,
    progress=terminal_progress,
) -> tuple[dict[str, list[np.ndarray]], dict[str, int]]:
    """Extend Quick Review matching with bounded explicit confirmations.

    These prototypes are used only by the shared, benchmark-gated recovery
    lane. They do not change production centroids or calibrated thresholds.
    The dual-model benchmark remains the authority for automatic recovery.
    """
    canonical_names = {
        name.casefold(): name for name in identity_db.identities
    }
    faces_by_source = reference_index.faces_by_source if reference_index is not None else None
    if faces_by_source is None:
        faces_by_source = _faces_by_source(cache)
    grouped: dict[str, list[identity_profiles.ReferenceSample]] = {}
    seen: set[tuple[str, str, int]] = set()
    for person, source, candidates in identity_confirmations.verified_records(
        confirmations_path, faces_by_source, canonical_names,
        reference_index=reference_index, progress=progress,
    ):
        source_key = os.path.realpath(str(source))
        face = max(candidates, key=lambda value: float(value.quality))
        key = (person.casefold(), source_key, int(face.face_index))
        if key in seen:
            continue
        seen.add(key)
        grouped.setdefault(person, []).append(identity_profiles.ReferenceSample(
            source=str(source),
            embedding=np.asarray(face.embedding, dtype=np.float32),
            quality=float(face.quality),
            pose_label=str(getattr(face, "pose_label", "unknown") or "unknown"),
        ))

    profiles: dict[str, list[np.ndarray]] = {}
    selected_counts: dict[str, int] = {}
    bounded_limit = max(0, int(limit_per_person))
    profile_progress = StageProgress("Building trusted recovery profiles", len(identity_db.identities), progress)
    for person in profile_progress.items(identity_db.identities):
        trusted = identity_profiles.select_diverse_samples(
            grouped.get(person, []),
            limit=bounded_limit,
        ) if bounded_limit else []
        values = [
            *identity_db.prototypes.get(person, []),
            *(identity_profiles.normalize_vector(sample.embedding) for sample in trusted),
        ]
        profiles[person] = values
        if trusted:
            selected_counts[person] = len(trusted)
    return profiles, selected_counts


def automatic_matches(clusters, identity_db, *, require_secondary=False):
    import recognition_policy
    return recognition_policy.automatic_matches(
        clusters, identity_db, require_secondary=require_secondary, settings=sys.modules[__name__])


def legacy_item_key(path: Path) -> str:
    canonical = str(path.expanduser().resolve(strict=False))
    try:
        stat = path.stat()
        signature = f"{canonical}\0{stat.st_size}\0{stat.st_mtime_ns}"
    except OSError:
        signature = canonical
    return hashlib.sha256(
        signature.encode("utf-8", errors="surrogateescape")
    ).hexdigest()[:20]


def item_content_sha256(path: Path) -> str:
    """Return a cached content identity without trusting path timestamps alone."""
    canonical = str(path.expanduser().resolve(strict=False))
    try:
        signature = sort_photos.content_identity.file_version(path)
    except OSError:
        return ""
    cached = _ITEM_CONTENT_CACHE.get(canonical)
    if cached is not None and cached[0] == signature:
        return cached[1]
    try:
        digest = sort_photos.content_identity.content_sha256(path)
    except OSError:
        return ""
    if len(_ITEM_CONTENT_CACHE) >= 8192:
        _ITEM_CONTENT_CACHE.clear()
    _ITEM_CONTENT_CACHE[canonical] = (signature, digest)
    return digest


def item_key(path: Path) -> str:
    """Identify one physical review item by canonical path and exact content."""
    canonical = str(path.expanduser().resolve(strict=False))
    content_sha256 = item_content_sha256(path)
    signature = f"{canonical}\0{content_sha256 or 'unavailable'}"
    return hashlib.sha256(
        signature.encode("utf-8", errors="surrogateescape")
    ).hexdigest()[:24]


def review_model_signature(
    identity_db: sort_photos.IdentityDB,
    hard_negatives_path: Path,
) -> str:
    """Fingerprint every input that can change a retained-unknown decision."""
    hard_negative_sha256 = ""
    if hard_negatives_path.is_file():
        try:
            hard_negative_sha256 = sort_photos.sha256_file(hard_negatives_path)
        except OSError:
            pass
    payload = {
        "policy_version": REVIEW_POLICY_VERSION,
        "primary_signature": secondary_identity_matcher.primary_signature(identity_db),
        "hard_negatives_sha256": hard_negative_sha256,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_decisions(path: Path) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {"version": DECISIONS_VERSION, "items": {}, "content_outcomes": {}}
    if not isinstance(payload, dict) or not isinstance(payload.get("items"), dict):
        return {"version": DECISIONS_VERSION, "items": {}, "content_outcomes": {}}
    outcomes = payload.get("content_outcomes", {})
    if not isinstance(outcomes, dict):
        outcomes = {}
    decisions = {
        "version": DECISIONS_VERSION,
        "items": payload["items"],
        "content_outcomes": outcomes,
        "content_rechecks": payload.get("content_rechecks", {}),
        "legacy_content_indexed": bool(payload.get("legacy_content_indexed")),
        "_needs_migration": int(payload.get("version", 1)) < DECISIONS_VERSION,
    }
    _index_decisions(decisions)
    return decisions


def _decision_order(record: dict) -> tuple[int, int]:
    # A legacy profile-reset timestamp is not a new user decision. Any real
    # decision for these bytes takes precedence over that reconstructed choice.
    stamp = int(record.get("decided_at_ns", 0)) or int(record.get("decided_at", 0)) * 1_000_000_000
    return (0 if record.get("restored_profile_reset") else 1, stamp)


def _index_decisions(decisions: dict) -> None:
    if decisions.get("_indexed"):
        return
    outcomes = decisions.setdefault("content_outcomes", {})
    path_index = {}
    if not decisions.get("legacy_content_indexed"):
        for key, record in decisions.setdefault("items", {}).items():
            if (not isinstance(record, dict) or record.get("content_sha256")
                    or not record.get("path")):
                continue
            path = Path(record["path"])
            if key == legacy_item_key(path):
                digest = item_content_sha256(path)
                if digest:
                    record["content_sha256"] = digest
        decisions["legacy_content_indexed"] = True
        decisions["_needs_migration"] = True
    records = list(outcomes.values()) + list(decisions.setdefault("items", {}).values())
    for record in records:
        if not isinstance(record, dict):
            continue
        if (record.get("outcome_state") == "requeued_model_changed"
                and not record.get("action")
                and record.get("previous_action") in RETAINED_ACTIONS):
            record.update(
                action=record["previous_action"],
                outcome_state=ACTION_OUTCOME_STATES[record["previous_action"]],
                model_signature=record.get("previous_model_signature", ""),
                restored_profile_reset=True,
            )
            decisions["_needs_migration"] = True
        digest = str(record.get("content_sha256", ""))
        previous = outcomes.get(digest)
        if digest and (not isinstance(previous, dict)
                       or _decision_order(record) > _decision_order(previous)):
            outcomes[digest] = record
    for key, record in decisions["items"].items():
        if not isinstance(record, dict) or not record.get("path"):
            continue
        canonical = os.path.realpath(str(record["path"]))
        previous = path_index.get(canonical)
        if previous is None or _decision_order(record) > _decision_order(previous[1]):
            path_index[canonical] = (key, record)
    decisions["_path_index"] = path_index
    decisions["_last_decision_ns"] = max(
        (_decision_order(record)[1] for record in records if isinstance(record, dict)), default=0
    )
    if not isinstance(decisions.get("content_rechecks"), dict):
        decisions["content_rechecks"] = {}
    decisions["_indexed"] = True


def save_decisions(path: Path, decisions: dict) -> None:
    _index_decisions(decisions)
    path.parent.mkdir(parents=True, exist_ok=True)
    if decisions.get("_needs_migration") and path.is_file():
        backup = path.with_name(f"{path.name}.pre-v3.{time.time_ns()}.bak")
        shutil.copy2(path, backup)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "version": DECISIONS_VERSION,
        "items": decisions.get("items", {}),
        "content_outcomes": decisions.get("content_outcomes", {}),
        "content_rechecks": decisions.get("content_rechecks", {}),
        "legacy_content_indexed": True,
    }
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    decisions["_needs_migration"] = False


def decision_for_path(
    decisions: dict, path: Path, *, known_key: str = "",
) -> tuple[str, dict] | None:
    """Use the latest decision for exact bytes, never a stale path's choice."""
    _index_decisions(decisions)
    items = decisions.setdefault("items", {})
    current_key = item_key(path)
    legacy_key = legacy_item_key(path)
    digest = item_content_sha256(path)
    if not digest and known_key and not path.exists():
        digest = str(items.get(known_key, {}).get("content_sha256", ""))
    candidates = []
    for key in dict.fromkeys([current_key, legacy_key, known_key]):
        record = items.get(key)
        if not isinstance(record, dict):
            continue
        recorded_digest = str(record.get("content_sha256", ""))
        if ((digest and recorded_digest == digest)
                or (not recorded_digest and key == legacy_key)
                or (not path.exists() and key == known_key)):
            if digest and not recorded_digest and key == legacy_key:
                record["content_sha256"] = digest
                previous = decisions["content_outcomes"].get(digest)
                if not previous or _decision_order(record) > _decision_order(previous):
                    decisions["content_outcomes"][digest] = record
                decisions["_needs_migration"] = True
            candidates.append((key, record))
    indexed = decisions["_path_index"].get(os.path.realpath(str(path)))
    if indexed and digest and indexed[1].get("content_sha256") == digest:
        candidates.append(indexed)
    value = decisions["content_outcomes"].get(digest)
    if digest and isinstance(value, dict):
        candidates.append((f"content:{digest}", value))
    return max(candidates, key=lambda pair: _decision_order(pair[1]), default=None)


def _store_decision(
    decisions: dict,
    *,
    key: str,
    path: Path,
    action: str,
    content_sha256: str = "",
    model_signature: str = "",
    **extra: object,
) -> None:
    _index_decisions(decisions)
    digest = content_sha256 or item_content_sha256(path)
    existing = decisions["content_outcomes"].get(digest, decisions["items"].get(key, {}))
    rejected_people = {
        str(value) for value in existing.get("rejected_people", []) if str(value)
    }
    rejected_people.update(
        str(value) for value in extra.pop("rejected_people", []) if str(value)
    )
    stamp = max(time.time_ns(), int(decisions.get("_last_decision_ns", 0)) + 1)
    decisions["_last_decision_ns"] = stamp
    record = {
        "rejected_people": sorted(rejected_people, key=str.casefold),
        "action": action,
        "outcome_state": str(
            extra.pop("outcome_state", ACTION_OUTCOME_STATES.get(action, "pending_review"))
        ),
        "path": str(path),
        "filename": path.name,
        "content_sha256": digest,
        "model_signature": model_signature,
        "decided_at": int(time.time()),
        "decided_at_ns": stamp,
        **extra,
    }
    decisions["items"][key] = record
    decisions.setdefault("_path_index", {})[os.path.realpath(str(path))] = (key, record)
    if digest:
        decisions.setdefault("content_outcomes", {})[digest] = dict(record)
        decisions.setdefault("content_rechecks", {}).pop(digest, None)


def record_decision(
    decisions: dict,
    item: UnknownItem,
    action: str,
    **extra: object,
) -> None:
    _store_decision(
        decisions,
        key=item.key,
        path=item.path,
        action=action,
        **extra,
    )


def record_path_decision(
    decisions: dict,
    *,
    key: str,
    path: Path,
    action: str,
    **extra: object,
) -> None:
    _store_decision(
        decisions,
        key=key,
        path=path,
        action=action,
        **extra,
    )


def record_rejection(decisions: dict, item: UnknownItem, person: str) -> None:
    resolved = decision_for_path(decisions, item.path)
    existing = dict(resolved[1] if resolved else {})
    rejected = {str(value) for value in existing.get("rejected_people", []) if str(value)}
    rejected.add(person)
    existing.update({
        "path": str(item.path),
        "filename": item.path.name,
        "content_sha256": item_content_sha256(item.path),
        "rejected_people": sorted(rejected, key=str.casefold),
        "last_rejected_at": int(time.time()),
    })
    decisions["items"][item.key] = existing
    digest = existing["content_sha256"]
    if digest:
        decisions["content_outcomes"][digest] = existing
    decisions["_path_index"][os.path.realpath(str(item.path))] = (item.key, existing)


def resolved_item(
    decisions: dict,
    path: Path,
    *,
    model_signature: str = "",
) -> bool:
    resolved = decision_for_path(decisions, path)
    if resolved is None:
        return False
    _key, record = resolved
    action = str(record.get("action", ""))
    if action in MOVING_ACTIONS:
        # A move/confirmation outcome cannot be complete while the source is
        # physically back in unknown_identity. Reconciliation owns that replay.
        return False
    if action in RETAINED_ACTIONS:
        return True
    return False


def needs_silent_recheck(decisions: dict, path: Path, model_signature: str) -> bool:
    resolved = decision_for_path(decisions, path)
    if not resolved or not model_signature:
        return False
    record = resolved[1]
    if (record.get("action") != "keep_unknown"
            or record.get("model_signature") == model_signature):
        return False
    checked = decisions.get("content_rechecks", {}).get(item_content_sha256(path), {})
    if checked.get("model_signature") != model_signature:
        return True
    checked_at = int(checked.get("checked_at_ns", 0)) or int(checked.get("checked_at", 0)) * 1_000_000_000
    # Backfilling another legacy copy may reveal a later old Keep Unknown
    # choice. A check performed after that choice still covers the same bytes.
    return (checked.get("decision_order") != list(_decision_order(record))
            and _decision_order(record)[1] > checked_at)


def mark_silent_recheck(decisions: dict, path: Path, model_signature: str) -> None:
    resolved = decision_for_path(decisions, path)
    if not resolved or resolved[1].get("action") != "keep_unknown":
        return
    digest = item_content_sha256(path)
    if digest:
        decisions.setdefault("content_rechecks", {})[digest] = {
            "model_signature": model_signature,
            "decision_order": list(_decision_order(resolved[1])),
            "checked_at": int(time.time()),
            "checked_at_ns": time.time_ns(),
        }


def automatic_check_allowed(decisions: dict, path: Path, model_signature: str) -> bool:
    resolved = decision_for_path(decisions, path)
    if not resolved or not resolved[1].get("action"):
        return True
    return needs_silent_recheck(decisions, path, model_signature)


def decision_for_item(decisions: dict, item: UnknownItem) -> dict:
    resolved = decision_for_path(decisions, item.path, known_key=item.key)
    return resolved[1] if resolved else {}


def collect_items(
    files: list[Path],
    identity_db: sort_photos.IdentityDB,
    *,
    workers: int,
    primary_det_size: int,
    fallback_det_size: int,
    hard_negatives: dict[str, list[np.ndarray]],
    review_prototypes: dict[str, list[np.ndarray]] | None = None,
    secondary_matcher: secondary_identity_matcher.SecondaryMatcher | None = None,
    progress_callback: Callable[[int, int, Counter], None] | None = None,
) -> tuple[list[UnknownItem], Counter, list[UnsupportedUnknown]]:
    items: list[UnknownItem] = []
    stats: Counter = Counter()
    unsupported: list[UnsupportedUnknown] = []
    compiled = identity_profiles.CompiledProfiles(
        identity_db.identities, review_prototypes or identity_db.prototypes,
        pose_prototypes=identity_db.pose_prototypes,
        appearance_prototypes=identity_db.appearance_prototypes,
        appearance_era_cutoffs=identity_db.appearance_era_cutoffs, hard_negatives=hard_negatives)
    results = recover_no_usable_faces.iter_detection_results(
        files,
        max(1, workers),
        max(320, primary_det_size),
        max(320, fallback_det_size),
        index_path=sort_photos.analysis_index_file(),
        cache_stats=stats,
    )
    for processed, (_index, path, status, faces, detector_error) in enumerate(results, 1):
        if detector_error:
            stats["errors"] += 1
            unsupported.append(UnsupportedUnknown(
                key=item_key(path),
                path=path,
                queue_kind="technical_review",
                detector_status=status or "processing_failed",
                error=str(detector_error),
            ))
        elif len(faces) == 1:
            items.append(automatic_review_item(
                item_key(path), path, faces[0], identity_db, hard_negatives,
                review_prototypes, secondary_matcher,
                compiled=compiled,
            ))
            stats["single_face"] += 1
        elif len(faces) > 1:
            stats["multi_face"] += 1
            unsupported.append(UnsupportedUnknown(
                key=item_key(path),
                path=path,
                queue_kind="multi_face_review",
                detector_status=status or "multiple_faces",
            ))
        elif status.startswith("face_quality_review:"):
            stats["quality_review"] += 1
            unsupported.append(UnsupportedUnknown(
                key=item_key(path),
                path=path,
                queue_kind="face_quality_review",
                detector_status=status,
            ))
        else:
            stats["no_face"] += 1
            unsupported.append(UnsupportedUnknown(
                key=item_key(path),
                path=path,
                queue_kind="no_usable_face",
                detector_status=status or "no_usable_face",
            ))
        if processed == 1 or processed % 50 == 0 or processed == len(files):
            print(
                f"Analyzing unknowns {processed}/{len(files)}: "
                f"single-face={stats['single_face']} multi-face={stats['multi_face']} "
                f"no-face={stats['no_face']} errors={stats['errors']}",
                flush=True,
            )
        if progress_callback is not None and (
            processed == 1 or processed % 10 == 0 or processed == len(files)
        ):
            progress_callback(processed, len(files), Counter(stats))
    return items, stats, unsupported


def _path_is_under(path_value: object, root: Path) -> bool:
    try:
        Path(str(path_value)).expanduser().resolve(strict=False).relative_to(root)
    except (OSError, ValueError, TypeError):
        return False
    return True


def review_progress(state: dict, decisions: dict | None = None) -> dict:
    decisions = decisions or load_decisions(state["decisions_path"])
    unknown_root = Path(state["unknown_root"]).resolve(strict=False)
    records = [
        value for value in decisions.get("items", {}).values()
        if isinstance(value, dict)
        and value.get("action") in RESOLVED_ACTIONS
        and _path_is_under(value.get("path", ""), unknown_root)
    ]
    actions = Counter(str(value.get("action", "")) for value in records)
    pending_raw = sum(
        not resolved_item(
            decisions,
            path,
            model_signature=str(state.get("model_signature", "")),
        )
        for path in recover_no_usable_faces.iter_images(unknown_root)
    )
    temporarily_skipped = len(state.get("temporarily_skipped", set()))
    pending = max(0, pending_raw - temporarily_skipped)
    confirmed = actions["confirmed"]
    kept_unknown = actions["keep_unknown"]
    junk = actions["moved_to_junk"]
    deferred = temporarily_skipped + actions["ignored"] + sum(
        actions[action] for action in RESOLVED_ACTIONS if action.startswith("routed_")
    )
    reviewed = confirmed + kept_unknown + junk + deferred
    return {
        "total": reviewed + pending,
        "reviewed": reviewed,
        "pending": pending,
        "confirmed": confirmed,
        "unknown": kept_unknown,
        "junk": junk,
        "deferred": deferred,
        "temporarily_skipped": temporarily_skipped,
    }


def save_session_state(state: dict, *, status: str = "reviewing") -> None:
    path = Path(state["session_path"])
    progress = review_progress(state)
    payload = {
        "version": 1,
        "status": status,
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "batch_number": int(state.get("batch_number", 0)),
        "batch_limit": int(state.get("batch_limit", 500)),
        "progress": progress,
        "summary": dict(state.get("summary", {})),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _remove_cache_sources(state: dict, source_paths: set[str]) -> None:
    if state.get("capture_cache_delta") and source_paths:
        state.setdefault("_cache_delta_removed_sources", set()).update(source_paths)
    cache = state.get("cache")
    if cache is None or not source_paths:
        return
    original_signatures = len(cache.file_signatures)
    original_faces = len(cache.faces)
    cache.file_signatures = {
        path: signature for path, signature in cache.file_signatures.items()
        if os.path.realpath(path) not in source_paths
    }
    cache.faces = [
        face for face in cache.faces
        if os.path.realpath(face.src_str) not in source_paths
    ]
    if (
        len(cache.file_signatures) != original_signatures
        or len(cache.faces) != original_faces
    ):
        state["cache_dirty"] = True


def route_unsupported_unknowns(
    state: dict,
    records: list[UnsupportedUnknown],
) -> Counter:
    """Move unsupported items to their recoverable specialist queues."""
    counts: Counter = Counter()
    if not records:
        return counts
    review_root = Path(state["unassigned_root"])
    destinations = {
        "multi_face_review": review_root / "multi_face_review",
        "no_usable_face": review_root / "no_usable_face",
        "face_quality_review": review_root / "face_quality_review",
        "technical_review": review_root / "processing_failed",
    }
    actions = {
        "multi_face_review": "routed_multi_face",
        "no_usable_face": "routed_no_usable_face",
        "face_quality_review": "routed_face_quality_review",
        "technical_review": "routed_technical_review",
    }
    decisions = load_decisions(state["decisions_path"])
    moved_sources: set[str] = set()
    for record in records:
        if not record.path.is_file():
            continue
        source_hash = item_content_sha256(record.path)
        source_text = os.path.realpath(str(record.path))
        destination = recover_no_usable_faces.move_review_file(
            record.path,
            destinations[record.queue_kind],
            status=f"unknown_review_{record.queue_kind}",
            extra={
                "review_action": "route_unsupported",
                "detector_status": record.detector_status,
                "detector_error": record.error,
            },
            dry_run=False,
        )
        moved_sources.add(source_text)
        record_path_decision(
            decisions,
            key=record.key,
            path=record.path,
            action=actions[record.queue_kind],
            content_sha256=source_hash,
            model_signature=str(state.get("model_signature", "")),
            destination=str(destination),
            detector_status=record.detector_status,
            detector_error=record.error,
        )
        save_decisions(state["decisions_path"], decisions)
        counts[record.queue_kind] += 1
    _remove_cache_sources(state, moved_sources)
    return counts


def _verified_organized_destination(
    state: dict,
    record: dict,
    digest: str,
) -> Path | None:
    people_root = Path(state["people_root"]).resolve()
    destination_text = str(record.get("destination", ""))
    if not destination_text:
        return None
    for candidate in [Path(destination_text).expanduser()]:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(people_root)
            if resolved.is_file() and sort_photos.sha256_file(resolved) == digest:
                return resolved
        except (OSError, ValueError):
            continue
    return None


def reconcile_unknown_queue(state: dict, *, dry_run: bool = False) -> Counter:
    """Repair replayed outcomes before any unknown-review batch is selected.

    Every move remains recoverable and ledgered. A prior confirmation is only
    trusted when its organized destination still exists and has the same exact
    bytes as the replayed source.
    """
    decisions = load_decisions(state["decisions_path"])
    current_model_signature = str(state.get("model_signature", ""))
    unknown_root = Path(state["unknown_root"])
    specialist_roots = {
        "routed_multi_face": Path(state["unassigned_root"]) / "multi_face_review",
        "routed_no_usable_face": Path(state["unassigned_root"]) / "no_usable_face",
        "routed_face_quality_review": Path(state["unassigned_root"]) / "face_quality_review",
        "routed_technical_review": Path(state["unassigned_root"]) / "processing_failed",
    }
    counts: Counter = Counter()
    moved_sources: set[str] = set()
    dirty = bool(decisions.get("_needs_migration"))

    for path in recover_no_usable_faces.iter_images(unknown_root):
        resolved = decision_for_path(decisions, path)
        if resolved is None:
            counts["pending"] += 1
            continue
        _record_key, record = resolved
        action = str(record.get("action", ""))
        digest = item_content_sha256(path)
        current_key = item_key(path)

        if action in RETAINED_ACTIONS:
            counts[f"retained_{action}"] += 1
            if needs_silent_recheck(decisions, path, current_model_signature):
                counts["silent_recheck_available"] += 1
            continue

        destination_root: Path | None = None
        destination: Path | None = None
        outcome_state = ""
        if action == "confirmed":
            destination = _verified_organized_destination(state, record, digest)
            if destination is None:
                if dry_run:
                    counts["would_requeue_unverified_confirmation"] += 1
                    continue
                decisions.setdefault("content_outcomes", {}).pop(digest, None)
                _store_decision(
                    decisions,
                    key=current_key,
                    path=path,
                    action="",
                    content_sha256=digest,
                    model_signature=current_model_signature,
                    outcome_state="requeued_unverified_confirmation",
                    previous_action=action,
                    previous_destination=str(record.get("destination", "")),
                )
                counts["requeued_unverified_confirmation"] += 1
                dirty = True
                continue
            destination_root = Path(state["replay_dir"])
            outcome_state = "verified_organized_replay"
        elif action in specialist_roots:
            destination_root = specialist_roots[action]
            outcome_state = "reconciled_specialist_replay"
        elif action == "moved_to_junk":
            destination_root = Path(state["junk_dir"])
            outcome_state = "reconciled_junk_replay"
        elif action:
            counts["pending"] += 1
            continue
        else:
            counts["pending"] += 1
            continue

        if dry_run:
            counts[f"would_reconcile_{action}"] += 1
            continue
        try:
            source_text = os.path.realpath(str(path))
            replay_destination = recover_no_usable_faces.move_review_file(
                path,
                destination_root,
                status=f"unknown_review_reconcile_{action}",
                extra={
                    "review_action": "reconcile_replayed_outcome",
                    "previous_action": action,
                    "verified_destination": str(destination or ""),
                    "content_sha256": digest,
                },
                dry_run=False,
            )
            moved_sources.add(source_text)
            record_path_decision(
                decisions,
                key=current_key,
                path=path,
                action=action,
                content_sha256=digest,
                model_signature=current_model_signature,
                outcome_state=outcome_state,
                destination=str(destination or replay_destination),
                replay_destination=str(replay_destination),
                reconciled_at=int(time.time()),
                person=str(record.get("person", "")),
                corrected_candidate=str(record.get("corrected_candidate", "")),
                rejected_people=list(record.get("rejected_people", [])),
                trusted_confirmation=bool(record.get("trusted_confirmation", False)),
            )
            counts[f"reconciled_{action}"] += 1
            dirty = True
        except Exception as error:  # noqa: BLE001
            counts["failed"] += 1
            counts[f"failed_{action}"] += 1
            print(f"Reconciliation failed for {path.name}: {error}", flush=True)
        if dirty and sum(counts.values()) % 25 == 0:
            save_decisions(state["decisions_path"], decisions)

    if (dirty or decisions.get("_needs_migration")) and not dry_run:
        save_decisions(state["decisions_path"], decisions)
    _remove_cache_sources(state, moved_sources)
    return counts


def write_review_reports(state: dict) -> None:
    output_dir = Path(state["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    decisions = load_decisions(state["decisions_path"])
    summary = dict(state.get("summary", {}))
    summary["progress"] = review_progress(state, decisions)
    state["summary"] = summary
    (output_dir / "latest_unknown_identity_review.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "unknown_identity_review.html").write_text(
        render_html(
            state.get("clusters", []),
            decisions,
            state["identity_names"],
            summary,
            interactive=False,
        ),
        encoding="utf-8",
    )
    save_session_state(state)


def load_next_batch(state: dict) -> dict:
    """Load the next reviewable batch and route blockers out of this queue."""
    if state.get("replay_dir"):
        reconcile_unknown_queue(state)
    decisions = load_decisions(state["decisions_path"])
    limit = max(1, int(state["batch_limit"]))
    accumulated_stats: Counter = Counter()
    routed_stats: Counter = Counter()
    items: list[UnknownItem] = []
    batch_files: list[Path] = []

    while not items:
        all_files = recover_no_usable_faces.iter_images(state["unknown_root"])
        pending_files = [
            path for path in all_files
            if not resolved_item(
                decisions,
                path,
                model_signature=str(state.get("model_signature", "")),
            )
            and item_key(path) not in state.get("temporarily_skipped", set())
        ]
        if not pending_files:
            state["clusters"] = []
            state["items_by_key"] = {}
            progress = review_progress(state, decisions)
            state["summary"] = {
                "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "queue_files": progress["total"],
                "pending_files": progress["pending"],
                "batch_files": 0,
                "clusters": 0,
                "progress": progress,
            }
            write_review_reports(state)
            return {"complete": progress["pending"] == 0, "summary": state["summary"]}

        batch_files = pending_files[:limit]
        batch_items, stats, unsupported = collect_items(
            batch_files,
            state["identity_db"],
            workers=state["workers"],
            primary_det_size=state["primary_det_size"],
            fallback_det_size=state["fallback_det_size"],
            hard_negatives=state["hard_negatives"],
            review_prototypes=state.get("review_prototypes"),
            secondary_matcher=state.get("secondary_matcher"),
            progress_callback=state.get("batch_progress_callback"),
        )
        accumulated_stats.update(stats)
        if state.get("route_unsupported", True):
            routed_stats.update(route_unsupported_unknowns(state, unsupported))
        decisions = load_decisions(state["decisions_path"])
        items = batch_items
        if unsupported and not state.get("route_unsupported", True):
            break
        if not unsupported and not items:
            break

    clusters = cluster_items(
        items,
        state["identity_db"],
        eps=float(state["cluster_eps"]),
        hard_negatives=state["hard_negatives"],
        review_prototypes=state.get("review_prototypes"),
    )
    state["clusters"] = clusters
    state["items_by_key"] = {item.key: item for item in items}
    auto_result = apply_automatic_review(
        state,
        clusters,
        preview=bool(state.get("auto_review_preview")),
    )
    if int(auto_result["confirmed"]):
        if state.get("cache_dirty"):
            sort_photos.save_cache(state["cache"])
            state["cache_dirty"] = False
        decisions = load_decisions(state["decisions_path"])
        items = [
            item for item in items
            if item.path.is_file()
            and not resolved_item(
                decisions,
                item.path,
                model_signature=str(state.get("model_signature", "")),
            )
        ]
        clusters = cluster_items(
            items,
            state["identity_db"],
            eps=float(state["cluster_eps"]),
            hard_negatives=state["hard_negatives"],
            review_prototypes=state.get("review_prototypes"),
        )
    state["batch_number"] = int(state.get("batch_number", 0)) + 1
    state["clusters"] = clusters
    state["items_by_key"] = {item.key: item for item in items}
    progress = review_progress(state, decisions)
    state["summary"] = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "queue_files": progress["total"],
        "pending_files": progress["pending"],
        "batch_files": len(batch_files),
        "single_face": accumulated_stats["single_face"],
        "multi_face": accumulated_stats["multi_face"],
        "no_face": accumulated_stats["no_face"],
        "quality_review": accumulated_stats["quality_review"],
        "errors": accumulated_stats["errors"],
        "sqlite_hits": accumulated_stats["sqlite_hits"],
        "detected": accumulated_stats["detected"],
        "clusters": len(clusters),
        "routed": dict(routed_stats),
        "auto_review": auto_result,
        "auto_sweep": dict(state.get("auto_sweep_summary", {})),
        "batch_number": state["batch_number"],
        "progress": progress,
    }
    write_review_reports(state)
    print(
        f"Batch {state['batch_number']} ready: {len(items)} reviewable face(s), "
        f"{len(clusters)} cluster(s), {int(auto_result['confirmed'])} safely auto-confirmed, "
        f"{sum(routed_stats.values())} routed to specialist queues.",
        flush=True,
    )
    return {"complete": False, "summary": state["summary"]}


def _run_automatic_sweep_batch(state: dict, files: list[Path]) -> dict[str, object]:
    decisions = load_decisions(state["decisions_path"])
    signature = str(state.get("model_signature", ""))
    files = [path for path in files if automatic_check_allowed(decisions, path, signature)]
    retained = {
        str(path) for path in files if resolved_item(decisions, path)
    }
    items, _stats, unsupported = collect_items(
        files,
        state["identity_db"],
        workers=state["workers"],
        primary_det_size=state["primary_det_size"],
        fallback_det_size=state["fallback_det_size"],
        hard_negatives=state["hard_negatives"],
        review_prototypes=state.get("review_prototypes"),
        secondary_matcher=state.get("secondary_matcher"),
    )
    routed = Counter()
    if state.get("route_unsupported", True):
        routed.update(route_unsupported_unknowns(
            state, [record for record in unsupported if str(record.path) not in retained]
        ))
    clusters = cluster_items(
        items,
        state["identity_db"],
        eps=float(state["cluster_eps"]),
        hard_negatives=state["hard_negatives"],
        review_prototypes=state.get("review_prototypes"),
    )
    state["clusters"] = clusters
    state["items_by_key"] = {item.key: item for item in items}
    batch_result = apply_automatic_review(state, clusters)
    decisions = load_decisions(state["decisions_path"])
    checked_paths = [item.path for item in items] + [
        record.path for record in unsupported if record.queue_kind != "technical_review"
    ]
    if not int(batch_result["failed"]):
        for path in checked_paths:
            if str(path) in retained and path.is_file():
                mark_silent_recheck(decisions, path, signature)
        if retained:
            save_decisions(state["decisions_path"], decisions)
    if state.get("cache_dirty") and not state.get("capture_cache_delta"):
        sort_photos.save_cache(state["cache"])
        state["cache_dirty"] = False
    secondary = state.get("secondary_matcher")
    if secondary is not None:
        secondary.flush()
        secondary.app = None
    destinations = [
        {
            "person_key": str(person_key),
            "sha256": str(digest),
            "path": str(destination),
        }
        for (person_key, digest), destination
        in state.get("destinations_by_hash", {}).items()
        if Path(destination).is_file()
    ]
    state["clusters"] = []
    state["items_by_key"] = {}
    return {
        "scanned": len(files),
        "confirmed": int(batch_result["confirmed"]),
        "failed": int(batch_result["failed"]),
        "completed_content_keys": [
            item_content_sha256(path) for path in checked_paths if path.is_file()
        ] if not int(batch_result["failed"]) else [],
        "routed": dict(routed),
        "destinations": destinations,
    }


def _run_isolated_automatic_sweep_batch(
    state: dict,
    files: list[Path],
) -> dict[str, object]:
    """Run one verifier chunk in a clean spawned Python process."""
    result_path = Path(state["output_dir"]) / f".auto_sweep.{uuid.uuid4().hex}.json"
    task_path = result_path.with_suffix(".task.json")
    task = {
        "result_path": str(result_path),
        "files": [str(path) for path in files],
        "output_dir": str(state["output_dir"]),
        "decisions_path": str(state["decisions_path"]),
        "unknown_root": str(state["unknown_root"]),
        "unassigned_root": str(state["unassigned_root"]),
        "people_root": str(state["people_root"]),
        "review_dir": str(state["review_dir"]),
        "junk_dir": str(state["junk_dir"]),
        "analysis_index": str(state["analysis_index"]),
        "workers": int(state["workers"]),
        "primary_det_size": int(state["primary_det_size"]),
        "fallback_det_size": int(state["fallback_det_size"]),
        "cluster_eps": float(state["cluster_eps"]),
        "route_unsupported": bool(state.get("route_unsupported", True)),
        "known_destinations": [
            [str(person_key), str(digest), str(destination)]
            for (person_key, digest), destination
            in state.get("destinations_by_hash", {}).items()
            if Path(destination).is_file()
        ],
    }
    temporary_task = task_path.with_suffix(task_path.suffix + ".tmp")
    temporary_task.write_text(
        json.dumps(task, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_task.replace(task_path)
    completed = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--auto-sweep-worker", str(task_path)],
        check=False,
    )
    task_path.unlink(missing_ok=True)
    try:
        payload = json.loads(result_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        payload = {}
    finally:
        result_path.unlink(missing_ok=True)
    if completed.returncode != 0 or payload.get("error"):
        raise RuntimeError(
            f"isolated automatic review batch failed (exit {completed.returncode}): "
            f"{payload.get('error', 'no worker result')}"
        )

    for record in payload.get("destinations", []):
        person_key = str(record.get("person_key", ""))
        digest = str(record.get("sha256", ""))
        destination = Path(str(record.get("path", "")))
        if not person_key or not digest or not destination.is_file():
            continue
        state.setdefault("destinations_by_hash", {})[(person_key, digest)] = destination
        try:
            person = destination.resolve().relative_to(
                Path(state["people_root"]).resolve()
            ).parts[0]
        except (OSError, ValueError, IndexError):
            continue
        state["existing_hashes"][person].add(digest)
    state["next_indexes"] = {}
    secondary_db = secondary_identity_matcher.load()
    if secondary_db is not None:
        state["secondary_matcher"] = secondary_identity_matcher.SecondaryMatcher(
            secondary_db
        )
    return payload


def apply_pending_automatic_cache_deltas(state: dict) -> dict[str, int]:
    """Merge durable worker cache deltas in one atomic cache save."""
    delta_paths = sorted(Path(state["output_dir"]).glob(".auto_sweep.cache.*.pkl"))
    if not delta_paths:
        return {"files": 0, "removed": 0, "entries": 0}
    cache = state.get("cache") or sort_photos.load_cache()
    removed_sources: set[str] = set()
    entries: list[tuple[Path, sort_photos.CachedFace, str]] = []
    valid_paths: list[Path] = []
    for path in delta_paths:
        try:
            with path.open("rb") as handle:
                payload = pickle.load(handle)
            removed_sources.update(str(value) for value in payload.get("removed_sources", []))
            entries.extend(payload.get("entries", []))
            valid_paths.append(path)
        except (OSError, ValueError, TypeError, pickle.PickleError):
            continue
    if not valid_paths:
        state["cache"] = cache
        return {"files": 0, "removed": 0, "entries": 0}

    original_signatures = len(cache.file_signatures)
    cache.file_signatures = {
        path: signature for path, signature in cache.file_signatures.items()
        if os.path.realpath(path) not in removed_sources
    }
    cache.faces = [
        face for face in cache.faces
        if os.path.realpath(face.src_str) not in removed_sources
    ]
    recovered = recover_no_usable_faces.merge_recovered_faces_into_cache(cache, entries)
    sort_photos.save_cache(cache)
    for path in valid_paths:
        path.unlink(missing_ok=True)
    state["cache"] = cache
    return {
        "files": len(valid_paths),
        "removed": max(0, original_signatures - len(cache.file_signatures)),
        "entries": len(recovered),
    }


def run_automatic_sweep(state: dict) -> dict[str, object]:
    """Scan every pending file once and file all benchmark-gated safe matches.

    The interactive page still displays a bounded batch, but safe candidates
    later in a large queue should not require repeatedly pressing Next Batch.
    Every automatic decision remains individually persisted, so interruption is
    resumable and already completed files are never repeated.
    """
    result: dict[str, object] = {
        "enabled": bool(state.get("auto_review_allowed")),
        "scanned": 0,
        "confirmed": 0,
        "failed": 0,
        "routed": {},
        "batches": 0,
    }
    if not state.get("auto_review_allowed") or state.get("auto_review_preview"):
        return result

    isolated_batches = bool(state.get("isolate_auto_sweep_batches"))
    if isolated_batches:
        recovered_deltas = apply_pending_automatic_cache_deltas(state)
        if recovered_deltas["files"]:
            print(
                f"Recovered {recovered_deltas['files']} interrupted sweep cache delta(s).",
                flush=True,
            )

    decisions = load_decisions(state["decisions_path"])
    all_pending_files = [
        path for path in recover_no_usable_faces.iter_images(state["unknown_root"])
        if automatic_check_allowed(decisions, path, str(state.get("model_signature", "")))
        and item_key(path) not in state.get("temporarily_skipped", set())
    ]
    sweep_state_path = (
        Path(state["output_dir"]) / "unknown_auto_sweep_state.json"
        if state.get("output_dir") else None
    )
    if sweep_state_path is None:
        sweep_state = {}
    else:
        try:
            sweep_state = json.loads(sweep_state_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            sweep_state = {}
    gate_signature = str(state.get("auto_review_gate", {}).get("signature", ""))
    model_signature = str(state.get("model_signature", ""))
    if (
        sweep_state.get("version") != AUTO_SWEEP_STATE_VERSION
        or sweep_state.get("gate_signature") != gate_signature
        or sweep_state.get("model_signature") != model_signature
    ):
        sweep_state = {
            "version": AUTO_SWEEP_STATE_VERSION,
            "gate_signature": gate_signature,
            "model_signature": model_signature,
            "completed_keys": [],
        }
    completed_keys = {str(value) for value in sweep_state.get("completed_keys", [])}
    pending_files = []
    seen_content = set()
    for path in all_pending_files:
        digest = item_content_sha256(path)
        # Byte-identical copies add no independent identity evidence.
        if digest and digest not in completed_keys and digest not in seen_content:
            pending_files.append(path)
            seen_content.add(digest)
    if not pending_files:
        previous = sweep_state.get("last_summary", {})
        if isinstance(previous, dict) and previous:
            result.update(previous)
            result["cached"] = True
        return result
    batch_size = min(
        AUTO_SWEEP_BATCH_SIZE,
        max(1, int(state.get("batch_limit", 500))),
    )
    routed_total: Counter = Counter()
    batch_total = max(1, (len(pending_files) + batch_size - 1) // batch_size)
    for offset in range(0, len(pending_files), batch_size):
        files = [path for path in pending_files[offset:offset + batch_size] if path.is_file()]
        if not files:
            continue
        batch_result = (
            _run_isolated_automatic_sweep_batch(state, files)
            if state.get("isolate_auto_sweep_batches")
            else _run_automatic_sweep_batch(state, files)
        )
        routed_total.update(batch_result.get("routed", {}))
        result["scanned"] = int(result["scanned"]) + len(files)
        result["confirmed"] = int(result["confirmed"]) + int(batch_result["confirmed"])
        result["failed"] = int(result["failed"]) + int(batch_result["failed"])
        result["batches"] = int(result["batches"]) + 1
        completed_keys.update(batch_result.get("completed_content_keys", []))
        sweep_state.update({
            "version": AUTO_SWEEP_STATE_VERSION,
            "gate_signature": gate_signature,
            "completed_keys": sorted(completed_keys),
            "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "status": "in_progress",
        })
        if sweep_state_path is not None:
            temporary_state = sweep_state_path.with_suffix(sweep_state_path.suffix + ".tmp")
            temporary_state.write_text(
                json.dumps(sweep_state, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            temporary_state.replace(sweep_state_path)
        print(
            f"Safe auto-sweep {int(result['batches'])}/{batch_total}: "
            f"scanned {int(result['scanned'])}/{len(pending_files)}, "
            f"filed {int(result['confirmed'])}, failed {int(result['failed'])}",
            flush=True,
        )
        del files
        gc.collect()

    state["clusters"] = []
    state["items_by_key"] = {}
    if isolated_batches:
        merged_deltas = apply_pending_automatic_cache_deltas(state)
        if merged_deltas["files"]:
            print(
                "Merged sweep cache updates once: "
                f"{merged_deltas['entries']} organized face(s), "
                f"{merged_deltas['removed']} stale source entry(s) removed.",
                flush=True,
            )
        elif state.get("cache") is None:
            state["cache"] = sort_photos.load_cache()
    result["routed"] = dict(routed_total)
    sweep_state.update({
        "completed_keys": sorted(completed_keys),
        "updated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "status": "complete",
        "last_summary": dict(result),
    })
    if sweep_state_path is not None:
        temporary_state = sweep_state_path.with_suffix(sweep_state_path.suffix + ".tmp")
        temporary_state.write_text(
            json.dumps(sweep_state, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary_state.replace(sweep_state_path)
    return result


def cluster_items(
    items: list[UnknownItem],
    identity_db: sort_photos.IdentityDB,
    *,
    eps: float = 0.30,
    maximum_items: int = MAX_CLUSTER_ACTION_ITEMS,
    hard_negatives: dict[str, list[np.ndarray]] | None = None,
    review_prototypes: dict[str, list[np.ndarray]] | None = None,
) -> list[UnknownCluster]:
    if not items:
        return []
    matrix = identity_profiles.normalize_matrix([item.face.embedding for item in items])
    if len(items) == 1:
        labels = np.asarray([-1], dtype=int)
    else:
        labels = DBSCAN(
            eps=float(eps), min_samples=2, metric="cosine", n_jobs=1
        ).fit_predict(matrix)

    grouped: dict[str, list[tuple[UnknownItem, np.ndarray]]] = {}
    for index, (item, embedding, label) in enumerate(zip(items, matrix, labels)):
        group_key = f"cluster:{int(label)}" if int(label) >= 0 else f"single:{index}"
        grouped.setdefault(group_key, []).append((item, embedding))

    clusters: list[UnknownCluster] = []
    chunk_size = max(1, int(maximum_items))
    for group in grouped.values():
        ordered = sorted(
            group,
            key=lambda pair: (-float(pair[0].face.quality), pair[0].path.name.casefold()),
        )
        for offset in range(0, len(ordered), chunk_size):
            chunk = ordered[offset:offset + chunk_size]
            chunk_items = tuple(pair[0] for pair in chunk)
            embeddings = np.stack([pair[1] for pair in chunk])
            qualities = np.asarray(
                [max(0.05, float(item.face.quality)) for item in chunk_items],
                dtype=np.float32,
            )
            qualities /= max(float(qualities.sum()), 1e-9)
            centroid = identity_profiles.normalize_vector(
                (embeddings * qualities[:, None]).sum(axis=0)
            )
            candidates = tuple(identity_profiles.rank_candidates(
                centroid,
                identity_db.identities,
                review_prototypes or identity_db.prototypes,
                hard_negatives=hard_negatives,
            )[:3])
            cohesion = float(np.max(1.0 - embeddings @ centroid))
            key_material = "\n".join(sorted(item.key for item in chunk_items))
            cluster_key = hashlib.sha256(key_material.encode("ascii")).hexdigest()[:20]
            clusters.append(UnknownCluster(
                key=cluster_key,
                items=chunk_items,
                candidates=candidates,
                cohesion=cohesion,
            ))
    return sorted(clusters, key=lambda cluster: (
        -len(cluster.items),
        cluster.candidates[0].distance if cluster.candidates else 1.0,
        cluster.key,
    ))


def canonical_person(
    value: str,
    identity_db: sort_photos.IdentityDB,
    *,
    allow_new: bool = False,
    people_root: Path | None = None,
) -> str:
    person = confirm_unknown_identity.canonical_person(value, identity_db)
    if person is not None:
        return person
    if not allow_new:
        raise ValueError(f"unknown person: {value}")

    normalized = " ".join(value.strip().split())
    if not normalized:
        raise ValueError("enter a person name before confirming")
    if len(normalized) > 100:
        raise ValueError("person name must be 100 characters or fewer")
    if (
        normalized in {".", ".."}
        or normalized.startswith((".", "_"))
        or any(character in normalized for character in "/\\:\0")
        or any(ord(character) < 32 for character in normalized)
    ):
        raise ValueError(f"unsafe person name: {value}")

    root = (people_root or recover_no_usable_faces.DEFAULT_PEOPLE).resolve()
    if root.is_dir():
        existing = next(
            (
                path.name
                for path in root.iterdir()
                if path.is_dir() and path.name.casefold() == normalized.casefold()
            ),
            None,
        )
        if existing:
            return existing
    destination = (root / normalized).resolve(strict=False)
    if destination.parent != root:
        raise ValueError(f"unsafe person name: {value}")
    return normalized


def apply_decision(
    state: dict,
    *,
    item_keys: list[str],
    action: str,
    person_value: str = "",
    trusted_confirmation: bool = True,
    decision_metadata: dict[str, object] | None = None,
) -> str:
    if action not in {
        "confirm", "keep_unknown", "ignore", "move_to_junk", "reject_candidate"
    }:
        raise ValueError("unknown review action")
    unique_keys = list(dict.fromkeys(key for key in item_keys if key))
    if not unique_keys:
        raise ValueError("no review items selected")
    if len(unique_keys) > MAX_BATCH_ACTION_ITEMS:
        raise ValueError(
            f"select no more than {MAX_BATCH_ACTION_ITEMS} images per review action"
        )
    items: list[UnknownItem] = []
    for key in unique_keys:
        item = state["items_by_key"].get(key)
        if item is None:
            raise ValueError("review item is no longer available")
        if not item.path.is_file():
            raise ValueError(f"file has moved: {item.path.name}")
        items.append(item)

    decisions = load_decisions(state["decisions_path"])
    if action == "reject_candidate":
        person = canonical_person(person_value, state["identity_db"])
        for item in items:
            identity_hard_negatives.record(
                state["hard_negatives_path"],
                person=person,
                source_path=item.path,
                face_index=item.face.face_index,
                embedding=item.face.embedding,
            )
            record_rejection(decisions, item, person)
        save_decisions(state["decisions_path"], decisions)
        state["hard_negatives"] = identity_hard_negatives.vectors_by_person(
            state["hard_negatives_path"]
        )
        state["model_signature"] = review_model_signature(
            state["identity_db"], Path(state["hard_negatives_path"])
        )
        refreshed_items, _stats, _unsupported = collect_items(
            [item.path for item in state["items_by_key"].values() if item.path.is_file()],
            state["identity_db"],
            workers=1,
            primary_det_size=1024,
            fallback_det_size=384,
            hard_negatives=state["hard_negatives"],
            review_prototypes=state.get("review_prototypes"),
            secondary_matcher=state.get("secondary_matcher"),
        )
        state["items_by_key"] = {item.key: item for item in refreshed_items}
        state["clusters"] = cluster_items(
            refreshed_items,
            state["identity_db"],
            hard_negatives=state["hard_negatives"],
            review_prototypes=state.get("review_prototypes"),
        )
        return f"Recorded: this face is not {person}. Candidate ranking updated."
    if action in {"keep_unknown", "ignore"}:
        saved_action = "keep_unknown" if action == "keep_unknown" else "ignored"
        for item in items:
            record_decision(
                decisions,
                item,
                saved_action,
                model_signature=str(state.get("model_signature", "")),
            )
        save_decisions(state["decisions_path"], decisions)
        return (
            f"Kept {len(items)} file(s) as unknown"
            if action == "keep_unknown"
            else f"Ignored {len(items)} file(s) for this review"
        )
    if action == "move_to_junk":
        moved_sources: set[str] = set()
        destinations: list[Path] = []
        for item in items:
            source_path = os.path.realpath(str(item.path))
            source_hash = item_content_sha256(item.path)
            destination = recover_no_usable_faces.move_review_file(
                item.path,
                state["junk_dir"],
                status="unknown_identity_junk_dashboard",
                extra={"review_action": "move_to_junk"},
                dry_run=False,
            )
            moved_sources.add(source_path)
            destinations.append(destination)
            record_decision(
                decisions,
                item,
                "moved_to_junk",
                content_sha256=source_hash,
                model_signature=str(state.get("model_signature", "")),
                destination=str(destination),
            )
            save_decisions(state["decisions_path"], decisions)
        _remove_cache_sources(state, moved_sources)
        return f"Moved {len(destinations)} file(s) to recoverable junk"

    person = canonical_person(
        person_value,
        state["identity_db"],
        allow_new=True,
        people_root=state.get("people_root"),
    )
    cache_entries: list[tuple[Path, sort_photos.CachedFace, str]] = []
    moved_sources: set[str] = set()
    destinations: list[Path] = []
    learned_rejections = 0
    for item in items:
        corrected_candidate = ""
        if (
            trusted_confirmation
            and item.candidates
            and item.candidates[0].name.casefold() != person.casefold()
        ):
            corrected_candidate = item.candidates[0].name
            if identity_hard_negatives.record(
                state["hard_negatives_path"],
                person=corrected_candidate,
                source_path=item.path,
                face_index=item.face.face_index,
                embedding=item.face.embedding,
                reason="explicit_user_correction_of_top_candidate",
            ):
                learned_rejections += 1
        source_hash = sort_photos.sha256_file(item.path)
        destination_key = (person.casefold(), source_hash)
        destination = state.setdefault("destinations_by_hash", {}).get(destination_key)
        already_present = destination is not None and Path(destination).is_file()
        if already_present:
            destination = Path(destination)
        else:
            destination, already_present = recover_no_usable_faces.copy_to_person(
                item.path,
                person,
                source_hash,
                state["existing_hashes"],
                state["next_indexes"],
                dry_run=False,
            )
        if already_present and destination is None:
            destination = confirm_unknown_identity.existing_person_path(
                person,
                source_hash,
                people_root=Path(state["people_root"]),
            )
        if destination is None:
            # A cached duplicate fingerprint can outlive its destination after
            # a manual move. Repair that stale entry and create a fresh,
            # verified organized copy rather than failing the whole batch.
            state["existing_hashes"][person].discard(source_hash)
            destination, already_present = recover_no_usable_faces.copy_to_person(
                item.path,
                person,
                source_hash,
                state["existing_hashes"],
                state["next_indexes"],
                dry_run=False,
            )
        if destination is None or not Path(destination).is_file():
            raise ValueError(
                f"could not create or locate a verified destination for {item.path.name}"
            )
        destination = Path(destination).resolve()
        state["destinations_by_hash"][destination_key] = destination
        if trusted_confirmation:
            identity_confirmations.record(
                state["confirmations_path"],
                person=person,
                organized_path=destination,
                content_sha256=source_hash,
                original_name=item.path.name,
                face=item.face,
                detector_version=sort_photos.config_fingerprint(),
            )
            evaluation_enrollment.enroll(
                source=destination,
                person=person,
                content_sha256=source_hash,
                pose_label=item.face.pose_label,
                quality=float(item.face.quality),
                path=Path(state.get(
                    "evaluation_path",
                    Path(state["confirmations_path"]).with_name(
                        "confirmed_unknown_identity_set.csv"
                    ),
                )),
            )
        recover_no_usable_faces.move_review_file(
            item.path,
            state["review_dir"],
            status="confirmed_identity_dashboard",
            extra={
                "person": person,
                "destination": str(destination),
                "content_sha256": source_hash,
            },
            dry_run=False,
        )
        moved_sources.add(os.path.realpath(str(item.path)))
        cache_entries.append((destination, item.face, person))
        destinations.append(destination)
        record_decision(
            decisions,
            item,
            "confirmed",
            content_sha256=source_hash,
            model_signature=str(state.get("model_signature", "")),
            person=person,
            destination=str(destination),
            trusted_confirmation=bool(trusted_confirmation),
            corrected_candidate=corrected_candidate,
            rejected_people=[corrected_candidate] if corrected_candidate else [],
            **dict(decision_metadata or {}),
        )
        save_decisions(state["decisions_path"], decisions)

    _remove_cache_sources(state, moved_sources)
    if state.get("capture_cache_delta") and cache_entries:
        state.setdefault("_cache_delta_entries", []).extend(cache_entries)
    recovered = recover_no_usable_faces.merge_recovered_faces_into_cache(
        state["cache"], cache_entries
    )
    identity_names = state.get("identity_names")
    if isinstance(identity_names, list) and not any(
        name.casefold() == person.casefold() for name in identity_names
    ):
        identity_names.append(person)
        identity_names.sort(key=str.casefold)
    persistence_warning = ""
    if recovered:
        state["cache_dirty"] = True
        if trusted_confirmation:
            state["identity_dirty"] = True
        paths = [Path(face.src_str) for face in recovered]
        detection_status = (
            "accepted_face_user_confirmation"
            if trusted_confirmation
            else "accepted_face_safe_automatic"
        )
        diagnostics = {str(path): detection_status for path in paths}
        try:
            sort_photos.persist_detection_batch(
                paths,
                recovered,
                diagnostics,
                state["analysis_index"],
            )
        except Exception as error:  # noqa: BLE001
            persistence_warning = f"; SQLite refresh deferred ({error})"
    if learned_rejections:
        state["hard_negatives"] = identity_hard_negatives.vectors_by_person(
            state["hard_negatives_path"]
        )
    learned_message = (
        f"; learned {learned_rejections} option-1 correction(s)"
        if learned_rejections
        else ""
    )
    return (
        f"Confirmed {len(destinations)} file(s) as {person}"
        f"{learned_message}{persistence_warning}"
    )


def apply_automatic_review(
    state: dict,
    clusters: list[UnknownCluster],
    *,
    preview: bool = False,
) -> dict[str, object]:
    matches = automatic_matches(
        clusters,
        state["identity_db"],
        require_secondary=True,
    )
    result: dict[str, object] = {
        "eligible": sum(len(match.item_keys) for match in matches),
        "confirmed": 0,
        "failed": 0,
        "lanes": dict(Counter(match.lane for match in matches)),
        "errors": [],
        "preview": bool(preview),
        "enabled": bool(state.get("auto_review_allowed")),
        "requested": bool(state.get("auto_review_requested")),
        "gate_message": str(state.get("auto_review_gate_message", "")),
        "samples": [
            {
                "person": match.person,
                "lane": match.lane,
                "files": len(match.item_keys),
                "distance": round(match.distance, 6),
                "margin": round(match.margin, 6),
            }
            for match in matches[:20]
        ],
    }
    if preview or not state.get("auto_review_allowed"):
        return result

    for match in matches:
        for offset in range(0, len(match.item_keys), MAX_BATCH_ACTION_ITEMS):
            decisions = load_decisions(state["decisions_path"])
            keys = list(match.item_keys[offset:offset + MAX_BATCH_ACTION_ITEMS])
            keys = [
                key for key in keys
                if key in state["items_by_key"]
                and state["items_by_key"][key].path.is_file()
                and automatic_check_allowed(
                    decisions, state["items_by_key"][key].path,
                    str(state.get("model_signature", "")),
                )
                and match.person.casefold() not in {
                    str(value).casefold() for value in decision_for_item(
                        decisions, state["items_by_key"][key]
                    ).get("rejected_people", [])
                }
            ]
            if not keys:
                continue
            try:
                apply_decision(
                    state,
                    item_keys=keys,
                    action="confirm",
                    person_value=match.person,
                    trusted_confirmation=False,
                    decision_metadata={
                        "automatic": True,
                        "auto_lane": match.lane,
                        "auto_support": match.support,
                        "auto_total": match.total,
                        "auto_distance": round(match.distance, 6),
                        "auto_margin": round(match.margin, 6),
                    },
                )
            except Exception as error:  # noqa: BLE001
                result["failed"] = int(result["failed"]) + len(keys)
                cast_errors = result["errors"]
                if isinstance(cast_errors, list):
                    cast_errors.append(str(error))
            else:
                result["confirmed"] = int(result["confirmed"]) + len(keys)
    return result


def _faces_by_source(cache: sort_photos.CacheState) -> dict[str, list[sort_photos.CachedFace]]:
    grouped: dict[str, list[sort_photos.CachedFace]] = {}
    for face in cache.faces:
        grouped.setdefault(os.path.realpath(face.src_str), []).append(face)
    return grouped


def _confirmed_case_face(
    case,
    faces: list[sort_photos.CachedFace],
    identity_db: sort_photos.IdentityDB,
) -> sort_photos.CachedFace | None:
    # Whole-image confirmation cannot select which face belongs to a name.
    return faces[0] if len(faces) == 1 else None


def learn_hard_negatives_from_confirmed_reviews(
    identity_db: sort_photos.IdentityDB,
    cache: sort_photos.CacheState,
    secondary_matcher: secondary_identity_matcher.SecondaryMatcher | None,
    *,
    hard_negatives_path: Path,
    review_prototypes: dict[str, list[np.ndarray]] | None = None,
    evaluation_path: Path = evaluation_enrollment.DEFAULT_PATH,
    state_path: Path | None = None,
) -> dict[str, object]:
    """Turn explicit option-1 corrections into precise lookalike guards."""
    report: dict[str, object] = {
        "available": False,
        "cases": 0,
        "usable": 0,
        "learned": 0,
        "errors": [],
    }
    if secondary_matcher is None or not evaluation_path.expanduser().is_file():
        return report
    try:
        evaluation_stat = evaluation_path.expanduser().stat()
        learning_signature = hashlib.sha256(
            (
                secondary_identity_matcher.primary_signature(identity_db)
                + "\0"
                + secondary_identity_matcher.identity_signature(secondary_matcher.db)
                + "\0"
                + str(evaluation_stat.st_size)
                + "\0"
                + str(evaluation_stat.st_mtime_ns)
                + "\0confirmed-review-learning-v3"
            ).encode("utf-8")
        ).hexdigest()
    except OSError:
        learning_signature = ""
    if state_path is not None and learning_signature:
        try:
            prior = json.loads(state_path.expanduser().read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError):
            prior = {}
        if prior.get("signature") == learning_signature:
            cached_report = dict(prior.get("report") or {})
            cached_report["cached"] = True
            return cached_report
    validation = identity_evaluation.evaluation_dataset.load_dataset(evaluation_path)
    report["available"] = True
    report["cases"] = len(validation.cases)
    report["errors"] = list(validation.errors)
    if validation.errors:
        return report

    faces_by_source = _faces_by_source(cache)
    hard_negatives = identity_hard_negatives.vectors_by_person(hard_negatives_path)
    learned = 0
    usable = 0
    for case in validation.cases:
        if not case.verified or case.expected_person not in identity_db.identities:
            continue
        source_faces = faces_by_source.get(os.path.realpath(str(case.source)), [])
        face = _confirmed_case_face(case, source_faces, identity_db)
        if face is None:
            continue
        usable += 1
        candidates = _rank_face_candidates(
            face,
            case.source,
            identity_db,
            hard_negatives,
            _without_exact_query_prototype(review_prototypes, face.embedding),
        )
        if not candidates or candidates[0].name.casefold() == case.expected_person.casefold():
            continue
        try:
            secondary = secondary_matcher.verify(
                face.crop_jpeg,
                candidates[0].name,
                excluded_source=case.source,
            )
        except TypeError:
            secondary = secondary_matcher.verify(face.crop_jpeg, candidates[0].name)
        item = UnknownItem("benchmark", case.source, face, candidates, secondary)
        evidence = _automatic_item_evidence(item, identity_db)
        if evidence is None or not bool(
            evidence["secondary"] or evidence["secondary_rescue"]
        ):
            continue
        if identity_hard_negatives.record(
            hard_negatives_path,
            person=candidates[0].name,
            source_path=case.source,
            face_index=face.face_index,
            embedding=face.embedding,
            reason="trusted_confirmation_disagrees_with_dual_matcher",
        ):
            learned += 1
    secondary_matcher.flush()
    report["usable"] = usable
    report["learned"] = learned
    report["cached"] = False
    if state_path is not None and learning_signature:
        state_path = state_path.expanduser()
        state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": 1,
            "signature": learning_signature,
            "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
            "report": report,
        }
        temporary = state_path.with_suffix(state_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(state_path)
    return report


def evaluate_automatic_policy_benchmark(
    identity_db: sort_photos.IdentityDB,
    cache: sort_photos.CacheState,
    secondary_matcher: secondary_identity_matcher.SecondaryMatcher,
    *,
    hard_negatives: dict[str, list[np.ndarray]],
    review_prototypes: dict[str, list[np.ndarray]] | None = None,
    evaluation_path: Path = evaluation_enrollment.DEFAULT_PATH,
    progress=terminal_progress,
    checkpoints=None,
) -> tuple[bool, dict[str, object]]:
    """Prove the dual-model automatic lane against explicit confirmations."""
    validation = identity_evaluation.evaluation_dataset.load_dataset(evaluation_path)
    report: dict[str, object] = {
        "available": evaluation_path.expanduser().is_file(),
        "cases": len(validation.cases),
        "evaluated": 0,
        "accepted": 0,
        "correct": 0,
        "incorrect": 0,
        "errors": list(validation.errors),
        "incorrect_rows": [],
        "leave_one_source_out": True,
    }
    if validation.errors:
        return False, report

    from evaluation_profiles import EvaluationProfiles
    prepared = EvaluationProfiles(identity_db)
    verifier_profiles = EvaluationProfiles(sort_photos.IdentityDB(
        identities=secondary_matcher.db.identities,
        prototypes=secondary_matcher.db.prototypes,
        prototype_sources=secondary_matcher.db.prototype_sources), negative_examples=[])
    faces_by_source = _faces_by_source(cache)
    groups = {}
    for case in validation.cases:
        groups.setdefault(case.group_id or case.content_sha256 or str(case.source), set()).add(str(case.source))
    def evaluate(case):
        result = {"evaluated": 0, "accepted": 0, "correct": 0, "incorrect": 0, "incorrect_rows": []}
        if not case.verified or case.expected_person not in identity_db.identities:
            return result
        source_faces = faces_by_source.get(os.path.realpath(str(case.source)), [])
        face = _confirmed_case_face(case, source_faces, identity_db)
        if face is None:
            return result
        result["evaluated"] = 1
        excluded = frozenset(groups[case.group_id or case.content_sha256 or str(case.source)])
        candidates = prepared.rank(face, excluded_sources=excluded, minimum_references=False)[:3]
        if not candidates:
            return result
        secondary = secondary_matcher.verify(
            face.crop_jpeg, candidates[0].name, excluded_source=case.source,
            excluded_sources=excluded, prepared=verifier_profiles)
        item = UnknownItem("benchmark", case.source, face, candidates, secondary)
        evidence = _automatic_item_evidence(item, identity_db)
        if evidence is None or not bool(
            evidence["secondary"] or evidence["secondary_rescue"]
        ):
            return result
        result["accepted"] = 1
        if candidates[0].name.casefold() == case.expected_person.casefold():
            result["correct"] = 1
            return result
        result["incorrect"] = 1
        result["incorrect_rows"].append({
                "source": str(case.source),
                "expected": case.expected_person,
                "predicted": candidates[0].name,
                "primary_distance": round(float(candidates[0].distance), 6),
                "primary_margin": round(
                    identity_profiles.candidate_margin(candidates), 6
                ),
                "secondary_distance": round(float(secondary.distance), 6),
                "secondary_margin": round(float(secondary.margin), 6),
        })
        return result
    results = evaluation_runtime.evaluate_blocks(validation.cases, evaluate, checkpoints=checkpoints,
        stage="Independent-match safety benchmark", progress=progress)
    for result in results:
        for key in ("evaluated", "accepted", "correct", "incorrect"):
            report[key] += result[key]
        report["incorrect_rows"].extend(result["incorrect_rows"][:max(0, 20 - len(report["incorrect_rows"]))])
    prepared.validate()
    verifier_profiles.validate()
    secondary_matcher.flush()
    evaluated = int(report["evaluated"])
    accepted = int(report["accepted"])
    incorrect = int(report["incorrect"])
    report["precision"] = (accepted - incorrect) / max(1, accepted)
    allowed = bool(
        evaluated >= AUTO_GATE_MIN_CONFIRMED_CASES
        and accepted > 0
        and incorrect == 0
    )
    return allowed, report


def automatic_review_gate_signature(
    identity_db: sort_photos.IdentityDB,
    *,
    secondary_db: secondary_identity_matcher.SecondaryIdentityDB | None = None,
    hard_negatives_path: Path = sort_photos.IDENTITY_HARD_NEGATIVES_FILE,
    review_prototypes: dict[str, list[np.ndarray]] | None = None,
) -> str:
    """Fingerprint every input that can change an automatic-review decision."""
    digest = hashlib.sha256()
    digest.update(secondary_identity_matcher.primary_signature(identity_db).encode("ascii"))
    for module_name in ("recognition_policy.py", "identity_assignment.py", "identity_evaluation.py",
                        "identity_profiles.py", "secondary_identity_matcher.py", "review_unknown_identities.py",
                        "identity_confirmations.py", "verified_references.py", "benchmark_inputs.py",
                        "evaluation_profiles.py", "evaluation_runtime.py", "evaluation_checkpoints.py",
                        "sort_photos.py", "recover_no_usable_faces.py", "appearance_profiles.py",
                        "evaluation_dataset.py", "benchmark_detection.py", "shadow_evaluation.py",
                        "identity_hard_negatives.py", "content_identity.py"):
        digest.update(sort_photos.content_identity.content_sha256(Path(__file__).with_name(module_name)).encode("ascii"))
    digest.update(secondary_identity_matcher.identity_signature(secondary_db).encode("ascii"))
    # Calibration and pose changes can alter decisions without changing the
    # core verifier fingerprint. Such changes must invalidate the safety gate.
    digest.update(json.dumps({
        "match_thresholds": identity_db.match_thresholds,
        "strict_thresholds": identity_db.strict_thresholds,
        "source_counts": identity_db.source_counts,
        "era_cutoffs": identity_db.appearance_era_cutoffs,
        "prototype_sources": identity_db.prototype_sources,
        "pose_sources": identity_db.pose_prototype_sources,
        "appearance_sources": identity_db.appearance_prototype_sources,
        "detector": sort_photos.config_fingerprint(),
        "settings": {name: value for name, value in vars(sort_photos).items()
                     if name.startswith("AUTO_PERSON_") and isinstance(value, (str, bool, int, float))},
        "recovery_settings": {name: value for name, value in vars(recover_no_usable_faces).items()
                              if name.startswith("MATCH_") and isinstance(value, (str, bool, int, float))},
    }, sort_keys=True).encode("utf-8"))
    for person in sorted(identity_db.identities, key=str.casefold):
        digest.update(person.encode("utf-8"))
        digest.update(np.asarray(identity_db.identities[person], dtype=np.float32).tobytes())
        for prototype in identity_db.prototypes.get(person, []):
            digest.update(np.asarray(prototype, dtype=np.float32).tobytes())
        for profiles in (identity_db.pose_prototypes, identity_db.appearance_prototypes):
            for label, values in sorted(profiles.get(person, {}).items()):
                digest.update(label.encode("utf-8"))
                for value in values:
                    digest.update(np.asarray(value, dtype=np.float32).tobytes())
    for person in sorted(review_prototypes or {}, key=str.casefold):
        digest.update(person.encode("utf-8", errors="surrogateescape"))
        for prototype in (review_prototypes or {}).get(person, []):
            digest.update(
                identity_profiles.normalize_vector(prototype)
                .astype(np.float32).tobytes()
            )
    digest.update(json.dumps({
        "minimum_cases": AUTO_GATE_MIN_CONFIRMED_CASES,
        "primary_distance": AUTO_JOINT_MAX_PRIMARY_DISTANCE,
        "secondary_distance": AUTO_JOINT_MAX_SECONDARY_DISTANCE,
        "primary_margin": AUTO_JOINT_MIN_PRIMARY_MARGIN,
        "secondary_margin": AUTO_JOINT_MIN_SECONDARY_MARGIN,
        "quality": AUTO_JOINT_MIN_QUALITY,
        "rescue_primary_distance": AUTO_RESCUE_MAX_PRIMARY_DISTANCE,
        "rescue_primary_margin": AUTO_RESCUE_MIN_PRIMARY_MARGIN,
        "rescue_secondary_distance": AUTO_RESCUE_MAX_SECONDARY_DISTANCE,
        "rescue_secondary_margin": AUTO_RESCUE_MIN_SECONDARY_MARGIN,
        "rescue_quality": AUTO_RESCUE_MIN_QUALITY,
    }, sort_keys=True).encode("ascii"))
    for path in (
        hard_negatives_path,
        evaluation_enrollment.DEFAULT_PATH,
        sort_photos.IDENTITY_CONFIRMATIONS_FILE,
        identity_evaluation.DEFAULT_PROTECTED_SET,
        identity_evaluation.DEFAULT_PROTECTED_BASELINE,
    ):
        try:
            value = f"{path.resolve()}\0{sort_photos.content_identity.content_sha256(path)}"
        except OSError:
            value = f"{path.expanduser().resolve(strict=False)}\0missing"
        digest.update(value.encode("utf-8", errors="surrogateescape"))
    return digest.hexdigest()


def prepare_automatic_review_gate(
    identity_db: sort_photos.IdentityDB,
    cache: sort_photos.CacheState,
    secondary_matcher: secondary_identity_matcher.SecondaryMatcher | None,
    *,
    requested: bool,
    output_dir: Path,
    review_prototypes: dict[str, list[np.ndarray]] | None = None,
    progress=terminal_progress,
) -> tuple[bool, dict[str, object], str]:
    """A failed safety evaluation disables auto-filing, not manual review."""
    try:
        if not requested:
            return _prepare_automatic_review_gate(identity_db, cache, secondary_matcher,
                requested=False, output_dir=output_dir, review_prototypes=review_prototypes, progress=progress)
        output_dir.mkdir(parents=True, exist_ok=True)
        with evaluation_runtime.exclusive_evaluation(output_dir):
            return _prepare_automatic_review_gate(
                identity_db, cache, secondary_matcher, requested=requested,
                output_dir=output_dir, review_prototypes=review_prototypes, progress=progress)
    except Exception as error:  # noqa: BLE001
        message = f"Safe auto-match unavailable: {type(error).__name__}: {error}. Manual review remains available."
        progress(message)
        # Do not cache interrupted/invalid evaluations as a completed verdict.
        return False, {"available": False, "requested": requested, "allowed": False,
                       "evaluation_incomplete": True, "failures": [str(error)]}, message


def _prepare_automatic_review_gate(
    identity_db: sort_photos.IdentityDB,
    cache: sort_photos.CacheState,
    secondary_matcher: secondary_identity_matcher.SecondaryMatcher | None,
    *,
    requested: bool,
    output_dir: Path,
    review_prototypes: dict[str, list[np.ndarray]] | None = None,
    progress=terminal_progress,
) -> tuple[bool, dict[str, object], str]:
    """Use a cached full-library safety gate before allowing automatic moves."""
    if not requested:
        return False, {"available": False, "requested": False}, "Safe auto-match is off."
    if secondary_matcher is None:
        return (
            False,
            {"available": False, "requested": True, "failures": ["secondary matcher unavailable"]},
            "Safe auto-match is off because the independent verifier is unavailable.",
        )

    progress("Safety benchmark: checking cached result and input versions...")
    policy_signature = automatic_review_gate_signature(
        identity_db,
        secondary_db=secondary_matcher.db,
        review_prototypes=review_prototypes,
    )
    primary_faces = evaluation_runtime.selected_faces(cache, output_dir, progress=progress)
    inputs = evaluation_runtime.GateInputs(cache, identity_db, secondary_matcher.db,
        datasets=(evaluation_enrollment.DEFAULT_PATH, identity_evaluation.DEFAULT_PROTECTED_SET),
        evidence_paths=(sort_photos.IDENTITY_HARD_NEGATIVES_FILE,
                        sort_photos.IDENTITY_CONFIRMATIONS_FILE,
                        identity_evaluation.DEFAULT_PROTECTED_BASELINE),
        negative_examples=identity_hard_negatives.load(sort_photos.IDENTITY_HARD_NEGATIVES_FILE)["examples"],
        primary_faces=primary_faces)
    signature = hashlib.sha256((policy_signature + inputs.fingerprint).encode()).hexdigest()
    def validate_inputs():
        inputs.validate()
        current_policy = automatic_review_gate_signature(identity_db, secondary_db=secondary_matcher.db,
                                                         review_prototypes=review_prototypes)
        if current_policy != policy_signature:
            raise RuntimeError("Safety benchmark inputs changed during evaluation; retry with the saved annotations")
    gate_path = output_dir / "unknown_auto_review_gate.json"
    try:
        cached = json.loads(gate_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        cached = {}
    if (
        cached.get("version") == AUTO_GATE_CACHE_VERSION
        and cached.get("signature") == signature
        and isinstance(cached.get("allowed"), bool)
    ):
        allowed = bool(cached["allowed"])
        validate_inputs()
        progress("Safety benchmark: reusing unchanged cached result.")
        return allowed, cached, (
            "Safe auto-match passed the cached full-library safety gate."
            if allowed
            else "Safe auto-match is blocked by the cached safety gate."
        )

    progress("Safety benchmark: inputs changed or no cached result; evaluating primary matcher...")
    output_dir.mkdir(parents=True, exist_ok=True)
    with BenchmarkCheckpoints(output_dir / "benchmark_checkpoints.sqlite3", signature) as store:
        store.prune()
        checkpoints = evaluation_runtime.GuardedCheckpoints(store, validate_inputs)
        try:
            primary_allowed, report = identity_evaluation.activation_gate(
                identity_db, identity_db, cache, confirmed_set=None, progress=progress,
                checkpoints=checkpoints, selected_faces=primary_faces, fail_fast=True)
            checkpoints.check()
            if primary_allowed:
                policy_allowed, policy_report = evaluate_automatic_policy_benchmark(
                    identity_db, cache, secondary_matcher,
                    hard_negatives=identity_hard_negatives.vectors_by_person(
                        sort_photos.IDENTITY_HARD_NEGATIVES_FILE),
                    review_prototypes=review_prototypes, progress=progress,
                    checkpoints=checkpoints)
            else:
                progress("Safety benchmark blocked; skipping independent matcher. Manual review remains available.")
                policy_allowed, policy_report = False, {"skipped": True, "reason": "primary safety gate blocked"}
            checkpoints.check()
        except BaseException:
            # Keep ordinary interruption checkpoints, discard any mixed-input work.
            try:
                checkpoints.check()
            except Exception:
                pass
            raise
    allowed = bool(primary_allowed and policy_allowed)
    report["automatic_policy"] = policy_report
    if primary_allowed and not policy_allowed:
        report.setdefault("failures", []).append(
            "dual-model automatic policy did not pass the confirmed benchmark"
        )
    validate_inputs()
    payload: dict[str, object] = {
        "version": AUTO_GATE_CACHE_VERSION,
        "signature": signature,
        "generated_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "allowed": bool(allowed),
        "report": report,
    }
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = gate_path.with_suffix(gate_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(gate_path)
    return bool(allowed), payload, (
        "Safe auto-match passed the full-library safety gate."
        if allowed
        else "Safe auto-match is blocked because the safety benchmark failed."
    )


class ReviewActionQueue:
    """Serializes live review actions and deduplicates overlapping submissions."""

    def __init__(self, state: dict, executor=apply_decision):
        self.state = state
        self.executor = executor
        self.pending: queue.Queue[dict | None] = queue.Queue()
        self.lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self.item_jobs: dict[str, str] = {}
        self.sequence = 0
        self.closed = False
        self.worker = threading.Thread(
            target=self._run,
            name="unknown-review-actions",
            daemon=True,
        )
        self.worker.start()

    @staticmethod
    def _public(job: dict, *, deduplicated: bool = False) -> dict:
        return {
            key: value for key, value in job.items()
            if key not in {"item_key_set"}
        } | {"deduplicated": deduplicated}

    def submit(
        self,
        *,
        item_keys: list[str],
        action: str,
        person_value: str = "",
    ) -> dict:
        if self.closed:
            raise ValueError("review action queue is closing")
        if action not in {
            "confirm", "keep_unknown", "ignore", "move_to_junk", "reject_candidate"
        }:
            raise ValueError("unknown review action")
        unique_keys = tuple(dict.fromkeys(key for key in item_keys if key))
        if not unique_keys:
            raise ValueError("no review items selected")
        if len(unique_keys) > MAX_BATCH_ACTION_ITEMS:
            raise ValueError(
                f"select no more than {MAX_BATCH_ACTION_ITEMS} images per review action"
            )
        person = person_value.strip()
        if action in {"confirm", "reject_candidate"}:
            person = canonical_person(
                person,
                self.state["identity_db"],
                allow_new=action == "confirm",
                people_root=self.state.get("people_root"),
            )

        key_set = frozenset(unique_keys)
        with self.lock:
            active_ids = {
                self.item_jobs[key] for key in unique_keys if key in self.item_jobs
            }
            if active_ids:
                if len(active_ids) == 1:
                    existing = self.jobs[next(iter(active_ids))]
                    if (
                        existing["action"] == action
                        and existing["person"] == person
                        and existing["item_key_set"] == key_set
                    ):
                        return self._public(existing, deduplicated=True)
                raise ReviewActionConflict(
                    "one or more selected images already have a queued or running action"
                )
            completed = [
                job for job in self.jobs.values()
                if job["status"] == "completed"
                and job["item_key_set"] & key_set
            ]
            for existing in completed:
                if (
                    existing["action"] == action
                    and existing["person"] == person
                    and existing["item_key_set"] == key_set
                ):
                    return self._public(existing, deduplicated=True)
            if action != "reject_candidate" and completed:
                raise ReviewActionConflict(
                    "one or more selected images were already reviewed in this session"
                )

            self.sequence += 1
            active_ahead = sum(
                job["status"] in {"queued", "running"}
                for job in self.jobs.values()
            )
            job = {
                "id": uuid.uuid4().hex,
                "sequence": self.sequence,
                "status": "queued",
                "position": active_ahead + 1,
                "action": action,
                "person": person,
                "item_keys": list(unique_keys),
                "item_key_set": key_set,
                "submitted_at": time.time(),
                "started_at": None,
                "finished_at": None,
                "message": "",
                "error": "",
            }
            self.jobs[job["id"]] = job
            for key in unique_keys:
                self.item_jobs[key] = job["id"]
            self.pending.put(job)
            return self._public(job)

    def submit_many(
        self,
        *,
        item_keys: list[str],
        action: str,
        person_value: str = "",
    ) -> list[dict]:
        unique_keys = list(dict.fromkeys(key for key in item_keys if key))
        if not unique_keys:
            raise ValueError("no review items selected")
        if len(unique_keys) > MAX_CLUSTER_ACTION_ITEMS:
            raise ValueError(
                f"select no more than {MAX_CLUSTER_ACTION_ITEMS} images per cluster action"
            )
        jobs = []
        for offset in range(0, len(unique_keys), MAX_BATCH_ACTION_ITEMS):
            jobs.append(self.submit(
                item_keys=unique_keys[offset:offset + MAX_BATCH_ACTION_ITEMS],
                action=action,
                person_value=person_value,
            ))
        return jobs

    def snapshot(
        self,
        job_ids: list[str] | None = None,
        *,
        active_only: bool = False,
    ) -> dict:
        with self.lock:
            requested = set(job_ids or self.jobs)
            jobs = [
                self._public(job)
                for job in sorted(self.jobs.values(), key=lambda value: value["sequence"])
                if job["id"] in requested
                and (not active_only or job["status"] in {"queued", "running"})
            ]
            counts = Counter(job["status"] for job in self.jobs.values())
        return {
            "jobs": jobs,
            "summary": {
                "queued": counts["queued"],
                "running": counts["running"],
                "completed": counts["completed"],
                "failed": counts["failed"],
            },
        }

    def is_idle(self) -> bool:
        with self.lock:
            return not any(
                job["status"] in {"queued", "running"}
                for job in self.jobs.values()
            )

    def _run(self) -> None:
        while True:
            job = self.pending.get()
            if job is None:
                self.pending.task_done()
                return
            with self.lock:
                job["status"] = "running"
                job["position"] = 0
                job["started_at"] = time.time()
            try:
                with self.state["lock"]:
                    message = self.executor(
                        self.state,
                        item_keys=job["item_keys"],
                        action=job["action"],
                        person_value=job["person"],
                    )
            except Exception as error:  # noqa: BLE001
                with self.lock:
                    job["status"] = "failed"
                    job["error"] = str(error)
            else:
                with self.lock:
                    job["status"] = "completed"
                    job["message"] = message
            finally:
                with self.lock:
                    job["finished_at"] = time.time()
                    for key in job["item_keys"]:
                        if self.item_jobs.get(key) == job["id"]:
                            self.item_jobs.pop(key, None)
                    queued = sorted(
                        (
                            value for value in self.jobs.values()
                            if value["status"] == "queued"
                        ),
                        key=lambda value: value["sequence"],
                    )
                    for position, queued_job in enumerate(queued, 1):
                        queued_job["position"] = position
                self.pending.task_done()

    def close(self, *, wait: bool = True) -> None:
        self.closed = True
        if wait:
            self.pending.join()
        self.pending.put(None)
        self.worker.join(timeout=30)


class BatchLoadJob:
    """Loads and analyzes the next review batch without blocking the dashboard."""

    def __init__(self, state: dict):
        self.state = state
        self.lock = threading.Lock()
        self.payload = {
            "status": "idle",
            "step": "Ready",
            "processed": 0,
            "total": 0,
            "single_face": 0,
            "multi_face": 0,
            "errors": 0,
            "complete": False,
            "message": "",
        }
        self.worker: threading.Thread | None = None

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.payload)

    def _set(self, **values: object) -> None:
        with self.lock:
            self.payload.update(values)

    def _progress(self, processed: int, total: int, stats: Counter) -> None:
        self._set(
            status="running",
            step=f"Analyzing {processed} of {total}",
            processed=processed,
            total=total,
            single_face=int(stats["single_face"]),
            multi_face=int(stats["multi_face"]),
            errors=int(stats["errors"]),
        )

    def start(self) -> tuple[dict, str | None]:
        state = self.state
        with state["lifecycle_lock"]:
            if state.get("finishing") or state.get("review_finished"):
                return self.snapshot(), "Finish Review is active or already complete."
            if state.get("batch_loading"):
                return self.snapshot(), None
            if not state["action_queue"].is_idle():
                return self.snapshot(), "Wait for queued actions before loading the next batch."
            state["batch_loading"] = True
            self._set(
                status="queued",
                step="Preparing next batch",
                processed=0,
                total=int(state.get("batch_limit", 500)),
                single_face=0,
                multi_face=0,
                errors=0,
                complete=False,
                message="",
            )
            self.worker = threading.Thread(
                target=self._run,
                name="unknown-review-batch-loader",
                daemon=True,
            )
            self.worker.start()
            return self.snapshot(), None

    def _run(self) -> None:
        state = self.state
        working_state = dict(state)
        working_state["batch_progress_callback"] = self._progress
        try:
            self._set(status="running", step="Discovering pending files")
            result = load_next_batch(working_state)
            with state["lock"]:
                for key in ("clusters", "items_by_key", "summary", "batch_number"):
                    state[key] = working_state[key]
                state["cache_dirty"] = bool(
                    state.get("cache_dirty") or working_state.get("cache_dirty")
                )
            complete = bool(result.get("complete"))
            self._set(
                status="completed",
                step="Queue complete" if complete else "Batch ready",
                complete=complete,
                message=(
                    "All pending unknown files are handled."
                    if complete
                    else f"Loaded {len(state['items_by_key'])} reviewable images."
                ),
            )
        except Exception as error:  # noqa: BLE001
            self._set(status="failed", step="Load failed", message=str(error))
        finally:
            with state["lifecycle_lock"]:
                state["batch_loading"] = False


class FinishReviewJob:
    """Runs the expensive final persistence and activation gate exactly once."""

    def __init__(self, state: dict):
        self.state = state
        self.lock = threading.Lock()
        self.payload = {
            "status": "idle",
            "step": "Ready",
            "message": "",
            "report": "",
            "started_at": None,
            "finished_at": None,
        }
        self.worker: threading.Thread | None = None

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.payload)

    def _set(self, **values: object) -> None:
        with self.lock:
            self.payload.update(values)

    def start(self) -> dict:
        with self.lock:
            if self.payload["status"] in {"queued", "running", "completed"}:
                return dict(self.payload)
            self.payload.update({
                "status": "queued",
                "step": "Waiting for queued review actions",
                "message": "",
                "started_at": time.time(),
            })
        self.state["finishing"] = True
        self.worker = threading.Thread(
            target=self._run,
            name="unknown-review-finish",
            daemon=True,
        )
        self.worker.start()
        return self.snapshot()

    def _run(self) -> None:
        state = self.state
        try:
            self._set(status="running", step="Waiting for queued actions")
            state["action_queue"].pending.join()
            with state["lock"]:
                self._set(step="Saving decisions and face cache")
                if state.get("cache_dirty"):
                    sort_photos.save_cache(state["cache"])
                    state["cache_dirty"] = False

                profile_refreshed = False
                activation_report: dict = {}
                if state.get("identity_dirty"):
                    self._set(step="Refreshing identity profiles")
                    refreshed_db = sort_photos.build_identity_db_from_person_folders(
                        Path(state["people_root"])
                    )
                    state["identity_db"] = refreshed_db
                    state["identity_dirty"] = False
                    profile_refreshed = True
                    self._set(step="Running safety benchmark")
                    gate_path = Path(state.get(
                        "activation_gate_path",
                        pipeline_paths.SOURCE_REVIEW
                        / "identity_evaluation"
                        / "latest_profile_activation_gate.json",
                    ))
                    try:
                        activation_report = json.loads(gate_path.read_text(encoding="utf-8"))
                    except (OSError, ValueError, TypeError):
                        activation_report = {
                            "available": False,
                            "message": "No activation-gate report was produced.",
                        }
                else:
                    self._set(step="Running safety benchmark")
                    _allowed, activation_report = identity_evaluation.activation_gate(
                        state["identity_db"],
                        state["identity_db"],
                        state["cache"],
                        confirmed_set=Path(state["evaluation_path"]),
                    )
                    activation_report["current_profile_check"] = True

                secondary = state.get("secondary_matcher")
                if secondary is not None:
                    secondary.flush()

                self._set(step="Writing final report")
                progress = review_progress(state)
                decisions = load_decisions(state["decisions_path"])
                automatic_confirmations = sum(
                    1
                    for value in decisions.get("items", {}).values()
                    if isinstance(value, dict)
                    and value.get("action") == "confirmed"
                    and value.get("automatic") is True
                )
                report = {
                    "version": 1,
                    "finished_at": dt.datetime.now(dt.timezone.utc).isoformat(),
                    "progress": progress,
                    "profile_refreshed": profile_refreshed,
                    "activation_gate": activation_report,
                    "safety": {
                        "automatic_confirmations": automatic_confirmations,
                        "thresholds_lowered": False,
                        "note": (
                            "Automatic confirmations required the safety gate, independent "
                            "matcher agreement, and strict cluster evidence. Only explicit "
                            "user confirmations were enrolled as trusted training examples."
                        ),
                    },
                }
                report_path = Path(state["output_dir"]) / "unknown_review_final_report.json"
                report_path.parent.mkdir(parents=True, exist_ok=True)
                temporary = report_path.with_suffix(report_path.suffix + ".tmp")
                temporary.write_text(
                    json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
                    encoding="utf-8",
                )
                temporary.replace(report_path)
                state["summary"]["progress"] = progress
                save_session_state(state, status="finished")
                state["review_finished"] = True
            self._set(
                status="completed",
                step="Complete",
                message=(
                    f"Review saved: {progress['reviewed']} of {progress['total']} handled; "
                    f"{progress['pending']} remain."
                ),
                report=str(report_path),
                finished_at=time.time(),
            )
        except Exception as error:  # noqa: BLE001
            self._set(
                status="failed",
                step="Failed safely",
                message=str(error),
                finished_at=time.time(),
            )
        finally:
            state["finishing"] = False


def candidate_summary(cluster: UnknownCluster) -> str:
    margin = identity_profiles.candidate_margin(cluster.candidates)
    values = " | ".join(
        f"{candidate.name}: {candidate.distance:.3f}"
        for candidate in cluster.candidates
    )
    return f"{values} | margin: {margin:.3f}"


def render_html(
    clusters: list[UnknownCluster],
    decisions: dict,
    identity_names: list[str],
    summary: dict,
    *,
    interactive: bool,
) -> str:
    cards: list[str] = []
    pending_clusters = 0
    pending_items = 0
    known_people_json = json.dumps(
        [name.casefold() for name in identity_names], ensure_ascii=True
    ).replace("</", "<\\/")
    all_people = "".join(
        f'<option value="{html.escape(name, quote=True)}"></option>'
        for name in identity_names
    )
    for cluster in clusters:
        unresolved = [
            item for item in cluster.items
            if not decision_for_item(decisions, item).get("action")
            and item.path.is_file()
        ]
        resolved = not unresolved
        if unresolved:
            pending_clusters += 1
            pending_items += len(unresolved)
        best_name = cluster.candidates[0].name if cluster.candidates else ""
        cluster_disabled = " disabled" if not interactive or not unresolved else ""
        suggestions = "".join(
            "<button class='suggestion secondary-button' data-action='confirm' data-scope='cluster' "
            f"data-shortcut='{index}' title='Confirm all as {html.escape(candidate.name, quote=True)}' "
            f"data-person='{html.escape(candidate.name, quote=True)}'{cluster_disabled}>"
            f"<kbd>{index}</kbd> {html.escape(candidate.name)} "
            f"<span>{candidate.distance:.3f}</span></button>"
            for index, candidate in enumerate(cluster.candidates, 1)
        )
        matcher_agreement = bool(
            unresolved
            and best_name
            and all(
                item.candidates
                and item.candidates[0].name.casefold() == best_name.casefold()
                and item.secondary is not None
                and item.secondary.accepted
                and str(item.secondary.predicted or "").casefold() == best_name.casefold()
                for item in unresolved
            )
        )
        agreement_badge = (
            "<span class='agreement-badge'>Both matchers agree; held for manual review by the remaining safety rules</span>"
            if matcher_agreement else ""
        )
        item_cards: list[str] = []
        for item in cluster.items:
            decision = decision_for_item(decisions, item)
            decided_action = str(decision.get("action", ""))
            exists = item.path.is_file()
            item_best = item.candidates[0].name if item.candidates else best_name
            face_url = "/face?" + urlencode({"item": item.key})
            full_url = "/image?" + urlencode({"item": item.key})
            disabled = " disabled" if not interactive or decided_action or not exists else ""
            selectable_disabled = " disabled" if decided_action or not exists else ""
            status = {
                "confirmed": f"Confirmed as {decision.get('person', '')}",
                "keep_unknown": "Kept as unknown",
                "ignored": "Ignored",
                "moved_to_junk": "Moved to recoverable junk",
            }.get(decided_action, "Pending" if exists else "File moved")
            item_margin = identity_profiles.candidate_margin(item.candidates)
            rejected_people = {
                str(value) for value in decision.get("rejected_people", []) if str(value)
            }
            secondary_text = ""
            if item.secondary is not None:
                result = "agrees" if item.secondary.accepted else "does not agree"
                secondary_text = (
                    f" · secondary {result}: {item.secondary.predicted or 'none'} "
                    f"d={item.secondary.distance:.3f} m={item.secondary.margin:.3f}"
                )
            negative_actions = "".join(
                "<button class='negative secondary-button' data-action='reject_candidate' data-scope='item' "
                f"data-person='{html.escape(candidate.name, quote=True)}'{disabled}>"
                f"Not {html.escape(candidate.name)}</button>"
                for candidate in item.candidates
                if candidate.name not in rejected_people
            )
            candidate_chips = "".join(
                f"<span class='candidate-chip'>{html.escape(candidate.name)} "
                f"<b>{candidate.distance:.3f}</b></span>"
                for candidate in item.candidates
            )
            item_search = " ".join((
                item.path.name,
                str(item.path),
                " ".join(candidate.name for candidate in item.candidates),
                status,
            )).casefold()
            item_status = "reviewed" if decided_action or not exists else "pending"
            item_cards.append(
                f"<article class='item {item_status}' data-item='{item.key}' "
                f"data-status='{item_status}' data-best='{html.escape(item_best, quote=True)}' "
                f"data-search='{html.escape(item_search, quote=True)}'>"
                f"<div class='image-area'><button class='preview-button' type='button' "
                f"data-full='{html.escape(full_url, quote=True)}' "
                f"data-name='{html.escape(item.path.name, quote=True)}'>"
                f"<img loading='lazy' decoding='async' src='{html.escape(face_url, quote=True)}' alt='Detected face'></button>"
                f"<label class='select-control' title='Select image'><input class='row-select' "
                f"type='checkbox' value='{item.key}'{selectable_disabled}><span></span></label>"
                f"<span class='status-badge'>{html.escape(status)}</span></div>"
                f"<div class='item-body'><strong class='filename' title='{html.escape(str(item.path), quote=True)}'>{html.escape(item.path.name)}</strong>"
                f"<div class='candidate-chips'>{candidate_chips}</div>"
                f"<p class='evidence'>margin={item_margin:.3f} &middot; quality={item.face.quality:.3f} &middot; pose={html.escape(item.face.pose_label)}{html.escape(secondary_text)}</p>"
                f"</div><details class='item-review'><summary>Review actions</summary><div class='review-body'>"
                f"<label>Confirmed person<input class='person' list='people' value='{html.escape(item_best, quote=True)}' aria-label='Confirmed person'{disabled}></label>"
                f"<div class='item-actions'><button class='primary-button' data-action='confirm' data-scope='item'{disabled}>Confirm Cluster</button>"
                f"<button class='secondary-button' data-action='keep_unknown' data-scope='item'{disabled}>Keep Unknown</button>"
                f"<button class='secondary-button' data-action='ignore' data-scope='item'{disabled}>Ignore</button></div>"
                f"<div class='negative-actions'>{negative_actions}</div></div></details></article>"
            )
        cluster_search = " ".join((
            " ".join(candidate.name for candidate in cluster.candidates),
            " ".join(item.path.name for item in cluster.items),
        )).casefold()
        cluster_kind = "group" if len(cluster.items) > 1 else "single"
        cards.append(
            f"<section class='cluster {'resolved-cluster' if resolved else ''}' data-cluster='{cluster.key}' "
            f"data-kind='{cluster_kind}' data-search='{html.escape(cluster_search, quote=True)}' "
            f"data-pending='{'0' if resolved else '1'}' "
            f"data-items='{html.escape(','.join(item.key for item in unresolved), quote=True)}'>"
            f"<div class='cluster-head'><div><h2>{len(cluster.items)} similar face(s)</h2>"
            f"<p>{html.escape(candidate_summary(cluster))}</p>"
            f"<p>Cohesion {cluster.cohesion:.3f}; every image in this visual cluster is shown.</p>{agreement_badge}</div>"
            f"<div class='cluster-actions'><button class='secondary-button select-cluster' type='button'{cluster_disabled}>Select cluster</button>"
            f"<button class='secondary-button' data-action='keep_unknown' data-scope='cluster'{cluster_disabled}><kbd>U</kbd> Confirm all as Unknown</button>"
            f"<button class='danger-button' data-action='move_to_junk' data-scope='cluster'{cluster_disabled}><kbd>J</kbd> Move all to Junk</button>"
            f"{suggestions}<div class='custom-person'><input class='cluster-person' list='people' placeholder='Existing or new person' aria-label='Person for this cluster'{cluster_disabled}>"
            f"<button class='primary-button confirm-custom' type='button'{cluster_disabled}>Confirm Cluster</button></div></div></div>"
            f"<div class='items'>{''.join(item_cards)}</div></section>"
        )
    mode_note = (
        "Actions are live. Confirmed files are copied and verified before the unknown source is archived recoverably."
        if interactive
        else "Static preview; launch from Face Terminal to use review actions."
    )
    queue_files = summary.get("queue_files", 0)
    batch_files = summary.get("batch_files", 0)
    sqlite_hits = summary.get("sqlite_hits", 0)
    detected = summary.get("detected", 0)
    progress = summary.get("progress", {})
    reviewed = int(progress.get("reviewed", 0))
    total = int(progress.get("total", queue_files))
    confirmed = int(progress.get("confirmed", 0))
    unknown = int(progress.get("unknown", 0))
    junk = int(progress.get("junk", 0))
    deferred = int(progress.get("deferred", 0))
    auto_review = summary.get("auto_review", {})
    auto_sweep = summary.get("auto_sweep", {})
    auto_confirmed = int(auto_review.get("confirmed", 0)) if isinstance(auto_review, dict) else 0
    auto_eligible = int(auto_review.get("eligible", 0)) if isinstance(auto_review, dict) else 0
    auto_preview = bool(auto_review.get("preview")) if isinstance(auto_review, dict) else False
    auto_enabled = bool(auto_review.get("enabled")) if isinstance(auto_review, dict) else False
    auto_requested = bool(auto_review.get("requested")) if isinstance(auto_review, dict) else False
    auto_gate_message = str(auto_review.get("gate_message", "")) if isinstance(auto_review, dict) else ""
    sweep_scanned = int(auto_sweep.get("scanned", 0)) if isinstance(auto_sweep, dict) else 0
    sweep_confirmed = int(auto_sweep.get("confirmed", 0)) if isinstance(auto_sweep, dict) else 0
    sweep_cached = bool(auto_sweep.get("cached")) if isinstance(auto_sweep, dict) else False
    if auto_preview:
        auto_status = f"Safe auto-match preview: {auto_eligible} eligible"
    elif sweep_scanned:
        auto_status = (
            f"Safe sweep: {sweep_confirmed} filed from {sweep_scanned} scanned"
            + (" (already current)" if sweep_cached else "")
        )
    elif auto_enabled:
        auto_status = f"Safe auto-match: {auto_confirmed} filed this batch"
    elif auto_requested:
        auto_status = auto_gate_message or "Safe auto-match unavailable"
    else:
        auto_status = "Safe auto-match off"
    return f"""<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width"><title>Unknown Identity Review</title>
<style>
:root{{color-scheme:dark;--bg:#0b0c0e;--panel:#17191d;--panel2:#22252b;--line:#343840;--text:#f5f5f7;--muted:#a8adb5;--blue:#0a84ff;--green:#30d158;--amber:#ffd60a;--red:#ff6961}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:14px -apple-system,BlinkMacSystemFont,"SF Pro Text",sans-serif;letter-spacing:0}}
button,input,select{{font:inherit}}button{{cursor:pointer}}button:disabled,input:disabled{{opacity:.35;cursor:not-allowed}}
kbd{{display:inline-grid;place-items:center;min-width:20px;height:20px;padding:0 5px;border:1px solid #5b606a;border-radius:4px;background:#2c3036;color:#fff;font:600 11px -apple-system,BlinkMacSystemFont,"SF Mono",monospace}}
header{{position:sticky;top:0;z-index:20;padding:15px 20px 12px;background:rgba(11,12,14,.95);border-bottom:1px solid var(--line);backdrop-filter:blur(18px)}}
.inner{{max-width:1800px;margin:auto}}h1{{font-size:25px;margin:0}}h2{{font-size:17px;margin:0 0 4px}}p{{margin:4px 0;color:var(--muted)}}
.metrics,.toolbar,.batch-row,.batch-actions,.cluster-head,.cluster-actions{{display:flex;align-items:center}}
.metrics{{gap:7px;flex-wrap:wrap;margin-top:10px}}.metric{{padding:5px 8px;border-radius:5px;background:var(--panel2);color:var(--muted)}}.metric strong{{color:var(--text)}}
.toolbar{{gap:8px;margin-top:10px;flex-wrap:wrap}}.toolbar input,.toolbar select,.batch-panel input{{min-height:36px;padding:7px 9px;border:1px solid var(--line);border-radius:6px;color:var(--text);background:var(--panel2)}}
.quick-nav{{display:flex;align-items:center;gap:8px;margin-top:10px;padding-top:10px;border-top:1px solid var(--line)}}.quick-nav .cluster-position{{min-width:130px;text-align:center;color:var(--muted)}}.quick-nav .finish-button{{margin-left:auto}}.shortcut-help{{color:var(--muted);font-size:12px}}
.search{{min-width:260px;flex:1}}.toolbar select{{width:auto}}input:focus,select:focus{{outline:2px solid var(--blue);outline-offset:-1px}}
.primary-button,.secondary-button,.danger-button{{display:inline-flex;align-items:center;justify-content:center;gap:6px;min-height:34px;padding:6px 9px;border:1px solid var(--line);border-radius:6px;background:var(--panel2);color:var(--text);text-decoration:none;white-space:nowrap}}
.primary-button{{border-color:var(--blue);background:var(--blue);color:white}}button[data-action=keep_unknown]{{color:#ffe680}}button.negative{{color:#ff9b96;font-size:11px;min-height:28px}}
.danger-button{{border-color:#873f3b;background:#3a1c1a;color:#ffb2ad}}
.density-switch{{display:flex;border:1px solid var(--line);border-radius:6px;overflow:hidden}}.density-switch button{{min-height:34px;padding:6px 9px;border:0;border-right:1px solid var(--line);background:var(--panel);color:var(--muted)}}.density-switch button:last-child{{border-right:0}}.density-switch button.active{{background:#3a3d45;color:white}}
.batch-panel{{display:none;margin-top:10px;padding:10px;border:1px solid #285987;border-radius:7px;background:#14263a}}.batch-panel.active{{display:block}}.batch-row{{justify-content:space-between;gap:12px}}.batch-actions{{gap:6px;flex-wrap:wrap}}.batch-person{{min-width:230px}}.batch-suggestion{{color:#b9d9ff}}
main{{max-width:1800px;margin:auto;padding:0 20px 42px}}.cluster{{padding:18px 0;border-bottom:1px solid var(--line)}}.cluster.hidden{{display:none}}body.quick-review .cluster{{display:none}}body.quick-review .cluster.quick-active{{display:block}}
.cluster-head{{justify-content:space-between;gap:18px;align-items:flex-start;margin-bottom:11px}}.cluster-actions{{gap:6px;flex-wrap:wrap;justify-content:flex-end}}.suggestion span{{color:#b5bac2}}
.custom-person{{display:flex;gap:6px}}.cluster-person{{min-width:220px;min-height:34px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:#0f1114;color:var(--text)}}.agreement-badge{{display:inline-block;margin-top:7px;padding:4px 7px;border-radius:4px;background:#173d25;color:#b5f4c3;font-size:12px;font-weight:700}}
.items{{display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:9px}}body.comfortable .items{{grid-template-columns:repeat(auto-fill,minmax(260px,1fr));gap:13px}}
.item{{min-width:0;overflow:hidden;border:1px solid var(--line);border-radius:7px;background:var(--panel);transition:border-color .12s,box-shadow .12s}}.item.hidden{{display:none}}.item.selected{{border-color:var(--blue);box-shadow:0 0 0 2px rgba(10,132,255,.3)}}.item.reviewed{{opacity:.52}}.item.processing{{border-color:#725f16;opacity:.68}}.item.processing .status-badge{{background:#725f16;color:#fff0a8}}
.image-area{{position:relative;aspect-ratio:1;background:#030304}}.preview-button{{width:100%;height:100%;padding:0;border:0;background:#030304}}.preview-button img{{display:block;width:100%;height:100%;object-fit:contain}}
.select-control{{position:absolute;top:7px;left:7px;width:30px;height:30px;padding:6px;border-radius:50%;background:rgba(0,0,0,.72)}}.select-control input{{position:absolute;opacity:0;pointer-events:none}}.select-control span{{display:block;width:18px;height:18px;border:2px solid white;border-radius:50%}}.select-control input:checked+span{{border-color:var(--blue);background:var(--blue)}}.select-control input:checked+span:after{{content:"";display:block;width:8px;height:4px;margin:4px 0 0 3px;border-left:2px solid white;border-bottom:2px solid white;transform:rotate(-45deg)}}
.status-badge{{position:absolute;top:7px;right:7px;max-width:70%;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:4px 6px;border-radius:4px;background:rgba(82,63,0,.9);color:#ffe680;font-size:11px;font-weight:700}}.reviewed .status-badge{{background:rgba(31,79,43,.9);color:#aff4bd}}
.item-body{{padding:9px}}.filename{{display:block;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.candidate-chips{{display:flex;gap:4px;flex-wrap:wrap;margin-top:7px}}.candidate-chip{{padding:3px 5px;border-radius:4px;background:#16354f;color:#d0e9ff;font-size:11px}}.candidate-chip b{{color:white}}.evidence{{font-size:11px;line-height:1.35;min-height:30px;margin-top:7px}}
.item-review{{border-top:1px solid var(--line)}}.item-review summary{{padding:8px 9px;color:#b9d9ff;cursor:pointer}}.review-body{{padding:0 9px 9px}}.review-body label{{color:var(--muted)}}input.person{{display:block;width:100%;height:35px;margin:5px 0 8px;padding:0 8px;border:1px solid var(--line);border-radius:5px;background:#0f1114;color:var(--text)}}.item-actions{{display:grid;grid-template-columns:1fr 1fr 1fr;gap:5px}}.negative-actions{{display:flex;gap:5px;flex-wrap:wrap;margin-top:7px}}
#toast{{display:none;position:fixed;right:18px;bottom:18px;max-width:460px;padding:12px 14px;border:1px solid var(--line);border-radius:7px;background:#22262b;z-index:40}}dialog{{width:min(92vw,1200px);height:min(92vh,920px);padding:0;border:1px solid #555;border-radius:8px;background:#050506;color:white}}dialog::backdrop{{background:rgba(0,0,0,.84)}}.viewer-bar{{height:48px;display:flex;align-items:center;justify-content:space-between;gap:12px;padding:0 12px;border-bottom:1px solid var(--line)}}.viewer-name{{overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}.viewer-image{{display:block;width:100%;height:calc(100% - 48px);object-fit:contain;background:#000}}.confirm-dialog{{width:min(92vw,520px);height:auto;padding:18px}}.confirm-dialog form{{margin:0}}.confirm-dialog p{{font-size:16px;line-height:1.45}}.confirm-actions{{display:flex;justify-content:flex-end;gap:8px;margin-top:18px}}
@media(max-width:760px){{header,main{{padding-left:11px;padding-right:11px}}.cluster-head,.batch-row{{display:block}}.cluster-actions,.batch-actions{{justify-content:flex-start;margin-top:9px}}.quick-nav{{flex-wrap:wrap}}.quick-nav .finish-button{{margin-left:0}}.custom-person{{width:100%}}.cluster-person{{min-width:0;flex:1}}.items{{grid-template-columns:repeat(auto-fill,minmax(148px,1fr));gap:7px}}body.comfortable .items{{grid-template-columns:repeat(auto-fill,minmax(220px,1fr))}}.item-actions{{grid-template-columns:1fr}}}}
</style></head><body class="quick-review"><header><div class="inner"><h1>Unknown Identity Quick Review</h1><p>{html.escape(mode_note)}</p>
<div class="metrics"><span class="metric">Reviewed <strong id="reviewedCount">{reviewed}</strong> of <strong id="totalCount">{total}</strong></span><span class="metric">Confirmed <strong id="confirmedCount">{confirmed}</strong></span><span class="metric">Unknown <strong id="unknownCount">{unknown}</strong></span><span class="metric">Junk <strong id="junkCount">{junk}</strong></span><span class="metric">Deferred <strong id="deferredCount">{deferred}</strong></span><span class="metric"><strong>{pending_clusters}</strong> clusters in this batch</span><span class="metric">Action queue <strong id="queueCount">0</strong></span><span class="metric">Batch <strong>{summary.get('batch_number', 1)}</strong> ({batch_files} files)</span><span class="metric">Cache <strong>{sqlite_hits}</strong> reused / <strong>{detected}</strong> analyzed</span><span class="metric">{html.escape(auto_status)}</span><span id="headerMessage" class="batch-suggestion"></span></div>
<div class="toolbar"><input id="search" class="search" type="search" placeholder="Search filename or suggested person"><select id="statusFilter"><option value="pending">Pending only</option><option value="all">All status</option><option value="reviewed">Reviewed only</option></select><select id="clusterFilter"><option value="all">All clusters</option><option value="group">Similar groups</option><option value="single">Single images</option></select><button id="selectVisible" class="secondary-button" type="button">Select visible</button><button id="clearSelection" class="secondary-button" type="button">Clear</button><div class="density-switch"><button class="active" type="button" data-density="compact">Compact</button><button type="button" data-density="comfortable">Large</button></div></div>
<div class="quick-nav"><button id="previousCluster" class="secondary-button" type="button" title="Previous cluster (Left Arrow)">Previous</button><strong id="clusterPosition" class="cluster-position">Cluster 0 of 0</strong><button id="nextCluster" class="secondary-button" type="button" title="Next cluster (Right Arrow)">Next</button><button id="skipCluster" class="secondary-button" type="button" title="Skip this cluster for this session (N)"><kbd>N</kbd> Skip</button><button id="loadNextBatch" class="secondary-button" type="button">Load Next Batch</button><span class="shortcut-help"><kbd>1</kbd> <kbd>2</kbd> <kbd>3</kbd> suggestions &nbsp; <kbd>U</kbd> unknown &nbsp; <kbd>J</kbd> junk</span><button id="finishReview" class="primary-button finish-button" type="button">Finish Review</button></div>
<section id="batchPanel" class="batch-panel"><div class="batch-row"><div><strong><span id="selectedCount">0</span> images selected</strong><span id="batchSuggestion" class="batch-suggestion"></span></div><div class="batch-actions"><input id="batchPerson" class="batch-person" list="people" placeholder="Choose person to confirm"><button id="useSuggestion" class="secondary-button" type="button" hidden></button><button id="confirmSelected" class="primary-button" type="button">Confirm selected</button><button id="keepSelected" class="secondary-button" type="button">Keep Unknown</button><button id="ignoreSelected" class="secondary-button" type="button">Defer</button></div></div></section>
</div></header><main>{''.join(cards) or '<p id="emptyBatch">No reviewable single-face unknowns in this batch.</p>'}</main>
<datalist id="people">{all_people}</datalist><div id="toast"></div><dialog id="viewer"><div class="viewer-bar"><strong id="viewerName" class="viewer-name"></strong><button id="closeViewer" class="secondary-button" type="button">Close</button></div><img id="viewerImage" class="viewer-image" alt="Full unknown image"></dialog><dialog id="confirmDialog" class="confirm-dialog"><form method="dialog"><h2>Finish Unknown Review?</h2><p id="confirmMessage"></p><div class="confirm-actions"><button class="secondary-button" value="cancel">Cancel</button><button class="primary-button" value="confirm">Finish Review</button></div></form></dialog>
<script>
const interactive={str(interactive).lower()};
const toast=document.getElementById('toast');
const queueCount=document.getElementById('queueCount');
const search=document.getElementById('search');
const statusFilter=document.getElementById('statusFilter');
const clusterFilter=document.getElementById('clusterFilter');
const batchPanel=document.getElementById('batchPanel');
const batchPerson=document.getElementById('batchPerson');
const useSuggestion=document.getElementById('useSuggestion');
const activeJobIds=new Set(JSON.parse(sessionStorage.getItem('unknownJobs')||'[]'));
const temporarilySkipped=new Set();
let activeIndex=0,lastChecked=null,polling=false,loadingBatch=false;
const allClusters=()=>[...document.querySelectorAll('.cluster')];
const cards=()=>[...document.querySelectorAll('.item')];
const checks=()=>[...document.querySelectorAll('.row-select')];
const selected=()=>checks().filter(input=>input.checked);
const availableClusters=()=>allClusters().filter(cluster=>
  cluster.dataset.pending==='1'&&!cluster.classList.contains('hidden')&&!temporarilySkipped.has(cluster.dataset.cluster)
);

function notify(message,error=false){{
  toast.textContent=message;toast.style.display='block';toast.style.borderColor=error?'var(--red)':'var(--line)';
  setTimeout(()=>toast.style.display='none',6000);
}}
function askConfirmation(message){{
  const dialog=document.getElementById('confirmDialog');
  document.getElementById('confirmMessage').textContent=message;dialog.returnValue='cancel';
  return new Promise(resolve=>{{dialog.addEventListener('close',()=>resolve(dialog.returnValue==='confirm'),{{once:true}});dialog.showModal();}});
}}
function persistJobs(){{sessionStorage.setItem('unknownJobs',JSON.stringify([...activeJobIds]));}}
function activeCluster(){{return document.querySelector('.cluster.quick-active');}}
function showCluster(index=activeIndex){{
  const available=availableClusters();allClusters().forEach(cluster=>cluster.classList.remove('quick-active'));
  if(!available.length){{document.getElementById('clusterPosition').textContent='No pending cluster';maybeLoadNextBatch();return;}}
  activeIndex=((index%available.length)+available.length)%available.length;
  available[activeIndex].classList.add('quick-active');
  document.getElementById('clusterPosition').textContent=`Cluster ${{activeIndex+1}} of ${{available.length}}`;
  window.scrollTo({{top:0,behavior:'smooth'}});
}}
function moveCluster(delta){{showCluster(activeIndex+delta);}}
function updateSelection(){{
  cards().forEach(card=>{{const checkbox=card.querySelector('.row-select');card.classList.toggle('selected',Boolean(checkbox&&checkbox.checked));}});
  const chosen=selected();document.getElementById('selectedCount').textContent=chosen.length;batchPanel.classList.toggle('active',chosen.length>0);
  const best=[...new Set(chosen.map(input=>input.closest('.item').dataset.best).filter(Boolean))];
  const suggestion=best.length===1?best[0]:'';
  document.getElementById('batchSuggestion').textContent=suggestion?` · common suggestion: ${{suggestion}}`:best.length>1?' · mixed suggestions':'';
  useSuggestion.hidden=!suggestion;useSuggestion.textContent=suggestion?`Use ${{suggestion}}`:'';useSuggestion.dataset.person=suggestion;
}}
function applyFilters(){{
  const query=search.value.trim().toLocaleLowerCase(),status=statusFilter.value,kind=clusterFilter.value;
  allClusters().forEach(cluster=>{{
    const statusAllowed=status==='all'||(status==='pending'&&cluster.dataset.pending==='1')||(status==='reviewed'&&cluster.dataset.pending==='0');
    const allowed=(kind==='all'||cluster.dataset.kind===kind)&&statusAllowed&&(!query||cluster.dataset.search.includes(query));
    cluster.classList.toggle('hidden',!allowed);
  }});
  localStorage.setItem('unknownSearch',search.value);localStorage.setItem('unknownStatus',status);localStorage.setItem('unknownCluster',kind);
  activeIndex=0;showCluster(0);
}}
function markJobCards(job){{
  const label=job.status==='running'?'Working':job.position>1?`Queued #${{job.position}}`:'Queued';
  job.item_keys.forEach(key=>{{const card=document.querySelector(`.item[data-item="${{key}}"]`);if(!card)return;card.classList.add('processing');const checkbox=card.querySelector('.row-select');if(checkbox){{checkbox.checked=false;checkbox.disabled=true;}}const badge=card.querySelector('.status-badge');if(badge)badge.textContent=label;}});
  const first=document.querySelector(`.item[data-item="${{job.item_keys[0]}}"]`);const cluster=first?.closest('.cluster');
  if(cluster){{cluster.dataset.pending='0';cluster.querySelectorAll('button,input').forEach(control=>control.disabled=true);}}
  updateSelection();
}}
function optimisticProgress(action,count){{
  const increment=id=>{{const node=document.getElementById(id);node.textContent=String(Number(node.textContent||0)+count);}};
  increment('reviewedCount');
  if(action==='confirm')increment('confirmedCount');else if(action==='keep_unknown')increment('unknownCount');else if(action==='move_to_junk')increment('junkCount');else increment('deferredCount');
}}
async function refreshProgress(){{
  try{{const response=await fetch('/progress');const result=await response.json();if(!response.ok)return;for(const [key,id] of Object.entries({{reviewed:'reviewedCount',total:'totalCount',confirmed:'confirmedCount',unknown:'unknownCount',junk:'junkCount',deferred:'deferredCount'}})){{document.getElementById(id).textContent=result.progress[key]??0;}}}}catch(_error){{}}
}}
function applyResolvedItems(keys){{
  keys.forEach(key=>{{const card=document.querySelector(`.item[data-item="${{key}}"]`);if(!card)return;card.classList.add('processing');card.querySelectorAll('button,input').forEach(control=>{{control.disabled=true;if(control.type==='checkbox')control.checked=false;}});const badge=card.querySelector('.status-badge');if(badge)badge.textContent='Decision saved';}});
  allClusters().forEach(cluster=>{{if(!cluster.querySelector('.row-select:not(:disabled)'))cluster.dataset.pending='0';}});
  updateSelection();
}}
async function pollJobs(){{
  if(polling)return;polling=true;let repeat=false,failed=false;
  try{{
    const ids=[...activeJobIds];const jobsURL=ids.length?'/jobs?'+new URLSearchParams({{ids:ids.join(',')}}):'/jobs';const response=await fetch(jobsURL);const result=await response.json();
    if(!response.ok)throw new Error(result.error||'Could not read action queue');
    queueCount.textContent=(result.summary.queued||0)+(result.summary.running||0);
    applyResolvedItems(result.resolved_item_keys||[]);
    const returnedJobIds=new Set(result.jobs.map(job=>job.id));
    ids.filter(id=>!returnedJobIds.has(id)).forEach(id=>activeJobIds.delete(id));
    result.jobs.forEach(job=>{{if(job.status==='queued'||job.status==='running'){{activeJobIds.add(job.id);markJobCards(job);}}else{{activeJobIds.delete(job.id);if(job.status==='failed'){{failed=true;notify(job.error||'Review action failed',true);}}else if(job.message)notify(job.message);}}}});
    persistJobs();repeat=activeJobIds.size>0||(result.summary.queued||0)+(result.summary.running||0)>0;
    if(failed){{setTimeout(()=>location.reload(),900);return;}}
    if(!repeat){{await refreshProgress();maybeLoadNextBatch();}}
  }}catch(error){{notify(error.message,true);repeat=true;}}
  finally{{polling=false;if(repeat)setTimeout(pollJobs,700);}}
}}
async function decide(action,keys,person=''){{
  if(!interactive){{notify('Launch with face unknown-review to use actions.',true);return;}}
  const values=Array.isArray(keys)?keys:keys.split(',').filter(Boolean);
  if(!values.length){{notify('This cluster has no pending images.',true);return;}}
  if(values.length>{MAX_CLUSTER_ACTION_ITEMS}){{notify('Select no more than {MAX_CLUSTER_ACTION_ITEMS} images.',true);return;}}
  if(action==='confirm'&&!person.trim()){{notify('Enter or choose a person.',true);return;}}
  try{{
    const response=await fetch('/decide',{{method:'POST',headers:{{'Content-Type':'application/x-www-form-urlencoded'}},body:new URLSearchParams({{action,person,item_keys:values.join(',')}})}});
    const result=await response.json();if(!response.ok)throw new Error(result.error||'Action failed');
    const jobs=result.jobs||[result.job];jobs.forEach(job=>{{activeJobIds.add(job.id);markJobCards(job);}});persistJobs();optimisticProgress(action,values.length);
    const firstJob=jobs[0];notify(jobs.length>1?`Queued safely in ${{jobs.length}} verified chunks.`:firstJob.position>1?`Queued safely at position ${{firstJob.position}}.`:'Action started.');showCluster(activeIndex);pollJobs();
  }}catch(error){{notify(error.message,true);}}
}}
async function skipCurrent(){{
  const cluster=activeCluster();if(!cluster)return;const keys=cluster.dataset.items.split(',').filter(Boolean);
  try{{const response=await fetch('/skip',{{method:'POST',headers:{{'Content-Type':'application/x-www-form-urlencoded'}},body:new URLSearchParams({{item_keys:keys.join(',')}})}});const result=await response.json();if(!response.ok)throw new Error(result.error||'Could not skip cluster');temporarilySkipped.add(cluster.dataset.cluster);notify('Skipped for this session.');showCluster(activeIndex);}}catch(error){{notify(error.message,true);}}
}}
async function maybeLoadNextBatch(manual=false){{
  if(!interactive||loadingBatch)return;
  if(availableClusters().length){{if(manual)notify('Finish or skip the current pending clusters first.');return;}}
  loadingBatch=true;document.getElementById('loadNextBatch').disabled=true;document.getElementById('finishReview').disabled=true;document.getElementById('clusterPosition').textContent='Loading next batch...';
  try{{const response=await fetch('/next-batch',{{method:'POST'}});const result=await response.json();if(response.status===409){{document.getElementById('clusterPosition').textContent=result.error||'Waiting for current operation...';loadingBatch=false;setTimeout(()=>maybeLoadNextBatch(),700);return;}}if(!response.ok)throw new Error(result.error||'Could not load next batch');pollBatchStatus();}}catch(error){{loadingBatch=false;document.getElementById('loadNextBatch').disabled=false;document.getElementById('finishReview').disabled=false;notify(error.message,true);document.getElementById('clusterPosition').textContent='Load failed';}}
}}
async function pollBatchStatus(){{
  try{{const response=await fetch('/batch-status');const result=await response.json();if(!response.ok)throw new Error(result.error||'Could not read batch progress');document.getElementById('clusterPosition').textContent=result.step||'Loading next batch...';if(result.status==='completed'){{loadingBatch=false;if(result.complete){{document.getElementById('clusterPosition').textContent='Queue complete';document.getElementById('loadNextBatch').disabled=false;document.getElementById('finishReview').disabled=false;notify('All pending unknown files are handled. Finish Review when ready.');}}else{{location.reload();}}return;}}if(result.status==='failed'){{loadingBatch=false;document.getElementById('loadNextBatch').disabled=false;document.getElementById('finishReview').disabled=false;document.getElementById('clusterPosition').textContent='Load failed';notify(result.message||'Could not load next batch',true);return;}}setTimeout(pollBatchStatus,500);}}catch(error){{document.getElementById('clusterPosition').textContent='Reconnecting to batch loader...';setTimeout(pollBatchStatus,1000);}}
}}
async function pollFinish(){{
  try{{const response=await fetch('/finish-status');const result=await response.json();if(!response.ok)throw new Error(result.error||'Could not read finish status');const button=document.getElementById('finishReview');button.textContent=result.step||'Finishing...';if(result.status==='completed'){{button.textContent='Review Finished';notify(`${{result.message}} Report: ${{result.report}}`);await refreshProgress();return;}}if(result.status==='failed'){{button.disabled=false;button.textContent='Finish Review';notify(result.message||'Finish failed safely',true);return;}}setTimeout(pollFinish,1000);}}catch(error){{notify(error.message,true);setTimeout(pollFinish,1500);}}
}}
async function finishReview(){{
  if(loadingBatch){{notify('Wait for the next batch to finish loading.',true);return;}}
  if(!await askConfirmation('Finish Review will wait for queued actions, save the cache, refresh identity profiles once, and run the safety benchmark. Continue?'))return;
  const button=document.getElementById('finishReview');button.disabled=true;button.textContent='Finishing...';
  try{{const response=await fetch('/finish',{{method:'POST'}});const result=await response.json();if(!response.ok)throw new Error(result.error||'Could not start Finish Review');pollFinish();}}catch(error){{button.disabled=false;button.textContent='Finish Review';notify(error.message,true);}}
}}

checks().forEach(input=>input.addEventListener('click',event=>{{if(event.shiftKey&&lastChecked){{const visible=checks().filter(item=>!item.disabled&&item.closest('.cluster')===activeCluster());const a=visible.indexOf(lastChecked),b=visible.indexOf(input);if(a>=0&&b>=0)visible.slice(Math.min(a,b),Math.max(a,b)+1).forEach(item=>item.checked=input.checked);}}lastChecked=input;updateSelection();}}));
[search,statusFilter,clusterFilter].forEach(control=>control.addEventListener('input',applyFilters));
document.getElementById('previousCluster').addEventListener('click',()=>moveCluster(-1));document.getElementById('nextCluster').addEventListener('click',()=>moveCluster(1));document.getElementById('skipCluster').addEventListener('click',skipCurrent);document.getElementById('loadNextBatch').addEventListener('click',()=>maybeLoadNextBatch(true));document.getElementById('finishReview').addEventListener('click',finishReview);
document.getElementById('selectVisible').addEventListener('click',()=>{{const cluster=activeCluster();if(!cluster)return;cluster.querySelectorAll('.row-select:not(:disabled)').forEach(input=>input.checked=true);updateSelection();}});document.getElementById('clearSelection').addEventListener('click',()=>{{checks().forEach(input=>input.checked=false);updateSelection();}});
document.querySelectorAll('.select-cluster').forEach(button=>button.addEventListener('click',()=>{{button.closest('.cluster').querySelectorAll('.row-select:not(:disabled)').forEach(input=>input.checked=true);updateSelection();}}));
document.querySelectorAll('[data-density]').forEach(button=>button.addEventListener('click',()=>{{const comfortable=button.dataset.density==='comfortable';document.body.classList.toggle('comfortable',comfortable);document.querySelectorAll('[data-density]').forEach(item=>item.classList.toggle('active',item===button));localStorage.setItem('unknownDensity',button.dataset.density);}}));
const viewer=document.getElementById('viewer');document.querySelectorAll('.preview-button').forEach(button=>button.addEventListener('click',()=>{{document.getElementById('viewerImage').src=button.dataset.full;document.getElementById('viewerName').textContent=button.dataset.name;viewer.showModal();}}));document.getElementById('closeViewer').addEventListener('click',()=>viewer.close());
document.querySelectorAll('button[data-action]').forEach(button=>button.addEventListener('click',()=>{{const cluster=button.closest('.cluster'),item=button.closest('.item'),scope=button.dataset.scope,keys=scope==='cluster'?cluster.dataset.items:item.dataset.item,person=button.dataset.person||(item?.querySelector('input.person')?.value||'');decide(button.dataset.action,keys,person);}}));
document.querySelectorAll('.confirm-custom').forEach(button=>button.addEventListener('click',()=>{{const cluster=button.closest('.cluster'),input=cluster.querySelector('.cluster-person');decide('confirm',cluster.dataset.items,input.value);}}));document.querySelectorAll('.cluster-person').forEach(input=>input.addEventListener('keydown',event=>{{if(event.key==='Enter'){{event.preventDefault();input.closest('.cluster').querySelector('.confirm-custom').click();}}}}));
document.getElementById('confirmSelected').addEventListener('click',()=>decide('confirm',selected().map(input=>input.value),batchPerson.value));document.getElementById('keepSelected').addEventListener('click',()=>decide('keep_unknown',selected().map(input=>input.value)));document.getElementById('ignoreSelected').addEventListener('click',()=>decide('ignore',selected().map(input=>input.value)));useSuggestion.addEventListener('click',()=>{{batchPerson.value=useSuggestion.dataset.person||'';}});
document.addEventListener('keydown',event=>{{
  if(event.metaKey||event.ctrlKey||event.altKey||['INPUT','TEXTAREA','SELECT'].includes(document.activeElement?.tagName)||document.querySelector('dialog[open]'))return;
  const cluster=activeCluster();if(!cluster)return;
  if(['1','2','3'].includes(event.key)){{event.preventDefault();cluster.querySelector(`button[data-shortcut="${{event.key}}"]`)?.click();}}
  else if(event.key.toLocaleLowerCase()==='u'){{event.preventDefault();cluster.querySelector('button[data-action="keep_unknown"]')?.click();}}
  else if(event.key.toLocaleLowerCase()==='j'){{event.preventDefault();cluster.querySelector('button[data-action="move_to_junk"]')?.click();}}
  else if(event.key.toLocaleLowerCase()==='n'){{event.preventDefault();skipCurrent();}}
  else if(event.key==='ArrowRight'){{event.preventDefault();moveCluster(1);}}
  else if(event.key==='ArrowLeft'){{event.preventDefault();moveCluster(-1);}}
}});
search.value=localStorage.getItem('unknownSearch')||'';statusFilter.value=localStorage.getItem('unknownStatus')||'pending';clusterFilter.value=localStorage.getItem('unknownCluster')||'all';document.querySelector(`[data-density="${{localStorage.getItem('unknownDensity')||'compact'}}"]`)?.click();applyFilters();updateSelection();pollJobs();
</script></body></html>"""


def make_handler(state: dict):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args) -> None:
            if not state.get("quiet"):
                super().log_message(fmt, *args)

        def send_bytes(
            self,
            data: bytes,
            content_type: str,
            status: int = 200,
            *,
            cache_control: str = "no-store",
        ) -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", cache_control)
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
            if parsed.path == "/jobs":
                job_ids = [
                    value for value in parse_qs(parsed.query).get("ids", [""])[0].split(",")
                    if value
                ]
                snapshot = state["action_queue"].snapshot(
                    job_ids or None,
                    active_only=not job_ids,
                )
                if (not snapshot["summary"].get("running")
                        and not snapshot["summary"].get("queued")
                        and state["lock"].acquire(blocking=False)):
                    try:
                        decisions = load_decisions(state["decisions_path"])
                        snapshot["resolved_item_keys"] = [
                            key for key, item in state["items_by_key"].items()
                            if decision_for_item(decisions, item).get("action") in RESOLVED_ACTIONS
                        ]
                    finally:
                        state["lock"].release()
                self.send_json(snapshot)
                return
            if parsed.path == "/progress":
                self.send_json({"progress": review_progress(state)})
                return
            if parsed.path == "/finish-status":
                self.send_json(state["finish_job"].snapshot())
                return
            if parsed.path == "/batch-status":
                self.send_json(state["batch_job"].snapshot())
                return
            if parsed.path in {"/face", "/image"}:
                self.serve_image(parsed.path, parsed.query)
                return
            if parsed.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            with state["lock"]:
                decisions = load_decisions(state["decisions_path"])
                page = render_html(
                    state["clusters"],
                    decisions,
                    state["identity_names"],
                    state["summary"],
                    interactive=True,
                )
            self.send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")

        def serve_image(self, route: str, query: str) -> None:
            key = parse_qs(query).get("item", [""])[0]
            item = state["items_by_key"].get(key)
            if item is None:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if route == "/face" and item.face.crop_jpeg:
                self.send_bytes(
                    item.face.crop_jpeg,
                    "image/jpeg",
                    cache_control="private, max-age=300",
                )
                return
            if not item.path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            image_path = item.path
            try:
                data = image_path.read_bytes()
            except OSError:
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            content_type = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
            self.send_bytes(data, content_type, cache_control="private, max-age=300")

        def do_POST(self) -> None:
            route = urlparse(self.path).path
            if route == "/next-batch":
                payload, conflict = state["batch_job"].start()
                if conflict:
                    self.send_json({"error": conflict, **payload}, HTTPStatus.CONFLICT)
                    return
                self.send_json(payload, HTTPStatus.ACCEPTED)
                return
            if route == "/finish":
                with state["lifecycle_lock"]:
                    if state.get("batch_loading"):
                        self.send_json(
                            {"error": "Wait for the next batch to finish loading."},
                            HTTPStatus.CONFLICT,
                        )
                        return
                    self.send_json(state["finish_job"].start(), HTTPStatus.ACCEPTED)
                return
            if route == "/skip":
                if state.get("batch_loading") or state.get("finishing"):
                    self.send_json(
                        {"error": "Wait for the current review operation to finish."},
                        HTTPStatus.CONFLICT,
                    )
                    return
                try:
                    length = min(int(self.headers.get("Content-Length", "0")), 64 * 1024)
                    params = parse_qs(self.rfile.read(length).decode("utf-8"))
                    keys = {
                        value for value in params.get("item_keys", [""])[0].split(",")
                        if value and value in state["items_by_key"]
                    }
                    if not keys:
                        raise ValueError("no available cluster items to skip")
                    state.setdefault("temporarily_skipped", set()).update(keys)
                    self.send_json({"skipped": len(keys)})
                except Exception as error:  # noqa: BLE001
                    self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            if route != "/decide":
                self.send_error(HTTPStatus.NOT_FOUND)
                return
            if state.get("finishing") or state.get("review_finished"):
                self.send_json(
                    {"error": "Review is being finalized; no new actions are accepted."},
                    HTTPStatus.CONFLICT,
                )
                return
            if state.get("batch_loading"):
                self.send_json(
                    {"error": "Wait for the next batch to finish loading."},
                    HTTPStatus.CONFLICT,
                )
                return
            try:
                length = min(int(self.headers.get("Content-Length", "0")), 64 * 1024)
                params = parse_qs(self.rfile.read(length).decode("utf-8"))
                jobs = state["action_queue"].submit_many(
                    item_keys=[
                        value for value in params.get("item_keys", [""])[0].split(",")
                        if value
                    ],
                    action=params.get("action", [""])[0],
                    person_value=params.get("person", [""])[0],
                )
            except ReviewActionConflict as error:
                self.send_json({"error": str(error)}, HTTPStatus.CONFLICT)
                return
            except Exception as error:  # noqa: BLE001
                self.send_json({"error": str(error)}, HTTPStatus.BAD_REQUEST)
                return
            self.send_json({"job": jobs[0], "jobs": jobs}, HTTPStatus.ACCEPTED)

    return Handler


def run_automatic_sweep_worker(task_path: Path) -> int:
    """Execute one bounded auto-sweep task in a fresh interpreter."""
    try:
        task = json.loads(task_path.expanduser().read_text(encoding="utf-8"))
        result_path = Path(task["result_path"])
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"ERROR: invalid automatic sweep task: {error}")
        return 2
    try:
        identity_db = sort_photos.load_identity_db()
        if identity_db is None or not identity_db.identities:
            raise RuntimeError("identity database is unavailable")
        identity_db = sort_photos.normalize_identity_db(identity_db)
        cache = sort_photos.load_cache()
        secondary_db = secondary_identity_matcher.load()
        if (
            secondary_db is None
            or secondary_db.primary_signature
            != secondary_identity_matcher.primary_signature(identity_db)
            or not secondary_identity_matcher.trusted_snapshot_is_current(
                secondary_db, sort_photos.IDENTITY_CONFIRMATIONS_FILE
            )
        ):
            raise RuntimeError("independent verifier is unavailable or out of date")
        review_prototypes, _counts = build_trusted_review_prototypes(
            identity_db,
            cache,
            confirmations_path=sort_photos.IDENTITY_CONFIRMATIONS_FILE,
        )
        people_root = Path(task["people_root"])
        existing_hashes = recover_no_usable_faces.load_existing_hashes(people_root)
        known_destinations: dict[tuple[str, str], Path] = {}
        for person_key, digest, destination_text in task.get("known_destinations", []):
            destination = Path(destination_text)
            if not destination.is_file():
                continue
            known_destinations[(str(person_key), str(digest))] = destination
            try:
                person = destination.resolve().relative_to(people_root.resolve()).parts[0]
            except (OSError, ValueError, IndexError):
                continue
            existing_hashes[person].add(str(digest))
        worker_state = {
            "clusters": [],
            "items_by_key": {},
            "identity_db": identity_db,
            "identity_names": sorted(identity_db.identities, key=str.casefold),
            "people_root": people_root,
            "unknown_root": Path(task["unknown_root"]),
            "unassigned_root": Path(task["unassigned_root"]),
            "output_dir": Path(task["output_dir"]),
            "decisions_path": Path(task["decisions_path"]),
            "review_dir": Path(task["review_dir"]),
            "junk_dir": Path(task["junk_dir"]),
            "confirmations_path": sort_photos.IDENTITY_CONFIRMATIONS_FILE,
            "evaluation_path": evaluation_enrollment.DEFAULT_PATH,
            "hard_negatives_path": sort_photos.IDENTITY_HARD_NEGATIVES_FILE,
            "hard_negatives": identity_hard_negatives.vectors_by_person(
                sort_photos.IDENTITY_HARD_NEGATIVES_FILE
            ),
            "model_signature": review_model_signature(
                identity_db, sort_photos.IDENTITY_HARD_NEGATIVES_FILE
            ),
            "review_prototypes": review_prototypes,
            "secondary_matcher": secondary_identity_matcher.SecondaryMatcher(secondary_db),
            "analysis_index": Path(task["analysis_index"]),
            "existing_hashes": existing_hashes,
            "destinations_by_hash": known_destinations,
            "next_indexes": {},
            "cache": cache,
            "cache_dirty": False,
            "identity_dirty": False,
            "workers": int(task["workers"]),
            "primary_det_size": int(task["primary_det_size"]),
            "fallback_det_size": int(task["fallback_det_size"]),
            "cluster_eps": float(task["cluster_eps"]),
            "route_unsupported": bool(task.get("route_unsupported", True)),
            "auto_review_allowed": True,
            "auto_review_requested": True,
            "auto_review_preview": False,
            "capture_cache_delta": True,
            "_cache_delta_removed_sources": set(),
            "_cache_delta_entries": [],
        }
        payload = _run_automatic_sweep_batch(
            worker_state,
            [Path(value) for value in task.get("files", []) if Path(value).is_file()],
        )
        if worker_state.get("cache_dirty"):
            delta_path = Path(task["output_dir"]) / (
                f".auto_sweep.cache.{uuid.uuid4().hex}.pkl"
            )
            temporary_delta = delta_path.with_suffix(delta_path.suffix + ".tmp")
            with temporary_delta.open("wb") as handle:
                pickle.dump(
                    {
                        "removed_sources": sorted(
                            worker_state.get("_cache_delta_removed_sources", set())
                        ),
                        "entries": worker_state.get("_cache_delta_entries", []),
                    },
                    handle,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
                handle.flush()
                os.fsync(handle.fileno())
            temporary_delta.replace(delta_path)
            payload["cache_delta"] = str(delta_path)
    except BaseException as error:  # noqa: BLE001
        payload = {"error": f"{type(error).__name__}: {error}"}

    try:
        result_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = result_path.with_suffix(result_path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(result_path)
    except OSError as error:
        print(f"ERROR: cannot write automatic sweep result: {error}")
        return 2
    return 1 if payload.get("error") else 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_UNKNOWN_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--decisions", type=Path, default=DEFAULT_DECISIONS)
    parser.add_argument("--max-files", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--primary-det-size", type=int, default=1024)
    parser.add_argument("--fallback-det-size", type=int, default=384)
    parser.add_argument("--cluster-eps", type=float, default=0.30)
    parser.add_argument("--serve", action="store_true")
    parser.add_argument("--open", action="store_true")
    parser.add_argument(
        "--auto-safe",
        action="store_true",
        help="Automatically file only benchmark-gated, independently verified matches.",
    )
    parser.add_argument(
        "--auto-safe-preview",
        action="store_true",
        help="Report safe automatic matches without moving any files.",
    )
    parser.add_argument(
        "--reconcile-only",
        action="store_true",
        help="Repair stale completed outcomes and exit before face analysis.",
    )
    parser.add_argument(
        "--reconcile-preview",
        action="store_true",
        help="Report stale-outcome repairs without moving or rewriting anything.",
    )
    parser.add_argument(
        "--reprocess-pending",
        action="store_true",
        help="Run benchmark-gated recovery across every pending file without serving the UI.",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8772)
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--auto-sweep-worker", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.auto_sweep_worker is not None:
        return run_automatic_sweep_worker(args.auto_sweep_worker)
    unknown_root = args.input.expanduser().resolve()
    if not unknown_root.is_dir():
        print(f"ERROR: unknown_identity folder is unavailable: {unknown_root}")
        return 2
    identity_db = sort_photos.load_identity_db()
    if identity_db is None or not identity_db.identities:
        print("ERROR: identity database is unavailable. Run `face rebuild-id` first.")
        return 2
    identity_db = sort_photos.normalize_identity_db(identity_db)
    hard_negatives_path = sort_photos.IDENTITY_HARD_NEGATIVES_FILE
    hard_negatives = identity_hard_negatives.vectors_by_person(hard_negatives_path)
    auto_requested = bool(
        args.auto_safe or args.auto_safe_preview or args.reprocess_pending
    )
    cache = sort_photos.load_cache()
    print("Preparing shared confirmed-reference index...", flush=True)
    reference_index = identity_confirmations.ReferenceIndex(faces_by_source=_faces_by_source(cache))
    secondary_matcher, secondary_status = prepare_secondary_verifier(
        identity_db,
        cache,
        requested=auto_requested,
        reference_index=reference_index,
    )
    review_prototypes, trusted_review_counts = build_trusted_review_prototypes(
        identity_db,
        cache,
        confirmations_path=sort_photos.IDENTITY_CONFIRMATIONS_FILE,
        reference_index=reference_index,
    )
    del reference_index
    decisions_path = args.decisions.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    analysis_index_path = sort_photos.analysis_index_file()
    relink_stats = evaluation_enrollment.relink_missing_sources(
        analysis_index_path=analysis_index_path,
        people_root=pipeline_paths.PEOPLE_ROOT,
        path=evaluation_enrollment.DEFAULT_PATH,
    )
    learned_hard_negatives = learn_hard_negatives_from_confirmed_reviews(
        identity_db,
        cache,
        secondary_matcher,
        hard_negatives_path=hard_negatives_path,
        review_prototypes=review_prototypes,
        state_path=output_dir / "confirmed_review_learning_state.json",
    ) if auto_requested else {"available": False, "learned": 0}
    if int(learned_hard_negatives.get("learned", 0)):
        hard_negatives = identity_hard_negatives.vectors_by_person(
            hard_negatives_path
        )
    auto_allowed, auto_gate, auto_gate_message = prepare_automatic_review_gate(
        identity_db,
        cache,
        secondary_matcher,
        requested=auto_requested,
        output_dir=output_dir,
        review_prototypes=review_prototypes,
    )
    # The full-library gate creates large temporary NumPy matrices. Release
    # them before the backlog sweep starts so review batches stay bounded on
    # 16 GB Macs.
    gc.collect()
    if auto_requested and secondary_matcher is None:
        auto_gate_message = f"Safe auto-match is off because verifier {secondary_status}."
    state = {
        "clusters": [],
        "items_by_key": {},
        "identity_db": identity_db,
        "identity_names": sorted(identity_db.identities, key=str.casefold),
        "people_root": pipeline_paths.PEOPLE_ROOT,
        "unknown_root": unknown_root,
        "unassigned_root": unknown_root.parent,
        "output_dir": output_dir,
        "session_path": output_dir / DEFAULT_SESSION.name,
        "summary": {},
        "decisions_path": decisions_path,
        "review_dir": pipeline_paths.SOURCE_REVIEW
        / "ready_to_delete"
        / "confirmed_unknown_identity_dashboard",
        "junk_dir": pipeline_paths.SOURCE_REVIEW
        / "ready_to_delete"
        / "unknown_identity_junk_dashboard",
        "replay_dir": pipeline_paths.SOURCE_REVIEW
        / "ready_to_delete"
        / "confirmed_unknown_identity_replays",
        "confirmations_path": sort_photos.IDENTITY_CONFIRMATIONS_FILE,
        "evaluation_path": evaluation_enrollment.DEFAULT_PATH,
        "hard_negatives_path": hard_negatives_path,
        "hard_negatives": hard_negatives,
        "model_signature": review_model_signature(
            identity_db, hard_negatives_path
        ),
        "review_prototypes": review_prototypes,
        "trusted_review_counts": trusted_review_counts,
        "secondary_matcher": secondary_matcher,
        "analysis_index": analysis_index_path,
        "existing_hashes": (
            {}
            if args.reconcile_only or args.reconcile_preview
            else recover_no_usable_faces.load_existing_hashes(pipeline_paths.PEOPLE_ROOT)
        ),
        "destinations_by_hash": {},
        "next_indexes": {},
        "cache": cache,
        "cache_dirty": False,
        "identity_dirty": False,
        "temporarily_skipped": set(),
        "batch_number": 0,
        "batch_limit": max(1, int(args.max_files)),
        "workers": max(1, int(args.workers)),
        "primary_det_size": max(320, int(args.primary_det_size)),
        "fallback_det_size": max(320, int(args.fallback_det_size)),
        "cluster_eps": float(args.cluster_eps),
        "route_unsupported": bool(args.serve or args.reprocess_pending),
        "auto_review_requested": auto_requested,
        "auto_review_allowed": bool(auto_allowed),
        "auto_review_preview": bool(args.auto_safe_preview),
        "auto_review_gate": auto_gate,
        "auto_review_gate_message": auto_gate_message,
        "learned_hard_negatives": learned_hard_negatives,
        "isolate_auto_sweep_batches": bool(args.serve or args.reprocess_pending),
        "batch_loading": False,
        "finishing": False,
        "review_finished": False,
        "lifecycle_lock": threading.Lock(),
        "lock": threading.Lock(),
        "quiet": args.quiet,
    }
    reconciliation = reconcile_unknown_queue(
        state,
        dry_run=bool(args.reconcile_preview),
    )
    state["reconciliation"] = dict(reconciliation)
    if state.get("cache_dirty"):
        sort_photos.save_cache(state["cache"])
        state["cache_dirty"] = False
    decisions = load_decisions(decisions_path)
    initial_progress = review_progress(state, decisions)

    print("Unknown Identity Quick Review")
    print("=" * 60)
    print(f"Total tracked:       {initial_progress['total']}")
    print(f"Previously reviewed: {initial_progress['reviewed']}")
    print(f"Pending:             {initial_progress['pending']}")
    if sum(reconciliation.values()):
        print(
            "Reconciliation:      "
            + ", ".join(
                f"{key.replace('_', ' ')}={value}"
                for key, value in sorted(reconciliation.items())
                if value
            )
        )
    print(f"Batch limit:         {state['batch_limit']}")
    print(f"Safe auto-match:     {auto_gate_message}")
    print(
        "Trusted review memory: "
        f"{sum(trusted_review_counts.values())} diverse examples across "
        f"{len(trusted_review_counts)} people"
    )
    if relink_stats["missing"]:
        print(
            "Benchmark paths:    "
            f"{relink_stats['relinked']} relinked, {relink_stats['unresolved']} unresolved"
        )
    if args.reconcile_only or args.reconcile_preview:
        return 1 if reconciliation.get("failed", 0) else 0
    auto_sweep = run_automatic_sweep(state)
    if int(auto_sweep.get("scanned", 0)) and not auto_sweep.get("cached"):
        print(
            "Full safe auto-sweep: "
            f"{auto_sweep['confirmed']} filed from {auto_sweep['scanned']} scanned; "
            f"{auto_sweep['failed']} failed"
        )
    state["auto_sweep_summary"] = auto_sweep
    load_next_batch(state)
    html_path = output_dir / "unknown_identity_review.html"
    print(f"Reviewable faces:    {len(state['items_by_key'])}")
    print(f"Clusters/cards:      {len(state['clusters'])}")
    print(f"SQLite reused:       {state['summary'].get('sqlite_hits', 0)}")
    print(f"Newly analyzed:      {state['summary'].get('detected', 0)}")
    print(
        "Automatic review:    "
        f"{auto_sweep.get('confirmed', 0)} filed from "
        f"{auto_sweep.get('scanned', 0)} full-queue checks"
        + (" (already current)" if auto_sweep.get("cached") else "")
        + (" (preview only)" if args.auto_safe_preview else "")
    )
    print(f"Static report:       {html_path}")
    if not args.serve:
        if args.open:
            webbrowser.open(html_path.resolve().as_uri())
        return 0

    state["action_queue"] = ReviewActionQueue(state)
    state["batch_job"] = BatchLoadJob(state)
    state["finish_job"] = FinishReviewJob(state)
    try:
        server = ThreadingHTTPServer((args.host, int(args.port)), make_handler(state))
    except OSError:
        server = ThreadingHTTPServer((args.host, 0), make_handler(state))
    url = f"http://{args.host}:{server.server_address[1]}/"
    print(f"Live review URL:     {url}")
    print("Keys: 1/2/3 confirm suggestion | U unknown | J junk | N skip")
    print("Use Finish Review in the page when done; Ctrl+C remains a safe fallback.")
    if args.open:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping unknown identity review...")
    finally:
        server.server_close()
        batch_worker = state["batch_job"].worker
        if batch_worker is not None and batch_worker.is_alive():
            print("Waiting for the next review batch to finish safely...")
            batch_worker.join()
        finish_worker = state["finish_job"].worker
        if finish_worker is not None and finish_worker.is_alive():
            print("Waiting for Finish Review to complete safely...")
            finish_worker.join()
        state["action_queue"].close(wait=True)
        if state["cache_dirty"]:
            sort_photos.save_cache(state["cache"])
            print("Updated face cache saved.")
        if state["identity_dirty"]:
            print("Refreshing identity profiles from confirmed examples...")
            sort_photos.build_identity_db_from_person_folders(pipeline_paths.PEOPLE_ROOT)
        if secondary_matcher is not None:
            secondary_matcher.flush()
    print("Unknown identity review closed safely.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

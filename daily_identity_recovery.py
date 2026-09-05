"""Plan individual intake recovery using Quick Review's existing safety gate.

This module never moves images or trains identities from automatic decisions.
Accepted records rejoin the sorter's normal verified-copy/duplicate pipeline.
"""

from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import identity_hard_negatives
import identity_profiles
import review_unknown_identities as review
import sort_photos


@dataclass
class RecoveryPlan:
    rows: list[dict] = field(default_factory=list)
    verifier_status: str = "not needed"
    gate_message: str = "not needed"

    @property
    def counts(self) -> dict[str, int]:
        return dict(Counter(str(row["reason"]) for row in self.rows))


def _source_key(path: Path) -> str:
    return os.path.realpath(str(path))


def _unresolved(records, name_map) -> dict[str, list]:
    sources: dict[str, list] = defaultdict(list)
    for record in records:
        sources[_source_key(record.src)].append(record)
    return {
        key: faces for key, faces in sources.items()
        if not any(sort_photos.is_real_person_label(name_map.get(face.cluster_id))
                   for face in faces)
    }


def _confirmation_person(decision: dict, digest: str, people_root: Path,
                         identity_db) -> str:
    if decision.get("action") != "confirmed" or not digest:
        return ""
    # Path/time decisions alone must never authorize reuse of different bytes.
    if decision.get("content_sha256") != digest:
        return ""
    destination = review._verified_organized_destination(
        {"people_root": people_root}, decision, digest
    )
    if destination is None:
        return ""
    parts = destination.relative_to(people_root.resolve()).parts
    canonical = {name.casefold(): name for name in identity_db.identities} if identity_db else {}
    requested = str(decision.get("person", "")).strip()
    person = canonical.get(requested.casefold(), requested)
    if (not sort_photos.is_real_person_label(person) or len(parts) < 3 or parts[1] != "photos"
            or parts[0].casefold() != person.casefold()):
        return ""
    return person


def apply_manual_overrides(records, name_map, identity_db, *, people_root: Path,
                           decisions_path: Path = review.DEFAULT_DECISIONS) -> int:
    """Manual content decisions take precedence before any automatic lane."""
    decisions = review.load_decisions(decisions_path)
    by_source: dict[str, list] = defaultdict(list)
    for face in records:
        by_source[_source_key(face.src)].append(face)
    next_id = max([0, *name_map, *(face.cluster_id for face in records)]) + 1
    handled = 0
    for faces in by_source.values():
        resolved = review.decision_for_path(decisions, faces[0].src)
        if not resolved:
            continue
        decision = resolved[1]
        action = decision.get("action", "")
        if action not in {"confirmed", "ignored", "moved_to_junk", "keep_unknown"}:
            continue
        digest = review.item_content_sha256(faces[0].src)
        person = (_confirmation_person(decision, digest, people_root, identity_db)
                  if len(faces) == 1 else "")
        name_map[next_id] = person or f"person_{next_id:03d}"
        for face in faces:
            face.cluster_id = next_id
            face.identity_review_reason = "" if person else "retained_user_decision"
        next_id += 1
        handled += 1
    return handled


def plan_recovery(
    records,
    name_map,
    identity_db,
    cache,
    *,
    people_root: Path,
    decisions_path: Path = review.DEFAULT_DECISIONS,
    gate_dir: Path = review.DEFAULT_OUTPUT_DIR,
    progress: Callable[[str], None] = print,
) -> RecoveryPlan:
    """Read detections/decisions without changing records, labels, or images.

    Verifier caches and benchmark reports may refresh. Group photographs remain
    review items; no cluster-level vote can carry another image into a folder.
    """
    plan = RecoveryPlan()
    sources = _unresolved(records, name_map)
    if not sources:
        return plan
    if identity_db is None or not identity_db.identities:
        plan.rows = [dict(source=key, status="held", reason="identity_db_unavailable")
                     for key in sorted(sources)]
        return plan
    identity_db = sort_photos.normalize_identity_db(identity_db)
    decisions = review.load_decisions(decisions_path)
    model_signature = review.review_model_signature(
        identity_db, sort_photos.IDENTITY_HARD_NEGATIVES_FILE
    )
    pending: list[tuple[object, dict, dict]] = []
    for key, faces in sorted(sources.items()):
        row = dict(source=key, status="held", reason="multiple_faces",
                   faces=len(faces), person="", distance=None, margin=None)
        plan.rows.append(row)
        if len(faces) != 1:
            continue
        face = faces[0]
        row["quality"] = float(face.quality)
        if not face.src.is_file():
            row["reason"] = "source_unavailable"
            continue
        digest = review.item_content_sha256(face.src)
        row["sha256"] = digest
        if not digest:
            row["reason"] = "source_hash_unavailable"
            continue
        resolved = review.decision_for_path(decisions, face.src)
        decision = resolved[1] if resolved else {}
        person = _confirmation_person(decision, digest, people_root, identity_db)
        if person:
            row.update(status="accepted", reason="verified_confirmation_replay", person=person)
            continue
        action = decision.get("action", "")
        if (action in {"ignored", "moved_to_junk"}
                or (action == "keep_unknown"
                    and not review.needs_silent_recheck(decisions, face.src, model_signature))):
            row["reason"] = "retained_user_decision"
            continue
        if not face.crop_jpeg:
            row["reason"] = "face_crop_unavailable"
            continue
        row["reason"] = "awaiting_verifier"
        pending.append((face, row, decision))
    if not pending:
        return plan

    matcher = None
    try:
        progress(f"Individual identity recovery: {len(pending)} images; checking independent verifier...")
        matcher, plan.verifier_status = review.prepare_secondary_verifier(
            identity_db, cache, requested=True
        )
        progress(f"Independent verifier: {plan.verifier_status}")
        if matcher is None:
            for _face, row, _decision in pending:
                row["reason"] = "verifier_unavailable"
            return plan
        prototypes, _trusted_counts = review.build_trusted_review_prototypes(
            identity_db, cache, confirmations_path=sort_photos.IDENTITY_CONFIRMATIONS_FILE
        )
        progress("Individual identity recovery: checking safety benchmark (cached when unchanged)...")
        allowed, _gate, plan.gate_message = review.prepare_automatic_review_gate(
            identity_db, cache, matcher, requested=True, output_dir=gate_dir,
            review_prototypes=prototypes,
        )
        progress(plan.gate_message)
        if not allowed:
            for _face, row, _decision in pending:
                row["reason"] = "safety_benchmark_blocked"
            return plan
        hard_negatives = identity_hard_negatives.vectors_by_person(
            sort_photos.IDENTITY_HARD_NEGATIVES_FILE
        )
        compiled = identity_profiles.CompiledProfiles(
            identity_db.identities, prototypes or identity_db.prototypes,
            pose_prototypes=identity_db.pose_prototypes,
            appearance_prototypes=identity_db.appearance_prototypes,
            appearance_era_cutoffs=identity_db.appearance_era_cutoffs, hard_negatives=hard_negatives)
        for number, (face, row, decision) in enumerate(pending, 1):
            item = review.automatic_review_item(
                row["source"], face.src, face, identity_db, hard_negatives,
                prototypes, matcher,
                compiled=compiled,
            )
            best = item.candidates[0] if item.candidates else None
            row.update(reason="insufficient_independent_evidence",
                       candidate=best.name if best else "",
                       distance=float(best.distance) if best else None,
                       margin=identity_profiles.candidate_margin(item.candidates))
            if item.secondary is not None:
                row.update(secondary_candidate=item.secondary.predicted,
                           secondary_distance=float(item.secondary.distance),
                           secondary_margin=float(item.secondary.margin))
            rejected = {str(name).casefold() for name in decision.get("rejected_people", [])}
            if best is not None and best.name.casefold() in rejected:
                row["reason"] = "user_rejected_candidate"
            elif (best is not None and identity_db.source_counts.get(best.name, 0)
                  < sort_photos.AUTO_PERSON_MATCH_MIN_REFERENCE_FACES):
                row["reason"] = "insufficient_identity_references"
            else:
                matches = review.automatic_matches(
                    [review.UnknownCluster(item.key, (item,), item.candidates, 0.0)],
                    identity_db, require_secondary=True,
                )
                if matches:
                    match = matches[0]
                    row.update(status="accepted", reason=match.lane, person=match.person,
                               distance=match.distance, margin=match.margin)
            if number % 25 == 0 or number == len(pending):
                recovered = sum(row["status"] == "accepted" for row in plan.rows)
                progress(f"Individual identity recovery {number}/{len(pending)}: "
                         f"{recovered} recoverable; originals unchanged during matching")
    except Exception as error:  # noqa: BLE001
        # Never partially adopt a new automatic policy after a failed run.
        plan.gate_message = f"Recovery unavailable: {error}"
        for _face, row, _decision in pending:
            row.update(status="held", reason="recovery_error", person="")
        progress(plan.gate_message)
    finally:
        if matcher is not None:
            try:
                matcher.flush()
            except Exception as error:  # noqa: BLE001
                progress(f"Verifier cache save failed (decisions remain in report): {error}")
            matcher.app = None
    return plan


def apply_plan(plan: RecoveryPlan, records, name_map, decision_recorder=None) -> int:
    """Split accepted individual records without relabeling their old cluster."""
    sources = _unresolved(records, name_map)
    next_id = max([0, *name_map.keys(), *(face.cluster_id for face in records)]) + 1
    assigned = 0
    for row in plan.rows:
        faces = sources.get(row["source"], [])
        if row["status"] != "accepted":
            for face in faces:
                face.identity_review_reason = row["reason"]
            continue
        if len(faces) != 1:
            continue
        face = faces[0]
        # Recheck exact bytes between planning and installation, including replays.
        try:
            digest = sort_photos.sha256_file(face.src)
        except OSError:
            digest = ""
        if not digest or digest != row.get("sha256"):
            face.identity_review_reason = "source_changed_during_recovery"
            row.update(status="held", reason=face.identity_review_reason, person="")
            continue
        if decision_recorder is not None:
            decision_recorder(face, row["person"], float(row["distance"] or 0),
                              float(row["margin"] or 0), "daily_" + row["reason"])
        name_map[next_id] = row["person"]
        face.cluster_id = next_id
        face.identity_review_reason = ""
        face.recovery_assigned = True
        next_id += 1
        assigned += 1
        sources.pop(row["source"], None)
    return assigned


def write_report(plan: RecoveryPlan, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps({
        "version": 1,
        "verifier_status": plan.verifier_status,
        "gate_message": plan.gate_message,
        "counts": plan.counts,
        "rows": plan.rows,
    }, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)

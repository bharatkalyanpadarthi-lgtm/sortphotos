"""Per-member daily identity assignment, independent of CLI orchestration."""

from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from typing import Callable, TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    from sort_photos import FaceRecord, IdentityDB


def assign_identity_labels(
    records: list[FaceRecord],
    name_map: dict[int, str],
    identity_db: IdentityDB | None,
    decision_recorder: Callable[[FaceRecord, str, float, float, str], None]
    | None = None,
    source_batch_root: Path | None = None,
    use_secondary_verifier: bool = True,
    *,
    pipeline,
    hard_negatives_override=None,
) -> int:
    if (
        not pipeline.AUTO_PERSON_MATCH_ENABLED
        or identity_db is None
        or (not identity_db.identities)
    ):
        return 0
    identity_db = pipeline.normalize_identity_db(identity_db)
    centroids = pipeline.compute_centroids(records)
    eligible_identities = {
        name: centroid
        for name, centroid in identity_db.identities.items()
        if identity_db.source_counts.get(name, 0)
        >= pipeline.AUTO_PERSON_MATCH_MIN_REFERENCE_FACES
    }
    if not eligible_identities:
        return 0
    eligible_prototypes = {
        name: identity_db.prototypes.get(name, [centroid])
        for name, centroid in eligible_identities.items()
    }
    hard_negatives = hard_negatives_override if hard_negatives_override is not None else pipeline.identity_hard_negatives.vectors_by_person(
        pipeline.IDENTITY_HARD_NEGATIVES_FILE
    )
    compiled = pipeline.identity_profiles.CompiledProfiles(
        eligible_identities, eligible_prototypes, pose_prototypes=identity_db.pose_prototypes,
        appearance_prototypes=identity_db.appearance_prototypes,
        appearance_era_cutoffs=identity_db.appearance_era_cutoffs, hard_negatives=hard_negatives)
    secondary_db = (
        pipeline.secondary_identity_matcher.load() if use_secondary_verifier else None
    )
    secondary_matcher = (
        pipeline.secondary_identity_matcher.SecondaryMatcher(secondary_db)
        if secondary_db is not None
        and secondary_db.primary_signature
        == pipeline.secondary_identity_matcher.primary_signature(identity_db)
        and pipeline.secondary_identity_matcher.trusted_snapshot_is_current(
            secondary_db, pipeline.IDENTITY_CONFIRMATIONS_FILE
        )
        else None
    )

    def secondary_agrees(record: FaceRecord, expected: str) -> bool:
        if secondary_matcher is None or record.quality < 0.32 or (not record.crop_jpeg):
            return False
        return secondary_matcher.verify(record.crop_jpeg, expected).accepted

    ranked_record_cache: dict[
        int, list[pipeline.identity_profiles.IdentityCandidate]
    ] = {}

    def rank_record(
        record: FaceRecord,
    ) -> list[pipeline.identity_profiles.IdentityCandidate]:
        cache_key = id(record)
        cached = ranked_record_cache.get(cache_key)
        if cached is not None:
            return cached
        lighting, captured_at = pipeline.appearance_profiles.query_attributes(
            record.crop_jpeg, record.src
        )
        ranked = pipeline.identity_profiles.rank_candidates(
            record.embedding,
            eligible_identities,
            eligible_prototypes,
            pose_label=record.pose_label,
            pose_prototypes=identity_db.pose_prototypes,
            lighting_label=lighting,
            capture_timestamp=captured_at,
            appearance_prototypes=identity_db.appearance_prototypes,
            appearance_era_cutoffs=identity_db.appearance_era_cutoffs,
            hard_negatives=hard_negatives,
            compiled=compiled,
        )
        ranked_record_cache[cache_key] = ranked
        return ranked

    def assign_members(cid: int, person: str, members: list[FaceRecord]) -> bool:
        if not members:
            return False
        accepted_ids = {id(record) for record in members}
        group = records_by_cluster.get(cid, [])
        if group and all((id(record) in accepted_ids for record in group)):
            name_map[cid] = person
            return True
        new_id = max([0, *name_map, *(record.cluster_id for record in records)]) + 1
        name_map[new_id] = person
        for record in group:
            if id(record) in accepted_ids:
                record.cluster_id = new_id
            else:
                record.identity_review_reason = "cluster_identity_dissent"
        return True

    def independent_source(record: FaceRecord) -> str:
        try:
            return pipeline.content_identity.content_sha256(record.src)
        except OSError:
            return str(record.src.resolve())

    batch_decisions: dict[
        tuple[str, int], pipeline.source_batch_consensus.ConsensusDecision
    ] = {}
    if source_batch_root is not None:
        batch_evidence: list[pipeline.source_batch_consensus.BatchEvidence] = []
        for record in records:
            batch_key = pipeline.source_batch_consensus.direct_batch_key(
                record.src, source_batch_root
            )
            if not batch_key:
                continue
            member_candidates = rank_record(record)
            if not member_candidates:
                continue
            member_best = member_candidates[0]
            member_threshold = min(
                pipeline.AUTO_PERSON_MATCH_DIST,
                identity_db.match_thresholds.get(
                    member_best.name, pipeline.AUTO_PERSON_MATCH_DIST
                ),
            )
            batch_evidence.append(
                pipeline.source_batch_consensus.BatchEvidence(
                    source=str(record.src),
                    face_index=int(record.face_index),
                    batch_key=batch_key,
                    embedding=np.asarray(record.embedding, dtype=np.float32),
                    candidate_name=member_best.name,
                    distance=float(member_best.distance),
                    margin=pipeline.identity_profiles.candidate_margin(
                        member_candidates
                    ),
                    quality=float(record.quality),
                    threshold=float(member_threshold),
                )
            )
        batch_decisions = pipeline.source_batch_consensus.consensus_decisions(
            batch_evidence
        )
    assigned = 0
    records_by_cluster: dict[int, list[FaceRecord]] = defaultdict(list)
    for record in records:
        if record.cluster_id != -1:
            records_by_cluster[record.cluster_id].append(record)
    for cid, current_name in sorted(name_map.items(), key=lambda kv: kv[0]):
        if cid == -1 or not current_name.startswith("person_") or cid not in centroids:
            continue
        cluster_records = records_by_cluster.get(cid, [])
        if any((record.identity_review_reason for record in cluster_records)):
            continue
        records_by_source: dict[str, FaceRecord] = {}
        for record in cluster_records:
            source = independent_source(record)
            if (
                source not in records_by_source
                or record.quality > records_by_source[source].quality
            ):
                records_by_source[source] = record
        independent_records = list(records_by_source.values())
        if not independent_records:
            continue
        cluster_candidates = pipeline.identity_profiles.rank_candidates(
            centroids[cid],
            eligible_identities,
            eligible_prototypes,
            pose_prototypes=identity_db.pose_prototypes,
            hard_negatives=hard_negatives,
            compiled=compiled,
        )
        if not cluster_candidates:
            continue
        best = cluster_candidates[0]
        matched_name = best.name
        cluster_margin = pipeline.identity_profiles.candidate_margin(cluster_candidates)
        consensus_threshold = min(
            pipeline.AUTO_PERSON_MATCH_DIST,
            identity_db.match_thresholds.get(
                matched_name, pipeline.AUTO_PERSON_MATCH_DIST
            ),
        )
        strict_threshold = min(
            pipeline.AUTO_PERSON_SINGLE_MATCH_DIST,
            identity_db.strict_thresholds.get(
                matched_name, pipeline.AUTO_PERSON_SINGLE_MATCH_DIST
            ),
        )
        if len(independent_records) == 1:
            record = independent_records[0]
            member_candidates = rank_record(record)
            member_best = member_candidates[0] if member_candidates else None
            member_margin = pipeline.identity_profiles.candidate_margin(
                member_candidates
            )
            batch_match = batch_decisions.get((str(record.src), int(record.face_index)))
            if batch_match is not None:
                members = [
                    member
                    for member in cluster_records
                    if (decision := batch_decisions.get((str(member.src), int(member.face_index))))
                    and decision.person == batch_match.person
                ]
                if not assign_members(cid, batch_match.person, members):
                    continue
                assigned += 1
                if decision_recorder is not None:
                    decision_recorder(
                        record,
                        batch_match.person,
                        batch_match.distance,
                        batch_match.margin,
                        batch_match.lane,
                    )
                pipeline.log.info(
                    "Existing-person source-batch match: %s -> %s (support %d, dist %.3f, margin %.3f).",
                    current_name,
                    batch_match.person,
                    batch_match.support,
                    batch_match.distance,
                    batch_match.margin,
                )
                continue
            primary_strict = (
                member_best is not None
                and member_best.name == matched_name
                and (best.distance <= strict_threshold)
                and (cluster_margin >= pipeline.AUTO_PERSON_SINGLE_MATCH_MARGIN)
                and (member_best.distance <= strict_threshold)
                and (member_margin >= pipeline.AUTO_PERSON_SINGLE_MATCH_MARGIN)
                and (record.quality >= pipeline.AUTO_PERSON_SINGLE_MIN_QUALITY)
            )
            secondary_verified = (
                not primary_strict
                and member_best is not None
                and (member_best.name == matched_name)
                and (member_best.distance <= consensus_threshold + 0.08)
                and (member_margin >= 0.03)
                and secondary_agrees(record, matched_name)
            )
            if primary_strict or secondary_verified:
                members = [
                    member
                    for member in cluster_records
                    if rank_record(member)
                    and rank_record(member)[0].name == matched_name
                    and (
                        rank_record(member)[0].distance
                        <= (strict_threshold if primary_strict else consensus_threshold + 0.08)
                    )
                    and (
                        pipeline.identity_profiles.candidate_margin(rank_record(member))
                        >= (pipeline.AUTO_PERSON_SINGLE_MATCH_MARGIN if primary_strict else 0.03)
                    )
                    and (member.quality >= pipeline.AUTO_PERSON_SINGLE_MIN_QUALITY if primary_strict
                         else secondary_agrees(member, matched_name))
                ]
                if not assign_members(cid, matched_name, members):
                    continue
                assigned += 1
                lane = "strict_single" if primary_strict else "secondary_agreement"
                if decision_recorder is not None:
                    try:
                        decision_recorder(
                            record,
                            matched_name,
                            member_best.distance,
                            member_margin,
                            lane,
                        )
                    except Exception as exc:
                        pipeline.log.warning(
                            "Could not persist strict identity history: %s", exc
                        )
                pipeline.log.info(
                    "Existing-person %s match: %s -> %s (dist %.3f, margin %.3f, quality %.3f).",
                    lane,
                    current_name,
                    matched_name,
                    member_best.distance,
                    member_margin,
                    record.quality,
                )
            continue
        if (
            best.distance > consensus_threshold
            or cluster_margin < pipeline.AUTO_PERSON_MATCH_MARGIN
        ):
            borderline_records: list[FaceRecord] = []
            if best.distance <= consensus_threshold + 0.08 and cluster_margin >= 0.03:
                for record in sorted(
                    independent_records, key=lambda value: -value.quality
                )[:3]:
                    member_candidates = rank_record(record)
                    if (
                        member_candidates
                        and member_candidates[0].name == matched_name
                        and secondary_agrees(record, matched_name)
                    ):
                        borderline_records.append(record)
            if len({str(record.src) for record in borderline_records}) >= 2:
                assign_members(cid, matched_name, borderline_records)
                assigned += 1
                if decision_recorder is not None:
                    for record in borderline_records:
                        decision_recorder(
                            record,
                            matched_name,
                            best.distance,
                            cluster_margin,
                            "secondary_cluster_agreement",
                        )
                pipeline.log.info(
                    "Existing-person secondary cluster match: %s -> %s (primary dist %.3f, margin %.3f, secondary support %d).",
                    current_name,
                    matched_name,
                    best.distance,
                    cluster_margin,
                    len(borderline_records),
                )
            continue
        agreeing = 0
        confident_votes = 0
        agreeing_members: list[FaceRecord] = []
        for record in independent_records:
            member_candidates = rank_record(record)
            if not member_candidates:
                continue
            member_best = member_candidates[0]
            member_margin = pipeline.identity_profiles.candidate_margin(
                member_candidates
            )
            member_threshold = min(
                pipeline.AUTO_PERSON_MATCH_DIST,
                identity_db.match_thresholds.get(
                    member_best.name, pipeline.AUTO_PERSON_MATCH_DIST
                ),
            )
            if (
                member_best.distance <= member_threshold
                and member_margin >= pipeline.AUTO_PERSON_MATCH_MARGIN
            ):
                confident_votes += 1
                if member_best.name == matched_name:
                    agreeing += 1
                    agreeing_members.append(record)
        agreement = agreeing / max(1, len(independent_records))
        if (
            agreeing < pipeline.AUTO_PERSON_MATCH_MIN_CLUSTER_FACES
            or len(independent_records) < pipeline.AUTO_PERSON_MATCH_MIN_CLUSTER_SOURCES
            or agreement < pipeline.AUTO_PERSON_MATCH_MIN_AGREEMENT
        ):
            pipeline.log.info(
                "Identity match held for review: %s -> %s (cluster dist %.3f, margin %.3f, agreement %d/%d, confident %d).",
                current_name,
                matched_name,
                best.distance,
                cluster_margin,
                agreeing,
                len(independent_records),
                confident_votes,
            )
            continue
        pipeline.log.info(
            "Existing-person match: %s -> %s (dist %.3f, margin %.3f, agreement %d/%d).",
            current_name,
            matched_name,
            best.distance,
            cluster_margin,
            agreeing,
            len(independent_records),
        )
        accepted_sources = {independent_source(record) for record in agreeing_members}
        accepted_members = []
        for record in cluster_records:
            ranked = rank_record(record)
            if (
                independent_source(record) in accepted_sources
                and ranked
                and (ranked[0].name == matched_name)
                and (ranked[0].distance <= consensus_threshold)
                and (
                    pipeline.identity_profiles.candidate_margin(ranked)
                    >= pipeline.AUTO_PERSON_MATCH_MARGIN
                )
            ):
                accepted_members.append(record)
        if not assign_members(cid, matched_name, accepted_members):
            continue
        assigned += 1
        if decision_recorder is not None:
            for record in accepted_members:
                try:
                    decision_recorder(
                        record,
                        matched_name,
                        best.distance,
                        cluster_margin,
                        "independent_consensus",
                    )
                except Exception as exc:
                    pipeline.log.warning(
                        "Could not persist consensus identity history: %s", exc
                    )
    if secondary_matcher is not None:
        secondary_matcher.flush()
    return assigned

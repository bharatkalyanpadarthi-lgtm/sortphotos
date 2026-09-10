"""Shared, pure acceptance policy for daily recovery and Quick Review."""

from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
import identity_profiles

if TYPE_CHECKING:
    from sort_photos import IdentityDB
    from review_unknown_identities import UnknownItem, UnknownCluster


@dataclass(frozen=True)
class AutomaticMatch:
    person: str
    item_keys: tuple[str, ...]
    lane: str
    support: int
    total: int
    distance: float
    margin: float


class RecoveryProfiles(dict):
    """Bounded recovery vectors with source provenance for held-out validation."""

    def __init__(self):
        super().__init__()
        self.sources = {}


def strict_lane_allowed(gate) -> bool:
    return bool(gate.get("allowed") is True and gate.get("strict_single_allowed") is True)


def _automatic_item_evidence(
    item: UnknownItem, identity_db: IdentityDB, *, settings
) -> dict[str, object] | None:
    if not item.candidates:
        return None
    best = item.candidates[0]
    margin = identity_profiles.candidate_margin(item.candidates)
    consensus_threshold = min(
        settings.recover_no_usable_faces.MATCH_MAX_DISTANCE,
        float(
            identity_db.match_thresholds.get(
                best.name, settings.recover_no_usable_faces.MATCH_MAX_DISTANCE
            )
        ),
    )
    strict_threshold = min(
        settings.sort_photos.AUTO_PERSON_SINGLE_MATCH_DIST,
        float(
            identity_db.strict_thresholds.get(
                best.name, settings.sort_photos.AUTO_PERSON_SINGLE_MATCH_DIST
            )
        ),
    )
    quality = float(item.face.quality)
    # Two models of one crop are not independent source-image consensus.
    # Neither agreement nor rescue may bypass calibration or verifier rejection.
    single_margin = settings.sort_photos.AUTO_PERSON_SINGLE_MATCH_MARGIN
    strict = (
        best.distance <= strict_threshold
        and margin >= settings.sort_photos.AUTO_PERSON_SINGLE_MATCH_MARGIN
        and (quality >= settings.sort_photos.AUTO_PERSON_SINGLE_MIN_QUALITY)
    )
    secondary = bool(
        item.secondary is not None
        and item.secondary.accepted
        and str(item.secondary.predicted or "").casefold() == best.name.casefold()
        and (
            best.distance
            <= min(settings.AUTO_JOINT_MAX_PRIMARY_DISTANCE, consensus_threshold)
        )
        and (margin >= max(settings.AUTO_JOINT_MIN_PRIMARY_MARGIN, single_margin))
        and (quality >= settings.AUTO_JOINT_MIN_QUALITY)
        and (item.secondary.distance <= settings.AUTO_JOINT_MAX_SECONDARY_DISTANCE)
        and (item.secondary.margin >= settings.AUTO_JOINT_MIN_SECONDARY_MARGIN)
    )
    secondary_rescue = bool(
        item.secondary is not None
        and item.secondary.accepted
        and str(item.secondary.predicted or "").casefold() == best.name.casefold()
        and (best.distance <= min(settings.AUTO_RESCUE_MAX_PRIMARY_DISTANCE, consensus_threshold))
        and (margin >= max(settings.AUTO_RESCUE_MIN_PRIMARY_MARGIN, single_margin))
        and (quality >= settings.AUTO_RESCUE_MIN_QUALITY)
        and (item.secondary.distance <= settings.AUTO_RESCUE_MAX_SECONDARY_DISTANCE)
        and (item.secondary.margin >= settings.AUTO_RESCUE_MIN_SECONDARY_MARGIN)
    )
    exceptional = (
        best.distance <= settings.recover_no_usable_faces.MATCH_EXCEPTIONAL_DISTANCE
        and margin >= settings.recover_no_usable_faces.MATCH_EXCEPTIONAL_MARGIN
    )
    consensus_vote = (
        best.distance <= consensus_threshold
        and margin >= settings.recover_no_usable_faces.MATCH_MIN_MARGIN
        and (
            quality >= settings.recover_no_usable_faces.MATCH_MIN_QUALITY or exceptional
        )
    )
    secondary_dissent = bool(
        item.secondary is not None
        and str(item.secondary.predicted or "")
        and (str(item.secondary.predicted).casefold() != best.name.casefold())
        and (item.secondary.distance <= 0.3)
        and (item.secondary.margin >= 0.08)
    )
    return {
        "person": best.name,
        "distance": float(best.distance),
        "margin": float(margin),
        "threshold": float(consensus_threshold),
        "strict": strict,
        "secondary": secondary,
        "secondary_rescue": secondary_rescue,
        "consensus_vote": consensus_vote,
        "secondary_dissent": secondary_dissent,
    }


def automatic_matches(
    clusters: list[UnknownCluster],
    identity_db: IdentityDB,
    *,
    require_secondary: bool = False,
    allow_strict_single: bool = False,
    settings,
) -> list[AutomaticMatch]:
    """Return only automatic matches supported by conservative evidence lanes."""
    matches: list[AutomaticMatch] = []
    assigned: set[str] = set()
    for cluster in clusters:
        independent = {}
        for item in cluster.items:
            key = settings.item_content_sha256(item.path) or str(item.path.resolve())
            previous = independent.get(key)
            if previous is None or item.face.quality > previous.face.quality:
                independent[key] = item
        evidence = {
            item.key: settings._automatic_item_evidence(item, identity_db)
            for item in cluster.items
        }
        cluster_best = cluster.candidates[0] if cluster.candidates else None
        cluster_margin = identity_profiles.candidate_margin(cluster.candidates)
        cluster_person = cluster_best.name if cluster_best is not None else ""
        cluster_threshold = (
            min(
                settings.recover_no_usable_faces.MATCH_MAX_DISTANCE,
                float(
                    identity_db.match_thresholds.get(
                        cluster_person,
                        settings.recover_no_usable_faces.MATCH_MAX_DISTANCE,
                    )
                ),
            )
            if cluster_person
            else 0.0
        )
        votes = [
            item
            for item in cluster.items
            if evidence[item.key] is not None
            and str(evidence[item.key]["person"]).casefold()
            == cluster_person.casefold()
            and bool(evidence[item.key]["consensus_vote"])
            and (not bool(evidence[item.key]["secondary_dissent"]))
        ]
        confident_dissent = any(
            (
                value is not None
                and str(value["person"]).casefold() != cluster_person.casefold()
                and bool(value["consensus_vote"])
                for value in evidence.values()
            )
        )
        all_votes = votes
        independent_keys = {item.key for item in independent.values()}
        votes = [item for item in votes if item.key in independent_keys]
        strict_anchors = [item for item in votes if bool(evidence[item.key]["strict"])]
        secondary_anchors = [
            item for item in votes if bool(evidence[item.key]["secondary"])
        ]
        all_vote_anchor = bool(
            len(votes) >= 3
            and sum((float(evidence[item.key]["distance"]) for item in votes))
            / len(votes)
            <= sum((float(evidence[item.key]["threshold"]) for item in votes))
            / len(votes)
            and (
                sum((float(evidence[item.key]["margin"]) for item in votes))
                / len(votes)
                >= 0.12
            )
        )
        cluster_is_safe = bool(
            cluster_person
            and len(independent) >= 2
            and (len(votes) >= settings.sort_photos.AUTO_PERSON_MATCH_MIN_CLUSTER_FACES)
            and (len(votes) / len(independent) >= settings.AUTO_CLUSTER_MIN_AGREEMENT)
            and (cluster_best is not None)
            and (cluster_best.distance <= cluster_threshold)
            and (cluster_margin >= settings.AUTO_CLUSTER_MIN_MARGIN)
            and (cluster.cohesion <= settings.AUTO_CLUSTER_MAX_COHESION)
            and (not confident_dissent)
            and (
                len(secondary_anchors) >= min(2, len(votes))
                if require_secondary
                else bool(strict_anchors or all_vote_anchor)
            )
        )
        if cluster_is_safe:
            item_keys = tuple((item.key for item in all_votes))
            matches.append(
                AutomaticMatch(
                    person=cluster_person,
                    item_keys=item_keys,
                    lane="cluster_consensus",
                    support=len(votes),
                    total=len(cluster.items),
                    distance=float(cluster_best.distance),
                    margin=float(cluster_margin),
                )
            )
            assigned.update(item_keys)
        for item in cluster.items:
            if item.key in assigned:
                continue
            value = evidence[item.key]
            if value is None or bool(value["secondary_dissent"]):
                continue
            lane = (
                "secondary_agreement"
                if bool(value["secondary"])
                else "trusted_verifier_rescue"
                if bool(value["secondary_rescue"])
                else "strict_single"
                if bool(value["strict"]) and (not require_secondary or allow_strict_single)
                else ""
            )
            if not lane:
                continue
            matches.append(
                AutomaticMatch(
                    person=str(value["person"]),
                    item_keys=(item.key,),
                    lane=lane,
                    support=1,
                    total=1,
                    distance=float(value["distance"]),
                    margin=float(value["margin"]),
                )
            )
            assigned.add(item.key)
    return matches

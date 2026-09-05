#!/usr/bin/env python3
"""Conservative identity consensus within independently sourced intake batches."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

import identity_profiles
import content_identity


@dataclass(frozen=True)
class BatchEvidence:
    source: str
    face_index: int
    batch_key: str
    embedding: np.ndarray
    candidate_name: str
    distance: float
    margin: float
    quality: float
    threshold: float


@dataclass(frozen=True)
class ConsensusDecision:
    person: str
    distance: float
    margin: float
    support: int
    batch_key: str
    lane: str = "source_batch_consensus"


def independent_key(value: BatchEvidence) -> str:
    try:
        return content_identity.content_sha256(Path(value.source))
    except OSError:
        return str(Path(value.source).resolve())


def direct_batch_key(path: Path, queue_root: Path) -> str:
    """Use an actual nested intake folder; never group an entire root queue."""
    try:
        relative = path.resolve(strict=False).relative_to(queue_root.resolve(strict=False))
    except ValueError:
        return ""
    if len(relative.parts) <= 1:
        return ""
    return f"folder:{relative.parts[0].casefold()}"


def _components(values: Sequence[BatchEvidence], link_distance: float) -> list[list[BatchEvidence]]:
    if not values:
        return []
    matrix = identity_profiles.normalize_matrix([value.embedding for value in values])
    parent = list(range(len(values)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    similarities = matrix @ matrix.T
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if 1.0 - float(similarities[left, right]) <= link_distance:
                union(left, right)
    groups: dict[int, list[BatchEvidence]] = {}
    for index, value in enumerate(values):
        groups.setdefault(find(index), []).append(value)
    return list(groups.values())


def consensus_decisions(
    evidence: Sequence[BatchEvidence],
    *,
    link_distance: float = 0.30,
    minimum_support: int = 2,
    minimum_agreement: float = 0.80,
    anchor_margin: float = 0.08,
    weak_margin: float = 0.04,
) -> dict[tuple[str, int], ConsensusDecision]:
    """Return only corroborated weak matches from a same-batch face cluster."""
    by_batch: dict[str, list[BatchEvidence]] = {}
    for value in evidence:
        if value.batch_key:
            by_batch.setdefault(value.batch_key, []).append(value)

    decisions: dict[tuple[str, int], ConsensusDecision] = {}
    for batch_key, batch_values in by_batch.items():
        for component in _components(batch_values, link_distance):
            all_members = component
            by_content: dict[str, BatchEvidence] = {}
            for value in component:
                key = independent_key(value)
                if key not in by_content or value.quality > by_content[key].quality:
                    by_content[key] = value
            component = list(by_content.values())
            unique_sources = set(by_content)
            if len(unique_sources) < minimum_support:
                continue
            names: dict[str, list[BatchEvidence]] = {}
            for value in component:
                names.setdefault(value.candidate_name, []).append(value)
            person, agreeing = max(
                names.items(), key=lambda item: (len({v.source for v in item[1]}), item[0].casefold())
            )
            support = len({value.source for value in agreeing})
            agreement = support / max(1, len(unique_sources))
            if support < minimum_support or agreement < minimum_agreement:
                continue
            anchor = any(
                value.distance <= value.threshold
                and value.margin >= anchor_margin
                and value.quality >= 0.45
                for value in agreeing
            )
            all_vote_lane = (
                support >= 3
                and all(value.distance <= value.threshold + 0.04 for value in agreeing)
                and sum(value.distance for value in agreeing) / len(agreeing)
                <= sum(value.threshold for value in agreeing) / len(agreeing)
            )
            if not anchor and not all_vote_lane:
                continue
            for value in all_members:
                if value.candidate_name != person:
                    continue
                if (
                    value.distance <= value.threshold + 0.05
                    and value.margin >= weak_margin
                    and value.quality >= 0.25
                ):
                    decisions[(value.source, value.face_index)] = ConsensusDecision(
                        person=person,
                        distance=value.distance,
                        margin=value.margin,
                        support=support,
                        batch_key=batch_key,
                    )
    return decisions

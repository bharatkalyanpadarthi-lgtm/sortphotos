#!/usr/bin/env python3
"""Pure identity-profile selection and matching helpers.

This module deliberately has no model or file-system dependencies. Keeping the
decision math isolated makes it possible to regression-test recognition policy
without loading InsightFace or touching the photo library.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ReferenceSample:
    source: str
    embedding: np.ndarray
    quality: float
    pose_label: str = "unknown"
    lighting_label: str = "unknown"
    capture_timestamp: float = 0.0


@dataclass(frozen=True)
class IdentityCandidate:
    name: str
    similarity: float
    distance: float
    raw_distance: float | None = None
    hard_negative_distance: float | None = None


def normalize_vector(value: np.ndarray) -> np.ndarray:
    vector = np.asarray(value, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(vector))
    return vector / max(norm, 1e-9)


def normalize_matrix(values: Sequence[np.ndarray]) -> np.ndarray:
    if not values:
        return np.zeros((0, 0), dtype=np.float32)
    matrix = np.stack([normalize_vector(value) for value in values]).astype(np.float32)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norms, 1e-9)


def dominant_identity_samples(
    samples: Sequence[ReferenceSample],
    *,
    link_distance: float = 0.36,
) -> list[ReferenceSample]:
    """Keep the best-supported face cluster from a person's reference images.

    Person folders can contain group photos. The target person normally appears
    across the most distinct source images, while bystanders form smaller
    components. Connected components preserve pose variation better than
    requiring every sample to be close to one global centroid.
    """
    if len(samples) <= 1:
        return list(samples)
    matrix = normalize_matrix([sample.embedding for sample in samples])
    parent = list(range(len(samples)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    similarities = matrix @ matrix.T
    for left in range(len(samples)):
        for right in range(left + 1, len(samples)):
            if 1.0 - float(similarities[left, right]) <= link_distance:
                union(left, right)

    components: dict[int, list[ReferenceSample]] = {}
    for index, sample in enumerate(samples):
        components.setdefault(find(index), []).append(sample)

    def component_score(group: list[ReferenceSample]) -> tuple[int, float, float]:
        source_count = len({sample.source for sample in group})
        quality_sum = sum(max(0.0, float(sample.quality)) for sample in group)
        best_quality = max((float(sample.quality) for sample in group), default=0.0)
        return source_count, quality_sum, best_quality

    return max(components.values(), key=component_score)


def select_diverse_samples(
    samples: Sequence[ReferenceSample],
    *,
    limit: int,
) -> list[ReferenceSample]:
    """Greedily select high-quality references that cover different appearances."""
    if limit <= 0 or len(samples) <= limit:
        return sorted(samples, key=lambda sample: (-sample.quality, sample.source))

    remaining = sorted(samples, key=lambda sample: (-sample.quality, sample.source))
    selected = [remaining.pop(0)]
    qualities = np.asarray([max(0.0, sample.quality) for sample in samples], dtype=np.float32)
    quality_min = float(qualities.min(initial=0.0))
    quality_span = max(float(qualities.max(initial=1.0)) - quality_min, 1e-6)

    while remaining and len(selected) < limit:
        selected_matrix = normalize_matrix([sample.embedding for sample in selected])

        def score(sample: ReferenceSample) -> tuple[float, float, str]:
            embedding = normalize_vector(sample.embedding)
            minimum_distance = float(np.min(1.0 - selected_matrix @ embedding))
            quality = (max(0.0, sample.quality) - quality_min) / quality_span
            combined = 0.58 * min(minimum_distance / 0.45, 1.0) + 0.42 * quality
            return combined, sample.quality, sample.source

        winner = max(remaining, key=score)
        selected.append(winner)
        remaining.remove(winner)
    return selected


def select_profile_prototypes(
    dominant_samples: Sequence[ReferenceSample],
    trusted_samples: Sequence[ReferenceSample],
    *,
    limit: int,
    trusted_limit: int = 2,
) -> list[ReferenceSample]:
    """Select stable core prototypes plus a few explicit user confirmations.

    Trusted examples are useful for rare side profiles, lighting, or styling,
    but they must not replace the repeatedly observed core of an identity.
    Reserving a small bounded number of slots gives those appearances coverage
    without allowing one confirmation to redefine the person's centroid.
    """
    if limit <= 0:
        return []

    trusted_by_source: dict[str, ReferenceSample] = {}
    for sample in trusted_samples:
        previous = trusted_by_source.get(sample.source)
        if previous is None or sample.quality > previous.quality:
            trusted_by_source[sample.source] = sample
    trusted = select_diverse_samples(
        list(trusted_by_source.values()),
        limit=min(limit, max(0, trusted_limit)),
    )
    trusted_sources = {sample.source for sample in trusted}
    core_pool = [
        sample for sample in dominant_samples
        if sample.source not in trusted_sources
    ]
    core = select_diverse_samples(core_pool, limit=max(0, limit - len(trusted)))
    selected = [*core, *trusted]

    if len(selected) < limit:
        selected_sources = {sample.source for sample in selected}
        remaining = [
            sample for sample in dominant_samples
            if sample.source not in selected_sources
        ]
        selected.extend(select_diverse_samples(remaining, limit=limit - len(selected)))
    return selected[:limit]


def weighted_centroid(samples: Sequence[ReferenceSample]) -> np.ndarray:
    if not samples:
        return np.zeros(0, dtype=np.float32)
    matrix = normalize_matrix([sample.embedding for sample in samples])
    weights = np.asarray([max(0.05, float(sample.quality)) for sample in samples], dtype=np.float32)
    weights /= max(float(weights.sum()), 1e-9)
    return normalize_vector((matrix * weights[:, None]).sum(axis=0))


def profile_similarity(
    embedding: np.ndarray,
    centroid: np.ndarray,
    prototypes: Sequence[np.ndarray] | None,
) -> float:
    query = normalize_vector(embedding)
    center = normalize_vector(centroid)
    center_similarity = float(center @ query)
    if not prototypes:
        return center_similarity
    prototype_matrix = normalize_matrix(list(prototypes))
    best_prototype = float(np.max(prototype_matrix @ query))
    # A single prototype cannot dominate the decision by itself. The centroid
    # keeps the score tied to the person's overall identity distribution.
    return max(center_similarity, 0.68 * best_prototype + 0.32 * center_similarity)


class CompiledProfiles:
    """Immutable normalized matrices reused throughout one matching batch.

    Construct a new snapshot after a profile refresh. Query-dependent pose and
    appearance selection retains exactly the scalar ranker's weighting.
    """

    def __init__(self, identities, prototypes=None, *, pose_prototypes=None,
                 appearance_prototypes=None, appearance_era_cutoffs=None,
                 hard_negatives=None):
        self.names = tuple(identities)
        self.centers = normalize_matrix(list(identities.values()))
        self.cutoffs = np.asarray([float((appearance_era_cutoffs or {}).get(name, 0))
                                   for name in self.names], dtype=np.float64)
        self.base = self._compile(prototypes or {})
        self.negatives = self._compile(hard_negatives or {})
        self.pose = self._compile_labels(pose_prototypes or {})
        self.appearance = self._compile_labels(appearance_prototypes or {})
        self.centers.setflags(write=False)
        self.cutoffs.setflags(write=False)

    def _compile(self, mapping):
        values, owners = [], []
        for index, name in enumerate(self.names):
            for value in mapping.get(name, ()):
                values.append(value)
                owners.append(index)
        matrix = normalize_matrix(values)
        owner_array = np.asarray(owners, dtype=np.intp)
        matrix.setflags(write=False)
        owner_array.setflags(write=False)
        return matrix, owner_array

    def _compile_labels(self, mapping):
        labels = {label for groups in mapping.values() for label in groups}
        return {label: self._compile({name: groups.get(label, ()) for name, groups in mapping.items()})
                for label in labels}

    def _similarities(self, query, compiled):
        best = np.full(len(self.names), -np.inf, dtype=np.float32)
        if compiled is not None:
            matrix, owners = compiled
            if len(owners):
                np.maximum.at(best, owners, matrix @ query)
        return best

    def rank(self, embedding, *, pose_label="unknown", lighting_label="unknown",
             capture_timestamp=0.0, hard_negative_radius=0.28, hard_negative_penalty=0.42):
        if not self.names:
            return []
        query = normalize_vector(embedding)
        centers = self.centers @ query
        best = self._similarities(query, self.base)
        best = np.maximum(best, self._similarities(query, self.pose.get(pose_label)))
        if lighting_label in {"low_light", "normal_light"}:
            best = np.maximum(best, self._similarities(query, self.appearance.get(lighting_label)))
        if capture_timestamp > 0:
            for label, selection in (("era_older", capture_timestamp < self.cutoffs),
                                     ("era_newer", capture_timestamp >= self.cutoffs)):
                values = self._similarities(query, self.appearance.get(label))
                best = np.maximum(best, np.where(selection & (self.cutoffs > 0), values, -np.inf))
        negative_best = self._similarities(query, self.negatives)
        candidates = []
        for index, name in enumerate(self.names):
            center, prototype = float(centers[index]), float(best[index])
            raw = 1.0 - max(center, 0.68 * prototype + 0.32 * center)
            negative = None if not np.isfinite(negative_best[index]) else 1.0 - float(negative_best[index])
            distance = raw
            radius = max(1e-6, float(hard_negative_radius))
            if negative is not None and negative < radius:
                distance += max(0.0, float(hard_negative_penalty)) * (1.0 - negative / radius)
            candidates.append(IdentityCandidate(name, 1.0 - distance, distance, raw, negative))
        return sorted(candidates, key=lambda item: (item.distance, item.name.casefold()))


def rank_candidates(
    embedding: np.ndarray,
    identities: Mapping[str, np.ndarray],
    prototypes: Mapping[str, Sequence[np.ndarray]] | None = None,
    *,
    pose_label: str = "unknown",
    pose_prototypes: Mapping[str, Mapping[str, Sequence[np.ndarray]]] | None = None,
    lighting_label: str = "unknown",
    capture_timestamp: float = 0.0,
    appearance_prototypes: Mapping[str, Mapping[str, Sequence[np.ndarray]]] | None = None,
    appearance_era_cutoffs: Mapping[str, float] | None = None,
    hard_negatives: Mapping[str, Sequence[np.ndarray]] | None = None,
    hard_negative_radius: float = 0.28,
    hard_negative_penalty: float = 0.42,
    compiled: CompiledProfiles | None = None,
) -> list[IdentityCandidate]:
    if compiled is not None:
        return compiled.rank(embedding, pose_label=pose_label, lighting_label=lighting_label,
            capture_timestamp=capture_timestamp, hard_negative_radius=hard_negative_radius,
            hard_negative_penalty=hard_negative_penalty)
    prototype_map = prototypes or {}
    pose_map = pose_prototypes or {}
    appearance_map = appearance_prototypes or {}
    era_cutoffs = appearance_era_cutoffs or {}
    negative_map = hard_negatives or {}
    query = normalize_vector(embedding)
    candidates: list[IdentityCandidate] = []
    for name, centroid in identities.items():
        profile_values = list(prototype_map.get(name) or ())
        specific_values = list(pose_map.get(name, {}).get(pose_label) or ())
        if specific_values:
            profile_values.extend(specific_values)
        appearance_labels: list[str] = []
        if lighting_label in {"low_light", "normal_light"}:
            appearance_labels.append(lighting_label)
        cutoff = float(era_cutoffs.get(name, 0.0) or 0.0)
        if capture_timestamp > 0 and cutoff > 0:
            appearance_labels.append(
                "era_older" if capture_timestamp < cutoff else "era_newer"
            )
        for label in appearance_labels:
            profile_values.extend(appearance_map.get(name, {}).get(label) or ())
        similarity = profile_similarity(query, centroid, profile_values)
        raw_distance = 1.0 - similarity
        negative_distance: float | None = None
        adjusted_distance = raw_distance
        negatives = list(negative_map.get(name) or ())
        if negatives:
            negative_matrix = normalize_matrix(negatives)
            negative_distance = float(np.min(1.0 - negative_matrix @ query))
            radius = max(1e-6, float(hard_negative_radius))
            if negative_distance < radius:
                proximity = 1.0 - negative_distance / radius
                adjusted_distance += max(0.0, float(hard_negative_penalty)) * proximity
        candidates.append(IdentityCandidate(
            name=name,
            similarity=1.0 - adjusted_distance,
            distance=adjusted_distance,
            raw_distance=raw_distance,
            hard_negative_distance=negative_distance,
        ))
    return sorted(candidates, key=lambda candidate: (candidate.distance, candidate.name.casefold()))


def calibrated_thresholds(
    samples: Sequence[ReferenceSample],
    centroid: np.ndarray,
    prototypes: Sequence[np.ndarray],
    *,
    maximum_consensus_distance: float,
) -> tuple[float, float]:
    """Return conservative (consensus, strict-single) distance thresholds."""
    distances = [
        1.0 - profile_similarity(sample.embedding, centroid, prototypes)
        for sample in samples
    ]
    if distances:
        observed = float(np.percentile(np.asarray(distances, dtype=np.float32), 90))
        consensus = float(np.clip(observed + 0.10, 0.28, maximum_consensus_distance))
    else:
        consensus = min(0.34, maximum_consensus_distance)
    strict = float(np.clip(consensus - 0.07, 0.20, 0.31))
    return consensus, strict


def impostor_aware_thresholds(
    target_name: str,
    identities: Mapping[str, np.ndarray],
    prototypes: Mapping[str, Sequence[np.ndarray]],
    *,
    base_consensus_distance: float,
    base_strict_distance: float,
    consensus_clearance: float = 0.02,
    strict_clearance: float = 0.04,
) -> tuple[float, float, float]:
    """Tighten thresholds using the closest competing identity evidence.

    The returned third value is the nearest observed impostor distance. A very
    close look-alike or duplicate identity name can intentionally reduce an
    automatic threshold to zero; review is safer than guessing in that case.
    """
    if target_name not in identities:
        return base_consensus_distance, base_strict_distance, 1.0
    target_centroid = identities[target_name]
    target_prototypes = prototypes.get(target_name, [target_centroid])
    impostor_distances: list[float] = []
    for other_name, other_centroid in identities.items():
        if other_name == target_name:
            continue
        evidence = [other_centroid, *prototypes.get(other_name, [])]
        impostor_distances.extend(
            1.0 - profile_similarity(value, target_centroid, target_prototypes)
            for value in evidence
        )
    if not impostor_distances:
        return base_consensus_distance, base_strict_distance, 1.0
    nearest = max(0.0, float(min(impostor_distances)))
    consensus = min(
        float(base_consensus_distance),
        max(0.0, nearest - max(0.0, float(consensus_clearance))),
    )
    strict = min(
        float(base_strict_distance),
        consensus,
        max(0.0, nearest - max(0.0, float(strict_clearance))),
    )
    return consensus, strict, nearest


def candidate_margin(candidates: Sequence[IdentityCandidate]) -> float:
    if not candidates:
        return 0.0
    if len(candidates) == 1:
        return 1.0
    return float(candidates[1].distance - candidates[0].distance)


def source_count(samples: Iterable[ReferenceSample]) -> int:
    return len({sample.source for sample in samples})

"""Run-scoped, read-only heldout profiles for identity evaluation.

Call validate() explicitly before trusting a completed run/checkpoint. Paths are
captured on first use (including failures), never refreshed within a session.
Only retained evidence variants are LRU-bounded; path validation metadata must
remain alive for the whole run. No query embeddings or crops are cached.
"""

from __future__ import annotations

import hashlib
import os
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

import appearance_profiles
import identity_hard_negatives
import identity_profiles


NUMERICAL_EPSILON = 1e-5
# Extra guards for the secondary verifier and independent joint/rescue policy.
# Scoring itself preserves scalar arithmetic even at unlisted custom cutoffs.
_POLICY_DISTANCE_BOUNDARIES = (0.18, 0.24, 0.30, 0.32, 0.38, 0.42, 0.58)
_POLICY_MARGIN_BOUNDARIES = (0.08, 0.14, 0.20)


class EvaluationInputsChanged(RuntimeError):
    """A captured input changed; discard this session and its results."""


def _version(path, *, follow_symlinks=True):
    try:
        value = os.stat(path, follow_symlinks=follow_symlinks)
    except OSError as error:
        return ("error", error.errno)
    return (value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns,
            value.st_ctime_ns, value.st_mode)


def _links(path):
    """Capture symlinks in prefixes and in their targets, including link chains."""
    pending, visited, result = [os.path.abspath(path)], set(), {}
    while pending:
        candidate = Path(pending.pop())
        for prefix in (*reversed(candidate.parents), candidate):
            text = str(prefix)
            if text in visited:
                continue
            visited.add(text)
            if not prefix.is_symlink():
                continue
            try:
                target = os.readlink(prefix)
            except OSError as error:
                result[text] = ("error", error.errno)
                continue
            result[text] = (_version(prefix, follow_symlinks=False), target)
            pending.append(os.path.abspath(os.path.join(prefix.parent, target)))
    return tuple(sorted(result.items()))


def _path_state(path):
    # same_holdout_group compares the literal realpath but expands ~ for hashing.
    expanded = os.path.expanduser(path)
    return (os.path.realpath(path), os.path.realpath(expanded),
            _version(expanded), _version(expanded, follow_symlinks=False),
            _links(expanded))


def _sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class _CapturedPath:
    state: tuple
    digest: str | None
    error: int | None


@dataclass(frozen=True)
class _Group:
    values: tuple
    matrix: np.ndarray
    tokens: tuple[int, ...]
    valid: bool = True

    def without(self, excluded):
        retained = [i for i, token in enumerate(self.tokens) if token not in excluded]
        if self.valid and len(retained) == len(self.values):
            return self
        if not self.valid:
            retained = []
        matrix = self.matrix[retained]
        matrix.setflags(write=False)
        return _Group(tuple(self.values[i] for i in retained), matrix,
                      tuple(self.tokens[i] for i in retained), self.valid)


@dataclass(frozen=True)
class _Profile:
    name: str
    centroid: np.ndarray
    center: np.ndarray
    base: _Group
    pose: dict
    appearance: dict
    negatives: _Group
    cutoff: float

    @property
    def tokens(self):
        return frozenset(token for group in
                         (self.base, self.negatives, *self.pose.values(),
                          *self.appearance.values()) for token in group.tokens)

    def without(self, excluded, *, recompute=False):
        base = self.base.without(excluded)
        if not base.values:
            return None
        centroid, center = self.centroid, self.center
        if recompute or base is not self.base:
            centroid = identity_profiles.normalize_vector(base.matrix.mean(axis=0))
            center = identity_profiles.normalize_vector(centroid)
            centroid.setflags(write=False)
            center.setflags(write=False)
        return replace(self, centroid=centroid, center=center, base=base,
                       pose={label: group.without(excluded) for label, group in self.pose.items()},
                       appearance={label: group.without(excluded)
                                   for label, group in self.appearance.items()},
                       negatives=self.negatives.without(excluded))

    def rank(self, query, negative_query, pose, light, timestamp, combine):
        groups = [("base", "", self.base), ("pose", pose, self.pose.get(pose))]
        if light in {"low_light", "normal_light"}:
            groups.append(("appearance", light, self.appearance.get(light)))
        if timestamp > 0 and self.cutoff > 0:
            label = "era_older" if timestamp < self.cutoff else "era_newer"
            groups.append(("appearance", label, self.appearance.get(label)))
        groups = [(kind, label, group) for kind, label, group in groups
                  if group is not None and group.values]
        # Keep the scalar ranker's row order and GEMV shape. Separate group
        # products can differ by an ULP, including at downstream policy cutoffs.
        best = float(np.max(combine(self.name, groups) @ query)) if groups else -np.inf
        center = float(self.center @ query)
        raw = 1.0 - max(center, 0.68 * best + 0.32 * center)
        negative = (float(np.min(1.0 - self.negatives.matrix @ negative_query))
                    if self.negatives.values else None)
        distance = raw
        if negative is not None and negative < 0.28:
            distance += 0.42 * (1.0 - negative / 0.28)
        return identity_profiles.IdentityCandidate(
            self.name, 1.0 - distance, distance, raw, negative)


class EvaluationProfiles:
    """Snapshot a DB and its hard negatives; rank without per-face file scans.

    max_cached_profiles bounds each of the affected-person and combined-matrix
    LRUs, not entire DB copies. Unchanged matrices are shared across variants.
    negative_examples=None loads the configured file; [] disables negatives.
    hard_negatives_path defaults lazily to sort_photos' configured evidence file.
    validate() checks versions, link targets and failed reads without rehashing.
    A new instance is required after a validation failure or DB/policy refresh.
    """

    def __init__(self, db, negative_examples=None, *, max_cached_profiles=128,
                 hard_negatives_path=None):
        import sort_photos

        if not isinstance(max_cached_profiles, int) or max_cached_profiles < 0:
            raise ValueError("max_cached_profiles must be a nonnegative integer")
        self._db = db
        self._limit = max_cached_profiles
        self._cache = OrderedDict()
        self._matrix_cache = OrderedDict()
        self._paths = {}
        self._hashes = {}
        self._timestamps = {}
        self._canonical_tokens = defaultdict(set)
        self._digest_tokens = defaultdict(set)
        self._source_tokens = {}
        self._next_token = 0
        self._negative_values = []
        self._minimum_refs = sort_photos.AUTO_PERSON_MATCH_MIN_REFERENCE_FACES
        self._eligible = {name for name in db.identities
                          if db.source_counts.get(name, 0) >= self._minimum_refs}
        self._thresholds = {
            name: (min(sort_photos.AUTO_PERSON_MATCH_DIST,
                       db.match_thresholds.get(name, sort_photos.AUTO_PERSON_MATCH_DIST)),
                   min(sort_photos.AUTO_PERSON_SINGLE_MATCH_DIST,
                       db.strict_thresholds.get(name, sort_photos.AUTO_PERSON_SINGLE_MATCH_DIST)))
            for name in db.identities}
        self._margins = (sort_photos.AUTO_PERSON_MATCH_MARGIN,
                         sort_photos.AUTO_PERSON_SINGLE_MATCH_MARGIN, *_POLICY_MARGIN_BOUNDARIES)
        if negative_examples is None:
            path = Path(hard_negatives_path if hard_negatives_path is not None
                        else sort_photos.IDENTITY_HARD_NEGATIVES_FILE)
            captured = self._capture(str(path))
            examples = identity_hard_negatives.load(path)["examples"]
            self._check(str(path), captured)
        else:
            examples = negative_examples
        full_negatives, held_negatives = defaultdict(list), defaultdict(list)
        for item in examples:
            origin = str(item.get("source_path") or "")
            token = self._register(origin, unique=True) if origin else -1
            value = identity_hard_negatives.decode_embedding(item)
            if value is None:
                continue
            value.setflags(write=False)
            person = str(item.get("person", ""))
            if person.strip():
                full_negatives[person.strip()].append((value, token))
            if origin:
                held_negatives[person].append((value, token))
                self._negative_values.append((token, value))
        self._held_negatives = {name: self._negative_group(rows)
                                for name, rows in held_negatives.items()}
        full_negatives = {name: self._negative_group(rows)
                          for name, rows in full_negatives.items()}
        self._empty = self._negative_group([])
        self._full, self._held, self._tokens = {}, {}, {}
        for name in sorted(db.identities, key=str.casefold):
            centroid = self._copy(db.identities[name])
            values = tuple(self._copy(value) for value in db.prototypes.get(name, [centroid]))
            pose = {label: tuple(self._copy(value) for value in values)
                    for label, values in db.pose_prototypes.get(name, {}).items()}
            appearance = {label: tuple(self._copy(value) for value in values)
                          for label, values in db.appearance_prototypes.get(name, {}).items()}
            compiled = identity_profiles.CompiledProfiles(
                {name: centroid}, {name: values}, pose_prototypes={name: pose},
                appearance_prototypes={name: appearance})
            # The scalar ranker normalizes centers once, CompiledProfiles twice.
            center = identity_profiles.normalize_vector(centroid)
            center.setflags(write=False)
            profile = _Profile(name, centroid, center,
                self._group(values, compiled.base[0], db.prototype_sources.get(name, [])),
                {label: self._group(values, compiled.pose[label][0],
                                    db.pose_prototype_sources.get(name, {}).get(label, []))
                 for label, values in pose.items()},
                {label: self._group(values, compiled.appearance[label][0],
                                    db.appearance_prototype_sources.get(name, {}).get(label, []))
                 for label, values in appearance.items()},
                full_negatives.get(name, self._empty),
                float(db.appearance_era_cutoffs.get(name, 0.0) or 0.0))
            self._full[name] = profile
            held = replace(profile, negatives=self._held_negatives.get(name, self._empty))
            self._held[name] = held.without(frozenset(), recompute=True)
            self._tokens[name] = self._held[name].tokens if self._held[name] else frozenset()

    @staticmethod
    def _copy(value):
        result = np.asarray(value).copy()
        result.setflags(write=False)
        return result

    @staticmethod
    def _negative_group(rows):
        values = tuple(value for value, _ in rows)
        matrix = identity_profiles.normalize_matrix(values)
        matrix.setflags(write=False)
        return _Group(values, matrix, tuple(token for _, token in rows))

    def _group(self, values, matrix, sources):
        tokens = tuple(self._register(str(source)) for source in sources)
        return _Group(values, matrix, tokens, len(values) == len(tokens))

    def _capture(self, path):
        path = str(path)
        if path in self._paths:
            return self._paths[path]
        state = _path_state(path)
        hash_key = state[1], state[2]
        if hash_key not in self._hashes:
            try:
                result = (_sha256(state[1]), None)
            except OSError as error:
                result = (None, error.errno)
            if _path_state(path) != state:
                raise EvaluationInputsChanged(f"Input changed during capture: {path}")
            self._hashes[hash_key] = result
        captured = _CapturedPath(state, *self._hashes[hash_key])
        self._paths[path] = captured
        return captured

    def _register(self, path, *, unique=False):
        if not unique and path in self._source_tokens:
            return self._source_tokens[path]
        captured = self._capture(path)
        token = self._next_token
        self._next_token += 1
        self._canonical_tokens[captured.state[0]].add(token)
        if captured.digest is not None:
            self._digest_tokens[captured.digest].add(token)
        if not unique:
            self._source_tokens[path] = token
        return token

    def _excluded(self, source, excluded_sources):
        excluded = set()
        for path in {str(source), *(str(path) for path in excluded_sources)}:
            captured = self._capture(path)
            excluded.update(self._canonical_tokens.get(captured.state[0], ()))
            if captured.digest is not None:
                excluded.update(self._digest_tokens.get(captured.digest, ()))
        return excluded

    def _query_negatives(self, query, excluded):
        for token, value in self._negative_values:
            if token not in excluded and np.allclose(value, query, atol=1e-6):
                excluded.add(token)

    def _profile(self, name, excluded):
        affected = self._tokens[name].intersection(excluded)
        if not affected:
            return self._held[name]
        key = name, affected
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        profile = self._held[name].without(affected)
        if self._limit:
            self._cache[key] = profile
            if len(self._cache) > self._limit:
                self._cache.popitem(last=False)
        return profile

    def _attributes(self, face):
        path = str(face.src_str)
        captured = self._capture(path)
        if path not in self._timestamps:
            # Bypass the process-global path/mtime/size EXIF cache for this run.
            self._timestamps[path] = (0.0 if captured.state[2][0] == "error" else
                appearance_profiles._capture_timestamp_cached.__wrapped__(
                    captured.state[1], captured.state[2][3], captured.state[2][2]))
            self._check(path, captured)
        return appearance_profiles.lighting_label(face.crop_jpeg), self._timestamps[path]

    def _combined_matrix(self, name, groups):
        if len(groups) == 1:
            return groups[0][2].matrix
        key = name, tuple((kind, label, group.tokens) for kind, label, group in groups)
        if key in self._matrix_cache:
            self._matrix_cache.move_to_end(key)
            return self._matrix_cache[key]
        matrix = np.concatenate([group.matrix for _, _, group in groups])
        matrix.setflags(write=False)
        if self._limit:
            self._matrix_cache[key] = matrix
            if len(self._matrix_cache) > self._limit:
                self._matrix_cache.popitem(last=False)
        return matrix

    def rank(self, face, excluded_sources=frozenset(), exclude_profile_source=True,
             minimum_references=True):
        """Return predict_face candidates; policy lanes can disable minrefs only."""
        query = identity_profiles.normalize_vector(face.embedding)
        # profile_similarity normalizes rank_candidates' normalized query again;
        # negative evidence instead uses the first normalization.
        profile_query = identity_profiles.normalize_vector(query)
        excluded = self._excluded(face.src_str, excluded_sources) if exclude_profile_source else set()
        if exclude_profile_source:
            self._query_negatives(query, excluded)
        profiles = [self._profile(name, excluded) if exclude_profile_source else profile
                    for name, profile in self._full.items()
                    if not minimum_references or name in self._eligible]
        profiles = [profile for profile in profiles if profile is not None]
        light, timestamp = self._attributes(face)
        pose = str(getattr(face, "pose_label", "unknown") or "unknown")
        candidates = sorted((profile.rank(profile_query, query, pose, light, timestamp,
                                           self._combined_matrix)
                             for profile in profiles), key=lambda item: (item.distance, item.name.casefold()))
        if self._near_boundary(candidates):
            return self._scalar_rank(face.embedding, profiles, pose, light, timestamp)
        return candidates

    def _near_boundary(self, candidates):
        for candidate in candidates[:1]:
            if any(abs(candidate.distance - threshold) <= NUMERICAL_EPSILON
                   for threshold in (*self._thresholds[candidate.name], *_POLICY_DISTANCE_BOUNDARIES)):
                return True
            if (candidate.hard_negative_distance is not None and
                    abs(candidate.hard_negative_distance - 0.28) <= NUMERICAL_EPSILON):
                return True
        if any(abs(right.distance - left.distance) <= 2 * NUMERICAL_EPSILON
               for left, right in zip(candidates[:3], candidates[1:4])):
            return True
        margin = identity_profiles.candidate_margin(candidates)
        return any(abs(margin - threshold) <= 2 * NUMERICAL_EPSILON for threshold in self._margins)

    @staticmethod
    def _scalar_rank(embedding, profiles, pose, light, timestamp):
        return identity_profiles.rank_candidates(
            embedding, {p.name: p.centroid for p in profiles},
            {p.name: p.base.values for p in profiles}, pose_label=pose,
            pose_prototypes={p.name: {label: group.values for label, group in p.pose.items()}
                             for p in profiles}, lighting_label=light, capture_timestamp=timestamp,
            appearance_prototypes={p.name: {label: group.values for label, group in p.appearance.items()}
                                   for p in profiles},
            appearance_era_cutoffs={p.name: p.cutoff for p in profiles},
            hard_negatives={p.name: p.negatives.values for p in profiles})

    def heldout_db(self, source, excluded_sources=frozenset()):
        """Equivalent to heldout_identity_db, deliberately without minrefs."""
        excluded = self._excluded(source, excluded_sources)
        profiles = [self._profile(name, excluded) for name in self._db.identities]
        profiles = [profile for profile in profiles if profile is not None]
        return replace(self._db, identities={p.name: p.centroid for p in profiles},
            prototypes={p.name: list(p.base.values) for p in profiles},
            pose_prototypes={p.name: {label: list(group.values) for label, group in p.pose.items()
                                      if group.valid} for p in profiles},
            appearance_prototypes={p.name: {label: list(group.values)
                for label, group in p.appearance.items() if group.valid} for p in profiles})

    def heldout_negatives(self, source, embedding, excluded_sources=frozenset()):
        """Equivalent to heldout_hard_negatives using this session's snapshot."""
        excluded = self._excluded(source, excluded_sources)
        self._query_negatives(identity_profiles.normalize_vector(embedding), excluded)
        return {name: [value for value, token in zip(group.values, group.tokens) if token not in excluded]
                for name, group in self._held_negatives.items()
                if any(token not in excluded for token in group.tokens)}

    @staticmethod
    def _check(path, captured):
        if _path_state(path) != captured.state:
            raise EvaluationInputsChanged(f"Captured input changed or disappeared: {path}")
        if captured.digest is None:
            try:
                with open(captured.state[1], "rb") as handle:
                    handle.read(1)
            except OSError as error:
                if error.errno == captured.error:
                    return
            raise EvaluationInputsChanged(f"Captured input readability changed: {path}")

    def validate(self):
        """Fail on changed/missing inputs or resolved failures; never rehash."""
        for path, captured in self._paths.items():
            self._check(path, captured)

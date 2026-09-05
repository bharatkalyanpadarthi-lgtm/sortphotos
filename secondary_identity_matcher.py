#!/usr/bin/env python3
"""Independent InsightFace verifier for borderline primary-model matches."""

from __future__ import annotations

import argparse
import hashlib
import os
import pickle
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import identity_confirmations
import identity_profiles
import pipeline_paths


MODEL_NAME = "buffalo_l"
CACHE_VERSION = 1
DEFAULT_MODEL_ROOT = pipeline_paths.configured_path(
    "secondary_model_root",
    pipeline_paths.DATA_ROOT / "models" / "insightface",
    "FACE_SECONDARY_MODEL_ROOT",
)
DEFAULT_CACHE = pipeline_paths.configured_path(
    "secondary_identity_cache",
    pipeline_paths.DATA_ROOT / "analysis_cache" / "secondary_identity_db.pkl",
    "FACE_SECONDARY_IDENTITY_CACHE",
)
TRUSTED_REBUILD_INTERVAL = 250


@dataclass
class SecondaryIdentityDB:
    version: int = CACHE_VERSION
    model_name: str = MODEL_NAME
    primary_signature: str = ""
    trusted_signature: str = ""
    trusted_example_count: int = 0
    identities: dict[str, np.ndarray] = field(default_factory=dict)
    prototypes: dict[str, list[np.ndarray]] = field(default_factory=dict)
    prototype_sources: dict[str, list[str]] = field(default_factory=dict)
    thresholds: dict[str, float] = field(default_factory=dict)
    source_counts: dict[str, int] = field(default_factory=dict)
    crop_embeddings: dict[str, np.ndarray] = field(default_factory=dict)


@dataclass(frozen=True)
class SecondaryVerification:
    accepted: bool
    predicted: str
    distance: float
    margin: float


def primary_signature(primary_db) -> str:
    digest = hashlib.sha256()
    digest.update(str(getattr(primary_db, "config_fingerprint", "")).encode("utf-8"))
    for name in sorted(primary_db.identities, key=str.casefold):
        digest.update(name.encode("utf-8", errors="surrogateescape"))
        sources = primary_db.prototype_sources.get(name, [])
        prototypes = primary_db.prototypes.get(name, [])
        for source, prototype in zip(sources, prototypes):
            digest.update(str(source).encode("utf-8", errors="surrogateescape"))
            digest.update(
                identity_profiles.normalize_vector(prototype).astype(np.float32).tobytes()
            )
    return digest.hexdigest()


def trusted_confirmation_signature(
    path: Path | None = None,
) -> str:
    """Fingerprint explicit confirmations used to train verifier profiles."""
    source = (path or (Path.home() / ".face_sort_cache" / "confirmed_identity_examples.json"))
    source = source.expanduser()
    digest = hashlib.sha256()
    digest.update(str(source.resolve(strict=False)).encode("utf-8", errors="surrogateescape"))
    try:
        with source.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        digest.update(b"\0missing")
    return digest.hexdigest()


def trusted_confirmation_count(path: Path | None = None) -> int:
    source = (
        path or (Path.home() / ".face_sort_cache" / "confirmed_identity_examples.json")
    ).expanduser()
    return len(identity_confirmations.load(source).get("examples", []))


def trusted_snapshot_is_current(
    db: SecondaryIdentityDB | None,
    path: Path | None = None,
    *,
    rebuild_interval: int = TRUSTED_REBUILD_INTERVAL,
) -> bool:
    """Allow bounded new confirmations to accumulate before a verifier rebuild.

    Review confirmations arrive interactively. Rebuilding a native model after
    every click would make the dashboard unusable. Exact snapshots are reused;
    append-only growth is batched, while removals or same-count edits force a
    rebuild immediately.
    """
    if db is None:
        return False
    source = (
        path or (Path.home() / ".face_sort_cache" / "confirmed_identity_examples.json")
    ).expanduser()
    current_signature = trusted_confirmation_signature(source)
    if getattr(db, "trusted_signature", "") == current_signature:
        return True
    stored_count = int(getattr(db, "trusted_example_count", 0))
    current_count = trusted_confirmation_count(source)
    added = current_count - stored_count
    return 0 < added < max(1, int(rebuild_interval))


def identity_signature(db: SecondaryIdentityDB | None) -> str:
    """Fingerprint verifier profiles without including its growing crop cache."""
    if db is None:
        return "missing"
    digest = hashlib.sha256()
    digest.update(str(db.version).encode("ascii"))
    digest.update(db.model_name.encode("utf-8"))
    digest.update(db.primary_signature.encode("ascii"))
    digest.update(str(getattr(db, "trusted_signature", "")).encode("ascii"))
    for name in sorted(db.identities, key=str.casefold):
        digest.update(name.encode("utf-8", errors="surrogateescape"))
        digest.update(
            identity_profiles.normalize_vector(db.identities[name])
            .astype(np.float32).tobytes()
        )
        for prototype in db.prototypes.get(name, []):
            digest.update(
                identity_profiles.normalize_vector(prototype)
                .astype(np.float32).tobytes()
            )
        for source in getattr(db, "prototype_sources", {}).get(name, []):
            digest.update(str(source).encode("utf-8", errors="surrogateescape"))
        digest.update(repr(float(db.thresholds.get(name, 0.0))).encode("ascii"))
    return digest.hexdigest()


def load(path: Path = DEFAULT_CACHE) -> SecondaryIdentityDB | None:
    try:
        main_module = sys.modules.get("__main__")
        if main_module is not None and not hasattr(main_module, "SecondaryIdentityDB"):
            setattr(main_module, "SecondaryIdentityDB", SecondaryIdentityDB)
        with path.expanduser().open("rb") as handle:
            value = pickle.load(handle)
    except (OSError, ValueError, TypeError, pickle.PickleError):
        return None
    if not isinstance(value, SecondaryIdentityDB) or value.version != CACHE_VERSION:
        return None
    if not hasattr(value, "trusted_signature"):
        value.trusted_signature = ""
    if not hasattr(value, "trusted_example_count"):
        value.trusted_example_count = 0
    if not hasattr(value, "prototype_sources"):
        value.prototype_sources = {}
    return value


def save(db: SecondaryIdentityDB, path: Path = DEFAULT_CACHE) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            pickle.dump(db, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        Path(temporary).replace(path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise


def build_app(model_root: Path = DEFAULT_MODEL_ROOT):
    from insightface.app import FaceAnalysis

    model_root = model_root.expanduser()
    model_root.mkdir(parents=True, exist_ok=True)
    app = FaceAnalysis(
        name=MODEL_NAME,
        root=str(model_root),
        providers=["CPUExecutionProvider"],
        allowed_modules=["detection", "recognition"],
    )
    app.prepare(ctx_id=-1, det_size=(320, 320))
    return app


def crop_key(crop_jpeg: bytes) -> str:
    return hashlib.sha256(crop_jpeg).hexdigest()


def embed_crop(crop_jpeg: bytes, app) -> np.ndarray | None:
    image = cv2.imdecode(np.frombuffer(crop_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        return None
    faces = app.get(image)
    if not faces:
        padded = cv2.copyMakeBorder(image, 80, 80, 80, 80, cv2.BORDER_REPLICATE)
        faces = app.get(padded)
    if not faces:
        return None
    height, width = image.shape[:2]

    def score(face) -> tuple[float, float]:
        bbox = np.asarray(face.bbox, dtype=np.float32)
        area = max(0.0, float((bbox[2] - bbox[0]) * (bbox[3] - bbox[1])))
        center_x = (float(bbox[0]) + float(bbox[2])) / 2.0
        center_y = (float(bbox[1]) + float(bbox[3])) / 2.0
        center_distance = abs(center_x - width / 2.0) + abs(center_y - height / 2.0)
        return area, -center_distance

    face = max(faces, key=score)
    embedding = getattr(face, "normed_embedding", None)
    if embedding is None:
        return None
    return identity_profiles.normalize_vector(np.asarray(embedding, dtype=np.float32))


class SecondaryMatcher:
    def __init__(self, db: SecondaryIdentityDB, *, app=None, cache_path: Path = DEFAULT_CACHE):
        self.db = db
        self.app = app
        self.cache_path = cache_path
        self.dirty = False
        self.compiled = identity_profiles.CompiledProfiles(db.identities, db.prototypes)

    def embedding(self, crop_jpeg: bytes) -> np.ndarray | None:
        key = crop_key(crop_jpeg)
        cached = self.db.crop_embeddings.get(key)
        if cached is not None:
            return np.asarray(cached, dtype=np.float32)
        if self.app is None:
            self.app = build_app()
        value = embed_crop(crop_jpeg, self.app)
        if value is not None:
            self.db.crop_embeddings[key] = value
            self.dirty = True
        return value

    def verify(
        self,
        crop_jpeg: bytes,
        expected_person: str,
        *,
        excluded_source: Path | str | None = None,
    ) -> SecondaryVerification:
        embedding = self.embedding(crop_jpeg)
        if embedding is None or expected_person not in self.db.identities:
            return SecondaryVerification(False, "", 1.0, 0.0)
        identities = self.db.identities
        prototypes = self.db.prototypes
        if excluded_source is not None:
            from identity_evaluation import same_holdout_group
            filtered_prototypes: dict[str, list[np.ndarray]] = {}
            filtered_identities = {}
            for name, values in prototypes.items():
                sources = self.db.prototype_sources.get(name, [])
                kept = [
                    value for value, source in zip(values, sources)
                    if not same_holdout_group(str(source), str(excluded_source))
                ] if len(sources) == len(values) else []
                filtered_prototypes[name] = kept
                if kept:
                    filtered_identities[name] = identity_profiles.normalize_vector(
                        np.mean(identity_profiles.normalize_matrix(kept), axis=0)
                    )
            identities = filtered_identities
            prototypes = filtered_prototypes
        candidates = identity_profiles.rank_candidates(
            embedding, identities, prototypes,
            compiled=self.compiled if excluded_source is None else None,
        )
        if not candidates:
            return SecondaryVerification(False, "", 1.0, 0.0)
        best = candidates[0]
        margin = identity_profiles.candidate_margin(candidates)
        threshold = min(0.30, self.db.thresholds.get(best.name, 0.30))
        accepted = (
            best.name == expected_person
            and best.distance <= threshold
            and margin >= 0.08
        )
        return SecondaryVerification(
            accepted, best.name, float(best.distance), float(margin)
        )

    def flush(self) -> None:
        if self.dirty:
            save(self.db, self.cache_path)
            self.dirty = False


def build_database(
    primary_db,
    cache,
    *,
    maximum_per_person: int = 16,
    maximum_core_per_person: int = 6,
    maximum_trusted_per_person: int = 12,
    confirmations_path: Path | None = None,
) -> SecondaryIdentityDB:
    """Build independent profiles from core references plus trusted corrections.

    The verifier previously saw only four core references. Explicit review
    confirmations now supply bounded, diverse difficult poses without changing
    global thresholds or allowing automatic labels to train the verifier.
    """
    signature = primary_signature(primary_db)
    confirmations_path = (
        confirmations_path
        or (Path.home() / ".face_sort_cache" / "confirmed_identity_examples.json")
    ).expanduser()
    trusted_signature = trusted_confirmation_signature(confirmations_path)
    trusted_example_count = trusted_confirmation_count(confirmations_path)
    existing = load()
    if (
        existing is not None
        and existing.primary_signature == signature
        and trusted_snapshot_is_current(existing, confirmations_path)
        and existing.identities
        and all(
            len(existing.prototype_sources.get(name, []))
            == len(existing.prototypes.get(name, []))
            for name in existing.identities
        )
    ):
        return existing
    db = existing or SecondaryIdentityDB()
    db.primary_signature = signature
    db.trusted_signature = trusted_signature
    db.trusted_example_count = trusted_example_count
    app = build_app()
    faces_by_source: dict[str, list] = defaultdict(list)
    faces_by_person: dict[str, list] = defaultdict(list)
    for face in cache.faces:
        faces_by_source[os.path.realpath(face.src_str)].append(face)
        if face.label and face.crop_jpeg:
            faces_by_person[str(face.label).casefold()].append(face)
    canonical_names = {
        name.casefold(): name for name in primary_db.identities
    }
    trusted_faces: dict[str, list] = defaultdict(list)
    trusted_seen: set[tuple[str, str, int]] = set()
    for record in identity_confirmations.load(confirmations_path).get("examples", []):
        person = canonical_names.get(str(record.get("person", "")).strip().casefold())
        if person is None:
            continue
        source = identity_confirmations.resolve_record(record, faces_by_source)
        if source is None:
            continue
        source_key = str(source)
        candidates = faces_by_source.get(source_key, [])
        candidates = identity_confirmations.selected_faces(record, candidates)
        if not candidates:
            continue
        face = max(candidates, key=lambda value: float(value.quality))
        key = (person.casefold(), source_key, int(face.face_index))
        if key in trusted_seen:
            continue
        trusted_seen.add(key)
        trusted_faces[person].append(face)
    identities: dict[str, np.ndarray] = {}
    prototypes: dict[str, list[np.ndarray]] = {}
    prototype_sources: dict[str, list[str]] = {}
    thresholds: dict[str, float] = {}
    counts: dict[str, int] = {}
    for person_index, person in enumerate(sorted(primary_db.identities, key=str.casefold), 1):
        primary_prototypes = primary_db.prototypes.get(person, [])
        sources = primary_db.prototype_sources.get(person, [])
        selected_faces: list = []
        used_faces: set[tuple[str, int]] = set()
        for primary_prototype, source in list(zip(primary_prototypes, sources))[:max(1, maximum_core_per_person)]:
            candidates = faces_by_source.get(os.path.realpath(source), [])
            if not candidates:
                # Newer identity databases use stable cache prototype IDs
                # rather than absolute paths. Resolve those IDs from the
                # labeled face cache so SSD moves and renames do not break the
                # independent verifier.
                candidates = faces_by_person.get(person.casefold(), [])
            candidates = [
                face for face in candidates
                if face.crop_jpeg
                and (os.path.realpath(face.src_str), int(face.face_index)) not in used_faces
            ]
            if not candidates:
                continue
            primary_face = min(
                candidates,
                key=lambda face: 1.0 - float(
                    identity_profiles.normalize_vector(face.embedding)
                    @ identity_profiles.normalize_vector(primary_prototype)
                ),
            )
            used_faces.add((os.path.realpath(primary_face.src_str), int(primary_face.face_index)))
            selected_faces.append(primary_face)

        trusted_samples = [
            identity_profiles.ReferenceSample(
                source=str(face.src_str),
                embedding=np.asarray(face.embedding, dtype=np.float32),
                quality=float(face.quality),
                pose_label=str(getattr(face, "pose_label", "unknown") or "unknown"),
            )
            for face in trusted_faces.get(person, [])
            if (os.path.realpath(face.src_str), int(face.face_index)) not in used_faces
        ]
        chosen_trusted = identity_profiles.select_diverse_samples(
            trusted_samples,
            limit=max(0, int(maximum_trusted_per_person)),
        )
        trusted_lookup = {
            (sample.source, bytes(identity_profiles.normalize_vector(sample.embedding).tobytes())): sample
            for sample in chosen_trusted
        }
        for face in trusted_faces.get(person, []):
            marker = (
                str(face.src_str),
                bytes(identity_profiles.normalize_vector(face.embedding).tobytes()),
            )
            if marker not in trusted_lookup:
                continue
            key = (os.path.realpath(face.src_str), int(face.face_index))
            if key in used_faces:
                continue
            used_faces.add(key)
            selected_faces.append(face)

        samples: list[identity_profiles.ReferenceSample] = []
        for primary_face in selected_faces:
            key = crop_key(primary_face.crop_jpeg)
            secondary = db.crop_embeddings.get(key)
            if secondary is None:
                secondary = embed_crop(primary_face.crop_jpeg, app)
                if secondary is not None:
                    db.crop_embeddings[key] = secondary
            if secondary is not None:
                samples.append(identity_profiles.ReferenceSample(
                    source=str(primary_face.src_str),
                    embedding=secondary,
                    quality=float(primary_face.quality),
                    pose_label=str(getattr(primary_face, "pose_label", "unknown")),
                ))
        if len(samples) >= 2:
            selected = identity_profiles.select_diverse_samples(
                samples, limit=max(2, int(maximum_per_person))
            )
            centroid = identity_profiles.weighted_centroid(selected)
            person_prototypes = [sample.embedding for sample in selected]
            consensus, strict = identity_profiles.calibrated_thresholds(
                samples, centroid, person_prototypes, maximum_consensus_distance=0.32
            )
            identities[person] = centroid
            prototypes[person] = person_prototypes
            prototype_sources[person] = [sample.source for sample in selected]
            thresholds[person] = min(0.30, strict + 0.03)
            counts[person] = len(samples)
        if person_index % 10 == 0:
            print(f"Secondary profiles {person_index}/{len(primary_db.identities)}", flush=True)
    db.identities = identities
    db.prototypes = prototypes
    db.prototype_sources = prototype_sources
    for person in identities:
        _consensus, strict, _nearest = identity_profiles.impostor_aware_thresholds(
            person,
            identities,
            prototypes,
            base_consensus_distance=min(0.32, thresholds.get(person, 0.30) + 0.04),
            base_strict_distance=thresholds.get(person, 0.30),
            consensus_clearance=0.02,
            strict_clearance=0.04,
        )
        thresholds[person] = min(thresholds.get(person, 0.30), strict + 0.03)
    db.thresholds = thresholds
    db.source_counts = counts
    save(db)
    return db


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if not args.build:
        parser.error("use --build")
    import sort_photos
    primary = sort_photos.load_identity_db()
    if primary is None:
        print("ERROR: primary identity DB is unavailable")
        return 2
    if args.force:
        DEFAULT_CACHE.unlink(missing_ok=True)
    db = build_database(primary, sort_photos.load_cache())
    print(f"Secondary identity DB: {len(db.identities)} people")
    print(f"Model root:            {DEFAULT_MODEL_ROOT}")
    print(f"Cache:                 {DEFAULT_CACHE}")
    return 0 if db.identities else 3


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build low-light and capture-era identity prototype buckets without redetection."""

from __future__ import annotations

import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import appearance_profiles
import evaluation_enrollment
import identity_confirmations
import identity_evaluation
import identity_profiles
import pipeline_paths
import sort_photos


DEFAULT_REPORT = (
    pipeline_paths.SOURCE_REVIEW
    / "identity_evaluation"
    / "latest_appearance_profile_refresh.json"
)


def refresh() -> int:
    incumbent = sort_photos.load_identity_db()
    if incumbent is None or not incumbent.identities:
        print("ERROR: primary identity database is unavailable")
        return 2
    cache = sort_photos.load_cache()
    faces_by_source: dict[str, list[sort_photos.CachedFace]] = defaultdict(list)
    faces_by_person_source: dict[str, dict[str, sort_photos.CachedFace]] = defaultdict(dict)
    for face in cache.faces:
        faces_by_source[os.path.realpath(face.src_str)].append(face)
        if face.label:
            current = faces_by_person_source[face.label].get(face.src_str)
            if current is None or face.quality > current.quality:
                faces_by_person_source[face.label][face.src_str] = face

    candidate = pickle.loads(pickle.dumps(incumbent, protocol=pickle.HIGHEST_PROTOCOL))
    candidate = sort_photos.normalize_identity_db(candidate)
    counts: dict[str, int] = defaultdict(int)
    people_with_era = 0
    for person in sorted(candidate.identities, key=str.casefold):
        prototypes = list(candidate.prototypes.get(person, []))
        sources = list(candidate.prototype_sources.get(person, []))
        samples: list[identity_profiles.ReferenceSample] = []
        for embedding, source in zip(prototypes, sources):
            faces = faces_by_source.get(os.path.realpath(source), [])
            if not faces:
                continue
            target = identity_profiles.normalize_vector(embedding)
            face = min(
                faces,
                key=lambda value: 1.0 - float(
                    identity_profiles.normalize_vector(value.embedding) @ target
                ),
            )
            samples.append(identity_profiles.ReferenceSample(
                source=source,
                embedding=target,
                quality=float(face.quality),
                pose_label=str(getattr(face, "pose_label", "unknown") or "unknown"),
                lighting_label=appearance_profiles.lighting_label(face.crop_jpeg),
                capture_timestamp=appearance_profiles.capture_timestamp(source),
            ))
        # Capture-era coverage is intentionally drawn from the wider labeled
        # cache because many selected profile prototypes have stripped EXIF.
        # Dominant-component filtering keeps isolated wrong-folder faces from
        # becoming era prototypes.
        dated_samples: list[identity_profiles.ReferenceSample] = []
        for source, face in faces_by_person_source.get(person, {}).items():
            captured_at = appearance_profiles.capture_timestamp(source)
            if captured_at <= 0:
                continue
            dated_samples.append(identity_profiles.ReferenceSample(
                source=source,
                embedding=identity_profiles.normalize_vector(face.embedding),
                quality=float(face.quality),
                pose_label=str(getattr(face, "pose_label", "unknown") or "unknown"),
                lighting_label=appearance_profiles.lighting_label(face.crop_jpeg),
                capture_timestamp=captured_at,
            ))
        dated_samples = identity_profiles.dominant_identity_samples(dated_samples)
        cutoff = appearance_profiles.era_cutoff(
            [sample.capture_timestamp for sample in dated_samples]
        )
        if cutoff > 0:
            people_with_era += 1
        values: dict[str, list] = {}
        value_sources: dict[str, list[str]] = {}
        for label in ("low_light", "normal_light", "era_older", "era_newer"):
            evidence = dated_samples if label.startswith("era_") else samples
            matching = [
                sample for sample in evidence
                if label in appearance_profiles.labels(
                    light=sample.lighting_label,
                    timestamp=sample.capture_timestamp,
                    person_era_cutoff=cutoff,
                )
            ]
            selected = identity_profiles.select_diverse_samples(matching, limit=2)
            if selected:
                values[label] = [sample.embedding for sample in selected]
                value_sources[label] = [sample.source for sample in selected]
                counts[label] += len(selected)
        candidate.appearance_prototypes[person] = values
        candidate.appearance_prototype_sources[person] = value_sources
        candidate.appearance_era_cutoffs[person] = cutoff
    candidate.calibration_version = sort_photos.IDENTITY_CALIBRATION_VERSION

    evaluation_enrollment.backfill_confirmations(
        identity_confirmations.load(sort_photos.IDENTITY_CONFIRMATIONS_FILE),
        cache.faces,
    )
    allowed, gate = identity_evaluation.activation_gate(
        candidate,
        incumbent,
        cache,
        confirmed_set=evaluation_enrollment.DEFAULT_PATH,
    )
    report = {
        "people": len(candidate.identities),
        "people_with_capture_era": people_with_era,
        "prototype_counts": dict(sorted(counts.items())),
        "activation_gate": gate,
        "activated": allowed,
    }
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print("Appearance Profile Refresh")
    print("=" * 60)
    print(f"People:                  {len(candidate.identities)}")
    print(f"People with capture era: {people_with_era}")
    for label, count in sorted(counts.items()):
        print(f"{label:24} {count}")
    if not allowed:
        print("ERROR: appearance profiles were blocked by the identity activation gate")
        print(f"Report: {DEFAULT_REPORT}")
        return 4
    sort_photos.save_identity_db(candidate)
    print("Appearance-aware identity database activated safely.")
    print(f"Report: {DEFAULT_REPORT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(refresh())

#!/usr/bin/env python3
"""Read-only accuracy evaluation for the Face Sort identity matcher."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

import appearance_profiles
import identity_profiles
import identity_hard_negatives
import evaluation_dataset
import pipeline_paths
import sort_photos
import content_identity
import benchmark_detection


DEFAULT_REPORT_DIR = pipeline_paths.SOURCE_REVIEW / "identity_evaluation"
DEFAULT_PROTECTED_SET = DEFAULT_REPORT_DIR / "protected_identity_benchmark.csv"
DEFAULT_PROTECTED_BASELINE = DEFAULT_REPORT_DIR / "protected_identity_baseline.json"


@dataclass(frozen=True)
class EvaluationResult:
    source: str
    expected: str
    predicted: str
    outcome: str
    distance: float
    margin: float
    quality: float


@dataclass(frozen=True)
class FacePrediction:
    predicted: str
    distance: float
    margin: float
    accepted: bool


def profile_without_source(
    db: sort_photos.IdentityDB,
    name: str,
    excluded_source: str,
    excluded_sources: frozenset[str] = frozenset(),
) -> tuple[np.ndarray | None, list[np.ndarray]]:
    centroid = db.identities[name]
    prototypes = list(db.prototypes.get(name, [centroid]))
    sources = list(db.prototype_sources.get(name, []))
    if len(sources) != len(prototypes):
        return None, []
    retained = [
        prototype for prototype, source in zip(prototypes, sources)
        if not same_holdout_group(source, excluded_source, excluded_sources)
    ]
    if not retained:
        return None, []
    retained_matrix = identity_profiles.normalize_matrix(retained)
    return identity_profiles.normalize_vector(retained_matrix.mean(axis=0)), retained


def same_holdout_group(source: str, query: str, excluded_sources: frozenset[str] = frozenset()) -> bool:
    excluded = excluded_sources | {query}
    canonical = os.path.realpath(source)
    if canonical in {os.path.realpath(path) for path in excluded}:
        return True
    try:
        digest = content_identity.content_sha256(source)
    except OSError:
        return False
    for path in excluded:
        try:
            if digest == content_identity.content_sha256(path):
                return True
        except OSError:
            continue
    return False


def heldout_identity_db(db, source, excluded_sources=frozenset()):
    """Remove the tested content/source group from every primary profile."""
    identities, prototypes = {}, {}
    for name in db.identities:
        center, values = profile_without_source(db, name, str(source), excluded_sources)
        if center is not None:
            identities[name], prototypes[name] = center, values

    def filtered(groups, provenance):
        output = {}
        for name in identities:
            output[name] = {}
            for label, values in groups.get(name, {}).items():
                sources = provenance.get(name, {}).get(label, [])
                if len(values) == len(sources):
                    output[name][label] = [value for value, origin in zip(values, sources)
                        if not same_holdout_group(origin, str(source), excluded_sources)]
        return output

    return replace(db, identities=identities, prototypes=prototypes,
        pose_prototypes=filtered(db.pose_prototypes, db.pose_prototype_sources),
        appearance_prototypes=filtered(db.appearance_prototypes, db.appearance_prototype_sources))


def heldout_hard_negatives(source, embedding, excluded_sources=frozenset()):
    result = {}
    for item in identity_hard_negatives.load(sort_photos.IDENTITY_HARD_NEGATIVES_FILE)["examples"]:
        origin = str(item.get("source_path") or "")
        if not origin or same_holdout_group(origin, str(source), excluded_sources):
            continue
        value = identity_hard_negatives.decode_embedding(item)
        if value is None or np.allclose(value, identity_profiles.normalize_vector(embedding), atol=1e-6):
            continue
        result.setdefault(str(item.get("person", "")), []).append(value)
    return result


def predict_face(
    face: sort_photos.CachedFace,
    db: sort_photos.IdentityDB,
    *,
    lane: str,
    exclude_profile_source: bool = True,
    excluded_sources: frozenset[str] = frozenset(),
) -> FacePrediction | None:
    identities: dict[str, np.ndarray] = {}
    profile_map: dict[str, list[np.ndarray]] = {}
    pose_map: dict[str, dict[str, list[np.ndarray]]] = {}
    appearance_map: dict[str, dict[str, list[np.ndarray]]] = {}
    for name in sorted(db.identities, key=str.casefold):
        if db.source_counts.get(name, 0) < sort_photos.AUTO_PERSON_MATCH_MIN_REFERENCE_FACES:
            continue
        if exclude_profile_source:
            centroid, prototypes = profile_without_source(db, name, face.src_str, excluded_sources)
            if centroid is None:
                continue
        else:
            centroid = db.identities[name]
            prototypes = list(db.prototypes.get(name, [centroid]))
        identities[name] = centroid
        profile_map[name] = prototypes
        person_pose: dict[str, list[np.ndarray]] = {}
        for pose, values in db.pose_prototypes.get(name, {}).items():
            sources = db.pose_prototype_sources.get(name, {}).get(pose, [])
            if exclude_profile_source and len(sources) == len(values):
                retained = [
                    value for value, source in zip(values, sources)
                    if not same_holdout_group(source, face.src_str, excluded_sources)
                ]
            else:
                retained = [] if exclude_profile_source else list(values)
            if retained:
                person_pose[pose] = retained
        pose_map[name] = person_pose
        person_appearance: dict[str, list[np.ndarray]] = {}
        for label, values in db.appearance_prototypes.get(name, {}).items():
            sources = db.appearance_prototype_sources.get(name, {}).get(label, [])
            if exclude_profile_source and len(sources) == len(values):
                retained = [
                    value for value, source in zip(values, sources)
                    if not same_holdout_group(source, face.src_str, excluded_sources)
                ]
            else:
                retained = [] if exclude_profile_source else list(values)
            if retained:
                person_appearance[label] = retained
        appearance_map[name] = person_appearance
    lighting, captured_at = appearance_profiles.query_attributes(
        face.crop_jpeg, face.src_str
    )
    candidates = identity_profiles.rank_candidates(
        face.embedding,
        identities,
        profile_map,
        pose_label=str(getattr(face, "pose_label", "unknown") or "unknown"),
        pose_prototypes=pose_map,
        lighting_label=lighting,
        capture_timestamp=captured_at,
        appearance_prototypes=appearance_map,
        appearance_era_cutoffs=db.appearance_era_cutoffs,
        hard_negatives=(heldout_hard_negatives(face.src_str, face.embedding, excluded_sources)
            if exclude_profile_source else identity_hard_negatives.vectors_by_person(
                sort_photos.IDENTITY_HARD_NEGATIVES_FILE)),
    )
    if not candidates:
        return None
    best = candidates[0]
    margin = identity_profiles.candidate_margin(candidates)
    if lane == "strict":
        threshold = min(
            sort_photos.AUTO_PERSON_SINGLE_MATCH_DIST,
            db.strict_thresholds.get(best.name, sort_photos.AUTO_PERSON_SINGLE_MATCH_DIST),
        )
        minimum_margin = sort_photos.AUTO_PERSON_SINGLE_MATCH_MARGIN
        quality_ok = face.quality >= sort_photos.AUTO_PERSON_SINGLE_MIN_QUALITY
    else:
        threshold = min(
            sort_photos.AUTO_PERSON_MATCH_DIST,
            db.match_thresholds.get(best.name, sort_photos.AUTO_PERSON_MATCH_DIST),
        )
        minimum_margin = sort_photos.AUTO_PERSON_MATCH_MARGIN
        quality_ok = True
    accepted = best.distance <= threshold and margin >= minimum_margin and quality_ok
    return FacePrediction(
        predicted=best.name if accepted else "",
        distance=best.distance,
        margin=margin,
        accepted=accepted,
    )


def evaluate_face(
    face: sort_photos.CachedFace,
    db: sort_photos.IdentityDB,
    *,
    lane: str,
) -> EvaluationResult | None:
    expected = str(face.label or "").strip()
    if not expected or expected not in db.identities:
        return None
    prediction = predict_face(face, db, lane=lane)
    if prediction is None:
        return None
    predicted = prediction.predicted
    outcome = "correct" if predicted == expected else ("incorrect" if predicted else "rejected")
    return EvaluationResult(
        source=face.src_str,
        expected=expected,
        predicted=predicted,
        outcome=outcome,
        distance=prediction.distance,
        margin=prediction.margin,
        quality=float(face.quality),
    )


def select_faces(faces: list[sort_photos.CachedFace], max_per_person: int) -> list[sort_photos.CachedFace]:
    grouped: dict[str, list[sort_photos.CachedFace]] = defaultdict(list)
    for face in faces:
        if face.label:
            grouped[str(face.label)].append(face)
    selected: list[sort_photos.CachedFace] = []
    for name in sorted(grouped, key=str.casefold):
        ordered_group = sorted(
            grouped[name],
            key=lambda face: (face.src_str, face.face_index),
        )
        # Dominant-component selection is quadratic. A deterministic spread of
        # references is ample for a holdout audit and keeps very large people
        # folders from making the evaluator progressively slower.
        pool_limit = max(160, max_per_person * 10) if max_per_person > 0 else 800
        if len(ordered_group) > pool_limit:
            indexes = np.linspace(0, len(ordered_group) - 1, num=pool_limit, dtype=int)
            ordered_group = [ordered_group[int(index)] for index in sorted(set(indexes.tolist()))]
        samples: list[identity_profiles.ReferenceSample] = []
        faces_by_sample: dict[int, sort_photos.CachedFace] = {}
        for face in ordered_group:
            sample = identity_profiles.ReferenceSample(
                source=face.src_str,
                embedding=face.embedding,
                quality=float(face.quality),
            )
            samples.append(sample)
            faces_by_sample[id(sample)] = face
        dominant = identity_profiles.dominant_identity_samples(samples)
        best_by_source: dict[str, sort_photos.CachedFace] = {}
        for sample in dominant:
            face = faces_by_sample[id(sample)]
            current = best_by_source.get(face.src_str)
            if current is None or face.quality > current.quality:
                best_by_source[face.src_str] = face
        ordered = sorted(
            best_by_source.values(),
            key=lambda face: (-face.quality, face.src_str, face.face_index),
        )
        selected.extend(ordered[:max_per_person] if max_per_person > 0 else ordered)
    return selected


def cache_metrics(
    db: sort_photos.IdentityDB,
    cache: sort_photos.CacheState,
    *,
    max_per_person: int = 100,
) -> dict[str, float | int]:
    results = [
        result
        for face in select_faces(cache.faces, max_per_person)
        if (result := evaluate_face(face, db, lane="strict")) is not None
    ]
    correct = sum(result.outcome == "correct" for result in results)
    incorrect = sum(result.outcome == "incorrect" for result in results)
    rejected = sum(result.outcome == "rejected" for result in results)
    accepted = correct + incorrect
    return {
        "evaluated": len(results),
        "correct": correct,
        "incorrect": incorrect,
        "rejected": rejected,
        "precision": correct / max(1, accepted),
        "recall": correct / max(1, len(results)),
    }


def activation_gate(
    candidate: sort_photos.IdentityDB,
    incumbent: sort_photos.IdentityDB,
    cache: sort_photos.CacheState,
    *,
    confirmed_set: Path | None = None,
    protected_set: Path = DEFAULT_PROTECTED_SET,
    protected_baseline: Path = DEFAULT_PROTECTED_BASELINE,
    require_protected: bool | None = None,
) -> tuple[bool, dict[str, object]]:
    """Block profile promotion when precision or known-person recall regresses."""
    current = cache_metrics(candidate, cache)
    prior = current if candidate is incumbent else cache_metrics(incumbent, cache)
    failures: list[str] = []
    if require_protected is None:
        require_protected = candidate is not incumbent
    if int(current["incorrect"]) > int(prior["incorrect"]):
        failures.append("new incorrect strict identity accepts")
    if int(current["incorrect"]) > 0 or float(current["precision"]) < 0.999:
        failures.append("strict accepted precision is below 99.9%")
    if float(current["recall"]) + 0.02 < float(prior["recall"]):
        failures.append("strict known-person recall dropped by more than 2 points")

    confirmed_summary: dict[str, object] = {"available": False}
    if confirmed_set is not None and confirmed_set.is_file():
        validation = evaluation_dataset.load_dataset(confirmed_set)
        confirmed_summary = {
            "available": True,
            "cases": len(validation.cases),
            "errors": list(validation.errors),
        }
        if validation.errors:
            failures.append("confirmed evaluation set is invalid")
        else:
            metrics, rows = evaluate_golden_set(
                validation.cases,
                cache,
                candidate,
                lane="consensus",
                exclude_profile_sources=True,
            )
            prior_metrics, prior_rows = evaluate_golden_set(
                validation.cases,
                cache,
                incumbent,
                lane="consensus",
                exclude_profile_sources=True,
            )
            incorrect_rows = [
                row for row in rows
                if row["identity_outcome"] == "incorrect"
            ]
            prior_incorrect_rows = [
                row for row in prior_rows
                if row["identity_outcome"] == "incorrect"
            ]
            confirmed_summary.update({
                "precision": metrics.identity_precision,
                "recall": metrics.known_case_recall,
                "incorrect": len(incorrect_rows),
                "prior_precision": prior_metrics.identity_precision,
                "prior_recall": prior_metrics.known_case_recall,
                "prior_incorrect": len(prior_incorrect_rows),
            })
            if len(incorrect_rows) > len(prior_incorrect_rows):
                failures.append("confirmed unknown benchmark has new incorrect accepts")
            if metrics.known_case_recall + 0.02 < prior_metrics.known_case_recall:
                failures.append("confirmed unknown benchmark recall dropped by more than 2 points")
    protected_summary: dict[str, object] = {"available": False}
    if require_protected and not protected_set.is_file():
        failures.append("profile promotion requires the verified protected benchmark")
    if require_protected and not protected_baseline.is_file():
        failures.append("profile promotion requires a fresh-detection protected baseline")
    if protected_baseline.is_file() and not protected_set.is_file():
        failures.append("protected benchmark is missing while its baseline exists")
    if protected_set.is_file():
        validation = evaluation_dataset.load_dataset(protected_set)
        protected_summary = {
            "available": True,
            "cases": len(validation.cases),
            "covered_case_types": sorted(validation.covered_types),
            "activation_ready": validation.activation_ready,
            "errors": list(validation.errors),
        }
        # Keep an incumbent available while a draft is completed, but do not
        # promote a new profile through an incomplete benchmark.
        if (require_protected or protected_baseline.is_file()) and not validation.activation_ready:
            failures.append("protected benchmark is incomplete or invalid")
        if validation.activation_ready:
            detected_faces = {}
            metrics, rows = evaluate_golden_set(
                validation.cases,
                cache,
                candidate,
                lane="strict",
                exclude_profile_sources=True,
                fresh_detection=bool(require_protected),
                detected_faces=detected_faces,
            )
            pipeline_metrics, pipeline_rows = evaluate_golden_set(
                validation.cases, cache, candidate, lane="pipeline",
                fresh_detection=bool(require_protected), detected_faces=detected_faces)
            protected_summary["pipeline_metrics"] = asdict(pipeline_metrics)
            if any(row["identity_outcome"] in {"incorrect", "false_accept"} for row in pipeline_rows):
                failures.append("protected daily filing planner has an incorrect accept")
            protected_summary["metrics"] = asdict(metrics)
            protected_summary["incorrect_rows"] = sum(
                row["identity_outcome"] in {"incorrect", "false_accept"}
                for row in rows
            )
            if int(protected_summary["incorrect_rows"]) > 0:
                failures.append("protected benchmark has an incorrect identity accept")
            if protected_baseline.is_file():
                try:
                    evidence = json.loads(protected_baseline.read_text(encoding="utf-8"))
                    if require_protected and evidence.get("evaluation_mode") != "fresh_detection":
                        failures.append("protected baseline has no fresh-detection provenance")
                    baseline = evaluation_dataset.load_baseline(protected_baseline)
                    regressions = evaluation_dataset.compare_to_baseline(metrics, baseline)
                except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                    regressions = [f"protected baseline is unreadable: {exc}"]
                protected_summary["regressions"] = regressions
                failures.extend(f"protected benchmark: {item}" for item in regressions)
    return not failures, {
        "candidate": current,
        "incumbent": prior,
        "confirmed": confirmed_summary,
        "protected": protected_summary,
        "failures": failures,
    }


def create_golden_template(
    path: Path,
    db: sort_photos.IdentityDB,
    *,
    max_per_person: int = 2,
) -> None:
    rows: list[dict[str, str]] = []
    for name in sorted(db.identities, key=str.casefold):
        sources = db.prototype_sources.get(name, [])[:max(1, max_per_person)]
        for source in sources:
            rows.append({
                "source": source,
                "expected_person": name,
                "case_types": "known",
                "expected_face": "true",
                "expected_nudity": "unknown",
                "verified": "false",
                "notes": "Verify the person and add applicable case types before activation.",
            })
    evaluation_dataset.write_template(path, rows)


def evaluate_golden_set(
    cases: tuple[evaluation_dataset.EvaluationCase, ...],
    cache: sort_photos.CacheState,
    db: sort_photos.IdentityDB,
    *,
    lane: str,
    exclude_profile_sources: bool = True,
    fresh_detection: bool = False,
    detected_faces: dict | None = None,
) -> tuple[evaluation_dataset.EvaluationMetrics, list[dict[str, object]]]:
    faces_by_source: dict[str, list[sort_photos.CachedFace]] = defaultdict(list)
    if fresh_detection:
        faces_by_source.update(benchmark_detection.detect_cases(
            cases, detected_faces=detected_faces))
    else:
        for face in cache.faces:
            faces_by_source[os.path.realpath(face.src_str)].append(face)
    groups: dict[str, set[str]] = defaultdict(set)
    for case in cases:
        groups[case.group_id or case.content_sha256 or str(case.source)].add(str(case.source))
    shadow = None
    if lane == "pipeline":
        import shadow_evaluation
        excluded = frozenset(str(case.source) for case in cases)
        heldout = heldout_identity_db(db, next(iter(excluded), ""), excluded)
        selected_faces = [face for case in cases
                          for face in faces_by_source.get(os.path.realpath(str(case.source)), [])]
        negatives = heldout_hard_negatives(next(iter(excluded), ""), np.zeros(512), excluded)
        shadow = shadow_evaluation.daily_plan(selected_faces, heldout, hard_negatives=negatives)

    known_cases = 0
    known_correct = 0
    identity_accepted = 0
    identity_correct = 0
    unknown_cases = 0
    unknown_rejected = 0
    expected_face_cases = 0
    missed_face_cases = 0
    no_face_cases = 0
    no_face_correct = 0
    nudity_cases = 0
    nudity_correct = 0
    safe_nudity_cases = 0
    nudity_false_positives = 0
    rows: list[dict[str, object]] = []

    with sort_photos.analysis_index.AnalysisIndex(sort_photos.analysis_index_file()) as index:
        for case_number, case in enumerate(cases, start=1):
            source_faces = faces_by_source.get(os.path.realpath(str(case.source)), [])
            identity_faces = source_faces
            if case.identity_face_id:
                identity_faces = [face for face in source_faces
                    if content_identity.face_identity(face) == case.identity_face_id]
                if len(identity_faces) != 1:
                    raise RuntimeError(f"Selected benchmark face is missing or ambiguous; "
                                       f"reverify the face selection: {case.source}")
            predictions = [
                prediction for face in identity_faces
                if (
                    prediction := predict_face(
                        face,
                        db,
                        lane=lane,
                        exclude_profile_source=exclude_profile_sources,
                        excluded_sources=frozenset(groups[case.group_id or case.content_sha256 or str(case.source)]),
                    )
                ) is not None
            ] if shadow is None else []
            accepted_names = [prediction.predicted for prediction in predictions if prediction.accepted]
            if shadow is not None:
                selected_indices = {face.face_index for face in identity_faces} if case.identity_face_id else None
                accepted_names = [item["person"] for item in shadow.get(str(case.source.resolve()), [])
                    if item["person"] and (selected_indices is None or item["face_index"] in selected_indices)]

            identity_outcome = "not_scored"
            expected_names = Counter(case.expected_people or ((case.expected_person,) if case.expected_person else ()))
            observed_names = Counter(accepted_names)
            if expected_names:
                known_cases += sum(expected_names.values())
                identity_accepted += len(accepted_names)
                correct = sum((expected_names & observed_names).values())
                known_correct += correct
                identity_correct += correct
                if observed_names - expected_names:
                    identity_outcome = "incorrect"
                elif observed_names == expected_names:
                    identity_outcome = "correct"
                elif accepted_names:
                    identity_outcome = "partial"
                else:
                    identity_outcome = "rejected"
            elif {"unknown", "no_face"} & case.case_types:
                unknown_cases += 1
                if not accepted_names:
                    unknown_rejected += 1
                    identity_outcome = "correct_rejection"
                else:
                    identity_accepted += len(accepted_names)
                    identity_outcome = "false_accept"

            if case.expected_face:
                expected_count = case.expected_face_count or max(1, sum(expected_names.values()))
                expected_face_cases += expected_count
                missed_face_cases += max(0, expected_count - len(source_faces))
            if "no_face" in case.case_types:
                no_face_cases += 1
                if not source_faces:
                    no_face_correct += 1

            observed_nudity = "unknown"
            if case.expected_nudity != "unknown":
                nudity_cases += 1
                if case.expected_nudity == "safe":
                    safe_nudity_cases += 1
                try:
                    file_hash = sort_photos.sha256_file(case.source)
                except OSError:
                    file_hash = ""
                observed_nudity = sort_photos.classified_nudity_status(
                    case.source,
                    file_hash=file_hash or None,
                    asset_index=index,
                )
                if observed_nudity == case.expected_nudity:
                    nudity_correct += 1
                if case.expected_nudity == "safe" and observed_nudity == "possible":
                    nudity_false_positives += 1

            rows.append({
                "source": str(case.source),
                "case_types": "|".join(sorted(case.case_types)),
                "expected_person": case.expected_person,
                "accepted_names": "|".join(accepted_names),
                "identity_outcome": identity_outcome,
                "faces_detected": len(source_faces),
                "identity_face_id": case.identity_face_id,
                "identity_faces_scored": len(identity_faces),
                "identity_faces_ignored": len(source_faces) - len(identity_faces),
                "expected_nudity": case.expected_nudity,
                "observed_nudity": observed_nudity,
                "evaluation_mode": "fresh_detection" if fresh_detection else "cached_matching_only",
                "decision_lane": lane,
            })
            if case_number % 100 == 0 or case_number == len(cases):
                print(f"Protected scoring: completed {case_number}/{len(cases)}", flush=True)

    metrics = evaluation_dataset.EvaluationMetrics(
        identity_precision=identity_correct / max(1, identity_accepted),
        known_case_recall=known_correct / max(1, known_cases),
        unknown_rejection_rate=unknown_rejected / max(1, unknown_cases),
        missed_face_rate=missed_face_cases / max(1, expected_face_cases),
        no_face_specificity=no_face_correct / max(1, no_face_cases),
        nudity_accuracy=nudity_correct / max(1, nudity_cases),
        nudity_false_positive_rate=nudity_false_positives / max(1, safe_nudity_cases),
        verified_cases=len(cases),
    )
    return metrics, rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=("strict", "consensus", "pipeline"), default="strict")
    parser.add_argument("--max-per-person", type=int, default=100)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--fail-below-precision", type=float, default=0.0)
    parser.add_argument("--golden-set", type=Path, default=None,
                        help="Manually verified protected evaluation CSV.")
    parser.add_argument("--create-golden-template", type=Path, default=None,
                        help="Write an unverified template from current profile sources.")
    parser.add_argument("--baseline", type=Path, default=None,
                        help="Fail when golden-set metrics regress from this baseline JSON.")
    parser.add_argument("--write-baseline", type=Path, default=None,
                        help="Write golden-set metrics as a protected activation baseline.")
    parser.add_argument("--fresh-detection", action="store_true",
                        help="Detect benchmark images again without changing the library or face cache.")
    args = parser.parse_args()
    if args.write_baseline is not None and not args.fresh_detection:
        parser.error("--write-baseline requires --fresh-detection; cached matching cannot prove detection quality")
    if args.lane == "pipeline" and args.golden_set is None:
        parser.error("--lane pipeline requires a protected --golden-set")

    db = sort_photos.load_identity_db()
    if db is None or not db.identities:
        print("ERROR: identity DB is missing. Run `face rebuild-id` first.")
        return 2
    if args.create_golden_template is not None:
        create_golden_template(args.create_golden_template, db)
        print(f"Golden-set template: {args.create_golden_template.expanduser().resolve()}")
        print("Review every row, add all required case types, then set verified=true.")
        return 0
    cache = sort_photos.CacheState() if args.fresh_detection else sort_photos.load_cache()
    if args.golden_set is not None:
        def dataset_signature() -> str:
            try:
                return content_identity.content_sha256(args.golden_set)
            except OSError:
                return ""

        dataset_sha256 = dataset_signature()
        validation = evaluation_dataset.load_dataset(args.golden_set)
        if validation.errors:
            print("ERROR: protected evaluation set is not valid:")
            for error in validation.errors:
                print(f"  - {error}")
            return 4
        if not dataset_sha256 or dataset_signature() != dataset_sha256:
            print("ERROR: benchmark changed while loading; retry with the saved annotations")
            return 7
        metrics, golden_rows = evaluate_golden_set(
            validation.cases, cache, db, lane=args.lane, fresh_detection=args.fresh_detection)
        report_dir = args.report_dir.expanduser().resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        csv_path = report_dir / "golden_set_results.csv"
        json_path = report_dir / "golden_set_summary.json"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            fieldnames = list(golden_rows[0]) if golden_rows else ["source"]
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(golden_rows)
        payload = asdict(metrics)
        payload["covered_case_types"] = sorted(validation.covered_types)
        payload["activation_ready"] = validation.activation_ready
        payload["evaluation_mode"] = "fresh_detection" if args.fresh_detection else "cached_matching_only"
        payload["detector_signature"] = sort_photos.config_fingerprint()
        payload["dataset_sha256"] = dataset_sha256
        payload["dataset_unchanged"] = dataset_signature() == dataset_sha256
        payload["identity_scoped_cases"] = sum(bool(case.identity_face_id) for case in validation.cases)
        json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        print("Protected Evaluation Set")
        print("=" * 60)
        print(f"Verified cases:        {metrics.verified_cases}")
        print(f"Identity precision:    {metrics.identity_precision:.2%}")
        print(f"Known-case recall:     {metrics.known_case_recall:.2%}")
        print(f"Unknown rejection:     {metrics.unknown_rejection_rate:.2%}")
        print(f"Missed-face rate:      {metrics.missed_face_rate:.2%}")
        print(f"No-face specificity:  {metrics.no_face_specificity:.2%}")
        print(f"Nudity accuracy:       {metrics.nudity_accuracy:.2%}")
        print(f"Nudity false positive:{metrics.nudity_false_positive_rate:>7.2%}")
        print(f"Dataset categories:    {'complete' if validation.activation_ready else 'incomplete'}")
        print(f"Results:               {csv_path}")
        print(f"Summary:               {json_path}")

        if not payload["dataset_unchanged"] or dataset_signature() != dataset_sha256:
            print("ERROR: benchmark annotations changed during evaluation; "
                  "report retained for diagnosis, activation blocked. Run again with the saved annotations.")
            return 7

        needs_activation_gate = args.baseline is not None or args.write_baseline is not None
        if needs_activation_gate and not validation.activation_ready:
            missing = evaluation_dataset.REQUIRED_CASE_TYPES - validation.covered_types
            print("ERROR: activation blocked; missing verified case types: "
                  + ", ".join(sorted(missing)))
            return 5
        if args.baseline is not None:
            try:
                baseline = evaluation_dataset.load_baseline(args.baseline)
            except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
                print(f"ERROR: could not load activation baseline: {exc}")
                return 5
            regressions = evaluation_dataset.compare_to_baseline(metrics, baseline)
            if regressions:
                print("ERROR: activation blocked by metric regressions:")
                for regression in regressions:
                    print(f"  - {regression}")
                return 6
        if args.write_baseline is not None:
            if args.lane != "strict":
                print("ERROR: the promotion baseline must use --lane strict; pipeline metrics are reported separately")
                return 6
            if any(row["identity_outcome"] in {"incorrect", "false_accept"} for row in golden_rows):
                print("ERROR: cannot establish a baseline containing false identity accepts")
                return 6
            evaluation_dataset.write_baseline(args.write_baseline, metrics,
                detector_signature=sort_photos.config_fingerprint(),
                dataset_sha256=dataset_sha256)
            print(f"Activation baseline written: {args.write_baseline.expanduser().resolve()}")
        return 0

    faces = select_faces(cache.faces, max(0, int(args.max_per_person)))
    results = [
        result for face in faces
        if (result := evaluate_face(face, db, lane=args.lane)) is not None
    ]
    counts = defaultdict(int)
    for result in results:
        counts[result.outcome] += 1
    accepted = counts["correct"] + counts["incorrect"]
    precision = counts["correct"] / max(1, accepted)
    recall = counts["correct"] / max(1, len(results))

    report_dir = args.report_dir.expanduser().resolve()
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / f"identity_{args.lane}_results.csv"
    json_path = report_dir / f"identity_{args.lane}_summary.json"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(EvaluationResult.__dataclass_fields__))
        writer.writeheader()
        for result in results:
            writer.writerow(asdict(result))
    summary = {
        "lane": args.lane,
        "evaluated": len(results),
        "correct": counts["correct"],
        "incorrect": counts["incorrect"],
        "rejected": counts["rejected"],
        "acceptedPrecision": precision,
        "acceptedRecall": recall,
        "resultCSV": str(csv_path),
    }
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    print("Identity Evaluation")
    print("=" * 60)
    print(f"Lane:               {args.lane}")
    print(f"Faces evaluated:    {len(results):,}")
    print(f"Correct:            {counts['correct']:,}")
    print(f"Incorrect:          {counts['incorrect']:,}")
    print(f"Rejected:           {counts['rejected']:,}")
    print(f"Accepted precision: {precision:.2%}")
    print(f"Accepted recall:    {recall:.2%}")
    print(f"Results:            {csv_path}")
    print(f"Summary:            {json_path}")
    if args.fail_below_precision > 0 and accepted and precision < args.fail_below_precision:
        print(f"ERROR: precision is below required {args.fail_below_precision:.2%}.")
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

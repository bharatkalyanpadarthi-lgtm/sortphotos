#!/usr/bin/env python3
"""Protected, manually verified evaluation-set format and activation gates."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import content_identity


REQUIRED_CASE_TYPES = {
    "known",
    "lookalike",
    "side_profile",
    "blurry_small",
    "group",
    "unknown",
    "no_face",
    "normal",
    "nude",
    "swimwear",
}
TRUE_VALUES = {"1", "true", "yes", "y", "verified"}
FIELDS = [
    "source", "expected_person", "case_types", "expected_face",
    "expected_nudity", "verified", "notes", "expected_people", "expected_face_count",
    "content_sha256", "group_id", "identity_face_id",
]


@dataclass(frozen=True)
class EvaluationCase:
    source: Path
    expected_person: str
    case_types: frozenset[str]
    expected_face: bool
    expected_nudity: str
    verified: bool
    notes: str = ""
    expected_people: tuple[str, ...] = ()
    content_sha256: str = ""
    group_id: str = ""
    expected_face_count: int | None = None
    identity_face_id: str = ""


@dataclass(frozen=True)
class EvaluationMetrics:
    identity_precision: float
    known_case_recall: float
    unknown_rejection_rate: float
    missed_face_rate: float
    no_face_specificity: float
    nudity_accuracy: float
    nudity_false_positive_rate: float
    verified_cases: int


@dataclass(frozen=True)
class DatasetValidation:
    cases: tuple[EvaluationCase, ...]
    errors: tuple[str, ...]
    covered_types: frozenset[str]

    @property
    def activation_ready(self) -> bool:
        return not self.errors and REQUIRED_CASE_TYPES.issubset(self.covered_types)


def parse_bool(value: str, default: bool = False) -> bool:
    clean = str(value or "").strip().casefold()
    if not clean:
        return default
    return clean in TRUE_VALUES


def parse_case_types(value: str) -> frozenset[str]:
    normalized = str(value or "").replace(",", "|").replace(";", "|")
    return frozenset(part.strip().casefold() for part in normalized.split("|") if part.strip())


def identity_scope_errors(face_id, types, expected_face, person, people, face_count):
    if not face_id:
        return []
    errors = []
    if not re.fullmatch(r"crop:[0-9a-f]{64}", face_id):
        errors.append("selected identity face has an invalid fingerprint")
    if not expected_face or len(people or ((person,) if person else ())) != 1:
        errors.append("selected-face identity scoring requires one known person")
    if "known" not in types or "group" in types or {"unknown", "no_face"} & types:
        errors.append("selected-face identity scoring cannot certify a group or unknown case")
    if face_count is None or face_count < 1:
        errors.append("selected-face identity scoring requires the total visible face count")
    return errors


def load_dataset(path: Path) -> DatasetValidation:
    errors: list[str] = []
    cases: list[EvaluationCase] = []
    covered: set[str] = set()
    try:
        handle = path.expanduser().open(newline="", encoding="utf-8")
    except OSError as exc:
        return DatasetValidation((), (f"could not open dataset: {exc}",), frozenset())
    with handle:
        reader = csv.DictReader(handle)
        required_columns = {
            "source", "expected_person", "case_types", "expected_face",
            "expected_nudity", "verified", "notes",
        }
        missing_columns = required_columns - set(reader.fieldnames or [])
        if missing_columns:
            return DatasetValidation(
                (),
                (f"missing columns: {', '.join(sorted(missing_columns))}",),
                frozenset(),
            )
        for row_number, row in enumerate(reader, start=2):
            source = Path(str(row.get("source") or "")).expanduser()
            types = parse_case_types(str(row.get("case_types") or ""))
            verified = parse_bool(str(row.get("verified") or ""))
            expected_face = parse_bool(str(row.get("expected_face") or ""), default=True)
            expected_nudity = str(row.get("expected_nudity") or "unknown").strip().casefold()
            expected_person = str(row.get("expected_person") or "").strip()
            expected_people = tuple(value.strip() for value in str(row.get("expected_people") or "").split("|") if value.strip())
            identity_face_id = str(row.get("identity_face_id") or "").strip()
            digest = str(row.get("content_sha256") or "").strip()
            try:
                actual_digest = content_identity.content_sha256(source)
                if digest and digest != actual_digest:
                    errors.append(f"row {row_number} content changed since verification")
                digest = digest or actual_digest
            except OSError:
                pass
            count_text = str(row.get("expected_face_count") or "").strip()
            try:
                face_count = int(count_text) if count_text else None
                if face_count is not None and face_count < 0:
                    raise ValueError()
            except ValueError:
                face_count = None
                errors.append(f"row {row_number} has invalid expected face count")
            if "group" in types and (not expected_people or face_count is None):
                errors.append(f"row {row_number} group needs all expected people and face count")
            if not verified:
                errors.append(f"row {row_number} is not manually verified")
            if not source.is_file():
                errors.append(f"row {row_number} source is missing: {source}")
            unknown_types = types - REQUIRED_CASE_TYPES
            if unknown_types:
                errors.append(
                    f"row {row_number} has unknown case types: {', '.join(sorted(unknown_types))}")
            if not types:
                errors.append(f"row {row_number} has no case type")
            if expected_nudity not in {"safe", "possible", "unknown"}:
                errors.append(f"row {row_number} has invalid expected_nudity: {expected_nudity}")
            if face_count is not None and (face_count < len(expected_people) or bool(face_count) != expected_face):
                errors.append(f"row {row_number} face count conflicts with labels")
            if expected_face and "unknown" not in types and not (expected_person or expected_people):
                errors.append(f"row {row_number} needs expected_person")
            errors.extend(f"row {row_number} {error}" for error in identity_scope_errors(
                identity_face_id, types, expected_face, expected_person, expected_people, face_count))
            case = EvaluationCase(
                source=source,
                expected_person=expected_person,
                case_types=types,
                expected_face=expected_face,
                expected_nudity=expected_nudity,
                verified=verified,
                notes=str(row.get("notes") or ""),
                expected_people=expected_people,
                content_sha256=digest,
                group_id=str(row.get("group_id") or digest),
                expected_face_count=face_count,
                identity_face_id=identity_face_id,
            )
            cases.append(case)
            if verified:
                covered.update(types)
    return DatasetValidation(tuple(cases), tuple(errors), frozenset(covered))


def write_template(path: Path, rows: Iterable[dict[str, str]]) -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = FIELDS
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})


def write_baseline(path: Path, metrics: EvaluationMetrics, *, detector_signature="", dataset_sha256="") -> None:
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"version": 2, "metrics": asdict(metrics),
               "evaluation_mode": "fresh_detection" if detector_signature else "unspecified",
               "detector_signature": detector_signature, "dataset_sha256": dataset_sha256}
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_baseline(path: Path) -> EvaluationMetrics:
    payload = json.loads(path.expanduser().read_text(encoding="utf-8"))
    return EvaluationMetrics(**payload.get("metrics", payload))


def compare_to_baseline(
    current: EvaluationMetrics,
    baseline: EvaluationMetrics,
    *,
    precision_drop: float = 0.005,
    recall_drop: float = 0.02,
    rejection_drop: float = 0.01,
    missed_face_increase: float = 0.01,
    nudity_accuracy_drop: float = 0.01,
    nudity_false_positive_increase: float = 0.005,
) -> list[str]:
    failures: list[str] = []
    checks = (
        (current.identity_precision + precision_drop < baseline.identity_precision,
         "identity precision regressed"),
        (current.known_case_recall + recall_drop < baseline.known_case_recall,
         "known-person recall regressed"),
        (current.unknown_rejection_rate + rejection_drop < baseline.unknown_rejection_rate,
         "unknown rejection regressed"),
        (current.missed_face_rate > baseline.missed_face_rate + missed_face_increase,
         "missed-face rate regressed"),
        (current.nudity_accuracy + nudity_accuracy_drop < baseline.nudity_accuracy,
         "nudity accuracy regressed"),
        (current.nudity_false_positive_rate
         > baseline.nudity_false_positive_rate + nudity_false_positive_increase,
         "nudity false-positive rate regressed"),
    )
    failures.extend(message for failed, message in checks if failed)
    return failures

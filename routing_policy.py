#!/usr/bin/env python3
"""Pure routing policy for nudity categories and duplicate separation."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path


DEFAULT_NUDITY_THRESHOLD = 0.80
DEFAULT_UNCERTAIN_THRESHOLD = 0.45
DEFAULT_CLASS_THRESHOLDS = {
    "FEMALE_BREAST_EXPOSED": 0.80,
    "BUTTOCKS_EXPOSED": 0.84,
    "FEMALE_GENITALIA_EXPOSED": 0.80,
    "MALE_GENITALIA_EXPOSED": 0.80,
    "ANUS_EXPOSED": 0.80,
}
EXPLICIT_CLASSES = frozenset(DEFAULT_CLASS_THRESHOLDS)
COVERED_EQUIVALENTS = {
    "FEMALE_BREAST_EXPOSED": "FEMALE_BREAST_COVERED",
    "FEMALE_GENITALIA_EXPOSED": "FEMALE_GENITALIA_COVERED",
    "BUTTOCKS_EXPOSED": "BUTTOCKS_COVERED",
    "ANUS_EXPOSED": "ANUS_COVERED",
}
COVERED_CLASSES = frozenset(COVERED_EQUIVALENTS.values())


def _detection_score(detection: dict) -> float:
    try:
        return float(detection.get("score", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _detection_box(detection: dict) -> tuple[float, float, float, float] | None:
    box = detection.get("box")
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    try:
        x, y, width, height = (float(value) for value in box)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0:
        return None
    return x, y, width, height


def _box_iou(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> float:
    left_x, left_y, left_width, left_height = left
    right_x, right_y, right_width, right_height = right
    overlap_width = max(
        0.0,
        min(left_x + left_width, right_x + right_width) - max(left_x, right_x),
    )
    overlap_height = max(
        0.0,
        min(left_y + left_height, right_y + right_height) - max(left_y, right_y),
    )
    intersection = overlap_width * overlap_height
    if intersection <= 0:
        return 0.0
    union = left_width * left_height + right_width * right_height - intersection
    return intersection / union if union > 0 else 0.0


def _covered_conflict_score(
    explicit_detection: dict,
    covered_class: str | None,
    detections: list[dict],
    scores: dict[str, float],
) -> float:
    if not covered_class:
        return 0.0
    candidates = [
        detection for detection in detections
        if str(detection.get("class", "")) == covered_class
    ]
    explicit_box = _detection_box(explicit_detection)
    boxed_candidates = [
        (detection, box)
        for detection in candidates
        if (box := _detection_box(detection)) is not None
    ]
    if explicit_box is not None and boxed_candidates:
        return max(
            (
                _detection_score(detection)
                for detection, box in boxed_candidates
                if _box_iou(explicit_box, box) >= 0.15
            ),
            default=0.0,
        )
    # Older detector output and synthetic tests may not contain boxes. Preserve
    # the conservative image-wide conflict behavior for those records.
    return scores.get(covered_class, 0.0)


def duplicate_nudity_status_for_path(path: Path) -> str:
    parts = [part.casefold() for part in path.parts]
    name = path.name.casefold()
    for index, part in enumerate(parts):
        next_part = parts[index + 1] if index + 1 < len(parts) else ""
        if part in {"photos", "all", "review"} and next_part in {
            "nude", "nudity_possible"
        }:
            return "possible"
        if part in {"photos_nude", "_possible_nudity", "nudity_possible"}:
            return "possible"
        if part in {"_uncertain_nudity", "uncertain_nudity"}:
            return "uncertain"
    if "nudity_possible" in name or "_nude" in name or "_nudity_" in name:
        return "possible"
    return "safe"


def nudity_decision(
    detections: list[dict],
    *,
    class_thresholds: dict[str, float] | None = None,
    default_threshold: float = DEFAULT_NUDITY_THRESHOLD,
    uncertain_threshold: float = DEFAULT_UNCERTAIN_THRESHOLD,
    exposed_over_covered_margin: float = 0.15,
) -> tuple[str, str, float, str]:
    """Return confirmed_nude, likely_safe, or needs_review."""
    thresholds = class_thresholds or DEFAULT_CLASS_THRESHOLDS
    scores: dict[str, float] = defaultdict(float)
    for detection in detections:
        class_name = str(detection.get("class", ""))
        try:
            score = float(detection.get("score", 0.0))
        except (TypeError, ValueError):
            continue
        scores[class_name] = max(scores[class_name], score)

    explicit: list[tuple[str, float, dict]] = []
    for class_name in EXPLICIT_CLASSES:
        candidates = [
            detection for detection in detections
            if str(detection.get("class", "")) == class_name
        ]
        if not candidates:
            continue
        best_detection = max(
            candidates,
            key=_detection_score,
        )
        explicit.append((class_name, scores[class_name], best_detection))
    covered_max = max((scores[name] for name in COVERED_CLASSES), default=0.0)
    if not explicit:
        if covered_max >= 0.70:
            return "likely_safe", "", 0.0, "strong_covered_evidence"
        return "needs_review", "", 0.0, "no_conclusive_anatomy_detection"

    best_class, best_score, best_detection = max(explicit, key=lambda item: item[1])
    covered_class = COVERED_EQUIVALENTS.get(best_class)
    best_covered_score = _covered_conflict_score(
        best_detection,
        covered_class,
        detections,
        scores,
    )
    confirmed: list[tuple[str, float]] = []
    for class_name, score, explicit_detection in explicit:
        threshold = thresholds.get(class_name, default_threshold)
        if score < threshold:
            continue
        equivalent = COVERED_EQUIVALENTS.get(class_name)
        covered_score = _covered_conflict_score(
            explicit_detection,
            equivalent,
            detections,
            scores,
        )
        if covered_score and score < covered_score + exposed_over_covered_margin:
            continue
        if class_name == "BUTTOCKS_EXPOSED" and covered_max >= 0.45:
            continue
        confirmed.append((class_name, score))

    if confirmed:
        confirmed_class, confirmed_score = max(confirmed, key=lambda item: item[1])
        return "confirmed_nude", confirmed_class, confirmed_score, "strong_explicit_evidence"
    if best_score < uncertain_threshold:
        if best_covered_score >= 0.70:
            return "likely_safe", best_class, best_score, "strong_same_anatomy_covered_evidence"
        return "needs_review", best_class, best_score, "weak_or_inconclusive_detection"
    if best_class == "BUTTOCKS_EXPOSED" and best_score < 0.78 and best_covered_score >= 0.60:
        return "likely_safe", best_class, best_score, "same_anatomy_covered_conflict"
    if best_score < 0.55 and best_covered_score >= max(0.65, best_score + 0.10):
        return "likely_safe", best_class, best_score, "same_anatomy_covered_evidence_dominates"
    return "needs_review", best_class, best_score, "explicit_not_confirmed"

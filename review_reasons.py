"""Cheap, read-only explanations using already computed matching evidence."""

import math

import identity_profiles
import recognition_policy


def explain(item, identity_db, *, settings, gate=None, enabled=False):
    def reason(code, message):
        return {"code": code, "message": message}

    if not item.candidates:
        return [reason("no_candidate", "No usable identity candidate was found.")]
    best = item.candidates[0]
    margin = identity_profiles.candidate_margin(item.candidates)
    quality = float(item.face.quality)
    if not all(math.isfinite(value) for value in (best.distance, margin, quality)):
        return [reason("invalid_evidence", "Matching evidence is incomplete; fresh analysis is needed.")]
    evidence = settings._automatic_item_evidence(item, identity_db)
    strict_allowed = recognition_policy.strict_lane_allowed(gate or {})
    eligible = not evidence["secondary_dissent"] and (
        evidence["secondary"] or evidence["secondary_rescue"]
        or (evidence["strict"] and strict_allowed))
    if eligible:
        return [reason("ready" if enabled else "safety_gate",
                       "Eligible for safe automatic filing; awaiting application."
                       if enabled else "Strong match; automatic filing is waiting for safety validation.")]
    reasons = []
    secondary = item.secondary
    if secondary is not None and secondary.predicted and secondary.predicted.casefold() != best.name.casefold():
        reasons.append(reason("matcher_disagreement",
                              f"Matchers disagree: primary suggests {best.name}; independent suggests {secondary.predicted}."))
    minimum_quality = min(settings.sort_photos.AUTO_PERSON_SINGLE_MIN_QUALITY,
                          settings.AUTO_JOINT_MIN_QUALITY, settings.AUTO_RESCUE_MIN_QUALITY)
    if quality < minimum_quality:
        reasons.append(reason("poor_face_quality", "Face quality is too low for safe automatic filing."))
    if len(item.candidates) > 1 and margin < settings.AUTO_JOINT_MIN_PRIMARY_MARGIN:
        reasons.append(reason("similar_alternative",
                              f"Similar-looking alternative: {item.candidates[1].name}; the separation is too small."))
    primary_limit = min(settings.AUTO_JOINT_MAX_PRIMARY_DISTANCE, float(evidence["threshold"]) + 0.08)
    if best.distance > primary_limit and not evidence["secondary_rescue"]:
        reasons.append(reason("weak_confidence", f"The match to {best.name} is not strong enough for automatic filing."))
    if evidence["strict"] and not strict_allowed:
        reasons.append(reason("strict_safety_gate", "Strong single-photo match; that filing path has not passed safety validation."))
    elif secondary is None:
        reasons.append(reason("independent_not_evaluated", "Independent confirmation is not available for this candidate."))
    elif not reasons:
        reasons.append(reason("weak_independent_confidence",
                              "The independent match is not strong or distinct enough for automatic filing."))
    return reasons

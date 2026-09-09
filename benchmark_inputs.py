"""Resolve explicit benchmark face selections without changing human labels."""

import os

import benchmark_detection
import content_identity
from pipeline_progress import StageProgress, terminal_progress


class BenchmarkInputError(RuntimeError):
    pass


def selected_identity_faces(case, faces):
    if not case.identity_face_id:
        return faces
    selected = [face for face in faces
                if content_identity.face_identity(face) == case.identity_face_id]
    if len(selected) != 1:
        raise BenchmarkInputError(
            f"Selected benchmark face is missing or ambiguous; "
            f"reverify the face selection: {case.source}")
    return selected


def prepare_selected_faces(cases, cache, *, force_fresh=False, progress=terminal_progress):
    """Refresh only scoped cases whose ordinary cache cannot reproduce the pin.

    A refresh must reproduce the exact user-selected crop fingerprint. Never
    replace it with the largest face, nearest identity, or a guessed index.
    Returned faces are evaluation-only and are never written to the face cache.
    """
    scoped = tuple(case for case in cases if case.identity_face_id)
    if not scoped:
        return {}
    keys = {os.path.realpath(str(case.source)) for case in scoped}
    cached = {key: [] for key in keys}
    for face in cache.faces:
        key = os.path.realpath(face.src_str)
        if key in cached:
            cached[key].append(face)
    refresh = []
    status = StageProgress("Checking saved benchmark face selections", len(scoped), progress)
    for case in status.items(scoped):
        try:
            digest = content_identity.content_sha256(case.source)
        except OSError as error:
            raise BenchmarkInputError(f"Benchmark source unavailable: {case.source}") from error
        if case.content_sha256 and digest != case.content_sha256:
            raise BenchmarkInputError(f"Benchmark content changed since verification: {case.source}")
        faces = cached[os.path.realpath(str(case.source))]
        try:
            selected_identity_faces(case, faces)
            stale = any(getattr(face, "content_sha256", "") not in {"", digest}
                        for face in faces)
            if force_fresh or stale:
                refresh.append(case)
        except BenchmarkInputError:
            refresh.append(case)
    if not refresh:
        return {}
    progress(f"Refreshing {len(refresh)} selected benchmark image(s); saved selections remain unchanged...")
    try:
        result = benchmark_detection.detect_cases(tuple(refresh))
    except (RuntimeError, OSError, ValueError) as error:
        raise BenchmarkInputError(f"Selected-face benchmark detection failed: {error}") from error
    for case in refresh:
        key = os.path.realpath(str(case.source))
        faces = result.get(key, [])
        digest = content_identity.content_sha256(case.source)
        if case.content_sha256 and digest != case.content_sha256:
            raise BenchmarkInputError(f"Benchmark content changed during verification: {case.source}")
        if any(os.path.realpath(face.src_str) != key
               or getattr(face, "content_sha256", "") != digest for face in faces):
            raise BenchmarkInputError(f"Invalid refreshed benchmark face provenance: {case.source}")
        selected_identity_faces(case, faces)
    progress(f"Verified {len(refresh)} exact saved face selection(s); no labels or originals changed.")
    return result

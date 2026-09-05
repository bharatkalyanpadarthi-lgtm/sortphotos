#!/usr/bin/env python3
"""
Fast synthetic integration tests for the photo pipeline.

These tests create a tiny temporary photo library and redirect cache files into
that temp folder. They are meant to catch cross-script regressions before a real
scan touches ~/Pictures.
"""

from __future__ import annotations

import contextlib
import csv
import io
import json
import os
import pickle
import queue
import shutil
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import urllib.parse
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import advanced_duplicate_matching  # noqa: E402
import analysis_index  # noqa: E402
import appearance_profiles  # noqa: E402
import asset_processing  # noqa: E402
import audit_cross_person_identity  # noqa: E402
import audit_nude_folders  # noqa: E402
import build_smart_albums  # noqa: E402
import cache_tools  # noqa: E402
import cleanup_empty_person_folders  # noqa: E402
import daily_runner  # noqa: E402
import delete_person_folder_duplicates  # noqa: E402
import evaluation_dataset  # noqa: E402
import evaluation_enrollment  # noqa: E402
import face  # noqa: E402
import generated_artifacts  # noqa: E402
import image_batch_runner  # noqa: E402
import identity_confirmations  # noqa: E402
import identity_hard_negatives  # noqa: E402
import identity_profiles  # noqa: E402
import identity_evaluation  # noqa: E402
import operation_ledger  # noqa: E402
import nudity_confirmations  # noqa: E402
import near_visual_review  # noqa: E402
import person_structure  # noqa: E402
import rename_person_folder_files  # noqa: E402
import relink_cache_from_old_cache  # noqa: E402
import recover_no_usable_faces  # noqa: E402
import review_unknown_identities  # noqa: E402
import review_identity_benchmark  # noqa: E402
import secondary_identity_matcher  # noqa: E402
import source_batch_consensus  # noqa: E402
import source_guard  # noqa: E402
import source_manifest  # noqa: E402
import separate_nudity_review  # noqa: E402
import place_nudity_inside_person_folders  # noqa: E402
import preflight_check  # noqa: E402
import quarantine_bad_person_images  # noqa: E402
import sort_photos  # noqa: E402
import sort_videos  # noqa: E402
import video_batch_runner  # noqa: E402

for _name in ("CacheState", "CachedFace", "FaceRecord", "LabelingState", "IdentityDB"):
    if hasattr(sort_photos, _name):
        setattr(sys.modules["__main__"], _name, getattr(sort_photos, _name))


@dataclass
class Result:
    ok: bool
    name: str
    detail: str


class SyntheticFailure(AssertionError):
    pass


def make_image(path: Path, color: tuple[int, int, int] = (90, 120, 180),
               size: tuple[int, int] = (32, 32), fmt: str | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, format=fmt)


def write_bad_image(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"this is not a readable image")


def make_validation_contact_sheet(path: Path) -> None:
    from PIL import ImageDraw

    image = Image.new("RGB", (974, 332), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), "05_same_scene/high_confidence page 1 images 1-2 of 2", fill=(20, 20, 20))
    for column, color in enumerate(((80, 120, 170), (150, 90, 120))):
        x = 12 + column * 190
        y = 54
        draw.rectangle((x, y, x + 189, y + 219), fill=(245, 245, 245))
        draw.rectangle((x + 20, y + 20, x + 169, y + 199), fill=color)
        draw.rectangle((x + 4, y + 4, x + 42, y + 26), fill="white", outline="black")
        draw.text((x + 8, y + 8), str(column + 1), fill="black")
        draw.text((x + 4, y + 224), f"{column + 1}: sample_photo", fill=(20, 20, 20))
    image.save(path, quality=90)


def make_legacy_contact_sheet(path: Path) -> None:
    from PIL import ImageDraw

    image = Image.new("RGB", (384, 280), "white")
    draw = ImageDraw.Draw(image)
    draw.text((12, 12), "00_START_HERE", fill=(20, 20, 20))
    for column, color in enumerate(((80, 120, 170), (150, 90, 120))):
        x = 12 + column * 180
        y = 54
        draw.rectangle((x, y, x + 179, y + 179), fill=(245, 245, 245))
        draw.rectangle((x + 20, y + 10, x + 159, y + 169), fill=color)
        draw.text((x + 4, y + 183), f"sample_{column + 1}", fill=(20, 20, 20))
    image.save(path, quality=88)


def assert_true(condition: bool, message: str) -> None:
    if not condition:
        raise SyntheticFailure(message)


@contextlib.contextmanager
def redirected_cache(cache_path: Path):
    old_sort_cache = sort_photos.CACHE_FILE
    old_sort_dir = sort_photos.CACHE_DIR
    old_cache_tools_cache = cache_tools.sort_photos.CACHE_FILE
    old_cache_tools_dir = cache_tools.sort_photos.CACHE_DIR
    sort_photos.CACHE_DIR = cache_path.parent
    sort_photos.CACHE_FILE = cache_path
    cache_tools.sort_photos.CACHE_DIR = cache_path.parent
    cache_tools.sort_photos.CACHE_FILE = cache_path
    try:
        yield
    finally:
        sort_photos.CACHE_FILE = old_sort_cache
        sort_photos.CACHE_DIR = old_sort_dir
        cache_tools.sort_photos.CACHE_FILE = old_cache_tools_cache
        cache_tools.sort_photos.CACHE_DIR = old_cache_tools_dir


@contextlib.contextmanager
def quiet_output():
    out = io.StringIO()
    err = io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        yield out, err


@contextlib.contextmanager
def redirected_label_state(label_state_path: Path):
    old_label_state = sort_photos.LABEL_STATE_FILE
    sort_photos.LABEL_STATE_FILE = label_state_path
    try:
        yield
    finally:
        sort_photos.LABEL_STATE_FILE = old_label_state


def dummy_cached_face(src: Path, label: str | None = None) -> sort_photos.CachedFace:
    return sort_photos.CachedFace(
        src_str=str(src),
        face_index=0,
        det_score=0.99,
        bbox_size=100.0,
        sharpness=100.0,
        yaw_proxy=0.0,
        quality=1.0,
        embedding=np.ones(512, dtype=np.float32),
        image_phash=np.zeros(64, dtype=bool),
        crop_jpeg=b"\xff\xd8\xff\xd9",
        label=label,
    )


def test_to_process_generated_names_are_visible(tmp: Path) -> None:
    inbox = tmp / "To Process"
    make_image(inbox / "all" / "a.jpg", (255, 0, 0))
    make_image(inbox / "review" / "b.jpg", (0, 255, 0))
    make_image(inbox / "_smart_albums" / "c.jpg", (0, 0, 255))
    make_image(inbox / "normal" / "d.jpg", (255, 255, 0))
    make_image(inbox / "normal" / "animated_input.gif", (120, 30, 180), fmt="GIF")

    visible = list(sort_photos.iter_images(
        inbox,
        excluded_dir_names=set(),
        always_excluded_dir_names=set(),
    ))
    assert_true(len(visible) == 5, f"expected 5 visible inbox images, got {len(visible)}")

    count = daily_runner.count_images(inbox, exclude_generated_dirs=False)
    assert_true(count == 5, f"daily preview should see 5 images, got {count}")

    gif = inbox / "normal" / "animated_input.gif"
    assert_true(sort_photos.imread_unicode(gif) is not None, "GIF intake image did not decode")


def test_unassigned_intake_is_preserved_by_reason(tmp: Path) -> None:
    inbox = tmp / "To Process"
    output = tmp / "sorted_all_pictures"
    no_face = inbox / "batch" / "no_face.jpg"
    unknown = inbox / "batch" / "unknown.jpg"
    copy_failed = inbox / "batch" / "copy_failed.jpg"
    processing_failed = inbox / "batch" / "processing_failed.jpg"
    unreadable = inbox / "batch" / "unreadable.jpg"
    quality_review = inbox / "batch" / "quality_review.jpg"
    organized = inbox / "batch" / "organized.jpg"
    for path in (no_face, unknown, copy_failed, processing_failed, quality_review, organized):
        make_image(path)
    write_bad_image(unreadable)

    def record(path: Path, cluster_id: int) -> sort_photos.FaceRecord:
        return sort_photos.FaceRecord(
            src=path,
            face_index=0,
            det_score=0.99,
            bbox_size=100.0,
            sharpness=100.0,
            yaw_proxy=0.0,
            embedding=np.ones(512, dtype=np.float32),
            cluster_id=cluster_id,
        )

    records = [record(unknown, -1), record(copy_failed, 1), record(organized, 2)]
    counts, report = sort_photos.archive_unassigned_sources(
        [no_face, unknown, copy_failed, processing_failed, unreadable, quality_review, organized],
        records,
        {-1: "unknown", 1: "Soundarya", 2: "Soundarya"},
        {organized},
        inbox,
        output,
        processed_sources={no_face, unknown, copy_failed, unreadable, quality_review, organized},
        detection_outcomes={str(quality_review.resolve()): "face_quality_review:face_too_small"},
    )

    review = output / "_source_review" / "unassigned_intake"
    assert_true(counts.get("no_usable_face") == 1, f"wrong no-face count: {counts}")
    assert_true(counts.get("unknown_identity") == 1, f"wrong unknown count: {counts}")
    assert_true(counts.get("copy_failed") == 1, f"wrong copy-failed count: {counts}")
    assert_true(counts.get("processing_failed") == 1, f"wrong processing-failed count: {counts}")
    assert_true(counts.get("unreadable_image") == 1, f"wrong unreadable count: {counts}")
    assert_true(counts.get("face_quality_review") == 1, f"wrong quality-review count: {counts}")
    assert_true((review / "no_usable_face" / "batch" / no_face.name).exists(),
                "no-face image was not preserved")
    assert_true((review / "unknown_identity" / "batch" / unknown.name).exists(),
                "unknown image was not preserved")
    assert_true((review / "copy_failed" / "batch" / copy_failed.name).exists(),
                "copy-failed image was not preserved")
    assert_true((review / "processing_failed" / "batch" / processing_failed.name).exists(),
                "unprocessed image was incorrectly classified as no-face")
    assert_true((review / "unreadable_image" / "batch" / unreadable.name).exists(),
                "unreadable image was incorrectly classified as no-face")
    assert_true((review / "face_quality_review" / "batch" / quality_review.name).exists(),
                "detected low-quality face was incorrectly classified as no-face")
    assert_true(organized.exists(), "organized input was incorrectly moved to review")
    assert_true(report is not None and report.exists(), "unassigned audit CSV was not written")
    assert_true(not (output / "_source_review" / "ready_to_delete" / "scanned_sources").exists(),
                "unassigned images were silently sent to scanned_sources")


def test_soft_face_uses_recovery_detection_tier(tmp: Path) -> None:
    image = tmp / "soft.jpg"
    make_image(image, size=(220, 220))
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    face = SimpleNamespace(
        det_score=0.90,
        bbox=np.asarray([40, 40, 180, 180], dtype=np.float32),
        normed_embedding=embedding,
        kps=None,
    )

    class FakeApp:
        def get(self, _image):
            return [face]

    old_sharpness = sort_photos.sharpness
    sort_photos.sharpness = lambda _crop: 10.0
    try:
        diagnostics: dict[str, str] = {}
        faces = sort_photos._detect_one_image(image, FakeApp(), diagnostics=diagnostics)
    finally:
        sort_photos.sharpness = old_sharpness

    assert_true(len(faces) == 1, "soft visible face was not recovered")
    assert_true(diagnostics.get(str(image)) == "accepted_face_recovery",
                f"wrong soft-face diagnostic: {diagnostics}")


def test_rotated_face_uses_alternate_view(tmp: Path) -> None:
    image = tmp / "rotated.jpg"
    make_image(image, size=(220, 300))
    embedding = np.zeros(512, dtype=np.float32)
    embedding[1] = 1.0
    face = SimpleNamespace(
        det_score=0.91,
        bbox=np.asarray([30, 50, 170, 190], dtype=np.float32),
        normed_embedding=embedding,
        kps=None,
    )

    class RotationOnlyApp:
        def __init__(self):
            self.calls = 0

        def get(self, _image):
            self.calls += 1
            return [] if self.calls == 1 else [face]

    old_sharpness = sort_photos.sharpness
    sort_photos.sharpness = lambda _crop: 12.0
    app = RotationOnlyApp()
    try:
        diagnostics: dict[str, str] = {}
        faces = sort_photos._detect_one_image(image, app, diagnostics=diagnostics)
    finally:
        sort_photos.sharpness = old_sharpness

    assert_true(len(faces) == 1 and app.calls >= 2,
                "rotated face was not recovered from an alternate view")
    assert_true(diagnostics.get(str(image)) == "accepted_face_recovery",
                f"wrong rotated-face diagnostic: {diagnostics}")


def test_tight_face_uses_padded_low_threshold_view(tmp: Path) -> None:
    image = tmp / "tight.jpg"
    make_image(image, size=(700, 900))
    embedding = np.zeros(512, dtype=np.float32)
    embedding[2] = 1.0
    face = SimpleNamespace(
        det_score=0.92,
        bbox=np.asarray([180, 220, 520, 620], dtype=np.float32),
        normed_embedding=embedding,
        kps=None,
    )

    class PaddingOnlyApp:
        def __init__(self):
            self.det_model = SimpleNamespace(det_thresh=0.44)
            self.calls = 0

        def get(self, candidate):
            self.calls += 1
            padded = candidate.shape[0] > 900 and candidate.shape[1] > 700
            low_threshold = self.det_model.det_thresh <= sort_photos.FACE_PRESENCE_MIN_DET_SCORE
            return [face] if padded and low_threshold else []

    old_sharpness = sort_photos.sharpness
    sort_photos.sharpness = lambda _crop: 12.0
    app = PaddingOnlyApp()
    try:
        diagnostics: dict[str, str] = {}
        faces = sort_photos._detect_one_image(image, app, diagnostics=diagnostics)
    finally:
        sort_photos.sharpness = old_sharpness

    assert_true(len(faces) == 1 and app.calls >= 2,
                "tightly cropped face was not recovered from padded context")
    assert_true(app.det_model.det_thresh == 0.44,
                "fallback detection threshold was not restored")
    assert_true(diagnostics.get(str(image)) == "accepted_face_recovery",
                f"wrong padded-face diagnostic: {diagnostics}")


def test_low_confidence_face_routes_to_quality_review(tmp: Path) -> None:
    image = tmp / "uncertain.jpg"
    make_image(image, size=(700, 900))
    embedding = np.zeros(512, dtype=np.float32)
    embedding[3] = 1.0
    face = SimpleNamespace(
        det_score=0.25,
        bbox=np.asarray([180, 220, 520, 620], dtype=np.float32),
        normed_embedding=embedding,
        kps=None,
    )

    class LowConfidenceApp:
        def __init__(self):
            self.det_model = SimpleNamespace(det_thresh=0.44)

        def get(self, candidate):
            padded = candidate.shape[0] > 900 and candidate.shape[1] > 700
            return [face] if padded and self.det_model.det_thresh <= 0.20 else []

    diagnostics: dict[str, str] = {}
    faces = sort_photos._detect_one_image(image, LowConfidenceApp(), diagnostics=diagnostics)
    assert_true(not faces, "low-confidence face was incorrectly accepted for matching")
    assert_true(
        diagnostics.get(str(image), "").startswith("face_quality_review:"),
        f"low-confidence face was mislabeled as no-face: {diagnostics}",
    )


def test_recovery_identity_match_is_strict(tmp: Path) -> None:
    del tmp
    alice = np.zeros(512, dtype=np.float32)
    bob = np.zeros(512, dtype=np.float32)
    alice[0] = 1.0
    bob[1] = 1.0
    db = sort_photos.IdentityDB(
        identities={"Alice": alice, "Bob": bob},
        source_counts={"Alice": 10, "Bob": 10},
    )
    names = ["Alice", "Bob"]
    matrix = np.stack([alice, bob])
    matched = recover_no_usable_faces.match_face(alice, names, matrix, db)
    assert_true(matched is not None and matched[0] == "Alice",
                f"exact recovery identity was not accepted: {matched}")
    ambiguous = (alice + bob) / np.linalg.norm(alice + bob)
    held = recover_no_usable_faces.match_face(ambiguous, names, matrix, db)
    assert_true(held is None, f"ambiguous recovery identity was accepted: {held}")

    pose = np.zeros(512, dtype=np.float32)
    pose[2] = 1.0
    profile_db = sort_photos.IdentityDB(
        identities={"Alice": alice, "Bob": bob},
        prototypes={"Alice": [pose], "Bob": [bob]},
        source_counts={"Alice": 10, "Bob": 10},
        match_thresholds={"Alice": 0.32, "Bob": 0.32},
    )
    prototype_match = recover_no_usable_faces.match_face(
        pose, names, matrix, profile_db, quality=0.9,
    )
    assert_true(
        prototype_match is not None and prototype_match[0] == "Alice",
        f"recovery ignored the person's alternate appearance prototype: {prototype_match}",
    )

    query = np.zeros(512, dtype=np.float32)
    query[0] = 0.85
    query[3] = np.sqrt(1.0 - 0.85**2)
    calibrated_db = sort_photos.IdentityDB(
        identities={"Alice": alice, "Bob": bob},
        prototypes={"Alice": [alice], "Bob": [bob]},
        source_counts={"Alice": 10, "Bob": 10},
        match_thresholds={"Alice": 0.10, "Bob": 0.32},
    )
    calibrated_hold = recover_no_usable_faces.match_face(
        query, names, matrix, calibrated_db, quality=0.9,
    )
    assert_true(calibrated_hold is None, "recovery bypassed a person's calibrated threshold")


def test_pose_profiles_and_hard_negatives_are_shared(tmp: Path) -> None:
    frontal = np.zeros(512, dtype=np.float32)
    profile = np.zeros(512, dtype=np.float32)
    rival = np.zeros(512, dtype=np.float32)
    frontal[0], profile[1], rival[2] = 1.0, 1.0, 1.0
    ranked = identity_profiles.rank_candidates(
        profile,
        {"Alice": frontal, "Bob": rival},
        {"Alice": [frontal], "Bob": [rival]},
        pose_label="left_profile",
        pose_prototypes={"Alice": {"left_profile": [profile]}},
    )
    assert_true(ranked[0].name == "Alice", "pose-specific prototype was not used")

    negatives = tmp / "negatives.json"
    identity_hard_negatives.record(
        negatives,
        person="Alice",
        source_path=tmp / "lookalike.jpg",
        face_index=0,
        embedding=profile,
    )
    guarded = identity_profiles.rank_candidates(
        profile,
        {"Alice": frontal, "Bob": rival},
        {"Alice": [profile], "Bob": [rival]},
        hard_negatives=identity_hard_negatives.vectors_by_person(negatives),
    )
    alice = next(candidate for candidate in guarded if candidate.name == "Alice")
    assert_true(
        alice.distance > float(alice.raw_distance or 0.0) + 0.3,
        "explicit negative evidence did not guard the rejected person",
    )

    bbox = np.asarray([0, 0, 100, 100], dtype=np.float32)
    left = [[25, 35], [75, 35], [42, 55]]
    right = [[25, 35], [75, 35], [58, 55]]
    assert_true(sort_photos.face_detection.pose_label_from_keypoints(left, bbox) == "left_profile",
                "left-profile pose was not classified")
    assert_true(sort_photos.face_detection.pose_label_from_keypoints(right, bbox) == "right_profile",
                "right-profile pose was not classified")


def test_appearance_profiles_improve_without_widening_thresholds(tmp: Path) -> None:
    del tmp
    frontal = np.zeros(512, dtype=np.float32)
    low_light = np.zeros(512, dtype=np.float32)
    rival = np.zeros(512, dtype=np.float32)
    frontal[0], low_light[1], rival[2] = 1.0, 1.0, 1.0
    ranked = identity_profiles.rank_candidates(
        low_light,
        {"Alice": frontal, "Bob": rival},
        {"Alice": [frontal], "Bob": [rival]},
        lighting_label="low_light",
        appearance_prototypes={"Alice": {"low_light": [low_light]}},
    )
    assert_true(ranked[0].name == "Alice", "low-light prototype was not used")
    labels = appearance_profiles.labels(
        light="low_light", timestamp=100.0, person_era_cutoff=200.0
    )
    assert_true(labels == ("low_light", "era_older"),
                f"appearance labels are incorrect: {labels}")


def test_source_batch_consensus_requires_face_and_batch_agreement(tmp: Path) -> None:
    del tmp
    def vector(x: float, y: float) -> np.ndarray:
        value = np.zeros(512, dtype=np.float32)
        value[:2] = [x, y]
        return value / max(float(np.linalg.norm(value)), 1e-9)

    evidence = [
        source_batch_consensus.BatchEvidence(
            source="a.jpg", face_index=0, batch_key="folder:batch-a",
            embedding=vector(1, 0), candidate_name="Alice", distance=0.20,
            margin=0.12, quality=0.8, threshold=0.28,
        ),
        source_batch_consensus.BatchEvidence(
            source="b.jpg", face_index=0, batch_key="folder:batch-a",
            embedding=vector(1, .03), candidate_name="Alice", distance=0.31,
            margin=0.06, quality=0.5, threshold=0.28,
        ),
        source_batch_consensus.BatchEvidence(
            source="c.jpg", face_index=0, batch_key="folder:batch-a",
            embedding=vector(0, 1), candidate_name="Bob", distance=0.22,
            margin=0.11, quality=0.8, threshold=0.28,
        ),
    ]
    decisions = source_batch_consensus.consensus_decisions(evidence)
    assert_true(("a.jpg", 0) in decisions and ("b.jpg", 0) in decisions,
                "same-face same-batch evidence did not form consensus")
    assert_true(("c.jpg", 0) not in decisions,
                "a different face was incorrectly absorbed into batch consensus")
    unbatched = [replace.batch_key for replace in evidence]
    assert_true(all(unbatched), "test fixture lost its batch keys")


def test_daily_individual_recovery_regressions(tmp: Path) -> None:
    del tmp
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "test_daily_identity_recovery", "-v"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True, text=True, timeout=60,
    )
    assert_true(result.returncode == 0, result.stdout + result.stderr)


def test_unknown_decision_persistence_regressions(tmp: Path) -> None:
    del tmp
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "test_unknown_decision_persistence", "-v"],
        cwd=Path(__file__).resolve().parent,
        capture_output=True, text=True, timeout=60,
    )
    assert_true(result.returncode == 0, result.stdout + result.stderr)


def test_architecture_hardening_regressions(tmp: Path) -> None:
    del tmp
    result = subprocess.run(
        [sys.executable, "-m", "unittest", "test_architecture_hardening", "-v"],
        cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=120)
    assert_true(result.returncode == 0, result.stdout + result.stderr)


def test_daily_identity_assignment_uses_source_batch_consensus(tmp: Path) -> None:
    def vector(x: float, y: float) -> np.ndarray:
        value = np.zeros(512, dtype=np.float32)
        value[:2] = [x, y]
        return value / max(float(np.linalg.norm(value)), 1e-9)

    alice, bob = vector(1, 0), vector(0, 1)
    weak_a, weak_b = vector(0.75, 0.0), vector(0.74, 0.01)
    # Add a shared orthogonal component so each match is just outside the
    # person's normal threshold but still within the bounded batch lane.
    weak_a[2], weak_b[2] = 0.66, 0.67
    weak_a /= np.linalg.norm(weak_a)
    weak_b /= np.linalg.norm(weak_b)
    db = sort_photos.normalize_identity_db(sort_photos.IdentityDB(
        identities={"Alice": alice, "Bob": bob},
        prototypes={"Alice": [alice], "Bob": [bob]},
        source_counts={"Alice": 6, "Bob": 6},
        match_thresholds={"Alice": 0.20, "Bob": 0.20},
        strict_thresholds={"Alice": 0.15, "Bob": 0.15},
    ))
    batch = tmp / "intake" / "camera-batch"
    first, second = batch / "one.jpg", batch / "two.jpg"
    make_image(first)
    make_image(second, color=(100, 130, 190))
    records = [
        sort_photos.FaceRecord(
            src=first, face_index=0, det_score=.99, bbox_size=120,
            sharpness=100, yaw_proxy=0, embedding=weak_a, quality=.7,
            cluster_id=10,
        ),
        sort_photos.FaceRecord(
            src=second, face_index=0, det_score=.99, bbox_size=120,
            sharpness=100, yaw_proxy=0, embedding=weak_b, quality=.7,
            cluster_id=11,
        ),
    ]
    names = {10: "person_010", 11: "person_011"}
    assigned = sort_photos.apply_identity_db_labels(
        records, names, db, source_batch_root=tmp / "intake"
    )
    assert_true(assigned == 2 and set(names.values()) == {"Alice"},
                f"Daily Run did not apply safe batch consensus: {assigned}, {names}")


def test_video_matcher_uses_pose_and_hard_negative_policy(tmp: Path) -> None:
    frontal = np.zeros(512, dtype=np.float32)
    profile = np.zeros(512, dtype=np.float32)
    rival = np.zeros(512, dtype=np.float32)
    frontal[0], profile[1], rival[2] = 1.0, 1.0, 1.0
    db = sort_photos.normalize_identity_db(sort_photos.IdentityDB(
        identities={"Alice": frontal, "Bob": rival},
        prototypes={"Alice": [frontal], "Bob": [rival]},
        pose_prototypes={"Alice": {"left_profile": [profile]}},
        source_counts={"Alice": 6, "Bob": 6},
        match_thresholds={"Alice": .34, "Bob": .34},
        strict_thresholds={"Alice": .34, "Bob": .34},
    ))
    frame = np.full((180, 180, 3), 120, dtype=np.uint8)
    face_record = SimpleNamespace(
        normed_embedding=profile,
        bbox=np.asarray([35, 30, 145, 155], dtype=np.float32),
        kps=np.asarray([[55, 70], [120, 70], [75, 100], [65, 125], [115, 125]], dtype=np.float32),
        det_score=.99,
    )
    matcher = sort_videos.VideoIdentityMatcher(db, {})
    name, tier, _distance, _margin = matcher.match(frame, face_record, tmp / "clip.mov")
    assert_true(name == "Alice" and tier == "strict",
                f"video matcher ignored pose profile: {name}, {tier}")
    guarded = sort_videos.VideoIdentityMatcher(db, {"Alice": [profile]})
    held, held_tier, _distance, _margin = guarded.match(
        frame, face_record, tmp / "clip.mov"
    )
    assert_true(held is None and held_tier == "none",
                "video matcher bypassed explicit hard-negative evidence")


def test_protected_benchmark_rows_are_atomic_and_unverified_by_default(tmp: Path) -> None:
    source = tmp / "candidate.jpg"
    make_image(source)
    dataset = tmp / "protected.csv"
    review_identity_benchmark.upsert_row(dataset, {
        "source": str(source), "expected_person": "Alice", "case_types": "known",
        "expected_face": "true", "expected_nudity": "unknown",
        "verified": "false", "notes": "review me",
    })
    rows = review_identity_benchmark.read_rows(dataset)
    validation = evaluation_dataset.load_dataset(dataset)
    assert_true(len(rows) == 1 and rows[0]["verified"] == "false",
                "benchmark candidate was not persisted safely")
    assert_true(validation.errors and not validation.activation_ready,
                "unverified benchmark was incorrectly eligible for activation")


def test_protected_benchmark_bulk_confirmation_is_atomic(tmp: Path) -> None:
    first = tmp / "first.jpg"
    second = tmp / "second.jpg"
    make_image(first)
    make_image(second, (140, 80, 120))
    dataset = tmp / "protected.csv"
    review_identity_benchmark.upsert_row(dataset, {
        "source": str(first), "expected_person": "Alice", "case_types": "known",
        "expected_face": "true", "expected_nudity": "safe",
        "verified": "false", "notes": "first",
    })
    review_identity_benchmark.upsert_row(dataset, {
        "source": str(second), "expected_person": "", "case_types": "known",
        "expected_face": "true", "expected_nudity": "safe",
        "verified": "false", "notes": "second",
    })
    try:
        review_identity_benchmark.bulk_update_rows(
            dataset, [str(first), str(second)], {}, verified=True
        )
    except ValueError:
        pass
    else:
        raise AssertionError("invalid multi-image benchmark confirmation was accepted")
    rows = review_identity_benchmark.read_rows(dataset)
    assert_true(all(row["verified"] == "false" for row in rows),
                "failed benchmark batch partially changed the dataset")

    changed = review_identity_benchmark.bulk_update_rows(
        dataset,
        [str(first), str(second)],
        {"expected_person": "Alice", "case_types": "known|side_profile"},
        verified=True,
    )
    rows = review_identity_benchmark.read_rows(dataset)
    assert_true(len(changed) == 2 and all(row["verified"] == "true" for row in rows),
                "valid multi-image benchmark confirmation was not saved")
    assert_true(all(row["case_types"] == "known|side_profile" for row in rows),
                "bulk benchmark labels were not normalized consistently")


def test_protected_benchmark_dashboard_supports_multi_image_review(tmp: Path) -> None:
    first = tmp / "first.jpg"
    second = tmp / "second.jpg"
    make_image(first, size=(120, 90))
    make_image(second, (140, 80, 120), size=(90, 120))
    dataset = tmp / "protected.csv"
    for source in (first, second):
        review_identity_benchmark.upsert_row(dataset, {
            "source": str(source), "expected_person": "Alice", "case_types": "known",
            "expected_face": "true", "expected_nudity": "safe",
            "verified": "false", "notes": "dashboard test",
        })

    server = review_identity_benchmark.BenchmarkServer(
        ("127.0.0.1", 0), review_identity_benchmark.Handler
    )
    server.dataset = dataset
    server.baseline = tmp / "baseline.json"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    root = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        with urllib.request.urlopen(root + "/", timeout=3) as response:
            page = response.read().decode("utf-8")
        assert_true(
            all(token in page for token in (
                'id="selectVisible"', 'id="verifySelected"',
                'id="applySelected"', 'data-density="compact"',
                'class="row-select"',
            )),
            "protected benchmark dashboard is missing multi-image review controls",
        )
        thumbnail_url = root + "/thumbnail?" + urllib.parse.urlencode({"path": str(first)})
        with urllib.request.urlopen(thumbnail_url, timeout=3) as response:
            thumbnail = response.read()
            content_type = response.headers.get_content_type()
        assert_true(content_type == "image/jpeg" and len(thumbnail) > 100,
                    "benchmark dashboard did not serve a review thumbnail")

        body = urllib.parse.urlencode([
            ("source", str(first)), ("source", str(second)), ("verified", "true"),
        ]).encode("utf-8")
        request = urllib.request.Request(root + "/bulk-save", data=body, method="POST")
        with urllib.request.urlopen(request, timeout=3) as response:
            result = json.loads(response.read())
        assert_true(result.get("ok") is True and result.get("count") == 2,
                    "dashboard multi-image confirmation endpoint failed")
        assert_true(all(row["verified"] == "true"
                        for row in review_identity_benchmark.read_rows(dataset)),
                    "dashboard batch confirmation did not persist every selected image")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)


def test_secondary_matcher_requires_independent_agreement(tmp: Path) -> None:
    alice = np.zeros(512, dtype=np.float32)
    bob = np.zeros(512, dtype=np.float32)
    alice[0], bob[1] = 1.0, 1.0
    crop = b"synthetic-crop"
    db = secondary_identity_matcher.SecondaryIdentityDB(
        identities={"Alice": alice, "Bob": bob},
        prototypes={"Alice": [alice], "Bob": [bob]},
        thresholds={"Alice": 0.30, "Bob": 0.30},
        crop_embeddings={secondary_identity_matcher.crop_key(crop): alice},
    )
    matcher = secondary_identity_matcher.SecondaryMatcher(
        db, cache_path=tmp / "secondary.pkl"
    )
    accepted = matcher.verify(crop, "Alice")
    rejected = matcher.verify(crop, "Bob")
    assert_true(accepted.accepted, "independent verifier rejected an exact agreement")
    assert_true(not rejected.accepted, "independent verifier accepted a disagreement")


def test_secondary_matcher_resolves_cache_prototype_ids(tmp: Path) -> None:
    first = np.zeros(512, dtype=np.float32)
    second = np.zeros(512, dtype=np.float32)
    first[0] = 1.0
    second[0], second[1] = 0.99, 0.1
    second = identity_profiles.normalize_vector(second)
    primary = sort_photos.IdentityDB(
        config_fingerprint="synthetic",
        identities={"Alice": first},
        prototypes={"Alice": [first, second]},
        prototype_sources={"Alice": ["cache:Alice:0", "cache:Alice:1"]},
        source_counts={"Alice": 2},
    )
    faces = [
        sort_photos.CachedFace(
            src_str=str(tmp / f"Alice_{index}.jpg"), face_index=0,
            det_score=0.99, bbox_size=120.0, sharpness=100.0,
            yaw_proxy=float(index), quality=0.9, embedding=embedding,
            image_phash=np.zeros(64, dtype=np.uint8), crop_jpeg=f"crop-{index}".encode(),
            label="Alice",
        )
        for index, embedding in enumerate((first, second))
    ]
    secondary_vectors = {
        b"crop-0": first,
        b"crop-1": second,
    }
    original = (
        secondary_identity_matcher.build_app,
        secondary_identity_matcher.embed_crop,
        secondary_identity_matcher.load,
        secondary_identity_matcher.save,
    )
    saved = []
    secondary_identity_matcher.build_app = lambda: object()
    secondary_identity_matcher.embed_crop = lambda crop, _app: secondary_vectors[crop]
    secondary_identity_matcher.load = lambda: None
    secondary_identity_matcher.save = lambda db: saved.append(db)
    try:
        built = secondary_identity_matcher.build_database(
            primary, sort_photos.CacheState(faces=faces), maximum_per_person=2
        )
    finally:
        (
            secondary_identity_matcher.build_app,
            secondary_identity_matcher.embed_crop,
            secondary_identity_matcher.load,
            secondary_identity_matcher.save,
        ) = original
    assert_true(
        "Alice" in built.identities
        and built.source_counts["Alice"] == 2
        and len(built.prototypes["Alice"]) == 2
        and saved,
        "secondary verifier did not resolve stable cache prototype IDs",
    )
    changed = sort_photos.IdentityDB(
        config_fingerprint="synthetic",
        identities={"Alice": second},
        prototypes={"Alice": [second, first]},
        prototype_sources={"Alice": ["cache:Alice:0", "cache:Alice:1"]},
        source_counts={"Alice": 2},
    )
    assert_true(
        secondary_identity_matcher.primary_signature(primary)
        != secondary_identity_matcher.primary_signature(changed),
        "secondary verifier signature ignored changed prototype embeddings",
    )
    before_cache_growth = secondary_identity_matcher.identity_signature(built)
    built.crop_embeddings["new-review-crop"] = first
    assert_true(
        secondary_identity_matcher.identity_signature(built) == before_cache_growth,
        "secondary identity signature incorrectly depends on the growing review cache",
    )


def test_confirmations_enroll_and_gate_identity_profiles(tmp: Path) -> None:
    image = tmp / "Alice" / "photos" / "Alice_00001.jpg"
    make_image(image)
    dataset = tmp / "confirmed.csv"
    evaluation_enrollment.enroll(
        source=image,
        person="Alice",
        content_sha256=sort_photos.sha256_file(image),
        pose_label="right_profile",
        quality=0.3,
        path=dataset,
    )
    validation = evaluation_dataset.load_dataset(dataset)
    assert_true(not validation.errors and {"known", "side_profile", "blurry_small"}.issubset(
        validation.covered_types
    ), "explicit confirmation was not enrolled as verified evaluation evidence")

    alice = np.zeros(512, dtype=np.float32)
    bob = np.zeros(512, dtype=np.float32)
    alice[0], bob[1] = 1.0, 1.0
    face_record = sort_photos.CachedFace(
        src_str=str(image), face_index=0, det_score=.99, bbox_size=120,
        sharpness=100, yaw_proxy=1, quality=.9, embedding=alice,
        image_phash=np.zeros(64, dtype=bool), crop_jpeg=b"jpeg", label="Alice",
        pose_label="frontal",
    )
    incumbent = sort_photos.normalize_identity_db(sort_photos.IdentityDB(
        identities={"Alice": alice, "Bob": bob},
        prototypes={"Alice": [alice], "Bob": [bob]},
        prototype_sources={"Alice": [str(tmp / "alice-reference.jpg")], "Bob": [str(tmp / "bob-reference.jpg")]},
        source_counts={"Alice": 5, "Bob": 5},
    ))
    bad = sort_photos.normalize_identity_db(sort_photos.IdentityDB(
        identities={"Alice": bob, "Bob": alice},
        prototypes={"Alice": [bob], "Bob": [alice]},
        prototype_sources={"Alice": [str(tmp / "alice-reference.jpg")], "Bob": [str(tmp / "bob-reference.jpg")]},
        source_counts={"Alice": 5, "Bob": 5},
    ))
    allowed, report = identity_evaluation.activation_gate(
        bad, incumbent, sort_photos.CacheState(faces=[face_record])
    )
    assert_true(not allowed and report["failures"],
                "activation gate promoted a matcher with a new false accept")


def test_confirmation_gate_allows_unchanged_safe_rejections(tmp: Path) -> None:
    image = tmp / "Alice" / "photos" / "Alice_00001.jpg"
    make_image(image)
    dataset = tmp / "confirmed.csv"
    evaluation_enrollment.enroll(
        source=image,
        person="Alice",
        content_sha256=sort_photos.sha256_file(image),
        path=dataset,
    )
    alice = np.zeros(512, dtype=np.float32)
    bob = np.zeros(512, dtype=np.float32)
    alice[0], bob[1] = 1.0, 1.0
    ambiguous = identity_profiles.normalize_vector(alice + bob)
    face = sort_photos.CachedFace(
        src_str=str(image), face_index=0, det_score=.99, bbox_size=120,
        sharpness=100, yaw_proxy=0, quality=.9, embedding=ambiguous,
        image_phash=np.zeros(64, dtype=bool), crop_jpeg=b"jpeg", label="Alice",
    )
    correct = sort_photos.CachedFace(
        src_str=str(tmp / "Bob.jpg"), face_index=0, det_score=.99, bbox_size=120,
        sharpness=100, yaw_proxy=0, quality=.9, embedding=bob,
        image_phash=np.zeros(64, dtype=bool), crop_jpeg=b"bob", label="Bob",
    )
    db = sort_photos.normalize_identity_db(sort_photos.IdentityDB(
        identities={"Alice": alice, "Bob": bob},
        prototypes={"Alice": [alice], "Bob": [bob]},
        prototype_sources={"Alice": [str(tmp / "alice-reference.jpg")], "Bob": [str(tmp / "bob-reference.jpg")]},
        source_counts={"Alice": 5, "Bob": 5},
    ))
    allowed, report = identity_evaluation.activation_gate(
        db,
        db,
        sort_photos.CacheState(faces=[face, correct]),
        confirmed_set=dataset,
        protected_set=tmp / "missing-protected.csv",
        protected_baseline=tmp / "missing-baseline.json",
    )
    assert_true(
        allowed
        and report["confirmed"]["recall"] == report["confirmed"]["prior_recall"]
        and report["confirmed"]["incorrect"] == 0,
        f"unchanged conservative rejection incorrectly blocked activation: {report}",
    )


def test_recovery_scans_nested_intake_batches(tmp: Path) -> None:
    root = tmp / "no_usable_face"
    top = root / "top.jpg"
    nested = root / "batch" / "nested.png"
    make_image(top)
    make_image(nested)
    found = recover_no_usable_faces.iter_images(root)
    assert_true(found == [nested, top], f"nested recovery inputs were missed: {found}")


def test_recovery_reuses_sqlite_face_detections(tmp: Path) -> None:
    image = tmp / "unknown_identity" / "candidate.jpg"
    make_image(image)
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    face = sort_photos.CachedFace(
        src_str=str(image), face_index=0, det_score=0.99, bbox_size=120.0,
        sharpness=100.0, yaw_proxy=0.0, quality=0.9, embedding=embedding,
        image_phash=np.zeros(64, dtype=np.uint8), crop_jpeg=b"", label=None,
    )
    database_path = tmp / "analysis.sqlite3"
    with analysis_index.AnalysisIndex(database_path) as database:
        database.replace_detections(
            image,
            sort_photos.config_fingerprint(),
            "accepted_face",
            [sort_photos.cached_face_to_index_record(face)],
        )
    stats: Counter[str] = Counter()
    results = list(recover_no_usable_faces.iter_detection_results(
        [image], 1, 1024, 384, index_path=database_path, cache_stats=stats,
    ))
    assert_true(len(results) == 1 and len(results[0][3]) == 1,
                "recovery did not hydrate the cached face detection")
    assert_true(stats["sqlite_hits"] == 1 and stats["detected"] == 0,
                f"recovery unnecessarily reran face detection: {stats}")

    destination = tmp / "people" / "Alice" / "photos" / "Alice_00001.jpg"
    make_image(destination)
    cache = sort_photos.CacheState(config_fingerprint=sort_photos.config_fingerprint())
    merged = recover_no_usable_faces.merge_recovered_faces_into_cache(
        cache, [
            (destination, face, "Alice"),
            (destination, face, "Alice"),
        ],
    )
    destination_text = str(destination.resolve())
    assert_true(
        len(merged) == 1
        and destination_text in cache.file_signatures
        and len(cache.faces) == 1
        and cache.faces[0].src_str == destination_text
        and cache.faces[0].label == "Alice",
        "recovered organized image was not merged into the face cache",
    )


def test_identity_auto_match_requires_independent_consensus(tmp: Path) -> None:
    def embedding(index: int) -> np.ndarray:
        value = np.zeros(512, dtype=np.float32)
        value[index] = 1.0
        return value

    def record(name: str, cluster_id: int, emb: np.ndarray) -> sort_photos.FaceRecord:
        return sort_photos.FaceRecord(
            src=tmp / name,
            face_index=0,
            det_score=0.99,
            bbox_size=120.0,
            sharpness=120.0,
            yaw_proxy=0.0,
            embedding=emb,
            quality=1.0,
            cluster_id=cluster_id,
        )

    db = sort_photos.IdentityDB(
        identities={"Alice": embedding(0), "Bob": embedding(1)},
        source_counts={"Alice": 10, "Bob": 10},
    )

    singleton_map = {1: "person_001"}
    assigned = sort_photos.apply_identity_db_labels(
        [record("one.jpg", 1, embedding(0))], singleton_map, db
    )
    assert_true(assigned == 1 and singleton_map[1] == "Alice",
                "exceptionally strong singleton identity was not accepted")

    close_bob = embedding(0).copy()
    close_bob[0] = 0.94
    close_bob[1] = 0.341
    close_bob /= np.linalg.norm(close_bob)
    close_db = sort_photos.IdentityDB(
        identities={"Alice": embedding(0), "Close Bob": close_bob},
        source_counts={"Alice": 10, "Close Bob": 10},
    )
    uncertain_map = {6: "person_006"}
    assigned = sort_photos.apply_identity_db_labels(
        [record("uncertain.jpg", 6, embedding(0))], uncertain_map, close_db
    )
    assert_true(assigned == 0 and uncertain_map[6] == "person_006",
                "single-source match with a weak identity margin was accepted")

    consensus_map = {2: "person_002"}
    assigned = sort_photos.apply_identity_db_labels(
        [record("two-a.jpg", 2, embedding(0)), record("two-b.jpg", 2, embedding(0))],
        consensus_map,
        db,
    )
    assert_true(assigned == 1 and consensus_map[2] == "Alice",
                "strong independent identity consensus was not accepted")

    ambiguous_map = {3: "person_003"}
    assigned = sort_photos.apply_identity_db_labels(
        [record("mixed-a.jpg", 3, embedding(0)), record("mixed-b.jpg", 3, embedding(1))],
        ambiguous_map,
        db,
    )
    assert_true(assigned == 0 and ambiguous_map[3] == "person_003",
                "mixed identity cluster was incorrectly auto-filed")

    repeated_map = {4: "person_004", 5: "person_005"}
    assigned = sort_photos.apply_identity_db_labels(
        [
            record("four-a.jpg", 4, embedding(0)),
            record("four-b.jpg", 4, embedding(0)),
            record("five-a.jpg", 5, embedding(0)),
            record("five-b.jpg", 5, embedding(0)),
        ],
        repeated_map,
        db,
    )
    assert_true(
        assigned == 2 and repeated_map == {4: "Alice", 5: "Alice"},
        f"separate valid clusters could not match the same person: {repeated_map}",
    )


def test_anchor_cluster_merge_requires_unique_person_margin(tmp: Path) -> None:
    def vector(x: float, y: float) -> np.ndarray:
        value = np.zeros(512, dtype=np.float32)
        value[0] = x
        value[1] = y
        return value / np.linalg.norm(value)

    def record(name: str, cluster_id: int, embedding: np.ndarray) -> sort_photos.FaceRecord:
        return sort_photos.FaceRecord(
            src=tmp / name,
            face_index=0,
            det_score=0.99,
            bbox_size=160.0,
            sharpness=150.0,
            yaw_proxy=0.0,
            embedding=embedding,
            quality=1.0,
            cluster_id=cluster_id,
        )

    alice = vector(1.0, 0.0)
    close_bob = vector(0.995, 0.1)
    ambiguous = vector(0.999, 0.05)
    records = [
        record("alice.jpg", 1, alice),
        record("bob.jpg", 2, close_bob),
        record("ambiguous.jpg", 3, ambiguous),
    ]
    names = {1: "Alice", 2: "Bob", 3: "person_003"}
    merged = sort_photos.anchor_cluster_merge(records, names, tmp / "clusters")
    assert_true(merged == 0 and names[3] == "person_003",
                "ambiguous anchor cluster merged without a safe second-place margin")

    clear = vector(0.98, 0.2)
    far_bob = vector(0.0, 1.0)
    records = [
        record("alice-1.jpg", 10, alice),
        record("alice-2.jpg", 11, vector(0.99, 0.03)),
        record("bob-2.jpg", 12, far_bob),
        record("clear.jpg", 13, clear),
    ]
    names = {10: "Alice", 11: "Alice", 12: "Bob", 13: "person_013"}
    merged = sort_photos.anchor_cluster_merge(records, names, tmp / "clusters-2")
    assert_true(merged == 1 and 13 not in names,
                "duplicate Alice anchors incorrectly consumed the identity margin")


def test_identity_refresh_reuses_cached_negative_detection(tmp: Path) -> None:
    people = tmp / "people"
    image = people / "Alice" / "photos" / "Alice_00001.jpg"
    make_image(image)
    cache_dir = tmp / "cache"
    cache_file = cache_dir / "cache.pkl"
    identity_file = cache_dir / "person_identity_db.pkl"
    build_file = cache_dir / "person_identity_db.building.pkl"
    cache_dir.mkdir(parents=True)
    cache = sort_photos.CacheState(
        config_fingerprint=sort_photos.config_fingerprint(),
        file_signatures={str(image): sort_photos.file_signature(image)},
        faces=[],
    )
    with cache_file.open("wb") as handle:
        pickle.dump(cache, handle, protocol=pickle.HIGHEST_PROTOCOL)

    old_values = (
        sort_photos.CACHE_DIR,
        sort_photos.CACHE_FILE,
        sort_photos.IDENTITY_DB_FILE,
        sort_photos.IDENTITY_DB_BUILD_FILE,
        sort_photos._build_app,
    )
    sort_photos.CACHE_DIR = cache_dir
    sort_photos.CACHE_FILE = cache_file
    sort_photos.IDENTITY_DB_FILE = identity_file
    sort_photos.IDENTITY_DB_BUILD_FILE = build_file
    sort_photos._build_app = lambda: (_ for _ in ()).throw(
        AssertionError("current cached no-face result was needlessly redetected")
    )
    try:
        db = sort_photos.build_identity_db_from_person_folders(people, force_rebuild=True)
        assert_true(not db.identities, "cached no-face image unexpectedly created an identity")
    finally:
        (
            sort_photos.CACHE_DIR,
            sort_photos.CACHE_FILE,
            sort_photos.IDENTITY_DB_FILE,
            sort_photos.IDENTITY_DB_BUILD_FILE,
            sort_photos._build_app,
        ) = old_values


def test_identity_database_promotion_is_atomic_and_backed_up(tmp: Path) -> None:
    cache_dir = tmp / "cache"
    identity_file = cache_dir / "person_identity_db.pkl"
    build_file = cache_dir / "person_identity_db.building.pkl"
    cache_dir.mkdir(parents=True)
    identity_file.write_bytes(b"previous-complete-database")
    vector = np.zeros(512, dtype=np.float32)
    vector[0] = 1.0
    db = sort_photos.IdentityDB(
        config_fingerprint=sort_photos.config_fingerprint(),
        identities={"Alice": vector},
        source_counts={"Alice": 5},
    )
    old_values = (
        sort_photos.CACHE_DIR,
        sort_photos.IDENTITY_DB_FILE,
        sort_photos.IDENTITY_DB_BUILD_FILE,
    )
    sort_photos.CACHE_DIR = cache_dir
    sort_photos.IDENTITY_DB_FILE = identity_file
    sort_photos.IDENTITY_DB_BUILD_FILE = build_file
    try:
        sort_photos.write_identity_db(db, build_file)
        assert_true(identity_file.read_bytes() == b"previous-complete-database",
                    "identity checkpoint replaced the active database")
        sort_photos.save_identity_db(db)
        backups = list(cache_dir.glob("person_identity_db.pkl.bak.profile_refresh_*"))
        assert_true(len(backups) == 1, f"identity backup was not retained: {backups}")
        assert_true(backups[0].read_bytes() == b"previous-complete-database",
                    "identity backup did not preserve the previous database")
        with identity_file.open("rb") as handle:
            promoted = pickle.load(handle)
        assert_true("Alice" in promoted.identities, "new identity database was not promoted")
    finally:
        (
            sort_photos.CACHE_DIR,
            sort_photos.IDENTITY_DB_FILE,
            sort_photos.IDENTITY_DB_BUILD_FILE,
        ) = old_values


def test_identity_profiles_select_quality_and_diversity(tmp: Path) -> None:
    del tmp

    def vector(x: float, y: float, z: float = 0.0) -> np.ndarray:
        value = np.zeros(512, dtype=np.float32)
        value[:3] = [x, y, z]
        return value / max(float(np.linalg.norm(value)), 1e-9)

    samples = [
        identity_profiles.ReferenceSample("front-a", vector(1.0, 0.00), 1.00),
        identity_profiles.ReferenceSample("front-b", vector(1.0, 0.01), 0.99),
        identity_profiles.ReferenceSample("profile", vector(0.92, 0.38), 0.86),
        identity_profiles.ReferenceSample("lighting", vector(0.93, 0.0, 0.36), 0.82),
    ]
    selected = identity_profiles.select_diverse_samples(samples, limit=3)
    sources = {sample.source for sample in selected}
    assert_true("front-a" in sources, f"best-quality reference was not selected: {sources}")
    assert_true("profile" in sources and "lighting" in sources,
                f"diverse poses were replaced by near-identical references: {sources}")


def test_confirmed_identity_examples_are_bounded_and_persistent(tmp: Path) -> None:
    def vector(index: int) -> np.ndarray:
        value = np.zeros(512, dtype=np.float32)
        value[index] = 1.0
        return value

    dominant = [
        identity_profiles.ReferenceSample(f"core-{index}", vector(index), 0.9)
        for index in range(6)
    ]
    trusted = [
        identity_profiles.ReferenceSample("trusted-profile", vector(20), 0.7),
        identity_profiles.ReferenceSample("trusted-low-light", vector(21), 0.6),
        identity_profiles.ReferenceSample("trusted-extra", vector(22), 0.5),
    ]
    selected = identity_profiles.select_profile_prototypes(
        dominant, trusted, limit=4, trusted_limit=2
    )
    sources = {sample.source for sample in selected}
    assert_true(len(selected) == 4, f"prototype limit was not preserved: {len(selected)}")
    assert_true(
        len(sources.intersection({"trusted-profile", "trusted-low-light", "trusted-extra"})) == 2,
        f"trusted prototype allowance was not bounded: {sources}",
    )
    assert_true(
        len(sources.intersection({sample.source for sample in dominant})) == 2,
        f"trusted examples displaced the identity core: {sources}",
    )
    production_selected = identity_profiles.select_profile_prototypes(
        dominant,
        trusted,
        limit=sort_photos.IDENTITY_MAX_PROTOTYPES_PER_PERSON,
        trusted_limit=min(
            sort_photos.IDENTITY_MAX_TRUSTED_PROTOTYPES_PER_PERSON,
            max(1, sort_photos.IDENTITY_MAX_PROTOTYPES_PER_PERSON // 2),
        ),
    )
    production_trusted = {
        sample.source for sample in production_selected
        if sample.source.startswith("trusted-")
    }
    production_core = {
        sample.source for sample in production_selected
        if sample.source.startswith("core-")
    }
    assert_true(
        len(production_trusted) <= sort_photos.IDENTITY_MAX_TRUSTED_PROTOTYPES_PER_PERSON,
        "production identity profile exceeded the trusted-example bound",
    )
    assert_true(
        len(production_core) >= len(production_trusted),
        "trusted examples displaced more than half of the production identity core",
    )


def test_reserved_review_labels_are_not_people(tmp: Path) -> None:
    assert_true(
        all(not sort_photos.is_real_person_label(value) for value in (
            "unknown", "Unknown", "__junk__", "Junk", "junk", "person_123"
        ))
        and sort_photos.is_real_person_label("Kajal Agarwal"),
        "reserved review labels can still enter the identity database",
    )

    people = tmp / "people"
    organized = people / "Alice" / "photos" / "Alice_00001.jpg"
    make_image(organized)
    registry = tmp / "confirmed_identity_examples.json"
    identity_confirmations.record(
        registry,
        person="Alice",
        organized_path=organized,
        content_sha256=sort_photos.sha256_file(organized),
        original_name="difficult.jpg",
    )
    paths = identity_confirmations.paths_for_person(registry, "Alice", people)
    assert_true(paths == [organized.resolve()], f"confirmed path was not restored: {paths}")
    assert_true(
        identity_confirmations.signature_for_person(registry, "Alice", people),
        "confirmed identity profile signature was empty",
    )


def test_unknown_review_clusters_and_confirms_safely(tmp: Path) -> None:
    def vector(x: float, y: float) -> np.ndarray:
        value = np.zeros(512, dtype=np.float32)
        value[:2] = [x, y]
        return value / max(float(np.linalg.norm(value)), 1e-9)

    alice = vector(1.0, 0.0)
    bob = vector(0.0, 1.0)
    db = sort_photos.IdentityDB(
        config_fingerprint=sort_photos.config_fingerprint(),
        identities={"Alice": alice, "Bob": bob},
        prototypes={"Alice": [alice], "Bob": [bob]},
        source_counts={"Alice": 5, "Bob": 5},
    )

    def unknown_item(name: str, embedding: np.ndarray) -> review_unknown_identities.UnknownItem:
        path = tmp / name
        make_image(path)
        face_record = sort_photos.CachedFace(
            src_str=str(path), face_index=0, det_score=0.99, bbox_size=120.0,
            sharpness=100.0, yaw_proxy=0.0, quality=0.9,
            embedding=embedding, image_phash=np.zeros(64, dtype=np.uint8),
            crop_jpeg=b"jpeg", label=None,
        )
        candidates = identity_profiles.rank_candidates(
            embedding, db.identities, db.prototypes
        )[:3]
        return review_unknown_identities.UnknownItem(
            review_unknown_identities.item_key(path), path, face_record, tuple(candidates)
        )

    first = unknown_item("alice-a.jpg", vector(1.0, 0.01))
    second = unknown_item("alice-b.jpg", vector(1.0, 0.03))
    third = unknown_item("bob.jpg", vector(0.01, 1.0))
    original_key = first.key
    first.path.write_bytes(first.path.read_bytes() + b"replacement")
    assert_true(
        review_unknown_identities.item_key(first.path) != original_key,
        "a replaced unknown file incorrectly inherited the prior review decision",
    )
    make_image(first.path)
    first = unknown_item("alice-a.jpg", vector(1.0, 0.01))
    clusters = review_unknown_identities.cluster_items(
        [first, second, third], db, eps=0.05
    )
    alice_cluster = next(cluster for cluster in clusters if len(cluster.items) == 2)
    assert_true(
        len(clusters[0].items) == 2,
        "unknown review did not place the largest visual cluster first",
    )
    assert_true(
        alice_cluster.candidates[0].name == "Alice",
        f"unknown cluster suggested the wrong person: {alice_cluster.candidates}",
    )
    page = review_unknown_identities.render_html(
        clusters,
        {"version": 1, "items": {}},
        ["Alice", "Bob"],
        {"queue_files": 3, "batch_files": 3, "sqlite_hits": 3, "detected": 0},
        interactive=True,
    )
    for label in (
        "Confirm all as Alice", "Confirm Cluster", "Keep Unknown", "Ignore",
        "Not Alice", "margin=", "quality=", 'id="selectVisible"',
        'id="confirmSelected"', 'id="keepSelected"',
        'id="ignoreSelected"', 'class=\'row-select\'',
        'data-density="compact"', 'id="queueCount"',
        "Action started.", "const jobsURL=ids.length?",
        "Confirm all as Unknown", "Move all to Junk",
        'id="confirmDialog"', "Finish Unknown Review?",
        "Unknown Identity Quick Review", 'id="finishReview"',
        "data-shortcut='1'", "maybeLoadNextBatch", "fetch('/next-batch'",
        "fetch('/skip'", "ArrowRight", "Finish Review",
        'id="loadNextBatch"', "returnedJobIds", "activeJobIds.delete(id)",
        "Wait for the next batch to finish loading.", "fetch('/batch-status'",
        "pollBatchStatus",
    ):
        assert_true(label in page, f"unknown review UI is missing {label}")
    assert_true(
        "await askConfirmation(verb)" not in page,
        "unknown review actions should queue immediately without confirmation",
    )
    assert_true(
        'id="legacyReviewScript"' not in page,
        "unknown review page should not include the obsolete legacy script",
    )

    decisions_path = tmp / "decisions.json"
    state = {
        "items_by_key": {first.key: first, second.key: second, third.key: third},
        "decisions_path": decisions_path,
        "identity_db": db,
    }
    message = review_unknown_identities.apply_decision(
        state, item_keys=[first.key, second.key], action="keep_unknown"
    )
    assert_true("Kept 2" in message and first.path.is_file() and second.path.is_file(),
                "multi-image keep-unknown action moved or lost a source")
    decisions = review_unknown_identities.load_decisions(decisions_path)["items"]
    assert_true(all(decisions[key]["action"] == "keep_unknown"
                    for key in (first.key, second.key)),
                "multi-image keep-unknown decision was not persisted")

    old_sorted = recover_no_usable_faces.DEFAULT_SORTED
    recover_no_usable_faces.DEFAULT_SORTED = tmp
    try:
        state.update({
            "junk_dir": tmp / "ready_to_delete" / "unknown_junk",
            "cache": sort_photos.CacheState(),
            "cache_dirty": False,
        })
        message = review_unknown_identities.apply_decision(
            state, item_keys=[third.key], action="move_to_junk"
        )
    finally:
        recover_no_usable_faces.DEFAULT_SORTED = old_sorted
    junk_files = list((tmp / "ready_to_delete" / "unknown_junk").glob("bob*.jpg"))
    decisions = review_unknown_identities.load_decisions(decisions_path)["items"]
    assert_true(
        "recoverable junk" in message
        and not third.path.exists()
        and len(junk_files) == 1
        and junk_files[0].is_file(),
        "unknown junk action did not preserve the original recoverably",
    )
    assert_true(
        decisions[third.key]["action"] == "moved_to_junk"
        and Path(decisions[third.key]["destination"]) == junk_files[0],
        "unknown junk decision or destination was not persisted",
    )


def test_unknown_review_auto_match_requires_independent_verifier(tmp: Path) -> None:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    db = sort_photos.IdentityDB(
        identities={"Alice": embedding},
        prototypes={"Alice": [embedding]},
        source_counts={"Alice": 5},
        match_thresholds={"Alice": 0.32},
        strict_thresholds={"Alice": 0.27},
    )
    source = tmp / "strict.jpg"
    make_image(source)
    face = sort_photos.CachedFace(
        src_str=str(source), face_index=0, det_score=0.99, bbox_size=120.0,
        sharpness=100.0, yaw_proxy=0.0, quality=0.9,
        embedding=embedding, image_phash=np.zeros(64, dtype=np.uint8),
        crop_jpeg=b"jpeg", label=None,
    )
    candidates = (
        identity_profiles.IdentityCandidate("Alice", 0.85, 0.15),
        identity_profiles.IdentityCandidate("Bob", 0.40, 0.60),
    )
    item = review_unknown_identities.UnknownItem(
        review_unknown_identities.item_key(source), source, face, candidates
    )
    cluster = review_unknown_identities.UnknownCluster("single", (item,), candidates, 0.0)
    matches = review_unknown_identities.automatic_matches(
        [cluster], db, require_secondary=True
    )
    assert_true(not matches, "single-image auto-match bypassed the independent verifier")

    verified = review_unknown_identities.UnknownItem(
        item.key,
        item.path,
        item.face,
        item.candidates,
        secondary_identity_matcher.SecondaryVerification(True, "Alice", 0.18, 0.20),
    )
    verified_cluster = review_unknown_identities.UnknownCluster(
        "verified", (verified,), candidates, 0.0
    )
    matches = review_unknown_identities.automatic_matches(
        [verified_cluster], db, require_secondary=True
    )
    assert_true(
        len(matches) == 1
        and matches[0].person == "Alice"
        and matches[0].lane == "secondary_agreement",
        f"independently verified single match was not accepted safely: {matches}",
    )

    jointly_verified = review_unknown_identities.UnknownItem(
        item.key,
        item.path,
        item.face,
        item.candidates,
        secondary_identity_matcher.SecondaryVerification(False, "Alice", 0.38, 0.12),
    )
    matches = review_unknown_identities.automatic_matches(
        [review_unknown_identities.UnknownCluster(
            "joint", (jointly_verified,), candidates, 0.0
        )],
        db,
        require_secondary=True,
    )
    assert_true(
        len(matches) == 1 and matches[0].lane == "secondary_agreement",
        f"strong dual-model agreement was blocked by a legacy absolute cutoff: {matches}",
    )

    weak_secondary = review_unknown_identities.UnknownItem(
        item.key,
        item.path,
        item.face,
        item.candidates,
        secondary_identity_matcher.SecondaryVerification(False, "Alice", 0.43, 0.12),
    )
    matches = review_unknown_identities.automatic_matches(
        [review_unknown_identities.UnknownCluster(
            "weak", (weak_secondary,), candidates, 0.0
        )],
        db,
        require_secondary=True,
    )
    assert_true(not matches, "weak secondary evidence bypassed the joint safety limits")

    rescue_candidates = (
        identity_profiles.IdentityCandidate("Alice", 0.50, 0.50),
        identity_profiles.IdentityCandidate("Bob", 0.44, 0.56),
    )
    rescued = review_unknown_identities.UnknownItem(
        item.key,
        item.path,
        item.face,
        rescue_candidates,
        secondary_identity_matcher.SecondaryVerification(True, "Alice", 0.18, 0.18),
    )
    matches = review_unknown_identities.automatic_matches(
        [review_unknown_identities.UnknownCluster(
            "rescue", (rescued,), rescue_candidates, 0.0
        )],
        db,
        require_secondary=True,
    )
    assert_true(
        len(matches) == 1 and matches[0].lane == "trusted_verifier_rescue",
        f"strict independent-verifier rescue did not recover a difficult face: {matches}",
    )


def test_unknown_review_refreshes_stale_secondary_verifier(tmp: Path) -> None:
    del tmp
    embedding = np.zeros(4, dtype=np.float32)
    embedding[0] = 1.0
    primary = sort_photos.IdentityDB(
        config_fingerprint="current",
        identities={"Alice": embedding},
        prototypes={"Alice": [embedding]},
        prototype_sources={"Alice": ["alice.jpg"]},
    )
    expected = secondary_identity_matcher.primary_signature(primary)
    stale = secondary_identity_matcher.SecondaryIdentityDB(
        primary_signature="stale",
        identities={"Alice": embedding},
    )
    refreshed = secondary_identity_matcher.SecondaryIdentityDB(
        primary_signature=expected,
        trusted_signature=secondary_identity_matcher.trusted_confirmation_signature(
            sort_photos.IDENTITY_CONFIRMATIONS_FILE
        ),
        identities={"Alice": embedding},
    )
    original_load = secondary_identity_matcher.load
    original_build = secondary_identity_matcher.build_database
    builds: list[bool] = []
    secondary_identity_matcher.load = lambda: stale
    secondary_identity_matcher.build_database = lambda _db, _cache: (
        builds.append(True) or refreshed
    )
    try:
        matcher, status = review_unknown_identities.prepare_secondary_verifier(
            primary,
            sort_photos.CacheState(),
            requested=True,
        )
    finally:
        secondary_identity_matcher.load = original_load
        secondary_identity_matcher.build_database = original_build
    assert_true(matcher is not None, "stale verifier was not refreshed")
    assert_true(status == "refreshed", f"unexpected verifier status: {status}")
    assert_true(len(builds) == 1, "stale verifier refresh did not run exactly once")


def test_unknown_review_does_not_refresh_unrequested_verifier(tmp: Path) -> None:
    del tmp
    embedding = np.zeros(4, dtype=np.float32)
    embedding[0] = 1.0
    primary = sort_photos.IdentityDB(
        config_fingerprint="current",
        identities={"Alice": embedding},
        prototypes={"Alice": [embedding]},
        prototype_sources={"Alice": ["alice.jpg"]},
    )
    stale = secondary_identity_matcher.SecondaryIdentityDB(
        primary_signature="stale",
        identities={"Alice": embedding},
    )
    original_load = secondary_identity_matcher.load
    original_build = secondary_identity_matcher.build_database
    secondary_identity_matcher.load = lambda: stale
    secondary_identity_matcher.build_database = lambda _db, _cache: (_ for _ in ()).throw(
        AssertionError("unrequested verifier refresh ran")
    )
    try:
        matcher, status = review_unknown_identities.prepare_secondary_verifier(
            primary,
            sort_photos.CacheState(),
            requested=False,
        )
    finally:
        secondary_identity_matcher.load = original_load
        secondary_identity_matcher.build_database = original_build
    assert_true(matcher is None, "unrequested stale verifier was loaded")
    assert_true(status == "not requested", f"unexpected verifier status: {status}")


def test_secondary_verifier_batches_interactive_confirmations(tmp: Path) -> None:
    confirmations = tmp / "confirmed.json"
    initial_signature = secondary_identity_matcher.trusted_confirmation_signature(
        confirmations
    )
    db = secondary_identity_matcher.SecondaryIdentityDB(
        trusted_signature=initial_signature,
        trusted_example_count=0,
    )
    assert_true(
        secondary_identity_matcher.trusted_snapshot_is_current(db, confirmations),
        "exact verifier confirmation snapshot was considered stale",
    )
    for index in range(3):
        source = tmp / "people" / "Alice" / "photos" / f"Alice_{index:05d}.jpg"
        make_image(source, (20 + index * 40, 80, 160))
        identity_confirmations.record(
            confirmations,
            person="Alice",
            organized_path=source,
            content_sha256=sort_photos.sha256_file(source),
            original_name=source.name,
        )
    assert_true(
        secondary_identity_matcher.trusted_snapshot_is_current(
            db, confirmations, rebuild_interval=4
        ),
        "small append-only confirmation batch forced an immediate verifier rebuild",
    )
    assert_true(
        not secondary_identity_matcher.trusted_snapshot_is_current(
            db, confirmations, rebuild_interval=3
        ),
        "completed confirmation batch did not request a verifier rebuild",
    )


def test_unknown_review_uses_bounded_confirmation_memory(tmp: Path) -> None:
    alice_core = np.zeros(4, dtype=np.float32)
    alice_core[0] = 1.0
    alice_profile = np.zeros(4, dtype=np.float32)
    alice_profile[1] = 1.0
    bob = np.zeros(4, dtype=np.float32)
    bob[2] = 1.0
    source = tmp / "people" / "Alice" / "photos" / "Alice_00001.jpg"
    make_image(source)
    face = sort_photos.CachedFace(
        src_str=str(source),
        face_index=0,
        det_score=0.99,
        bbox_size=120.0,
        sharpness=100.0,
        yaw_proxy=0.0,
        quality=0.9,
        embedding=alice_profile,
        image_phash=np.zeros(64, dtype=np.uint8),
        crop_jpeg=b"profile",
        label="Alice",
    )
    cache = sort_photos.CacheState(faces=[face])
    db = sort_photos.IdentityDB(
        identities={"Alice": alice_core, "Bob": bob},
        prototypes={"Alice": [alice_core], "Bob": [bob]},
    )
    confirmations = tmp / "confirmed.json"
    identity_confirmations.record(
        confirmations,
        person="Alice",
        organized_path=source,
        content_sha256=sort_photos.sha256_file(source),
        original_name=source.name,
    )
    profiles, counts = review_unknown_identities.build_trusted_review_prototypes(
        db,
        cache,
        confirmations_path=confirmations,
        limit_per_person=2,
    )
    ranked = identity_profiles.rank_candidates(
        alice_profile,
        db.identities,
        profiles,
    )
    assert_true(
        ranked[0].name == "Alice" and ranked[0].distance < 0.40,
        f"trusted review memory did not recover the confirmed appearance: {ranked}",
    )
    assert_true(counts == {"Alice": 1}, f"trusted review counts are wrong: {counts}")
    assert_true(
        len(db.prototypes["Alice"]) == 1,
        "review-only confirmation memory mutated the production identity profile",
    )


def test_unknown_review_auto_sweep_scans_entire_queue(tmp: Path) -> None:
    unknown = tmp / "unknown"
    files = [unknown / f"image-{index}.jpg" for index in range(5)]
    for index, path in enumerate(files):
        make_image(path, (20 + index * 20, 80, 120))
    calls: list[int] = []
    original_collect = review_unknown_identities.collect_items
    original_apply = review_unknown_identities.apply_automatic_review
    review_unknown_identities.collect_items = lambda batch, *_args, **_kwargs: (
        calls.append(len(batch)) or [], Counter(), [
            review_unknown_identities.UnsupportedUnknown(
                key=review_unknown_identities.item_key(path), path=path,
                queue_kind="no_usable_face", detector_status="no_face",
            ) for path in batch
        ]
    )
    review_unknown_identities.apply_automatic_review = lambda *_args, **_kwargs: {
        "confirmed": 0,
        "failed": 0,
    }
    state = {
        "auto_review_allowed": True,
        "auto_review_preview": False,
        "decisions_path": tmp / "decisions.json",
        "output_dir": tmp / "reports",
        "unknown_root": unknown,
        "temporarily_skipped": set(),
        "batch_limit": 2,
        "identity_db": sort_photos.IdentityDB(),
        "workers": 1,
        "primary_det_size": 320,
        "fallback_det_size": 320,
        "hard_negatives": {},
        "cluster_eps": 0.30,
        "route_unsupported": False,
        "cache": sort_photos.CacheState(),
        "cache_dirty": False,
        "auto_review_gate": {"signature": "test-gate"},
    }
    Path(state["output_dir"]).mkdir(parents=True, exist_ok=True)
    try:
        result = review_unknown_identities.run_automatic_sweep(state)
        repeated = review_unknown_identities.run_automatic_sweep(state)
    finally:
        review_unknown_identities.collect_items = original_collect
        review_unknown_identities.apply_automatic_review = original_apply
    assert_true(calls == [2, 2, 1], f"auto-sweep skipped later queue batches: {calls}")
    assert_true(
        result["scanned"] == 5 and result["batches"] == 3,
        f"auto-sweep summary is incomplete: {result}",
    )
    assert_true(
        calls == [2, 2, 1]
        and repeated.get("cached") is True
        and repeated.get("scanned") == 5,
        f"unchanged auto-sweep was not reused: calls={calls}, result={repeated}",
    )


def test_unknown_review_auto_cluster_rejects_identity_dissent(tmp: Path) -> None:
    alice = np.zeros(512, dtype=np.float32)
    alice[0] = 1.0
    bob = np.zeros(512, dtype=np.float32)
    bob[1] = 1.0
    db = sort_photos.IdentityDB(
        identities={"Alice": alice, "Bob": bob},
        prototypes={"Alice": [alice], "Bob": [bob]},
        source_counts={"Alice": 5, "Bob": 5},
        match_thresholds={"Alice": 0.32, "Bob": 0.32},
        strict_thresholds={"Alice": 0.27, "Bob": 0.27},
    )

    def item(index: int, person: str = "Alice") -> review_unknown_identities.UnknownItem:
        source = tmp / f"face-{index}.jpg"
        make_image(source, color=(90 + index * 10, 120, 180))
        embedding = alice if person == "Alice" else bob
        face = sort_photos.CachedFace(
            src_str=str(source), face_index=0, det_score=0.99, bbox_size=120.0,
            sharpness=100.0, yaw_proxy=0.0, quality=0.9,
            embedding=embedding, image_phash=np.zeros(64, dtype=np.uint8),
            crop_jpeg=b"jpeg", label=None,
        )
        other = "Bob" if person == "Alice" else "Alice"
        candidates = (
            identity_profiles.IdentityCandidate(person, 0.82, 0.18),
            identity_profiles.IdentityCandidate(other, 0.40, 0.60),
        )
        secondary = secondary_identity_matcher.SecondaryVerification(
            True, person, 0.20, 0.18
        ) if index < 2 or person == "Bob" else None
        return review_unknown_identities.UnknownItem(
            review_unknown_identities.item_key(source), source, face, candidates, secondary
        )

    good_items = (item(0), item(1), item(2))
    cluster_candidates = (
        identity_profiles.IdentityCandidate("Alice", 0.81, 0.19),
        identity_profiles.IdentityCandidate("Bob", 0.45, 0.55),
    )
    good_cluster = review_unknown_identities.UnknownCluster(
        "good", good_items, cluster_candidates, 0.04
    )
    matches = review_unknown_identities.automatic_matches(
        [good_cluster], db, require_secondary=True
    )
    assert_true(
        len(matches) == 1
        and matches[0].lane == "cluster_consensus"
        and len(matches[0].item_keys) == 3,
        f"safe corroborated cluster was not accepted: {matches}",
    )

    mixed_items = (good_items[0], good_items[1], item(3, "Bob"))
    mixed_cluster = review_unknown_identities.UnknownCluster(
        "mixed", mixed_items, cluster_candidates, 0.04
    )
    mixed_matches = review_unknown_identities.automatic_matches(
        [mixed_cluster], db, require_secondary=True
    )
    assert_true(
        all(match.lane != "cluster_consensus" for match in mixed_matches)
        and any(
            match.person == "Bob" and mixed_items[2].key in match.item_keys
            for match in mixed_matches
        )
        and all(len(match.item_keys) == 1 for match in mixed_matches),
        f"mixed-identity cluster was bulk auto-confirmed: {mixed_matches}",
    )


def test_unknown_review_automatic_confirmation_is_not_trusted_training(tmp: Path) -> None:
    sorted_root = tmp / "sorted"
    people = sorted_root / "photos_by_person"
    unknown = sorted_root / "_source_review" / "unassigned_intake" / "unknown_identity"
    source = unknown / "automatic.jpg"
    make_image(source)
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    face = sort_photos.CachedFace(
        src_str=str(source), face_index=0, det_score=0.99, bbox_size=120.0,
        sharpness=100.0, yaw_proxy=0.0, quality=0.9,
        embedding=embedding, image_phash=np.zeros(64, dtype=np.uint8),
        crop_jpeg=b"jpeg", label=None,
    )
    db = sort_photos.IdentityDB(
        identities={"Alice": embedding}, prototypes={"Alice": [embedding]},
        source_counts={"Alice": 5},
    )
    candidate = identity_profiles.IdentityCandidate("Alice", 0.85, 0.15)
    item = review_unknown_identities.UnknownItem(
        review_unknown_identities.item_key(source), source, face, (candidate,)
    )
    old_values = (
        recover_no_usable_faces.DEFAULT_PEOPLE,
        recover_no_usable_faces.DEFAULT_SORTED,
        recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES,
    )
    old_nudity = sort_photos.NUDITY_SORT_ENABLED
    recover_no_usable_faces.DEFAULT_PEOPLE = people
    recover_no_usable_faces.DEFAULT_SORTED = sorted_root
    recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES = 0
    sort_photos.NUDITY_SORT_ENABLED = False
    try:
        state = {
            "items_by_key": {item.key: item},
            "decisions_path": tmp / "decisions.json",
            "identity_db": db,
            "existing_hashes": {"Alice": set()},
            "next_indexes": {},
            "confirmations_path": tmp / "trusted-confirmations.json",
            "evaluation_path": tmp / "trusted-benchmark.csv",
            "review_dir": sorted_root / "_source_review" / "ready_to_delete" / "confirmed",
            "analysis_index": tmp / "analysis.sqlite3",
            "cache": sort_photos.CacheState(),
            "cache_dirty": False,
            "identity_dirty": False,
        }
        review_unknown_identities.apply_decision(
            state,
            item_keys=[item.key],
            action="confirm",
            person_value="Alice",
            trusted_confirmation=False,
            decision_metadata={"automatic": True, "auto_lane": "secondary_agreement"},
        )
    finally:
        (
            recover_no_usable_faces.DEFAULT_PEOPLE,
            recover_no_usable_faces.DEFAULT_SORTED,
            recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES,
        ) = old_values
        sort_photos.NUDITY_SORT_ENABLED = old_nudity
    decision = review_unknown_identities.load_decisions(
        tmp / "decisions.json"
    )["items"][item.key]
    assert_true(
        decision["automatic"] is True
        and decision["trusted_confirmation"] is False
        and not (tmp / "trusted-confirmations.json").exists()
        and not (tmp / "trusted-benchmark.csv").exists()
        and state["cache_dirty"]
        and not state["identity_dirty"],
        "automatic match contaminated trusted identity training data",
    )


def test_unknown_review_manual_correction_teaches_option_one_rejection(tmp: Path) -> None:
    sorted_root = tmp / "sorted"
    people = sorted_root / "photos_by_person"
    unknown = sorted_root / "_source_review" / "unassigned_intake" / "unknown_identity"
    source = unknown / "correction.jpg"
    make_image(source)
    alice = np.zeros(512, dtype=np.float32)
    alice[0] = 1.0
    bob = np.zeros(512, dtype=np.float32)
    bob[1] = 1.0
    face = sort_photos.CachedFace(
        src_str=str(source), face_index=0, det_score=0.99, bbox_size=120.0,
        sharpness=100.0, yaw_proxy=0.0, quality=0.9,
        embedding=alice, image_phash=np.zeros(64, dtype=np.uint8),
        crop_jpeg=b"jpeg", label=None,
    )
    candidates = (
        identity_profiles.IdentityCandidate("Bob", 0.78, 0.22),
        identity_profiles.IdentityCandidate("Alice", 0.62, 0.38),
    )
    item = review_unknown_identities.UnknownItem(
        review_unknown_identities.item_key(source), source, face, candidates
    )
    db = sort_photos.IdentityDB(
        identities={"Alice": alice, "Bob": bob},
        prototypes={"Alice": [alice], "Bob": [bob]},
        source_counts={"Alice": 5, "Bob": 5},
    )
    negatives_path = tmp / "hard-negatives.json"
    old_values = (
        recover_no_usable_faces.DEFAULT_PEOPLE,
        recover_no_usable_faces.DEFAULT_SORTED,
        recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES,
    )
    old_nudity = sort_photos.NUDITY_SORT_ENABLED
    recover_no_usable_faces.DEFAULT_PEOPLE = people
    recover_no_usable_faces.DEFAULT_SORTED = sorted_root
    recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES = 0
    sort_photos.NUDITY_SORT_ENABLED = False
    state = {
        "items_by_key": {item.key: item},
        "decisions_path": tmp / "decisions.json",
        "identity_db": db,
        "identity_names": ["Alice", "Bob"],
        "people_root": people,
        "existing_hashes": {"Alice": set(), "Bob": set()},
        "destinations_by_hash": {},
        "next_indexes": {},
        "confirmations_path": tmp / "trusted-confirmations.json",
        "evaluation_path": tmp / "trusted-benchmark.csv",
        "hard_negatives_path": negatives_path,
        "hard_negatives": {},
        "review_dir": sorted_root / "_source_review" / "ready_to_delete" / "confirmed",
        "analysis_index": tmp / "analysis.sqlite3",
        "cache": sort_photos.CacheState(),
        "cache_dirty": False,
        "identity_dirty": False,
    }
    try:
        message = review_unknown_identities.apply_decision(
            state,
            item_keys=[item.key],
            action="confirm",
            person_value="Alice",
            trusted_confirmation=True,
        )
    finally:
        (
            recover_no_usable_faces.DEFAULT_PEOPLE,
            recover_no_usable_faces.DEFAULT_SORTED,
            recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES,
        ) = old_values
        sort_photos.NUDITY_SORT_ENABLED = old_nudity
    vectors = identity_hard_negatives.vectors_by_person(negatives_path)
    decision = review_unknown_identities.load_decisions(
        tmp / "decisions.json"
    )["items"][item.key]
    assert_true(
        len(vectors.get("Bob", [])) == 1
        and decision["corrected_candidate"] == "Bob"
        and "learned 1 option-1 correction" in message,
        "manual correction did not teach the rejected option-1 identity",
    )


def test_confirmed_review_learning_excludes_its_own_prototype(tmp: Path) -> None:
    query = np.zeros(512, dtype=np.float32)
    query[0] = 1.0
    alice = np.zeros(512, dtype=np.float32)
    alice[0] = 0.5
    alice[1] = np.sqrt(0.75)
    bob = np.zeros(512, dtype=np.float32)
    bob[0] = 0.8
    bob[2] = 0.6
    source = tmp / "explicit-alice.jpg"
    make_image(source)
    face = sort_photos.CachedFace(
        src_str=str(source), face_index=0, det_score=0.99, bbox_size=120.0,
        sharpness=100.0, yaw_proxy=0.0, quality=0.9,
        embedding=query, image_phash=np.zeros(64, dtype=np.uint8),
        crop_jpeg=b"jpeg", label=None,
    )
    db = sort_photos.IdentityDB(
        identities={"Alice": alice, "Bob": bob},
        prototypes={"Alice": [alice], "Bob": [bob]},
        source_counts={"Alice": 5, "Bob": 5},
        match_thresholds={"Alice": 0.32, "Bob": 0.32},
        strict_thresholds={"Alice": 0.27, "Bob": 0.27},
    )

    class AgreeingMatcher:
        def __init__(self):
            self.db = secondary_identity_matcher.SecondaryIdentityDB(
                identities={"Alice": alice, "Bob": bob}
            )

        def verify(self, _crop, expected_person, excluded_source=None):
            if excluded_source is None:
                return secondary_identity_matcher.SecondaryVerification(
                    True, "Alice", 0.05, 0.40
                )
            return secondary_identity_matcher.SecondaryVerification(
                True, expected_person, 0.10, 0.30
            )

        def flush(self):
            return None

    benchmark = tmp / "confirmed.csv"
    evaluation_enrollment.enroll(
        source=source,
        person="Alice",
        content_sha256=sort_photos.sha256_file(source),
        path=benchmark,
    )
    negatives = tmp / "hard-negatives.json"
    report = review_unknown_identities.learn_hard_negatives_from_confirmed_reviews(
        db,
        sort_photos.CacheState(faces=[face]),
        AgreeingMatcher(),
        hard_negatives_path=negatives,
        review_prototypes={"Alice": [query], "Bob": [bob]},
        evaluation_path=benchmark,
        state_path=tmp / "learning-state.json",
    )
    vectors = identity_hard_negatives.vectors_by_person(negatives)
    assert_true(
        report["learned"] == 1
        and len(vectors.get("Bob", [])) == 1
        and not vectors.get("Alice"),
        "leave-one-source-out learning did not reject the competing identity",
    )


def test_unknown_review_joint_policy_is_benchmark_gated(tmp: Path) -> None:
    alice = np.zeros(512, dtype=np.float32)
    alice[0] = 1.0
    bob = np.zeros(512, dtype=np.float32)
    bob[1] = 1.0
    source = tmp / "confirmed.jpg"
    make_image(source)
    face = sort_photos.CachedFace(
        src_str=str(source), face_index=0, det_score=0.99, bbox_size=120.0,
        sharpness=100.0, yaw_proxy=0.0, quality=0.9,
        embedding=alice, image_phash=np.zeros(64, dtype=np.uint8),
        crop_jpeg=b"jpeg", label=None,
    )
    db = sort_photos.IdentityDB(
        identities={"Alice": alice, "Bob": bob},
        prototypes={"Alice": [alice], "Bob": [bob]},
        prototype_sources={"Alice": [str(tmp / "alice-reference.jpg")], "Bob": [str(tmp / "bob-reference.jpg")]},
        source_counts={"Alice": 5, "Bob": 5},
        match_thresholds={"Alice": 0.32, "Bob": 0.32},
        strict_thresholds={"Alice": 0.27, "Bob": 0.27},
    )

    class AgreeingMatcher:
        def __init__(self):
            self.db = secondary_identity_matcher.SecondaryIdentityDB(
                identities={"Alice": alice, "Bob": bob}
            )

        def verify(self, _crop, expected_person, *, excluded_source=None):
            return secondary_identity_matcher.SecondaryVerification(
                False, expected_person, 0.30, 0.15
            )

        def flush(self):
            return None

    matcher = AgreeingMatcher()
    old_minimum = review_unknown_identities.AUTO_GATE_MIN_CONFIRMED_CASES
    review_unknown_identities.AUTO_GATE_MIN_CONFIRMED_CASES = 1
    try:
        correct_set = tmp / "correct.csv"
        evaluation_enrollment.enroll(
            source=source, person="Alice", content_sha256=sort_photos.sha256_file(source), path=correct_set
        )
        allowed, report = review_unknown_identities.evaluate_automatic_policy_benchmark(
            db,
            sort_photos.CacheState(faces=[face]),
            matcher,
            hard_negatives={},
            evaluation_path=correct_set,
        )
        assert_true(
            allowed and report["accepted"] == 1 and report["incorrect"] == 0,
            f"correct dual-model benchmark case did not pass: {report}",
        )

        incorrect_set = tmp / "incorrect.csv"
        evaluation_enrollment.enroll(
            source=source, person="Bob", content_sha256=sort_photos.sha256_file(source), path=incorrect_set
        )
        allowed, report = review_unknown_identities.evaluate_automatic_policy_benchmark(
            db,
            sort_photos.CacheState(faces=[face]),
            matcher,
            hard_negatives={},
            evaluation_path=incorrect_set,
        )
        assert_true(
            not allowed and report["incorrect"] == 1,
            f"incorrect dual-model benchmark case did not block activation: {report}",
        )
    finally:
        review_unknown_identities.AUTO_GATE_MIN_CONFIRMED_CASES = old_minimum


def test_confirmed_benchmark_relinks_moved_source_by_hash(tmp: Path) -> None:
    people = tmp / "photos_by_person"
    replacement = people / "Alice" / "photos" / "Alice_00001.jpg"
    make_image(replacement)
    digest = sort_photos.sha256_file(replacement)
    benchmark = tmp / "confirmed.csv"
    evaluation_enrollment.enroll(
        source=tmp / "old" / "Alice_00001.jpg",
        person="Alice",
        content_sha256=digest,
        path=benchmark,
    )
    index_path = tmp / "analysis.sqlite3"
    connection = sqlite3.connect(index_path)
    try:
        connection.execute("CREATE TABLE assets(path TEXT, sha256 TEXT)")
        connection.execute(
            "INSERT INTO assets(path, sha256) VALUES(?, ?)",
            (str(replacement), digest),
        )
        connection.commit()
    finally:
        connection.close()
    stats = evaluation_enrollment.relink_missing_sources(
        analysis_index_path=index_path,
        people_root=people,
        path=benchmark,
    )
    validation = evaluation_dataset.load_dataset(benchmark)
    assert_true(
        stats == {"rows": 1, "missing": 1, "relinked": 1, "unresolved": 0}
        and validation.cases[0].source == replacement.resolve(),
        f"moved confirmed benchmark source was not safely relinked: {stats}",
    )


def test_unknown_review_routes_unsupported_files_recoverably(tmp: Path) -> None:
    unknown = tmp / "_source_review" / "unassigned_intake" / "unknown_identity"
    records = []
    for queue_kind, filename in (
        ("multi_face_review", "multi.jpg"),
        ("no_usable_face", "no-face.jpg"),
        ("face_quality_review", "quality.jpg"),
        ("technical_review", "error.jpg"),
    ):
        source = unknown / filename
        make_image(source)
        records.append(review_unknown_identities.UnsupportedUnknown(
            key=review_unknown_identities.item_key(source),
            path=source,
            queue_kind=queue_kind,
            detector_status=queue_kind,
            error="synthetic detector failure" if queue_kind == "technical_review" else "",
        ))

    old_sorted = recover_no_usable_faces.DEFAULT_SORTED
    recover_no_usable_faces.DEFAULT_SORTED = tmp
    try:
        state = {
            "unknown_root": unknown,
            "unassigned_root": unknown.parent,
            "decisions_path": tmp / "decisions.json",
            "cache": sort_photos.CacheState(),
            "cache_dirty": False,
            "temporarily_skipped": set(),
        }
        counts = review_unknown_identities.route_unsupported_unknowns(state, records)
    finally:
        recover_no_usable_faces.DEFAULT_SORTED = old_sorted

    expected_destinations = {
        "multi_face_review": unknown.parent / "multi_face_review" / "multi.jpg",
        "no_usable_face": unknown.parent / "no_usable_face" / "no-face.jpg",
        "face_quality_review": unknown.parent / "face_quality_review" / "quality.jpg",
        "technical_review": unknown.parent / "processing_failed" / "error.jpg",
    }
    assert_true(
        all(counts[kind] == 1 for kind in expected_destinations),
        f"unsupported review routing counts are wrong: {counts}",
    )
    assert_true(
        all(path.is_file() for path in expected_destinations.values())
        and not any(record.path.exists() for record in records),
        "unsupported files were not moved into recoverable specialist queues",
    )
    decisions = review_unknown_identities.load_decisions(state["decisions_path"])["items"]
    assert_true(
        {decisions[record.key]["action"] for record in records} == {
            "routed_multi_face",
            "routed_no_usable_face",
            "routed_face_quality_review",
            "routed_technical_review",
        },
        "specialist queue decisions were not persisted",
    )
    progress = review_unknown_identities.review_progress(state)
    assert_true(
        progress["reviewed"] == 4
        and progress["deferred"] == 4
        and progress["pending"] == 0,
        f"global unsupported-routing progress is wrong: {progress}",
    )


def test_unknown_review_content_identity_ignores_timestamp_only_changes(tmp: Path) -> None:
    source = tmp / "unknown" / "same-content.jpg"
    make_image(source)
    original_key = review_unknown_identities.item_key(source)
    original_digest = review_unknown_identities.item_content_sha256(source)
    stat = source.stat()
    os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 2_000_000_000))
    assert_true(
        review_unknown_identities.item_key(source) == original_key
        and review_unknown_identities.item_content_sha256(source) == original_digest,
        "timestamp-only change replaced a content-aware unknown decision key",
    )
    source.write_bytes(source.read_bytes() + b"changed")
    assert_true(
        review_unknown_identities.item_key(source) != original_key,
        "changed bytes inherited the previous unknown decision key",
    )


def test_unknown_review_reconciles_replays_and_preserves_manual_decisions(tmp: Path) -> None:
    sorted_root = tmp / "sorted"
    people = sorted_root / "photos_by_person"
    unknown = sorted_root / "_source_review" / "unassigned_intake" / "unknown_identity"
    confirmed = unknown / "confirmed-replay.jpg"
    multi = unknown / "multi-replay.jpg"
    retained = unknown / "retained-old-model.jpg"
    unverified = unknown / "missing-destination.jpg"
    organized = people / "Alice" / "photos" / "Alice_00001.jpg"
    make_image(confirmed, (80, 100, 140))
    organized.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(confirmed, organized)
    make_image(multi, (110, 80, 150))
    make_image(retained, (70, 140, 90))
    make_image(unverified, (150, 90, 70))

    decisions_path = tmp / "decisions.json"
    decisions = review_unknown_identities.load_decisions(decisions_path)
    review_unknown_identities.record_path_decision(
        decisions,
        key=review_unknown_identities.item_key(confirmed),
        path=confirmed,
        action="confirmed",
        model_signature="old-model",
        person="Alice",
        destination=str(organized),
    )
    review_unknown_identities.record_path_decision(
        decisions,
        key=review_unknown_identities.item_key(multi),
        path=multi,
        action="routed_multi_face",
        model_signature="old-model",
        destination=str(unknown.parent / "multi_face_review" / multi.name),
    )
    review_unknown_identities.record_path_decision(
        decisions,
        key=review_unknown_identities.item_key(retained),
        path=retained,
        action="keep_unknown",
        model_signature="old-model",
    )
    review_unknown_identities.record_path_decision(
        decisions,
        key=review_unknown_identities.item_key(unverified),
        path=unverified,
        action="confirmed",
        model_signature="old-model",
        person="Alice",
        destination=str(people / "Alice" / "photos" / "missing.jpg"),
    )
    review_unknown_identities.save_decisions(decisions_path, decisions)

    old_sorted = recover_no_usable_faces.DEFAULT_SORTED
    recover_no_usable_faces.DEFAULT_SORTED = sorted_root
    try:
        state = {
            "unknown_root": unknown,
            "unassigned_root": unknown.parent,
            "decisions_path": decisions_path,
            "people_root": people,
            "replay_dir": sorted_root / "_source_review" / "ready_to_delete" / "replays",
            "junk_dir": sorted_root / "_source_review" / "ready_to_delete" / "junk",
            "model_signature": "new-model",
            "cache": sort_photos.CacheState(),
            "cache_dirty": False,
        }
        counts = review_unknown_identities.reconcile_unknown_queue(state)
    finally:
        recover_no_usable_faces.DEFAULT_SORTED = old_sorted

    assert_true(
        counts["reconciled_confirmed"] == 1
        and counts["reconciled_routed_multi_face"] == 1
        and counts["retained_keep_unknown"] == 1
        and counts["silent_recheck_available"] == 1
        and counts["requeued_unverified_confirmation"] == 1
        and not confirmed.exists()
        and organized.is_file()
        and len(list(state["replay_dir"].glob("confirmed-replay*.jpg"))) == 1
        and not multi.exists()
        and len(list((unknown.parent / "multi_face_review").glob("multi-replay*.jpg"))) == 1
        and retained.is_file()
        and unverified.is_file(),
        f"unknown replay reconciliation was unsafe or incomplete: {counts}",
    )
    repaired = review_unknown_identities.load_decisions(decisions_path)
    assert_true(
        review_unknown_identities.resolved_item(
            repaired, retained, model_signature="new-model"
        )
        and review_unknown_identities.needs_silent_recheck(repaired, retained, "new-model")
        and not review_unknown_identities.resolved_item(
            repaired, unverified, model_signature="new-model"
        ),
        "manual choice was reopened or unverified confirmation was silently trusted",
    )


def test_unknown_review_loads_successive_batches_automatically(tmp: Path) -> None:
    unknown = tmp / "unassigned_intake" / "unknown_identity"
    first_path = unknown / "first.jpg"
    second_path = unknown / "second.jpg"
    make_image(first_path)
    make_image(second_path, (91, 120, 180))
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    identity_db = sort_photos.IdentityDB(
        identities={"Alice": embedding},
        prototypes={"Alice": [embedding]},
        source_counts={"Alice": 5},
    )

    def fake_collect(files, identity_db_value, **_kwargs):
        items = []
        for path in files:
            face = sort_photos.CachedFace(
                src_str=str(path), face_index=0, det_score=0.99, bbox_size=120.0,
                sharpness=100.0, yaw_proxy=0.0, quality=0.9,
                embedding=embedding, image_phash=np.zeros(64, dtype=np.uint8),
                crop_jpeg=b"jpeg", label=None,
            )
            candidate = identity_profiles.rank_candidates(
                embedding, identity_db_value.identities, identity_db_value.prototypes
            )[0]
            items.append(review_unknown_identities.UnknownItem(
                review_unknown_identities.item_key(path), path, face, (candidate,)
            ))
        return items, Counter({"sqlite_hits": len(items)}), []

    state = {
        "unknown_root": unknown,
        "unassigned_root": unknown.parent,
        "output_dir": tmp / "reports",
        "session_path": tmp / "reports" / "session.json",
        "decisions_path": tmp / "reports" / "decisions.json",
        "identity_db": identity_db,
        "identity_names": ["Alice"],
        "batch_limit": 1,
        "batch_number": 0,
        "workers": 1,
        "primary_det_size": 320,
        "fallback_det_size": 320,
        "cluster_eps": 0.30,
        "hard_negatives": {},
        "temporarily_skipped": set(),
        "route_unsupported": True,
        "cache": sort_photos.CacheState(),
        "cache_dirty": False,
        "summary": {},
    }
    original_collect = review_unknown_identities.collect_items
    review_unknown_identities.collect_items = fake_collect
    try:
        first_batch = review_unknown_identities.load_next_batch(state)
        first_item = next(iter(state["items_by_key"].values()))
        decisions = review_unknown_identities.load_decisions(state["decisions_path"])
        review_unknown_identities.record_decision(decisions, first_item, "keep_unknown")
        review_unknown_identities.save_decisions(state["decisions_path"], decisions)

        second_batch = review_unknown_identities.load_next_batch(state)
        second_item = next(iter(state["items_by_key"].values()))
        decisions = review_unknown_identities.load_decisions(state["decisions_path"])
        review_unknown_identities.record_decision(decisions, second_item, "keep_unknown")
        review_unknown_identities.save_decisions(state["decisions_path"], decisions)
        completed = review_unknown_identities.load_next_batch(state)
    finally:
        review_unknown_identities.collect_items = original_collect

    assert_true(
        not first_batch["complete"]
        and not second_batch["complete"]
        and first_item.path != second_item.path
        and state["batch_number"] == 2,
        "continuous unknown review did not advance to the next file batch",
    )
    assert_true(
        completed["complete"]
        and completed["summary"]["progress"]["reviewed"] == 2
        and completed["summary"]["progress"]["pending"] == 0,
        f"continuous unknown review did not finish cleanly: {completed}",
    )


def test_unknown_review_batch_loader_is_async_and_reports_progress(tmp: Path) -> None:
    class IdleQueue:
        @staticmethod
        def is_idle() -> bool:
            return True

    started = threading.Event()
    release = threading.Event()
    state = {
        "action_queue": IdleQueue(),
        "lifecycle_lock": threading.Lock(),
        "lock": threading.Lock(),
        "finishing": False,
        "review_finished": False,
        "batch_loading": False,
        "batch_limit": 500,
        "clusters": ["old-cluster"],
        "items_by_key": {"old": "item"},
        "summary": {"batch_number": 1},
        "batch_number": 1,
        "cache_dirty": False,
    }
    original_load = review_unknown_identities.load_next_batch

    def fake_load(working_state: dict) -> dict:
        working_state["batch_progress_callback"](
            200,
            500,
            Counter({"single_face": 150, "multi_face": 50}),
        )
        started.set()
        assert_true(release.wait(2), "synthetic batch loader was not released")
        working_state["clusters"] = ["new-cluster"]
        working_state["items_by_key"] = {"new": "item"}
        working_state["summary"] = {"batch_number": 2}
        working_state["batch_number"] = 2
        return {"complete": False, "summary": working_state["summary"]}

    review_unknown_identities.load_next_batch = fake_load
    try:
        job = review_unknown_identities.BatchLoadJob(state)
        initial, conflict = job.start()
        assert_true(conflict is None, f"batch loader unexpectedly conflicted: {conflict}")
        assert_true(initial["status"] in {"queued", "running"}, "batch loader did not start")
        assert_true(started.wait(2), "batch loader did not report analysis progress")
        snapshot = job.snapshot()
        assert_true(
            snapshot["processed"] == 200
            and snapshot["total"] == 500
            and snapshot["single_face"] == 150
            and snapshot["multi_face"] == 50,
            f"batch loader reported incorrect progress: {snapshot}",
        )
        assert_true(
            state["clusters"] == ["old-cluster"],
            "batch loader exposed a partial batch before completion",
        )
        release.set()
        assert_true(job.worker is not None, "batch loader worker was not created")
        job.worker.join(timeout=3)
        final = job.snapshot()
        assert_true(
            final["status"] == "completed"
            and state["clusters"] == ["new-cluster"]
            and state["batch_number"] == 2
            and not state["batch_loading"],
            f"batch loader did not publish atomically: {final}",
        )
    finally:
        release.set()
        review_unknown_identities.load_next_batch = original_load


def test_unknown_review_finish_runs_safety_gate_once(tmp: Path) -> None:
    unknown = tmp / "unassigned_intake" / "unknown_identity"
    unknown.mkdir(parents=True)
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    identity_db = sort_photos.IdentityDB(
        identities={"Alice": embedding},
        prototypes={"Alice": [embedding]},
        source_counts={"Alice": 5},
    )

    class IdleQueue:
        def __init__(self):
            self.pending = queue.Queue()

    calls = []

    def fake_gate(candidate, incumbent, cache, *, confirmed_set):
        calls.append((candidate, incumbent, cache, confirmed_set))
        return True, {"allowed": True, "failures": []}

    state = {
        "action_queue": IdleQueue(),
        "lock": threading.Lock(),
        "unknown_root": unknown,
        "output_dir": tmp / "reports",
        "session_path": tmp / "reports" / "session.json",
        "decisions_path": tmp / "reports" / "decisions.json",
        "evaluation_path": tmp / "confirmed.csv",
        "people_root": tmp / "photos_by_person",
        "identity_db": identity_db,
        "cache": sort_photos.CacheState(),
        "cache_dirty": False,
        "identity_dirty": False,
        "temporarily_skipped": set(),
        "batch_limit": 500,
        "batch_number": 1,
        "summary": {},
        "finishing": False,
        "review_finished": False,
    }
    original_gate = review_unknown_identities.identity_evaluation.activation_gate
    review_unknown_identities.identity_evaluation.activation_gate = fake_gate
    try:
        finish = review_unknown_identities.FinishReviewJob(state)
        first = finish.start()
        assert_true(first["status"] in {"queued", "running", "completed"},
                    "Finish Review did not start")
        assert_true(finish.worker is not None, "Finish Review worker was not created")
        finish.worker.join(timeout=5)
        completed = finish.snapshot()
        finish.start()
    finally:
        review_unknown_identities.identity_evaluation.activation_gate = original_gate

    report_path = tmp / "reports" / "unknown_review_final_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert_true(
        completed["status"] == "completed"
        and len(calls) == 1
        and report["activation_gate"]["allowed"]
        and report["safety"]["automatic_confirmations"] == 0
        and not report["safety"]["thresholds_lowered"],
        f"Finish Review was not safely gated exactly once: {completed}",
    )


def test_unknown_review_new_person_name_confirms_entire_cluster(tmp: Path) -> None:
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    db = sort_photos.IdentityDB(
        identities={"Alice": embedding},
        prototypes={"Alice": [embedding]},
        source_counts={"Alice": 5},
    )
    people = tmp / "photos_by_person"
    unknown = tmp / "unknown_identity"

    def make_item(filename: str) -> review_unknown_identities.UnknownItem:
        path = unknown / filename
        color = (80, 120, 170) if filename == "new-a.jpg" else (120, 80, 170)
        make_image(path, color)
        face = sort_photos.CachedFace(
            src_str=str(path), face_index=0, det_score=0.99, bbox_size=120.0,
            sharpness=100.0, yaw_proxy=0.0, quality=0.9,
            embedding=embedding, image_phash=np.zeros(64, dtype=np.uint8),
            crop_jpeg=b"jpeg", label=None,
        )
        candidate = identity_profiles.rank_candidates(
            embedding, db.identities, db.prototypes
        )[0]
        return review_unknown_identities.UnknownItem(
            review_unknown_identities.item_key(path), path, face, (candidate,)
        )

    first = make_item("new-a.jpg")
    second = make_item("new-b.jpg")
    old_recovery_values = (
        recover_no_usable_faces.DEFAULT_PEOPLE,
        recover_no_usable_faces.DEFAULT_SORTED,
        recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES,
    )
    old_nudity = sort_photos.NUDITY_SORT_ENABLED
    recover_no_usable_faces.DEFAULT_PEOPLE = people
    recover_no_usable_faces.DEFAULT_SORTED = tmp
    recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES = 0
    sort_photos.NUDITY_SORT_ENABLED = False
    action_queue = None
    try:
        state = {
            "items_by_key": {first.key: first, second.key: second},
            "clusters": [review_unknown_identities.UnknownCluster(
                key="new-cluster", items=(first, second),
                candidates=first.candidates, cohesion=0.99,
            )],
            "decisions_path": tmp / "decisions.json",
            "identity_db": db,
            "identity_names": ["Alice"],
            "people_root": people,
            "existing_hashes": recover_no_usable_faces.load_existing_hashes(people),
            "next_indexes": {},
            "confirmations_path": tmp / "confirmed.json",
            "review_dir": tmp / "ready_to_delete" / "confirmed",
            "analysis_index": tmp / "analysis.sqlite3",
            "cache": sort_photos.CacheState(
                config_fingerprint=sort_photos.config_fingerprint()
            ),
            "cache_dirty": False,
            "identity_dirty": False,
            "lock": threading.Lock(),
        }
        action_queue = review_unknown_identities.ReviewActionQueue(state)
        job = action_queue.submit(
            item_keys=[first.key, second.key],
            action="confirm",
            person_value="New Person",
        )
        action_queue.close(wait=True)
        result = action_queue.snapshot([job["id"]])["jobs"][0]
    finally:
        if action_queue is not None and action_queue.worker.is_alive():
            action_queue.close(wait=True)
        (
            recover_no_usable_faces.DEFAULT_PEOPLE,
            recover_no_usable_faces.DEFAULT_SORTED,
            recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES,
        ) = old_recovery_values
        sort_photos.NUDITY_SORT_ENABLED = old_nudity

    organized = list((people / "New Person" / "photos").glob("*"))
    assert_true(
        result["status"] == "completed"
        and len(organized) == 2
        and not first.path.exists()
        and not second.path.exists(),
        f"new-person cluster confirmation did not organize every image: {result}",
    )
    assert_true(
        state["identity_names"] == ["Alice", "New Person"]
        and state["identity_dirty"],
        "new person was not registered for the live dashboard/profile refresh",
    )


def test_unknown_review_action_queue_serializes_and_deduplicates(tmp: Path) -> None:
    del tmp
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    identity_db = sort_photos.IdentityDB(
        identities={"Alice": embedding},
        prototypes={"Alice": [embedding]},
        source_counts={"Alice": 1},
    )
    started = threading.Event()
    release = threading.Event()
    events: list[str] = []

    def executor(_state: dict, *, item_keys: list[str], action: str,
                 person_value: str = "") -> str:
        events.append(f"start:{item_keys[0]}")
        if item_keys == ["first"]:
            started.set()
            assert_true(release.wait(timeout=2), "queued action test timed out")
        events.append(f"end:{item_keys[0]}")
        return f"{action}:{person_value}:{item_keys[0]}"

    state = {"identity_db": identity_db, "lock": threading.Lock()}
    action_queue = review_unknown_identities.ReviewActionQueue(state, executor)
    try:
        first = action_queue.submit(
            item_keys=["first"], action="confirm", person_value="Alice"
        )
        assert_true(started.wait(timeout=2), "first review action did not start")
        duplicate = action_queue.submit(
            item_keys=["first"], action="confirm", person_value="Alice"
        )
        assert_true(
            duplicate["id"] == first["id"] and duplicate["deduplicated"],
            "identical active review action was queued twice",
        )
        try:
            action_queue.submit(item_keys=["first"], action="ignore")
        except review_unknown_identities.ReviewActionConflict:
            pass
        else:
            raise AssertionError("conflicting active review action was not rejected")
        second = action_queue.submit(item_keys=["second"], action="keep_unknown")
        assert_true(second["position"] == 2, "second review action has wrong queue position")
        release.set()
    finally:
        action_queue.close(wait=True)
    snapshot = action_queue.snapshot([first["id"], second["id"]])
    assert_true(
        [job["status"] for job in snapshot["jobs"]] == ["completed", "completed"],
        f"queued review actions did not complete: {snapshot}",
    )
    assert_true(
        events == ["start:first", "end:first", "start:second", "end:second"],
        f"review actions were not executed serially: {events}",
    )


def test_unknown_review_large_cluster_is_chunked_safely(tmp: Path) -> None:
    del tmp
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    identity_db = sort_photos.IdentityDB(
        identities={"Alice": embedding},
        prototypes={"Alice": [embedding]},
        source_counts={"Alice": 1},
    )
    chunk_sizes = []

    def executor(_state: dict, *, item_keys: list[str], action: str,
                 person_value: str = "") -> str:
        del action, person_value
        chunk_sizes.append(len(item_keys))
        return f"handled {len(item_keys)}"

    action_queue = review_unknown_identities.ReviewActionQueue(
        {"identity_db": identity_db, "lock": threading.Lock()}, executor
    )
    try:
        jobs = action_queue.submit_many(
            item_keys=[f"item-{index}" for index in range(121)],
            action="keep_unknown",
        )
    finally:
        action_queue.close(wait=True)
    snapshot = action_queue.snapshot([job["id"] for job in jobs])
    assert_true(
        len(jobs) == 3
        and chunk_sizes == [50, 50, 21]
        and all(job["status"] == "completed" for job in snapshot["jobs"]),
        f"large cluster was not serialized into safe chunks: {chunk_sizes}",
    )


def test_unknown_review_confirmation_uses_recoverable_pipeline(tmp: Path) -> None:
    sorted_root = tmp / "sorted"
    people = sorted_root / "photos_by_person"
    unknown = sorted_root / "_source_review" / "unassigned_intake" / "unknown_identity"
    source = unknown / "difficult.jpg"
    second_source = unknown / "difficult-profile.jpg"
    make_image(source)
    make_image(second_source, (120, 90, 170))
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    face_record = sort_photos.CachedFace(
        src_str=str(source), face_index=0, det_score=0.99, bbox_size=120.0,
        sharpness=100.0, yaw_proxy=0.0, quality=0.9,
        embedding=embedding, image_phash=np.zeros(64, dtype=np.uint8),
        crop_jpeg=b"jpeg", label=None,
    )
    db = sort_photos.IdentityDB(
        config_fingerprint=sort_photos.config_fingerprint(),
        identities={"Alice": embedding},
        prototypes={"Alice": [embedding]},
        source_counts={"Alice": 5},
    )
    candidate = identity_profiles.rank_candidates(
        embedding, db.identities, db.prototypes
    )[0]
    item = review_unknown_identities.UnknownItem(
        review_unknown_identities.item_key(source),
        source,
        face_record,
        (candidate,),
    )
    second_face = sort_photos.CachedFace(
        src_str=str(second_source), face_index=0, det_score=0.98, bbox_size=115.0,
        sharpness=90.0, yaw_proxy=0.2, quality=0.85,
        embedding=embedding, image_phash=np.zeros(64, dtype=np.uint8),
        crop_jpeg=b"jpeg-2", label=None,
    )
    second_item = review_unknown_identities.UnknownItem(
        review_unknown_identities.item_key(second_source),
        second_source,
        second_face,
        (candidate,),
    )
    old_recovery_values = (
        recover_no_usable_faces.DEFAULT_PEOPLE,
        recover_no_usable_faces.DEFAULT_SORTED,
        recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES,
    )
    old_nudity = sort_photos.NUDITY_SORT_ENABLED
    recover_no_usable_faces.DEFAULT_PEOPLE = people
    recover_no_usable_faces.DEFAULT_SORTED = sorted_root
    recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES = 0
    sort_photos.NUDITY_SORT_ENABLED = False
    try:
        confirmations_path = tmp / "confirmed.json"
        state = {
            "items_by_key": {item.key: item, second_item.key: second_item},
            "decisions_path": tmp / "decisions.json",
            "identity_db": db,
            "existing_hashes": {"Alice": set()},
            "next_indexes": {},
            "confirmations_path": confirmations_path,
            "review_dir": sorted_root / "_source_review" / "ready_to_delete" / "confirmed",
            "analysis_index": tmp / "analysis.sqlite3",
            "cache": sort_photos.CacheState(
                config_fingerprint=sort_photos.config_fingerprint()
            ),
            "cache_dirty": False,
            "identity_dirty": False,
        }
        message = review_unknown_identities.apply_decision(
            state,
            item_keys=[item.key, second_item.key],
            action="confirm",
            person_value="Alice",
        )
        confirmed_paths = identity_confirmations.paths_for_person(
            confirmations_path, "Alice", people
        )
        assert_true("Confirmed 2" in message and len(confirmed_paths) == 2,
                    "multi-image unknown confirmation was not organized and registered")
        assert_true(not source.exists() and not second_source.exists()
                    and all(path.is_file() for path in confirmed_paths),
                    "multi-image unknown sources were not archived after verified copies")
        assert_true(state["cache_dirty"] and state["identity_dirty"],
                    "confirmation did not request cache/profile persistence")
        archived = list(state["review_dir"].rglob("difficult*.jpg"))
        assert_true(len(archived) == 2, "unknown sources were not moved recoverably")
    finally:
        (
            recover_no_usable_faces.DEFAULT_PEOPLE,
            recover_no_usable_faces.DEFAULT_SORTED,
            recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES,
        ) = old_recovery_values
        sort_photos.NUDITY_SORT_ENABLED = old_nudity


def test_unknown_review_exact_duplicates_share_review_destination(tmp: Path) -> None:
    sorted_root = tmp / "sorted"
    people = sorted_root / "photos_by_person"
    unknown = sorted_root / "_source_review" / "unassigned_intake" / "unknown_identity"
    first_source = unknown / "duplicate-a.jpg"
    second_source = unknown / "duplicate-b.jpg"
    make_image(first_source, (90, 120, 160))
    second_source.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(first_source, second_source)

    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0
    db = sort_photos.IdentityDB(
        config_fingerprint=sort_photos.config_fingerprint(),
        identities={"Alice": embedding},
        prototypes={"Alice": [embedding]},
        source_counts={"Alice": 5},
    )
    candidate = identity_profiles.rank_candidates(
        embedding, db.identities, db.prototypes
    )[0]

    def make_item(path: Path) -> review_unknown_identities.UnknownItem:
        face = sort_photos.CachedFace(
            src_str=str(path), face_index=0, det_score=0.99, bbox_size=120.0,
            sharpness=100.0, yaw_proxy=0.0, quality=0.9,
            embedding=embedding, image_phash=np.zeros(64, dtype=np.uint8),
            crop_jpeg=b"jpeg", label=None,
        )
        return review_unknown_identities.UnknownItem(
            review_unknown_identities.item_key(path), path, face, (candidate,)
        )

    first_item = make_item(first_source)
    second_item = make_item(second_source)
    old_recovery_values = (
        recover_no_usable_faces.DEFAULT_PEOPLE,
        recover_no_usable_faces.DEFAULT_SORTED,
        recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES,
    )
    original_nudity_router = sort_photos.maybe_move_to_nudity_subfolder

    def route_to_uncertain(path: Path, person_dir: Path):
        destination = sort_photos.unique_dest(
            person_dir / sort_photos.NUDITY_UNCERTAIN_DIR,
            path.name,
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.replace(destination)
        return destination, sort_photos.NUDITY_UNCERTAIN_DIR

    recover_no_usable_faces.DEFAULT_PEOPLE = people
    recover_no_usable_faces.DEFAULT_SORTED = sorted_root
    recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES = 0
    sort_photos.maybe_move_to_nudity_subfolder = route_to_uncertain
    try:
        decisions_path = tmp / "decisions.json"
        state = {
            "items_by_key": {
                first_item.key: first_item,
                second_item.key: second_item,
            },
            "decisions_path": decisions_path,
            "identity_db": db,
            "people_root": people,
            "existing_hashes": {"Alice": set()},
            "destinations_by_hash": {},
            "next_indexes": {},
            "confirmations_path": tmp / "confirmed.json",
            "review_dir": sorted_root / "_source_review" / "ready_to_delete" / "confirmed",
            "analysis_index": tmp / "analysis.sqlite3",
            "cache": sort_photos.CacheState(
                config_fingerprint=sort_photos.config_fingerprint()
            ),
            "cache_dirty": False,
            "identity_dirty": False,
        }
        message = review_unknown_identities.apply_decision(
            state,
            item_keys=[first_item.key, second_item.key],
            action="confirm",
            person_value="Alice",
        )
        decisions = review_unknown_identities.load_decisions(decisions_path)["items"]
        destinations = {
            decisions[first_item.key]["destination"],
            decisions[second_item.key]["destination"],
        }
        destination = Path(next(iter(destinations)))
        assert_true(
            "Confirmed 2" in message and len(destinations) == 1,
            "exact duplicate confirmations did not reuse one verified destination",
        )
        assert_true(
            destination.is_file()
            and destination.parent
            == (people / "Alice" / "review" / "uncertain_nudity").resolve(),
            f"duplicate review destination was unavailable or misplaced: {destination}",
        )
        assert_true(
            not first_source.exists() and not second_source.exists(),
            "duplicate unknown sources were not archived after confirmation",
        )
    finally:
        (
            recover_no_usable_faces.DEFAULT_PEOPLE,
            recover_no_usable_faces.DEFAULT_SORTED,
            recover_no_usable_faces.MIN_INTERNAL_FREE_BYTES,
        ) = old_recovery_values
        sort_photos.maybe_move_to_nudity_subfolder = original_nudity_router


def test_impostor_aware_thresholds_tighten_lookalikes(tmp: Path) -> None:
    del tmp
    alice = np.zeros(512, dtype=np.float32)
    alice[0] = 1.0
    lookalike = np.zeros(512, dtype=np.float32)
    lookalike[0] = 0.98
    lookalike[1] = 0.20
    lookalike /= np.linalg.norm(lookalike)
    consensus, strict, nearest = identity_profiles.impostor_aware_thresholds(
        "Alice",
        {"Alice": alice, "Lookalike": lookalike},
        {"Alice": [alice], "Lookalike": [lookalike]},
        base_consensus_distance=0.40,
        base_strict_distance=0.27,
    )
    assert_true(nearest < 0.03, f"closest impostor was not detected: {nearest}")
    assert_true(consensus < nearest and strict <= consensus,
                f"lookalike thresholds were not safely tightened: {(consensus, strict, nearest)}")

    bob = np.zeros(512, dtype=np.float32)
    bob[1] = 1.0
    consensus, strict, _nearest = identity_profiles.impostor_aware_thresholds(
        "Alice",
        {"Alice": alice, "Bob": bob},
        {"Alice": [alice], "Bob": [bob]},
        base_consensus_distance=0.40,
        base_strict_distance=0.27,
    )
    assert_true(abs(consensus - 0.40) < 1e-6 and abs(strict - 0.27) < 1e-6,
                "distant identity unnecessarily tightened thresholds")


def test_legacy_identity_db_is_upgraded_in_memory(tmp: Path) -> None:
    del tmp
    centroid = np.zeros(512, dtype=np.float32)
    centroid[0] = 1.0
    db = sort_photos.IdentityDB(identities={"Alice": centroid}, source_counts={"Alice": 4})
    del db.prototypes
    del db.match_thresholds
    del db.strict_thresholds
    del db.intrinsic_match_thresholds
    del db.intrinsic_strict_thresholds
    del db.nearest_impostor_distances
    del db.source_signatures
    del db.calibration_version
    db.version = 1
    upgraded = sort_photos.normalize_identity_db(db)
    assert_true(upgraded.version == sort_photos.IDENTITY_DB_VERSION,
                "legacy identity DB version was not upgraded")
    assert_true(len(upgraded.prototypes["Alice"]) == 1,
                "legacy centroid was not retained as a prototype")
    assert_true("Alice" in upgraded.match_thresholds and "Alice" in upgraded.strict_thresholds,
                "legacy identity DB did not receive safe default thresholds")
    assert_true(hasattr(upgraded, "nearest_impostor_distances"),
                "legacy identity DB did not receive calibration evidence storage")


def test_identity_evaluation_reports_correct_and_rejected(tmp: Path) -> None:
    del tmp
    alice = np.zeros(512, dtype=np.float32)
    bob = np.zeros(512, dtype=np.float32)
    alice[0] = 1.0
    bob[1] = 1.0
    db = sort_photos.normalize_identity_db(sort_photos.IdentityDB(
        identities={"Alice": alice, "Bob": bob},
        source_counts={"Alice": 5, "Bob": 5},
        prototype_sources={"Alice": ["independent-alice-ref.jpg"], "Bob": ["independent-bob-ref.jpg"]},
    ))

    def cached(source: str, embedding: np.ndarray, label: str) -> sort_photos.CachedFace:
        return sort_photos.CachedFace(
            src_str=source,
            face_index=0,
            det_score=0.99,
            bbox_size=180.0,
            sharpness=160.0,
            yaw_proxy=1.0,
            quality=1.0,
            embedding=embedding,
            image_phash=np.zeros(64, dtype=bool),
            crop_jpeg=b"",
            label=label,
        )

    correct = identity_evaluation.evaluate_face(cached("alice.jpg", alice, "Alice"), db, lane="strict")
    assert_true(correct is not None and correct.outcome == "correct",
                f"exact identity evaluation was not correct: {correct}")
    ambiguous_embedding = (alice + bob) / np.linalg.norm(alice + bob)
    rejected = identity_evaluation.evaluate_face(
        cached("ambiguous.jpg", ambiguous_embedding, "Alice"), db, lane="strict")
    assert_true(rejected is not None and rejected.outcome == "rejected",
                f"ambiguous identity evaluation was not rejected: {rejected}")

    dominant_faces = [
        cached("alice-1.jpg", alice, "Alice"),
        cached("alice-2.jpg", alice, "Alice"),
        cached("alice-3.jpg", alice, "Alice"),
        cached("group-bystander.jpg", bob, "Alice"),
    ]
    selected = identity_evaluation.select_faces(dominant_faces, max_per_person=10)
    assert_true(
        len(selected) == 3 and all(np.allclose(face.embedding, alice) for face in selected),
        "identity evaluator included a disconnected group-photo bystander",
    )


def test_protected_evaluation_set_requires_verified_coverage(tmp: Path) -> None:
    known = tmp / "known.jpg"
    unknown = tmp / "unknown.jpg"
    no_face = tmp / "no-face.jpg"
    for path in (known, unknown, no_face):
        make_image(path)
    dataset = tmp / "golden.csv"
    evaluation_dataset.write_template(dataset, [
        {
            "source": str(known),
            "expected_person": "Alice",
            "case_types": "known|lookalike|side_profile|blurry_small|group|normal|swimwear",
            "expected_face": "true",
            "expected_nudity": "safe",
            "verified": "true",
            "notes": "verified known case",
            "expected_people": "Alice|Bob",
            "expected_face_count": "2",
        },
        {
            "source": str(unknown),
            "expected_person": "",
            "case_types": "unknown|nude",
            "expected_face": "true",
            "expected_nudity": "possible",
            "verified": "true",
            "notes": "verified unknown case",
        },
        {
            "source": str(no_face),
            "expected_person": "",
            "case_types": "no_face",
            "expected_face": "false",
            "expected_nudity": "unknown",
            "verified": "true",
            "notes": "verified no-face case",
        },
    ])
    validation = evaluation_dataset.load_dataset(dataset)
    assert_true(validation.activation_ready,
                f"complete verified golden set was not activation-ready: {validation.errors}")

    baseline = evaluation_dataset.EvaluationMetrics(
        identity_precision=0.99,
        known_case_recall=0.95,
        unknown_rejection_rate=0.98,
        missed_face_rate=0.02,
        no_face_specificity=0.99,
        nudity_accuracy=0.98,
        nudity_false_positive_rate=0.01,
        verified_cases=3,
    )
    regressed = evaluation_dataset.EvaluationMetrics(
        identity_precision=0.95,
        known_case_recall=0.90,
        unknown_rejection_rate=0.90,
        missed_face_rate=0.08,
        no_face_specificity=0.95,
        nudity_accuracy=0.92,
        nudity_false_positive_rate=0.08,
        verified_cases=3,
    )
    failures = evaluation_dataset.compare_to_baseline(regressed, baseline)
    assert_true(len(failures) >= 5, f"metric regressions did not block activation: {failures}")

    text = dataset.read_text(encoding="utf-8").replace(
        "safe,true,verified known case",
        "safe,false,verified known case",
        1,
    )
    dataset.write_text(text, encoding="utf-8")
    invalid = evaluation_dataset.load_dataset(dataset)
    assert_true(not invalid.activation_ready and invalid.errors,
                "unverified golden-set row did not block activation")


def test_original_copy_is_verified_before_source_archive(tmp: Path) -> None:
    source = tmp / "source.jpg"
    destination = tmp / "person" / "photo.jpg"
    make_image(source, (20, 80, 160))
    sort_photos._atomic_copy(source, destination)
    expected = sort_photos.sha256_file(source)
    actual = sort_photos.verify_original_copy(source, destination, expected)
    assert_true(actual == expected and destination.exists(), "valid original copy failed verification")

    destination.write_bytes(b"corrupted copy")
    try:
        sort_photos.verify_original_copy(source, destination, expected)
    except OSError:
        pass
    else:
        raise SyntheticFailure("corrupted original copy incorrectly passed verification")
    assert_true(source.exists(), "copy verification removed the source image")
    assert_true(not destination.exists(), "unverified destination was left in the person library")


def test_daily_order_is_safe(tmp: Path) -> None:
    del tmp
    names = [step["name"] for step in daily_runner.step_list(50)]
    index = {name: i for i, name in enumerate(names)}
    required = [
        ("video-process", "process"),
        ("structure", "cache-rehydrate"),
        ("rename", "cache-rehydrate"),
        ("exact-dedupe", "cache-rehydrate"),
        ("advanced-dedupe", "cache-rehydrate"),
        ("cleanup-empty", "cache-rehydrate"),
        ("cache-rehydrate", "integration-audit"),
        ("integration-audit", "status"),
    ]
    for before, after in required:
        assert_true(before in index, f"daily step missing: {before}")
        assert_true(after in index, f"daily step missing: {after}")
        assert_true(index[before] < index[after], f"{before} must run before {after}")


def test_video_person_decisions_accept_one_recognized_frame(tmp: Path) -> None:
    del tmp
    matched = sort_videos.decide_person(
        Counter({"Alice": 6}),
        sampled_frames=10,
        face_frames=8,
        detected_faces=8,
        confident_matches=6,
    )
    assert_true(matched.status == "matched" and matched.person == "Alice",
                "repeated dominant video identity was not accepted")

    multiple = sort_videos.decide_person(
        Counter({"Alice": 6, "Bob": 4}),
        sampled_frames=10,
        face_frames=8,
        detected_faces=12,
        confident_matches=10,
    )
    assert_true(multiple.status == "multiple_people" and not multiple.person,
                "multi-person video was incorrectly assigned to one person")

    single = sort_videos.decide_person(
        Counter({"Alice": 1}),
        sampled_frames=10,
        face_frames=8,
        detected_faces=8,
        confident_matches=1,
    )
    assert_true(single.status == "matched" and single.person == "Alice",
                "one strict recognized frame was not accepted")

    no_face = sort_videos.decide_person(
        Counter(),
        sampled_frames=8,
        face_frames=0,
        detected_faces=0,
        confident_matches=0,
    )
    assert_true(no_face.status == "no_usable_face",
                "no-face video did not enter the review bucket")


def test_video_support_consensus_recovers_difficult_pose(tmp: Path) -> None:
    del tmp
    recovered = sort_videos.decide_person(
        Counter({"Alice": 1}),
        support_votes=Counter({"Alice": 6}),
        sampled_frames=10,
        face_frames=8,
        detected_faces=8,
        confident_matches=1,
    )
    assert_true(
        recovered.status == "matched" and recovered.person == "Alice",
        "repeated unambiguous support did not recover a difficult-pose video",
    )

    single = sort_videos.decide_person(
        Counter(),
        support_votes=Counter({"Alice": 1}),
        sampled_frames=10,
        face_frames=8,
        detected_faces=8,
        confident_matches=0,
    )
    assert_true(
        single.status == "matched" and single.person == "Alice",
        "one supporting recognized frame was not accepted",
    )

    conflicting = sort_videos.decide_person(
        Counter({"Alice": 1}),
        support_votes=Counter({"Bob": 6}),
        sampled_frames=10,
        face_frames=8,
        detected_faces=8,
        confident_matches=1,
    )
    assert_true(
        conflicting.status == "multiple_people",
        "conflicting strict and supporting identities were not held for review",
    )


def test_video_fallback_model_is_packaged(tmp: Path) -> None:
    del tmp
    assert_true(sort_videos.YUNET_MODEL.is_file(), "YuNet fallback model is missing")
    assert_true(
        sort_videos.YUNET_MODEL.stat().st_size > 200_000,
        "YuNet fallback model is unexpectedly truncated",
    )


def test_video_workers_are_memory_bounded(tmp: Path) -> None:
    del tmp
    paths = [Path(f"video-{index}.mov") for index in range(14)]
    batches = video_batch_runner.chunks(paths, 6)
    assert_true([len(batch) for batch in batches] == [6, 6, 2], "video worker chunking is incorrect")
    video_step = next(step for step in daily_runner.step_list(50) if step["name"] == "video-process")
    assert_true(
        Path(str(video_step["cmd"][1])).name == "video_batch_runner.py",
        "daily video analysis bypasses the memory-bounded supervisor",
    )


def test_image_workers_are_memory_bounded_and_resumable(tmp: Path) -> None:
    intake = tmp / "To Process"
    make_image(intake / "z-last.jpg")
    make_image(intake / "a-first.jpg")
    make_image(intake / "nested" / "m-middle.png")
    make_image(intake / ".hidden" / "ignored.jpg")
    (intake / "notes.txt").write_text("not an image", encoding="utf-8")

    assert_true(image_batch_runner.count_images(intake) == 3,
                "image supervisor count did not match visible intake images")
    selected = sort_photos.bounded_input_images(
        [intake / "z-last.jpg", intake / "nested" / "m-middle.png", intake / "a-first.jpg"],
        2,
    )
    assert_true(
        [path.name for path in selected] == ["a-first.jpg", "m-middle.png"],
        "bounded input selection is not deterministic",
    )

    process_step = next(
        step for step in daily_runner.step_list(50) if step["name"] == "process"
    )
    process_command = [str(part) for part in process_step["cmd"]]
    assert_true(
        Path(process_command[1]).name == "image_batch_runner.py",
        "daily image intake bypasses the memory-bounded supervisor",
    )
    assert_true(
        "--max-images-per-process" in process_command,
        "daily image supervisor has no per-process input limit",
    )

    worker_command = image_batch_runner.build_sort_command(
        intake,
        tmp / "sorted",
        max_images=123,
        detection_batch_size=17,
        detect_workers=1,
    )
    max_index = worker_command.index("--max-input-images")
    assert_true(worker_command[max_index + 1] == "123",
                "image worker did not forward its bounded input size")
    assert_true("--archive-scanned-sources" in worker_command,
                "image worker would not remove completed files from the inbox")


def test_image_supervisor_repeats_until_inbox_is_drained(tmp: Path) -> None:
    intake = tmp / "intake"
    output = tmp / "sorted"
    for index in range(5):
        make_image(intake / f"image-{index}.jpg")

    old_run = image_batch_runner.subprocess.run
    calls = 0

    class FakeResult:
        returncode = 0

    def fake_run(_command, check=False):  # noqa: ANN001
        nonlocal calls
        del check
        calls += 1
        visible = sorted(intake.glob("*.jpg"))
        for path in visible[:2]:
            path.unlink()
        return FakeResult()

    image_batch_runner.subprocess.run = fake_run
    try:
        result = image_batch_runner.run_batches(
            intake,
            output,
            max_images=2,
            detection_batch_size=1,
            detect_workers=1,
        )
    finally:
        image_batch_runner.subprocess.run = old_run

    assert_true(result == 0, "image supervisor did not complete a draining inbox")
    assert_true(calls == 3, f"image supervisor used {calls} workers instead of 3")
    assert_true(image_batch_runner.count_images(intake) == 0,
                "image supervisor left completed files in the inbox")


def test_image_supervisor_stops_when_worker_makes_no_progress(tmp: Path) -> None:
    intake = tmp / "intake"
    make_image(intake / "blocked.jpg")
    old_run = image_batch_runner.subprocess.run

    class FakeResult:
        returncode = 0

    image_batch_runner.subprocess.run = lambda _command, check=False: FakeResult()
    try:
        result = image_batch_runner.run_batches(
            intake,
            tmp / "sorted",
            max_images=1,
            detection_batch_size=1,
            detect_workers=1,
        )
    finally:
        image_batch_runner.subprocess.run = old_run

    assert_true(result == 5, "image supervisor did not stop a no-progress loop")
    assert_true((intake / "blocked.jpg").exists(),
                "no-progress handling modified the blocked source")


def test_video_destinations_keep_matches_and_review_separate(tmp: Path) -> None:
    source = tmp / "intake" / "clip.mov"
    people = tmp / "people"
    review = tmp / "review"
    matched = sort_videos.VideoDecision(
        "matched", "Alice", "test", {"Alice": 4}, 8, 5, 5, 4,
    )
    matched_dest = sort_videos.destination_for(
        source, matched, people_dir=people, review_root=review,
    )
    assert_true(matched_dest == people / "Alice" / "videos" / "clip.mov",
                f"matched video destination is incorrect: {matched_dest}")

    ambiguous = sort_videos.VideoDecision(
        "multiple_people", "", "test", {"Alice": 4, "Bob": 4}, 8, 6, 8, 8,
    )
    review_dest = sort_videos.destination_for(
        source, ambiguous, people_dir=people, review_root=review,
    )
    assert_true(review_dest == review / "multiple_people" / "clip.mov",
                f"multi-person video destination is incorrect: {review_dest}")


def test_video_low_space_pauses_only_matched_copy(tmp: Path) -> None:
    destination = tmp / "people" / "Alice" / "videos" / "clip.mov"
    matched = sort_videos.VideoDecision(
        "matched", "Alice", "test", {"Alice": 4}, 8, 5, 5, 4,
    )
    unknown = sort_videos.VideoDecision(
        "unknown_identity", "", "test", {}, 8, 5, 5, 0,
    )
    no_capacity = lambda _destination, _size: False
    assert_true(
        sort_videos.should_pause_for_space(
            matched, destination, 100, capacity_check=no_capacity,
        ),
        "matched video did not pause before crossing the disk reserve",
    )
    assert_true(
        not sort_videos.should_pause_for_space(
            unknown, destination, 100, capacity_check=no_capacity,
        ),
        "review-only video unnecessarily triggered the internal-disk pause",
    )


def test_video_frame_sampling_matches_known_identity(tmp: Path) -> None:
    video_path = tmp / "known-person.avi"
    writer = sort_videos.cv2.VideoWriter(
        str(video_path),
        sort_videos.cv2.VideoWriter_fourcc(*"MJPG"),
        5.0,
        (320, 240),
    )
    assert_true(writer.isOpened(), "synthetic video writer could not open")
    for value in range(30):
        frame = np.full((240, 320, 3), value * 3, dtype=np.uint8)
        writer.write(frame)
    writer.release()

    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0

    class FakeApp:
        def get(self, _frame):
            return [SimpleNamespace(
                det_score=0.99,
                bbox=np.asarray([40, 40, 150, 170], dtype=np.float32),
                normed_embedding=embedding,
            )]

    decision = sort_videos.analyze_video(
        video_path,
        FakeApp(),
        ["Alice"],
        np.stack([embedding]),
        max_samples=8,
    )
    assert_true(decision.status == "matched" and decision.person == "Alice",
                f"sampled video did not match the repeated known face: {decision}")


def test_daily_commands_are_non_destructive_for_duplicates(tmp: Path) -> None:
    del tmp
    for step in daily_runner.step_list(50):
        cmd = [str(part) for part in step.get("cmd", [])]
        script = Path(cmd[1]).name if len(cmd) > 1 and cmd[1].endswith(".py") else ""
        if step["name"] == "process":
            assert_true("--skip-output-cleanup" in cmd, "daily process must skip sort_photos automatic cleanup")
        if step["name"] == "rename":
            assert_true("--simple" in cmd, "daily rename must use simple filename mode")
        assert_true(step["name"] != "smart-albums",
                    "daily must not rebuild smart albums; smart folders are disabled for now")
        if script in {"delete_person_folder_duplicates.py", "advanced_duplicate_matching.py"}:
            assert_true("--apply" not in cmd, f"daily {step['name']} must not apply duplicate moves")
            assert_true("--quarantine-bad" not in cmd, f"daily {step['name']} must not quarantine in duplicate scan")


def test_preflight_names_external_review_and_ignores_own_launcher(tmp: Path) -> None:
    del tmp
    processes = """
1 0 /sbin/launchd
10 1 python face.py
11 10 /tmp/venv/bin/python /repo/review_unknown_identities.py --serve --open
98 1 python face.py daily
99 98 /tmp/venv/bin/python /repo/daily_runner.py
100 99 /tmp/venv/bin/python /repo/preflight_check.py
200 1 /bin/zsh -c echo face.py
201 1 caffeinate -dimsu /tmp/venv/bin/python /repo/face.py daily --resume
"""
    conflicts = preflight_check.process_conflicts(processes, current_pid=100)
    assert_true(
        len(conflicts) == 1
        and conflicts[0].startswith("11: Unknown Identity Quick Review")
        and "10: Face command" not in conflicts[0],
        f"preflight did not identify the external review worker cleanly: {conflicts}",
    )


def test_daily_empty_inbox_gate_is_fast_and_media_aware(tmp: Path) -> None:
    inbox = tmp / "To Process"
    legacy_videos = tmp / "videos"
    inbox.mkdir()
    legacy_videos.mkdir()
    (inbox / "notes.txt").write_text("not media", encoding="utf-8")
    assert_true(
        not daily_runner.intake_has_media(inbox, legacy_videos),
        "non-media files should not trigger a daily library scan",
    )
    nested = inbox / "new batch"
    nested.mkdir()
    (nested / "photo.JPG").write_bytes(b"test")
    assert_true(
        daily_runner.intake_has_media(inbox, legacy_videos),
        "a nested supported image should trigger daily ingest",
    )
    (nested / "photo.JPG").unlink()
    (legacy_videos / "clip.MOV").write_bytes(b"test")
    assert_true(
        daily_runner.intake_has_media(inbox, legacy_videos),
        "a legacy supported video should trigger daily ingest",
    )


def test_face_main_menu_is_streamlined(tmp: Path) -> None:
    del tmp
    import face

    visible = [key for _heading, keys in face.MENU_GROUPS for key in keys]
    assert_true(
        visible == [
            "daily", "dry-run", "status", "health",
            "review-dashboard", "unknown-review", "benchmark-review", "cross-person-audit", "confirm-unknown",
            "recover-unknown", "recover-no-face", "recover-videos", "nudity",
        ],
        f"unexpected main menu actions: {visible}",
    )
    for key in (
        "review", "finish", "duplicate-review", "nudity-audit",
        "scrap-smart-albums", "repair", "integration-audit",
    ):
        assert_true(key in face.ADVANCED_MENU_KEYS, f"{key} should remain advanced")
    for key in ("dry-run", "status", "health", "review-dashboard"):
        action = face.find_action_by_key(key)
        assert_true(bool(action and action.get("read_only")), f"{key} should use the fast read-only path")
    cross_person = face.find_action_by_key("cross-person-audit")
    assert_true(bool(cross_person and cross_person.get("allow_original_count_decrease")),
                "cross-person review should allow explicit recoverable membership moves")
    duplicate_review = face.find_action_by_key("duplicate-review")
    duplicate_steps = [step.get("script") for step in (duplicate_review or {}).get("steps", [])]
    assert_true(
        duplicate_steps == ["advanced_duplicate_matching.py", "near_visual_review.py"],
        "duplicate review must refresh its report before opening the action server",
    )

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "sort_photos.py"), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert_true(proc.returncode == 0, f"sort_photos --help failed: {proc.stdout[-500:]}")
    assert_true("--skip-output-cleanup" in proc.stdout, "sort_photos does not accept --skip-output-cleanup")
    assert_true("--max-input-images" in proc.stdout, "sort_photos does not accept bounded intake slices")


def test_sort_post_process_duplicate_steps_are_report_only(tmp: Path) -> None:
    out = tmp / "sorted"
    (out / "photos_by_person").mkdir(parents=True)
    commands: list[list[str]] = []
    old_run = sort_photos.subprocess.run

    class FakeResult:
        returncode = 0

    def fake_run(cmd, check=False):  # noqa: ANN001
        del check
        commands.append([str(part) for part in cmd])
        return FakeResult()

    sort_photos.subprocess.run = fake_run
    try:
        sort_photos.run_post_process(out)
    finally:
        sort_photos.subprocess.run = old_run

    duplicate_commands = [
        cmd for cmd in commands
        if len(cmd) > 1 and Path(cmd[1]).name in {
            "delete_person_folder_duplicates.py",
            "advanced_duplicate_matching.py",
        }
    ]
    assert_true(len(duplicate_commands) == 2, f"expected 2 duplicate report commands, got {duplicate_commands}")
    for cmd in duplicate_commands:
        assert_true("--apply" not in cmd, f"post-process duplicate command must not apply: {cmd}")
        assert_true("--quarantine-bad" not in cmd, f"post-process duplicate command must not quarantine: {cmd}")
    rename_commands = [cmd for cmd in commands if len(cmd) > 1 and Path(cmd[1]).name == "rename_person_folder_files.py"]
    assert_true(len(rename_commands) == 1, f"expected one rename command, got {rename_commands}")
    assert_true("--simple" in rename_commands[0], f"post-process rename must use simple mode: {rename_commands[0]}")


def test_simple_rename_plan_preserves_indices_and_flattens(tmp: Path) -> None:
    person = tmp / "people" / "Person Name"
    make_image(person / "photos" / "Person_Name_0001_photo_portrait_q_high.jpg", (20, 120, 210))
    make_image(person / "photos" / "GIFs" / "Person_Name_0002_photo_square_q_review.gif",
               (210, 80, 20), fmt="GIF")
    make_image(person / "photos" / "nude" / "Person_Name_0003_nudity_possible_portrait_q_good.png",
               (80, 210, 20))
    make_image(person / "review" / "duplicates" / "Person_Name_0004_photo_portrait_q_good.jpg",
               (20, 210, 120))

    actions = rename_person_folder_files.plan_for_person_simple(person)
    destinations = {
        Path(action["destination"]).relative_to(person).as_posix()
        for action in actions
    }

    assert_true("photos/Person_Name_00001.jpg" in destinations,
                f"simple rename did not remove quality/orientation tokens: {destinations}")
    assert_true("photos/Person_Name_00002.gif" in destinations,
                f"nested GIF was not flattened into photos/: {destinations}")
    assert_true("photos/nude/Person_Name_00003.png" in destinations,
                f"nude image was not kept under photos/nude: {destinations}")
    assert_true(all("review/" not in str(action["source"]) for action in actions),
                f"simple rename should not touch review files: {actions}")


def test_simple_repair_assigns_unique_person_wide_indices(tmp: Path) -> None:
    person = tmp / "people" / "Person Name"
    make_image(person / "photos" / "Person_Name_00001.jpg", (20, 120, 210))
    make_image(person / "photos" / "Person_Name_00001_copy2.png", (210, 80, 20))
    make_image(person / "photos" / "nude" / "Person_Name_001.gif", (80, 210, 20), fmt="GIF")
    make_image(person / "photos" / "Person_Name_00002.jpg", (120, 20, 210))
    make_image(person / "photos" / "Person_Name_005.png", (20, 210, 120))

    actions = rename_person_folder_files.plan_for_person_simple(
        person,
        repair_numbering=True,
    )
    destinations = [Path(action["destination"]) for action in actions]
    destination_stems = [path.stem for path in destinations]

    assert_true(len(destination_stems) == len(set(destination_stems)),
                f"repair produced duplicate destination names: {destination_stems}")
    assert_true(all("_copy" not in stem.casefold() for stem in destination_stems),
                f"repair retained copy suffixes: {destination_stems}")
    assert_true(any(stem == "Person_Name_00005" for stem in destination_stems),
                f"repair did not normalize a unique three-digit name: {destination_stems}")
    reassigned = [action for action in actions
                  if action["index_source"] == "reassigned_duplicate_index"]
    assert_true(len(reassigned) == 2,
                f"expected two reused-index entries to be reassigned: {actions}")


def test_new_output_numbering_is_person_wide_and_five_digits(tmp: Path) -> None:
    person = tmp / "people" / "Person Name"
    photos = person / "photos"
    nude = photos / "nude"
    make_image(photos / "Person_Name_00005.jpg", (20, 120, 210))
    make_image(nude / "Person_Name_00009.png", (210, 80, 20))
    incoming = tmp / "incoming.jpg"
    make_image(incoming, (80, 210, 20))

    state: dict[Path, int] = {}
    first = sort_photos.next_numbered_dest(photos, person, "Person Name", incoming, state)
    second = sort_photos.next_numbered_dest(nude, person, "Person Name", incoming, state)

    assert_true(first.name == "Person_Name_00010.jpg", f"unexpected first name: {first}")
    assert_true(second.name == "Person_Name_00011.jpg", f"unexpected second name: {second}")


def test_rename_plan_relink_preserves_unchanged_cache_entries(tmp: Path) -> None:
    old_path = tmp / "people" / "Person" / "photos" / "Person_00001.jpg"
    unchanged = tmp / "people" / "Person" / "photos" / "Person_00002.jpg"
    make_image(old_path, (20, 120, 210))
    make_image(unchanged, (210, 80, 20))
    old_sig = sort_photos.file_signature(old_path)
    unchanged_sig = sort_photos.file_signature(unchanged)
    new_path = old_path.with_name("Person_00003.jpg")
    old_path.rename(new_path)

    cache = sort_photos.CacheState(
        version=sort_photos.CACHE_VERSION,
        config_fingerprint=sort_photos.config_fingerprint(),
        file_signatures={str(old_path): old_sig, str(unchanged): unchanged_sig},
    )
    plan = tmp / "rename.csv"
    with plan.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "destination"])
        writer.writeheader()
        writer.writerow({"source": str(old_path), "destination": str(new_path)})

    cache_path = tmp / "cache.pkl"
    with cache_path.open("wb") as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    original_cache_path = relink_cache_from_old_cache.sort_photos.CACHE_FILE
    relink_cache_from_old_cache.sort_photos.CACHE_FILE = cache_path
    try:
        result = relink_cache_from_old_cache.relink_from_rename_plan(cache, plan, apply=True)
    finally:
        relink_cache_from_old_cache.sort_photos.CACHE_FILE = original_cache_path

    assert_true(result == 0, f"rename-plan relink failed with {result}")
    with cache_path.open("rb") as f:
        relinked = pickle.load(f)
    assert_true(str(new_path) in relinked.file_signatures,
                "renamed cache path was not relinked")
    assert_true(str(unchanged) in relinked.file_signatures,
                "unchanged cache path was discarded")
    assert_true(str(old_path) not in relinked.file_signatures,
                "stale pre-rename cache path remained")


def test_generated_person_views_are_excluded_from_scanners(tmp: Path) -> None:
    people = tmp / "people"
    person = people / "Person"
    make_image(person / "photos" / "Person_0001.jpg", (20, 120, 210))
    for dirname in [
        "all",
        "_smart_albums",
        "_smart_albums_v2",
        "_smart_albums_simple_preview",
        "review",
        "_duplicates",
        "_near_visual_review",
    ]:
        make_image(person / dirname / "view.jpg", (210, 80, 20))

    sort_seen = list(sort_photos.iter_images(people, excluded_dir_names=set()))
    dup_seen = delete_person_folder_duplicates.iter_images(person)
    advanced_seen = advanced_duplicate_matching.iter_images(people)
    cache_seen = cache_tools.person_folder_images(people)
    rename_seen = rename_person_folder_files.iter_images(person)

    assert_true(len(sort_seen) == 1, f"sort_photos saw generated views: {sort_seen}")
    assert_true(len(dup_seen) == 1, f"delete duplicate scan saw generated views: {dup_seen}")
    assert_true(len(advanced_seen) == 1, f"advanced duplicate scan saw generated views: {advanced_seen}")
    assert_true(len(cache_seen) == 1, f"cache rehydrate scan saw generated views: {cache_seen}")
    assert_true(len(rename_seen) == 1, f"rename scanner saw generated views: {rename_seen}")
    for dirname in [
        "_smart_albums_v2",
        "_smart_albums",
        "_smart_albums_simple_preview",
        "all",
        "_duplicates",
        "_near_visual_review",
    ]:
        assert_true(dirname in cleanup_empty_person_folders.SKIP_DIRS,
                    f"cleanup-empty does not skip generated folder {dirname}")


def test_generated_contact_sheet_detector_is_precise(tmp: Path) -> None:
    people = tmp / "sorted" / "photos_by_person"
    photos = people / "Person" / "photos"
    validation = photos / "Person_00001.jpg"
    legacy = photos / "Person_00002.jpg"
    ordinary = photos / "Person_00003.jpg"
    validation.parent.mkdir(parents=True, exist_ok=True)
    make_validation_contact_sheet(validation)
    make_legacy_contact_sheet(legacy)
    make_image(ordinary, (35, 90, 140), size=(974, 332))

    matches = generated_artifacts.detect_paths(
        [validation, legacy, ordinary],
        people_dir=people,
        index_path=tmp / "analysis.sqlite3",
        progress_every=0,
    )
    matched = {Path(row.path).name: row.detection_lane for row in matches}
    assert_true(matched == {
        validation.name: "validation_layout",
        legacy.name: "legacy_layout",
    }, f"generated-sheet detector mismatch: {matched}")


def test_generated_contact_sheet_ledger_hash_recovers_nonstandard_layout(tmp: Path) -> None:
    people = tmp / "sorted" / "photos_by_person"
    path = people / "Person" / "photos" / "Person_00001.png"
    make_image(path, (90, 120, 180), size=(752, 1392))
    decoded = asset_processing.load_decoded_asset(path)
    assert_true(decoded is not None, "fixture did not decode")
    matches = generated_artifacts.detect_paths(
        [path],
        people_dir=people,
        index_path=tmp / "analysis.sqlite3",
        known_hashes={decoded.sha256},
        progress_every=0,
    )
    assert_true(
        len(matches) == 1 and matches[0].detection_lane == "historical_ledger_hash",
        f"historical generated hash was not recovered: {matches}",
    )


def test_generated_contact_sheet_cleanup_is_recoverable(tmp: Path) -> None:
    sorted_root = tmp / "sorted"
    people = sorted_root / "photos_by_person"
    path = people / "Person" / "photos" / "Person_00001.jpg"
    path.parent.mkdir(parents=True, exist_ok=True)
    make_validation_contact_sheet(path)
    matches = generated_artifacts.detect_paths(
        [path],
        people_dir=people,
        index_path=tmp / "analysis.sqlite3",
        progress_every=0,
    )
    applied = generated_artifacts.apply_matches(
        matches,
        people_dir=people,
        sorted_root=sorted_root,
        day="2026-08-27",
    )
    destination = (
        people / ".photo_app_trash" / "2026-08-27" /
        "Person" / "photos" / path.name
    )
    assert_true(not path.exists(), "generated sheet remained in canonical photos")
    assert_true(destination.exists(), "generated sheet was not recoverably preserved")
    assert_true(
        applied[0].status == "moved_to_recoverable_app_trash",
        f"unexpected cleanup status: {applied}",
    )
    events = operation_ledger.iter_events(sorted_root)
    assert_true(
        any(event.get("operation") == "generated_artifacts.remove_contact_sheet" for event in events),
        "cleanup move was not recorded in the operation ledger",
    )


def test_intake_generated_sheet_is_diverted_before_face_sorting(tmp: Path) -> None:
    sorted_root = tmp / "sorted"
    inbox = tmp / "To Process"
    generated = inbox / "batch" / "contact_sheet_page_001.jpg"
    ordinary = inbox / "batch" / "portrait.jpg"
    make_image(generated, (90, 120, 180), size=(752, 1392))
    make_image(ordinary, (20, 90, 180), size=(640, 960))
    kept, matches = generated_artifacts.partition_intake_artifacts(
        [generated, ordinary],
        index_path=tmp / "analysis.sqlite3",
    )
    assert_true(kept == [ordinary], f"intake partition was incorrect: {kept}")
    moved = generated_artifacts.archive_intake_matches(
        matches,
        input_dir=inbox,
        sorted_root=sorted_root,
    )
    destination = (
        sorted_root / "_source_review" / "ready_to_delete" /
        "generated_contact_sheets" / "batch" / generated.name
    )
    assert_true(moved == 1 and destination.exists(), "generated intake was not archived")
    assert_true(ordinary.exists(), "ordinary intake image was changed")


def test_person_staging_folders_are_excluded_from_library_scanners(tmp: Path) -> None:
    people = tmp / "people"
    person = people / "Person"
    canonical = person / "photos" / "Person_00001.jpg"
    staging = person / "Person Best" / "Person_Best_001.png"
    make_image(canonical, (20, 120, 210))
    make_image(staging, (210, 80, 20))

    sort_seen = list(sort_photos.iter_person_original_images(people))
    dup_seen = delete_person_folder_duplicates.iter_images(person)
    advanced_seen = advanced_duplicate_matching.iter_images(people)
    cache_seen = cache_tools.person_folder_images(people)
    rename_seen = rename_person_folder_files.iter_images(person)

    for label, seen in [
        ("sort person-original", sort_seen),
        ("exact duplicate", dup_seen),
        ("advanced duplicate", advanced_seen),
        ("cache rehydrate", [path for path, _person in cache_seen]),
        ("rename", rename_seen),
    ]:
        assert_true(seen == [canonical], f"{label} scanner saw staging content: {seen}")


def test_source_manifest_restore_from_ledger(tmp: Path) -> None:
    sorted_root = tmp / "sorted"
    people = sorted_root / "photos_by_person"
    manifest = sorted_root / "_source_review" / "source_manifest" / "last_known_good_originals.json"
    report_dir = sorted_root / "_source_review" / "source_manifest" / "reports"
    ready = sorted_root / "_source_review" / "ready_to_delete"
    original = people / "Person" / "photos" / "Person_0001.jpg"
    make_image(original, (24, 80, 160))

    source_manifest.promote_current(
        label="synthetic_restore",
        reason="synthetic restore baseline",
        people_dir=people,
        manifest_path=manifest,
    )
    held = ready / "person_folder_duplicates" / "Person" / "photos" / original.name
    operation_ledger.move_path(
        original,
        held,
        sorted_root=sorted_root,
        operation="synthetic.move_original",
        reason="synthetic missing original test",
        run_id="synthetic_restore",
    )
    assert_true(not original.exists(), "synthetic original should be missing before restore")

    ok, report, rows = source_manifest.restore_from_manifest(
        people_dir=people,
        manifest_path=manifest,
        search_roots=[ready],
        conflict_dir=ready / "source_manifest_recovery_conflicts",
        report_dir=report_dir,
        label="synthetic_restore",
        apply=True,
        last_failed_run=True,
    )
    assert_true(ok, f"restore did not report success: {rows}")
    assert_true(report.exists(), "restore report was not written")
    assert_true(original.exists(), "manifest restore did not recreate the protected original")
    validation = source_manifest.validate_current(
        label="synthetic_restore_validate",
        people_dir=people,
        manifest_path=manifest,
        report_dir=report_dir,
    )
    assert_true(validation.ok, f"manifest is not valid after restore: missing={validation.missing}")


def test_source_manifest_restore_dry_run_is_non_destructive(tmp: Path) -> None:
    sorted_root = tmp / "sorted"
    people = sorted_root / "photos_by_person"
    manifest = sorted_root / "_source_review" / "source_manifest" / "last_known_good_originals.json"
    report_dir = sorted_root / "_source_review" / "source_manifest" / "reports"
    ready = sorted_root / "_source_review" / "ready_to_delete"
    original = people / "Person" / "photos" / "Person_0001.jpg"
    make_image(original, (30, 90, 180))
    source_manifest.promote_current(
        label="synthetic_restore_dry_run",
        reason="synthetic dry-run baseline",
        people_dir=people,
        manifest_path=manifest,
    )
    held = ready / "person_folder_duplicates" / "Person" / "photos" / original.name
    operation_ledger.move_path(
        original,
        held,
        sorted_root=sorted_root,
        operation="synthetic.move_original",
        reason="synthetic dry-run missing original test",
        run_id="synthetic_restore_dry_run",
    )

    ok, _report, rows = source_manifest.restore_from_manifest(
        people_dir=people,
        manifest_path=manifest,
        search_roots=[ready],
        conflict_dir=ready / "source_manifest_recovery_conflicts",
        report_dir=report_dir,
        label="synthetic_restore_dry_run",
        apply=False,
    )
    assert_true(ok, f"dry-run restore should have a valid plan: {rows}")
    assert_true(not original.exists(), "dry-run restore unexpectedly recreated the original")
    assert_true(any(row.get("status") == "planned" for row in rows),
                f"dry-run restore did not produce a planned row: {rows}")


def test_source_manifest_accepts_recoverable_app_trash_only(tmp: Path) -> None:
    sorted_root = tmp / "sorted"
    people = sorted_root / "photos_by_person"
    manifest = sorted_root / "_source_review" / "source_manifest" / "last_known_good_originals.json"
    report_dir = sorted_root / "_source_review" / "source_manifest" / "reports"
    original = people / "Person" / "photos" / "Person_0001.jpg"
    make_image(original, (45, 105, 190))
    source_manifest.promote_current(
        label="synthetic_app_trash",
        reason="synthetic app-trash baseline",
        people_dir=people,
        manifest_path=manifest,
    )

    trashed = people / ".photo_app_trash" / "2026-07-21" / "Person" / "photos" / original.name
    trashed.parent.mkdir(parents=True, exist_ok=True)
    original.rename(trashed)
    validation = source_manifest.validate_current(
        label="synthetic_app_trash_validate",
        people_dir=people,
        manifest_path=manifest,
        report_dir=report_dir,
    )
    assert_true(validation.ok, f"recoverable app-trash move was blocked: {validation.missing}")
    assert_true(len(validation.app_trashed) == 1,
                f"app-trash move was not classified: {validation.app_trashed}")

    outside = sorted_root / "untracked" / original.name
    outside.parent.mkdir(parents=True, exist_ok=True)
    trashed.rename(outside)
    missing_validation = source_manifest.validate_current(
        label="synthetic_app_trash_missing_validate",
        people_dir=people,
        manifest_path=manifest,
        report_dir=report_dir,
    )
    assert_true(not missing_validation.ok, "untracked missing original was incorrectly accepted")
    assert_true(len(missing_validation.missing) == 1,
                f"untracked missing original was not reported: {missing_validation.missing}")

    collision_trash = people / ".photo_app_trash" / "2026-07-21" / "Person" / "photos" / "Person_0001-1.jpg"
    collision_trash.parent.mkdir(parents=True, exist_ok=True)
    outside.rename(collision_trash)
    receipt = people / ".photo_app_trash" / source_manifest.APP_TRASH_RECEIPT_NAME
    receipt.write_text(json.dumps({
        "sourceRelativePath": "Person/photos/Person_0001.jpg",
        "destinationRelativePath": collision_trash.relative_to(people).as_posix(),
    }) + "\n", encoding="utf-8")
    receipt_validation = source_manifest.validate_current(
        label="synthetic_app_trash_receipt_validate",
        people_dir=people,
        manifest_path=manifest,
        report_dir=report_dir,
    )
    assert_true(receipt_validation.ok, "receipt-mapped app-trash collision was blocked")
    assert_true(receipt_validation.app_trashed[0].get("match_source") == "facefolders_receipt",
                f"receipt mapping was not used: {receipt_validation.app_trashed}")


def test_source_manifest_prefers_active_rename_over_stale_trash_copy(tmp: Path) -> None:
    sorted_root = tmp / "sorted"
    people = sorted_root / "photos_by_person"
    manifest = sorted_root / "_source_review" / "source_manifest" / "manifest.json"
    report_dir = sorted_root / "_source_review" / "source_manifest" / "reports"
    original = people / "Person" / "photos" / "Person_0001.jpg"
    make_image(original, (45, 105, 190))
    source_manifest.promote_current(
        label="synthetic_active_rename",
        reason="synthetic active rename baseline",
        people_dir=people,
        manifest_path=manifest,
    )
    stale_trash = (
        people / ".photo_app_trash" / "2026-07-21"
        / "Person" / "photos" / original.name
    )
    stale_trash.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original, stale_trash)
    active = people / "Person" / "review" / "uncertain_nudity" / original.name
    active.parent.mkdir(parents=True, exist_ok=True)
    original.rename(active)
    validation = source_manifest.validate_current(
        label="synthetic_active_rename_validate",
        people_dir=people,
        manifest_path=manifest,
        report_dir=report_dir,
    )
    assert_true(validation.ok, f"active relocation failed validation: {validation.missing}")
    assert_true(len(validation.renamed) == 1, f"active relocation not recognized: {validation.renamed}")
    assert_true(not validation.app_trashed,
                f"stale trash copy overrode active relocation: {validation.app_trashed}")
    assert_true(not validation.extra, f"active relocation was left as extra: {validation.extra}")


def seed_cache(cache_path: Path, people: Path, names: list[str]) -> list[Path]:
    paths: list[Path] = []
    cache = sort_photos.CacheState(
        version=sort_photos.CACHE_VERSION,
        config_fingerprint=sort_photos.config_fingerprint(),
    )
    for i, name in enumerate(names, start=1):
        path = people / name / "photos" / f"{name}_{i:04d}.jpg"
        make_image(path, (40 * i % 255, 80, 130))
        cache.file_signatures[str(path)] = sort_photos.file_signature(path)
        paths.append(path)
    sort_photos.save_cache(cache)
    assert_true(cache_path.exists(), "seed cache was not written")
    return paths


def test_person_rehydrate_preserves_global_cache(tmp: Path) -> None:
    people = tmp / "people"
    cache_file = tmp / "cache" / "cache.pkl"
    with redirected_cache(cache_file):
        alice, bob = seed_cache(cache_file, people, ["Alice", "Bob"])
        with quiet_output():
            rc = cache_tools.rehydrate(
                people_dir=people,
                person="Alice",
                apply=True,
                replace=False,
                max_images=None,
                batch_size=50,
            )
        assert_true(rc == 0, f"person rehydrate exited {rc}")
        cache = sort_photos.load_cache()
        assert_true(str(alice) in cache.file_signatures, "Alice cache entry missing")
        assert_true(str(bob) in cache.file_signatures, "Bob cache entry was dropped")
        assert_true(len(cache.file_signatures) == 2, "person rehydrate changed cache coverage")


def test_full_rehydrate_keeps_cached_candidates(tmp: Path) -> None:
    people = tmp / "people"
    cache_file = tmp / "cache" / "cache.pkl"
    with redirected_cache(cache_file):
        paths = seed_cache(cache_file, people, ["Alice", "Bob"])
        before = cache_file.read_bytes()
        with quiet_output():
            rc = cache_tools.rehydrate(
                people_dir=people,
                person=None,
                apply=False,
                replace=False,
                max_images=None,
                batch_size=50,
            )
        assert_true(rc == 0, f"dry-run rehydrate exited {rc}")
        assert_true(cache_file.read_bytes() == before, "dry-run rehydrate wrote the cache")
        cache = sort_photos.load_cache()
        assert_true(set(cache.file_signatures) == {str(p) for p in paths},
                    "full dry-run rehydrate changed cache paths")


def test_cache_coverage_reports_missing_files(tmp: Path) -> None:
    people = tmp / "people"
    cache_file = tmp / "cache" / "cache.pkl"
    with redirected_cache(cache_file):
        seed_cache(cache_file, people, ["Alice"])
        make_image(people / "Bob" / "photos" / "Bob_0001.jpg", (10, 20, 30))
        summary = cache_tools.coverage_summary(people)
        assert_true(summary["total"] == 2, f"expected 2 originals, got {summary}")
        assert_true(summary["cached"] == 1, f"expected 1 cached file, got {summary}")
        assert_true(summary["missing"] == 1, f"expected 1 missing cache file, got {summary}")
        assert_true(cache_tools.print_coverage(people, min_coverage=0.75) == 2,
                    "coverage guard did not fail below threshold")


def test_resume_with_no_eligible_clusters_does_not_finalize(tmp: Path) -> None:
    output = tmp / "sorted"
    input_dir = tmp / "To Process"
    (output / "face_clusters").mkdir(parents=True)
    src = input_dir / "tiny.jpg"
    make_image(src, (20, 40, 60))
    state = sort_photos.LabelingState(
        version=sort_photos.LABEL_STATE_VERSION,
        output_dir=str(output),
        input_dir=str(input_dir),
        config_fingerprint=sort_photos.config_fingerprint(),
        faces=[dummy_cached_face(src)],
        cluster_ids=[1],
        name_map={1: "person_001"},
    )
    old_finish = sort_photos.finish_pipeline

    def fail_finish(*_args, **_kwargs):  # noqa: ANN001
        raise SyntheticFailure("finish_pipeline was called with no eligible clusters")

    sort_photos.finish_pipeline = fail_finish
    try:
        with quiet_output():
            rc = sort_photos.do_resume(
                state,
                do_review=False,
                use_ai=False,
                min_cluster_size=20,
                finish_labeled=False,
            )
    finally:
        sort_photos.finish_pipeline = old_finish
    assert_true(rc == 0, f"do_resume exited {rc}")


def test_partial_finish_preserves_library_cache_for_temp_sources(tmp: Path) -> None:
    people = tmp / "sorted" / "photos_by_person"
    cache_file = tmp / "cache" / "cache.pkl"
    label_state = tmp / "cache" / "labeling_state.pkl"
    with redirected_cache(cache_file), redirected_label_state(label_state):
        library_file = seed_cache(cache_file, people, ["Alice"])[0]
        before = sort_photos.load_cache()
        temp_src = tmp / "To Process" / "new.jpg"
        make_image(temp_src, (200, 40, 80))
        rec = sort_photos.cached_to_record(dummy_cached_face(temp_src))
        rec.cluster_id = 1

        old_organize = sort_photos.organize_originals
        old_manifest = sort_photos.write_manifest
        old_post = sort_photos.run_post_process
        sort_photos.organize_originals = lambda *_args, **_kwargs: None
        sort_photos.write_manifest = lambda *_args, **_kwargs: None
        sort_photos.run_post_process = lambda *_args, **_kwargs: None
        try:
            with quiet_output():
                sort_photos.finish_pipeline(
                    [rec],
                    {1: "Alice"},
                    tmp / "sorted",
                    input_dir=tmp / "To Process",
                    do_review=False,
                    interactive_was_run=False,
                    preserve_labeling_state=True,
                )
        finally:
            sort_photos.organize_originals = old_organize
            sort_photos.write_manifest = old_manifest
            sort_photos.run_post_process = old_post

        after = sort_photos.load_cache()
        assert_true(str(library_file) in after.file_signatures,
                    "library cache entry was dropped by partial finish")
        assert_true(after.file_signatures == before.file_signatures,
                    "partial finish rewrote the library cache from temp sources")


def test_cache_signature_mismatch_is_not_preserved(tmp: Path) -> None:
    image = tmp / "Person" / "photos" / "Person_0001.jpg"
    make_image(image, (20, 30, 40))
    old_sig = sort_photos.file_signature(image)
    make_image(image, (90, 110, 130))
    assert_true(not cache_tools.signature_matches_current_file(str(image), old_sig),
                "changed image still matched old cache signature")


def test_generated_views_do_not_recover_bad_images(tmp: Path) -> None:
    people = tmp / "people"
    bad = people / "BadCase" / "all" / "bad.jpg"
    good = people / "GoodCase" / "all" / "good.jpg"
    write_bad_image(bad)
    make_image(good, (12, 90, 140))

    with quiet_output():
        stats = person_structure.audit_or_repair(
            people_dir=people,
            review_root=tmp / "review",
            apply=True,
            quiet=True,
        )
    assert_true(stats.generated_only == 1, f"expected 1 readable recovery, got {stats.generated_only}")
    assert_true(stats.generated_only_unreadable == 1,
                f"expected 1 unreadable generated image, got {stats.generated_only_unreadable}")
    assert_true((people / "GoodCase" / "photos" / "GoodCase_recovered_0001.jpg").exists(),
                "readable generated image was not recovered")
    assert_true(not (people / "BadCase" / "photos" / "BadCase_recovered_0001.jpg").exists(),
                "unreadable generated image was incorrectly recovered")


def test_duplicate_matching_accepts_pillow_readable_images(tmp: Path) -> None:
    root = tmp / "people"
    fallback = root / "Person" / "photos" / "fallback.heic"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), (160, 30, 60)).save(fallback, format="GIF")

    img = advanced_duplicate_matching.imread(fallback)
    assert_true(img is not None and isinstance(img, np.ndarray),
                "Pillow-readable image failed duplicate decoder fallback")

    with quiet_output():
        infos, errors, stats = advanced_duplicate_matching.collect(
            root=root,
            scope="per-folder",
            cache_path=tmp / "fingerprints.json",
            quarantine_errors=True,
            bad_dir=tmp / "bad",
        )
    assert_true(len(infos) == 1, f"expected 1 duplicate info, got {len(infos)}")
    assert_true(len(errors) == 0, f"readable fallback image was reported as bad: {errors}")
    assert_true(stats["bad_moved"] == 0, "readable fallback image was quarantined")
    assert_true(fallback.exists(), "readable fallback image was moved out of place")


def test_near_visual_candidates_are_never_move_actions(tmp: Path) -> None:
    a = advanced_duplicate_matching.ImageInfo(
        path=tmp / "a.jpg",
        scope="Person",
        nudity_status="safe",
        size_bytes=1,
        width=100,
        height=100,
        sha256="a",
        pixel_sha256="pa",
        phash=0,
    )
    b = advanced_duplicate_matching.ImageInfo(
        path=tmp / "b.jpg",
        scope="Person",
        nudity_status="safe",
        size_bytes=1,
        width=100,
        height=100,
        sha256="b",
        pixel_sha256="pb",
        phash=1,
    )
    groups = advanced_duplicate_matching.build_duplicates(
        [a, b],
        near_threshold=5,
        move_near=True,
        include_same_pixels=False,
        include_near_visual=True,
    )
    assert_true(any(g.kind == "visually_similar" for g in groups),
                f"expected a near-visual group, got {groups}")
    assert_true(all(g.action != "move" for g in groups if g.kind == "visually_similar"),
                f"near-visual candidate became a move action: {groups}")


def duplicate_review_state(tmp: Path) -> tuple[dict, near_visual_review.ReviewGroup, Path, Path]:
    sorted_root = tmp / "sorted"
    people = sorted_root / "photos_by_person"
    keeper = people / "Alice" / "photos" / "Alice_00001.jpg"
    candidate = people / "Alice" / "photos" / "Alice_00002.jpg"
    make_image(keeper, color=(40, 80, 120))
    candidate.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(keeper, candidate)
    digest = operation_ledger.sha256_file(keeper)
    group = near_visual_review.ReviewGroup(
        group_id="1",
        kind="exact_file",
        person="Alice",
        scope="Alice",
        keeper=keeper,
        confidence=100,
        candidates=[
            near_visual_review.Candidate(keeper, "exact_file", "keep", 100, None, 32, 32, keeper.stat().st_size, digest),
            near_visual_review.Candidate(candidate, "exact_file", "move", 100, None, 32, 32, candidate.stat().st_size, digest),
        ],
    )
    state = {
        "root": people,
        "sorted_root": sorted_root,
        "review_dir": sorted_root / "_source_review" / "ready_to_delete" / "manual_duplicate_review",
        "thumb_dir": tmp / "thumbs",
        "thumbnails": {},
        "candidate_kinds": near_visual_review.candidate_kinds_by_path([group]),
        "candidate_keepers": near_visual_review.candidate_keepers_by_path([group]),
        "candidate_hashes": near_visual_review.candidate_hashes_by_path([group]),
        "decisions": tmp / "duplicate_decisions.json",
        "report": tmp / "duplicates.csv",
        "limit": None,
        "quiet": True,
        "lock": threading.Lock(),
        "moved_sources": set(),
    }
    return state, group, keeper, candidate


def test_duplicate_review_revalidates_and_moves_exact_files_recoverably(tmp: Path) -> None:
    state, group, keeper, candidate = duplicate_review_state(tmp)
    message = near_visual_review.apply_decision(candidate, "move", state)
    assert_true("Moved" in message, "exact duplicate was not moved")
    assert_true(keeper.exists() and not candidate.exists(), "duplicate review moved the keeper or left the candidate")
    recovered = list(state["review_dir"].rglob(candidate.name))
    assert_true(len(recovered) == 1, "moved duplicate is not in recoverable review storage")
    assert_true(str(candidate.resolve(strict=False)) in state["moved_sources"],
                "moved duplicate was not queued for cache cleanup")
    decisions = near_visual_review.load_decisions(state["decisions"])
    record = decisions["items"].get(str(candidate.resolve(strict=False)), {})
    assert_true(record.get("verified_sha256") == operation_ledger.sha256_file(keeper),
                "duplicate decision did not persist its verified hash")
    events = operation_ledger.iter_events(state["sorted_root"])
    assert_true(any(event.get("operation") == "duplicate_review.move_candidate"
                    and event.get("status") == "moved" for event in events),
                "duplicate move was not recorded in the correct library ledger")

    cached_face = sort_photos.CachedFace(
        src_str=str(candidate.resolve(strict=False)), face_index=0, det_score=0.9,
        bbox_size=100.0, sharpness=100.0, yaw_proxy=0.0, quality=0.9,
        embedding=np.zeros(512, dtype=np.float32), image_phash=np.zeros(64, dtype=np.uint8),
        crop_jpeg=b"", label="Alice",
    )
    cache = sort_photos.CacheState(
        config_fingerprint=sort_photos.config_fingerprint(),
        file_signatures={str(candidate.resolve(strict=False)): (1.0, 1), str(keeper): (1.0, 1)},
        faces=[cached_face],
    )
    changed = near_visual_review.prune_cache_sources(cache, state["moved_sources"])
    assert_true(changed and str(candidate.resolve(strict=False)) not in cache.file_signatures,
                "moved duplicate path remained in the face cache")
    assert_true(not cache.faces and str(keeper) in cache.file_signatures,
                "duplicate cache cleanup removed the keeper or retained the moved face")

    stale = keeper.parent / "Alice_00003.jpg"
    shutil.copy2(keeper, stale)
    stale_key = str(stale.resolve())
    state["candidate_kinds"][stale_key] = "exact_file"
    state["candidate_keepers"][stale_key] = str(keeper.resolve())
    state["candidate_hashes"][stale_key] = operation_ledger.sha256_file(stale)
    make_image(stale, color=(220, 30, 30))
    try:
        near_visual_review.apply_decision(stale, "move", state)
    except ValueError as exc:
        assert_true("changed since" in str(exc), f"wrong stale-report error: {exc}")
    else:
        raise SyntheticFailure("stale duplicate report moved a changed file")
    assert_true(stale.exists(), "stale-report protection moved the changed candidate")

    page = near_visual_review.render_html(
        [group], state["root"], state["report"], state["review_dir"],
        {"version": 1, "items": {}}, static_images=True,
    )
    assert_true("face duplicate-review" in page and "face option 8" not in page,
                "static duplicate review still points to the obsolete menu number")

    near_candidate = near_visual_review.Candidate(
        stale, "visually_similar", "review", 80, 4, 32, 32, stale.stat().st_size,
    )
    near_card = near_visual_review.render_card(
        near_candidate, state["root"], False, {}, static_images=True,
    )
    assert_true("Move to Ready To Delete" not in near_card and ">Keep<" in near_card,
                "near-visual review card exposes a destructive move action")


def test_duplicate_review_server_serves_and_accepts_keep(tmp: Path) -> None:
    state, _group, _keeper, candidate = duplicate_review_state(tmp)
    server = near_visual_review.ThreadingHTTPServer(
        ("127.0.0.1", 0), near_visual_review.make_handler(state),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8")
        assert_true("Duplicate Review" in page, "live duplicate review page did not render")
        payload = urllib.parse.urlencode({"action": "keep", "path": str(candidate)}).encode("utf-8")
        response = urllib.request.urlopen(base + "/decide", data=payload, timeout=5)
        result = json.loads(response.read().decode("utf-8"))
        assert_true("Kept" in result.get("message", ""), "live duplicate keep decision failed")
        assert_true(candidate.exists(), "keep action moved the duplicate candidate")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_intake_fingerprint_accepts_pillow_readable_images(tmp: Path) -> None:
    fallback = tmp / "incoming" / "fallback.heic"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), (35, 140, 80)).save(fallback, format="GIF")

    stats: Counter = Counter()
    fp = sort_photos.image_duplicate_fingerprint(fallback, cache_entries={}, stats=stats)
    assert_true(fp is not None, "intake duplicate fingerprint rejected a Pillow-readable image")
    assert_true(stats.get("decode_errors", 0) == 0,
                f"intake duplicate fingerprint counted a decode error: {stats}")


def test_sqlite_analysis_index_reuses_and_invalidates_fingerprints(tmp: Path) -> None:
    image = tmp / "asset.jpg"
    make_image(image, color=(40, 90, 160), size=(32, 24))
    database = tmp / "analysis.sqlite3"
    first_stats: Counter = Counter()
    with analysis_index.AnalysisIndex(database) as index:
        first = sort_photos.image_duplicate_fingerprint(
            image, cache_entries={}, stats=first_stats, asset_index=index)
    assert_true(first is not None and first_stats["cache_misses"] == 1,
                f"first SQLite fingerprint was not calculated: {first_stats}")

    second_stats: Counter = Counter()
    with analysis_index.AnalysisIndex(database) as index:
        second = sort_photos.image_duplicate_fingerprint(
            image, cache_entries={}, stats=second_stats, asset_index=index)
    assert_true(second == first and second_stats["sqlite_hits"] == 1,
                f"unchanged fingerprint did not come from SQLite: {second_stats}")

    make_image(image, color=(170, 40, 80), size=(40, 30))
    changed_stats: Counter = Counter()
    with analysis_index.AnalysisIndex(database) as index:
        changed = sort_photos.image_duplicate_fingerprint(
            image, cache_entries={}, stats=changed_stats, asset_index=index)
    assert_true(changed is not None and changed != first,
                "changed asset incorrectly reused its old fingerprint")
    assert_true(changed_stats["cache_misses"] == 1 and not changed_stats["sqlite_hits"],
                f"changed asset was not invalidated: {changed_stats}")


def test_operation_ledger_sqlite_mirror_never_blocks_moves(tmp: Path) -> None:
    sorted_root = tmp / "sorted"
    sorted_root.mkdir()
    database = tmp / "analysis.sqlite3"
    original_sorted = operation_ledger.DEFAULT_SORTED
    original_database = operation_ledger.ANALYSIS_INDEX_PATH
    blocker = analysis_index.AnalysisIndex(database)
    try:
        blocker.connection.execute("BEGIN IMMEDIATE")
        operation_ledger.DEFAULT_SORTED = sorted_root
        operation_ledger.ANALYSIS_INDEX_PATH = database
        event = {
            "run_id": "sqlite-lock-regression",
            "operation": "test.move",
            "status": "planned",
            "source_path": str(tmp / "source.jpg"),
            "dest_path": str(tmp / "dest.jpg"),
        }
        started = time.monotonic()
        ledger = operation_ledger.append_event(event, sorted_root=sorted_root)
        elapsed = time.monotonic() - started
        assert_true(ledger.exists(), "authoritative JSON operation ledger was not written")
        assert_true(
            elapsed < 1.0,
            f"SQLite mirror delayed the authoritative operation ledger for {elapsed:.2f}s",
        )
    finally:
        operation_ledger.DEFAULT_SORTED = original_sorted
        operation_ledger.ANALYSIS_INDEX_PATH = original_database
        blocker.connection.rollback()
        blocker.close()

    # A failed short-timeout mirror connection must not leak another lock.
    with analysis_index.AnalysisIndex(database, timeout=0.25) as index:
        index.record_operation(event)


def test_operation_ledger_batch_mirror_is_transactional(tmp: Path) -> None:
    sorted_root = tmp / "sorted"
    database = tmp / "analysis.sqlite3"
    run_id = "batch-mirror-regression"
    original_database = operation_ledger.ANALYSIS_INDEX_PATH
    events = [
        {
            "run_id": run_id,
            "operation": "test.bulk_move",
            "status": status,
            "source_path": str(tmp / "source.jpg"),
            "dest_path": str(tmp / "dest.jpg"),
        }
        for status in ("planned", "moved")
    ]
    ledger = operation_ledger.ledger_path(sorted_root, run_id)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    ledger.write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    try:
        operation_ledger.ANALYSIS_INDEX_PATH = database
        mirrored, malformed = operation_ledger.mirror_run_to_sqlite(
            sorted_root=sorted_root,
            run_id=run_id,
        )
        assert_true((mirrored, malformed) == (2, 0),
                    f"unexpected batch mirror result: {(mirrored, malformed)}")
        with analysis_index.AnalysisIndex(database) as index:
            count = index.connection.execute(
                "SELECT COUNT(*) FROM operation_history WHERE run_id=?",
                (run_id,),
            ).fetchone()[0]
        assert_true(count == 2, f"batch mirror persisted {count} of 2 events")
    finally:
        operation_ledger.ANALYSIS_INDEX_PATH = original_database


def test_shared_asset_analysis_reads_and_decodes_once(tmp: Path) -> None:
    image = tmp / "asset.jpg"
    make_image(image, color=(30, 100, 190), size=(64, 40))
    original_read_bytes = Path.read_bytes
    calls = 0

    def counting_read_bytes(path: Path) -> bytes:
        nonlocal calls
        if path == image:
            calls += 1
        return original_read_bytes(path)

    Path.read_bytes = counting_read_bytes
    try:
        decoded = asset_processing.load_decoded_asset(image)
    finally:
        Path.read_bytes = original_read_bytes
    assert_true(decoded is not None, "shared asset analysis could not decode image")
    assert_true(calls == 1, f"shared asset analysis read the source {calls} times")
    assert_true(decoded.width == 64 and decoded.height == 40,
                f"shared asset dimensions were wrong: {decoded}")
    assert_true(len(decoded.sha256) == 64 and len(decoded.pixel_sha256) == 64,
                "shared asset hashes were not calculated")
    assert_true(decoded.phash_bits.size == 64,
                "shared asset perceptual hash was not calculated")


def test_detection_persists_shared_fingerprint_for_later_stages(tmp: Path) -> None:
    image = tmp / "detected.jpg"
    make_image(image, color=(60, 110, 180), size=(96, 96))
    embedding = np.zeros(512, dtype=np.float32)
    embedding[0] = 1.0

    class FakeApp:
        def get(self, _image):
            return [SimpleNamespace(
                det_score=0.99,
                bbox=np.array([8.0, 8.0, 88.0, 88.0], dtype=np.float32),
                kps=np.array([
                    [30.0, 35.0], [66.0, 35.0], [48.0, 50.0],
                    [35.0, 68.0], [61.0, 68.0],
                ], dtype=np.float32),
                normed_embedding=embedding,
            )]

    fingerprints: dict[str, dict[str, int | str]] = {}
    diagnostics: dict[str, str] = {}
    faces = sort_photos._detect_one_image(
        image,
        FakeApp(),
        diagnostics=diagnostics,
        fingerprints=fingerprints,
    )
    database = tmp / "analysis.sqlite3"
    sort_photos.persist_detection_batch(
        [image], faces, diagnostics, database, fingerprints,
    )
    stats: Counter = Counter()
    with analysis_index.AnalysisIndex(database) as index:
        fingerprint = sort_photos.image_duplicate_fingerprint(
            image, cache_entries={}, stats=stats, asset_index=index,
        )
    assert_true(fingerprint is not None, "detection fingerprint was not persisted")
    assert_true(stats["sqlite_hits"] == 1 and stats["cache_misses"] == 0,
                f"later analysis decoded an already analysed asset: {stats}")


def test_sqlite_migration_cli_dispatches_without_people_dir(tmp: Path) -> None:
    database = tmp / "migration.sqlite3"
    command = [
        sys.executable,
        str(SCRIPT_DIR / "cache_tools.py"),
        "migrate-sqlite",
        "--apply",
        "--database",
        str(database),
        "--max-images",
        "0",
    ]
    process = subprocess.run(
        command,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert_true(process.returncode == 0,
                f"SQLite migration CLI failed: {process.stdout[-500:]}")
    assert_true(database.exists(), "SQLite migration CLI did not create its database")


def test_sqlite_analysis_index_round_trips_analysis_history(tmp: Path) -> None:
    image = tmp / "asset.jpg"
    make_image(image, color=(80, 120, 160), size=(48, 32))
    database = tmp / "analysis.sqlite3"
    face = dummy_cached_face(image, label="Alice")
    indexed_face = sort_photos.cached_face_to_index_record(face)
    with analysis_index.AnalysisIndex(database) as index:
        index.replace_detections(
            image,
            sort_photos.config_fingerprint(),
            "accepted_face",
            [indexed_face],
            width=48,
            height=32,
            orientation=1,
        )
        index.record_nudity(
            image,
            model=sort_photos.NUDITY_ANALYSIS_VERSION,
            status="safe",
            detections=[{"class": "FACE", "score": 0.99}],
        )
        index.record_identity(
            image,
            face_index=0,
            identity_name="Alice",
            distance=0.08,
            margin=0.44,
            lane="strict_single",
            run_id="test-run",
        )
        event = {
            "run_id": "test-run",
            "operation": "test.move",
            "status": "moved",
            "source_path": str(image),
            "dest_path": str(tmp / "dest.jpg"),
        }
        index.record_operation(event)
        index.record_operation(event)

        cached = index.cached_detections(image, sort_photos.config_fingerprint())
        assert_true(cached is not None and cached.status == "accepted_face",
                    "SQLite detection result was not readable")
        assert_true(len(cached.detections) == 1,
                    f"SQLite face detections were not preserved: {cached}")
        restored = sort_photos.index_record_to_cached_face(image, cached.detections[0])
        assert_true(restored.label == "Alice" and np.allclose(restored.embedding, face.embedding),
                    "SQLite face embedding did not round trip")
        nudity = index.cached_nudity(image, sort_photos.NUDITY_ANALYSIS_VERSION)
        assert_true(nudity is not None and nudity[0] == "safe",
                    f"SQLite nudity evidence was not readable: {nudity}")
        latest = index.latest_identity(image, 0)
        assert_true(latest is not None and latest["identity_name"] == "Alice",
                    f"SQLite identity history was not readable: {latest}")
        operation_count = index.connection.execute(
            "SELECT COUNT(*) FROM operation_history"
        ).fetchone()[0]
        assert_true(operation_count == 1,
                    "duplicate operation event was inserted more than once")

    make_image(image, color=(170, 30, 60), size=(49, 33))
    with analysis_index.AnalysisIndex(database) as index:
        assert_true(
            index.cached_detections(image, sort_photos.config_fingerprint()) is None,
            "changed asset reused stale SQLite detections",
        )
        assert_true(
            index.cached_nudity(image, sort_photos.NUDITY_ANALYSIS_VERSION) is None,
            "changed asset reused stale SQLite nudity evidence",
        )


def test_sqlite_detection_replace_deduplicates_face_indices(tmp: Path) -> None:
    image = tmp / "asset.jpg"
    make_image(image)
    database = tmp / "analysis.sqlite3"
    lower_quality = sort_photos.cached_face_to_index_record(
        dummy_cached_face(image, label="Alice")
    )
    better_quality = analysis_index.DetectionRecord(
        **{
            **lower_quality.__dict__,
            "quality": lower_quality.quality + 0.1,
            "crop_jpeg": b"better-crop",
        }
    )
    with analysis_index.AnalysisIndex(database) as index:
        index.replace_detections(
            image,
            sort_photos.config_fingerprint(),
            "accepted_face",
            [lower_quality, better_quality],
        )
        cached = index.cached_detections(image, sort_photos.config_fingerprint())
    assert_true(cached is not None and len(cached.detections) == 1,
                "duplicate face indices were not collapsed")
    assert_true(cached.detections[0].crop_jpeg == b"better-crop",
                "SQLite did not keep the best duplicate face detection")


def test_nudity_check_uses_normalized_fallback(tmp: Path) -> None:
    class FakeNudityDetector:
        def __init__(self):
            self.paths: list[str] = []

        def detect(self, path: str) -> list[dict]:
            self.paths.append(path)
            if Path(path).suffix.lower() != ".jpg":
                raise AttributeError("'NoneType' object has no attribute 'shape'")
            return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.99}]

    person_dir = tmp / "people" / "Person"
    image = person_dir / "photos" / "candidate.heic"
    image.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (32, 32), (180, 80, 120)).save(image, format="GIF")

    old_detector = sort_photos._NUDITY_DETECTOR
    old_enabled = sort_photos.NUDITY_SORT_ENABLED
    detector = FakeNudityDetector()
    sort_photos._NUDITY_DETECTOR = detector
    sort_photos.NUDITY_SORT_ENABLED = True
    try:
        with quiet_output():
            dest, status = sort_photos.maybe_move_to_nudity_subfolder(image, person_dir)
    finally:
        sort_photos._NUDITY_DETECTOR = old_detector
        sort_photos.NUDITY_SORT_ENABLED = old_enabled

    assert_true(status == sort_photos.NUDITY_POSSIBLE_DIR,
                f"nudity fallback did not classify possible nudity: {status}")
    assert_true(dest.exists(), "nudity fallback destination does not exist")
    assert_true(dest.parent == person_dir / "photos" / "nude",
                f"nudity fallback moved to wrong folder: {dest}")
    assert_true(len(detector.paths) == 2, f"expected original + fallback detector calls, got {detector.paths}")


def test_intake_preserves_nude_variant_of_existing_normal(tmp: Path) -> None:
    class ExposedDetector:
        def __init__(self):
            self.calls = 0

        def detect(self, path: str) -> list[dict]:
            del path
            self.calls += 1
            return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.99}]

    output = tmp / "sorted"
    person_dir = output / "photos_by_person" / "Person"
    existing = person_dir / "photos" / "Person_00001_photo.jpg"
    incoming = tmp / "intake" / "candidate.jpg"
    make_image(existing, color=(120, 70, 90))
    incoming.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(existing, incoming)

    old_detector = sort_photos._NUDITY_DETECTOR
    old_enabled = sort_photos.NUDITY_SORT_ENABLED
    old_fingerprint_cache = sort_photos.FINGERPRINT_CACHE_FILE
    detector = ExposedDetector()
    sort_photos._NUDITY_DETECTOR = detector
    sort_photos.NUDITY_SORT_ENABLED = True
    sort_photos.FINGERPRINT_CACHE_FILE = tmp / "fingerprints.json"
    sort_photos._NUDITY_STATUS_CACHE.clear()
    try:
        with quiet_output():
            kept, duplicates, _review = sort_photos.filter_existing_person_duplicates(
                [incoming], output)
        assert_true(kept == [incoming], "nude variant was discarded as a normal duplicate")
        assert_true(not duplicates, "nude variant was incorrectly archived as an exact duplicate")

        copied = person_dir / "photos" / "Person_00002_photo.jpg"
        shutil.copy2(incoming, copied)
        file_hash = sort_photos.sha256_file(incoming)
        with quiet_output():
            destination, status = sort_photos.maybe_move_to_nudity_subfolder(
                copied, person_dir, file_hash=file_hash)
        assert_true(status == sort_photos.NUDITY_POSSIBLE_DIR,
                    f"nude variant received wrong placement status: {status}")
        assert_true(destination.parent == person_dir / "photos" / "nude",
                    f"nude variant was not placed in photos/nude: {destination}")
        assert_true(detector.calls == 1,
                    f"content classification was not reused during placement: {detector.calls} calls")
    finally:
        sort_photos._NUDITY_DETECTOR = old_detector
        sort_photos.NUDITY_SORT_ENABLED = old_enabled
        sort_photos.FINGERPRINT_CACHE_FILE = old_fingerprint_cache
        sort_photos._NUDITY_STATUS_CACHE.clear()


def test_intake_filters_only_same_nudity_category(tmp: Path) -> None:
    class ExposedDetector:
        def detect(self, path: str) -> list[dict]:
            del path
            return [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.99}]

    output = tmp / "sorted"
    person_dir = output / "photos_by_person" / "Person"
    existing = person_dir / "photos" / "nude" / "Person_00001_photo.jpg"
    incoming = tmp / "intake" / "candidate.jpg"
    make_image(existing, color=(120, 70, 90))
    incoming.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(existing, incoming)

    old_detector = sort_photos._NUDITY_DETECTOR
    old_enabled = sort_photos.NUDITY_SORT_ENABLED
    old_fingerprint_cache = sort_photos.FINGERPRINT_CACHE_FILE
    sort_photos._NUDITY_DETECTOR = ExposedDetector()
    sort_photos.NUDITY_SORT_ENABLED = True
    sort_photos.FINGERPRINT_CACHE_FILE = tmp / "fingerprints.json"
    sort_photos._NUDITY_STATUS_CACHE.clear()
    try:
        with quiet_output():
            kept, duplicates, _review = sort_photos.filter_existing_person_duplicates(
                [incoming], output)
        assert_true(not kept, "same-category exact nude duplicate was retained")
        assert_true(duplicates == [incoming], "same-category exact nude duplicate was not detected")
    finally:
        sort_photos._NUDITY_DETECTOR = old_detector
        sort_photos.NUDITY_SORT_ENABLED = old_enabled
        sort_photos.FINGERPRINT_CACHE_FILE = old_fingerprint_cache
        sort_photos._NUDITY_STATUS_CACHE.clear()


def test_intake_collapses_same_category_duplicates_within_batch(tmp: Path) -> None:
    output = tmp / "sorted"
    existing = output / "photos_by_person" / "Existing" / "photos" / "Existing_00001.jpg"
    first = tmp / "intake" / "batch_a" / "candidate.jpg"
    second = tmp / "intake" / "batch_b" / "candidate_copy.jpg"
    make_image(existing, color=(20, 80, 140))
    make_image(first, color=(170, 40, 90))
    second.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(first, second)

    old_fingerprint_cache = sort_photos.FINGERPRINT_CACHE_FILE
    old_nudity_enabled = sort_photos.NUDITY_SORT_ENABLED
    sort_photos.FINGERPRINT_CACHE_FILE = tmp / "fingerprints.json"
    sort_photos.NUDITY_SORT_ENABLED = False
    sort_photos._NUDITY_STATUS_CACHE.clear()
    try:
        with quiet_output():
            kept, duplicates, _review = sort_photos.filter_existing_person_duplicates(
                [first, second], output)
        assert_true(kept == [first], f"first same-batch occurrence was not retained: {kept}")
        assert_true(
            duplicates == [second],
            f"same-category duplicate in one intake batch reached clustering: {duplicates}",
        )
    finally:
        sort_photos.FINGERPRINT_CACHE_FILE = old_fingerprint_cache
        sort_photos.NUDITY_SORT_ENABLED = old_nudity_enabled
        sort_photos._NUDITY_STATUS_CACHE.clear()


def test_cross_person_identity_audit_flags_without_move(tmp: Path) -> None:
    alice = tmp / "people" / "Alice" / "photos" / "Alice_00001.jpg"
    bob = tmp / "people" / "Bob" / "photos" / "Bob_00001.jpg"
    make_image(alice)
    bob.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(alice, bob)
    embedding = np.asarray([1.0, 0.0], dtype=np.float32)
    face_record = sort_photos.CachedFace(
        src_str=str(alice),
        face_index=0,
        det_score=1.0,
        bbox_size=0.5,
        sharpness=100.0,
        yaw_proxy=0.0,
        quality=1.0,
        embedding=embedding,
        image_phash=np.zeros(64, dtype=np.uint8),
        crop_jpeg=b"",
    )
    identity_db = sort_photos.IdentityDB(
        identities={
            "Alice": embedding,
            "Bob": np.asarray([0.0, 1.0], dtype=np.float32),
        },
        prototypes={
            "Alice": [embedding],
            "Bob": [np.asarray([0.0, 1.0], dtype=np.float32)],
        },
        strict_thresholds={"Alice": 0.27, "Bob": 0.27},
    )
    groups = {
        "sha256:test": [
            audit_cross_person_identity.Membership("Alice", alice),
            audit_cross_person_identity.Membership("Bob", bob),
        ]
    }
    results = audit_cross_person_identity.analyze_groups(
        groups,
        {str(alice): [face_record]},
        identity_db,
    )
    assert_true(len(results) == 1, "cross-person conflict group was not reported")
    assert_true(results[0].status == "high_confidence_conflict",
                f"strict conflict received wrong status: {results[0].status}")
    assert_true(results[0].predicted_person == "Alice", "strict identity prediction was wrong")
    assert_true(results[0].wrong_people == ("Bob",), "wrong membership was not isolated")
    assert_true(alice.exists() and bob.exists(), "report-only audit modified an original")


def cross_person_review_state(
    tmp: Path,
) -> tuple[dict, audit_cross_person_identity.AuditGroup, Path, Path]:
    sorted_root = tmp / "sorted"
    people = sorted_root / "photos_by_person"
    alice = people / "Alice" / "photos" / "Alice_00001.jpg"
    bob = people / "Bob" / "photos" / "Bob_00001.jpg"
    make_image(alice)
    bob.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(alice, bob)
    content_key = f"sha256:{operation_ledger.sha256_file(alice)}"
    group = audit_cross_person_identity.AuditGroup(
        content_key=content_key,
        memberships=(
            audit_cross_person_identity.Membership("Alice", alice),
            audit_cross_person_identity.Membership("Bob", bob),
        ),
        representative=alice,
        detected_faces=1,
        strict_matches=(
            audit_cross_person_identity.StrictMatch(
                person="Alice", distance=0.1, margin=0.3, quality=0.9,
            ),
        ),
        status="high_confidence_conflict",
        predicted_person="Alice",
    )
    cache = sort_photos.CacheState(
        config_fingerprint=sort_photos.config_fingerprint(),
        file_signatures={
            str(alice): sort_photos.file_signature(alice),
            str(bob): sort_photos.file_signature(bob),
        },
    )
    state = {
        "people_dir": people,
        "sorted_root": sorted_root,
        "review_dir": sorted_root / "_source_review" / "ready_to_delete" / "identity",
        "decisions_path": tmp / "decisions.json",
        "groups_by_key": {content_key: group},
        "cache": cache,
        "cache_dirty": False,
        "affected_people": set(),
    }
    return state, group, alice, bob


def test_cross_person_review_keep_ignore_and_manual_correct(tmp: Path) -> None:
    state, group, alice, bob = cross_person_review_state(tmp)
    kept = audit_cross_person_identity.apply_decision(
        state,
        content_key=group.content_key,
        person="Alice",
        path_text=str(alice),
        action="keep",
    )
    ignored = audit_cross_person_identity.apply_decision(
        state,
        content_key=group.content_key,
        person="Bob",
        path_text=str(bob),
        action="ignore",
    )
    assert_true("Kept" in kept and "Ignored" in ignored, "non-moving decisions were not recorded")
    assert_true(alice.exists() and bob.exists(), "keep/ignore moved an original")

    message = audit_cross_person_identity.apply_decision(
        state,
        content_key=group.content_key,
        person="Bob",
        path_text=str(bob),
        action="correct",
    )
    assert_true("Set Bob as correct" in message, "manual correct-person override was not honored")
    assert_true(bob.exists() and not alice.exists(), "manual correction kept the wrong membership")
    recovered = list(state["review_dir"].rglob("Alice_00001.jpg"))
    assert_true(len(recovered) == 1, "corrected membership was not moved to recoverable storage")
    assert_true(str(alice) not in state["cache"].file_signatures,
                "corrected membership remained in the face cache")
    assert_true(state["cache_dirty"], "corrected membership did not mark cache dirty")
    events = operation_ledger.iter_events(state["sorted_root"])
    assert_true(any(event.get("operation") == "cross_person_identity.corrected"
                    and event.get("status") == "moved" for event in events),
                "corrected membership move was not ledgered")


def test_cross_person_review_discard_is_recoverable_and_ui_has_actions(tmp: Path) -> None:
    state, group, alice, bob = cross_person_review_state(tmp)
    message = audit_cross_person_identity.apply_decision(
        state,
        content_key=group.content_key,
        person="Bob",
        path_text=str(bob),
        action="discard",
    )
    assert_true("recoverable review storage" in message, "discard did not describe recovery")
    assert_true(alice.exists() and not bob.exists(), "discard moved the wrong membership")
    recovered = list(state["review_dir"].rglob("Bob_00001.jpg"))
    assert_true(len(recovered) == 1, "discarded membership is not recoverable")

    summary = {"current_files": 2}
    page = audit_cross_person_identity.render_html(
        [group],
        summary,
        {"version": 1, "items": {}},
        state["people_dir"],
        interactive=True,
        thumb_dir=tmp / "thumbs",
    )
    for label in ("Keep Here", "Correct Person", "Ignore", "Discard Copy"):
        assert_true(label in page, f"cross-person review UI is missing {label}")


def test_cross_person_review_server_serves_and_accepts_decisions(tmp: Path) -> None:
    state, group, alice, _bob = cross_person_review_state(tmp)
    state.update({
        "groups": [group],
        "summary": {"current_files": 2},
        "thumb_dir": tmp / "thumbs",
        "lock": threading.Lock(),
        "quiet": True,
    })
    server = audit_cross_person_identity.ThreadingHTTPServer(
        ("127.0.0.1", 0),
        audit_cross_person_identity.make_handler(state),
    )
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_address[1]}"
    try:
        page = urllib.request.urlopen(base + "/", timeout=5).read().decode("utf-8")
        assert_true("Correct Person" in page, "live cross-person page did not render")
        image = urllib.request.urlopen(
            base + "/image?" + urllib.parse.urlencode({"content_key": group.content_key}),
            timeout=5,
        ).read()
        assert_true(bool(image), "live cross-person thumbnail endpoint returned no data")
        payload = urllib.parse.urlencode({
            "action": "keep",
            "content_key": group.content_key,
            "person": "Alice",
            "path": str(alice),
        }).encode("utf-8")
        response = urllib.request.urlopen(base + "/decide", data=payload, timeout=5)
        result = json.loads(response.read().decode("utf-8"))
        assert_true("Kept" in result.get("message", ""), "live keep decision failed")
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def test_perceptual_dedup_keeps_cross_nudity_variants(tmp: Path) -> None:
    normal = tmp / "normal.jpg"
    nude = tmp / "nude.jpg"
    phash = np.zeros(64, dtype=np.uint8)
    items = [(normal, 100.0, phash), (nude, 90.0, phash.copy())]
    keepers, duplicates = sort_photos.dedup_within_bucket(
        items,
        threshold=sort_photos.PHASH_THRESHOLD,
        category_for_path=lambda path: "possible" if path == nude else "safe",
    )
    assert_true(keepers == [normal, nude], f"cross-category visual variants were collapsed: {keepers}")
    assert_true(not duplicates, f"cross-category visual variant became a duplicate: {duplicates}")


def test_explicit_nude_copy_does_not_reclassify_normal_copy(tmp: Path) -> None:
    class CoveredDetector:
        def detect(self, path: str) -> list[dict]:
            del path
            return [{"class": "FEMALE_BREAST_COVERED", "score": 0.95}]

    normal = tmp / "candidate.jpg"
    nude = tmp / "candidate_nude.jpg"
    make_image(normal, color=(80, 130, 170))
    shutil.copy2(normal, nude)
    phash = np.zeros(64, dtype=np.uint8)

    old_detector = sort_photos._NUDITY_DETECTOR
    old_enabled = sort_photos.NUDITY_SORT_ENABLED
    sort_photos._NUDITY_DETECTOR = CoveredDetector()
    sort_photos.NUDITY_SORT_ENABLED = True
    sort_photos._NUDITY_STATUS_CACHE.clear()
    try:
        keepers, duplicates = sort_photos.dedup_within_bucket(
            [(nude, 100.0, phash), (normal, 90.0, phash.copy())],
            threshold=sort_photos.PHASH_THRESHOLD,
            category_for_path=lambda path: sort_photos.classified_nudity_status(
                path, file_hash=sort_photos.sha256_file(path)),
        )
        assert_true(keepers == [nude, normal],
                    f"explicit nude and normal copies were collapsed: {keepers}")
        assert_true(not duplicates,
                    f"explicit nude and normal copies became duplicates: {duplicates}")
    finally:
        sort_photos._NUDITY_DETECTOR = old_detector
        sort_photos.NUDITY_SORT_ENABLED = old_enabled
        sort_photos._NUDITY_STATUS_CACHE.clear()


def test_nudity_policy_rejects_swimwear_false_positive(tmp: Path) -> None:
    del tmp
    decision = sort_photos.nudity_decision([
        {"class": "BUTTOCKS_EXPOSED", "score": 0.71},
        {"class": "FEMALE_BREAST_COVERED", "score": 0.53},
    ])
    assert_true(decision[0] != "confirmed_nude",
                f"swimwear-like conflicting evidence was marked nude: {decision}")


def test_nudity_policy_holds_borderline_detection_for_review(tmp: Path) -> None:
    del tmp
    decision = sort_photos.nudity_decision([
        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.68},
    ])
    assert_true(decision[0] == "needs_review",
                f"borderline explicit detection bypassed review: {decision}")


def test_nudity_policy_does_not_assume_missed_detection_is_safe(tmp: Path) -> None:
    del tmp
    decision = sort_photos.nudity_decision([
        {"class": "FACE_FEMALE", "score": 0.91},
    ])
    assert_true(decision[0] == "needs_review",
                f"missing anatomy detection was incorrectly treated as safe: {decision}")


def test_nudity_policy_accepts_strong_non_conflicting_detection(tmp: Path) -> None:
    del tmp
    decision = sort_photos.nudity_decision([
        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.86},
        {"class": "FACE_FEMALE", "score": 0.82},
    ])
    assert_true(decision[0] == "confirmed_nude",
                f"strong explicit evidence was not retained: {decision}")


def test_nudity_policy_accepts_separate_exposed_and_covered_regions(tmp: Path) -> None:
    del tmp
    decision = sort_photos.nudity_decision([
        {
            "class": "FEMALE_BREAST_EXPOSED",
            "score": 0.856,
            "box": [183, 830, 253, 259],
        },
        {
            "class": "FEMALE_BREAST_COVERED",
            "score": 0.763,
            "box": [443, 839, 247, 262],
        },
    ])
    assert_true(
        decision[0] == "confirmed_nude",
        f"separate anatomy regions incorrectly conflicted: {decision}",
    )


def test_nudity_policy_holds_overlapping_covered_conflict(tmp: Path) -> None:
    del tmp
    decision = sort_photos.nudity_decision([
        {
            "class": "FEMALE_BREAST_EXPOSED",
            "score": 0.856,
            "box": [180, 830, 250, 260],
        },
        {
            "class": "FEMALE_BREAST_COVERED",
            "score": 0.763,
            "box": [190, 840, 245, 250],
        },
    ])
    assert_true(
        decision[0] == "needs_review",
        f"same-region covered conflict was ignored: {decision}",
    )


def test_nudity_policy_ignores_malformed_detector_scores(tmp: Path) -> None:
    del tmp
    decision = sort_photos.nudity_decision([
        {"class": "FEMALE_BREAST_EXPOSED", "score": "invalid"},
        {"class": "FEMALE_BREAST_EXPOSED", "score": 0.86},
    ])
    assert_true(
        decision[0] == "confirmed_nude",
        f"malformed detector score disrupted valid evidence: {decision}",
    )


def test_ambiguous_explicit_detection_routes_to_uncertain_review(tmp: Path) -> None:
    person_dir = tmp / "photos_by_person" / "Person"
    source = person_dir / "photos" / "Person_00001.jpg"
    make_image(source)
    previous = sort_photos.ROUTE_UNCERTAIN_NUDITY_TO_NUDE
    sort_photos.ROUTE_UNCERTAIN_NUDITY_TO_NUDE = False
    try:
        destination, status = sort_photos.maybe_move_to_nudity_subfolder(
            source,
            person_dir,
            preclassified_status="uncertain",
        )
    finally:
        sort_photos.ROUTE_UNCERTAIN_NUDITY_TO_NUDE = previous
    assert_true(
        status == sort_photos.NUDITY_UNCERTAIN_DIR,
        f"ambiguous explicit detection returned the wrong status: {status}",
    )
    assert_true(
        destination.parent == person_dir / "review" / "uncertain_nudity",
        f"ambiguous explicit detection went to the wrong folder: {destination}",
    )
    assert_true(destination.exists(), "ambiguous review image was not preserved")


def test_user_preference_routes_uncertain_nudity_to_nude(tmp: Path) -> None:
    person_dir = tmp / "photos_by_person" / "Person"
    source = person_dir / "photos" / "Person_00001.jpg"
    make_image(source)
    previous = sort_photos.ROUTE_UNCERTAIN_NUDITY_TO_NUDE
    sort_photos.ROUTE_UNCERTAIN_NUDITY_TO_NUDE = True
    try:
        destination, status = sort_photos.maybe_move_to_nudity_subfolder(
            source,
            person_dir,
            preclassified_status="uncertain",
        )
    finally:
        sort_photos.ROUTE_UNCERTAIN_NUDITY_TO_NUDE = previous
    assert_true(
        status == sort_photos.NUDITY_POSSIBLE_DIR,
        f"saved uncertain-nudity preference returned the wrong status: {status}",
    )
    assert_true(
        destination.parent == person_dir / "photos" / "nude",
        f"saved uncertain-nudity preference used the wrong folder: {destination}",
    )
    assert_true(destination.exists(), "user-routed uncertain image was not preserved")


def test_bad_image_audit_accepts_heif_with_legacy_suffix(tmp: Path) -> None:
    import pillow_heif

    disguised = tmp / "legacy-name.jpg"
    heif = pillow_heif.from_pillow(Image.new("RGB", (24, 24), (80, 120, 160)))
    heif.save(str(disguised))
    ok, detail = quarantine_bad_person_images.can_decode_image(disguised)
    assert_true(ok, f"valid HEIF with legacy suffix was rejected: {detail}")


def test_nudity_and_bad_image_menu_never_auto_quarantine(tmp: Path) -> None:
    del tmp
    nudity = next(action for action in face.ACTIONS if action["key"] == "nudity")
    nudity_scripts = [step["script"] for step in nudity.get("steps", [])]
    assert_true(
        "quarantine_bad_person_images.py" not in nudity_scripts,
        "nudity scan still invokes the bad-image quarantine",
    )
    assert_true(
        "audit_nude_folders.py" in nudity_scripts,
        "nudity scan no longer revalidates existing nude folders",
    )
    bad_images = next(action for action in face.ACTIONS if action["key"] == "bad-images")
    command_args = [
        arg
        for step in bad_images.get("steps", [])
        for arg in step.get("args", [])
    ]
    assert_true("--apply" not in command_args,
                "bad-image menu action is destructive by default")

    review = next(action for action in face.ACTIONS if action["key"] == "review")
    review_args = [str(arg) for arg in review.get("args", [])]
    assert_true("--resume-label" not in review_args,
                "unknown review still requires an ephemeral labeling snapshot")
    assert_true(
        any("unassigned_intake/unknown_identity" in arg for arg in review_args),
        "unknown review is not sourced from the persistent review folder",
    )

    uncertain = next(action for action in face.ACTIONS if action["key"] == "uncertain-nudity")
    assert_true(
        bool(uncertain.get("allow_original_count_decrease")),
        "confirmed uncertain-nudity moves cannot promote their protected path changes",
    )


def test_nude_audit_destinations_are_outside_nude_folder(tmp: Path) -> None:
    people = tmp / "people"
    source = people / "Person" / "photos" / "nude" / "nested" / "photo.jpg"
    make_image(source)
    safe = audit_nude_folders.destination_for(source, people, "likely_safe")
    review = audit_nude_folders.destination_for(source, people, "needs_review")
    confirmed = audit_nude_folders.destination_for(source, people, "confirmed_nude")
    assert_true(safe == people / "Person" / "photos" / "nested" / "photo.jpg",
                f"safe nude-audit destination is wrong: {safe}")
    assert_true(
        review
        == people / "Person" / "review" / "uncertain_nudity" / "nested" / "photo.jpg",
        f"review original did not enter recoverable review: {review}",
    )
    assert_true(confirmed is None, "confirmed nude image should stay in place")


def test_uncertain_nudity_only_promotes_confirmed_files(tmp: Path) -> None:
    people = tmp / "photos_by_person"
    uncertain = people / "Person" / "review" / "uncertain_nudity"
    confirmed = uncertain / "confirmed.jpg"
    ambiguous = uncertain / "ambiguous.jpg"
    safe = uncertain / "safe.jpg"
    for path in (confirmed, ambiguous, safe):
        make_image(path)

    confirmed_destination = audit_nude_folders.destination_for(
        confirmed, people, "confirmed_nude", "uncertain"
    )
    ambiguous_destination = audit_nude_folders.destination_for(
        ambiguous, people, "needs_review", "uncertain"
    )
    safe_destination = audit_nude_folders.destination_for(
        safe, people, "likely_safe", "uncertain"
    )
    assert_true(
        confirmed_destination == people / "Person" / "photos" / "nude" / confirmed.name,
        f"confirmed uncertain file has wrong destination: {confirmed_destination}",
    )
    assert_true(ambiguous_destination is None,
                "ambiguous uncertain file should remain in review")
    assert_true(safe_destination is None,
                "safe uncertain file should remain in review")


def test_manual_nudity_confirmation_survives_move(tmp: Path) -> None:
    source = tmp / "photos_by_person" / "Person" / "review" / "uncertain_nudity" / "photo.jpg"
    destination = tmp / "photos_by_person" / "Person" / "photos" / "nude" / "photo.jpg"
    confirmations = tmp / "manual_nudity_confirmations.jsonl"
    make_image(source)
    file_hash = nudity_confirmations.sha256_file(source)
    recorded_hash = nudity_confirmations.record_confirmation(
        source,
        destination,
        person="Person",
        reason="synthetic user confirmation",
        sha256=file_hash,
        confirmations_path=confirmations,
    )
    assert_true(recorded_hash == file_hash, "manual confirmation changed the content hash")
    assert_true(
        nudity_confirmations.is_confirmed(
            source, confirmations_path=confirmations
        ),
        "manual confirmation did not recognize its source path",
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    source.replace(destination)
    assert_true(
        nudity_confirmations.is_confirmed(
            destination, confirmations_path=confirmations
        ),
        "manual confirmation did not survive the destination move",
    )
    renamed = destination.with_name("renamed.jpg")
    destination.replace(renamed)
    assert_true(
        nudity_confirmations.is_confirmed(
            renamed, sha256=file_hash, confirmations_path=confirmations
        ),
        "hash fallback did not preserve a manual confirmation after rename",
    )


def test_source_guards_protect_uncertain_nudity_without_generated_review(tmp: Path) -> None:
    people = tmp / "photos_by_person"
    normal = people / "Person" / "photos" / "normal.jpg"
    uncertain = people / "Person" / "review" / "uncertain_nudity" / "uncertain.jpg"
    generated = people / "Person" / "review" / "duplicates" / "duplicate.jpg"
    for path in (normal, uncertain, generated):
        make_image(path)
    protected = set(source_manifest.image_files(people))
    assert_true(normal in protected, "normal original was not protected")
    assert_true(uncertain in protected, "uncertain nudity original was not protected")
    assert_true(generated not in protected, "generated duplicate review entered source manifest")
    assert_true(source_guard.person_original_counts(people) == {"Person": 2},
                "source count guard did not use the same protected roots")


def test_manual_nudity_scan_uses_shared_high_precision_policy(tmp: Path) -> None:
    del tmp
    confirmed = separate_nudity_review.classify_detections(
        [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.86}],
        0.10,
        0.05,
    )
    review = separate_nudity_review.classify_detections(
        [{"class": "FEMALE_BREAST_EXPOSED", "score": 0.68}],
        0.10,
        0.05,
    )
    ordinary = separate_nudity_review.classify_detections(
        [{"class": "FACE_FEMALE", "score": 0.95}],
        0.10,
        0.05,
    )
    assert_true(confirmed[0] == "confirmed_nude", f"strong detection not confirmed: {confirmed}")
    assert_true(review[0] == "needs_review", f"borderline detection bypassed review: {review}")
    assert_true(ordinary[0] == "likely_safe", f"ordinary image was flagged: {ordinary}")


def test_manual_nudity_scan_decodes_heif_with_legacy_jpg_suffix(tmp: Path) -> None:
    import pillow_heif

    disguised = tmp / "legacy.jpg"
    heif = pillow_heif.from_pillow(Image.new("RGB", (24, 32), (80, 120, 160)))
    heif.save(str(disguised))
    decoded, error = separate_nudity_review.decode_for_detector(disguised)
    assert_true(decoded is not None, f"legacy HEIF could not be decoded: {error}")
    assert_true(decoded.shape[:2] == (32, 24),
                f"legacy HEIF dimensions changed: {decoded.shape}")


def test_nudity_report_placement_never_promotes_legacy_candidates(tmp: Path) -> None:
    del tmp
    previous = place_nudity_inside_person_folders.ROUTE_UNCERTAIN_NUDITY_TO_NUDE
    try:
        place_nudity_inside_person_folders.ROUTE_UNCERTAIN_NUDITY_TO_NUDE = False
        assert_true(
            place_nudity_inside_person_folders.target_subdir("confirmed_nude", False) == "photos/nude",
            "confirmed classification did not map to nude",
        )
        assert_true(
            place_nudity_inside_person_folders.target_subdir("needs_review", False)
            == "review/uncertain_nudity",
            "ambiguous classification did not map to recoverable review",
        )
        for legacy in ("possible_nudity", "uncertain"):
            assert_true(
                place_nudity_inside_person_folders.target_subdir(legacy, True) is None,
                f"legacy low-threshold category could still enter nude: {legacy}",
            )
    finally:
        place_nudity_inside_person_folders.ROUTE_UNCERTAIN_NUDITY_TO_NUDE = previous


def test_manual_nudity_placement_respects_saved_user_preference(tmp: Path) -> None:
    del tmp
    previous = place_nudity_inside_person_folders.ROUTE_UNCERTAIN_NUDITY_TO_NUDE
    try:
        place_nudity_inside_person_folders.ROUTE_UNCERTAIN_NUDITY_TO_NUDE = True
        assert_true(
            place_nudity_inside_person_folders.target_subdir("needs_review", False)
            == "photos/nude",
            "saved user preference did not route manual-scan review to nude",
        )
    finally:
        place_nudity_inside_person_folders.ROUTE_UNCERTAIN_NUDITY_TO_NUDE = previous


def test_nudity_review_export_cannot_move_protected_original(tmp: Path) -> None:
    source = tmp / "photos_by_person" / "Person" / "photos" / "photo.jpg"
    destination = tmp / "review_export" / "photo.jpg"
    make_image(source)
    try:
        separate_nudity_review.copy_or_move(source, destination, move=True)
    except ValueError:
        pass
    else:
        raise AssertionError("deprecated review export moved a protected original")
    assert_true(source.exists(), "protected original disappeared after rejected review move")
    assert_true(not destination.exists(), "rejected review move created a destination")


def test_incremental_smart_albums_skip_without_heavy_models(tmp: Path) -> None:
    people = tmp / "people"
    person = people / "Person"
    make_image(person / "photos" / "Person_0001.jpg", (20, 120, 210))
    (person / "_smart_albums").mkdir(parents=True, exist_ok=True)
    (person / "_smart_albums" / "_smart_album_index.csv").write_text(
        "album,path\n",
        encoding="utf-8",
    )
    (person / "_smart_albums_v2" / "_data").mkdir(parents=True, exist_ok=True)
    (person / "_smart_albums_v2" / "_data" / "image_index.csv").write_text(
        "path\n",
        encoding="utf-8",
    )
    state_path = tmp / "smart_state.json"
    state = {
        "version": 1,
        "people": {
            str(person.resolve()): {
                "signature": build_smart_albums.person_content_signature(person),
                "updated_at": 0,
            }
        },
    }
    state_path.write_text(json.dumps(state), encoding="utf-8")

    cmd = [
        sys.executable,
        str(SCRIPT_DIR / "build_smart_albums.py"),
        str(people),
        "--incremental",
        "--smart-state",
        str(state_path),
        "--quiet",
    ]
    proc = subprocess.run(cmd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    assert_true(proc.returncode == 0, f"smart-albums dry-run exited {proc.returncode}: {proc.stdout[-500:]}")
    assert_true("No smart albums needed rebuilding." in proc.stdout,
                "incremental smart-albums did not skip unchanged folder")
    assert_true("Initializing InsightFace" not in proc.stdout and "Applied providers" not in proc.stdout,
                "smart-albums skip path loaded heavy face models")


TESTS = [
    ("Daily individual recovery is gated and preserves originals", test_daily_individual_recovery_regressions),
    ("Unknown decisions persist across copies and profile updates", test_unknown_decision_persistence_regressions),
    ("Architecture hardening and score equivalence regressions", test_architecture_hardening_regressions),
    ("To Process generated-like names are scanned", test_to_process_generated_names_are_visible),
    ("Unassigned intake is preserved by reason", test_unassigned_intake_is_preserved_by_reason),
    ("Soft visible faces use recovery detection", test_soft_face_uses_recovery_detection_tier),
    ("Rotated faces use alternate detection views", test_rotated_face_uses_alternate_view),
    ("Tightly cropped faces use padded detection views", test_tight_face_uses_padded_low_threshold_view),
    ("Low-confidence faces route to quality review", test_low_confidence_face_routes_to_quality_review),
    ("Recovery identity matching remains strict", test_recovery_identity_match_is_strict),
    ("Pose profiles and hard negatives share matcher policy", test_pose_profiles_and_hard_negatives_are_shared),
    ("Appearance profiles preserve strict matcher policy", test_appearance_profiles_improve_without_widening_thresholds),
    ("Source batch consensus requires independent agreement", test_source_batch_consensus_requires_face_and_batch_agreement),
    ("Daily Run uses conservative source-batch consensus", test_daily_identity_assignment_uses_source_batch_consensus),
    ("Video matcher shares pose and hard-negative policy", test_video_matcher_uses_pose_and_hard_negative_policy),
    ("Protected benchmark requires explicit verification", test_protected_benchmark_rows_are_atomic_and_unverified_by_default),
    ("Protected benchmark batch confirmation is atomic", test_protected_benchmark_bulk_confirmation_is_atomic),
    ("Protected benchmark dashboard supports multi-image review", test_protected_benchmark_dashboard_supports_multi_image_review),
    ("Secondary verifier requires independent agreement", test_secondary_matcher_requires_independent_agreement),
    ("Secondary verifier resolves cache prototype IDs", test_secondary_matcher_resolves_cache_prototype_ids),
    ("Confirmed unknowns enroll and profile activation is gated", test_confirmations_enroll_and_gate_identity_profiles),
    ("Confirmation gate allows unchanged safe rejections", test_confirmation_gate_allows_unchanged_safe_rejections),
    ("Recovery scans nested intake batches", test_recovery_scans_nested_intake_batches),
    ("Recovery reuses SQLite face detections", test_recovery_reuses_sqlite_face_detections),
    ("Identity auto-match uses strict single and consensus lanes", test_identity_auto_match_requires_independent_consensus),
    ("Anchor cluster merge requires unique-person margin", test_anchor_cluster_merge_requires_unique_person_margin),
    ("Identity refresh reuses cached negative detections", test_identity_refresh_reuses_cached_negative_detection),
    ("Identity DB promotion is atomic and backed up", test_identity_database_promotion_is_atomic_and_backed_up),
    ("Identity profiles select quality and diversity", test_identity_profiles_select_quality_and_diversity),
    ("Reserved review labels are excluded from identities", test_reserved_review_labels_are_not_people),
    ("Confirmed identity examples are bounded and persistent", test_confirmed_identity_examples_are_bounded_and_persistent),
    ("Unknown review clusters and keeps files safely", test_unknown_review_clusters_and_confirms_safely),
    ("Unknown auto-review requires independent verification", test_unknown_review_auto_match_requires_independent_verifier),
    ("Unknown review refreshes stale independent verifier", test_unknown_review_refreshes_stale_secondary_verifier),
    ("Unknown review skips unrequested verifier refresh", test_unknown_review_does_not_refresh_unrequested_verifier),
    ("Secondary verifier batches interactive confirmations", test_secondary_verifier_batches_interactive_confirmations),
    ("Unknown review uses bounded confirmation memory", test_unknown_review_uses_bounded_confirmation_memory),
    ("Unknown review auto-sweep scans entire queue", test_unknown_review_auto_sweep_scans_entire_queue),
    ("Unknown auto-review rejects mixed identity clusters", test_unknown_review_auto_cluster_rejects_identity_dissent),
    ("Unknown auto-review does not contaminate trusted training", test_unknown_review_automatic_confirmation_is_not_trusted_training),
    ("Unknown review learns corrected option-one candidates", test_unknown_review_manual_correction_teaches_option_one_rejection),
    ("Confirmed review learning excludes its own prototype", test_confirmed_review_learning_excludes_its_own_prototype),
    ("Unknown dual-model policy is benchmark gated", test_unknown_review_joint_policy_is_benchmark_gated),
    ("Confirmed benchmark follows safe file moves", test_confirmed_benchmark_relinks_moved_source_by_hash),
    ("Unknown review routes unsupported files recoverably", test_unknown_review_routes_unsupported_files_recoverably),
    ("Unknown review content identity ignores timestamps", test_unknown_review_content_identity_ignores_timestamp_only_changes),
    ("Unknown review reconciles replays and preserves manual decisions", test_unknown_review_reconciles_replays_and_preserves_manual_decisions),
    ("Unknown review loads successive batches automatically", test_unknown_review_loads_successive_batches_automatically),
    ("Unknown review batch loading is asynchronous", test_unknown_review_batch_loader_is_async_and_reports_progress),
    ("Unknown review Finish runs safety gate once", test_unknown_review_finish_runs_safety_gate_once),
    ("Unknown review action queue serializes and deduplicates", test_unknown_review_action_queue_serializes_and_deduplicates),
    ("Unknown review large clusters use safe queued chunks", test_unknown_review_large_cluster_is_chunked_safely),
    ("Unknown review confirmation uses recoverable pipeline", test_unknown_review_confirmation_uses_recoverable_pipeline),
    ("Unknown review exact duplicates reuse review destinations", test_unknown_review_exact_duplicates_share_review_destination),
    ("Impostor-aware thresholds tighten lookalikes", test_impostor_aware_thresholds_tighten_lookalikes),
    ("Legacy identity DB upgrades in memory", test_legacy_identity_db_is_upgraded_in_memory),
    ("Identity evaluation reports correct and rejected", test_identity_evaluation_reports_correct_and_rejected),
    ("Protected evaluation set gates verified coverage", test_protected_evaluation_set_requires_verified_coverage),
    ("Video person decisions accept one recognized frame", test_video_person_decisions_accept_one_recognized_frame),
    ("Video supporting evidence accepts one recognized frame", test_video_support_consensus_recovers_difficult_pose),
    ("Video fallback detector model is packaged", test_video_fallback_model_is_packaged),
    ("Video analysis uses memory-bounded workers", test_video_workers_are_memory_bounded),
    ("Image intake uses resumable memory-bounded workers", test_image_workers_are_memory_bounded_and_resumable),
    ("Image supervisor drains successive worker batches", test_image_supervisor_repeats_until_inbox_is_drained),
    ("Image supervisor stops safely on no progress", test_image_supervisor_stops_when_worker_makes_no_progress),
    ("Video matched and review destinations stay separate", test_video_destinations_keep_matches_and_review_separate),
    ("Video low-space handling pauses before moving matched originals", test_video_low_space_pauses_only_matched_copy),
    ("Video frame sampling matches a known identity", test_video_frame_sampling_matches_known_identity),
    ("Original copy is verified before source archive", test_original_copy_is_verified_before_source_archive),
    ("Daily step ordering is safe", test_daily_order_is_safe),
    ("Daily duplicate commands are report-only", test_daily_commands_are_non_destructive_for_duplicates),
    ("Preflight identifies active review workers", test_preflight_names_external_review_and_ignores_own_launcher),
    ("Daily empty-inbox gate is fast and media-aware", test_daily_empty_inbox_gate_is_fast_and_media_aware),
    ("Face main menu is streamlined", test_face_main_menu_is_streamlined),
    ("sort_photos post-process duplicate steps are report-only", test_sort_post_process_duplicate_steps_are_report_only),
    ("Simple rename preserves indices and flattens photos", test_simple_rename_plan_preserves_indices_and_flattens),
    ("Simple numbering repair is person-wide", test_simple_repair_assigns_unique_person_wide_indices),
    ("New output numbering is person-wide", test_new_output_numbering_is_person_wide_and_five_digits),
    ("Rename-plan relink preserves unchanged cache", test_rename_plan_relink_preserves_unchanged_cache_entries),
    ("Generated person views are excluded from scanners", test_generated_person_views_are_excluded_from_scanners),
    ("Generated contact sheets are detected without false positives", test_generated_contact_sheet_detector_is_precise),
    ("Historical contact-sheet hashes recover nonstandard layouts", test_generated_contact_sheet_ledger_hash_recovers_nonstandard_layout),
    ("Generated contact-sheet cleanup is recoverable", test_generated_contact_sheet_cleanup_is_recoverable),
    ("Generated intake sheets are diverted before face sorting", test_intake_generated_sheet_is_diverted_before_face_sorting),
    ("Person staging folders are excluded from library scanners", test_person_staging_folders_are_excluded_from_library_scanners),
    ("Source manifest restore uses operation ledger", test_source_manifest_restore_from_ledger),
    ("Source manifest restore dry-run is non-destructive", test_source_manifest_restore_dry_run_is_non_destructive),
    ("Source manifest accepts recoverable app trash only", test_source_manifest_accepts_recoverable_app_trash_only),
    ("Source manifest prefers active rename over stale trash", test_source_manifest_prefers_active_rename_over_stale_trash_copy),
    ("Person rehydrate preserves global cache", test_person_rehydrate_preserves_global_cache),
    ("Cache dry-run is non-destructive", test_full_rehydrate_keeps_cached_candidates),
    ("Cache coverage reports missing files", test_cache_coverage_reports_missing_files),
    ("Resume with no eligible clusters does not finalize", test_resume_with_no_eligible_clusters_does_not_finalize),
    ("Partial finish preserves library cache", test_partial_finish_preserves_library_cache_for_temp_sources),
    ("Changed cache signatures are refreshed", test_cache_signature_mismatch_is_not_preserved),
    ("Generated bad views are not recovered", test_generated_views_do_not_recover_bad_images),
    ("Duplicate matching accepts Pillow-readable images", test_duplicate_matching_accepts_pillow_readable_images),
    ("Near-visual candidates are never move actions", test_near_visual_candidates_are_never_move_actions),
    ("Duplicate review revalidates exact files and moves recoverably", test_duplicate_review_revalidates_and_moves_exact_files_recoverably),
    ("Duplicate review server serves and accepts keep", test_duplicate_review_server_serves_and_accepts_keep),
    ("Intake fingerprint accepts Pillow-readable images", test_intake_fingerprint_accepts_pillow_readable_images),
    ("SQLite analysis index reuses and invalidates fingerprints", test_sqlite_analysis_index_reuses_and_invalidates_fingerprints),
    ("Operation ledger SQLite mirror never blocks moves", test_operation_ledger_sqlite_mirror_never_blocks_moves),
    ("Operation ledger batch mirror is transactional", test_operation_ledger_batch_mirror_is_transactional),
    ("Shared asset analysis reads and decodes once", test_shared_asset_analysis_reads_and_decodes_once),
    ("Detection shares its fingerprint with later stages", test_detection_persists_shared_fingerprint_for_later_stages),
    ("SQLite migration CLI dispatches independently", test_sqlite_migration_cli_dispatches_without_people_dir),
    ("SQLite analysis index round-trips analysis history", test_sqlite_analysis_index_round_trips_analysis_history),
    ("SQLite detection replacement deduplicates face indices", test_sqlite_detection_replace_deduplicates_face_indices),
    ("Nudity check uses normalized fallback", test_nudity_check_uses_normalized_fallback),
    ("Intake preserves nude variant of existing normal", test_intake_preserves_nude_variant_of_existing_normal),
    ("Intake filters only same nudity category", test_intake_filters_only_same_nudity_category),
    ("Intake collapses same-category duplicates within a batch", test_intake_collapses_same_category_duplicates_within_batch),
    ("Cross-person identity audit is strict and report-only", test_cross_person_identity_audit_flags_without_move),
    ("Cross-person review supports keep, ignore, and manual correction", test_cross_person_review_keep_ignore_and_manual_correct),
    ("Cross-person discard is recoverable and UI actions exist", test_cross_person_review_discard_is_recoverable_and_ui_has_actions),
    ("Cross-person live review serves and accepts decisions", test_cross_person_review_server_serves_and_accepts_decisions),
    ("Perceptual dedup keeps cross-nudity variants", test_perceptual_dedup_keeps_cross_nudity_variants),
    ("Explicit nude copy stays separate from normal copy", test_explicit_nude_copy_does_not_reclassify_normal_copy),
    ("Nudity policy rejects swimwear false positives", test_nudity_policy_rejects_swimwear_false_positive),
    ("Nudity policy holds borderline detections for review", test_nudity_policy_holds_borderline_detection_for_review),
    ("Nudity policy does not assume detector misses are safe", test_nudity_policy_does_not_assume_missed_detection_is_safe),
    ("Nudity policy accepts strong explicit evidence", test_nudity_policy_accepts_strong_non_conflicting_detection),
    ("Nudity policy separates exposed and covered regions", test_nudity_policy_accepts_separate_exposed_and_covered_regions),
    ("Nudity policy preserves overlapping covered conflicts", test_nudity_policy_holds_overlapping_covered_conflict),
    ("Nudity policy ignores malformed detector scores", test_nudity_policy_ignores_malformed_detector_scores),
    ("Ambiguous nudity routes to uncertain review", test_ambiguous_explicit_detection_routes_to_uncertain_review),
    ("Saved preference routes uncertain nudity to nude", test_user_preference_routes_uncertain_nudity_to_nude),
    ("Bad-image audit accepts HEIF content with legacy suffixes", test_bad_image_audit_accepts_heif_with_legacy_suffix),
    ("Nudity and bad-image menu never auto-quarantine", test_nudity_and_bad_image_menu_never_auto_quarantine),
    ("Nude audit destinations leave confirmed files untouched", test_nude_audit_destinations_are_outside_nude_folder),
    ("Uncertain nudity only promotes confirmed files", test_uncertain_nudity_only_promotes_confirmed_files),
    ("Manual nudity confirmation survives move and rename", test_manual_nudity_confirmation_survives_move),
    ("Source guards protect uncertain nudity originals", test_source_guards_protect_uncertain_nudity_without_generated_review),
    ("Manual nudity scan uses the shared high-precision policy", test_manual_nudity_scan_uses_shared_high_precision_policy),
    ("Manual nudity scan decodes legacy-suffix HEIF", test_manual_nudity_scan_decodes_heif_with_legacy_jpg_suffix),
    ("Nudity placement rejects legacy low-threshold categories", test_nudity_report_placement_never_promotes_legacy_candidates),
    ("Manual nudity placement respects saved user preference", test_manual_nudity_placement_respects_saved_user_preference),
    ("Nudity review export cannot move protected originals", test_nudity_review_export_cannot_move_protected_original),
    ("Incremental smart albums skip heavy dry-run work", test_incremental_smart_albums_skip_without_heavy_models),
]


def run_test(name: str, func) -> Result:
    with tempfile.TemporaryDirectory(prefix="photo_pipeline_synthetic_") as td:
        tmp = Path(td)
        try:
            func(tmp)
            return Result(True, name, "passed")
        except Exception as exc:  # noqa: BLE001
            return Result(False, name, str(exc))


def main() -> int:
    print("Synthetic Integration Tests")
    print("=" * 60)
    results = [run_test(name, func) for name, func in TESTS]
    width = max(len(r.name) for r in results) if results else 1
    for result in results:
        level = "OK" if result.ok else "FAIL"
        print(f"[{level:4}] {result.name:<{width}}  {result.detail}")
    failures = sum(1 for r in results if not r.ok)
    print()
    print(f"Result: {failures} failure(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

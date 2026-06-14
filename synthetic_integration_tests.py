#!/usr/bin/env python3
"""
Fast synthetic integration tests for the photo pipeline.

These tests create a tiny temporary photo library and redirect cache files into
that temp folder. They are meant to catch cross-script regressions before a real
scan touches ~/Pictures.
"""

from __future__ import annotations

import contextlib
import io
import json
import subprocess
import sys
import tempfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import advanced_duplicate_matching  # noqa: E402
import build_smart_albums  # noqa: E402
import cache_tools  # noqa: E402
import cleanup_empty_person_folders  # noqa: E402
import daily_runner  # noqa: E402
import delete_person_folder_duplicates  # noqa: E402
import operation_ledger  # noqa: E402
import person_structure  # noqa: E402
import rename_person_folder_files  # noqa: E402
import source_manifest  # noqa: E402
import sort_photos  # noqa: E402

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


def test_daily_order_is_safe(tmp: Path) -> None:
    del tmp
    names = [step["name"] for step in daily_runner.step_list(50)]
    index = {name: i for i, name in enumerate(names)}
    required = [
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

    proc = subprocess.run(
        [sys.executable, str(SCRIPT_DIR / "sort_photos.py"), "--help"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    assert_true(proc.returncode == 0, f"sort_photos --help failed: {proc.stdout[-500:]}")
    assert_true("--skip-output-cleanup" in proc.stdout, "sort_photos does not accept --skip-output-cleanup")


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


def test_intake_fingerprint_accepts_pillow_readable_images(tmp: Path) -> None:
    fallback = tmp / "incoming" / "fallback.heic"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), (35, 140, 80)).save(fallback, format="GIF")

    stats: Counter = Counter()
    fp = sort_photos.image_duplicate_fingerprint(fallback, cache_entries={}, stats=stats)
    assert_true(fp is not None, "intake duplicate fingerprint rejected a Pillow-readable image")
    assert_true(stats.get("decode_errors", 0) == 0,
                f"intake duplicate fingerprint counted a decode error: {stats}")


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
    ("To Process generated-like names are scanned", test_to_process_generated_names_are_visible),
    ("Daily step ordering is safe", test_daily_order_is_safe),
    ("Daily duplicate commands are report-only", test_daily_commands_are_non_destructive_for_duplicates),
    ("sort_photos post-process duplicate steps are report-only", test_sort_post_process_duplicate_steps_are_report_only),
    ("Simple rename preserves indices and flattens photos", test_simple_rename_plan_preserves_indices_and_flattens),
    ("Generated person views are excluded from scanners", test_generated_person_views_are_excluded_from_scanners),
    ("Source manifest restore uses operation ledger", test_source_manifest_restore_from_ledger),
    ("Source manifest restore dry-run is non-destructive", test_source_manifest_restore_dry_run_is_non_destructive),
    ("Person rehydrate preserves global cache", test_person_rehydrate_preserves_global_cache),
    ("Cache dry-run is non-destructive", test_full_rehydrate_keeps_cached_candidates),
    ("Cache coverage reports missing files", test_cache_coverage_reports_missing_files),
    ("Resume with no eligible clusters does not finalize", test_resume_with_no_eligible_clusters_does_not_finalize),
    ("Partial finish preserves library cache", test_partial_finish_preserves_library_cache_for_temp_sources),
    ("Changed cache signatures are refreshed", test_cache_signature_mismatch_is_not_preserved),
    ("Generated bad views are not recovered", test_generated_views_do_not_recover_bad_images),
    ("Duplicate matching accepts Pillow-readable images", test_duplicate_matching_accepts_pillow_readable_images),
    ("Near-visual candidates are never move actions", test_near_visual_candidates_are_never_move_actions),
    ("Intake fingerprint accepts Pillow-readable images", test_intake_fingerprint_accepts_pillow_readable_images),
    ("Nudity check uses normalized fallback", test_nudity_check_uses_normalized_fallback),
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

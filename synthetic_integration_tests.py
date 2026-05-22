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
import daily_runner  # noqa: E402
import person_structure  # noqa: E402
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


def test_to_process_generated_names_are_visible(tmp: Path) -> None:
    inbox = tmp / "To Process"
    make_image(inbox / "all" / "a.jpg", (255, 0, 0))
    make_image(inbox / "review" / "b.jpg", (0, 255, 0))
    make_image(inbox / "_smart_albums" / "c.jpg", (0, 0, 255))
    make_image(inbox / "normal" / "d.jpg", (255, 255, 0))

    visible = list(sort_photos.iter_images(
        inbox,
        excluded_dir_names=set(),
        always_excluded_dir_names=set(),
    ))
    assert_true(len(visible) == 4, f"expected 4 visible inbox images, got {len(visible)}")

    count = daily_runner.count_images(inbox, exclude_generated_dirs=False)
    assert_true(count == 4, f"daily preview should see 4 images, got {count}")


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
        ("cache-rehydrate", "all-views"),
        ("cache-rehydrate", "smart-albums"),
        ("integration-audit", "status"),
    ]
    for before, after in required:
        assert_true(before in index, f"daily step missing: {before}")
        assert_true(after in index, f"daily step missing: {after}")
        assert_true(index[before] < index[after], f"{before} must run before {after}")


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


def test_intake_fingerprint_accepts_pillow_readable_images(tmp: Path) -> None:
    fallback = tmp / "incoming" / "fallback.heic"
    fallback.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (24, 24), (35, 140, 80)).save(fallback, format="GIF")

    stats: Counter = Counter()
    fp = sort_photos.image_duplicate_fingerprint(fallback, cache_entries={}, stats=stats)
    assert_true(fp is not None, "intake duplicate fingerprint rejected a Pillow-readable image")
    assert_true(stats.get("decode_errors", 0) == 0,
                f"intake duplicate fingerprint counted a decode error: {stats}")


def test_incremental_smart_albums_skip_without_heavy_models(tmp: Path) -> None:
    people = tmp / "people"
    person = people / "Person"
    make_image(person / "photos" / "Person_0001.jpg", (20, 120, 210))
    (person / "_smart_albums").mkdir(parents=True, exist_ok=True)
    (person / "_smart_albums" / "_smart_album_index.csv").write_text(
        "album,path\n",
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
    ("Person rehydrate preserves global cache", test_person_rehydrate_preserves_global_cache),
    ("Cache dry-run is non-destructive", test_full_rehydrate_keeps_cached_candidates),
    ("Generated bad views are not recovered", test_generated_views_do_not_recover_bad_images),
    ("Duplicate matching accepts Pillow-readable images", test_duplicate_matching_accepts_pillow_readable_images),
    ("Intake fingerprint accepts Pillow-readable images", test_intake_fingerprint_accepts_pillow_readable_images),
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

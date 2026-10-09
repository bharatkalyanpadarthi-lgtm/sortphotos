"""Daily efficiency regressions; all libraries, caches and reports are temporary."""

import contextlib
import io
import json
import os
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import analysis_index
import cache_tools
import content_identity
import daily_duplicates
import daily_inventory
import daily_runner
import face
import rename_person_folder_files as rename
import review_unknown_identities as review
import secondary_identity_matcher as secondary
import sort_photos as sorter
import unknown_triage
import source_manifest


class Fixtures(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="face-incremental-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.people = self.root / "people"
        self.people.mkdir()
        self.database = self.root / "inventory.sqlite3"

    def image(self, name="Alice/photos/Alice_00001.jpg", data=b"original"):
        path = self.people / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def baseline(self):
        entries = daily_inventory.capture(self.people)
        with daily_inventory.Inventory(self.database, self.people) as store:
            store.checkpoint("baseline", store.changes(entries))
            store.publish("baseline", entries, None)

    def changes(self):
        with daily_inventory.Inventory(self.database, self.people) as store:
            return store.changes(daily_inventory.capture(self.people))


class InventoryTests(Fixtures):
    def test_first_baseline_is_full_then_unchanged_is_empty(self):
        self.image()
        self.assertTrue(self.changes()["full"])
        self.baseline()
        self.assertFalse(self.changes()["full"])
        self.assertEqual(self.changes()["people"], [])

    def test_add_move_remove_and_external_change_are_scoped(self):
        original = self.image()
        self.image("Bob/photos/Bob_00001.jpg")
        self.baseline()
        moved = original.with_name("Alice_00002.jpg")
        original.rename(moved)
        result = self.changes()
        self.assertEqual(result["people"], ["Alice"])
        self.assertIn("Alice/photos/Alice_00001.jpg", result["removed"])
        self.assertIn("Alice/photos/Alice_00002.jpg", result["added"])

    def test_same_size_and_restored_mtime_replacement_is_detected(self):
        image = self.image()
        self.baseline()
        previous = image.stat()
        image.write_bytes(b"modified")
        os.utime(image, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        self.assertIn("Alice/photos/Alice_00001.jpg", self.changes()["changed"])

    def test_failed_run_does_not_publish_pending_changes(self):
        self.image()
        self.baseline()
        self.image("Bob/photos/Bob_00001.jpg")
        with daily_inventory.Inventory(self.database, self.people) as store:
            store.checkpoint("failed", self.changes())
        self.assertEqual(self.changes()["people"], ["Bob"])

    def test_unprocessed_external_changes_remain_pending_after_scoped_publish(self):
        self.image()
        self.image("Bob/photos/Bob_00001.jpg")
        self.baseline()
        self.image("Alice/photos/Alice_00002.jpg")
        self.image("Bob/photos/Bob_00002.jpg")
        with daily_inventory.Inventory(self.database, self.people) as store:
            store.publish("partial", daily_inventory.capture(self.people), {"Alice"})
        self.assertEqual(self.changes()["people"], ["Bob"])

    def test_removed_person_is_in_scope(self):
        original = self.image()
        self.baseline()
        original.unlink()
        original.parent.rmdir()
        original.parent.parent.rmdir()
        self.assertEqual(self.changes()["people"], ["Alice"])

    def test_policy_change_requires_full_reconciliation(self):
        self.image()
        self.baseline()
        with daily_inventory.Inventory(self.database, self.people, "new-model") as store:
            self.assertTrue(store.changes(daily_inventory.capture(self.people))["full"])

    def test_corrupt_index_is_preserved_and_falls_back_to_full(self):
        self.database.write_bytes(b"not sqlite")
        with contextlib.redirect_stdout(io.StringIO()), daily_inventory.open_inventory(
            self.database, self.people, "1"
        ) as store:
            self.assertTrue(store.changes(daily_inventory.capture(self.people))["full"])
        self.assertEqual(len(list(self.root.glob("inventory.sqlite3.corrupt.*"))), 1)

    def test_unavailable_storage_is_not_an_empty_library(self):
        with self.assertRaises(OSError):
            daily_inventory.capture(self.root / "disconnected")

    def test_future_inventory_schema_is_not_downgraded(self):
        self.baseline()
        with daily_inventory.Inventory(self.database, self.people) as store:
            store.db.execute("UPDATE metadata SET value='999' WHERE key='version'")
            store.db.commit()
            with self.assertRaisesRegex(ValueError, "newer version"):
                store.changes({})

    def test_scope_rejects_traversal_and_explicit_empty_stays_empty(self):
        scope = self.root / "scope.json"
        for value in [["../Alice"], ["/Alice"], ["A\\B"], [".hidden"], "Alice", [12]]:
            scope.write_text(json.dumps(value))
            with self.assertRaises(ValueError):
                daily_inventory.read_people_file(scope)
        scope.write_text("[]")
        self.assertEqual(daily_inventory.read_people_file(scope), set())

    def test_generated_artifacts_are_not_inventory_assets(self):
        self.image("Alice/_smart_albums/contact.jpg")
        entries = daily_inventory.capture(self.people)
        self.assertFalse(any("contact.jpg" in path for path in entries))


class DuplicateTests(Fixtures):
    def report(self, people=None):
        with contextlib.redirect_stdout(io.StringIO()):
            return daily_duplicates.refresh(self.people, self.root / "duplicates.csv", people,
                                             self.root / "hashes.sqlite3")

    def test_normal_and_nude_versions_are_separate(self):
        self.image()
        self.image("Alice/photos/nude/Alice_00002.jpg")
        self.image("Bob/photos/Bob_00001.jpg")
        self.assertEqual(self.report()["exact_candidates"], 0)

    def test_exact_duplicates_and_hardlinks_are_distinguished(self):
        first = self.image()
        self.image("Alice/photos/Alice_00002.jpg")
        link = self.people / "Alice/photos/Alice_00003.jpg"
        os.link(first, link)
        result = self.report()
        self.assertEqual(result["exact_candidates"], 1)
        self.assertEqual(result["already_hardlinked"], 1)
        self.assertTrue(first.exists())
        self.assertTrue(link.exists())

    def test_incremental_report_preserves_unaffected_people(self):
        for name in ("Alice", "Bob"):
            self.image(f"{name}/photos/{name}_00001.jpg", name.encode())
            self.image(f"{name}/photos/{name}_00002.jpg", name.encode())
        self.assertEqual(self.report()["exact_candidates"], 2)
        self.image("Alice/photos/Alice_00003.jpg", b"Alice")
        result = self.report({"Alice"})
        self.assertEqual(result["scanned"], 3)
        self.assertEqual(result["exact_candidates"], 3)

    def test_unchanged_report_uses_indexed_hashes_without_decoding(self):
        self.image()
        self.report()
        with patch.object(content_identity, "content_sha256", side_effect=AssertionError("rehash")), \
             patch.object(daily_duplicates.matching, "imread", side_effect=AssertionError("decode")):
            self.report()

    def test_changed_bytes_do_not_reuse_hash_with_restored_mtime(self):
        first = self.image()
        self.image("Alice/photos/Alice_00002.jpg")
        self.assertEqual(self.report()["exact_candidates"], 1)
        previous = first.stat()
        first.write_bytes(b"modified")
        os.utime(first, ns=(previous.st_atime_ns, previous.st_mtime_ns))
        self.assertEqual(self.report({"Alice"})["exact_candidates"], 0)

    def test_missing_report_rebuilds_all_people_not_only_scope(self):
        self.image()
        self.image("Bob/photos/Bob_00001.jpg")
        self.assertEqual(self.report({"Alice"})["scanned"], 2)

    def test_damaged_report_rebuilds_instead_of_preserving_invalid_rows(self):
        self.image()
        self.image("Bob/photos/Bob_00001.jpg")
        report = self.root / "duplicates.csv"
        for text in ("scope,group_id,file_path,action,type\nBob,1,path,move,exact_file\n",
                     "scope,group_id,file_path,action,type,keeper_path,size_bytes\nBob,1,path,move,exact_file,keep,invalid\n"):
            report.write_text(text)
            self.assertEqual(self.report({"Alice"})["scanned"], 2)

    def test_report_records_hash_hits_and_misses(self):
        self.image()
        first = self.report()
        second = self.report()
        self.assertEqual((first["hash_hits"], first["hash_misses"]), (0, 1))
        self.assertEqual((second["hash_hits"], second["hash_misses"]), (1, 0))


class CacheReuseTests(Fixtures):
    def setUp(self):
        super().setUp()
        self.cache = sorter.CacheState(config_fingerprint="detector-v1")
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        self.stack.enter_context(patch.object(sorter, "config_fingerprint", return_value="detector-v1"))
        self.stack.enter_context(patch.object(sorter, "load_cache", side_effect=lambda: self.cache))
        self.save = self.stack.enter_context(patch.object(cache_tools, "save_cache_with_backup", side_effect=self.saved))
        self.worker = self.stack.enter_context(patch.object(cache_tools.subprocess, "run", side_effect=AssertionError("unexpected detection")))

    def saved(self, cache, backup):
        self.cache = cache
        return self.root / "backup.pkl"

    def cached_face(self, path, index=0, label=None, quality=0.8):
        return sorter.CachedFace(src_str=str(path), face_index=index, det_score=.99, bbox_size=100,
            sharpness=100, yaw_proxy=0, quality=quality, embedding=np.eye(2, dtype=np.float32)[index % 2],
            image_phash=np.zeros(64, dtype=bool), crop_jpeg=b"crop" + str(index).encode(), label=label)

    def seed_index(self, path, faces, status="accepted_face", config="detector-v1"):
        with analysis_index.AnalysisIndex(self.root / "analysis.sqlite3") as index:
            index.replace_detections(path, config, status, [sorter.cached_face_to_index_record(face) for face in faces])

    def refresh(self, replace=False, **kwargs):
        with contextlib.redirect_stdout(io.StringIO()):
            return cache_tools.rehydrate(self.people, None, True, replace, None, 1,
                                          index_path=self.root / "analysis.sqlite3", **kwargs)

    def test_unchanged_cache_never_saves_or_backs_up(self):
        path = self.image()
        self.cache.file_signatures[str(path)] = sorter.file_signature(path)
        self.cache.faces.append(self.cached_face(path, label="Alice"))
        self.assertEqual(self.refresh(), 0)
        self.save.assert_not_called()
        self.worker.assert_not_called()

    def test_verified_copy_reuses_detection_even_when_source_has_gone(self):
        source = self.root / "source.jpg"
        source.write_bytes(b"original")
        self.seed_index(source, [self.cached_face(source)])
        source.unlink()
        destination = self.image()
        self.assertEqual(self.refresh(), 0)
        self.worker.assert_not_called()
        self.save.assert_called_once()
        self.assertEqual(self.cache.faces[0].src_str, str(destination))
        self.assertEqual(self.cache.faces[0].label, "Alice")

    def test_selected_group_face_is_preserved_across_rename(self):
        source = self.root / "selected.jpg"
        source.write_bytes(b"original")
        self.seed_index(source, [self.cached_face(source, 1, "Alice", 0.5)], "organized_face")
        destination = self.image()
        self.refresh()
        self.assertEqual(self.cache.faces[0].face_index, 1)
        self.assertEqual(self.cache.faces[0].crop_jpeg, b"crop1")
        self.worker.assert_not_called()

    def test_ambiguous_multiface_analysis_is_not_reused_as_a_single_identity(self):
        source = self.root / "group.jpg"
        source.write_bytes(b"original")
        self.seed_index(source, [self.cached_face(source, 0), self.cached_face(source, 1)])
        self.image()
        self.worker.side_effect = None
        self.worker.return_value = SimpleNamespace(returncode=9, stdout="fixture failure")
        self.assertEqual(self.refresh(), 9)
        self.save.assert_not_called()

    def test_no_face_analysis_is_reused_without_model_loading(self):
        path = self.image()
        self.seed_index(path, [], "no_usable_face")
        self.assertEqual(self.refresh(), 0)
        self.assertIn(str(path), self.cache.file_signatures)
        self.worker.assert_not_called()

    def test_missing_face_records_are_not_mistaken_for_no_face_result(self):
        path = self.image()
        self.seed_index(path, [], "accepted_face")
        self.worker.side_effect = None
        self.worker.return_value = SimpleNamespace(returncode=9, stdout="fixture failure")
        self.assertEqual(self.refresh(), 9)
        self.save.assert_not_called()

    def test_replaced_content_during_reuse_never_receives_old_detections(self):
        path = self.image()
        self.seed_index(path, [self.cached_face(path)])
        original = analysis_index.AnalysisIndex.cached_detections
        def replace_after_lookup(index, source, config):
            result = original(index, source, config)
            source.write_bytes(b"different bytes")
            return result
        with patch.object(analysis_index.AnalysisIndex, "cached_detections", replace_after_lookup):
            with self.assertRaisesRegex(RuntimeError, "Source changed"):
                self.refresh()
        self.save.assert_not_called()
        with analysis_index.AnalysisIndex(self.root / "analysis.sqlite3") as index:
            self.assertIsNone(index.cached_detections(path, "detector-v1"))

    def test_changed_detector_requires_fresh_analysis(self):
        path = self.image()
        self.seed_index(path, [self.cached_face(path)], config="old-model")
        self.worker.side_effect = None
        self.worker.return_value = SimpleNamespace(returncode=9, stdout="fixture failure")
        self.assertEqual(self.refresh(), 9)
        self.worker.assert_called_once()

    def test_scoped_refresh_preserves_other_person_cache(self):
        alice = self.image()
        bob = self.image("Bob/photos/Bob_00001.jpg")
        self.cache.file_signatures[str(bob)] = sorter.file_signature(bob)
        self.cache.faces = [self.cached_face(bob, label="Bob")]
        self.seed_index(alice, [self.cached_face(alice)])
        self.refresh(people={"Alice"})
        self.assertEqual(set(self.cache.file_signatures), {str(alice), str(bob)})
        self.assertEqual({face.label for face in self.cache.faces}, {"Alice", "Bob"})

    def test_scoped_replace_preserves_other_person_cache(self):
        alice = self.image()
        bob = self.image("Bob/photos/Bob_00001.jpg")
        self.cache.file_signatures[str(bob)] = sorter.file_signature(bob)
        self.cache.faces = [self.cached_face(bob, label="Bob")]
        self.seed_index(alice, [self.cached_face(alice)])
        self.refresh(replace=True, people={"Alice"})
        self.assertEqual(set(self.cache.file_signatures), {str(alice), str(bob)})

    def test_organized_copy_records_selected_face_for_rehydration(self):
        source = self.root / "intake.jpg"
        source.write_bytes(b"original")
        record = sorter.FaceRecord(source, 1, .99, 120, 120, 0, np.eye(2, dtype=np.float32)[1],
                                  quality=1, cluster_id=1, crop_jpeg=b"selected face")
        record.content_sha256 = content_identity.content_sha256(source)
        with patch.object(sorter, "analysis_index_file", return_value=self.root / "analysis.sqlite3"), \
             patch.object(sorter, "DEDUP_DUPLICATES", False), \
             patch.object(sorter, "classified_nudity_status", return_value="safe"), \
             patch.object(sorter, "maybe_move_to_nudity_subfolder", side_effect=lambda path, *_a, **_kw: (path, None)):
            organized = sorter.organize_originals([record], {1: "Alice"}, self.people)
        self.assertIn(source, organized)
        self.assertEqual(self.refresh(), 0)
        self.worker.assert_not_called()
        self.assertEqual(self.cache.faces[0].face_index, 1)
        self.assertEqual(self.cache.faces[0].crop_jpeg, b"selected face")

    def test_removed_person_prunes_stale_entries_without_dropping_other_people(self):
        bob = self.image("Bob/photos/Bob_00001.jpg")
        gone = self.people / "Alice/photos/Alice_00001.jpg"
        self.cache.file_signatures = {str(gone): (0, 1), str(bob): sorter.file_signature(bob)}
        self.cache.faces = [self.cached_face(gone, label="Alice"), self.cached_face(bob, label="Bob")]
        self.refresh(people={"Alice"})
        self.assertEqual(set(self.cache.file_signatures), {str(bob)})

    def write_worker_output(self, command, **kwargs):
        job = pickle.loads(Path(command[-1]).read_bytes())
        paths = [Path(path) for path in job["input_paths"]]
        fingerprints = {str(path): {"sha256": content_identity.content_sha256(path), "pixel_sha256": "",
                                   "phash": 0, "width": 1, "height": 1} for path in paths}
        Path(job["output_path"]).write_bytes(pickle.dumps({"faces": [self.cached_face(path) for path in paths],
            "diagnostics": {}, "fingerprints": fingerprints}))
        return SimpleNamespace(returncode=0)

    def test_multiple_detection_batches_write_one_compatibility_snapshot(self):
        self.image()
        self.image("Alice/photos/Alice_00002.jpg", b"second")
        self.worker.side_effect = self.write_worker_output
        self.refresh()
        self.assertEqual(self.worker.call_count, 2)
        self.save.assert_called_once()

    def test_interrupted_detection_resumes_completed_sqlite_batch(self):
        self.image()
        self.image("Alice/photos/Alice_00002.jpg", b"second")
        calls = []
        def interrupt(command, **kwargs):
            calls.append(command)
            return self.write_worker_output(command, **kwargs) if len(calls) == 1 else SimpleNamespace(returncode=9, stdout="interrupted")
        self.worker.side_effect = interrupt
        self.assertEqual(self.refresh(), 9)
        self.save.assert_not_called()
        self.worker.reset_mock()
        self.worker.side_effect = self.write_worker_output
        self.assertEqual(self.refresh(), 0)
        self.worker.assert_called_once()
        self.save.assert_called_once()

    def test_unreadable_worker_result_does_not_claim_success_or_cache_no_face(self):
        image = self.image()
        def unreadable(command, **kwargs):
            job = pickle.loads(Path(command[-1]).read_bytes())
            Path(job["output_path"]).write_bytes(pickle.dumps({"faces": [],
                "diagnostics": {str(image): "unreadable: fixture"}}))
            return SimpleNamespace(returncode=0)
        self.worker.side_effect = unreadable
        self.assertEqual(self.refresh(), 2)
        self.assertNotIn(str(image), self.cache.file_signatures)


class PlanningTests(Fixtures):
    def test_photo_and_video_stages_skip_independently(self):
        inbox = self.root / "inbox"
        inbox.mkdir()
        (inbox / "photo.jpg").write_bytes(b"photo")
        with patch.object(daily_runner, "TO_PROCESS", inbox), patch.object(daily_runner, "LEGACY_VIDEO_INBOX", self.root / "videos"):
            self.assertIsNone(daily_runner.skip_reason({"name": "process"}, {}))
            self.assertEqual(daily_runner.skip_reason({"name": "video-process"}, {}), "no new videos")
            (inbox / "photo.jpg").unlink()
            (inbox / "video.mov").write_bytes(b"video")
            self.assertEqual(daily_runner.skip_reason({"name": "process"}, {}), "no new photos")
            self.assertIsNone(daily_runner.skip_reason({"name": "video-process"}, {}))

    def test_scoped_commands_and_no_change_skips(self):
        state = {"incremental": {"full": False, "people": [], "scope_path": "scope.json"}}
        step = {"name": "structure", "cmd": ["python", "person_structure.py"]}
        self.assertIn("no changed", daily_runner.skip_reason(step, state))
        self.assertEqual(daily_runner.scoped_command(step, state)[-2:], ["--people-file", "scope.json"])
        state["incremental"]["full"] = True
        self.assertIsNone(daily_runner.skip_reason(step, state))
        self.assertEqual(daily_runner.scoped_command(step, state), step["cmd"])

    def test_duplicate_pass_is_single_report_only_engine(self):
        steps = daily_runner.step_list(50)
        self.assertNotIn("exact-dedupe", [step["name"] for step in steps])
        duplicate = next(step for step in steps if step["name"] == "advanced-dedupe")
        self.assertTrue(duplicate["cmd"][1].endswith("daily_duplicates.py"))
        self.assertNotIn("--apply", duplicate["cmd"])
        self.assertFalse(duplicate["mutates"])

    def test_summary_preserves_timings_skips_and_nested_counts(self):
        state = {"run_id": "fixture", "started_at": 1, "steps": {"video-process": {"status": "skipped", "reason": "no new videos"}},
                 "timings_seconds": {"process": 12.5}, "safety_timings_seconds": {"process": .5}}
        path = self.root / "summary.json"
        daily_runner.write_summary(path, state, {"person_counts": {"Alice": 1}},
                                   {"person_counts": {"Alice": 2}}, "completed")
        saved = json.loads(path.read_text())
        self.assertEqual(saved["timings_seconds"], {"process": 12.5})
        self.assertEqual(saved["safety_timings_seconds"], {"process": .5})
        self.assertEqual(saved["steps"]["video-process"]["reason"], "no new videos")

    def test_diagnostics_records_only_current_step_counters(self):
        log = self.root / "worker.log"
        log.write_bytes(b"Verified detections reused: 99\n")
        offset = log.stat().st_size
        with log.open("ab") as handle:
            handle.write(b"Indexed content hashes: 50 reused; 2 newly hashed\n"
                         b"Verified detections reused: 3\nSafety benchmark reason: profiles changed\n")
        diagnostics = daily_runner.worker_diagnostics(log, offset)
        self.assertEqual(diagnostics["counters"], {"hash_hits": 50, "hash_misses": 2, "reused_detections": 3})
        self.assertIn("profiles changed", diagnostics["benchmark"][0])

    def test_unchanged_rename_does_not_validate_or_promote_again(self):
        self.image()
        with patch("sys.argv", ["rename", str(self.people), "--simple", "--apply", "--quiet"]), \
             patch.object(rename.source_manifest, "validate_current", side_effect=AssertionError("redundant scan")), \
             patch.object(rename.source_manifest, "promote_current", side_effect=AssertionError("redundant promotion")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(rename.main(), 0)

    def test_daily_launcher_delegates_manifest_ownership_to_runner(self):
        action = next(action for action in face.ACTIONS if action["key"] == "daily")
        with patch.object(face, "source_review_storage_check", return_value=0), \
             patch.object(face.daily_runner, "intake_has_media", return_value=True), \
             patch.object(face, "cache_guard_check", return_value=0), \
             patch.object(face, "run_steps", return_value=0) as run, \
             patch.object(face.source_manifest, "validate_current", side_effect=AssertionError("duplicate wrapper guard")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(face._run_action(action), 0)
        run.assert_called_once()

    def test_legacy_report_stamp_invalidates_when_source_is_replaced(self):
        source = self.root / "labels.pkl"
        source.write_bytes(b"old")
        output = self.root / "triage"
        output.mkdir()
        (output / "unknown_triage.csv").write_text("header")
        (output / "unknown_triage.html").write_text("report")
        (output / "input_signature.json").write_text(json.dumps(unknown_triage.input_signature(source)))
        self.assertTrue(unknown_triage.report_current(source, output))
        source.write_bytes(b"new")
        self.assertFalse(unknown_triage.report_current(source, output))

    def test_legacy_report_changed_during_render_is_not_marked_current(self):
        source = self.root / "labels.pkl"
        source.write_bytes(b"before")
        output = self.root / "triage"
        output.mkdir()
        with patch("sys.argv", ["unknown-triage", "--state", str(source), "--output-dir", str(output)]), \
             patch.object(unknown_triage, "load_state", return_value={}), \
             patch.object(unknown_triage, "build_clusters", return_value=[]), \
             patch.object(unknown_triage, "write_html", side_effect=lambda *args: source.write_bytes(b"after")), \
             contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(unknown_triage.main(), 0)
        self.assertFalse((output / "input_signature.json").exists())

    def test_preview_skips_photo_and_video_stages_independently(self):
        before = {"to_process_images": 1, "organized_images": 1}
        with patch.object(daily_runner, "cleanup_holding_count", return_value=0), \
             contextlib.redirect_stdout(io.StringIO()) as output:
            daily_runner.print_dry_run(daily_runner.step_list(50), before,
                {"message": "normal", "batch_size": 50}, False)
        self.assertIn("skip: no new videos", output.getvalue())
        self.assertIn("changed folders only", output.getvalue())
        self.assertIn("included in final summary", output.getvalue())

    def test_verified_manifest_counts_match_photos_only_guard(self):
        self.image()
        self.image("Alice/photos/nude/Alice_00002.jpg", b"second")
        self.image("Alice/review/uncertain_nudity/Alice_00003.jpg", b"third")
        (self.people / "Bob").mkdir()
        entries = source_manifest.collect_entries(self.people)
        with patch.object(daily_runner, "PEOPLE", self.people), \
             patch.object(daily_runner, "original_person_counts", side_effect=AssertionError("extra scan")):
            self.assertEqual(daily_runner.verified_original_counts(SimpleNamespace(current_entries=entries)),
                             {"Alice": 2, "Bob": 0})

    def test_benchmark_signature_ignores_ui_but_tracks_policy_components(self):
        db = sorter.IdentityDB()
        original = content_identity.content_sha256
        def policy_only(path):
            self.assertNotEqual(Path(path).name, "review_unknown_ui.py")
            return original(path)
        parts = {}
        with patch.object(content_identity, "content_sha256", side_effect=policy_only):
            first = review.automatic_review_gate_signature(db, secondary_db=secondary.SecondaryIdentityDB(), components=parts)
            second = review.automatic_review_gate_signature(db, secondary_db=secondary.SecondaryIdentityDB())
        self.assertEqual(first, second)
        self.assertIn("automatic matching safety thresholds", parts)
        self.assertIn("recognition code: recognition_policy.py", parts)


class DailyWorkflowTests(Fixtures):
    """Exercise real orchestration and original protection, using fixture workers."""

    def setUp(self):
        super().setUp()
        self.sorted = self.root / "sorted"
        self.people = self.sorted / "photos_by_person"
        self.people.mkdir(parents=True)
        self.inbox = self.root / "inbox"
        self.inbox.mkdir()
        self.review = self.sorted / "_source_review"
        self.manifest_path = self.root / "protected.json"
        self.summary = self.review / "daily_run_summaries"
        self.stack = contextlib.ExitStack()
        self.addCleanup(self.stack.close)
        values = {"SORTED": self.sorted, "PEOPLE": self.people, "SOURCE_REVIEW": self.review,
                  "READY": self.review / "ready_to_delete", "TO_PROCESS": self.inbox,
                  "LEGACY_VIDEO_INBOX": self.root / "videos", "STATE_FILE": self.root / "state.json",
                  "SUMMARY_DIR": self.summary, "ADV_REPORT": self.review / "duplicates.csv"}
        for key, value in values.items():
            self.stack.enter_context(patch.object(daily_runner, key, value))
        self.stack.enter_context(patch.object(daily_runner, "labeling_remaining", return_value={"clusters": 0, "faces": 0}))
        self.stack.enter_context(patch.object(daily_runner, "memory_profile", return_value={"ok": True,
            "message": "normal", "available_mb": 8192, "batch_size": 50}))
        self.stack.enter_context(patch.object(daily_runner, "inventory_policy", return_value="fixture-policy"))
        self.stack.enter_context(patch.object(daily_runner, "intake_has_media",
            side_effect=lambda: daily_runner.tree_contains_media(self.inbox, daily_runner.IMAGE_EXTS)))
        self.stack.enter_context(patch.object(unknown_triage, "DEFAULT_STATE", self.root / "no-legacy.pkl"))
        validate, promote = source_manifest.validate_current, source_manifest.promote_current
        self.validate = self.stack.enter_context(patch.object(source_manifest, "validate_current",
            side_effect=lambda **kwargs: validate(**kwargs, manifest_path=self.manifest_path,
                                                  report_dir=self.root / "manifest_reports")))
        self.promote = self.stack.enter_context(patch.object(source_manifest, "promote_current",
            side_effect=lambda **kwargs: promote(**kwargs, manifest_path=self.manifest_path)))
        self.worker = self.stack.enter_context(patch.object(daily_runner, "run_command", side_effect=self.run_worker))
        self.calls = []
        self.fail_process = False
        self.remove_at_audit = None
        self.sequence = 0
        self.stack.enter_context(patch.object(daily_runner, "run_id", side_effect=self.next_id))

    def next_id(self):
        self.sequence += 1
        return f"fixture-{self.sequence}"

    def protect(self):
        source_manifest.save_manifest(source_manifest.build_manifest(self.people), self.manifest_path)

    def run_worker(self, command, log_path, *, verbose, step_name):
        scope = daily_inventory.read_people_file(Path(command[-1])) if "--people-file" in command else None
        self.calls.append((step_name, scope))
        if step_name == "process":
            if self.fail_process:
                return 9
            for path in self.inbox.glob("*.jpg"):
                person = path.stem.split("_", 1)[0]
                destination = self.people / person / "photos" / path.name
                destination.parent.mkdir(parents=True, exist_ok=True)
                path.rename(destination)
        elif step_name == "advanced-dedupe":
            daily_duplicates.refresh(self.people, daily_runner.ADV_REPORT, scope, self.root / "hashes.sqlite3")
        elif step_name == "integration-audit" and self.remove_at_audit:
            self.remove_at_audit.unlink()
        return 0

    def run_daily(self, *args):
        with patch("sys.argv", ["daily", *args]), contextlib.redirect_stdout(io.StringIO()):
            return daily_runner.main()

    def test_full_baseline_then_single_person_incremental_run(self):
        original = self.image()
        self.protect()
        (self.inbox / "Alice_00002.jpg").write_bytes(b"second")
        self.assertEqual(self.run_daily(), 0)
        self.assertFalse(daily_runner.STATE_FILE.exists())
        self.assertTrue(original.exists())
        self.calls.clear()
        (self.inbox / "Bob_00001.jpg").write_bytes(b"third")
        self.assertEqual(self.run_daily(), 0)
        self.assertNotIn("video-process", [name for name, _scope in self.calls])
        scopes = [scope for name, scope in self.calls if name in daily_runner.SCOPED_STEPS]
        self.assertTrue(scopes)
        self.assertTrue(all(scope == {"Bob"} for scope in scopes))
        self.assertEqual(daily_runner.original_person_counts(), {"Alice": 2, "Bob": 1})
        saved = json.loads((self.summary / "daily_run_fixture-2.json").read_text())
        self.assertEqual(saved["incremental"]["people"], ["Bob"])
        self.assertTrue(saved["timings_seconds"])

    def test_failed_worker_never_publishes_baseline_and_resume_completes(self):
        self.image()
        self.protect()
        (self.inbox / "Alice_00002.jpg").write_bytes(b"second")
        self.fail_process = True
        self.assertEqual(self.run_daily(), 9)
        with daily_inventory.Inventory(self.root / "daily_inventory.sqlite3", self.people, "fixture-policy") as store:
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM baseline").fetchone()[0], 0)
        self.assertTrue(daily_runner.STATE_FILE.exists())
        self.fail_process = False
        self.assertEqual(self.run_daily("--resume"), 0)
        self.assertFalse(daily_runner.STATE_FILE.exists())
        self.assertEqual(daily_runner.original_person_counts(), {"Alice": 2})

    def test_final_guard_detects_external_loss_after_readonly_stage(self):
        original = self.image()
        self.protect()
        (self.inbox / "Alice_00002.jpg").write_bytes(b"second")
        self.remove_at_audit = original
        with contextlib.redirect_stderr(io.StringIO()):
            self.assertEqual(self.run_daily(), daily_runner.SOURCE_GUARD_EXIT)
        self.promote.assert_not_called()
        self.assertTrue(daily_runner.STATE_FILE.exists())
        with daily_inventory.Inventory(self.root / "daily_inventory.sqlite3", self.people, "fixture-policy") as store:
            self.assertEqual(store.db.execute("SELECT COUNT(*) FROM baseline").fetchone()[0], 0)


if __name__ == "__main__":
    unittest.main()

"""Protected examples survive intake moves without changing human annotations."""

import csv
import os
import pickle
import sqlite3
import tempfile
import threading
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import benchmark_detection
import content_identity
import evaluation_dataset as dataset
import evaluation_runtime as runtime
import identity_evaluation as evaluation
import protected_benchmark_assets as assets
import secondary_identity_matcher as secondary
import sort_photos as sorter


class ProtectedBenchmarkAssetsTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "unassigned_intake" / "unknown_identity" / "image.jpg"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"verified unknown image")
        self.digest = content_identity.content_sha256(self.source)
        self.csv = self.root / "benchmark.csv"
        self.row = dict(source=str(self.source), expected_person="", case_types="unknown",
            expected_face="true", expected_nudity="unknown", verified="true", notes="human label",
            content_sha256=self.digest, group_id="source batch", identity_face_id="")
        dataset.write_template(self.csv, [self.row])
        self.target = assets.asset_path(self.csv, self.source, self.digest)

    def test_copy_is_separate_and_keeps_csv_baseline_binding_and_metadata(self):
        before = self.csv.read_bytes()
        self.assertEqual(assets.pin_dataset(self.csv), {"protected": 1, "unresolved": []})
        self.assertNotEqual(self.source.stat().st_ino, self.target.stat().st_ino)
        self.assertEqual(self.source.stat().st_mtime_ns, self.target.stat().st_mtime_ns)
        self.assertEqual(self.csv.read_bytes(), before)
        case = dataset.load_dataset(self.csv).cases[0]
        self.assertEqual((case.source, case.original_source), (self.target, self.source))
        self.assertEqual((case.group_id, case.notes, case.content_sha256),
                         ("source batch", "human label", self.digest))

    def test_move_keeps_verified_example_and_labels_without_restoring_intake_file(self):
        assets.pin_dataset(self.csv)
        self.source.rename(self.root / "archived.jpg")
        validation = dataset.load_dataset(self.csv)
        self.assertEqual(validation.errors, ())
        self.assertEqual(validation.cases[0].source, self.target)
        self.assertFalse(self.source.exists())

    def test_replaced_original_is_not_silently_masked_by_protected_copy(self):
        assets.pin_dataset(self.csv)
        self.source.write_bytes(b"different image")
        validation = dataset.load_dataset(self.csv)
        self.assertTrue(any("content changed" in item for item in validation.errors))
        self.assertTrue(assets.pin_dataset(self.csv)["unresolved"])
        self.assertEqual(content_identity.content_sha256(self.target), self.digest)

    def test_corrupt_copy_is_rejected_and_never_overwritten(self):
        assets.pin_dataset(self.csv)
        self.target.write_bytes(b"corrupt")
        with self.assertRaisesRegex(ValueError, "corrupt"):
            assets.pin(self.csv, self.source, self.digest)
        self.assertTrue(dataset.load_dataset(self.csv).errors)
        self.assertEqual(self.target.read_bytes(), b"corrupt")

    def test_wrong_recovery_candidate_cannot_be_published(self):
        other = self.root / "other.jpg"
        other.write_bytes(b"not the same image")
        self.source.unlink()
        result = assets.pin_dataset(self.csv, replacements={self.digest: other})
        self.assertTrue(result["unresolved"])
        self.assertFalse(self.target.exists())
        self.assertEqual(other.read_bytes(), b"not the same image")

    def test_interruption_does_not_expose_partial_copy(self):
        with patch.object(assets.shutil, "copy2", side_effect=OSError("interrupted")):
            self.assertTrue(assets.pin_dataset(self.csv)["unresolved"])
        self.assertFalse(self.target.exists())
        self.assertEqual(list(self.target.parent.iterdir()), [])
        self.assertEqual(content_identity.content_sha256(self.source), self.digest)

    def test_changed_source_during_copy_is_rejected(self):
        real_copy = assets.shutil.copy2
        def changed(source, target):
            result = real_copy(source, target)
            Path(source).write_bytes(b"changed concurrently")
            return result
        with patch.object(assets.shutil, "copy2", side_effect=changed):
            self.assertTrue(assets.pin_dataset(self.csv)["unresolved"])
        self.assertFalse(self.target.exists())

    def test_unverified_and_hashless_rows_do_not_gain_trust(self):
        for update in ({"verified": "false"}, {"content_sha256": ""},
                       {"content_sha256": "../../escape"}):
            with self.subTest(update=update):
                dataset.write_template(self.csv, [{**self.row, **update}])
                assets.pin_dataset(self.csv)
                self.assertFalse(self.target.exists())

    def test_missing_invalid_and_incomplete_inputs_fail_before_any_scoring(self):
        baseline = self.root / "baseline.json"
        baseline.write_text("{}")
        db = sorter.IdentityDB()
        for mode in ("missing", "changed", "incomplete"):
            with self.subTest(mode=mode):
                self.source.write_bytes(b"verified unknown image")
                dataset.write_template(self.csv, [self.row])
                if mode == "missing":
                    self.source.unlink()
                elif mode == "changed":
                    self.source.write_bytes(b"replacement")
                with patch.object(evaluation, "cache_metrics") as scoring, \
                     patch.object(evaluation.benchmark_inputs, "prepare_selected_faces") as detect:
                    allowed, report = evaluation.activation_gate(db, db, sorter.CacheState(),
                        protected_set=self.csv, protected_baseline=baseline, progress=lambda _: None)
                self.assertFalse(allowed)
                self.assertTrue(report["candidate"]["skipped"])
                scoring.assert_not_called()
                detect.assert_not_called()

    def test_stale_baseline_blocks_profile_promotion_before_scoring(self):
        baseline = self.root / "baseline.json"
        metrics = dataset.EvaluationMetrics(1, 1, 1, 0, 1, 1, 0, 1)
        dataset.write_baseline(baseline, metrics, detector_signature="old detector",
                               dataset_sha256="old dataset")
        validation = dataset.DatasetValidation((), (), frozenset(dataset.REQUIRED_CASE_TYPES))
        with patch.object(dataset, "load_dataset", return_value=validation), \
             patch.object(evaluation, "cache_metrics") as scoring, \
             patch.object(evaluation.benchmark_inputs, "prepare_selected_faces") as detect:
            allowed, report = evaluation.activation_gate(sorter.IdentityDB(), sorter.IdentityDB(),
                sorter.CacheState(), protected_set=self.csv, protected_baseline=baseline,
                progress=lambda _: None)
        self.assertFalse(allowed)
        self.assertTrue(any("different dataset" in error for error in report["failures"]))
        self.assertTrue(any("different detector" in error for error in report["failures"]))
        scoring.assert_not_called()
        detect.assert_not_called()

    def test_original_path_still_selects_a_protected_case_for_diagnostic_exclusion(self):
        assets.pin_dataset(self.csv)
        self.source.unlink()
        case = dataset.load_dataset(self.csv).cases[0]
        other = replace(case, source=self.root / "other.jpg", original_source=None,
                        content_sha256="other hash", group_id="other group")
        retained, excluded = evaluation.split_excluded_cases((case, other), [self.source])
        self.assertEqual(retained, (other,))
        self.assertEqual(excluded, (case,))

    def test_non_object_baseline_is_reported_without_scoring_or_crashing(self):
        baseline = self.root / "baseline.json"
        baseline.write_text("[]")
        validation = dataset.DatasetValidation((), (), frozenset(dataset.REQUIRED_CASE_TYPES))
        with patch.object(dataset, "load_dataset", return_value=validation), \
             patch.object(evaluation, "cache_metrics") as scoring:
            allowed, report = evaluation.activation_gate(sorter.IdentityDB(), sorter.IdentityDB(),
                sorter.CacheState(), protected_set=self.csv, protected_baseline=baseline,
                progress=lambda _: None)
        self.assertFalse(allowed)
        self.assertTrue(any("baseline must be an object" in error for error in report["failures"]))
        scoring.assert_not_called()

    def test_verified_hash_recovery_ignores_stale_index_and_finds_archive(self):
        index = self.root / "index.sqlite3"
        archived = self.root / "archive"
        archived.mkdir()
        recovered = archived / "newname.png"
        self.source.rename(recovered)
        wrong = self.root / "wrong.jpg"
        wrong.write_bytes(b"wrong index content")
        with sqlite3.connect(index) as connection:
            connection.execute("CREATE TABLE assets(path TEXT, sha256 TEXT, byte_size INTEGER)")
            connection.execute("CREATE TABLE content_versions(path TEXT, sha256 TEXT)")
            connection.execute("INSERT INTO assets VALUES(?,?,?)",
                (str(wrong), self.digest, recovered.stat().st_size))
        matches = assets.recover_candidates(self.csv, index, [archived])
        self.assertEqual(matches, {self.digest: recovered})
        self.assertEqual(assets.pin_dataset(self.csv, replacements=matches)["unresolved"], [])
        self.assertTrue(recovered.exists())

    def test_detector_cache_reuses_only_identical_bytes_and_rebinds_copied_faces(self):
        assets.pin_dataset(self.csv)
        cache = self.root / "detector.pkl"
        face = SimpleNamespace(src_str=str(self.source), content_sha256=self.digest,
                               face_index=0, embedding=np.array([1, 0]))
        benchmark_detection._save_cache(cache, sorter.config_fingerprint(),
            {str(self.source): {"sha256": self.digest, "faces": [face]}})
        self.source.unlink()
        cases = dataset.load_dataset(self.csv).cases
        with patch.object(benchmark_detection.subprocess, "run") as worker:
            faces = benchmark_detection.detect_cases(cases, cache_path=cache)[str(self.target)]
        worker.assert_not_called()
        self.assertEqual(faces[0].src_str, str(self.target))
        self.assertEqual(face.src_str, str(self.source))
        self.assertEqual(faces[0].content_sha256, self.digest)
        with cache.open("rb") as handle:
            self.assertEqual(pickle.load(handle)["entries"][str(self.source)]["faces"][0].src_str,
                             str(self.source))

    def test_malformed_detector_provenance_is_not_reused_by_hash(self):
        assets.pin_dataset(self.csv)
        cache = self.root / "detector.pkl"
        face = SimpleNamespace(src_str="wrong source", content_sha256=self.digest)
        benchmark_detection._save_cache(cache, sorter.config_fingerprint(),
            {str(self.source): {"sha256": self.digest, "faces": [face]}})
        with patch.object(benchmark_detection.subprocess, "run", side_effect=RuntimeError("fresh worker needed")):
            with self.assertRaisesRegex(RuntimeError, "fresh worker needed"):
                benchmark_detection.detect_cases(dataset.load_dataset(self.csv).cases, cache_path=cache)

    def test_source_move_reuses_gate_but_concurrent_move_invalidates_current_run(self):
        assets.pin_dataset(self.csv)
        db, secondary_db = sorter.IdentityDB(), secondary.SecondaryIdentityDB()
        cache = sorter.CacheState()
        with patch.object(runtime.benchmark_inputs, "protected_case_faces", return_value={}):
            first = runtime.GateInputs(cache, db, secondary_db, datasets=[self.csv],
                evidence_paths=[], negative_examples=[], primary_faces=[])
            self.source.rename(self.root / "archive.jpg")
            with self.assertRaisesRegex(RuntimeError, "input changed"):
                first.validate()
            second = runtime.GateInputs(cache, db, secondary_db, datasets=[self.csv],
                evidence_paths=[], negative_examples=[], primary_faces=[])
            self.assertEqual(first.fingerprint, second.fingerprint)
            self.target.write_bytes(b"corrupt")
            with self.assertRaisesRegex(RuntimeError, "input changed"):
                second.validate()

    def test_protected_copy_remains_held_out_from_training_by_content(self):
        assets.pin_dataset(self.csv)
        self.assertTrue(evaluation.same_holdout_group(str(self.source), str(self.target)))

    def test_replaced_original_blocks_even_a_previously_cached_gate(self):
        assets.pin_dataset(self.csv)
        with patch.object(runtime.benchmark_inputs, "protected_case_faces", return_value={}):
            before = runtime.GateInputs(sorter.CacheState(), sorter.IdentityDB(),
                secondary.SecondaryIdentityDB(), datasets=[self.csv], evidence_paths=[],
                negative_examples=[], primary_faces=[])
            before.validate()
            self.source.write_bytes(b"replacement")
            with self.assertRaisesRegex(RuntimeError, "Verified benchmark source changed"):
                runtime.GateInputs(sorter.CacheState(), sorter.IdentityDB(),
                    secondary.SecondaryIdentityDB(), datasets=[self.csv], evidence_paths=[],
                    negative_examples=[], primary_faces=[])

    def test_bulk_review_still_edits_annotations_after_source_moves(self):
        import review_identity_benchmark as review
        assets.pin_dataset(self.csv)
        self.source.unlink()
        changed = review.bulk_update_rows(self.csv, [str(self.source)],
                                         {"notes": "Reviewed again"}, verified=True)
        self.assertEqual(changed[0]["source"], str(self.source))
        self.assertEqual(changed[0]["content_sha256"], self.digest)
        self.assertEqual(dataset.load_dataset(self.csv).errors, ())

    def test_browser_uses_copy_and_old_forms_preserve_hash_and_source_group(self):
        import review_identity_benchmark as review
        from urllib.parse import urlencode
        from urllib.request import Request, urlopen
        assets.pin_dataset(self.csv)
        self.source.unlink()
        server = review.BenchmarkServer(("127.0.0.1", 0), review.Handler)
        server.dataset = self.csv
        server.baseline = self.root / "baseline.json"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            base = f"http://127.0.0.1:{server.server_port}"
            with urlopen(base, timeout=5) as response:
                page = response.read().decode()
            self.assertIn(urlencode({"path": str(self.target)}), page)
            with urlopen(base + "/media?" + urlencode({"path": str(self.target)}), timeout=5) as response:
                self.assertEqual(response.read(), b"verified unknown image")
            form = {key: value for key, value in self.row.items()
                    if key not in {"content_sha256", "group_id", "identity_face_id"}}
            form["notes"] = "Browser confirmation"
            with urlopen(Request(base + "/save", data=urlencode(form).encode()), timeout=5) as response:
                response.read()
            row = review.read_rows(self.csv)[0]
            self.assertEqual(row["content_sha256"], self.digest)
            self.assertEqual(row["group_id"], "source batch")
            self.assertEqual(row["notes"], "Browser confirmation")
            self.assertFalse(self.source.exists())
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

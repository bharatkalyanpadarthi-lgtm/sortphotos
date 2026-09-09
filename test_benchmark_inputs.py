"""Synthetic regressions for selected-face preflight and recoverable gate errors."""

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import benchmark_inputs
import content_identity
import evaluation_dataset
import identity_evaluation as evaluation
import review_unknown_identities as review
import sort_photos as sorter


class BenchmarkInputTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.source = self.root / "image.jpg"
        self.source.write_bytes(b"synthetic source")
        self.digest = content_identity.content_sha256(self.source)
        self.main = self.face(b"confirmed crop", 0)
        self.other = self.face(b"background", 1)
        self.case = evaluation_dataset.EvaluationCase(
            self.source, "Alice", frozenset({"known"}), True, "unknown", True,
            content_sha256=self.digest, expected_face_count=2,
            identity_face_id=content_identity.face_identity(self.main))
        self.messages = []

    def face(self, crop, number):
        return SimpleNamespace(src_str=str(self.source), crop_jpeg=crop,
                               face_index=number, content_sha256=self.digest)

    def prepare(self, faces, **kwargs):
        return benchmark_inputs.prepare_selected_faces(
            (self.case,), SimpleNamespace(faces=faces), progress=self.messages.append, **kwargs)

    def test_matching_selection_uses_cache_without_detection(self):
        with patch.object(benchmark_inputs.benchmark_detection, "detect_cases") as detect:
            self.assertEqual(self.prepare([self.main, self.other]), {})
        detect.assert_not_called()

    def test_refresh_recovers_only_exact_saved_selection_without_cache_edits(self):
        stale = self.face(b"different crop", 0)
        cache = SimpleNamespace(faces=[stale, self.other])
        ordinary = replace(self.case, identity_face_id="")
        result = {str(self.source): [self.main, self.other]}
        with patch.object(benchmark_inputs.benchmark_detection, "detect_cases", return_value=result) as detect:
            actual = benchmark_inputs.prepare_selected_faces(
                (ordinary, self.case), cache, progress=self.messages.append)
        self.assertEqual(actual, result)
        self.assertEqual(detect.call_args.args[0], (self.case,))
        self.assertEqual(cache.faces, [stale, self.other])
        self.assertEqual(self.case.identity_face_id, content_identity.face_identity(self.main))
        self.assertEqual(self.source.read_bytes(), b"synthetic source")

    def test_fresh_mode_does_not_reuse_matching_ordinary_cache(self):
        with patch.object(benchmark_inputs.benchmark_detection, "detect_cases",
                          return_value={str(self.source): [self.main, self.other]}) as detect:
            self.prepare([self.main, self.other], force_fresh=True)
        detect.assert_called_once()

    def test_missing_or_duplicate_fresh_fingerprint_is_never_guessed(self):
        for faces in ([self.other], [self.main, self.main], []):
            with self.subTest(faces=len(faces)), \
                 patch.object(benchmark_inputs.benchmark_detection, "detect_cases",
                              return_value={str(self.source): faces}):
                with self.assertRaisesRegex(benchmark_inputs.BenchmarkInputError, "missing or ambiguous"):
                    self.prepare([])

    def test_replaced_image_is_rejected_before_detection(self):
        self.source.write_bytes(b"different source")
        with patch.object(benchmark_inputs.benchmark_detection, "detect_cases") as detect:
            with self.assertRaisesRegex(benchmark_inputs.BenchmarkInputError, "content changed"):
                self.prepare([])
        detect.assert_not_called()

    def test_wrong_worker_provenance_is_rejected(self):
        self.main.content_sha256 = "incorrect"
        with patch.object(benchmark_inputs.benchmark_detection, "detect_cases",
                          return_value={str(self.source): [self.main]}):
            with self.assertRaisesRegex(benchmark_inputs.BenchmarkInputError, "provenance"):
                self.prepare([])

    def test_worker_failure_reports_input_error(self):
        with patch.object(benchmark_inputs.benchmark_detection, "detect_cases",
                          side_effect=RuntimeError("worker failed")):
            with self.assertRaisesRegex(benchmark_inputs.BenchmarkInputError, "detection failed"):
                self.prepare([])

    def test_failure_happens_before_expensive_cache_metrics(self):
        dataset = self.root / "benchmark.csv"
        dataset.touch()
        validation = evaluation_dataset.DatasetValidation((self.case,), (), evaluation_dataset.REQUIRED_CASE_TYPES)
        with patch.object(evaluation.evaluation_dataset, "load_dataset", return_value=validation), \
             patch.object(benchmark_inputs.benchmark_detection, "detect_cases",
                          return_value={str(self.source): [self.other]}), \
             patch.object(evaluation, "cache_metrics") as scoring:
            with self.assertRaises(benchmark_inputs.BenchmarkInputError):
                db = sorter.IdentityDB()
                evaluation.activation_gate(db, db, sorter.CacheState(),
                    protected_set=dataset, protected_baseline=self.root / "missing.json",
                    progress=self.messages.append)
        scoring.assert_not_called()

    def test_prepared_faces_reused_for_both_protected_scoring_lanes(self):
        dataset = self.root / "benchmark.csv"
        dataset.touch()
        validation = evaluation_dataset.DatasetValidation((self.case,), (), evaluation_dataset.REQUIRED_CASE_TYPES)
        overrides = {str(self.source): [self.main, self.other]}
        metrics = evaluation_dataset.EvaluationMetrics(1, 1, 1, 0, 1, 1, 0, 1)
        with patch.object(evaluation.evaluation_dataset, "load_dataset", return_value=validation), \
             patch.object(benchmark_inputs.benchmark_detection, "detect_cases", return_value=overrides) as detect, \
             patch.object(evaluation, "cache_metrics", return_value={"incorrect": 0, "precision": 1, "recall": 1}), \
             patch.object(evaluation, "evaluate_golden_set", return_value=(metrics, [])) as scoring:
            db = sorter.IdentityDB()
            allowed, _ = evaluation.activation_gate(db, db, sorter.CacheState(),
                protected_set=dataset, protected_baseline=self.root / "missing.json",
                progress=self.messages.append)
        self.assertTrue(allowed)
        detect.assert_called_once()
        self.assertEqual([call.kwargs["lane"] for call in scoring.call_args_list], ["strict", "pipeline"])
        self.assertTrue(all(call.kwargs["selected_face_overrides"] == overrides
                            for call in scoring.call_args_list))

    def test_late_invalid_case_is_preflighted_before_any_scoring(self):
        first = replace(self.case, identity_face_id="")
        with patch.object(evaluation, "predict_face") as scoring:
            with self.assertRaises(benchmark_inputs.BenchmarkInputError):
                evaluation.evaluate_golden_set((first, self.case),
                    sorter.CacheState(faces=[self.other]), sorter.IdentityDB(), lane="strict")
        scoring.assert_not_called()

    def test_refreshed_scope_counts_background_face_but_does_not_score_it(self):
        refreshed = {str(self.source): [self.main, self.other]}
        with patch.object(evaluation, "predict_face", return_value=evaluation.FacePrediction("Alice", 0, 1, True)) as scoring, \
             patch.object(sorter, "analysis_index_file", return_value=self.root / "index.sqlite"):
            _, rows = evaluation.evaluate_golden_set((self.case,), sorter.CacheState(),
                sorter.IdentityDB(), lane="strict", selected_face_overrides=refreshed)
        self.assertEqual(scoring.call_count, 1)
        self.assertIs(scoring.call_args.args[0], self.main)
        self.assertEqual(rows[0]["faces_detected"], 2)
        self.assertEqual(rows[0]["identity_faces_ignored"], 1)
        self.assertTrue(rows[0]["selected_face_refreshed"])
        self.assertEqual(rows[0]["evaluation_mode"], "cached_matching_only")

    def test_changed_override_cannot_enter_scoring(self):
        self.source.write_bytes(b"replacement")
        with patch.object(evaluation, "predict_face") as scoring:
            with self.assertRaises(benchmark_inputs.BenchmarkInputError):
                evaluation.evaluate_golden_set((self.case,), sorter.CacheState(),
                    sorter.IdentityDB(), lane="strict",
                    selected_face_overrides={str(self.source): [self.main]})
        scoring.assert_not_called()

    def test_gate_exception_keeps_manual_review_available_without_auto_moves(self):
        cached = self.root / "unknown_auto_review_gate.json"
        cached.write_text("existing diagnostic state")
        with patch.object(review, "_prepare_automatic_review_gate",
                          side_effect=benchmark_inputs.BenchmarkInputError("reverify saved face")):
            allowed, report, message = review.prepare_automatic_review_gate(
                object(), object(), object(), requested=True,
                output_dir=self.root, progress=self.messages.append)
        self.assertFalse(allowed)
        self.assertTrue(report["evaluation_incomplete"])
        self.assertIn("Manual review remains available", message)
        self.assertEqual(cached.read_text(), "existing diagnostic state")
        self.assertEqual(review.run_automatic_sweep({"auto_review_allowed": allowed})["scanned"], 0)

    def test_user_interrupt_is_not_swallowed(self):
        with patch.object(review, "_prepare_automatic_review_gate", side_effect=KeyboardInterrupt):
            with self.assertRaises(KeyboardInterrupt):
                review.prepare_automatic_review_gate(object(), object(), object(),
                    requested=True, output_dir=self.root, progress=self.messages.append)

    def test_changing_annotations_cannot_cache_an_outdated_verdict(self):
        with patch.object(review, "automatic_review_gate_signature", side_effect=["before", "after"]), \
             patch.object(review.evaluation_runtime, "GateInputs", return_value=SimpleNamespace(fingerprint="inputs", validate=lambda: None)), \
             patch.object(evaluation, "activation_gate", return_value=(True, {})), \
             patch.object(review, "evaluate_automatic_policy_benchmark", return_value=(True, {})), \
             patch.object(review.identity_hard_negatives, "vectors_by_person", return_value={}):
            allowed, report, message = review.prepare_automatic_review_gate(
                object(), SimpleNamespace(faces=[]), SimpleNamespace(db=object()), requested=True,
                output_dir=self.root, progress=self.messages.append)
        self.assertFalse(allowed)
        self.assertTrue(report["evaluation_incomplete"])
        self.assertIn("inputs changed", message)
        self.assertFalse((self.root / "unknown_auto_review_gate.json").exists())


if __name__ == "__main__":
    unittest.main()

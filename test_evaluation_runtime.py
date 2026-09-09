"""Temporary-file tests for resume, invalidation and fail-closed gate reuse."""

import csv
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import evaluation_runtime as runtime
from evaluation_checkpoints import BenchmarkCheckpoints
import identity_evaluation as evaluation
import review_unknown_identities as review
import secondary_identity_matcher as secondary
import sort_photos as sorter


class EvaluationRuntimeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / "image.jpg"
        self.path.write_bytes(b"original")
        self.db = sorter.IdentityDB()
        self.secondary = secondary.SecondaryIdentityDB()
        self.face = SimpleNamespace(src_str=str(self.path), face_index=0,
            embedding=np.array([1, 0], dtype=np.float32), quality=0.8,
            label="Alice", crop_jpeg=b"crop", content_sha256="sha", pose_label="frontal")
        self.cache = SimpleNamespace(config_fingerprint="detector1", faces=[self.face])
        self.dataset = self.root / "cases.csv"
        with self.dataset.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["source", "expected_person"])
            writer.writeheader()
            writer.writerow({"source": str(self.path), "expected_person": "Alice"})

    def inputs(self):
        return runtime.GateInputs(self.cache, self.db, self.secondary,
            datasets=[self.dataset], evidence_paths=[], negative_examples=[], primary_faces=[self.face])

    def test_reuse_exact_inputs_and_invalidate_changed_embedding_label_crop_or_config(self):
        before = self.inputs().fingerprint
        self.assertEqual(before, self.inputs().fingerprint)
        for field, value in (("label", "Bob"), ("quality", 0.1), ("crop_jpeg", b"other"),
                             ("embedding", np.array([0, 1], dtype=np.float32)), ("pose_label", "left")):
            with self.subTest(field=field):
                original = getattr(self.face, field)
                setattr(self.face, field, value)
                self.assertNotEqual(before, self.inputs().fingerprint)
                setattr(self.face, field, original)
        self.cache.config_fingerprint = "new detector"
        self.assertNotEqual(before, self.inputs().fingerprint)

    def test_source_replacement_same_size_mtime_invalidates_saved_work(self):
        before = self.inputs()
        stat = self.path.stat()
        self.path.write_bytes(b"modified")
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(before.fingerprint, self.inputs().fingerprint)
        with self.assertRaisesRegex(RuntimeError, "input changed"):
            before.validate()

    def test_missing_reference_appearing_and_provenance_path_change_invalidates(self):
        reference = self.root / "new.jpg"
        self.db.pose_prototype_sources = {"Alice": {"left": [str(reference)]}}
        before = self.inputs()
        reference.write_bytes(b"new")
        with self.assertRaises(RuntimeError):
            before.validate()
        self.assertNotEqual(before.fingerprint, self.inputs().fingerprint)

    def test_irrelevant_non_sampled_face_does_not_invalidate(self):
        before = self.inputs().fingerprint
        self.cache.faces.append(SimpleNamespace(src_str=str(self.root / "unselected.jpg")))
        self.assertEqual(before, self.inputs().fingerprint)

    def test_metadata_touch_reuses_but_annotation_change_invalidates(self):
        before = self.inputs().fingerprint
        os.utime(self.dataset, None)
        self.assertEqual(before, self.inputs().fingerprint)
        stat = self.dataset.stat()
        self.dataset.write_text(self.dataset.read_text().replace("Alice", "Carol"))
        os.utime(self.dataset, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(before, self.inputs().fingerprint)

    def test_symlink_retarget_invalidates(self):
        other = self.root / "other.jpg"
        other.write_bytes(b"other")
        link = self.root / "reference.jpg"
        link.symlink_to(self.path)
        versions = runtime.InputVersions()
        versions.add(link)
        link.unlink()
        link.symlink_to(other)
        with self.assertRaises(RuntimeError):
            versions.validate()

    def test_interrupted_block_resumes_completed_rows_including_skips(self):
        calls = []
        messages = []
        def evaluate(value):
            calls.append(value)
            if value == 5:
                raise KeyboardInterrupt
            return None if value % 2 else {"value": value}
        path = self.root / "checkpoints.sqlite3"
        with BenchmarkCheckpoints(path, "inputs1") as checkpoints:
            with self.assertRaises(KeyboardInterrupt):
                runtime.evaluate_blocks(list(range(10)), evaluate, checkpoints=checkpoints,
                    stage="Synthetic", block_size=4, progress=messages.append)
        calls.clear()
        with BenchmarkCheckpoints(path, "inputs1") as checkpoints:
            results = runtime.evaluate_blocks(list(range(10)), lambda value: calls.append(value) or
                (None if value % 2 else {"value": value}), checkpoints=checkpoints,
                stage="Synthetic", block_size=4, progress=messages.append)
        self.assertEqual(calls, list(range(4, 10)))
        self.assertEqual(results, [None if n % 2 else {"value": n} for n in range(10)])
        self.assertTrue(any("4 reused" in message for message in messages))
        calls.clear()
        with BenchmarkCheckpoints(path, "changed inputs") as checkpoints:
            runtime.evaluate_blocks(list(range(10)), lambda value: calls.append(value),
                checkpoints=checkpoints, stage="Synthetic", block_size=4, progress=None)
        self.assertEqual(len(calls), 10)

    def test_incomplete_block_is_not_trusted(self):
        with BenchmarkCheckpoints(self.root / "db.sqlite3", "inputs") as checkpoints:
            checkpoints.put("Test/block/4/0", {"count": 4, "rows": ["partial"]})
            self.assertEqual(runtime.evaluate_blocks([1, 2, 3, 4], lambda n: n,
                checkpoints=checkpoints, stage="Test", block_size=4, progress=None), [1, 2, 3, 4])

    def test_concurrent_gate_is_rejected_and_interrupt_releases_lock(self):
        with self.assertRaises(KeyboardInterrupt):
            with runtime.exclusive_evaluation(self.root):
                with self.assertRaisesRegex(RuntimeError, "already running"):
                    with runtime.exclusive_evaluation(self.root):
                        self.fail("concurrent gate started")
                raise KeyboardInterrupt
        with runtime.exclusive_evaluation(self.root):
            pass

    def test_memory_mutation_is_rejected_without_filesystem_change(self):
        inputs = self.inputs()
        self.face.embedding[0] = 0.5
        with self.assertRaisesRegex(RuntimeError, "cached faces changed"):
            inputs.validate()

    def test_detected_policy_drift_discards_namespace_even_after_policy_restored(self):
        current = ["A"]
        def validate():
            if current[0] != "A":
                raise RuntimeError("policy changed")
        with BenchmarkCheckpoints(self.root / "db.sqlite3", "A") as store:
            guarded = runtime.GuardedCheckpoints(store, validate)
            guarded.put("completed-before-drift", {"result": "A"})
            current[0] = "B"
            with self.assertRaisesRegex(RuntimeError, "policy changed"):
                guarded.put("wrong-block", {"result": "B"})
            current[0] = "A"
            self.assertIsNone(guarded.get("completed-before-drift"))
            self.assertIsNone(guarded.get("wrong-block"))

    def test_interrupting_validation_keeps_previously_validated_work(self):
        path = self.root / "db.sqlite3"
        with BenchmarkCheckpoints(path, "A") as store:
            store.put("completed", {"valid": True})
            def interrupted():
                raise KeyboardInterrupt
            with self.assertRaises(KeyboardInterrupt):
                runtime.GuardedCheckpoints(store, interrupted).get("completed")
        with BenchmarkCheckpoints(path, "A") as store:
            self.assertEqual(store.get("completed"), {"valid": True})

    def test_unchanged_sample_selection_is_reused_and_embedding_change_invalidates(self):
        self.face.label = "Alice"
        with patch.object(evaluation, "select_faces", return_value=[self.face]) as select:
            for _ in range(2):
                self.assertEqual(runtime.selected_faces(self.cache, self.root,
                    progress=lambda _: None), [self.face])
            self.assertEqual(select.call_count, 1)
            self.face.embedding[0] = 0.5
            runtime.selected_faces(self.cache, self.root, progress=lambda _: None)
            self.assertEqual(select.call_count, 2)

    def test_failed_primary_gate_skips_independent_and_is_reused(self):
        inputs = SimpleNamespace(fingerprint="inputs", validate=lambda: None)
        messages = []
        with patch.object(review, "automatic_review_gate_signature", return_value="policy"), \
             patch.object(review.evaluation_runtime, "GateInputs", return_value=inputs), \
             patch.object(evaluation, "activation_gate", return_value=(False, {"failures": ["wrong identity"]})) as primary, \
             patch.object(review, "evaluate_automatic_policy_benchmark") as independent:
            for _ in range(2):
                allowed, report, _ = review.prepare_automatic_review_gate(self.db, sorter.CacheState(),
                    SimpleNamespace(db=self.secondary), requested=True, output_dir=self.root,
                    progress=messages.append)
                self.assertFalse(allowed)
                self.assertTrue(report["report"]["automatic_policy"]["skipped"])
            self.assertEqual(primary.call_count, 1)
            independent.assert_not_called()

    def test_primary_fail_fast_does_not_start_protected_scoring(self):
        metrics = dict(incorrect=1, correct=10, rejected=2, precision=0.9, recall=0.8)
        with patch.object(evaluation, "cache_metrics", return_value=metrics), \
             patch.object(evaluation, "evaluate_golden_set") as protected:
            allowed, report = evaluation.activation_gate(self.db, self.db, sorter.CacheState(),
                protected_set=self.root / "missing.csv", protected_baseline=self.root / "missing.json",
                fail_fast=True, progress=lambda _: None)
            self.assertFalse(allowed)
            self.assertTrue(report["protected"]["skipped"])
            protected.assert_not_called()

    def test_prepared_secondary_verification_preserves_holdout_decision(self):
        from evaluation_profiles import EvaluationProfiles
        alice, bob = np.eye(2, dtype=np.float32)
        sources = {}
        for name in ("Alice", "Bob"):
            reference = self.root / f"{name}.jpg"
            reference.write_bytes(name.encode())
            sources[name] = [str(reference)]
        db = secondary.SecondaryIdentityDB(identities={"Alice": alice, "Bob": bob},
            prototypes={"Alice": [alice], "Bob": [bob]}, prototype_sources=sources)
        matcher = secondary.SecondaryMatcher(db)
        prepared = EvaluationProfiles(sorter.IdentityDB(identities=db.identities,
            prototypes=db.prototypes, prototype_sources=sources), negative_examples=[])
        with patch.object(matcher, "embedding", return_value=alice):
            for peers in (frozenset(), frozenset(sources["Alice"])):
                original = matcher.verify(b"crop", "Alice", excluded_source=self.path, excluded_sources=peers)
                optimized = matcher.verify(b"crop", "Alice", excluded_source=self.path,
                                           excluded_sources=peers, prepared=prepared)
                self.assertEqual(original, optimized)
        prepared.validate()

    def test_policy_signature_covers_provenance_runtime_settings_and_evidence_content(self):
        negative = self.root / "negative.json"
        negative.write_text('{"examples": []}')
        before = review.automatic_review_gate_signature(self.db, hard_negatives_path=negative)
        with patch.object(sorter, "AUTO_PERSON_SINGLE_MATCH_DIST", 0.123):
            self.assertNotEqual(before, review.automatic_review_gate_signature(self.db, hard_negatives_path=negative))
        self.db.pose_prototype_sources = {"Alice": {"left": [str(self.path)]}}
        self.assertNotEqual(before, review.automatic_review_gate_signature(self.db, hard_negatives_path=negative))
        self.db.pose_prototype_sources.clear()
        stat = negative.stat()
        negative.write_text('{"examples": {}}')
        os.utime(negative, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertNotEqual(before, review.automatic_review_gate_signature(self.db, hard_negatives_path=negative))


if __name__ == "__main__":
    unittest.main()

"""Bounded-work checks for review startup, isolated workers and completion."""

from collections import defaultdict
from contextlib import ExitStack
import json
import hashlib
from pathlib import Path
import queue
import sqlite3
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

import analysis_index
import benchmark_snapshot
import content_identity
import identity_confirmations
import review_unknown_identities as review
import secondary_identity_matcher as secondary
import shared_profile_snapshot
import sort_photos as sorter


class QuickReviewPerformanceTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.source = self.root / "Alice/photos/reference.jpg"
        self.source.parent.mkdir(parents=True)
        self.source.write_bytes(b"reference")
        self.database = self.root / "analysis.sqlite3"
        with analysis_index.AnalysisIndex(self.database) as index:
            self.digest = index.content_sha256(self.source)
        vector = np.array([1., 0.], dtype=np.float32)
        self.db = sorter.IdentityDB(identities={"Alice": vector},
            prototypes={"Alice": [vector]}, source_counts={"Alice": 5})
        self.cache = sorter.CacheState()
        self.progress = dict(total=1, reviewed=1, pending=0, confirmed=0,
                             unknown=1, junk=0, deferred=0, temporarily_skipped=0)

    def worker(self, *, accepted=True, replaced=False, build_error=False, binding=None,
               snapshot_hash=None, snapshot_read_error=None):
        task = self.root / "task.json"
        result = self.root / "result.json"
        task.write_text(json.dumps(dict(result_path=str(result), output_dir=str(self.root),
            analysis_index=str(self.database), people_root=str(self.root),
            unknown_root=str(self.root / "unknown"), unassigned_root=str(self.root),
            decisions_path=str(self.root / "decisions.json"), review_dir=str(self.root / "review"),
            junk_dir=str(self.root / "junk"), workers=1, primary_det_size=320,
            fallback_det_size=384, cluster_eps=.4, files=[])))
        payload = json.loads(task.read_text())
        payload['snapshot_binding'] = binding if binding is not None else {
            'version': 1, 'gate_signature': 'current-gate',
            'policy_signature': 'current-policy', 'model_signature': 'current-model'}
        payload['shared_profile_snapshot'] = {'path': str(self.root / 'snapshot.json'),
            'sha256': snapshot_hash or hashlib.sha256(b'snapshot').hexdigest()}
        task.write_text(json.dumps(payload))
        record = dict(person="Alice", organized_path=str(self.source), content_sha256=self.digest)
        if replaced:
            self.source.write_bytes(b"different content")
        indexes = []

        def build(*_args, reference_index, **_kwargs):
            indexes.append(reference_index)
            self.assertEqual(reference_index._database, self.database)
            resolved = reference_index.resolve(record)
            self.assertEqual(resolved, None if replaced else self.source)
            if not replaced:
                self.assertEqual(reference_index.hashes_read, 0)
                self.assertEqual(reference_index.hashes_reused, 1)
            connection = reference_index._analysis.connection
            indexes.append(connection)
            if build_error:
                raise RuntimeError("reference preparation failed")
            return {}, {}

        with ExitStack() as stack:
            for owner, name, value in (
                (sorter, "load_identity_db", self.db),
                (sorter, "load_cache", self.cache),
                (secondary, "load", SimpleNamespace(primary_signature="current")),
                (secondary, "primary_signature", "current"),
                (secondary, "trusted_snapshot_is_current", True),
                (secondary, "SecondaryMatcher", object()),
                (review, "prepare_automatic_review_gate", (accepted, {'signature': 'current-gate'}, "gate blocked")),
                (review, "automatic_review_gate_signature", 'current-policy'),
                (review, "review_model_signature", 'current-model'),
                (review.recover_no_usable_faces, "load_existing_hashes", defaultdict(set)),
            ):
                stack.enter_context(patch.object(owner, name, return_value=value))
            stack.enter_context(patch.object(review, "build_trusted_review_prototypes", side_effect=build))
            stack.enter_context(patch.object(benchmark_snapshot, "freeze_profile", return_value=self.db))
            stack.enter_context(patch.object(benchmark_snapshot, "FrozenRecoveryProfiles", return_value={}))
            stack.enter_context(patch.object(secondary, "SnapshotSecondaryMatcher", return_value=SimpleNamespace(db=object())))
            stack.enter_context(patch.object(shared_profile_snapshot, "encode", return_value=b'snapshot'))
            stack.enter_context(patch.object(shared_profile_snapshot, "read",
                side_effect=snapshot_read_error, return_value=(self.db, object(), {}, {})))
            batch = stack.enter_context(patch.object(review, "_run_automatic_sweep_batch",
                return_value=dict(scanned=0, confirmed=0, failed=0)))
            code = review.run_automatic_sweep_worker(task)
        self.assertEqual(len(indexes), 2)
        with self.assertRaises(sqlite3.ProgrammingError):
            indexes[1].execute("SELECT 1")
        return code, json.loads(result.read_text()), batch

    def test_worker_reuses_verified_hash_and_closes_its_index(self):
        code, result, batch = self.worker()
        self.assertEqual(code, 0)
        self.assertEqual(result["scanned"], 0)
        batch.assert_called_once()

    def test_sweep_writes_reason_and_cache_diagnostics_without_changing_decisions(self):
        state = {'auto_review_allowed': True, 'auto_review_preview': False,
                 'unknown_root': self.root, 'decisions_path': self.root / 'decisions.json',
                 'model_signature': 'model', 'output_dir': self.root,
                 'auto_review_gate': {'signature': 'gate'}, 'batch_limit': 500}
        payload = {'confirmed': 0, 'failed': 0, 'routed': {}, 'completed_content_keys': ['digest'],
                   'detection_stats': {'sqlite_hits': 1}, 'eligible_lanes': {},
                   'held_diagnostics': [{'path': str(self.source), 'sha256': 'digest',
                                         'reason_codes': ['weak_confidence']}]}
        with patch.object(review, 'load_decisions', return_value={}), \
             patch.object(review.recover_no_usable_faces, 'iter_images', return_value=[self.source]), \
             patch.object(review, 'automatic_check_allowed', return_value=True), \
             patch.object(review, 'item_content_sha256', return_value='digest'), \
             patch.object(review, '_run_automatic_sweep_batch', return_value=payload):
            result = review.run_automatic_sweep(state)
        report = json.loads(Path(result['diagnostics_path']).read_text())
        self.assertEqual(report['reason_counts'], {'weak_confidence': 1})
        self.assertEqual(report['detection_stats']['sqlite_hits'], 1)
        self.assertTrue(self.source.is_file())

    def test_shared_snapshot_directory_is_cleaned_after_failure(self):
        state = {'isolate_auto_sweep_batches': True}
        directories = []
        def fail(current):
            directories.append(Path(current['_shared_snapshot_dir']))
            self.assertTrue(directories[0].is_dir())
            raise RuntimeError('interrupted')
        with patch.object(review, '_run_automatic_sweep', side_effect=fail):
            with self.assertRaisesRegex(RuntimeError, 'interrupted'):
                review.run_automatic_sweep(state)
        self.assertFalse(directories[0].exists())
        self.assertNotIn('_shared_snapshot_dir', state)

    def test_worker_rejects_replaced_reference_content(self):
        code, _result, _batch = self.worker(replaced=True)
        self.assertEqual(code, 0)

    def test_worker_retains_gate_before_filing(self):
        code, result, batch = self.worker(accepted=False)
        self.assertEqual(code, 1)
        self.assertIn("gate blocked", result["error"])
        batch.assert_not_called()

    def test_worker_rejects_cross_process_gate_drift_before_filing(self):
        code, result, batch = self.worker(binding={'version': 1, 'gate_signature': 'old-gate',
            'policy_signature': 'current-policy', 'model_signature': 'current-model'})
        self.assertEqual(code, 1)
        self.assertIn('snapshot changed', result['error'])
        batch.assert_not_called()

    def test_worker_rejects_missing_binding_before_filing(self):
        code, result, batch = self.worker(binding={})
        self.assertEqual(code, 1)
        self.assertIn('invalid automatic sweep snapshot binding', result['error'])
        batch.assert_not_called()

    def test_worker_accepts_matching_cross_process_binding(self):
        code, result, batch = self.worker(binding={'version': 1, 'gate_signature': 'current-gate',
            'policy_signature': 'current-policy', 'model_signature': 'current-model'})
        self.assertEqual(code, 0)
        batch.assert_called_once()

    def test_worker_rejects_changed_live_snapshot_before_filing(self):
        code, result, batch = self.worker(snapshot_hash='0' * 64)
        self.assertEqual(code, 1)
        self.assertIn('no longer matches live profiles', result['error'])
        batch.assert_not_called()

    def test_worker_rejects_corrupted_shared_snapshot_before_filing(self):
        code, result, batch = self.worker(snapshot_read_error=ValueError('snapshot hash mismatch'))
        self.assertEqual(code, 1)
        self.assertIn('snapshot hash mismatch', result['error'])
        batch.assert_not_called()

    def test_reference_exception_closes_index_and_prevents_filing(self):
        code, result, batch = self.worker(build_error=True)
        self.assertEqual(code, 1)
        self.assertIn("reference preparation failed", result["error"])
        batch.assert_not_called()

    def test_worker_and_parent_select_identical_references_among_exact_copies(self):
        alias = self.source.with_name("aaa-copy.jpg")
        alias.write_bytes(self.source.read_bytes())
        with analysis_index.AnalysisIndex(self.database) as index:
            index.content_sha256(alias)
        faces = [sorter.CachedFace(str(path), 0, 1, 100, 100, 0, .9,
                    np.array([1., 0.], dtype=np.float32), np.array([True, False]),
                    b"crop", label="Alice", content_sha256=self.digest)
                 for path in (self.source, alias)]
        self.cache.faces = faces
        confirmations = self.root / "confirmations.json"
        confirmations.write_text(json.dumps({"version": 1, "examples": [dict(
            person="Alice", organized_path=str(self.source.with_name("old.jpg")),
            content_sha256=self.digest, face_id=content_identity.face_identity(faces[0]))]}))
        original_build = review.build_trusted_review_prototypes
        with patch.object(sorter, "IDENTITY_CONFIRMATIONS_FILE", confirmations), \
             identity_confirmations.ReferenceIndex(
                 faces_by_source=review._faces_by_source(self.cache),
                 database=self.database, progress=None) as parent_index:
            parent = original_build(self.db, self.cache, confirmations_path=confirmations,
                                    reference_index=parent_index, progress=None)
            task = self.root / "reference-worker.json"
            result = self.root / "reference-result.json"
            task.write_text(json.dumps(dict(result_path=str(result), output_dir=str(self.root),
                                           analysis_index=str(self.database))))

            def build(*args, **kwargs):
                worker = original_build(*args, **kwargs, progress=None)
                self.assertEqual(parent[1], worker[1])
                self.assertEqual(parent[0].sources, worker[0].sources)
                for person in parent[0]:
                    np.testing.assert_array_equal(parent[0][person], worker[0][person])
                self.assertIn(str(alias), worker[0].sources["Alice"])
                return worker

            with patch.object(sorter, "load_identity_db", return_value=self.db), \
                 patch.object(sorter, "load_cache", return_value=self.cache), \
                 patch.object(secondary, "load", return_value=SimpleNamespace(primary_signature="current")), \
                 patch.object(secondary, "primary_signature", return_value="current"), \
                 patch.object(secondary, "trusted_snapshot_is_current", return_value=True), \
                 patch.object(secondary, "SecondaryMatcher", return_value=object()), \
                 patch.object(review, "build_trusted_review_prototypes", side_effect=build), \
                 patch.object(review, "prepare_automatic_review_gate", return_value=(False, {}, "blocked")):
                self.assertEqual(review.run_automatic_sweep_worker(task), 1)
            self.assertIn("blocked", json.loads(result.read_text())["error"])

    def state(self):
        return dict(output_dir=self.root, session_path=self.root / "session.json",
            decisions_path=self.root / "decisions.json", summary={}, items_by_key={},
            clusters=[], identity_names=["Alice"], identity_db=self.db,
            unknown_root=self.root / "unknown", people_root=self.root,
            evaluation_path=self.root / "confirmed.csv", cache=self.cache,
            action_queue=SimpleNamespace(pending=queue.Queue()), lock=threading.Lock(),
            cache_dirty=False, identity_dirty=False)

    def test_report_and_session_use_one_progress_scan(self):
        state = self.state()
        with patch.object(review, "review_progress", return_value=self.progress) as progress, \
             patch.object(review, "load_decisions", return_value={"items": {}}) as decisions, \
             patch.object(review, "render_html", return_value="report"):
            review.write_review_reports(state)
        progress.assert_called_once()
        decisions.assert_called_once()
        self.assertEqual(json.loads(state["session_path"].read_text())["progress"], self.progress)

    def test_batch_progress_is_shared_without_another_scan(self):
        state = self.state()
        with patch.object(review, "review_progress", side_effect=AssertionError("repeated scan")), \
             patch.object(review, "load_decisions", side_effect=AssertionError("repeated load")), \
             patch.object(review, "render_html", return_value="report"):
            review.write_review_reports(state, decisions={"items": {}}, progress=self.progress)
        for name in ("session.json", "latest_unknown_identity_review.json"):
            self.assertEqual(json.loads((self.root / name).read_text())["progress"], self.progress)

    def test_empty_progress_is_not_treated_as_a_cache_miss(self):
        with patch.object(review, "review_progress", side_effect=AssertionError("repeated scan")):
            review.save_session_state(self.state(), progress={})
        self.assertEqual(json.loads((self.root / "session.json").read_text())["progress"], {})

    def test_finish_reuses_primary_stage_path_and_does_not_rebuild_clean_profiles(self):
        state = self.state()
        job = review.FinishReviewJob(state)
        with patch.object(review.identity_evaluation, "activation_gate", return_value=(True, {})) as gate, \
             patch.object(sorter, "build_identity_db_from_person_folders") as build, \
             patch.object(review, "review_progress", return_value=self.progress) as progress, \
             patch.object(review, "load_decisions", return_value={"items": {}}) as decisions:
            job._run()
        self.assertEqual(job.snapshot()["status"], "completed")
        self.assertEqual(gate.call_args.kwargs["stage_cache_dir"], self.root / "primary_stages")
        self.assertEqual(gate.call_args.kwargs["confirmed_set"], state["evaluation_path"])
        build.assert_not_called()
        progress.assert_called_once()
        decisions.assert_called_once()
        report = json.loads((self.root / "unknown_review_final_report.json").read_text())
        self.assertEqual(report["progress"], self.progress)
        self.assertTrue(report["activation_gate"]["current_profile_check"])

    def test_dirty_finish_keeps_profile_builder_gate_without_running_another_gate(self):
        state = self.state()
        state["identity_dirty"] = True
        state["activation_gate_path"] = self.root / "activation.json"
        state["activation_gate_path"].write_text('{"allowed":true}')
        job = review.FinishReviewJob(state)
        with patch.object(review.identity_evaluation, "activation_gate") as gate, \
             patch.object(sorter, "build_identity_db_from_person_folders", return_value=self.db) as build, \
             patch.object(review, "review_progress", return_value=self.progress), \
             patch.object(review, "load_decisions", return_value={"items": {}}):
            job._run()
        self.assertEqual(job.snapshot()["status"], "completed")
        gate.assert_not_called()
        build.assert_called_once_with(self.root)

    def test_finish_gate_error_is_not_marked_successful(self):
        state = self.state()
        job = review.FinishReviewJob(state)
        with patch.object(review.identity_evaluation, "activation_gate", side_effect=RuntimeError("gate error")):
            job._run()
        self.assertEqual(job.snapshot()["status"], "failed")
        self.assertFalse(state.get("review_finished", False))
        self.assertFalse((self.root / "unknown_review_final_report.json").exists())


if __name__ == "__main__":
    unittest.main()

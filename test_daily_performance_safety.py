"""Isolated parity, bounded-work and failure tests for daily speed improvements."""

import copy
from contextlib import ExitStack
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import analysis_index
import benchmark_snapshot
import content_identity
import evaluation_runtime as runtime
import evaluation_enrollment
import identity_evaluation as evaluation
import primary_stage_cache
import review_unknown_identities as review
import secondary_identity_matcher as secondary
import sort_photos as sorter
from verified_references import ReferenceIndex


class Fixtures(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.path = self.root / "Alice/photos/query.jpg"
        self.path.parent.mkdir(parents=True)
        self.path.write_bytes(b"query")
        self.reference = self.path.with_name("reference.jpg")
        self.reference.write_bytes(b"reference")
        self.bob = self.root / "Bob.jpg"
        self.bob.write_bytes(b"Bob")
        a, b = np.eye(2, dtype=np.float32)
        self.face = sorter.CachedFace(str(self.path), 0, 1, 100, 100, 0, .9,
            a.copy(), np.array([True, False]), b"crop", label="Alice",
            content_sha256=content_identity.content_sha256(self.path))
        self.cache = sorter.CacheState(config_fingerprint="detector", faces=[self.face])
        self.db = sorter.IdentityDB(identities={"Alice": a, "Bob": b},
            prototypes={"Alice": [a.copy()], "Bob": [b.copy()]},
            prototype_sources={"Alice": [str(self.reference)], "Bob": [str(self.bob)]},
            source_counts={"Alice": 5, "Bob": 5})
        self.index_path = self.root / "analysis.sqlite3"
        negative = patch.object(sorter, "IDENTITY_HARD_NEGATIVES_FILE", self.root / "negatives.json")
        negative.start()
        self.addCleanup(negative.stop)


class PersistentReferenceTests(Fixtures):
    def record(self):
        return dict(person="Alice", organized_path=str(self.path.with_name("old.jpg")),
                    content_sha256=content_identity.content_sha256(self.path))

    def seed(self):
        with analysis_index.AnalysisIndex(self.index_path) as index:
            index.content_sha256(self.path)

    def test_index_hit_avoids_person_scan_and_reading_unchanged_bytes(self):
        self.seed()
        record = self.record()
        with ReferenceIndex([self.path], database=self.index_path, progress=None) as index, \
             patch.object(index, "_lookup", side_effect=AssertionError("full person scan")), \
             patch.object(content_identity, "content_sha256", side_effect=AssertionError("rehash")):
            self.assertEqual(index.resolve(record), self.path)
            self.assertEqual(index.hashes_read, 0)

    def test_hash_reuse_survives_a_new_index_instance(self):
        self.seed()
        record = self.record()
        for _ in range(2):
            with ReferenceIndex([self.path], database=self.index_path, progress=None) as index:
                self.assertEqual(index.resolve(record), self.path)
                self.assertEqual(index.hashes_read, 0)
                self.assertGreater(index.hashes_reused, 0)

    def test_same_size_mtime_replacement_rejects_stale_hash_hint(self):
        self.seed()
        record, stat = self.record(), self.path.stat()
        self.path.write_bytes(b"wrong")
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        with ReferenceIndex([self.path], database=self.index_path, progress=None) as index:
            self.assertIsNone(index.resolve(record))

    def test_wrong_person_hint_is_not_accepted(self):
        other = self.root / "Bob/photos/duplicate.jpg"
        other.parent.mkdir(parents=True)
        other.write_bytes(self.path.read_bytes())
        with analysis_index.AnalysisIndex(self.index_path) as index:
            index.content_sha256(other)
        record = self.record()
        self.path.unlink()
        with ReferenceIndex([self.path, other], database=self.index_path, progress=None) as index:
            self.assertIsNone(index.resolve(record))

    def test_missing_or_corrupt_index_falls_back_without_weakening_verification(self):
        self.index_path.write_bytes(b"not SQLite")
        with ReferenceIndex([self.path], database=self.index_path, progress=None) as index:
            self.assertEqual(index.resolve(self.record()), self.path)
            self.assertEqual(index.hashes_read, 1)

    def test_unindexed_move_falls_back_then_next_session_reuses_it(self):
        self.seed()
        record = self.record()
        renamed = self.path.with_name("renamed.jpg")
        self.path.rename(renamed)
        for _ in range(2):
            with ReferenceIndex([renamed], database=self.index_path, progress=None) as index:
                self.assertEqual(index.resolve(record), renamed)
        with ReferenceIndex([renamed], database=self.index_path, progress=None) as index, \
             patch.object(index, "_lookup", side_effect=AssertionError("rescanned")):
            self.assertEqual(index.resolve(record), renamed)

    def test_closed_lookup_releases_connection(self):
        with ReferenceIndex([self.path], database=self.index_path, progress=None) as index:
            index.resolve(self.record())
            connection = index._analysis.connection
        with self.assertRaises(sqlite3.ProgrammingError):
            connection.execute("SELECT 1")


class BatchEnrollmentTests(Fixtures):
    def examples(self):
        import identity_confirmations
        faces, records = [], []
        for number in range(80):
            path = self.path.with_name(f"image-{number}.jpg")
            path.write_bytes(str(number).encode())
            face = copy.deepcopy(self.face)
            face.src_str = str(path)
            face.content_sha256 = content_identity.content_sha256(path)
            faces.append(face)
            records.append(dict(person="Alice", organized_path=str(path),
                content_sha256=face.content_sha256, face_id=content_identity.face_identity(face)))
        return faces, records

    def test_batch_reads_and_writes_csv_once_and_matches_individual_enrollment(self):
        faces, records = self.examples()
        batch, individual = self.root / "batch.csv", self.root / "individual.csv"
        for face in faces:
            evaluation_enrollment.enroll(source=Path(face.src_str), person="Alice",
                content_sha256=face.content_sha256, quality=face.quality,
                pose_label=face.pose_label, path=individual)
        with patch.object(evaluation_enrollment, "_read", wraps=evaluation_enrollment._read) as read, \
             patch.object(evaluation_enrollment, "_write", wraps=evaluation_enrollment._write) as write:
            self.assertEqual(evaluation_enrollment.backfill_confirmations(
                {"examples": records}, faces, path=batch), len(faces))
            read.assert_called_once()
            write.assert_called_once()
        self.assertEqual(batch.read_bytes(), individual.read_bytes())

    def test_noop_backfill_preserves_manual_fields_and_does_not_rewrite(self):
        faces, records = self.examples()
        path = self.root / "cases.csv"
        evaluation_enrollment.backfill_confirmations({"examples": records}, faces, path=path)
        rows = evaluation_enrollment._read(path)
        rows[0].update(notes="Human annotation", group_id="protected family", verified="false",
                       case_types="known|lookalike", expected_face_count="3",
                       identity_face_id="crop:" + "a" * 64)
        evaluation_enrollment._write(path, rows)
        before = path.read_bytes()
        with patch.object(evaluation_enrollment, "_read", wraps=evaluation_enrollment._read) as read, \
             patch.object(evaluation_enrollment, "_write", side_effect=AssertionError("no-op rewrite")):
            evaluation_enrollment.backfill_confirmations({"examples": records}, faces, path=path)
            read.assert_called_once()
        self.assertEqual(before, path.read_bytes())

    def test_changed_source_is_not_enrolled_from_stale_confirmation(self):
        faces, records = self.examples()
        Path(faces[0].src_str).write_bytes(b"changed")
        path = self.root / "cases.csv"
        count = evaluation_enrollment.backfill_confirmations({"examples": records}, faces, path=path)
        self.assertEqual(count, len(faces) - 1)
        self.assertNotIn(records[0]["content_sha256"],
                         {row["content_sha256"] for row in evaluation_enrollment._read(path)})


class ImmutableBenchmarkTests(Fixtures):
    def inputs(self):
        return runtime.GateInputs(self.cache, self.db, secondary.SecondaryIdentityDB(),
            datasets=[], evidence_paths=[], negative_examples=[], primary_faces=[self.face])

    def test_snapshot_arrays_and_fields_are_irreversibly_read_only(self):
        frozen = benchmark_snapshot.FrozenFace(self.face)
        with self.assertRaises(AttributeError):
            frozen.label = "Bob"
        with self.assertRaises(ValueError):
            frozen.embedding[0] = .5
        with self.assertRaises(ValueError):
            frozen.embedding.setflags(write=True)
        with self.assertRaises(TypeError):
            vars(frozen)["label"] = "Bob"
        np.testing.assert_array_equal(frozen.embedding, self.face.embedding)

    def test_snapshot_preserves_face_order_and_multiplicity(self):
        cache, selected = benchmark_snapshot.freeze_cache(self.cache,
            [self.face, self.face], [self.face])
        self.assertEqual(len(cache.faces), 2)
        self.assertIs(cache.faces[0], cache.faces[1])
        self.assertIs(selected[0], cache.faces[0])

    def test_original_mutation_cannot_change_scored_snapshot_and_final_check_rejects_it(self):
        inputs = self.inputs()
        self.face.embedding[:] = [0, 1]
        self.face.label = "Bob"
        frozen = inputs.snapshot_primary_faces[0]
        np.testing.assert_array_equal(frozen.embedding, [1, 0])
        self.assertEqual(frozen.label, "Alice")
        inputs.validate_snapshot()
        with self.assertRaisesRegex(RuntimeError, "cached faces changed"):
            inputs.validate()

    def test_transient_mutation_cannot_poison_a_checkpoint(self):
        inputs = self.inputs()
        self.face.embedding[:] = [0, 1]
        self.assertEqual(evaluation.evaluate_face(inputs.snapshot_primary_faces[0], self.db,
                                                 lane="strict").outcome, "correct")
        self.face.embedding[:] = [1, 0]
        inputs.validate()

    def test_block_guard_does_not_rehash_live_cache_but_final_guard_does(self):
        inputs = self.inputs()
        with patch.object(inputs, "_memory_signature", wraps=inputs._memory_signature) as signature:
            for _ in range(100):
                inputs.validate_snapshot()
            signature.assert_not_called()
            inputs.validate()
            signature.assert_called_once()

    def test_block_guard_still_rejects_file_replacement_and_detector_change(self):
        inputs = self.inputs()
        self.cache.config_fingerprint = "different"
        with self.assertRaisesRegex(RuntimeError, "detector changed"):
            inputs.validate_snapshot()
        self.cache.config_fingerprint = "detector"
        self.reference.write_bytes(b"replaced")
        with self.assertRaisesRegex(RuntimeError, "input changed"):
            inputs.validate_snapshot()


class SharedPrimaryTests(Fixtures):
    def setUp(self):
        super().setUp()
        self.directory = self.root / "stages"
        self.messages = []
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        for owner, key in ((sorter, "IDENTITY_HARD_NEGATIVES_FILE"),
                           (sorter, "IDENTITY_CONFIRMATIONS_FILE"),
                           (evaluation_enrollment, "DEFAULT_PATH"),
                           (evaluation, "DEFAULT_PROTECTED_SET"), (evaluation, "DEFAULT_PROTECTED_BASELINE")):
            self.stack.enter_context(patch.object(owner, key, self.root / (key + ".missing")))

    def metrics(self, db=None):
        return evaluation.cache_metrics(db or self.db, self.cache, selected_faces=[self.face],
            stage_directory=self.directory, progress=self.messages.append)

    def test_profile_and_recovery_share_identical_primary_rows(self):
        with patch.object(evaluation, "evaluate_face", wraps=evaluation.evaluate_face) as scoring:
            first = self.metrics()
            second = self.metrics(copy.deepcopy(self.db))
        self.assertEqual(first, second)
        self.assertEqual(first["incorrect"], 0)
        self.assertEqual(scoring.call_count, 1)
        self.assertTrue(any("1 reused" in message for message in self.messages))

    def test_profile_and_embedding_changes_never_reuse_old_scores(self):
        with patch.object(evaluation, "evaluate_face", wraps=evaluation.evaluate_face) as scoring:
            self.metrics()
            changed = copy.deepcopy(self.db)
            changed.strict_thresholds["Alice"] = .001
            self.metrics(changed)
            self.face.embedding[:] = [0, 1]
            self.metrics()
        self.assertEqual(scoring.call_count, 3)

    def test_source_replacement_or_hard_negative_change_invalidates_cache(self):
        with patch.object(evaluation, "evaluate_face", wraps=evaluation.evaluate_face) as scoring:
            self.metrics()
            self.reference.write_bytes(b"other reference")
            self.metrics()
            sorter.IDENTITY_HARD_NEGATIVES_FILE.write_text('{"version": 1, "examples": []}')
            self.metrics()
        self.assertEqual(scoring.call_count, 3)

    def test_live_memory_drift_blocks_publication_and_discards_namespace(self):
        with self.assertRaisesRegex(RuntimeError, "cached faces changed"):
            with primary_stage_cache.session(self.db, self.cache, [self.face], self.directory) as (_, _, checkpoints):
                self.face.label = "Bob"
                checkpoints.put("poison", {"value": "snapshot Alice"})
        self.face.label = "Alice"
        with primary_stage_cache.session(self.db, self.cache, [self.face], self.directory) as (_, _, checkpoints):
            self.assertIsNone(checkpoints.get("poison"))

    def test_policy_drift_discards_completed_blocks_even_after_restoration(self):
        initial = sorter.AUTO_PERSON_SINGLE_MATCH_DIST
        with self.assertRaisesRegex(RuntimeError, "policy changed"):
            with primary_stage_cache.session(self.db, self.cache, [self.face], self.directory) as (_, _, checkpoints):
                checkpoints.put("before", {"valid": True})
                with patch.object(sorter, "AUTO_PERSON_SINGLE_MATCH_DIST", initial / 2):
                    checkpoints.put("after", {"valid": False})
        with primary_stage_cache.session(self.db, self.cache, [self.face], self.directory) as (_, _, checkpoints):
            self.assertIsNone(checkpoints.get("before"))

    def test_live_profile_drift_invalidates_immutable_snapshot_work(self):
        import benchmark_snapshot
        frozen = benchmark_snapshot.freeze_profile(self.db)
        with self.assertRaisesRegex(RuntimeError, "Live primary profile changed"):
            with primary_stage_cache.session(frozen, self.cache, [self.face], self.directory,
                                             live_profile=self.db) as (_, _, checkpoints):
                self.db.strict_thresholds["Alice"] = .001
                checkpoints.put("changed", {"valid": False})

    def test_interruption_preserves_completed_immutable_work(self):
        with self.assertRaises(KeyboardInterrupt):
            with primary_stage_cache.session(self.db, self.cache, [self.face], self.directory) as (_, _, checkpoints):
                checkpoints.put("complete", {"valid": True})
                raise KeyboardInterrupt
        with primary_stage_cache.session(self.db, self.cache, [self.face], self.directory) as (_, _, checkpoints):
            self.assertEqual(checkpoints.get("complete"), {"valid": True})

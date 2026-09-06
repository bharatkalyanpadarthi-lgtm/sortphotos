"""Isolated regressions for the September architecture audit."""

import errno
import csv
import hashlib
import json
import os
import pickle
import sqlite3
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import analysis_index
import benchmark_detection
import content_identity
import copy_journal
import daily_identity_recovery
import evaluation_dataset
import face_detection
import identity_confirmations
import identity_evaluation
import identity_profiles
import review_identity_benchmark as benchmark
import review_unknown_identities as review
import secondary_identity_matcher
import source_batch_consensus
import shadow_evaluation
import sort_photos as sorter


class ArchitectureTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="face_architecture_test_")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.alice = np.eye(512, dtype=np.float32)[0]
        self.bob = np.zeros(512, dtype=np.float32)
        self.bob[:2] = [0.72, np.sqrt(1 - 0.72**2)]

    def face(self, name, embedding):
        path = self.root / name
        path.write_bytes(name.encode())
        return sorter.FaceRecord(
            path, 0, 0.99, 120, 120, 0, embedding, quality=1, cluster_id=1
        )

    def test_classification_write_cannot_validate_stale_detection(self):
        path = self.root / "image.jpg"
        path.write_bytes(b"old image")
        with analysis_index.AnalysisIndex(self.root / "index.sqlite") as index:
            index.replace_detections(path, "detector", "no_face_detected", [])
            path.write_bytes(b"new image containing a face")
            index.record_nudity(path, model="classifier", status="safe", detections=[])
            self.assertIsNone(index.cached_detections(path, "detector"))

    def test_detection_write_cannot_validate_stale_classification(self):
        path = self.root / "image.jpg"
        path.write_bytes(b"old image")
        with analysis_index.AnalysisIndex(self.root / "index.sqlite") as index:
            index.record_nudity(path, model="classifier", status="safe", detections=[])
            path.write_bytes(b"new image with different contents")
            index.replace_detections(path, "detector", "no_face_detected", [])
            self.assertIsNone(index.cached_nudity(path, "classifier"))

    def test_confirmed_reference_requires_original_content(self):
        people = self.root / "people"
        path = people / "Alice" / "photos" / "00001.jpg"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"confirmed original")
        confirmed = self.root / "confirmed.json"
        identity_confirmations.record(
            confirmed,
            person="Alice",
            organized_path=path,
            content_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            original_name=path.name,
        )
        self.assertIn(
            path, identity_confirmations.paths_for_person(confirmed, "Alice", people)
        )
        path.write_bytes(b"unconfirmed replacement")
        self.assertNotIn(
            path, identity_confirmations.paths_for_person(confirmed, "Alice", people)
        )

    def test_cluster_consensus_never_files_dissenting_member(self):
        records = [
            self.face(f"image-{i}.jpg", vector)
            for i, vector in enumerate([self.alice] * 4 + [self.bob])
        ]
        names = {1: "person_001"}
        db = sorter.IdentityDB(
            identities={"Alice": self.alice, "Bob": self.bob},
            source_counts={"Alice": 10, "Bob": 10},
        )
        with (
            patch.object(
                sorter.identity_hard_negatives, "vectors_by_person", return_value={}
            ),
            patch.object(
                sorter.appearance_profiles,
                "query_attributes",
                return_value=("unknown", 0.0),
            ),
        ):
            sorter.apply_identity_db_labels(
                records, names, db, use_secondary_verifier=False
            )
        self.assertNotEqual(names.get(records[-1].cluster_id), "Alice")
        self.assertIn(
            str(records[-1].src), daily_identity_recovery._unresolved(records, names)
        )

    def test_legacy_checkpoint_does_not_claim_missing_copy(self):
        record = self.face("image.jpg", self.alice)
        destination = self.root / "output"
        destination.mkdir()
        sorter._save_checkpoint(destination, {f"Alice||{record.src}||sharp||main"})
        with (
            patch.object(sorter, "DEDUP_DUPLICATES", False),
            patch.object(
                sorter, "analysis_index_file", return_value=self.root / "index.sqlite"
            ),
            patch.object(sorter, "classified_nudity_status", return_value="safe"),
            patch.object(
                sorter,
                "maybe_move_to_nudity_subfolder",
                side_effect=lambda p, *_a, **_kw: (p, None),
            ),
            patch.object(
                sorter, "_atomic_copy", side_effect=OSError("simulated failed copy")
            ),
        ):
            organized = sorter.organize_originals([record], {1: "Alice"}, destination)
        self.assertNotIn(record.src, organized)
        self.assertTrue(record.src.is_file())

    def test_large_recovery_contains_rotations(self):
        image = np.zeros((800, 800, 3), dtype=np.uint8)
        with patch.object(
            face_detection.cv2, "rotate", wraps=face_detection.cv2.rotate
        ) as rotate:
            list(face_detection.fallback_detection_views(image, max_dimension=1600))
        self.assertGreaterEqual(rotate.call_count, 3)

    def test_recovery_settings_invalidate_detection_signature(self):
        original = sorter.config_fingerprint()
        with patch.object(sorter, "RECOVERY_MIN_FACE_PX", 999):
            self.assertNotEqual(original, sorter.config_fingerprint())

    def test_missing_protected_dataset_blocks_existing_baseline(self):
        baseline = self.root / "baseline.json"
        baseline.write_text("{}")
        db = sorter.IdentityDB()
        with patch.object(
            identity_evaluation,
            "cache_metrics",
            return_value={"incorrect": 0, "precision": 1.0, "recall": 0.5},
        ):
            allowed, _ = identity_evaluation.activation_gate(
                db,
                db,
                sorter.CacheState(),
                protected_set=self.root / "missing.csv",
                protected_baseline=baseline,
            )
        self.assertFalse(allowed)

    def test_baseline_is_bound_to_evaluated_annotations(self):
        for change_at in (None, "load", "evaluate"):
            with self.subTest(change_at=change_at):
                folder = self.root / str(change_at)
                folder.mkdir()
                dataset = folder / "benchmark.csv"
                dataset.write_text("original annotations")
                digest = content_identity.content_sha256(dataset)
                baseline = folder / "baseline.json"
                validation = evaluation_dataset.DatasetValidation(
                    (), (), frozenset(evaluation_dataset.REQUIRED_CASE_TYPES))
                metrics = evaluation_dataset.EvaluationMetrics(1, 1, 1, 0, 1, 1, 0, 1)

                def load(_path):
                    if change_at == "load":
                        dataset.write_text("new annotations")
                    return validation

                def evaluate(*_args, **_kwargs):
                    if change_at == "evaluate":
                        dataset.write_text("new annotations")
                    return metrics, [{"source": "fixture", "identity_outcome": "correct"}]

                with (
                    patch("sys.argv", ["identity_evaluation.py", "--golden-set", str(dataset),
                        "--write-baseline", str(baseline), "--report-dir", str(folder),
                        "--fresh-detection"]),
                    patch.object(sorter, "load_identity_db", return_value=SimpleNamespace(
                        identities={"Alice": self.alice})),
                    patch.object(sorter, "load_cache", side_effect=AssertionError("live cache accessed")),
                    patch.object(evaluation_dataset, "load_dataset", side_effect=load),
                    patch.object(identity_evaluation, "evaluate_golden_set", side_effect=evaluate) as run,
                ):
                    result = identity_evaluation.main()
                if change_at is None:
                    self.assertEqual(result, 0)
                    self.assertEqual(json.loads(baseline.read_text())["dataset_sha256"], digest)
                else:
                    self.assertEqual(result, 7)
                    self.assertFalse(baseline.exists())
                if change_at == "load":
                    run.assert_not_called()
                else:
                    report = json.loads((folder / "golden_set_summary.json").read_text())
                    self.assertEqual(report["dataset_sha256"], digest)
                    self.assertEqual(report["dataset_unchanged"], change_at is None)

    def test_replaced_bytes_with_same_mtime_and_size_invalidate_all_hashes(self):
        path = self.root / "replace.jpg"
        path.write_bytes(b"old")
        stamp = path.stat().st_mtime_ns
        digest = review.item_content_sha256(path)
        with analysis_index.AnalysisIndex(self.root / "index.sqlite") as index:
            index.replace_detections(path, "v1", "no_face", [])
        path.write_bytes(b"new")
        os.utime(path, ns=(stamp, stamp))
        self.assertNotEqual(digest, review.item_content_sha256(path))
        with analysis_index.AnalysisIndex(self.root / "index.sqlite") as index:
            self.assertIsNone(index.cached_detections(path, "v1"))

    def test_wrong_analysis_content_cannot_commit(self):
        path = self.root / "image.jpg"
        path.write_bytes(b"new")
        with analysis_index.AnalysisIndex(self.root / "index.sqlite") as index:
            with self.assertRaises(ValueError):
                index.replace_detections(path, "detector", "no_face", [], expected_sha256="wrong")
            with self.assertRaises(ValueError):
                index.record_nudity(path, model="classifier", status="safe", detections=[], expected_sha256="wrong")
            self.assertIsNone(index.cached_detections(path, "detector"))
            self.assertIsNone(index.cached_nudity(path, "classifier"))

    def test_future_database_is_not_downgraded(self):
        path = self.root / "future.sqlite"
        with sqlite3.connect(path) as db:
            db.execute("CREATE TABLE metadata(key TEXT PRIMARY KEY, value TEXT)")
            db.execute("INSERT INTO metadata VALUES('schema_version', '999')")
        with self.assertRaises(ValueError):
            analysis_index.AnalysisIndex(path)
        with sqlite3.connect(path) as db:
            self.assertEqual(db.execute("SELECT value FROM metadata").fetchone()[0], "999")

    def test_reference_face_provenance_and_renamed_file(self):
        people = self.root / "people"
        path = people / "Alice" / "photos" / "old.jpg"
        path.parent.mkdir(parents=True)
        path.write_bytes(b"confirmed")
        registry = self.root / "confirmed.json"
        first, second = SimpleNamespace(crop_jpeg=b"face-one", face_index=0), SimpleNamespace(crop_jpeg=b"face-two", face_index=1)
        identity_confirmations.record(registry, person="Alice", organized_path=path,
            content_sha256=content_identity.content_sha256(path), original_name=path.name, face=second)
        moved = path.with_name("new.jpg")
        path.rename(moved)
        examples = identity_confirmations.examples_for_person(registry, "Alice", people)
        self.assertEqual(set(examples), {moved})
        self.assertEqual(identity_confirmations.selected_faces(examples[moved], [first, second]), [second])
        self.assertEqual(identity_confirmations.selected_faces({}, [first, second]), [])

    def test_copy_journal_reopens_and_revalidates_destination(self):
        path = self.root / "original.jpg"
        path.write_bytes(b"original")
        digest = content_identity.content_sha256(path)
        op = copy_journal.CopyJournal.operation_id("Alice", digest, "safe")
        journal = copy_journal.CopyJournal(self.root)
        journal.planned(op, digest)
        self.assertIsNone(journal.verified_destination(op, digest))
        journal.completed(op, digest, path)
        journal.close()
        journal = copy_journal.CopyJournal(self.root)
        self.addCleanup(journal.close)
        self.assertEqual(journal.verified_destination(op, digest), path)
        path.write_bytes(b"corrupt!")
        self.assertIsNone(journal.verified_destination(op, digest))
        with self.assertRaises(ValueError):
            journal.completed(op, digest, path)
        path.unlink()
        self.assertIsNone(journal.verified_destination(op, digest))

    def test_duplicate_bytes_are_not_independent_batch_votes(self):
        files = [self.root / "a.jpg", self.root / "copy.jpg"]
        for path in files:
            path.write_bytes(b"identical")
        evidence = [source_batch_consensus.BatchEvidence(str(path), 0, "batch", self.alice,
                    "Alice", .28, .1, .9, .3) for path in files]
        self.assertEqual(source_batch_consensus.consensus_decisions(evidence), {})

    def test_conflicting_prior_labels_cannot_merge(self):
        first, second = self.face("a.jpg", self.alice), self.face("b.jpg", self.alice)
        first.prior_label, second.prior_label = "Alice", "Bob"
        first.cluster_id, second.cluster_id = 1, 2
        self.assertEqual(sorter.merge_close_clusters([first, second]), 0)

    def test_unlabelled_member_does_not_inherit_prior_label(self):
        first, second = self.face("a.jpg", self.alice), self.face("b.jpg", self.alice)
        first.prior_label = "Alice"
        self.assertNotEqual(sorter.make_initial_name_map([first, second])[1], "Alice")

    def test_transformed_boxes_and_geometric_dedup(self):
        matrix = np.array([[1, 0, 150], [0, 1, 200], [0, 0, 1]], dtype=float)
        box, points = face_detection.original_geometry([10, 20, 50, 60], [[20, 30]], matrix)
        np.testing.assert_allclose(box, [160, 220, 200, 260])
        np.testing.assert_allclose(points, [[170, 230]])
        self.assertTrue(face_detection.same_detection(box, [161, 221, 201, 261]))
        self.assertFalse(face_detection.same_detection(box, [300, 220, 340, 260]))

    def test_holdout_removes_identical_reference_and_no_centroid_fallback(self):
        query, copy = self.root / "query.jpg", self.root / "copy.jpg"
        query.write_bytes(b"same")
        copy.write_bytes(b"same")
        db = sorter.normalize_identity_db(sorter.IdentityDB(
            identities={"Alice": self.alice}, prototypes={"Alice": [self.alice]},
            prototype_sources={"Alice": [str(copy)]}, source_counts={"Alice": 5}))
        center, prototypes = identity_evaluation.profile_without_source(db, "Alice", str(query))
        self.assertIsNone(center)
        self.assertEqual(prototypes, [])

    def test_secondary_holdout_cannot_reuse_excluded_centroid(self):
        source = self.root / "query.jpg"
        source.write_bytes(b"query")
        db = secondary_identity_matcher.SecondaryIdentityDB(
            identities={"Alice": self.alice}, prototypes={"Alice": [self.alice]},
            prototype_sources={"Alice": [str(source)]})
        matcher = secondary_identity_matcher.SecondaryMatcher(db, cache_path=self.root / "secondary.pkl")
        with patch.object(matcher, "embedding", return_value=self.alice):
            self.assertFalse(matcher.verify(b"crop", "Alice", excluded_source=source).accepted)

    def test_group_extra_accepted_person_is_a_false_accept(self):
        source = self.root / "group.jpg"
        source.write_bytes(b"group")
        face = SimpleNamespace(src_str=str(source))
        case = evaluation_dataset.EvaluationCase(source, "Alice", frozenset({"group"}), True,
            "unknown", True, expected_people=("Alice",), expected_face_count=2)
        predictions = [identity_evaluation.FacePrediction("Alice", 0, 1, True),
                       identity_evaluation.FacePrediction("Bob", 0, 1, True)]
        with patch.object(identity_evaluation, "predict_face", side_effect=predictions), \
             patch.object(sorter, "analysis_index_file", return_value=self.root / "index.sqlite"):
            metrics, rows = identity_evaluation.evaluate_golden_set((case,),
                sorter.CacheState(faces=[face, face]), sorter.IdentityDB(), lane="strict")
        self.assertEqual(rows[0]["identity_outcome"], "incorrect")
        self.assertEqual(metrics.identity_precision, .5)

    def test_selected_face_ignores_background_identity_not_detection(self):
        source = self.root / "foreground.jpg"
        source.write_bytes(b"foreground and background")
        main = SimpleNamespace(src_str=str(source), face_index=4, crop_jpeg=b"foreground")
        background = SimpleNamespace(src_str=str(source), face_index=1, crop_jpeg=b"background")
        case = evaluation_dataset.EvaluationCase(source, "Alice", frozenset({"known"}),
            True, "unknown", True, expected_face_count=2,
            identity_face_id=content_identity.face_identity(main))
        for lane in ("strict", "pipeline"):
            for predicted, outcome in (("Alice", "correct"), ("Wrong", "incorrect")):
                with self.subTest(lane=lane, predicted=predicted), \
                     patch.object(identity_evaluation, "predict_face", return_value=
                         identity_evaluation.FacePrediction(predicted, 0, 1, True)) as predict, \
                     patch.object(identity_evaluation, "heldout_identity_db", return_value=sorter.IdentityDB()), \
                     patch.object(identity_evaluation, "heldout_hard_negatives", return_value={}), \
                     patch.object(shadow_evaluation, "daily_plan", return_value={str(source): [
                         {"face_index": background.face_index, "person": "Bob"},
                         {"face_index": main.face_index, "person": predicted}]}) as planner, \
                     patch.object(sorter, "analysis_index_file", return_value=self.root / "index.sqlite"):
                    metrics, rows = identity_evaluation.evaluate_golden_set((case,),
                        sorter.CacheState(faces=[background, main]), sorter.IdentityDB(), lane=lane)
                self.assertEqual(rows[0]["identity_outcome"], outcome)
                self.assertEqual(rows[0]["faces_detected"], 2)
                self.assertEqual(rows[0]["identity_faces_scored"], 1)
                self.assertEqual(rows[0]["identity_faces_ignored"], 1)
                self.assertEqual(rows[0]["accepted_names"], predicted)
                self.assertEqual(metrics.missed_face_rate, 0)
                if lane == "pipeline":
                    self.assertEqual(planner.call_args.args[0], [background, main])
                    predict.assert_not_called()
                else:
                    predict.assert_called_once()
                    self.assertIs(predict.call_args.args[0], main)

    def test_missing_or_ambiguous_selected_face_blocks_evaluation(self):
        source = self.root / "selected.jpg"
        source.write_bytes(b"selected image")
        main = SimpleNamespace(src_str=str(source), face_index=0, crop_jpeg=b"foreground")
        other = SimpleNamespace(src_str=str(source), face_index=1, crop_jpeg=b"other")
        case = evaluation_dataset.EvaluationCase(source, "Alice", frozenset({"known"}),
            True, "unknown", True, expected_face_count=2,
            identity_face_id=content_identity.face_identity(main))
        for faces in ([], [other], [main, main]):
            with self.subTest(face_count=len(faces)), \
                 patch.object(sorter, "analysis_index_file", return_value=self.root / "index.sqlite"), \
                 patch.object(identity_evaluation, "predict_face") as predict:
                with self.assertRaisesRegex(RuntimeError, "missing or ambiguous"):
                    identity_evaluation.evaluate_golden_set((case,), sorter.CacheState(faces=faces),
                        sorter.IdentityDB(), lane="strict")
                predict.assert_not_called()

    def test_selected_face_dataset_roundtrip_and_validation(self):
        source = self.root / "source.jpg"
        source.write_bytes(b"image")
        path = self.root / "cases.csv"
        row = dict(source=str(source), expected_person="Alice", case_types="known",
            expected_face="true", expected_nudity="unknown", verified="true", notes="Ignore background",
            expected_face_count="2", identity_face_id="crop:" + "a" * 64)
        evaluation_dataset.write_template(path, [row])
        validation = evaluation_dataset.load_dataset(path)
        self.assertFalse(validation.errors)
        self.assertEqual(validation.cases[0].identity_face_id, row["identity_face_id"])
        self.assertEqual(benchmark.read_rows(path)[0]["identity_face_id"], row["identity_face_id"])
        for updates in ({"identity_face_id": "invalid"}, {"expected_face_count": ""},
                        {"case_types": "group|known", "expected_people": "Alice | Bob"},
                        {"expected_person": "", "expected_people": "Alice | Bob"},
                        {"expected_face": "false"}):
            invalid = dict(row, **updates)
            evaluation_dataset.write_template(path, [invalid])
            self.assertTrue(evaluation_dataset.load_dataset(path).errors)
            self.assertTrue(benchmark._verification_errors(benchmark._normalized_row(invalid)))
        row.pop("identity_face_id")
        evaluation_dataset.write_template(path, [row])
        self.assertEqual(evaluation_dataset.load_dataset(path).cases[0].identity_face_id, "")
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerow(row)
        legacy = evaluation_dataset.load_dataset(path)
        self.assertFalse(legacy.errors)
        self.assertEqual(legacy.cases[0].identity_face_id, "")

    def test_old_review_form_preserves_selected_face_until_explicit_clear(self):
        from urllib.parse import urlencode
        from urllib.request import Request, urlopen

        source = self.root / "foreground.jpg"
        source.write_bytes(b"image")
        dataset = self.root / "cases.csv"
        row = dict(source=str(source), expected_person="Alice", case_types="known",
            expected_face="true", expected_nudity="unknown", verified="true", notes="Foreground only",
            expected_face_count="2", identity_face_id="crop:" + "a" * 64)
        evaluation_dataset.write_template(dataset, [row])
        server = benchmark.BenchmarkServer(("127.0.0.1", 0), benchmark.Handler)
        server.dataset = dataset
        server.baseline = self.root / "baseline.json"
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            url = f"http://127.0.0.1:{server.server_port}/save"
            old_form = dict(row)
            old_form.pop("identity_face_id")
            with urlopen(Request(url, data=urlencode(old_form).encode()), timeout=5) as response:
                html = response.read().decode()
            self.assertEqual(benchmark.read_rows(dataset)[0]["identity_face_id"], row["identity_face_id"])
            self.assertIn("Confirmed face only", html)
            self.assertIn(f'value="{row["identity_face_id"]}" selected', html)
            with urlopen(Request(url, data=urlencode(dict(row, identity_face_id="")).encode()), timeout=5) as response:
                response.read()
            self.assertEqual(benchmark.read_rows(dataset)[0]["identity_face_id"], "")
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)

    def test_compiled_profiles_match_scalar_scoring(self):
        rng = np.random.default_rng(42)
        identities = {f"Person {i}": rng.normal(size=32) for i in range(8)}
        prototypes = {name: [rng.normal(size=32) for _ in range(4)] for name in identities}
        pose = {name: {"left_profile": [rng.normal(size=32)]} for name in identities}
        appearance = {name: {label: [rng.normal(size=32)] for label in
                      ("low_light", "normal_light", "era_newer", "era_older")} for name in identities}
        negatives = {name: [vector] for name, vector in identities.items()}
        settings = dict(pose_prototypes=pose, appearance_prototypes=appearance,
                        appearance_era_cutoffs={name: 100 for name in identities}, hard_negatives=negatives)
        compiled = identity_profiles.CompiledProfiles(identities, prototypes, **settings)
        for timestamp in (0, 20, 150):
            for lighting in ("unknown", "low_light", "normal_light"):
                for query in [rng.normal(size=32), identities["Person 0"]]:
                    options = dict(settings, pose_label="left_profile", lighting_label=lighting,
                                   capture_timestamp=timestamp)
                    expected = identity_profiles.rank_candidates(query, identities, prototypes, **options)
                    actual = identity_profiles.rank_candidates(query, identities, prototypes, compiled=compiled, **options)
                    self.assertEqual([item.name for item in actual], [item.name for item in expected])
                    np.testing.assert_allclose([item.distance for item in actual],
                                               [item.distance for item in expected], atol=1e-6)

    def test_worker_results_from_replaced_source_are_not_cached(self):
        record = self.face("source.jpg", self.alice)
        digest = content_identity.content_sha256(record.src)
        cached = sorter.record_to_cached(record, None)
        record.src.write_bytes(b"replacement")
        diagnostics = {}
        paths, faces = sorter.validate_detection_batch([record.src], [cached],
            {str(record.src): {"sha256": digest}}, diagnostics)
        self.assertEqual(paths, [])
        self.assertEqual(faces, [])
        self.assertIn("source_changed", diagnostics[str(record.src)])

    def test_shadow_uses_daily_consensus_without_copy_or_label_leakage(self):
        records = [self.face(f"shadow-{i}.jpg", vector) for i, vector in
                   enumerate([self.alice] * 4 + [self.bob])]
        faces = [sorter.record_to_cached(record, "IncorrectPriorLabel") for record in records]
        db = sorter.normalize_identity_db(sorter.IdentityDB(
            identities={"Alice": self.alice, "Bob": self.bob},
            source_counts={"Alice": 5, "Bob": 5}))
        with patch.object(sorter, "organize_originals", side_effect=AssertionError("shadow cannot copy")), \
             patch.object(sorter, "save_cache", side_effect=AssertionError("shadow cannot write cache")):
            plan = shadow_evaluation.daily_plan(faces, db, hard_negatives={})
        self.assertNotIn("Alice", [item["person"] for item in plan[str(records[-1].src)]])
        self.assertTrue(all(face.label == "IncorrectPriorLabel" for face in faces))
        self.assertTrue(all(record.src.is_file() for record in records))

    def test_changed_source_is_not_filed_from_old_recognition(self):
        record = self.face("source.jpg", self.alice)
        record.content_sha256 = content_identity.content_sha256(record.src)
        record.src.write_bytes(b"replacement")
        with patch.object(sorter, "analysis_index_file", return_value=self.root / "index.sqlite"):
            organized = sorter.organize_originals([record], {1: "Alice"}, self.root / "output")
        self.assertEqual(organized, set())
        self.assertTrue(record.src.exists())

    def test_verified_manual_name_survives_pending_profile_promotion(self):
        record = self.face("manual.jpg", self.alice)
        people = self.root / "people"
        destination = people / "New Person" / "photos" / "confirmed.jpg"
        destination.parent.mkdir(parents=True)
        destination.write_bytes(record.src.read_bytes())
        digest = content_identity.content_sha256(record.src)
        decision = dict(action="confirmed", person="New Person", content_sha256=digest)
        with patch.object(review, "_verified_organized_destination", return_value=destination):
            self.assertEqual(daily_identity_recovery._confirmation_person(
                decision, digest, people, sorter.IdentityDB()), "New Person")

    def test_benchmark_reseed_preserves_reviewed_labels_and_adds_new_cases(self):
        dataset = self.root / "benchmark.csv"
        first = self.face("reviewed.jpg", self.alice).src
        second = self.face("new.jpg", self.bob).src
        reviewed = {field: "" for field in benchmark.FIELDS}
        reviewed.update(
            source=str(first), expected_person="", expected_people="Alice | Bob",
            expected_face_count="2", case_types="group|known|normal",
            expected_face="true", expected_nudity="safe", verified="true",
            content_sha256=content_identity.content_sha256(first),
            group_id="manually-grouped-shoot", notes="Human-reviewed group",
        )
        benchmark.write_rows(dataset, [reviewed])
        stale = dict(reviewed, expected_person="Wrong Person", expected_people="",
                     expected_face_count="", case_types="known",
                     expected_nudity="unknown", group_id="", notes="Old enrollment")
        new = dict(reviewed, source=str(second), expected_person="Bob",
                   expected_people="Bob", expected_face_count="1", case_types="known",
                   content_sha256=content_identity.content_sha256(second))
        with (
            patch.object(sorter, "load_identity_db", return_value=None),
            patch.object(sorter, "load_cache", return_value=SimpleNamespace(faces=[])),
            patch.object(benchmark.pipeline_paths, "SOURCE_REVIEW", self.root / "review"),
            patch.object(benchmark.identity_hard_negatives, "load", return_value={}),
            patch.object(benchmark.evaluation_enrollment, "_read", return_value=[stale, new]),
        ):
            self.assertEqual(benchmark.seed_dataset(dataset), 2)
            rows = {row["source"]: row for row in benchmark.read_rows(dataset)}
            self.assertEqual(rows[str(first)], reviewed)
            self.assertEqual(rows[str(second)], new)
            signature = dataset.stat().st_mtime_ns
            self.assertEqual(benchmark.seed_dataset(dataset), 2)
            self.assertEqual(dataset.stat().st_mtime_ns, signature)
            self.assertEqual({row["source"]: row for row in benchmark.read_rows(dataset)}, rows)

    def test_benchmark_reseed_does_not_reverify_changed_or_pending_cases(self):
        dataset = self.root / "benchmark.csv"
        source = self.face("changed.jpg", self.alice).src
        pending = {field: "" for field in benchmark.FIELDS}
        pending.update(source=str(source), expected_person="Alice", case_types="known",
                       expected_face="true", expected_nudity="unknown", verified="false",
                       content_sha256=content_identity.content_sha256(source))
        benchmark.write_rows(dataset, [pending])
        source.write_bytes(b"replacement content must be reviewed")
        incoming = dict(pending, verified="true", expected_person="Bob",
                        content_sha256=content_identity.content_sha256(source))
        with (
            patch.object(sorter, "load_identity_db", return_value=None),
            patch.object(sorter, "load_cache", return_value=SimpleNamespace(faces=[])),
            patch.object(benchmark.pipeline_paths, "SOURCE_REVIEW", self.root / "review"),
            patch.object(benchmark.identity_hard_negatives, "load", return_value={}),
            patch.object(benchmark.evaluation_enrollment, "_read", return_value=[incoming]),
        ):
            benchmark.seed_dataset(dataset)
        self.assertEqual(benchmark.read_rows(dataset), [pending])
        validation = evaluation_dataset.load_dataset(dataset)
        self.assertFalse(validation.activation_ready)
        self.assertTrue(any("content changed" in error for error in validation.errors))

    def test_benchmark_second_launcher_does_not_seed_active_dataset(self):
        with (
            patch.object(benchmark.sys, "argv", ["review_identity_benchmark.py"]),
            patch.object(benchmark, "seed_dataset") as seed,
            patch.object(benchmark, "BenchmarkServer", side_effect=OSError(errno.EADDRINUSE, "in use")),
            patch.object(benchmark, "urlopen") as request,
        ):
            response = request.return_value.__enter__.return_value
            response.status = 200
            response.read.return_value = b"Protected Face Benchmark"
            self.assertEqual(benchmark.main(), 0)
            seed.assert_not_called()

    def benchmark_cases(self, count=3):
        cases = []
        for i in range(count):
            record = self.face(f"benchmark-{i}.jpg", self.alice)
            cases.append(evaluation_dataset.EvaluationCase(
                source=record.src, expected_person="Alice", case_types=frozenset({"known"}),
                expected_face=True, expected_nudity="unknown", verified=True,
                content_sha256=content_identity.content_sha256(record.src)))
        return cases

    def fake_benchmark_worker(self, command, **kwargs):
        with Path(command[-1]).open("rb") as handle:
            job = pickle.load(handle)
        payload = {"faces": [], "diagnostics": {}, "fingerprints": {}}
        for source in job["input_paths"]:
            payload["diagnostics"][source] = "no_face_detected"
            payload["fingerprints"][source] = {
                "sha256": content_identity.content_sha256(Path(source))}
        with Path(job["output_path"]).open("wb") as handle:
            pickle.dump(payload, handle)
        return SimpleNamespace(returncode=0, stdout="")

    def test_benchmark_detection_uses_bounded_workers_without_live_cache_writes(self):
        cases = self.benchmark_cases()
        memo = {}
        with (
            patch.object(benchmark_detection.subprocess, "run", side_effect=self.fake_benchmark_worker) as run,
            patch.object(sorter, "save_cache", side_effect=AssertionError("must not write live cache")),
            patch.object(sorter, "persist_detection_batch", side_effect=AssertionError("must not write index")),
            patch.object(sorter, "_build_app", side_effect=AssertionError("no model in parent")),
        ):
            result = benchmark_detection.detect_cases(cases, detected_faces=memo, batch_size=2)
            self.assertEqual(run.call_count, 2)
            self.assertEqual(result, {str(case.source): [] for case in cases})
            self.assertEqual(result, memo)
            benchmark_detection.detect_cases(cases, detected_faces=memo, batch_size=2)
            self.assertEqual(run.call_count, 2)

    def test_benchmark_failed_worker_cannot_return_partial_success(self):
        cases = self.benchmark_cases()
        calls = 0

        def worker(command, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 2:
                return SimpleNamespace(returncode=-9, stdout="worker killed")
            return self.fake_benchmark_worker(command, **kwargs)

        memo = {}
        with patch.object(benchmark_detection.subprocess, "run", side_effect=worker):
            with self.assertRaisesRegex(RuntimeError, "exit -9"):
                benchmark_detection.detect_cases(cases, detected_faces=memo, batch_size=2)
        self.assertEqual(memo, {})

    def test_benchmark_rejects_source_changed_during_worker(self):
        cases = self.benchmark_cases(1)

        def worker(command, **kwargs):
            result = self.fake_benchmark_worker(command, **kwargs)
            cases[0].source.write_bytes(b"changed during detection")
            return result

        with patch.object(benchmark_detection.subprocess, "run", side_effect=worker):
            with self.assertRaisesRegex(RuntimeError, "changed benchmark image"):
                benchmark_detection.detect_cases(cases)

    def test_benchmark_missing_worker_output_is_not_no_face(self):
        with patch.object(benchmark_detection.subprocess, "run",
                          return_value=SimpleNamespace(returncode=0, stdout="")):
            with self.assertRaisesRegex(RuntimeError, "batch failed"):
                benchmark_detection.detect_cases(self.benchmark_cases(1))

    def test_benchmark_gate_reports_killed_process_and_blocks_duplicate_runs(self):
        server = SimpleNamespace(gate_lock=threading.Lock(), dataset=self.root / "benchmark.csv",
                                 baseline=self.root / "baseline.json", last_gate_output="")
        with (
            patch.object(benchmark.evaluation_dataset, "load_dataset",
                         return_value=SimpleNamespace(activation_ready=True)),
            patch.object(benchmark.subprocess, "Popen") as popen,
        ):
            child = popen.return_value.__enter__.return_value
            child.stdout = ["Protected detection: completed 25/100\n"]
            child.wait.return_value = -9
            message = benchmark.BenchmarkServer.run_gate(server)
            self.assertIn("interrupted by signal 9", message)
            self.assertIn("no successful result", server.last_gate_output)
            self.assertFalse(server.gate_lock.locked())
            server.gate_lock.acquire()
            self.assertIn("already running", benchmark.BenchmarkServer.run_gate(server))
            self.assertEqual(popen.call_count, 1)
            server.gate_lock.release()


if __name__ == "__main__":
    unittest.main()

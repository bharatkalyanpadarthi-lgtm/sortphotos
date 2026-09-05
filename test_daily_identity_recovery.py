"""Isolated regressions for the daily/Quick Review recovery integration."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import daily_identity_recovery as recovery
import identity_evaluation
import review_unknown_identities as review
import secondary_identity_matcher as secondary
import sort_photos as sorter


class RecoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="daily_recovery_test_")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.people = self.root / "people"
        self.decisions = self.root / "decisions.json"
        self.cache = sorter.CacheState()
        self.alice, self.bob, self.unknown = np.eye(512, dtype=np.float32)[:3]
        self.db = sorter.normalize_identity_db(sorter.IdentityDB(
            identities={"Alice": self.alice, "Bob": self.bob},
            prototypes={"Alice": [self.alice], "Bob": [self.bob]},
            source_counts={"Alice": 6, "Bob": 6},
        ))
        self.matcher = type("Matcher", (), {
            "app": None,
            "flush": lambda _self: None,
            "verify": lambda _self, _crop, expected: secondary.SecondaryVerification(
                True, expected, 0.10, 0.50
            ),
        })()
        self.gate = patch.object(review, "prepare_automatic_review_gate",
                                 return_value=(True, {}, "passed")).start()
        self.preparer = patch.object(review, "prepare_secondary_verifier",
                                     return_value=(self.matcher, "ready")).start()
        patch.object(review, "build_trusted_review_prototypes",
                     return_value=(self.db.prototypes, {})).start()
        patch.object(recovery.identity_hard_negatives, "vectors_by_person", return_value={}).start()
        patch.object(review, "review_model_signature", return_value="model-v1").start()
        patch.object(review.appearance_profiles, "query_attributes",
                     return_value=("unknown", 0.0)).start()
        self.addCleanup(patch.stopall)

    def face(self, name, embedding, cluster=4, quality=0.8):
        path = self.root / "intake" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        return sorter.FaceRecord(
            src=path, face_index=0, det_score=.99, bbox_size=100,
            sharpness=100, yaw_proxy=0, quality=quality, embedding=embedding,
            crop_jpeg=b"synthetic-crop", cluster_id=cluster,
        )

    def plan(self, faces, names):
        return recovery.plan_recovery(
            faces, names, self.db, self.cache, people_root=self.people,
            decisions_path=self.decisions, gate_dir=self.root / "gate",
            progress=lambda _message: None,
        )

    def store_outcome(self, face, **outcome):
        digest = sorter.sha256_file(face.src)
        existing = review.load_decisions(self.decisions)
        existing["content_outcomes"][digest] = {"content_sha256": digest, **outcome}
        review.save_decisions(self.decisions, existing)

    def test_recovers_individuals_without_carrying_cluster_or_multi_face(self):
        faces = [self.face("alice.jpg", self.alice), self.face("uncertain.jpg", self.unknown),
                 self.face("bob.jpg", self.bob, -1), self.face("named.jpg", self.alice, 90),
                 self.face("group.jpg", self.alice, 6), self.face("group.jpg", self.bob, 6)]
        faces[-1].face_index = 1
        names = {4: "person_004", 6: "person_006", 90: "Already Chosen", -1: "unknown"}
        before = [face.src.read_bytes() for face in faces]
        plan = self.plan(faces, names)
        self.assertEqual([face.cluster_id for face in faces], [4, 4, -1, 90, 6, 6])
        self.assertEqual(sum(row["status"] == "accepted" for row in plan.rows), 2)
        self.assertEqual(recovery.apply_plan(plan, faces, names), 2)
        self.assertEqual(names[faces[0].cluster_id], "Alice")
        self.assertEqual(names[faces[2].cluster_id], "Bob")
        self.assertEqual(names[4], "person_004")
        self.assertEqual(faces[1].cluster_id, 4)
        self.assertEqual(names[90], "Already Chosen")
        self.assertEqual(faces[-1].cluster_id, 6)
        self.assertEqual(before, [face.src.read_bytes() for face in faces])
        self.assertEqual(self.preparer.call_count, 1)

    def test_failed_gate_and_missing_verifier_fail_closed(self):
        face = self.face("alice.jpg", self.alice)
        names = {4: "person_004"}
        self.gate.return_value = False, {}, "blocked"
        plan = self.plan([face], names)
        self.assertEqual(plan.counts, {"safety_benchmark_blocked": 1})
        self.assertEqual(recovery.apply_plan(plan, [face], names), 0)
        self.preparer.return_value = None, "refresh failed"
        self.assertEqual(self.plan([face], names).counts, {"verifier_unavailable": 1})
        self.assertEqual(names, {4: "person_004"})

    def test_verifier_error_discards_partial_automatic_results(self):
        faces = [self.face("a.jpg", self.alice), self.face("b.jpg", self.bob)]
        calls = []

        def verify(_crop, person):
            calls.append(person)
            if len(calls) == 2:
                raise RuntimeError("model interrupted")
            return secondary.SecondaryVerification(True, person, .10, .50)

        self.matcher.verify = verify
        plan = self.plan(faces, {4: "person_004"})
        self.assertEqual(plan.counts, {"recovery_error": 2})
        self.assertTrue(all(row["status"] == "held" for row in plan.rows))

    def test_confirmation_reuse_requires_same_bytes_and_correct_person_destination(self):
        faces = [self.face(name, self.alice) for name in
                 ("replay.jpg", "missing.jpg", "changed.jpg", "outside.jpg", "wrong-person.jpg")]
        for face in faces:
            destination = self.people / "Alice" / "photos" / face.src.name
            if face.src.name == "outside.jpg":
                destination = self.root / "outside" / face.src.name
            if face.src.name == "wrong-person.jpg":
                destination = self.people / "Bob" / "photos" / face.src.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            if face.src.name != "missing.jpg":
                destination.write_bytes(b"different" if face.src.name == "changed.jpg"
                                        else face.src.read_bytes())
            self.store_outcome(face, action="confirmed", person="Alice", destination=str(destination))
        self.gate.return_value = False, {}, "blocked"
        plan = self.plan(faces, {4: "person_004"})
        accepted = [row for row in plan.rows if row["status"] == "accepted"]
        self.assertEqual(len(accepted), 1)
        self.assertEqual(Path(accepted[0]["source"]).name, "replay.jpg")
        self.assertEqual(accepted[0]["reason"], "verified_confirmation_replay")

    def test_manual_decisions_and_rejections_are_not_overridden(self):
        faces = [self.face(name, self.alice) for name in
                 ("ignored.jpg", "junk.jpg", "keep.jpg", "reject.jpg", "old-model.jpg")]
        self.store_outcome(faces[0], action="ignored")
        self.store_outcome(faces[1], action="moved_to_junk")
        self.store_outcome(faces[2], action="keep_unknown", model_signature="model-v1")
        self.store_outcome(faces[3], action="keep_unknown", model_signature="old",
                           rejected_people=["Alice"])
        self.store_outcome(faces[4], action="keep_unknown", model_signature="old")
        plan = self.plan(faces, {4: "person_004"})
        self.assertEqual(plan.counts["retained_user_decision"], 3)
        self.assertEqual(plan.counts["user_rejected_candidate"], 1)
        self.assertEqual(sum(row["status"] == "accepted" for row in plan.rows), 1)

    def test_changed_source_and_already_labeled_records_cannot_be_adopted(self):
        face = self.face("alice.jpg", self.alice)
        names = {4: "person_004"}
        plan = self.plan([face], names)
        face.src.write_bytes(b"different longer contents")
        self.assertEqual(recovery.apply_plan(plan, [face], names), 0)
        self.assertEqual(face.identity_review_reason, "source_changed_during_recovery")
        plan = self.plan([face], names)
        names[4] = "Manual Choice"
        self.assertEqual(recovery.apply_plan(plan, [face], names), 0)
        self.assertEqual(names[4], "Manual Choice")

    def test_weak_or_dissenting_evidence_stays_in_review(self):
        faces = [self.face("low-quality.jpg", self.alice, quality=.1),
                 self.face("ambiguous.jpg", (self.alice+self.bob)/np.sqrt(2)),
                 self.face("dissent.jpg", self.alice)]
        self.matcher.verify = lambda _crop, _person: secondary.SecondaryVerification(
            True, "Bob", .10, .50
        )
        plan = self.plan(faces, {4: "person_004"})
        self.assertTrue(all(row["status"] == "held" for row in plan.rows))

    def test_empty_work_does_not_load_verifier_and_report_is_machine_readable(self):
        face = self.face("named.jpg", self.alice)
        plan = self.plan([face], {4: "Alice"})
        self.preparer.assert_not_called()
        output = self.root / "report.json"
        recovery.write_report(plan, output)
        self.assertEqual(json.loads(output.read_text())["counts"], {})

    def test_gate_signature_changes_with_thresholds_and_pose_profiles(self):
        before = review.automatic_review_gate_signature(self.db)
        self.db.match_thresholds["Alice"] = .123
        calibrated = review.automatic_review_gate_signature(self.db)
        self.assertNotEqual(before, calibrated)
        self.db.pose_prototypes["Alice"] = {"left_profile": [self.bob]}
        self.assertNotEqual(calibrated, review.automatic_review_gate_signature(self.db))

    def test_same_database_benchmark_is_not_evaluated_twice(self):
        metrics = {"incorrect": 0, "precision": 1.0, "recall": .5}
        with patch.object(identity_evaluation, "cache_metrics", return_value=metrics) as evaluate:
            allowed, _report = identity_evaluation.activation_gate(
                self.db, self.db, self.cache,
                protected_set=self.root / "absent.csv",
                protected_baseline=self.root / "absent.json",
            )
        self.assertTrue(allowed)
        self.assertEqual(evaluate.call_count, 1)

    def test_daily_cannot_use_ungated_legacy_secondary_matching(self):
        face = self.face("uncertain.jpg", self.unknown)
        with patch.object(secondary, "load", side_effect=AssertionError("ungated load")):
            self.assertEqual(sorter.apply_identity_db_labels(
                [face], {4: "person_004"}, self.db, use_secondary_verifier=False
            ), 0)

    def test_later_anchor_merge_cannot_override_recovery_rejection(self):
        strong = self.face("strong.jpg", self.alice)
        weak = self.face("weak.jpg", self.alice, quality=.1)
        names = {4: "person_004"}
        plan = self.plan([strong, weak], names)
        self.assertEqual(recovery.apply_plan(plan, [strong, weak], names), 1)
        self.assertEqual(sorter.anchor_cluster_merge(
            [strong, weak], names, self.root / "clusters"
        ), 0)
        self.assertEqual(weak.cluster_id, 4)
        self.assertEqual(names[4], "person_004")

    def test_recovered_original_uses_verified_copy_and_duplicate_resume(self):
        strong = self.face("strong.jpg", self.alice)
        weak = self.face("weak.jpg", self.unknown)
        names = {4: "person_004"}
        plan = self.plan([strong, weak], names)
        recovery.apply_plan(plan, [strong, weak], names)
        with patch.object(sorter, "analysis_index_file", return_value=self.root / "analysis.sqlite"), \
                patch.object(sorter, "classified_nudity_status", return_value="safe"), \
                patch.object(sorter, "maybe_move_to_nudity_subfolder",
                             side_effect=lambda path, *_args, **_kwargs: (path, "")), \
                patch.object(sorter, "DEDUP_DUPLICATES", False):
            for _attempt in range(2):
                organized = sorter.organize_originals([strong, weak], names, self.people)
                self.assertEqual(organized, {strong.src})
        copied = list((self.people / "Alice" / "photos").glob("*.jpg"))
        self.assertEqual(len(copied), 1)
        self.assertEqual(copied[0].read_bytes(), strong.src.read_bytes())
        self.assertTrue(weak.src.is_file())
        self.assertFalse((self.people / "person_004").exists())

    def test_new_recovery_is_not_an_anchor_for_other_unverified_faces(self):
        named = self.face("group.jpg", self.alice, 90)
        unverified = self.face("group.jpg", self.bob, 4)
        unverified.face_index = 1
        strong = self.face("bob.jpg", self.bob, -1)
        faces = [named, unverified, strong]
        names = {90: "Alice", 4: "person_004", -1: "unknown"}
        plan = self.plan(faces, names)
        self.assertEqual(recovery.apply_plan(plan, faces, names), 1)
        self.assertEqual(sorter.anchor_cluster_merge(faces, names, self.root / "clusters"), 0)
        self.assertEqual(names[4], "person_004")

    def test_recovery_guards_survive_saved_session_and_legacy_sessions_load(self):
        face = self.face("alice.jpg", self.alice)
        face.identity_review_reason = "held"
        face.recovery_assigned = True
        with patch.object(sorter, "CACHE_DIR", self.root / "cache"), \
                patch.object(sorter, "LABEL_STATE_FILE", self.root / "cache" / "state.pkl"):
            sorter.save_labeling_state([face], {4: "person_004"}, self.root, face.src.parent)
            state = sorter.load_labeling_state()
        restored = sorter.cached_to_record(state.faces[0])
        sorter.restore_recovery_state(restored, state, 0)
        self.assertEqual(restored.identity_review_reason, "held")
        self.assertTrue(restored.recovery_assigned)
        del state.recovery_metadata
        legacy = sorter.cached_to_record(state.faces[0])
        sorter.restore_recovery_state(legacy, state, 0)
        self.assertFalse(legacy.recovery_assigned)


if __name__ == "__main__":
    unittest.main(verbosity=2)

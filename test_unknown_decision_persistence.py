"""Offline regression tests: decisions survive model changes and duplicate copies."""

import json
import tempfile
import threading
import unittest
from collections import Counter
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.request import urlopen

import review_unknown_identities as review


class DecisionPersistenceTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.unknown = self.root / "unknown"
        self.unknown.mkdir()
        self.first = self.unknown / "first.jpg"
        self.copy = self.unknown / "copy.jpg"
        self.first.write_bytes(b"same image bytes")
        self.copy.write_bytes(self.first.read_bytes())
        self.path = self.root / "decisions.json"
        self.decisions = review.load_decisions(self.path)

    def record(self, path=None, action="keep_unknown", signature="old", **extra):
        path = path or self.first
        review.record_path_decision(
            self.decisions, key=review.item_key(path), path=path, action=action,
            model_signature=signature, **extra,
        )

    def reload(self):
        review.save_decisions(self.path, self.decisions)
        self.decisions = review.load_decisions(self.path)

    def test_keep_and_ignore_survive_profile_changes(self):
        for action in ("keep_unknown", "ignored"):
            self.record(action=action)
            self.reload()
            self.assertTrue(review.resolved_item(self.decisions, self.copy, model_signature="new"))
            self.assertEqual(review.needs_silent_recheck(self.decisions, self.copy, "new"), action == "keep_unknown")

    def test_latest_choice_wins_for_every_copy_even_same_clock_tick(self):
        with patch.object(review.time, "time_ns", return_value=10), patch.object(review.time, "time", return_value=1):
            self.record(action="confirmed", person="Alice")
            self.record(self.copy, action="ignored")
            self.assertEqual(review.decision_for_path(self.decisions, self.first)[1]["action"], "ignored")
            self.record(action="keep_unknown")
        self.reload()
        self.assertEqual(review.decision_for_path(self.decisions, self.copy)[1]["action"], "keep_unknown")

    def test_legacy_path_decision_backfills_content_and_survives_rename(self):
        record = dict(action="keep_unknown", content_sha256=review.item_content_sha256(self.first),
                      path=str(self.first), decided_at=1, model_signature="old")
        self.path.write_text(json.dumps(dict(version=1, items={review.legacy_item_key(self.first): record})))
        self.decisions = review.load_decisions(self.path)
        self.first.rename(self.unknown / "renamed.jpg")
        self.assertTrue(review.resolved_item(self.decisions, self.copy, model_signature="new"))

    def test_changed_bytes_at_same_path_do_not_inherit_decision(self):
        self.record(action="confirmed", person="Alice")
        self.reload()
        self.first.write_bytes(b"different image content")
        self.assertIsNone(review.decision_for_path(self.decisions, self.first))
        self.assertEqual(review.decision_for_path(self.decisions, self.copy)[1]["person"], "Alice")

    def test_hashless_legacy_decision_requires_original_signature(self):
        key = review.legacy_item_key(self.first)
        self.path.write_text(json.dumps(dict(version=1, items={key: dict(action="ignored", path=str(self.first))})))
        self.decisions = review.load_decisions(self.path)
        self.assertTrue(review.resolved_item(self.decisions, self.first))
        self.first.write_bytes(b"new bytes")
        self.assertIsNone(review.decision_for_path(self.decisions, self.first))

    def test_newer_hashless_legacy_choice_applies_to_all_identical_copies(self):
        digest = review.item_content_sha256(self.first)
        older = dict(action="keep_unknown", content_sha256=digest, decided_at=1, path=str(self.copy))
        latest = dict(action="ignored", decided_at=2, path=str(self.first))
        self.path.write_text(json.dumps(dict(version=2, items={
            review.item_key(self.copy): older, review.legacy_item_key(self.first): latest,
        }, content_outcomes={digest: older})))
        self.decisions = review.load_decisions(self.path)
        self.assertEqual(review.decision_for_path(self.decisions, self.copy)[1]["action"], "ignored")
        self.reload()
        self.assertEqual(review.decision_for_path(self.decisions, self.copy)[1]["action"], "ignored")

    def test_legacy_content_migration_does_not_guess_when_signature_changed(self):
        legacy_key = review.legacy_item_key(self.first)
        self.first.write_bytes(b"different bytes after original choice")
        self.path.write_text(json.dumps(dict(version=2, items={legacy_key: dict(
            action="confirmed", person="Alice", decided_at=2, path=str(self.first)
        )})))
        self.decisions = review.load_decisions(self.path)
        self.assertIsNone(review.decision_for_path(self.decisions, self.first))

    def legacy_reset(self, *, later_choice=False):
        digest = review.item_content_sha256(self.first)
        reset = dict(action="", previous_action="keep_unknown", previous_model_signature="old",
                     model_signature="new", outcome_state="requeued_model_changed", decided_at=100,
                     path=str(self.first), content_sha256=digest)
        outcomes = {digest: dict(action="ignored", decided_at=101, content_sha256=digest)} if later_choice else {}
        raw = json.dumps(dict(version=2, items={review.item_key(self.first): reset}, content_outcomes=outcomes))
        self.path.write_text(raw)
        return raw

    def test_profile_reset_migration_is_reversible_and_idempotent(self):
        raw = self.legacy_reset()
        self.decisions = review.load_decisions(self.path)
        self.assertEqual(self.path.read_text(), raw, "read-only load changed the live file")
        self.assertTrue(review.resolved_item(self.decisions, self.copy, model_signature="new"))
        self.assertTrue(review.needs_silent_recheck(self.decisions, self.copy, "new"))
        self.reload()
        backups = list(self.root.glob("*.bak"))
        self.assertEqual(len(backups), 1)
        self.assertEqual(backups[0].read_text(), raw)
        self.reload()
        self.assertEqual(len(list(self.root.glob("*.bak"))), 1)

    def test_stale_reset_never_shadows_latest_content_choice(self):
        self.legacy_reset(later_choice=True)
        self.decisions = review.load_decisions(self.path)
        self.assertEqual(review.decision_for_path(self.decisions, self.first)[1]["action"], "ignored")
        self.assertFalse(review.needs_silent_recheck(self.decisions, self.copy, "new"))

    def test_silent_recheck_once_per_content_per_model_preserves_choice_time(self):
        self.record()
        before = dict(review.decision_for_path(self.decisions, self.first)[1])
        review.mark_silent_recheck(self.decisions, self.first, "new")
        self.reload()
        self.assertFalse(review.needs_silent_recheck(self.decisions, self.copy, "new"))
        self.assertTrue(review.needs_silent_recheck(self.decisions, self.copy, "next"))
        self.assertEqual(review.decision_for_path(self.decisions, self.copy)[1], before)
        self.assertTrue(review.resolved_item(self.decisions, self.copy, model_signature="next"))

    def test_old_legacy_copy_does_not_repeat_a_later_completed_content_check(self):
        digest = review.item_content_sha256(self.first)
        older = dict(action="keep_unknown", content_sha256=digest, decided_at=1, model_signature="old", path=str(self.copy))
        newer = dict(action="keep_unknown", decided_at=2, model_signature="old", path=str(self.first))
        self.path.write_text(json.dumps(dict(version=2, items={
            review.item_key(self.copy): older, review.legacy_item_key(self.first): newer,
        }, content_outcomes={digest: older}, content_rechecks={digest: dict(
            model_signature="new", decision_order=[1, 1_000_000_000], checked_at=3,
        )})))
        self.decisions = review.load_decisions(self.path)
        self.assertFalse(review.needs_silent_recheck(self.decisions, self.first, "new"))
        self.assertFalse(review.needs_silent_recheck(self.decisions, self.copy, "new"))
        self.assertTrue(review.needs_silent_recheck(self.decisions, self.copy, "next"))

    def test_new_explicit_choice_invalidates_old_silent_checkpoint(self):
        self.record()
        review.mark_silent_recheck(self.decisions, self.first, "new")
        self.record(self.copy, action="ignored")
        self.assertFalse(review.needs_silent_recheck(self.decisions, self.first, "next"))
        self.assertFalse(self.decisions["content_rechecks"])

    def test_verified_recovery_supersedes_retained_decision(self):
        self.record()
        self.record(self.copy, action="confirmed", signature="new", person="Alice")
        self.reload()
        record = review.decision_for_path(self.decisions, self.first)[1]
        self.assertEqual(record["action"], "confirmed")
        self.assertFalse(review.automatic_check_allowed(self.decisions, self.first, "next"))

    def test_unverified_confirmation_state_does_not_resurrect_old_confirmation(self):
        self.record(action="confirmed", person="Alice")
        self.record(self.copy, action="", outcome_state="requeued_unverified_confirmation")
        self.reload()
        self.assertEqual(review.decision_for_path(self.decisions, self.first)[1]["action"], "")

    def test_candidate_rejection_applies_to_identical_copies(self):
        item = SimpleNamespace(path=self.first, key=review.item_key(self.first))
        review.record_rejection(self.decisions, item, "Alice")
        self.assertEqual(review.decision_for_path(self.decisions, self.copy)[1]["rejected_people"], ["Alice"])

    def batch_state(self):
        return dict(decisions_path=self.path, model_signature="new", identity_db=None,
                    workers=1, primary_det_size=320, fallback_det_size=320, hard_negatives={},
                    cluster_eps=0.3, auto_review_allowed=True, route_unsupported=True,
                    unknown_root=self.unknown, output_dir=self.root, batch_limit=500,
                    auto_review_gate={"signature": "gate"})

    def test_negative_silent_result_stays_hidden(self):
        self.record()
        self.reload()
        item = SimpleNamespace(path=self.first, key=review.item_key(self.first))
        with patch.object(review, "collect_items", return_value=([item], Counter(), [])), \
             patch.object(review, "cluster_items", return_value=[]), \
             patch.object(review, "apply_automatic_review", return_value=dict(confirmed=0, failed=0)):
            review._run_automatic_sweep_batch(self.batch_state(), [self.first])
        current = review.load_decisions(self.path)
        self.assertTrue(review.resolved_item(current, self.copy, model_signature="new"))
        self.assertFalse(review.needs_silent_recheck(current, self.copy, "new"))

    def test_silent_detector_failure_preserves_choice_and_allows_retry(self):
        self.record()
        self.reload()
        failed = SimpleNamespace(path=self.first, key=review.item_key(self.first), queue_kind="technical_review")
        with patch.object(review, "collect_items", return_value=([], Counter(), [failed])), \
             patch.object(review, "cluster_items", return_value=[]), \
             patch.object(review, "route_unsupported_unknowns") as route, \
             patch.object(review, "apply_automatic_review", return_value=dict(confirmed=0, failed=0)):
            result = review._run_automatic_sweep_batch(self.batch_state(), [self.first])
        route.assert_called_once_with(unittest.mock.ANY, [])
        self.assertEqual(result["completed_content_keys"], [])
        current = review.load_decisions(self.path)
        self.assertTrue(review.needs_silent_recheck(current, self.copy, "new"))
        self.assertTrue(review.resolved_item(current, self.copy))

    def test_gate_failure_never_rechecks_or_mutates_decisions(self):
        self.record()
        self.reload()
        before = self.path.read_bytes()
        state = self.batch_state()
        state["auto_review_allowed"] = False
        with patch.object(review, "collect_items") as collect:
            review.run_automatic_sweep(state)
        collect.assert_not_called()
        self.assertEqual(self.path.read_bytes(), before)

    def test_sweep_checks_identical_bytes_only_once_and_resumes(self):
        self.record()
        self.reload()
        digest = review.item_content_sha256(self.first)
        with patch.object(review, "_run_automatic_sweep_batch", return_value=dict(
                confirmed=0, failed=0, completed_content_keys=[digest])) as batch:
            result = review.run_automatic_sweep(self.batch_state())
            self.assertEqual(result["scanned"], 1)
            batch.assert_called_once()
            second = review.run_automatic_sweep(self.batch_state())
            self.assertTrue(second["cached"])
            batch.assert_called_once()

    def test_loaded_cards_use_content_decision(self):
        self.record(self.copy, action="ignored")
        item = SimpleNamespace(path=self.first, key=review.item_key(self.first))
        self.assertEqual(review.decision_for_item(self.decisions, item)["action"], "ignored")

    def test_automatic_confirmation_respects_latest_choice_and_rejected_person(self):
        item = SimpleNamespace(path=self.first, key=review.item_key(self.first))
        match = SimpleNamespace(item_keys=(item.key,), lane="test", person="Alice",
                                support=1, total=1, distance=0.1, margin=0.3)
        for action, signature, rejected, expected in (
            ("ignored", "old", [], 0),
            ("keep_unknown", "new", [], 0),
            ("keep_unknown", "old", ["Alice"], 0),
            ("keep_unknown", "old", [], 1),
        ):
            with self.subTest(action=action, signature=signature, rejected=rejected):
                self.decisions = review.load_decisions(self.root / "missing.json")
                self.record(action=action, signature=signature, rejected_people=rejected)
                self.reload()
                state = self.batch_state()
                state["items_by_key"] = {item.key: item}
                with patch.object(review, "automatic_matches", return_value=[match]), \
                     patch.object(review, "apply_decision") as apply:
                    result = review.apply_automatic_review(state, [])
                self.assertEqual(apply.call_count, expected)
                self.assertEqual(result["confirmed"], expected)

    def test_rendered_duplicate_card_is_not_selectable(self):
        self.record(self.copy, action="ignored")
        face = SimpleNamespace(quality=0.8, yaw_proxy=0, bbox_size=100, sharpness=100, pose_label="frontal")
        item = review.UnknownItem(review.item_key(self.first), self.first, face, ())
        cluster = review.UnknownCluster("cluster", (item,), (), 0)
        page = review.render_html([cluster], self.decisions, [], {}, interactive=True)
        self.assertIn("Ignored", page)
        self.assertIn("data-pending='0'", page)
        self.assertIn("applyResolvedItems", page)

    def test_failed_auto_move_does_not_mark_silent_check_complete(self):
        self.record()
        self.reload()
        item = SimpleNamespace(path=self.first, key=review.item_key(self.first))
        with patch.object(review, "collect_items", return_value=([item], Counter(), [])), \
             patch.object(review, "cluster_items", return_value=[]), \
             patch.object(review, "apply_automatic_review", return_value=dict(confirmed=0, failed=1)):
            result = review._run_automatic_sweep_batch(self.batch_state(), [self.first])
        self.assertEqual(result["completed_content_keys"], [])
        current = review.load_decisions(self.path)
        self.assertTrue(review.needs_silent_recheck(current, self.copy, "new"))

    def test_job_poll_does_not_block_behind_profile_refresh(self):
        lock = threading.Lock()
        lock.acquire()
        state = dict(lock=lock, quiet=True, action_queue=SimpleNamespace(
            snapshot=lambda *args, **kwargs: dict(jobs=[], summary=dict(running=0, queued=0))
        ))
        server = review.ThreadingHTTPServer(("127.0.0.1", 0), review.make_handler(state))
        worker = threading.Thread(target=server.serve_forever, daemon=True)
        worker.start()
        try:
            with urlopen(f"http://127.0.0.1:{server.server_port}/jobs", timeout=1) as response:
                payload = json.load(response)
            self.assertEqual(payload["summary"]["running"], 0)
        finally:
            lock.release()
            server.shutdown()
            server.server_close()
            worker.join(timeout=2)


if __name__ == "__main__":
    unittest.main()

import tempfile
import unittest
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import evaluation_dataset
import identity_evaluation
import identity_profiles
import recognition_policy
import review_reasons
import review_unknown_identities as review
import secondary_identity_matcher as secondary
import sort_photos as sorter


class ReviewExplanationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.path = self.root / 'case.jpg'
        self.path.write_bytes(b'synthetic fixture')
        self.db = sorter.IdentityDB(identities={'Alice': np.array([1., 0.])},
                                   match_thresholds={'Alice': .30}, strict_thresholds={'Alice': .20})

    def item(self, distance=.10, margin=.60, quality=.9, verifier=None):
        candidates = (identity_profiles.IdentityCandidate('Alice', 1-distance, distance),
                      identity_profiles.IdentityCandidate('Bob', 1-distance-margin, distance+margin))
        face = SimpleNamespace(quality=quality, crop_jpeg=b'crop', pose_label='frontal',
                               src_str=str(self.path), embedding=np.array([1., 0.]))
        return review.UnknownItem('item', self.path, face, candidates, verifier)

    def codes(self, item, **options):
        return {row['code'] for row in review_reasons.explain(item, self.db, settings=review, **options)}

    def test_strict_single_requires_explicit_lane_permission(self):
        item = self.item()
        cluster = review.UnknownCluster('cluster', (item,), item.candidates, 0.)
        self.assertEqual(review.automatic_matches([cluster], self.db, require_secondary=True), [])
        matches = review.automatic_matches([cluster], self.db, require_secondary=True, allow_strict_single=True)
        self.assertEqual(matches[0].lane, 'strict_single')
        self.assertFalse(recognition_policy.strict_lane_allowed({'allowed': True}))
        self.assertFalse(recognition_policy.strict_lane_allowed({'allowed': False, 'strict_single_allowed': True}))

    def test_confident_matcher_disagreement_still_blocks_strict_single(self):
        item = self.item(verifier=secondary.SecondaryVerification(True, 'Bob', .1, .4))
        cluster = review.UnknownCluster('cluster', (item,), item.candidates, 0.)
        self.assertEqual(review.automatic_matches([cluster], self.db,
                         require_secondary=True, allow_strict_single=True), [])
        self.assertIn('matcher_disagreement', self.codes(item))

    def test_low_quality_close_alternative_and_weak_confidence_are_distinct(self):
        self.assertIn('poor_face_quality', self.codes(self.item(quality=.1)))
        self.assertIn('similar_alternative', self.codes(self.item(margin=.01)))
        self.assertIn('weak_confidence', self.codes(self.item(distance=.7)))

    def test_eligible_match_explains_gate_not_bad_quality(self):
        item = self.item(verifier=secondary.SecondaryVerification(True, 'Alice', .1, .4))
        self.assertEqual(self.codes(item), {'safety_gate'})
        self.assertEqual(self.codes(item, enabled=True), {'ready'})

    def test_strict_and_missing_verifier_have_clear_reasons(self):
        self.assertIn('strict_safety_gate', self.codes(self.item()))
        self.assertIn('independent_not_evaluated', self.codes(self.item(distance=.29)))

    def test_failed_gate_never_files_even_with_strong_candidate(self):
        item = self.item()
        state = {'identity_db': self.db, 'auto_review_allowed': False,
                 'auto_review_gate': {'allowed': False, 'strict_single_allowed': True}}
        with patch.object(review, 'apply_decision') as apply:
            result = review.apply_automatic_review(state, [review.UnknownCluster('cluster', (item,), item.candidates, 0.)])
        self.assertEqual(result['confirmed'], 0)
        apply.assert_not_called()

    def test_reason_html_escapes_candidate_names(self):
        item = self.item()
        reasons = {item.key: [{'code': 'matcher_disagreement', 'message': 'Disagrees with <script>bad</script>'}]}
        page = review.render_html([review.UnknownCluster('c', (item,), item.candidates, 0.)],
                                  {}, ['Alice'], {'review_reasons': reasons}, interactive=True)
        self.assertIn('Disagrees with &lt;script&gt;bad&lt;/script&gt;', page)
        self.assertNotIn('Disagrees with <script>', page)

    def test_unknown_false_accept_blocks_both_lane_verdicts(self):
        item = self.item(verifier=secondary.SecondaryVerification(True, 'Alice', .1, .4))
        known = evaluation_dataset.EvaluationCase(self.path, 'Alice', frozenset({'known'}), True, 'safe', True)
        unknown = evaluation_dataset.EvaluationCase(self.path, '', frozenset({'unknown'}), True, 'safe', True)
        datasets = [evaluation_dataset.DatasetValidation((known,), (), frozenset({'known'})),
                    evaluation_dataset.DatasetValidation((unknown,), (), frozenset({'unknown'}))]
        matcher = SimpleNamespace(db=secondary.SecondaryIdentityDB(),
                                  verify=lambda *a, **k: item.secondary, flush=lambda: None)
        prepared = SimpleNamespace(rank=lambda *a, **k: item.candidates, validate=lambda: None)
        with patch.object(evaluation_dataset, 'load_dataset', side_effect=datasets), \
             patch('evaluation_profiles.EvaluationProfiles', return_value=prepared), \
             patch.object(review, 'AUTO_GATE_MIN_CONFIRMED_CASES', 1):
            allowed, report = review.evaluate_automatic_policy_benchmark(
                self.db, sorter.CacheState(faces=[item.face]), matcher, hard_negatives={},
                evaluation_path=self.path, protected_path=self.path, progress=lambda _: None)
        self.assertFalse(allowed)
        self.assertFalse(report['strict_single_allowed'])
        self.assertEqual(report['incorrect'], 1)
        self.assertEqual(report['strict_single']['incorrect'], 1)

    def test_recovery_prototypes_without_provenance_fail_closed(self):
        validation = evaluation_dataset.DatasetValidation((), (), frozenset())
        with patch.object(evaluation_dataset, 'load_dataset', return_value=validation):
            allowed, report = review.evaluate_automatic_policy_benchmark(
                self.db, sorter.CacheState(), object(), hard_negatives={},
                review_prototypes={'Alice': [np.array([1., 0.])]})
        self.assertFalse(allowed)
        self.assertIn('provenance', report['errors'][0])

    def test_diagnostic_runs_independent_after_primary_failure_without_enabling(self):
        inputs = SimpleNamespace(fingerprint='inputs', validate=lambda: None)
        matcher = SimpleNamespace(db=secondary.SecondaryIdentityDB())
        with patch.object(review, 'automatic_review_gate_signature', return_value='policy'), \
             patch.object(review.evaluation_runtime, 'GateInputs', return_value=inputs), \
             patch.object(review.identity_evaluation, 'activation_gate', return_value=(False, {'failures': ['incorrect']})), \
             patch.object(review, 'evaluate_automatic_policy_benchmark', return_value=(True, {'strict_single_allowed': True})) as independent:
            allowed, report, _ = review.prepare_automatic_review_gate(
                self.db, sorter.CacheState(), matcher, requested=True,
                output_dir=self.root, progress=lambda _: None, diagnose_independent=True)
        independent.assert_called_once()
        self.assertFalse(allowed)
        self.assertFalse(report['strict_single_allowed'])

    def test_validation_only_cli_does_not_reconcile_or_file(self):
        args = SimpleNamespace(auto_sweep_worker=None, input=self.root, auto_safe=False,
                               auto_safe_preview=False, reprocess_pending=False, validate_auto_only=True,
                               decisions=self.root / 'decisions', output_dir=self.root)
        with patch.object(review, 'parse_args', return_value=args), \
             patch.object(sorter, 'load_identity_db', return_value=self.db), \
             patch.object(sorter, 'load_cache', return_value=sorter.CacheState()), \
             patch.object(review, 'prepare_secondary_verifier', return_value=(object(), 'ready')), \
             patch.object(review, 'build_trusted_review_prototypes', return_value=({}, {})), \
             patch.object(review, 'prepare_automatic_review_gate', return_value=(False, {}, 'blocked')) as gate, \
             patch.object(review, 'reconcile_unknown_queue') as reconcile, \
             patch.object(review, 'run_automatic_sweep') as sweep:
            self.assertEqual(review.main(), 1)
        self.assertTrue(gate.call_args.kwargs['diagnose_independent'])
        reconcile.assert_not_called()
        sweep.assert_not_called()

    def test_background_worker_revalidates_gate_before_touching_destinations(self):
        task = self.root / 'task.json'
        result = self.root / 'result.json'
        task.write_text(json.dumps({'result_path': str(result), 'output_dir': str(self.root)}))
        secondary_db = secondary.SecondaryIdentityDB(primary_signature='profile')
        with patch.object(sorter, 'load_identity_db', return_value=self.db), \
             patch.object(sorter, 'load_cache', return_value=sorter.CacheState()), \
             patch.object(secondary, 'load', return_value=secondary_db), \
             patch.object(secondary, 'primary_signature', return_value='profile'), \
             patch.object(secondary, 'trusted_snapshot_is_current', return_value=True), \
             patch.object(secondary, 'SecondaryMatcher', return_value=object()), \
             patch.object(review, 'build_trusted_review_prototypes', return_value=({}, {})), \
             patch.object(review, 'prepare_automatic_review_gate', return_value=(False, {}, 'gate failed')) as gate, \
             patch.object(review, '_run_automatic_sweep_batch') as batch, \
             patch.object(review.recover_no_usable_faces, 'load_existing_hashes') as destinations:
            self.assertEqual(review.run_automatic_sweep_worker(task), 1)
        self.assertIn('gate failed', json.loads(result.read_text())['error'])
        gate.assert_called_once()
        batch.assert_not_called()
        destinations.assert_not_called()

    def test_approved_merged_names_match_but_distinct_people_do_not(self):
        key = identity_evaluation.evaluation_identity_key
        for old, current in [('Yami Gautham', 'Yami Gautam'), ('Kruthi shetty', 'Kriti Shetty'),
                             ('Kiara', 'Kiara Advani'), ('Kriti', 'Kriti Kharbhandha'),
                             ('Rakul', 'Rakul Preeth Singh'), ('Priya Bhavani', 'Priya Bhavani Sankar')]:
            self.assertEqual(key(old), key(current))
        self.assertNotEqual(key('Preity'), key('Preethi'))
        self.assertNotEqual(key('Kriti Shetty'), key('Kriti Kharbhandha'))
        self.assertNotEqual(key('Kamal'), key('Kamal Haasan'))

    def test_correct_known_and_rejected_unknown_allow_strict_lane(self):
        item = self.item(verifier=secondary.SecondaryVerification(True, 'Alice', .1, .5))
        other = self.root / 'unknown.jpg'
        other.write_bytes(b'unknown fixture')
        unknown_face = SimpleNamespace(**(vars(item.face) | {'src_str': str(other)}))
        known = evaluation_dataset.EvaluationCase(self.path, 'Alice', frozenset({'known'}), True, 'safe', True)
        unknown = evaluation_dataset.EvaluationCase(other, '', frozenset({'unknown'}), True, 'safe', True)
        validations = [evaluation_dataset.DatasetValidation((known,), (), frozenset({'known'})),
                       evaluation_dataset.DatasetValidation((unknown,), (), frozenset({'unknown'}))]
        candidates = (identity_profiles.IdentityCandidate('Alice', .1, .9),)
        prepared = SimpleNamespace(rank=lambda face, **k: candidates if face is unknown_face else item.candidates,
                                   validate=lambda: None)
        matcher = SimpleNamespace(db=secondary.SecondaryIdentityDB(),
                                  verify=lambda *a, **k: item.secondary, flush=lambda: None)
        with patch.object(evaluation_dataset, 'load_dataset', side_effect=validations), \
             patch('evaluation_profiles.EvaluationProfiles', return_value=prepared), \
             patch.object(review, 'AUTO_GATE_MIN_CONFIRMED_CASES', 1):
            allowed, report = review.evaluate_automatic_policy_benchmark(
                self.db, sorter.CacheState(faces=[item.face, unknown_face]), matcher,
                hard_negatives={}, evaluation_path=self.path, protected_path=other, progress=lambda _: None)
        self.assertTrue(allowed)
        self.assertTrue(report['strict_single_allowed'])
        self.assertEqual(report['unknown_evaluated'], 1)


if __name__ == '__main__':
    unittest.main()

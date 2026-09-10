"""Reference integration tests use synthetic vectors and temporary files only."""

import copy
import hashlib
import os
import sqlite3
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np

import analysis_index
import face_reference_library as refs
import identity_evaluation
import evaluation_enrollment
import sort_photos as sorter


class FaceReferenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.people, self.refs = self.root / 'people', self.root / 'references'
        self.refs.mkdir()
        self.database = self.root / 'refs.sqlite3'
        self.db = sorter.IdentityDB(config_fingerprint=sorter.config_fingerprint())
        for name, vector in [('Alice', [1., 0.]), ('Bob', [0., 1.])]:
            folder = self.people / name
            folder.mkdir(parents=True)
            value = np.asarray(vector, dtype=np.float32)
            self.db.identities[name] = value
            self.db.prototypes[name] = [value]
            self.db.prototype_sources[name] = [str(folder / 'core.jpg')]
            self.db.source_counts[name] = 3
            self.db.source_signatures[name] = sorter.identity_source_signature(folder, [])
        sorter.normalize_identity_db(self.db)
        self.db.calibration_version = sorter.IDENTITY_CALIBRATION_VERSION
        self.patches = [patch.object(sorter, 'IDENTITY_HARD_NEGATIVES_FILE', self.root / 'negatives.json'),
                        patch.object(sorter, 'IDENTITY_CONFIRMATIONS_FILE', self.root / 'confirmations.json'),
                        patch.object(sorter, 'IDENTITY_DB_BUILD_FILE', self.root / 'building.pkl'),
                        patch.object(sorter, 'IDENTITY_DB_FILE', self.root / 'identity.pkl'),
                        patch.object(sorter.pipeline_paths, 'SOURCE_REVIEW', self.root / 'review')]
        for item in self.patches:
            item.start()
            self.addCleanup(item.stop)

    def file(self, name='one.jpg', person='Alice', data=b'new reference'):
        path = self.refs / person / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def store_detection(self, path, *, faces=1, quality=.9, embedding=(.98, .2), config=None):
        records = []
        for number in range(faces):
            face = sorter.CachedFace(str(path), number, .99, 100., 100., 0., quality,
                np.asarray(embedding, dtype=np.float32), np.array([]), b'face', label=None)
            records.append(sorter.cached_face_to_index_record(face))
        with analysis_index.AnalysisIndex(self.database) as index:
            index.replace_detections(path, config or sorter.config_fingerprint(),
                                     'accepted_face' if faces else 'no_face_detected', records)

    def detector(self, paths, _database, **kwargs):
        for path in paths:
            self.store_detection(path)

    def prepare(self, detector=None, db=None):
        return refs.prepare(self.refs, self.people, db or self.db, database=self.database,
                            detector=detector or self.detector, progress=lambda _: None)

    def test_cold_analyzes_once_and_warm_run_decodes_nothing(self):
        self.file()
        detector = Mock(side_effect=self.detector)
        first = self.prepare(detector)
        self.assertEqual(first.counts['analyzed'], 1)
        self.assertEqual(first.counts['selected'], 1)
        self.assertEqual(detector.call_count, 1)
        warm = self.prepare(Mock(side_effect=AssertionError('unneeded decode')))
        self.assertEqual(warm.counts['analyzed'], 0)
        self.assertEqual(warm.counts['reused'], 1)
        self.assertEqual(warm.signatures, first.signatures)

    def test_unchanged_selection_is_not_ranked_again(self):
        self.file()
        first = self.prepare()
        with patch.object(refs.identity_profiles.CompiledProfiles, 'rank', side_effect=AssertionError('repeated matching')):
            warm = self.prepare()
        self.assertEqual(warm.signatures, first.signatures)

    def test_selection_cache_invalidates_when_core_identity_changes(self):
        self.file()
        self.prepare()
        changed = copy.deepcopy(self.db)
        changed.identities['Alice'] = np.array([-1., 0.])
        changed.prototypes['Alice'] = [changed.identities['Alice']]
        self.assertFalse(self.prepare(db=changed).samples)

    def test_selection_cache_invalidates_when_matching_code_changes(self):
        self.file()
        first = self.prepare()
        replacement = self.root / 'matching-policy.py'
        replacement.write_text('updated matching policy')
        with patch.object(refs.identity_profiles, '__file__', str(replacement)), \
             patch.object(refs.identity_profiles.CompiledProfiles, 'rank', return_value=[]) as rank:
            changed = self.prepare(Mock(side_effect=AssertionError('unchanged image decoded')))
        self.assertTrue(first.samples)
        self.assertFalse(changed.samples)
        rank.assert_called_once()

    def test_duplicate_and_rename_reuse_content_without_redetection(self):
        first = self.file()
        second = self.file('copy.jpg')
        detector = Mock(side_effect=self.detector)
        snapshot = self.prepare(detector)
        self.assertEqual(len(detector.call_args.args[0]), 1)
        self.assertEqual(snapshot.counts['duplicate_content'], 1)
        renamed = first.with_name('renamed.jpg')
        first.rename(renamed)
        second.unlink()
        warm = self.prepare(Mock(side_effect=AssertionError('renamed bytes decoded')))
        self.assertEqual(warm.counts['analyzed'], 0)
        self.assertEqual(warm.samples['Alice'][0].source, str(renamed))

    def test_replacement_with_same_size_mtime_and_new_detector_invalidates_cache(self):
        source = self.file(data=b'before')
        self.prepare()
        stat = source.stat()
        source.write_bytes(b'after!')
        os.utime(source, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertEqual(self.prepare().counts['analyzed'], 1)
        with patch.object(sorter, 'config_fingerprint', return_value='new-model'):
            self.assertEqual(self.prepare().counts['analyzed'], 1)

    def test_bad_quality_group_and_no_face_never_become_prototypes(self):
        for name, faces, quality in [('group.jpg', 2, .9), ('none.jpg', 0, .9), ('low.jpg', 1, .2)]:
            self.store_detection(self.file(name, data=name.encode()), faces=faces, quality=quality)
        snapshot = self.prepare(Mock(side_effect=AssertionError('negative cache ignored')))
        self.assertFalse(snapshot.samples)
        self.assertEqual(snapshot.counts['multiple_faces'], 1)
        self.assertEqual(snapshot.counts['no_usable_face'], 1)
        self.assertEqual(snapshot.counts['low_quality'], 1)

    def test_wrong_label_and_cross_label_duplicate_are_held(self):
        wrong = self.file(person='Bob')
        snapshot = self.prepare()
        self.assertFalse(snapshot.samples)
        self.assertEqual(snapshot.rows[0]['status'], 'held_for_identity_verification')
        self.file(person='Alice', data=wrong.read_bytes())
        snapshot = self.prepare()
        self.assertEqual(snapshot.counts['conflicting_labels'], 2)
        self.assertFalse(snapshot.samples)

    def test_unknown_names_are_not_detected_or_added(self):
        self.file(person='Unverified Person')
        snapshot = self.prepare(Mock(side_effect=AssertionError('unmapped name decoded')))
        self.assertEqual(snapshot.counts['unmapped_name'], 1)
        self.assertFalse(snapshot.samples)

    def test_generated_and_symlink_files_are_not_scanned(self):
        self.file('review/no.jpg')
        self.file('_report/no.jpg')
        original = self.file('okay.jpg')
        self.file('_hidden.jpg')
        (original.parent / 'link.jpg').symlink_to(original)
        paths = list(refs.reference_files(self.refs, self.people, self.db.identities))
        self.assertEqual(paths, [(original, 'Alice')])

    def test_completed_alias_uses_existing_canonical_person(self):
        self.file(person='Old Alice')
        with patch.object(refs.person_aliases, 'canonical_folder', return_value='Alice'):
            snapshot = self.prepare()
        self.assertEqual(set(snapshot.samples), {'Alice'})

    def test_supplements_cannot_self_authorize_and_never_replace_core(self):
        self.file()
        snapshot = self.prepare()
        candidate = refs.augment(self.db, snapshot)
        self.assertEqual(candidate.source_counts, self.db.source_counts)
        np.testing.assert_array_equal(candidate.identities['Alice'], self.db.identities['Alice'])
        np.testing.assert_array_equal(candidate.prototypes['Alice'][0], self.db.prototypes['Alice'][0])
        self.assertEqual(self.prepare(db=candidate).signatures, snapshot.signatures)
        stripped = refs.core_profiles(candidate)
        self.assertEqual(len(stripped.prototypes['Alice']), 1)
        self.assertFalse(stripped.reference_sources)
        self.assertEqual(len(refs.augment(candidate, snapshot).prototypes['Alice']), 2)

    def test_removal_drops_supplement_but_preserves_core_and_originals(self):
        path = self.file()
        candidate = refs.augment(self.db, self.prepare())
        path.unlink()
        result = refs.augment(candidate, self.prepare(db=candidate))
        self.assertEqual(len(result.prototypes['Alice']), 1)
        self.assertEqual(result.source_counts, self.db.source_counts)

    def test_source_mutation_during_validation_prevents_promotion(self):
        path = self.file()
        snapshot = self.prepare()
        path.write_bytes(b'changed')
        with self.assertRaises(RuntimeError):
            snapshot.validate()

    def test_concurrent_profile_change_invalidates_candidate_snapshot(self):
        self.file()
        snapshot = self.prepare()
        sorter.IDENTITY_DB_FILE.write_bytes(b'new database from another process')
        with self.assertRaises(RuntimeError):
            snapshot.validate()

    def test_unchanged_snapshot_never_calls_profile_builder(self):
        self.file()
        snapshot = self.prepare()
        active = refs.augment(self.db, snapshot)
        with patch.object(sorter, 'build_identity_db_from_person_folders', side_effect=AssertionError('unneeded gate')):
            self.assertIs(refs.refresh(active, self.people, root=self.refs, database=self.database, progress=lambda _: None), active)

    def test_missing_reference_volume_keeps_active_database(self):
        with patch.object(sorter, 'build_identity_db_from_person_folders', side_effect=AssertionError('volume missing')):
            result = refs.refresh(self.db, self.people, root=self.root/'unmounted', database=self.database, progress=lambda _: None)
        self.assertIs(result, self.db)

    def test_locked_index_keeps_active_profiles_available(self):
        with patch.object(refs, 'prepare', side_effect=sqlite3.OperationalError('database locked')):
            self.assertIs(refs.refresh(self.db, self.people, root=self.refs, database=self.database,
                                      progress=lambda _: None), self.db)

    def test_failed_candidate_is_not_rebenchmarked_on_every_start(self):
        self.file()
        self.prepare()
        with patch.object(sorter, 'build_identity_db_from_person_folders', return_value=self.db) as build:
            refs.refresh(self.db, self.people, root=self.refs, database=self.database, progress=lambda _: None)
            refs.refresh(self.db, self.people, root=self.refs, database=self.database, progress=lambda _: None)
            self.assertEqual(build.call_count, 1)
            refs.refresh(self.db, self.people, root=self.refs, database=self.database, retry=True, progress=lambda _: None)
            self.assertEqual(build.call_count, 2)

    def test_incremental_index_never_reuses_a_different_content_or_model(self):
        source = self.file()
        self.store_detection(source)
        different = self.file('different.jpg', data=b'different')
        with analysis_index.AnalysisIndex(self.database) as index:
            self.assertIsNone(index.reusable_detections(source, 'different-model'))
            self.assertIsNone(index.reusable_detections(different, sorter.config_fingerprint()))

    def test_dashboard_counts_active_supplements_not_legacy_cache(self):
        import status_report
        self.file()
        active = refs.augment(self.db, self.prepare())
        with patch.object(sorter, 'load_identity_db', return_value=active):
            self.assertEqual(status_report.reference_summary(), (1, 1))
        with patch.object(sorter, 'load_identity_db', return_value=None):
            self.assertEqual(status_report.reference_summary(), (0, 0))

    def test_review_preview_and_validation_never_refresh_references(self):
        import review_unknown_identities as review
        flags = dict(validate_auto_only=False, reconcile_preview=False, reconcile_only=False,
                     auto_safe_preview=False, auto_safe=False, reprocess_pending=False)
        for flag in [None, 'validate_auto_only', 'reconcile_preview', 'reconcile_only', 'auto_safe_preview']:
            args = SimpleNamespace(input=self.refs, auto_sweep_worker=None,
                                   **(flags | ({flag: True} if flag else {})))
            with self.subTest(flag=flag), patch.object(review, 'parse_args', return_value=args), \
                 patch.object(sorter, 'load_identity_db', return_value=self.db), \
                 patch.object(refs, 'refresh', return_value=self.db) as refresh, \
                 patch.object(sorter, 'load_cache', side_effect=RuntimeError('stop before analysis')):
                with self.assertRaisesRegex(RuntimeError, 'stop before analysis'):
                    review.main()
                self.assertEqual(refresh.call_count, 0 if flag else 1)

    def test_supplements_are_bounded_and_hashes_remain_unchanged(self):
        for number in range(8):
            source = self.file(f'{number}.jpg', data=str(number).encode())
            self.store_detection(source, embedding=(.99, .05 + number * .02))
        snapshot = self.prepare()
        self.assertLessEqual(len(snapshot.samples['Alice']), refs.MAX_SUPPLEMENTS)
        for row in snapshot.rows:
            self.assertEqual(hashlib.sha256(Path(row['source']).read_bytes()).hexdigest(), row['sha256'])

    def test_reference_only_change_uses_gate_and_failed_gate_never_saves(self):
        self.file()
        snapshot = self.prepare()
        with patch.object(sorter, 'load_identity_db', return_value=self.db), \
             patch.object(sorter, 'load_cache', return_value=sorter.CacheState()), \
             patch.object(sorter, 'save_identity_db') as save, \
             patch.object(evaluation_enrollment, 'backfill_confirmations'), \
             patch.object(identity_evaluation, 'activation_gate', return_value=(False, {'failures': ['test failure']})) as gate:
            result = sorter.build_identity_db_from_person_folders(self.people, reference_snapshot=snapshot)
            self.assertIs(result, self.db)
            gate.assert_called_once()
            save.assert_not_called()
            gate.return_value = (True, {'failures': []})
            result = sorter.build_identity_db_from_person_folders(self.people, reference_snapshot=snapshot)
            save.assert_called_once_with(result)
            self.assertEqual(result.reference_signatures, snapshot.signatures)
            for name in self.db.identities:
                self.assertLessEqual(result.strict_thresholds[name], self.db.strict_thresholds[name])


if __name__ == '__main__':
    unittest.main()

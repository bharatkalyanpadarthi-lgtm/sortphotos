import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import identity_hard_negatives as negatives
import relink_cache_from_old_cache as relink


class ReferenceSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def face(self, path, data=None):
        return SimpleNamespace(src_str=str(path), content_sha256=(
            hashlib.sha256(data).hexdigest() if data is not None else ""))

    def test_same_size_and_timestamp_does_not_prove_face_content(self):
        first, second = self.root / 'first', self.root / 'second'
        first.write_bytes(b'aaaa')
        second.write_bytes(b'bbbb')
        os.utime(second, ns=(first.stat().st_atime_ns, first.stat().st_mtime_ns))
        self.assertEqual(relink.file_sig(first), relink.file_sig(second))
        self.assertEqual(relink.content_verified_faces(second, [self.face(first, b'aaaa')]), [])

    def test_renamed_identical_bytes_reuse_face(self):
        destination = self.root / 'renamed'
        destination.write_bytes(b'original')
        face = self.face(self.root / 'old', b'original')
        self.assertEqual(relink.content_verified_faces(destination, [face]), [face])

    def test_legacy_unhashed_face_requires_fresh_detection(self):
        path = self.root / 'photo'
        path.write_bytes(b'original')
        self.assertEqual(relink.content_verified_faces(path, [self.face(path)]), [])

    def test_mixed_provenance_drops_signature_to_allow_redetection(self):
        path = self.root / 'photo'
        path.write_bytes(b'original')
        good, bad = self.face(path, b'original'), self.face(path, b'incorrect')
        preserved = self.face(self.root / 'unchanged')
        cache = SimpleNamespace(faces=[good, bad, preserved], file_signatures={str(path): (1, 8)})
        relink.validate_relocated_faces(cache, {str(path)})
        self.assertNotIn(str(path), cache.file_signatures)
        self.assertEqual(cache.faces, [preserved])

    def test_rejection_is_person_specific_and_not_a_similarity_threshold(self):
        path = self.root / 'negatives.json'
        vector = np.array([1., 0., 0.], dtype=np.float32)
        negatives.record(path, person='Alice', source_path=self.root / 'photo', face_index=0, embedding=vector)
        evidence = negatives.ReferenceRejections(path)
        self.assertTrue(evidence.rejects('ALICE', vector * 2))
        self.assertFalse(evidence.rejects('Bob', vector))
        self.assertFalse(evidence.rejects('Alice', np.array([1., .05, 0.])))
        self.assertFalse(evidence.rejects('Alice', np.array([1., 0.])))

    def test_rejection_change_only_invalidates_affected_person(self):
        path = self.root / 'negatives.json'
        before = negatives.ReferenceRejections(path)
        negatives.record(path, person='Alice', source_path=self.root / 'photo', face_index=0,
                         embedding=np.array([1., 0.], dtype=np.float32))
        after = negatives.ReferenceRejections(path)
        self.assertNotEqual(before.signature('Alice'), after.signature('Alice'))
        self.assertEqual(before.signature('Bob'), after.signature('Bob'))

    def test_removed_incorrect_rejection_clears_training_exclusion(self):
        path = self.root / 'negatives.json'
        vector = np.array([1., 0.], dtype=np.float32)
        negatives.record(path, person='Alice', source_path=self.root / 'photo', face_index=0, embedding=vector)
        negatives.save(path, {'examples': []})
        evidence = negatives.ReferenceRejections(path)
        self.assertFalse(evidence.rejects('Alice', vector))
        self.assertEqual(evidence.signature('Alice'), '')


if __name__ == '__main__':
    unittest.main()

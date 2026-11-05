"""Data-only cross-process profile snapshots; no live library writes."""

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np

import recognition_policy
import secondary_identity_matcher as secondary
import shared_profile_snapshot as snapshot
import sort_photos


class SharedProfileSnapshotTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.path = Path(tmp.name) / 'snapshot.json'
        self.vector = np.array([1., 0.], dtype=np.float32)
        self.primary = sort_photos.IdentityDB(identities={'Alice': self.vector},
            prototypes={'Alice': [self.vector]}, prototype_sources={'Alice': ['source']})
        self.secondary = secondary.SecondaryIdentityDB(identities={'Alice': self.vector},
            prototypes={'Alice': [self.vector]}, prototype_sources={'Alice': ['source']},
            crop_embeddings={'private-cache': self.vector})
        self.recovery = recognition_policy.RecoveryProfiles()
        self.recovery['Alice'] = [self.vector]
        self.recovery.sources['Alice'] = ['source']

    def test_round_trip_is_deterministic_and_frozen(self):
        data = snapshot.encode(self.primary, self.secondary, self.recovery, {'Alice': [self.vector]})
        self.assertEqual(data, snapshot.encode(self.primary, self.secondary, self.recovery, {'Alice': [self.vector]}))
        digest = snapshot.write(self.path, data)
        primary, independent, recovery, negatives = snapshot.read(self.path, digest)
        self.vector[:] = 0
        self.assertEqual(float(primary.identities['Alice'][0]), 1.)
        self.assertEqual(float(independent.prototypes['Alice'][0][0]), 1.)
        self.assertEqual(recovery.sources['Alice'], ('source',))
        self.assertEqual(float(negatives['Alice'][0][0]), 1.)
        self.assertEqual(len(independent.crop_embeddings), 0)
        with self.assertRaises(ValueError):
            primary.identities['Alice'].setflags(write=True)
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_tampering_and_symlink_rejected(self):
        data = snapshot.encode(self.primary, self.secondary, self.recovery, {})
        digest = snapshot.write(self.path, data)
        self.path.write_bytes(data + b' ')
        with self.assertRaises(ValueError):
            snapshot.read(self.path, digest)
        link = self.path.with_name('link')
        link.symlink_to(self.path)
        with self.assertRaises(OSError):
            snapshot.read(link, hashlib.sha256(self.path.read_bytes()).hexdigest())


if __name__ == '__main__':
    unittest.main()

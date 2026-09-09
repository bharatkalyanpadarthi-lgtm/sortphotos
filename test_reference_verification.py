"""Isolated safety and bounded-work checks; never use the real photo library."""

import hashlib
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import content_identity
import identity_confirmations
from pipeline_progress import StageProgress
from verified_references import ReferenceIndex


class ReferenceFixtures:
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()

    def file(self, name, data=b"trusted", person="Alice"):
        path = self.root / person / "photos" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        return path

    def record(self, path, *, size=True, original=None, person="Alice"):
        result = dict(person=person, organized_path=str(original or path),
                      content_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        if size:
            result["byte_size"] = path.stat().st_size
        return result


class ReferenceIndexTests(ReferenceFixtures, unittest.TestCase):
    def test_many_renamed_legacy_records_hash_each_candidate_only_once(self):
        paths = [self.file(f"new-{n}.jpg", str(n).encode()) for n in range(180)]
        records = [self.record(path, original=path.with_name(f"old-{n}.jpg"), size=False)
                   for n, path in enumerate(paths)]
        index = ReferenceIndex(paths, progress=None)
        with patch.object(content_identity, "content_sha256", wraps=content_identity.content_sha256) as hashing:
            for record, path in zip(records, paths):
                self.assertEqual(index.resolve(record), path)
            self.assertEqual(hashing.call_count, len(paths))
            # Simulate eviction of the separate global hash LRU between stages.
            content_identity._hash_version.cache_clear()
            index.refresh()
            for record, path in zip(records, paths):
                self.assertEqual(index.resolve(record), path)
            self.assertEqual(hashing.call_count, len(paths))
        self.assertEqual(index.hashes_read, len(paths))

    def test_size_index_avoids_reading_nonmatching_files(self):
        wrong_size = self.file("large.jpg", b"x" * 200)
        target = self.file("renamed.jpg", b"okay")
        record = self.record(target, original=self.root / "gone.jpg")
        index = ReferenceIndex([wrong_size, target], progress=None)
        self.assertEqual(index.resolve(record), target)
        self.assertEqual(index.hashes_read, 1)

    def test_wrong_person_and_nonphoto_locations_are_not_fallbacks(self):
        target = self.file("image.jpg", person="Bob")
        record = self.record(target, original=self.root / "gone.jpg")
        review = self.root / "Alice" / "review" / "image.jpg"
        review.parent.mkdir(parents=True)
        review.write_bytes(target.read_bytes())
        index = ReferenceIndex([target, review], progress=None)
        self.assertIsNone(index.resolve(record))
        self.assertEqual(index.hashes_read, 0)

    def test_earlier_photos_directory_does_not_hide_person_bucket(self):
        self.root = self.root / "photos" / "library"
        path = self.file("new.jpg")
        record = self.record(path, original=self.root / "missing.jpg")
        self.assertEqual(ReferenceIndex([path], progress=None).resolve(record), path)

    def test_small_size_buckets_do_not_flood_terminal(self):
        messages = []
        paths = [self.file(f"{n}.jpg", b"x" * n) for n in range(1, 101)]
        index = ReferenceIndex(paths, progress=messages.append)
        for path in paths:
            self.assertEqual(index.resolve(self.record(path, original=self.root / "missing.jpg")), path)
        self.assertLess(len(messages), 10)

    def test_direct_reference_replacement_with_same_size_and_mtime_is_rejected(self):
        path = self.file("image.jpg", b"before")
        record = self.record(path)
        index = ReferenceIndex([path], progress=None)
        self.assertEqual(index.resolve(record), path)
        stat = path.stat()
        path.write_bytes(b"after!")
        os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        self.assertIsNone(index.resolve(record))

    def test_index_hit_is_reverified_after_replacement(self):
        path = self.file("new.jpg")
        record = self.record(path, original=self.root / "gone.jpg")
        index = ReferenceIndex([path], progress=None)
        self.assertEqual(index.resolve(record), path)
        replacement = self.file("replacement.jpg", b"changed")
        replacement.replace(path)
        self.assertIsNone(index.resolve(record))

    def test_refresh_recovers_a_file_moved_between_known_candidate_paths(self):
        path = self.file("old.jpg")
        new = path.parent / "nude" / "new.jpg"
        new.parent.mkdir()
        record = self.record(path)
        index = ReferenceIndex([path, new], progress=None)
        self.assertEqual(index.resolve(record), path)
        path.rename(new)
        index.refresh()
        self.assertEqual(index.resolve(record), new)

    def test_unreadable_candidate_can_recover_next_stage(self):
        path = self.file("new.jpg")
        record = self.record(path, original=self.root / "gone.jpg")
        index = ReferenceIndex([path], progress=None)
        with patch.object(content_identity, "content_sha256", side_effect=OSError("unavailable")):
            self.assertIsNone(index.resolve(record))
        index.refresh()
        self.assertEqual(index.resolve(record), path)

    def test_file_changing_during_verification_is_not_trusted(self):
        path = self.file("image.jpg", b"before")
        record = self.record(path)
        index = ReferenceIndex([path], progress=None)

        def changing(source):
            source.write_bytes(b"changed")
            return record["content_sha256"]

        with patch.object(content_identity, "content_sha256", side_effect=changing):
            self.assertIsNone(index.resolve(record))

    def test_invalid_digest_or_size_does_not_authorize_a_file(self):
        path = self.file("image.jpg")
        record = self.record(path)
        index = ReferenceIndex([path], progress=None)
        self.assertIsNone(index.resolve({**record, "content_sha256": "bad"}))
        self.assertIsNone(index.resolve({**record, "byte_size": "bad"}))

    def test_missing_results_are_not_persisted_across_runs(self):
        path = self.file("original.jpg")
        record = self.record(path)
        path.unlink()
        self.assertIsNone(ReferenceIndex([path], progress=None).resolve(record))
        self.file("original.jpg")
        self.assertEqual(ReferenceIndex([path], progress=None).resolve(record), path)

    def test_same_bytes_resolve_in_original_candidate_order(self):
        first, second = self.file("z.jpg"), self.file("a.jpg")
        record = self.record(first, original=self.root / "missing.jpg")
        self.assertEqual(ReferenceIndex([first, second], progress=None).resolve(record), first)

    def test_reused_hash_does_not_bypass_selected_face_provenance(self):
        path = self.file("image.jpg")
        first = SimpleNamespace(crop_jpeg=b"face-a", content_sha256="")
        second = SimpleNamespace(crop_jpeg=b"face-b", content_sha256="")
        record = {**self.record(path), "face_id": content_identity.face_identity(second)}
        registry = self.root / "confirmed.json"
        identity_confirmations.save(registry, {"examples": [record]})
        index = ReferenceIndex([path], progress=None)
        kwargs = dict(reference_index=index, progress=None)
        rows = list(identity_confirmations.verified_records(
            registry, {str(path): [first, second]}, {"alice": "Alice"}, **kwargs))
        self.assertEqual(rows, [("Alice", path, [second])])
        second.crop_jpeg = b"other-face"
        self.assertEqual(list(identity_confirmations.verified_records(
            registry, {str(path): [first, second]}, {"alice": "Alice"}, **kwargs)), [])


class ProgressTests(unittest.TestCase):
    def test_progress_is_throttled_and_finishes_with_actual_total(self):
        messages = []
        with patch("pipeline_progress.time.monotonic", return_value=0) as clock:
            status = StageProgress("References", 100, messages.append)
            clock.return_value = 1
            status.update(1)
            self.assertEqual(len(messages), 1)
            clock.return_value = 6
            status.update(20, "reused")
            status.update(100)
        self.assertIn("0/100", messages[0])
        self.assertIn("20/100", messages[1])
        self.assertIn("reused", messages[1])
        self.assertIn("100/100", messages[2])

    def test_skipped_records_still_advance_progress(self):
        messages = []
        for _item in StageProgress("Cases", 5, messages.append).items(range(5)):
            continue
        self.assertIn("5/5", messages[-1])

    def test_aborted_stage_does_not_claim_completion(self):
        messages = []
        with self.assertRaises(RuntimeError):
            for _item in StageProgress("Cases", 5, messages.append).items(range(5)):
                raise RuntimeError("interrupted")
        self.assertFalse(any("5/5" in message for message in messages))

    def test_empty_stage_reports_zero_work(self):
        messages = []
        self.assertEqual(list(StageProgress("Cases", 0, messages.append).items([])), [])
        self.assertIn("0/0", messages[-1])


class IntegrationTests(ReferenceFixtures, unittest.TestCase):
    def test_secondary_and_recovery_profiles_share_verified_content(self):
        import numpy as np
        import review_unknown_identities as review
        import secondary_identity_matcher as secondary
        import sort_photos as sorter

        paths = [self.file(f"new-{n}.jpg", str(n).encode()) for n in range(3)]
        vector = np.eye(4, dtype=np.float32)[0]
        faces = [sorter.CachedFace(
            src_str=str(path), face_index=0, quality=.9, det_score=.99, bbox_size=100,
            sharpness=100, yaw_proxy=0, embedding=vector, crop_jpeg=f"crop-{n}".encode(),
            label="Alice", image_phash=np.zeros(64, dtype=np.uint8),
        ) for n, path in enumerate(paths)]
        db = sorter.IdentityDB(identities={"Alice": vector}, prototypes={"Alice": [vector]},
                               prototype_sources={"Alice": [str(paths[0])]})
        cache = sorter.CacheState(faces=faces)
        registry = self.root / "confirmed.json"
        identity_confirmations.save(registry, {"examples": [
            self.record(path, original=path.with_name(f"old-{n}.jpg"), size=False)
            for n, path in enumerate(paths)]})
        index = ReferenceIndex(faces_by_source={str(path): [face] for path, face in zip(paths, faces)},
                               progress=None)
        messages = []
        with patch.object(secondary, "load", return_value=None), \
             patch.object(secondary, "build_app", return_value=object()), \
             patch.object(secondary, "embed_crop", return_value=vector), \
             patch.object(secondary, "save") as save:
            secondary.build_database(db, cache, confirmations_path=registry,
                                     reference_index=index, progress=messages.append)
            self.assertEqual(save.call_count, 1)
        self.assertEqual(index.hashes_read, len(paths))
        content_identity._hash_version.cache_clear()
        with patch.object(content_identity, "content_sha256", side_effect=AssertionError("rehash")), \
             patch.object(review, "_faces_by_source", side_effect=AssertionError("reindex")):
            profiles, counts = review.build_trusted_review_prototypes(
                db, cache, confirmations_path=registry,
                reference_index=index, progress=messages.append)
        self.assertIn("Alice", counts)
        self.assertGreater(len(profiles["Alice"]), 1)
        self.assertTrue(any("3/3" in value and "usable" in value for value in messages))
        self.assertTrue(any("Building secondary profiles: 1/1" in value for value in messages))

    def test_gate_progress_distinguishes_cached_and_new_evaluations(self):
        import review_unknown_identities as review
        matcher = SimpleNamespace(db=object())
        messages = []
        with patch.object(review, "automatic_review_gate_signature", return_value="unchanged"), \
             patch.object(review.identity_evaluation, "activation_gate", return_value=(True, {})) as primary, \
             patch.object(review, "evaluate_automatic_policy_benchmark", return_value=(True, {})) as policy, \
             patch.object(review.identity_hard_negatives, "vectors_by_person", return_value={}):
            for _ in range(2):
                allowed, _, _ = review.prepare_automatic_review_gate(
                    object(), object(), matcher, requested=True,
                    output_dir=self.root, progress=messages.append)
                self.assertTrue(allowed)
            self.assertEqual(primary.call_count, 1)
            self.assertEqual(policy.call_count, 1)
            self.assertIs(primary.call_args.kwargs["progress"].__self__, messages)
        self.assertTrue(any("inputs changed" in value for value in messages))
        self.assertTrue(any("reusing unchanged" in value for value in messages))


if __name__ == "__main__":
    unittest.main()

"""Trusted appearance coverage without relaxing recognition decisions."""

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import identity_profiles as profiles
import review_unknown_identities as review
import secondary_identity_matcher as secondary
import sort_photos as sorter


def vector(angle):
    return np.array([np.cos(angle), np.sin(angle), 0.], dtype=np.float32)


def sample(name, angle, quality=.9):
    return profiles.ReferenceSample(name, vector(angle), quality)


class SelectionTests(unittest.TestCase):
    def test_prioritizes_missing_appearance_over_repeated_core(self):
        values = [sample(f"front-{n}", .01 * n, .99) for n in range(8)]
        values.append(sample("confirmed-profile", 1., .8))
        selected = profiles.select_complementary_samples(
            values, existing_embeddings=[vector(0)], limit=2)
        self.assertEqual(selected[0].source, "confirmed-profile")
        self.assertEqual(len(selected), 2)

    def test_next_slot_covers_another_missing_appearance(self):
        values = [sample("left", 1.), sample("left-copy", 1.01), sample("right", -1.)]
        selected = profiles.select_complementary_samples(
            values, existing_embeddings=[vector(0)], limit=2)
        self.assertEqual({row.source for row in selected}, {"left", "right"})

    def test_quality_still_matters(self):
        values = [sample("poor-outlier", 1.5, .01), sample("good-profile", .9, .9)]
        selected = profiles.select_complementary_samples(
            values, existing_embeddings=[vector(0)], limit=1)
        self.assertEqual(selected[0].source, "good-profile")

    def test_zero_and_negative_limits_select_nothing(self):
        for limit in (0, -1):
            self.assertEqual(profiles.select_complementary_samples(
                [sample("trusted", .5)], existing_embeddings=[], limit=limit), [])

    def test_small_histories_keep_all_verified_sources(self):
        values = [sample("b", .1), sample("a", .2)]
        selected = profiles.select_complementary_samples(
            values, existing_embeddings=[vector(0)], limit=24)
        self.assertEqual([row.source for row in selected], ["a", "b"])

    def test_no_core_falls_back_to_existing_diversity_policy(self):
        values = [sample(str(n), n * .2) for n in range(8)]
        selected = profiles.select_complementary_samples(values, existing_embeddings=[], limit=3)
        expected = profiles.select_diverse_samples(values, limit=3)
        self.assertEqual([row.source for row in selected], [row.source for row in expected])

    def test_invalid_embeddings_and_quality_are_not_selected(self):
        values = [sample("valid", .5), profiles.ReferenceSample("nan", vector(1), np.nan),
                  profiles.ReferenceSample("zero", np.zeros(3), .9),
                  profiles.ReferenceSample("wrong-model", np.ones(4), .9),
                  profiles.ReferenceSample("bad-vector", np.array([np.inf, 0., 0.]), .9)]
        selected = profiles.select_complementary_samples(
            values, existing_embeddings=[vector(0)], limit=4)
        self.assertEqual([row.source for row in selected], ["valid"])

    def test_selection_is_deterministic_and_does_not_mutate_inputs(self):
        values = [sample(str(n), n * .2) for n in range(12)]
        before = [row.embedding.copy() for row in values]
        anchors = [vector(0)]
        selected = profiles.select_complementary_samples(values, existing_embeddings=anchors, limit=4)
        reverse = profiles.select_complementary_samples(list(reversed(values)), existing_embeddings=anchors, limit=4)
        self.assertEqual([row.source for row in selected], [row.source for row in reverse])
        for old, row in zip(before, values):
            np.testing.assert_array_equal(old, row.embedding)
        np.testing.assert_array_equal(anchors[0], vector(0))

    def test_large_history_normalizes_candidates_once_not_per_slot(self):
        values = [sample(str(n), n * .01) for n in range(1000)]
        with patch.object(profiles, "normalize_vector", wraps=profiles.normalize_vector) as normalizer:
            selected = profiles.select_complementary_samples(
                values, existing_embeddings=[vector(0)], limit=24)
        self.assertEqual(len(selected), 24)
        self.assertEqual(len({row.source for row in selected}), 24)
        self.assertEqual(normalizer.call_count, len(values) + 1)


class ProfileIntegrationTests(unittest.TestCase):
    def fixtures(self):
        faces = [SimpleNamespace(src_str=f"/fixture/front-{n}.jpg", face_index=0,
                                 quality=.99, embedding=vector(.01 * n),
                                 pose_label="frontal", crop_jpeg=f"crop-{n}".encode(), label="Alice")
                 for n in range(8)]
        faces.append(SimpleNamespace(src_str="/fixture/profile.jpg", face_index=0, quality=.8,
                                     embedding=vector(1), pose_label="left_profile",
                                     crop_jpeg=b"profile", label="Alice"))
        core = faces[0]
        db = sorter.IdentityDB(identities={"Alice": core.embedding},
                              prototypes={"Alice": [core.embedding]},
                              prototype_sources={"Alice": [core.src_str]})
        records = [("Alice", Path(face.src_str), [face]) for face in faces]
        return db, SimpleNamespace(faces=faces), records

    def test_primary_recovery_preserves_established_selection_and_provenance(self):
        db, cache, records = self.fixtures()
        with patch.object(review.identity_confirmations, "verified_records", return_value=records), \
             patch.object(review.identity_hard_negatives, "ReferenceRejections") as rejected:
            rejected.return_value.rejects.return_value = False
            result, counts = review.build_trusted_review_prototypes(
                db, cache, limit_per_person=2, progress=lambda _: None)
        self.assertIn("/fixture/profile.jpg", result.sources["Alice"])
        self.assertEqual(result.sources["Alice"][0], db.prototype_sources["Alice"][0])
        np.testing.assert_array_equal(result["Alice"][0], db.prototypes["Alice"][0])
        self.assertEqual(counts, {"Alice": 2})
        self.assertEqual(len(result.sources["Alice"]), len(result["Alice"]))
        self.assertEqual(result.sources["Alice"],
                         ["/fixture/front-0.jpg", "/fixture/front-0.jpg", "/fixture/profile.jpg"])

    def test_rejected_confirmation_cannot_become_a_recovery_reference(self):
        db, cache, records = self.fixtures()
        with patch.object(review.identity_confirmations, "verified_records", return_value=records), \
             patch.object(review.identity_hard_negatives, "ReferenceRejections") as rejected:
            rejected.return_value.rejects.return_value = True
            result, counts = review.build_trusted_review_prototypes(db, cache, progress=lambda _: None)
        self.assertEqual(counts, {})
        self.assertEqual(result.sources, db.prototype_sources)

    def test_secondary_refresh_reuses_crop_vectors_without_loading_model(self):
        db, cache, records = self.fixtures()
        existing = secondary.SecondaryIdentityDB(primary_signature="old-selection-policy",
            crop_embeddings={secondary.crop_key(face.crop_jpeg): face.embedding for face in cache.faces})
        with patch.object(secondary, "load", return_value=existing), \
             patch.object(secondary, "trusted_confirmation_signature", return_value="confirmed"), \
             patch.object(secondary, "trusted_confirmation_count", return_value=len(records)), \
             patch.object(secondary.identity_confirmations, "verified_records", return_value=records), \
             patch.object(secondary, "build_app", side_effect=AssertionError("unnecessary model load")), \
             patch.object(secondary, "embed_crop", side_effect=AssertionError("unnecessary inference")), \
             patch.object(secondary, "save"):
            result = secondary.build_database(db, cache, maximum_trusted_per_person=2,
                                              progress=lambda _: None)
        self.assertIn("/fixture/profile.jpg", result.prototype_sources["Alice"])
        self.assertEqual(result.primary_signature, secondary.primary_signature(db))

    def test_selection_change_invalidates_secondary_profile_signature(self):
        db, _, _ = self.fixtures()
        original = secondary.primary_signature(db)
        with patch.object(secondary, "PROFILE_SELECTION_VERSION", secondary.PROFILE_SELECTION_VERSION + 1):
            self.assertNotEqual(secondary.primary_signature(db), original)

    def test_database_rebuild_does_not_defer_small_confirmation_batch(self):
        db, cache, records = self.fixtures()
        existing = secondary.SecondaryIdentityDB(primary_signature=secondary.primary_signature(db),
            trusted_signature="before-confirmation", trusted_example_count=len(records) - 1,
            identities=db.identities, prototypes=db.prototypes, prototype_sources=db.prototype_sources,
            crop_embeddings={secondary.crop_key(face.crop_jpeg): face.embedding for face in cache.faces})
        with patch.object(secondary, "load", return_value=existing), \
             patch.object(secondary, "trusted_confirmation_signature", return_value="after-confirmation"), \
             patch.object(secondary, "trusted_confirmation_count", return_value=len(records)), \
             patch.object(secondary.identity_confirmations, "verified_records", return_value=records) as verified, \
             patch.object(secondary, "build_app", side_effect=AssertionError("unnecessary model load")), \
             patch.object(secondary, "save"):
            result = secondary.build_database(db, cache, progress=lambda _: None)
        verified.assert_called_once()
        self.assertEqual(result.trusted_signature, "after-confirmation")

    def test_secondary_build_loads_model_once_for_only_missing_crops(self):
        db, cache, records = self.fixtures()
        existing = secondary.SecondaryIdentityDB(primary_signature="old-selection-policy",
            crop_embeddings={secondary.crop_key(face.crop_jpeg): face.embedding for face in cache.faces[:-1]})
        with patch.object(secondary, "load", return_value=existing), \
             patch.object(secondary, "trusted_confirmation_signature", return_value="confirmed"), \
             patch.object(secondary, "trusted_confirmation_count", return_value=len(records)), \
             patch.object(secondary.identity_confirmations, "verified_records", return_value=records), \
             patch.object(secondary, "build_app", return_value=object()) as model, \
             patch.object(secondary, "embed_crop", return_value=vector(1)) as infer, \
             patch.object(secondary, "save"):
            secondary.build_database(db, cache, progress=lambda _: None)
        model.assert_called_once()
        infer.assert_called_once_with(b"profile", model.return_value)

    def test_new_review_session_learns_even_one_new_confirmation(self):
        db, cache, _ = self.fixtures()
        existing = secondary.SecondaryIdentityDB(
            primary_signature=secondary.primary_signature(db), identities=db.identities)
        refreshed = secondary.SecondaryIdentityDB(
            primary_signature=secondary.primary_signature(db), identities=db.identities)
        def is_current(value, _path, *, rebuild_interval=250):
            return value is refreshed or rebuild_interval > 1
        with patch.object(secondary, "load", return_value=existing), \
             patch.object(secondary, "trusted_snapshot_is_current", side_effect=is_current), \
             patch.object(secondary, "build_database", return_value=refreshed) as build:
            matcher, status = review.prepare_secondary_verifier(db, cache, requested=True,
                                                               progress=lambda _: None)
        self.assertEqual(status, "refreshed")
        self.assertIs(matcher.db, refreshed)
        build.assert_called_once()


if __name__ == "__main__":
    unittest.main()

"""Synthetic-only regressions for the prepared evaluation matcher."""

import gc
import os
import tempfile
import time
import unittest
import weakref
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import appearance_profiles
import evaluation_profiles as prepared
import identity_evaluation as scalar
import identity_hard_negatives
import identity_profiles
import sort_photos


class EvaluationProfilesTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.negative_path = self.root / "negatives.json"
        self.negative_path.write_text('{"examples": []}', encoding="utf-8")
        setting = patch.object(sort_photos, "IDENTITY_HARD_NEGATIVES_FILE", self.negative_path)
        setting.start()
        self.addCleanup(setting.stop)
        self.rng = np.random.default_rng(9831)
        self.db = sort_photos.IdentityDB()
        for index, name in enumerate(("Alice", "bob", "CAROL")):
            vectors = [self.vector(index, noise=0.018) * (1.4 + i / 3) for i in range(4)]
            self.db.identities[name] = vectors[0] + vectors[1] * 2
            self.db.prototypes[name] = vectors
            self.db.prototype_sources[name] = [str(self.file(f"{name}-{i}.bin")) for i in range(4)]
            self.db.source_counts[name] = 4
        self.face = self.make_face(self.file("query.bin"), self.vector(0, noise=0.015))

    def file(self, name, contents=None):
        path = self.root / name
        path.write_bytes(contents if contents is not None else name.encode("ascii"))
        return path

    def vector(self, index, *, noise=0.0):
        vector = self.rng.normal(0, noise, 512).astype(np.float32)
        vector[index] += 1
        return identity_profiles.normalize_vector(vector)

    @staticmethod
    def make_face(source, embedding, *, pose="unknown", quality=0.9):
        return SimpleNamespace(src_str=str(source), embedding=embedding, pose_label=pose,
                               crop_jpeg=b"", quality=quality, label="untouched")

    def write_negatives(self, rows):
        examples = []
        for person, origin, vector in rows:
            encoded, dimension, digest = identity_hard_negatives.encode_embedding(vector)
            examples.append(dict(person=person, source_path=str(origin) if origin else "",
                                 embedding_base64=encoded, embedding_dimension=dimension,
                                 embedding_sha256=digest))
        identity_hard_negatives.save(self.negative_path, {"examples": examples})

    def scalar_candidates(self, face, *, lane="strict", **kwargs):
        candidates = []
        ranker = identity_profiles.rank_candidates

        def record(*args, **settings):
            result = ranker(*args, **settings)
            candidates.extend(result)
            return result

        with patch.object(identity_profiles, "rank_candidates", side_effect=record):
            prediction = scalar.predict_face(face, self.db, lane=lane, **kwargs)
        return candidates, prediction

    def prediction(self, face, candidates, lane):
        if not candidates:
            return None
        best = candidates[0]
        strict = lane == "strict"
        maximum = (sort_photos.AUTO_PERSON_SINGLE_MATCH_DIST if strict
                   else sort_photos.AUTO_PERSON_MATCH_DIST)
        thresholds = self.db.strict_thresholds if strict else self.db.match_thresholds
        margin = identity_profiles.candidate_margin(candidates)
        required_margin = (sort_photos.AUTO_PERSON_SINGLE_MATCH_MARGIN if strict
                           else sort_photos.AUTO_PERSON_MATCH_MARGIN)
        accepted = (best.distance <= min(maximum, thresholds.get(best.name, maximum))
                    and margin >= required_margin and
                    (not strict or face.quality >= sort_photos.AUTO_PERSON_SINGLE_MIN_QUALITY))
        return scalar.FacePrediction(best.name if accepted else "", best.distance, margin, accepted)

    def assert_candidates(self, actual, expected, *, exact=True):
        self.assertEqual([c.name for c in actual], [c.name for c in expected])
        for left, right in zip(actual, expected):
            for key in ("similarity", "distance", "raw_distance", "hard_negative_distance"):
                a, b = getattr(left, key), getattr(right, key)
                if a is None or b is None or exact:
                    self.assertEqual(a, b, (left.name, key))
                else:
                    self.assertAlmostEqual(a, b, delta=1e-6, msg=f"{left.name}: {key}")

    def assert_parity(self, session, face=None, **kwargs):
        face = face or self.face
        actual = session.rank(face, **kwargs)
        for lane in ("strict", "consensus"):
            expected, prediction = self.scalar_candidates(face, lane=lane, **kwargs)
            self.assert_candidates(actual, expected)
            ours = self.prediction(face, actual, lane)
            if prediction is None:
                self.assertIsNone(ours)
            else:
                self.assertEqual((ours.predicted, ours.accepted),
                                 (prediction.predicted, prediction.accepted))
                self.assertAlmostEqual(ours.distance, prediction.distance, delta=1e-6)
                self.assertAlmostEqual(ours.margin, prediction.margin, delta=2e-6)
        return actual

    def assert_databases(self, actual, expected):
        self.assertEqual(list(actual.identities), list(expected.identities))
        for name in expected.identities:
            np.testing.assert_array_equal(actual.identities[name], expected.identities[name])
        for field in ("prototypes", "pose_prototypes", "appearance_prototypes"):
            left, right = getattr(actual, field), getattr(expected, field)
            self.assertEqual(left.keys(), right.keys())
            for name in right:
                groups = [(left[name], right[name])] if field == "prototypes" else []
                if field != "prototypes":
                    self.assertEqual(left[name].keys(), right[name].keys())
                    groups = [(left[name][label], right[name][label]) for label in right[name]]
                for values, reference in groups:
                    self.assertEqual(len(values), len(reference))
                    for a, b in zip(values, reference):
                        np.testing.assert_array_equal(a, b)

    def test_random_512d_scalar_predictions_and_distances_both_modes(self):
        session = prepared.EvaluationProfiles(self.db)
        for i in range(12):
            face = self.make_face(self.face.src_str, self.vector(i % 3, noise=0.007 + i / 200),
                                  quality=0.4 if i % 2 else 0.9)
            for holdout in (False, True):
                with self.subTest(i=i, holdout=holdout):
                    self.assert_parity(session, face, exclude_profile_source=holdout)
        session.validate()

    def test_source_counts_and_strict_provenance_rejection(self):
        self.db.source_counts["bob"] = 2
        self.db.prototype_sources["CAROL"] = []
        session = prepared.EvaluationProfiles(self.db)
        self.assertEqual([c.name for c in self.assert_parity(session)], ["Alice"])
        self.assertEqual({c.name for c in self.assert_parity(session, exclude_profile_source=False)},
                         {"Alice", "CAROL"})
        expected = scalar.heldout_identity_db(self.db, self.face.src_str)
        actual = session.heldout_db(self.face.src_str)
        self.assert_databases(actual, expected)
        self.assertIn("bob", actual.identities)
        self.assertNotIn("CAROL", actual.identities)

    def test_policy_minrefs_opt_out_keeps_holdout_and_pose_appearance_semantics(self):
        self.db.source_counts.clear()
        source = self.db.prototype_sources["Alice"][0]
        self.db.pose_prototypes["Alice"] = {"side": [self.vector(0), self.vector(1)]}
        self.db.pose_prototype_sources["Alice"] = {"side": [source, self.face.src_str]}
        self.db.appearance_prototypes["bob"] = {"low_light": [self.vector(1)]}
        self.db.appearance_prototype_sources["bob"] = {"low_light": [source]}
        session = prepared.EvaluationProfiles(self.db, negative_examples=[])
        face = self.make_face(source, self.face.embedding, pose="side")
        self.assertEqual(session.rank(face), [])
        held = scalar.heldout_identity_db(self.db, source)
        expected = identity_profiles.rank_candidates(
            face.embedding, held.identities, held.prototypes, pose_label="side",
            pose_prototypes=held.pose_prototypes, lighting_label="low_light",
            appearance_prototypes=held.appearance_prototypes)
        with patch.object(session, "_attributes", return_value=("low_light", 0)):
            actual = session.rank(face, minimum_references=False)
        self.assert_candidates(actual, expected)
        self.assert_databases(session.heldout_db(source), held)

    def test_explicit_negative_examples_and_disabled_evidence_do_not_load_file(self):
        origin = self.file("negative.bin")
        near = self.face.embedding + self.vector(8) * 0.1
        self.write_negatives([("Alice", origin, near)])
        examples = identity_hard_negatives.load(self.negative_path)["examples"]
        with patch.object(identity_hard_negatives, "load", side_effect=AssertionError("unexpected load")):
            disabled = prepared.EvaluationProfiles(self.db, negative_examples=[])
            injected = prepared.EvaluationProfiles(self.db, negative_examples=examples)
        self.assertNotIn(str(self.negative_path), disabled._paths)
        self.assertNotIn(str(self.negative_path), injected._paths)
        self.assertEqual(disabled.heldout_negatives(self.face.src_str, self.face.embedding), {})
        self.assertTrue(all(c.hard_negative_distance is None for c in disabled.rank(self.face)))
        self.assert_parity(injected)
        self.negative_path.unlink()
        disabled.validate()
        injected.validate()

    def test_implicit_centroid_prototype_and_empty_profiles(self):
        self.db.prototypes.pop("Alice")
        self.db.prototype_sources["Alice"] = [str(self.file("implicit.bin"))]
        self.db.prototypes["bob"] = []
        self.db.prototype_sources["bob"] = []
        session = prepared.EvaluationProfiles(self.db)
        self.assert_parity(session)
        self.assert_parity(session, exclude_profile_source=False)
        self.assert_databases(session.heldout_db(self.face.src_str),
                              scalar.heldout_identity_db(self.db, self.face.src_str))

    def test_copied_content_and_symlink_alias_hold_out_every_person(self):
        origin = Path(self.db.prototype_sources["Alice"][0])
        copied = self.file("copy.bin", origin.read_bytes())
        alias = self.root / "alias.bin"
        alias.symlink_to(origin)
        self.db.prototype_sources["bob"][1] = str(copied)
        session = prepared.EvaluationProfiles(self.db)
        for source in (origin, copied, alias):
            self.assert_parity(session, self.make_face(source, self.face.embedding))
            held = session.heldout_db(source)
            self.assertEqual(len(held.prototypes["Alice"]), 3)
            self.assertEqual(len(held.prototypes["bob"]), 3)
        extras = frozenset(self.db.prototype_sources["CAROL"])
        self.assert_parity(session, excluded_sources=extras)
        self.assertNotIn("CAROL", session.heldout_db(self.face.src_str, extras).identities)
        session.validate()

    def test_canonical_holdout_works_even_when_source_is_missing(self):
        missing = self.root / "missing.bin"
        self.db.prototype_sources["Alice"][0] = str(missing)
        self.db.prototype_sources["bob"][0] = str(self.root / "also-missing.bin")
        session = prepared.EvaluationProfiles(self.db)
        face = self.make_face(missing, self.face.embedding)
        self.assert_parity(session, face)
        held = session.heldout_db(missing)
        self.assertEqual(len(held.prototypes["Alice"]), 3)
        self.assertEqual(len(held.prototypes["bob"]), 4)
        session.validate()

    def test_pose_lighting_era_and_mismatched_group_provenance(self):
        keep = str(self.file("specific-keep.bin"))
        query = self.face.src_str
        for name in self.db.identities:
            self.db.pose_prototypes[name] = {"side": [self.face.embedding, self.vector(0, noise=0.05)],
                                             "bad": [self.face.embedding]}
            self.db.pose_prototype_sources[name] = {"side": [query, keep], "bad": []}
            self.db.appearance_prototypes[name] = {
                "low_light": [self.face.embedding, self.vector(0, noise=0.04)],
                "normal_light": [self.face.embedding], "era_older": [self.face.embedding],
                "era_newer": [self.vector(0, noise=0.06)], "bad": [self.face.embedding]}
            self.db.appearance_prototype_sources[name] = {
                "low_light": [query, keep], "normal_light": [query],
                "era_older": [query], "era_newer": [keep], "bad": []}
            self.db.appearance_era_cutoffs[name] = 100
        session = prepared.EvaluationProfiles(self.db)
        for pose in ("side", "bad", "unknown"):
            for light, timestamp in (("low_light", 50), ("normal_light", 100), ("unknown", 0)):
                with self.subTest(pose=pose, light=light, timestamp=timestamp), \
                     patch.object(appearance_profiles, "query_attributes", return_value=(light, timestamp)), \
                     patch.object(session, "_attributes", return_value=(light, timestamp)):
                    face = self.make_face(query, self.face.embedding, pose=pose)
                    self.assert_parity(session, face)
                    self.assert_parity(session, face, exclude_profile_source=False)
        self.assert_databases(session.heldout_db(query), scalar.heldout_identity_db(self.db, query))
        held = session.heldout_db(query)
        self.assertEqual(held.appearance_prototypes["Alice"]["normal_light"], [])
        self.assertNotIn("bad", held.pose_prototypes["Alice"])

    def test_negatives_source_copy_query_equal_and_no_holdout_name_semantics(self):
        origin = self.file("negative.bin")
        copied = self.file("negative-copy.bin", origin.read_bytes())
        other = self.file("negative-other.bin")
        near = identity_profiles.normalize_vector(self.face.embedding + self.vector(5) * 0.13)
        almost_equal = self.face.embedding.copy()
        almost_equal[9] += 1e-7
        self.write_negatives([
            ("Alice", origin, near), ("Alice", other, self.face.embedding),
            ("Alice", other, almost_equal), ("bob", self.face.src_str, near),
            ("CAROL", None, near), (" Alice ", other, near), ("", other, near),
            ("not in DB", other, near)])
        session = prepared.EvaluationProfiles(self.db)
        actual = self.assert_parity(session)
        self.assertIsNotNone(next(c for c in actual if c.name == "Alice").hard_negative_distance)
        self.assert_parity(session, exclude_profile_source=False)
        excluded = frozenset({str(copied)})
        self.assert_parity(session, excluded_sources=excluded)
        expected = scalar.heldout_hard_negatives(self.face.src_str, self.face.embedding, excluded)
        ours = session.heldout_negatives(self.face.src_str, self.face.embedding, excluded)
        self.assertEqual(ours.keys(), expected.keys())
        for name in expected:
            np.testing.assert_array_equal(ours[name], expected[name])
        self.assertNotIn("Alice", ours)
        self.assertIn(" Alice ", ours)
        self.assertIn("", ours)

    def test_query_equal_negatives_do_not_leak_across_cache_keys(self):
        vector = identity_profiles.normalize_vector(self.face.embedding)
        other_face = self.make_face(self.face.src_str, vector + self.vector(8) * 0.05)
        self.write_negatives([("Alice", self.file("negative.bin"), vector)])
        session = prepared.EvaluationProfiles(self.db)
        first = self.assert_parity(session)
        second = self.assert_parity(session, other_face)
        third = self.assert_parity(session)
        self.assertIsNone(next(c for c in first if c.name == "Alice").hard_negative_distance)
        self.assertIsNotNone(next(c for c in second if c.name == "Alice").hard_negative_distance)
        self.assert_candidates(first, third, exact=True)

    def test_mutation_deletion_replacement_and_missing_then_created(self):
        for change in ("write", "delete", "replace", "create"):
            with self.subTest(change=change):
                source = self.file(f"mutate-{change}.bin", b"original")
                if change == "create":
                    source.unlink()
                session = prepared.EvaluationProfiles(self.db)
                session.rank(self.make_face(source, self.face.embedding))
                session.validate()
                if change == "delete":
                    source.unlink()
                elif change == "replace":
                    self.file("replacement.bin", b"original").replace(source)
                else:
                    source.write_bytes(b"modified")
                with self.assertRaisesRegex(prepared.EvaluationInputsChanged, str(source)):
                    session.validate()

    def test_same_size_same_mtime_write_still_fails_and_new_session_is_fresh(self):
        source = Path(self.db.prototype_sources["Alice"][0])
        before = source.stat()
        copied = self.file("old-copy.bin", source.read_bytes())
        session = prepared.EvaluationProfiles(self.db)
        source.write_bytes(b"x" * before.st_size)
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        with self.assertRaises(prepared.EvaluationInputsChanged):
            session.validate()
        fresh = prepared.EvaluationProfiles(self.db)
        held = fresh.heldout_db(copied)
        self.assertEqual(len(held.prototypes["Alice"]), 4)
        fresh.validate()

    def test_negative_evidence_file_is_captured_including_absence(self):
        for missing in (False, True):
            with self.subTest(missing=missing):
                if missing:
                    self.negative_path.unlink()
                session = prepared.EvaluationProfiles(self.db)
                session.validate()
                self.write_negatives([("Alice", self.file("negative.bin"), self.face.embedding)])
                with self.assertRaises(prepared.EvaluationInputsChanged):
                    session.validate()

    def test_reference_exclusion_and_negative_origins_are_validated(self):
        negative = self.file("negative.bin")
        excluded = self.file("excluded.bin")
        self.write_negatives([("Alice", negative, self.vector(0))])
        for source in (Path(self.db.prototype_sources["Alice"][0]), negative, excluded):
            with self.subTest(source=source):
                session = prepared.EvaluationProfiles(self.db)
                session.heldout_negatives(self.face.src_str, self.face.embedding, frozenset({str(excluded)}))
                source.write_bytes(source.read_bytes() + b"!")
                with self.assertRaises(prepared.EvaluationInputsChanged):
                    session.validate()

    def test_symlink_retarget_and_intermediate_same_destination_retarget(self):
        original = Path(self.db.prototype_sources["Alice"][0])
        copy = self.file("same-content.bin", original.read_bytes())
        inner = self.root / "inner.bin"
        outer = self.root / "outer.bin"
        alias = self.root / "same-destination.bin"
        inner.symlink_to(original)
        outer.symlink_to(inner)
        alias.symlink_to(original)
        for target in (copy, alias):
            with self.subTest(target=target):
                inner.unlink()
                inner.symlink_to(original)
                session = prepared.EvaluationProfiles(self.db)
                session.rank(self.make_face(outer, self.face.embedding))
                session.validate()
                inner.unlink()
                inner.symlink_to(target)
                with self.assertRaises(prepared.EvaluationInputsChanged):
                    session.validate()

    def test_input_changing_during_initial_hash_is_rejected(self):
        source = Path(self.db.prototype_sources["Alice"][0])
        original = prepared._sha256

        def mutate(path):
            digest = original(path)
            if path == str(source):
                source.write_bytes(b"changed during capture")
            return digest

        with patch.object(prepared, "_sha256", side_effect=mutate):
            with self.assertRaisesRegex(prepared.EvaluationInputsChanged, "during capture"):
                prepared.EvaluationProfiles(self.db)

    def test_timestamp_is_run_scoped_and_bypasses_global_file_cache(self):
        session = prepared.EvaluationProfiles(self.db)
        timestamp_reader = appearance_profiles._capture_timestamp_cached
        with patch.object(timestamp_reader, "__wrapped__", return_value=1234.0) as read_timestamp:
            session.rank(self.face)
            session.rank(self.face)
            self.assertEqual(session._timestamps[self.face.src_str], 1234.0)
            read_timestamp.assert_called_once()

    def test_failed_read_becoming_readable_is_detected_without_rehashing(self):
        source = Path(self.db.prototype_sources["Alice"][0])
        original = prepared._sha256

        def fail_once(path):
            if path == str(source):
                raise PermissionError(13, "synthetic unreadable path")
            return original(path)

        with patch.object(prepared, "_sha256", side_effect=fail_once):
            session = prepared.EvaluationProfiles(self.db)
        with patch.object(prepared, "_sha256", side_effect=AssertionError("must not rehash")):
            with self.assertRaisesRegex(prepared.EvaluationInputsChanged, "readability"):
                session.validate()

    def test_hash_once_per_canonical_path_and_no_io_on_repeated_query(self):
        alias = self.root / "alias.bin"
        alias.symlink_to(self.face.src_str)
        original = prepared._sha256
        with patch.object(prepared, "_sha256", wraps=original) as hashing:
            session = prepared.EvaluationProfiles(self.db)
            session.rank(self.face)
            session.rank(self.make_face(alias, self.face.embedding))
            with patch.object(os.path, "realpath", side_effect=AssertionError("repeated realpath")):
                for _ in range(4):
                    session.rank(self.face)
            session.validate()
            paths = [call.args[0] for call in hashing.call_args_list]
            self.assertEqual(len(paths), len(set(paths)))
            self.assertEqual(paths.count(self.face.src_str), 1)

    def test_normalization_and_compilation_reused_for_irrelevant_sources(self):
        source = self.db.prototype_sources["Alice"][0]
        self.db.pose_prototypes["Alice"] = {"side": [self.vector(0, noise=0.05)]}
        self.db.pose_prototype_sources["Alice"] = {"side": [source]}
        self.face.pose_label = "side"
        session = prepared.EvaluationProfiles(self.db)
        session.rank(self.face)
        original = identity_profiles.normalize_vector
        with patch.object(identity_profiles, "normalize_matrix", side_effect=AssertionError("matrix renormalized")), \
             patch.object(identity_profiles, "CompiledProfiles", side_effect=AssertionError("recompiled")), \
             patch.object(identity_profiles, "normalize_vector", wraps=original) as normalize, \
             patch.object(session, "_scalar_rank", wraps=session._scalar_rank) as fallback:
            for i in range(10):
                source = self.file(f"irrelevant-{i}.bin")
                session.rank(self.make_face(source, self.face.embedding, pose="side"))
            self.assertEqual(normalize.call_count, 20)
            fallback.assert_not_called()
        self.assertEqual(len(session._cache), 0)
        self.assertEqual(len(session._matrix_cache), 1)

    def test_combined_matrix_cache_bounded_across_holdouts(self):
        self.db.pose_prototypes["Alice"] = {"side": [self.vector(0)]}
        self.db.pose_prototype_sources["Alice"] = {"side": [self.face.src_str]}
        face = self.make_face(self.file("pose-query.bin"), self.face.embedding, pose="side")
        session = prepared.EvaluationProfiles(self.db, max_cached_profiles=2)
        for source in self.db.prototype_sources["Alice"]:
            self.assert_parity(session, face, excluded_sources=frozenset({source}))
            self.assertLessEqual(len(session._cache), 2)
            self.assertLessEqual(len(session._matrix_cache), 2)

    def test_affected_variants_lru_bounded_and_unaffected_matrices_shared(self):
        session = prepared.EvaluationProfiles(self.db, max_cached_profiles=2)
        sources = self.db.prototype_sources["Alice"]
        session.rank(self.face, excluded_sources=frozenset({sources[0]}))
        first_key = next(iter(session._cache))
        first = session._cache[first_key]
        reference = weakref.ref(first)
        self.assertIs(first.negatives.matrix, session._held["Alice"].negatives.matrix)
        self.assertIsNot(first.base.matrix, session._held["Alice"].base.matrix)
        del first
        for source in sources[1:]:
            self.assert_parity(session, excluded_sources=frozenset({source}))
            self.assertLessEqual(len(session._cache), 2)
        gc.collect()
        self.assertIsNone(reference())
        self.assertNotIn(first_key, session._cache)
        self.assertEqual({key[0] for key in session._cache}, {"Alice"})
        no_cache = prepared.EvaluationProfiles(self.db, max_cached_profiles=0)
        self.assert_parity(no_cache, excluded_sources=frozenset({sources[0]}))
        self.assertFalse(no_cache._cache)
        with self.assertRaises(ValueError):
            prepared.EvaluationProfiles(self.db, max_cached_profiles=-1)

    def test_same_effective_exclusions_share_cache_entry_and_lru_refreshes(self):
        source = Path(self.db.prototype_sources["Alice"][0])
        copy = self.file("copy.bin", source.read_bytes())
        other, third = self.db.prototype_sources["Alice"][1:3]
        session = prepared.EvaluationProfiles(self.db, max_cached_profiles=2)
        session.rank(self.face, excluded_sources=frozenset({str(source)}))
        first_key = next(iter(session._cache))
        first = session._cache[first_key]
        session.rank(self.face, excluded_sources=frozenset({other}))
        session.rank(self.face, excluded_sources=frozenset({str(copy)}))
        self.assertIs(session._cache[first_key], first)
        session.rank(self.face, excluded_sources=frozenset({third}))
        self.assertIn(first_key, session._cache)
        self.assertEqual(len(session._cache), 2)

    def test_distance_boundary_uses_exact_scalar_fallback(self):
        candidates, _ = self.scalar_candidates(self.face)
        self.db.strict_thresholds[candidates[0].name] = candidates[0].distance
        self.db.match_thresholds[candidates[0].name] = candidates[0].distance
        session = prepared.EvaluationProfiles(self.db)
        with patch.object(session, "_scalar_rank", wraps=session._scalar_rank) as fallback:
            actual = self.assert_parity(session)
            fallback.assert_called_once()
        self.assert_candidates(actual, candidates, exact=True)

    def test_margin_and_tie_boundaries_use_exact_scalar_fallback(self):
        candidates, _ = self.scalar_candidates(self.face)
        margin = identity_profiles.candidate_margin(candidates)
        with patch.object(sort_photos, "AUTO_PERSON_SINGLE_MATCH_MARGIN", margin):
            session = prepared.EvaluationProfiles(self.db)
            with patch.object(session, "_scalar_rank", wraps=session._scalar_rank) as fallback:
                actual = self.assert_parity(session)
                fallback.assert_called_once()
            self.assert_candidates(actual, candidates, exact=True)
        self.db.identities["bob"] = self.db.identities["Alice"]
        self.db.prototypes["bob"] = self.db.prototypes["Alice"]
        session = prepared.EvaluationProfiles(self.db)
        with patch.object(session, "_scalar_rank", wraps=session._scalar_rank) as fallback:
            self.assert_parity(session)
            fallback.assert_called_once()

    def test_irrelevant_tail_ties_do_not_trigger_full_scalar_fallback(self):
        session = prepared.EvaluationProfiles(self.db)
        candidates = [identity_profiles.IdentityCandidate("Alice", 0.9, 0.1)]
        candidates.extend(identity_profiles.IdentityCandidate("Alice", 1 - distance, distance)
                          for distance in (0.7, 0.8, 0.9, 0.95, 0.95, 0.95))
        self.assertFalse(session._near_boundary(candidates))

    def test_float64_inputs_and_tiny_norms_keep_scalar_arithmetic(self):
        for name in self.db.identities:
            self.db.identities[name] = self.db.identities[name].astype(np.float64)
            self.db.prototypes[name] = [v.astype(np.float64) for v in self.db.prototypes[name]]
        self.db.prototypes["Alice"][0] *= 1e-12
        session = prepared.EvaluationProfiles(self.db)
        self.assert_parity(session)
        self.assert_parity(session, exclude_profile_source=False)
        self.assert_databases(session.heldout_db(self.face.src_str),
                              scalar.heldout_identity_db(self.db, self.face.src_str))

    def test_inputs_labels_and_original_arrays_are_not_modified(self):
        content = {path: path.read_bytes() for path in self.root.iterdir()}
        centroid = self.db.identities["Alice"].copy()
        prototype = self.db.prototypes["Alice"][0].copy()
        session = prepared.EvaluationProfiles(self.db)
        session.rank(self.face)
        held = session.heldout_db(self.face.src_str)
        session.heldout_negatives(self.face.src_str, self.face.embedding)
        session.validate()
        self.assertEqual(self.face.label, "untouched")
        np.testing.assert_array_equal(self.db.identities["Alice"], centroid)
        np.testing.assert_array_equal(self.db.prototypes["Alice"][0], prototype)
        self.assertTrue(self.db.prototypes["Alice"][0].flags.writeable)
        self.assertFalse(held.prototypes["Alice"][0].flags.writeable)
        self.assertEqual(content, {path: path.read_bytes() for path in self.root.iterdir()})

    def test_empty_db_and_all_primary_references_held_out(self):
        self.db = sort_photos.IdentityDB()
        session = prepared.EvaluationProfiles(self.db)
        self.assert_parity(session)
        self.assert_databases(session.heldout_db(self.face.src_str),
                              scalar.heldout_identity_db(self.db, self.face.src_str))
        session.validate()


def synthetic_timing():
    """Opt-in timing fixture, never run by the unit-test suite; no photo inputs."""
    fixture = EvaluationProfilesTests()
    fixture.setUp()
    try:
        fixture.db = sort_photos.IdentityDB()
        for index in range(100):
            name = f"Person {index:03d}"
            values = [fixture.vector(index, noise=0.018) * (1 + i / 8) for i in range(8)]
            fixture.db.identities[name] = values[0] + values[1]
            fixture.db.prototypes[name] = values
            fixture.db.prototype_sources[name] = [str(fixture.file(f"person-{index}-{i}.bin"))
                                                  for i in range(8)]
            fixture.db.source_counts[name] = 8
        faces = [fixture.make_face(fixture.db.prototype_sources[f"Person {i:03d}"][0],
                    fixture.vector(i, noise=(0.008, 0.018, 0.04, 0.10)[i % 4]),
                    quality=0.4 if i % 5 == 0 else 0.9) for i in range(50)]
        start = time.perf_counter()
        expected = [scalar.predict_face(face, fixture.db, lane="strict") for face in faces]
        scalar_seconds = time.perf_counter() - start
        start = time.perf_counter()
        session = prepared.EvaluationProfiles(fixture.db)
        preparation_seconds = time.perf_counter() - start
        with patch.object(session, "_scalar_rank", wraps=session._scalar_rank) as fallback:
            start = time.perf_counter()
            actual = [fixture.prediction(face, session.rank(face), "strict") for face in faces]
            rank_seconds = time.perf_counter() - start
            fallback_count = fallback.call_count
        start = time.perf_counter()
        session.validate()
        validation_seconds = time.perf_counter() - start
        for left, right in zip(actual, expected):
            fixture.assertEqual(left, right)
        return dict(people=100, prototypes_per_person=8, queries=50, dimensions=512,
                    scalar_seconds=scalar_seconds, preparation_seconds=preparation_seconds,
                    rank_seconds=rank_seconds, validation_seconds=validation_seconds,
                    prepared_total_seconds=preparation_seconds + rank_seconds + validation_seconds,
                    exact_predictions=True, accepted=sum(p.accepted for p in actual),
                    rejected=sum(not p.accepted for p in actual), scalar_fallbacks=fallback_count)
    finally:
        fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()

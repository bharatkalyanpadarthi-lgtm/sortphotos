import unittest
import json
import tempfile
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import recover_labeled_sources_from_cache as recovery
import person_aliases


class PersonAliasRulesTests(unittest.TestCase):
    def test_preity_aliases_resolve_to_existing_canonical_folder(self):
        aliases, removed = recovery.load_rules()
        people = {"preity": "Preity", "preethi": "Preethi"}
        for name in ("Preity", "Priety", "Preethi", "PREETHI"):
            with self.subTest(name=name):
                self.assertEqual("Preity", recovery.canonical_label(
                    name, aliases, removed, people, create_missing_people=False))

    def test_recovery_does_not_recreate_the_merged_alias(self):
        aliases, removed = recovery.load_rules()
        self.assertIsNone(recovery.canonical_label(
            "Preethi", aliases, removed, {}, create_missing_people=False))
        self.assertEqual("Preity", recovery.canonical_label(
            "Preethi", aliases, removed, {}, create_missing_people=True))


class CompletedMergeRoutingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.people = self.root / "people"
        (self.people / "Preity" / "photos").mkdir(parents=True)
        self.rules = self.root / "rules.json"
        self.rules.write_text(json.dumps({"merge": {"Preity": ["Preity", "Preethi"]}}))

    def resolve(self, name):
        return person_aliases.canonical_folder(name, self.people, rules_path=self.rules)

    def test_completed_merge_routes_old_name_only(self):
        self.assertEqual("Preity", self.resolve("Preethi"))
        self.assertEqual("Preity", self.resolve("PREETHI"))
        self.assertEqual("Unrelated", self.resolve("Unrelated"))

    def test_two_existing_folders_are_not_implicitly_merged(self):
        (self.people / "Preethi").mkdir()
        self.assertEqual("Preethi", self.resolve("Preethi"))

    def test_missing_target_does_not_create_a_merge(self):
        self.assertEqual("Preethi", person_aliases.canonical_folder(
            "Preethi", self.root / "missing", rules_path=self.rules))

    def test_rule_updates_invalidate_cached_aliases(self):
        self.assertEqual("Preity", self.resolve("Preethi"))
        self.rules.write_text('{"merge": {}}')
        self.assertEqual("Preethi", self.resolve("Preethi"))

    def test_unsafe_rules_fail_closed(self):
        self.rules.write_text(json.dumps({"merge": {"../outside": ["Preethi"]}}))
        with self.assertRaises(ValueError):
            self.resolve("Preethi")

    def test_manual_and_quick_review_use_canonical_destination(self):
        import confirm_unknown_identity
        import review_unknown_identities
        db = SimpleNamespace(identities={"Preity": object(), "Preethi": object()})
        self.assertEqual("Preity", confirm_unknown_identity.canonical_person(
            "Preethi", db, people_root=self.people))
        self.assertEqual("Preity", review_unknown_identities.canonical_person(
            "Preethi", db, people_root=self.people))

    def test_daily_filing_preserves_bytes_and_does_not_recreate_alias(self):
        import sort_photos as sorter
        source = self.root / "incoming.jpg"
        source.write_bytes(b"unchanged original image bytes")
        record = SimpleNamespace(cluster_id=1, src=source, quality=1.0,
                                 sharpness=100000, image_phash=None)
        names = {1: "Preethi"}
        with (
            patch.object(sorter, "DEDUP_DUPLICATES", False),
            patch.object(sorter, "analysis_index_file", return_value=self.root / "analysis.sqlite3"),
            patch.object(sorter, "classified_nudity_status", return_value="safe"),
            patch.object(sorter, "maybe_move_to_nudity_subfolder",
                         side_effect=lambda p, *_a, **_kw: (p, None)),
        ):
            organized = sorter.organize_originals([record], names, self.people)
        self.assertEqual({source}, organized)
        self.assertEqual("Preity", names[1])
        self.assertFalse((self.people / "Preethi").exists())
        destination, = (self.people / "Preity" / "photos").glob("*.jpg")
        self.assertEqual(source.read_bytes(), destination.read_bytes())

    def test_recovery_preview_uses_canonical_folder(self):
        import recover_no_usable_faces as missed
        source = self.root / "incoming.jpg"
        source.write_bytes(b"original")
        with patch.object(missed, "DEFAULT_PEOPLE", self.people):
            path, duplicate = missed.copy_to_person(source, "Preethi", "digest",
                defaultdict(set), {}, dry_run=True)
        self.assertFalse(duplicate)
        self.assertTrue(path.is_relative_to(self.people / "Preity" / "photos"))
        self.assertFalse((self.people / "Preethi").exists())


if __name__ == "__main__":
    unittest.main()

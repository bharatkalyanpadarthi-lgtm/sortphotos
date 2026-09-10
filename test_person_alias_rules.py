import unittest

import recover_labeled_sources_from_cache as recovery


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


if __name__ == "__main__":
    unittest.main()

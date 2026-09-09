"""Menu/dispatch regressions with no production photo or cache access."""

import io
import unittest
from collections import Counter
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import face


class FaceMenuTests(unittest.TestCase):
    def choose(self, *answers):
        output = io.StringIO()
        with patch("builtins.input", side_effect=answers), redirect_stdout(output), \
                patch.object(face.daily_runner, "tree_contains_media", side_effect=AssertionError("menu scanned queues")), \
                patch.object(face.Path, "rglob", side_effect=AssertionError("menu traversed photos")), \
                patch.object(face, "run_action", side_effect=AssertionError("menu ran processing")):
            action = face.show_menu()
        return action, output.getvalue()

    def test_seven_stable_choices_without_storage_scans(self):
        keys = ["daily", "dry-run", "status", "unknown-review",
                "recover-no-face", "recover-videos", "nudity"]
        for index, key in enumerate(keys, 1):
            with self.subTest(key=key):
                action, output = self.choose(str(index))
                self.assertEqual(action["key"], key)
                self.assertIn("[7] Run High-Precision Nudity Check", output)
                self.assertNotIn("[8]", output)

    def test_all_commands_have_one_menu_location_and_unambiguous_names(self):
        keys = [key for _, group in face.MENU_GROUPS for key in group]
        keys += [key for groups in face.TOOL_MENUS.values() for _, group in groups for key in group]
        self.assertEqual(Counter(keys), Counter(a["key"] for a in face.ACTIONS))
        names = [name for a in face.ACTIONS for name in [a["key"], *a.get("aliases", [])]]
        self.assertEqual(len(names), len(set(names)))
        for action in face.ACTIONS:
            for name in [action["key"], *action.get("aliases", [])]:
                self.assertIs(face.find_action_by_key(name), action)
            for step in action.get("steps", [action]):
                self.assertTrue((face.SCRIPT_DIR / step["script"]).is_file())

    def test_review_tools_diagnostics_and_back(self):
        self.assertEqual(self.choose("r", "2")[0]["key"], "benchmark-review")
        self.assertEqual(self.choose("d", "1")[0]["key"], "health")
        self.assertEqual(self.choose("a", "b", "4")[0]["key"], "unknown-review")
        self.assertEqual(self.choose("review-tools", "back", "diagnostics", "2")[0]["key"], "repair")

    def test_legacy_keywords_work_in_all_menus(self):
        for section in ("r", "d", "a"):
            for key in ("process", "review", "finish", "all-views", "process-all", "recover-unknown"):
                with self.subTest(section=section, key=key):
                    self.assertIs(self.choose(section, key)[0], face.find_action_by_key(key))

    def test_help_invalid_input_and_quit_do_not_run_work(self):
        action, output = self.choose("?", "invalid", "q")
        self.assertIsNone(action)
        for title in ("Review Tools:", "Diagnostics:", "Legacy / Compatibility:"):
            self.assertIn(title, output)
        self.assertIn("Unknown choice: invalid", output)
        self.assertIsNone(self.choose("d", "q")[0])

    def test_eof_and_interrupt_exit(self):
        for error in (EOFError, KeyboardInterrupt):
            with patch("builtins.input", side_effect=error), redirect_stdout(io.StringIO()):
                self.assertIsNone(face.show_menu())

    def test_daily_aliases_share_empty_inbox_fast_path(self):
        with patch.object(face, "source_review_storage_check", return_value=0), \
                patch.object(face.daily_runner, "intake_has_media", return_value=False), \
                patch.object(face, "cache_guard_check", side_effect=AssertionError("extra cache scan")), \
                patch.object(face, "run_steps", side_effect=AssertionError("ran empty ingest")), \
                redirect_stdout(io.StringIO()):
            for key in ("daily", "process", "process-new", "process-move", "sort", "go", "run", "end-to-end"):
                with self.subTest(key=key):
                    self.assertIs(face.find_action_by_key(key), face.find_action_by_key("daily"))
                    self.assertEqual(face.run_action(face.find_action_by_key(key)), 0)

    def test_daily_controls_and_nonempty_inbox_still_reach_guards(self):
        with patch.object(face, "source_review_storage_check", return_value=0), \
                patch.object(face.daily_runner, "intake_has_media", return_value=False) as intake, \
                patch.object(face, "cache_guard_check", return_value=92) as guard, \
                patch.object(face, "run_steps", side_effect=AssertionError("guard bypassed")):
            for key in ("daily", "process", "sort"):
                for flag in ("--resume", "--restart", "--full-maintenance", "--dry-run"):
                    self.assertEqual(face.run_action(face.find_action_by_key(key), [flag]), 92)
            self.assertEqual(guard.call_count, 12)
            intake.return_value = True
            self.assertEqual(face.run_action(face.find_action_by_key("process")), 92)

    def test_alias_forwards_arguments_to_daily_runner_once(self):
        with patch.object(face.subprocess, "run", return_value=SimpleNamespace(returncode=0)) as run, \
                redirect_stdout(io.StringIO()):
            self.assertEqual(face.run_steps(face.find_action_by_key("sort"), ["--resume"]), 0)
        run.assert_called_once()
        self.assertEqual(run.call_args.args[0], [face.sys.executable, str(face.SCRIPT_DIR / "daily_runner.py"), "--resume"])

    def test_review_and_recovery_keep_distinct_pipelines(self):
        quick = face.find_action_by_key("unknown-review")
        recovery = face.find_action_by_key("recover-unknown")
        self.assertEqual(quick["script"], "review_unknown_identities.py")
        self.assertEqual(quick["args"], ["--serve", "--open", "--auto-safe"])
        self.assertEqual(recovery["script"], "recover_no_usable_faces.py")
        self.assertNotIn("steps", quick)
        self.assertNotIn("steps", recovery)

    def test_keyword_listing_is_available_without_storage(self):
        with patch.object(face.sys, "argv", ["face", "commands"]), \
                patch.object(face, "source_review_storage_check", side_effect=AssertionError("requires SSD")), \
                redirect_stdout(io.StringIO()) as output:
            self.assertEqual(face.main(), 0)
        self.assertIn("process-new", output.getvalue())

    def test_cli_submenu_runs_only_selected_action(self):
        selected = face.find_action_by_key("benchmark-review")
        with patch.object(face.sys, "argv", ["face", "review-tools"]), \
                patch.object(face, "show_menu", return_value=selected) as menu, \
                patch.object(face, "run_action", return_value=0) as run:
            self.assertEqual(face.main(), 0)
        menu.assert_called_once_with("r")
        run.assert_called_once_with(selected)


if __name__ == "__main__":
    unittest.main()

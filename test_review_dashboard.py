"""Saved dashboard navigation must not trigger recognition or library walks."""

import io
import json
import os
import subprocess
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import review_dashboard as dashboard


class ReviewDashboardTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="face-dashboard-test-"))).resolve()
        for key in ("SOURCE_REVIEW", "PEOPLE", "UNKNOWN_HTML", "UNKNOWN_SUMMARY",
                    "DUPLICATE_REVIEW_HTML", "ADV_REPORT", "REF_REPORT", "DASHBOARD"):
            self.stack.enter_context(patch.object(dashboard, key, self.root / key))
        self.output = self.root / "dashboard.html"
        dashboard.UNKNOWN_HTML.write_text("<h1>Unknown Identity Quick Review</h1>")
        dashboard.UNKNOWN_SUMMARY.write_text(json.dumps({"progress": {"pending": 12}}))
        self.stack.enter_context(redirect_stdout(io.StringIO()))

    def run_dashboard(self, *args):
        with patch.object(dashboard.sys, "argv", ["review_dashboard", "--output", str(self.output), *args]):
            return dashboard.main()

    def test_first_open_uses_modern_reports_without_scanning_or_subprocesses(self):
        with patch.object(dashboard, "count_images", side_effect=AssertionError("image scan")), \
                patch.object(dashboard, "count_files", side_effect=AssertionError("folder scan")), \
                patch.object(dashboard.subprocess, "run", side_effect=AssertionError("processing started")), \
                patch.object(Path, "rglob", side_effect=AssertionError("recursive traversal")):
            self.assertEqual(self.run_dashboard(), 0)
        page = self.output.read_text()
        self.assertIn("Unknown Identity Quick Review", page)
        self.assertIn("Open saved report (read-only)", page)
        self.assertIn(dashboard.UNKNOWN_HTML.as_uri(), page)
        self.assertIn('class="count">12<', page)
        self.assertNotIn("unknown_triage", page)

    def test_unchanged_dashboard_reopens_without_rebuilding(self):
        self.run_dashboard()
        with patch.object(dashboard, "write_dashboard", side_effect=AssertionError("rebuilt unchanged report")), \
                patch.object(dashboard.webbrowser, "open") as browser:
            self.assertEqual(self.run_dashboard("--open", "--no-refresh"), 0)
        browser.assert_called_once_with(self.output.as_uri())

    def test_updated_report_invalidates_dashboard_without_folder_scan(self):
        self.run_dashboard()
        dashboard.UNKNOWN_SUMMARY.write_text(json.dumps({"progress": {"pending": 5}}))
        stamp = self.output.stat().st_mtime_ns + 1_000_000
        os.utime(dashboard.UNKNOWN_SUMMARY, ns=(stamp, stamp))
        with patch.object(dashboard, "count_images", side_effect=AssertionError("image scan")):
            self.run_dashboard()
        self.assertIn('class="count">5<', self.output.read_text())

    def test_removed_report_drops_stale_link(self):
        self.run_dashboard()
        dashboard.UNKNOWN_HTML.unlink()
        self.assertFalse(dashboard.dashboard_is_current(self.output))
        self.run_dashboard()
        self.assertNotIn(dashboard.UNKNOWN_HTML.as_uri(), self.output.read_text())

    def test_partial_or_legacy_dashboard_is_rebuilt(self):
        for text in ("<h1>Old unknown triage</h1>", "<html>partial"):
            self.output.write_text(text)
            self.assertFalse(dashboard.dashboard_is_current(self.output))
            self.run_dashboard()
            self.assertTrue(dashboard.dashboard_is_current(self.output))

    def test_explicit_refresh_only_rebuilds_duplicate_preview_and_counts(self):
        with patch.object(dashboard.subprocess, "run") as run, \
                patch.object(dashboard, "count_images", return_value=4) as images, \
                patch.object(dashboard, "count_files", return_value=8):
            self.assertEqual(self.run_dashboard("--refresh"), 0)
        run.assert_called_once()
        command = run.call_args.args[0]
        self.assertEqual(Path(command[1]).name, "near_visual_review.py")
        self.assertEqual(command[2:], ["--html-only", "--quiet"])
        self.assertTrue(run.call_args.kwargs["check"])
        images.assert_called_once()
        self.assertIn('class="count">8<', self.output.read_text())

    def test_refresh_failure_keeps_existing_dashboard(self):
        self.output.write_text("saved dashboard")
        with patch.object(dashboard.subprocess, "run", side_effect=subprocess.CalledProcessError(7, ["preview"])):
            self.assertEqual(self.run_dashboard("--refresh"), 7)
        self.assertEqual(self.output.read_text(), "saved dashboard")

    def test_missing_or_partial_summary_is_not_reported_as_zero_pending(self):
        for value in ("{", "null", "{}", '{"progress":null}', '{"progress":{"pending":true}}',
                      '{"progress":{"pending":-2}}', '{"progress":{"pending":"12"}}'):
            dashboard.UNKNOWN_SUMMARY.write_text(value)
            self.assertIsNone(dashboard.saved_unknown_count())
        dashboard.UNKNOWN_SUMMARY.unlink()
        self.assertIsNone(dashboard.saved_unknown_count())

    def test_refresh_flags_are_unambiguous(self):
        with redirect_stdout(io.StringIO()), patch.object(dashboard.sys, "stderr", io.StringIO()), \
                self.assertRaises(SystemExit) as error:
            self.run_dashboard("--refresh", "--no-refresh")
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()

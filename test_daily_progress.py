"""Presentation and subprocess tests; only temporary fixtures are processed."""

import io
import os
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import daily_runner
from daily_progress import CommandProgress, OutputLines


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.output = []
        self.progress = CommandProgress("Processing photos", emit=self.output.append, clock=lambda: self.now)

    def test_internal_model_and_per_person_messages_are_hidden(self):
        for line in ("Applied providers: CPUExecutionProvider", "find model: /models/face.onnx",
                     "00:01:00 INFO Indexing moved references for Person: 2/3",
                     "00:01:00 INFO Verifying moved references for Person: 3/3",
                     "Cache saved: /cache/file.pkl", "[OK  ] source folder /private/path"):
            self.progress.consume(line)
        self.assertEqual(self.output, [])

    def test_stage_progress_has_real_counts_and_percentages(self):
        self.progress.consume("23:49:12 INFO Primary-match safety benchmark: 448/6993 (0m 05s); 0 reused; 448 evaluated")
        self.assertEqual(self.output, ["  Matching safety check (primary): 448/6,993 (6%)"])
        self.progress.consume("Primary-match safety benchmark: 6993/6993 (1m 00s)")
        self.assertIn("6,993/6,993 (100%)", self.output[-1])

    def test_throttles_fast_updates_but_keeps_completion(self):
        for count in range(2001):
            self.progress.consume(f"Verifying confirmed references: {count}/2000")
        self.assertEqual(len(self.output), 2)
        self.assertIn("2,000/2,000 (100%)", self.output[-1])

    def test_pending_progress_is_flushed_and_heartbeat_is_honest(self):
        self.progress.consume("Verifying confirmed references: 1/10")
        self.progress.consume("Verifying confirmed references: 4/10")
        self.now = 5
        self.progress.tick()
        self.assertIn("4/10 (40%)", self.output[-1])
        self.now = 35
        self.progress.tick()
        self.assertIn("waiting for the next update", self.output[-1])
        self.assertIn("4/10 (40%)", self.output[-1])
        self.assertNotIn("100%", self.output[-1])

    def test_no_counts_invents_no_percentage(self):
        self.now = 31
        self.progress.tick()
        self.assertIn("Processing photos", self.output[-1])
        self.assertNotIn("%", self.output[-1])

    def test_failed_gate_is_not_reported_as_success(self):
        self.progress.consume("Safety benchmark: reusing unchanged cached result.")
        self.progress.consume("Safe auto-match is blocked because the safety benchmark failed.")
        self.assertIn("reusing the saved result", self.output[0])
        self.assertIn("paused by the safety check", self.output[1])
        self.assertNotIn("passed", " ".join(self.output))

    def test_unavailable_recovery_and_cache_read_errors_are_visible(self):
        self.progress.consume("Safe auto-match unavailable: model changed")
        self.progress.consume("Recovery unavailable: verifier failed")
        self.progress.consume("Read/signature errors:   2")
        self.assertEqual(len(self.output), 3)
        self.assertIn("2 files could not be read", self.output[-1])

    def test_cache_reuse_and_batch_progress_remain_clear(self):
        self.progress.consume("Already cached candidate files: 58000")
        self.progress.consume("Remaining candidate files:      25")
        self.progress.consume("[1/2] Detecting images 1-25...")
        self.assertIn("58,000", self.output[0])
        self.assertIn("25", self.output[1])
        self.assertIn("batch 1/2, checking photos 1-25", self.output[2])

    def test_warnings_errors_and_low_space_remain_visible(self):
        for i in range(5):
            self.progress.consume(f"00:10:01 WARNING Warning number {i}")
        self.progress.consume("00:10:02 ERROR Copy failed: source.jpg")
        self.progress.consume("insufficient_space: 20 GB reserve would be exceeded")
        self.progress.consume("[FAIL] duplicate running process")
        self.assertEqual(len(self.output), 8)
        self.assertIn("Warning number 4", self.output[4])

    def test_failure_exposes_tail_even_without_known_error_format(self):
        self.progress.consume("new worker format: unexpected stop reason")
        self.assertEqual(self.output, [])
        self.progress.finish(9)
        self.assertIn("unexpected stop reason", self.output[-1])

    def test_batch_counts_are_not_claimed_to_be_organized_totals(self):
        self.progress.consume("Worker batch 2 complete: 4,000 image(s) left the inbox; 115 remain.")
        self.assertIn("4,000 photos left the inbox; 115 remain", self.output[-1])
        self.assertNotIn("organized", self.output[-1])

    def test_carriage_returns_split_chunks_and_limit_memory(self):
        output = []
        lines = OutputLines(output.append, limit=8)
        lines.feed("a\rb\nc")
        lines.feed("d\r\ne")
        lines.finish()
        self.assertEqual(output, ["a", "b", "cd", "e"])
        lines.feed("x" * 1000)
        self.assertLessEqual(len(lines.pending), 8)

    def test_malformed_counts_do_not_show_false_progress(self):
        for line in ("Worker: 5/0", "Worker: 5/2", "Worker: ,/,", f"Worker: {'9' * 5000}/2"):
            self.progress.consume(line)
        self.assertEqual(self.output, [])


class WorkerOutputTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="face-progress-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.log = self.root / "daily_run_fixture.log"

    def test_compact_output_keeps_complete_raw_diagnostics_in_log(self):
        script = "import os; os.write(1,b'Applied providers: CPU\\nWorker: 50%|xx| 1/2 [00:00]\\rWorker: 100%|xxx| 2/2 [00:01]\\nERROR: useful failure\\n'); raise SystemExit(7)"
        with redirect_stdout(io.StringIO()) as output:
            result = daily_runner.run_command([sys.executable, "-c", script], self.log, verbose=False, step_name="process")
        self.assertEqual(result, 7)
        # Failure tails intentionally retain unknown diagnostics for debugging.
        self.assertIn("ERROR: useful failure", output.getvalue())
        self.assertIn("2/2 (100%)", output.getvalue())
        self.assertIn(b"Applied providers: CPU\n", self.log.read_bytes())
        self.assertIn(b"[00:00]\rWorker", self.log.read_bytes())

    def test_verbose_output_is_unfiltered_and_unbuffered(self):
        script = "import os; print(os.environ['PYTHONUNBUFFERED']); print('Applied providers: CPU')"
        with redirect_stdout(io.StringIO()) as output:
            result = daily_runner.run_command([sys.executable, "-c", script], self.log)
        self.assertEqual(result, 0)
        self.assertEqual(output.getvalue(), "1\nApplied providers: CPU\n")

    def test_progress_reaches_console_before_child_exits_without_newline(self):
        ack = self.root / "ack"
        def emit(message):
            if "1/2 (50%)" in message:
                ack.touch()
        script = (
            "import os,time,pathlib; os.write(1,b'Worker: 50%|xx| 1/2 [00:00]\\r'); "
            f"ack=pathlib.Path({str(ack)!r}); stop=time.monotonic()+3\n"
            "while not ack.exists() and time.monotonic()<stop: time.sleep(0.02)\n"
            "raise SystemExit(0 if ack.exists() else 8)"
        )
        with patch.object(daily_runner, "CommandProgress", side_effect=lambda label: CommandProgress(label, emit=emit)):
            result = daily_runner.run_command([sys.executable, "-c", script], self.log, verbose=False)
        self.assertEqual(result, 0)
        self.assertTrue(ack.exists())

    def test_utf8_split_across_chunks_is_not_corrupted(self):
        script = "import os,time; os.write(1,b'caf\\xc3'); time.sleep(.03); os.write(1,b'\\xa9\\n')"
        with redirect_stdout(io.StringIO()) as output:
            self.assertEqual(daily_runner.run_command([sys.executable, "-c", script], self.log), 0)
        self.assertEqual(output.getvalue(), "caf\u00e9\n")

    def test_interrupt_stops_owned_worker_and_preserves_resume_status(self):
        selector = SimpleNamespace(register=lambda *args: None, close=lambda: None)
        def interrupt(**kwargs):
            raise KeyboardInterrupt
        selector.select = interrupt
        with patch.object(daily_runner.selectors, "DefaultSelector", return_value=selector), \
                redirect_stdout(io.StringIO()) as output:
            result = daily_runner.run_command([sys.executable, "-c", "import time; time.sleep(60)"], self.log, verbose=False)
        self.assertEqual(result, 130)
        self.assertIn("completed work remains saved", output.getvalue())


class DailyPresentationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        root = Path(self.stack.enter_context(tempfile.TemporaryDirectory(prefix="face-daily-display-")))
        self.stack.enter_context(patch.object(daily_runner, "SUMMARY_DIR", root))
        self.stack.enter_context(patch.object(daily_runner, "STATE_FILE", root / "state.json"))
        self.snapshot = {"to_process_images": 3, "person_original_images": 10}
        self.stack.enter_context(patch.object(daily_runner, "snapshot", return_value=self.snapshot))
        self.stack.enter_context(patch.object(daily_runner, "original_person_counts", return_value={"Person": 10}))
        self.stack.enter_context(patch.object(daily_runner, "memory_profile", return_value={"ok": True, "message": "normal", "available_mb": 8192, "batch_size": 50}))
        self.stack.enter_context(patch.object(daily_runner, "intake_has_media", return_value=True))
        self.stack.enter_context(patch.object(daily_runner, "ensure_source_guard_baseline", return_value={}))
        self.guard = self.stack.enter_context(patch.object(daily_runner, "check_source_guard", return_value=(True, {}, [])))
        self.manifest = self.stack.enter_context(patch.object(daily_runner, "check_source_manifest", return_value=SimpleNamespace(ok=True)))
        self.promote = self.stack.enter_context(patch.object(daily_runner.source_manifest, "promote_current", return_value=root / "manifest.json"))
        self.steps = [{"name": "process", "desc": "Old technical name", "cmd": ["worker"]}]
        self.stack.enter_context(patch.object(daily_runner, "step_list", return_value=self.steps))

    def test_compact_is_default_without_removing_safety_checks(self):
        with patch.object(sys, "argv", ["daily_runner"]), \
                patch.object(daily_runner, "run_command", return_value=0) as run, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(daily_runner.main(), 0)
        run.assert_called_once()
        self.assertEqual(run.call_args.kwargs, {"verbose": False, "step_name": "process"})
        self.assertEqual(self.guard.call_count, 3)
        self.assertEqual(self.manifest.call_count, 3)
        self.promote.assert_called_once()
        self.assertIn("[1/1] Processing new photos", output.getvalue())
        self.assertNotIn("Step timing:", output.getvalue())
        self.assertNotIn("Old technical name", output.getvalue())

    def test_failure_saves_checkpoint_and_does_not_claim_completion(self):
        with patch.object(sys, "argv", ["daily_runner"]), \
                patch.object(daily_runner, "run_command", return_value=9), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(daily_runner.main(), 9)
        self.assertTrue(daily_runner.STATE_FILE.exists())
        self.promote.assert_not_called()
        self.assertIn("face daily --resume", output.getvalue())
        self.assertNotIn("Daily run complete", output.getvalue())

    def test_resume_reuses_completed_steps_without_running_workers(self):
        state = {"run_id": "resumed", "started_at": 0, "before": self.snapshot,
                 "steps": {"process": {"status": "completed"}},
                 "memory": {"ok": True, "batch_size": 50, "message": "normal"}}
        with patch.object(sys, "argv", ["daily_runner", "--resume"]), \
                patch.object(daily_runner, "load_state", return_value=state), \
                patch.object(daily_runner, "run_command") as run, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(daily_runner.main(), 0)
        run.assert_not_called()
        self.assertIn("already completed", output.getvalue())

    def test_failed_starting_manifest_still_blocks_workers(self):
        self.manifest.return_value = SimpleNamespace(ok=False)
        with patch.object(sys, "argv", ["daily_runner"]), \
                patch.object(daily_runner, "run_command") as run, redirect_stdout(io.StringIO()) as output:
            self.assertEqual(daily_runner.main(), daily_runner.SOURCE_GUARD_EXIT)
        run.assert_not_called()
        self.promote.assert_not_called()
        self.assertIn("protected source manifest failed", output.getvalue())

    def test_summary_distinguishes_pending_reviews_from_current_run(self):
        before = {"to_process_images": 10, "person_original_images": 100}
        after = {"to_process_images": 0, "person_original_images": 110,
                 "unassigned_unknown_identity": 700, "unassigned_copy_failed": 2}
        with redirect_stdout(io.StringIO()) as output:
            daily_runner.print_summary(before, after, Path("summary.json"), verbose=False)
        text = output.getvalue()
        self.assertIn("including earlier runs", text)
        self.assertIn("700 unknown identity", text)
        self.assertIn("technical attention: 2", text)
        self.assertNotIn("New organized images", text)


if __name__ == "__main__":
    unittest.main()

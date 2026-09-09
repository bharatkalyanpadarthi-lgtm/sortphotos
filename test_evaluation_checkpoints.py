"""Synthetic checkpoint tests; no application imports or production data."""

import hashlib
import json
import sqlite3
import subprocess
import sys
import tempfile
import textwrap
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from threading import Barrier
from unittest.mock import patch

from evaluation_checkpoints import BenchmarkCheckpoints


def checksum(signature, stage, payload):
    key = json.dumps([signature, stage], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(key + b"\0" + payload.encode("utf-8")).hexdigest()


class BenchmarkCheckpointTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / "checkpoints.sqlite3"

    def execute(self, sql, parameters=()):
        connection = sqlite3.connect(self.path)
        try:
            with connection:
                return connection.execute(sql, parameters).fetchall()
        finally:
            connection.close()

    def test_roundtrip_reopen_and_independent_stages(self):
        values = {
            "metrics": {"ok": True, "missing": None, "score": 0.75, "count": 3},
            "rows": [{"label": "caf\u00e9", "values": [False, -4, "text"]}],
            "empty-object": {},
            "empty-array": [],
        }
        with BenchmarkCheckpoints(self.path, "run") as cache:
            self.assertIsNone(cache.get("missing"))
            for stage, payload in values.items():
                self.assertIsNone(cache.put(stage, payload))
                self.assertEqual(cache.get(stage), payload)
            cache.get("metrics")["count"] = 999
            self.assertEqual(cache.get("metrics"), values["metrics"])
        with BenchmarkCheckpoints(self.path, "run") as cache:
            for stage, payload in values.items():
                self.assertEqual(cache.get(stage), payload)
            cache.put("metrics", {"updated": True})
            self.assertEqual(cache.get("metrics"), {"updated": True})
            self.assertEqual(cache.get("rows"), values["rows"])

    def test_signatures_are_separate_namespaces(self):
        with BenchmarkCheckpoints(self.path, "first'") as cache:
            cache.put("stage'", {"run": 1})
        with BenchmarkCheckpoints(self.path, "second") as cache:
            self.assertIsNone(cache.get("stage'"))
            cache.put("stage'", {"run": 2})
        with BenchmarkCheckpoints(self.path, "first'") as cache:
            self.assertEqual(cache.get("stage'"), {"run": 1})
        with BenchmarkCheckpoints(self.path, "second") as cache:
            self.assertEqual(cache.get("stage'"), {"run": 2})

    def test_completed_stages_survive_later_interruption(self):
        with self.assertRaises(KeyboardInterrupt):
            with BenchmarkCheckpoints(self.path, "run") as cache:
                cache.put("completed", {"ok": True})
                raise KeyboardInterrupt()
        with BenchmarkCheckpoints(self.path, "run") as cache:
            self.assertEqual(cache.get("completed"), {"ok": True})

    def test_interrupted_put_rolls_back_insert_and_replacement(self):
        with BenchmarkCheckpoints(self.path, "run") as cache:
            cache.put("existing", {"old": True})

        class InterruptedConnection(sqlite3.Connection):
            def execute(self, sql, parameters=()):
                cursor = super().execute(sql, parameters)
                if sql.startswith("INSERT INTO checkpoints"):
                    raise KeyboardInterrupt()
                return cursor

        connect = sqlite3.connect
        with patch("evaluation_checkpoints.sqlite3.connect",
                   side_effect=lambda path: connect(path, factory=InterruptedConnection)):
            with BenchmarkCheckpoints(self.path, "run") as cache:
                for stage in ("existing", "new"):
                    with self.subTest(stage=stage), self.assertRaises(KeyboardInterrupt):
                        cache.put(stage, {"partial": True})
                    self.assertEqual(cache.get("existing"), {"old": True})
                    self.assertIsNone(cache.get("new"))
        with BenchmarkCheckpoints(self.path, "run") as cache:
            self.assertEqual(cache.get("existing"), {"old": True})
            self.assertIsNone(cache.get("new"))

    def test_process_exit_during_put_recovers_previous_checkpoint(self):
        with BenchmarkCheckpoints(self.path, "run") as cache:
            cache.put("stage", {"old": True})
        script = textwrap.dedent("""
            import os
            import sqlite3
            import sys
            from pathlib import Path
            import evaluation_checkpoints

            class InterruptedConnection(sqlite3.Connection):
                def execute(self, sql, parameters=()):
                    cursor = super().execute(sql, parameters)
                    if sql.startswith("INSERT INTO checkpoints"):
                        os._exit(23)
                    return cursor

            connect = sqlite3.connect
            evaluation_checkpoints.sqlite3.connect = (
                lambda path: connect(path, factory=InterruptedConnection))
            with evaluation_checkpoints.BenchmarkCheckpoints(Path(sys.argv[1]), "run") as cache:
                cache.put("stage", {"partial": "x" * 1000000})
        """)
        result = subprocess.run(
            [sys.executable, "-B", "-c", script, str(self.path)],
            cwd=Path(__file__).resolve().parent, capture_output=True, text=True, timeout=15,
        )
        self.assertEqual(result.returncode, 23, result.stderr)
        with BenchmarkCheckpoints(self.path, "run") as cache:
            self.assertEqual(cache.get("stage"), {"old": True})
            cache.put("stage", {"resumed": True})
            self.assertEqual(cache.get("stage"), {"resumed": True})

    def test_malformed_and_incompatible_rows_are_misses(self):
        invalid = [
            '{"partial":', '{"ok":true} trailing', "null", "true", "12", '"text"',
            '{"value":NaN}', '{"value":Infinity}', '{"value":-Infinity}',
            '[{"value":1e999}]', '{"ok":true,"ok":false}',
            '[{"nested":{"value":1,"value":2}}]', b'{"blob":true}',
            "[" * 2000 + "]" * 2000,
        ]
        with BenchmarkCheckpoints(self.path, "run") as cache:
            cache.put("bad", {})
            cache.put("good", [1])
            for raw in invalid:
                with self.subTest(raw=repr(raw)[:80]):
                    digest = checksum("run", "bad", raw) if isinstance(raw, str) else "0" * 64
                    self.execute(
                        "UPDATE checkpoints SET payload = ?, checksum = ? WHERE stage = 'bad'",
                        (raw, digest),
                    )
                    self.assertIsNone(cache.get("bad"))
                    self.assertEqual(cache.get("good"), [1])

    def test_checksum_rejects_valid_json_changes(self):
        with BenchmarkCheckpoints(self.path, "run") as cache:
            cache.put("stage", {"count": 1})
            original = self.execute("SELECT payload, checksum FROM checkpoints")[0]
            self.assertEqual(original[1], checksum("run", "stage", original[0]))
            self.execute("UPDATE checkpoints SET payload = ?", ('{"count":9}',))
            self.assertIsNone(cache.get("stage"))
            cache.put("stage", {"count": 9})
            self.assertEqual(cache.get("stage"), {"count": 9})

    def test_checksum_rejects_damaged_digests(self):
        with BenchmarkCheckpoints(self.path, "run") as cache:
            cache.put("stage", [])
            for digest in ("", "0" * 64, "not-a-checksum", b"0" * 64):
                with self.subTest(digest=digest):
                    self.execute("UPDATE checkpoints SET checksum = ?", (digest,))
                    self.assertIsNone(cache.get("stage"))

    def test_checksum_binds_signature_and_stage(self):
        with BenchmarkCheckpoints(self.path, "first") as cache:
            cache.put("stage", {"ok": True})
            self.execute("UPDATE checkpoints SET stage = 'other'")
            self.assertIsNone(cache.get("other"))
            self.execute("UPDATE checkpoints SET stage = 'stage'")
        with BenchmarkCheckpoints(self.path, "second") as cache:
            self.execute("UPDATE checkpoints SET signature = 'second'")
            self.assertIsNone(cache.get("stage"))

    def test_open_does_not_scan_checkpoint_contents(self):
        with BenchmarkCheckpoints(self.path, "run") as cache:
            cache.put("stage", [])
        statements = []
        connect = sqlite3.connect

        def traced_connect(path):
            connection = connect(path)
            connection.set_trace_callback(statements.append)
            return connection

        with patch("evaluation_checkpoints.sqlite3.connect", side_effect=traced_connect):
            with BenchmarkCheckpoints(self.path, "run"):
                pass
        for sql in statements:
            normalized = sql.lower()
            self.assertNotIn("integrity_check", normalized)
            self.assertNotIn("quick_check", normalized)
            if "from checkpoints" in normalized:
                self.assertIn("limit 0", normalized)

    def test_open_never_evicts_other_live_namespaces(self):
        with ExitStack() as stack:
            caches = [stack.enter_context(BenchmarkCheckpoints(self.path, str(i)))
                      for i in range(6)]
            for i, cache in enumerate(caches):
                cache.put("stage", {"run": i})
            for i, cache in enumerate(caches):
                self.assertEqual(cache.get("stage"), {"run": i})
        self.assertEqual(self.execute("SELECT COUNT(*) FROM namespaces"), [(6,)])

    def test_concurrent_opens_serialize_recency_without_eviction(self):
        barrier = Barrier(6)

        def run(signature):
            barrier.wait(timeout=10)
            with BenchmarkCheckpoints(self.path, signature) as cache:
                cache.put("stage", [signature])
                self.assertEqual(cache.get("stage"), [signature])

        with BenchmarkCheckpoints(self.path, "live") as live:
            live.put("stage", ["live"])
            with ThreadPoolExecutor(max_workers=6) as executor:
                futures = [executor.submit(run, str(i)) for i in range(6)]
                for future in futures:
                    future.result(timeout=15)
            self.assertEqual(live.get("stage"), ["live"])
            live.put("resumed", ["still live"])
        self.assertEqual(self.execute("SELECT opened_order FROM namespaces ORDER BY opened_order"),
                         [(i,) for i in range(1, 8)])
        self.assertEqual(self.execute("SELECT COUNT(*) FROM checkpoints"), [(8,)])

    def test_explicit_prune_keeps_current_and_two_most_recent_others(self):
        for signature in ("a", "b", "c", "d", "e"):
            with BenchmarkCheckpoints(self.path, signature) as cache:
                cache.put("stage", {"signature": signature})
                cache.put("other", [])
        with BenchmarkCheckpoints(self.path, "b") as cache:
            self.assertIsNone(cache.prune())
            self.assertEqual(cache.get("stage"), {"signature": "b"})
            cache.prune()
        self.assertEqual(self.execute("SELECT signature FROM namespaces ORDER BY signature"),
                         [("b",), ("d",), ("e",)])
        self.assertEqual(self.execute("SELECT DISTINCT signature FROM checkpoints ORDER BY signature"),
                         [("b",), ("d",), ("e",)])
        self.assertEqual(self.execute("SELECT COUNT(*) FROM checkpoints"), [(6,)])

    def test_reopening_refreshes_transactional_recency(self):
        for signature in ("a", "b", "c", "a", "d"):
            with BenchmarkCheckpoints(self.path, signature) as cache:
                cache.put("stage", [signature])
        self.assertEqual(
            self.execute("SELECT signature, opened_order FROM namespaces ORDER BY opened_order"),
            [("b", 2), ("c", 3), ("a", 4), ("d", 5)],
        )
        with BenchmarkCheckpoints(self.path, "d") as cache:
            cache.prune()
        self.assertEqual(self.execute("SELECT signature FROM namespaces ORDER BY signature"),
                         [("a",), ("c",), ("d",)])

    def test_prune_preserves_current_even_if_it_was_opened_long_ago(self):
        with BenchmarkCheckpoints(self.path, "current") as current:
            current.put("stage", ["current"])
            for signature in ("a", "b", "c", "d"):
                with BenchmarkCheckpoints(self.path, signature) as cache:
                    cache.put("stage", [signature])
            current.prune()
            self.assertEqual(current.get("stage"), ["current"])
            self.assertEqual(self.execute("SELECT signature FROM namespaces ORDER BY signature"),
                             [("c",), ("current",), ("d",)])

    def test_interrupted_prune_rolls_back_namespaces_and_payloads(self):
        for signature in ("a", "b", "c", "d"):
            with BenchmarkCheckpoints(self.path, signature) as cache:
                cache.put("stage", [signature])

        class InterruptedConnection(sqlite3.Connection):
            def execute(self, sql, parameters=()):
                cursor = super().execute(sql, parameters)
                if sql.startswith("DELETE FROM namespaces"):
                    raise KeyboardInterrupt()
                return cursor

        connect = sqlite3.connect
        with patch("evaluation_checkpoints.sqlite3.connect",
                   side_effect=lambda path: connect(path, factory=InterruptedConnection)):
            with BenchmarkCheckpoints(self.path, "d") as cache:
                with self.assertRaises(KeyboardInterrupt):
                    cache.prune()
        self.assertEqual(self.execute("SELECT COUNT(*) FROM namespaces"), [(4,)])
        self.assertEqual(self.execute("SELECT COUNT(*) FROM checkpoints"), [(4,)])
        with BenchmarkCheckpoints(self.path, "a") as cache:
            self.assertEqual(cache.get("stage"), ["a"])

    def test_interrupted_open_rolls_back_namespace_recency(self):
        with BenchmarkCheckpoints(self.path, "run"):
            pass
        before = self.execute("SELECT signature, opened_order FROM namespaces")

        class InterruptedConnection(sqlite3.Connection):
            def execute(self, sql, parameters=()):
                cursor = super().execute(sql, parameters)
                if sql.startswith("INSERT INTO namespaces"):
                    raise KeyboardInterrupt()
                return cursor

        connect = sqlite3.connect
        with patch("evaluation_checkpoints.sqlite3.connect",
                   side_effect=lambda path: connect(path, factory=InterruptedConnection)):
            for signature in ("run", "new"):
                with self.subTest(signature=signature), self.assertRaises(KeyboardInterrupt):
                    with BenchmarkCheckpoints(self.path, signature):
                        self.fail("Interrupted open was accepted")
        self.assertEqual(self.execute("SELECT signature, opened_order FROM namespaces"), before)

    def test_invalid_payloads_never_insert_or_replace(self):
        cycle = []
        cycle.append(cycle)
        invalid = [
            None, True, 3, "scalar", (1, 2), {1: "coerced key"},
            {"nested": {False: 1}}, {"tuple": (1, 2)}, {"set": {1}},
            {"object": object()}, {"bytes": b"data"}, cycle,
            {"value": float("nan")}, [float("inf")],
            {"nested": [float("-inf")]},
        ]
        with BenchmarkCheckpoints(self.path, "run") as cache:
            cache.put("existing", {"old": True})
            for payload in invalid:
                for stage in ("existing", "new"):
                    with self.subTest(payload=repr(payload), stage=stage):
                        with self.assertRaises((TypeError, ValueError, RecursionError)):
                            cache.put(stage, payload)
                        self.assertEqual(cache.get("existing"), {"old": True})
                        self.assertIsNone(cache.get("new"))
            cache.put("new", ["valid"])
        with BenchmarkCheckpoints(self.path, "run") as cache:
            self.assertEqual(cache.get("existing"), {"old": True})
            self.assertEqual(cache.get("new"), ["valid"])

    def test_unsupported_schema_is_not_overwritten(self):
        with BenchmarkCheckpoints(self.path, "run") as cache:
            cache.put("stage", ["preserved"])
        for version in ("0", "999", "invalid"):
            with self.subTest(version=version):
                self.execute("UPDATE metadata SET value = ? WHERE key = 'schema_version'", (version,))
                before = self.path.read_bytes()
                with self.assertRaisesRegex(ValueError, "Unsupported checkpoint schema"):
                    with BenchmarkCheckpoints(self.path, "run"):
                        self.fail("Unsupported schema was accepted")
                self.assertEqual(self.path.read_bytes(), before)

    def test_missing_schema_version_is_not_recreated(self):
        with BenchmarkCheckpoints(self.path, "run"):
            pass
        self.execute("DELETE FROM metadata")
        before = self.path.read_bytes()
        with self.assertRaisesRegex(ValueError, "Unsupported checkpoint schema"):
            with BenchmarkCheckpoints(self.path, "run"):
                self.fail("Missing schema version was accepted")
        self.assertEqual(self.path.read_bytes(), before)

    def test_incompatible_database_schema_propagates(self):
        self.execute("CREATE TABLE unrelated (value TEXT)")
        before = self.path.read_bytes()
        with self.assertRaises(sqlite3.DatabaseError):
            with BenchmarkCheckpoints(self.path, "run"):
                self.fail("Unrelated database was accepted")
        self.assertEqual(self.path.read_bytes(), before)

    def test_previous_undeployed_layout_is_rejected_without_changes(self):
        self.execute("CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.execute("INSERT INTO metadata VALUES ('schema_version', '1')")
        self.execute(
            "CREATE TABLE checkpoints (signature TEXT NOT NULL, stage TEXT NOT NULL, "
            "payload TEXT NOT NULL, PRIMARY KEY (signature, stage))"
        )
        self.execute("INSERT INTO checkpoints VALUES ('run', 'stage', '{}')")
        before = self.path.read_bytes()
        with self.assertRaises(sqlite3.DatabaseError):
            with BenchmarkCheckpoints(self.path, "run"):
                self.fail("Old checksum-free layout was accepted")
        self.assertEqual(self.path.read_bytes(), before)

    def test_corrupt_database_propagates_without_replacement(self):
        corrupt = b"not a SQLite database\x00" * 100
        self.path.write_bytes(corrupt)
        with self.assertRaises(sqlite3.DatabaseError):
            with BenchmarkCheckpoints(self.path, "run"):
                self.fail("Corrupt database was accepted")
        self.assertEqual(self.path.read_bytes(), corrupt)

    def test_database_read_errors_are_not_cache_misses(self):
        with BenchmarkCheckpoints(self.path, "run") as cache:
            self.execute("DROP TABLE checkpoints")
            with self.assertRaises(sqlite3.DatabaseError):
                cache.get("stage")

    def test_requires_context_and_string_keys(self):
        cache = BenchmarkCheckpoints(self.path, "run")
        with self.assertRaises(RuntimeError):
            cache.get("stage")
        with self.assertRaises(RuntimeError):
            cache.prune()
        with cache:
            with self.assertRaises(RuntimeError):
                cache.__enter__()
            with self.assertRaises(TypeError):
                cache.get(1)
            with self.assertRaises(TypeError):
                cache.put(1, {})
        with self.assertRaises(RuntimeError):
            cache.put("stage", {})
        with self.assertRaises(RuntimeError):
            cache.prune()
        with self.assertRaises(TypeError):
            BenchmarkCheckpoints(self.path, 1)


if __name__ == "__main__":
    unittest.main()

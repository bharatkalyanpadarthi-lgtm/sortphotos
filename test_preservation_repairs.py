"""Disposable-file regressions for the September 10 preservation audit."""
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import analysis_index
import audit_nude_folders
import cleanup_empty_person_folders
import file_operations
import integration_audit
import operation_ledger
import pipeline_writer
import rename_transaction
import review_unknown_identities as review
import source_manifest
import sort_photos


class PreservationRepairs(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()

    def file(self, name, content=b"original"):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        return path

    def test_copy_never_replaces_occupied_file_or_link(self):
        src = self.file("src")
        dest = self.file("dest", b"other")
        for hardlinks in (True, False):
            with self.assertRaises(FileExistsError):
                file_operations.atomic_copy(src, dest, use_hardlinks=hardlinks)
            self.assertEqual(dest.read_bytes(), b"other")
        dest.unlink()
        dest.symlink_to(self.root / "missing")
        with self.assertRaises(FileExistsError):
            file_operations.atomic_copy(src, dest, use_hardlinks=False)
        self.assertTrue(dest.is_symlink())

    def test_copy_verifies_and_syncs_before_publication(self):
        src = self.file("src")
        dest = self.root / "dest"
        with patch.object(file_operations.os, "fsync", wraps=os.fsync) as sync:
            file_operations.atomic_copy(src, dest, use_hardlinks=False)
        self.assertGreaterEqual(sync.call_count, 2)
        self.assertEqual(dest.read_bytes(), src.read_bytes())

    def test_corrupt_copy_keeps_original_and_no_destination(self):
        src = self.file("src")
        dest = self.root / "dest"
        with patch.object(file_operations.shutil, "copyfileobj", side_effect=lambda _src, dst, *_: dst.write(b"corrupted")):
            with self.assertRaises(OSError):
                file_operations.atomic_copy(src, dest, use_hardlinks=False)
        self.assertFalse(dest.exists())
        self.assertEqual(src.read_bytes(), b"original")

    def test_move_no_clobber_and_verified_ledger(self):
        src = self.file("src")
        dest = self.file("dest", b"other")
        with self.assertRaises(FileExistsError):
            operation_ledger.move_path(src, dest, sorted_root=self.root, operation="test", reason="test")
        dest.unlink()
        operation_ledger.move_path(src, dest, sorted_root=self.root, operation="test", reason="test")
        events = operation_ledger.iter_events(self.root)
        self.assertEqual([e["status"] for e in events], ["planned", "verified", "moved"])
        self.assertEqual(events[-1]["dest"]["sha256"], hashlib.sha256(dest.read_bytes()).hexdigest())
        self.assertFalse(src.exists())

    def test_move_rejects_stale_supplied_source_hash(self):
        src = self.file("src")
        with self.assertRaises(ValueError):
            operation_ledger.move_path(src, self.root / "dest", sorted_root=self.root,
                                       operation="test", reason="test", source_sha256="wrong")
        self.assertTrue(src.exists())

    def test_rename_cycle_preserves_every_original(self):
        a = self.file("a.jpg", b"A")
        b = self.file("b.jpg", b"B")
        rename_transaction.apply([(a, b), (b, a)])
        self.assertEqual(a.read_bytes(), b"B")
        self.assertEqual(b.read_bytes(), b"A")
        self.assertFalse(list(self.root.glob(".rename*")))

    def test_failed_rename_staging_rolls_back(self):
        a = self.file("a.jpg", b"A")
        b = self.file("b.jpg", b"B")
        real = rename_transaction.rename_exclusive
        calls = 0
        def fail_second(src, dest):
            nonlocal calls
            calls += 1
            if calls == 2:
                raise OSError("interrupted staging")
            real(src, dest)
        with patch.object(rename_transaction, "rename_exclusive", side_effect=fail_second):
            with self.assertRaises(OSError):
                rename_transaction.apply([(a, self.root / "c.jpg"), (b, self.root / "d.jpg")])
        self.assertEqual(a.read_bytes(), b"A")
        self.assertEqual(b.read_bytes(), b"B")

    def test_interrupted_commit_recovers_from_journal(self):
        a = self.file("a.jpg", b"A")
        b = self.file("b.jpg", b"B")
        real = rename_transaction.rename_exclusive
        calls = 0
        def fail_fourth(src, dest):
            nonlocal calls
            calls += 1
            if calls == 4:
                raise OSError("interrupted commit")
            real(src, dest)
        with patch.object(rename_transaction, "rename_exclusive", side_effect=fail_fourth):
            with self.assertRaises(OSError):
                rename_transaction.apply([(a, b), (b, a)])
        rename_transaction.recover_under(self.root)
        self.assertEqual(a.read_bytes(), b"B")
        self.assertEqual(b.read_bytes(), b"A")

    def test_old_unjournaled_staging_is_preserved_and_blocks_cleanup(self):
        staged = self.file("Alice/photos/.rename_tmp_123_1.jpg")
        with self.assertRaises(ValueError):
            rename_transaction.recover_under(self.root)
        self.assertTrue(staged.exists())

    def test_review_original_and_hidden_original_are_not_empty(self):
        self.file("Alice/review/uncertain_nudity/a.jpg")
        self.file("Bob/photos/.rename_tmp_1.jpg")
        for name in ("Alice", "Bob"):
            self.assertTrue(cleanup_empty_person_folders.has_real_source_files(self.root / name))

    def test_audit_requires_review_preservation_for_cleanup(self):
        findings = []
        integration_audit.check_scanner_scope(findings)
        self.assertFalse(any(item.level == "FAIL" for item in findings))

    def test_manifest_rejects_same_size_replacement(self):
        people = self.root / "people"
        path = self.file("people/Alice/photos/a.jpg", b"AAAA")
        before = source_manifest.build_manifest(people)
        path.write_bytes(b"BBBB")
        comparison = source_manifest.compare_manifest(before, source_manifest.collect_entries(people), people_dir=people)
        self.assertEqual(len(comparison["size_changed"]), 1)

    def test_manifest_hash_resolves_rename_without_mtime_dependency(self):
        people = self.root / "people"
        path = self.file("people/Alice/photos/a.jpg")
        before = source_manifest.build_manifest(people)
        dest = path.with_name("b.jpg")
        path.rename(dest)
        os.utime(dest, (1, 1))
        result = source_manifest.compare_manifest(before, source_manifest.collect_entries(people), people_dir=people)
        self.assertEqual(len(result["renamed"]), 1)
        self.assertFalse(result["missing"])

    def test_relocation_needs_actual_verified_copy(self):
        people = self.root / "photos_by_person"
        src = self.file("photos_by_person/Alice/photos/a.jpg")
        expected = source_manifest.entry_for_path(src, people)
        dest = self.root / "_source_review" / "a.jpg"
        operation_ledger.move_path(src, dest, sorted_root=self.root, operation="test", reason="test")
        events = operation_ledger.iter_events(self.root)
        self.assertEqual(source_manifest.verified_relocation(expected, people, events), dest)
        dest.write_bytes(b"wrong!!!")
        self.assertIsNone(source_manifest.verified_relocation(expected, people, events))

    def test_manifest_only_authorizes_current_verified_relocation(self):
        people = self.root / "photos_by_person"
        src = self.file("photos_by_person/Alice/photos/a.jpg")
        manifest_path = self.root / "baseline.json"
        report_dir = self.root / "reports"
        source_manifest.save_manifest(source_manifest.build_manifest(people), manifest_path)
        dest = self.root / "_source_review" / "a.jpg"
        operation_ledger.move_path(src, dest, sorted_root=self.root,
                                   operation="test", reason="test", run_id="earlier-run")
        def check(run_id=None):
            return source_manifest.validate_current(people_dir=people,
                manifest_path=manifest_path, report_dir=report_dir,
                relocation_run_id=run_id)
        self.assertEqual(len(check().missing), 1)
        self.assertEqual(len(check("other-run").missing), 1)
        self.assertEqual(len(check("earlier-run").app_trashed), 1)
        with self.assertRaises(ValueError):
            source_manifest.promote_current(people_dir=people, manifest_path=manifest_path)
        dest.write_bytes(b"wrong!!!")
        self.assertEqual(len(check("earlier-run").missing), 1)

    def test_source_symlink_never_archives_external_original(self):
        src = self.file("outside.jpg")
        inbox = self.root / "inbox"
        inbox.mkdir()
        link = inbox / "link.jpg"
        link.symlink_to(src)
        for archive in (sort_photos.archive_organized_sources, sort_photos.archive_scanned_sources):
            self.assertEqual(archive({link}, inbox, self.root / "sorted"), 0)
        self.assertTrue(src.is_file())

    def test_content_lookup_does_not_hold_write_lock(self):
        source = self.file("image.jpg")
        with analysis_index.AnalysisIndex(self.root / "index.sqlite") as index:
            self.assertTrue(index.content_sha256(source))
            self.assertFalse(index.connection.in_transaction)

    def test_nudity_audit_honors_saved_uncertain_routing(self):
        path = self.root / "Alice/photos/nude/a.jpg"
        with patch.object(sort_photos, "ROUTE_UNCERTAIN_NUDITY_TO_NUDE", True):
            self.assertIsNone(audit_nude_folders.destination_for(path, self.root, "needs_review"))

    def test_stale_review_content_cannot_be_confirmed(self):
        path = self.file("review.jpg")
        key = review.item_key(path)
        item = SimpleNamespace(path=path, key=key)
        path.write_bytes(b"replaced")
        with self.assertRaisesRegex(ValueError, "changed since review"):
            review.apply_decision({"items_by_key": {key: item}}, item_keys=[key], action="keep_unknown")

    def test_writer_lease_excludes_unrelated_process_and_allows_child(self):
        lock = self.root / "writer.lock"
        script = "import pipeline_writer; from pathlib import Path;\nwith pipeline_writer.writer_lease(Path(__import__('sys').argv[1])): print('acquired')"
        with pipeline_writer.writer_lease(lock):
            child = subprocess.run([sys.executable, "-B", "-c", script, str(lock)], capture_output=True, text=True, **pipeline_writer.child_process_options())
            self.assertEqual(child.returncode, 0, child.stderr)
            env = dict(os.environ)
            env.pop(pipeline_writer.OWNER_ENV, None)
            other = subprocess.run([sys.executable, "-B", "-c", script, str(lock)], capture_output=True, text=True, env=env)
            self.assertNotEqual(other.returncode, 0)
        with pipeline_writer.writer_lease(lock):
            pass


if __name__ == "__main__":
    unittest.main()

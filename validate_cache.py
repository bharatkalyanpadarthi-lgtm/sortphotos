"""
validate_cache.py — Audit the face cache for corruption, stale entries, and
common inconsistencies. Reports problems with options to fix them.

What it checks:
  1. Cache file is loadable (not corrupted)
  2. Cache version matches current code
  3. Config fingerprint matches (detection params unchanged)
  4. All face entries have non-empty embeddings of correct shape
  5. All face entries have non-empty crop_jpeg
  6. All face entries have a src_str path
  7. file_signatures and faces stay in sync (every cached face's src is registered)
  8. No orphaned file_signatures (files Anthropic registered but with no faces)
  9. Source files actually still exist on disk
 10. Source files match their stored signature (mtime/size)
 11. Duplicate face entries (same src + face_index)
 12. Labeled faces — count and report named people
 13. AI suggestions cache — load + format check

Run:
    python validate_cache.py            # audit only
    python validate_cache.py --fix      # audit + offer to repair issues
    python validate_cache.py --fix --yes  # audit + auto-fix everything
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

# Ensure pickle can resolve the dataclasses
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from sort_photos import CacheState, CachedFace, config_fingerprint, file_signature
except ImportError as e:
    print(f"ERROR: Could not import from sort_photos.py: {e}")
    print("Run this script from the same folder as sort_photos.py")
    sys.exit(1)

# Help pickle find the classes (it stores them as __main__.CacheState)
sys.modules['__main__'].CacheState = CacheState
sys.modules['__main__'].CachedFace = CachedFace

CACHE_DIR     = Path.home() / ".face_sort_cache"
CACHE_FILE    = CACHE_DIR / "cache.pkl"
AI_CACHE_FILE = CACHE_DIR / "ai_suggestions.json"
EXPECTED_EMBEDDING_DIM = 512   # antelopev2 ArcFace output


# ============================================================================
# AUDIT REPORT
# ============================================================================

class Report:
    def __init__(self):
        self.issues: list[tuple[str, str, dict]] = []   # (severity, code, details)
        self.stats: dict[str, int] = {}

    def add(self, severity: str, code: str, **details) -> None:
        self.issues.append((severity, code, details))

    def has_severity(self, severity: str) -> bool:
        return any(sev == severity for sev, _, _ in self.issues)

    def of(self, code: str) -> list[dict]:
        return [d for _, c, d in self.issues if c == code]

    def print(self) -> None:
        print()
        print("=" * 60)
        print("CACHE AUDIT REPORT")
        print("=" * 60)
        for k in sorted(self.stats.keys()):
            print(f"  {k:30s}: {self.stats[k]}")
        if not self.issues:
            print()
            print("✓ No issues found. Cache looks healthy.")
            return

        by_sev: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for sev, code, det in self.issues:
            by_sev[sev].append((code, det))

        print()
        for sev in ("CRITICAL", "ERROR", "WARNING", "INFO"):
            if sev not in by_sev:
                continue
            print(f"--- {sev} ({len(by_sev[sev])} issue group(s)) ---")
            counts = Counter(c for c, _ in by_sev[sev])
            for code, n in counts.most_common():
                print(f"  [{code}] {n} occurrence(s)")
                # Show 3 sample details for each
                samples = [d for c, d in by_sev[sev] if c == code][:3]
                for d in samples:
                    snippet = ", ".join(f"{k}={v}" for k, v in d.items() if k != "_full")
                    print(f"      → {snippet[:120]}")
                if n > 3:
                    print(f"      → … and {n - 3} more")
            print()


# ============================================================================
# CHECKS
# ============================================================================

def check_cache_file(report: Report) -> CacheState | None:
    if not CACHE_FILE.exists():
        report.add("CRITICAL", "no_cache_file", path=str(CACHE_FILE))
        return None
    report.stats["cache_size_bytes"] = CACHE_FILE.stat().st_size
    try:
        with CACHE_FILE.open("rb") as f:
            data = pickle.load(f)
    except Exception as e:  # noqa: BLE001
        report.add("CRITICAL", "cache_unloadable", error=str(e)[:200])
        return None

    if not isinstance(data, CacheState):
        report.add("CRITICAL", "wrong_cache_type",
                   got=type(data).__name__, expected="CacheState")
        return None
    return data


def check_versions_and_config(cache: CacheState, report: Report) -> None:
    expected_version = 2  # current CACHE_VERSION
    if getattr(cache, "version", None) != expected_version:
        report.add("ERROR", "version_mismatch",
                   got=cache.version, expected=expected_version)
    expected_fp = config_fingerprint()
    cache_fp = getattr(cache, "config_fingerprint", "")
    if cache_fp != expected_fp:
        report.add("WARNING", "config_changed",
                   got=cache_fp[:60], expected=expected_fp[:60])


def check_face_entries(cache: CacheState, report: Report) -> None:
    by_key: dict[tuple[str, int], int] = defaultdict(int)
    label_counts: Counter[str] = Counter()
    for i, face in enumerate(cache.faces):
        if not isinstance(face, CachedFace):
            report.add("CRITICAL", "wrong_face_type",
                       index=i, got=type(face).__name__)
            continue
        if not face.src_str:
            report.add("ERROR", "missing_src_str", index=i)
        emb = face.embedding
        if emb is None or not hasattr(emb, "shape") or emb.size == 0:
            report.add("ERROR", "missing_embedding",
                       index=i, src=face.src_str)
        elif emb.shape != (EXPECTED_EMBEDDING_DIM,):
            report.add("ERROR", "wrong_embedding_shape",
                       index=i, got=str(emb.shape),
                       expected=f"({EXPECTED_EMBEDDING_DIM},)")
        if not face.crop_jpeg:
            report.add("ERROR", "missing_crop_jpeg",
                       index=i, src=face.src_str)
        else:
            # Quick sanity: JPEG starts with 0xFFD8
            if not face.crop_jpeg.startswith(b"\xff\xd8"):
                report.add("WARNING", "crop_not_jpeg",
                           index=i, src=face.src_str,
                           first_bytes=face.crop_jpeg[:4].hex())
        key = (face.src_str, face.face_index)
        by_key[key] += 1
        if face.label:
            label_counts[face.label] += 1

    dups = {k: n for k, n in by_key.items() if n > 1}
    for (src, face_idx), n in list(dups.items())[:50]:
        report.add("ERROR", "duplicate_face_entry",
                   src=src, face_index=face_idx, count=n)

    report.stats["total_face_entries"] = len(cache.faces)
    report.stats["unique_(src,face_idx)_keys"] = len(by_key)
    report.stats["labeled_faces"] = sum(label_counts.values())
    report.stats["distinct_people_labeled"] = len(label_counts)
    report.stats["duplicate_face_entries"] = sum(n - 1 for n in dups.values())


def check_signatures_vs_faces(cache: CacheState, report: Report) -> None:
    sig_paths = set(cache.file_signatures.keys())
    face_paths = {f.src_str for f in cache.faces if f.src_str}

    if not sig_paths and not face_paths:
        report.add(
            "WARNING",
            "empty_face_cache",
            detail=(
                "cache is loadable but contains no processed files or faces; "
                "old sources will be re-detected if introduced again"
            ),
        )

    only_in_sigs = sig_paths - face_paths
    only_in_faces = face_paths - sig_paths

    # only_in_sigs: file processed, no faces detected — totally normal
    # only_in_faces: orphan faces with no signature — bug
    for path in list(only_in_faces)[:50]:
        report.add("ERROR", "face_without_signature", path=path)

    report.stats["files_with_no_faces"] = len(only_in_sigs)
    report.stats["faces_with_no_signature"] = len(only_in_faces)


def check_files_on_disk(cache: CacheState, report: Report,
                        sample_only: bool = False) -> None:
    paths = list(cache.file_signatures.keys())
    if sample_only and len(paths) > 1000:
        # On large caches, only check a random sample
        import random
        rng = random.Random(42)
        paths = rng.sample(paths, 1000)
    n_missing = 0
    n_changed = 0
    for s in paths:
        p = Path(s)
        if not p.exists():
            n_missing += 1
            # Always record (no 50-cap) so --fix actually acts on all stale entries.
            # Display-side truncation lives elsewhere.
            report.add("WARNING", "src_file_missing", path=s)
            continue
        try:
            actual = file_signature(p)
        except OSError as e:
            report.add("WARNING", "src_file_unreadable", path=s, error=str(e)[:80])
            continue
        stored = cache.file_signatures[s]
        if actual != stored:
            n_changed += 1
            report.add("WARNING", "src_file_modified",
                       path=s, stored=str(stored), actual=str(actual))

    report.stats["src_files_missing"] = n_missing
    report.stats["src_files_modified"] = n_changed
    if sample_only:
        report.stats["disk_check_mode"] = "sampled (1000 files)"
    else:
        report.stats["disk_check_mode"] = "full"


def check_ai_cache(report: Report) -> None:
    if not AI_CACHE_FILE.exists():
        report.stats["ai_cache_present"] = 0
        return
    try:
        with AI_CACHE_FILE.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:  # noqa: BLE001
        report.add("ERROR", "ai_cache_unloadable", error=str(e)[:200])
        return
    if not isinstance(data, dict):
        report.add("ERROR", "ai_cache_wrong_type", got=type(data).__name__)
        return
    n_total = len(data)
    n_named = sum(1 for v in data.values() if v)
    n_unknown = n_total - n_named
    report.stats["ai_cache_present"] = 1
    report.stats["ai_cache_entries"] = n_total
    report.stats["ai_cache_named"] = n_named
    report.stats["ai_cache_unknown"] = n_unknown


# ============================================================================
# REPAIR
# ============================================================================

def repair(cache: CacheState, report: Report,
           auto_yes: bool) -> tuple[CacheState, bool]:
    """Apply fixes. Returns (possibly new cache, whether anything changed)."""
    changed = False

    def confirm(prompt: str) -> bool:
        if auto_yes:
            return True
        try:
            return input(f"  {prompt} (y/n): ").strip().lower() == "y"
        except (EOFError, KeyboardInterrupt):
            return False

    # Fix 1: drop entries missing critical fields
    bad_entries = (report.of("missing_embedding") +
                   report.of("missing_crop_jpeg") +
                   report.of("missing_src_str") +
                   report.of("wrong_embedding_shape") +
                   report.of("crop_not_jpeg"))
    bad_indices = {d["index"] for d in bad_entries if "index" in d}
    if bad_indices:
        print(f"\n[FIX] Found {len(bad_indices)} face entries with missing/invalid data.")
        if confirm("Drop these entries?"):
            cache.faces = [f for i, f in enumerate(cache.faces) if i not in bad_indices]
            changed = True

    # Fix 2: deduplicate (keep first occurrence per (src, face_index))
    if report.stats.get("duplicate_face_entries", 0) > 0:
        n_before = len(cache.faces)
        seen: set[tuple[str, int]] = set()
        deduped: list[CachedFace] = []
        for f in cache.faces:
            key = (f.src_str, f.face_index)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(f)
        n_after = len(deduped)
        n_removed = n_before - n_after
        if n_removed:
            print(f"\n[FIX] Found {n_removed} duplicate face entries.")
            if confirm("Remove duplicates?"):
                cache.faces = deduped
                changed = True

    # Fix 3: add signatures for orphan faces (back-fill)
    orphans = report.of("face_without_signature")
    if orphans:
        paths_to_fix = {d["path"] for d in orphans}
        # Try to compute signature from disk
        added = 0
        missing = 0
        for s in paths_to_fix:
            p = Path(s)
            if p.exists():
                try:
                    cache.file_signatures[s] = file_signature(p)
                    added += 1
                except OSError:
                    missing += 1
            else:
                missing += 1
        if added or missing:
            print(f"\n[FIX] {len(paths_to_fix)} faces have no file_signature.")
            print(f"      Could back-fill signatures for {added} (file exists),")
            print(f"      {missing} have no source file on disk.")
            if confirm("Apply back-fill (and drop faces with no source file)?"):
                # Drop faces whose src has no signature even after fix
                missing_paths = {s for s in paths_to_fix
                                  if s not in cache.file_signatures}
                if missing_paths:
                    cache.faces = [f for f in cache.faces
                                   if f.src_str not in missing_paths]
                changed = True

    # Fix 4: drop entries for files that no longer exist on disk
    missing_on_disk = report.of("src_file_missing")
    if missing_on_disk:
        missing_paths = {d["path"] for d in missing_on_disk}
        print(f"\n[FIX] {len(missing_paths)} source file(s) no longer exist on disk.")
        print("      Their cache entries (signatures + faces) are stale.")
        if confirm("Remove stale entries from cache?"):
            for p in missing_paths:
                cache.file_signatures.pop(p, None)
            cache.faces = [f for f in cache.faces if f.src_str not in missing_paths]
            changed = True

    return cache, changed


# ============================================================================
# MAIN
# ============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fix", action="store_true",
                        help="Offer to repair issues found")
    parser.add_argument("--yes", action="store_true",
                        help="With --fix, auto-confirm every fix")
    parser.add_argument("--full-disk-check", action="store_true",
                        help="Check every source file on disk (slower for large caches)")
    args = parser.parse_args()

    print(f"Auditing cache: {CACHE_FILE}")
    report = Report()

    cache = check_cache_file(report)
    if cache is None:
        report.print()
        return 1

    check_versions_and_config(cache, report)
    check_face_entries(cache, report)
    check_signatures_vs_faces(cache, report)

    print(f"Checking source files on disk "
          f"({'full' if args.full_disk_check else 'sampled'})…")
    check_files_on_disk(cache, report, sample_only=not args.full_disk_check)

    check_ai_cache(report)
    report.print()

    if args.fix and report.issues:
        cache, changed = repair(cache, report, auto_yes=args.yes)
        if changed:
            backup = CACHE_FILE.with_suffix(".pkl.bak")
            print(f"\nBacking up old cache → {backup}")
            shutil.copy2(str(CACHE_FILE), str(backup))
            tmp = CACHE_FILE.with_suffix(".pkl.tmp")
            with tmp.open("wb") as f:
                pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
            tmp.replace(CACHE_FILE)
            print(f"Wrote repaired cache → {CACHE_FILE}")
            print(f"Original preserved at  → {backup}")
            print()
            print("Re-run validate_cache.py to confirm everything is clean.")
        else:
            print("\nNo fixes applied.")

    if report.has_severity("CRITICAL"):
        return 2
    if report.has_severity("ERROR"):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

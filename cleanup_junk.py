#!/usr/bin/env python3
"""
cleanup_junk.py — Move source images of junk-labeled faces to a holding folder.

After you've marked clusters as junk by typing 'j' during labeling
(sort_photos.py), this script:

  1. Reads BOTH the finalized cache (~/.face_sort_cache/cache.pkl) AND the
     in-progress labeling state — so 'j' marks count even before you've
     finished a full labeling pass.
  2. Collects every source image that has at least one face labeled "__junk__".
     (Per your design choice: a source image with even one junk face is moved
     to the holding folder, even if it also contains labeled people.)
  3. Moves those source images to ~/Pictures/sorted_all_pictures/junk_to_review/ (preserving
     subfolder structure relative to ~/Pictures/To Process when possible).

Default mode is --dry-run: prints what WOULD happen, nothing moves. Use --apply
to commit. Move semantics, not delete: original files end up in the holding
folder, fully recoverable until you delete that folder yourself in Finder.

Usage:
    python cleanup_junk.py             # dry-run, default thresholds
    python cleanup_junk.py --apply     # actually move files
    python cleanup_junk.py --apply --also-remove-cluster-folder
                                       # also delete face_clusters/__junk__/

After --apply, run:
    python validate_cache.py --fix     # drop now-missing entries from cache
"""

import argparse
import shutil
import sys
from pathlib import Path

# Re-use the cache schema. Pickle was written with sort_photos as __main__,
# so we alias the dataclasses into __main__ before any load_*() call.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sort_photos  # noqa: F401
from sort_photos import (  # type: ignore
    CACHE_FILE,
    load_cache,
    load_labeling_state,
)

_main = sys.modules["__main__"]
for _name in ("CacheState", "CachedFace", "FaceRecord", "LabelingState"):
    if hasattr(sort_photos, _name) and not hasattr(_main, _name):
        setattr(_main, _name, getattr(sort_photos, _name))

JUNK_LABEL = "__junk__"
DEFAULT_INPUT = Path.home() / "Pictures"
DEFAULT_OUTPUT = Path.home() / "Pictures" / "sorted_all_pictures"
JUNK_HOLDING_DIR = DEFAULT_OUTPUT / "junk_to_review"


def collect_junk_src_paths(verbose: bool = False) -> tuple[set[str], dict[str, int]]:
    """Return (set of source paths, stats dict). Reads cache + labeling state."""
    junk_srcs: set[str] = set()
    stats = {"from_cache": 0, "from_state": 0, "junk_faces_total": 0}

    # 1. Finalized cache
    if CACHE_FILE.exists():
        cache = load_cache()
        for f in cache.faces:
            if f.label == JUNK_LABEL:
                stats["junk_faces_total"] += 1
                if f.src_str:
                    if f.src_str not in junk_srcs:
                        stats["from_cache"] += 1
                    junk_srcs.add(f.src_str)
        if verbose:
            print(f"  cache.pkl: {stats['from_cache']} unique source paths "
                  f"with junk faces")
    else:
        if verbose:
            print(f"  cache.pkl not found at {CACHE_FILE} — skipping")

    # 2. In-progress labeling state (covers junk marked but not yet finalized)
    state = load_labeling_state()
    if state is not None:
        before = len(junk_srcs)
        for face, cid in zip(state.faces, state.cluster_ids):
            label = state.name_map.get(cid)
            if label == JUNK_LABEL:
                stats["junk_faces_total"] += 1
                if face.src_str:
                    junk_srcs.add(face.src_str)
        added = len(junk_srcs) - before
        stats["from_state"] = added
        if verbose:
            print(f"  labeling_state.pkl: {added} additional unique source paths")
    else:
        if verbose:
            print(f"  labeling_state.pkl not found — skipping")

    return junk_srcs, stats


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--apply", action="store_true",
                   help="Move files. Default is dry-run.")
    p.add_argument("--input-root", type=Path, default=DEFAULT_INPUT,
                   help=f"Source-path root for preserving subfolder structure (default: {DEFAULT_INPUT}).")
    p.add_argument("--holding", type=Path, default=JUNK_HOLDING_DIR,
                   help=f"Destination holding folder (default: {JUNK_HOLDING_DIR}).")
    p.add_argument("--also-remove-cluster-folder", action="store_true",
                   help="Also remove ~/Pictures/sorted_all_pictures/face_clusters/__junk__/ after moving sources.")
    p.add_argument("--quiet", action="store_true",
                   help="Don't list every path in dry-run output.")
    args = p.parse_args()

    print("Scanning for junk-labeled faces...")
    junk_srcs, stats = collect_junk_src_paths(verbose=True)
    print()
    print(f"Total junk face entries: {stats['junk_faces_total']}")
    print(f"Unique source images:    {len(junk_srcs)}")
    print()

    if not junk_srcs:
        print("Nothing to clean up. Mark clusters with 'j' during labeling first.")
        return 0

    paths = sorted(junk_srcs)
    moved = 0
    skipped_missing = 0
    failed = 0

    for src_str in paths:
        src = Path(src_str)
        if not src.exists():
            if not args.quiet:
                print(f"  [missing on disk]  {src_str}")
            skipped_missing += 1
            continue

        # Compute destination — preserve relative path under input-root if possible.
        try:
            rel = src.resolve().relative_to(args.input_root.resolve())
        except ValueError:
            rel = Path(src.name)
        dst = args.holding / rel

        if args.apply:
            dst.parent.mkdir(parents=True, exist_ok=True)
            # Collision-avoidance — extremely unlikely given full src paths,
            # but cheap to handle.
            if dst.exists():
                stem = dst.stem
                suffix = "".join(dst.suffixes)
                k = 1
                while True:
                    candidate = dst.with_name(f"{stem}__dup{k}{suffix}")
                    if not candidate.exists():
                        dst = candidate
                        break
                    k += 1
            try:
                shutil.move(str(src), str(dst))
                moved += 1
            except OSError as exc:
                print(f"  FAILED: {src} -> {dst}: {exc}")
                failed += 1
        else:
            if not args.quiet:
                print(f"  [would move]  {src}")
            moved += 1  # would-be move

    print()
    if args.apply:
        print(f"Moved:                 {moved}")
        if skipped_missing:
            print(f"Skipped (missing):     {skipped_missing}")
        if failed:
            print(f"Failed:                {failed}")
        print(f"Holding folder:        {args.holding}")
        if args.also_remove_cluster_folder:
            cluster_junk = DEFAULT_OUTPUT / "face_clusters" / JUNK_LABEL
            if cluster_junk.exists():
                try:
                    shutil.rmtree(cluster_junk)
                    print(f"Removed cluster folder: {cluster_junk}")
                except OSError as exc:
                    print(f"Could NOT remove {cluster_junk}: {exc}")
        print()
        print("Next steps:")
        print(f"  1. Open {args.holding} in Finder and review.")
        print(f"  2. When satisfied, delete that folder (Cmd+Delete).")
        print(f"  3. Run `python validate_cache.py --fix` to drop now-missing")
        print(f"     entries from the cache so future sorts skip them.")
    else:
        print(f"Would move:            {moved}")
        if skipped_missing:
            print(f"Already missing:       {skipped_missing}")
        print()
        print("DRY-RUN — no files moved. Re-run with --apply to commit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

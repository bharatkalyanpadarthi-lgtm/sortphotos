#!/usr/bin/env python3
"""
mark_small_unlabeled_junk.py — Auto-mark small unlabeled clusters as junk.

Scans ~/Pictures/sorted_all_pictures/face_clusters/ for folders still named person_NNN
(i.e., never labeled by you) that contain fewer than --threshold face crops
(default 5). For each such folder this script:

  1. Parses the crop filenames to identify the matching faces in cache.pkl
     and sets their label to "__junk__".
  2. Moves the crop files into face_clusters/__junk__/.

After this runs, run cleanup_junk.py (dry-run first, then --apply) to move
the underlying source images to ~/Pictures/sorted_all_pictures/junk_to_review/ for final
review and deletion.

Default mode is dry-run. Use --apply to commit changes.

Safety guards:
  - Only operates on folders whose name matches the literal pattern person_NNN.
  - Skips any face that turns out to already have a real label (defensive —
    shouldn't happen, but if it does we don't clobber it).
  - Backs up cache.pkl before writing.

Usage:
    python mark_small_unlabeled_junk.py                    # dry-run, threshold 5
    python mark_small_unlabeled_junk.py --threshold 3      # only mark folders with <3
    python mark_small_unlabeled_junk.py --apply            # commit
    python mark_small_unlabeled_junk.py --apply --threshold 3
"""

import argparse
import re
import sys
from collections import defaultdict
from pathlib import Path

# Re-use the cache schema from sort_photos.py so unpickling works.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sort_photos  # noqa: F401
from sort_photos import CACHE_FILE, load_cache, save_cache  # type: ignore

_main = sys.modules["__main__"]
for _name in ("CacheState", "CachedFace", "FaceRecord", "LabelingState"):
    if hasattr(sort_photos, _name) and not hasattr(_main, _name):
        setattr(_main, _name, getattr(sort_photos, _name))

JUNK_LABEL = "__junk__"
DEFAULT_CLUSTERS_DIR = Path.home() / "Pictures" / "sorted_all_pictures" / "face_clusters"
PERSON_FOLDER_RE = re.compile(r"^person_\d+$")
# Crop filename pattern used by sort_photos.py:
#     "{src_stem}__face{face_index}_{tag}.jpg"
CROP_FILENAME_RE = re.compile(
    r"^(?P<stem>.+?)__face(?P<idx>\d+)_.*\.(?:jpg|jpeg|png|webp)$",
    re.IGNORECASE,
)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--threshold", type=int, default=5,
                   help="Mark unlabeled clusters with fewer than this many crops as junk. Default 5 (catches folders with 1-4 crops).")
    p.add_argument("--clusters-dir", type=Path, default=DEFAULT_CLUSTERS_DIR,
                   help=f"Path to face_clusters/ (default: {DEFAULT_CLUSTERS_DIR}).")
    p.add_argument("--apply", action="store_true",
                   help="Commit changes. Default is dry-run.")
    p.add_argument("--quiet", action="store_true",
                   help="Don't list every affected folder in the report.")
    args = p.parse_args()

    if not args.clusters_dir.is_dir():
        print(f"ERROR: {args.clusters_dir} is not a directory", file=sys.stderr)
        return 1

    print(f"Scanning {args.clusters_dir} ...")
    person_folders = sorted(
        d for d in args.clusters_dir.iterdir()
        if d.is_dir() and PERSON_FOLDER_RE.match(d.name)
    )

    small: list[tuple[Path, list[Path]]] = []
    for folder in person_folders:
        crops = [
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS
        ]
        if 0 < len(crops) < args.threshold:
            small.append((folder, crops))

    print(f"Total person_NNN folders: {len(person_folders)}")
    print(f"Below threshold ({args.threshold}): {len(small)}")
    print()

    if not small:
        print("Nothing to mark.")
        return 0

    # Distribution by size for transparency
    size_dist: dict[int, int] = defaultdict(int)
    for _, crops in small:
        size_dist[len(crops)] += 1
    print("Size distribution of affected folders:")
    for size in sorted(size_dist.keys()):
        print(f"  {size} crops: {size_dist[size]} folders")
    print()

    # Build (src_stem, face_index) -> [cache index] lookup
    print(f"Loading cache from {CACHE_FILE} ...")
    if not CACHE_FILE.exists():
        print(f"ERROR: cache not found at {CACHE_FILE}", file=sys.stderr)
        return 1
    cache = load_cache()
    by_key: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, fc in enumerate(cache.faces):
        if fc.src_str:
            by_key[(Path(fc.src_str).stem, fc.face_index)].append(i)

    target_indices: set[int] = set()
    folders_with_no_match: list[Path] = []
    crops_unparseable = 0
    crops_no_cache_match = 0
    crops_total = 0

    for folder, crops in small:
        folder_hits = 0
        for crop in crops:
            crops_total += 1
            m = CROP_FILENAME_RE.match(crop.name)
            if not m:
                crops_unparseable += 1
                continue
            stem = m.group("stem")
            idx = int(m.group("idx"))
            hits = by_key.get((stem, idx), [])
            if not hits:
                crops_no_cache_match += 1
                continue
            folder_hits += len(hits)
            target_indices.update(hits)
        if folder_hits == 0:
            folders_with_no_match.append(folder)

    print(f"Crop -> cache match summary:")
    print(f"  Crops examined:        {crops_total}")
    print(f"  Cache faces matched:   {len(target_indices)}")
    print(f"  Unparseable filenames: {crops_unparseable}")
    print(f"  No cache match:        {crops_no_cache_match}")
    if folders_with_no_match:
        print(f"  Folders with 0 cache matches: {len(folders_with_no_match)}")
        if not args.quiet:
            for f in folders_with_no_match[:10]:
                print(f"    {f.name}")
            if len(folders_with_no_match) > 10:
                print(f"    ... and {len(folders_with_no_match) - 10} more")
    print()

    if not target_indices and crops_total > 0:
        print("No cache entries matched any of these crops. Likely a filename-format")
        print("mismatch — investigate before running with --apply.")
        return 1

    if not args.quiet:
        print("Sample folders that would be marked junk (first 15):")
        for folder, crops in small[:15]:
            print(f"  {folder.name:<20} {len(crops)} crops")
        if len(small) > 15:
            print(f"  ... and {len(small) - 15} more")
        print()

    if args.apply:
        # 1. Update cache labels (defensive: skip any face that already has a real label)
        n_set_junk = 0
        n_already_labeled = 0
        for i in target_indices:
            current = cache.faces[i].label
            if current and current != JUNK_LABEL:
                n_already_labeled += 1
                continue
            cache.faces[i].label = JUNK_LABEL
            n_set_junk += 1

        bak = CACHE_FILE.with_name(CACHE_FILE.name + ".bak.mark_small_junk")
        print(f"Backing up cache to {bak} ...")
        bak.write_bytes(CACHE_FILE.read_bytes())
        print(f"Saving cache ...")
        save_cache(cache)
        print(f"  Set to '{JUNK_LABEL}': {n_set_junk}")
        if n_already_labeled:
            print(f"  Skipped (already had real label): {n_already_labeled}")
        print()

        # 2. Move crops into __junk__/ and remove now-empty person folders
        junk_dir = args.clusters_dir / JUNK_LABEL
        junk_dir.mkdir(parents=True, exist_ok=True)
        moved = 0
        move_failed = 0
        folders_removed = 0
        for folder, crops in small:
            for crop in crops:
                target = junk_dir / crop.name
                if target.exists():
                    stem = target.stem
                    suffix = "".join(target.suffixes)
                    k = 1
                    while True:
                        cand = junk_dir / f"{stem}__dup{k}{suffix}"
                        if not cand.exists():
                            target = cand
                            break
                        k += 1
                try:
                    crop.rename(target)
                    moved += 1
                except OSError as exc:
                    print(f"  failed to move {crop.name}: {exc}")
                    move_failed += 1
            # Try to remove the now-empty folder (and its montage if present)
            montage = args.clusters_dir / f"{folder.name}_montage.jpg"
            if montage.exists():
                try:
                    montage.unlink()
                except OSError:
                    pass
            try:
                folder.rmdir()
                folders_removed += 1
            except OSError:
                pass

        print(f"Crops moved into {junk_dir}: {moved}")
        if move_failed:
            print(f"Crop moves failed:               {move_failed}")
        print(f"Empty person folders removed:     {folders_removed}/{len(small)}")
        print()
        print("Next steps:")
        print(f"  1. Run `python cleanup_junk.py` (dry-run) to see which source images would move.")
        print(f"  2. Run `python cleanup_junk.py --apply` to actually move them to junk_to_review/.")
        print(f"  3. Review junk_to_review/ in Finder and Cmd+Delete when satisfied.")
        print(f"  4. Run `python validate_cache.py` to confirm cache health.")
    else:
        print("DRY-RUN — no changes written.")
        print(f"Would mark {len(target_indices)} cache faces as '{JUNK_LABEL}'.")
        print(f"Would move {crops_total} crops into {args.clusters_dir / JUNK_LABEL}/.")
        print(f"Would remove {len(small)} person_NNN folders (after they're empty).")
        print(f"Re-run with --apply to commit.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

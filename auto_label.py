#!/usr/bin/env python3
"""
auto_label.py — Propagate existing labels to unlabeled faces via embedding similarity.

For each labeled person, computes a centroid (mean L2-normalized embedding).
For each unlabeled face, finds the most similar centroid. If the similarity
exceeds --threshold AND beats the second-best centroid by at least --margin,
the face is auto-labeled.

Default mode is --dry-run: prints a report, no writes. Use --apply to commit
changes (the existing cache is backed up to cache.pkl.bak.autolabel first).

Usage:
    python auto_label.py                                # dry-run, defaults (cos>=0.55, margin>=0.07)
    python auto_label.py --threshold 0.50               # more aggressive
    python auto_label.py --apply                        # write changes
    python auto_label.py --apply --limit 100            # safe smoke test on first 100
    python auto_label.py --external-centroids celeb.pkl # also match against an external celeb DB
        (build that pickle via build_celeb_centroids.py against a folder of reference photos)
"""

import argparse
import pickle
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

# Re-use the cache schema (CacheState / CachedFace) from sort_photos.py so
# pickle.load can resolve the dataclasses.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import sort_photos  # noqa: F401
from sort_photos import CACHE_FILE, MODEL_NAME, load_cache, save_cache  # type: ignore

# face.py launches sort_photos.py via subprocess, so its dataclasses got
# pickled under the module name "__main__". When THIS script is __main__,
# pickle can't find them — alias the classes into __main__ before any
# load_cache() call so unpickling resolves correctly.
_main = sys.modules["__main__"]
for _name in (
    "CacheState", "CachedFace", "FaceRecord", "LabelingState",
):
    if hasattr(sort_photos, _name) and not hasattr(_main, _name):
        setattr(_main, _name, getattr(sort_photos, _name))


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def build_centroids(cache):
    """Return (names, centroids[N,D], counts[N]) for each labeled person."""
    by_label: dict[str, list[np.ndarray]] = defaultdict(list)
    for f in cache.faces:
        if f.label and f.embedding is not None and f.embedding.size > 0:
            by_label[f.label].append(np.asarray(f.embedding, dtype=np.float32))

    names = sorted(by_label.keys())
    if not names:
        return [], np.zeros((0, 0), dtype=np.float32), []

    centroids = np.stack([
        l2_normalize(np.mean(np.stack(by_label[n], axis=0), axis=0))
        for n in names
    ])
    counts = [len(by_label[n]) for n in names]
    return names, centroids, counts


def run_apply_loop(args: argparse.Namespace) -> int:
    if not args.apply:
        print("ERROR: --loop requires --apply, otherwise it would only repeat dry-runs.",
              file=sys.stderr)
        return 2

    script = Path(__file__).resolve()
    total_assigned = 0
    for i in range(1, args.max_loops + 1):
        cmd = [
            sys.executable, str(script),
            "--threshold", str(args.threshold),
            "--margin", str(args.margin),
            "--min-examples", str(args.min_examples),
            "--apply",
        ]
        if args.limit:
            cmd.extend(["--limit", str(args.limit)])
        if args.external_centroids:
            cmd.extend(["--external-centroids", str(args.external_centroids)])

        print()
        print(f"=== auto-label loop pass {i}/{args.max_loops} ===")
        proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
        print(proc.stdout, end="")
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        if proc.returncode != 0:
            return proc.returncode

        m = re.search(r"Assigned:\s+(\d+)", proc.stdout)
        assigned = int(m.group(1)) if m else 0
        total_assigned += assigned
        if assigned == 0:
            print(f"\nAuto-label loop complete: no new assignments on pass {i}.")
            break
    else:
        print(f"\nAuto-label loop stopped after --max-loops={args.max_loops}.")

    print(f"Total assigned across loop: {total_assigned}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--threshold", type=float, default=0.55,
                   help="Min cosine similarity to top centroid (default 0.55).")
    p.add_argument("--margin", type=float, default=0.07,
                   help="Top centroid must beat 2nd-best by at least this much (default 0.07).")
    p.add_argument("--min-examples", type=int, default=5,
                   help="Ignore people with fewer than this many labeled faces — centroids would be too noisy. Default 5.")
    p.add_argument("--apply", action="store_true",
                   help="Write changes to cache. Default is dry-run.")
    p.add_argument("--limit", type=int, default=0,
                   help="Process only the first N unlabeled faces (debug). 0=all.")
    p.add_argument("--external-centroids", type=Path, default=None,
                   help="Pickle file from build_celeb_centroids.py. Centroids in it are added to the matching pool. Names already in your cache take priority (cache version preferred).")
    p.add_argument("--loop", action="store_true",
                   help="With --apply, repeat auto-label passes until a pass assigns 0 faces.")
    p.add_argument("--max-loops", type=int, default=5,
                   help="Maximum passes for --loop (default 5).")
    args = p.parse_args()

    if args.loop:
        return run_apply_loop(args)

    if not CACHE_FILE.exists():
        print(f"ERROR: cache not found at {CACHE_FILE}", file=sys.stderr)
        return 1

    print(f"Loading cache from {CACHE_FILE} ...")
    cache = load_cache()

    total = len(cache.faces)
    labeled = [f for f in cache.faces if f.label]
    unlabeled = [f for f in cache.faces if not f.label]

    print(f"Total face entries:   {total}")
    print(f"Already labeled:      {len(labeled)}")
    print(f"Unlabeled:            {len(unlabeled)}")
    print()

    names, centroids, counts = build_centroids(cache)
    if len(names) == 0:
        print("No labeled faces found — nothing to learn from. Label some clusters first.")
        return 1

    print(f"Distinct labeled people: {len(names)}")
    print()
    print(f"{'Person':<32} {'Examples':>10}  Status")
    print(f"{'-'*32} {'-'*10}  {'-'*16}")
    skip_set: set[str] = set()
    for n, c in zip(names, counts):
        status = "OK"
        if c < args.min_examples:
            status = f"SKIP (n<{args.min_examples})"
            skip_set.add(n)
        print(f"{n:<32} {c:>10}  {status}")
    print()

    keep_idx = [i for i, n in enumerate(names) if n not in skip_set]
    keep_names = [names[i] for i in keep_idx]
    keep_C = centroids[keep_idx] if keep_idx else np.zeros((0, 0), dtype=np.float32)
    n_cache_centroids = len(keep_names)

    # External centroids (e.g., celeb DB built via build_celeb_centroids.py)
    n_external_added = 0
    if args.external_centroids:
        if not args.external_centroids.exists():
            print(f"ERROR: --external-centroids file not found: {args.external_centroids}",
                  file=sys.stderr)
            return 1
        with args.external_centroids.open("rb") as f:
            ext = pickle.load(f)
        ext_model = ext.get("model")
        if ext_model != MODEL_NAME:
            print(f"WARNING: external centroids built with model '{ext_model}', "
                  f"but your cache uses '{MODEL_NAME}'. Cosine similarities are only "
                  f"meaningful when both sides use the same model. Aborting.",
                  file=sys.stderr)
            return 1
        ext_names = list(ext["names"])
        ext_C = np.asarray(ext["centroids"], dtype=np.float32)
        ext_counts = list(ext["counts"])
        existing_set = set(keep_names)
        added_names: list[str] = []
        added_rows: list[np.ndarray] = []
        added_counts: list[int] = []
        for i, n in enumerate(ext_names):
            if n in existing_set:
                continue
            added_names.append(n)
            added_rows.append(ext_C[i])
            added_counts.append(ext_counts[i])
        if added_rows:
            new_block = np.stack(added_rows).astype(np.float32)
            if keep_C.size == 0:
                keep_C = new_block
            else:
                keep_C = np.vstack([keep_C, new_block])
            keep_names.extend(added_names)
            n_external_added = len(added_names)
        print(f"External centroids loaded: {len(ext_names)} total, "
              f"{n_external_added} added (skipped {len(ext_names) - n_external_added} "
              f"that already exist in your cache).")
        print()

    if not keep_names:
        print("No usable centroids (cache + external). Label some clusters or supply a "
              "valid --external-centroids file.")
        return 1

    if args.limit > 0:
        unlabeled = unlabeled[: args.limit]

    print(f"Scoring {len(unlabeled)} unlabeled faces against {len(keep_names)} centroids "
          f"({n_cache_centroids} from cache, {n_external_added} from external)...")
    print(f"Threshold: cos>={args.threshold:.3f}   Margin: >={args.margin:.3f}")
    print()

    external_set = set(keep_names[n_cache_centroids:])  # names that came from external

    auto_counts: dict[str, int] = defaultdict(int)
    n_assigned = 0
    n_low_sim = 0
    n_low_margin = 0
    n_no_emb = 0

    for f in unlabeled:
        if f.embedding is None or f.embedding.size == 0:
            n_no_emb += 1
            continue
        emb = l2_normalize(np.asarray(f.embedding, dtype=np.float32))
        sims = keep_C @ emb  # (P,)
        best_idx = int(np.argmax(sims))
        best = float(sims[best_idx])
        if len(sims) > 1:
            sims_other = np.delete(sims, best_idx)
            second = float(np.max(sims_other))
        else:
            second = -1.0

        if best < args.threshold:
            n_low_sim += 1
            continue
        if (best - second) < args.margin:
            n_low_margin += 1
            continue

        new_label = keep_names[best_idx]
        auto_counts[new_label] += 1
        n_assigned += 1
        if args.apply:
            f.label = new_label

    print("Results:")
    print(f"  Assigned:               {n_assigned}")
    print(f"  Skipped (low sim):      {n_low_sim}")
    print(f"  Skipped (low margin):   {n_low_margin}")
    print(f"  Skipped (no embedding): {n_no_emb}")
    print()
    if auto_counts:
        print("Per-person auto-label counts (top first):")
        for n in sorted(auto_counts, key=lambda x: -auto_counts[x]):
            tag = "  [external]" if n in external_set else ""
            print(f"  {n:<30} +{auto_counts[n]}{tag}")
        print()

    if args.apply:
        bak = CACHE_FILE.with_name(CACHE_FILE.name + ".bak.autolabel")
        print(f"Backing up cache to {bak} ...")
        bak.write_bytes(CACHE_FILE.read_bytes())
        print(f"Writing updated cache to {CACHE_FILE} ...")
        save_cache(cache)
        print("Done. Run `python validate_cache.py` to confirm cache is healthy.")
    else:
        print("DRY-RUN — no changes written. Re-run with --apply to commit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

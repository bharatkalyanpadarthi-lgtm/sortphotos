#!/usr/bin/env python3
"""
build_celeb_centroids.py — Compute per-actress face centroids from a reference dataset.

Input: a directory laid out one subfolder per person:

    reference_db/
        Rashmika_Mandanna/
            img1.jpg
            img2.jpg
            ...
        Pooja_Hegde/
            ...

Output: a pickle file containing per-actress centroids, computed with the SAME
InsightFace model that sort_photos.py uses (antelopev2). This file can be fed
to auto_label.py via --external-centroids to match your unlabeled faces against
actresses you've never manually labeled.

Folder name conventions:
  - Underscores in folder names become spaces in the saved label
    ("Pooja_Hegde" -> "Pooja Hegde")
  - Folders with fewer than --min-images usable photos are skipped

Usage:
    python build_celeb_centroids.py REF_DIR OUTPUT.pkl [--min-images 3] [--max-per-person 20]

Example:
    python build_celeb_centroids.py ~/Downloads/bollywood_db celeb_centroids.pkl
"""

import argparse
import gc
import pickle
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sort_photos  # noqa: F401
from sort_photos import (  # type: ignore
    CACHE_DIR,
    MODEL_NAME,
    MIN_DET_SCORE,
    REFERENCE_CENTROIDS_FILE,
    imread_unicode,
    _build_app,
)

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".heic", ".heif"}
DEFAULT_REF_DIR = Path.home() / "Pictures" / "Face References"
DEFAULT_CHUNK_SIZE = 25


def direct_crop_embedding(img, app) -> np.ndarray | None:
    """Fallback for already-cropped face images from this pipeline.

    FaceAnalysis detection can fail on tight 256x256 face crops because there
    is little surrounding context. For square-ish small crops, feed the image
    directly to the recognition model and L2-normalize the output.
    """
    if img is None:
        return None
    h, w = img.shape[:2]
    ratio = w / max(1, h)
    if max(w, h) > 512 or ratio < 0.75 or ratio > 1.33:
        return None
    rec = app.models.get("recognition")
    if rec is None:
        return None
    try:
        feat = rec.get_feat([img])
    except Exception:
        return None
    if feat is None or len(feat) == 0:
        return None
    return l2_normalize(np.asarray(feat[0], dtype=np.float32))


def best_face_embedding(img, app) -> tuple[np.ndarray | None, str]:
    """Highest-confidence face in img -> embedding and method."""
    if img is None:
        return None, "decode_failed"
    fallback = direct_crop_embedding(img, app)
    if fallback is not None:
        return fallback, "crop_direct"
    faces = app.get(img)
    if not faces:
        return None, "no_face"
    valid = [f for f in faces if float(getattr(f, "det_score", 0)) >= MIN_DET_SCORE]
    if not valid:
        return None, "low_score"
    best = max(valid, key=lambda f: float(f.det_score))
    emb = getattr(best, "normed_embedding", None)
    if emb is None:
        return None, "no_embedding"
    return np.asarray(emb, dtype=np.float32), "detected"


def l2_normalize(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-9:
        return v.astype(np.float32)
    return (v / n).astype(np.float32)


def reference_person_dirs(ref_dir: Path) -> list[Path]:
    return [
        d for d in sorted(ref_dir.iterdir())
        if d.is_dir() and not d.name.startswith("_")
    ]


def save_payload(output: Path,
                 names: list[str],
                 centroids: list[np.ndarray],
                 counts: list[int]) -> None:
    payload = {
        "version": 1,
        "model": MODEL_NAME,
        "names": names,
        "centroids": np.stack(centroids).astype(np.float32),
        "counts": counts,
    }

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(output)


def build_in_chunks(args: argparse.Namespace, actresses: list[Path]) -> int:
    """Run short-lived worker processes and merge their centroid outputs.

    ONNX Runtime can keep CPU memory arenas alive for the life of the process.
    Rebuilding a large reference set in chunks releases that memory between
    batches and prevents macOS from pausing Terminal for application memory.
    """
    chunk_size = max(1, int(args.chunk_size))
    parts_dir = args.output.parent / f".{args.output.stem}_parts"
    shutil.rmtree(parts_dir, ignore_errors=True)
    parts_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(actresses)} subfolders.")
    print(f"Building in chunks of {chunk_size} people to keep memory bounded.\n")

    part_paths: list[Path] = []
    script = Path(__file__).resolve()
    for start in range(0, len(actresses), chunk_size):
        end = min(start + chunk_size, len(actresses))
        part_path = parts_dir / f"part_{start:04d}_{end:04d}.pkl"
        part_paths.append(part_path)
        cmd = [
            sys.executable,
            str(script),
            str(args.ref_dir),
            str(part_path),
            "--min-images", str(args.min_images),
            "--max-per-person", str(args.max_per_person),
            "--det-size", str(args.det_size),
            "--chunk-size", "0",
            "--person-start", str(start),
            "--person-limit", str(chunk_size),
            "--allow-empty",
        ]
        print(f"Chunk {start // chunk_size + 1}: people {start + 1}-{end}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"\nChunk failed with exit code {result.returncode}: {start + 1}-{end}",
                  file=sys.stderr)
            return result.returncode
        print()

    names: list[str] = []
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    seen: set[str] = set()
    for part_path in part_paths:
        if not part_path.exists():
            continue
        with part_path.open("rb") as f:
            payload = pickle.load(f)
        part_names = list(payload.get("names", []))
        part_centroids = np.asarray(payload.get("centroids", []), dtype=np.float32)
        part_counts = list(payload.get("counts", []))
        if part_centroids.ndim != 2 or len(part_names) != len(part_centroids):
            print(f"Skipping malformed chunk output: {part_path}", file=sys.stderr)
            continue
        for i, name in enumerate(part_names):
            clean = str(name).strip()
            if not clean or clean in seen:
                continue
            seen.add(clean)
            names.append(clean)
            centroids.append(l2_normalize(part_centroids[i]))
            counts.append(int(part_counts[i]) if i < len(part_counts) else 0)

    if not names:
        print("\nNo actresses passed the threshold. Nothing saved.")
        return 1

    save_payload(args.output, names, centroids, counts)
    shutil.rmtree(parts_dir, ignore_errors=True)

    print(f"Saved {len(names)} centroids to {args.output}")
    print(f"Model: {MODEL_NAME}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("ref_dir", type=Path, nargs="?", default=DEFAULT_REF_DIR,
                   help=f"Reference dir with one subfolder per person. Default: {DEFAULT_REF_DIR}")
    p.add_argument("output", type=Path, nargs="?", default=REFERENCE_CENTROIDS_FILE,
                   help=f"Output pickle path. Default: {REFERENCE_CENTROIDS_FILE}")
    p.add_argument("--min-images", type=int, default=3,
                   help="Skip actresses with fewer than this many usable photos. Default 3.")
    p.add_argument("--max-per-person", type=int, default=20,
                   help="Cap photos used per actress. Default 20 for higher-quality reference matching.")
    p.add_argument("--det-size", type=int, default=1024,
                   help="InsightFace detection size. Default 1024 for quality.")
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE,
                   help=f"People per worker process. Default {DEFAULT_CHUNK_SIZE}; use 0 to disable chunking.")
    p.add_argument("--person-start", type=int, default=0,
                   help=argparse.SUPPRESS)
    p.add_argument("--person-limit", type=int, default=0,
                   help=argparse.SUPPRESS)
    p.add_argument("--allow-empty", action="store_true",
                   help=argparse.SUPPRESS)
    args = p.parse_args()

    if not args.ref_dir.is_dir():
        args.ref_dir.mkdir(parents=True, exist_ok=True)
        print(f"Created reference folder: {args.ref_dir}")
        print("Add photos like:")
        print(f"  {args.ref_dir}/Rashmika Mandanna/img1.jpg")
        print(f"  {args.ref_dir}/Pooja Hegde/img1.jpg")
        print("Then rerun: python face.py refs")
        return 1

    actresses = reference_person_dirs(args.ref_dir)
    if not actresses:
        print(f"No subfolders found in {args.ref_dir}", file=sys.stderr)
        return 1

    if args.person_start or args.person_limit:
        start = max(0, int(args.person_start))
        limit = max(0, int(args.person_limit))
        actresses = actresses[start:start + limit if limit else None]

    if args.chunk_size > 0 and not args.person_start and not args.person_limit and len(actresses) > args.chunk_size:
        return build_in_chunks(args, actresses)

    sort_photos.DET_SIZE = (max(320, int(args.det_size)), max(320, int(args.det_size)))
    print(f"Initializing InsightFace ({MODEL_NAME}, det_size={sort_photos.DET_SIZE[0]})...")
    app = _build_app()
    print()

    print(f"Found {len(actresses)} subfolders.\n")

    by_name: dict[str, list[np.ndarray]] = defaultdict(list)
    no_face_count: dict[str, int] = defaultdict(int)
    method_count: dict[str, Counter[str]] = defaultdict(Counter)

    for ad in actresses:
        name = ad.name.replace("_", " ").strip()
        if not ad.is_dir():
            print(f"  {name:<32} skipped; folder disappeared")
            continue
        try:
            images = sorted(
                p for p in ad.iterdir()
                if p.is_file() and p.suffix.lower() in IMAGE_EXTS
            )
        except FileNotFoundError:
            print(f"  {name:<32} skipped; folder disappeared")
            continue
        images = images[: args.max_per_person]
        if not images:
            print(f"  {name:<32} no images")
            continue
        for img_path in images:
            try:
                img = imread_unicode(img_path)
                emb, method = best_face_embedding(img, app)
                if emb is not None:
                    by_name[name].append(emb)
                    method_count[name][method] += 1
                else:
                    no_face_count[name] += 1
            except Exception as exc:  # noqa: BLE001
                print(f"    skipped {img_path.name}: {exc}")
            finally:
                img = None
                emb = None
        methods = " ".join(f"{k}={v}" for k, v in sorted(method_count[name].items()))
        if methods:
            methods = f" ({methods})"
        print(f"  {name:<32} usable={len(by_name[name]):<4} no_face={no_face_count[name]}{methods}")
        gc.collect()

    names: list[str] = []
    centroids: list[np.ndarray] = []
    counts: list[int] = []
    skipped_low_n: list[tuple[str, int]] = []
    for name in sorted(by_name.keys()):
        embs = by_name[name]
        if len(embs) < args.min_images:
            skipped_low_n.append((name, len(embs)))
            continue
        centroid = l2_normalize(np.mean(np.stack(embs, axis=0), axis=0))
        names.append(name)
        centroids.append(centroid)
        counts.append(len(embs))

    if skipped_low_n:
        print()
        print(f"Skipped (n < {args.min_images}):")
        for n, c in skipped_low_n:
            print(f"  {n:<32} {c} usable")

    if not names:
        print("\nNo actresses passed the threshold. Nothing saved.")
        return 0 if args.allow_empty else 1

    save_payload(args.output, names, centroids, counts)

    print()
    print(f"Saved {len(names)} centroids to {args.output}")
    print(f"Model: {MODEL_NAME}")
    print()
    print("Use with auto_label.py:")
    print(f"  python auto_label.py --external-centroids {args.output} --threshold 0.45")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

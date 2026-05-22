#!/usr/bin/env python3
"""
Inspect and rehydrate the face detector cache.

The detector cache speeds up repeat scans only when it points at files that
still exist. Since the daily workflow moves source files out of To Process, this
tool can rebuild a useful cache from the current organized person folders.

Run:
  python cache_tools.py status
  python cache_tools.py rehydrate
  python cache_tools.py rehydrate --apply
"""

from __future__ import annotations

import argparse
import pickle
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sort_photos  # noqa: E402

for _name in ("CacheState", "CachedFace", "FaceRecord", "LabelingState", "IdentityDB"):
    if hasattr(sort_photos, _name):
        setattr(sys.modules["__main__"], _name, getattr(sort_photos, _name))


DEFAULT_PEOPLE = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
IMAGE_EXTS = sort_photos.IMAGE_EXTS


def human_size(n: int) -> str:
    units = ["B", "KB", "MB", "GB", "TB"]
    value = float(n)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} TB"


def person_dirs(people_dir: Path, person: str | None = None) -> list[Path]:
    if not people_dir.exists():
        return []
    dirs = [
        p for p in people_dir.iterdir()
        if p.is_dir() and not p.name.startswith("_") and p.name != "all"
    ]
    if person:
        wanted = person.casefold()
        dirs = [p for p in dirs if p.name.casefold() == wanted]
    return sorted(dirs, key=lambda p: p.name.casefold())


def person_folder_images(people_dir: Path, person: str | None = None) -> list[tuple[Path, str]]:
    out: list[tuple[Path, str]] = []
    seen: set[tuple[int, int] | Path] = set()
    for person_dir in person_dirs(people_dir, person):
        photos_dir = person_dir / "photos"
        roots = [photos_dir] if photos_dir.exists() else [person_dir]
        excluded = {"_smart_albums", "all", "review", "best", "quality", "duplicates"}
        for root in roots:
            for image in sort_photos.iter_images(root, excluded_dir_names=excluded):
                try:
                    stat = image.stat()
                    key: tuple[int, int] | Path = (stat.st_dev, stat.st_ino)
                except OSError:
                    try:
                        key = image.resolve()
                    except OSError:
                        key = image
                if key in seen:
                    continue
                seen.add(key)
                out.append((image, person_dir.name))
    return sorted(out, key=lambda item: (item[1].casefold(), str(item[0]).casefold()))


def cache_summary() -> dict[str, int | str]:
    cache = sort_photos.load_cache()
    paths = set(cache.file_signatures)
    face_paths = {face.src_str for face in cache.faces if face.src_str}
    existing_paths = {p for p in paths if Path(p).exists()}
    missing_paths = paths - existing_paths
    labeled = sum(1 for face in cache.faces if face.label)
    cache_size = sort_photos.CACHE_FILE.stat().st_size if sort_photos.CACHE_FILE.exists() else 0
    return {
        "cache_file": str(sort_photos.CACHE_FILE),
        "cache_size": cache_size,
        "file_signatures": len(paths),
        "existing_files": len(existing_paths),
        "missing_files": len(missing_paths),
        "face_entries": len(cache.faces),
        "face_paths": len(face_paths),
        "labeled_faces": labeled,
        "config_ok": int(cache.config_fingerprint == sort_photos.config_fingerprint()),
    }


def labeling_state_summary() -> dict[str, int | str]:
    state = sort_photos.load_labeling_state()
    if state is None:
        return {
            "present": 0, "faces": 0, "unique_sources": 0,
            "existing_sources": 0, "missing_sources": 0,
            "input_dir": "", "output_dir": "",
        }
    sources = {face.src_str for face in state.faces if face.src_str}
    existing = {src for src in sources if Path(src).exists()}
    return {
        "present": 1,
        "faces": len(state.faces),
        "unique_sources": len(sources),
        "existing_sources": len(existing),
        "missing_sources": len(sources) - len(existing),
        "input_dir": state.input_dir,
        "output_dir": state.output_dir,
    }


def print_status(people_dir: Path, person: str | None = None) -> int:
    cache = cache_summary()
    state = labeling_state_summary()
    candidates = person_folder_images(people_dir, person)
    candidate_size = 0
    for path, _person in candidates:
        try:
            candidate_size += path.stat().st_size
        except OSError:
            pass

    print("Face Cache Status")
    print("=" * 60)
    print(f"Cache file:              {cache['cache_file']}")
    print(f"Cache size:              {human_size(int(cache['cache_size']))}")
    print(f"Cache files:             {cache['file_signatures']}")
    print(f"Cache existing/missing:  {cache['existing_files']} / {cache['missing_files']}")
    print(f"Cache faces/labeled:     {cache['face_entries']} / {cache['labeled_faces']}")
    print(f"Config matches code:     {'yes' if cache['config_ok'] else 'no'}")
    print()
    print("Saved Labeling State")
    print(f"Present:                 {'yes' if state['present'] else 'no'}")
    print(f"Faces:                   {state['faces']}")
    print(f"Unique sources:          {state['unique_sources']}")
    print(f"Existing/missing source: {state['existing_sources']} / {state['missing_sources']}")
    if state["input_dir"]:
        print(f"Input dir:               {state['input_dir']}")
    if state["output_dir"]:
        print(f"Output dir:              {state['output_dir']}")
    print()
    print("Rehydrate Candidates")
    print(f"People folder:           {people_dir}")
    print(f"Person filter:           {person or 'all'}")
    print(f"Images from people:      {len(candidates)} ({human_size(candidate_size)})")
    print()
    if state["present"] and state["existing_sources"] == 0 and state["faces"]:
        print("Note: saved labeling-state source paths are gone, so rehydrate should use")
        print("the current person folders rather than stale To Process paths.")
    return 0


def backup_cache() -> Path | None:
    if not sort_photos.CACHE_FILE.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup = sort_photos.CACHE_FILE.with_name(f"{sort_photos.CACHE_FILE.name}.bak.rehydrate_{stamp}")
    shutil.copy2(sort_photos.CACHE_FILE, backup)
    return backup


def save_cache_with_backup(cache: sort_photos.CacheState, backup: Path | None) -> Path | None:
    if backup is None:
        backup = backup_cache()
    sort_photos.save_cache(cache)
    return backup


def rehydrate(people_dir: Path, person: str | None, apply: bool,
              replace: bool, max_images: int | None, batch_size: int) -> int:
    candidates = person_folder_images(people_dir, person)
    all_candidate_paths = {str(path) for path, _person in candidates}
    if max_images is not None:
        candidates = candidates[:max(0, max_images)]

    print("Face Cache Rehydrate")
    print("=" * 60)
    print(f"People folder:      {people_dir}")
    print(f"Person filter:      {person or 'all'}")
    print(f"Images to inspect:  {len(candidates)}")
    print(f"Mode:               {'APPLY' if apply else 'DRY-RUN'}")
    print(f"Cache strategy:     {'replace' if replace else 'merge existing/resume'}")
    print(f"Batch size:         {batch_size}")
    print()
    if not candidates:
        print("No candidate images found.")
        return 0
    if not apply:
        print("DRY-RUN - no model loaded and no cache written. Re-run with --apply.")
        return 0

    old_cache = sort_photos.load_cache()
    new_cache = sort_photos.CacheState(
        version=sort_photos.CACHE_VERSION,
        config_fingerprint=sort_photos.config_fingerprint(),
    )
    candidate_paths = {str(path) for path, _person in candidates}
    retain_paths = candidate_paths if max_images is None else all_candidate_paths

    kept_existing_faces = 0
    kept_existing_files = 0
    if not replace and old_cache.config_fingerprint == sort_photos.config_fingerprint():
        for src, sig in old_cache.file_signatures.items():
            if src not in retain_paths:
                continue
            if not Path(src).exists():
                continue
            new_cache.file_signatures[src] = sig
            kept_existing_files += 1
        for face in old_cache.faces:
            if face.src_str not in retain_paths:
                continue
            if not Path(face.src_str).exists():
                continue
            new_cache.faces.append(face)
            kept_existing_faces += 1

    remaining = [
        (path, label) for path, label in candidates
        if str(path) not in new_cache.file_signatures
    ]
    if len(remaining) != len(candidates):
        print(f"Already cached candidate files: {len(candidates) - len(remaining)}")
        print(f"Remaining candidate files:      {len(remaining)}")
        print()

    detected_images = 0
    no_face_images = 0
    read_errors = 0
    backup: Path | None = None
    batch_size = max(1, int(batch_size))
    n_batches = (len(remaining) + batch_size - 1) // batch_size
    tmp_dir = Path(tempfile.mkdtemp(prefix="cache_rehydrate_"))
    script_path = Path(sort_photos.__file__).resolve()

    try:
        for batch_index in range(n_batches):
            start = batch_index * batch_size
            end = min(start + batch_size, len(remaining))
            batch = remaining[start:end]
            image_paths = [path for path, _label in batch]
            labels_by_path = {str(path): label for path, label in batch}
            job_path = tmp_dir / f"job_{batch_index:04d}.pkl"
            out_path = tmp_dir / f"out_{batch_index:04d}.pkl"
            with job_path.open("wb") as f:
                pickle.dump({
                    "input_paths": [str(path) for path in image_paths],
                    "output_path": str(out_path),
                    "det_size": sort_photos.DET_SIZE[0],
                }, f, protocol=pickle.HIGHEST_PROTOCOL)

            print(f"[{batch_index + 1}/{n_batches}] Detecting images {start + 1}-{end}...")
            proc = subprocess.run(
                [sys.executable, str(script_path), "--detect-batch", str(job_path)],
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            if proc.returncode != 0:
                print(f"ERROR: detection worker failed for batch {batch_index + 1} (exit {proc.returncode}).")
                if proc.stdout:
                    print(proc.stdout[-4000:])
                print("Cache has been saved through the last completed batch.")
                return proc.returncode
            if not out_path.exists():
                print(f"ERROR: detection worker produced no output for batch {batch_index + 1}.")
                print("Cache has been saved through the last completed batch.")
                return 2

            with out_path.open("rb") as f:
                batch_faces: list[sort_photos.CachedFace] = pickle.load(f)

            faces_by_path: dict[str, list[sort_photos.CachedFace]] = {}
            for face in batch_faces:
                faces_by_path.setdefault(face.src_str, []).append(face)

            batch_detected = 0
            batch_no_face = 0
            for image in image_paths:
                try:
                    new_cache.file_signatures[str(image)] = sort_photos.file_signature(image)
                except OSError:
                    read_errors += 1
                    continue
                faces = faces_by_path.get(str(image), [])
                if not faces:
                    no_face_images += 1
                    batch_no_face += 1
                    continue
                best = max(faces, key=lambda face: face.quality)
                best.label = labels_by_path[str(image)]
                new_cache.faces.append(best)
                detected_images += 1
                batch_detected += 1

            backup = save_cache_with_backup(new_cache, backup)
            print(
                f"    saved cache: files={len(new_cache.file_signatures)} "
                f"faces={len(new_cache.faces)} batch_faces={batch_detected} "
                f"batch_no_face={batch_no_face}"
            )

            try:
                job_path.unlink()
                out_path.unlink()
            except OSError:
                pass
    finally:
        try:
            shutil.rmtree(tmp_dir)
        except OSError:
            pass

    print()
    print("Rehydrate complete.")
    if backup:
        print(f"Backup:                 {backup}")
    print(f"Cache written:           {sort_photos.CACHE_FILE}")
    print(f"Existing files kept:     {kept_existing_files}")
    print(f"Existing faces kept:     {kept_existing_faces}")
    print(f"Candidate files stored:  {len(candidate_paths & set(new_cache.file_signatures))}")
    print(f"Detected/labeled faces:  {detected_images}")
    print(f"No-face images cached:   {no_face_images}")
    print(f"Read/signature errors:   {read_errors}")
    print(f"Total cache files:       {len(new_cache.file_signatures)}")
    print(f"Total cache faces:       {len(new_cache.faces)}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    status = sub.add_parser("status", help="Show detector cache usefulness.")
    status.add_argument("--people-dir", type=Path, default=DEFAULT_PEOPLE)
    status.add_argument("--person", default=None)

    rebuild = sub.add_parser("rehydrate", help="Rebuild cache from current person folders.")
    rebuild.add_argument("--people-dir", type=Path, default=DEFAULT_PEOPLE)
    rebuild.add_argument("--person", default=None)
    rebuild.add_argument("--apply", action="store_true")
    rebuild.add_argument("--replace", action="store_true",
                         help="Replace existing cache instead of merging existing live entries.")
    rebuild.add_argument("--max-images", type=int, default=None,
                         help="Debug/safety limit for the number of images to inspect.")
    rebuild.add_argument("--batch-size", type=int, default=50,
                         help="Images per detection subprocess. Default 50.")

    args = parser.parse_args()
    people_dir = args.people_dir.expanduser().resolve()
    if args.command == "status":
        return print_status(people_dir, args.person)
    if args.command == "rehydrate":
        return rehydrate(
            people_dir, args.person, args.apply, args.replace,
            args.max_images, args.batch_size,
        )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

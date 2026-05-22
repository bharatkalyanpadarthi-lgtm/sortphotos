#!/usr/bin/env python3
"""
Rebuild face cache and identity DB by relinking old cached embeddings to
current photos_by_person files.

This avoids slow re-detection after a restore/rename. It only uses exact
(mtime, size) file-signature matches and never modifies image files.
"""

from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import numpy as np

import sort_photos

for _name in ("CacheState", "CachedFace", "FaceRecord", "LabelingState", "IdentityDB"):
    if hasattr(sort_photos, _name):
        setattr(sys.modules["__main__"], _name, getattr(sort_photos, _name))


CACHE_DIR = Path.home() / ".face_sort_cache"
DEFAULT_OLD_CACHE = CACHE_DIR / "cache.pkl.bak.mark_small_junk"
PEOPLE_ROOT = Path.home() / "Pictures" / "sorted_all_pictures" / "photos_by_person"
RULES_FILE = Path(__file__).with_name("person_folder_rules.json")
IMAGE_EXTS = sort_photos.IMAGE_EXTS


def load_rules() -> tuple[dict[str, str], set[str]]:
    if not RULES_FILE.exists():
        return {}, set()
    data = json.loads(RULES_FILE.read_text(encoding="utf-8"))
    aliases: dict[str, str] = {}
    for dest, sources in data.get("merge", {}).items():
        aliases[dest.casefold()] = dest
        for src in sources:
            aliases[src.casefold()] = dest
    for src, dest in data.get("rename", {}).items():
        aliases[src.casefold()] = dest
    removed = {x.casefold() for x in data.get("remove", [])}
    return aliases, removed


def canonical_label(label: str | None, aliases: dict[str, str], removed: set[str]) -> str | None:
    if not label or label == "__junk__":
        return None
    key = label.casefold()
    if key in removed:
        return None
    mapped = aliases.get(key, label)
    if mapped.casefold() in removed:
        return None
    return mapped


def person_photo_files(people_root: Path):
    for person_dir in sorted([p for p in people_root.iterdir() if p.is_dir()], key=lambda p: p.name.casefold()):
        for subdir_name in ("photos", "photos/nude", "photos_nude"):
            subdir = person_dir / subdir_name
            if not subdir.exists():
                continue
            for path in subdir.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    yield person_dir.name, path


def file_sig(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return int(st.st_mtime), int(st.st_size)


def backup(path: Path, suffix: str) -> Path | None:
    if not path.exists():
        return None
    stamp = time.strftime("%Y%m%d_%H%M%S")
    backup_path = path.with_name(f"{path.name}.bak.{suffix}_{stamp}")
    shutil.copy2(path, backup_path)
    return backup_path


def save_pickle(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as fh:
        pickle.dump(obj, fh, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(path)


def l2norm(v: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(v, axis=1, keepdims=True)
    norm[norm == 0] = 1.0
    return v / norm


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-cache", type=Path, default=DEFAULT_OLD_CACHE)
    parser.add_argument("--people-root", type=Path, default=PEOPLE_ROOT)
    parser.add_argument("--identity-max-faces", type=int, default=80)
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    old_cache_path = args.old_cache.expanduser()
    people_root = args.people_root.expanduser()
    if not old_cache_path.exists():
        print(f"ERROR: old cache not found: {old_cache_path}")
        return 1
    if not people_root.exists():
        print(f"ERROR: people folder not found: {people_root}")
        return 1

    aliases, removed = load_rules()
    print(f"Loading old cache: {old_cache_path}")
    with old_cache_path.open("rb") as fh:
        old_cache: sort_photos.CacheState = pickle.load(fh)

    faces_by_sig: dict[tuple[int, int], list[sort_photos.CachedFace]] = defaultdict(list)
    for face in old_cache.faces:
        sig_raw = old_cache.file_signatures.get(face.src_str)
        if not sig_raw:
            continue
        faces_by_sig[(int(sig_raw[0]), int(sig_raw[1]))].append(face)

    new_cache = sort_photos.CacheState(
        version=sort_photos.CACHE_VERSION,
        config_fingerprint=sort_photos.config_fingerprint(),
    )
    identity_vectors: dict[str, list[np.ndarray]] = defaultdict(list)
    seen_face_keys: set[tuple[str, int, str]] = set()
    counts = Counter()

    print(f"Scanning current people photos: {people_root}")
    for person, path in person_photo_files(people_root):
        sig = file_sig(path)
        if sig is None:
            counts["stat_error"] += 1
            continue
        matches = faces_by_sig.get(sig, [])
        if not matches:
            counts["no_old_match"] += 1
            continue
        added_for_file = 0
        for old_face in matches:
            label = canonical_label(old_face.label, aliases, removed)
            if label is None or label.casefold() != person.casefold():
                continue
            key = (str(path), int(old_face.face_index), person)
            if key in seen_face_keys:
                continue
            seen_face_keys.add(key)
            new_face = replace(old_face, src_str=str(path), label=person)
            new_cache.faces.append(new_face)
            identity_vectors[person].append(np.asarray(new_face.embedding, dtype=np.float32))
            added_for_file += 1
        if added_for_file:
            new_cache.file_signatures[str(path)] = sig
            counts["matched_files"] += 1
            counts["matched_faces"] += added_for_file
        else:
            counts["signature_but_no_person_face"] += 1

    identity_db = sort_photos.IdentityDB(config_fingerprint=sort_photos.config_fingerprint())
    max_faces = max(0, int(args.identity_max_faces))
    for person, vectors in sorted(identity_vectors.items(), key=lambda x: x[0].casefold()):
        if not vectors:
            continue
        selected = vectors[:max_faces] if max_faces else vectors
        centroid = np.mean(np.stack(selected), axis=0)
        identity_db.identities[person] = l2norm(centroid[None, :])[0]
        identity_db.source_counts[person] = len(selected)

    print()
    print("Fast cache relink")
    print(f"  Current matched files: {counts['matched_files']:,}")
    print(f"  Current matched faces: {counts['matched_faces']:,}")
    print(f"  Identity DB people:    {len(identity_db.identities):,}")
    print(f"  No old-cache match:    {counts['no_old_match']:,}")
    print(f"  Sig but no person face:{counts['signature_but_no_person_face']:,}")
    print(f"  Mode:                  {'APPLY' if args.apply else 'DRY-RUN'}")

    if not args.apply:
        print()
        print("DRY-RUN only. Re-run with --apply to write cache and identity DB.")
        return 0

    cache_backup = backup(sort_photos.CACHE_FILE, "relink")
    identity_backup = backup(sort_photos.IDENTITY_DB_FILE, "relink")
    save_pickle(sort_photos.CACHE_FILE, new_cache)
    save_pickle(sort_photos.IDENTITY_DB_FILE, identity_db)
    print()
    if cache_backup:
        print(f"Backed up old cache:      {cache_backup}")
    if identity_backup:
        print(f"Backed up old identity DB:{identity_backup}")
    print(f"Wrote cache:              {sort_photos.CACHE_FILE}")
    print(f"Wrote identity DB:        {sort_photos.IDENTITY_DB_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

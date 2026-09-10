#!/usr/bin/env python3
"""
Rebuild face cache and identity DB by relinking old cached embeddings to
current photos_by_person files.

This avoids slow re-detection after a restore/rename. File signatures narrow
candidates, but face transfers require content hashes. Image files never change.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import replace
from pathlib import Path

import pipeline_paths

import numpy as np

import operation_ledger
import sort_photos
import content_identity

for _name in ("CacheState", "CachedFace", "FaceRecord", "LabelingState", "IdentityDB"):
    if hasattr(sort_photos, _name):
        setattr(sys.modules["__main__"], _name, getattr(sort_photos, _name))


CACHE_DIR = Path.home() / ".face_sort_cache"
DEFAULT_OLD_CACHE = CACHE_DIR / "cache.pkl.bak.mark_small_junk"
PEOPLE_ROOT = pipeline_paths.PEOPLE_ROOT
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
    for person_dir in sorted(
        [
            p for p in people_root.iterdir()
            if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")
        ],
        key=lambda p: p.name.casefold(),
    ):
        # "photos" is scanned recursively and already includes photos/nude.
        # Listing photos/nude separately double-counts nude originals.
        for subdir_name in ("photos", "photos_nude"):
            subdir = person_dir / subdir_name
            if not subdir.exists():
                continue
            for path in subdir.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                    yield person_dir.name, path


def file_sig(path: Path) -> tuple[int, int] | None:
    try:
        return sort_photos.file_signature(path)
    except OSError:
        return None


def signatures_equal(left: tuple[float, int] | tuple[int, int],
                     right: tuple[float, int] | tuple[int, int]) -> bool:
    return abs(float(left[0]) - float(right[0])) < 0.000001 and int(left[1]) == int(right[1])


def content_verified_faces(path: Path, faces: list) -> list:
    candidates = [face for face in faces if getattr(face, "content_sha256", "")]
    if not candidates:
        return []
    try:
        digest = content_identity.content_sha256(path)
    except OSError:
        return []
    return [face for face in candidates if face.content_sha256 == digest]


def validate_relocated_faces(cache, relocated: set[str]) -> None:
    grouped = defaultdict(list)
    for face in cache.faces:
        if face.src_str in relocated:
            grouped[face.src_str].append(face)
    invalid = {path for path, faces in grouped.items()
               if len(content_verified_faces(Path(path), faces)) != len(faces)}
    for path in invalid:
        cache.file_signatures.pop(path, None)
    cache.faces = [face for face in cache.faces if face.src_str not in invalid]
    if invalid:
        print(f"  Deferred for fresh detection (unverified content): {len(invalid):,}")


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


def load_rename_plan(path: Path) -> dict[str, str]:
    mapping: dict[str, str] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source = str(row.get("source") or "").strip()
            destination = str(row.get("destination") or "").strip()
            if source and destination:
                mapping[source] = destination
    return mapping


def relink_from_rename_plan(old_cache: sort_photos.CacheState,
                            rename_plan: Path,
                            *,
                            apply: bool) -> int:
    mapping = load_rename_plan(rename_plan)
    new_cache = sort_photos.CacheState(
        version=sort_photos.CACHE_VERSION,
        config_fingerprint=sort_photos.config_fingerprint(),
    )
    counts = Counter()

    for old_src, old_sig in old_cache.file_signatures.items():
        new_src = mapping.get(str(old_src))
        if not new_src:
            old_path = Path(old_src)
            if not old_path.exists():
                counts["unchanged_missing"] += 1
                continue
            current_sig = sort_photos.file_signature(old_path)
            if not signatures_equal(old_sig, current_sig):
                counts["unchanged_signature_mismatch"] += 1
                continue
            new_cache.file_signatures[old_src] = current_sig
            counts["preserved_files"] += 1
            continue
        new_path = Path(new_src)
        if not new_path.exists():
            counts["destination_missing"] += 1
            continue
        new_sig = sort_photos.file_signature(new_path)
        if not signatures_equal(old_sig, new_sig):
            counts["signature_mismatch"] += 1
            continue
        new_cache.file_signatures[new_src] = new_sig
        counts["mapped_files"] += 1

    for face in old_cache.faces:
        new_src = mapping.get(str(face.src_str))
        if new_src and new_src in new_cache.file_signatures:
            new_cache.faces.append(replace(face, src_str=new_src))
            counts["mapped_faces"] += 1
            continue
        if not new_src and face.src_str in new_cache.file_signatures:
            new_cache.faces.append(face)
            counts["preserved_faces"] += 1
            continue
        if new_src or face.src_str not in new_cache.file_signatures:
            counts["unmapped_faces"] += 1

    print()
    validate_relocated_faces(new_cache, set(mapping.values()))
    print("Cache relink from rename plan")
    print(f"  Rename plan:          {rename_plan}")
    print(f"  Plan mappings:        {len(mapping):,}")
    print(f"  Mapped cache files:   {counts['mapped_files']:,}")
    print(f"  Mapped faces:         {counts['mapped_faces']:,}")
    print(f"  Preserved files:      {counts['preserved_files']:,}")
    print(f"  Preserved faces:      {counts['preserved_faces']:,}")
    print(f"  Destination missing:  {counts['destination_missing']:,}")
    print(f"  Signature mismatch:   {counts['signature_mismatch']:,}")
    print(f"  Unchanged missing:    {counts['unchanged_missing']:,}")
    print(f"  Unchanged mismatch:   {counts['unchanged_signature_mismatch']:,}")
    print(f"  Unmapped faces:       {counts['unmapped_faces']:,}")
    print(f"  Mode:                 {'APPLY' if apply else 'DRY-RUN'}")

    if not apply:
        print()
        print("DRY-RUN only. Re-run with --apply to write cache.")
        return 0
    if (
        counts["destination_missing"]
        or counts["signature_mismatch"]
        or counts["unchanged_missing"]
        or counts["unchanged_signature_mismatch"]
        or counts["unmapped_faces"]
    ):
        print("ERROR: rename-plan relink has missing destinations or signature mismatches.", file=sys.stderr)
        return 2

    cache_backup = backup(sort_photos.CACHE_FILE, "rename_relink")
    save_pickle(sort_photos.CACHE_FILE, new_cache)
    print()
    if cache_backup:
        print(f"Backed up old cache: {cache_backup}")
    print(f"Wrote cache:         {sort_photos.CACHE_FILE}")
    return 0


def load_move_ledger_mapping(sorted_root: Path,
                             operation: str,
                             run_id: str | None = None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for event in operation_ledger.iter_events(sorted_root):
        if event.get("status") != "moved":
            continue
        if operation and event.get("operation") != operation:
            continue
        if run_id and event.get("run_id") != run_id:
            continue
        source = str(event.get("source_path") or "")
        dest = str(event.get("dest_path") or "")
        if source and dest:
            mapping[source] = dest
    return mapping


def relink_from_move_mapping(old_cache: sort_photos.CacheState,
                             mapping: dict[str, str],
                             *,
                             apply: bool,
                             label: str) -> int:
    new_cache = sort_photos.CacheState(
        version=sort_photos.CACHE_VERSION,
        config_fingerprint=sort_photos.config_fingerprint(),
    )
    counts = Counter()

    for old_src, old_sig in old_cache.file_signatures.items():
        new_src = mapping.get(str(old_src), str(old_src))
        new_path = Path(new_src)
        if not new_path.exists():
            counts["missing"] += 1
            continue
        new_sig = sort_photos.file_signature(new_path)
        if not signatures_equal(old_sig, new_sig):
            counts["signature_mismatch"] += 1
            continue
        new_cache.file_signatures[new_src] = new_sig
        if new_src == str(old_src):
            counts["kept_files"] += 1
        else:
            counts["mapped_files"] += 1

    for face in old_cache.faces:
        new_src = mapping.get(str(face.src_str), str(face.src_str))
        if new_src not in new_cache.file_signatures:
            counts["dropped_faces"] += 1
            continue
        if new_src == face.src_str:
            new_cache.faces.append(face)
            counts["kept_faces"] += 1
        else:
            new_cache.faces.append(replace(face, src_str=new_src))
            counts["mapped_faces"] += 1

    print()
    validate_relocated_faces(new_cache, {destination for source, destination in mapping.items() if source != destination})
    print(f"Cache relink from {label}")
    print(f"  Move mappings:        {len(mapping):,}")
    print(f"  Kept cache files:     {counts['kept_files']:,}")
    print(f"  Mapped cache files:   {counts['mapped_files']:,}")
    print(f"  Kept faces:           {counts['kept_faces']:,}")
    print(f"  Mapped faces:         {counts['mapped_faces']:,}")
    print(f"  Missing files:        {counts['missing']:,}")
    print(f"  Signature mismatch:   {counts['signature_mismatch']:,}")
    print(f"  Dropped faces:        {counts['dropped_faces']:,}")
    print(f"  Mode:                 {'APPLY' if apply else 'DRY-RUN'}")

    if not apply:
        print()
        print("DRY-RUN only. Re-run with --apply to write cache.")
        return 0
    if counts["missing"] or counts["signature_mismatch"]:
        print("ERROR: ledger relink has missing files or signature mismatches.", file=sys.stderr)
        return 2

    cache_backup = backup(sort_photos.CACHE_FILE, label)
    save_pickle(sort_photos.CACHE_FILE, new_cache)
    print()
    if cache_backup:
        print(f"Backed up old cache: {cache_backup}")
    print(f"Wrote cache:         {sort_photos.CACHE_FILE}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-cache", type=Path, default=None,
                        help="Cache used as the relink source. Rename-plan and ledger modes default to "
                             "the live cache; full recovery defaults to the historical recovery cache.")
    parser.add_argument("--people-root", type=Path, default=PEOPLE_ROOT)
    parser.add_argument("--rename-plan-csv", type=Path, default=None,
                        help="Exact rename plan CSV from rename_person_folder_files.py. "
                             "When set, relink cache paths directly from source to destination.")
    parser.add_argument("--ledger-run-id", default=None,
                        help="Relink cache paths using moved events from one operation ledger run id.")
    parser.add_argument("--ledger-operation", default="place_nudity_inside_person_folders.move_to_nude",
                        help="Ledger operation to use with --ledger-run-id. "
                             "Default: place_nudity_inside_person_folders.move_to_nude.")
    parser.add_argument("--sorted-root", type=Path, default=pipeline_paths.SORTED_ROOT,
                        help="Sorted root for operation ledgers.")
    parser.add_argument("--identity-max-faces", type=int, default=80)
    parser.add_argument("--include-unmatched-signatures", action="store_true",
                        help="Deprecated compatibility option; unmatched files "
                             "always remain pending for fresh detection. File "
                             "signatures alone cannot prove a no-face result.")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    if args.old_cache is not None:
        old_cache_path = args.old_cache.expanduser()
    elif args.rename_plan_csv is not None or args.ledger_run_id:
        old_cache_path = sort_photos.CACHE_FILE
    else:
        old_cache_path = DEFAULT_OLD_CACHE
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

    if args.rename_plan_csv is not None:
        return relink_from_rename_plan(
            old_cache,
            args.rename_plan_csv.expanduser(),
            apply=bool(args.apply),
        )

    if args.ledger_run_id:
        mapping = load_move_ledger_mapping(
            args.sorted_root.expanduser(),
            args.ledger_operation,
            args.ledger_run_id,
        )
        return relink_from_move_mapping(
            old_cache,
            mapping,
            apply=bool(args.apply),
            label="ledger_relink",
        )

    faces_by_sig: dict[tuple[int, int], list[sort_photos.CachedFace]] = defaultdict(list)
    for face in old_cache.faces:
        sig_raw = old_cache.file_signatures.get(face.src_str)
        if not sig_raw:
            continue
        faces_by_sig[(float(sig_raw[0]), int(sig_raw[1]))].append(face)

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
        matches = content_verified_faces(path, faces_by_sig.get(sig, []))
        if not matches:
            counts["no_old_match"] += 1
            # A signature cannot prove that a different file has no face.
            counts["skipped_unmatched"] += 1
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
        candidates = vectors[:max_faces] if max_faces else vectors
        samples = [
            sort_photos.identity_profiles.ReferenceSample(
                source=f"cache:{person}:{index}", embedding=vector, quality=1.0)
            for index, vector in enumerate(candidates)
        ]
        selected = sort_photos.identity_profiles.select_diverse_samples(
            samples, limit=sort_photos.IDENTITY_MAX_PROTOTYPES_PER_PERSON)
        centroid = sort_photos.identity_profiles.weighted_centroid(selected)
        prototypes = [
            sort_photos.identity_profiles.normalize_vector(sample.embedding)
            for sample in selected
        ]
        consensus, strict = sort_photos.identity_profiles.calibrated_thresholds(
            samples,
            centroid,
            prototypes,
            maximum_consensus_distance=sort_photos.AUTO_PERSON_MATCH_DIST,
        )
        identity_db.identities[person] = centroid
        identity_db.prototypes[person] = prototypes
        identity_db.prototype_sources[person] = [sample.source for sample in selected]
        identity_db.match_thresholds[person] = consensus
        identity_db.strict_thresholds[person] = strict
        identity_db.source_counts[person] = len(candidates)

    print()
    print("Fast cache relink")
    print(f"  Current matched files: {counts['matched_files']:,}")
    print(f"  Current matched faces: {counts['matched_faces']:,}")
    print(f"  Identity DB people:    {len(identity_db.identities):,}")
    print(f"  No old-cache match:    {counts['no_old_match']:,}")
    print(f"  Matched no-face files: {counts['matched_no_face_files']:,}")
    print(f"  Skipped unmatched:     {counts['skipped_unmatched']:,}")
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

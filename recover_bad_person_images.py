#!/usr/bin/env python3
"""
Recover quarantined bad person images from valid backup/source files.

The recovery uses the old face cache to map each bad placeholder file back to:
  - the person folder it belonged to
  - the old source file signature
  - the old full-image perceptual hash

It then searches available source/backup folders for decodable images that match
by exact signature or by perceptual hash. Recovered files are hardlinked/copied
back into photos_by_person/<person>/photos. Bad quarantined files are never
deleted by this script.

Default is dry-run. Use --apply to restore matched files.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import pickle
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

import sort_photos

for _name in ("CacheState", "CachedFace", "FaceRecord", "LabelingState"):
    if hasattr(sort_photos, _name):
        setattr(sys.modules["__main__"], _name, getattr(sort_photos, _name))


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff", ".heic", ".heif", ".gif"}
CACHE_DIR = Path.home() / ".face_sort_cache"
DEFAULT_OLD_CACHE = CACHE_DIR / "cache.pkl.bak.mark_small_junk"
SORTED = Path.home() / "Pictures" / "sorted_all_pictures"
PEOPLE_ROOT = SORTED / "photos_by_person"
BAD_DIR = SORTED / "_source_review" / "ready_to_delete" / "bad_person_images"
REPORT_DIR = SORTED / "_source_review" / "recovery_reports"
RULES_FILE = Path(__file__).with_name("person_folder_rules.json")
DEFAULT_RECOVERED_BACKUP = Path("/Volumes/ssd 1/Photos Recovered/Photos & Videos  Backup/photo_source_review_backup")
DEFAULT_EXTERNAL_BACKUP = Path("/Volumes/Photos & Videos  Backup/photo_source_review_backup")
DEFAULT_CANDIDATE_CACHE = CACHE_DIR / "bad_person_recovery_candidate_phashes.json"


@dataclass
class Target:
    bad_path: Path
    person: str
    signature: tuple[int, int]
    phashes: set[int] = field(default_factory=set)


@dataclass
class Candidate:
    root_name: str
    path: Path
    signature: tuple[int, int]
    phash: int


class BKNode:
    def __init__(self, value: int) -> None:
        self.value = value
        self.children: dict[int, "BKNode"] = {}


class BKTree:
    def __init__(self, values: set[int]) -> None:
        self.root: BKNode | None = None
        for value in values:
            self.add(value)

    def add(self, value: int) -> None:
        if self.root is None:
            self.root = BKNode(value)
            return
        node = self.root
        while True:
            dist = hamming_int(value, node.value)
            child = node.children.get(dist)
            if child is None:
                node.children[dist] = BKNode(value)
                return
            node = child

    def query(self, value: int, threshold: int) -> list[tuple[int, int]]:
        if self.root is None:
            return []
        found: list[tuple[int, int]] = []
        stack = [self.root]
        while stack:
            node = stack.pop()
            dist = hamming_int(value, node.value)
            if dist <= threshold:
                found.append((node.value, dist))
            low = dist - threshold
            high = dist + threshold
            for child_dist, child in node.children.items():
                if low <= child_dist <= high:
                    stack.append(child)
        return found


@contextlib.contextmanager
def suppress_native_stderr(enabled: bool = True):
    if not enabled:
        yield
        return
    try:
        fd = sys.stderr.fileno()
    except Exception:
        yield
        return
    saved_fd = os.dup(fd)
    try:
        with open(os.devnull, "w", encoding="utf-8") as devnull:
            os.dup2(devnull.fileno(), fd)
            yield
    finally:
        os.dup2(saved_fd, fd)
        os.close(saved_fd)


def hamming_int(a: int, b: int) -> int:
    return int((a ^ b).bit_count())


def file_signature(path: Path) -> tuple[int, int] | None:
    try:
        st = path.stat()
    except OSError:
        return None
    return int(st.st_mtime), int(st.st_size)


def decode_image(path: Path) -> np.ndarray | None:
    try:
        header = path.read_bytes()[:32]
    except OSError:
        return None
    if not looks_like_image_container(header):
        return None
    with suppress_native_stderr():
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            img = cv2.imdecode(data, cv2.IMREAD_COLOR)
            if img is not None and img.size > 0:
                return img
        except Exception:
            pass
        try:
            from PIL import Image, ImageFile

            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(path) as im:
                im.load()
                arr = np.array(im.convert("RGB"))
            return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        except Exception:
            return None


def looks_like_image_container(header: bytes) -> bool:
    if header.startswith(b"\xff\xd8"):
        return True
    if header.startswith(b"\x89PNG\r\n\x1a\n"):
        return True
    if header.startswith((b"GIF87a", b"GIF89a", b"BM", b"II*\x00", b"MM\x00*")):
        return True
    if len(header) >= 12 and header[:4] == b"RIFF" and header[8:12] == b"WEBP":
        return True
    if len(header) >= 12 and header[4:8] == b"ftyp" and header[8:12] in {
        b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"avif",
    }:
        return True
    return False


def phash64(img: np.ndarray) -> int:
    return sort_photos.phash_to_int(sort_photos.perceptual_hash(img))


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


def current_people(people_root: Path) -> dict[str, str]:
    return {
        p.name.casefold(): p.name
        for p in people_root.iterdir()
        if p.is_dir() and not p.name.startswith("_") and not p.name.startswith(".")
    }


def canonical_label(label: str, aliases: dict[str, str], removed: set[str], people: dict[str, str]) -> str | None:
    key = label.casefold()
    if key in removed:
        return None
    mapped = aliases.get(key, label)
    if mapped.casefold() in removed:
        return None
    return people.get(mapped.casefold(), mapped)


def iter_images(root: Path, skip_roots: list[Path] | None = None):
    skips = [p.resolve() for p in (skip_roots or []) if p.exists()]
    if not root.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root):
        base = Path(dirpath)
        resolved = base.resolve()
        if any(resolved == s or s in resolved.parents for s in skips):
            dirnames[:] = []
            continue
        dirnames[:] = [d for d in dirnames if not d.startswith(".") and not d.endswith(".photoslibrary")]
        for filename in filenames:
            path = base / filename
            if path.is_file() and path.suffix.lower() in IMAGE_EXTS:
                yield path


def load_cache(path: Path) -> sort_photos.CacheState:
    with path.open("rb") as fh:
        return pickle.load(fh)


def build_targets(bad_dir: Path,
                  old_cache: sort_photos.CacheState,
                  people_root: Path) -> tuple[list[Target], dict[tuple[int, int], list[Target]], dict[int, list[Target]]]:
    aliases, removed = load_rules()
    people = current_people(people_root)
    faces_by_sig: dict[tuple[int, int], list[sort_photos.CachedFace]] = defaultdict(list)
    for face in old_cache.faces:
        sig_raw = old_cache.file_signatures.get(face.src_str)
        if not sig_raw:
            continue
        faces_by_sig[(int(sig_raw[0]), int(sig_raw[1]))].append(face)

    targets: list[Target] = []
    by_sig: dict[tuple[int, int], list[Target]] = defaultdict(list)
    by_phash: dict[int, list[Target]] = defaultdict(list)
    for bad_path in iter_images(bad_dir):
        try:
            person_hint = bad_path.relative_to(bad_dir).parts[0]
        except (ValueError, IndexError):
            continue
        person = canonical_label(person_hint, aliases, removed, people)
        if not person:
            continue
        sig = file_signature(bad_path)
        if sig is None:
            continue
        phashes: set[int] = set()
        for face in faces_by_sig.get(sig, []):
            label = canonical_label(str(face.label or ""), aliases, removed, people)
            if label and label.casefold() != person.casefold():
                continue
            bits = getattr(face, "image_phash", None)
            if bits is not None and getattr(bits, "size", 0):
                phashes.add(sort_photos.phash_to_int(bits))
        if not phashes:
            for face in faces_by_sig.get(sig, []):
                bits = getattr(face, "image_phash", None)
                if bits is not None and getattr(bits, "size", 0):
                    phashes.add(sort_photos.phash_to_int(bits))
        target = Target(bad_path=bad_path, person=person, signature=sig, phashes=phashes)
        targets.append(target)
        by_sig[sig].append(target)
        for phash in phashes:
            by_phash[phash].append(target)
    return targets, by_sig, by_phash


def cache_key(path: Path, sig: tuple[int, int]) -> str:
    return f"{path}|{sig[0]}:{sig[1]}"


def load_candidate_cache(path: Path) -> dict:
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("version") == 1 and isinstance(data.get("items"), dict):
            return data
    except Exception:
        pass
    return {"version": 1, "items": {}}


def save_candidate_cache(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(data, fh)
    tmp.replace(path)


def default_roots() -> list[tuple[str, Path]]:
    roots = [
        ("to_process", Path.home() / "Pictures" / "To Process"),
        ("source_review", SORTED / "_source_review"),
        ("external_backup", DEFAULT_EXTERNAL_BACKUP),
        ("recovered_backup", DEFAULT_RECOVERED_BACKUP),
    ]
    return [(name, path) for name, path in roots if path.exists()]


def scan_candidates(roots: list[tuple[str, Path]],
                    target_sigs: set[tuple[int, int]],
                    phash_tree: BKTree,
                    threshold: int,
                    bad_dir: Path,
                    cache_path: Path,
                    use_cache: bool,
                    quiet: bool) -> tuple[dict[str, tuple[str, Candidate, int, str]], Counter]:
    candidate_cache = load_candidate_cache(cache_path) if use_cache else {"version": 1, "items": {}}
    items = candidate_cache["items"]
    stats = Counter()
    matches: dict[str, tuple[str, Candidate, int, str]] = {}
    skipped_roots = [bad_dir]

    for root_name, root in roots:
        if not quiet:
            print(f"Scanning {root_name}: {root}", flush=True)
        for path in iter_images(root, skip_roots=skipped_roots):
            sig = file_signature(path)
            if sig is None:
                stats["stat_error"] += 1
                continue
            stats["seen"] += 1
            need_hash = sig in target_sigs or threshold >= 0
            if not need_hash:
                continue
            key = cache_key(path, sig)
            cached = items.get(key)
            phash: int | None = None
            valid = False
            if cached:
                stats["cache_hits"] += 1
                valid = bool(cached.get("valid"))
                phash = int(cached["phash"], 16) if valid and cached.get("phash") else None
            else:
                stats["cache_misses"] += 1
                img = decode_image(path)
                valid = img is not None
                if valid:
                    phash = phash64(img)
                items[key] = {
                    "valid": valid,
                    "phash": "" if phash is None else f"{phash:016x}",
                }
            if not valid or phash is None:
                stats["bad_candidates"] += 1
                continue

            candidate = Candidate(root_name=root_name, path=path, signature=sig, phash=phash)
            if sig in target_sigs:
                stats["exact_signature_candidates"] += 1
                for target in SIG_TARGETS.get(sig, []):
                    consider_match(matches, str(target.bad_path), ("exact_signature", candidate, 0, root_name))
                    stats["match_exact_signature"] += 1
            for ph, dist in phash_tree.query(phash, threshold):
                for target in PHASH_TARGETS.get(ph, []):
                    if sig == target.signature or ph in target.phashes:
                        consider_match(matches, str(target.bad_path), ("phash", candidate, dist, root_name))
                        stats["match_phash"] += 1

            if not quiet and stats["seen"] % 5000 == 0:
                print(f"  scanned={stats['seen']:,} valid={stats['seen'] - stats['bad_candidates']:,} "
                      f"matched_targets={len(matches):,}", flush=True)

    if use_cache:
        save_candidate_cache(cache_path, candidate_cache)
    return matches, stats


def match_rank(match: tuple[str, Candidate, int, str]) -> tuple:
    kind, candidate, dist, root_name = match
    root_rank = {
        "to_process": 0,
        "source_review": 1,
        "external_backup": 2,
        "recovered_backup": 3,
    }
    kind_rank = {"exact_signature": 0, "phash": 1}
    return (
        kind_rank.get(kind, 9),
        dist,
        root_rank.get(root_name, 99),
        len(str(candidate.path)),
        str(candidate.path).casefold(),
    )


def consider_match(matches: dict[str, tuple[str, Candidate, int, str]],
                   key: str,
                   candidate: tuple[str, Candidate, int, str]) -> None:
    current = matches.get(key)
    if current is None or match_rank(candidate) < match_rank(current):
        matches[key] = candidate


def sha256_file(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def existing_person_hashes(person_dir: Path) -> set[str]:
    hashes: set[str] = set()
    for sub in ("photos", "photos/nude", "photos_nude", "review/nudity_possible", "review/uncertain_nudity"):
        folder = person_dir / sub
        if not folder.exists():
            continue
        for path in iter_images(folder):
            img = decode_image(path)
            if img is None:
                continue
            try:
                hashes.add(sha256_file(path))
            except OSError:
                continue
    return hashes


def unique_dest(dest: Path) -> Path:
    if not dest.exists():
        return dest
    stem = dest.stem
    suffix = "".join(dest.suffixes)
    parent = dest.parent
    i = 2
    while True:
        candidate = parent / f"{stem}__{i}{suffix}"
        if not candidate.exists():
            return candidate
        i += 1


def restore_file(src: Path, dest: Path, apply: bool) -> str:
    if not apply:
        return "dry_run"
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dest)
        return "hardlinked"
    except OSError:
        shutil.copy2(src, dest)
        return "copied"


def choose_match(options: list[tuple[str, Candidate, int, str]]) -> tuple[str, Candidate, int, str] | None:
    if not options:
        return None
    rank = {
        "to_process": 0,
        "source_review": 1,
        "external_backup": 2,
        "recovered_backup": 3,
    }
    kind_rank = {"exact_signature": 0, "phash": 1}
    return sorted(
        options,
        key=lambda x: (
            kind_rank.get(x[0], 9),
            x[2],
            rank.get(x[3], 99),
            len(str(x[1].path)),
            str(x[1].path).casefold(),
        ),
    )[0]


PHASH_TARGETS: dict[int, list[Target]] = {}
SIG_TARGETS: dict[tuple[int, int], list[Target]] = {}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bad-dir", type=Path, default=BAD_DIR)
    parser.add_argument("--people-root", type=Path, default=PEOPLE_ROOT)
    parser.add_argument("--old-cache", type=Path, default=DEFAULT_OLD_CACHE)
    parser.add_argument("--root", action="append", type=Path, default=[],
                        help="Extra candidate root. Can be passed more than once.")
    parser.add_argument("--phash-threshold", type=int, default=0,
                        help="Perceptual hash distance allowed. Default 0 is safest.")
    parser.add_argument("--candidate-cache", type=Path, default=DEFAULT_CANDIDATE_CACHE)
    parser.add_argument("--no-candidate-cache", action="store_true")
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    bad_dir = args.bad_dir.expanduser().resolve()
    people_root = args.people_root.expanduser().resolve()
    old_cache_path = args.old_cache.expanduser()
    if not bad_dir.exists():
        print(f"ERROR: bad image folder not found: {bad_dir}")
        return 1
    if not people_root.exists():
        print(f"ERROR: people folder not found: {people_root}")
        return 1
    if not old_cache_path.exists():
        print(f"ERROR: old cache not found: {old_cache_path}")
        return 1

    old_cache = load_cache(old_cache_path)
    targets, by_sig, by_phash = build_targets(bad_dir, old_cache, people_root)
    global PHASH_TARGETS, SIG_TARGETS
    PHASH_TARGETS = by_phash
    SIG_TARGETS = by_sig
    target_phashes = set(by_phash)
    phash_tree = BKTree(target_phashes)

    roots = default_roots()
    for i, root in enumerate(args.root, start=1):
        path = root.expanduser()
        if path.exists():
            roots.insert(0, (f"extra_root_{i}", path))

    print(f"Bad image folder:       {bad_dir}")
    print(f"People folder:          {people_root}")
    print(f"Targets:                {len(targets):,}")
    print(f"Target signatures:      {len(by_sig):,}")
    print(f"Target phashes:         {len(target_phashes):,}")
    print(f"Candidate roots:        {len(roots)}")
    for name, root in roots:
        print(f"  {name:<18} {root}")
    print(f"pHash threshold:        {args.phash_threshold}")
    print(f"Mode:                   {'APPLY' if args.apply else 'DRY-RUN'}")
    print()

    matches, scan_stats = scan_candidates(
        roots=roots,
        target_sigs=set(by_sig),
        phash_tree=phash_tree,
        threshold=max(0, args.phash_threshold),
        bad_dir=bad_dir,
        cache_path=args.candidate_cache.expanduser(),
        use_cache=not args.no_candidate_cache,
        quiet=args.quiet,
    )

    existing_hashes_by_person: dict[str, set[str]] = defaultdict(set)
    rows: list[dict[str, str]] = []
    totals = Counter()
    restored_person_phash: set[tuple[str, int]] = set()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report = REPORT_DIR / f"recover_bad_person_images_{time.strftime('%Y%m%d_%H%M%S')}.csv"

    for target in targets:
        choice = matches.get(str(target.bad_path))
        row = {
            "person": target.person,
            "bad_file": str(target.bad_path),
            "status": "",
            "match_kind": "",
            "distance": "",
            "source_root": "",
            "source": "",
            "dest": "",
        }
        if choice is None:
            totals["missing"] += 1
            row["status"] = "missing"
            rows.append(row)
            continue
        kind, candidate, dist, root_name = choice
        person_dir = people_root / target.person
        person_hashes = existing_hashes_by_person[target.person]
        if not person_hashes:
            person_hashes.update(existing_person_hashes(person_dir))
        try:
            digest = sha256_file(candidate.path)
        except OSError as e:
            totals["restore_error"] += 1
            row["status"] = f"error:{type(e).__name__}:{e}"
            rows.append(row)
            continue
        if digest in person_hashes:
            totals["already_present"] += 1
            row["status"] = "already_present"
            row["match_kind"] = kind
            row["distance"] = str(dist)
            row["source_root"] = root_name
            row["source"] = str(candidate.path)
            rows.append(row)
            continue
        dedupe_keys = [(target.person, ph) for ph in target.phashes] or [(target.person, candidate.phash)]
        if any(key in restored_person_phash for key in dedupe_keys):
            totals["duplicate_target_skipped"] += 1
            row["status"] = "duplicate_target_skipped"
            row["match_kind"] = kind
            row["distance"] = str(dist)
            row["source_root"] = root_name
            row["source"] = str(candidate.path)
            rows.append(row)
            continue
        dest = unique_dest(person_dir / "photos" / f"recovered_bad__{candidate.path.name}")
        try:
            status = restore_file(candidate.path, dest, args.apply)
        except Exception as e:  # noqa: BLE001
            totals["restore_error"] += 1
            row["status"] = f"error:{type(e).__name__}:{e}"
            row["source"] = str(candidate.path)
            row["dest"] = str(dest)
            rows.append(row)
            continue
        totals[status] += 1
        person_hashes.add(digest)
        for key in dedupe_keys:
            restored_person_phash.add(key)
        row["status"] = status
        row["match_kind"] = kind
        row["distance"] = str(dist)
        row["source_root"] = root_name
        row["source"] = str(candidate.path)
        row["dest"] = str(dest)
        rows.append(row)

    with report.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["person", "bad_file", "status", "match_kind", "distance",
                        "source_root", "source", "dest"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print("---- Recovery Results ----")
    print(f"Candidates scanned:     {scan_stats['seen']:,}")
    print(f"Bad candidates skipped: {scan_stats['bad_candidates']:,}")
    print(f"Matched targets:        {len(matches):,}")
    print(f"Restored/restorable:    {totals['hardlinked'] + totals['copied'] + totals['dry_run']:,}")
    print(f"Already present:        {totals['already_present']:,}")
    print(f"Duplicate skipped:      {totals['duplicate_target_skipped']:,}")
    print(f"Missing:                {totals['missing']:,}")
    print(f"Restore errors:         {totals['restore_error']:,}")
    print(f"Report:                 {report}")
    if not args.apply:
        print()
        print("DRY-RUN only. Re-run with --apply to restore matched files.")
    return 1 if totals["restore_error"] else 0


if __name__ == "__main__":
    raise SystemExit(main())

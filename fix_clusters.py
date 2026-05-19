"""
fix_clusters.py — Quick fix-up tool for sort_photos.py output.

Use this AFTER sort_photos.py whenever you spot mistakes:
  * Merge two clusters that should be one person (e.g. person_001 + Syamala)
  * Rename a cluster (e.g. person_003 → "Anushka")
  * Label clusters you missed during the original interactive labeling

Updates everything consistently:
  - face_clusters/ folders + montages
  - photos_by_person/ folders (including nested _blurred and _duplicates)
  - _clusters.csv manifest
  - the persistent cache at ~/.face_sort_cache/cache.pkl

Run:
    python fix_clusters.py
    python fix_clusters.py /path/to/sorted

Inside the tool:
    list                       show all clusters with face / photo counts
    open N                     open cluster #N's montage in Preview
    show N                     open cluster #N's photos folder in Finder
    merge N M                  merge cluster #N into cluster #M
    rename N <new_name>        rename cluster #N (auto-merges if name exists)
    q                          save changes and quit
"""

from __future__ import annotations

import argparse
import csv
import pickle
import shutil
import subprocess
import sys
from pathlib import Path

DEFAULT_OUTPUT = Path.home() / "Pictures" / "sorted_all_pictures"
CACHE_FILE     = Path.home() / ".face_sort_cache" / "cache.pkl"

INVALID_NAME_CHARS = '/\\:*?"<>|'
PHOTO_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".webp", ".tif", ".tiff", ".bmp"}


# ---------- helpers ----------

def sanitize(name: str) -> str:
    name = name.strip()
    for ch in INVALID_NAME_CHARS:
        name = name.replace(ch, "_")
    return name


def list_clusters(output_dir: Path) -> list[tuple[str, int, int]]:
    """(name, face_count, original_photo_count) sorted by photo count desc.

    With the nested layout, photos_by_person/<name>/ already contains
    everything for that person (sharp + _blurred + _duplicates), so a
    single rglob picks them all up.
    """
    clusters_dir = output_dir / "face_clusters"
    photos_dir   = output_dir / "photos_by_person"
    if not clusters_dir.exists():
        print(f"Error: {clusters_dir} not found.")
        sys.exit(1)
    out = []
    for d in sorted(clusters_dir.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        face_count = sum(1 for f in d.iterdir()
                         if f.is_file() and f.suffix.lower() == ".jpg")
        orig_count = 0
        person_folder = photos_dir / name
        if person_folder.exists():
            for f in person_folder.rglob("*"):
                if f.is_file() and f.suffix.lower() in PHOTO_EXTS:
                    orig_count += 1
        out.append((name, face_count, orig_count))
    out.sort(key=lambda x: -x[2])
    return out


def open_montage(name: str, clusters_dir: Path) -> bool:
    montage = clusters_dir / f"{name}_montage.jpg"
    if montage.exists():
        subprocess.run(["open", str(montage)], check=False)
        return True
    return False


def open_folder(path: Path) -> bool:
    if path.exists():
        subprocess.run(["open", str(path)], check=False)
        return True
    return False


def merge_folders(src: Path, dst: Path) -> int:
    """Move all files (recursively) from src into dst."""
    if not src.exists():
        return 0
    dst.mkdir(parents=True, exist_ok=True)
    moved = 0
    for f in list(src.iterdir()):
        if f.is_dir():
            moved += merge_folders(f, dst / f.name)
        elif f.is_file():
            target = dst / f.name
            i = 2
            while target.exists():
                target = dst / f"{f.stem}__{i}{f.suffix}"
                i += 1
            shutil.move(str(f), str(target))
            moved += 1
    try:
        src.rmdir()
    except OSError:
        pass
    return moved


def merge_cluster(source_name: str, dest_name: str, output_dir: Path) -> str:
    """Merge all data for source_name into dest_name.
    With nested layout, photos_by_person/<name>/ is one tree containing
    everything (sharp + _blurred + _duplicates) — one merge_folders call
    moves it all.
    """
    clusters_dir = output_dir / "face_clusters"
    photos_dir   = output_dir / "photos_by_person"

    parts = []
    n = merge_folders(clusters_dir / source_name, clusters_dir / dest_name)
    if n:
        parts.append(f"face_clusters: {n}")
    n = merge_folders(photos_dir / source_name, photos_dir / dest_name)
    if n:
        parts.append(f"photos: {n}")
    src_montage = clusters_dir / f"{source_name}_montage.jpg"
    if src_montage.exists():
        src_montage.unlink()
        parts.append("montage removed")
    return ", ".join(parts) if parts else "(nothing to move)"


def rename_cluster(old_name: str, new_name: str, output_dir: Path) -> str:
    """Rename folders + montage. With nested layout, renaming the person
    folder automatically renames its nested _blurred and _duplicates."""
    clusters_dir = output_dir / "face_clusters"
    photos_dir   = output_dir / "photos_by_person"

    parts = []
    pairs = [
        (clusters_dir / old_name, clusters_dir / new_name),
        (photos_dir   / old_name, photos_dir   / new_name),
        (clusters_dir / f"{old_name}_montage.jpg",
         clusters_dir / f"{new_name}_montage.jpg"),
    ]
    for src, dst in pairs:
        if src.exists() and not dst.exists():
            src.rename(dst)
            parts.append(src.name)
    return ", ".join(parts) if parts else "(nothing to rename)"


def chain_update(name_changes: dict[str, str], old: str, new: str) -> None:
    name_changes[old] = new
    for k, v in list(name_changes.items()):
        if v == old:
            name_changes[k] = new


def update_csv(csv_path: Path, name_changes: dict[str, str]) -> int:
    if not csv_path.exists() or not name_changes:
        return 0
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames or []
        for row in reader:
            old = row.get("person")
            if old in name_changes:
                row["person"] = name_changes[old]
            rows.append(row)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def update_cache(name_changes: dict[str, str]) -> int:
    if not CACHE_FILE.exists() or not name_changes:
        return 0
    try:
        with CACHE_FILE.open("rb") as f:
            cache = pickle.load(f)
    except Exception as e:  # noqa: BLE001
        print(f"  Warning: could not load cache: {e}")
        return 0
    changed = 0
    for face in cache.faces:
        if face.label in name_changes:
            new_label = name_changes[face.label]
            face.label = new_label if (
                new_label and not new_label.startswith("person_") and new_label != "unknown"
            ) else None
            changed += 1
    if changed:
        tmp = CACHE_FILE.with_suffix(".pkl.tmp")
        with tmp.open("wb") as f:
            pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(CACHE_FILE)
    return changed


# ---------- main loop ----------

def print_clusters(clusters):
    print(f"\n{'#':>3}  {'name':<25}  {'faces':>5}  {'photos':>6}")
    print("-" * 50)
    for i, (name, fc, oc) in enumerate(clusters, 1):
        print(f"{i:>3}  {name:<25}  {fc:>5}  {oc:>6}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output", nargs="?", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()

    output_dir = Path(args.output).expanduser()
    if not output_dir.exists():
        print(f"Output folder not found: {output_dir}")
        return 1

    clusters_dir = output_dir / "face_clusters"
    photos_dir   = output_dir / "photos_by_person"
    csv_path     = output_dir / "_clusters.csv"

    name_changes: dict[str, str] = {}

    print("Cluster fix-up tool.")
    print("Tip: open Finder to face_clusters/ and photos_by_person/ in another window.")
    print("Commands:")
    print("  list                        show all clusters")
    print("  open N                      open cluster #N's face montage")
    print("  show N                      open cluster #N's photos folder")
    print("  merge N M                   merge cluster #N into cluster #M")
    print("  rename N <new_name>         rename cluster #N (auto-merges if name exists)")
    print("  q                           save and quit\n")

    print_clusters(list_clusters(output_dir))

    while True:
        try:
            line = input("\n> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        cmd_parts = line.split(maxsplit=2)
        cmd = cmd_parts[0].lower()

        if cmd in ("q", "quit", "exit"):
            break

        if cmd == "list":
            print_clusters(list_clusters(output_dir))
            continue

        if cmd in ("open", "show"):
            if len(cmd_parts) < 2:
                print("  Usage: open N   /   show N")
                continue
            try:
                n = int(cmd_parts[1]) - 1
                clusters = list_clusters(output_dir)
                name = clusters[n][0]
                if cmd == "open":
                    if open_montage(name, clusters_dir):
                        print(f"  Opened montage for '{name}'.")
                    else:
                        print(f"  No montage for '{name}'.")
                else:
                    if open_folder(photos_dir / name):
                        print(f"  Opened photos folder for '{name}'.")
                    else:
                        print(f"  No photos folder for '{name}'.")
            except (ValueError, IndexError):
                print("  Invalid cluster number.")
            continue

        if cmd == "merge":
            if len(cmd_parts) < 2:
                print("  Usage: merge N M")
                continue
            nums = (cmd_parts[1] + (" " + cmd_parts[2] if len(cmd_parts) > 2 else "")).split()
            if len(nums) != 2:
                print("  Usage: merge N M")
                continue
            try:
                clusters = list_clusters(output_dir)
                n1 = int(nums[0]) - 1
                n2 = int(nums[1]) - 1
                if n1 == n2:
                    print("  Same cluster, nothing to merge.")
                    continue
                src_name = clusters[n1][0]
                dst_name = clusters[n2][0]
                open_montage(src_name, clusters_dir)
                open_montage(dst_name, clusters_dir)
                ans = input(f"  Confirm: merge '{src_name}' INTO '{dst_name}'? (y/n): ").strip().lower()
                if ans != "y":
                    print("  Cancelled.")
                    continue
                result = merge_cluster(src_name, dst_name, output_dir)
                chain_update(name_changes, src_name, dst_name)
                print(f"  → merged ({result})")
            except (ValueError, IndexError):
                print("  Invalid cluster number.")
            continue

        if cmd == "rename":
            if len(cmd_parts) < 3:
                print("  Usage: rename N <new_name>")
                continue
            try:
                clusters = list_clusters(output_dir)
                n = int(cmd_parts[1]) - 1
                old_name = clusters[n][0]
                new_name = sanitize(cmd_parts[2])
                if not new_name or old_name == new_name:
                    print("  Nothing to do.")
                    continue
                existing = {c[0] for c in clusters}
                if new_name in existing:
                    open_montage(old_name, clusters_dir)
                    open_montage(new_name, clusters_dir)
                    ans = input(f"  '{new_name}' exists. Merge '{old_name}' INTO it? (y/n): ").strip().lower()
                    if ans != "y":
                        print("  Cancelled.")
                        continue
                    result = merge_cluster(old_name, new_name, output_dir)
                    chain_update(name_changes, old_name, new_name)
                    print(f"  → merged ({result})")
                else:
                    result = rename_cluster(old_name, new_name, output_dir)
                    chain_update(name_changes, old_name, new_name)
                    print(f"  → renamed ({result})")
            except (ValueError, IndexError):
                print("  Invalid cluster number.")
            continue

        print(f"  Unknown command: {cmd}")

    if name_changes:
        print(f"\nApplying {len(name_changes)} change(s)…")
        n_csv = update_csv(csv_path, name_changes)
        n_cache = update_cache(name_changes)
        print(f"  CSV: updated {n_csv} rows.")
        print(f"  Cache: updated {n_cache} face label(s).")
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

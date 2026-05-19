"""
Interactive labeling: walks through each clustered person, opens the montage
in Preview, asks you for a name, then renames the folder + montage and updates
the CSV manifest so organize_originals.py picks up the new names.

Run AFTER cluster_faces_v2.py.
Run BEFORE organize_originals.py (or re-run organize_originals after this).
"""

from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from pathlib import Path

CLUSTERS_DIR = Path.home() / "Pictures" / "face_clusters_v2"
CSV_PATH     = CLUSTERS_DIR / "_clusters.csv"

INVALID_CHARS = '/\\:*?"<>|'


def sanitize(name: str) -> str:
    name = name.strip()
    for ch in INVALID_CHARS:
        name = name.replace(ch, "_")
    return name


def count_faces(folder: Path) -> int:
    return sum(1 for p in folder.iterdir() if p.is_file() and p.suffix.lower() == ".jpg")


def merge_into(src_folder: Path, dst_folder: Path) -> None:
    """Move all files from src_folder into dst_folder (handling name collisions)."""
    for f in src_folder.iterdir():
        if not f.is_file():
            continue
        dest = dst_folder / f.name
        i = 2
        while dest.exists():
            dest = dst_folder / f"{f.stem}__{i}{f.suffix}"
            i += 1
        shutil.move(str(f), str(dest))
    try:
        src_folder.rmdir()
    except OSError:
        pass  # not empty for some reason — leave it


def main() -> int:
    if not CLUSTERS_DIR.exists():
        print(f"ERROR: {CLUSTERS_DIR} does not exist. Run cluster_faces_v2.py first.")
        return 2

    # Person folders, in size order (already sorted by cluster_faces_v2 — person_001 = largest).
    person_folders = sorted([
        p for p in CLUSTERS_DIR.iterdir()
        if p.is_dir() and p.name.startswith("person_")
    ])
    if not person_folders:
        print("No person folders found. Run cluster_faces_v2.py first.")
        return 2

    total = len(person_folders)
    print(f"\nFound {total} person clusters to label.")
    print("For each one, the montage opens in Preview.")
    print("  - Type the person's name and press Enter")
    print("  - Press Enter (empty) to skip (keeps person_NNN)")
    print("  - Type the same name twice → folders auto-merge")
    print("  - Type 'q' to quit and save progress so far\n")

    name_map: dict[str, str] = {}

    for idx, folder in enumerate(person_folders, start=1):
        # Folder may have been renamed/removed during a merge in a previous iteration
        if not folder.exists():
            continue

        montage = CLUSTERS_DIR / f"{folder.name}_montage.jpg"
        n = count_faces(folder)

        if montage.exists():
            subprocess.run(["open", str(montage)], check=False)

        print(f"[{idx}/{total}] {folder.name}  ({n} faces)")
        try:
            raw = input("  Name: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nInterrupted — saving progress.")
            break

        if raw.lower() == "q":
            print("Quitting — saving progress.")
            break
        if not raw:
            print(f"  → kept as '{folder.name}'\n")
            continue

        new_name = sanitize(raw)
        if not new_name:
            print(f"  Invalid name, kept as '{folder.name}'\n")
            continue

        target = CLUSTERS_DIR / new_name
        target_montage = CLUSTERS_DIR / f"{new_name}_montage.jpg"

        if target.exists() and target.resolve() != folder.resolve():
            # Same name as an already-labeled folder → merge
            print(f"  '{new_name}' already exists — merging {folder.name} into it...")
            merge_into(folder, target)
            if montage.exists():
                montage.unlink()
            name_map[folder.name] = new_name
            print(f"  → merged into '{new_name}'\n")
        else:
            folder.rename(target)
            if montage.exists():
                montage.rename(target_montage)
            name_map[folder.name] = new_name
            print(f"  → '{new_name}'\n")

    # Update CSV manifest
    if name_map and CSV_PATH.exists():
        rows = []
        with CSV_PATH.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames
            for row in reader:
                if row.get("person") in name_map:
                    row["person"] = name_map[row["person"]]
                rows.append(row)
        with CSV_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"Updated _clusters.csv ({len(rows)} rows).")

    print(f"\nDone. Labeled {len(name_map)} cluster(s).")
    print("Next: re-run organize_originals.py to apply names to your originals folder.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

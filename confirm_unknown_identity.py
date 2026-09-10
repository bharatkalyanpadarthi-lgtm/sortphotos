#!/usr/bin/env python3
"""Confirm difficult known faces from unknown_identity without loosening safety.

The command requires exactly one detected face per file.  Confirmed examples
are organized using the normal atomic-copy and nudity-routing pipeline, kept in
the face cache, and registered as bounded supplemental identity prototypes.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import identity_confirmations
import evaluation_enrollment
import pipeline_paths
import person_aliases
import recover_no_usable_faces
import sort_photos


UNKNOWN_ROOT = (
    pipeline_paths.SOURCE_REVIEW / "unassigned_intake" / "unknown_identity"
)


def canonical_person(value: str, identity_db: sort_photos.IdentityDB,
                     *, people_root: Path | None = None) -> str | None:
    root = people_root or pipeline_paths.PEOPLE_ROOT
    wanted = person_aliases.canonical_folder(value.strip(), root).casefold()
    return next(
        (name for name in identity_db.identities if name.casefold() == wanted),
        None,
    )


def resolve_unknown(value: str, unknown_root: Path) -> Path:
    supplied = Path(value).expanduser()
    if supplied.is_file():
        candidate = supplied.resolve()
    else:
        wanted = supplied.name.casefold()
        wanted_stem = supplied.stem.casefold()
        matches = [
            path.resolve()
            for path in unknown_root.rglob("*")
            if path.is_file()
            and (
                path.name.casefold() == wanted
                or path.stem.casefold() == wanted
                or path.stem.casefold() == wanted_stem
            )
        ]
        if not matches:
            raise ValueError(f"unknown file not found: {value}")
        if len(matches) > 1:
            choices = "\n  ".join(str(path) for path in matches[:10])
            raise ValueError(f"ambiguous unknown filename: {value}\n  {choices}")
        candidate = matches[0]
    try:
        candidate.relative_to(unknown_root.resolve())
    except ValueError as error:
        raise ValueError(f"file is outside unknown_identity: {candidate}") from error
    return candidate


def existing_person_path(
    person: str,
    digest: str,
    people_root: Path | None = None,
) -> Path | None:
    """Find any verified copy already held inside a person's library.

    Confirmed files can be routed to ``photos``, ``photos/nude``, or a
    person-level review folder.  Searching only identity-training sources made
    exact duplicates fail when the first copy was placed in nudity review.
    """
    person_root = (people_root or pipeline_paths.PEOPLE_ROOT) / person
    for candidate in sort_photos.iter_images(
        person_root,
        excluded_dir_names=set(),
    ):
        try:
            if sort_photos.sha256_file(candidate) == digest:
                return candidate.resolve()
        except OSError:
            continue
    return None


def prompt_if_needed(person: str | None, files: list[str]) -> tuple[str, list[str]]:
    if not person:
        person = input("Confirmed person name: ").strip()
    if not files:
        entered = input("Unknown filename(s), separated by commas: ").strip()
        files = [item.strip() for item in entered.split(",") if item.strip()]
    return person, files


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--person", help="Exact canonical person folder name")
    parser.add_argument("files", nargs="*", help="Unknown file paths, names, or stems")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    requested_person, requested_files = prompt_if_needed(args.person, args.files)
    identity_db = sort_photos.load_identity_db()
    if identity_db is None:
        print("Identity database is unavailable. Run `face rebuild-id` first.")
        return 2
    person = canonical_person(requested_person, identity_db)
    if person is None:
        print(f"Unknown person: {requested_person}")
        close = [
            name for name in sorted(identity_db.identities)
            if requested_person.casefold() in name.casefold()
            or name.casefold() in requested_person.casefold()
        ]
        if close:
            print("Possible names: " + ", ".join(close[:10]))
        return 2
    if not requested_files:
        print("No unknown files were supplied.")
        return 2

    try:
        files = [resolve_unknown(value, UNKNOWN_ROOT) for value in requested_files]
    except ValueError as error:
        print(error)
        return 2

    print("Confirm Unknown Identity")
    print("=" * 60)
    print(f"Person: {person}")
    print(f"Files:  {len(files)}")
    print(f"Mode:   {'DRY RUN' if args.dry_run else 'SAFE CONFIRMATION'}")

    existing_hashes = recover_no_usable_faces.load_existing_hashes(
        pipeline_paths.PEOPLE_ROOT
    )
    next_indexes: dict[Path, int] = {}
    cache_entries: list[tuple[Path, sort_photos.CachedFace, str]] = []
    confirmed: list[tuple[Path, str, str]] = []
    failures = 0

    results = recover_no_usable_faces.iter_detection_results(
        files,
        1,
        1024,
        384,
        index_path=sort_photos.analysis_index_file(),
    )
    for _index, source, status, faces, detector_error in results:
        if detector_error:
            print(f"FAILED {source.name}: detector error: {detector_error}")
            failures += 1
            continue
        if len(faces) != 1:
            print(
                f"SKIPPED {source.name}: detected {len(faces)} faces ({status}); "
                "multi-face images require visual face selection"
            )
            failures += 1
            continue
        source_hash = sort_photos.sha256_file(source)
        destination, already_present = recover_no_usable_faces.copy_to_person(
            source,
            person,
            source_hash,
            existing_hashes,
            next_indexes,
            dry_run=args.dry_run,
        )
        if already_present:
            destination = existing_person_path(person, source_hash)
            if destination is None:
                print(f"FAILED {source.name}: duplicate exists but its path was not found")
                failures += 1
                continue
        if destination is None:
            print(f"FAILED {source.name}: no organized destination")
            failures += 1
            continue

        if args.dry_run:
            print(f"WOULD CONFIRM {source.name} -> {destination}")
            continue

        identity_confirmations.record(
            sort_photos.IDENTITY_CONFIRMATIONS_FILE,
            person=person,
            organized_path=destination,
            content_sha256=source_hash,
            original_name=source.name,
            face=faces[0],
            detector_version=sort_photos.config_fingerprint(),
        )
        evaluation_enrollment.enroll(
            source=destination,
            person=person,
            content_sha256=source_hash,
            pose_label=faces[0].pose_label,
            quality=float(faces[0].quality),
        )
        cache_entries.append((destination, faces[0], person))
        confirmed.append((destination, source_hash, source.name))
        recover_no_usable_faces.move_review_file(
            source,
            pipeline_paths.SOURCE_REVIEW
            / "ready_to_delete"
            / "confirmed_unknown_identity",
            status="confirmed_identity",
            extra={
                "person": person,
                "destination": str(destination),
                "content_sha256": source_hash,
            },
            dry_run=False,
        )
        print(f"CONFIRMED {source.name} -> {destination.name}")

    if confirmed and not args.dry_run:
        persisted = recover_no_usable_faces.persist_recovered_faces(
            cache_entries, rebuild_identity=False
        )
        sort_photos.build_identity_db_from_person_folders(pipeline_paths.PEOPLE_ROOT)
        print(f"Trusted examples recorded: {len(confirmed)}")
        print(f"Face-cache entries updated: {persisted}")
        print(f"Identity profile refreshed: {person}")
    elif args.dry_run:
        print("Dry run complete; no files or profiles changed.")

    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

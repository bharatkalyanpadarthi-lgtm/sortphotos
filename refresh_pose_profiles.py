#!/usr/bin/env python3
"""Safely split legacy profile prototypes into left/right pose buckets."""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

import evaluation_enrollment
import identity_confirmations
import identity_evaluation
import identity_profiles
import pipeline_paths
import sort_photos


DEFAULT_REPORT = (
    pipeline_paths.SOURCE_REVIEW
    / "identity_evaluation"
    / "latest_pose_profile_refresh.json"
)


def worker(task_path: Path, output_path: Path) -> int:
    with task_path.open("rb") as handle:
        tasks = pickle.load(handle)
    app = sort_photos._build_app()
    results: list[dict[str, str]] = []
    for task in tasks:
        source = Path(task["source"])
        pose = "profile_unknown"
        if source.is_file():
            faces = sort_photos._detect_one_image(source, app)
            if faces:
                target = identity_profiles.normalize_vector(task["embedding"])
                face = min(
                    faces,
                    key=lambda value: 1.0 - float(
                        identity_profiles.normalize_vector(value.embedding) @ target
                    ),
                )
                pose = str(face.pose_label or "profile_unknown")
        results.append({
            "person": str(task["person"]),
            "source": str(source),
            "pose": pose,
        })
    output_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    return 0


def refresh(batch_size: int) -> int:
    incumbent = sort_photos.load_identity_db()
    if incumbent is None:
        print("ERROR: primary identity database is unavailable")
        return 2
    tasks: list[dict[str, object]] = []
    for person, groups in incumbent.pose_prototypes.items():
        values = groups.get("profile_unknown", [])
        sources = incumbent.pose_prototype_sources.get(person, {}).get(
            "profile_unknown", []
        )
        for embedding, source in zip(values, sources):
            tasks.append({"person": person, "source": source, "embedding": embedding})
    print("Pose Profile Refresh")
    print("=" * 60)
    print(f"Legacy profile prototypes: {len(tasks)}")
    print(f"Worker batch size:         {batch_size}")
    if not tasks:
        print("Pose profiles are already current.")
        return 0

    resolved: dict[tuple[str, str], str] = {}
    with tempfile.TemporaryDirectory(prefix="face_pose_refresh_") as temporary:
        root = Path(temporary)
        for offset in range(0, len(tasks), batch_size):
            batch = tasks[offset:offset + batch_size]
            task_path = root / f"tasks_{offset:05d}.pkl"
            output_path = root / f"results_{offset:05d}.json"
            with task_path.open("wb") as handle:
                pickle.dump(batch, handle, protocol=pickle.HIGHEST_PROTOCOL)
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                str(task_path),
                str(output_path),
            ]
            result = subprocess.run(command, check=False)
            if result.returncode != 0 or not output_path.is_file():
                print(f"ERROR: pose worker failed at item {offset + 1}")
                return 3
            for item in json.loads(output_path.read_text(encoding="utf-8")):
                resolved[(str(item["person"]), str(item["source"]))] = str(item["pose"])
            print(f"Classified {min(offset + len(batch), len(tasks))}/{len(tasks)}", flush=True)

    candidate = pickle.loads(pickle.dumps(incumbent, protocol=pickle.HIGHEST_PROTOCOL))
    for person in list(candidate.pose_prototypes):
        old_values = candidate.pose_prototypes[person].pop("profile_unknown", [])
        old_sources = candidate.pose_prototype_sources.get(person, {}).pop(
            "profile_unknown", []
        )
        for embedding, source in zip(old_values, old_sources):
            pose = resolved.get((person, source), "profile_unknown")
            if pose not in {"frontal", "left_profile", "right_profile"}:
                pose = "profile_unknown"
            candidate.pose_prototypes[person].setdefault(pose, []).append(embedding)
            candidate.pose_prototype_sources.setdefault(person, {}).setdefault(
                pose, []
            ).append(source)

    cache = sort_photos.load_cache()
    evaluation_enrollment.backfill_confirmations(
        identity_confirmations.load(sort_photos.IDENTITY_CONFIRMATIONS_FILE),
        cache.faces,
    )
    allowed, gate = identity_evaluation.activation_gate(
        candidate,
        incumbent,
        cache,
        confirmed_set=evaluation_enrollment.DEFAULT_PATH,
    )
    report = {
        "tasks": len(tasks),
        "resolved": {
            pose: sum(value == pose for value in resolved.values())
            for pose in ("frontal", "left_profile", "right_profile", "profile_unknown")
        },
        "activation_gate": gate,
        "activated": allowed,
    }
    DEFAULT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    DEFAULT_REPORT.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if not allowed:
        print("ERROR: pose refresh was blocked by the identity activation gate")
        print(f"Report: {DEFAULT_REPORT}")
        return 4
    sort_photos.save_identity_db(candidate)
    print("Pose-aware identity database activated safely.")
    print(f"Report: {DEFAULT_REPORT}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=40)
    parser.add_argument("--worker", nargs=2, metavar=("TASKS", "OUTPUT"))
    args = parser.parse_args()
    if args.worker:
        return worker(Path(args.worker[0]), Path(args.worker[1]))
    return refresh(max(1, int(args.batch_size)))


if __name__ == "__main__":
    raise SystemExit(main())

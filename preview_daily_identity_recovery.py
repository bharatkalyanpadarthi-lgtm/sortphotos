#!/usr/bin/env python3
"""Preview daily recovery from a prior review CSV. Never move or label images."""

import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np

import daily_identity_recovery as recovery
import sort_photos as sorter


def load_report_records(report: Path, index: Path):
    records, skipped = [], []
    connection = sqlite3.connect(index.resolve().as_uri() + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        with report.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row["outcome"] != "unknown_identity":
                    continue
                path = Path(row["review_path"]).resolve()
                rows = []
                if path.is_file():
                    digest = sorter.sha256_file(path)
                    for source in (path, Path(row["source_path"]).resolve()):
                        asset = connection.execute(
                            "SELECT sha256 FROM assets WHERE path=?", (str(source),)
                        ).fetchone()
                        if asset is None or asset["sha256"] != digest:
                            continue
                        rows = connection.execute(
                            "SELECT * FROM face_detections WHERE asset_path=? "
                            "AND detector_config=? ORDER BY face_index",
                            (str(source), sorter.config_fingerprint()),
                        ).fetchall()
                        if rows:
                            break
                if not rows or len(rows) != int(row["detected_faces"]):
                    skipped.append(dict(source=str(path), status="held",
                                        reason="verified_cached_detection_unavailable"))
                    continue
                for face in rows:
                    records.append(sorter.FaceRecord(
                        src=path, face_index=int(face["face_index"]),
                        det_score=float(face["det_score"]), bbox_size=float(face["bbox_size"]),
                        sharpness=float(face["sharpness"]), yaw_proxy=float(face["yaw_proxy"]),
                        quality=float(face["quality"]), pose_label=str(face["pose_label"]),
                        embedding=np.frombuffer(face["embedding"], dtype=np.float32).copy(),
                        crop_jpeg=bytes(face["crop_jpeg"]), cluster_id=-1,
                    ))
    finally:
        connection.close()
    return records, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("review_report", type=Path)
    args = parser.parse_args()
    report = args.review_report.expanduser().resolve()
    print("Read-only preview: reusing hash-verified detections; no images will move.", flush=True)
    records, skipped = load_report_records(report, sorter.analysis_index_file())
    print(f"Loaded {len(records)} cached face records; {len(skipped)} items need fresh detection.", flush=True)
    print("Loading identity/reference cache for the safety benchmark...", flush=True)
    plan = recovery.plan_recovery(
        records, {-1: "unknown"}, sorter.load_identity_db(), sorter.load_cache(),
        people_root=sorter.DEFAULT_OUTPUT / "photos_by_person",
        progress=lambda message: print(message, flush=True),
    )
    plan.rows.extend(skipped)
    destination = sorter.DEFAULT_OUTPUT / "_source_review" / "unassigned_intake" / "reports"
    destination = destination / (report.stem + "_recovery_preview.json")
    recovery.write_report(plan, destination)
    print(f"Preview counts: {plan.counts}", flush=True)
    print(f"Report: {destination}", flush=True)
    print("No images moved; no identity labels or review decisions changed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

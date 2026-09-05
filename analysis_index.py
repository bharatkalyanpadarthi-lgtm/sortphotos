#!/usr/bin/env python3
"""Incremental SQLite index for reusable photo-analysis metadata."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import content_identity


SCHEMA_VERSION = 4


@dataclass(frozen=True)
class AssetFingerprint:
    sha256: str
    pixel_sha256: str
    phash: int
    width: int
    height: int


@dataclass(frozen=True)
class DetectionRecord:
    face_index: int
    det_score: float
    bbox_size: float
    sharpness: float
    yaw_proxy: float
    quality: float
    embedding: bytes
    embedding_dimension: int
    image_phash: bytes
    image_phash_bits: int
    crop_jpeg: bytes
    label: str | None = None
    pose_label: str = "unknown"
    bbox: tuple[float, ...] = ()
    keypoints: tuple[tuple[float, ...], ...] = ()


@dataclass(frozen=True)
class CachedDetectionSet:
    status: str
    width: int
    height: int
    orientation: int
    detections: tuple[DetectionRecord, ...]


class AnalysisIndex:
    def __init__(self, path: Path, *, timeout: float = 30.0):
        self.path = path.expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        timeout = max(0.0, float(timeout))
        self.connection = sqlite3.connect(self.path, timeout=timeout)
        try:
            self.connection.execute("PRAGMA journal_mode=WAL")
            self.connection.execute("PRAGMA synchronous=NORMAL")
            self.connection.execute(
                f"PRAGMA busy_timeout={max(0, int(timeout * 1000))}"
            )
            self.connection.execute("PRAGMA foreign_keys=ON")
            self._create_schema()
        except Exception:
            self.connection.close()
            raise

    def _create_schema(self) -> None:
        if self.connection.execute("SELECT 1 FROM sqlite_master WHERE name='metadata'").fetchone():
            version = self.connection.execute("SELECT value FROM metadata WHERE key='schema_version'").fetchone()
            if version and int(version[0]) > SCHEMA_VERSION:
                raise ValueError("Analysis index was created by a newer version; refusing to downgrade it")
        self.connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS metadata (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS content_versions (
                path TEXT PRIMARY KEY,
                signature TEXT NOT NULL,
                sha256 TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS assets (
                path TEXT PRIMARY KEY,
                byte_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                sha256 TEXT NOT NULL,
                pixel_sha256 TEXT NOT NULL,
                phash_hex TEXT NOT NULL,
                width INTEGER NOT NULL,
                height INTEGER NOT NULL,
                nudity_status TEXT,
                detection_status TEXT,
                identity_name TEXT,
                identity_distance REAL,
                updated_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS assets_sha256_idx ON assets(sha256);
            CREATE INDEX IF NOT EXISTS assets_identity_idx ON assets(identity_name);
            CREATE TABLE IF NOT EXISTS asset_analysis (
                path TEXT PRIMARY KEY,
                byte_size INTEGER NOT NULL,
                mtime_ns INTEGER NOT NULL,
                width INTEGER NOT NULL DEFAULT 0,
                height INTEGER NOT NULL DEFAULT 0,
                orientation INTEGER NOT NULL DEFAULT 1,
                detection_config TEXT,
                detection_status TEXT,
                nudity_model TEXT,
                nudity_status TEXT,
                nudity_json TEXT,
                updated_at REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS face_detections (
                asset_path TEXT NOT NULL,
                face_index INTEGER NOT NULL,
                detector_config TEXT NOT NULL,
                det_score REAL NOT NULL,
                bbox_size REAL NOT NULL,
                sharpness REAL NOT NULL,
                yaw_proxy REAL NOT NULL,
                quality REAL NOT NULL,
                embedding BLOB NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                image_phash BLOB NOT NULL,
                image_phash_bits INTEGER NOT NULL,
                crop_jpeg BLOB NOT NULL,
                label TEXT,
                pose_label TEXT NOT NULL DEFAULT 'unknown',
                PRIMARY KEY(asset_path, face_index, detector_config),
                FOREIGN KEY(asset_path) REFERENCES asset_analysis(path) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS face_detections_config_idx
                ON face_detections(detector_config);
            CREATE TABLE IF NOT EXISTS identity_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                asset_path TEXT NOT NULL,
                face_index INTEGER NOT NULL,
                identity_name TEXT NOT NULL,
                distance REAL NOT NULL,
                margin REAL NOT NULL,
                lane TEXT NOT NULL,
                run_id TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS identity_history_asset_idx
                ON identity_history(asset_path, created_at);
            CREATE TABLE IF NOT EXISTS operation_history (
                event_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                operation TEXT NOT NULL,
                status TEXT NOT NULL,
                source_path TEXT NOT NULL,
                dest_path TEXT NOT NULL,
                event_json TEXT NOT NULL,
                created_at REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS operation_history_run_idx
                ON operation_history(run_id, created_at);
            CREATE INDEX IF NOT EXISTS operation_history_dest_idx
                ON operation_history(dest_path, created_at);
            """
        )
        detection_columns = {
            str(row[1])
            for row in self.connection.execute("PRAGMA table_info(face_detections)")
        }
        if "pose_label" not in detection_columns:
            self.connection.execute(
                "ALTER TABLE face_detections "
                "ADD COLUMN pose_label TEXT NOT NULL DEFAULT 'unknown'"
            )
        for column in ("bbox_json", "keypoints_json"):
            if column not in detection_columns:
                self.connection.execute(f"ALTER TABLE face_detections ADD COLUMN {column} TEXT NOT NULL DEFAULT '[]'")
        analysis_columns = {str(row[1]) for row in self.connection.execute(
            "PRAGMA table_info(asset_analysis)")}
        for column in ("detection_sha256", "nudity_sha256"):
            if column not in analysis_columns:
                self.connection.execute(f"ALTER TABLE asset_analysis ADD COLUMN {column} TEXT")
        self.connection.execute(
            "INSERT INTO metadata(key, value) VALUES('schema_version', ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (str(SCHEMA_VERSION),),
        )
        self.connection.commit()

    @staticmethod
    def canonical_path(path: Path | str) -> str:
        return str(Path(path).expanduser().resolve(strict=False))

    @staticmethod
    def current_signature(path: Path) -> tuple[int, int] | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        return int(stat.st_size), int(stat.st_mtime_ns)

    def fingerprint(self, path: Path) -> AssetFingerprint | None:
        try:
            stat = path.stat()
        except OSError:
            return None
        row = self.connection.execute(
            "SELECT sha256, pixel_sha256, phash_hex, width, height "
            "FROM assets WHERE path=? AND byte_size=? AND mtime_ns=?",
            (self.canonical_path(path), int(stat.st_size), int(stat.st_mtime_ns)),
        ).fetchone()
        if row is None:
            return None
        if str(row[0]) != self.content_sha256(path):
            return None
        try:
            return AssetFingerprint(
                sha256=str(row[0]),
                pixel_sha256=str(row[1]),
                phash=int(str(row[2]), 16),
                width=int(row[3]),
                height=int(row[4]),
            )
        except (TypeError, ValueError):
            return None

    def upsert_fingerprint(self, path: Path, fingerprint: AssetFingerprint) -> None:
        stat = path.stat()
        if fingerprint.sha256 != self.content_sha256(path):
            raise ValueError(f"Fingerprint belongs to different file content: {path}")
        canonical = self.canonical_path(path)
        self.connection.execute(
            """
            INSERT INTO assets(
                path, byte_size, mtime_ns, sha256, pixel_sha256, phash_hex,
                width, height, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                nudity_status=CASE WHEN assets.sha256=excluded.sha256 THEN assets.nudity_status END,
                detection_status=CASE WHEN assets.sha256=excluded.sha256 THEN assets.detection_status END,
                identity_name=CASE WHEN assets.sha256=excluded.sha256 THEN assets.identity_name END,
                identity_distance=CASE WHEN assets.sha256=excluded.sha256 THEN assets.identity_distance END,
                byte_size=excluded.byte_size,
                mtime_ns=excluded.mtime_ns,
                sha256=excluded.sha256,
                pixel_sha256=excluded.pixel_sha256,
                phash_hex=excluded.phash_hex,
                width=excluded.width,
                height=excluded.height,
                updated_at=excluded.updated_at
            """,
            (
                canonical, int(stat.st_size), int(stat.st_mtime_ns),
                fingerprint.sha256, fingerprint.pixel_sha256,
                f"{fingerprint.phash:016x}", fingerprint.width,
                fingerprint.height, time.time(),
            ),
        )

    def update_classification(
        self,
        path: Path,
        *,
        nudity_status: str | None = None,
        detection_status: str | None = None,
        identity_name: str | None = None,
        identity_distance: float | None = None,
    ) -> None:
        assignments: list[str] = []
        values: list[object] = []
        for column, value in (
            ("nudity_status", nudity_status),
            ("detection_status", detection_status),
            ("identity_name", identity_name),
            ("identity_distance", identity_distance),
        ):
            if value is not None:
                assignments.append(f"{column}=?")
                values.append(value)
        if not assignments:
            return
        assignments.append("updated_at=?")
        values.append(time.time())
        values.append(self.canonical_path(path))
        self.connection.execute(
            f"UPDATE assets SET {', '.join(assignments)} WHERE path=?", values)

    def cached_detections(
        self,
        path: Path,
        detector_config: str,
    ) -> CachedDetectionSet | None:
        signature = self.current_signature(path)
        if signature is None:
            return None
        canonical = self.canonical_path(path)
        digest = self.content_sha256(path)
        row = self.connection.execute(
            "SELECT width, height, orientation, detection_status "
            "FROM asset_analysis WHERE path=? AND byte_size=? AND mtime_ns=? "
            "AND detection_config=? AND detection_sha256=?",
            (canonical, signature[0], signature[1], detector_config, digest),
        ).fetchone()
        if row is None or row[3] is None:
            return None
        records = self.connection.execute(
            "SELECT face_index, det_score, bbox_size, sharpness, yaw_proxy, quality, "
            "embedding, embedding_dimension, image_phash, image_phash_bits, crop_jpeg, label, "
            "pose_label, bbox_json, keypoints_json "
            "FROM face_detections WHERE asset_path=? AND detector_config=? "
            "ORDER BY face_index",
            (canonical, detector_config),
        ).fetchall()
        detections = tuple(DetectionRecord(
            face_index=int(record[0]),
            det_score=float(record[1]),
            bbox_size=float(record[2]),
            sharpness=float(record[3]),
            yaw_proxy=float(record[4]),
            quality=float(record[5]),
            embedding=bytes(record[6]),
            embedding_dimension=int(record[7]),
            image_phash=bytes(record[8]),
            image_phash_bits=int(record[9]),
            crop_jpeg=bytes(record[10]),
            label=str(record[11]) if record[11] is not None else None,
            pose_label=str(record[12] or "unknown"),
            bbox=tuple(json.loads(record[13])),
            keypoints=tuple(tuple(point) for point in json.loads(record[14])),
        ) for record in records)
        return CachedDetectionSet(
            status=str(row[3]),
            width=int(row[0]),
            height=int(row[1]),
            orientation=int(row[2]),
            detections=detections,
        )

    def replace_detections(
        self,
        path: Path,
        detector_config: str,
        status: str,
        detections: Iterable[DetectionRecord],
        *,
        width: int = 0,
        height: int = 0,
        orientation: int = 1,
        expected_sha256: str | None = None,
    ) -> None:
        signature = self.current_signature(path)
        if signature is None:
            return
        digest = self.content_sha256(path)
        if not digest or (expected_sha256 and digest != expected_sha256):
            raise ValueError(f"Detection source changed before commit: {path}")
        # Recovery and review flows can merge detections from several passes.
        # Keep the best record for each stable face index before the atomic replace.
        unique_detections: dict[int, DetectionRecord] = {}
        for record in detections:
            face_index = int(record.face_index)
            current = unique_detections.get(face_index)
            if current is None or (record.quality, record.det_score) > (
                current.quality,
                current.det_score,
            ):
                unique_detections[face_index] = record
        normalized_detections = [
            unique_detections[face_index]
            for face_index in sorted(unique_detections)
        ]
        canonical = self.canonical_path(path)
        now = time.time()
        self.connection.execute(
            """
            INSERT INTO asset_analysis(
                path, byte_size, mtime_ns, width, height, orientation,
                detection_config, detection_status, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                byte_size=excluded.byte_size,
                mtime_ns=excluded.mtime_ns,
                width=CASE WHEN excluded.width > 0 THEN excluded.width ELSE asset_analysis.width END,
                height=CASE WHEN excluded.height > 0 THEN excluded.height ELSE asset_analysis.height END,
                orientation=excluded.orientation,
                detection_config=excluded.detection_config,
                detection_status=excluded.detection_status,
                updated_at=excluded.updated_at
            """,
            (canonical, signature[0], signature[1], int(width), int(height),
             int(orientation), detector_config, status, now),
        )
        self.connection.execute(
            "DELETE FROM face_detections WHERE asset_path=? AND detector_config=?",
            (canonical, detector_config),
        )
        self.connection.executemany(
            """
            INSERT INTO face_detections(
                asset_path, face_index, detector_config, det_score, bbox_size,
                sharpness, yaw_proxy, quality, embedding, embedding_dimension,
                image_phash, image_phash_bits, crop_jpeg, label, pose_label, bbox_json, keypoints_json
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [(
                canonical, int(record.face_index), detector_config,
                float(record.det_score), float(record.bbox_size),
                float(record.sharpness), float(record.yaw_proxy),
                float(record.quality), sqlite3.Binary(record.embedding),
                int(record.embedding_dimension), sqlite3.Binary(record.image_phash),
                int(record.image_phash_bits), sqlite3.Binary(record.crop_jpeg),
                record.label, str(record.pose_label or "unknown"),
                json.dumps(record.bbox), json.dumps(record.keypoints),
            ) for record in normalized_detections],
        )
        self.connection.execute(
            "UPDATE asset_analysis SET detection_sha256=? WHERE path=?", (digest, canonical))

    def record_nudity(
        self,
        path: Path,
        *,
        model: str,
        status: str,
        detections: Any,
        expected_sha256: str | None = None,
    ) -> None:
        signature = self.current_signature(path)
        if signature is None:
            return
        digest = self.content_sha256(path)
        if not digest or (expected_sha256 and digest != expected_sha256):
            raise ValueError(f"Classification source changed before commit: {path}")
        canonical = self.canonical_path(path)
        payload = json.dumps(detections, sort_keys=True, default=str)
        self.connection.execute(
            """
            INSERT INTO asset_analysis(
                path, byte_size, mtime_ns, nudity_model, nudity_status,
                nudity_json, updated_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(path) DO UPDATE SET
                byte_size=excluded.byte_size,
                mtime_ns=excluded.mtime_ns,
                nudity_model=excluded.nudity_model,
                nudity_status=excluded.nudity_status,
                nudity_json=excluded.nudity_json,
                updated_at=excluded.updated_at
            """,
            (canonical, signature[0], signature[1], model, status, payload, time.time()),
        )
        self.connection.execute(
            "UPDATE asset_analysis SET nudity_sha256=? WHERE path=?", (digest, canonical))

    def cached_nudity(self, path: Path, model: str) -> tuple[str, Any] | None:
        signature = self.current_signature(path)
        if signature is None:
            return None
        row = self.connection.execute(
            "SELECT nudity_status, nudity_json FROM asset_analysis "
            "WHERE path=? AND byte_size=? AND mtime_ns=? AND nudity_model=? AND nudity_sha256=?",
            (self.canonical_path(path), signature[0], signature[1], model, self.content_sha256(path)),
        ).fetchone()
        if row is None or row[0] is None:
            return None
        try:
            detections = json.loads(str(row[1] or "[]"))
        except json.JSONDecodeError:
            detections = []
        return str(row[0]), detections

    def cached_nudity_detections_by_sha256(self, sha256: str) -> Any | None:
        """Reuse raw NudeNet output while allowing routing policy upgrades."""
        if not sha256:
            return None
        row = self.connection.execute(
            "SELECT aa.nudity_json FROM asset_analysis AS aa "
            "WHERE aa.nudity_sha256=? AND aa.nudity_model LIKE 'nudenet-policy-%' "
            "AND aa.nudity_json IS NOT NULL "
            "ORDER BY aa.updated_at DESC LIMIT 1",
            (sha256,),
        ).fetchone()
        if row is None:
            return None
        try:
            detections = json.loads(str(row[0] or "[]"))
        except json.JSONDecodeError:
            return None
        return detections if isinstance(detections, list) else None

    def content_sha256(self, path: Path) -> str | None:
        try:
            canonical = self.canonical_path(path)
            version = content_identity.file_version(path)
            signature = json.dumps(version)
            row = self.connection.execute(
                "SELECT sha256 FROM content_versions WHERE path=? AND signature=?",
                (canonical, signature),
            ).fetchone()
            if row is not None:
                return str(row[0])
            digest = content_identity.content_sha256(path)
            if content_identity.file_version(path) != version:
                return None
            self.connection.execute(
                "INSERT INTO content_versions VALUES(?, ?, ?) ON CONFLICT(path) DO UPDATE SET "
                "signature=excluded.signature, sha256=excluded.sha256", (canonical, signature, digest))
            return digest
        except OSError:
            return None

    @staticmethod
    def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str | None:
        """Hash a cache miss without decoding the image or changing the index."""
        digest = hashlib.sha256()
        try:
            with path.open("rb") as handle:
                while chunk := handle.read(chunk_size):
                    digest.update(chunk)
        except OSError:
            return None
        return digest.hexdigest()

    def record_identity(
        self,
        path: Path,
        *,
        face_index: int,
        identity_name: str,
        distance: float,
        margin: float,
        lane: str,
        run_id: str,
    ) -> None:
        self.connection.execute(
            """
            INSERT INTO identity_history(
                asset_path, face_index, identity_name, distance, margin,
                lane, run_id, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (self.canonical_path(path), int(face_index), identity_name,
             float(distance), float(margin), lane, run_id, time.time()),
        )

    def latest_identity(self, path: Path, face_index: int) -> dict[str, Any] | None:
        row = self.connection.execute(
            "SELECT identity_name, distance, margin, lane, run_id, created_at "
            "FROM identity_history WHERE asset_path=? AND face_index=? "
            "ORDER BY created_at DESC LIMIT 1",
            (self.canonical_path(path), int(face_index)),
        ).fetchone()
        if row is None:
            return None
        return {
            "identity_name": str(row[0]),
            "distance": float(row[1]),
            "margin": float(row[2]),
            "lane": str(row[3]),
            "run_id": str(row[4]),
            "created_at": float(row[5]),
        }

    def record_operation(self, event: dict[str, Any]) -> None:
        payload = json.dumps(event, sort_keys=True, separators=(",", ":"), default=str)
        event_id = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        self.connection.execute(
            """
            INSERT OR IGNORE INTO operation_history(
                event_id, run_id, operation, status, source_path, dest_path,
                event_json, created_at
            ) VALUES(?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (event_id, str(event.get("run_id") or ""),
             str(event.get("operation") or ""), str(event.get("status") or ""),
             str(event.get("source_path") or ""), str(event.get("dest_path") or ""),
             payload, time.time()),
        )

    def source_batch_context(self, path: Path, *, maximum_hops: int = 12) -> str:
        """Trace recoverable moves back to their intake parent and run."""
        current = self.canonical_path(path)
        seen = {current}
        oldest_source = current
        oldest_run = ""
        for _hop in range(max(1, int(maximum_hops))):
            row = self.connection.execute(
                "SELECT source_path, run_id FROM operation_history "
                "WHERE dest_path=? AND status IN ('moved', 'copied', 'completed') "
                "ORDER BY created_at DESC LIMIT 1",
                (current,),
            ).fetchone()
            if row is None:
                break
            source = self.canonical_path(str(row[0]))
            if not source or source in seen:
                break
            seen.add(source)
            oldest_source = source
            oldest_run = str(row[1] or oldest_run)
            current = source
        parent = str(Path(oldest_source).parent).casefold()
        if not oldest_run or parent in {"", ".", "/"}:
            return ""
        return f"ledger:{oldest_run}:{parent}"

    def prune_missing(self, paths: set[str]) -> int:
        rows = self.connection.execute("SELECT path FROM assets").fetchall()
        stale = [str(row[0]) for row in rows if str(row[0]) not in paths]
        self.connection.executemany("DELETE FROM assets WHERE path=?", [(path,) for path in stale])
        canonical_paths = {self.canonical_path(path) for path in paths}
        analysis_rows = self.connection.execute("SELECT path FROM asset_analysis").fetchall()
        stale_analysis = [
            str(row[0]) for row in analysis_rows if str(row[0]) not in canonical_paths
        ]
        self.connection.executemany(
            "DELETE FROM asset_analysis WHERE path=?",
            [(path,) for path in stale_analysis],
        )
        return len(stale)

    def commit(self) -> None:
        self.connection.commit()

    def close(self) -> None:
        self.connection.commit()
        self.connection.close()

    def __enter__(self) -> "AnalysisIndex":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if exc_type is None:
            self.connection.commit()
        else:
            self.connection.rollback()
        self.connection.close()

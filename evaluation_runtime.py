"""Versioned inputs and bounded, resumable blocks for read-only benchmarks."""

from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
from contextlib import contextmanager

import numpy as np

import content_identity
from pipeline_progress import StageProgress, terminal_progress
from evaluation_checkpoints import BenchmarkCheckpoints


@contextmanager
def exclusive_evaluation(directory):
    """Avoid duplicate expensive gates and prune only without other gate users."""
    with (Path(directory) / ".benchmark_evaluation.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another safety benchmark is already running; manual review remains available") from error
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


class InputVersions:
    def __init__(self):
        self.versions = {}

    @staticmethod
    def version(path):
        canonical = os.path.realpath(path)
        try:
            return canonical, content_identity.file_version(Path(path))
        except FileNotFoundError:
            return canonical, None

    def add(self, path):
        path = os.path.abspath(os.path.expanduser(str(path)))
        if path not in self.versions:
            self.versions[path] = self.version(path)
        return self.versions[path][0]

    def validate(self):
        for path, version in self.versions.items():
            if self.version(path) != version:
                raise RuntimeError(f"Safety benchmark input changed during evaluation: {path}")

    def signature(self, *, excluding=()):
        values = {path: value for path, value in self.versions.items() if path not in excluding}
        return hashlib.sha256(json.dumps(values, sort_keys=True).encode()).hexdigest()


def selection_signature(cache):
    digest = hashlib.sha256()
    for index, face in enumerate(cache.faces):
        if not getattr(face, "label", None):
            continue
        digest.update(json.dumps((index, face.label, face.src_str, face.face_index, float(face.quality)),
                                 allow_nan=False).encode())
        values = np.asarray(face.embedding)
        digest.update(str(values.dtype).encode() + repr(values.shape).encode() + values.tobytes())
    return digest.hexdigest()


def selected_faces(cache, directory, *, progress=terminal_progress):
    """Cache the unchanged dominant-component sample selection independently."""
    import identity_evaluation
    memory_signature = selection_signature(cache)
    digest = hashlib.sha256(memory_signature.encode())
    for name in ("identity_evaluation.py", "identity_profiles.py", "evaluation_runtime.py"):
        digest.update(content_identity.content_sha256(Path(__file__).with_name(name)).encode())
    signature = digest.hexdigest()
    # The caller holds exclusive_evaluation, including during selection.
    with BenchmarkCheckpoints(Path(directory) / "benchmark_selection.sqlite3", signature) as store:
        store.prune()
        saved = store.get("samples-100")
        indexes = saved.get("indexes") if isinstance(saved, dict) else None
        if (isinstance(indexes, list)
                and all(type(i) is int and 0 <= i < len(cache.faces) for i in indexes)
                and len(set(indexes)) == len(indexes)):
            progress(f"Safety benchmark: reusing {len(indexes)} selected samples.")
            return [cache.faces[i] for i in indexes]
        result = identity_evaluation.select_faces(cache.faces, 100, progress=progress)
        if memory_signature != selection_signature(cache):
            raise RuntimeError("Benchmark cache changed during sample selection")
        by_id = {id(face): i for i, face in enumerate(cache.faces)}
        store.put("samples-100", {"indexes": [by_id[id(face)] for face in result]})
        return result


class GuardedCheckpoints:
    """Validate at block boundaries and invalidate contaminated namespaces."""
    def __init__(self, store, validate):
        self.store, self.validate = store, validate

    def check(self):
        try:
            self.validate()
        except Exception:
            self.store.invalidate()
            raise

    def get(self, stage):
        value = self.store.get(stage)
        if value is not None:
            self.check()
        return value

    def put(self, stage, payload):
        self.check()
        self.store.put(stage, payload)


class GateInputs:
    """Bind reusable results to actual tested faces and live reference files.

    The detector cache itself can grow during ingest. Only selected primary
    samples and faces belonging to benchmark cases affect this evaluation.
    """

    def __init__(self, cache, db, secondary_db, *, datasets, evidence_paths,
                 negative_examples, primary_faces):
        self.primary_faces = primary_faces
        self.cache = cache
        self.face_ids = tuple(id(face) for face in cache.faces)
        self.tested_ids = set()
        self.selection_fingerprint = selection_signature(cache)
        self.files = InputVersions()
        controls = {}
        def control(path):
            self.files.add(path)
            key = os.path.abspath(os.path.expanduser(str(path)))
            try:
                controls[key] = content_identity.content_sha256(path)
            except FileNotFoundError:
                controls[key] = None
        case_sources = set()
        for path in datasets:
            control(path)
            if not Path(path).is_file():
                continue
            with Path(path).open(newline="", encoding="utf-8-sig") as handle:
                for row in csv.DictReader(handle):
                    if row.get("source"):
                        case_sources.add(self.files.add(row["source"]))
        for path in evidence_paths:
            control(path)
        for database in (db, secondary_db):
            for sources in getattr(database, "prototype_sources", {}).values():
                for source in sources:
                    self.files.add(source)
            for field in ("pose_prototype_sources", "appearance_prototype_sources"):
                for labels in getattr(database, field, {}).values():
                    for sources in labels.values():
                        for source in sources:
                            self.files.add(source)
        for example in negative_examples:
            if example.get("source_path"):
                self.files.add(example["source_path"])

        digest = hashlib.sha256()
        digest.update(str(cache.config_fingerprint).encode())
        selected_ids = {id(face) for face in primary_faces}
        canonical = {}
        for face in cache.faces:
            source = canonical.setdefault(face.src_str, None)
            if source is None:
                source = canonical[face.src_str] = os.path.realpath(face.src_str)
            if id(face) not in selected_ids and source not in case_sources:
                continue
            self.tested_ids.add(id(face))
            self.files.add(face.src_str)
            fields = {key: value for key, value in vars(face).items()
                      if not isinstance(value, (np.ndarray, bytes))}
            digest.update(json.dumps(fields, sort_keys=True, allow_nan=False).encode())
            for key, value in sorted(vars(face).items()):
                if isinstance(value, np.ndarray):
                    digest.update(key.encode() + str(value.dtype).encode()
                                  + repr(value.shape).encode() + value.tobytes())
                elif isinstance(value, bytes):
                    digest.update(key.encode() + hashlib.sha256(value).digest())
        # Sample order participates in block checkpoint identity.
        digest.update(json.dumps([(face.src_str, face.face_index) for face in primary_faces]).encode())
        digest.update(json.dumps(controls, sort_keys=True).encode())
        digest.update(self.files.signature(excluding=controls).encode())
        self.fingerprint = digest.hexdigest()
        self.memory_fingerprint = self._memory_signature()
        self.validate()

    def _memory_signature(self):
        digest = hashlib.sha256(selection_signature(self.cache).encode())
        digest.update(str(self.cache.config_fingerprint).encode())
        for face in self.cache.faces:
            # Byte buffers are immutable; hashing only tested crops avoids
            # reading the full library's crops on every checkpoint boundary.
            if id(face) not in self.tested_ids:
                continue
            for key, value in sorted(vars(face).items()):
                digest.update(key.encode())
                if isinstance(value, np.ndarray):
                    digest.update(str(value.dtype).encode() + repr(value.shape).encode() + value.tobytes())
                elif isinstance(value, bytes):
                    digest.update(hashlib.sha256(value).digest())
                else:
                    digest.update(repr(value).encode())
        return digest.hexdigest()

    def validate(self):
        self.files.validate()
        if (tuple(id(face) for face in self.cache.faces) != self.face_ids
                or self._memory_signature() != self.memory_fingerprint):
            raise RuntimeError("Safety benchmark cached faces changed during evaluation")


def evaluate_blocks(items, evaluate, *, checkpoints=None, stage, progress=terminal_progress,
                    block_size=64):
    """Commit completed blocks only; an interrupted block is recomputed safely."""
    if block_size < 1:
        raise ValueError("block_size must be positive")
    status = StageProgress(stage, len(items), progress)
    results = []
    reused = 0
    for start in range(0, len(items), block_size):
        block = items[start:start + block_size]
        key = f"{stage}/block/{block_size}/{start}"
        saved = checkpoints.get(key) if checkpoints is not None else None
        if isinstance(saved, dict) and saved.get("count") == len(block) and isinstance(saved.get("rows"), list) and len(saved["rows"]) == len(block):
            rows = saved["rows"]
            reused += len(block)
        else:
            rows = []
            for offset, item in enumerate(block, 1):
                rows.append(evaluate(item))
                if offset < len(block):
                    status.update(start + offset, f"{reused} reused; {start + offset - reused} evaluated")
            if checkpoints is not None:
                checkpoints.put(key, {"count": len(block), "rows": rows})
        results.extend(rows)
        detail = f"{reused} reused; {start + len(block) - reused} evaluated"
        if checkpoints is not None:
            detail += "; completed blocks saved"
        status.update(start + len(block), detail,
                      force=bool(reused and start == 0))
    return results

"""Fresh benchmark detection using short-lived, read-only sorting workers."""

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import content_identity
import file_operations
import pipeline_paths
import sort_photos

CACHE_VERSION = 1
DEFAULT_CACHE_PATH = (
    pipeline_paths.SOURCE_REVIEW
    / "identity_evaluation"
    / "protected_detection_cache.pkl"
)


def _load_cache(path: Path, detector_signature: str) -> dict:
    try:
        with path.open("rb") as handle:
            payload = pickle.load(handle)
        if (payload.get("version") != CACHE_VERSION
                or payload.get("detector_signature") != detector_signature
                or not isinstance(payload.get("entries"), dict)):
            return {}
        return payload["entries"]
    except (OSError, ValueError, TypeError, pickle.PickleError, EOFError, AttributeError):
        return {}


def _save_cache(path: Path, detector_signature: str, entries: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        pickle.dump({
            "version": CACHE_VERSION,
            "detector_signature": detector_signature,
            "entries": entries,
        }, handle, protocol=pickle.HIGHEST_PROTOCOL)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    file_operations.sync_directory(path.parent)


def detect_cases(cases, *, detected_faces=None, batch_size=25, cache_path=None):
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    result = {} if detected_faces is None else dict(detected_faces)
    expected = {}
    for case in cases:
        key = os.path.realpath(str(case.source))
        digest = content_identity.content_sha256(case.source)
        if case.content_sha256 and digest != case.content_sha256:
            raise RuntimeError(f"Benchmark content changed: {case.source}")
        expected[key] = (case.source, digest)
    detector_signature = sort_photos.config_fingerprint()
    cache_file = Path(cache_path or DEFAULT_CACHE_PATH)
    cache_entries = _load_cache(cache_file, detector_signature)
    reused = 0
    for key, (_path, digest) in expected.items():
        entry = cache_entries.get(key, {})
        if not isinstance(entry, dict):
            continue
        faces = entry.get("faces")
        if entry.get("sha256") != digest or not isinstance(faces, list):
            continue
        if any(
            os.path.realpath(str(getattr(face, "src_str", ""))) != key
            or getattr(face, "content_sha256", "") != digest
            for face in faces
        ):
            continue
        result[key] = faces
        reused += 1
    if reused:
        print(f"Protected detection: reused {reused}/{len(expected)} unchanged image(s).",
              flush=True)
    pending = [key for key in expected if key not in result]
    env = os.environ.copy()
    env.update(sort_photos.DETECTION_WORKER_ENV_LIMITS)

    with tempfile.TemporaryDirectory(prefix="face_benchmark_detection_") as temporary:
        root = Path(temporary)
        for start in range(0, len(pending), batch_size):
            keys = pending[start:start + batch_size]
            paths = [expected[key][0] for key in keys]
            job, output = root / "job.pkl", root / "result.pkl"
            output.unlink(missing_ok=True)
            with job.open("wb") as handle:
                pickle.dump({"input_paths": [str(path) for path in paths],
                             "output_path": str(output),
                             "det_size": sort_photos.DET_SIZE[0]}, handle)
            print(f"Protected detection: starting {start + 1}-{start + len(keys)}"
                  f"/{len(pending)} (isolated batch)", flush=True)
            from pipeline_writer import child_process_options
            process = subprocess.run(
                [sys.executable, str(Path(sort_photos.__file__).resolve()),
                 "--detect-batch", str(job)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env=env, check=False, **child_process_options(),
            )
            if process.returncode != 0 or not output.is_file():
                raise RuntimeError(
                    f"Benchmark detection batch failed (exit {process.returncode}); "
                    f"no baseline can be certified.\n{process.stdout[-2000:]}")
            sort_photos.install_pickle_class_aliases()
            with output.open("rb") as handle:
                payload = pickle.load(handle)
            fingerprints = payload.get("fingerprints", {})
            diagnostics = payload.get("diagnostics", {})
            batch = {key: [] for key in keys}
            for key, path in zip(keys, paths):
                digest = expected[key][1]
                if (fingerprints.get(str(path), {}).get("sha256") != digest
                        or content_identity.content_sha256(path) != digest):
                    raise RuntimeError(f"Incomplete or changed benchmark image: {path}")
                status = diagnostics.get(str(path), "")
                if not (status in {"accepted_face", "accepted_face_recovery", "no_face_detected"}
                        or status.startswith("face_quality_review:")):
                    raise RuntimeError(f"Benchmark detection failed for {path}: {status}")
            for face in payload.get("faces", []):
                key = os.path.realpath(face.src_str)
                if key not in batch or face.content_sha256 != expected[key][1]:
                    raise RuntimeError("Benchmark worker returned invalid face provenance")
                batch[key].append(face)
            result.update(batch)
            for key in keys:
                cache_entries[key] = {
                    "sha256": expected[key][1],
                    "faces": batch[key],
                }
            _save_cache(cache_file, detector_signature, cache_entries)
            print(f"Protected detection: completed {start + len(keys)}/{len(pending)}",
                  flush=True)

    if detected_faces is not None:
        detected_faces.update(result)
    return {key: result[key] for key in expected}

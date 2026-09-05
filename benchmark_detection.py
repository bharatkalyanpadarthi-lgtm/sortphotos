"""Fresh benchmark detection using short-lived, read-only sorting workers."""

import os
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import content_identity
import sort_photos


def detect_cases(cases, *, detected_faces=None, batch_size=25):
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
            process = subprocess.run(
                [sys.executable, str(Path(sort_photos.__file__).resolve()),
                 "--detect-batch", str(job)],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                env=env, check=False,
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
            print(f"Protected detection: completed {start + len(keys)}/{len(pending)}",
                  flush=True)

    if detected_faces is not None:
        detected_faces.update(result)
    return {key: result[key] for key in expected}

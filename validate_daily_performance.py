"""Opt-in read-only old/new benchmark comparison; never ingest or file photos.

Use separate report directories with --module-root pointing to each checkout.
Native-model derived caches are redirected into the report directory. Inputs
must remain unchanged for a completed result to be considered comparable.
"""

import argparse
import inspect
import json
from pathlib import Path
import shutil
import sqlite3
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--module-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path.insert(0, str(args.module_root.resolve()))
    import sort_photos as sorter
    import review_unknown_identities as review
    import secondary_identity_matcher as secondary
    import benchmark_detection
    import content_identity
    import evaluation_enrollment
    import identity_evaluation
    import analysis_index
    import source_manifest

    output = args.report_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    evidence = [sorter.CACHE_FILE, sorter.IDENTITY_DB_FILE, secondary.DEFAULT_CACHE,
        sorter.IDENTITY_CONFIRMATIONS_FILE, sorter.IDENTITY_HARD_NEGATIVES_FILE,
        evaluation_enrollment.DEFAULT_PATH, identity_evaluation.DEFAULT_PROTECTED_SET,
        identity_evaluation.DEFAULT_PROTECTED_BASELINE]

    def versions():
        return {str(path): content_identity.content_sha256(path) if path.is_file() else None
                for path in evidence}

    before = versions()
    model = sorter.load_identity_db()
    verifier = secondary.load()
    if model is None or verifier is None or verifier.primary_signature != secondary.primary_signature(model):
        raise RuntimeError("Complete the active profile update before comparing benchmark implementations")
    cache = sorter.load_cache()
    benchmark_cache = output / "protected_detection_cache.pkl"
    if not benchmark_cache.exists():
        shutil.copy2(benchmark_detection.DEFAULT_CACHE_PATH, benchmark_cache)
    benchmark_detection.DEFAULT_CACHE_PATH = benchmark_cache
    # Copy hash metadata only, not images or the large detection/embedding DB.
    reference_database = output / "reference_lookup.sqlite3"
    with analysis_index.AnalysisIndex(reference_database) as destination, sqlite3.connect(
        sorter.analysis_index_file().resolve().as_uri() + "?mode=ro", uri=True) as original:
        destination.connection.executemany("INSERT OR REPLACE INTO content_versions VALUES(?,?,?)",
            original.execute("SELECT path,signature,sha256 FROM content_versions"))

    matcher = secondary.SecondaryMatcher(verifier, cache_path=output / "secondary_cache.pkl")
    options = dict(faces_by_source=review._faces_by_source(cache), progress=print)
    if "database" in inspect.signature(review.identity_confirmations.ReferenceIndex).parameters:
        options["database"] = reference_database
    start = time.perf_counter()
    reference_index = review.identity_confirmations.ReferenceIndex(**options)
    try:
        prototypes, counts = review.build_trusted_review_prototypes(model, cache,
            confirmations_path=sorter.IDENTITY_CONFIRMATIONS_FILE, reference_index=reference_index)
    finally:
        if hasattr(reference_index, "close"):
            reference_index.close()
    reference_seconds = time.perf_counter() - start
    timings, gates = [], []
    for _ in range(2):
        start = time.perf_counter()
        allowed, gate, message = review.prepare_automatic_review_gate(model, cache, matcher,
            requested=True, output_dir=output / "gate", review_prototypes=prototypes)
        timings.append(round(time.perf_counter() - start, 3))
        gates.append(gate)
        print(message, flush=True)
        if not allowed:
            raise RuntimeError(f"Benchmark did not pass: {gate}")
    after = versions()
    result = dict(module_root=str(args.module_root.resolve()), stable_inputs=before == after,
        input_digests=before, reference_seconds=round(reference_seconds, 3),
        reference_hashes_read=reference_index.hashes_read,
        reference_hashes_reused=reference_index.hashes_reused,
        trusted_counts=counts, gate_seconds=timings, report=gates[-1]["report"])
    source_manifest.write_json_atomic(output / "comparison.json", result)
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"report", "input_digests", "trusted_counts"}}, indent=2))
    if before != after:
        raise RuntimeError("Live inputs changed; report is diagnostic only, rerun after review finishes")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

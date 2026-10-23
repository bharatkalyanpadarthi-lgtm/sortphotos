"""Share identical strict primary scoring, never a whole policy verdict."""

from contextlib import contextmanager
import hashlib

import evaluation_runtime as runtime
from evaluation_checkpoints import BenchmarkCheckpoints


@contextmanager
def session(db, cache, faces, directory):
    import review_unknown_identities as review
    import identity_hard_negatives
    import sort_photos

    directory.mkdir(parents=True, exist_ok=True)
    with runtime.exclusive_evaluation(directory):
        # A deliberately conservative policy key. Independent model changes
        # do not affect this key; primary evidence/code/source changes do.
        policy = review.automatic_review_gate_signature(db)
        inputs = runtime.GateInputs(cache, db, None, datasets=(),
            evidence_paths=(sort_photos.IDENTITY_HARD_NEGATIVES_FILE,),
            negative_examples=identity_hard_negatives.load(
                sort_photos.IDENTITY_HARD_NEGATIVES_FILE)["examples"], primary_faces=faces)
        signature = hashlib.sha256((policy + inputs.fingerprint).encode()).hexdigest()

        def validate():
            inputs.validate_snapshot()
            if review.automatic_review_gate_signature(db) != policy:
                raise RuntimeError("Primary benchmark policy changed during evaluation")

        with BenchmarkCheckpoints(directory / "primary_scores.sqlite3", signature) as store:
            store.prune()
            checkpoints = runtime.GuardedCheckpoints(store, validate)
            try:
                yield inputs.snapshot_cache, inputs.snapshot_primary_faces, checkpoints
                inputs.validate()
                checkpoints.check()
            except BaseException:
                try:
                    inputs.validate()
                    checkpoints.check()
                except Exception:
                    store.invalidate()
                raise

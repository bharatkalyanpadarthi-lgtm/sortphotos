# Recognition hardening, 2026-09-05

## Changes

- Daily assignment accepts individual agreeing members, never a cluster's dissenters.
  Manual content decisions are consulted before automatic assignment, including
  confirmations of a new person whose profile has not yet been promoted.
- Exact content copies count as one independent consensus vote. Existing image
  thresholds and the separate one-recognized-frame video policy are unchanged.
- SQLite schema 4 ties detection and classification to independent content hashes.
  Same-size replacements, interrupted workers, and results for changed source bytes
  cannot silently become current. Older analysis rows miss safely; future database
  versions are refused rather than downgraded.
- Positive confirmations verify the original hash and selected face crop. Renames
  resolve by content. Ambiguous legacy group confirmations are not training evidence.
- Copy outcomes use a durable SQLite journal. A resume hint is not success: the
  destination must still exist, be inside the library, and match the source hash.
- Large images retain rotation recovery. Recovery boxes/keypoints are mapped to the
  original image and overlapping detections are merged geometrically. Tile recovery
  completes its current tile group instead of stopping after the first detected face.
- Recovery settings and model bytes participate in invalidation. Automatic-review
  gates also include the policy implementation fingerprint. Manual decisions remain
  independent of those invalidations.
- Matching snapshots normalize prototypes once per batch. SQLite hydration reconciles
  the cache once, and copy planning reuses validated indexed hashes. Daily summaries
  record step timings. Pixels are shared for decode, face quality, face detection,
  perceptual hashing and pixel hashing; separate later classifier passes still exist.
- Daily assignment, shared automatic acceptance, copy outcomes, and non-mutating
  evaluation now have dedicated modules. Existing CLI facades remain compatible.

## Evaluation

Protected cases support `content_sha256`, `group_id`, `expected_people` (pipe-separated)
and `expected_face_count`. All known identities in a group must be annotated; an
extra accepted person is an error. Missing or changed benchmark files fail validation.
Holdouts exclude matching bytes across renamed copies and selected source groups,
including primary prototypes, pose/appearance examples and secondary references.
Profiles with no remaining reference provenance cannot fall back to a leaked centroid.

New profile promotion requires complete protected coverage and a fresh-detection
baseline. An incumbent can remain usable while the draft is completed. The actual
daily cluster/assignment planner is evaluated without copying or archiving anything;
the independently gated secondary policy remains a separate measurement.

Examples, after manually verifying the protected CSV:

```zsh
.venv/bin/python identity_evaluation.py --golden-set /path/to/protected.csv --fresh-detection
.venv/bin/python identity_evaluation.py --golden-set /path/to/protected.csv --fresh-detection --lane pipeline
.venv/bin/python identity_evaluation.py --golden-set /path/to/protected.csv --fresh-detection --write-baseline /path/to/protected_identity_baseline.json
```

Use `face benchmark-review` for the existing review interface. Its group case form
now accepts all known people, total face count and source group. `Run Gate` creates
a first baseline using fresh detection, not cached matching alone.

## Validation

- Synthetic suite: 140 top-level checks, zero failures.
- Clean Git source snapshot: the same 140 checks pass using the existing tested
  Python environment, including the packaged video fallback model check.
- Focused architecture, daily recovery and manual-decision suites are included in
  that run. They cover the audit reproductions, copy interruption, file replacement,
  hash/provenance, group scoring, shadow execution and scalar/compiled equivalence.
- Python compilation and undefined-name checks pass for the changed modules.
- Runtime snapshot: 11 direct dependencies and 10 model files verified unchanged.
- Read-only integration audit: zero failures; cache coverage and report-only
  duplicate warnings remain. It found 58,127 protected originals, 58,217 current
  originals, 90 new, and no missing protected originals.
- Microbenchmark: 147 identities, eight prototypes each, 100 queries. Scalar scoring
  0.395s; compiled scoring 0.017s, about 23x faster. Rankings matched; maximum score
  difference was 3.8e-8. This is not a full-ingest throughput claim.

Run validation with:

```zsh
.venv/bin/python synthetic_integration_tests.py
.venv/bin/python -m unittest test_architecture_hardening test_daily_identity_recovery test_unknown_decision_persistence
.venv/bin/python verify_runtime.py
```

`requirements-macos-py312.lock` records the installed package versions, and
`runtime_manifest.json` records observed model checksums. No large recognition
weights, photo library, face embeddings, credentials, or local review decisions
belong in the repository. The small existing YuNet video fallback is bundled with
its upstream license, so a clean checkout retains video recovery behavior.
This is an environment snapshot, not a claim that a fresh installation on every
platform has been tested. External weights must be installed separately.

## Deliberately Not Claimed Complete

The real protected set still needs human-verified group, lookalike, normal and
swimwear coverage and its first valid baseline. No human identities or category
labels were fabricated. Live recognition precision/recall, a real video benchmark,
alternative-model promotion and a small real ingest canary remain unproven until
that evidence is ready. Source-batch/near-duplicate groups need annotation for an
independent calibration/test split; cached-library measurements are regression
checks, not an independent generalization estimate.

No production ingest, library move, model replacement, cache rebuild, automatic
training activation, or live Quick Review restart was performed during this work.
The existing large orchestration files remain; the safety-critical decisions have
been extracted, not every CLI/UI function. Do not replace that with a claim that
every possible bug or recognition error has been eliminated.

New terminal processes load these changes. Let any already-running review queue
finish before restarting its server; saved decisions are preserved.

## Benchmark persistence repair, 2026-09-06

Dashboard startup previously overwrote existing benchmark rows with older unknown
review enrollments. This could erase verified case types, nudity labels and source
groups even though the verification count stayed unchanged. Enrollment now adds
only absent candidates. Existing annotations, including pending cases and stale
content hashes requiring review, remain intact. Unchanged startup data is not
rewritten, and a second dashboard launcher checks the occupied port before seeding.

Three new regressions reproduce those failures and pass after the repair. The
focused suites pass 67 tests; the full synthetic suite passes all 140 checks.
Lost local annotations were restored from a pre-restart backup only where the
current row exactly matched the older enrollment and its original hash still
matched. Newer group annotations were retained. Two real-dataset seed runs kept
the same CSV hash, with 3,094 verified cases and all ten categories covered.
This establishes dataset readiness, not a passing recognition baseline.

The subsequent fresh-detection run was killed by macOS after 376 images;
memorystatus reported the evaluator at 40,935 MB with no paging space. The
benchmark had kept one native detector alive for its entire run, unlike daily
ingest's bounded workers. Fresh benchmark detection now invokes the existing
sorting worker in isolated batches of 25, validates every returned fingerprint,
and refuses partial, missing or failed worker output. It does not save the live
face cache or detection index. The CLI no longer loads the full library face
cache when fresh detection is requested.

The dashboard streams batch output and explicitly reports nonzero/signal exits;
a second gate click cannot start a concurrent evaluator. A 50-image real-data
canary spanning the former stopping point completed in two isolated batches.
The focused suites pass 72 tests, and all 140 synthetic checks pass. One synthetic
gate test now supplies temporary protected-set paths instead of implicitly
reading the user's real benchmark. These checks do not substitute for the full
protected activation result.

The evaluation CLI now binds its report and baseline to the dataset hash read at
the start. Changed annotations during loading or scoring block activation instead
of stamping old metrics with a newer CSV hash. Reports retain their evaluated
hash and indicate whether the dataset remained unchanged. Targeted tests cover
unchanged input and changes in both phases. Scoring progress is reported every
100 cases; category coverage is no longer labelled "Activation ready".

A concurrent live run exposed another synthetic-test isolation gap: two gate
tests could open the user's analysis database. They now use temporary analysis
databases and explicit temporary confirmed/protected datasets. All 140 synthetic
checks and 73 focused tests pass after these changes.

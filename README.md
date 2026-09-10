# Face Photo Sorting Pipeline

Recognition safety, performance changes, validation commands and remaining human
benchmark work are documented in [Architecture Hardening](ARCHITECTURE_HARDENING.md).

## Face Menu

The seven main choices have stable numbers, even when a review queue is empty:

1. Daily Ingest / Cache Run
2. Preview Daily Run
3. Status Dashboard
4. Quick Review Unknown Faces
5. Recover Missed Faces
6. Recover Missed Videos
7. Run High-Precision Nudity Check

Use `r` for Review Tools (saved dashboard, protected benchmark, cross-person
audit, manual confirmation, duplicates and uncertain-nudity review), `d` for
Diagnostics, and `a` for Advanced recovery and legacy tools. `b` returns to the
main menu. Opening a menu does not scan the SSD or run any processing.

All previous command keywords and aliases still work. `face process`,
`face process-new`, `face process-move` and `face sort` now resolve directly to
`face daily`, including its empty-inbox shortcut. `face commands` lists every
keyword. You can also open a submenu with `face review-tools`,
`face diagnostics` or `face advanced`.

`face review-dashboard` opens saved reports without rerunning recognition or
scanning photo folders. Its Unknown Identity link is the current Quick Review
**read-only report**; use `face unknown-review` for live decisions. Run
`face review-dashboard --refresh` explicitly to regenerate the duplicate
preview and count folders. The old `--no-refresh` flag remains accepted.

Health Check is a thorough diagnostic run, not a quick status read or cleanup.
Recover Known Faces remains an advanced tool with its own matching path; it
is not automatically chained after Quick Review. Legacy `process-all` still
scans `~/Pictures` by default, and `all-views` creates legacy hardlinked views.
Neither is part of normal daily ingest. This menu cleanup does not change
recognition thresholds, benchmark gates, photo routing or safety checks.

### Daily Progress

Option 1 shows compact progress by default: the current step, actual batch or
file counts where available, cache reuse, warnings and failures. Internal
model/provider setup, per-person reference lookups and repeated report paths
stay in the full worker log. Updates are throttled; percentages describe the
named substage, not a guessed completion percentage for the whole run.

The terminal prints the log location and a short final summary. Review totals
are explicitly labelled as including earlier runs. Safety checks, processing
order and recognition policies are unchanged. No extra library scans are
performed for progress reporting.

For troubleshooting, use `face daily --verbose` (or
`face daily --resume --verbose`). Full worker diagnostics are saved regardless
of display mode. Resume remains `face daily --resume`; interrupted steps do
not get marked completed.

## Storage Layout

The external SSD is the source of truth for photo data:

- Sorted library: `/Volumes/SSD 2TB/Photo Sort Data/sorted_all_pictures`
- Face references: `/Volumes/SSD 2TB/Photo Sort Data/Face References`
- Source review: `/Volumes/SSD 2TB/Photo Sort Data/_source_review`
- Intake: `/Volumes/SSD 2TB/To Process SSD`
- Incremental analysis DB: `/Volumes/SSD 2TB/Photo Sort Data/analysis_cache/analysis_index.sqlite3`

Per-Mac configuration lives in `~/.config/face-sort/paths.json`. Python tools
must use `pipeline_paths.py` instead of embedding paths under `~/Pictures`.
Compatibility links remain at `~/Pictures/sorted_all_pictures` and
`~/Pictures/Face References`, so older commands and Finder shortcuts still
work. Connect and mount `SSD 2TB` before running `face`.

Run the read-only migration/preflight validation with:

```sh
face health
```

Persistent detector and duplicate-cache paths can be audited after a storage
move with `python migrate_storage_paths.py`; it is a dry run unless `--apply`
is supplied.

## Identity Matching

Known people use a quality-weighted centroid plus diverse face prototypes.
Reference refreshes are incremental by person: unchanged folders reuse their
existing profile, while changed folders are rebuilt from current cached face
embeddings. Checkpoints are written to a staging database and the previous
complete database is backed up before atomic promotion.

Each profile keeps its intrinsic reference-spread threshold and then tightens
it against the nearest competing person's centroid and prototypes. Identity
decisions require both an allowed distance and a second-place person margin;
the same rule protects strict single-image matching, consensus matching,
Stage-B reassignment and anchor-cluster merging.

Refresh profiles after meaningful person-folder changes:

```sh
face rebuild-id
```

Run the read-only strict-match benchmark:

```sh
face identity-eval --lane strict --max-per-person 20 --fail-below-precision 0.98
```

Single-image auto-filing requires a high-quality, strongly separated match.
Otherwise the item remains available for review. Clusters with multiple
independent source images use the separate consensus lane. Video evidence uses
its dedicated frame-level decision policy.

## Incremental Analysis Cache

Duplicate fingerprints and reusable asset metadata are stored in
`~/.face_sort_cache/analysis_index.sqlite3`. Entries are reused only while
exact file size and modification time match; changed files are recalculated.
The index stores fingerprints, face detections and embeddings, identity
decision history, NudeNet evidence and an operation-history mirror. The JSONL
operation ledger remains authoritative for recovery, and the previous JSON
fingerprint cache remains a migration fallback.

Migrate current valid detector-cache records without rerunning the model:

```sh
face analysis-migrate --apply
```

Newly detected assets are read and decoded once to derive the face input,
content hash, pixel hash, dimensions and perceptual hash. NudeNet remains an
isolated model stage because its public API accepts a file path; its result is
reused from SQLite until the file signature or model-policy version changes.

## Protected Evaluation Gate

Generate the manually verified golden-set CSV template:

```sh
face identity-eval --create-golden-template \
  ~/.face_sort_cache/identity_golden_set.csv
```

After filling and verifying all required categories, run the combined identity,
face-presence and nudity gate:

```sh
face identity-eval \
  --golden-set ~/.face_sort_cache/identity_golden_set.csv \
  --baseline ~/.face_sort_cache/identity_golden_baseline.json
```

The gate refuses activation until the set includes verified known people,
look-alikes, side profiles, blurry/small faces, groups, unknown people,
no-face images, normal images, nudity and swimwear.

Explicit confirmations from `face unknown-review` are enrolled automatically
in `confirmed_unknown_identity_set.csv`. Identity profile rebuilds compare the
candidate matcher with the active matcher and refuse promotion if strict
precision drops below 99.9%, a new false accept appears, or recall regresses by
more than two percentage points.

Build the independent borderline verifier once (its model and cache live on
the configured external data volume):

```sh
face secondary-id
```

## Unknown Identity Learning

Daily ingest now uses the same independent-verifier preparation and safety
benchmark as Quick Review for images left unresolved by normal clustering.
It reuses detected faces instead of decoding every source again. Each recovered
image gets its own assignment; the other members of a rejected cluster are not
automatically carried into that person's folder. Multi-face images stay held.
The existing distance, margin, and quality limits are unchanged.

Daily recovery and Quick Review share a run-scoped confirmed-reference index
between the independent verifier and trusted-profile builder. Renamed files
are looked up by person, byte size (when recorded), and SHA-256 rather than
searching the person's entire folder for every confirmation. Each unchanged
candidate is hashed at most once within this shared verification run, even if
the separate bounded hash cache evicts it. No image pixels are retained.
The index checks size, modification/change timestamps, device, and inode before
reusing a hash, revalidates lookup hits, and still requires the confirmed face.
Lookup buckets refresh between stages; new candidate paths enter on the next
run. This is an in-memory run index, not a permanent exemption from verification.

Terminal progress now names reference indexing/verification, secondary and
trusted profiles, sample selection, and primary/independent/protected benchmark
scoring. It reports actual completed/total counts, elapsed time, and hashes
read/reused. Updates are throttled to avoid flooding the terminal. A cached
safety-gate hit is explicit; changed inputs still require evaluation. These
changes apply to newly started processes, not an already-running worker.

Saved foreground-face selections are checked before bulk benchmark scoring.
If the normal face cache has different crops, a read-only worker re-detects
only the affected benchmark images and must reproduce the exact saved crop
fingerprint. The user annotation, original image, and production face cache
remain unchanged; background faces are still counted for detection metrics.
The refreshed detections are reused by both protected scoring lanes. Missing,
ambiguous, changed-content, or failed detection results block automatic filing
without crashing the manual Quick Review dashboard. Interrupted evaluations
and results whose inputs changed mid-run are not cached as completed verdicts.
Benchmark scoring now prepares normalized reference matrices and indexed
source/content holdouts once per session. Primary, protected, and independent
matching reuse these snapshots instead of resolving and hashing every reference
for every comparison. The protected lanes share one primary snapshot. Missing
provenance still excludes a reference; identical content and explicit source
groups remain held out. Snapshot validation catches replacements and changed
symlinks before results can authorize filing. Distance/quality thresholds and
sample coverage are not reduced.

Dominant-component sample selection also has its own checksummed cache keyed
by selection code, face labels, quality, embeddings and order. An unchanged
launch does not repeat the pairwise component-selection work. Annotation files
are compared by content, so a no-op rewrite alone does not trigger re-evaluation.

Daily/Quick Review gates keep resumable work in
`identity_audits/unknown_review/benchmark_checkpoints.sqlite3` under the configured
source-review directory. Primary and independent scoring commit blocks of 64
cases, including skipped/rejected outcomes. Protected scoring reuses completed
stages. An interrupted block is recomputed; previous complete blocks are reused
only for the same versioned inputs. Progress distinguishes reused cases from
new work. Checkpoints alone never authorize automatic filing.
File and in-memory inputs are revalidated at block boundaries; detected drift
discards the affected namespace, while an ordinary interruption retains valid
work. A per-directory advisory lock prevents concurrent gates. SQLite retains
the current namespace plus two recent namespaces, reusing freed pages.

The final gate signature covers decision code/settings, profile provenance,
annotation/evidence content, tested cached faces/crops, detector configuration,
and current source/reference versions. Changed inputs require fresh validation;
unchanged completed pass or fail verdicts are reused. Routine gates stop after
a definitive primary/protected failure instead of running further expensive
checks that cannot authorize filing. Manual review remains available. Explicit
diagnostic/profile-promotion evaluations still run full checks, and promotion
still requires fresh protected detection. No unvalidated profile is enabled to
make a daily run appear faster.

Run the isolated reference safety, reuse, progress, and recovery regressions:

```sh
.venv/bin/python -m unittest discover -p 'test_*.py'
```

Confirmed exact-content replays are reused only while the matching organized
copy exists under that person's `photos` folder (including `photos/nude`).
Missing/different copies, ignored/junk decisions, and explicitly rejected
candidates cannot authorize automatic reuse. Newly automatic matches do not
become trusted training examples. Ordinary verified copying, duplicate checks,
source guards, and recoverable archiving still handle the actual files.

The daily run writes `*_identity_recovery.json` beside its intake-review CSV,
with per-image decisions and verifier/benchmark status. Preview a previous
review report without moving images or changing labels:

```sh
.venv/bin/python preview_daily_identity_recovery.py /path/to/daily_run_report.csv
```

The preview reads SQLite detections only after verifying the current image
hash. It can refresh the verifier cache and write benchmark/report files, but
does not apply its proposed assignments. Failed or missing safety benchmarks
leave new automatic matches in review; thresholds are not relaxed to pass.

`face unknown-review` opens the continuous Quick Review dashboard. Before the
page appears, a cached full-library safety gate is checked. Faces are filed
automatically through independently agreeing matchers or a separately validated
strong single-photo lane. Cluster consensus still requires independent support.
Recovery respects per-person thresholds, the independent verifier's accepted
verdict, and the existing strict single-image separation margin. Two models of
the same image do not substitute for independent source-image consensus; the
rescue lane cannot override either model's calibration. Both matchers are
validated with source-group holdouts; verified unknown cases must also be
rejected. The strong single-photo lane stays off unless its own evaluation has
zero incorrect accepts and the complete gate passes. Global
thresholds are never lowered, and automatically filed faces do not become
trusted confirmation examples. When a manual confirmation corrects option 1,
that face is retained as a hard-negative lookalike guard so the same mistake is
less likely in later batches. The dashboard then presents one
visual cluster at a time, automatically advances after a decision, and
loads successive 500-file batches without restarting the command. Every
single-face item shows the top three people with distance, second-place margin,
quality, pose and, for borderline matches, the independent `buffalo_l` result.
Each remaining item also shows why it needs review: weak confidence, a similar
alternative, poor face quality, matcher disagreement, independent-verifier
rejection, or a blocked safety gate.
These explanations reuse computed evidence and do not rerun either detector.

To validate without reconciling, moving, or filing photos:
```bash
python review_unknown_identities.py --validate-auto-only
```
This diagnostic checks the independent matcher even when the primary check
fails. A diagnostic success alone never overrides a failed primary check.

- `1`, `2`, or `3` confirms the entire cluster as that suggested person.
- Enter confirms an existing or new person name for the entire cluster.
- `U` keeps the cluster unknown, `J` moves it to recoverable junk, and `N`
  temporarily skips it for the current session.
- Left and Right Arrow move between pending clusters.
- `Finish Review` drains queued actions, saves decisions and cache, refreshes
  identity profiles once, and runs the protected activation gate.
- Unsupported items are moved recoverably to `multi_face_review`,
  `no_usable_face`, `face_quality_review`, or `processing_failed`; they no
  longer block later unknown batches.
- `Confirm` safely organizes and enrolls trusted examples.
- `Not Person` stores persistent hard-negative look-alike evidence.
- Manual decisions are stored by exact content hash. The latest decision applies
  to every identical copy, including renamed files; changed bytes never inherit
  the old file's identity. Legacy decisions are migrated with a backup on save.
- `Keep Unknown` hides the image from manual review across profile updates.
  When profiles change, automatic review can silently recheck those bytes once.
  Only an independently verified, benchmark-gated match can file the image;
  rejected or failed checks leave the manual choice intact. Failures can retry.
- `Ignore` remains ignored across profile updates and is not auto-rechecked.
- same-batch consensus requires a real operation-ledger or nested-folder batch,
  a face-embedding cluster, independent sources and agreement; filename
  proximity alone is never accepted.
- the primary and secondary matchers must agree before a borderline face is
  auto-filed.

Pose profiles retain frontal, left-profile and right-profile prototypes. The
one-time legacy migration is memory bounded:

```sh
face pose-refresh
```

## Module Boundaries

- `sort_photos.py`: CLI orchestration, clustering and compatibility API.
- `identity_profiles.py`: reference selection, prototypes and calibrated matching.
- `identity_hard_negatives.py`: explicit look-alike rejection evidence.
- `secondary_identity_matcher.py`: independent borderline verification.
- `source_batch_consensus.py`: same-batch, same-face corroboration policy.
- `appearance_profiles.py`: cached low-light and capture-era prototype labels.
- `face_detection.py`: face geometry, quality and recovery views.
- `asset_processing.py`: single-read decode and reusable image metadata.
- `analysis_index.py`: incremental SQLite analysis/history store.
- `routing_policy.py`: duplicate-category and nudity routing decisions.
- `file_operations.py`: atomic copy and source-preserving verification.
- `operation_ledger.py`: authoritative recoverable operation journal.
- `evaluation_dataset.py` / `identity_evaluation.py`: protected activation gates.
- `evaluation_enrollment.py`: automatic benchmark enrollment for confirmations.
- `review_identity_benchmark.py`: local human-verification dashboard for the
  protected identity, face-detection, and nudity benchmark.

Review and activate the protected benchmark:

```bash
face benchmark-review
```

The first run seeds unverified candidates. Nothing becomes trusted until the
`Manually verified` box is selected. `Run activation gate` creates the first
baseline only after every required case type has verified coverage; later runs
block identity-profile regressions against that baseline.

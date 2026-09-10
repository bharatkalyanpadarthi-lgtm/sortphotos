# Face References

Daily image sorting and Quick Review share the same validated identity database.
The configured Face References folder can supplement those profiles. It no longer
injects the legacy `reference_centroids.pkl` into daily cluster matching.

## Normal Use

Continue using `face` for daily sorting or Quick Review. Their startup checks
reuse the reference analysis cache. To explicitly refresh references, use:

```sh
face refs
```

The first scan detects references in bounded subprocess batches. Subsequent
checks reuse SQLite detections and verified content hashes. Renaming a file or
adding an identical copy does not require detecting its face again. A changed
file or detector version invalidates that cached analysis. Verified selections
are cached too; matching-code or profile changes invalidate those decisions.
An unchanged profile does not require another activation benchmark. Preview and
validation-only commands do not refresh or activate reference profiles.

## Safety

- A reference-folder label is a hint, not a manual confirmation.
- Only established canonical identities or completed, configured merge aliases
  are eligible. Unknown folder names are held, not guessed.
- Group images, low-quality faces, conflicting labels, and weak identity matches
  are not admitted. The existing strict distance, margin, quality, minimum-source
  and hard-negative checks remain in effect.
- Each identity can gain at most two supplemental prototypes. Its core vectors,
  manual confirmations, centroid and source counts are preserved.
- Reference supplements cannot authorize further supplements through themselves.
- Every changed profile passes the normal protected activation gate before the
  active database is replaced. Interrupted or failed validation leaves the
  previous database available. Original and reference image files are untouched.
- A rejected candidate is not repeatedly benchmarked on every startup. It is
  retried when its evidence or validation inputs change, or with `face refs --retry`.

The status dashboard reports active verified examples rather than the size of the
legacy cache. Detailed scan reasons are kept in `face_reference_analysis.sqlite3`
beside the configured analysis index, under `metadata.last_reference_scan`.
The activation report remains in `_source_review/identity_evaluation/`.

Keep the SSD connected. If the reference folder or index is unavailable, the
last validated profiles remain available. Ambiguous references still require
verification; this is not a promise of perfect recognition.

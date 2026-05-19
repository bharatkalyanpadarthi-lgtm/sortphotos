# Photo Sorting Pipeline — User Guide

A reference for sorting your photo collection by person, using `sort_photos.py` and `fix_clusters.py` on your Mac Mini M4.

---

## What you have

Two scripts in `~/scripts/face-crop/`:

| Script | Purpose |
|---|---|
| `sort_photos.py` | Main pipeline: detect faces → cluster by identity → label → organize originals into per-person folders, with sharp/blurred split and duplicate detection |
| `fix_clusters.py` | Quick fix-up tool: manually merge or rename clusters without re-running the pipeline |

Plus a persistent cache at `~/.face_sort_cache/cache.pkl` that remembers everything across runs (file fingerprints, embeddings, labels, crops). This is what makes future runs fast and self-improving.

---

## TL;DR — Daily commands

**Run the main pipeline:**
```bash
cd ~/scripts/face-crop
source .venv/bin/activate
python sort_photos.py
```

**Fix mistakes after a run:**
```bash
cd ~/scripts/face-crop
source .venv/bin/activate
python fix_clusters.py
```

**Start completely over (wipe memory):**
```bash
cd ~/scripts/face-crop
source .venv/bin/activate
python sort_photos.py --reset-cache
```

---

## Folder layout

**Input** (you put photos here): `~/Pictures/To Process/`

**Output** (script creates this): `~/Pictures/sorted/`
```
sorted/
├── face_clusters/                ← face crops + montages, named by person
│   ├── Syamala/
│   ├── Syamala_montage.jpg
│   ├── Yami Gautam/
│   └── Yami Gautam_montage.jpg
│
├── photos_by_person/             ← YOUR ORIGINAL PHOTOS, sorted by person
│   ├── Syamala/                  ← sharp keepers
│   │   ├── IMG_1260.PNG
│   │   ├── IMG_1262.PNG
│   │   └── _duplicates/          ← near-duplicate burst shots
│   ├── Yami Gautam/
│   ├── unknown/                  ← faces that didn't match anyone
│   └── _blurred/                 ← shots with blurry faces
│       ├── Syamala/
│       │   └── _duplicates/
│       └── Yami Gautam/
│
└── _clusters.csv                 ← full manifest of every face → person mapping
```

---

## The batching workflow (for 20K+ images)

Why batch: 20,000 images all at once takes 3–4 hours of detection and uses a lot of RAM. If anything crashes mid-way, you lose progress. Batching avoids this and the cache makes it seamless.

### Setup folders once

```bash
mkdir -p ~/Pictures/photo_pool
mkdir -p ~/Pictures/processed_batches
```

Move ALL your 20K images into `~/Pictures/photo_pool/`. This is your master pool.

### Per-batch loop (repeat 4 times for 20K, 5K per batch)

**Step 1** — empty the input folder:
```bash
rm -rf ~/Pictures/To\ Process
mkdir -p ~/Pictures/To\ Process
```

**Step 2** — move 5,000 images from the pool to the input folder:
```bash
ls ~/Pictures/photo_pool | head -5000 | while read f; do
  mv "~/Pictures/photo_pool/$f" "~/Pictures/To Process/"
done
```

**Step 3** — run the pipeline:
```bash
cd ~/scripts/face-crop
source .venv/bin/activate
python sort_photos.py
```

(Answer `y` when asked if you want to wipe the previous output.)

The script will:
- Detect faces in the new 5K images (~30 min on M4)
- Cluster them, using your cache from previous batches as anchors
- Auto-recognize people from earlier batches without prompting
- Prompt you only for genuinely new people
- Run anchor-cluster-merge and close-pair review at the end
- Copy originals into `photos_by_person/`
- Save updated cache

**Step 4** — fix any mistakes:
```bash
python fix_clusters.py
```

Look at any `person_NNN` folders that should have been merged into existing labels (e.g. you spot a Syamala variant that didn't merge). Use `merge N M` or `rename N <name>`.

**Step 5** — move processed batch out of input folder:
```bash
mv ~/Pictures/To\ Process/* ~/Pictures/processed_batches/
```

Now go back to Step 1 for the next 5K.

### Final consolidation pass

After all 4 batches, the cache has labels for everyone. Move all 20K originals back into `~/Pictures/To Process/`:

```bash
mv ~/Pictures/processed_batches/* ~/Pictures/To\ Process/
```

Run one final time:
```bash
python sort_photos.py
```

This run is fast — detection is skipped entirely (everything cached), clustering uses all known labels as anchors, and you get one unified, fully-labeled `photos_by_person/` with all 20K sorted.

---

## Using `sort_photos.py` in detail

### Command-line flags

```bash
python sort_photos.py                          # default: ~/Pictures/To Process
python sort_photos.py /path/to/input           # custom input folder
python sort_photos.py /path/to/in /path/to/out # custom in + out
python sort_photos.py --no-label               # skip interactive labeling
python sort_photos.py --no-dedup               # skip duplicate detection
python sort_photos.py --no-review              # skip close-pair review
python sort_photos.py --reset-cache            # wipe cache, start fresh
```

### What happens during a run

1. **Cache load**: reads `~/.face_sort_cache/cache.pkl`
2. **Scan**: lists all images in input folder
3. **Cache hit check**: matches each file by (mtime, size). Cached files skip detection.
4. **Detection**: runs InsightFace on new/changed images only
5. **Stage A clustering**: strict DBSCAN
6. **Auto-merge by prior labels**: clusters sharing a known label get fused
7. **Centroid merge**: very-close clusters fused
8. **Anchor pass**: unknown faces pulled toward labeled embeddings
9. **Stage B**: remaining unknowns assigned to nearest cluster
10. **Final centroid merge**
11. **Initial naming**: clusters with prior labels get those names; others get `person_001`, `person_002`, etc.
12. **Cluster crops + montages written**
13. **Interactive labeling**: prompts only for `person_NNN` clusters
14. **Anchor-cluster merge**: unlabeled clusters auto-folded into close labeled ones
15. **Close-pair review**: shows remaining suspicious cluster pairs
16. **Originals copy**: with sharp/blurred split + duplicate detection
17. **Manifest written**
18. **Cache saved**

### During interactive labeling

For each unlabeled cluster, the montage opens in Preview. You see:
```
[3/12] person_003  (8 faces)
  Name: _
```

Type one of:
- A **new name** (e.g. `Sravani`) → folder gets renamed, label remembered for future runs
- An **existing name** (e.g. `Syamala`) → auto-merges into that folder
- **Empty Enter** → skip (keeps `person_NNN`, won't be remembered)
- **`q`** → stop labeling, continue with the rest of the pipeline

### During close-pair review

After labeling, the script offers any cluster pairs whose centroids are within 0.50 cosine distance for review:
```
distance=0.421  |  'Syamala'  vs  'person_005'
  Same person? (y/n/q): _
```

Both montages open in Preview. Type `y` to merge, `n` (or Enter) to keep separate, `q` to stop reviewing.

---

## Using `fix_clusters.py` in detail

Run this any time after `sort_photos.py` if you spot mistakes. It updates everything consistently — folders, CSV manifest, AND the cache.

### Inside the tool

You'll see a numbered list of clusters:
```
  #  name                       faces  photos
--------------------------------------------------
  1  Syamala                       43      78
  2  Yami Gautam                   35      62
 11  person_001                    12      18
```

Available commands:

| Command | What it does |
|---|---|
| `list` | Re-show the cluster list (numbers shift after merges) |
| `open N` | Open cluster #N's face montage in Preview |
| `show N` | Open cluster #N's photos folder in Finder |
| `merge N M` | Merge cluster #N into cluster #M |
| `rename N <name>` | Rename cluster #N (auto-merges if name already exists) |
| `q` | Save changes and quit |

### Common scenarios

**Scenario 1 — Merging a `person_NNN` into an existing label:**
```
> open 11        # look at person_001 montage
> open 1         # look at Syamala montage
> merge 11 1     # confirm with y
```

**Scenario 2 — Labeling a `person_NNN` you weren't prompted for:**
```
> open 13
> rename 13 Anushka
```

**Scenario 3 — Renaming because of typo:**
```
> rename 5 Sravani    # was previously "Sraavani"
```

**Scenario 4 — Browsing originals before deciding:**
```
> show 13              # opens Finder to that person's photos folder
```

When done, type `q`. The script applies all your changes to the CSV manifest and the cache. Future `sort_photos.py` runs will respect the new labels.

---

## Tuning knobs

If results aren't right, edit the top of `sort_photos.py`. The most useful knobs:

| Knob | Default | When to change |
|---|---|---|
| `MIN_DET_SCORE` | 0.55 | Lower (0.45) to catch more borderline faces; raise (0.65) to be stricter |
| `MIN_FACE_PX` | 70 | Lower (50) to include smaller faces in background |
| `MIN_SHARPNESS` | 40 | Lower (30) to include more blurry faces; raise (60) to keep only sharp |
| `STAGE_B_MAX_DIST` | 0.50 | Raise (0.55) if too many faces end up in `unknown`; lower (0.45) if wrong people get pulled in |
| `MERGE_CENTROID_DIST` | 0.38 | Raise (0.42) to merge more aggressively; lower (0.35) if different people getting merged |
| `ANCHOR_CLUSTER_MERGE_DIST` | 0.42 | Raise (0.45) for more aggressive anchor merging; lower (0.40) if labeled person is "absorbing" wrong clusters |
| `REVIEW_CLOSE_PAIRS_DIST` | 0.50 | Raise (0.55) to review more borderline pairs |
| `QUALITY_THRESHOLD` | 0.50 | Raise (0.55) to mark more photos as "blurred"; lower (0.40) if too strict |
| `PHASH_THRESHOLD` | 8 | Lower (4) for stricter dedup; raise (12) for more aggressive grouping |

Order to try if recall (missing photos of known people) is too low:
1. Raise `STAGE_B_MAX_DIST` to `0.55`
2. Raise `ANCHOR_CLUSTER_MERGE_DIST` to `0.45`
3. Raise `REVIEW_CLOSE_PAIRS_DIST` to `0.55`

Order to try if precision is too low (different people getting merged):
1. Lower `MERGE_CENTROID_DIST` to `0.34`
2. Lower `ANCHOR_CLUSTER_MERGE_DIST` to `0.38`
3. Lower `STAGE_B_MAX_DIST` to `0.45`

---

## Common operational tasks

### "I added 100 new photos to my collection"

```bash
# Drop the new photos into ~/Pictures/To Process (don't remove the existing ones — cache will skip them)
cd ~/scripts/face-crop
source .venv/bin/activate
python sort_photos.py
```

The cache will skip the existing photos and only process the new 100. Known people get auto-labeled. New people prompt for names.

### "I noticed cluster X and Y are the same person"

```bash
cd ~/scripts/face-crop
source .venv/bin/activate
python fix_clusters.py
# Inside: merge X Y
```

### "I want to wipe everything and start completely over"

```bash
rm -rf ~/Pictures/sorted
rm -rf ~/.face_sort_cache
python sort_photos.py
```

### "I want to keep my labels but re-detect because I changed thresholds"

The cache stores a config fingerprint. If you change `MIN_DET_SCORE`, `MIN_FACE_PX`, `MIN_SHARPNESS`, `MODEL_NAME`, `DET_SIZE`, `CROP_SIZE`, or `PADDING_RATIO`, the script automatically re-detects but **keeps your labels**.

### "I want to re-do labeling from scratch on existing photos"

```bash
# Wipe just the cache labels, keep file metadata + embeddings
# (currently the script doesn't separate these — easiest is to wipe and re-run)
python sort_photos.py --reset-cache
```

### "I changed my mind about a label"

Use `fix_clusters.py` to rename:
```
> rename 5 NewName
```

---

## Output details

### `face_clusters/<name>/` contents

JPEGs of cropped faces, named like:
```
IMG_1234__face0_q87_c92.jpg
```

Where:
- `face0` = which face in the source image (0 = first detected)
- `q87` = quality score 0–99 (higher = sharper, more frontal, larger)
- `c92` = centroid similarity 0–99 (higher = more confidently belongs to this cluster)

Best shots sort to the top. If you see a `c<70` shot, it's borderline — possibly a wrong assignment.

### `_clusters.csv` columns

| Column | Meaning |
|---|---|
| `source` | Full path to original image |
| `face_index` | Which face in that image (0, 1, 2, ...) |
| `det_score` | Detection confidence 0–1 |
| `quality` | Combined quality score 0–1 |
| `person` | Cluster label (real name or `person_NNN`) |
| `centroid_similarity` | Cosine similarity to cluster centroid |
| `from_cache` | `yes` if loaded from cache, `no` if newly detected |

Open in Numbers or Excel to filter, search, or audit.

---

## Troubleshooting

### Pipeline crashes mid-run

Detection is the slow step. If it crashes, the cache hasn't been saved yet — but file detection is the only loss. Just re-run:
```bash
python sort_photos.py
```

Files that were processed will still be in `~/Pictures/To Process/` and will be re-detected from scratch.

### "antelopev2" model issues

If you ever see `assert 'detection' in self.models` errors:
```bash
ls ~/.insightface/models/antelopev2/
# should show .onnx files directly. If you see another antelopev2/ folder inside:
mv ~/.insightface/models/antelopev2/antelopev2/* ~/.insightface/models/antelopev2/
rmdir ~/.insightface/models/antelopev2/antelopev2
```

### CoreML errors

The script uses CPU-only deliberately — Apple Silicon has a known CoreML bug with RetinaFace. Don't change `PROVIDERS` away from `["CPUExecutionProvider"]`.

### Cache file getting huge

Each face uses ~30 KB cached. For 60K faces, expect ~2 GB. Check:
```bash
ls -lh ~/.face_sort_cache/cache.pkl
```

If it's a problem, you can wipe and rebuild — but you'll lose your labels. Better long-term: I can refactor to evict old crops while keeping embeddings + labels. Tell me if cache size becomes painful.

### "I labeled the wrong cluster"

Use `fix_clusters.py` to rename. The cache propagates the new label.

### "Same person split into 5 clusters even after merge thresholds maxed"

Some people genuinely have very different embeddings across photos (heavy makeup change, age range, etc.). The fix is `fix_clusters.py` — manually merge them once. After the merge, the cache has anchor embeddings covering all the variants, and future runs handle them automatically.

### "Different people getting merged"

Lower `MERGE_CENTROID_DIST` and `ANCHOR_CLUSTER_MERGE_DIST` (try 0.34 and 0.38), then `--reset-cache` and re-run.

---

## File locations cheat sheet

| What | Where |
|---|---|
| Scripts | `~/scripts/face-crop/sort_photos.py`, `~/scripts/face-crop/fix_clusters.py` |
| Python virtualenv | `~/scripts/face-crop/.venv/` |
| Input photos | `~/Pictures/To Process/` |
| Output | `~/Pictures/sorted/` |
| Persistent cache | `~/.face_sort_cache/cache.pkl` |
| Face detection model | `~/.insightface/models/antelopev2/` |

---

## Activating the environment (every session)

Always run these before invoking any script in a fresh Terminal:

```bash
cd ~/scripts/face-crop
source .venv/bin/activate
```

You'll know the venv is active when your prompt starts with `(.venv)`.

---

## Realistic expectations

- **Recall on people you've labeled**: 95%+ after a few labeling rounds
- **Profile shots vs frontal of someone you've only seen frontally**: model limit, will sometimes split
- **Twins**: model can't reliably separate them
- **Children at very different ages**: may split (a 3-year-old vs 6-year-old looks different to the model)
- **Heavy makeup change / glasses on/off**: usually handled
- **Lighting changes (outdoor pink saree vs indoor blue saree)**: handled if you label both variants once; the cache learns

The system gets better over time as you correct mistakes and re-run. The cache is doing real work for you.

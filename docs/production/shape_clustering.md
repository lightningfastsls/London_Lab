# Production Shape Clustering — the shape "letter" per USV

> **What this is** — The system that assigns each USV a *shape* label ("letter")
> based on the geometry of its frequency contour (a chevron vs a jump vs a flat
> tone), with pitch, duration, and time-position factored OUT by registration.
> **Status: CURRENT.** Two production representations coexist by the
> 2026-06-04 **HYBRID** decision: (1) a **soft-DTW K=20 alphabet** —
> `models/shape_kmeans/k20_softdtw.joblib` (+ `k20_softdtw_letters.parquet`) —
> the labeled-set metric that best matches human shape labels on *jump*/*complex*;
> (2) an **elastic-FPCA coordinate system** — `models/shape_fpca/elastic_fpca.joblib`
> (+ `elastic_fpca_scores.parquet`, 67,337 rows × {5 amp + 3 phase}) — the O(N)
> full-corpus coordinate producer consumed by the downstream grammar/manifold work.
> **Historical / superseded:** the registration-Euclidean K=20 model `k20.joblib`
> (NOT on disk — see Gotchas). The whole **VAE family is CLOSED for shape
> clustering** — see [Contour-VAE](production_vae.md).

The shape pipeline answers "what *shape* is this call's pitch contour?" after
deliberately discarding *how high* (pitch), *how long* (duration), and *where in
the file* (time) it is. Those nuisance axes are removed by **registration**
(subtract mean kHz, resample the active span to 50 points). What remains is a
50-point unit-less shape curve per call. Two methods then operate on those curves:
**soft-DTW** (elastic warp-aligned distance — the best human-anchored *metric*)
and **elastic FPCA** (a low-dimensional *coordinate system* over the same curves).

Sibling docs: [CNN detection pipeline](cnn_detection_pipeline.md) ·
[batch detection](batch_detection.md) · [corpus constants](corpus_constants.md) ·
[Contour-VAE](production_vae.md).

---

## 1. Operate

### 1.1 What you can do with these artifacts

| Goal | Use | Artifact |
|------|-----|----------|
| Look up the shape letter (0–19) already assigned to a call | Read the parquet | `models/shape_kmeans/k20_softdtw_letters.parquet` |
| Get a call's continuous shape coordinates (amp/phase PCs) | Read the parquet | `models/shape_fpca/elastic_fpca_scores.parquet` |
| Assign a letter to a NEW ridge | Load joblib, soft-DTW to centroids | `models/shape_kmeans/k20_softdtw.joblib` |
| Re-evaluate which representation best matches humans | Run the gate harness | `scripts/experiments/eval_shape_human_anchored.py` |
| Rebuild the alphabet / FPCA from ridges | Run the builders | `build_softdtw_alphabet.py` / `build_elastic_fpca.py` |

Both parquet files are **per-call lookup tables keyed by `(wav_stem, call_id)`**.
For the common case — "what letter / what coordinates does call X have?" — you do
**not** run any model; you read the parquet. Re-running a model is only needed to
score a call that is not already in the 67,337-row corpus.

### 1.2 Output schemas (read these first)

**`models/shape_kmeans/k20_softdtw_letters.parquet`** — 67,337 rows × 4 cols
(verified on disk):

| Column | Type | Meaning |
|--------|------|---------|
| `wav_stem` | str | Recording stem, e.g. `2024-09-30_11-22-17_0000053` |
| `call_id` | int | **1-indexed** call number within that recording |
| `cohort` | str | `5970` / `3452` / `9252` / `lab_131204` |
| `softdtw_letter` | int32 | Shape letter, integer in **[0, 19]** (the K=20 cluster id) |

Letters are bare integers, not glyphs — there is no A/B/C mapping. A letter is
just "which of the 20 soft-DTW barycenters is this ridge closest to". Letters are
**not** ordered or named; cluster 6 is not "more chevron" than cluster 5.

**`models/shape_fpca/elastic_fpca_scores.parquet`** — 67,337 rows × 11 cols
(verified on disk):

| Column | Type | Meaning |
|--------|------|---------|
| `wav_stem` | str | Recording stem |
| `call_id` | int | **1-indexed** call number |
| `cohort` | str | Cohort id |
| `amp_pc1` … `amp_pc5` | float64 | Amplitude (vertical) FPCA scores — the *shape* axes |
| `phase_pc1` … `phase_pc3` | float64 | Phase (warp) FPCA scores — the *timing/warp* axes |

The 5 amplitude PCs capture how the curve's *shape* deviates from the elastic
Karcher mean; the 3 phase PCs capture how its *timing* (warp) deviates. These 8
numbers are the coordinate system adopted for the downstream grammar / manifold
analyses (WS-B/C/D/E).

### 1.3 Environment

```bash
# repo root = /home/shachar/projects/mickey_london_lab ; interpreter = .venv/bin/python
.venv/bin/python -c "import tslearn, fdasrsf, joblib, pandas; \
  print('tslearn', tslearn.__version__, '| fdasrsf', fdasrsf.__version__)"
# Expect fdasrsf 2.6.9 (prebuilt wheel — no Fortran/C toolchain needed).
```

`tslearn` (soft-DTW) and `fdasrsf` 2.6.9 (elastic FPCA) are the two non-stdlib
dependencies. Both are in `.venv`.

### 1.4 Worked example — look up and assign a shape letter

**(a) Look up an existing call's letter and coordinates** (no model needed):

```bash
.venv/bin/python - <<'PY'
import pandas as pd
letters = pd.read_parquet("models/shape_kmeans/k20_softdtw_letters.parquet")
coords  = pd.read_parquet("models/shape_fpca/elastic_fpca_scores.parquet")

stem, cid = "2024-09-30_11-22-17_0000053", 1
row_l = letters[(letters.wav_stem == stem) & (letters.call_id == cid)]
row_c = coords [(coords.wav_stem  == stem) & (coords.call_id  == cid)]
print("soft-DTW letter:", int(row_l.softdtw_letter.iloc[0]))
print("amp PCs:", row_c[[f"amp_pc{i}" for i in range(1,6)]].to_numpy().round(3))
PY
```

**(b) Assign a letter to a NEW 50-point registered ridge.** The soft-DTW model
expects a **registered ridge**: a length-50 float curve, mean-subtracted and
resampled to the active span (the exact transform `register_one` in the ridge
builder applies — see §2.1). You feed it as shape `(1, 50, 1)`:

```bash
.venv/bin/python - <<'PY'
import joblib, numpy as np
from tslearn.metrics import cdist_soft_dtw_normalized

bundle = joblib.load("models/shape_kmeans/k20_softdtw.joblib")
km     = bundle["kmeans"]              # tslearn TimeSeriesKMeans
gamma  = bundle["gamma"]              # 1.0
centroids = km.cluster_centers_       # (20, 50, 1)

# my_ridge: a length-50 registered ridge (see §2.1 for how to build one)
my_ridge = np.load("/path/to/registered_ridge_50pt.npy").astype(np.float64)
D = cdist_soft_dtw_normalized(my_ridge[None, :, None], centroids, gamma=gamma)  # (1, 20)
print("assigned letter:", int(D.argmin(axis=1)[0]))
PY
```

> This reproduces exactly the streamed-assignment step in
> `build_softdtw_alphabet.py:117-121`. The same `gamma=1.0` and
> `cdist_soft_dtw_normalized` must be used, or distances will not be comparable to
> how the corpus was assigned.

### 1.5 Re-evaluate the representations (the standing GATE harness)

The single source of truth for "which shape representation is best" is
`scripts/experiments/eval_shape_human_anchored.py`. It scores each representation
by **leave-one-out kNN retrieval purity** against the human shape labels, with
1000× bootstrap CIs, and decides on **non-overlapping CIs** (never point
estimates). Its 5 core functions are unit-tested as a frozen spec:

```bash
.venv/bin/python -m pytest tests/experiments/test_eval_shape_human_anchored.py -q
# Expect: 33 passed.

.venv/bin/python scripts/experiments/eval_shape_human_anchored.py
# Writes results/shape_retrospective/human_anchored_eval_v2.{json,html}
```

**Flags** (all have defaults; defaults reproduce the current scorecard):

| Flag | Default | Meaning / when to change |
|------|---------|--------------------------|
| `--meta` | `…/jobs/9a954f32/tmp/shape_data/true_registered_ridges_meta.npz` | The registered-ridge meta (`shapes`, `wav_stem`, `call_id`, `cohort`). **NOT git-tracked** — see Gotchas. |
| `--lab` | `…/jobs/9a954f32/tmp/shape_data/true_registered_ridges.npz` | The companion file holding `lab_shape` (incumbent K=20 alphabet) + `chevron_valley`. |
| `--human` | `data/manual_shape_labels.csv` | The human gold labels. |
| `--out-json` | `results/shape_retrospective/human_anchored_eval_v2.json` | Scorecard JSON. |
| `--out-html` | `results/shape_retrospective/human_anchored_eval_v2.html` | HTML report (prints a `file://wsl.localhost/...` URL). |
| `--k` | `10` | kNN neighbours for purity. |
| `--n-boot` | `1000` | Bootstrap resamples for CIs. |
| `--seed` | `42` | RNG seed (bootstrap + random control). |
| `--softdtw-gamma` | `1.0` | Soft-DTW smoothing γ. Must match the alphabet's γ. |
| `--no-softdtw` | off | Skip the elastic soft-DTW block (debug only — disables GATE 1 read). |
| `--fpca-lambda` | `0.0` | Elasticity penalty λ for the SRVF elastic-FPCA distance used *inside the eval*. **NB:** the production FPCA *artifact* was built at λ=0.05 (§2.2). |
| `--no-elasticfpca` | off | Skip the SRVF elastic-FPCA block. |

**How to read the scorecard.** Each cell is `purity [ci_lo, ci_hi]`. Compare the
**candidate's `ci_lo` against the incumbent's `ci_hi`** — if they do not overlap,
that's a real win. Rows: `registration_euclidean(IDENTITY)` is the incumbent /
baseline; `soft_dtw(ELASTIC)` is the production metric; `elastic_fpca(SRVF-WARP)`
is the coordinate-system candidate; `random_control(BASE RATE)` is the floor.

**Current result** (`human_anchored_eval_v2.json`, 611 matched labels across all 4
cohorts, verified on disk):

| representation | chevron | jump | flat | complex |
|----------------|---------|------|------|---------|
| registration (IDENTITY) | 0.186 [.143,.232] | 0.415 [.377,.453] | 0.419 [.384,.456] | 0.194 [.148,.236] |
| **soft-DTW (ELASTIC)** | 0.214 [.168,.261] | **0.522 [.480,.570]** | 0.396 [.362,.433] | **0.243 [.199,.284]** |
| elastic-FPCA (SRVF-warp) | 0.230 [.177,.280] | 0.499 [.457,.540] | 0.381 [.346,.417] | 0.170 [.136,.206] |
| random (BASE RATE) | 0.080 | 0.332 | 0.193 | 0.118 |

**The headline result:** soft-DTW beats registration on **jump** with
non-overlapping CIs → GATE 1 verdict `PROCEED`. (On **complex** soft-DTW also
*improves* the point estimate — 0.243 vs 0.194 — but the CIs OVERLAP
(.199 < .236), so the gate scores complex as a *tie*, not a beat: the v2
`gate1.beats` list is `["jump"]` only. Treat the complex gain as suggestive, not
yet established.) (The originally-reported
Phase-1 figures were **jump 0.452 vs registration 0.327** on the first 200-label
lab-only set; the result has only strengthened as labels expanded: 0.463 vs 0.373
at 551 labels, 0.522 vs 0.415 at the current 611. The mechanism is unchanged — a
step/jump is a discontinuity that warp-alignment lines up cheaply.) Secondary:
NMI(incumbent K=20 alphabet vs human family) = **0.178**.

### 1.6 Rebuild the artifacts (rarely needed)

Both builders read a registered-ridge meta npz and write **parallel**
artifacts — they never overwrite the incumbent `k20.joblib`. **Watch the default
`--meta` paths: they differ.** `build_softdtw_alphabet.py` and the eval default to
job `9a954f32` (`build_softdtw_alphabet.py:61`, `eval_shape_human_anchored.py:241`),
but `build_elastic_fpca.py` defaults to job `57976676`
(`build_elastic_fpca.py:347`). The two meta npz are content-identical
(`shapes (67337, 50)`, verified equal), so the artifacts are consistent — but pass
`--meta` explicitly if either job dir has been purged.

```bash
# Soft-DTW alphabet (CPU-bound; full fit ~1–3 h. Smoke first.)
.venv/bin/python scripts/experiments/build_softdtw_alphabet.py --smoke   # validates pipeline, writes *_SMOKE
.venv/bin/python scripts/experiments/build_softdtw_alphabet.py           # full: K=20, subsample 8000, γ=1.0

# Elastic FPCA (full-corpus coordinate producer)
.venv/bin/python scripts/experiments/build_elastic_fpca.py --lam 0.05    # production λ (see Gotchas)
```

`build_softdtw_alphabet.py` flags (defaults at `:64-70`): `--k 20`,
`--subsample 8000`, `--max-iter 15`, `--n-init 2`, `--gamma 1.0`,
`--assign-batch 500`, `--seed 42`, `--smoke`.
`build_elastic_fpca.py` flags (defaults at `:350-356`): `--lam 0.0` (production
artifact used **0.05**), `--n-amp 5`, `--n-phase 3`, `--seed 42`,
`--subset N` (smoke), `--max-itr 20`, `--no-parallel`.

### 1.7 Troubleshooting / Gotchas

- **`k20.joblib` (registration model) is NOT on disk.** Only `k20_softdtw.joblib`
  and the FPCA artifacts exist under `models/shape_kmeans/` / `models/shape_fpca/`.
  Every script *comment* that says it preserves the incumbent `k20.joblib` is
  describing intent, not a file you will find. The registration representation is
  reproducible from the ridge meta (`X_reg = Sh[rows]` is just the raw ridges,
  `eval_shape_human_anchored.py:302`); it is the IDENTITY control, no model file.

- **The ridge meta npz is NOT git-tracked.** Both builders and the eval default to
  `…/.claude/jobs/<id>/tmp/shape_data/true_registered_ridges*.npz`. Those live in
  ephemeral job tmp dirs (currently present in jobs `9a954f32` and `57976676`). If
  they are gone, regenerate them with the rig builder
  `archive/cleaning_legacy/stack3/scripts/experiments/rig_R1_true_ridges.py`,
  which reads the contour-masked patches from the rig
  (`/data/shachar/contour_vae/results/masked_patches/`). **Before regenerating,
  check the rig** — the masked patches + the VAE artifacts live only there
  (see `docs/DATA_LOCATIONS.md`). Do not reinvent the render.

- **FPCA join key gotcha (the most dangerous one).** To join the FPCA scores to a
  detection table, the key is **`(wav_stem, call_id - 1) == (wav_stem, det_index)`**
  — `call_id` is 1-indexed, `det_index` is 0-indexed (the `offset=-1` in
  `build_join`, `eval_shape_human_anchored.py:72-101`). Do **not** join on a raw
  `id`. Also: **`(wav_stem, call_id)` is NON-UNIQUE** in upstream tables — dedupe
  first (the harness keeps the FIRST ridge row per composite id,
  `eval_shape_human_anchored.py:91-94`). Joining without dedup silently fans out
  rows.

- **`--fpca-lambda` in the eval (0.0) ≠ the production FPCA artifact (0.05).** The
  eval's internal SRVF-distance λ defaults to 0.0; the *shipped*
  `elastic_fpca.joblib` was built at **λ=0.05** (verified in the joblib:
  `lam: 0.05`). The 0.05 value won GATE A. If you rebuild the artifact, pass
  `--lam 0.05`.

- **Letters are integers in [0,19], not glyphs and not ordered.** Don't assume
  letter 0 is "flat" — inspect the centroid (`km.cluster_centers_[i]`) to see its
  shape.

- **Soft-DTW is O(N²) and CPU-bound.** A full 67k×67k matrix is ~36 GB and is
  NEVER built. The alphabet is fit on a cohort-balanced subsample (~8000, realized
  4840 in the shipped model) and the rest are assigned by streamed
  centroid-vs-batch distance (`build_softdtw_alphabet.py:7-17, 112-124`). For new
  per-call assignment use the centroid approach in §1.4(b), not a pairwise matrix.

- **Human labels: 758 in the CSV, 611 matched in the eval.** Not all labels join
  (different `call_id` format / unmatched recordings) and `unclear` labels are
  dropped (`eval_shape_human_anchored.py:289-292`). The matched-count print
  `[JOIN] matched N/…` is the number that actually scored.

- **Use the right metric for the right job.** Soft-DTW is the **labeled-set
  metric** (best on jump/complex). Elastic-FPCA scores are the **full-corpus
  coordinate system** for downstream analysis. This split *is* the HYBRID decision
  — soft-DTW does not give coordinates (it's O(N²) distances only); FPCA does, but
  loses to soft-DTW on `complex`.

---

## 2. Internals

### 2.1 Data flow

```
contour-masked patches (rig: /data/shachar/contour_vae/results/masked_patches/)
        │  rig_R1_true_ridges.py  (archive/cleaning_legacy/stack3/scripts/experiments/)
        │    per call: band-crop rows [35:205] (20–120 kHz) → Viterbi ridge (track_ridge)
        │              → register: subtract mean kHz (kill pitch), resample active span
        │                to N_RESAMPLE=50 (kill duration + time-position)
        ▼
true_registered_ridges_meta.npz   shapes (67337, 50) float32  + wav_stem/call_id/cohort
true_registered_ridges.npz        lab_shape (incumbent K=20), chevron_valley, srvf, …
        │
        ├── build_softdtw_alphabet.py ──► k20_softdtw.joblib + k20_softdtw_letters.parquet
        ├── build_elastic_fpca.py     ──► elastic_fpca.joblib + elastic_fpca_scores.parquet
        └── eval_shape_human_anchored.py (vs data/manual_shape_labels.csv) ──► v2 scorecard
```

**Registration** is the core nuisance-removal step (in `rig_R1_true_ridges.py`):
band-crop to USV rows `BAND0,BAND1 = 35,205` (170 bins, 20–120 kHz), Viterbi
ridge-track to a per-call pitch curve, subtract the mean kHz (removes absolute
pitch), and resample the active span to `N_RESAMPLE = 50` points (removes duration
and time-position). The 50-point unit-less curve is what every downstream method
consumes. Signal-processing band/STFT constants are upstream of this doc — see
[corpus constants](corpus_constants.md); do not re-derive them here.

### 2.2 Key functions (file:line)

**Soft-DTW alphabet** — `scripts/experiments/build_softdtw_alphabet.py`:
- `stratified_cohort_subsample(cohort, n_total, seed)` `:38` — cohort-balanced
  subsample (quota = `n_total // n_cohorts`; small cohorts contribute all rows,
  freed quota NOT redistributed). Shipped model subsampled to 4840 rows.
- Fit: `TimeSeriesKMeans(n_clusters=k, metric="softdtw", metric_params={"gamma":gamma}, …)`
  `:103-107`. Centroids = soft-DTW barycenters, shape `(20, 50, 1)`.
- Streamed assignment: `cdist_soft_dtw_normalized(batch, centroids, gamma)` →
  `argmin` `:117-121`. This is the assignment you reproduce for new calls.
- Output bundle keys (`:129-130`): `kmeans`, `k`, `gamma`, `metric`,
  `subsample_idx`, `seed`.

**Elastic FPCA** — `scripts/experiments/build_elastic_fpca.py`:
- `elastic_amplitude_distance(f1, f2, time, lam, method)` `:49` — Fisher-Rao SRVF
  amplitude distance WITH warp optimization (`fdasrsf.elastic_distance(...)[0]`).
  This is the function the eval harness imports as its elastic-FPCA metric.
- `elastic_amplitude_distance_matrix(X, lam, …)` `:83` — symmetric `(n,n)` matrix;
  symmetrizes `(D+D.T)/2`, zero diagonal (raw library is ~1% directional).
- `elastic_karcher_align(X, lam, max_itr, parallel)` `:245` — aligns all ridges to
  the elastic Karcher mean via `fdasrsf.fdawarp.srsf_align`; returns
  `mean_f / aligned_f / aligned_q / warps`.
- `amplitude_fpca(aligned_q, n_components)` `:170` — vertical FPCA on aligned SRVFs
  → 5 amplitude PCs. `phase_fpca(warps, n_components)` `:218` — horizontal FPCA in
  ψ = √(dγ/dt) space (`_warp_to_psi` `:204`) → 3 phase PCs.
- `build(...)` `:277` — full-corpus producer; model bundle keys (`:328-336`):
  `lam, seed, n_amp, n_phase, time_grid, amp_mean, amp_components,
  amp_recon_errors, phase_mean, phase_components, karcher_mean_f, n_rows`.
- Constants: `DEFAULT_TIME_POINTS = 50` `:40`, `TIME_GRID = linspace(0,1,50)`
  `:41`, `DEFAULT_METHOD = "DP2"` `:43`. **`DEFAULT_LAMBDA = 0.0` `:42` is a
  placeholder** — the shipped artifact was built at λ=0.05.

**Eval / gate** — `scripts/experiments/eval_shape_human_anchored.py`:
- `group_family(label)` `:58` — maps Grimsley display labels to families
  {chevron, jump, flat, complex}; unknown labels pass through unchanged.
- `build_join(wav_stem, call_id, human_df, offset=-1)` `:72` — the join with the
  `offset=-1` (1-indexed→0-indexed) correction and first-occurrence dedup.
- `loo_knn_purity(X, labels, target, k=10)` `:129` / `knn_purity_from_distance(D, …)`
  `:162` — purity from an embedding vs from a precomputed distance matrix (the
  soft-DTW / elastic path).
- `bootstrap_purity_ci(...)` `:178` / `bootstrap_purity_ci_from_distance(...)` `:199`
  — 1000× bootstrap CIs.
- `gate1_read(results, base)` `:418` — encodes "beats iff `ci_lo > ident.ci_hi`,
  regresses iff `ci_hi < ident.ci_lo`"; PROCEED if soft-DTW beats identity on jump
  OR complex AND chevron/flat don't regress.

### 2.3 Invariants

- **These 5 functions are a frozen SPEC** (`group_family`, `build_join`,
  `loo_knn_purity`, `knn_purity_from_distance`, `bootstrap_purity_ci`), tested by
  `tests/experiments/test_eval_shape_human_anchored.py` (**33 tests**). Do NOT
  change their signatures.
- **Decisions are on non-overlapping bootstrap CIs, never point estimates**
  (`gate1_read`). The harness exists to replace the *circular* `shape η²` /
  `chevron_valley` metric (which was computed on the same KMeans it graded);
  `shape η²` is deliberately NOT computed here.
- **Parallel artifacts only.** Builders must never overwrite the incumbent
  `k20.joblib` — they write `k20_softdtw.*` and `elastic_fpca.*`.
- **γ must match.** New per-call soft-DTW assignment must use the same
  `gamma=1.0` and `cdist_soft_dtw_normalized` as the alphabet build.
- **Letters/coordinates are keyed by `(wav_stem, call_id)`; `call_id` is
  1-indexed.** Joins to 0-indexed detection tables need `call_id - 1`, and the key
  is non-unique upstream → dedupe first.

### 2.4 Where to change things

| Change | Edit |
|--------|------|
| Different K, γ, subsample size for the alphabet | `build_softdtw_alphabet.py` CLI defaults `:64-70` |
| Different FPCA λ / #amp / #phase axes | `build_elastic_fpca.py` CLI defaults `:350-356`; rebuild at λ=0.05 |
| The decision rule / families / k / bootstrap | `eval_shape_human_anchored.py` (`FAMILIES :217`, `gate1_read :418`); update the SPEC tests in lockstep |
| How a ridge is registered (band, resample length) | `archive/cleaning_legacy/stack3/scripts/experiments/rig_R1_true_ridges.py` (`BAND0/BAND1`, `N_RESAMPLE`) — touches the contour-masked render; check the rig first |
| Add the wild-cohort gold labels (Phase 2 → production swap) | `data/manual_shape_labels.csv` + re-run the gate; see `docs/handoffs/2026-06-03_elastic-shape-phase2-labels.md` |

### 2.5 Provenance & related decisions

- **Plans:** `PLAN_elastic_shape_clustering.md` (GATE 1 — soft-DTW vs
  registration), `PLAN_continuum_repertoire_program.md` (§WS-A GATE A — the
  HYBRID decision: adopt elastic-FPCA λ=0.05 as the coordinate system, carry
  soft-DTW as the complex-sensitive metric).
- **Handoffs:** `docs/handoffs/2026-06-04_ws-a-elastic-fpca-implementation.md`,
  `docs/handoffs/2026-06-04_ws-bc-continuous-grammar-manifold.md`,
  `docs/handoffs/2026-06-03_elastic-shape-phase2-labels.md`.
- **VAE closure:** the VAE family was closed for shape clustering — see
  [Contour-VAE](production_vae.md). Use that VAE for cohort / latent-geometry
  analysis, NOT for shape.
- **Production swap is GATED.** The soft-DTW alphabet is the current *metric*, but
  swapping it in as *the* production shape alphabet (over registration) is gated on
  Phase-2 label expansion to ≥2 wild cohorts holding the win on non-overlapping
  CIs (the current 611-label set already spans all 4 cohorts, strengthening the
  case — but read the Phase-2 handoff for the formal gate).

# Contour-VAE (32-D latent) — Cohort / Latent-Geometry Analysis Tool

> **What this is** — A continuous-latent variational autoencoder (`ImageVAE`,
> `latent_dim=32`) trained on contour-masked USV patches. It embeds every call
> into a 32-D vector (`z_0..z_31`). It was the **first VAE treated as production**
> and is the basis of the cross-cohort (wild-vs-lab) latent-geometry analysis.
> **Status: CURRENT for cohort / latent-geometry analysis. SUPERSEDED for *shape*
> clustering** — the VAE family was closed for shape work in favour of
> registration / soft-DTW (see [shape clustering](#deprecation--scope-boundaries)).
> **Production artifact:** `models/contour_vae_combined/best.pt` (32-D latent,
> trained on 69,293 patches). Config: `models/contour_vae_combined/hyperparams.json`.
> **Use it for:** cohort separation, per-dim effect sizes, latent KNN. **Do not
> use it for:** clustering calls by *shape*.

This VAE is a **pitch / duration sorter**. Its latent geometry cleanly separates
recording cohorts (cage signature) and orders calls by pitch and duration, but it
does **not** recover shape categories. For shape, use
`models/shape_kmeans/k20_softdtw.joblib` instead (see
[Deprecation & scope boundaries](#deprecation--scope-boundaries)).

Related docs:
[cleaning subsystems](../modules/cleaning-subsystems.md) (Stack 4 produces the input
patches) · [CNN classifier](../modules/cnn-classifier.md) (upstream detector).

---

## 1. Operate

There are **two distinct pipelines** under this banner, with different code,
different inputs, and different embedding files. Read this first so you run the
right one:

| Pipeline | Embedder | Embedding file | Producer |
|---|---|---|---|
| **A — PyTorch contour-VAE (this model)** | `models/contour_vae_combined/best.pt` | `latents.parquet` (z_0..z_31 + manifest joins) | `scripts/train_contour_vae_v2.py` |
| **B — DeepSqueak/MATLAB VAE** | MATLAB DeepSqueak VAE | `vae_embeddings.csv` (z_0..z_31 + mat_file/begin_s) | `scripts/deepsqueak_train_vae.m` |

The **analysis script `scripts/analyze_vae_embeddings.py` reads the *CSV* form
(`vae_embeddings.csv`, pipeline B)**, not `latents.parquet`. Both are 32-D and
share the `z_*` naming convention, but their schemas differ (table below). If you
want to analyse *this* PyTorch model's latents with `analyze_vae_embeddings.py`,
you must first rename/remap `latents.parquet` columns to the CSV schema
(`wav_stem`→`mat_file`, add `begin_s`) — see Gotchas.

### Required environment

- Interpreter: `.venv/bin/python` (Linux/WSL).
- Training (pipeline A) needs CUDA + the 16 GiB `patches.npz`; **run it on the GPU
  rig**, not the box (see the patches.npz gotcha below).
- Analysis (`analyze_vae_embeddings.py`) needs `umap-learn`, `hdbscan`, `scipy`,
  `pandas`, `matplotlib` (all in `.venv`). It runs fine on the box because it
  operates on the small CSV of embeddings, not the 16 GiB patches.

### 1a. Generate embeddings (training — pipeline A)

This is the step that *produced* the production model. You only re-run it to
retrain. Source: `scripts/train_contour_vae_v2.py:580-613` (CLI),
`:616-915` (`main`).

```bash
# Run ON THE RIG (cloudyclaude). The 16 GiB patches.npz only exists there.
.venv/bin/python scripts/train_contour_vae_v2.py \
    --patches-npz      results/masked_patches/combined_all_cohorts/patches.npz \
    --manifest-parquet results/masked_patches/combined_all_cohorts/patches_manifest.parquet \
    --output-model-dir   models/contour_vae_combined \
    --output-results-dir results/contour_vae_combined
```

| Flag | Default | What it does / when to change |
|---|---|---|
| `--patches-npz` | *(required)* | NPZ with keys `patches` (N,F,T raw power) and `freqs_kHz` (F,). Must be **uncompressed (ZIP_STORED)** so `mmap_mode='r'` works (`:638-641`). |
| `--manifest-parquet` | *(required)* | One row per patch; must contain `patch_idx, wav_stem, call_id, window_idx` (`:589-591`). Joined **positionally** to latents, never on `patch_idx` — see Internals. |
| `--output-model-dir` | *(required)* | Receives `best.pt`, `last.pt`, `hyperparams.json` (`:592-593`). |
| `--output-results-dir` | *(required)* | Receives `training_log.csv`, `latents.parquet`, `reconstructions/` (`:594-595`). |
| `--latent-dim` | `32` | Latent dimensionality (`:596-597`). The production model is 32. Changing it makes the model incompatible with `z_0..z_31` analyses. |
| `--batch-size` | `32` | Mini-batch (`:598-599`). |
| `--lr` | `2.5e-4` | Adam learning rate (`:600-601`). |
| `--max-epochs` | `500` | Cap; early stopping usually cuts it short (`:602-603`). |
| `--patience` | `50` | Early-stop patience on `val_recon` (`:604-605`). |
| `--seed` | `42` | Seeds torch/numpy/train-val split; `cudnn.deterministic=True` (`:606-607`, `:619-624`). |
| `--beta` | `1.0` | KL weight in the ELBO. `1.0` = standard VAE; do **not** turn this into a β-VAE until standard ELBO is verified working (`:608-610`). |
| `--n-recon-pngs` | `20` | Number of input/recon QA PNG pairs from the val set (`:611-612`). |

**Training internals you should know before reading outputs:**

- **Input preprocessing** per patch (`MaskedPatchDataset.__getitem__`,
  `:402-413`): (1) band-crop to the corpus USV band, (2) `log1p(power)`,
  (3) per-patch min/max rescale to `[0,1]`, (4) zero-pad to 256×256,
  (5) add channel dim. Band crop is computed at runtime from the NPZ's
  `freqs_kHz` against `corpus.USV_FREQ_MIN_HZ`/`USV_FREQ_MAX_HZ`
  (`_compute_band_slice`, `:297-306`) — for the canonical render this gives
  band slice `[35, 205]` = **170 freq bins (`F_in`)**, **234 time bins (`T_in`)**
  (`hyperparams.json`).
- **Padding** (`PaddingSpec`, `:309-365`): symmetric on freq (43 top / 43 bottom,
  centres the band), **right-pad-only on time** (0 left / 22 right, preserves
  onset at column 0). The reverse `crop` is applied only for visualization.
- **Split**: 80/20 train/val, `train_test_split(..., random_state=seed)`
  (`:684-690`). For the production run: `n_total=69293`, `n_train=55434`,
  `n_val=13859`.
- **Loss / early stop**: ELBO = BCE(recon, target) summed over pixels / batch +
  `beta` · KL summed over dims / batch (`image_vae_loss`, `:269-289`). `best.pt`
  is saved at the **lowest `val_recon`** (not lowest total loss) — `:832-836`.

#### Training outputs (pipeline A)

| Output | Path | Contents |
|---|---|---|
| Best model | `<model-dir>/best.pt` | `state_dict` at min `val_recon`. **This is the production weight file.** |
| Last model | `<model-dir>/last.pt` | `state_dict` at final epoch (diagnostic only). |
| Config | `<model-dir>/hyperparams.json` | Full config dump, written *before* training so it survives a crash (`:738-808`). |
| Epoch log | `<results-dir>/training_log.csv` | One row/epoch: `epoch, train_total, train_recon, train_kl, val_total, val_recon, val_kl, lr`. |
| **Latents** | `<results-dir>/latents.parquet` | `(N, 37)`: `patch_idx, z_0..z_31, wav_stem, call_id, window_idx, cohort`. Encoded from the **posterior mean** (`encode_mean` / `model.encode` mu), not a sample (`:464-476`). |
| QA PNGs | `<results-dir>/reconstructions/recon_NN_patchXXXX.png` | 20 input-vs-recon pairs, cropped back to the 170×234 band region. |

`latents.parquet` schema (verified, 69,293 rows × 37 cols):
`patch_idx (int64)`, `z_0..z_31 (float32, posterior mean)`,
`wav_stem`, `call_id`, `window_idx`, `cohort`.

### 1b. Run the cross-cohort analysis (pipeline B input)

Source: `scripts/analyze_vae_embeddings.py:428-594` (`main`). This is the script
behind the wild-vs-lab deck.

```bash
.venv/bin/python scripts/analyze_vae_embeddings.py \
    --embeddings   results/vae_5970_lab/vae_embeddings.csv \
    --scattoni-csv results/traditional_taxonomy/classified_traditional.csv \
    --out-dir      results/vae_analysis/

# Exercise the full pipeline with synthetic data (no real embeddings needed):
.venv/bin/python scripts/analyze_vae_embeddings.py --smoke-test
```

| Flag | Default | What it does |
|---|---|---|
| `--embeddings` | `results/vae_5970_lab/vae_embeddings.csv` | Embedding CSV. Must have `z_*` columns + a `cohort` column (`:430-432`, `load_embeddings` `:78-94`). |
| `--scattoni-csv` | `results/traditional_taxonomy/classified_traditional.csv` | Optional Scattoni-7 labels for type-coloured UMAP. Best-effort tolerance join (5 ms) on `mat_file`+`begin_s`; silently skipped if columns don't align (`:97-145`). |
| `--out-dir` | `results/vae_analysis` | Output dir (`:436-438`). |
| `--seed` | `42` | Seeds UMAP `random_state` and KNN sampling (`:439`). |
| `--smoke-test` | off | Generate synthetic 2-cohort embeddings (5000 each, d=32, Cohort B shifted +0.5 on dims 0-7) and run end-to-end (`:440-441`, `make_synthetic_embeddings` `:397-421`). |
| `--min-cluster-size` | `50` | HDBSCAN `min_cluster_size`, applied to both the UMAP-2D and the raw-32D clustering (`:442-443`). |

**Hardcoded analysis parameters** (not CLI flags — change in code if needed):

- **UMAP**: `n_neighbors=15, min_dist=0.1, n_components=2, metric="euclidean",
  random_state=seed` (`run_umap`, `:152-160`).
- **HDBSCAN**: `min_cluster_size` from `--min-cluster-size` (default 50),
  `prediction_data=False` (`run_hdbscan`, `:163-168`); run **twice** — once on
  the UMAP-2D coords, once on the raw 32-D latent (`:487-488`).
- **Dead-dim threshold**: a latent dim with `std < 0.05` across the dataset is
  flagged as collapsed/dead (`plot_dead_dims` `:378-390`, summary `:585`).
- **KNN sanity**: 10 queries per cohort, `k=5`, Euclidean in latent space
  (`knn_sanity` `:220-249`, called `:527`).

#### Analysis outputs (`--out-dir`)

| Output | What it is |
|---|---|
| `figs/01_umap_cohort.png` | UMAP 2-D scatter coloured by cohort. |
| `figs/02_umap_scattoni.png` | UMAP coloured by Scattoni-7 type (only if join succeeded). |
| `figs/03_umap_duration.png`, `03b_umap_tonality.png` | UMAP coloured by `call_length_s`/`duration_s` and `tonality` (only if those columns exist). |
| `figs/04_density_continuum.png` | Hexbin density — the "continuum" view. |
| `figs/05_cohort_split.png` | One hexbin panel per cohort on shared axes. |
| `figs/06_cohen_d_per_dim.png` | Per-dim Cohen's d between the first two cohorts, sorted by \|d\|; 0.2/0.5/0.8 reference lines. |
| `figs/07_emd_per_dim.png` | Per-dim 1-Wasserstein (EMD) between cohorts. |
| `figs/08_hdbscan_clusters.png` | HDBSCAN labels on the UMAP plane (noise = grey). |
| `figs/10_dead_dim_diagnostic.png` | Per-dim std bar chart with the std=0.05 dead-dim line. |
| `umap_embeddings.csv` | `cohort, mat_file, call_id, begin_s, end_s, umap_1, umap_2, hdbscan_umap, hdbscan_32d` (whichever metadata cols exist). |
| `knn_sanity.csv` | One row per (query, neighbour): `query_cohort, query_idx, query_mat_file, query_begin_s, rank, neighbor_idx, neighbor_cohort, neighbor_mat_file, neighbor_begin_s, latent_distance`. |
| `cross_cohort_summary.json` | All numbers in one file (schema below). |
| `synthetic_embeddings.csv` | Only with `--smoke-test`. |

`cross_cohort_summary.json` keys (`:565-588`): `embeddings_path`, `smoke_test`,
`n_total`, `latent_dim`, `cohorts` (per-cohort counts), `umap_params`,
`hdbscan_on_umap` / `hdbscan_on_32d` (each: `min_cluster_size, n_clusters,
n_noise, noise_frac`), `knn_same_cohort_frac`, `dead_dim_count_at_std_0p05`,
`per_dim_std` (32 values), `pairwise` (per cohort-pair: `cohens_d_per_dim`,
`emd_per_dim`, `sum_emd`, `max_abs_cohens_d`, `n_dims_d_above_0p2`,
`umap_density_jsd`).

#### How to read the key diagnostics

- **`knn_same_cohort_frac`**: 1.0 = total cohort separation, ~0.5 = full overlap
  (`:531-533`). High values mean the latent geometry is dominated by cage/cohort.
- **`dead_dim_count_at_std_0p05`**: number of collapsed latent dims. Many dead
  dims = the 32-D latent is effectively lower-dimensional (mode collapse) — read
  this *before* trusting per-dim effect sizes.
- **Cohen's d**: \|d\|≥0.2 small, ≥0.5 medium, ≥0.8 large. **No p-values** — N=1
  *couple* per cohort, so these are effect sizes, not inferential statistics
  (`cohens_d` docstring `:175-185`; see auto-memory
  `feedback_cross_animal_population_strata`).
- **`umap_density_jsd`**: `jensenshannon(...)**2` (the function returns the
  *distance* = sqrt(JSD); the code squares it back to a divergence) — `:198-217`.

### Worked example (smoke test, no GPU, no rig)

```bash
.venv/bin/python scripts/analyze_vae_embeddings.py --smoke-test \
    --out-dir results/vae_analysis_smoke/
```

Produces synthetic 5970 vs lab_131204 embeddings with a deliberate +0.5 shift on
dims 0-7 (≈ Cohen's d 0.5 there, ~0 elsewhere), then writes all figures + the
three CSV/JSON outputs to `results/vae_analysis_smoke/`. Use it to verify the
analysis environment works before pointing it at real embeddings.

### Troubleshooting / Gotchas

- **NEVER full-scan `patches.npz` on the box.** The combined-cohort
  `results/masked_patches/combined_all_cohorts/patches.npz` is **~16 GiB**
  (DATA_LOCATIONS.md:36; ~11 GiB working set) and lives **only on the GPU rig**
  at `/data/shachar/contour_vae/` — it is **not in git** and not on the box.
  Loading it whole OOM-crashed WSL once. Training opens it with `mmap_mode='r'`
  and slices per-patch precisely to avoid this (`train_contour_vae_v2.py:638-641`,
  `MaskedPatchDataset.__getitem__:402-403`). If you must touch it, **stream/slice
  on the rig**, never `np.load(...)["patches"]` into RAM.
- **patches.npz must be uncompressed** (ZIP_STORED) or `mmap_mode='r'` fails
  (`:636-641`). `assemble_combined_patches.py` guarantees this.
- **Two embedding files, easy to confuse.** `analyze_vae_embeddings.py` reads
  `vae_embeddings.csv` (mat_file/begin_s schema, MATLAB pipeline). *This model*
  emits `latents.parquet` (wav_stem/call_id schema). They are not interchangeable
  without remapping columns.
- **Manifest join is positional, not by key.** In a combined-cohort manifest,
  `patch_idx` restarts at 0 per cohort and is **not globally unique** — merging on
  it cross-joins and scrambles the z↔wav mapping. `_build_latents_df` aligns
  positionally (`z_all[i]` ↔ `manifest.iloc[i]`) and *asserts* equal lengths
  (`:479-506`). Preserve patch order end-to-end.
- **Reconstruction PNGs look "mostly zeros" — that's the padding, not the model.**
  PNGs are cropped back to the 170×234 band region precisely so the zero borders
  don't mislead you (`_save_reconstruction_pngs:509-572`).
- **Scattoni join silently skips** if column names don't match — the script
  proceeds without type labels and omits `02_umap_scattoni.png` (`:97-145`). Check
  the printed "Scattoni join: N/M matched" line.
- **`mean_power_db` / `tonality` separation between cohorts is a cage artifact**,
  not biology, unless cross-cage calibrated (auto-memory
  `feedback_rig_artifact_mean_power_db`). The same caution applies to any cohort
  separation this VAE reports.
- **`last.pt` is absent locally.** Only `best.pt` + `hyperparams.json` are checked
  into `models/contour_vae_combined/` on the box; `last.pt` is not. The results
  dir (`results/contour_vae_combined/latents.parquet`, `training_log.csv`,
  `reconstructions/`) **is present on the box** (it is the small per-call output,
  not the 16 GiB patches); only the source `patches.npz` is rig-only.

---

## 2. Internals

### Data flow

```
WAV → CNN+hysteresis detection → DeepSqueak focus-STFT + contour mask (Stack 4,
  cleaning-subsystems.md) → window_calls_to_patches → patches.npz (N,257,234 raw
  power) + patches_manifest.parquet
        │  (canonical render: global Hann n_fft=512 / hop=128 @ 300 kHz,
        │   fixed 234-bin windows, raw power — NOT dB, NOT event-cropped)
        ▼
  train_contour_vae_v2.py: band-crop[35:205]→170×234 → log1p → per-patch
    min/max [0,1] → zero-pad 256×256 → ImageVAE → best.pt
        ▼
  encode_all (posterior mean) → latents.parquet (z_0..z_31 + manifest)
        ▼
  (separate MATLAB path → vae_embeddings.csv) → analyze_vae_embeddings.py
    → UMAP/HDBSCAN/Cohen-d/EMD/JSD/KNN → figs + summary JSON
```

**Canonical patch render** (do not reinvent — a chat diverged 6 ways on
2026-05-29; see DATA_LOCATIONS.md:16-26 and `feedback_cleaning_pipeline_impl_on_rig`):
global Hann `n_fft=512` / `hop=128` @ 300 kHz, fixed 234-bin windows, raw power
`(N, 257, 234)`. The 257 freq bins are `n_fft/2 + 1`.

### Model architecture (`ImageVAE`, `train_contour_vae_v2.py:170-261`)

A self-contained port of `../vae-pytorch-pivot/usv_language/models/image_vae.py`
(inlined so the script has no cross-worktree dependency). `n_params=7,746,673`.

- **Encoder** (`:186-199`): 4× stride-2 conv `1→32→64→128→256`, each
  `BatchNorm2d` + `LeakyReLU(0.2)`. Input 256×256 → bottleneck `(256, 16, 16)`.
- **Latent head** (`:201`): single `Linear(256·16·16 → 2·latent_dim)`; split into
  `(mu, logvar)`; `logvar.clamp(-10, 10)` for KL numerical stability
  (`encode:227-237`).
- **Decoder** (`:204-223`): `Linear(latent_dim → 256·16·16)` → 4× stride-2
  `ConvTranspose2d` `256→128→64→32→16` (BN+ReLU) → 2× stride-1 refinement tconv
  `16→16→1` → `sigmoid` (`decode:244-249`).
- **`ImageVAEConfig`** (`:116-167`): `image_size=256` (must be power of 2 ≥16,
  validated in `__post_init__`), `in_channels=1`, `latent_dim=32`,
  `base_channels=32`, `beta=1.0`. `bottleneck_spatial = image_size//16 = 16`,
  `bottleneck_channels = base_channels·8 = 256`.

### Key function signatures (file:line)

- `ImageVAE.encode(x) -> (mu, logvar)` — `train_contour_vae_v2.py:227`
- `ImageVAE.reparameterize(mu, logvar) -> z` — `:239`
- `ImageVAE.decode(z) -> sigmoid(...)` — `:244`
- `ImageVAE.encode_mean(x) -> mu` (deterministic embedding) — `:257`
- `image_vae_loss(x_recon, x, mu, logvar, beta=1.0) -> (loss, recon, kl)` — `:269`
- `MaskedPatchDataset.__getitem__(idx) -> (1,256,256) tensor` — `:402`
- `PaddingSpec.for_shape(f_in, t_in, image_size)` / `.pad` / `.crop` — `:327/346/358`
- `_encode_all(model, dataset, device, batch_size) -> (N, latent_dim)` — `:464`
- `_build_latents_df(z_all, manifest, latent_dim)` (positional join) — `:479`
- `run_umap(Z, seed=42)` — `analyze_vae_embeddings.py:152`
- `run_hdbscan(X, min_cluster_size=50)` — `:163`
- `cohens_d(a, b)` / `per_dim_cohens_d` — `:175` / `:188`
- `umap_density_jsd(...)` (returns JSD = sqrt-distance squared back) — `:198`
- `knn_sanity(Z, df, n_queries_per_cohort=10, k=5, seed=42)` — `:220`

### Invariants

- **Corpus constants are imported, never restated.** `corpus.USV_FREQ_MIN_HZ=20000`,
  `USV_FREQ_MAX_HZ=120000`, `SAMPLE_RATE_HZ=300000`, `STFT_N_FFT=512`,
  `STFT_HOP=128` (`src/usv_spectrogram/corpus.py:30-36`). The band crop is derived
  from these against the NPZ's `freqs_kHz`; if you re-render patches with a
  different band, `F_in` and the padding change and the model must be retrained.
- **`latent_dim=32` is load-bearing.** Every analysis assumes columns `z_0..z_31`.
- **Embedding = posterior mean**, not a reparameterized sample (`_encode_all`
  calls `model.encode` and keeps `mu` only, `:474-475`) — embeddings are
  deterministic given the model.
- **`best.pt` = min `val_recon`**, not min total ELBO (`:832-836`).
- **Order is the join key.** Patches, manifest rows, and latent rows must stay in
  the same order (positional alignment assertion, `:490-493`).

### Where to change things

- Retrain / change capacity: edit `ImageVAEConfig` defaults
  (`train_contour_vae_v2.py:139-143`) or pass `--latent-dim` / `--beta`.
- Change preprocessing (band, normalization, padding): `MaskedPatchDataset`
  (`:368-413`) and `PaddingSpec` (`:309-365`). Any change here invalidates
  `best.pt`.
- Change clustering/UMAP params: `run_umap` (`:152-160`) and `run_hdbscan`
  (`:163-168`) in `analyze_vae_embeddings.py`. UMAP `n_neighbors`/`min_dist` and
  the dead-dim 0.05 threshold are hardcoded.
- Add a cohort pairing: the analysis only compares `cohorts[0]` vs `cohorts[1]`
  (`:499-501`); extend `pair_results` to compare more pairs.

---

## Deprecation & scope boundaries

**This VAE is GOOD for:** cohort / latent-geometry analysis (the wild-vs-lab
deck), per-dim effect sizes, latent-space KNN, mode-collapse diagnostics. It is a
**pitch / duration sorter** — its latent axes track pitch and duration and cleanly
separate recording cages.

**This VAE is NOT for shape clustering.** The VAE family was **closed for shape
clustering** after extensive falsification (auto-memory
`project_shape_registration_clustering`): the production shape model is
**`models/shape_kmeans/k20_softdtw.joblib`** (soft-DTW warp-aligned k-means;
present on the box at that path) — *not* any VAE latent. For shape work see the
shape-clustering / elastic-FPCA program, not this document.

**Two production pointers, do not cross them:**
- Cohort / latent geometry → `models/contour_vae_combined/best.pt` (this doc).
- Shape categories → `models/shape_kmeans/k20_softdtw.joblib`.

**Historical note:** v1 (`scripts/train_contour_vae.py`) used latent 8, a smaller
channel ladder, and z-scored MSE — it produced noisy reconstructions and is
superseded by this v2 model (`train_contour_vae_v2.py:12-26`).

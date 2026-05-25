# DATA_LOCATIONS — canonical paths for non-git data artifacts

**Purpose.** The contour-VAE / shape-representation work produces large,
regeneratable data artifacts that are deliberately kept **out of git** (see
`.gitignore` "Shape-representation v2 artifacts" + "Contour-VAE pipeline"
sections). This file is the canonical map so future sessions don't re-hunt.
Written during the 2026-05-25 post-merge reconciliation (R-C).

**Canonical rig root:** `shachar@100.113.224.57:/data/shachar/contour_vae`
(`cloudyclaude`, 3× RTX 3060 Ti; ~32 GB total). The rig cannot reach GitHub —
code moves box→rig by rsync (see `reference_gpu_rig_cloudyclaude` memory note).
The other two rig roots (`/opt/mickey_london_lab`, `/data/mickey_london_lab`)
are NOT canonical for this work; `/data/mickey_london_lab` holds the older
`ridge_tracker`.

## On the rig — `/data/shachar/contour_vae/`

| Artifact | Path (under canonical root) | Size |
|---|---|---|
| Masked patches, 5970 | `results/masked_patches/5970_focus/patches.npz` | 2.8 G |
| Masked patches, 3452 | `results/masked_patches/3452_focus/patches.npz` | 94 M |
| Masked patches, 9252 | `results/masked_patches/9252_focus/patches.npz` | 134 M |
| Masked patches, lab 131204 | `results/masked_patches/lab_131204_focus/patches.npz` | 13 G |
| Masked patches, combined | `results/masked_patches/combined_all_cohorts/patches.npz` | **16 G** |
| Trained contour-VAE | `models/contour_vae_combined/best.pt` | 30 M |
| VAE latents | `results/contour_vae_combined/latents.parquet` | 14 M |
| Registered ridges (TRUE) | `results/latent_transitions/shape_registered_TRUE/true_registered_ridges.npz` | 11 M |
| Registered ridges + identity meta | `results/latent_transitions/shape_alphabet/true_registered_ridges_meta.npz` | 10 M |
| Per-call shape letters | `results/latent_transitions/shape_alphabet/shape_call_letters.parquet` | 276 K |
| M10 image-VAE registered images | `results/latent_transitions/m10_image_vae/registered_images.npz` | 131 M |
| Per-cohort classified detections | `classified_detections_{3452,9252,full,lab_131204_clean}.csv` | (root) |

> **NEVER full-scan `combined_all_cohorts/patches.npz` (16 G) on the box** — it
> OOM-crashed WSL once. Stream/slice on the rig.

## Clustering models — now on the rig (relocated 2026-05-26 during R-D)

These small but valuable clustering models originally lived **only** in the
`latent-analysis-b-a-c` worktree (git-ignored). They were rsynced to the rig
before that worktree was retired, so they are now under the canonical root:

| Artifact | Rig path (under `/data/shachar/contour_vae/`) | Size |
|---|---|---|
| Shape K=20 alphabet (registration→shape, η²=0.75) | `models/shape_kmeans/k20.joblib` | 268 K |
| Latent K=20 model | `models/latent_kmeans/k20.joblib` | 276 K |
| Latent K=20 labels | `models/latent_kmeans/k20_labels.npy` | 544 K |
| Latent dispersion / repertoire / 3452-vs-9252 outputs | `results/{latent_dispersion,latent_repertoire,compare_3452_9252}/` | ~0.8 M |

## Code (in git, mirrored to rig)

- Production patch-generation pipeline: `scripts/{contour_mask_utils,deepsqueak_focus_stft,mass_apply_contour_mask,sweep_contour_mask,window_calls_to_patches}.py`
- Rig experiment drivers: `scripts/experiments/rig_*.py` (see that dir's README)
- Latent-analysis: `scripts/analyze_latent_{dispersion,repertoire_jsd,transitions}.py`, `scripts/{shape_registered_clustering,compare_3452_vs_9252,transition_alphabet_compare}.py`

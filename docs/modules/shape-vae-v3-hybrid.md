# Module: Shape-VAE v3 — Pathway B+A (hybrid)

**File:** `scripts/experiments/train_shape_vae_v3_hybrid.py`
**Spec:** `docs/handoffs/2026-05-27_shape-vae-BA-hybrid.md` · `PLAN_geometric_shape_clustering_vae.md` §3 Option B+A
**Status:** CPU build DONE (2026-05-27); rig train/eval gated. Tests: 75 (50 spec + 25 hardening).

## Purpose
A learned, unsupervised latent in which geometrically-similar USVs are neighbors, **invariant to
absolute pitch and time position**, AND a **navigable generative shape-map** (decode a latent point →
watch the shape morph). One of three sibling pathways; the most ambitious / highest-variance because it
re-adds the reconstruction objective that the 2026-05-26 denoised retrain showed is a shape dead-end.

## Loss
```
L = λ_nt·NT-Xent(z, z_aug)            # contrastive — the clustering driver
  + λ_recon·BCE(x_recon, x)           # reconstruction (generative half), kept modest
  + β·KL                              # KL — kept LOW (β=1.0 was the dead-end's prime suspect)
  + λ_lc·‖z(x) − z(shift(x))‖²        # latent-consistency — explicit invariance pressure
  + λ_d·MSE(dF/dt_decoded, dF/dt_true)# ridge-derivative — shape fidelity
```
`z`/`z_aug` are posterior **means** (deterministic embeddings for contrastive/consistency); the decoder
reconstructs from a **sampled** `z`. The decoded ridge for the derivative term is a **soft-argmax
expected-frequency** proxy over the band rows (differentiable); the **target** dF/dt is the cached
Viterbi ridge (`track_ridge`), never computed in the loop.

## Loss-weight schedule (user-locked: contrastive-dominant anneal)
`annealed_weights(cfg, epoch)`: `λ_nt`/`λ_lc` constant; `λ_recon`/`β`/`λ_d` ramp linearly 0→full over
`[recon_anneal_start, recon_anneal_start+recon_anneal_epochs]`. Prevents reconstruction from hijacking
the latent before clusters form.

## Augmentation (the dF/dt invariance, realized as the positive pair)
`augment_pitch_time_shift`: per-sample integer pitch shift (vertical) + time shift (horizontal),
**non-wrapping** (pad+slice), with the pitch shift **clamped in-band** via a ceil/floor integer
interval so the ridge never crosses 20–120 kHz. `time_warp_range` is RESERVED (declared, not yet
applied). In-band span per patch derived from energy extent.

## Inputs / reuse (do NOT rebuild)
- Denoised patches: `results/denoised_patches/combined_denoised/patches.npz` (rig, 16.7 GB).
- **Shared Track-0 cache** (built by `extract_ridge_targets_v3.py`, also used by Pathway A):
  `dFdt_true (N,T-1)`, `valid_mask (N,T-1)` in kHz/frame → trainer ×1000 → Hz/frame.
- Base architecture: FROZEN `ImageVAE`/`ImageVAEConfig`/`image_vae_loss`/`PaddingSpec`/`_compute_band_slice`
  imported from `scripts/train_contour_vae_v2.py` (never modified).

## CLI (rig)
`--patches-npz --ridge-npz --output-model-dir --output-results-dir` plus all loss weights
(`--lambda-nt --lambda-recon --beta --lambda-lc --lambda-deriv`), `--nt-temperature --softargmax-temp`,
augmentation (`--max-df-hz --max-dt-frames`), anneal (`--recon-anneal-start --recon-anneal-epochs`),
and `--epochs --batch-size --lr --workers --device --seed`. Outputs `best.pt`/`last.pt`/`hyperparams.json`
+ `training_log.csv`.

## Eval gates (shared `eval_shape_vae_v3.py`, DEFERRED)
shape η² (clear 0.12 decisively, target ≥0.50, stretch ≥0.58) · k-NN purity + geometric-type NMI
(>0.04, target >0.20) · pitch/onset η² LOW · **navigable-map decode-grid** (unique B+A payoff) ·
HDBSCAN continuum check · UMAP vs registration + production VAE. **Kill** if shape η² < 0.12 after
tuning, or if contrastive-only Pathway B matches it with fewer knobs.

## Public functions (unit-tested)
`ShapeVAEv3Config`, `soft_argmax_ridge`, `nt_xent`, `latent_consistency`, `derivative_loss`,
`augment_pitch_time_shift`, `annealed_weights`, `hybrid_loss` (assembly reference — see its docstring's
BAND-INPUT contract). Plus `ShapeVAEv3Hybrid` (model) and `DenoisedPatchRidgeDataset` (rig-only).

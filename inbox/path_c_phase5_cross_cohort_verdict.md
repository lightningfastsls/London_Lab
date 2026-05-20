---
title: Path C Phase 5 cross-cohort diagnostic — CLEAN verdict
date_captured: 2026-05-20
source_type: methodology-validation
status: applied
---

## TL;DR

The Phase 5 cross-cohort cage-confound diagnostic, run on a contour-masked
VAE trained jointly on all 4 cohorts (5970, 3452, 9252, lab_131204; 69,293
patches), returned a **CLEAN** verdict: **0 of 32 latent dimensions** cross
the pre-registered firing thresholds. The contour-mask methodology generalizes
from within-cohort (validated earlier on 5970 alone) to cross-cohort. The
combined latent space is safe to treat as a single space for downstream
biological analysis — cohort identity is not encoded.

## Decision rule (pre-registered, locked)

A latent dim `z_K` **fires** if BOTH conditions hold:
- `|Cohen's d| > 1.5` between any pair of cohorts on `z_K`
- `|Pearson r| > 0.4` between `z_K` and `principal_freq_hz`

Verdict logic:
- 0 fire = **CLEAN** (this run)
- 1–3 fire = partial leak (tighten mask)
- 4+ fire = cage signature dominates (methodology rethink)

Source of truth: `scripts/cage_confound_diagnostic.py` (locked — do not modify).

## Observed values

- **Dims that fire both**: 0 of 32
- **Dims with |d| > 1.5 alone**: 0 of 32 (max observed |d| = 0.77 on z_20, between 3452 and lab_131204)
- **Dims with |r| > 0.4 alone**: 0 of 32 (max observed |r| = 0.36 on z_24)
- **Sample size**: 69,293 patches (lab_131204 55,863 | 5970 12,440 | 9252 584 | 3452 406)
- **Latent dim**: 32

Top 10 dims by max |Cohen's d| (all below threshold):

| dim  | max |d| | pair                       | r vs principal_freq |
|------|---------|----------------------------|---------------------|
| z_20 | 0.77    | (3452, lab_131204)         | −0.17               |
| z_31 | 0.73    | (3452, lab_131204)         | −0.24               |
| z_24 | 0.65    | (5970, 9252)               | +0.36               |
| z_30 | 0.59    | (5970, lab_131204)         | −0.08               |
| z_27 | 0.54    | (9252, lab_131204)         | −0.22               |
| z_26 | 0.54    | (9252, lab_131204)         | −0.04               |
| z_2  | 0.54    | (9252, lab_131204)         | −0.12               |
| z_18 | 0.46    | (5970, lab_131204)         | +0.15               |
| z_15 | 0.41    | (9252, lab_131204)         | +0.04               |
| z_0  | 0.40    | (3452, lab_131204)         | −0.11               |

The strongest cohort separations are (3452, lab_131204) and (9252, lab_131204) — i.e. the cohort with the smallest sample (3452, n=406) vs the largest (lab_131204, n=55,863). This is consistent with sample-size-induced Cohen's d inflation (small-sample mean estimates have higher variance), and is in any case well below the firing threshold.

## What the verdict supports

1. **The contour-masked VAE methodology removes cross-cohort cage signature.**
   The contour mask (±5 kHz around the rlowess-smoothed DeepSqueak ridge,
   tonality threshold = 0) preserves enough call-specific signal for the
   VAE to learn meaningful latents, while stripping the broadband and tonal
   cage artifacts that otherwise separate cohorts.

2. **The combined-cohort latent space is operationally a single space.**
   Downstream analyses (call-type clustering, sequential structure mining,
   individual differences, behavioral context regression) can be performed
   on the combined latents without cohort being a confound.

3. **The N=4 cohort co-embedding holds across different wild dyads and a
   lab cohort.** Recording-environment heterogeneity (different microphones,
   placements, ambient noise) does not leak through the contour mask.

## What the verdict does NOT support

1. The VAE is **sufficient** for any specific downstream question — that's
   a separate test per question.
2. Within-cohort biology is preserved — cohort-internal structure (e.g.
   individual differences within 5970's dyad) was not the Phase 5 question.
3. Generalization to cohorts NOT in the training set — the verdict is
   in-distribution. An out-of-distribution test (encode a held-out cohort,
   check distribution overlap) is separate.

## Pipeline configuration captured at verdict-time

- **STFT canonical**: `corpus.SAMPLE_RATE_HZ = 300_000`, `STFT_HOP = 128`,
  `STFT_N_FFT` per corpus. Adaptive window per DeepSqueak's
  `CreateFocusSpectrogram` formula.
- **Contour extraction**: `scripts/deepsqueak_focus_stft.py` Python port,
  with the 2026-05-19 freq-offset fix (use `focus.fr_hz[freq_lo_idx]`,
  not `call_box.freq_start_kHz`, as the freq-offset anchor — see
  `notes/...freq-offset-1-based-vs-0-based...md` once authored).
- **Mask**: hard bandwidth ±5 kHz, tonality_threshold = 0,
  `scripts/contour_mask_utils.py`.
- **Patch shape**: (257, 234) float32 power; band-cropped to 170 freq bins
  (USV band 20–120 kHz per corpus), zero-padded to (256, 256) for the VAE.
- **VAE architecture**: 4× stride-2 channel-doubling encoder + mirrored
  decoder + sigmoid output + BCE-reconstruction ELBO; latent_dim=32;
  base_channels=32; bottleneck=(256, 16, 16); 7,746,673 params.
- **Training hyperparams**: batch_size=32, lr=2.5e-4, beta=1.0,
  max_epochs=500, patience=50, seed=42, with **logvar clamp [-10, 10]**
  in the encoder for numerical stability.
- **Training outcome**: early-stopped at epoch 108, best epoch 58, best
  val_recon = 203.01, final train_recon = 188.25, final KL ≈ 26 (train/val).
- **Compute**: rig (cloudyclaude), 1× RTX 3060 Ti via `CUDA_VISIBLE_DEVICES=2`,
  ~5 hours wall-clock for training.

## Numerical-stability fixes that made this verdict possible

These are not corpus constants — they're implementation details of the
training script that, without them, would have produced a degenerate or
crashing model. Captured here because they're load-bearing:

1. **Encoder `logvar.clamp(-10, 10)`** in `ImageVAE.encode`. Without it,
   exp(logvar) in the KL term overflows to Inf → KL=NaN → model trains
   without latent regularization → effectively a deterministic autoencoder,
   not a VAE. Phase 5 diagnostic interpretation requires proper VAE.
2. **BCE input `nan_to_num + clamp(0, 1)`** in `image_vae_loss`. Defensive
   guard against numerical edge cases that produce out-of-range x_recon.
3. **`MaskedPatchDataset` lazy per-getitem preprocessing**. The original
   eager log1p in `__init__` cached ~11 GB on small-RAM hosts; the lazy
   version is O(1) memory at the cost of ~12 min CPU over a 90 min run.
4. **`assemble_combined_patches.py` memmap-streaming concat**. Original
   in-memory concat had ~32 GB peak; memmap version has ~1.5 GB peak.
   Required on the rig (31 GB RAM, heavily co-tenanted).
5. **`np.load(..., mmap_mode='r')`** in train_contour_vae_v2.py:587. Avoids
   loading the full 16 GB combined patches.npz into RAM at startup.

Without #1, training would have produced a non-VAE. Without #2–5, training
would have OOM-crashed before producing any model on the rig.

## Caveats worth recording for downstream readers

- **Class imbalance**: lab_131204 = 80.6% of combined patches. Per-dim
  Cohen's d on pairs involving lab_131204 may be biased toward "lab is
  different" simply because lab dominates the cohort distribution.
  Re-running with stratified per-cohort sampling would address this.
- **Single seed**: training ran with seed=42 only. The Phase 5 verdict
  could in principle be seed-sensitive. A second seed (or 3-seed mean)
  would harden the result, though the margin (max |d|=0.77 vs threshold
  1.5) suggests the verdict is robust.
- **Patch_idx bug in latents.parquet**: the original training run wrote a
  corrupted latents.parquet (cross-joined on per-cohort patch_idx values).
  Re-encoded post-hoc via `re_encode_latents.py` (in `$CLAUDE_JOB_DIR`).
  Source-side fix applied 2026-05-20 in
  `scripts/train_contour_vae_v2.py` (§1a) and
  `scripts/assemble_combined_patches.py` (§1b), with regression tests in
  `tests/test_train_contour_vae_v2.py` + `tests/test_assemble_combined_patches.py`.
  See `docs/handoffs/2026-05-20_path-c-cleanup.md` for the cleanup record.

## Artifacts

- `models/contour_vae_combined/best.pt` (31 MB, epoch 58)
- `models/contour_vae_combined/last.pt` (31 MB, epoch 108)
- `models/contour_vae_combined/hyperparams.json`
- `results/contour_vae_combined/training_log.csv` (per-epoch history)
- `results/contour_vae_combined/latents.parquet` (69,293 × 32 latents +
  provenance)
- `results/contour_vae_combined/reconstructions/` (20 sanity-check PNGs)
- `results/phase5_cross_cohort/index.html` (Phase 5 report)
- `results/phase5_cross_cohort/diagnostic_result.json` (verdict + parameters)
- `results/phase5_cross_cohort/per_dim_diagnostic.csv` (per-dim Cohen's d,
  pairs, r, fires)
- `results/phase5_cross_cohort/umap_cohort.png` (latent UMAP colored by
  cohort)

All on the rig at `/data/shachar/contour_vae/...`; subset SCPd to local
worktree per cleanup handoff §4.

## Wikilinks (for /reflect to traverse)

- [[cage-confound-diagnostic]] — the locked diagnostic rule
- [[contour-mask-methodology]] — the masking approach
- [[image-vae-architecture]] — the model
- [[vae-cross-cohort-validation]] — this verdict's category
- [[cross-animal-population-strata]] — population-level interpretation caveats
  (compare to feedback memo on wild-vs-lab strata)
- [[cohort-imbalance-sampling]] — the lab_131204 dominance caveat
- [[freq-offset-1-based-vs-0-based]] — the 2026-05-19 fix that enabled
  this verdict by removing systematic downward bias in the Python contours

## Supersedes

Nothing directly. Extends the within-cohort 5970 methodology test (the
"Refinement F" run captured in earlier worktree handoffs).

## Cleanup work captured separately

See `docs/handoffs/2026-05-20_path-c-cleanup.md` for the post-verdict
cleanup tasks: source-side bug fixes, regression tests, artifact pull-back,
backup cleanup, and SSH credential audit.

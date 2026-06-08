# Pathway B (contrastive shape encoder) — CLOSED (KILL). Canonical shape clustering = registration.

**Date:** 2026-05-28  **Predecessor:** `docs/handoffs/2026-05-27_shape-vae-B-contrastive-invariance.md` (now CLOSED).
**HTML report:** `$CLAUDE_JOB_DIR/session_report.html` (this session's wrap).

## Decision (locked)

Pathway B's 2-D contrastive shape encoder, on the full 69,293 denoised-patch corpus with the recommended defaults (NT-Xent τ=0.2, pitch ±15 kHz, time ±20 fr, warp 0.9–1.1×, 40 epochs), produced **shape η² = 0.044** (held-out n=6,929). Independently verified to ~1e-8 by a fresh agent.

Per the original handoff's kill criteria:
> shape η² < 0.12 AND k-NN purity no better than production ⇒ KILL; ship registration (0.75).

Both conditions met (shape η² 0.044 ≪ 0.12; chevron purity 0.293 below the 3-class base rate of ~0.33). **The contrastive 2-D image direction is closed.**

## What this leaves canonical

- **Production shape clustering = registration → KMeans** on the registered ridge (`models/shape_kmeans/k20.joblib`, shape η² 0.58–0.75 on TRUE ridges). Already productionized 2026-05-25 (`docs/handoffs/2026-05-25_productionize-shape-registration.md`).
- The leaderboard now has **5 falsified 2-D image attempts** (production 0.099, denoised retrain 0.081, Pathway B contrastive 0.044, M10 image-VAE 0.009) and **2 successful preprocessing approaches** (registration 0.58–0.75, M8 1-D-on-registered 0.42, M9 1-D contrastive 0.34). The empirical pattern is clear: shape lives in the 1-D registered ridge, not the 2-D pixel grid.

## Methodological lesson to encode

1. **Shift augmentation creates LOCAL invariance only.** ±15 kHz makes positive pairs locally pitch-invariant, but USVs span 20–120 kHz; the encoder still represents pitch centroid (pitch η² = 0.306 ≠ 0).
2. **The 2-D image objective itself is the wrong substrate for shape**, even contrastively. The "no reconstruction term → won't spend capacity on pitch" argument is structurally clean and empirically wrong: 2-D negatives in NT-Xent can be pushed apart by pitch/extent (the easy cues) regardless of reconstruction.
3. **1-D registered ridge is the right substrate.** Once you've removed pitch/position/duration by construction, even a tiny VAE (M8: 0.42) or contrastive (M9: 0.34) gets useful shape η².

## Open follow-ups (not blocking)

- **Sibling Pathway A** (`train_shape_vae_v3_deriv.py`, λ_d=1.0 β=0.1, 60 ep) finished while waiting. Its scorecard is under `/data/shachar/contour_vae/results/shape_vae_v3_deriv/run_ld1_beta0p1/` on the rig. Score it with the same eval harness to confirm whether the VAE-family is FULLY closed. Predicted: similar failure (A still has reconstruction on 2-D image; the dominant failure mode applies).
- **UMAP figure** for Pathway B: skipped (rig venv lacks `umap-learn`). Re-runnable after `pip install umap-learn` on the rig if needed for a figure in a talk; verdict unchanged.
- **Memory note** `[[project_shape_registration_clustering]]` was appended this session to record Pathway B's KILL.

## Files NOT to touch

- `models/shape_kmeans/k20.joblib` (production registration alphabet)
- `scripts/experiments/rig_R2_shape_alphabet.py` (productionization driver)
- `src/usv_spectrogram/corpus.py`, `ExtractionConfig`, detection pipeline
- `train_contour_vae_v2.py` (frozen baseline)

## Done means

This closure is itself the deliverable: the contrastive frontier is documented as falsified, registration confirmed canonical, methodological lessons captured. Re-opening Pathway B would require a substantively different design (different augmentation, different loss, or different input substrate) — not a hyperparameter sweep.

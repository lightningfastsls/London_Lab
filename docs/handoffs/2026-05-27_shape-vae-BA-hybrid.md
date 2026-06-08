# HANDOFF — Pathway B+A: Hybrid contrastive encoder + VAE decoder + derivative term

**Date:** 2026-05-27  **Status:** READY — APPROVAL_PENDING (no code until the receiving chat presents a CLAUDE.md approval request)
**Owner thread:** shape-vs-pitch clustering (Shachar + Mickey)
**Canonical plan:** `PLAN_geometric_shape_clustering_vae.md` §3 Option B+A (executable spec).
**Sibling pathways (separate handoffs, run independently):** `2026-05-27_shape-vae-A-derivative-loss.md`, `2026-05-27_shape-vae-B-contrastive-invariance.md`.

## The idea — clustering power AND a navigable generative shape-map
Combine the two sibling pathways into one model:
- **Contrastive invariance** (from B): positive pair = pitch/time-shifted patch → the encoder learns
  pitch/position don't matter (clusters chevron-with-chevron, jump-with-jump).
- **VAE decoder** (the "A" half): a light reconstruction head so the latent is *generative* — you can
  decode a point and watch the shape morph → the **navigable 2-D shape-map** Mickey wants (chevron
  region → jump region), not just an opaque embedding.
- **Ridge-derivative term**: `MSE(dF/dt_decoded, dF/dt_true)` on the decoded ridge for shape fidelity.

**Run this when you want BOTH a clustering win and a generative/navigable latent.** If you only want
clusters, B-contrastive is simpler and has fewer failure modes; if you only want to test the literal
derivative hypothesis, A is the clean test. B+A is the ambitious, higher-variance option.

## ⚠️ Honest prior — why this is the riskiest of the three
It reintroduces the reconstruction objective that the 2026-05-26 denoised dead-end (shape η² 0.081)
showed spends capacity on pitch/duration pixel-variance. The bet here is that the **contrastive +
shift-augmentation + latent-consistency** terms dominate the reconstruction term and pull the latent
toward shape anyway. That is plausible but unproven — **weight the contrastive/consistency terms
heavily relative to reconstruction, and keep β LOW** (KL over-smoothing is the prime suspect from the
dead-end, which used β=1.0). More knobs = more ways to fail; budget for tuning.

## Prerequisites — shared Track 0 (denoised patches + ridge F(t))
1. **Denoised patch corpus already exists on the rig — do NOT rebuild** (16.7 GB):
   `results/denoised_patches/combined_denoised/patches.npz` (69,293 × 257 × 234), manifest
   `mask_kind=denoised`.
2. **Per-patch true ridge F(t) required** (derivative *target*, as in pathway A): extract once via
   `src/usv_spectrogram/features/ridge_tracker.py::track_ridge`, cache (~24 min/corpus; rich denoised
   patches are ~50× slower than masked — compute once, never in the train loop).
3. Base: `ImageVAE` / `ImageVAEConfig` in `scripts/train_contour_vae_v2.py` (256², `latent_dim=32`).
   **New script `scripts/experiments/train_shape_vae_v3_hybrid.py` — do NOT overwrite
   `train_contour_vae_v2.py`.**

## Design
- **Loss:**
  `L = λ_nt·NT-Xent(z, z_aug) + λ_r·recon(BCE/MSE) + β·KL + λ_lc·‖z(x) − z(shift(x))‖² + λ_d·MSE(dF/dt_decoded, dF/dt_true)`
  - **NT-Xent** over the (patch, shifted-patch) positive pair — the clustering driver.
  - **Latent-consistency** `‖z(x) − z(shift(x))‖²` — a second, explicit invariance pressure on the
    encoder (cheap insurance that shifts collapse in latent space).
  - **recon + KL** — the generative/navigable half. Keep `λ_r` modest and `β` low.
  - **Derivative term** on the decoded ridge (decode → `track_ridge`/soft-argmax proxy → finite-diff →
    match true dF/dt). Same differentiability note as pathway A: Viterbi `track_ridge` is non-diff →
    use a soft-argmax expected-frequency proxy for the in-loss ridge, or a detached-target schedule.
  - Expose **all** weights (`λ_nt`, `λ_r`, `β`, `λ_lc`, `λ_d`) as CLI flags. Suggested staging:
    start contrastive-dominant (high `λ_nt`, tiny `λ_r`/`β`), then anneal in recon/derivative once
    clusters form — prevents the reconstruction term from hijacking early training.
- **Augmentation (shared with B):** vertical `Δf ±15 kHz` (wrap/zero-pad), horizontal `Δt ±N frames`,
  optional time-warp 0.9–1.1×; **clamp the ridge inside 20–120 kHz** (corpus band). No destructive crop.

## Eval gates (held-out split; print params, thresholds, sort keys, row counts)
Reuse `eta2(v, lab)` / `register_one(crop, freqs_khz)`; build `scripts/eval_shape_vae_v3.py` (shared
with pathway A — one eval script can score all three pathways for a clean head-to-head).
1. **shape η²** — clear 0.12 decisively; target ≥ 0.50, stretch ≥ 0.58. Below 0.12 ⇒ KILL.
2. **k-NN neighbor purity** + **geometric-type NMI** (> 0.04, target > 0.20).
3. **Pitch / onset η² LOW** (invariance worked).
4. **Navigable-map check (the B+A payoff):** decode a grid/geodesic through the latent → does the shape
   morph *smoothly* (chevron → jump)? This is the unique deliverable vs pathway B. Qualitative figure.
5. **Continuum vs discrete (Track D):** HDBSCAN on the latent — crisp clusters or continuum? If
   continuum, deliver the **navigable shape-map** framing (this pathway is built for exactly that).
6. **UMAP** coloured by shape-type vs registration + production VAE.

## Kill criteria
- shape η² < 0.12 after tuning ⇒ KILL.
- If it clears 0.12 but the contrastive-only pathway B matched/beat it with fewer knobs ⇒ prefer B and
  retire the hybrid (don't pay for a decoder you don't need).
- If the decoded latent doesn't morph smoothly (no navigable map) AND clustering ≤ B ⇒ no unique value.

## Orchestration — how the receiving chat should run this (skills + tools)
1. **Start:** `/execute docs/handoffs/2026-05-27_shape-vae-BA-hybrid.md`. Read in full. Orchestrator role.
2. **Knowledge check:** `/kcheck` on "contour VAE contrastive derivative navigable shape map invariance";
   re-read memory `project_shape_registration_clustering` and `project_c06_empty_cluster`.
3. **Approval gate:** CLAUDE.md approval request before code. The **loss-weight schedule** (contrastive-
   dominant vs joint-from-start) and the differentiable-ridge proxy are genuine decisions — surface both.
   This pathway has the most hyperparameters; agree a tuning budget with the user up front.
4. **Tests first (/implement Step 0):** spawn **test-architect** for `train_shape_vae_v3_hybrid.py`
   (each loss term assembles + is finite, latent-consistency is shift-invariant by construction,
   augmentation in-band) and the shared `eval_shape_vae_v3.py` BEFORE implementation.
5. **Compute dispatch:** rig, **background job** / subagent — this is the longest-training pathway; do
   not block the orchestrator. **Verify the scorecard + the navigable-map figure with a fresh
   `Explore`/general-purpose agent.**
6. **Review:** **master-reviewer** against this handoff + plan §4 (esp. gate 4/5 navigable-map +
   continuum), then **test-hardener**.
7. **Close:** **`/wrap-session`** → HTML head-to-head (all three pathways vs registration 0.75 vs
   production 0.12 vs denoised 0.081) **plus** the decoded shape-morph figure. **Print the
   `file://wsl.localhost/Ubuntu/<path>` URL** in the final message (mandatory; never `explorer.exe`).

## Compute / data movement (rig)
- Rig `shachar@100.113.224.57` (cloudyclaude, 3× RTX 3060 Ti), repo `/data/mickey_london_lab`,
  venv `/data/mickey_london_lab/.venv`, `PYTHONPATH=/data/mickey_london_lab/src`. Run from
  `/data/mickey_london_lab`.
- **Read-only SSH pre-authorized; training launches gated** — get the user's per-session OK first.
- Box→rig ~0.5–0.7 MB/s (NAT) → operate on the rig. **Box OOM-crashes WSL** — never full-scan
  `patches.npz` locally. `docker stop llama-large` (as shachar) frees a GPU if contended.

## Relevant constraints (flattened — do not violate)
- **Corpus constants:** import sr / USV band (20–120 kHz) / STFT from `src/usv_spectrogram/corpus.py`;
  load `data/corpus_facts/{dataset}.json`. Never redeclare. Augmentation clamp uses the band.
- **DO NOT TOUCH:** production detection pipeline (`scripts/run_batch_detection.py`,
  `app/core/sliding_inference.py`, `postprocessing/`), `ExtractionConfig`, `corpus.py` constants
  (CNN-frozen), committed `models/`, `train_contour_vae_v2.py` (frozen baseline).
- **Print discipline:** params, thresholds, sort keys, filter row counts on every eval run.
- **Continuum caveat:** shape-space is a continuum even after registration — this pathway is the one
  built to *embrace* that (navigable map). Don't force K crisp clusters as the success criterion.

## Reusable assets (do NOT rebuild)
- Denoised corpus (rig): `results/denoised_patches/combined_denoised/patches.npz` + manifest.
- Baseline scorecards (rig): `results/eval_shape/score_{masked_baseline,denoised}.json`; caches
  `desc_{masked,denoised}.npz`.
- Registration ceiling: `models/shape_kmeans/k20.joblib`, `results/latent_transitions/shape_alphabet/`
  (rig-only; see `docs/DATA_LOCATIONS.md`).
- Box-side helpers: worktree `contour-vae-denoised-retrain/scripts/` (uncommitted) — stage by exact path.

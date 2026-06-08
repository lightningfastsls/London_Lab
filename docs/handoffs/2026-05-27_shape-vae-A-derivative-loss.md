# HANDOFF — Pathway A: Derivative-loss contour-VAE (the literal dF/dt hypothesis)

**Date:** 2026-05-27  **Status:** READY — APPROVAL_PENDING (no code until the receiving chat presents a CLAUDE.md approval request)
**Owner thread:** shape-vs-pitch clustering (Shachar + Mickey)
**Canonical plan:** `PLAN_geometric_shape_clustering_vae.md` §3 Option A (this handoff is the executable spec for that option).
**Sibling pathways (separate handoffs, run independently):** `2026-05-27_shape-vae-B-contrastive-invariance.md`, `2026-05-27_shape-vae-BA-hybrid.md`.

## The idea (Shachar's proposal, done faithfully)
Add a **ridge-derivative term** to the VAE loss so the model is penalised for getting the
**slope dF/dt** (and optionally curvature d²F/dt²) of the call's frequency contour wrong.
dF/dt is invariant to absolute pitch (a constant shift differentiates away) and to time
position, so in principle it pushes the latent toward *shape* and away from pitch/duration.

**This is the literal reading of "the derivative loss signal I proposed" — and it has NOT yet
been tested faithfully.** Read the two prior near-misses so you don't repeat them:
- **M8** (`scripts/experiments/rig_M8_contour_vae.py`) put the derivative term on a **1-D ridge
  that was already registered** (pitch removed by preprocessing). The term had nothing left to
  do → `deriv` 0.425 ≤ `plain` 0.497. It never tested "can the loss *create* invariance."
- **M10** (`scripts/experiments/rig_M10_image_vae.py`) used a **pixel gradient** `MSE(∂I/∂t)+MSE(∂I/∂f)`
  as the "derivative." A pixel gradient is **not** dF/dt and is **not** pitch-invariant — shift a
  call up 5 kHz and every lit pixel moves rows, so the term is large for an identical shape. → 0.009.

**The faithful test (this handoff):** keep the **2-D denoised image** as input (so jumps /
sub-harmonics survive) and compute the derivative on the **decoded ridge** — decode the recon,
run `track_ridge()`, finite-difference F(t), and match it to the **true** ridge dF/dt. The
derivative acts on the 1-D ridge *extracted from* the 2-D recon, never on raw pixels.

## ⚠️ Honest prior — the bar this must clear
The 2026-05-26 denoised retrain (no derivative term) was a **dead end**: shape η² **0.081** vs
0.099 masked baseline vs **0.75** registration ceiling; latent still sorted by pitch(0.527)/
dur(0.404). The recorded conclusion: an un-registered image-VAE spends latent capacity on the
dominant *pixel*-variance axes (pitch=vertical position, duration=horizontal extent) and ignores
the low-variance ridge curve. **Your derivative term is the one mechanism that could re-weight the
objective toward that low-variance curve.** That is exactly why it is worth a clean test — but it
is also why it may not be enough on its own. Treat this as a hypothesis test, not a sure win.

## Prerequisites — shared Track 0 (denoised patches + ridge F(t))
1. **Denoised patch corpus already exists on the rig — do NOT rebuild** (16.7 GB):
   `results/denoised_patches/combined_denoised/patches.npz` (69,293 × 257 × 234), manifest
   `mask_kind=denoised`. Built by `mass_apply_denoised_patches.py` (STFT → band-crop →
   `src/usv_spectrogram/features/spectrogram_filter.py::prefilter_spectrogram` → no contour mask).
2. **MISSING and required for this pathway:** the per-patch **true ridge F(t)** array, used as the
   derivative *target*. Extract once with `src/usv_spectrogram/features/ridge_tracker.py::track_ridge`
   (the same Viterbi tracker that feeds registration) and store as a parallel array in a new
   `.npz` next to the patches. NOTE: `track_ridge` is ~50× slower on denoised (rich) patches than
   on masked ridges (≈24 min/corpus) — compute once, cache, never recompute in the train loop.
3. Base architecture: `ImageVAE` / `ImageVAEConfig` in `scripts/train_contour_vae_v2.py`
   (256², `latent_dim=32`, BCE+β·KL). **New script `scripts/experiments/train_shape_vae_v3_deriv.py`
   — do NOT overwrite `train_contour_vae_v2.py`** (it is the frozen baseline).

## Design
- **Loss:** `L = recon (BCE/MSE) + β·KL + λ_d·MSE(dF/dt_decoded, dF/dt_true) [+ λ_c·MSE(d²F/dt²)]`.
  - `dF/dt_decoded`: decode → `track_ridge()` on the recon crop → finite-diff. Differentiability:
    `track_ridge` (Viterbi) is non-differentiable, so either (a) use a **soft argmax / expected
    frequency per column** on the decoded magnitude as a differentiable ridge proxy, then diff it,
    or (b) train in two terms with a detached-target schedule. State which you chose and why.
  - Expose `λ_d`, `λ_c`, `β` as CLI flags. Start `β` LOW (the dead-end used β=1.0; KL over-smoothing
    is a prime suspect). Consider `latent_dim` ≥ 32.
- **Optional (Track-B idea #2):** mask the recon loss to the call region (weight by the prefilter
  mask returned by `prefilter_spectrogram`) so the objective isn't dominated by getting background
  right. Flag-gated; report with/without.
- **No augmentation in this pathway** — invariance is meant to come from the derivative term, not
  from shift-augmentation. (Shift-augmentation is the *separate* B / B+A pathways. Keeping them
  separate is the whole point of "different handoffs for different ideas" — do not merge them here,
  or you lose the ability to attribute any gain to the derivative term specifically.)

## Eval gates (held-out split; print params, thresholds, sort keys, row counts)
Reuse `eta2(v, lab)` and `register_one(crop, freqs_khz)` from `scripts/experiments/rig_M8_contour_vae.py`
/ `rig_R2_shape_alphabet.py`. Build `scripts/eval_shape_vae_v3.py` (may fork the existing
`eval_denoised_vae_shape.py` logic — it already does register_one/eta2 with a `desc_*.npz` cache).
1. **shape η²** — must clear **0.12 decisively**; target ≥ 0.50; stretch ≥ 0.58. Below 0.12 ⇒ KILL.
2. **Pitch & duration η²** — must DROP vs the dead-end's 0.527 / 0.404 (the direct test that the
   derivative term moved the latent off pitch/position).
3. **Geometric-type NMI** vs `syllable_type` (chevron / jump / up-FM / down-FM / flat / complex) —
   must beat production VAE's **0.04**; target > 0.20. *Verify the label column exists in
   `classified_detections_*` before relying on it.*
4. **UMAP of the latent** coloured by shape-type, side-by-side vs registration + production-VAE.

## Kill criteria (avoid sunk cost)
- shape η² < 0.12 after tuning `λ_d`/`β` ⇒ KILL; the derivative term cannot rescue the image-VAE
  objective and registration (0.75) is the answer — say so plainly and stop.
- Clears 0.12 but ≪ registration's 0.58 AND shows no multi-component/jump capture registration
  lacks ⇒ a learned model that merely under-performs registration at higher cost. Report, don't ship.

## Orchestration — how the receiving chat should run this (skills + tools)
1. **Start:** `/execute docs/handoffs/2026-05-27_shape-vae-A-derivative-loss.md`. Read this in full
   first. This is orchestrator work — you are the conductor, not the implementer.
2. **Knowledge check:** run `/kcheck` on "contour VAE shape clustering derivative loss ridge
   registration" before designing the loss — this thread has a dense constraint history (M8/M10/
   denoised dead-end). Also re-read memory `project_shape_registration_clustering`,
   `project_c06_empty_cluster`, `project_shape_registration_clustering`.
3. **Approval gate:** present a CLAUDE.md approval request (Intent/Context/Scope/Plan/Assumptions/
   Risks/Validation/Learning) BEFORE any code. The differentiable-ridge choice (soft-argmax vs
   detached target) is a genuine design decision — surface it for the user.
4. **Tests first (/implement Step 0):** spawn **test-architect** to write tests for
   `train_shape_vae_v3_deriv.py` (loss assembles, derivative term shape-correct, ridge cache loads)
   and `eval_shape_vae_v3.py` (eta2/register_one wired) BEFORE writing the implementation.
5. **Compute dispatch:** training runs on the **rig** (see below). Launch it as a **background job**
   (`Bash run_in_background`) or a dedicated subagent; do not block the orchestrator on a multi-hour
   GPU run. Monitor; when it finishes, **verify the results with a fresh `Explore`/general-purpose
   agent** (fresh eyes on the scorecard) rather than trusting the training log.
6. **Review:** after results, spawn **master-reviewer** (check against this handoff + the plan §4
   gates), then **test-hardener** for edge cases.
7. **Close:** run **`/wrap-session`** → HTML report with the head-to-head η² table (this pathway vs
   registration 0.75 vs production 0.12 vs denoised dead-end 0.081). **Include the
   `file://wsl.localhost/Ubuntu/<path>` URL in your final message** (mandatory — SendUserFile alone
   is not enough; never `explorer.exe`).

## Compute / data movement (rig)
- Rig `shachar@100.113.224.57` (cloudyclaude, 3× RTX 3060 Ti), repo `/data/mickey_london_lab`
  (non-git rsync copy), venv `/data/mickey_london_lab/.venv`, `PYTHONPATH=/data/mickey_london_lab/src`.
  Run from `/data/mickey_london_lab` (full `src`); `/data/shachar/contour_vae/src` is stale.
- **Read-only SSH inspection is pre-authorized; compute writes/launches are gated** — get the user's
  per-session OK before kicking off a training run. A scoped-null SSH result ≠ "absent on rig".
- Box→rig transfer is ~0.5–0.7 MB/s (NAT) — **regenerate/operate on the rig**, don't push 17 GB.
- The **box OOM-crashes WSL** under load (12-core/11 GiB); never full-scan `patches.npz` locally.
- Free a GPU if `docker llama-server` contends: `docker stop llama-large` as shachar; leave GPU0's
  contour-VAE and the embedding service alone.

## Relevant constraints (flattened — do not violate)
- **Corpus constants:** import sample rate / USV band (20–120 kHz) / STFT from
  `src/usv_spectrogram/corpus.py`; load `data/corpus_facts/{dataset}.json`. Never redeclare them.
- **DO NOT TOUCH:** production detection pipeline (`scripts/run_batch_detection.py`,
  `app/core/sliding_inference.py`, `postprocessing/`), `ExtractionConfig`, `corpus.py` constants
  (all CNN-frozen), committed `models/`, and `train_contour_vae_v2.py` (frozen baseline).
- **Print discipline:** every eval run prints params, thresholds, sort keys, and filter row counts.
- **Continuum caveat:** shape-space is a continuum even after registration (UMAP→HDBSCAN). Evaluate
  with η² / NMI / neighbor purity, not "did we get K crisp clusters."

## Reusable assets (do NOT rebuild)
- Denoised corpus (rig): `results/denoised_patches/combined_denoised/patches.npz` + manifest.
- Dead-end baseline scorecards (rig): `results/eval_shape/score_{masked_baseline,denoised}.json`;
  descriptor caches `desc_{masked,denoised}.npz`.
- Registration ceiling assets: `models/shape_kmeans/k20.joblib`,
  `results/latent_transitions/shape_alphabet/` (see `docs/DATA_LOCATIONS.md`; rig-only).
- Box-side helper scripts live in worktree `contour-vae-denoised-retrain/scripts/` (uncommitted) —
  `mass_apply_denoised_patches.py`, `eval_denoised_vae_shape.py`. Stage by exact path if adopting.

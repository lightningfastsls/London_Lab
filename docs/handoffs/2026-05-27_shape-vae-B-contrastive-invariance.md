# HANDOFF — Pathway B: Invariance-augmentation contrastive encoder (the untested frontier)

**Date:** 2026-05-27  **Status:** READY — APPROVAL_PENDING (no code until the receiving chat presents a CLAUDE.md approval request)
**Owner thread:** shape-vs-pitch clustering (Shachar + Mickey)
**Canonical plan:** `PLAN_geometric_shape_clustering_vae.md` §3 Option B-contrastive (executable spec).
**Sibling pathways (separate handoffs, run independently):** `2026-05-27_shape-vae-A-derivative-loss.md`, `2026-05-27_shape-vae-BA-hybrid.md`.

## The idea — "M9 done right," and the one VAE-family idea not yet falsified
A 2-D conv encoder trained with **NT-Xent contrastive loss**, where the **positive pair is
(patch, pitch-shifted + time-shifted patch)**. Because the only difference between the pair is
absolute pitch and time position, the encoder is *forced to learn that those don't matter* — which
is exactly the invariance dF/dt gives you, but realised through the data instead of the loss math.
This directly optimises the literal goal: **similar shapes close, dissimilar far** → chevron-with-
chevron, jump-with-jump.

**Why this is the live frontier:** every other learned attempt has been falsified for shape
clustering — M8 1-D-deriv (0.425), M10 pixel-edge (0.009), the 2026-05-26 denoised retrain (0.081).
The recorded conclusion from the dead-end is that an un-registered *reconstruction* objective spends
capacity on pitch/duration pixel-variance. **A contrastive objective has no reconstruction term** —
it never has to "spend capacity" representing pitch/position, and the shift-augmentation actively
*punishes* doing so. This is the structural reason it can succeed where reconstruction VAEs failed.

**Prior near-miss to fix:** M9 (`scripts/experiments/rig_M9_contrastive.py`) was the right *family*
but ran on the **1-D ridge only** with **time-warp + noise** augmentations — never pitch/time shift,
never 2-D geometry. It still scored 0.34 (best learned). This pathway = M9 on the **2-D denoised
image** with **pitch/time-shift** augmentation. Two fixes to the only learned method that nearly worked.

## Prerequisites — shared Track 0 (denoised patches; ridge F(t) only for eval)
1. **Denoised patch corpus already exists on the rig — do NOT rebuild** (16.7 GB):
   `results/denoised_patches/combined_denoised/patches.npz` (69,293 × 257 × 234), manifest
   `mask_kind=denoised`. (STFT → band-crop → `prefilter_spectrogram` → no contour mask.)
2. This pathway needs **no ridge target for training** (no reconstruction/derivative term). The true
   ridge F(t) is needed only at **eval** time to compute shape η² — extract via
   `src/usv_spectrogram/features/ridge_tracker.py::track_ridge` once and cache (~24 min/corpus; the
   `desc_denoised.npz` cache makes reruns instant).
3. **New script `scripts/experiments/train_shape_encoder_contrastive.py`.** You may reuse the conv
   stack from `ImageVAE` (`scripts/train_contour_vae_v2.py`) as the encoder backbone, but **do NOT
   overwrite `train_contour_vae_v2.py`** and **do NOT** add a decoder (this pathway is encoder-only).

## Design
- **Encoder:** 2-D conv (reuse `ImageVAE`'s 1→32→64→128→256 stack) → projection head → embedding.
- **Loss:** NT-Xent (normalized temperature-scaled cross-entropy). Standard SimCLR-style: each batch
  patch + its augmented positive; all other patches are negatives. Expose temperature `τ` as a flag.
- **Augmentation (the invariance idea, shared with B+A):**
  - Vertical shift `Δf ∈ ±U kHz` (U ≈ ±15 kHz), wrap or zero-pad. **Clamp so the ridge stays inside
    20–120 kHz** (corpus band) — never push a call out of band.
  - Horizontal shift `Δt ∈ ±N frames`.
  - Optional mild time-warp 0.9–1.1× so duration is *partially* invariant too (decide whether you
    want duration-invariance; flag it).
  - Do NOT add destructive crop-registration (that mangled M10). Augment, don't crop.
- No KL, no β, no reconstruction — fewer knobs than the VAE pathways, and no over-smoothing failure mode.

## Eval gates (held-out split; print params, thresholds, sort keys, row counts)
Reuse `eta2(v, lab)` and `register_one(crop, freqs_khz)`; build `scripts/eval_shape_encoder.py`.
1. **k-NN neighbor purity** (the most literal test of the goal): for a sample of chevrons/jumps,
   fraction of k=10 embedding-NN sharing the geometric type. Report per-type.
2. **Geometric-type NMI** (k-means on embedding vs `syllable_type`) — must beat production VAE's
   **0.04**; target > 0.20. *Verify the label column exists before relying on it.*
3. **shape η²** on registered-ridge shape — must clear **0.12**; target ≥ 0.50, stretch ≥ 0.58.
4. **Pitch / onset η² must be LOW** — the direct test that shift-augmentation made the embedding stop
   caring about mean frequency / position. This is the metric this pathway should win most cleanly.
5. **Jump / multi-component capture** (the 2-D payoff a 1-D ridge can't give): do frequency-jump and
   two-component calls form their own embedding neighborhood? Qualitative.
6. **UMAP** of the embedding coloured by shape-type, side-by-side vs registration + production VAE.

## Kill criteria
- shape η² < 0.12 AND k-NN purity no better than production ⇒ KILL; ship registration (0.75).
- Clears 0.12 and pitch η² drops but still ≪ registration AND no jump-capture edge ⇒ report, don't ship.
- **Success that beats registration OR matches it while capturing jumps/sub-harmonics registration
  can't represent ⇒ this is the win Mickey asked for ("finally get an algorithm to cluster shapes").**

## Orchestration — how the receiving chat should run this (skills + tools)
1. **Start:** `/execute docs/handoffs/2026-05-27_shape-vae-B-contrastive-invariance.md`. Read in full.
   Orchestrator role — conduct, don't hand-code the whole thing inline.
2. **Knowledge check:** `/kcheck` on "contrastive shape clustering invariance augmentation USV";
   re-read memory `project_shape_registration_clustering` (M9 details + dead-end conclusion).
3. **Approval gate:** CLAUDE.md approval request before code. The augmentation magnitude (`U`, `N`,
   whether to include time-warp) is a real design decision tied to *which invariances you want* —
   surface it (e.g. duration-invariance may or may not be desired for the shape goal).
4. **Tests first (/implement Step 0):** spawn **test-architect** for
   `train_shape_encoder_contrastive.py` (NT-Xent shape/sign correctness, augmentation stays in-band,
   positive-pair construction) and `eval_shape_encoder.py` BEFORE implementation.
5. **Compute dispatch:** train on the **rig** as a **background job** or subagent; don't block the
   orchestrator. **Verify the scorecard with a fresh `Explore`/general-purpose agent.**
6. **Review:** **master-reviewer** against this handoff + plan §4, then **test-hardener**.
7. **Close:** **`/wrap-session`** → HTML head-to-head (this pathway vs registration 0.75 vs production
   0.12 vs denoised 0.081 vs M9 0.34). **Print the `file://wsl.localhost/Ubuntu/<path>` URL** in the
   final message (mandatory; never `explorer.exe`).

## Compute / data movement (rig)
- Rig `shachar@100.113.224.57` (cloudyclaude, 3× RTX 3060 Ti), repo `/data/mickey_london_lab`,
  venv `/data/mickey_london_lab/.venv`, `PYTHONPATH=/data/mickey_london_lab/src`. Run from
  `/data/mickey_london_lab`.
- **Read-only SSH pre-authorized; training launches gated** — get the user's per-session OK first.
- Box→rig ~0.5–0.7 MB/s (NAT) → operate on the rig. **Box OOM-crashes WSL** — never full-scan
  `patches.npz` locally. Free GPU contention with `docker stop llama-large` (as shachar) if needed.

## Relevant constraints (flattened — do not violate)
- **Corpus constants:** import sr / USV band (20–120 kHz) / STFT from `src/usv_spectrogram/corpus.py`;
  load `data/corpus_facts/{dataset}.json`. Never redeclare. The augmentation clamp uses the band.
- **DO NOT TOUCH:** production detection pipeline, `ExtractionConfig`, `corpus.py` constants
  (CNN-frozen), committed `models/`, `train_contour_vae_v2.py` (frozen baseline).
- **Print discipline:** params, thresholds, sort keys, filter row counts on every eval run.
- **Continuum caveat:** shape-space is a continuum even after registration. The honest deliverable may
  be "geometrically similar calls are neighbors" (purity/NMI), not "K disjoint clusters." Frame it so.

## Reusable assets (do NOT rebuild)
- Denoised corpus (rig): `results/denoised_patches/combined_denoised/patches.npz` + manifest.
- Baseline scorecards (rig): `results/eval_shape/score_{masked_baseline,denoised}.json`; caches
  `desc_{masked,denoised}.npz`. M9 outputs: `/data/shachar/contour_vae/results/latent_transitions/m9_contrastive/`.
- Registration ceiling: `models/shape_kmeans/k20.joblib`, `results/latent_transitions/shape_alphabet/`
  (rig-only; see `docs/DATA_LOCATIONS.md`).

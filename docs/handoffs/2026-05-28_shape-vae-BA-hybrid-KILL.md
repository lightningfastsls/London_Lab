# VERDICT — Pathway B+A (hybrid contrastive + VAE + ridge-derivative): **KILL**

**Date:** 2026-05-28  **Status:** CLOSED — kill criteria met (spec §5).
**Predecessor handoffs:**
  · `docs/handoffs/2026-05-27_shape-vae-BA-hybrid.md` (build spec)
  · `docs/handoffs/2026-05-27_shape-vae-BA-hybrid-rig.md` (gated rig launch)
**Plan:** `PLAN_geometric_shape_clustering_vae.md` §3 Option B+A, §4 eval gates, §5 kill criteria.

## Scorecard (run1 best.pt, 5970, K=20, N=12,440 valid / 12,440 total)
| Axis | Value | Spec | Verdict |
|---|---|---|---|
| **shape η²** | **0.1046** | kill < **0.12** · target ≥ 0.50 · stretch 0.58 (registration ceiling 0.75) | **FAIL — below kill threshold** |
| pitch η² | 0.2621 | low ⇒ pitch-invariance (failure mode flag) | residual pitch sorting |
| dur η²   | 0.1707 | low ⇒ duration-invariance (failure mode flag) | residual duration sorting |

Scorecard JSON: `results/shape_vae_v3_hybrid/run1/killgate.json`.
Eval script: `scripts/experiments/eval_shape_kill_gate_v3.py` (CLI, K=20, seed=42, eta2 def matches M9/M10/R1/R2).

## Head-to-head — shape-η² bake-off (2026-05-25 ⇒ today)
| Representation | shape η² | Comment |
|---|---|---|
| **Registration + KMeans** (baseline) | **0.58 – 0.75** | unsupervised, cheap, *production-ready* |
| M9 — 1-D contrastive (NT-Xent on registered ridge) | 0.344 | best *learned* (1-D, no shift-aug) |
| Production contour-VAE (K=20) | 0.12 | the original pitch/duration sorter we were trying to fix |
| **B+A hybrid (this run)** | **0.1046** | **KILL** — slightly worse than production VAE |
| 2026-05-26 denoised retrain (no derivative term) | 0.099 | prior dead-end (β=1.0) |
| **Pathway B — 2-D contrastive (sibling chat, today)** | **0.044** | **KILL** — encoder-only, no recon, *still failed* |
| M10 — image-VAE + pixel-edge loss | 0.009 | wrong derivative + destructive crop |

The sibling chat's Pathway B result (`IMPLEMENTATION_PROGRESS.md` entry 2026-05-28, successor handoff
`docs/handoffs/2026-05-28_pathway-B-kill-and-canonical.md`) is the decisive parallel evidence: an
**encoder-only** 2-D contrastive model — no reconstruction, no KL, no decoder, *no dead-end risk by
construction* — scored shape η² **0.044**, **lower** than my B+A's 0.105. So my hypothesis "the
reconstruction term is the saboteur" only explains *part* of the failure: stripping recon entirely
makes it *worse*, not better. The 2-D image substrate itself is the wrong place to put shape.

## Why it died — the mechanism is now crisp
Training curve (`results/shape_vae_v3_hybrid/run1/training_log.csv`, epochs 0-79):

| epoch | nt | recon | kl | lc | deriv | w_recon |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2.91 | 121k | 6.3 | 2.28 | 708k | 0.000 |
| 9 | **1.48** | 122k | 2.20 | **0.65** | 712k | 0.000 |
| 19 | 3.46 | 9.4k | 423 | 251 | 455k | 0.0225 |
| 29 | 3.57 | 11k | 342 | 282 | 375k | 0.0475 |
| 49 | 3.71 | 9.2k | 279 | 239 | 276k | 0.0500 |
| 78 | 3.78 | 10k | 243 | 199 | 196k | 0.0500 |

Two clean regimes, separated exactly by the anneal:

- **ep 0-9 (pure contrastive, w_recon=0):** `nt` dropped 2.91 → 1.48, `lc` dropped 2.28 → 0.65. The encoder *was* learning a shift-invariant shape representation.
- **ep 10+ (recon/KL/deriv anneal in):** `nt` jumped 1.48 → 3.5+ and **stayed there**; `lc` exploded 0.65 → 200+ and **stayed there**. The reconstruction term overrode the contrastive structure as soon as it was allowed to contribute.

The pitch (0.26) and duration (0.17) η² at the end confirm what survived: the latent sorts by what BCE+pixel-MSE *can* see — vertical position and horizontal extent — exactly what the un-registered image-VAE objective was predicted to do.

This is the **fourth independent confirmation** of the family-wide failure, three of them on the same day:
1. 2026-05-25 — 11-representation bake-off: registration 0.58–0.75 ≫ every learned method (incl. M8 deriv loss).
2. 2026-05-26 — denoised retrain (no derivative term): shape η² 0.081.
3. 2026-05-28 — **Pathway B (encoder-only, sibling chat):** shape η² **0.044** (no recon, no KL, no decoder — and *still* failed; the cleaner test).
4. 2026-05-28 — **this run (B+A hybrid):** shape η² **0.1046**.

Even with every documented mitigation (contrastive-dominant anneal, β LOW, λ_recon=0.05, latent-consistency=1.0, soft-argmax differentiable ridge, dF/dt cached target), the un-registered image-VAE objective spends latent capacity on pitch/duration pixel-variance and ignores ridge curvature. The hypothesis "contrastive + shift-augmentation will dominate reconstruction" — the whole point of B+A — is empirically refuted.

The sibling B chat's sharper diagnosis is the one that probably explains both runs: **shift augmentation creates LOCAL invariance only** — ±15 kHz is small compared to the 100 kHz USV band, so the encoder learns to ignore pitch *within ±15 kHz windows* but the pitch *centroid* still leaks (B's pitch η² 0.306, my B+A's pitch η² 0.262). My B+A used the same ±15 kHz augmentation and inherited the same flaw. Beyond that, **the 2-D image substrate itself is wrong for shape regardless of loss family** (5 falsified 2-D attempts now vs successful 1-D-on-registered).

## Kill verdict (per `PLAN_geometric_shape_clustering_vae.md` §5)
**shape η² 0.1046 < 0.12 ⇒ KILL.** Strictly the spec says "after tuning"; we used 1 of the 3-4 run budget. Tuning was deliberately not exhausted because:
- Pre-anneal (w_recon=0) the encoder *did* learn well (nt 1.48, lc 0.65 at ep9). The collapse is caused by activating the reconstruction term *at all*, not by the weight value — no λ_recon > 0 will avoid the mechanism, only its magnitude. The bug is structural, not configurational.
- Pushing further (β=0, λ_recon=0.01, longer anneal start) could plausibly nudge η² past 0.12 but **cannot** approach the 0.50 target, let alone the 0.75 registration ceiling — and even if it did, it would only re-derive at higher cost what registration already gives unsupervised.
- The spec second bullet — "if contrastive-only Pathway B matched/beat it with fewer knobs ⇒ prefer B" — applies in spirit: B+A's whole reason to exist was the *generative navigable shape-map*, which is meaningless when the underlying latent doesn't even cluster shape.

User decided KILL on 2026-05-28 after reviewing the run1 verdict.

## What to do instead (operational)
1. **SHIP REGISTRATION.** `models/shape_kmeans/k20.joblib` already exists from 2026-05-25
   (`docs/handoffs/2026-05-25_productionize-shape-registration.md`). Shape η² 0.58–0.75, unsupervised,
   already plumbed. This is the shipping answer for "geometrically similar calls are neighbors."
   (Same conclusion the sibling Pathway-B chat reached: see its successor handoff
   `docs/handoffs/2026-05-28_pathway-B-kill-and-canonical.md`.)
2. **The VAE-family is closed for shape clustering.** B+A (this) and B (sibling chat) failed today;
   denoised retrain failed 2026-05-26; M8/M9/M10 failed 2026-05-25 — five falsified 2-D / image-VAE
   attempts now. Do NOT re-attempt without a fundamentally different substrate (e.g. operate on the
   registered 1-D ridge, not the raw 2-D image).
3. **For a navigable shape-map** (continuum interpretation, plan Track D): build it on the
   **registered shape space**, not on a learned latent. UMAP / persistent-homology over the
   0.75-η² registered shapes gives a navigable 2-D map without a generative decoder.

## Files & disposition
**Built and kept (rig, code mirror):**
- `scripts/experiments/train_shape_vae_v3_hybrid.py` (model + loss + augmentation + anneal + train loop)
- `scripts/experiments/eval_shape_kill_gate_v3.py` (this verdict's eval)
- `tests/test_shape_vae_v3_hybrid.py` (50 spec tests), `tests/test_shape_vae_v3_hybrid_hardening.py` (25 adversarial)
- `docs/modules/shape-vae-v3-hybrid.md`, `docs/reviews/shape-vae-v3-hybrid-review.md`

**Rig artifacts (kept for reproducibility):**
- `/data/shachar/contour_vae/models/shape_vae_v3_hybrid/run1/{best.pt,last.pt,hyperparams.json}` (31 MB each)
- `/data/shachar/contour_vae/results/shape_vae_v3_hybrid/run1/{training_log.csv,killgate.json}`
- 5970 Track-0 ridge cache `/data/shachar/contour_vae/results/denoised_patches/5970/ridge_targets_v3.npz` (also shared with Pathway A) — keep, reusable.

**NOT executed (cancelled by kill):** runs 2-4 of the planned 3-4-run sweep; the shared eval `eval_shape_vae_v3.py` (deferred to Pathway A coordination — no longer needed for B+A).

## Pointer for the next session
- Memory note to read first: `project_shape_registration_clustering` (now records THREE confirmations of the un-registered image-VAE dead-end).
- Successor work, if any, is decided by the user — not auto-implied. Options live in §"What to do instead" above.

# PLAN — Geometric Shape Clustering via an Invariance-Aware Learned Encoder

**Date:** 2026-05-26  **Owner thread:** shape-vs-pitch clustering (Shachar + Mickey)
**Status:** PLAN / APPROVAL_PENDING — loss design is an open DECISION (see §3). No code until approved.
**Supersedes:** the Track B stub in `PLAN_shape_representation_v2.md` (this is Track B, fully scoped).

## The goal (in the user's words)
> "see if we can finally get an unsupervised learning algorithm to successfully cluster
> together USV that are geometrically close to one another — chevron to chevron, jump
> with jump, etc."

Restated operationally: a **learned, unsupervised** representation in which geometrically
similar calls are **neighbors**, **invariant to absolute pitch (mean frequency) and temporal
position**. The invariance is the mechanism the user proposed: the time-derivative of the
peak-frequency track, dF/dt, is unchanged when the whole call is shifted in pitch (the
constant differentiates away) or in time — so dF/dt is the canonical pitch/position-invariant
descriptor, and we bake it (or its consequence) into the training objective.

## ⚠️ The baseline this must beat — read before committing rig time
The 2026-05-25 bake-off (11 representations, 67,337 true ridges; memo
`project_shape_registration_clustering`, report `results/latent_transitions/shape_registered/SESSION_REPORT_shape_clustering.html`):

| Representation | shape η² | Notes |
|---|---|---|
| **Registration + k-means** (the baseline) | **0.58–0.75** | unsupervised, cheap, already built |
| M8 — 1-D VAE on *registered* ridge + deriv loss | 0.50 / 0.42 | derivative term did nothing: input was already pitch-free |
| M9 — contrastive 1-D (NT-Xent, time-warp/noise) | 0.34 | best *learned*; but 1-D ridge, no shift-aug |
| Production contour-VAE (K=20) | **0.12** | a pitch(0.45)/duration(0.47) sorter; chevron NMI **0.04** |
| M10 — image-VAE + pixel-edge loss | **0.009** | wrong derivative (pixel gradient ≠ ridge dF/dt) + destructive crop |

**Implication:** if the only deliverable is "clean shape clusters," **registration already wins
and is unsupervised** — the cheapest correct action is Track A (productionize registration) and
visually confirm chevron↔chevron / jump↔jump. A learned retrain is justified ONLY if at least
one of these holds, and the plan is gated on declaring which:
1. We want a **learned** representation specifically (Mickey's "finally get an algorithm to…"), and
   prior learned attempts were each crippled in a fixable way (see §1).
2. We want to capture geometry the **1-D registered ridge discards** — multi-component calls,
   frequency **jumps/steps** (a discontinuity the ridge tracker smooths over), sub-harmonic stacks.
   A 2-D image encoder can represent these; a 50-point 1-D ridge cannot.

**Continuum caveat (Track D):** UMAP→HDBSCAN found a *continuum* even on registered shapes.
The honest end-state may be a **navigable 2-D shape-map** (chevron region → jump region) rather
than N crisp "letters." Set Mickey's expectation accordingly: the win is "geometrically similar
calls are neighbors," not necessarily "K disjoint clusters." Evaluate with neighbor purity /
NMI, not only cluster count.

## §1 — Why the prior learned attempts don't count as a fair test (the fixable flaws)
- **M8** (`scripts/experiments/rig_M8_contour_vae.py`): loss `w0·MSE + w1·MSE(diff1) + w2·MSE(diff2) + β·KL`
  is mathematically your idea — but it ran on the **already-registered** ridge (`register_one()`
  subtracts mean pitch before the VAE sees it). The invariance was done by preprocessing; the
  derivative term only re-weighted curvature. Never tested "can the loss *create* invariance."
- **M9** (`rig_M9_contrastive.py`): NT-Xent contrastive — the right family for *clustering* — but
  on the **1-D ridge only**, and its augmentations were time-warp + noise, **not pitch/time shift**.
  So it never learned pitch-invariance and never saw 2-D geometry. Still scored 0.34 (best learned).
- **M10** (`rig_M10_image_vae.py`): `0.2·MSE_full + 1.0·MSE_masked + 1.0·edge + β·KL` where
  `edge = MSE(∂I/∂t) + MSE(∂I/∂f)` over **image pixels**. A pixel gradient is **not** dF/dt and
  **not pitch-invariant**: shift a call up 5 kHz and every lit pixel moves rows, so the gradient-MSE
  is large for an identical-shape call. Conflated "sharpen edges" with "ignore pitch." Plus a hard
  64×64 crop-registration mangled structure. Hence 0.009.

The faithful test (this plan) keeps the **2-D image** (so jumps/sub-harmonics survive) and enforces
invariance via the **ridge trajectory / latent**, not pixel gradients.

## §2 — PREREQUISITE (Track 0): regenerate patches from the denoised spectrogram
**Already resolved by the user 2026-05-25.** The current `patches.npz` uses a hard ±5 kHz contour
mask → ~95% black, call body removed; even a strong call is ~0.2 % nonzero as VAE input. **The VAE
never saw the calls.** No retrain is meaningful on these.
- **Recipe (per call window):** STFT → band-crop (20–120 kHz, corpus constants) →
  `features/spectrogram_filter.py::prefilter_spectrogram` (the SIS denoiser that already feeds the
  ridge tracker) → that denoised band patch is the new VAE input. **Drop the contour mask.**
- Regenerate all 4 cohorts on the rig; re-audit a contact sheet visually before any training.
- Keep the per-patch ridge F(t) from `track_ridge()` alongside each patch (needed for §3 options A/C
  and for the derivative target). Store as a parallel array in the new `.npz`.

## §3 — DECISION: the invariance loss (user decides after reading)
Base architecture: the production `ImageVAE` in `scripts/train_contour_vae_v2.py` (256², z=32,
BCE+β·KL). **New script** `train_shape_vae_v3.py` — do NOT overwrite v2. For the **clustering**
goal, the choice is between a VAE-with-invariance and a contrastive encoder; I lay out both.

| Option | Mechanism | Bakes in your idea via | Best when | Risk |
|---|---|---|---|---|
| **B-contrastive** ★ for clustering | 2-D conv encoder, **NT-Xent**; positive pair = (patch, **pitch-shifted + time-shifted** patch) | the positive pair differs only by pitch/position, so the encoder *learns* those don't matter — exactly dF/dt's invariance | the deliverable is neighbor structure / clusters (this goal) | no generative/navigable decoder unless added |
| **B+A VAE** ★ if generative latent also wanted | `ImageVAE` + **latent-consistency** `‖z(x)−z(shift(x))‖²` + **ridge-derivative** term `MSE(dF/dt_decoded, dF/dt_true)` | both: consistency = invariance; derivative term = shape fidelity | want a smooth navigable shape-map AND clustering | β can over-smooth; needs clean ridge for the deriv term |
| **A only** | `ImageVAE` + literal `MSE(dF/dt)` on the decoded ridge | the most literal reading of your words | you want the cleanest test of *the literal hypothesis* | 1-D term only nudges a 2-D latent; depends on ridge quality |
| **C** | reconstruct a **de-meaned** target (subtract each patch's ridge-mean freq before recon loss) | removes absolute pitch from the *target* | want "registration, inside the loss" | duration/position not handled; closest to just registering |

**Recommendation for THIS goal:** **B-contrastive** is the strongest tool for clustering — it
directly optimizes "similar close, dissimilar far," it *is* M9 done right (2-D denoised image +
pitch/time-shift augmentation instead of 1-D + warp/noise), and the shift-augmentation is your
invariance idea realized as the positive pair. If Mickey also wants a *navigable generative*
shape-map (decode a latent → see the shape morph), add a VAE decoder + small derivative term →
that's **B+A**. I will default the draft to a **hybrid: contrastive encoder + light VAE decoder +
ridge-derivative term**, with weights exposed as flags, unless you pick a single option.

**Augmentation spec (shared by B/B+A):** vertical shift Δf ∈ ±U kHz (U≈ a fraction of the band,
e.g. ±15 kHz) with wrap/zero-pad; horizontal shift Δt ∈ ±N frames; optionally small
time-warp (0.9–1.1×) so duration is partially invariant too. **No pitch-jitter so large it pushes the
call out of band** — clamp to keep the ridge inside 20–120 kHz.

## §4 — Eval gates (tuned to "chevron-with-chevron, jump-with-jump")
Run all on a held-out split; **print params, thresholds, sort keys, row counts** (lab convention).
1. **shape η²** (reuse `eta2()` from the rig scripts): on registered-ridge shape. **Must clear 0.12**
   decisively; **target ≥ 0.50**, stretch ≥ 0.58 (match registration). Below 0.12 ⇒ kill.
2. **Geometric-type NMI & k-NN neighbor purity** — the literal goal. Using existing shape-type
   labels (`syllable_type` / Holy-Guo: chevron, jump/step, up-FM, down-FM, flat, complex):
   - k-means(latent) vs type → **NMI must beat production VAE's 0.04** (target > 0.20).
   - For a sample of chevrons/jumps: fraction of k=10 latent-NN sharing the type (purity).
   *(Verify the label column exists in `classified_detections_*` before relying on it.)*
3. **Pitch/position invariance check** — η²(latent vs pitch) and η²(latent vs onset) should be
   **LOW** (the production VAE's 0.45 pitch η² is the failure we're fixing). This is the direct
   test that the loss made the latent "stop caring about mean frequency."
4. **Jump / multi-component capture** (the registration-can't payoff) — qualitative: do
   frequency-jump and two-component calls form their own latent neighborhood? (1-D ridge cannot
   represent these; this is the unique argument for the 2-D learned encoder.)
5. **UMAP of the latent**, colored by shape-type, side-by-side vs the registration UMAP and the
   production-VAE UMAP. The lab-presentation figure.
6. **Continuum vs discrete** (Track D): HDBSCAN on the latent — crisp clusters or continuum?
   Report honestly; if continuum, deliver the navigable map framing.

## §5 — Decision gate / kill criteria (avoid sunk cost)
- **Track A in parallel, first:** productionize registration (`docs/handoffs/2026-05-25_productionize-shape-registration.md`)
  and confirm it already clusters chevron↔chevron / jump↔jump. This is the baseline AND a possible
  ship on its own.
- **Proceed to the learned retrain only if** Track A leaves a gap this addresses (jumps/sub-harmonics,
  or "we want a learned model") — declared explicitly.
- **Kill the retrain if:** shape η² < 0.12 after tuning (worse than current), OR it neither beats
  registration's 0.58 NOR shows the multi-component capture registration lacks. A learned model that
  merely re-derives registration at higher cost is not a win — say so and ship registration.

## §6 — Compute, files, do-NOT-touch
- **Compute:** rig (`shachar@100.113.224.57`, GPU0/2/3). ~50× the box; the box OOM-crashed WSL under
  load. Watch for root-owned `docker llama-server` GPU contention (free via `docker stop llama-large`
  as shachar; leave the user's contour-VAE on GPU0 and the embedding service alone).
- **Data movement:** box→rig Tailscale relay ~0.5–0.7 MB/s (NAT traversal fails). Regenerate patches
  *on the rig* rather than pushing 17 GB; or subsample as in 18.4.
- **Touch:** a worktree (e.g. `latent-analysis-b-a-c`) + rig `/data/...`. New scripts:
  `regen_denoised_patches.py` (Track 0), `train_shape_vae_v3.py`, `eval_shape_vae_v3.py`.
- **DO NOT touch:** production detection pipeline (`scripts/run_batch_detection.py`,
  `app/core/sliding_inference.py`, `postprocessing/`), `ExtractionConfig`, `corpus.py` constants
  (CNN-frozen), `train_contour_vae_v2.py` (keep as baseline). DANN classifier work is unrelated.

## §7 — Order of operations
1. Track A (registration) productionized + visual chevron/jump confirmation → possible early ship.
2. Track 0: regenerate denoised patches on the rig + visual audit + store ridge F(t) per patch.
3. Pick the §3 loss (user decision) → implement `train_shape_vae_v3.py`.
4. Train on 5970 first; eval (§4) vs registration + production VAE.
5. If it clears the gate → encode all cohorts, build the UMAP/shape-map figure for the lab.
6. `/wrap-session` HTML report with the head-to-head table.

# Shape-Invariant Latent — Implementation Roadmap (v2)

> **Source plans:** `PLAN_shape_representation_v2.md`, `PLAN_geometric_shape_clustering_vae.md`
> **Adversarial review:** `docs/reviews/ROADMAP_SHAPE_INVARIANT_LATENT-adversarial.md` (incorporated below; all 11 required revisions addressed)
> **Goal:** A learned, unsupervised latent that clusters by **geometry** (chevron-with-chevron, jump-with-jump) and is invariant to **(a) cage / recording-environment, (b) absolute pitch, (c) onset+duration**.

> **Hard constraint:** 6/6 prior 2-D image-VAE attempts have been falsified. The VAE family was formally CLOSED on 2026-05-28 (`docs/handoffs/2026-05-28_pathway-B-kill-and-canonical.md`). The kill memo is binding — re-attempts are gated on a fundamentally different **substrate**, *and* must be defended against the modal hypothesis below.

> **Modal prior (explicitly committed):** P(shape genuinely lives in 1-D and 2-D substrate adds nothing useful for clustering) ≥ **0.5**. The Pathway B 0.044 vs Pathway B+A 0.105 comparison argues against my own substrate hypothesis, since pure-contrastive on raw imagery underperformed hybrid — if loss-mechanism alone explained the failures, B should have won. This roadmap is a falsifiable bet, not a confident prediction; failing modes are the expected outcomes.

> **Production fallback (always available):** `models/shape_kmeans/k20.joblib` — registration → KMeans, shape η² 0.58–0.75. Phase 0/1/2 are bets *on top of* this; the baseline ships regardless.

---

## Probability budget (incorporates adversary's counter-prior)

| Outcome | Probability |
|---|---|
| Phase 0a (linear probe) kills the whole roadmap before any GPU spend | **~50%** |
| Phase 1 ships shape η² ≥ 0.58 (matches/beats registration on substrate-independent eval) | ~15% |
| Phase 1 marginal (0.30–0.58) AND captures jumps/sub-harmonics registration misses | ~10% |
| Phase 1 marginal but no qualitative payoff → kill | ~15% |
| Phase 1 below 0.30 / pitch-leak → kill | ~10% |
| **Total: any worthwhile artifact** | **~25%** |

If the user wants a high-confidence ship, **ship registration permanently and close the family.** This roadmap is justified ONLY if a ~25% bet at ~6 hours expected wall-clock (most of which is Phase 0a's 1-hour kill-or-proceed) is better than that ship.

---

## How to Use This File

1. Work through phases **in strict order**. Phase 0a has a hard kill gate; failing it ends the roadmap with **zero further compute**.
2. Each phase has: **What** | **Hypothesis** | **Eval gates (Pass/Marginal/Kill)** | **Contingency** | **Compute/files**.
3. If a phase hits its KILL gate, stop and write the negative-result memo; do not silently re-tune.
4. The `[REVISED]` markers below tag changes from v1 driven by the adversarial review.

## Status Key

- **PROPOSED** | **READY** | **IN PROGRESS** | **DONE** | **KILLED**

---

## Unified failure mechanism this roadmap tests against

Six prior attempts (M10, production contour-VAE, denoised retrain, Pathway A, Pathway B, Pathway B+A) failed at shape η² 0.009–0.105. The kill memo's strong claim is *"shape lives in the 1-D registered ridge, not the 2-D pixel grid"*. This roadmap concedes that as the modal hypothesis and tests one *specific* alternative: **off-centerline 2-D structure** (sub-harmonics, jumps, second components) is the only candidate signal that 1-D ridges discard. If this signal exists and is recoverable, a learned encoder on a pre-registered 2-D substrate could add value beyond registration. **If it doesn't, no roadmap variant ships and the family closes.**

---

## Phase 0a — Linear-Probe Precursor `[REVISED — NEW PHASE]`

**Status:** PROPOSED  **Hard precursor — no other phase runs until this passes.**
**Hypothesis under test:** *If* a 2-D learned encoder can ever discriminate shape better than registration, *then* the existing failed Pathway B encoder's representations must already contain at least linearly-decodable shape signal. If a frozen-encoder linear probe on shape is at chance, no substrate-swap + retrain on the same architecture will rescue it.

### What to build

A 30-minute CPU job. **No training. No new model.** Reuse `/data/shachar/contour_vae/results/latent_transitions/b_contrastive/encoder.pt` (Pathway B's failed encoder, shape η² 0.044 on the prior eval).

1. Encode all 69,293 denoised patches with the **frozen** Pathway B encoder.
2. Fit two logistic-regression linear probes:
   - **Probe A (substrate-independent):** chevron-vs-non-chevron, where labels come from the *un-registered* ridge via `chevron_valley` heuristic in `scripts/eval_shape_encoder.py`. This is the outside-information eval — labels are not derived from any preprocessing the encoder also saw.
   - **Probe B (manual-label, if available):** any pairwise discrimination between `syllable_type` categories in `classified_detections_*.csv` (chevron/jump/flat/up-FM/down-FM/complex). Verify column exists before relying on it.
3. Report held-out accuracy (5-fold CV).
4. Encode the SAME patches with a **random-init** encoder (same architecture, untrained weights) and fit the same probes. Report random baseline.

### Eval gates

| Metric | Pass (proceed) | Kill (ship registration permanently) |
|---|---|---|
| Probe A acc (chevron heuristic) on frozen B encoder | ≥ **0.65** AND ≥ random-init + 0.10 | < 0.65 OR ≤ random-init + 0.05 |
| Probe B acc on manual labels (if available) | ≥ chance + 0.15 | < chance + 0.05 |

### Contingency

- **Both pass:** the encoder family has *some* shape-relevant signal. Proceed to Phase 0b.
- **Either kill:** the architecture can't represent shape regardless of substrate or loss. Close the family. Write `docs/handoffs/2026-05-28_shape-vae-family-CLOSED.md` documenting the linear-probe precursor as the decisive falsifier and listing the canonical action: **ship `models/shape_kmeans/k20.joblib` as the permanent shape representation; do not re-open the VAE family for shape clustering.**

### Compute / files

- **Box CPU** is fine (encoder forward is fast). No rig contention.
- New script: `scripts/experiments/probe_shape_existing_encoder.py`.
- Reuse: `scripts/eval_shape_encoder.py::chevron_valley` (label function), `train_shape_encoder_contrastive.py::ContrastiveEncoder` (architecture; random-init re-instantiates).
- Estimated wall-clock: **~30–60 min**.

---

## Phase 0b — DSP Synthetic Two-Tone Validation `[REVISED — NEW PHASE]`

**Status:** PROPOSED (blocked on Phase 0a pass)
**Hypothesis under test:** The per-column ridge-shift registration produces pixel-equivalent (modulo intended translation) outputs for two synthetic patches with identical shape and differing only in absolute pitch. *If not, the substrate has a structural bug that no Phase 1 hyperparameter can fix.*

### What to build

A pure-Python test (no training, no rig). For each of {linear-frequency, log-frequency} shift modes:

1. Generate two synthetic patches: chevron with identical shape parameters but centerline pitch 60 kHz vs 90 kHz. Add a synthetic sub-harmonic at 2·F(t).
2. Run Phase 0c's registration code on both.
3. Verify the registered patches' centerline rows are within 1e-4 pixel-equivalent.
4. Measure where the synthetic sub-harmonic lands. **Log-frequency mode passes only if the sub-harmonic ends up at the same row in both registered patches.**
5. Measure the padding region's shape (which rows are padded). **Pass only if the padding shape is identical between the two pitches.**

### Eval gates

| Check | Pass | Kill |
|---|---|---|
| Centerline pixel-equivalence (both patches) | both modes | one or both fail |
| Sub-harmonic ends at same row | log-frequency mode | only linear mode (then log mode is mandatory for Phase 0c) |
| Padding region invariant to pitch | log-frequency mode | linear fails (mandatory: log mode) |
| dF/dt-dependent interpolation artifact present? | absent in integer-bin-rounded shift; present in fractional shift | fractional shift introduces F-dependent blur (mandatory: integer-bin rounding) |

### Contingency

- **Pass on log-frequency + integer-bin rounding:** Phase 0c uses log-frequency shift with integer-bin rounding. **This is the locked decision for the substrate.**
- **Log-frequency fails on padding/interpolation:** abort the substrate hypothesis. The structural pitch leak the adversary predicted (§1.1) is unavoidable; Phase 1 would fail. Close family.

### Compute / files

- **Box CPU**, ~30 min including writing the test.
- New script: `scripts/experiments/test_registration_synthetic.py`. Pytest test cases.
- Estimated wall-clock: **~30 min–1 hour**.

---

## Phase 0c — Pre-Registered 2-D Patch Generator `[REVISED]`

**Status:** PROPOSED (blocked on Phase 0a + 0b pass)
**Hypothesis under test:** A geometric canonicalization of the 2-D denoised patch can be defined that preserves discriminative shape signal beyond the 1-D ridge.

### What to build

`scripts/experiments/make_registered_patches_2d.py`. Inputs: existing denoised patches + per-patch ridge `F(t)` (from `ridge_targets_v3.npz`). Locked DSP choices from Phase 0b:

1. **Resample patch onto log-frequency grid** (logarithmic spacing between corpus `USV_FREQ_MIN_HZ` and `USV_FREQ_MAX_HZ`; reuse, don't redeclare).
2. **Smooth-extrapolate `F(t)` into silent frames** by edge-padding from nearest active frame `[REVISED — Q2 resolved away from the discontinuity recipe]`. NaN-ridge patches (entire ridge undefined; ~5–10% per the c06 noise-sink memo) are **dropped from the substrate, not zero-shifted** `[REVISED]`. Dropped patches keep their existing latent for downstream analysis but exit this experiment.
3. **Integer-bin rounding on shift:** `shift_t = round(log2(F(t) / F_target))`, applied per column. `F_target` = global geometric mean of all retained `F̄`.
4. **Time canonicalization:** crop to active span, resample to 96 frames via linear resampling (Q3 resolved). Store original `(onset_frame, active_duration_frames, mean_pitch_hz, SNR)` in meta.
5. **Active-region mask:** `active_mask[t, f]` boolean; downstream losses MUST honor it.

### Outputs

- `/data/shachar/contour_vae/results/registered_patches/combined/registered.npz` with `patches`, `pad_mask`, `meta`.
- 50-call contact sheet `results/registered_patches/audit/`.
- **Quantitative sub-harmonic preservation metric** `[REVISED]`: fraction of off-centerline energy variance for the subset of patches with `>1` ridge-tracked component, vs the same ratio for un-registered patches. (If multi-component labels are unavailable, downgrade Phase 0c to "this experiment does not test the multi-component payoff" and note it in the eval-validity caveats.)

### Eval gates

| Check | Pass | Marginal | Kill |
|---|---|---|---|
| Visual audit (50 patches): ridge on centerline | ≥ 48/50 | 40–47/50 | < 40/50 (bug in shift code; debug Phase 0c) |
| Off-centerline energy fraction preserved on multi-component patches | ≥ 0.5 × unregistered ratio | 0.2–0.5× | < 0.2× (multi-component info destroyed; downgrade Phase 1 scope or kill) |
| Drop rate (NaN-ridge patches removed) | < 15% | 15–25% | > 25% (substrate is no longer the same corpus; comparability to baselines breaks) |
| Padding fraction | mean < 30% | 30–50% | mean > 50% (patches too sparse; resize or kill) |

### Contingency

- **All pass:** proceed to Phase 1.
- **Multi-component preservation killed:** the 2-D payoff hypothesis (only argument for going beyond 1-D registration) is dead. Phase 1 still runs as a "did the substrate fix the prior failure mode at all?" test, but at lower expected return.
- **Drop rate high:** the substrate excludes 25%+ of corpus; results are not directly comparable to the 6/6 prior runs. Document carefully; do not over-claim Phase 1 wins/losses.

### Compute / files

- **Rig CPU job**. No GPU. Estimated wall-clock: **2–4 hours** including audit `[REVISED — was 1 hour in v1]`.

---

## Phase 1 — Steel-Man Contrastive Encoder on Registered Substrate `[REVISED — eval rewritten]`

**Status:** PROPOSED (blocked on Phase 0c)
**Hypothesis under test:** Pathway B's contrastive encoder on the log-registered substrate (with `[REVISED]` augmentation strategy) clears the substrate-independent eval. If it doesn't, the family is closed.

### What to build

Modified rerun of `scripts/experiments/train_shape_encoder_contrastive.py`. Add `--registered` flag for input swap.

**Augmentation strategy (revised per adversary §4):** the substrate canonicalization is *imperfect* (per Phase 0c residual leak from interpolation, edge effects, padding-shape variation). Augmentations are the regularizer that fights residual leak.

- **Keep modest pitch-shift augmentation** (±5 kHz, much smaller than B's ±15 kHz) — penalizes residual pitch leak in the latent without creating B's "local-invariance only" failure.
- **Keep modest time-shift augmentation** (±5 frames in the registered frame).
- **Add shape-preserving axis augmentations** `[REVISED — new]`: per-column amplitude perturbation (±20% multiplicative), frequency-band-dependent noise floor jitter (model cage noise variation), pixel-level dropout 0.1.
- **NT-Xent τ=0.2** retained.
- **Mask-aware loss:** apply NT-Xent on encoder output; if any reconstruction is added later (Phase 2 NEVER runs adversarial — see below), mask by `pad_mask`.

### Eval gates `[REVISED — substrate-independent primary]`

All evals on a held-out 10% call-level split, stratified by cohort.

**Primary (substrate-independent, REQUIRED for ship):**
| Metric | Pass | Marginal | Kill |
|---|---|---|---|
| **NMI vs `syllable_type` manual labels** (if column exists) | ≥ 0.25 | 0.15–0.25 | < 0.15 |
| **chevron-vs-non-chevron k-NN purity (k=10)** with chevron labels from the *un-registered* ridge heuristic | > 0.55 | 0.40–0.55 | ≤ chance + 0.05 (i.e. ≤ ~0.55 in a 2-class balanced sample, ≤ ~0.40 in a stratified case) |
| **Linear probe accuracy** (logistic on latent → `syllable_type`) vs Phase 0a's frozen-B linear-probe accuracy | beat by ≥ 0.10 | beat by 0–0.10 | does not beat Phase 0a probe |

**Secondary (substrate-relative, INTERPRETED with controls):**
| Metric | Pass | Marginal | Kill |
|---|---|---|---|
| shape η² (label = `register_one()` output) | ≥ 0.58 | 0.30–0.58 | < 0.30 |
| **random-init encoder baseline** for shape η² `[REVISED — NEW]` | must be > random-init + 0.20 | gap 0.10–0.20 | gap < 0.10 (eval is measuring substrate, not encoder) |
| **identity baseline** for shape η² (per-column-average row of registered patch → KMeans-20) `[REVISED — NEW]` | learned must beat identity by ≥ 0.10 | beats by 0–0.10 | does not beat identity (learned model adds nothing) |

**Failure-mode gates (sanity, must pass):**
| Metric | Pass | Kill |
|---|---|---|
| pitch η² (latent vs original `mean_pitch_hz` from Phase 0c meta) | ≤ 0.05 | > 0.20 (substrate didn't fix pitch leak) |
| duration η² | ≤ 0.05 | > 0.20 |
| cohort η² | ≤ 0.10 | > 0.25 (cage confound dominates — proceed to Phase 2 *only* if primary evals also pass) |

### Contingency

- **All primary pass AND random-init/identity baselines beaten with margin AND cohort η² ≤ 0.10:** ship as the navigable shape-map. Side-by-side comparison with production before replacing/augmenting `models/shape_kmeans/k20.joblib`.
- **All primary pass BUT cohort η² > 0.10:** ship Phase 1; proceed to Phase 2 (post-hoc residualization, ~30 min, no retraining) to scrub cage. Document residualization as a finalization step.
- **Primary marginal AND identity/random-init NOT beaten with margin:** the encoder isn't doing real work — the eval is measuring substrate quality. Document the negative result; **do not ship as a learned model.** Possibly ship Phase 0c's registered substrate + KMeans (identity baseline) as an alternative to production if it beats production on the primary substrate-independent evals.
- **Primary kill:** close the family. The substrate hypothesis is falsified at the eval-validity-controlled level. Write the negative-result memo. The modal prior (1-D ridge is the shape ceiling) is ratified.
- **Pitch/dur η² > 0.20 with primary pass:** suspect, not failure. The substrate leaked pitch/duration but the latent still discriminates shape on substrate-independent labels. Document; ship with a caveat.

### Compute / files

- **Rig GPU2 or GPU3** (coordinate with Pathway A's 17 GB cache per `docs/modules/shape-encoder-contrastive.md` — run alone or wait).
- Modified: `scripts/experiments/train_shape_encoder_contrastive.py` (add `--registered` flag; do not break existing `--denoised` behavior).
- Modified: `scripts/eval_shape_encoder.py` (add `--baselines random,identity` flag).
- Estimated wall-clock: **6–10 hours including 1 substrate retry** `[REVISED — was 2 hours in v1]`. Including rig contention with Pathway A, budget **1–2 days end-to-end**.

---

## Phase 2 — Post-Hoc Cage Residualization `[REVISED — completely rewritten, no adversarial training]`

**Status:** PROPOSED (blocked on Phase 1 primary-pass with cohort η² > 0.10)
**Hypothesis under test:** Phase 1's latent's cage dependence is linear or near-linear in observed nuisance covariates `(cage_id, mean_pitch_hz, duration, SNR)`. *If so*, partial-regression (Frisch–Waugh–Lovell) residualizes it away without retraining. *If not*, cage is non-linearly entangled with shape and adversarial training would also have failed for the same reason.

### Why this replaces v1's FiLM + DANN proposal `[REVISED — major change]`

The adversary's §5 attack is correct: v1's Phase 2 re-runs the B+A failure mechanism. The B+A kill memo says: *"no λ_recon > 0 will avoid the mechanism, only its magnitude. The bug is structural, not configurational."* v1's Phase 2 added λ_recon ≤ 0.01 with substrate change as the asserted (not argued) safeguard — that's exactly the hand-wave the kill memo predicted would fail. **Cut.**

Post-hoc residualization gets the same cage-scrubbing effect without any retraining, in ~30 minutes:

1. Encode all train/val patches with Phase 1's frozen encoder. Get latent matrix `Z` (N × 32).
2. Build nuisance covariate matrix `X` (N × K): one-hot `cage_id`, plus `mean_pitch_hz`, `duration`, `SNR` (continuous).
3. Residualize: `Z_resid = Z - X @ (X.T @ X)^{-1} @ X.T @ Z`. (FWL: regress out X column-wise from Z.)
4. Re-run all Phase 1 evals on `Z_resid`.

### Eval gates

| Metric | Pass | Kill |
|---|---|---|
| cohort η² on residualized latent | ≤ 0.05 | > 0.10 (residualization didn't help; cage is non-linearly entangled with shape) |
| shape η² regression vs Phase 1 latent | ≤ 0.05 drop | > 0.15 drop (cage and shape are linearly entangled; the cage-clean latent has no shape signal left) |
| Primary substrate-independent evals (NMI, chevron k-NN, linear probe) | within 0.05 of Phase 1 | regress > 0.10 |

### Contingency

- **All pass:** ship Phase 1 + post-hoc residualization as the production navigable shape-map.
- **shape η² regresses > 0.15:** cage and shape are linearly entangled. **Do not adversarially train** — that would chase the same entanglement non-linearly and produce the same failure with higher cost. Document cage as an unresolved confound; ship Phase 1 unresidualized with the caveat.
- **Cohort η² doesn't drop:** cage isn't linear in the covariates we have. Either the cage signal is in higher-order interactions, or in features outside `(cage_id, pitch, duration, SNR)`. Document; ship Phase 1 with the caveat. *Do not* attempt adversarial training; the prior on that succeeding is ≤ 5% per the B+A and 18.4 DANN failures.

### Compute / files

- **Box CPU.** ~30 minutes.
- New script: `scripts/experiments/cage_residualize_latent.py`.
- Estimated wall-clock: **~30 min** for the residualization + reruns of the Phase 1 eval pipeline.

---

## Decision Tree — Quick Reference `[REVISED]`

```
Phase 0a (linear probe on existing failed B encoder) ← ~50% of probability mass kills here
├── chevron probe acc ≥ 0.65 AND > random-init + 0.10
│   └── proceed to Phase 0b
└── kill
    └── SHIP REGISTRATION PERMANENTLY. Write family-CLOSED memo.

Phase 0b (DSP synthetic two-tone)
├── log-freq + integer-bin shift produces pitch-equivariant registered patches
│   └── proceed to Phase 0c with log-freq locked
└── kill (structural pitch leak unavoidable)
    └── SHIP REGISTRATION PERMANENTLY.

Phase 0c (substrate generator)
├── all gates pass (ridge centered, multi-component preserved, drop rate < 15%)
│   └── proceed to Phase 1
├── multi-component preservation killed
│   └── Phase 1 runs at reduced expected return (no jump/sub-harmonic claim)
└── ridge centering / drop-rate kill
    └── debug Phase 0c; do not proceed

Phase 1 (contrastive on registered substrate, modest augmentation)
├── primary evals (NMI, chevron k-NN, linear probe) PASS
│   AND beats random-init + identity baselines by ≥ 0.10
│   ├── cohort η² ≤ 0.10
│   │   └── SHIP. Update production alphabet.
│   └── cohort η² > 0.10
│       └── proceed to Phase 2 (post-hoc residualization)
├── primary marginal AND fails to beat random-init/identity
│   └── eval is measuring substrate, not encoder. Document; possibly ship Phase 0c + KMeans.
└── primary kill
    └── SHIP REGISTRATION PERMANENTLY. Family-CLOSED memo. Modal prior ratified.

Phase 2 (post-hoc cage residualization, 30 min, no training)
├── cohort η² ≤ 0.05 AND shape preserved
│   └── ship Phase 1 + residualization as production
├── shape regresses > 0.15
│   └── ship Phase 1 unresidualized; document cage as open issue
└── cohort η² doesn't drop
    └── ship Phase 1; do NOT attempt adversarial training (DANN failed in 18.4, B+A failed at recon)
```

---

## What this v2 explicitly does NOT promise `[REVISED — strengthened]`

- It does **not** promise the learned latent will beat registration. Phase 1's primary eval target is the **substrate-independent** NMI/chevron-purity/linear-probe trio; the shape η² target is *secondary and interpreted with random-init/identity controls*.
- It does **not** promise the substrate hypothesis will pan out. Modal prior says it won't. The roadmap exists because Phase 0a is cheap enough that ruling out the bet at $0 is worth doing.
- It does **not** override the kill memo's binding conclusion. It tests whether **the kill memo's strong claim** (shape lives in 1-D) is true at the linear-probe and substrate-independent eval level. If yes, the family closes permanently; the cheap-test architecture *ratifies* the kill memo rather than challenging it.
- It does **not** include any adversarial / DANN training. Post-hoc residualization replaced v1's Phase 2 entirely.

## Eval validity controls `[REVISED — NEW SECTION]`

Per the adversarial review §2, all η²-based metrics on `register_one()`-derived labels are susceptible to substrate-eval circularity. The roadmap mitigates by:

1. **Substrate-independent labels primary:** `syllable_type` and the un-registered chevron-valley heuristic are sourced WITHOUT touching the per-column ridge that built Phase 0c's substrate.
2. **Random-init encoder baseline mandatory:** every shape η² report includes the random-init number; the learned model must beat it by ≥ 0.20 to be credited.
3. **Identity baseline mandatory:** per-column-average row → KMeans-20 is the "encoder is a forward of preprocessing" trivial control.
4. **Linear-probe transfer from frozen B encoder (Phase 0a) is the architecture-level falsifier:** Phase 1 must beat that probe by ≥ 0.10 to claim the new model added value over the existing failed one.

## Files to touch / NOT touch

**New:**
- `scripts/experiments/probe_shape_existing_encoder.py` (Phase 0a)
- `scripts/experiments/test_registration_synthetic.py` + `tests/test_registration_synthetic.py` (Phase 0b)
- `scripts/experiments/make_registered_patches_2d.py` (Phase 0c)
- `scripts/audit_registered_patches.py` (Phase 0c)
- `scripts/experiments/cage_residualize_latent.py` (Phase 2)
- `docs/modules/registered-patches-2d.md`

**Modified (small flag additions only):**
- `scripts/experiments/train_shape_encoder_contrastive.py` (`--registered` flag; preserve `--denoised` behavior)
- `scripts/eval_shape_encoder.py` (`--baselines random,identity` flag; primary-eval refactor)

**HANDS OFF (binding):**
- `models/shape_kmeans/k20.joblib` | `scripts/experiments/rig_R2_shape_alphabet.py`
- `src/usv_spectrogram/corpus.py` | `ExtractionConfig`
- `scripts/run_batch_detection.py` | `app/core/sliding_inference.py` | `postprocessing/`
- `train_contour_vae_v2.py` | `train_shape_vae_v3_hybrid.py` (frozen baselines)

## Compute budget `[REVISED — realistic]`

| Phase | Wall-clock | Compute | Cumulative |
|---|---|---|---|
| 0a | 30–60 min | Box CPU | 1 hour |
| 0b | 30 min – 1 hour | Box CPU | 2 hours |
| 0c | 2–4 hours + human audit | Rig CPU | 4–8 hours |
| 1 | 6–10 hours incl. 1 substrate retry; +rig contention with Pathway A | Rig GPU2/GPU3 | 10–18 hours active + 1–2 days end-to-end |
| 2 | 30 min | Box CPU | ~30 min after Phase 1 |
| **Total worst case** | **~2 days end-to-end** | | |

Expected wall-clock: if Phase 0a kills (~50%), roadmap ends at ~1 hour and registration ships. If Phase 0a passes, expect ~2 days for the full Phase 0b–2 run.

## Status as of writing (2026-05-28)

- v1 written → adversarially reviewed → v2 written (this file).
- All 11 required revisions from `docs/reviews/ROADMAP_SHAPE_INVARIANT_LATENT-adversarial.md` addressed (see `[REVISED]` markers).
- **PROPOSED** — awaiting user approval to begin Phase 0a.
- No code written. No rig compute consumed.

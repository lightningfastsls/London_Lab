# Adversarial review — ROADMAP_SHAPE_INVARIANT_LATENT v1

**Roadmap reviewed:** `docs/plans/ROADMAP_SHAPE_INVARIANT_LATENT.md` (v1, 2026-05-28, PROPOSED).
**Reviewer mandate:** red-team. Author prior on Phase 1 success ≈ 50%. My counter-prior is **15–20%**, with most of the variance falling on shape η² landing in the marginal 0.30–0.45 band (i.e. expensive ties), not the ≥0.58 ship band.
**Overall verdict:** **REVISE-BEFORE-START.** The roadmap has one structurally correct insight (substrate matters) wrapped in two design errors (a self-fulfilling eval and a Phase 2 that re-imports the B+A failure mechanism) and one missing cheap-first test. There is a real experiment in here, but not the one the v1 plan describes.

This document is harsh on purpose. It does not recommend killing the entire line of work, but it does recommend killing **Phase 2 as currently scoped** and gating Phase 1 behind a far cheaper precursor that costs ~30 min of CPU.

---

## TL;DR — the strongest attacks

1. **Phase 0's shape η² eval is self-fulfilling.** The "shape" ground truth in `desc_denoised.npz` is produced by `register_one()` from `rig_R2_shape_alphabet.py` (lines 47–64). Phase 0's substrate is constructed by shifting each time-column by the **same** per-column ridge `F(t)` that `register_one` computes. The encoder is being scored on its ability to recover what we preprocessed in. The "sanity η² ≥ 0.50" gate in Phase 0 is therefore trivially satisfied (or trivially broken — there's no middle outcome that means what the roadmap thinks it means).
2. **Phase 2 re-runs the B+A failure with extra noise.** The B+A kill memo (`docs/handoffs/2026-05-28_shape-vae-BA-hybrid-KILL.md` lines 38–54) shows that the moment `w_recon` anneals in at 0.0225, NT-Xent collapses from 1.48 to 3.5+ and stays there for the rest of training. The kill memo's exact words (line 68): *"no λ_recon > 0 will avoid the mechanism, only its magnitude. The bug is structural, not configurational."* Phase 2 adds back λ_recon ≤ 0.01 with the assertion that "registered substrate to remove the dominant pixel-variance the recon would otherwise latch onto" (roadmap line 162) saves it. That is a one-line hand-wave against a structural finding the author of the original memo (Claude) reached after observing the exact training curve.
3. **The stronger version of the kill memo's claim is not rebutted.** The kill memo (line 18) says *"shape lives in the 1-D registered ridge, not the 2-D pixel grid."* The roadmap's substrate hypothesis (line 36) only rebuts the weaker *"2-D pixel variance is dominated by pitch/duration"*. Per-column ridge subtraction produces a 2-D image whose **discriminative content is approximately the 1-D ridge plus near-padding zeros plus second-component residuals.** If those residuals carry shape information ≥ what the 1-D ridge already captures, registration would already be at 0.95+ rather than 0.75 — its ceiling already tells you the 2-D residual is mostly noise.
4. **The "remove pitch/time augmentation" decision in Phase 1 collapses NT-Xent's positive pair to noise-only.** When inputs are already canonicalized in pitch and time, the augmentations the roadmap keeps (small Gaussian noise, ±5% warp, pixel dropout) are dramatically weaker than what Pathway B used. NT-Xent with weak augmentation is known to collapse toward a mean encoding or memorize patch identity; either is fatal. The roadmap acknowledges no model of what the optimal augmentation magnitude on registered substrate should be.
5. **The cheap precursor that should run FIRST is not in the plan.** Take Pathway B's existing trained encoder (`/data/shachar/contour_vae/results/latent_transitions/b_contrastive/encoder.pt`). Encode registered patches (or even un-registered patches) and fit a **linear probe** on shape from the activations. If a linear probe on the already-trained-and-failed encoder doesn't clear shape η² 0.30, no amount of substrate-swap-then-retrain will. This is ~30 min of CPU. **It is irresponsible to spend 3+ rig-hours on Phase 1 without running this first.**

---

## §1 — Mechanism attack: does Phase 0's substrate change actually rebut the kill memo?

The kill memo's mechanism statement (`docs/handoffs/2026-05-28_pathway-B-kill-and-canonical.md` line 18) is stronger than the roadmap concedes. The kill memo distinguishes two claims:

- **Weak claim (roadmap addresses):** 2-D pixel variance is dominated by pitch/duration; remove those by construction and the encoder has nothing to be invariant to.
- **Strong claim (roadmap does NOT address):** Shape *lives in* the 1-D registered ridge. The 2-D image substrate, registered or not, does not carry strictly more shape-relevant information than the 1-D ridge once mean pitch is removed.

The strong claim is supported empirically: M8 (1-D VAE on registered ridge) hits 0.42 and M9 (1-D contrastive on registered ridge) hits 0.34. Registration → KMeans on the 1-D ridge hits 0.58–0.75. Every method that uses the 1-D registered ridge sits in a coherent 0.34–0.75 band; every method that uses 2-D imagery sits in a 0.009–0.105 band. The cleanest model that explains both bands is "the discriminative shape signal is 1-D, period."

Phase 0 does not refute the strong claim. It bets that **off-centerline 2-D structure** (sub-harmonics, jumps, second components, harmonic stacks) adds enough discriminative shape signal beyond the 1-D ridge to be worth training on. That bet has four problems:

### 1.1 — Per-column ridge subtraction does not produce harmonic-invariant patches

The roadmap's Q1 (line 88) flags this but waves it away. Concretely:

- **Linear shift** (default per Phase 0 step 2): a sub-harmonic at `2·F(t)` ends up at row `F(t)` in the registered patch. For a call with mean pitch 60 kHz, the sub-harmonic sits at row 60. For a call with mean pitch 90 kHz, the sub-harmonic sits at row 90. **Two calls with identical shape but different absolute pitch produce registered patches that differ by 30 rows of harmonic offset.** The encoder will exploit this — it is exactly the same pitch leak that killed Pathway B (pitch η² 0.306) and B+A (pitch η² 0.262).
- **Log shift** (the roadmap's "principled" alternative): a harmonic at `log(2·F(t)) = log(2) + log(F(t))` does sit at constant offset `log(2)` from the centerline. But (a) you now need a log-frequency resampling step before subtraction, which introduces its own interpolation artifacts that vary with absolute F; (b) the **noise floor** in dB is not log-uniform in linear frequency — log resampling will stretch low-frequency noise rows more than high-frequency rows, producing a structured artifact that varies with F; (c) the call body width in dB is not the same in log vs linear space — the encoder learns "wide-in-log" vs "narrow-in-log" as a shape feature, but this varies with F.
- **Padding fraction varies with F.** After per-column shift, the in-band region of the patch (rows where the original column had finite data) is bounded by `[b_lo - F_min, b_hi - F_max]`. For a low-pitch call this active region is a different shape than for a high-pitch call. The encoder learns the *shape of the padding region*, which is monotonic in F. Even with the `pad_mask` passed downstream, a 2-D conv encoder's pooled representation depends on where the padding is.

**Verdict on §1.1:** The substrate change does *not* produce truly pitch-invariant 2-D patches. It produces patches with the *centerline* pitch-invariant but a *padding shape* + *harmonic offset* monotonic in F. Phase 1's `pitch η² ≤ 0.05` gate (roadmap line 116) is the gate this fails first.

### 1.2 — "Sub-harmonics survive registration" is asserted, not measured

Phase 0's eval gate row 2 ("are sub-harmonics / jumps visible in registered patches? yes, qualitatively preserved") is qualitative — a visual audit of 50 patches. There is no quantitative measure of how much sub-harmonic *information* survived. A 50-patch contact sheet cannot distinguish "sub-harmonic visible" from "sub-harmonic visible but at a position that the encoder cannot exploit for shape clustering." Phase 0 should include a quantitative SHIM: for the subset of patches that the manual c07.2 labels mark as multi-component, does the post-registration patch's *off-centerline energy* correlate with the labeled component count? If not, sub-harmonics are formally preserved but informationally lost.

### 1.3 — Ridge-tracker-failure patches will produce garbage substrate

The c06 noise-sink cluster (memory note `project_c06_empty_cluster`) contains patches where `track_ridge()` produces all-NaN or near-all-NaN ridges. The roadmap's Q2 default ("leave NaN-ridge frames at zero-shift") means the substrate generator silently emits the raw patch for these. So:

- Phase 0 produces a mixed dataset: ~80% genuinely registered, ~20% raw (the noise/c06/failed-ridge patches).
- NT-Xent on a mixed substrate where some pairs are registered-vs-registered (small augmentation effect) and some are raw-vs-raw (full pitch/duration variance) is degenerate: the model can trivially distinguish the two regimes (active-mask coverage differs) and use that as a shortcut. The encoder learns "is this a real ridge?" as its top axis.
- The cleanest fix is to **drop** all failed-ridge patches from the registered substrate — but that subsets the training corpus by 20%+ and discards the very patches whose multi-component / unusual structure was supposedly the unique 2-D payoff.

### 1.4 — The 0.044 (B) → 0.105 (B+A) progression argues *against* the substrate hypothesis

The kill memo's comparison table (B+A KILL memo, head-to-head table line 22–28): pure contrastive on 2-D denoised got 0.044. Adding a VAE decoder + recon + KL + ridge-derivative term *raised* it to 0.105. **Adding reconstruction helped pure contrastive on raw 2-D imagery.** The roadmap's framing implies pure contrastive is structurally cleaner ("no recon = no shape-irrelevant pixel variance to chase"), but B's empirical result is worse than B+A's. If pure contrastive on raw imagery fails, what's the mechanistic argument that pure contrastive on *registered* imagery doesn't also fail — given that registration removes the very pixel-variance the contrastive negatives could push apart on?

Steel-manned: the roadmap *could* argue that B's negatives separated on pitch (the easy cue), so removing pitch frees the encoder to find shape negatives. But B's pitch η² ended at 0.306 — it *did* sort negatives by pitch. The argument requires that with pitch removed by construction, the encoder will find a strong-enough secondary signal to drive negatives apart. There is no evidence in the corpus for what that secondary signal is. If it's shape, registration would already be at 0.95+.

---

## §2 — Eval validity: shape η² on registered substrate is self-fulfilling

This is the single largest design flaw in the roadmap.

### 2.1 — The eval label and the substrate share a common cause

`scripts/eval_shape_encoder.py` (line 87) sources its ground-truth shape labels from `results/eval_shape/desc_denoised.npz`. Tracing `desc_denoised.npz`'s origin: it's produced by running `register_one()` (`rig_R2_shape_alphabet.py` lines 47–64, also reproduced in `eval_shape_kill_gate_v3.py` line 77). `register_one` calls `track_ridge()` per patch, subtracts the active-frame mean pitch, resamples to 50 points, and returns the 50-D shape vector.

Phase 0's substrate generator calls `track_ridge()` per patch (Phase 0 step 2: "shift the column by `-F(t)` rows so the ridge ends up on row R/2"). It reuses the *exact same ridge* — the spec even says "consume the existing denoised patches + per-patch ridge `F(t)` (already cached in `ridge_targets_v3.npz` from Pathway A)" (roadmap line 47).

So the eval's "shape" target is `register_one(F(t))` and Phase 1's substrate is also a function of `F(t)`. The encoder is being scored on whether its embedding is a useful function of the same `F(t)` that generated the labels.

### 2.2 — Why this is self-fulfilling

Suppose Phase 1's encoder is the **identity map on the per-column-average row of the registered patch**. That's a trivial encoder — it doesn't learn anything; it just averages each time column of the input. By construction, the per-column average of `P[t, f - F(t)]` is the original row `P[t, f]` averaged after the F(t) shift. With high probability this is approximately the registered ridge shape vector that `register_one` produces. KMeans(20) on this trivial encoder's output will hit shape η² very close to 0.58–0.75 — *not because the encoder learned shape* but because the substrate already canonicalized to shape and the eval label is shape derived from the same canonicalization.

The roadmap's Phase 0 "sanity η² check" (line 69, "redo registration-→-KMeans shape clustering using a per-time-column-average of the 2-D registered patch as the 1-D ridge surrogate. Should approximately match the existing 0.58–0.75") explicitly anticipates this — but treats it as a *passing* sanity check rather than recognizing it as a fundamental eval-validity problem.

### 2.3 — What's the right control?

The roadmap has **no negative control** that distinguishes "encoder learned shape" from "encoder forwarded the preprocessing". Required controls:

- **Random-init encoder baseline.** Encode all patches with an *untrained* `ContrastiveEncoder` (random weights). Compute shape η². If random-init also clears 0.30, the eval is measuring substrate quality, not encoder quality.
- **Identity baseline.** Treat each registered patch's per-column-average row as a 50-D vector. KMeans(20). Shape η². If this clears 0.50, again, the eval is measuring substrate canonicalization.
- **Pixel-shuffled substrate.** Train Phase 1 on registered patches where rows are randomly permuted within each column. If shape η² is similar, the encoder isn't using row-position structure — meaning the registered substrate's shape signal is the *column average*, and a learned encoder adds nothing.

Without these baselines, a "shape η² = 0.60" result for Phase 1 means nothing — it could be the encoder doing real work or it could be substrate leakage.

### 2.4 — A better shape eval

If you want to score the encoder, not the preprocessing, use a **held-out manual label** (the existing chevron/jump/flat/up-FM/down-FM hand-labels referenced in `PLAN_geometric_shape_clustering_vae.md` §4 — though note that script flags `syllable_type` may not exist; the chevron/valley heuristic is the fallback). The eval should be: cluster on latent → measure NMI against hand-labels. Shape η² *on the registered ridge* is contaminated by the substrate.

---

## §3 — Phase 0 DSP / preprocessing risks

### 3.1 — Per-column interpolation artifacts

Phase 0 step 2 specifies "linear interpolation; pad with zeros." Linear interpolation of a column where the call is concentrated in 2–3 bins (typical for a ~3 kHz wide tonal USV ridge with the corpus 1.46 kHz/bin resolution) introduces a 2-bin-wide blur band that varies with `dy = -F(t)`. If `F(t)` is an integer number of bins, no interp. If it's mid-bin, you get blur. Across time columns, this produces a **fractional-bin shift pattern monotonic in dF/dt**. The encoder learns dF/dt-of-the-rounding-error as a feature. This is exactly the kind of subtle DSP artifact that disproportionately leaks pitch-velocity into the latent (pitch leakage), and the substrate audit (visual contact sheet) will not catch it because the human eye doesn't see fractional-bin shifts.

**Required fix:** shift by integer bins only (round `F(t)` to nearest bin index) AND verify on a synthetic two-tone test that the registered output has no dF/dt-dependent interpolation pattern. Better yet, do the registration in **log-frequency space** with integer log-bin shifts (relates to Q1) but only after demonstrating on synthetic data that this isn't worse.

### 3.2 — Discontinuities at silent-frame boundaries

The roadmap (Q2) defaults silent frames to zero-shift. So the registered patch has a band of registered-shifted content (where ridge was active), bordered by an unshifted band (where ridge was NaN), with a sharp discontinuity at the boundary. The discontinuity's vertical position is `F(t_onset)` — i.e. **the discontinuity position encodes the onset pitch**. The encoder learns this as a feature.

Mitigations the roadmap should specify:
- Smooth-extrapolate `F(t)` into adjacent silent frames (NaN-fill from edge) before shifting.
- OR: shift the entire patch by the *active-region average* `F̄` and then within the active region apply the residual shift `F(t) - F̄`. The second shift is smaller and the discontinuity at the silent-frame boundary is smaller.

Neither is in the roadmap. The first variant chosen will materially affect Phase 1's pitch η².

### 3.3 — Edge effects at active-region boundaries

Phase 0 step 3 crops to active span then resamples. Calls with a slow onset / decay have ridge-tracker-detected onset that depends on `silence_threshold` (`RidgeConfig` default 0.02 × max). Two calls with identical shape but different SNR will have different detected active spans. After resampling to 96 frames, the *normalized rate of change* dF/dτ (τ=fraction of normalized duration) is different — encoding SNR into the shape representation. This is the same mechanism that puts SNR (a cage proxy) into the latent. Phase 1's cohort η² gate (≤ 0.10) is what catches this.

### 3.4 — Failed-ridge patches

(Already discussed in §1.3.) Phase 0's Q2 must be resolved to "drop NaN-ridge patches from substrate" or the substrate is silently bimodal. Either choice has cost; the choice must be explicit.

---

## §4 — "Remove pitch/time augmentation" decision

Roadmap line 103–104: "since pitch/timing are canonicalized in the input, the pitch/time-shift augmentations become identity transformations and should be REMOVED to avoid double-canonicalization."

This is wrong in two ways.

### 4.1 — The canonicalization is not perfect, so the augmentations are not identities

§3.1–§3.3 above show that Phase 0's registered substrate has residual pitch leak (interpolation artifacts, silent-frame discontinuities, harmonic offsets, padding-shape variation). A pitch-shift augmentation on the registered substrate is *not* an identity — it is a perturbation that mostly cancels the encoder's residual pitch sensitivity. *Removing the augmentation removes the regularization that fights the residual leak.* This is the wrong direction.

The author appears to assume that if input is canonicalized, augmentation is redundant. But the encoder still has residual freedom to use the leaked-pitch features as a discriminator. The augmentation is what tells the encoder "these residual differences are not signal." Removing it removes the teaching signal.

### 4.2 — NT-Xent with weak positive-pair perturbation collapses or memorizes

The roadmap keeps only: small Gaussian noise, tiny time-warp 0.95–1.05× (since duration is already normalized to 96 frames, a 5% warp is 1–5 frames), pixel-level dropout. These are *very* weak perturbations on a 96 × 256 patch. NT-Xent with weak perturbations has two well-known failure modes:

- **Representation collapse**: positives become trivially close (small perturbation = near-identical embedding) so the loss can be minimized by mapping every input to a constant. Loss curve becomes "low and uninformative."
- **Patch memorization**: encoder learns a near-identity hash from input pixels to embedding; clustering on memorized hashes gives random partitions modulo pixel-level noise. Shape η² near chance.

A reasonable Phase 1 augmentation budget on registered substrate, if you keep the substrate, is *more* augmentation on shape-preserving axes that the substrate did NOT canonicalize: harmonic-amplitude perturbation, additive noise that varies with frequency (not just spectrally flat Gaussian), per-column phase jitter. None of these are in the roadmap.

### 4.3 — Concrete failure mode prediction

I predict: Phase 1 as specified produces a learned encoder with **shape η² in the 0.25–0.40 marginal band** primarily because the encoder is partly memorizing the substrate's structure (which the eval already encodes as shape labels) rather than learning shape. The gate `pitch η² ≤ 0.05` is what fails first, in the 0.10–0.25 range — *not* because the encoder is "sorting by pitch" per se, but because the substrate's residual pitch leak is recovered by the encoder. The roadmap's contingency (line 128: "registration in Phase 0 is leaking pitch... debug Phase 0") is then triggered, but the leak is structural to per-column shifting — debugging Phase 0 produces a different substrate which then needs Phase 1 re-trained. This is the failure cascade Phase 0's open Q5 asks about and the roadmap has no cheap detection for.

---

## §5 — Phase 2: re-runs the B+A failure with extra moving parts

### 5.1 — B+A's kill memo is unambiguous about the recon mechanism

From `docs/handoffs/2026-05-28_shape-vae-BA-hybrid-KILL.md` lines 49–54:

> Two clean regimes, separated exactly by the anneal:
>
> - **ep 0-9 (pure contrastive, w_recon=0):** `nt` dropped 2.91 → 1.48, `lc` dropped 2.28 → 0.65. The encoder *was* learning a shift-invariant shape representation.
> - **ep 10+ (recon/KL/deriv anneal in):** `nt` jumped 1.48 → 3.5+ and **stayed there**; `lc` exploded 0.65 → 200+ and **stayed there**. The reconstruction term overrode the contrastive structure as soon as it was allowed to contribute.

And line 68:

> Pre-anneal (w_recon=0) the encoder *did* learn well (nt 1.48, lc 0.65 at ep9). The collapse is caused by activating the reconstruction term *at all*, not by the weight value — no λ_recon > 0 will avoid the mechanism, only its magnitude. The bug is structural, not configurational.

Phase 2 adds back `λ_recon ≤ 0.01` recon + β·KL + adversary heads, with the assertion that registered substrate prevents the collapse (roadmap line 162). This is the v1 author asserting that the same Claude who wrote "no λ_recon > 0 will avoid the mechanism, only its magnitude" is wrong because the substrate is different. The author owes a mechanistic argument, not an assertion.

### 5.2 — The mechanism-of-collapse argument was substrate-independent

The B+A kill memo's mechanism is: "BCE+pixel-MSE recon spends latent capacity on what it *can* see — vertical position and horizontal extent." On the registered substrate, pixel position is canonicalized — but the recon term still **reconstructs pixels**. A registered call's reconstruction is dominated by:

- The padded region (large fraction; the recon loss can be reduced by predicting "0 everywhere" for padding, getting low loss). If the pad-mask correctly excludes padding from the recon loss (which the roadmap specifies — "reconstruction masked by `pad_mask`"), this is mitigated. But the masked area still dominates: a registered patch is 96 × 256 = 24,576 pixels; the active region is roughly 96 × ~50 (the ridge band width × time length) ≈ 4,800 pixels; the recon loss is computed over ~20% of the patch.
- Within the active region, pixel intensity is dominated by **energy** (mean spectral power), which varies with cage (recording level, mic position, SNR). The recon term learns to encode cage-level energy in z. So Phase 2's adversary head `D_cage(z) → cage_id` has to push against a recon term that's actively pulling cage info into z. Adversarial games with the recon as the adversary's adversary are exactly the regime that *causes* training to be unstable.

Phase 2 risk is structurally worse than B+A: B+A had recon + NT-Xent racing. Phase 2 has recon + NT-Xent + 3 adversary heads + FiLM conditioning. The author should expect:

- Adversary head dominates → encoder produces a degenerate z that's uninformative about cage but also uninformative about shape (shape η² regresses).
- Recon dominates → cage leaks back, adversary head is at chance, shape η² may hold but cohort η² doesn't drop. The roadmap calls this case "ship Phase 1" but it took 4 hours of GPU to discover that.
- Both fight to a draw → NaN or chaotic training curve, the standard DANN failure mode.

### 5.3 — DANN already failed in this repo

Memory note `project_lab_cnn_classifier_scope` and `project_lab_transfer_v1_vs_dann_patchsweep` document the DANN failure in module 18.4: balanced acc dropped from 0.81 (v1) to 0.58, noise_recall from 0.64 to 0.16. The roadmap (line 163) acknowledges this and says "mechanism was different (domain-shift collapse in a supervised classifier)." That's a correct disclaimer but doesn't constitute a model of why this DANN will succeed. The convergence story is "monitor adversary accuracy; if it stays near chance for >10 epochs, the gradient-reversal isn't biting." That's a *diagnostic*, not a recipe. What's the prior probability the gradient-reversal converges in this setup? The roadmap doesn't say. From the 18.4 evidence, it should be ≤ 25%.

### 5.4 — Phase 2 should be cut

My recommendation: **delete Phase 2 from v1 entirely.** Replace with a non-adversarial cage-scrubbing approach: fit Phase 1's latent, then post-hoc residualize against `cage_id` via partial-regression (Frisch–Waugh–Lovell) at inference. This gets you the cage-clean latent without any of the recon/adversary games. If Phase 1 ships, this is a 30-min post-hoc step. If the residualization destroys shape η², you've learned something and you can revisit DANN with eyes open. Either way, Phase 2 as written is 4 GPU-hours buying ~25% probability of a marginal improvement — bad expected value.

---

## §6 — Cage confound location

The roadmap assumes cage information lives in things Phase 0 doesn't touch (noise floor, mic response, room reverb). Per memory note `feedback_rig_artifact_mean_power_db`, mean spectral power and tonality are cage artifacts. But the cleaning pipeline (`prefilter_spectrogram` per memory `project_cleaning_pipeline_inventory`) already removes:

- Per-recording baseline subtraction (`--subtract-baseline`)
- Global MAD normalization
- (Dormant) per-recording Z-norm

So the cage signal in the *denoised* patches is partly scrubbed already. The residual cage signal that survives `prefilter_spectrogram` is mostly **frequency-dependent SNR** (some cages have more noise at 30–40 kHz from fans, others at 80–100 kHz from mic resonance), not absolute energy. This shows up as the *texture* of the off-ridge band in the patch.

Phase 0's registration moves the ridge to the centerline but keeps the off-ridge band texture intact. So the residual cage signal *does* survive into Phase 1's substrate. But the adversary in Phase 2 is supposed to push this out of z — the question is whether the residual cage signal is even strong enough for the adversary to find. From the B kill memo's cohort η² ~0.10–0.15 range, it's borderline detectable. Adversarial training on a borderline signal is a known instability regime: the adversary head's gradient is noisy and the encoder's response is correspondingly noisy. The expected outcome is "adversary at chance, no biting, ship Phase 1." This is the most-probable Phase 2 outcome and the roadmap's contingency for it (line 179) is correct — but the roadmap should *pre-commit* to that outcome being the modal expectation, and budget compute accordingly (i.e., budget zero for Phase 2 as a "let's try DANN" exploratory run).

---

## §7 — Opportunity cost / wall-clock

The roadmap estimates 3 hours total for Phase 0+1 (1 hr CPU + 2 hr GPU) and 4 hours for Phase 2. Realistic estimates:

- **Phase 0:** 1 hr CPU is plausible for the substrate generation. But the visual audit (50-patch contact sheet) is a human-in-the-loop checkpoint — call it a half day of attention to actually look. The "sanity η² ≥ 0.50" check requires reading the new substrate back into memory and running KMeans, which on 70k × 96 × 256 is non-trivial RAM (call it another 30 min + risk of OOM on box; per memory `project_c06_empty_cluster`, "11 GiB box: never full-scan patches.npz"). Realistic: 2–4 hours wall + half a day for audit decisions.
- **Phase 1:** 2 hr GPU is *training* time; the encode+eval cycle is another 30–60 min; if the first run's pitch η² is high (likely per §1.1), at least one Phase 0 re-spin is needed (back to step 1, +2 hr), then re-train (+2 hr). Realistic: 6–10 hours wall, including 1 retry.
- **Phase 2:** 4 hr GPU is optimistic for adversarial training. DANN training notoriously needs hyperparam search on `λ_adv` warmup (try 3 values × 4 hr = 12 hr); per memory `project_lab_cnn_classifier_scope`, the 18.4 DANN failed silently for ~2 days of debugging before being abandoned. Realistic: 1–2 days wall.

**Rig contention.** The memory note `project_cnn_iteration_eval_5970` and `feedback_orchestrator_mode` mention the rig has GPU0 used by user's contour-VAE work and 17 GB cache shared with Pathway A. The shape encoder module doc (`docs/modules/shape-encoder-contrastive.md` line 47: "the 16 GB patch corpus + a concurrent 17 GB Pathway-A job exceed the rig's 31 GB → run B alone (wait for A) or it gets OOM-killed.") This applies directly: Phase 1's registered patches need ~10 GB, training needs ~6 GB GPU memory + RAM for batches. Coordinating with Pathway A traffic adds 1–2 days of dead waiting time.

**Compared to:** post-hoc residualization of Phase 1's latent against cage_id is ~30 min. Linear probe on the existing failed Pathway B encoder is ~30 min CPU. The pre-commit cheap-test gates have a 50–100× compute advantage over the proposed roadmap.

---

## §8 — Selection bias in the kill memos

The roadmap's framing of the kill memos is: "6 image-VAE attempts failed, but they all shared the un-registered-substrate failure mode; the substrate is the unifying explanation." This is a *post-hoc unification* — the 6 attempts varied in many ways (recon weight, KL weight, derivative term, contrastive vs reconstructive, with and without masking, with and without augmentation magnitude). Calling "substrate" the common cause is not a falsifiable claim until you've controlled for the other variables.

The cleanest counter-evidence is the B (0.044) vs B+A (0.105) comparison. Pure contrastive (B) is structurally the simplest: no recon, no KL, no decoder. Yet it scored *worse* than B+A's hybrid (B+A had recon + KL + derivative + contrastive + latent-consistency). The roadmap's framing predicts B should outperform B+A because B has no recon term competing with shape. The data shows the opposite. This suggests the failure mechanism is *not* "recon term steals capacity" but something else — possibly that the encoder needs the recon term to anchor *some* signal and pure contrastive collapses faster than hybrid. If that's true, then Phase 1 (a re-run of B on substrate-swapped input, with even *less* augmentation than B used) is structurally vulnerable to the same collapse — registration substrate doesn't help with the collapse mechanism.

There's a more parsimonious explanation for the 6/6 failures: **the corpus genuinely doesn't contain 2-D shape signal that's richer than the 1-D ridge.** This isn't a failure of the encoder, the loss, or the substrate — it's a fact about the data. Mickey's USVs are predominantly tonal calls where the ridge captures essentially all of the shape variation. The 2-D residual after registration is noise + cage texture + rare multi-component events that are too few to drive clustering. If this is the world, no Phase 0/1/2 variant ships. The roadmap should acknowledge this as the modal hypothesis (priorprob ≥ 50%).

---

## §9 — What's missing from the roadmap

1. **Cheap precursor test** (§ TL;DR #5). Take `encoder.pt` from `/data/shachar/contour_vae/results/latent_transitions/b_contrastive/`. Load the registered shapes from `true_registered_ridges.npz`. Fit a logistic linear probe from encoder embedding → chevron/valley/other label. If the linear probe doesn't clear 0.50 accuracy, the encoder doesn't have shape-relevant features in its representation, period — and a re-train on a *similar* encoder on a *similar* substrate isn't going to. Cost: ~30 min CPU. Information value: replaces the entire Phase 0+1 bet with a yes/no.
2. **Random-init negative control** for the eval (§2.3). Without this, "shape η² = X" is uninterpretable.
3. **Quantitative sub-harmonic retention measure** for Phase 0 (§1.2). Visual audit of 50 patches is not adequate evidence.
4. **DSP-level audit of per-column shift artifacts** on synthetic two-tone test (§3.1).
5. **Pre-commit budget on Phase 2 cancellation.** If Phase 1 ships at all, *do not* run Phase 2 as written. Use post-hoc residualization or live with the cage confound.
6. **Acknowledgment that registration may genuinely be ceiling.** The most likely outcome is that shape lives in 1-D and there is no 2-D shape signal to recover. The roadmap's "what this explicitly does NOT promise" section (line 226) gestures at this but doesn't commit to it as the *prior* expectation.

---

## §10 — Verdict

**REVISE-BEFORE-START.** Do not begin Phase 0 implementation on v1.

The roadmap contains one real insight (substrate matters, registration as preprocessing has been proven by 1-D-on-registered results to ~0.42–0.75) and two structural design errors (self-fulfilling eval, Phase 2 = B+A redux). The fixes are tractable but require a v2 that addresses them upfront rather than via debugging mid-Phase-1.

**The single highest-value change:** require the cheap linear-probe-on-existing-B-encoder test as Phase 0a before any new training. If the existing B encoder's representations have ≥ 0.50 linear probe accuracy for chevron/valley, the substrate hypothesis is plausible and Phase 0+1 are worth running. If not, the bet is dead at the encoder-can-find-shape-given-good-substrate level and registration ships permanently.

**Probability assessment** for the roadmap as written, by my counts:
- Phase 1 ships at shape η² ≥ 0.58: **15%**.
- Phase 1 lands marginal (0.30–0.58) AND captures multi-component / jumps that registration misses: **15%**.
- Phase 1 lands marginal but adds no qualitative payoff: **30%**.
- Phase 1 falls below 0.30 kill gate: **35%**.
- Phase 1 fails because Phase 0 has substrate bugs that take ≥ 2 spins to find: **40%** (overlaps with the above).
- Phase 2 ships (cohort η² ≤ 0.10 AND shape preserved): **10%** of the conditional-on-Phase-1-ship paths, i.e. **~3%** unconditional.

Net: ~30% probability of a worthwhile shipping artifact at ~10–20 hours wall-clock, vs. the **canonical action which is to ship registration permanently and close the family** with the cheap linear-probe + post-hoc cage residualization as a 1-hour finalization.

If forced to advocate, I would advocate **shipping registration as permanent, running the linear probe and post-hoc residualization as a 1-hour wrap, and writing a "no further VAE-family work on shape clustering" decision note.** That action dominates the v1 roadmap on expected return.

If the v2 author insists on running Phase 0+1, the linear probe must be a hard precursor and Phase 2 must be cut. With those two changes, the experiment is a defensible ~6-hour bet rather than an indefensible ~3-day bet.

---

## REQUIRED revisions before Phase 0 can start

- [ ] **Add Phase 0a (precursor):** linear probe on the existing failed Pathway B encoder against (a) chevron/valley heuristic labels (`scripts/eval_shape_encoder.py::chevron_valley`), (b) any available manual labels. **Gate:** probe accuracy must clear 0.50 on chevron-vs-other for Phase 0 to proceed. If it doesn't, ship registration permanently.
- [ ] **Replace the shape η² eval with at least one substrate-independent metric.** Specifically: NMI between learned-latent clusters and hand-labeled types (chevron/jump/flat/up-FM/down-FM/complex if `syllable_type` exists, else chevron/valley heuristic on the *un-registered* ridge as an outside-information check). The eval must score the encoder, not the preprocessing.
- [ ] **Add random-init encoder baseline** to Phase 1 eval. Compute shape η² for a random-init `ContrastiveEncoder`. If random-init clears 0.30, the eval is measuring substrate quality, not encoder quality, and must be redesigned.
- [ ] **Add identity baseline** to Phase 1 eval. Per-column-average of registered patch → KMeans(20) → shape η². This is the trivial-encoder baseline; the learned model must beat it by ≥ 0.10.
- [ ] **Resolve Q1, Q2, Q3 with explicit choices and named risks**, not "default proposal." Specifically Q2 must commit to either "drop failed-ridge patches" or "smooth-extrapolate F(t) into silent frames" — zero-shift on silents is the discontinuity-bug recipe of §3.2.
- [ ] **DSP synthetic test for §3.1.** Generate 2 synthetic patches with identical chevron shape but different absolute pitch (60 kHz centerline vs 90 kHz centerline). Run Phase 0 substrate generation. Verify the registered patches are pixel-equivalent modulo translation. If they aren't, fix the shift code before generating any real-data substrate.
- [ ] **Cut Phase 2 from v1.** Replace with: "after Phase 1 ships, run post-hoc partial-regression residualization of latent against `cage_id`. If cohort η² doesn't drop below 0.10 after residualization, document cage as an unresolved open issue and do not attempt adversarial scrubbing."
- [ ] **Quantitative sub-harmonic preservation gate for Phase 0.** Replace the qualitative "visible in registered patches" check with: fraction of off-centerline energy variance on multi-component-labeled patches ≥ 0.5 × the same fraction in the unregistered patches. (Definitive measure depends on availability of multi-component labels; if not available, the sub-harmonic preservation claim should be downgraded to "this experiment does not test the multi-component payoff.")
- [ ] **Augmentation budget on registered substrate must be specified with a mechanism**, not "remove pitch/time aug because canonicalized." Either (a) keep modest pitch/time aug to penalize encoder's recovery of leaked pitch, (b) add new augmentations on shape-preserving axes (per-column amplitude perturbation, frequency-dependent noise) and justify them. v1's reduction to "Gaussian noise + 5% warp + dropout" is too weak to drive NT-Xent and will likely produce a collapsed representation.
- [ ] **Realistic compute budget.** Phase 0: 1 hr CPU + 0.5 day human audit. Phase 1: 6–10 hr wall (assume 1 substrate retry). With the linear-probe precursor potentially short-circuiting the whole thing, total expected wall-clock is 1–2 days, not the v1 claim of 3 hours.
- [ ] **Explicit prior on the 1-D-shape-is-ceiling hypothesis.** State up front: P(shape genuinely lives in 1-D and 2-D substrate adds nothing) ≥ 0.5. Phase 1 results that fail to beat the M9 0.34 baseline ratify this prior and close the family.

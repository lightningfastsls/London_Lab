# Handoff: By-construction time/frequency-invariant shape representations for USV contours

**Purpose.** Implement and benchmark four representations that enforce invariance to time-position/duration/warp and (where chosen) absolute frequency *by construction*, then test each against the incumbent soft-DTW baseline on the existing kNN shape-label benchmark. Goal is truth-finding: we want to learn *where each invariance helps and where it hurts*, not to crown a single winner.

This covers Methods 2–5 from the design discussion:
- **M2** — Wavelet scattering (deformation-stable, translation-invariant front-end)
- **M3** — Sublevel-set persistent homology of the contour
- **M4** — Within-call recurrence matrix + RQA
- **M5** — Classical curve signatures (turning function, curvature scale space) — simplest, used to validate the harness

---

## Prereqs / assumed repo state

Assume these exist from prior work; if any are missing, build them in Phase 0 before anything else:
- A loader returning, per call: the **registered 50-pt contour** (mean-pitch subtracted, active span resampled), a **call id**, **cohort/cage id**, and where available the **human shape label** (~12-class ethological taxonomy) on the labeled subset.
- Access to the **higher-resolution contour** (active span *before* the 50-pt downsample) and the **per-call narrowband spectrogram / audio segment** — M2 needs more temporal resolution than 50 points, and M2b needs the spectrogram.
- The **kNN retrieval-purity harness**: leave-one-out, purity against human labels, bootstrap CIs. Baselines already include **Euclidean on registered curves** and **soft-DTW** (the bar to beat).

Total corpus ~67k calls across 4 recording environments (cage = strong nuisance axis). The labeled subset for purity is smaller; sweep parameters on a ~5k stratified sample, run final numbers on the full labeled set.

---

## Common interface (make all four methods interchangeable)

Every method implements one of two contracts; the harness accepts either:

```python
# Preferred: per-call feature vector -> (N, d) matrix, then ANN kNN
def encode(contour, *, aux=None) -> np.ndarray   # shape (d,)

# Fallback for distance-native methods (soft-DTW baseline):
def pairwise(contour_i, contour_j) -> float
```

M2/M3/M4/M5 are all feature-vector methods, so they produce an `(N, d)` matrix. Do **not** materialize a 67k×67k matrix anywhere. For retrieval, use approximate NN (`pynndescent` or `faiss`) on the feature matrix with both Euclidean and cosine; report whichever is better per method.

Store outputs as `features/{method}__{paramhash}.npy` plus a small JSON of the params, so sweeps are reproducible and the synthesis step can diff them.

---

## Cross-cutting design rules (these are the traps — enforce them as code, not comments)

1. **Never grant time-reversal invariance.** An up-sweep and its mirror are biologically distinct. M3 (persistence) and M5 (turning function) are reversal-blind by default. For each method add a **reversal unit test**: `dist(x, reverse(x))` must be large (top-decile of the pairwise distribution). If a method fails it, append an explicit direction feature (signed net slope, or the antisymmetric part of the turning function) and re-test.

2. **Keep duration and modulation-depth as separate scalar side-channels.** Every invariant representation here also discards absolute duration and (optionally) frequency excursion — both are behaviorally live (e.g., duration is context-modulated). Store `duration_ms` and `freq_range`/`freq_std` per call in a side-channel array. Run the purity benchmark **twice**: invariant vector alone, and invariant vector ⊕ z-scored side-channels. Report both.

3. **Stratify by cage.** Run purity **within-cohort** and **pooled**, report both. If a method only "wins" pooled, suspect cage leakage, not shape.

4. **Decide frequency-scale invariance explicitly per method.** Mean subtraction already removes additive pitch. Whether to *also* divide out excursion magnitude (full frequency invariance) is a flag, `scale_invariant: bool`, defaulting to **False** (keep depth as signal). Test both settings for M4/M5 where it's cheap.

---

## M5 — Classical curve signatures (DO THIS FIRST)

Simplest method; use it to shake out the harness, the reversal test, and the side-channel logic.

- **Input.** Registered 50-pt contour (also try 128-pt).
- **Library.** Pure numpy/scipy; no heavy dep.
- **Pipeline.**
  - *Turning function:* arc-length parameterize the contour `(t, f(t))`, compute cumulative tangent angle vs normalized arc length; sample to fixed length. Distance = L2 after arc-length normalization. Translation/scale-invariant by construction.
  - *Curvature scale space (CSS):* Gaussian-smooth the contour at a geometric ladder of scales, take curvature zero-crossings; feature = CSS maxima map. Use as a secondary descriptor.
- **Params to sweep.** Resample length {50,128}; CSS scale ladder; `scale_invariant {False,True}`.
- **Output.** Fixed-length turning-function vector (+ optional CSS feature).
- **Traps.** Reversal-blind and start-point sensitive — apply rule (1); pin the start point to the registered onset.
- **Scale.** Trivial.

**Prediction to log:** if turning-function purity ≈ elastic on *clean* calls but loses on jump/step, that confirms shape itself is low-dimensional and warping is the only hard part — i.e., M5 isolates "warp alignment" as the entire value of soft-DTW.

---

## M4 — Within-call recurrence matrix + RQA

- **Input.** Registered contour (50-pt fine; also try 128-pt).
- **Library.** `pyrqa` (GPU) or `pyunicorn` for RQA scalars; numpy for the raw matrix.
- **Pipeline.**
  - Self-distance matrix `R[i,j] = |f(t_i) - f(t_j)|` (additive-pitch invariant automatically). Optionally a delay-embedded recurrence for higher-order structure.
  - Threshold at fixed **recurrence rate** (not fixed ε) for cross-call comparability; sweep the rate.
  - Feature = RQA scalars (DET, LAM, L_mean, TT, entropy) → small vector. Optionally add a coarse pooled/downsampled `R` for a richer vector.
- **Params to sweep.** Recurrence rate; embedding dim/delay (start un-embedded); pooling size.
- **Output.** RQA scalar vector (≈6–10 d), optional + pooled-matrix block.
- **Traps.** Recurrence matrix is reversal-*invariant* → apply rule (1). Fixed-rate thresholding is essential for comparing calls of different shape complexity.
- **Scale.** Per-call, embarrassingly parallel; cheap.

**Prediction to log:** should track soft-DTW's wins on relational structure at a fraction of the cost; if it matches soft-DTW on complex/flat/chevron, it's a cheap stand-in for the elastic distance.

---

## M3 — Sublevel-set persistent homology of the contour

- **Input.** Registered contour as a 1-D array (50-pt; also 128-pt). Pure shape; reversal/direction handled below.
- **Library.** `giotto-tda` (`CubicalPersistence` on the 1-D array gives sublevel-set persistence; built-in `PersistenceImage` / `PersistenceLandscape` vectorizers). `ripser`/`homcloud` as alternates.
- **Pipeline.**
  - Compute **sublevel-set** persistence of `f` (pairs minima→maxima) **and superlevel** (run on `-f`) so both valley and peak structure are captured.
  - Work in **(birth, lifetime)** coordinates (lifetime = death − birth) to be invariant to additive pitch; this also makes prominences the salient quantity.
  - Vectorize via persistence image (fixed grid) → fixed-length feature.
- **Params to sweep.** Resolution {50,128}; persistence-image bandwidth & grid; sublevel-only vs sublevel⊕superlevel.
- **Output.** Concatenated persistence-image vector.
- **Traps.** Reversal- and order-blind by construction → **expected to conflate up- vs down-ramps**; this is a *diagnostic*, not a bug. Apply rule (1) (append signed net slope) only for the head-to-head purity run; keep a pure-persistence variant to characterize what topology alone captures.
- **Scale.** Per-call; cheap. Cubical persistence on a 50–128 length array is microseconds.

**Prediction to log:** strong on peak/valley/jump-*count* families (chevron, valley, multi-jump), weak on sweep-direction families (flat-up vs flat-down). If adding the slope side-channel closes that specific gap, we've cleanly separated "configuration of extrema" from "direction" as two orthogonal shape factors.

---

## M2 — Wavelet scattering

Two variants. **2a is the faithful primary** (shape lives in the 1-D contour); **2b is the diagnostic** that tests whether the seven prior VAE failures were about *pixels-as-learned-objective* or about spectrograms per se.

### M2a — Scattering1D on the contour (primary)
- **Input.** **Higher-resolution** contour (active span before 50-pt downsample, or resampled to 256). Scattering needs more samples than 50.
- **Library.** `kymatio` (`Scattering1D`), GPU.
- **Pipeline.** `Scattering1D(J, Q, T)` on the pitch-normalized contour → scattering coefficients → flatten (drop order-0 if it just re-encodes mean). Deformation-stable and time-translation-invariant by construction.
- **Params to sweep.** `J` (max scale), `Q` (wavelets/octave), `T` (invariance scale — larger T = more translation invariance, less detail; sweep T relative to contour length). Try with/without order-0.
- **Output.** Scattering feature vector (moderate d; PCA to ~30–50 before kNN).
- **Traps.** Short signals → keep J modest. Pitch must be normalized upstream (it is). Time-translation-invariant but not reversal-invariant — should pass rule (1) natively; verify.
- **Scale.** GPU-batched over 67k; fast.

### M2b — Joint time–frequency scattering on the spectrogram (diagnostic)
- **Input.** Per-call narrowband **spectrogram** (needs audio/spectrogram access, not just the contour).
- **Library.** `kymatio` `TimeFrequencyScattering1D` (joint TF scattering) — gives **frequency-transposition invariance** natively (the principled, *non-learned* way to get the invariance the pixel VAE never achieved).
- **Pipeline.** Joint TF scattering → flatten → PCA → kNN.
- **Params to sweep.** Time and frequency `J/Q`; transposition-invariance scale.
- **Output.** Joint-TF scattering vector.
- **Traps.** This *is* a spectrogram representation — but a fixed transform with guaranteed invariances, not a learned pixel objective. Keep it clearly labeled as the VAE-diagnostic arm.
- **Scale.** Heavier than 2a but GPU-batched is fine.

**Predictions to log:**
- 2a: competitive with elastic overall, **wins specifically on the noisy/oscillatory pocket and on warped jump/step** (where deformation-stability pays off).
- 2b: if it **matches or beats soft-DTW**, that's strong evidence the seven VAE failures were about the *learned pixel objective*, not about spectrograms — and the principled path is a small learned encoder *on top of a scattering front-end* (separate future handoff).

---

## Evaluation (identical for all four)

1. Leave-one-out kNN retrieval purity vs human shape labels, bootstrap CIs. k swept {1,5,15}.
2. Baselines on the same split: Euclidean-on-contour, **soft-DTW (incumbent)**.
3. **Per-family purity breakdown** (flat, chevron, jump/step, complex, short, noisy pocket, …) — this is where the predictions above are confirmed or falsified. A single aggregate number hides the interesting result.
4. Each method reported in 4 settings: {within-cohort, pooled} × {invariant-only, invariant⊕side-channels}.
5. Reversal unit test must pass (or the direction-augmented variant is used and noted).

Output one tidy table: rows = methods × settings, cols = overall purity + per-family purity, with CIs, soft-DTW row bolded as the bar.

---

## Execution order

- **Phase 0** — Harness, loader, baselines, side-channel store, reversal unit test, ANN kNN. Validate against the known soft-DTW numbers.
- **Phase 1** — M5 (turning function). Confirms harness end-to-end on the cheapest method.
- **Phase 2** — M4 (recurrence + RQA).
- **Phase 3** — M3 (persistence).
- **Phase 4** — M2a (Scattering1D), then M2b (joint-TF, exploratory).
- **Phase 5** — Synthesis: the comparison table + per-family analysis + a short written read on *which invariance bought what*, and a recommendation on which representation(s) to carry into the downstream manifold/sequence/biology pipelines.

---

## Dependencies

```
numpy scipy scikit-learn
kymatio            # M2 (GPU build if available)
giotto-tda         # M3 (or ripser / homcloud)
pyrqa              # M4 (or pyunicorn)
pynndescent        # ANN kNN (or faiss-gpu)
matplotlib         # diagnostics only
```

## Definition of done

For each of M2a/M2b/M3/M4/M5: a reproducible `encode()` saved to `features/`, an entry in the comparison table with CIs and per-family breakdown, a passing (or explicitly-handled) reversal test, and a one-paragraph note on whether its logged prediction held. No method is "rejected" on aggregate purity alone — a method that loses overall but wins a family (e.g., persistence on extrema-configuration) is a kept result, because the point is to map invariances to shape factors.

# Spectrogram Cleaning Subsystems

There are **four independent spectrogram/audio "cleaning" implementations** in this
repo. They look similar but operate on different data representations, serve different
pipelines, and use different sample-rate conventions. **They are genuinely distinct and
must not be force-merged** (verified 2026-05-27; ref memory note
`project_cleaning_stacks_three_distinct`). This page names each, its call sites, and
which pipeline it is canonical for.

> **Why this doc exists:** the cleaning sprawl was the load-bearing "this doesn't make
> sense" item that triggered the 2026-05-27 repo housekeeping
> (`docs/handoffs/2026-05-27_repo-housekeeping.md`, finding H1). Every claim below was
> verified against the code at HEAD `75395217`, not assumed.

## At a glance

| Stack | File | Operates on | Canonical for | Lineage |
|-------|------|-------------|---------------|---------|
| **1. Classifier 4-layer** | `src/usv_spectrogram/classifier/cleaning_pipeline.py` | dB spectrogram | **Pre-CNN cleaning for Module 18.x** (lab VocalMat classifier train/eval) | New (18.x). Wraps/reproduces our own production stack. Default `sample_rate_hz=250_000` (VocalMat-aligned). |
| **2a. Production app — notch** | `src/usv_spectrogram/app/core/notch.py` | **time-domain audio** | **Live PyQt6 + `run_batch_detection.py`** equipment-tonal removal | Adaptive soft-notch (`docs/handoffs/2026-05-11_adaptive-soft-notch.md`). 300 kHz. |
| **2b. Production app — denoise** | `src/usv_spectrogram/app/core/denoise.py` | linear-magnitude spectrogram | **Live + batch detection** temporal baseline subtraction | Boll 1979. Kernel derived from `corpus.STFT_HOP`/`SAMPLE_RATE_HZ` (300 kHz). |
| **3. SIS prefilter** | `src/usv_spectrogram/features/spectrogram_filter.py` | linear-magnitude spectrogram | **SIS-benchmark** (ridge / Oren / AMVOC), modules 17.3/17.5/17.6 | Separate; `ROADMAP_SIS_BENCHMARK §17.2`. 300 kHz. Returns `(cleaned, mask)`. |
| **4. DeepSqueak contour port** | `scripts/deepsqueak_focus_stft.py` + `scripts/contour_mask_utils.py` | per-call focus STFT → power-spectrogram mask | **Contour-masked VAE** patch generation | Line-by-line port of DeepSqueak `CreateFocusSpectrogram.m` + `CalculateStats.m`. **Tracked on `main` at HEAD** (earlier handoff/memory said worktree-only — that is now stale; verified `git ls-files` 2026-05-27). |

## Call-site map

| Stack | Imported / called by (non-test) |
|---|---|
| **1. Classifier 4-layer** (`clean_spectrogram` / `CleaningConfig`) | `scripts/cnn_prepare_training_data.py`, `scripts/prepare_held_out_844.py`, `scripts/cnn_cleaning_validation.py`, `scripts/benchmark_baseline_modes.py`, `scripts/compare_baseline_modes_visual.py`, `scripts/cnn_wild_topup.py`, `scripts/profile_prep_phases.py`, `scripts/experiments/patch_duration_sweep.py`. Re-exported in `classifier/__init__.py`. |
| **2a. notch** (`auto_soft_notch` / `TonalLibrary`) | `app/core/audio_loader.py` (live PyQt6/batch path); `scripts/run_batch_detection.py` (behind `--soft-notch`). |
| **2b. denoise** (`subtract_temporal_baseline`) | `app/core/audio_loader.py` (behind `subtract_baseline`); `scripts/preview_spectral_subtraction.py`; **wrapped by** stack 1 Layer 2. |
| **3. SIS prefilter** (`prefilter_spectrogram` / `FilterConfig`) | Re-exported in `features/__init__.py`; consumed by `features/ridge_tracker.py` and the SIS 17.3/17.5/17.6 modules. |
| **4. contour port** (`deepsqueak_focus_stft` + `contour_mask_utils`) | `scripts/sweep_contour_mask.py`, `scripts/mass_apply_contour_mask.py` → `scripts/train_contour_vae_v2.py` / `window_calls_to_patches.py` / `assemble_combined_patches.py`. |

**Production detection path (the canonical one):** `run_batch_detection.py` →
`audio_loader.SpectrogramAudioLoader` → (optional `auto_soft_notch` on audio) → STFT →
(optional `subtract_temporal_baseline`) → `sliding_inference._apply_mad_normalization`.
The classifier `cleaning_pipeline` is **not** in this path; it is the pre-CNN
*training/eval* stack for Module 18.x.

## What each does

**1. Classifier 4-layer (`cleaning_pipeline.py`).** Fixed layer order
(`clean_spectrogram`): **soft-notch → baseline subtraction → global MAD →
per-recording Z-score**. Each layer independently toggleable for ablation. Verified
internals (2026-05-27):
- Layer 3 (global MAD) **reproduces `sliding_inference.py:_apply_mad_normalization`
  byte-for-byte** — confirmed identical math: same `2.0`/`4.0` MAD scales,
  clip-before-normalize, `+1e-12` guard, `vmax>vmin` degenerate fallback. It
  *reproduces rather than imports* because the upstream is a method on a PyQt6/CNN-heavy
  class and cannot be cleanly imported as a free function.
- Layer 2 (baseline) is a true **wrapper** around `denoise.subtract_temporal_baseline`,
  with an in-module fallback for minimal worktree checkouts.
- Layer 1 (soft-notch) only reuses the `TonalLibrary` **data format**; it
  re-implements the filter in the **spectrogram domain** (the production notch is
  time-domain audio) and no-ops when no tonal library is supplied. The "wrapper" label
  in the docstring overstates this layer.
- Layer 4 (Z-score) is a 2D analogue that **explicitly diverges** from the 1D
  `postprocessing/normalization.normalize_scores_per_recording` (global median+MAD vs a
  bottom-50%-percentile noise slice).
- *Consolidation verdict:* **No.** Only Layer 2 genuinely delegates; Layers 1 and 4 are
  intentionally different from their nominal upstreams.

**2a / 2b. Production app (`notch.py`, `denoise.py`).** The live detection cleaning used
by the PyQt6 app and `run_batch_detection.py`. `notch.auto_soft_notch` removes equipment
tonals in **time-domain audio** before STFT; `denoise.subtract_temporal_baseline` removes
a per-bin temporal noise floor in linear magnitude after STFT. *Consolidation verdict:*
**No** — this is the production source of truth that stack 1 partially wraps.

**3. SIS prefilter (`spectrogram_filter.py`).** A stateless `prefilter_spectrogram`
(median filter → per-column robust noise floor → amplitude+band mask) returning
`(cleaned, mask)`, built for the SIS-benchmark ridge/Oren/AMVOC modules. Different output
contract (returns a boolean mask) and different lineage. *Consolidation verdict:* **No.**

**4. DeepSqueak contour port (`deepsqueak_focus_stft.py` + `contour_mask_utils.py`).**
A faithful Python port of DeepSqueak's per-call adaptive focus-STFT contour extraction,
plus hard/soft bandwidth masking of the power spectrogram; feeds the contour-masked VAE.
Distinct from stack 3: it computes a per-call adaptive-window STFT (not the canonical
hop-128 grid) and masks around the extracted tonal ridge rather than applying a global
noise-floor threshold. *Consolidation verdict:* **No.**

## Decision matrix — which cleaning do I use?

| You are working on... | Use this stack |
|----------------------|----------------|
| Live PyQt6 detection or `run_batch_detection.py` | **2a/2b** (`notch.py` + `denoise.py`) |
| Module 18.x lab VocalMat classifier train/eval patches | **1** (`cleaning_pipeline.py`) |
| SIS benchmark (ridge / Oren / AMVOC) | **3** (`spectrogram_filter.py`) |
| Contour-masked VAE patches | **4** (`deepsqueak_focus_stft.py` + `contour_mask_utils.py`) |

---

## VocalMat MATLAB equivalence (H1b adjudication, 2026-05-27)

**The standing question:** *"matlab and our cleaning pipeline is the same, we ported the
same cleaning code … or the cleaning at inference didn't do the right cleaning."* Does
VocalMat's MATLAB spectrogram rendering equal our python cleaning?

**Key structural fact: there is no single "our pipeline" to compare against.** Two
distinct render paths exist, with *opposite* verdicts:

| Path | Source | Uses `cleaning_pipeline.py`? | Equivalent to VocalMat? |
|------|--------|------------------------------|-------------------------|
| **(A) v1 transfer-test inference** (the run that produced noise-recall **0.64**) | `.claude/worktrees/vocalmat-classifier-test/scripts/vocalmat_test/run_inference.py` | **No** | **YES — a faithful port** |
| **(B) Module 18.2b training-data prep** for our own WAV cohorts | `scripts/cnn_prepare_training_data.py` → `clean_spectrogram` | **Yes** | **NO — fundamentally different** |

**Verdict adjudication:** the user's recollection is **correct for Path A and incorrect
for Path B.** A faithful VocalMat port really exists — it lives in `run_inference.py`,
*not* in the module named `cleaning_pipeline.py`. The port reproduces VocalMat's rendering
(STFT hamming-256/hop-128/nfft-1024 at 250 kHz → `10·log10(P)` → `>45 kHz` crop →
`mat2gray` min-max → `flipud` → resize 227×227 → 3-channel). Source of truth checked:
`github.com/ahof1704/VocalMat` `vocalmat_classifier/vocalmat_classifier.m:255-257` (the
saved CNN image) + `vocalmat_identifier/vocalmat_identifier.m:130-175` (the `A_total`
build). VocalMat's `imadjust`/`imbinarize`/morphology operate on a *separate* `B`
variable used only for blob segmentation — **not** on the saved CNN training image, which
gets only `mat2gray`.

**Where Path B diverges from VocalMat (and therefore from Path A):**

| Step | VocalMat (training PNGs) | Path A `run_inference.py` | Path B `cleaning_pipeline.py` prep |
|------|--------------------------|---------------------------|------------------------------------|
| dB conversion | `10·log10(P)` (power) | `10·log10(P)` ✓ | `20·log10(\|S\|)` — **2× dB scale** ✗ |
| Frequency band | **> 45 kHz only** | > 45 kHz ✓ | 0 → Nyquist, **no crop** ✗ |
| Denoising | none on CNN image | none ✓ | **Boll baseline subtraction** ✗ |
| Contrast/normalize | `mat2gray` (min-max) | `mat2gray` ✓ | **global MAD + per-recording Z-score** ✗ |
| Resize | `imresize` (bicubic) | Pillow BICUBIC ≈✓ | Pillow **BILINEAR** ✗ |

**Consequence (the load-bearing part):** because the v1 transfer test renders *both* its
training images and its inference patches in VocalMat style (Path A on both sides), its
0.64 noise-recall is **NOT** caused by a train/inference cleaning mismatch — train and
inference are matched. The poor noise-recall is therefore a **genuine domain gap** between
VocalMat's clean lab cohort and our recordings (or an AlexNet transfer limit), consistent
with `project_lab_transfer_v1_vs_dann_patchsweep` (κ=0.13, bal-acc flat under
patch-duration sweep). This redirects the next experiment away from "fix the cleaning"
toward **human-in-the-loop fine-tune of v1** (Path B in
`docs/handoffs/2026-05-27_lab-classifier-transfer-solve.md`).

**The `cleaning_pipeline.py` docstring's lineage claim is accurate**: it attributes itself
to our own `sliding_inference.py` MAD + Boll baseline, *not* to VocalMat — and that is
true. It is genuinely our production-detection cleaning stack, never a VocalMat port. Path
B would only be appropriate for cleaning **our** detection patches, never for matching the
VocalMat-rendered training corpus.

**Evidence basis & limit:** this is a *static* MATLAB-source-vs-python-code comparison,
which is the authoritative comparison. The cheaper empirical demo-WAV render proxy was
**not run** — the local `data/vocalmat_sample/` holds only a `.gitignore` placeholder (no
demo WAV present), so a render would require an OSF download; the static comparison is
decisive without it. VocalMat `.m` sources fetched from
`github.com/ahof1704/VocalMat` (master).

## Invariants / cross-references
- Stack 1 Layer 3 ≡ `sliding_inference.py:_apply_mad_normalization` — keep in sync
  (`feedback_cnn_inference_global_mad`: global MAD on the whole spectrogram, never
  per-window).
- Stack 1's default sample rate is **250 kHz** (VocalMat-aligned), NOT the 300 kHz corpus
  canonical (ADR-001). Both are accepted; 250 kHz applies to that pipeline only.
- Stacks 2a/2b derive kernels from `corpus.STFT_HOP` / `SAMPLE_RATE_HZ` — never hardcode.
- Related notes: `project_cleaning_stacks_three_distinct`,
  `project_cleaning_pipeline_inventory`, `project_lab_transfer_v1_vs_dann_patchsweep`.

# CNN Data Preparation (Module 18.2b)

> Build a unified training-data manifest for the lab CNN classifier:
> walk the VocalMat OSF download, process our own 300 kHz lab/wild WAVs
> (resample → cleaning stack → STFT → 0.22 s patches), emit sanity
> patches for human review, and write recording-level-grouped 80/10/10
> train/val/test splits.

## What it is

Module 18.2b is the data-prep stage that turns three heterogeneous inputs
(VocalMat 227×227 PNGs, lab 131204 WAVs at 300 kHz, wild 5970 WAVs at
300 kHz) into a single uniform corpus the ResNet-18 classifier in Module
18.3 can train on. The hard constraint is **no recording-level leakage**:
a recording in train must not appear in val or test, or the cage-confound
issue documented in the VAE comparison memo reappears as data leakage.

## Architecture

```
src/usv_spectrogram/classifier/
  resample.py              # 300 → 250 kHz polyphase + custom Kaiser FIR
  dataset.py               # GRIMSLEY_12_CLASSES + stratified split builder
scripts/
  cnn_prepare_training_data.py   # CLI orchestrator + smoke entry point
tests/classifier/
  test_resample.py               #  9 tests
  test_dataset.py                # 11 tests
  test_cnn_prepare_training_data.py   # 6 tests (incl. <30 s smoke)
data/
  vocalmat_full/.gitignore       # 12 GB download — never committed
  lab_cnn_training/              # output of the prep CLI (also gitignored)
docs/modules/
  cnn-data-preparation.md        # this file
```

## Resampling (`resample.py`)

Our hardware records at 300 kHz (`corpus.SAMPLE_RATE_HZ`, ADR-001).
VocalMat is anchored at 250 kHz. A 5/6 rational resample bridges them:

- Polyphase: `scipy.signal.resample_poly(samples, up=5, down=6, window=...)`
- Anti-aliasing FIR: 481-tap Kaiser (β=14) low-pass at 120 kHz cutoff,
  designed at the intermediate 1.5 MHz rate. The cutoff sits 5 kHz below
  the new Nyquist (125 kHz) as a guard band.

### Why a custom FIR

`scipy.signal.resample_poly`'s **default** Kaiser window is short (121
taps, β=5) and only delivers ~32 dB rejection 15 kHz above the cutoff —
the ROADMAP `test_140khz_tone_anti_aliased_below_40db` test demands
≥ 40 dB at that distance. The custom FIR (longer + higher β) closes that
gap. The slowdown (~4× the convolution work) is invisible against the
per-WAV STFT + cleaning cost and the prep run is offline-only.

### Public API

```python
from usv_spectrogram.classifier.resample import (
    SOURCE_SAMPLE_RATE_HZ,   # 300_000 — re-exported from corpus
    TARGET_SAMPLE_RATE_HZ,   # 250_000 — VocalMat-aligned
    RESAMPLE_UP, RESAMPLE_DOWN,
    resample_to_vocalmat,
)
```

`resample_to_vocalmat(samples) -> np.ndarray` is mono-only; passing a 2-D
array (stereo OR an accidental `(1, N)` row vector) raises `ValueError`.
Output is always `float32`.

## Stratified split (`dataset.py`)

`build_stratified_split` is the no-leakage allocator. Inputs are a per-call
manifest with columns `path`, `class`, `source_recording`, `duration_ms`;
outputs are three CSVs (one per split) plus `class_weights` for weighted
cross-entropy and `oversample_targets` for in-batch replacement-sampling.

### Algorithm: LPT-greedy by deficit

For each class independently:

1. Gather unique `source_recording` IDs.
2. Deterministically shuffle (seed-driven entropy).
3. Stable-sort *descending by call count* (Longest-Processing-Time first).
4. Walk the sorted list; place each recording into whichever split has
   the largest remaining `target_calls − actual_calls` deficit.

Why descending: large recordings placed first leave the small ones as
fine-tuners. A naive "fill-train-then-val-then-test" greedy can overshoot
test by 7+ percentage points when the last recording happens to be large;
LPT keeps every class within ±5 % of the 80/10/10 target.

Ties in call count are broken by the shuffle order, so different seeds
still produce different splits when the distribution has ties — which it
does in the multinomial-with-exponential-weights synthetic fixture and in
the real VocalMat distribution.

### Class weights and oversample targets

- `class_weights[cls] = 1 / count(cls)` normalized so the mean is 1.0 →
  sum equals `n_classes = 12`. Plugged into
  `torch.nn.CrossEntropyLoss(weight=...)`.
- `oversample_targets[cls] = max(actual_count, median_count)`. Minority
  classes are brought up to the median via *replacement* sampling on the
  training set only; majority classes are never reduced.

ROADMAP **D5** (keep all 12 classes + weighted CE + focal loss +
oversampling) is encoded here. Revisit only if v1 confusion matrix shows
per-class precision < 0.20.

### Recording-level grouping is HARD

The split satisfies
`set(train.source_recording) ∩ (set(val.source_recording) ∪ set(test.source_recording)) == ∅`
unconditionally. `test_recording_level_no_leakage` enforces this — the
test reads back the three CSVs and intersects on `source_recording`.

## Label assignment for lab/wild WAVs (Option A architecture)

Lab and wild WAV patches are **NOT** assigned a placeholder label and **NOT**
mixed into the supervised train/val/test manifests. Master-reviewer flagged
this in WARNING 2 of `docs/reviews/cnn-data-preparation-review.md`: a
placeholder "Noise" label would let real-USV calls leak into Module 18.3's
supervised CNN as Noise, silently corrupting the training signal.

Instead:

- Lab and wild patches are written to `output_dir/patches/{lab,wild}/`.
- A separate `output_dir/domain_unlabeled.csv` lists them with columns
  `path, cohort, source_recording, duration_ms`. Module 18.4 (DANN
  cage-invariance training) consumes this file as the unlabeled domain
  side.
- A random sample (up to 50 per cohort) is copied into
  `output_dir/sanity_patches/` for human review.
- The supervised manifest `output_dir/manifest_all.csv` and the
  `{train,val,test}/manifest.csv` splits contain **only VocalMat rows**.

## End-to-end CLI (`cnn_prepare_training_data.py`)

Steps:

1. Walk `--vocalmat-source` for snake_case class folders, emit one manifest
   row per PNG. `source_recording` is unique per PNG (no recording-session
   metadata in VocalMat OSF metadata).
2. For each `--lab-wav-dirs` / `--wild-wav-dirs` WAV:
   - load (soundfile, mono)
   - `resample_to_vocalmat` if at 300 kHz; pass-through at 250 kHz
   - `_spectrogram_db` (VocalMat STFT: Hamming-256 / hop-128 / NFFT-1024)
   - `clean_spectrogram(..., CleaningConfig())` — all four layers
     (soft-notch is silent no-op without a calibrated tonal library)
   - global MAD-normalize before patching (C2: never per-window)
   - slice every `--patch-duration-s` worth of frames into a 227×227 RGB
     patch, save PNG
3. Emit up to 50 random patches per cohort (`vocalmat`, `lab`, `wild`)
   into `output_dir/sanity_patches/` for the **human review checkpoint**
   before Module 18.3 starts.
4. Build the stratified split → write
   `output_dir/{train,val,test}/manifest.csv`.

### Required CLI flags

```
--vocalmat-source <dir>            VocalMat OSF download root (12 classes)
--lab-wav-dirs <dir> [<dir> ...]   one or more lab WAV folders (300 kHz)
--wild-wav-dirs <dir> ...          optional wild WAV folders (300 kHz)
--output-dir <dir>                 split-manifest + sanity-patches root
--patch-duration-s 0.22            ROADMAP D1: 0.22 primary
--workers <int>                    sequential by default; arg reserved
--skip-checksum-verify             skip VocalMat hash check (not impl'd)
--seed 1729                        controls shuffle + sanity sampling
```

### Importable entry point

```python
from cnn_prepare_training_data import main
exit_code = main([...])   # returns int 0 on success
```

The smoke test in `tests/classifier/test_cnn_prepare_training_data.py`
uses this in-process path (faster + better tracebacks than subprocess).

## Constants — naming convention

VocalMat STFT params share *numeric* values with the corpus STFT (both use
hop 128) but the **semantics differ**:

- `corpus.STFT_HOP = 128` is anchored to the production detection pipeline
  at 300 kHz / NFFT 512.
- `_VOCALMAT_STFT_HOP = 128` is anchored to the VocalMat paper at 250 kHz
  / NFFT 1024 / Hamming 256.

Keep them separate. A future tweak to either should not silently change
the other. The classifier patch geometry is a CNN training-grid invariant
(C5 + the "CNN FREEZE" rule in the corpus-invariant hook).

## Cross-phase constraints honoured

- **C1** (250 kHz internal, never modify `corpus.py`): `resample.py`
  imports `corpus.SAMPLE_RATE_HZ` as `SOURCE_SAMPLE_RATE_HZ` and
  re-exports `TARGET_SAMPLE_RATE_HZ` / `RESAMPLE_UP` / `RESAMPLE_DOWN`
  from `classifier/__init__.py` — no redeclaration.
- **C2** (global MAD before crop): `_wav_to_patches` runs
  `clean_spectrogram` (which does global MAD as Layer 3) on the *whole*
  spectrogram, then crops patches. Never crop first.
- **C3** (4-layer stack reused, not rebuilt): `CleaningConfig` + the
  Module 18.1 `clean_spectrogram` entry point are imported as a
  black box.
- **C4** (soft-notch no-op without tonal library): default
  `CleaningConfig()` has `tonal_library_path=None`; the soft-notch layer
  silently no-ops. Valid for VocalMat and wild-5970; lab 131204 prep
  would set a calibrated library.
- **C5** (new code in `classifier/` only; production detection
  untouched): all 3 new files live under `src/usv_spectrogram/classifier/`
  or `scripts/`. None of the "Do NOT touch" canary files are modified.
- **C6** (cage vs rig terminology): see `feedback-cage-not-rig-terminology`
  in the user-memory index.

## Test summary

| File | Tests | Coverage |
|------|-------|----------|
| `test_resample.py` | 9 | ROADMAP plan items 1-5 + 4 boundary cases |
| `test_dataset.py` | 11 | ROADMAP plan items 6-9 + 7 robustness checks |
| `test_cnn_prepare_training_data.py` | 6 | ROADMAP plan item 10 + 5 CLI checks |

100/100 classifier-package tests green (`pytest tests/classifier/ -q`).

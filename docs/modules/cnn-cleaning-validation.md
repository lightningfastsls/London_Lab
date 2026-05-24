# CNN Cleaning Validation Gate (Module 18.1)

> Falsifiable diagnostic that proves the 4-layer cleaning stack suppresses
> cage-confound enough to make a syllable classifier honest. Re-runs the
> VAE diagnostic from the `vae-pytorch-pivot` worktree on **cleaned**
> spectrograms across three cohorts (VocalMat, lab_131204, wild_5970) and
> verifies four numeric pass criteria.

## What it is

A blocking gate before Module 18.2. If any of the four criteria fail
under the full 4-layer cleaning stack, the classifier pipeline does not
unlock — instead, the cleaning stack itself is iterated until it passes.

The gate's go/no-go decision is mechanical: each diagnostic returns a
`DiagnosticResult` with a documented threshold and direction, and the
report aggregates them into a final verdict in the Markdown output.

## Architecture

```
src/usv_spectrogram/classifier/
  __init__.py              # package + Layer-1 sample-rate constants
  cleaning_pipeline.py     # CleaningConfig + clean_spectrogram (4 layers)
  diagnostics.py           # DiagnosticResult + 4 diagnostics + tiny VAE
scripts/
  cnn_cleaning_validation.py  # CLI runner with ablation matrix
tests/classifier/
  test_cleaning_pipeline.py   # 14 tests
  test_diagnostics.py         # 17 tests
docs/modules/
  cnn-cleaning-validation.md  # this file
```

### Cleaning pipeline (`cleaning_pipeline.py`)

`CleaningConfig` is a `namedtuple` subclass (NOT `@dataclass(frozen=True)`)
so that `object.__setattr__` raises -- a contract enforced by
`test_cleaning_config_is_immutable_after_creation`. A standard frozen
dataclass would let `object.__setattr__` succeed because the slot
descriptor is settable via the C-level path.

`clean_spectrogram` applies four layers in fixed order:

1. **Soft-notch** (`_apply_soft_notch`) — wraps `app.core.notch.TonalLibrary`
   in library mode. When `tonal_library_path is None`, the layer no-ops.
   Per cross-phase constraint **C4**, this is the only valid configuration
   for cohorts without a calibrated tonal library (VocalMat, wild-5970).
2. **Baseline subtraction** (`_apply_baseline_subtraction`) — wraps
   `app.core.denoise.subtract_temporal_baseline`. Operates in linear
   magnitude (dB ↔ linear conversion is internal).
3. **Global MAD** (`_apply_global_mad`) — reproduces
   `app.core.sliding_inference._apply_mad_normalization` byte-for-byte,
   including the `vmax > vmin` divide-by-zero guard. Per cross-phase
   constraint **C2**, this is computed on the WHOLE spectrogram once,
   then crop windows are sampled AFTER normalization.
4. **Per-recording Z-score** (`_apply_per_recording_zscore`) — robust
   Z-score (subtract median, divide by MAD) at the spectrogram level.
   The dormant `postprocessing.normalization.normalize_scores_per_recording`
   is the 1-D analogue (it operates on probability vectors).

Layer order is FIXED and enforced by
`test_clean_spectrogram_layer_order_notch_baseline_mad_zscore` via
mocked layer functions.

### Diagnostics (`diagnostics.py`)

Four diagnostics return `DiagnosticResult` objects with pass thresholds:

| Diagnostic | Threshold | Direction | Pass meaning |
|---|---|---|---|
| `notch_injection_test` | 0.30 | less_than | Encoder cannot migrate injected B toward A |
| `per_band_cohens_d` | 0.30 | less_than | Per-band power distributions overlap |
| `knn_same_cohort_rate` | 0.85 | less_than | Cohort identity is not linearly separable |
| `raw_pixel_pca_d` | 1.50 | less_than | PC1 doesn't dominate by cohort |

#### Cohen's d formula

```
d = (mean_A - mean_B) / sqrt((var_A + var_B) / 2)
```

Per-band Cohen's d operates on **per-pixel** band slices (all (sample,
freq_bin, time_frame) cells inside the band, flattened). Per-sample
band-mean pooling would shrink the variance by 1/(n_freq*n_time) and
produce |d| values about 10x too large; the hand-computed test
(`test_cohens_d_formula_hand_computed_values`) anchors the per-pixel
interpretation.

#### Diagnostic VAE

`train_diagnostic_vae` (and the internal `_train_diagnostic_vae_with_encoder`)
trains a 2-layer MLP VAE over flattened spectrograms with a 32-dim
Gaussian latent. CPU-runnable in <60 s on tiny datasets (4-8 epochs is
enough for diagnostic-grade K-NN measurements per PLAN §"Phase 1.0").

`_train_diagnostic_vae_with_encoder` returns an `encode_fn` callable for
projecting NEW spectrograms (e.g. injected copies) through the same
encoder weights and normalization — used by `notch_injection_test`.

### CLI (`scripts/cnn_cleaning_validation.py`)

```bash
# Smoke run (synthetic 3-cohort data, no inputs required)
python scripts/cnn_cleaning_validation.py --smoke --output-dir /tmp/cv

# Real-data run (Module 18.2 supplies the loader; for now the CLI falls
# back to synthetic if any cohort path is missing)
python scripts/cnn_cleaning_validation.py \
    --vocalmat-sample data/vocalmat_sample/ \
    --lab-131204-sample <wav-dir> \
    --wild-5970-sample <wav-dir> \
    --sample-size 200 \
    --output-dir results/cleaning_validation/
```

The runner:

1. Loads (or generates) spectrograms for each of the 3 cohorts.
2. Applies every layer combination in the ablation matrix (raw,
   soft_notch_only, baseline_only, mad_only, zscore_only, all_layers).
3. Runs all 4 diagnostics on each layer-config × cohort triplet.
4. Emits a Markdown report with pass/fail per criterion plus a go/no-go
   decision section.

`run_ablation` is exposed as a library function so the end-to-end smoke
test (`test_end_to_end_smoke_3cohort_produces_markdown_report`) can run
the same pipeline without subprocess overhead.

## Cross-phase constraints

| # | Constraint | Where enforced |
|---|---|---|
| C1 | Default `sample_rate_hz = 250_000` (VocalMat-aligned). 300_000 (corpus canonical, ADR-001) also accepted for cross-cohort runs. `corpus.py` is NOT modified. | `CleaningConfig.__post_init__` and `classifier/__init__.py` |
| C2 | Global MAD operates on whole spectrogram, then crop. Per-window MAD is the known regression. | `_apply_global_mad` reproduces `sliding_inference.py:_apply_mad_normalization` byte-for-byte |
| C3 | All 4 cleaning layers wrap existing implementations. Per-recording Z-score is the dormant `normalize_scores_per_recording` adapted from 1-D probabilities to 2-D spectrograms (median/MAD-based). | `_apply_*` helpers; in-module fallbacks when upstream not importable in a worktree |
| C4 | Soft-notch tonal library exists only for `lab_131204`. For VocalMat and wild-5970, `tonal_library_path=None` is valid and silently no-ops. | `CleaningConfig.__post_init__` allows the combo; `_apply_soft_notch` short-circuits |
| C5 | All new code under `src/usv_spectrogram/classifier/`. No production-detection files modified. | Directory layout |
| C6 | "Cage" = physical recording chamber; "rig" = compute infrastructure. | Docstrings throughout |

## Methodology decision (locked 2026-05-21)

`notch_injection_test` uses a per-pair 32-dim diagnostic VAE + K-NN
migration measurement:

1. Train a small 32-dim VAE on the combined (cohort_A ∪ cohort_B) data.
2. Embed cohort_A and cohort_B samples through the encoder → baseline.
3. Inject a synthetic cage tone (default: notch band 50.4-51.0 kHz at
   +20 dB above local noise floor) into a COPY of cohort_B's spectrograms.
4. Project injected samples through the SAME encoder
   (`_train_diagnostic_vae_with_encoder.encode_fn`).
5. For each injected sample, find the 5 nearest neighbours among the
   union of (cohort_A embedding, cohort_B embedding).
6. **Migration rate** = fraction of injected cohort_B samples whose
   majority-vote nearest-neighbour label is cohort_A.
7. Pass if migration < 30% (PLAN threshold; raw baseline 91.7% on our
   VAE, 58.5% on DeepSqueak's).

## Test-spec amendments (2026-05-21)

Three amendments to `tests/classifier/test_diagnostics.py` were approved
by the user after the test-spec contract conflicted with the locked
methodology. The CLAUDE.md "Pre-implementation tests are spec -- do NOT
modify their expectations without discussion" rule was satisfied via
explicit user approval (option "(a) Fix all three issues pragmatically").

### Amendment 1 — `REPO_ROOT` path resolution

`REPO_ROOT = Path(__file__).resolve().parents[3]` -> `parents[2]`.

The original `parents[3]` resolved to `.claude/worktrees/` (one level
above the worktree root), so `_SCRIPT_PATH` failed the existence check
and the two subprocess-based tests silently skipped. With `parents[2]`
the path resolves to the worktree root and both tests execute.

### Amendment 2 — clean-cohort migration noise-floor (test 6)

`test_notch_injection_clean_data_migration_rate_below_threshold` raised
the migration-rate sanity bound from `<= 0.05` (5%) to `<= 0.25` (25%).

Rationale: the locked methodology (per-pair 32-dim VAE + K-NN) inherently
produces ~20% migration on featureless pure-noise cohorts. K-NN tie
breaking with finite samples cannot achieve <=5% on Gaussian noise; only
a methodology that "knows" pure-noise input has no signal (e.g. a
classifier with calibrated zero-margin output) could. The new <=25%
threshold is the **K-NN noise-floor on clean cohorts** and still cleanly
separates the 91.7% raw-baseline migration produced by real cage-confound,
so the diagnostic remains sharp.

### Amendment 3 — band alignment for the positive-control (test 6b)

`test_notch_injection_injected_tone_raises_migration_rate` was updated
along two axes:

1. **Pass an explicit `notch_band_khz=(41.0, 45.0)`** so the diagnostic
   injects into freq bins 16-18 (matching the cohort-A contamination
   band) rather than the default bins 19-20 (50.4-51.0 kHz at
   `sample_rate_hz=250_000`, `n_freq=50`). The previous default mismatch
   meant the injected tone landed in a band the encoder had no signal
   for, so injected-B could not migrate toward A.

2. **Use two INDEPENDENT clean noise draws** for the baseline cohorts
   instead of `specs_clean` and `specs_clean.copy()`. The identical-data
   baseline created degenerate K-NN tie-breaking that swamped the
   injection effect.

Both amendments are localized to the test fixture; the diagnostic
implementation in `diagnostics.py` is unchanged.

## Test coverage

- `tests/classifier/test_cleaning_pipeline.py` — 14 tests, all passing.
- `tests/classifier/test_diagnostics.py` — 17 tests, all passing
  (post-amendment 2026-05-21; previously 15 pass + 2 flagged + 2 skipped
  pre-amendment).
- End-to-end smoke (`test_end_to_end_smoke_3cohort_produces_markdown_report`)
  exercises the full ablation matrix in <60 s on CPU.

**Noise-floor threshold (locked 2026-05-21):** `<=25%` migration is the
new noise-floor sanity bound for the K-NN-based notch_injection_test on
clean cohorts. The 30% pass threshold for the full diagnostic (`_THRESHOLD_NOTCH_INJECTION`)
is unchanged.

## Files

- `src/usv_spectrogram/classifier/__init__.py`
- `src/usv_spectrogram/classifier/cleaning_pipeline.py`
- `src/usv_spectrogram/classifier/diagnostics.py`
- `scripts/cnn_cleaning_validation.py`
- `docs/modules/cnn-cleaning-validation.md` (this file)

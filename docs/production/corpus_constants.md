# Corpus Constants — Canonical Signal-Processing Parameters

> **What this is:** The single source of truth for the physical signal-processing
> constants (sample rate, USV frequency band, STFT parameters) that *every* other
> module in this lab depends on, plus the empirical per-dataset facts layer.
> **Status:** Live / current. Layer 1 landed 2026-04-17; values are LOCKED to the
> production CNN training grid.
> **Production artifact (Layer 1 code):** `src/usv_spectrogram/corpus.py` (52 lines).
> **Production artifact (Layer 2 data):** `data/corpus_facts/{5970,3452,9252,lab_131204}.json`.
> **Danger rule:** Changing `USV_FREQ_MIN_HZ`/`USV_FREQ_MAX_HZ` in `corpus.py` without
> retraining the CNN silently corrupts inference. See [CNN freeze](#the-cnn-freeze-the-one-rule-you-must-not-break).

This is the foundation document. Most other production docs point back here for
their sample rate, band, and STFT numbers. If you change anything described here,
assume you are touching every downstream pipeline.

---

## 1. Operate

There is nothing to "run" in the traditional sense for Layer 1 — `corpus.py` is a
set of constants you **import**. Layer 2 (`data/corpus_facts/*.json`) is generated
by a script and read as data. This section covers both: how to consume the
constants in code, and how to regenerate / read the empirical facts.

### 1.1 Consuming Layer 1 constants (the normal case)

Import from `usv_spectrogram.corpus`. Never re-type the numbers.

```python
from usv_spectrogram.corpus import (
    SAMPLE_RATE_HZ,      # 300_000
    USV_FREQ_MIN_HZ,     # 20_000
    USV_FREQ_MAX_HZ,     # 120_000
    STFT_N_FFT,          # 512
    STFT_HOP,            # 128
)
from usv_spectrogram import corpus

# Always pass sr explicitly to any DSP call — never rely on a library default:
import librosa
y, sr = librosa.load(path, sr=SAMPLE_RATE_HZ)   # forces 300 kHz, no resample surprise
D = librosa.stft(y, n_fft=STFT_N_FFT, hop_length=STFT_HOP, window="hann")

# Derived quantities are functions (the formula stays visible), not constants:
corpus.nyquist_hz()              # 150000
corpus.stft_freq_resolution_hz() # 585.9375   (Hz per frequency bin)
corpus.stft_time_resolution_ms() # 1.7066...  (window length in ms)
corpus.stft_hop_ms()             # 0.4266...  (hop length in ms)
```

To run this on the repo interpreter (note `PYTHONPATH=src`, the package lives under `src/`):

```bash
PYTHONPATH=src .venv/bin/python -c \
  "from usv_spectrogram import corpus as c; \
   print(c.nyquist_hz(), c.stft_freq_resolution_hz(), \
         c.stft_time_resolution_ms(), c.stft_hop_ms())"
# -> 150000 585.9375 1.7066666666666666 0.42666666666666664
```

#### Layer 1 constant reference

Source: `src/usv_spectrogram/corpus.py:30-36`. Values verified against source.

| Constant | Value | Unit | Meaning | Source line |
|---|---|---|---|---|
| `SAMPLE_RATE_HZ` | `300_000` | Hz | Recording sample rate (LMT rig). Nyquist = 150 kHz. | `corpus.py:30` |
| `USV_FREQ_MIN_HZ` | `20_000` | Hz | Lower edge of the analyzed USV band. Locked to CNN training. | `corpus.py:32` |
| `USV_FREQ_MAX_HZ` | `120_000` | Hz | Upper edge of the analyzed USV band. Locked to CNN training. | `corpus.py:33` |
| `STFT_N_FFT` | `512` | samples | FFT window length for the detection/analysis STFT. | `corpus.py:35` |
| `STFT_HOP` | `128` | samples | STFT hop (75% overlap: `1 - 128/512`). | `corpus.py:36` |

Derived helper functions (defined `corpus.py:39-52`):

| Function | Returns | Formula | Why it matters |
|---|---|---|---|
| `nyquist_hz()` | `150000` | `SAMPLE_RATE_HZ // 2` | Hard ceiling on representable frequency; 120 kHz band leaves headroom. |
| `stft_freq_resolution_hz()` | `585.9375` | `SAMPLE_RATE_HZ / STFT_N_FFT` | **Frequency resolution** = Hz per FFT bin. Smaller `n_fft` -> coarser frequency, finer time. |
| `stft_time_resolution_ms()` | `1.7067` | `(STFT_N_FFT / SAMPLE_RATE_HZ) * 1000` | Window duration in ms; the smallest time event a single frame can represent. |
| `stft_hop_ms()` | `0.4267` | `(STFT_HOP / SAMPLE_RATE_HZ) * 1000` | Time step between successive frames; the spectrogram's effective time grid spacing. |

> **Intuition for a new lab member (the time/frequency trade-off).** An STFT chops
> the signal into windows of `n_fft` samples and runs an FFT on each. A *longer*
> window (bigger `n_fft`) gives you *finer frequency* resolution (`sr/n_fft` gets
> smaller) but *coarser time* resolution (you average over a longer slice). A
> *shorter* window does the opposite. At `n_fft=512` and 300 kHz we get ~586 Hz per
> bin and a ~1.7 ms window — a deliberate compromise: USVs are short (10–500 ms)
> and narrow-band, so we favor time resolution to catch onset/offset while keeping
> just enough frequency resolution to tell subtypes apart. `hop=128` (75% overlap)
> oversamples in time so the spectrogram looks smooth rather than blocky.

### 1.2 Regenerating Layer 2 empirical facts (`data/corpus_facts/`)

Layer 2 stores *measured* per-dataset statistics that are expensive to recompute
(median inter-call interval, bout counts, transition mutual information, labeling
distributions). These are produced by `scripts/audit_corpus.py` (575 lines) from
already-computed classified CSVs and Phase A1/A2 outputs.

Single dataset:

```bash
.venv/bin/python scripts/audit_corpus.py \
    --dataset 5970 \
    --output data/corpus_facts/5970.json
```

All registered datasets at once (outputs to `data/corpus_facts/<dataset>.json`):

```bash
.venv/bin/python scripts/audit_corpus.py --all
```

Validate the lab tonal-line libraries instead of corpus facts:

```bash
.venv/bin/python scripts/audit_corpus.py --lab-tonal-lines
```

#### `audit_corpus.py` flags

Defined in the argparse block at `scripts/audit_corpus.py:130-160`. Exactly one of
`--dataset`, `--all`, `--lab-tonal-lines` is required (mutually exclusive group).

| Flag | Required? | Default | What it does |
|---|---|---|---|
| `--dataset <name>` | one-of group | — | Process a single dataset. Choices are the keys of `DATASET_REGISTRY`: `3452`, `5970`, `9252` (`audit_corpus.py:131-132`). **Requires `--output`** (`audit_corpus.py:558`). |
| `--all` | one-of group | — | Process every dataset in `DATASET_REGISTRY`; writes to `--output-dir`. Warns (does not fail) when a dataset's input files are missing. |
| `--lab-tonal-lines` | one-of group | — | Validate every JSON in `data/lab_tonal_lines/` against the `TonalLibrary` schema instead of generating corpus facts. Non-zero exit on failure. |
| `--output <path>` | with `--dataset` | — | Output JSON path. Required when `--dataset` is used (`audit_corpus.py:558`). Ignored by `--all`. |
| `--output-dir <dir>` | no | `data/corpus_facts/` | Output directory for `--all` (`audit_corpus.py:156-160`). |

> Note the registry contains `3452`, `5970`, `9252` but **not** `lab_131204`. The
> `data/corpus_facts/lab_131204.json` file was created **manually** (its
> `"generator"` field says so) for the noise-filter section, not by this script.
> Do not expect `--dataset lab_131204` to work.

### 1.3 Reading the Layer 2 facts (output schema)

Each `data/corpus_facts/<dataset>.json` is a versioned artifact (commit it). The two
schema variants seen in the repo:

**Wild-dyad schema** (`5970.json`, `3452.json`, `9252.json`) — keys and meaning,
read from `data/corpus_facts/5970.json`:

| Top-level key | Field | Meaning |
|---|---|---|
| `dataset` | — | Dataset id (e.g. `"5970"`). |
| `generated_at_utc` | — | ISO-8601 generation timestamp. |
| `generator` | — | `"scripts/audit_corpus.py"` for the three wild dyads. |
| `sources` | `classified_csv`, `hdbscan_csv`, `detection_csv`, `ici_*_npy`, `sequential_summary_csv`, `sis_baselines_csv` | Exact input paths every stat was derived from (traceability). |
| `counts` | `n_calls_raw`, `n_calls_after_dropna_file`, `n_files`, `n_sessions` | Corpus size before/after dropping rows with no file id. |
| `timing` | `median_ici_gap_ms`, `median_ioi_ms`, `median_call_duration_ms`, `q25_ici_gap_ms`, `q75_ici_gap_ms`, `n_cross_file_pairs_over_10s`, `n_negative_gaps`, `n_ici_samples` | ICI = inter-call interval (silent gap, onset−prev_offset). IOI = inter-onset interval (onset−prev_onset). All in ms. |
| `bout_detection_a2` | `threshold_s`, `derivation`, `n_bouts`, `n_within_bout_pairs`, `n_cross_bout_pairs_excluded` | Bout = run of calls separated by < `threshold_s`. Threshold derived as 3 × median(IOI). |
| `sequential_structure_mi` | `scattoni_7_bout_aware.mi_lag1_bits`, `marginal_entropy_bits`, `conditional_entropy_bits`, `n_pairs`, `method`, `canonical_for_downstream` | Lag-1 mutual information (bits) between successive Scattoni-7 call types, bout-aware. `canonical_for_downstream: true` marks the entry you should cite. |
| `labeling_distributions` | `scattoni_7`, `hdbscan` | Per-type call counts for the 7-class taxonomy and for HDBSCAN clusters. |
| `references` | `median_within_bout_silent_gap_ms`, `inter_bout_threshold_ms_range`, `hertz_2020_*_sis_bits` | Literature anchors (not measured here) for sanity comparison. |

**Lab schema** (`lab_131204.json`) is different — it carries a `noise_filter`
block (algorithm `v2_per_bin_floor_excess`, `contrast_db: 3.0`, `reject_fraction:
0.5`) and validated-chunk records rather than timing/MI. Read
`data/corpus_facts/lab_131204.json` directly before relying on it; it is the
exception, not the template.

#### The measured numbers (current files, for orientation)

These are *Layer 2 empirical facts* — they change as the datasets evolve. Read the
JSON for the live value; this table is a snapshot so you know what "normal" looks like.

| Fact | 5970 | 3452 | 9252 | Source field |
|---|---|---|---|---|
| `n_calls_raw` | 7921 | 401 | 604 | `counts.n_calls_raw` |
| `n_files` | 1338 | 110 | 318 | `counts.n_files` |
| median ICI gap (ms) | 86.68 | 171.92 | 1586.71 | `timing.median_ici_gap_ms` |
| median IOI (ms) | 192.99 | 212.26 | 1625.57 | `timing.median_ioi_ms` |
| median call duration (ms) | 60.12 | 17.10 | 22.94 | `timing.median_call_duration_ms` |
| bout threshold (s) | 0.6 | 0.6 | 0.6 | `bout_detection_a2.threshold_s` |
| n_bouts | 1238 | 73 | 104 | `bout_detection_a2.n_bouts` |
| lag-1 MI (bits, bout-aware) | 0.0921 | 0.1974 | 0.1941 | `sequential_structure_mi.scattoni_7_bout_aware.mi_lag1_bits` |

> **Do not over-read cross-dataset differences.** 5970, 3452, 9252 are all *wild
> dyads*, and several of these quantities (e.g. mean power, tonality, and arguably
> ICI on small-N sets like 3452/9252) are cage/recording-environment artifacts, not
> biology. The 9252 ICI of ~1.6 s is dominated by long cross-file gaps (note
> `q75_ici_gap_ms` ≈ 176 s and `n_cross_file_pairs_over_10s = 237`). Cite the
> population stratum when comparing.

### 1.4 Troubleshooting / Gotchas

- **`ModuleNotFoundError: No module named 'usv_spectrogram'`** — the package lives
  under `src/`. Either run with `PYTHONPATH=src .venv/bin/python ...`, or rely on
  the installed editable package if the venv has it. Scripts under `scripts/` add
  the repo root / `src` to the path themselves; ad-hoc `-c` one-liners do not.
- **An import suddenly fails with an `AssertionError` mentioning "drifted from
  corpus"** — this is the **drift guard** in
  `src/usv_spectrogram/detection/extraction_config.py:148-162` firing, not a bug.
  Someone changed a `corpus.py` constant without updating the matching
  `ExtractionConfig` literal (or vice versa). See
  [CNN freeze](#the-cnn-freeze-the-one-rule-you-must-not-break) for the correct
  change sequence. Do **not** "fix" it by editing the assertion.
- **`error: --output is required with --dataset`** — pass `--output <path>` when
  using `--dataset`; only `--all` infers the path (`audit_corpus.py:558`).
- **`--dataset lab_131204` rejected** — `lab_131204` is not in `DATASET_REGISTRY`;
  its facts JSON was hand-authored. Only `3452`, `5970`, `9252` are script-driven.
- **`--all` silently skipped a dataset** — `--all` *warns* rather than fails when a
  dataset's input CSVs/NPYs are missing. Check stderr; a missing
  `results/...` input means that dataset's pipeline hasn't been run yet.
- **Wrong frequency mapping in a spectrogram you generated** — confirm you used the
  detection/analysis STFT (`n_fft=512`), not the *visualization* STFT.
  `SpectrogramConfig` deliberately uses different parameters (n_fft=2048, zero-pad
  to 4096, ~61 Hz/bin) for display — see ADR-002. Mixing the two grids produces a
  spectrogram the CNN cannot read.
- **You added a new config module and re-typed `sample_rate=300000`** — don't.
  Import from `corpus`. Re-declaring is exactly the bug this module was built to
  kill (four modules previously disagreed; one used 250 kHz — see Internals §2.1).

---

## 2. Internals

### 2.1 Why this module exists (the bug it killed)

Before the 2026-04-17 refactor, four config classes each declared the same physical
constants with **different** values (from `docs/modules/corpus-constants.md:14-19`):

| Module | Sample rate | freq_min | freq_max |
|---|---|---|---|
| `SpectrogramConfig` | 250_000 | 30_000 | 125_000 |
| `DetectionConfig` | 300_000 | 25_000 | 110_000 |
| `ExtractionConfig` | 300_000 | 20_000 | 120_000 |
| `AnalysisConfig` | n/a | 20_000 | 120_000 |

`SpectrogramConfig`'s 250 kHz contradicted ADR-001 (300 kHz): any
`SpectrogramConfig()` loaded against a real 300 kHz WAV would raise a sample-rate
mismatch. The refactor collapsed all four onto the single canonical set in
`corpus.py` and added a *drift assertion* so they can never silently diverge again.

### 2.2 Three-layer architecture

```
Layer 1 — CORPUS CONSTANTS (physical facts)
  src/usv_spectrogram/corpus.py
  SAMPLE_RATE_HZ, USV_FREQ_{MIN,MAX}_HZ, STFT_N_FFT, STFT_HOP + derived helpers

Layer 2 — EMPIRICAL DATA (per dataset)
  data/corpus_facts/{5970,3452,9252,lab_131204}.json
  Generated by scripts/audit_corpus.py (lab_131204 hand-authored)

Layer 3 — ANALYSIS PARAMETERS (per-module)
  SpectrogramConfig / DetectionConfig / ExtractionConfig / AnalysisConfig
  Import Layer 1 constants instead of redeclaring them.
```

Layer 1 = facts of physics and the CNN grid (rarely change). Layer 2 = facts of a
*dataset* (change as data evolves). Layer 3 = knobs of an *algorithm* (tuned freely,
but borrow their physical constants from Layer 1).

### 2.3 Key signatures (file:line)

`src/usv_spectrogram/corpus.py`:

| Symbol | Line | Definition |
|---|---|---|
| `SAMPLE_RATE_HZ: Final[int]` | `30` | `= 300_000` |
| `USV_FREQ_MIN_HZ: Final[int]` | `32` | `= 20_000` |
| `USV_FREQ_MAX_HZ: Final[int]` | `33` | `= 120_000` |
| `STFT_N_FFT: Final[int]` | `35` | `= 512` |
| `STFT_HOP: Final[int]` | `36` | `= 128` |
| `nyquist_hz() -> int` | `39-40` | `SAMPLE_RATE_HZ // 2` |
| `stft_freq_resolution_hz() -> float` | `43-44` | `SAMPLE_RATE_HZ / STFT_N_FFT` |
| `stft_time_resolution_ms() -> float` | `47-48` | `(STFT_N_FFT / SAMPLE_RATE_HZ) * 1000.0` |
| `stft_hop_ms() -> float` | `51-52` | `(STFT_HOP / SAMPLE_RATE_HZ) * 1000.0` |

The constants are typed `Final` (`from typing import Final`, `corpus.py:28`) — a
static-analysis signal that they are not to be reassigned. (Note: mypy is not
configured in this repo, so `Final` is documentation-grade, not enforced at CI.)

### 2.4 The CNN freeze (the one rule you must not break)

The production CNN `models/hard_neg_retrain/best_model.pt` was trained on
256-pixel-tall spectrograms covering **exactly 20–120 kHz**. The pixel grid is
fixed at training time; the Hz-per-pixel mapping is implied by the band edges.

If you change `USV_FREQ_MIN_HZ` or `USV_FREQ_MAX_HZ` in `corpus.py` **without
retraining**, every spectrogram keeps the same 256 rows but each row now means a
*different* frequency. The CNN sees features in the wrong rows and produces
**silently-wrong inference** — no crash, no error, just degraded/garbage detections.
This is called out at `corpus.py:15-23` and is one of the Red-Flag locked changes
in `CLAUDE.md`.

#### The drift guard (mechanism)

`ExtractionConfig` (in `src/usv_spectrogram/detection/extraction_config.py`)
**intentionally hardcodes** the CNN-training literals — it does *not* import them
from `corpus.py` (`extraction_config.py:36-37` for the band;
`extraction_config.py:30-32` for sr/n_fft/hop). At the bottom of the file, a
module-level block late-imports the corpus values under aliases
(`extraction_config.py:139-144`) and asserts every literal still matches
(`extraction_config.py:148-162`):

```python
assert _FIELDS["freq_min_hz"].default == _CORPUS_USV_FREQ_MIN_HZ, (...)
assert _FIELDS["freq_max_hz"].default == _CORPUS_USV_FREQ_MAX_HZ, (...)
assert _FIELDS["sample_rate"].default == _CORPUS_SAMPLE_RATE_HZ, (...)
assert _FIELDS["n_fft"].default       == _CORPUS_STFT_N_FFT,     (...)
assert _FIELDS["hop_length"].default  == _CORPUS_STFT_HOP,       (...)
```

So if `corpus.py` and the CNN-grid literals ever disagree, **importing
`ExtractionConfig` fails at import time** — a loud failure, by design, in place of
silent inference corruption.

#### Correct change sequence (never reverse it)

If the corpus band genuinely must change:

1. **Retrain** the CNN on the new band.
2. **Update `ExtractionConfig`** literals (`extraction_config.py:36-37`) to match.
3. **Update `corpus.py`** constants (`corpus.py:32-33`).
4. The drift assertion passes on the next import.

Doing 3 before 1–2 is exactly the failure mode the guard exists to catch.

### 2.5 Invariants

- **Always specify `sr=300000` explicitly** to any DSP call (or pass
  `SAMPLE_RATE_HZ`). Never rely on librosa's default 22050 Hz — at the wrong sample
  rate, every frequency and duration is wrong by a factor of `300000/default`.
  (ADR-001; `CLAUDE.md` "Signal Processing Conventions".)
- **Never redeclare** `sample_rate`, `freq_min_hz`, `freq_max_hz`, `n_fft`, or
  `hop_length` in a new module. Import from `corpus`. The single exception is
  `ExtractionConfig`, which hardcodes them *on purpose* and is policed by the drift
  assertion.
- **`corpus.py` constants are LOCKED to the CNN.** Band edges cannot move without a
  retrain (§2.4).
- **Layer 2 JSONs are versioned artifacts** — regenerate with `audit_corpus.py` and
  commit; don't hand-edit the script-generated ones (`5970/3452/9252`). The drift
  check inside `5970.json` (`scattoni_7_bout_aware` vs deprecated `scattoni_7_sis_17_1`
  must stay within 1e-3 bits) exists to catch silent pipeline changes.
- **The visualization STFT is deliberately different.** `SpectrogramConfig` uses
  n_fft=2048 / zero-pad 4096 (~61 Hz/bin) for display only (ADR-002 note). It is
  *not* governed by `STFT_N_FFT`; do not "unify" them.

### 2.6 Where to change things

| You want to... | Edit | Then |
|---|---|---|
| Change sample rate or STFT params globally | `src/usv_spectrogram/corpus.py:30-36` | Update `ExtractionConfig` literals to match (drift assert) + review every Layer-3 config. Sample-rate change implies re-rendering all training data. |
| Change the analyzed band (20–120 kHz) | `corpus.py:32-33` | **Retrain CNN first** (§2.4), then `extraction_config.py:36-37`, then `corpus.py`. |
| Add a derived quantity | `corpus.py` (new helper fn, keep the formula visible) | — |
| Register a new dataset for Layer 2 | `DATASET_REGISTRY` in `scripts/audit_corpus.py:68` | Run `audit_corpus.py --dataset <name> --output data/corpus_facts/<name>.json`, commit the JSON. |
| Tune a per-algorithm knob | the relevant Layer-3 config, **not** here | Borrow physical constants from `corpus`. |

### 2.7 ADR references

- **ADR-001 (Sample Rate — 300 kHz):** `docs/human/DECISIONS.md:11`. Status:
  Accepted. Rationale: USVs reach ~120 kHz; Nyquist needs ≥240 kHz; hardware runs at
  300 kHz (Nyquist 150 kHz) for headroom. Mandates `sr=300000` everywhere and never
  trusting librosa defaults. Notes the legacy 250 kHz `SpectrogramConfig` value as
  outdated.
- **ADR-002 (STFT Parameters):** `docs/human/DECISIONS.md:38`. Status: Accepted.
  n_fft=512, hop=128, Hann; band 20–25k to 110–120k (detection historically used
  25k–110k, VQ-VAE/analysis 20k–120k — the corpus now standardizes on 20k–120k).
  Derived: frame 1.707 ms, hop 0.427 ms, 585.9 Hz/bin, ~171 bins in 20–120 kHz.
  Explicitly states the visualization STFT (n_fft=2048, ~61 Hz/bin) is intentionally
  different.

> Historical note on the band: ADR-002 records that the *detection* pipeline once
> used 25–110 kHz while VQ-VAE used 20–120 kHz. The 2026-04-17 unification settled
> the corpus on **20–120 kHz** to match the production CNN's training grid. If you
> read an older module citing 25–110 kHz, treat `corpus.py` as the truth.

---

## Related docs

- [CNN classifier / detection pipeline](../modules/cnn-classifier.md) — consumes
  these constants; the model whose grid locks the band.
- [Cleaning subsystems](../modules/cleaning-subsystems.md) — references the STFT grid.
- [SIS baselines](../modules/sis-baselines.md) — consumes the Layer 2 MI facts.
- `docs/modules/corpus-constants.md` — the original module doc (also covers the
  `data/lab_tonal_lines/<rig_id>.json` per-rig tonal libraries in detail).
- `docs/human/DECISIONS.md` — ADR-001 (sample rate), ADR-002 (STFT), ADR-003
  (detection thresholds).
- `docs/handoffs/corpus-constants-unification-2026-04-17.md` — the refactor spec.

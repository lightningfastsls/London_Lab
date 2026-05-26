# Handoff: Adaptive Soft-Notch Pre-CNN Filter for Lab Detection (Hybrid Library + Drift Audit)

Date: 2026-05-11
From: Claude Code (this chat)
To: Claude Code (implementing chat)
Status: SPEC — not yet implemented

## Resolved Decisions (locked before implementation)

These decisions were agreed during spec discussion and should not be
reopened by the implementing chat without going back to the user:

| Decision | Outcome |
|---|---|
| Filter design | Butterworth band-stop + zero-phase `sosfiltfilt` (IIR with linear-phase result via forward+backward) |
| Cut formulation | Complementary-bandpass subtraction: `audio - α * bandpass(audio)` where `α = 1 - 10^(-cut_db/20)` |
| Cut depth source | Always measured per-chunk locally (peak − local median + safety margin), never from library |
| Width / frequency source | From library when in library mode; from per-chunk discovery in audit / pure auto-detect mode |
| Architecture mode | Hybrid: library-mode primary + per-chunk audit secondary (no auto-filter of unmatched audit detections — log-only) |
| Default-off discipline | `--soft-notch` defaults off; wild-mouse runs must produce byte-identical output without it |
| Auto-discovery in standalone script | Yes — `notch_filter_wav.py` auto-detects bands; no `--center` / `--width` flags needed (optional `--manual-band` override retained) |
| Layer of new fact | Layer-2 (rig-specific empirical) at `data/lab_tonal_lines/<rig_id>.json`, with full audit wiring |
| Rig-id naming convention | `<rig_id>.json` where the calibration script's default heuristic strips leading `USV_` and trailing `_chunked_*` from the WAV directory name. Override via `--rig-id`. |
| Per-batch persistence | Two sidecar files in `<output-dir>/`: `soft_notch_applied.parquet` (one row per applied filter or audit detection) + `soft_notch_summary.json` (batch-level metadata) |

## Goal

Replace the current frequency-blind, hand-tuned hard band-stop in
`scripts/notch_filter_wav.py` with an **adaptive soft-notch** that runs
automatically inside batch detection. The filter operates in a **hybrid
two-mode architecture**:

- **Primary (library) mode:** at batch start, load a per-rig **tonal
  library** (`data/lab_tonal_lines/<rig_id>.json`, a Layer-2 empirical
  fact). For every WAV chunk, apply soft-notches at the library's
  frequencies, widths, and depths. Fast, deterministic, reviewable.
- **Drift audit:** for each chunk, *also* run the per-chunk PSD-peak
  detector. Reconcile its detections against the library. **Unmatched**
  detections are logged but NOT filtered by default — the library is
  the source of truth and the audit is purely a stale-library alarm.
- **Library generation:** a new calibration script
  (`scripts/calibrate_lab_tonal_lines.py`) samples chunks from a rig,
  aggregates the per-chunk detector's findings across chunks, and
  writes the library JSON. Run once per rig / recording session /
  equipment-change.

The library mode embodies an empirical fact ("rig X has tonals at A, B,
C with widths W and depths D"); the audit mode catches the case where
that fact has gone stale.

Each individual soft-notch:

1. Measures the tonal's actual center frequency, −3 dB width, and
   amplitude above the local PSD median (during calibration), OR reads
   them from the library (at batch detection time).
2. Attenuates the band by a calibrated, per-tonal depth — just enough
   to bring the tonal down to the surrounding median level, no further.
   The filter never drives a band below the natural baseline, so any
   USV that happens to cross the band sees finite attenuation rather
   than total erasure.

The standalone CLI (`scripts/notch_filter_wav.py`) is rewritten as a thin
wrapper around the same core. It defaults to per-chunk auto-detect mode
(no library required) for single-file experiments and probe runs, with
an optional `--library` flag for testing library entries against a
specific WAV.

**Acceptance:**

- A tonal library `data/lab_tonal_lines/lab_131204.json` is produced by
  `scripts/calibrate_lab_tonal_lines.py` and contains ≥1 entry near
  51.09 kHz with `detection_rate ≥ 0.95` (the tonal is recurrent across
  the sampled chunks). Schema validates against the JSON schema added
  to the calibration script's docstring.
- Lab batch detection re-run on `USV_lab_131204_chunked_2s_hot/` with
  `--soft-notch data/lab_tonal_lines/lab_131204.json` enabled shows
  ≥30% reduction in `auto_accept` noise rate vs the same run without
  `--soft-notch`.
- Wild-mouse batches (5970, 3452, 9252) re-run with `--soft-notch` off
  produce byte-identical results to the current production. Same default-
  off discipline as `--subtract-baseline`.
- USV recall on a held-out lab set (TBD by user) does not regress more
  than 2% relative.
- Drift audit fires (logs a "library may be stale" warning) when the
  library is run against a deliberately-mismatched recording: e.g.,
  applying `lab_131204.json` to a synthetic WAV with an injected tonal
  at 73 kHz. Unit test #8 below.

## Motivation

### Evidence from this session

Probing `USV_lab_131204_chunked_2s_hot/131204_1400_m3fm3_chunk_243.wav`
(fs=300 kHz, 2.00 s, FLOAT subtype) yielded:

| Tonal | Center | −3 dB width | Peak PSD | Above median |
|---|---|---|---|---|
| #1 (known) | 51.09 kHz | 110 Hz | −91.7 dB | +15.9 dB |
| #2 (newly discovered) | 46.58 kHz | 40 Hz | similar | similar |

USV-band median (20–120 kHz, excluding ±1.5 kHz around the peak): **−107.6 dB**.

The current hand-tuned hard band-stop in `scripts/notch_filter_wav.py`
applied `--center 50000 --width 5000`, which:

- Was **~50× wider than the actual #1 tonal** (5000 Hz of kill zone vs
  110 Hz of real tonal). The "extra" 4.9 kHz was innocent bystander
  spectrum that any USV crossing the band lost completely.
- **Missed tonal #2 at 46.58 kHz entirely.** The notch passband 47.5–52.5 kHz
  did not include it. Equipment harmonics are common, single-band designs
  are fragile.
- Drove the band to roughly −∞ dB (effectively −90 dB) when the tonal
  only needed to come down ~16 dB to stop dominating the spectrogram.

### Why "hard cut at fixed width and depth" is the wrong model

A hard band-stop is *content-blind*: it removes the band's energy
unconditionally, regardless of whether that energy is an equipment line
or a USV harmonic. For wild-mouse data this is acceptable because the
50 kHz region is dominated by tonal noise. For lab data the cost is the
loss of any real USV that sweeps through the band.

### Why "adaptive soft cut" works

The cut depth at each band should be exactly the amount that brings the
band's stationary content down to "looks like the rest of the spectrum
here." For a tonal that sits +16 dB above the local median, that's a 16 dB
cut. For a tonal that sits +30 dB above, it's a 30 dB cut. The depth is
data-driven, never hand-tuned.

Because the cut is *finite*, a USV crossing the band loses the same dB
that the tonal does (so e.g. 16 dB) but is not erased. Detectability is
preserved.

Because the width is also data-driven (measured −3 dB width × small
expansion margin), the kill zone is the tonal and nothing else.

## Design

### Two-mode architecture

```
                   ┌─────────────────────────────┐
                   │  data/lab_tonal_lines/      │
                   │     <rig_id>.json           │  ← built by
                   │  (Layer-2 corpus fact)      │   calibrate_lab_tonal_lines.py
                   └────────────┬────────────────┘
                                │  load at batch start
                                ▼
        WAV ──► AudioLoader ──► apply_soft_notches(audio, library) ──► STFT ──► CNN
                    │                       ▲
                    │                       │  filter using library entries (FAST PATH)
                    │
                    └──► discover_tonals(audio) ──► reconcile(library, detections)
                                                          │
                                                          ├──► matched: noise audit metadata
                                                          └──► unmatched: WARN "library may be stale"
                                                               (NOT auto-filtered by default)
```

### Library entry semantics

A library entry is a deliberate, calibrated statement: "rig X always
emits a tonal at center `c_hz`, with width `w_hz`, that sits roughly
`d_db` above the local PSD median." When the library is applied:

- `center_hz`, `width_hz`: taken from the library directly. No
  per-chunk re-measurement (this is the speed win).
- `cut_depth_db`: each chunk locally measures the band's actual peak
  PSD and local median, then computes
  `cut_depth_db = peak_db - local_median_db + safety_margin_db`.
  Library carries `mean_above_median_db` as a *sanity reference*: if a
  chunk measures `> 2σ` away from the library mean, the audit logs a
  "tonal intensity drift" warning. The filter still applies; the
  warning is informational.

Rationale: tonal *frequency* and *width* are rig-physical and stable;
tonal *intensity* varies chunk-to-chunk (e.g., gain settings, ambient
loading). Per-chunk depth measurement keeps the cut amount adaptive to
the moment, while frequency/width come from the deterministic library.

### Algorithm: per-WAV adaptive soft-notch

Given a freshly loaded WAV `audio` at sample rate `fs` and a loaded
library `tonal_library` (may be empty for pure auto-detect mode):

**Library-mode steps** (fast path, default when library is supplied):

L1. For each library entry `(center_hz, width_hz, mean_above_median_db)`:
    measure the local PSD median in a 4 kHz neighborhood (excluding
    the band itself) and the band's peak PSD. Compute
    `cut_depth_db = peak_db - local_median_db + safety_margin_db`.
L2. Apply the soft-notch (see step 5 below) using the library's
    `center_hz`, `width_hz`, and the per-chunk computed `cut_depth_db`.
L3. If `abs(peak_db - local_median_db - mean_above_median_db) > intensity_drift_db`
    (default 6 dB ≈ 2σ for stable rigs), log a "tonal intensity drift"
    warning for this chunk and tonal.

**Audit steps** (always run when library mode is active; primary path
when no library is supplied):

A1. **PSD estimate**
   ```
   f, pxx = scipy.signal.welch(audio, fs=fs, nperseg=8192)
   pxx_db = 10 * log10(pxx + ε)
   ```
   `nperseg=8192` → ~37 Hz/bin at 300 kHz fs. Fine enough to resolve
   the 110 Hz tonals seen in lab data.

2. **Local-median baseline**
   For each frequency bin in the USV band (`corpus.USV_FREQ_MIN_HZ` to
   `corpus.USV_FREQ_MAX_HZ`), compute the rolling median of `pxx_db`
   over a frequency window wide enough to skip any individual tonal
   (default 4 kHz). This is the "what the spectrum would look like here
   if no tonal were present" estimate.

3. **Peak discovery**
   Find frequency bins where `pxx_db - rolling_median > discovery_threshold_db`
   (default 10 dB). Cluster contiguous bins into peak candidates. For
   each candidate, the center is the argmax bin and the −3 dB width is
   the contiguous span where PSD ≥ peak − 3 dB.

4. **Per-tonal cut parameters**
   For each detected tonal `t`:
   - `center_hz = peak frequency`
   - `width_hz = max(min_width_hz, measured_width_hz × width_safety_factor)`
     (default `min_width_hz=200`, `width_safety_factor=2.0` — covers the
     skirts and small per-recording drift)
   - `cut_depth_db = (peak_db - local_median_db) + safety_margin_db`
     (default `safety_margin_db=0` — bring the tonal exactly to median;
     positive values cut *below* median, negative values leave the tonal
     partially intact)

5. **Cascaded soft-notch**
   For each detected tonal, design a Butterworth band-pass at
   `[center − width/2, center + width/2]` order 4 (effective order 8
   via `sosfiltfilt`). Then:
   ```
   alpha = 1.0 - 10**(-cut_depth_db / 20.0)
   audio = audio - alpha * sosfiltfilt(bandpass, audio)
   ```
   `alpha=1` reduces to the hard band-stop (when `cut_depth_db = ∞`).
   `alpha=0.9` = −20 dB cut. Cascade through all detected tonals in
   sequence. Order does not matter mathematically (linear operations) but
   we apply lowest-frequency first for predictable logging.

6. **Pass to downstream pipeline**
   The cleaned audio array replaces the original input to STFT / CNN
   inference. No downstream change required — same array shape and dtype.

### Why complementary-bandpass subtraction (not iirpeak / RBJ biquad)

The complementary-bandpass formulation reuses the existing Butterworth
band-pass machinery and is mathematically identical to a parametric EQ
cut. RBJ peaking-EQ biquads would give a slightly more compact filter
but introduce a different design path with its own gain-vs-Q
calibration. Reusing the same `scipy.signal.butter` + `sosfiltfilt` path
keeps the code surface small and lets us share validation utilities with
the existing batch pipeline.

### Why local-median baseline (not file-wide max or percentile)

The user's first instinct — "find the max dB in the wav file" — would
work if equipment tonals were *always* the file maximum. In practice a
loud USV cluster can be brighter than the equipment line. Anchoring to
the *local* PSD median (a frequency neighborhood of the candidate tonal)
gives a stable reference that depends only on the surrounding noise
floor, not on whatever transients happened to fire during the chunk.

### Why this fits the existing pipeline

This is the time-domain analogue of the `--subtract-baseline` feature
already in `audio_loader.py` (see
`docs/handoffs/2026-05-08_pre-cnn-spectral-subtraction-lab.md` and
`src/usv_spectrogram/app/core/denoise.py`). Both target stationary
equipment tonals; both are default-off lab opt-ins; both run before the
CNN sees the signal.

The difference: `--subtract-baseline` operates on the *spectrogram*
fed to the CNN, after STFT. The adaptive soft-notch operates on the
*audio* before STFT, so its output can also be written to a WAV file
for inspection / external tooling. The two are not mutually exclusive
and can stack.

## Files to Modify

### 1. `src/usv_spectrogram/app/core/notch.py` *(new)*

Pure-NumPy/SciPy module. No project imports beyond `corpus.py` for
USV-band constants. Public surface:

```python
@dataclass(frozen=True)
class DetectedTonal:
    """A tonal measured from a single PSD (audit-mode output)."""
    center_hz: float
    width_hz: float            # measured -3 dB width
    peak_db: float
    local_median_db: float
    above_median_db: float     # peak_db - local_median_db

@dataclass(frozen=True)
class LibraryEntry:
    """A calibrated tonal that should always be filtered for this rig."""
    center_hz: float
    width_hz: float
    mean_above_median_db: float
    stdev_above_median_db: float
    n_chunks_seen: int
    detection_rate: float      # fraction of calibration chunks where seen

@dataclass(frozen=True)
class TonalLibrary:
    rig_id: str
    calibrated_at: str          # ISO 8601
    n_chunks_sampled: int
    sample_files: list[str]
    entries: list[LibraryEntry]

    @classmethod
    def load(cls, path: Path) -> "TonalLibrary": ...
    def save(self, path: Path) -> None: ...

@dataclass(frozen=True)
class ReconciliationResult:
    matched: list[tuple[LibraryEntry, DetectedTonal]]
    unmatched_detections: list[DetectedTonal]      # in audit but not library
    unmatched_library_entries: list[LibraryEntry]  # expected but not seen
    intensity_drifts: list[tuple[LibraryEntry, float]]  # (entry, |measured - mean|/stdev)

def discover_tonals(
    audio: np.ndarray,
    fs_hz: float,
    *,
    usv_band_min_hz: float = USV_FREQ_MIN_HZ,
    usv_band_max_hz: float = USV_FREQ_MAX_HZ,
    discovery_threshold_db: float = 10.0,
    median_window_hz: float = 4_000.0,
    nperseg: int = 8192,
) -> list[DetectedTonal]: ...

def reconcile(
    library: TonalLibrary,
    detections: Sequence[DetectedTonal],
    *,
    freq_tolerance_hz: float = 200.0,
    intensity_drift_sigma: float = 2.0,
) -> ReconciliationResult: ...

def apply_soft_notches(
    audio: np.ndarray,
    fs_hz: float,
    tonals: Sequence[LibraryEntry | DetectedTonal],
    *,
    min_width_hz: float = 200.0,
    width_safety_factor: float = 2.0,
    safety_margin_db: float = 0.0,
    order: int = 4,
) -> np.ndarray: ...

def auto_soft_notch(
    audio: np.ndarray,
    fs_hz: float,
    library: TonalLibrary | None = None,
    **kwargs,
) -> tuple[np.ndarray, ReconciliationResult]:
    """Library-mode if library is provided; pure auto-detect otherwise.

    Returns (cleaned_audio, reconciliation_result). When library is None,
    reconciliation.matched is empty and all detected tonals are filtered
    (audit-only mode).
    """
```

`apply_soft_notches` returns audio with the *same dtype and shape* as
input (incl. multi-channel). When `tonals` mixes `LibraryEntry` and
`DetectedTonal` items, `LibraryEntry.width_hz` is preferred over
measured width for library entries (deterministic kill zone), but
`cut_depth_db` is always computed per-chunk from the local PSD.

### 1b. `scripts/calibrate_lab_tonal_lines.py` *(new)*

Standalone script that builds a `TonalLibrary` from a sample of WAV
chunks. Surface:

```
python scripts/calibrate_lab_tonal_lines.py \
    --wav-dir USV_lab_131204_chunked_2s_hot/ \
    --rig-id lab_131204 \
    --sample-size 50 \
    --min-detection-rate 0.5 \
    --output data/lab_tonal_lines/lab_131204.json
```

Algorithm:

1. Sample `--sample-size` chunks uniformly at random from `--wav-dir`.
2. For each chunk, run `discover_tonals`.
3. Cluster detections across chunks by frequency proximity (default
   tolerance 200 Hz — same as `reconcile` freq tolerance).
4. For each cluster: compute mean center, mean width, mean and stdev
   of `above_median_db`, count of chunks seen.
5. Keep clusters with `detection_rate >= --min-detection-rate` (default
   0.5 — a tonal must appear in at least half the sampled chunks to be
   library-worthy).
6. Write `TonalLibrary` JSON to `--output`.

Prints a human-readable summary at end (rig_id, n_sampled, n_entries,
top 5 entries by detection_rate).

### 1c. `data/lab_tonal_lines/` *(new directory + JSON files)*

Layer-2 empirical-fact directory. Schema documented in `TonalLibrary`
dataclass + reflected in the JSON file by the calibration script's
`save()` method. Each rig has its own file (`<rig_id>.json`).
Version-controlled. Naming convention: the calibration script's
default `rig_id` heuristic strips a leading `USV_` and a trailing
`_chunked_*` from the WAV directory name (e.g.,
`USV_lab_131204_chunked_2s_hot/` → `lab_131204`). The default can
always be overridden via `--rig-id`.

### 1f. Per-batch sidecar files in `<output-dir>/` *(new outputs from run_batch_detection.py)*

When `--soft-notch` is active, batch detection writes two sidecar
artifacts next to the existing detection parquet:

**`soft_notch_applied.parquet`** — one row per (recording, chunk,
applied_filter_or_audit_detection):

| Column | Type | Meaning |
|---|---|---|
| `recording_path` | str | Source WAV path |
| `chunk_idx` | int | Chunk index within the recording |
| `center_hz` | float | Tonal center frequency (Hz) |
| `width_hz` | float | Applied filter width or measured width (Hz) |
| `peak_db` | float | Measured PSD peak at tonal |
| `local_median_db` | float | Measured PSD median in the 4 kHz neighborhood |
| `cut_depth_db` | float | Actual cut applied (0 for unmatched audit detections) |
| `source` | str | `"library"` (applied) / `"audit"` (logged only) |
| `is_drift` | bool | True for audit detections that did not match any library entry |
| `intensity_drift_sigma` | float | abs(measured − library_mean) / library_stdev; NaN if `source != "library"` |

**`soft_notch_summary.json`** — single batch-level summary:

```json
{
  "library_path": "data/lab_tonal_lines/lab_131204.json",
  "library_rig_id": "lab_131204",
  "library_calibrated_at": "2026-05-12T09:14:33",
  "library_n_entries": 2,
  "batch_n_chunks": 8643,
  "n_chunks_with_unmatched": 47,
  "unmatched_rate": 0.0054,
  "stale_library_warning_fired": false,
  "stale_library_warning_reason": null
}
```

Rationale: persisting both files makes the soft-notch behavior fully
auditable after the fact. The parquet allows joining onto detection
results to ask "did this false positive happen on a chunk with a
drift event?"; the summary JSON gives a quick pass/fail signal per
batch without loading the parquet.

### 1d. `scripts/audit_corpus.py` *(extend)*

Add a `lab_tonal_lines` key that, when run, enumerates every JSON in
`data/lab_tonal_lines/` and verifies:

- JSON parses against the `TonalLibrary` schema.
- All entries have `center_hz` inside `[USV_FREQ_MIN_HZ, USV_FREQ_MAX_HZ]`.
- All entries have `width_hz` in `[20, 5000]` Hz (sanity bounds).
- All entries have `detection_rate ∈ [0, 1]`.
- File mtime is within 365 days (warns if older — calibration may be stale).

This is the Layer-2 wiring required by `docs/modules/corpus-constants.md`.

### 1e. `docs/modules/corpus-constants.md` *(extend)*

Add a section "Layer-2 fact: `data/lab_tonal_lines/<rig_id>.json`"
documenting:
- What the file contains (calibrated rig-specific tonal frequencies).
- How to regenerate it (`scripts/calibrate_lab_tonal_lines.py`).
- When to regenerate (equipment change, rig change, audit warning).
- Field-by-field schema reference to `notch.TonalLibrary`.

### 2. `src/usv_spectrogram/app/core/audio_loader.py`

Add `auto_soft_notch: bool = False` kwarg to `AudioLoader.__init__`,
mirroring the existing `subtract_baseline` flag. When True, call
`notch.auto_soft_notch(audio, fs)` *before* `_compute_spectrogram` and
*before* `_compute_sonic_spectrogram` so both spectrograms reflect the
cleaned audio. Capture the returned `list[DetectedTonal]` and surface
it via `AudioData` (new optional field `applied_tonals: list[DetectedTonal] | None`).

The flag must default to False so wild-mouse runs are byte-identical.

### 3. `scripts/run_batch_detection.py`

Add `--soft-notch` flag, lab-only opt-in (default off). The flag's
argument is **either** a path to a tonal library JSON **or** the
literal `auto`:

```
--soft-notch data/lab_tonal_lines/lab_131204.json   # library mode (preferred)
--soft-notch auto                                    # per-chunk auto-detect (fallback)
```

Pass through to `AudioLoader`. Log a single line per WAV summarizing
which library entries fired plus any audit findings, e.g.:

```
soft-notch: chunk_243.wav  lib=lab_131204 (2 entries)  applied=2 unmatched_detections=0
  [51.09 kHz: -15.9 dB, 200 Hz; 46.58 kHz: -12.3 dB, 200 Hz]
```

When audit fires:

```
soft-notch: chunk_500.wav  lib=lab_131204 (2 entries)  applied=2
  WARN unmatched detection at 73.2 kHz, +14 dB above median, 180 Hz width
       library may be stale — consider recalibrating
```

The `--soft-notch` and `--subtract-baseline` flags are independent and
may be combined. The pipeline applies `--soft-notch` first (time domain)
then `--subtract-baseline` (spectrogram domain) so both stationary
discrimination strategies stack.

### 4. `scripts/notch_filter_wav.py` *(REPLACE)*

Rewrite to use `notch.auto_soft_notch` as the engine. Surface:

```
python scripts/notch_filter_wav.py --input FILE [--probe]
    [--library data/lab_tonal_lines/<rig>.json]
    [--discovery-threshold-db 10.0]
    [--safety-margin-db 0.0]
    [--width-safety-factor 2.0]
    [--plot]
    [--suffix _notch]
```

- Default behavior (no `--library`): pure auto-detect mode. Discover
  tonals, filter, save `<stem>_notch.wav`, print summary line.
- `--library`: load the tonal library and apply library mode (same as
  batch detection). Audit findings are still printed. Useful for
  testing whether a library is suitable for a new WAV.
- `--probe`: discover and print + plot, but do **not** write a WAV.
  This is the answer to the "probe mode that lists everything" feature
  the user agreed to in spec discussion.
- `--plot`: save `<stem>_notch.png` PSD before/after with detected
  bands shaded. Library entries shaded blue; per-chunk detections
  shaded red.
- Removed flags: `--center` / `--width` / `--order` / `--cut-depth-db` /
  `--self-test`. The auto-detector and library replace them.
  *Override path*: if the user supplies an optional `--manual-band
  center,width[,depth_db]` argument (repeatable), those bands are added
  to the filter set before filtering. This preserves the ability to
  filter a tonal that the auto-detector misses (e.g., one buried in a
  noisy band that does not stick out +10 dB above the local median).
- Self-test moves to a pytest in `tests/test_notch.py` (see Validation
  Plan below) so it runs in CI.

### 5. `tests/test_notch.py` *(new)*

See Validation Plan section.

### 6. `docs/modules/recording-triage.md` *(update)* or new module doc

Document the `--soft-notch` flag, library generation workflow, and the
discovery + reconciliation algorithm. Cross-link to this handoff for
the design rationale.

## Parameters and Defaults

### Filter parameters (apply to both modes)

| Parameter | Default | Reasoning |
|---|---|---|
| `min_width_hz` | 200 Hz | Floor on the cut width. Even when a tonal measures as 110 Hz wide, expand to 200 Hz to cover skirts and small per-recording drift. |
| `width_safety_factor` | 2.0× | Multiplier on measured −3 dB width. Combined with `min_width_hz`, gives a kill zone that comfortably covers the tonal without bleeding. |
| `safety_margin_db` | 0.0 dB | Bring tonal exactly to local median. Positive values cut further (more aggressive); negative values leave tonal partially intact. |
| `order` | 4 | Butterworth order per band-pass. Effective order 8 via `sosfiltfilt`. Same as current `notch_filter_wav.py`. |
| `nperseg` | 8192 | Welch segment length. ~37 Hz/bin at 300 kHz — resolves 110 Hz tonals. |

### Audit-mode parameters (per-chunk discovery)

| Parameter | Default | Reasoning |
|---|---|---|
| `discovery_threshold_db` | 10.0 dB | High enough to ignore broadband USV bursts (which average <5 dB above local median), low enough to catch the 51.09 kHz tonal at +15.9 dB. |
| `median_window_hz` | 4 kHz | Wider than any plausible tonal, narrower than the USV band. Provides a smooth local-baseline estimate. |

### Library-mode parameters (reconciliation + calibration)

| Parameter | Default | Reasoning |
|---|---|---|
| `freq_tolerance_hz` (reconcile) | 200 Hz | A library entry and a per-chunk detection match if their centers are within this. Equal to `min_width_hz` so reconciliation tolerance matches the kill-zone width. |
| `intensity_drift_sigma` (reconcile) | 2.0σ | Per-chunk `above_median_db` more than 2σ away from library's `mean_above_median_db` triggers an intensity-drift warning. |
| `min_detection_rate` (calibration) | 0.5 | A tonal must appear in ≥50% of sampled chunks to be promoted to a library entry. Rejects transient noise sources. |
| `sample_size` (calibration) | 50 chunks | Sampling 50 × 2 s = 100 s of audio per rig. Empirically enough to estimate `mean_above_median_db` to ~1 dB stdev for stable equipment lines. Override for noisier rigs. |
| `cluster_tolerance_hz` (calibration) | 200 Hz | Detections within this frequency span across chunks are clustered as the same tonal. Equal to `freq_tolerance_hz`. |

### Stale-library warning (run_batch_detection only)

| Parameter | Default | Reasoning |
|---|---|---|
| `unmatched_detection_rate_warning` | 0.10 (10%) | If more than 10% of chunks in a batch produce an unmatched audit detection at consistent frequency, emit a "library may be stale" warning. Single chunks with noise spikes don't trigger; persistent new tonals do. |

All defaults are configurable via the CLI for tuning and via the
`AudioLoader` constructor / `auto_soft_notch` kwargs for programmatic
use. **All defaults must be re-validated empirically on real lab data
before declaring the feature stable.**

## Corpus Invariance

### Layer-1 (physical constants) — consumption only

This change does not introduce new physical constants. It consumes:

- `corpus.USV_FREQ_MIN_HZ`, `corpus.USV_FREQ_MAX_HZ` for the scan range
  in `notch.discover_tonals` and as validation bounds in
  `scripts/audit_corpus.py`.
- The WAV file's own `fs` at runtime (not redeclared anywhere).

### Layer-2 (empirical fact) — NEW: tonal libraries

This change DOES introduce a new Layer-2 empirical-fact category:
`data/lab_tonal_lines/<rig_id>.json`. Per `docs/modules/corpus-constants.md`,
Layer-2 facts require:

1. ✅ **Audit-script wiring:** `scripts/audit_corpus.py` extended with
   a `lab_tonal_lines` key (see "Files to Modify" §1d).
2. ✅ **Schema documentation:** `docs/modules/corpus-constants.md`
   extended with a section describing the file format and regeneration
   workflow (see "Files to Modify" §1e).
3. ✅ **Versioned in git:** files live under `data/` and are committed,
   so calibration history is reviewable via `git blame` and rig-specific
   changes are visible in PR diffs.
4. ✅ **Reproducible generator:** `scripts/calibrate_lab_tonal_lines.py`
   is the canonical regeneration path. The JSON includes
   `calibrated_at` (ISO 8601), `n_chunks_sampled`, and `sample_files`
   so runs are reproducible.

### Operational tuning parameters — NOT corpus facts

Filter parameters (`discovery_threshold_db`, `min_width_hz`,
`freq_tolerance_hz`, `unmatched_detection_rate_warning`, etc.) are
operational tuning knobs, not corpus facts. They belong in CLI args /
function defaults, not in `corpus.py` or `data/corpus_facts/`. They may
change as the algorithm is empirically tuned; corpus facts may not.

### CNN-input warning

This feature modifies the audio that the CNN sees. It MUST stay
default-off so wild-mouse runs remain identical. Enabling it for
wild-mouse data without retraining is a silent distribution shift. The
lab use case is acceptable only because the lab CNN was never trained
on the equipment tonals being suppressed — i.e., removing them brings
lab audio *closer* to the wild-data distribution the CNN was trained
on.

## Validation Plan

### Unit tests (`tests/test_notch.py`)

**Discovery + filtering (pure auto-detect mode):**

1. **Two-tone synthetic survives.** Generate 30 kHz + 50 kHz sines at
   equal amplitude. Run `auto_soft_notch` with `library=None`. Assert:
   30 kHz tone survives within 1 dB of original power; 50 kHz tone
   reduced by at least 10 dB (the discovery threshold).
2. **Pure noise → no tonals detected.** White noise input. Assert:
   `discover_tonals` returns empty list. Audio passes through unchanged
   to within float precision.
3. **USV-burst not flagged as tonal.** Synthesize a 50 ms FM sweep that
   crosses 51 kHz. Assert: not flagged as a tonal (because it's
   transient, time-averaged PSD is low).
4. **Tonal-on-USV preserves USV.** Mix a USV-shaped FM sweep with a
   continuous 51 kHz tone. Assert: detected tonal at 51 kHz; cut depth
   matches `peak − median`; the USV sweep's time-domain envelope
   survives with ≥ −cut_depth_db reduction at the crossing point but
   not full erasure.
5. **Multi-tonal cascade.** Two tones at 46 and 51 kHz. Assert: both
   detected, both cut.
6. **Self-consistency on the reference WAV.** Load
   `USV_lab_131204_chunked_2s_hot/131204_1400_m3fm3_chunk_243.wav`,
   run `auto_soft_notch` with `library=None`, assert ≥1 tonal detected
   at 51.09 ± 0.2 kHz with measured `above_median_db` between 13 and
   18 (allowing for PSD estimator variance). This becomes a regression
   test against future algorithm changes.
7. **Wild-data byte-equivalence flag check.** Mock a wild-mouse
   `AudioLoader.load` call with `auto_soft_notch=False` (default).
   Assert no modification to audio array vs the same call without the
   notch module imported. Catches accidental coupling of the new code
   path to the default-off code path.

**Library mode + reconciliation:**

8. **Library hits → matched.** Build a `TonalLibrary` with one entry
   at 51 kHz. Apply to a synthetic two-tone (30 + 51 kHz). Assert:
   reconciliation result has 1 `matched`, 0 `unmatched_detections`,
   0 `unmatched_library_entries`. 51 kHz tone filtered; 30 kHz tone
   survives.
9. **Library miss → `unmatched_detection` warning.** Library has an
   entry at 51 kHz; signal contains tones at 51 kHz AND 73 kHz (drift).
   Assert: 51 kHz matched; 73 kHz appears in `unmatched_detections`.
   The 73 kHz tone is **not** auto-filtered (library is source of
   truth). 51 kHz IS filtered.
10. **Library expects, signal lacks → `unmatched_library_entries`.**
    Library has entries at 51 kHz and 46 kHz; signal contains only
    51 kHz. Assert: 46 kHz appears in `unmatched_library_entries`.
    No crash; 51 kHz still filtered.
11. **Intensity drift triggers warning.** Library entry has
    `mean_above_median_db=15`, `stdev=1`. Signal contains the tonal
    at `+22 dB` above median. Assert: matched but with
    `intensity_drifts` non-empty (|22 − 15| / 1 = 7σ ≫ 2σ).
12. **TonalLibrary round-trip.** Build a library, save to a temp JSON,
    reload. Assert all fields equal. Catches JSON schema regressions.
13. **TonalLibrary schema validation.** Hand-write a malformed JSON
    (missing `entries`, or `detection_rate > 1.0`). Assert `load`
    raises a clear error.

**Calibration script integration:**

14. **End-to-end calibration on synthetic chunks.** Generate 20
    synthetic 2 s chunks each with a 51 kHz tone of varying intensity
    plus random USV-like bursts. Run `calibrate_lab_tonal_lines.py`
    against this directory. Assert: output JSON has exactly 1 entry
    at 51 kHz ± tolerance, `detection_rate == 1.0`,
    `mean_above_median_db` close to ground truth.
15. **Calibration rejects sporadic tonals.** Same as #14 but inject
    a 73 kHz tone only into 2 of 20 chunks. Assert: 73 kHz NOT
    promoted to library (detection_rate=0.10 < 0.5 default).

### Integration validation (post-implementation, before declaring done)

1. Run `scripts/run_batch_detection.py --soft-notch` on a 10-chunk
   subset of `USV_lab_131204_chunked_2s_hot/` and on the original
   reference WAV. Compare CNN `auto_accept` counts vs the same run
   without `--soft-notch`.
2. Spot-check 20 random auto-accepts visually using the existing
   `scripts/audit_lab_auto_accepts.py` flow. Verify the per-chunk
   "soft-notch:" log lines match what the spectrogram shows.
3. Wild-mouse byte-equivalence check: re-run a 100-recording slice of
   the 5970 batch with the new code, `--soft-notch` omitted. Diff the
   resulting parquet against the prior 5970 batch parquet. Must be
   byte-identical.

## Risks

| Risk | Mitigation |
|---|---|
| Auto-detector flags a real USV harmonic as a tonal. | `discovery_threshold_db=10` plus the time-stationarity assumption (Welch averages out USV transients). Unit test #3 specifically checks this. |
| `safety_margin_db=0` is too aggressive on bands the CNN was actually using as features. | Default is "bring to median, no further" — minimum possible cut. Negative `safety_margin_db` is the escape valve if even this is too much. |
| Performance: Welch + multiple `sosfiltfilt` calls per WAV. | At 2 s × 300 kHz = 600k samples and 1–3 tonals per chunk, expected runtime is ~50–100 ms per chunk. Compare to STFT + CNN inference which dominate at ~seconds. Acceptable. |
| Default-off discipline violated, wild-mouse runs change. | Add the byte-equivalence integration check above. Reviewer must verify. |
| Tonal width estimation fails on weak / noisy tonals. | `min_width_hz=200` floor protects against width=0 corner cases. Unit test #2 (pure noise) protects against false positives. |
| User runs the script with `--manual-band` overlapping an auto-detected band. | Cascade is linear so order does not matter mathematically. Log both for clarity; warn if center frequencies are within 100 Hz. |

## Open Questions

1. **Should `--soft-notch` also be available in the PyQt6 app's `AudioLoader`
   instantiations, or only in batch detection?** Default suggestion: yes,
   same flag in `app/main_window.py` (off by default, toggleable in the
   processing config panel). Defer if scope creep.
2. **Is `discovery_threshold_db=10` too generous on quiet recordings?**
   Should it scale with the file's overall SNR? Defer — gather data
   from validation pass first, tune after.
3. ~~Should applied notches be persisted to the batch detection output?~~
   **RESOLVED** (see Resolved Decisions and Files-to-Modify §1f):
   Yes — `soft_notch_applied.parquet` + `soft_notch_summary.json` in
   `<output-dir>/` per batch run.
4. **Cooperation with `--subtract-baseline`.** Both target stationary
   content. When both are enabled, are we double-counting? Likely no
   (one acts in audio domain, other in spectrogram domain), but the
   integration validation pass should include a `--soft-notch
   --subtract-baseline` combined run to confirm no pathological
   interaction.
5. **Calibration cadence and rig identification.** How do we know when
   to recalibrate? Proposed triggers (any one): (a)
   `unmatched_detection_rate_warning` fires on a batch run, (b) new
   recording session begins after equipment maintenance, (c) the
   `audit_corpus.py` 365-day mtime check warns. Decision deferred —
   document the triggers in the module doc after empirical data from
   the first lab batch.
6. ~~Rig-id naming convention.~~
   **RESOLVED** (see Resolved Decisions): default heuristic strips
   leading `USV_` and trailing `_chunked_*` from the WAV directory
   name (`USV_lab_131204_chunked_2s_hot/` → `lab_131204`). Override
   via `--rig-id`. Multi-rig labs can add a `rig.json` sidecar later;
   deferred until needed.
7. **Auto-filter mode for unmatched detections.** Default behavior is
   "library is source of truth; unmatched = log-only audit warning."
   Should there be an opt-in `--soft-notch-auto-filter-drift` flag
   that filters unmatched detections too? Trade-off: catches drift
   automatically but masks it from review. **Recommended: no, keep
   audit-only.** Drift should be a human decision (recalibrate vs.
   ignore vs. investigate), not silent auto-suppression.
8. **Multi-channel WAV behavior.** Apply the same filter independently
   to each channel? Compute the PSD on a mixdown for discovery? Most
   USV recordings are mono, but the spec should specify. **Recommended:
   compute PSD on channel 0 for discovery; apply identical filter to
   each channel.** Documented in `apply_soft_notches` docstring.

## Relevant Constraints (from vault)

To be filled by the implementing agent via `node ops/scripts/vault-search.mjs --query "soft notch adaptive lab equipment tonal"` or `/kcheck` before touching `audio_loader.py`. Likely candidates from the prior handoff:

- `notes/per-frequency-bin normalization removes frequency-dependent energy bias in spectrogram input.md`
- `notes/per-recording normalization compensates for varying noise floors across recording sessions.md`
- The `2026-05-08_pre-cnn-spectral-subtraction-lab.md` handoff for full
  rationale on default-off discipline.

## Implementation Order (suggested)

1. **Tests first** — write `tests/test_notch.py` (all 15 tests) using
   synthetic signals; have them fail with `NotImplementedError`.
   Follows the test-architect pattern from `CLAUDE.md`.
2. **Core module** — build `src/usv_spectrogram/app/core/notch.py`
   (`DetectedTonal`, `LibraryEntry`, `TonalLibrary`,
   `ReconciliationResult`, `discover_tonals`, `reconcile`,
   `apply_soft_notches`, `auto_soft_notch`) until tests 1–13 pass.
3. **Calibration script** — write
   `scripts/calibrate_lab_tonal_lines.py` until tests 14–15 pass.
4. **First library** — run calibration against
   `USV_lab_131204_chunked_2s_hot/` to produce
   `data/lab_tonal_lines/lab_131204.json`. Commit alongside this
   implementation as the first real Layer-2 fact in this category.
5. **Standalone CLI** — rewrite `scripts/notch_filter_wav.py` on top
   of `notch.py`. Verify with `--probe` on the reference WAV
   (`131204_1400_m3fm3_chunk_243.wav`); confirm visual output matches
   the calibration library entry.
6. **AudioLoader + batch flag** — add `auto_soft_notch` to
   `audio_loader.py` and `--soft-notch` to `run_batch_detection.py`.
7. **Audit wiring** — extend `scripts/audit_corpus.py` with the
   `lab_tonal_lines` key; extend `docs/modules/corpus-constants.md`
   with the new Layer-2 fact section.
8. **Wild-mouse byte-equivalence check** — re-run a 100-recording
   slice of the 5970 batch with the new code, `--soft-notch` omitted.
   Diff the resulting parquet against the prior 5970 batch parquet.
   Must be byte-identical.
9. **Lab-batch integration validation** — re-run
   `USV_lab_131204_chunked_2s_hot/` with
   `--soft-notch data/lab_tonal_lines/lab_131204.json` and measure
   `auto_accept` noise rate reduction vs the same batch without
   `--soft-notch`. Spot-check 20 audit log lines visually.
10. **Drift-detection synthetic test** — inject a fake 73 kHz tone
    into a small batch and verify the "library may be stale" warning
    fires.
11. **Docs + patterns** — write the module doc
    (`docs/modules/soft-notch.md` or extend `recording-triage.md`),
    update `docs/architecture/patterns.md` if the
    library-plus-drift-audit pattern is worth generalizing.

## Out of Scope (for this handoff)

- iSTFT-based in-band temporal-baseline subtraction. Possible Phase-2
  improvement if soft-notch is still too lossy at USV–tonal crossings.
- CNN retraining on soft-notch-cleaned lab data. Decide after seeing
  the false-positive reduction numbers.
- Auto-triggered recalibration. When the stale-library warning fires
  enough times, the system could in principle re-run calibration
  itself. Deferred — manual recalibration keeps the human in the loop
  on every library change.
- Cross-rig library sharing or inheritance (e.g., "if rigs A and B
  share 90% of tonals, share the library"). Premature until we have
  ≥2 rigs calibrated and can measure overlap.

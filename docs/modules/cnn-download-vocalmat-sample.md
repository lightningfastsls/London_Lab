# Module 18.2a — VocalMat Sample Downloader + Real-Data Loader

**Purpose.** Unlock Module 18.1's deferred real-data exit criteria by:

1. Downloading a small balanced VocalMat sample from OSF (`bk2uj`) for
   the cleaning-validation gate.
2. Adding a real-data loader to the gate so it consumes the sample plus
   our own lab/wild WAVs and produces the binding GO/NO-GO verdict.

Scope split: this module is the small download (~200 PNGs/class) + the
loader. Module 18.2b is the full pull (all 12,221 PNGs) and runs only
after the gate passes.

## Files

| File | Lines | Purpose |
|------|-------|---------|
| `scripts/cnn_download_vocalmat_sample.py` | ~520 | OSF small-sample downloader (stdlib urllib) |
| `tests/test_cnn_download_vocalmat_sample.py` | ~220 | 11 download-script tests |
| `data/vocalmat_sample/.gitignore` | 2 | `*` so the 113 MB of downloaded PNGs never enter git |
| `scripts/cnn_cleaning_validation.py` (modified) | +~220 | New `_load_real_cohorts()` + helpers; wired into `main()` |
| `tests/classifier/test_cleaning_real_data_loader.py` | ~230 | 12 loader tests (sibling to spec tests; spec tests untouched) |

## Architecture

```
OSF bk2uj                         data/vocalmat_sample/
   Dataset/                          noise/*.png      (200)
     noise/*.png       --(A)-->     step_up/*.png   (200)
     step_up/*.png                  ...
     ...                            mult_steps/*.png (74)
                                    manifest.csv

USV_lab_131204/*.wav     --(B)-->   STFT(250 kHz, n_fft=512)
                                    --(C)-->   227x227 dB-scale array
5970 USV/*.wav           --(B)-->   STFT(250 kHz, n_fft=512)
                                    --(C)-->   227x227 dB-scale array

(A) cnn_download_vocalmat_sample.py — stdlib urllib + page_size=100
(B) scipy.signal.resample_poly 300->250 kHz + STFT
(C) _resize_2d bilinear, two-pass np.interp
```

All three cohorts emerge at `(sample_size, 227, 227)` float32. The
cleaning gate's 4 diagnostics then consume them uniformly.

## Constants (sourced, not redeclared)

Per CLAUDE.md corpus protocol, the loader **imports** canonical constants
rather than redeclaring them:

| Constant | Source | Value |
|----------|--------|-------|
| `STFT_N_FFT` | `usv_spectrogram.corpus` | 512 (ADR-002) |
| `STFT_HOP` | `usv_spectrogram.corpus` | 128 (ADR-002) |
| `SAMPLE_RATE_HZ` | `usv_spectrogram.corpus` | 300 000 (ADR-001, used in tests for synthetic WAV source rate) |
| `TARGET_SAMPLE_RATE_HZ` | `usv_spectrogram.classifier` | 250 000 (VocalMat-aligned, C1) |

The two new pipeline-level parameters introduced here are:

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `_REAL_TARGET_SHAPE` | `(227, 227)` | VocalMat AlexNet input convention; matches Module 18.2b's planned patch size. Not a corpus invariant. |
| `_REAL_WINDOW_DURATION_S` | `0.22` s | Per ROADMAP §"Decisions" D1 (patch size 0.22s primary). Aligns gate windows with eventual training patches. |

## Key Decisions

- **OSF backend: stdlib `urllib`, not osfclient.** Initial attempt used
  osfclient but discovered three blocking limitations: hardcoded
  `per_page=10`, no socket timeout (which caused a 3-hour silent hang),
  and no retry on 429. Direct REST with `page[size]=100`, explicit 30 s
  timeout, and exponential-backoff retry is faster, more reliable, and
  zero-dependency.
- **Sequential enumeration, 4-worker parallel download.** The OSF
  v2 API rate-limits aggressively (~10 rps); we hit 429s at 12-thread
  parallel enumeration. The download endpoint
  (`osf.io/download/<id>/`) is bandwidth-bound rather than API-rate-
  limited, so 4 parallel download threads are safe.
- **VocalMat PNG → luminance scalar.** VocalMat ships 227×227 RGB
  spectrogram renderings (MATLAB perceptual colormap). The gate's
  pixel-distribution diagnostics need scalar intensity; we use PIL's
  `convert('L')` luminance (0.299 R + 0.587 G + 0.114 B) divided by 255
  for a `[0, 1]` float32 surrogate. After the cleaning stack's MAD
  layer, all three cohorts are in `[0, 1]` regardless of input units,
  so the "all_layers" ablation is unit-aligned. The "raw" ablation
  shows unit-driven separation — that's expected and is exactly what
  the gate's GO/NO-GO metric measures (does cleaning collapse the
  separation?).
- **Tests use dependency injection.** `OSFVocalMatSource` is one
  concrete implementation of a small `VocalMatSource` Protocol; tests
  use `FakeVocalMatSource` (synthetic file metadata, in-memory
  payloads). No test touches OSF.

## CLI (downloader)

```bash
# Dry-run (enumerate plan, no fetch)
python scripts/cnn_download_vocalmat_sample.py --dry-run

# Default: 200 per class, sequential enum, 4-worker parallel download
python scripts/cnn_download_vocalmat_sample.py \
    --output-dir data/vocalmat_sample/ \
    --n-per-class 200 \
    --workers 4

# Module 18.2b bridge: full pull (all 12,221 files)
python scripts/cnn_download_vocalmat_sample.py --full
```

## Wet-run outcome (2026-05-22)

Sample pulled: **2,196 / 2,210** PNGs (99.4%), 113 MB on disk.
Failures: 4 in `complex`, 8 in `rev_chevron` — all transient timeouts
or 403s. The loader tolerates manifest rows pointing to missing files
because it samples by filesystem glob, not by manifest.

Real-data cleaning-validation report: see
`docs/handoffs/cleaning-validation-report.md` for the GO/NO-GO verdict.

## Future modifiers — invariants to preserve

1. **Stable sample seed = 1729.** Re-runs with the same seed pick the
   same VocalMat files. Changing the seed re-shuffles which 200 of
   1,814 are chosen, which changes the cleaning gate's exact result
   (within statistical noise but not bit-for-bit).
2. **`page_size=100` is OSF's documented cap.** Going larger triggers
   400-level errors; going smaller multiplies API calls and risks
   429s.
3. **`workers=4` for downloads.** Higher concurrency on the
   `/download/` endpoint may still trigger rate-limits at scale; 4 is
   the tested-safe value. If you change it, run a small `--n-per-class`
   first to verify.
4. **Loader's `(227, 227)` target shape.** Module 18.2b assumes this
   shape for patch generation. Changing it here cascades to 18.2b.
5. **`_REAL_WINDOW_DURATION_S = 0.22`.** Aligns to ROADMAP D1. If
   18.2b later adopts a `0.08 s` variant per D1's "defer 0.08s variant
   unless 18.5 (wild transfer) fails", the diagnostic gate should be
   re-run with the matching window length.

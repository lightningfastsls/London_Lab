# CNN Retrain Plan — Matched Windows + Dataset Expansion

**Date:** 2026-03-27
**Origin:** User conversation with Web Claude, transcribed for Claude Code execution
**Prerequisite:** `docs/handoffs/path-b-retrain-cnn-matched-windows.md` (training/inference window mismatch)

---

## Context & Motivation

The production CNN (`models/production/best_model.pt`, trained 2026-02-02) has two problems:

1. **Training/inference window mismatch** — training spectrograms are 105–270px wide, but inference feeds 100px windows. This compresses probabilities (max ~0.58 on real USVs) and forces absurdly low thresholds (0.04/0.03).
2. **Small dataset** — only ~376 unique positive labels, 476 noise labels in `splits/`. With jitter expansion: 376 USV + 1276 Not USV in `data/full_training_dataset/`. Not enough to generalize well across ~25,000 remaining WAV files.

**Goal:** Retrain with (a) matched window sizes and (b) significantly more training data.

**All recordings are wild mice dyads** (folders 5970, 3452, 2379 are different dyad pairs, not different strains). This simplifies domain shift concerns — augmentation doesn't need to account for strain-specific spectral differences.

---

## Current Infrastructure (already implemented — explore before building)

| Component | File | Status |
|-----------|------|--------|
| CNN architecture (small, ~101K params) | `src/usv_spectrogram/models/cnn_classifier.py` | Done |
| Trainer (AdamW, early stopping, class weights) | `src/usv_spectrogram/models/trainer.py` | Done |
| Data loader (variable-size padding, per-image norm) | `src/usv_spectrogram/models/data_loader.py` | Done |
| Dataset assembler (jitter + 3-source negatives) | `src/usv_spectrogram/dataset/assembler.py` | Done |
| Spectrogram extractor (training mode) | `src/usv_spectrogram/detection/spectrogram_extractor.py` | Done |
| Energy detector | `src/usv_spectrogram/detection/energy_detector.py` | Done |
| Labeling tool (Streamlit) | `src/usv_spectrogram/labeling/labeling_app.py` | Done |
| Training script | `scripts/train_cnn.py` | Done |
| Assembly script | `scripts/assemble_training_data.py` | Done |
| ROC curve evaluation | `scripts/generate_roc_curve.py` | Done |
| Splits (by recording, ADR-004) | `src/usv_spectrogram/dataset/splits.py` | Done |
| Quality checks | `src/usv_spectrogram/dataset/quality_checks.py` | Done |
| Sliding inference | `src/usv_spectrogram/app/core/sliding_inference.py` | Done |
| Noise-labeled files (104 recordings) | `USV_Detections/noise_labeled_files/` | Available |
| Existing positive labels | `splits/{train,val,test}.csv` (376 USV total) | Available |
| Existing detections (5970 group) | `USV_Detections/5970/` (949 detection JSONs) | Available |
| Existing detections (3452 group) | `USV_Detections/3452/` (39 detection JSONs) | Available |

**Key parameters (ADR-002):** sr=300,000 Hz, n_fft=512, hop_length=128, freq_range=25–110 kHz

---

## Strategy (priority order)

### Lever 1: Rolling Window Over Long Detections (biggest impact)

**Problem:** Many labeled USV detections span >40ms — often because they captured multiple USVs or a long syllable. The current jitter places a 40ms window at random positions within the detection. But for a 120ms detection, a systematic sliding window with 10ms stride produces ~9 windows, each a legitimate training example (the CNN will encounter exactly these partial views at inference time).

**Implementation:**
- Add a `rolling_window` mode to `DatasetAssembler` alongside existing jitter
- For each positive detection with `duration_ms > jitter_window_ms`:
  - Slide a 40ms window from `start_ms` to `end_ms - 40ms` with configurable stride (default 10ms)
  - Each window becomes a positive training sample
- For detections where `duration_ms <= jitter_window_ms`: keep existing jitter behavior (random placement, 5 versions)

**On label accuracy for long detections:** Not every 40ms slice of a 120ms detection will contain a USV — some may fall in a gap between two USVs that were merged. The user accepts 0–3 mislabeled windows per long detection as tolerable noise. CNNs are robust to small label noise rates (<5%). **Do NOT require manual review of each window** — that defeats the purpose.

**Minimum overlap policy:** Windows at the very edges (where USV barely enters/exits) are actually the most valuable for training — they're the hardest inference cases. No minimum overlap threshold needed here because these are labeled detections, not random positions.

**Expected yield:** If average long detection is ~100ms, each produces ~7 windows instead of 5 jittered versions. For short detections (<40ms), jitter still gives 5 versions. Net effect: moderate increase in unique positive views, with more realistic edge-case representation.

### Lever 2: Noise File Slicing for Negatives

**Problem:** Current negatives come from 3 sources (random position, inter-USV gap, low-energy) but are limited by the number of labeled recordings. There are 104 noise-labeled recording files in `USV_Detections/noise_labeled_files/` — entire recordings confirmed to contain only noise.

**Implementation:**
- Add a `noise_file_negatives` source to `DatasetAssembler`
- For each noise-labeled recording JSON:
  - Find the corresponding WAV file
  - Slice into 40ms windows with a stride of 20–30ms (noise is more homogeneous, larger stride is fine)
  - Each window becomes a negative training sample
- **Source diversity check:** Verify noise files come from multiple recording sessions/dyads — if they're all from one quiet session, the CNN may learn "quiet = no USV" rather than "no USV-shaped spectral contour = no USV"

**Expected yield:** 104 recordings × (recording_duration / stride) = potentially thousands of negatives. May need to subsample to maintain class balance.

### Lever 3: Fix Training/Inference Window Mismatch

**This is mandatory regardless of other changes.** See `docs/handoffs/path-b-retrain-cnn-matched-windows.md` for full analysis.

**Recommended approach (Option A from handoff):**
- Set `jitter_window_ms = 42.7` (or set total extraction to match 100 STFT columns = 42.67ms at hop_length=128, sr=300000)
- Set `jitter_context_padding_ms = 0.0` (the 40ms window IS the inference window, no extra context)
- Training images will be ~100px wide = exactly what `SlidingInference` feeds
- **No changes to inference code needed**

**Important:** The rolling window (Lever 1) and noise slicing (Lever 2) should also use this 42.7ms window, not 40ms. Keep window size consistent across all sample generation.

### Lever 4: Standard Augmentations (lower priority, do after Levers 1–3)

After the dataset is expanded:
- **Time/frequency shifts:** Small random shifts (already partially covered by jitter's random placement)
- **Gain variation:** ±3dB random gain applied to spectrogram (simulates recording level differences)
- **Cutout/masking (optional):** Small masks (10–15% of area), preferably frequency-axis bands only. Avoid masking >15% to limit risk of occluding the USV. **Do NOT mask a full third of the spectrogram** — too high a chance of destroying the signal.

These are training-time augmentations (applied in `USVDataset.__getitem__` or a transform pipeline), not pre-computed like jitter.

---

## Implementation Steps

### Phase 1: Window Size Alignment

1. **Read** `src/usv_spectrogram/app/core/sliding_inference.py` to confirm current inference window (expected: 100 columns = 42.67ms)
2. **Read** `src/usv_spectrogram/dataset/assembler.py` fully to understand jitter implementation
3. **Decision:** Confirm 42.7ms as the unified window size (or adjust if inference uses a different column count)
4. **Modify** `AssemblyConfig` defaults: `jitter_window_ms=42.7`, `jitter_context_padding_ms=0.0`
5. **Test:** Run assembler in dry_run mode to verify spectrogram dimensions match inference expectations

### Phase 2: Rolling Window for Long Detections

1. **Add** `rolling_stride_ms: float = 10.0` to `AssemblyConfig`
2. **Add** `rolling_window_threshold_ms: float` (defaults to `jitter_window_ms`) — detections longer than this use rolling window instead of jitter
3. **Implement** rolling window logic in `DatasetAssembler._create_positives()` (or new method `_create_rolling_positives()`)
4. **Each rolling window sample** gets a unique candidate_id (e.g., `{original_id}_roll_{offset_ms:05.0f}`)
5. **Quality check:** Rolling samples from the same detection must stay in the same split (extend ADR-004 recording-level grouping)
6. **Test:** Verify a 120ms detection produces expected number of windows

### Phase 3: Noise File Negatives

1. **Read** noise label JSONs to understand format (they're in `USV_Detections/noise_labeled_files/`)
2. **Map** each noise JSON to its WAV file path
3. **Add** `noise_files_dir: Optional[Path] = None` to `AssemblyConfig`
4. **Implement** `_create_noise_negatives()` in `DatasetAssembler`:
   - Load WAV, slice into 42.7ms windows with 25ms stride
   - Create `Candidate` objects for each slice
   - Extract spectrograms using same `SpectrogramExtractor` config
5. **Source diversity audit:** Log which dyads/sessions contribute noise files. Warn if <3 unique sources.
6. **Subsample** if noise negatives exceed 2× the positive count (configurable cap)

### Phase 4: Assemble & Train

1. **Run** full assembly: `python scripts/assemble_training_data.py` with new config
2. **Verify** assembly report: check class balance, split sizes, no recording leakage
3. **Train:** `python scripts/train_cnn.py` with existing hyperparameters (small model, 100 epochs, patience 15)
4. **Evaluate:**
   - Run `scripts/generate_roc_curve.py --split test`
   - Check: probabilities should span 0–1 (no compression)
   - Check: Youden's J optimal threshold should be near 0.5
   - Check: AUC ≥ 0.95

### Phase 5: Validate & Deploy

1. **Compare** ROC curves: old model vs. new model on same test set
2. **Spot check:** Run inference on a few known-USV and known-noise recordings
3. **Update** `models/production/best_model.pt` with new checkpoint
4. **Update** `optimal_threshold` in checkpoint metadata to new Youden's J value
5. **Verify** `SlidingInference` works with new model (probabilities in expected range)
6. **Run** app on a test recording to confirm end-to-end behavior

---

## Validation Criteria

| Metric | Current | Target | Hard Fail |
|--------|---------|--------|-----------|
| Training samples (positive) | ~376 unique | >1500 unique views | <500 |
| Training samples (negative) | ~1276 | >2000 | <1000 |
| Probability range (true USV) | 0.03–0.58 | 0.5–0.95 | max <0.7 |
| Probability range (noise) | 0.01–0.15 | 0.01–0.3 | min >0.4 |
| ROC AUC (test) | 0.97 | ≥0.95 | <0.90 |
| Youden's J threshold | 0.437 (compressed) | ~0.5 | <0.2 or >0.8 |
| Window match | Mismatched | Training = Inference = 42.7ms | Any mismatch |

---

## What NOT To Do

- **Do NOT manually review each rolling window** — accept 0–3 mislabels per long detection as noise
- **Do NOT mask >15% of spectrogram** in augmentation — too likely to destroy the USV signal
- **Do NOT change the CNN architecture** — the small model (~101K params) is appropriate for this dataset size
- **Do NOT change STFT parameters** (n_fft=512, hop=128, sr=300000) — these are ADR-002
- **Do NOT modify test expectations to make tests pass** — fix code instead
- **Do NOT use `git add -A`** when committing — stage specific files

---

## Open Questions (for user decision)

1. **Window size precision:** Is 42.7ms (100 STFT columns) the right target, or should we round to 40ms and adjust SlidingInference instead? (Check inference code first)
2. **Rolling window stride:** 10ms gives good overlap but many near-duplicates. Would 15ms be better? (10ms recommended — the near-duplicates differ at edges which is where discrimination matters)
3. **Noise subsampling cap:** How many noise negatives to keep? Suggestion: cap at 2× positives, but keep all sources represented.
4. **Multi-USV long detections:** Some >40ms detections contain 2–3 USVs. The rolling window naturally handles this (some windows get USV, some get gap). Is 0–3 mislabels per long detection acceptable? (User confirmed: yes)

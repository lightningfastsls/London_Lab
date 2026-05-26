# Handoff: Lab-Data CNN Fine-Tune
Date: 2026-05-09
From: Claude Code (subtraction-experiment chat)
To: Claude Code (next chat — fine-tune session)

## Goal

Fine-tune the production CNN (`models/hard_neg_retrain/best_model.pt`) on
labeled lab-cohort examples so it stops hallucinating USVs from equipment-line
patterns it never saw during wild training. Replace the abandoned pre-CNN
spectral-subtraction approach with model-side learning of the noise
distribution.

**Acceptance criteria:**

| Metric | Baseline (current production) | Target | Disaster |
|---|---|---|---|
| Lab `auto_accept` noise rate (eyeball audit, n≥30) | ~14% | <5% | unchanged |
| Wild 5970 detection-count delta vs production | n/a | <1% | >5% |
| Cleanest-bucket real-USV recall | 100% (12/15 in subtracted pilot) | ≥90% | <80% |
| Lab `auto_accept` event count, full 26,309-chunk batch | 32,724 | 25,000–32,000 | <15,000 |

## Why fine-tuning, not more preprocessing

This session attempted pre-CNN spectral subtraction as a lab-only opt-in.
Two methods were implemented, tested, and piloted on the same 382-chunk
labeled set:

### Method 1 — `percentile` (Boll 1979 floor subtraction)

**Code:** `src/usv_spectrogram/app/core/denoise.py` `method="percentile"`

**Pilot result:** 382-chunk paired comparison vs baseline:
- Per-chunk tier shift: auto_accept 220 → 146, auto_reject 53 → 131
- Tier transitions on 72 already-labeled events:

```
                       PILOT
                  GONE  DEMOTED-AR  DEMOTED-MR  KEPT-AA   TOTAL
cleanest             0           0           3       12     15  ← regression-clean ✓
typical              9           2           3        1     15  ← 14/15 demoted ✓
borderline           0           5           1        4     10
high_overlap        0           2          10       10     22  ← residual band cases
long_event           2           2           5        1     10
TOTAL              11          11          22       28     72
```

**Failure mode discovered:** amplitude-modulated bands (e.g.
`131209_1000_m1fm1_chunk_278` at 50/62 kHz) leave 39–79% of original peak
energy after subtraction. Plain p10 strips the band's *floor*, not its
*variance*. The wild-trained CNN still reads the residual horizontal-line
texture as USV-shaped energy.

### Method 2 — `median_envelope` (sliding median per bin, ~0.5 s kernel)

**Code:** `src/usv_spectrogram/app/core/denoise.py` `method="median_envelope"`

**Pilot result on the same 382 chunks: catastrophic over-correction.**
- Per-chunk tier: auto_accept **220 → 0**
- `chunk_173` (clean real USVs): 13 USVs at p=1.000 → **0 events** in pilot
- Pre-CNN energy filter logs: `Windows skipped by energy filter: 459 (100.0%)`
  — many chunks now fall below the CNN's input energy gate

**Failure mode:** envelope subtraction reduces all cells (not just the 10%
floor cells the percentile method hits), pulling the spectrogram's absolute
dB level down 20–30 dB. Two downstream effects:

1. The energy filter — calibrated on wild-data energy levels — rejects
   100% of windows in many chunks.
2. Even when windows pass, MAD-normalized window distributions are more
   peaked than the wild training distribution. The CNN reads "weird outliers"
   instead of "USV-shaped energy."

### Conclusion

There is no free lunch with input preprocessing on a wild-trained CNN
applied to lab data. Percentile preserves USVs but leaves residual bands;
envelope kills bands but kills USVs. The root cause is **out-of-distribution
generalization**: the model has no exposure to lab equipment-line patterns
in its training set. Solving this at the input is leaky; solving it at the
model is principled.

User's framing this session: *"the only thing we can actually do is hard
retraining."*

## Two fine-tune strategies — start with A, escalate to B

### A. Hard-negative augmentation (LIGHT, recommended first)

Re-run the existing `hard_neg_retrain` training loop with lab band-pattern
false positives mixed into the negative set. Reuses all the training
machinery already in place.

**Plan:**
1. Mine ~500 lab false-positives:
   - All events flagged `noise_band_overlap` in
     `results/batch_lab_131204_full/merged_events_with_filter.parquet`
     (filter SEF > 0.5)
   - User-labeled noise events from `audit_2026-05-08/typical/` and
     `audit_2026-05-08/long_event_survivors/` (labels in prior chat
     transcript — re-elicit if needed)
2. Cross-check labels: spot-check 50 mined examples for label correctness
   (poisoned negatives = poisoned model)
3. Add lab positives: re-extract spectrogram windows for the 12 cleanest-bucket
   real USVs, plus another ~100 from chunks the user labels as real USVs
   on a fresh pass through `audit_post_subtraction/cleanest/` and
   `labeled_paired_comparison/KEPT-AA/cleanest_*`
4. Run training loop with `models/hard_neg_retrain/best_model.pt` as
   initialization. Output → `models/lab_finetune_v1/best_model.pt`

**Expected effort:** 1–2 days.

### B. Full lab+wild mixed retrain (HEAVY, fallback if A insufficient)

Build a balanced lab+wild training set (~200 lab positives + ~500 lab
negatives + existing wild data), train from `hard_neg_retrain` checkpoint
with class-balanced sampling and SpecAugment-style augmentation (frequency
masking, time masking — see vault note "spectrogram SpecAugment-style
augmentation").

**Expected effort:** 3–5 days, including more user labeling.

## Files to read first (in order)

1. `docs/handoffs/2026-05-08_pre-cnn-spectral-subtraction-lab.md` — the
   subtraction-attempt handoff this work supersedes
2. `docs/handoffs/v2-full-pipeline-results.md` — how the current production
   model was trained (hard_neg_retrain pipeline)
3. `src/usv_spectrogram/models/trainer.py` — training loop and
   hard-negative integration
4. `src/usv_spectrogram/models/cnn_classifier.py` — model architecture
5. `data/corpus_facts/lab_131204.json` — lab cohort facts including the
   noise-filter parameters and validated stationary bins

## Files NOT to modify (regression guards)

- `models/hard_neg_retrain/*` — leave production model intact; fine-tuned
  model goes to a new directory (`models/lab_finetune_v1/` suggested)
- `src/usv_spectrogram/corpus.py` — corpus invariants frozen
- `src/usv_spectrogram/detection/extraction_config.py` — STFT/freq grid
  must match training grid (CNN FREEZE)
- `src/usv_spectrogram/app/core/denoise.py` and the
  `--subtract-baseline`/`--subtraction-method` plumbing —
  **leave in place but default off.** Could become a training-time
  augmentation knob later (apply percentile subtraction to half of
  training batches as a noise-robustness regularizer).

## Available data inventory

### Labeled events (your sources of truth)

| Location | Events | Status |
|---|---|---|
| `results/batch_lab_131204_full/audit_2026-05-08/{cleanest,typical,borderline,high_overlap_survivors,long_event_survivors}/` | 72 events | User labeled in prior chat transcript. **Step 1 of fine-tune: re-elicit labels in structured form (CSV/JSON) so they're reusable.** |
| `results/batch_lab_131204_subtracted_pilot/labeled_paired_comparison/{KEPT-AA,DEMOTED-MR,DEMOTED-AR,GONE}/` | 72 paired PNGs | Auto-categorized by tier transition. Useful for cold-review labeling pass. |
| `results/batch_lab_131204_subtracted_pilot/audit_post_subtraction/{...}/` | 56 fresh PNGs | Unlabeled, candidates for new labels. |

### Lab data

- WAVs: `USV_lab_131204/` (17 raw recordings)
- 2-sec chunks: `USV_lab_131204_chunked_2s_full/` (26,309 chunks)
- Chunk manifest: `USV_lab_131204_chunked_2s_full/chunk_manifest.csv`
- Production batch: `results/batch_lab_131204_full/` (baseline w/o subtraction)

### Pilot results (do not need to re-run)

- `results/batch_lab_131204_subtracted_pilot/` — percentile subtraction,
  382 chunks. Useful as a comparison reference.
- `results/batch_lab_131204_envelope_pilot/` — envelope subtraction,
  382 chunks. Documents the OOD-failure case; do not use for training.

### Wild data (for regression test)

- `5970_manual_review_reviewed/` — wild WAVs
- `results/` — production batch results to compare against post-fine-tune

## Vault constraints (flatten into your reasoning)

The next chat will have vault access; these are the constraints already
relevant.

1. **Mid-c CNN architecture: [32, 96, 192] filters, dense_units=64, ~207K
   params, sample-to-param ratio ~71:1**
   *(Note: `notes/mid-c-cnn-balances-capacity-and-inference-speed-for-14k-samples.md`)*
   The hard_neg_retrain model uses this. Keep it for the fine-tune unless
   you have a specific reason — bigger architectures with our data volume
   risk overfitting.

2. **Class weight 3× over raw class ratio → effective pos_weight ~35.4**
   *(Note: `notes/3x class weight boost compensates for USV class imbalance...`)*
   The current training loop uses recall-biased weighting. Lab fine-tune
   should *probably* lower this — lab data has more aggressive negatives,
   and recall-biased loss will compound the FP problem we're trying to
   fix. Consider pos_weight ~5–15× for the lab fine-tune.

3. **Modern CNNs are systematically miscalibrated**
   *(Note: `notes/modern CNNs are systematically miscalibrated...`)*
   `models/hard_neg_retrain/temperature.json` was fit on wild data.
   After fine-tune: refit `TemperatureScaler` on a held-out lab set,
   write to `models/lab_finetune_v1/temperature.json`. Do NOT reuse
   the wild temperature.

4. **FP filter is event-level and trained on wild events**
   `models/hard_neg_retrain/fp_filter.pkl` won't generalize. After
   fine-tune: refit `FalsePositiveFilter` on lab events.

5. **Active learning cycle is the established methodology**
   *(Note: `notes/active learning cycle automates the label-train-evaluate-mine loop...`)*
   The label-train-evaluate-mine loop already exists. Lab fine-tune is
   one cycle of this loop, not a new methodology.

6. **CORPUS-INVARIANT: ExtractionConfig.freq_min/max/n_fft/hop are
   frozen** — fine-tune must use the same training grid. Spectrogram
   extraction code must not change.

## Validation plan (in order)

1. **Synthetic data validation** of training data assembly: spot-check
   50 mined hard negatives by rendering PNGs. If any look like real
   USVs, the labeling pipeline is broken — fix before training.
2. **Held-out lab eval**: 80/20 split on labeled lab data. Train on 80%,
   evaluate precision/recall on the held-out 20%. Report numbers.
3. **Wild regression test**: run fine-tuned model on
   `5970_manual_review_reviewed/`. Compare event count, tier
   distribution, and per-chunk max_confidence to production
   `models/hard_neg_retrain/`. **Counts must match within 1%
   absolute.** This is the hard regression guard.
4. **Lab batch re-run + audit**: full 26,309-chunk lab batch with the
   new model. Re-run `scripts/audit_lab_auto_accepts.py`. User
   eyeballs the 5 buckets and confirms typical-bucket noise rate <20%
   (current ~73%).

## Production batch invocation (post-fine-tune)

```bash
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir USV_lab_131204_chunked_2s_full/ \
    --model models/lab_finetune_v1/best_model.pt \
    --output-dir results/batch_lab_131204_finetune_v1/ \
    --temperature models/lab_finetune_v1/temperature.json \
    --fp-filter models/lab_finetune_v1/fp_filter.pkl \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization_v2.json \
    --workers 4
    # NOTE: omit --subtract-baseline. The fine-tuned model handles bands
    # natively — no input preprocessing needed.
```

Wild-data invocation should NOT change the model path:
```bash
# Wild runs continue to use the production model:
--model models/hard_neg_retrain/best_model.pt
```

## Open questions / known unknowns

1. **Lab cohort generalization:** 131204 is one recording session.
   Cohort 9252 (the new lab data, 8 sessions) may have different
   equipment-line patterns. Fine-tune-only-on-131204 may not transfer.
   *Mitigate:* if user has 9252 spot-labels, include them; otherwise
   plan a follow-up 9252-fine-tune cycle.

2. **Training data quantity threshold:** literature suggests 100–500
   hard examples can shift a model substantially. Start with ~200
   labeled lab events; add more if held-out eval shows underfitting.

3. **Fine-tune schedule:** start with frozen-backbone + last-layer
   training (cheaper, less risk of catastrophic forgetting on wild),
   escalate to full-backbone with low LR if precision insufficient.
   See trainer's existing args.

4. **Combining fine-tune with subtraction at inference:** could test
   whether `--subtract-baseline --subtraction-method=percentile` on
   the *fine-tuned* model further improves — the percentile method's
   residual-band issue might be moot if the model has learned to
   ignore the residual itself. Worth an ablation late in the
   project.

5. **Hysteresis-config refit:** `models/hard_neg_retrain/hysteresis_optimization_v2.json`
   was fit on wild data. Fine-tune may shift confidence distributions;
   the optimal hysteresis thresholds may change. *Defer until end* —
   only refit if final lab batch shows obvious threshold mismatch.

## Worth remembering for Claude

- **The wild-trained CNN's failure on lab data is OOD generalization
  on equipment-line patterns**, not a tunable threshold problem. The
  model has no training-set exposure to bands that look like horizontal-
  line texture; it pattern-matches them as USVs. This is a model-side
  problem, not a feature-engineering problem.
- **Pre-CNN preprocessing has fundamental tradeoffs** because the
  model's input distribution expectations are baked into its trained
  weights. Changing the input shifts the model off-distribution. This
  is why both percentile (residual) and envelope (over-correction)
  failed in different ways. **Future Claude instances should not
  propose more input-preprocessing variants as a way around the OOD
  problem — the user explicitly chose hard retraining.**
- **The user values quantitative regression guards.** Wild-mouse
  byte-identical guarantees were validated for subtraction; same
  standard applies for fine-tuning (must not regress wild detection
  counts). The wild regression test is non-negotiable.
- **Lab data is N=1 cohort (131204) right now.** A fine-tune on this
  cohort alone risks overfitting to its specific equipment-line pattern.
  Plan for a follow-up validation pass on 9252 when its detection
  batch finishes.

## References

- `docs/handoffs/2026-05-08_pre-cnn-spectral-subtraction-lab.md` — the
  prior session's spec; this handoff supersedes the production-path
  intent (subtraction not promoted), but the experimental code and
  pilot results are kept for reference and ablation use.
- `docs/handoffs/v2-full-pipeline-results.md` — the precedent for this
  kind of work (hard_neg_retrain pipeline that produced the current
  production model).
- `data/corpus_facts/lab_131204.json` — lab corpus facts including
  validated stationary bins on chunk 194 (50.4, 51.0, 63.3, 63.9 kHz).
- `notes/mid-c-cnn-balances-capacity-and-inference-speed-for-14k-samples.md`
- `notes/3x class weight boost compensates for USV class imbalance in CNN training.md`
- `notes/modern CNNs are systematically miscalibrated — confidence does not match accuracy.md`
- `notes/active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement.md`

# Lab 131204 — Human-Reviewed Detections (generalization test)

This folder preserves the **human-reviewed** subset of the wild→lab detector
generalization test (hard_neg CNN + soft-notch applied to lab cohort 131204,
which the detector was never trained on). Reviewed in session 2026-07-05.

## Contents
- `reviewed_chunk_pngs/` — 60 spectrogram PNGs, one per model-blind random-sample
  chunk that was manually reviewed. Across these 60 chunks there are **168
  detection events**. Each PNG shows the full 2 s chunk with detection boxes.
  Extracted from the original `random_sample_review.html` review gallery.
- `random_sample_60_manifest.csv` — the review manifest: per-chunk stem, tier,
  `n_events`, `max_confidence`, `noise_floor_p90`.
- `generalization_proof_summary.html` — the summary writeup (verified rates:
  ~96–98% hard-tier, 100%/99.7% hot-tier per prior analysis).

## Deliberately excluded (kept out of git)
- Raw `flagged_wavs/*.wav` audio (regenerable from source recordings; large).
- The 27.9 MB monolithic `random_sample_review.html` (these 60 PNGs replace it).
- The full 24 GB `USV_lab_131204/` WAV corpus and 423 MB batch PNG renders.

## Regenerating the rest
Detection tables are committed under `raven_tables_lab_131204/`,
`results/batch_lab_131204_full/*.parquet`, and
`classified_detections_lab_131204_clean.csv`. All rendered pictures can be
re-derived from those tables + the source WAVs.

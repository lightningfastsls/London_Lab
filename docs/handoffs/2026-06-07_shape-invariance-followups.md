# Successor handoff — shape-invariance bake-off follow-ups

**Prior session (2026-06-07):** executed `usv_shape_invariance_handoff.md` end-to-end. Built `scripts/experiments/shape_invariance/` (loader/harness/reversal/io) + M2a/M2b/M3/M4/M5 methods, benchmarked vs soft-DTW on the 611-label human-anchored kNN-purity gate. Result: soft-DTW not dethroned under the strict non-overlapping-CI rule, but 4/5 tie it cheaply; outcome recorded in memory `project_shape_registration_clustering` (entry 'BY-CONSTRUCTION INVARIANCE BAKE-OFF'). Full report: `results/shape_invariance/shape_invariance_comparison.html`.

## Open thread 1 — proper JTFS for M2b
- **Why:** kymatio 0.3.0 only exposes isotropic `Scattering2D`; the handoff specified `TimeFrequencyScattering1D` (separable frequential wavelet → real frequency-transposition invariance). The VAE-diagnostic conclusion ('spectrograms aren't the blocker') holds qualitatively but the principled invariance was not exercised.
- **Do:** check for a kymatio build exposing `TimeFrequencyScattering1D` (or wavespin); swap it into `methods/m2b_jtfs.py` (the spectrogram-render + harness plumbing is reusable), re-run on the 611 labeled calls (CPU).
- **Decision gate:** JTFS beats soft-DTW on jump/complex non-overlapping → strong VAE-diagnostic 'principled non-learned path wins' result, worth a small learned-encoder-on-scattering-front-end follow-up. JTFS only ties → consistent with the Scattering2D substitute; close M2b.

## Open thread 2 — resolve the M2a/M2b near-miss
- **Why:** M2a jump 0.595 [.554,.641] / complex 0.309 and M2b jump 0.587 LEAD soft-DTW (0.522/0.243) on point estimates, CIs overlap by ~0.016 → power-limited near-miss (mirrors the 5970-jump near-miss in Phase 3).
- **Do:** expand the labeled jump+complex set, then re-run `.venv/bin/python scripts/experiments/shape_invariance/run_phase0_m5.py`-style harness eval (or the per-method drivers) on the larger gold set.
- **Decision gate:** M2a/M2b lower CI > soft-DTW upper CI on jump → scattering genuinely beats the elastic bar (headline change). Still overlapping → 'validated as parity at scale, not better' and close.

## Open thread 3 — downstream carry
- M4-RQA (cheap O(1)/call elastic stand-in) and M3's extrema-vs-direction factorization are candidates for the WS-B/C/D continuum pipeline alongside the elastic-FPCA coordinates (`models/shape_fpca/elastic_fpca_scores.parquet`).
- **Optional GPU:** full-corpus 67k M2b joint-TF scattering as a downstream coordinate system — ~minutes on one 3060 Ti when the rig is free (the only GPU-favorable task).

## Files NOT to touch
- `scripts/experiments/eval_shape_human_anchored.py` — the locked SPEC harness (5 funcs + 33 tests). Reused, never edit its signatures.
- `models/shape_kmeans/k20.joblib`, `k20_softdtw.*`, `models/shape_fpca/*` — production/incumbent artifacts, untouched this session.

## Data
- Ridges staged at an ephemeral job dir this session (`$JOB/tmp/shape_data/true_registered_ridges*.npz`). Canonical source = the shape-alphabet pipeline; re-stage from `results/latent_transitions/shape_alphabet/true_registered_ridges_meta.npz` + the lab npz if the job dir is gone.
- Human labels: `data/manual_shape_labels.csv` (758 rows, 611 join after drop-unclear, all cohort lab_131204).
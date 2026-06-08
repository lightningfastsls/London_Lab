# Successor handoff — M2a/M2b jump near-miss, label expansion (Thread 2)

**Predecessor:** `docs/handoffs/2026-06-07_shape-invariance-followups.md` (Thread 1
done 2026-06-08 — proper JTFS benchmarked on rig GPU 0, **TIE → M2b CLOSED**; see
memory `project_shape_registration_clustering` entry "M2b PROPER-JTFS FOLLOW-UP" and
`results/shape_invariance/m2b_jtfs_real_result.json`). Thread 3 (full-corpus 67k JTFS)
is **mooted** by the tie — do not run it. **Only Thread 2 remains open.**

## The open question
M2a (Scattering1D) and M2b (JTFS) LEAD soft-DTW on **jump** by point estimate but
the CIs overlap, so it is a power-limited near-miss, not a win:

| method | jump | complex |
|---|---|---|
| soft-DTW (bar) | 0.522 [.480,.570] | **0.243 [.199,.284]** |
| M2a Scattering1D | 0.595 [.554,.641] | 0.309 |
| M2b proper JTFS | 0.583 [.542,.624] | 0.167 |

On jump, M2a/M2b lower-CI (~0.54–0.55) sits just under soft-DTW's upper-CI (0.570)
→ overlap ~0.016. The decision needs **more labels**, which is a human task.

## Input expected
A larger human-labeled jump+complex gold set, appended to
`data/manual_shape_labels.csv` (current: 758 rows, 611 after join+drop-`unclear`,
4 cohorts {lab:182, 5970:204, 9252:140, 3452:85}; jump=204, complex=67). Label on
the **normal-spectrogram** substrate (`render_human_view_patches.py` /
`alpha3_human_patches`), NOT the contour-masked VAE render — see
`feedback_shape_labeling_substrate`. Do **not** select calls by a shape metric
(jump-score / max-step conflate jump with complex/FM and would bias the eval) —
draw a random/stratified top-up, as the Phase-3 5970 top-up did.

## Analysis (re-run, no new code needed)
The harness and all five methods already exist and are CPU-cheap on the labeled set
(JTFS features are cached per call — only re-extract if new calls are added):

```bash
# 1. M2a (cheap, box CPU) on the grown labels:
.venv/bin/python scripts/experiments/shape_invariance/run_phase0_m5.py   # baselines refresh
#    (M2a driver: methods/m2a_scattering1d.py via its run path)
# 2. M2b proper JTFS only if NEW calls were added — re-stage segments + rig GPU pass:
#    Stage A: extract_m2b_jtfs_segments.py --out <dir>     (box)
#    Stage B: extract_m2b_jtfs_gpu.py (rig cuda:0, kymatio-main + sph_harm shim)
#    Stage C: eval_m2b_jtfs_real.py --feat-dir <dir>        (box; gate auto-applied)
```

## Decision gate
| outcome | action |
|---|---|
| M2a/M2b jump lower-CI > soft-DTW jump upper-CI (non-overlap) | **Headline change** — scattering genuinely beats the elastic bar on jump; write it up, consider a learned-encoder-on-scattering follow-up |
| still overlapping at the larger N | "validated as parity at scale, not better" — close the scattering family; soft-DTW stays the jump method |
| complex stays below soft-DTW for both | confirms soft-DTW's irreducible edge = warp on multi-segment calls (already the standing conclusion) |

## Files to touch / NOT touch
- **NOT touch (locked SPEC):** `scripts/experiments/eval_shape_human_anchored.py`
  (5 funcs + 33 tests). Reused by the harness; never edit its signatures.
- **NOT touch (incumbent/production):** `models/shape_kmeans/k20.joblib`,
  `k20_softdtw.*`, `models/shape_fpca/*`.
- **Reuse as-is:** the `scripts/experiments/shape_invariance/` package (loader,
  harness, io, reversal, methods/*). M2b JTFS needs kymatio-main isolated to a
  throwaway dir + the `sph_harm`→`sph_harm_y` shim on the rig.

## Data / housekeeping
- Staged ridges: `/home/shachar/.claude/jobs/b619c2bb/tmp/shape_data/true_registered_ridges*.npz`
  (re-stage from `results/latent_transitions/shape_alphabet/` if the job dir is gone).
- Rig staging from this session: `/data/shachar/jtfs_m2b/` (~320 MB segments +
  small features) — safe to delete once Thread 2 is settled.

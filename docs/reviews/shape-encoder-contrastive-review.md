# Shape Encoder Contrastive (Pathway B) Module Review

**Review date:** 2026-05-27
**Reviewer:** master-reviewer
**Handoff:** `docs/handoffs/2026-05-27_shape-vae-B-contrastive-invariance.md`
**Canonical plan:** `PLAN_geometric_shape_clustering_vae.md` §3 (B-contrastive) + §4 (eval gates)
**Files reviewed:**
- `scripts/experiments/train_shape_encoder_contrastive.py`
- `scripts/eval_shape_encoder.py`
**Tests reviewed:** `tests/test_shape_encoder_contrastive.py` (20), `tests/test_eval_shape_encoder.py` (15)
**Prior art read:** `scripts/experiments/rig_M9_contrastive.py`, `scripts/experiments/rig_M10_image_vae.py`
**Tier:** 3 (contrastive loss math, DSP augmentation geometry, eval scorecard alignment)

---

## Verdict: CHANGES NEEDED — **no blockers**

The NT-Xent loss, augmentation geometry, in-band clamp, encoder-only design, and
row-index alignment are all **correct**. All 35 pre-implementation tests pass. The
scorecard produced on the rig is **trustworthy as-is**. Findings are WARNING/NIT level.

## Math/logic trace (verified)

- **NT-Xent**: numerically identical to M9's `ntxent` (concat → L2-normalize → sim/τ →
  mask diagonal → CE to cross targets). B=1 edge case → loss 0 for perfect positive. No sign/index error.
- **Augmentation geometry**: `gy[i]=i−dy` (+dy moves content DOWN), `gx[j]=(j−dx)/s`
  (+dx right; s∈[0.9,1.1] = ±10% duration warp). Identity at df=0,dt=0,warp=1. Confirmed empirically.
- **In-band clamp**: in-band call rows [r0,r1] ⊂ [b_lo,b_hi] → dy∈[b_lo−r0, b_hi−r1] keeps
  call in band; edge-touch forces 0 at that bound; out-of-band/empty → full-frame fallback [0,H−1]. Correct.
- **Encoder-only**: no decoder / KL / reconstruction anywhere (grep-confirmed). KMeans clusters on
  the pre-projection EMBEDDING; contrastive loss on the projection head (SimCLR standard).
- **Row alignment**: `embs[i]` = embedding of patch i; `desc.row[j]` = surviving patch index;
  `keep=isin(row,val_idx)`, `Z=embs[row[keep]]`, axes `d[...][keep]` all align. No off-by-one.
- **Do-NOT-touch**: only imports corpus constants (never redeclares); `train_contour_vae_v2.py`,
  `ExtractionConfig`, detection pipeline untouched.
- **Print discipline**: params/thresholds/row counts printed at eval. Compliant.

## Findings

**WARNING-1** — `main()` PARAM print did `USV_FREQ_MIN_HZ/1e3` which is `None/1e3` (TypeError) on the
box when the corpus import fails. (Rig is safe — import succeeds.)
**WARNING-2** — `eta2()` emitted `RuntimeWarning: Mean of empty slice` when all labels <0 (returns
0.0 correctly, but noisy). Both files.
**WARNING-3** — no `IMPLEMENTATION_PROGRESS.md` entry yet.
**WARNING-4** — no `docs/modules/shape-encoder-contrastive.md`.
**NIT-1** — STRONG-verdict curvature threshold 0.30 was an undocumented magic number.
**NIT-2** — `load_batch` sorts the batch for memmap I/O; pairing is preserved but undocumented.

---

## Fixes Applied (2026-05-27, implementer)

| Finding | Fix | File:where |
|---|---|---|
| WARNING-1 | `band_str` guard: print "from-patch-freqs" when corpus constants are None instead of dividing None | `train_shape_encoder_contrastive.py` `main()` PARAM block |
| WARNING-2 | Early `if len(v) == 0: return 0.0` before `v.mean(0)` | `eta2()` in BOTH `train_shape_encoder_contrastive.py` and `eval_shape_encoder.py` |
| NIT-1 | Comment: "0.30 = empirical midpoint between production 0.099 and M9 0.344; advisory only" | `eval_shape_encoder.py` verdict block |
| NIT-2 | Comment: sort is for contiguous memmap access; NT-Xent pairs by position i↔i+B independent of shuffle | `train_shape_encoder_contrastive.py` `load_batch` |
| WARNING-3 | Dated entry appended at session wrap | `IMPLEMENTATION_PROGRESS.md` |
| WARNING-4 | Module doc created at session wrap | `docs/modules/shape-encoder-contrastive.md` |

**Re-test after fixes:** `35 passed` (no warnings) — `tests/test_shape_encoder_contrastive.py` + `tests/test_eval_shape_encoder.py`.
No blockers, so self-verification of WARNING/NIT fixes is sufficient per the review's own escalation rule.

# Shape VAE v3 Hybrid (B+A) Module Review

**Date:** 2026-05-27
**Reviewer:** master-reviewer (Sonnet 4.6)
**File under review:** `scripts/experiments/train_shape_vae_v3_hybrid.py`
**Test file:** `tests/test_shape_vae_v3_hybrid.py`
**Spec:** `docs/handoffs/2026-05-27_shape-vae-BA-hybrid.md`, `PLAN_geometric_shape_clustering_vae.md` §3 Option B+A
**Review tier:** Tier 3 (DSP math, ML training objective, contrastive loss, differentiable ridge proxy)

---

## Test Run

```
.venv/bin/python -m pytest tests/test_shape_vae_v3_hybrid.py -q
50 passed
```

`python -m py_compile scripts/experiments/train_shape_vae_v3_hybrid.py` exits 0.

---

## Findings

### SHOULD-FIX 1 — `hybrid_loss` tests a different computation than the train loop
`hybrid_loss` is unit-tested but is not called in the rig train loop (the loop computes terms inline
against the SHARED Track-0 cache, which stores the *pre-differenced* dF/dt target, not a full ridge).
The inline path that runs on the rig is therefore not covered by the unit tests, and `hybrid_loss`'s
internal soft-argmax would leak softmax mass onto zero-padded rows if ever called with the full
256-row canvas.

### SHOULD-FIX 2 — `time_warp_range` config field is stored but never applied
Misleading in a tuning context: a hyperparameter dumped to `hyperparams.json` that has no effect.

### NICE-TO-HAVE 1 — test header says 48 tests, 50 exist (cosmetic).
### NICE-TO-HAVE 2 — `IMPLEMENTATION_PROGRESS.md` not appended; `docs/modules/shape-vae-v3-hybrid.md` missing.

## Verified-Correct (explicitly checked)
- Corpus constants imported, never redeclared; `freq_per_bin = SAMPLE_RATE_HZ/STFT_N_FFT` derived.
- Frozen base (`train_contour_vae_v2.py`) unmodified; only imported.
- In-band augmentation clamp math correct (ceil lower / floor upper, full-band edge → df=0); non-wrapping pad+slice confirmed.
- Unit consistency: cached dF/dt ×1000 (kHz→Hz) matches the Hz soft-argmax ridge before the masked MSE.
- Band/time slicing aligns: `x_recon[:, :, pad_f_top:pad_f_top+f_band, :t_in]` → (B,t_in) ridge → (B,t_in-1) dF/dt == cached width t_in-1 (dim check at load time).
- Latent semantics: posterior means (mu, mu_aug) for NT-Xent/consistency, sampled z for decode — standard and internally consistent.
- Print discipline met (params/weights/anneal/band/row counts).
- No hard dependency on the (deferred) eval script.

## Verdict
**CHANGES NEEDED** — no BLOCKERs. Two SHOULD-FIX items before rig launch. Self-verification permitted (no blocker).

---

## Fixes Applied (implementor, 2026-05-27)

**SHOULD-FIX 1 — resolved by documentation (option B, reviewer-accepted).**
- `train_shape_vae_v3_hybrid.py` `hybrid_loss` docstring: added a "SCOPE / why the rig train loop does
  NOT call this" section and an explicit **BAND-INPUT CONTRACT** ("pass band-restricted `x_recon` +
  matching `freqs_hz`; do NOT pass the zero-padded 256-row canvas"). Rationale: the tested contract
  takes a *full ridge* target and diffs internally, whereas the shared Track-0 cache stores the
  *pre-differenced* dF/dt — so the rig loop must compute the derivative term inline. A code refactor to
  force the loop through `hybrid_loss` is not possible without either breaking the locked test
  expectations or fabricating an absolute ridge from the derivative cache. The inline path reuses the
  same unit-tested pure functions (`soft_argmax_ridge`, `nt_xent`, `latent_consistency`,
  `image_vae_loss`); only the final weighted sum + masked-dF/dt MSE are inline, and both are exercised
  by the actual rig smoke-run (gated). Coverage gap acknowledged and documented.

**SHOULD-FIX 2 — resolved by labeling the field RESERVED.**
- `time_warp_range` cannot be removed: `test_config_default_values` asserts it exists `== (0.9, 1.1)`,
  and applying time-warp unconditionally would break `test_augment_zero_shift_unchanged` (which requires
  identity when df/dt maxes are 0). Added a comment marking the field RESERVED / not-yet-applied
  (the spec calls time-warp "optional"), deferred to a future sweep knob (successor handoff).

**NICE-TO-HAVE 2 — addressed:** `IMPLEMENTATION_PROGRESS.md` appended; `docs/modules/shape-vae-v3-hybrid.md` created.
**NICE-TO-HAVE 1 — left as-is** (cosmetic test-header count; the test file is test-architect spec output).

**Re-verification after fixes:** `py_compile` OK; `pytest tests/test_shape_vae_v3_hybrid.py -q` → 50 passed.

*Reviewed by master-reviewer (Sonnet 4.6); fixes self-verified by the implementing session, 2026-05-27.*

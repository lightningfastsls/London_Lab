# Shape-VAE family — CLOSED (linear-probe precursor ratifies the kill). Canonical = registration.

**Date:** 2026-06-02
**Roadmap executed:** `docs/plans/ROADMAP_SHAPE_INVARIANT_LATENT.md` (Phase 0a only — hard kill-gate hit).
**Predecessor (binding):** `docs/handoffs/2026-05-28_pathway-B-kill-and-canonical.md`.
**Roadmap's predicted filename for this memo:** `2026-05-28_shape-vae-family-CLOSED.md` — written here under the actual run date (2026-06-02) instead.
**Probe script:** `scripts/experiments/probe_shape_existing_encoder.py`.
**Machine result JSON:** `/data/shachar/contour_vae/results/latent_transitions/b_contrastive/score_phase0a_linear_probe.json` (rig).

---

## Decision (locked)

The shape-VAE family is **CLOSED** for shape clustering. The roadmap's Phase 0a linear-probe
precursor — the cheapest possible architecture-level falsifier — was hit and **KILLED** the
roadmap before any Phase 0b/0c/1/2 compute. **Ship `models/shape_kmeans/k20.joblib` (registration
→ KMeans, shape η² 0.58–0.75) as the permanent shape representation. Do not re-open the VAE family
for shape clustering.**

This ratifies the 2026-05-28 Pathway B kill rather than challenging it. The roadmap was an explicit,
falsifiable bet (the author budgeted P ≥ 0.5 that the kill memo's "shape lives in 1-D" claim was
correct, and ~50% probability that Phase 0a would kill the whole thing at ~1 hour CPU). It did.

## What Phase 0a tested and why

The 2026-05-28 kill rested on a *clustering* metric (shape η² 0.044 ≪ 0.12). Phase 0a asked a
narrower, stronger, **architecture-level** question that a substrate-swap (Phase 0c) could not dodge:

> Does the frozen Pathway B encoder's representation contain *linearly-decodable* chevron-shape
> signal — any more than a **random-init encoder of the same architecture** does?

If a frozen-encoder linear probe is at chance (and no better than random weights), then no input-
substrate change + retrain on this architecture can rescue shape clustering — the architecture
itself cannot encode the distinction. Probe + random-init control, 5-fold stratified CV, ~2 minutes
on rig GPU 0.

## Result (decisive)

Probe A — chevron-vs-non-chevron (labels from the **un-registered** ridge `chevron_valley` heuristic,
i.e. substrate-independent; n = 69,290 labelled patches, chevron = 17,980 / 25.9%):

| Encoder | Accuracy | Balanced accuracy |
|---|---|---|
| **frozen Pathway B** | 0.741 | **0.517** |
| **random-init** (same architecture, untrained, eval-mode) | 0.740 | 0.501 |
| majority baseline (always non-chevron) | 0.741 | 0.500 |
| **gap (frozen − random)** | **+0.001** | +0.016 |

- **Probe B (manual `syllable_type`): N/A** — `classified_detections_*.csv` has only the DeepSqueak
  `Cluster_NN` `label` column; no human syllable-type labels exist. Reported unavailable, not fabricated.

**Gate:** PASS requires acc ≥ 0.65 **AND** gap ≥ +0.10; KILL at acc < 0.65 **OR** gap ≤ +0.05.
Gap = **+0.001** → **KILL**.

### Why this is the strongest possible falsification

- The frozen encoder's accuracy *equals the majority-class prior* (0.741). The class is 74/26
  imbalanced, so raw accuracy is uninformative on its own — which is exactly why the roadmap gated
  on the **gap vs random-init**, not the absolute number. Both ride the prior; only *learned* signal
  shows up in the gap.
- **Balanced accuracy** (chance = 0.50) is the honest metric: frozen-B = **0.517**, random-init = 0.501.
  120 epochs of NT-Xent contrastive training bought **+0.016 balanced accuracy over untrained weights**.
- The trained encoder is **statistically indistinguishable from a random projection** for chevron
  geometry. It did not learn a *weak* shape representation — it learned essentially *none*.

## What this leaves canonical

- **Production shape clustering = registration → KMeans** (`models/shape_kmeans/k20.joblib`,
  shape η² 0.58–0.75 on TRUE ridges; productionized 2026-05-25). **Untouched by this work; ships as-is.**
- The 2-D image direction is now falsified at **two independent levels**: the clustering level
  (6/6 attempts, shape η² 0.009–0.105 per the roadmap's tally) *and* the architecture level (this
  linear-probe precursor: trained ≈ random). The empirical conclusion from the kill memo —
  **shape lives in the 1-D registered ridge, not the 2-D pixel grid** — is ratified.

## What this does NOT claim

- It does not test every conceivable 2-D architecture — only the Pathway B `ContrastiveEncoder`
  (the architecture the roadmap proposed to re-use on a swapped substrate). It falsifies *that*
  re-attempt path, which was the roadmap's entire premise.
- It does not touch the registration pipeline's validity; registration remains the canonical win.

## Reproduction

```bash
# on the rig (canonical root /data/shachar/contour_vae); CPU is fine, GPU 0 if idle
cd /data/mickey_london_lab
.venv/bin/python scripts/experiments/probe_shape_existing_encoder.py \
    --train-module-dir /data/mickey_london_lab/scripts/experiments \
    --device cuda:0
# writes results/latent_transitions/b_contrastive/score_phase0a_linear_probe.json
```

## Files NOT to touch (binding, carried forward)

- `models/shape_kmeans/k20.joblib` | `scripts/experiments/rig_R2_shape_alphabet.py`
- `src/usv_spectrogram/corpus.py` | `ExtractionConfig` | detection pipeline
- `train_contour_vae_v2.py` | `train_shape_vae_v3_hybrid.py` (frozen baselines)

## Done means

Closure is the deliverable: the roadmap's hard kill-gate was executed and hit; the shape-VAE family
is documented as falsified at the architecture level; registration is confirmed canonical. Re-opening
would require a substantively different architecture *and* a defense against this linear-probe result —
not a substrate swap or hyperparameter sweep on the existing encoder.

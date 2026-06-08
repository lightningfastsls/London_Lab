# Shape-clustering retrospective — the real miss was MEASUREMENT (circular metric + unused human gold set)

**Source:** session 2026-06-02. 35-agent verification workflow + human-anchored re-scoring.
Report: `results/shape_retrospective/shape_clustering_retrospective.html`.
Scripts: `results/shape_retrospective/{human_anchored_eval.py, score_existing_shape_labels.py, supervised_ceiling.py}`.

## Core claim (corrected; supersedes an earlier circular-metric draft)
The unsupervised-shape goal was "won" by registration→KMeans (η² 0.58–0.75) ONLY on a doubly-circular metric:
`shape η²` is computed on the same registered ridge KMeans optimizes, and checked against `chevron_valley`, a
heuristic also derived from that ridge. Scored against the REAL 204-row human gold set (which existed, unused),
the win is modest and partial.

## Evidence (human-anchored, NON-circular)
- Human gold set EXISTS: `data/manual_shape_labels.csv`, 204 rows, 12-class vocabulary, built by
  `scripts/labeling/hand_label_200.py`. CLOSED handoff's "Probe B: N/A — no human labels" is STALE/WRONG.
  All 204 are lab cohort 131204; only 4 true "Chevron" (25 chevron-family). Join meta on
  `wav_stem__det{call_id-1}` (det is 0-indexed, call_id 1-indexed) → 200/204 matched.
- Human kNN purity (LOO, k=10), registered ridge vs base rate:
  chevron 0.364 (base 0.137, ~2.6x), flat 0.274 (base 0.126, ~2x) = ABOVE chance;
  jump 0.327 (base 0.346), complex 0.117 (base 0.066) = AT/below chance.
  SRVF best on chevron (0.388), worst on jump (0.189). NMI(registration KMeans-20 vs human) = 0.241.
- The heuristic `chevron_valley` vs human chevron-family: precision 0.30, recall 0.56 — a blurry proxy that the
  ENTIRE bake-off optimized.
- Learned encoder (alpha3 `data/alpha3_a6/a6_gamma_binding.json`, 2-D substrate, 185-anchor): chevron 0.124 vs
  random 0.084 — also near base rate. 2-D VAE family genuinely dead (Phase-0a frozen 0.517 vs random 0.501).

## RETRACTION (integrity)
An earlier draft of this report claimed chevron recoverable at 0.84 and "~50 labels solves it." Those used the
CIRCULAR `chevron_valley` target (`score_existing_shape_labels.py`, `supervised_ceiling.py`) and are WITHDRAWN as
evidence about real shape clustering. The human-anchored 0.36 is the valid number.

## What we could have done better (ranked)
- A1 (lead, TESTED THIS SESSION — WORKS): per-pair elastic warp alignment (DTW/soft-DTW via tslearn) on the 182
  human-labeled registered ridges beats registration's Euclidean metric on EVERY family, human-anchored kNN purity:
  chevron 0.36→0.40, **jump 0.327→0.45 (at-chance→clearly above)**, flat 0.27→0.29, complex 0.12→0.18. Mechanism:
  a step/jump is a discontinuity at variable internal position; DTW warps to align it, Euclidean can't. Registration
  only does GLOBAL pitch+time normalization, never internal-landmark warp. Script: `results/shape_retrospective/
  a1_elastic_test.py`. Next: full-corpus soft-DTW k-means (tslearn installed) or fdasrsf SRVF geodesic, on an
  expanded human set. Likely loses η² (different geometry) — score on humans, not η².
- B1/B2: make the 204 human labels the standing eval anchor; replace circular η² with human kNN-purity + random
  control (ROADMAP_SHAPE_INVARIANT_LATENT specified this; closure memo reverted to η²).
- B3: ship a navigable continuum shape-map, not K=20 letters (continuum confirmed; map specified, never rendered).
- Expand the human gold set to wild cohorts + >=20 per family — every decimal here is throttled by N=4 chevrons.

## Uncertainty
Small/imbalanced N; chevron purity swung 0.16<->0.36 on a join-offset choice. Trust the RANK (chevron/flat >
jump/complex) and the heuristic audit, not single decimals.

## Links
[[project_shape_registration_clustering]] — reframes WHY the VAE kill happened: substrate fine, but the eval was
circular and the human anchor unused. 2-D image-VAE family still correctly dead.

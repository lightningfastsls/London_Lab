# Build a navigable 2-D shape-map; decide the transition alphabet

**Predecessor.** `docs/handoffs/2026-05-25_productionize-shape-registration.md` is DONE. The registered-shape K=20 alphabet is built/validated (`models/shape_kmeans/k20.joblib`, ARI=0.9995 vs the bake-off `lab_shape`), and the latent-vs-shape transition comparison is in `results/latent_transitions/alphabet_compare/`.

**Finding that motivates this.** The shape alphabet has LOWER transition MI than the latent alphabet everywhere (combined 0.207→0.058 bits, -72%; well-powered cohorts 5970 -41%, lab_matched -68%, lab_swap -66%). The latent "grammar" was largely pitch/duration autocorrelation. Hard K=20 letters sit on a continuum (bake-off UMAP→HDBSCAN = 23 fuzzy blobs). So a 2-D navigable shape-map is the more honest representation than either alphabet.

**Input expected (rig).** `/data/shachar/contour_vae/results/latent_transitions/shape_alphabet/true_registered_ridges_meta.npz` — keys `shapes` (67337,50), `patch_label`, `cohort`, `wav_stem`, `call_id`, `abs_time_start_s`. This is the identity-tagged ridge set produced 2026-05-25 (byte-faithful, `allclose`, to the cached R1 ridges). Per-call shape letters are in `results/latent_transitions/shape_alphabet/shape_call_letters.parquet`.

**Analysis (rig, ~minutes).**
```python
import numpy as np, pandas as pd, umap
d = np.load('results/latent_transitions/shape_alphabet/true_registered_ridges_meta.npz', allow_pickle=True)
Sh = d['shapes']                       # (67337, 50) registered ridges
emb = umap.UMAP(n_neighbors=30, min_dist=0.05, n_components=2, random_state=42).fit_transform(Sh)
# render: 2-D scatter colored by cohort; overlay a hex-binned mean-ridge thumbnail grid
#   so the map is browsable as 'shape regions', not dots. Aggregate per call (mean ridge)
#   for the call-level map; per patch for the fine-grained map.
```
Color by cohort and by the K=20 shape-letter to see how the letters tile the continuum. A PHATE embedding is a good second view (better at continuous trajectories).

**Decision gate.**

| Outcome | Action |
|---|---|
| 2-D map shows clear navigable regions (chevron / up / down / flat zones) | Adopt the map as the repertoire representation; treat K=20 letters as a coarse index into it |
| Map is a smooth blob with no regions | Confirms continuum; report shape as continuous axes (e.g. modulation depth, curvature), drop hard letters for the grammar analysis |
| User wants a hard alphabet anyway | Re-point `scripts/analyze_latent_transitions.py` to consume `shape_call_letters.parquet` (add a `--shape-letters` arg; keep the latent path as default) — but document that shape-grammar MI is ~30-60% lower than latent |

**Files to touch / NOT touch.** Do NOT modify the production detection pipeline or `ExtractionConfig`. Do NOT overwrite `models/latent_kmeans/` or the `centroids_n20/` latent artifacts. `scripts/analyze_latent_transitions.py` stays unchanged until the alphabet decision is made. New work in the `latent-analysis-b-a-c` worktree + rig `/data/shachar/contour_vae`.

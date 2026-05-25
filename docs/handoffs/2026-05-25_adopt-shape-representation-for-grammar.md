# Adopt a shape representation for the transition/idiom grammar

**Predecessor.** `docs/handoffs/2026-05-25_shape-map-and-alphabet-decision.md` is DONE.
The 2-D shape-map is built (`results/latent_transitions/shape_map/`, see
`shape_map_report.html`). **Verdict: CONTINUUM** — the registered-shape K=20 letters
do NOT tile compact regions of the map (letter silhouette in 2-D = **-0.017**;
HDBSCAN finds only 3 coarse lobes, sizes [46860, 13790, 556]). The map is
*navigable by morphology* (flat core / valley top / chevron bottom / up-sweep
right lobe) but has no discrete category boundaries.

**Decision required (user).** The grammar/idiom analysis
(`scripts/analyze_latent_transitions.py`) currently keys off the **latent** K=20
alphabet. Three options, in recommended order:

| Option | What it means | Cost | When to pick |
|---|---|---|---|
| **A — continuous axes (recommended)** | Drop hard letters for the grammar. Derive 2–3 interpretable shape coordinates (curvature / modulation depth / terminal-sweep, or the UMAP-2 axis), bin into continuous cells, rebuild transitions over those. | ~half day | The honest representation given silhouette ≈ 0. |
| **B — keep latent alphabet (status quo)** | Leave `analyze_latent_transitions.py` unchanged; treat the shape-map as a *descriptive* repertoire view only. | 0 | If the grammar result must stay comparable to prior A2 runs. |
| **C — hard shape alphabet anyway** | Wire the registered-shape letters into the grammar (gate row 3). | ~1–2 hrs | Only if a discrete shape alphabet is explicitly wanted; document the MI caveat. |

**Input expected.** Already local in this worktree:
- `results/latent_transitions/shape_alphabet/shape_call_letters.parquet` —
  per-call (47,026) cols `cohort, wav_stem, call_id, begin_time_s, shape_letter, n_patches`.
- `results/latent_transitions/shape_map/shape_map_embeddings.npz` — keys
  `emb_patch (67337,2)`, `patch_letter`, `patch_cohort`, `emb_call (47026,2)`,
  `call_letter`, `call_cohort` (use `emb_call` / UMAP-2 as a continuous shape axis for Option A).
- `models/shape_kmeans/k20.joblib` — KMeans(20), `n_features_in_=50`.

**Analysis — Option C (only if chosen).**
```python
# Add to scripts/analyze_latent_transitions.py:
#   --shape-letters PATH   (default None -> keep latent alphabet)
# When set, build the per-call symbol sequence from shape_call_letters.parquet
# instead of the latent K-means labels. Key on (cohort, wav_stem, call_id),
# order by begin_time_s within (cohort, wav_stem). Keep the latent path as the
# DEFAULT so prior A2 runs stay reproducible.
import pandas as pd
sl = pd.read_parquet("results/latent_transitions/shape_alphabet/shape_call_letters.parquet")
seq = (sl.sort_values(["cohort","wav_stem","begin_time_s"])
         .groupby(["cohort","wav_stem"])["shape_letter"].apply(list))
# -> feed `seq` into the existing transition-matrix / MI / idiom machinery.
```
Document in the run output: **shape-grammar MI is ~30–60% lower than latent-grammar
MI** (combined 0.207→0.058 bits, -72%; 5970 -41%, lab_matched -68%, lab_swap -66%).
That gap is the whole reason this decision exists — the latent "grammar" was largely
pitch/duration autocorrelation.

**Analysis — Option A (recommended).**
```python
import numpy as np
d = np.load("results/latent_transitions/shape_map/shape_map_embeddings.npz", allow_pickle=True)
axis = d["emb_call"][:, 1]          # UMAP-2 ≈ curvature axis (valley→flat→chevron)
# bin into ~6-10 continuous shape cells (quantile bins), rebuild transitions over bins;
# OR fit 2-3 named axes by regressing ridge features (curvature, terminal slope,
# modulation depth) and bin those. Report transition MI on the continuous cells.
```

**Decision gate (for the implementer).**

| If user picks | Then |
|---|---|
| A | Build continuous shape cells; rebuild transition matrices; compare MI to latent baseline; report per-cohort. |
| B | No code change. Mark the shape-map as descriptive-only in the repertoire docs. |
| C | Add `--shape-letters` (default off); re-run grammar; ALWAYS print the MI-gap caveat. |

**Files to touch / NOT touch.**
- Touch only on an explicit decision: `scripts/analyze_latent_transitions.py`
  (add an opt-in arg; **latent path stays the default**).
- Do NOT modify the production detection pipeline or `ExtractionConfig`.
- Do NOT overwrite `models/latent_kmeans/` or `results/latent_transitions/centroids_n20/`.
- New work stays in the `latent-analysis-b-a-c` worktree.

**Vault context (for a Codex/no-vault session).**
- Shape-space is a continuum even after registration (`project_shape_registration_clustering`).
- `mean_power_db` / tonality are cage artifacts; never cite as biology without
  cross-cage calibration (`feedback_rig_artifact_mean_power_db`).
- Every analysis run must print params, thresholds, sort keys, filter row counts
  (`feedback_analysis_print_params`).
- Default user-facing outputs to HTML; include the `file://wsl.localhost/Ubuntu/...`
  URL in the message (`feedback_html_user_facing_default`, `feedback_wsl_file_viewing`).

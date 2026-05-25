# Productionize registration-based shape clustering

**Context.** The 2026-05-25 bake-off (11 representations, 67,337 true ridges) showed the production contour-VAE K=20 clusters by pitch/duration (shape η² 0.12), and that **registering the ridge** (subtract mean freq, resample active span to 50 pts) before K-means lifts shape η² to 0.58 on real data — beating every learned encoder (M8 VAE 0.50, M9 contrastive 0.34, M10 image VAE 0.009). The win is preprocessing, not modeling. Full memo: memory note `project_shape_registration_clustering.md`.

**Goal.** Replace the raw-latent K=20 alphabet used by the transition/idiom analysis with a registered-shape alphabet.

**Input expected.** Real contour-masked patches on the rig: `/data/shachar/contour_vae/results/masked_patches/{5970,3452,9252,lab_131204}_focus/patches.npz` (keys `patches` (N,257,234), `freqs_kHz`). True registered ridges already cached: `.../shape_registered_TRUE/true_registered_ridges.npz` (key `shapes` (67337,50)).

**Analysis (the incantation).**
```bash
# on rig
ssh shachar@100.113.224.57
cd /data/shachar/contour_vae
PYTHONPATH=/data/mickey_london_lab/src /data/mickey_london_lab/.venv/bin/python - <<'PY'
import numpy as np; from sklearn.cluster import KMeans
d=np.load('results/latent_transitions/shape_registered_TRUE/true_registered_ridges.npz',allow_pickle=True)
Sh=d['shapes']                      # (67337,50) registered ridges = the shape alphabet input
km=KMeans(20,n_init=10,random_state=42).fit(Sh)
# persist km + per-call shape-letter; then re-run transition matrices on these letters
PY
```
Then re-point `scripts/analyze_latent_transitions.py` at the registered-shape labels instead of the latent K-means (`models/latent_kmeans/k20.joblib`).

**Decision gate.**

| Outcome | Action |
|---|---|
| Registered-shape transition MI/entropy differs materially from latent-based | Report both; the latent version was pitch-confounded — prefer registered |
| UMAP→HDBSCAN still shows a continuum (likely) | Consider a 2-D shape-map (navigable) instead of forcing K=20 letters |
| Need sub-harmonic distinctions | Separate track — image VAE with explicit sub-harmonic objective (NOT this handoff) |

**Files to touch / not touch.** Do NOT modify the production detection pipeline or `ExtractionConfig`. Work lives in the `latent-analysis-b-a-c` worktree + rig `/data/shachar/contour_vae`. Don't overwrite the existing `centroids_n20/` latent-alphabet artifacts — write the registered-shape alphabet to a parallel dir.
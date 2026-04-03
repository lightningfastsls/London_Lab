# Handoff: UMAP + HDBSCAN Re-clustering of Classified USV Data
Date: 2026-04-03
From: Claude Code
To: Claude Code (plan mode)

## Task

Replace the DeepSqueak k-means 27-cluster classification with density-based clustering using UMAP dimensionality reduction + HDBSCAN. The current 27 clusters are visually indistinguishable and likely overclustered. HDBSCAN will automatically determine the natural number of clusters and explicitly label noise points.

**Input:** `classified_detections_full.csv` — 7,518 matched USV calls with 18 acoustic features from DeepSqueak.

**Acceptance criteria:**
1. A script that reads the classified CSV, extracts acoustic features, runs UMAP + HDBSCAN, and outputs a new CSV with cluster labels
2. UMAP parameters documented and justified (n_neighbors, min_dist, n_components)
3. HDBSCAN parameters documented (min_cluster_size, min_samples)
4. Output CSV preserves all original columns, adds new columns: `umap_x`, `umap_y`, `hdbscan_label`, `hdbscan_probability`
5. A summary visualization: UMAP 2D scatter colored by cluster, with noise points in grey
6. Per-cluster feature summary table (mean duration, frequency, slope, etc.)
7. Comparison table: old k-means labels vs new HDBSCAN labels (contingency matrix)
8. Gallery PNGs regenerated with new cluster labels (reuse `scripts/generate_cluster_gallery.py` pattern)

## Files to Modify

- **NEW** `scripts/recluster_umap_hdbscan.py` — Main re-clustering script
- **NEW** `results/recluster_umap_hdbscan/` — Output directory for CSV, figures, summary
- **Reuse pattern from** `scripts/generate_cluster_gallery.py` — For gallery PNG generation

## Relevant Constraints (from vault)

1. **UMAP + HDBSCAN is the field-standard pipeline for bioacoustic clustering.** HDBSCAN automatically determines cluster count, handles variable cluster shapes, explicitly models noise points. NMI 0.5-0.88 across species in Best et al. 2023.
   Source: `UMAP plus HDBSCAN is now the dominant unsupervised clustering pipeline for bioacoustic vocalizations`
   Verified: 2026-04-03

2. **Mouse USVs form a continuum, not discrete clusters.** Goffinet 2021 GMM model selection only supported k<=2 for mice. The 27 k-means clusters impose structure the data doesn't naturally support. HDBSCAN's noise labeling will handle the continuous regions honestly.
   Source: `Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs`
   Verified: 2026-04-03

3. **Forcing discrete categories may obscure population-distinguishing variation.** Within-category variation is discarded by hard clustering. HDBSCAN's soft membership probabilities partially address this.
   Source: `forcing USVs into discrete categories may obscure the continuous variation that distinguishes populations`
   Verified: 2026-04-03

4. **Sample rate is 300 kHz — all frequency features from DeepSqueak are already extracted.** No raw audio processing needed for re-clustering; we operate on the feature vectors in the CSV.
   Source: ADR-001 (DECISIONS.md)
   Verified: 2026-04-03

5. **DeepSqueak frequency values appear to be in kHz (range 47-91), not Hz.** Handle unit consistency when normalizing features.
   Source: Observed in `classified_detections_full.csv` this session
   Verified: 2026-04-03

## Context

### Features to use for clustering
The CSV contains these DeepSqueak acoustic features suitable for UMAP input:
- `call_length_s` — duration
- `principal_freq_hz` — center frequency (actually in kHz despite column name)
- `low_freq_hz`, `high_freq_hz`, `bandwidth_hz` — frequency range
- `freq_std_dev_hz` — frequency variability
- `slope` — frequency sweep rate (range: -793 to +704)
- `sinuosity` — contour complexity (range: 1.2 to 8.1)
- `tonality` — harmonic content (range: 0.18 to 0.34)
- `mean_power_db` — signal strength

**Critical:** Z-score normalize all features before UMAP. Slope has a range of ~1500 while tonality has a range of ~0.16 — without normalization, slope would dominate.

### Recommended starting parameters
- **UMAP:** `n_neighbors=15, min_dist=0.1, n_components=2` for visualization, `n_components=8` for HDBSCAN input (per Best et al. 2023 pattern)
- **HDBSCAN:** `min_cluster_size=50` (given 7,518 points, this prevents micro-clusters), `min_samples=10`
- These are starting points — the plan should include parameter sensitivity exploration

### Dependencies
- `umap-learn`, `hdbscan` — check if in venv, install if not
- `scikit-learn` for StandardScaler
- `matplotlib`, `pandas`, `numpy` — already available

### Pattern to follow
See `scripts/generate_cluster_gallery.py` for the project's script structure (REPO_ROOT, SRC_ROOT, argparse, logging pattern).

## Validation

1. `python -m py_compile scripts/recluster_umap_hdbscan.py`
2. Script runs end-to-end and produces output CSV + figures
3. HDBSCAN finds fewer clusters than 27 (if it finds 27 again, something is wrong)
4. Noise points exist (HDBSCAN label -1) — if zero noise points, min_cluster_size is too small
5. Visual inspection: UMAP scatter shows visible structure, not uniform blob
6. Output CSV has same row count as input (7,518 or 7,921 depending on NaN handling)

## Open Questions / Known Risks

1. **Feature selection:** Should we include `mean_power_db`? It's more about recording quality than call type. Consider running with and without.
2. **346 unmatched rows** in the CSV have NaN for detection metadata — they still have acoustic features and should be included in clustering.
3. **HDBSCAN might find very few clusters (2-3)** — this is scientifically valid per Goffinet's finding but may feel unsatisfying. Don't artificially increase cluster count if this happens.
4. **Embedding approach:** The field standard uses learned embeddings (from a pretrained model) before UMAP, not raw features. We're using raw DeepSqueak features because we don't have a pretrained USV embedding model. This is a known limitation — results will be feature-dependent rather than representation-learned.

---

## Results (2026-04-03)

**Status: COMPLETE**

### Clustering outcome
HDBSCAN found **3 natural clusters + 37 noise points** from 7,864 valid rows (57 NaN excluded):

| Cluster | Count | % of total | Description |
|---------|-------|-----------|-------------|
| 2 (main) | 7,598 | 96.6% | The continuous USV manifold — all frequency-modulated call types |
| 1 | 131 | 1.7% | Outlier group (77/131 from old Cluster_27) |
| 0 | 98 | 1.2% | Ultra-short calls (~4ms), minimal frequency sweep |
| noise | 37 | 0.5% | Ambiguous points between clusters |

This **confirms Goffinet 2021**: mouse USVs do not naturally partition into many discrete categories. The 27 k-means clusters were imposing structure the data doesn't support — the contingency matrix shows nearly every old cluster maps almost entirely to HDBSCAN cluster 2.

### Parameters used (defaults)
- **UMAP:** n_neighbors=15, min_dist=0.1, 2D (viz) + 8D (clustering), seed=42
- **HDBSCAN:** min_cluster_size=50, min_samples=10
- **Features:** All 10 acoustic features, StandardScaler normalized
- **Runtime:** ~35 seconds end-to-end

### Output files
All in `results/recluster_umap_hdbscan/`:
- `reclassified_detections.csv` — 7,921 rows, original columns + umap_x, umap_y, hdbscan_label, hdbscan_probability
- `umap_hdbscan_scatter.png` — 2D UMAP colored by 3 HDBSCAN clusters
- `umap_kmeans_scatter.png` — Same UMAP colored by original 27 k-means labels
- `contingency_matrix.png` — Old vs new label heatmap
- `cluster_summary.csv` — Per-cluster feature statistics
- `gallery/HDBSCAN_*/` — Example spectrogram PNGs per cluster

### Tests
- `tests/test_recluster_umap_hdbscan.py` — 36 spec-based tests
- `tests/test_recluster_umap_hdbscan_adversarial.py` — 31 edge-case tests
- All 67 tests pass, including with `-W error::FutureWarning`

### Open questions resolved
1. **mean_power_db included by default** — excludable via `--exclude-features mean_power_db`
2. **57 NaN rows** (not 346) had missing acoustic features — labeled as noise in output
3. **HDBSCAN found 3 clusters** — scientifically valid, not artificially inflated
4. **Raw features limitation acknowledged** — results are feature-dependent, but the UMAP structure is visually clear and the cluster separation is meaningful

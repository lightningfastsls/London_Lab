# USV Clustering Exploration - Quick Start Guide

## Overview

This pipeline discovers acoustic subtypes in USV vocalizations using CNN embeddings and unsupervised clustering. It expands your dataset from 596 manually-labeled USVs to ~2500 samples and identifies 5-8 distinct acoustic clusters.

## Prerequisites

Dependencies already installed:
- ✅ PyTorch (for CNN inference)
- ✅ scikit-learn (for t-SNE, K-means, metrics)
- ✅ umap-learn (for UMAP visualization)
- ✅ hdbscan (for density-based clustering)
- ✅ tqdm (for progress bars)

## Pipeline Execution (5 Steps)

### Step 0: Auto-Detect USVs from Unlabeled WAV Files

**What it does:** Uses trained CNN to automatically detect USVs at high confidence (prob>0.90), expanding dataset from 596 to ~2500 samples.

```powershell
.\.venv\Scripts\python.exe scripts/batch_detect_for_clustering.py \
  --wav-dir "5970 USV" \
  --threshold 0.90 \
  --output-dir analysis/clustering/auto_detected
```

**Expected output:**
- `analysis/clustering/auto_detected/spectrograms/*.png` (~1500-2500 clean spectrograms)
- `analysis/clustering/auto_detected/detections.csv` (metadata)

**Validation:**
```powershell
# Check detection count
wc -l analysis/clustering/auto_detected/detections.csv
# Expected: 1500-2500 lines
```

---

### Step 1: Extract CNN Embeddings

**What it does:** Extracts 128-dimensional embeddings from CNN's global_pool layer for both labeled and auto-detected USVs.

```powershell
.\.venv\Scripts\python.exe scripts/clustering_extract_features.py \
  --model checkpoints/best_model.pt \
  --labeled-csv-dir splits \
  --auto-detected-csv analysis/clustering/auto_detected/detections.csv \
  --output-dir analysis/clustering
```

**Expected output:**
- `analysis/clustering/embeddings_all.csv` (~2500 rows × 132 columns)
  - Columns: candidate_id, source_file, data_source, embedding_0...embedding_127

**Validation:**
```powershell
# Check sample count
wc -l analysis/clustering/embeddings_all.csv
# Expected: ~2500 lines (596 labeled + ~1900 auto-detected)
```

---

### Step 2: Visualize Embeddings (t-SNE and UMAP)

**What it does:** Reduces 128D embeddings to 2D for visualization to explore structure before clustering.

```powershell
.\.venv\Scripts\python.exe scripts/clustering_visualize.py \
  --embeddings analysis/clustering/embeddings_all.csv \
  --method tsne umap \
  --color-by data_source \
  --output-dir analysis/clustering
```

**Expected output:**
- `analysis/clustering/tsne_plot.png` (2D t-SNE scatter plot)
- `analysis/clustering/umap_plot.png` (2D UMAP scatter plot)

**Validation:** Open plots and verify:
- ✅ 5-10 separable clusters visible (not random scatter)
- ✅ Labeled and auto-detected samples mix well (not separated)

---

### Step 3: Cluster Embeddings (HDBSCAN Recommended)

**What it does:** Applies clustering algorithm to discover natural acoustic groupings.

**Option A: HDBSCAN (Recommended - automatic cluster count + outlier detection)**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_cluster.py \
  --embeddings analysis/clustering/embeddings_all.csv \
  --method hdbscan \
  --min-cluster-size 50 \
  --output-dir analysis/clustering
```

**Option B: K-means (Fixed cluster count)**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_cluster.py \
  --embeddings analysis/clustering/embeddings_all.csv \
  --method kmeans \
  --k 5 \
  --output-dir analysis/clustering
```

**Expected output:**
- `analysis/clustering/hdbscan/cluster_assignments.csv` (cluster labels per sample)
- `analysis/clustering/hdbscan/cluster_metrics.txt` (quality metrics)

**Validation:** Check metrics in cluster_metrics.txt:
- ✅ Silhouette score > 0.3 (acceptable structure)
- ✅ 5-8 main clusters (interpretable range)
- ✅ All clusters have >5% of samples (not too small)

---

### Step 4: Analyze Clusters & Tier 2 QC

**What it does:** Extracts exemplar spectrograms for each cluster and generates quality report for manual validation.

```powershell
.\.venv\Scripts\python.exe scripts/clustering_analyze.py \
  --embeddings analysis/clustering/embeddings_all.csv \
  --clusters analysis/clustering/hdbscan/cluster_assignments.csv \
  --spectrograms-labeled spectrograms_training \
  --spectrograms-auto analysis/clustering/auto_detected/spectrograms \
  --n-exemplars 5 \
  --output-dir analysis/clustering/hdbscan
```

**Expected output:**
- `exemplars_cluster_0.png` through `exemplars_cluster_N.png` (1×5 grids)
- `cluster_noise.png` (HDBSCAN outliers, if any)
- `recording_diversity.csv` (per-recording entropy)
- `cluster_quality_report.txt` (Tier 2 QC checklist)

**Tier 2 QC (~5 minutes):**
1. Open each `exemplars_cluster_*.png` file
2. Visual inspection:
   - ✅ **Valid cluster:** Exemplars show consistent acoustic pattern
   - ❌ **Noise cluster:** Exemplars are inconsistent or artifacts
   - 🤔 **Mixed cluster:** Some exemplars good, some bad
3. Document findings in `cluster_quality_report.txt`

**Validation:**
- ✅ 5-8 valid USV clusters (distinct acoustic subtypes)
- ✅ Exemplars within each cluster look visually similar
- ✅ Different clusters show distinct patterns
- ✅ At least 80% of samples are in valid clusters (not noise)

---

## Output Directory Structure

```
analysis/clustering/
├── auto_detected/
│   ├── spectrograms/
│   │   ├── 2024-09-30_11-18-17_0000001_det000.png
│   │   ├── 2024-09-30_11-18-17_0000001_det001.png
│   │   └── ... (~1500-2500 PNGs)
│   └── detections.csv
├── labeled_usvs.csv (temporary - combined splits)
├── embeddings_all.csv (~2500 samples × 132 columns)
├── tsne_plot.png
├── umap_plot.png
└── hdbscan/
    ├── cluster_assignments.csv
    ├── cluster_metrics.txt
    ├── exemplars_cluster_0.png
    ├── exemplars_cluster_1.png
    ├── ...
    ├── cluster_noise.png
    ├── recording_diversity.csv
    └── cluster_quality_report.txt
```

## Scientific Questions Answered

1. **How many distinct USV acoustic subtypes exist?**
   - Silhouette score provides statistical evidence
   - Exemplar grids provide visual confirmation

2. **What are the characteristic features of each subtype?**
   - Exemplar spectrograms show representative samples
   - Visual patterns: duration, frequency, harmonics, modulation

3. **Are some recordings more acoustically diverse than others?**
   - Recording diversity CSV ranks by entropy
   - Identifies recordings with more varied vocalizations

4. **Can we identify representative exemplars for each subtype?**
   - 5 exemplars per cluster (nearest to centroid)
   - Use for labeling, presentation, or further analysis

## Troubleshooting

**Step 0: No detections found**
- Check WAV files exist in directory
- Try lower threshold (--threshold 0.85)
- Verify model path is correct

**Step 1: Import errors**
- Check dependencies installed: `pip list | grep -E "umap|hdbscan|tqdm"`
- Re-install if needed: `pip install umap-learn hdbscan tqdm`

**Step 2: Plots show random scatter (no structure)**
- May need more samples (run Step 0 on more WAV files)
- Try different perplexity for t-SNE (--perplexity 50)

**Step 3: Silhouette score < 0.3**
- Weak clustering structure - embeddings may not separate well
- Try K-means with different k values (--k 3, --k 8)
- Consider collecting more diverse data

**Step 4: Exemplars look inconsistent**
- May indicate mixed cluster - document in QC report
- Check if auto-detected samples have high false positive rate
- Consider re-running Step 0 with higher threshold (--threshold 0.95)

## Next Steps After Clustering

1. **Annotate clusters with semantic labels**
   - Review literature on USV call types
   - Assign names like "short-flat", "long-harmonic", "chevron", etc.

2. **Build cluster-specific classifiers**
   - Train multi-class CNN (8 classes instead of binary)
   - Enables automatic call type classification

3. **Analyze cluster associations with behavior**
   - Link cluster distribution to experimental conditions
   - Identify which call types correlate with specific behaviors

4. **Generate synthetic training data**
   - Use cluster exemplars as templates
   - Augment with time-stretching, pitch-shifting

5. **Publish results**
   - Cluster exemplars make excellent figures
   - Recording diversity metrics useful for methods section

## Implementation Details

All modules created:
- `src/usv_spectrogram/clustering/feature_extractor.py` - CNN embedding extraction
- `src/usv_spectrogram/clustering/visualizer.py` - t-SNE/UMAP visualization
- `src/usv_spectrogram/clustering/clusterer.py` - K-means/HDBSCAN clustering
- `src/usv_spectrogram/clustering/analyzer.py` - Exemplar extraction & QC

For details, see `IMPLEMENTATION_PROGRESS.md` Session 15.

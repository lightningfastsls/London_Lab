# USV Clustering Pipeline - Complete Script Documentation

**Version:** 1.0
**Date:** 2026-01-31
**Author:** Claude Code Implementation

---

## Table of Contents

1. [Pipeline Overview](#pipeline-overview)
2. [Script 0: Test CNN on New Data (Validation)](#script-0-test-cnn-on-new-data)
3. [Script 1: Batch Detection for Clustering](#script-1-batch-detection-for-clustering)
4. [Script 2: Extract CNN Embeddings](#script-2-extract-cnn-embeddings)
5. [Script 3: Visualize Embeddings](#script-3-visualize-embeddings)
6. [Script 4: Cluster Embeddings](#script-4-cluster-embeddings)
7. [Script 5: Analyze Clusters](#script-5-analyze-clusters)
8. [Complete Workflow Example](#complete-workflow-example)
9. [Troubleshooting Guide](#troubleshooting-guide)

---

## Pipeline Overview

The USV clustering pipeline discovers acoustic subtypes in mouse vocalizations using CNN embeddings and unsupervised learning. The pipeline consists of 6 scripts executed sequentially:

```
Script 0: Test CNN on New Data (VALIDATION - Run first!)
    ↓
Script 1: Batch Detection (Auto-detect USVs from WAV files)
    ↓
Script 2: Feature Extraction (Extract 128D CNN embeddings)
    ↓
Script 3: Visualization (t-SNE and UMAP plots)
    ↓
Script 4: Clustering (K-means or HDBSCAN)
    ↓
Script 5: Analysis (Extract exemplars and QC report)
```

**Prerequisites:**
- Trained CNN model: `checkpoints/best_model.pt`
- Python environment with dependencies: `torch`, `scikit-learn`, `umap-learn`, `hdbscan`, `tqdm`

---

## Script 0: Test CNN on New Data

**File:** `scripts/test_cnn_on_new_data.py`

**Purpose:** Validate that the CNN performs well on new USV data before running the full clustering pipeline. This script samples WAV files, runs detection, and generates spectrograms for manual review.

### Command-Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--source-dirs` | list[str] | `['USV_1', 'USV_2', 'USV_3', 'USV_4', 'USV_5']` | Source directories to sample WAV files from |
| `--n-per-dir` | int | `50` | Number of WAV files to sample per directory |
| `--model` | Path | `checkpoints/best_model.pt` | Path to trained CNN model |
| `--threshold` | float | `0.90` | Detection probability threshold (0.0-1.0) |
| `--output-dir` | Path | `analysis/clustering_test` | Output directory for all results |
| `--review-dir` | Path | `test_spectrograms` | Directory for review spectrograms (limited subset) |
| `--max-review` | int | `30` | Maximum spectrograms to save for manual review |
| `--device` | str | `cpu` | Device for CNN inference (`cpu` or `cuda`) |
| `--seed` | int | `42` | Random seed for reproducible sampling |

### Input Requirements

**Required:**
- Source directories containing WAV files (e.g., `USV_1/`, `USV_2/`, etc.)
- Trained CNN model at `--model` path
- WAV files must be 300 kHz sample rate (or adjust in code)

**File Structure:**
```
USV_1/
  usv_lmt_034/
    2024-09-30_18-57-01_0002402.wav
    2024-09-30_18-58-36_0002417.wav
    ...
USV_2/
  ...
```

### Output Files

| File | Location | Description |
|------|----------|-------------|
| `sampled_files_manifest.csv` | `{output-dir}/` | CSV listing which WAV files were randomly sampled |
| `all_detections.csv` | `{output-dir}/` | All CNN detections with metadata (candidate_id, start/end times, probabilities) |
| `spectrograms/*.png` | `{output-dir}/spectrograms/` | All detected USV spectrograms (training mode, clean images) |
| Review spectrograms | `{review-dir}/` | Top N highest-confidence detections for manual review (old files deleted first) |

**Output CSV Schemas:**

**sampled_files_manifest.csv:**
```
wav_file,source_dir
C:\...\USV_1\usv_lmt_034\2024-09-30_18-57-01_0002402.wav,USV_1
C:\...\USV_2\usv_lmt_035\2024-09-30_19-02-36_0002439.wav,USV_2
...
```

**all_detections.csv:**
```
candidate_id,source_file,source_path,start_time_sec,end_time_sec,duration_ms,max_prob,mean_prob,spectrogram_path
2024-09-30_18-57-01_0002402_det000,2024-09-30_18-57-01_0002402.wav,C:\...\USV_1\...,0.450,0.520,70.0,0.985,0.972,C:\...\spectrograms\...
...
```

### Usage Examples

**Basic usage (recommended for first test):**
```powershell
.\.venv\Scripts\python.exe scripts/test_cnn_on_new_data.py
```

**Sample more files (100 per directory = 500 total):**
```powershell
.\.venv\Scripts\python.exe scripts/test_cnn_on_new_data.py --n-per-dir 100
```

**Lower threshold (more permissive):**
```powershell
.\.venv\Scripts\python.exe scripts/test_cnn_on_new_data.py --threshold 0.85
```

**More review samples:**
```powershell
.\.venv\Scripts\python.exe scripts/test_cnn_on_new_data.py --max-review 50
```

**Sample from specific directories only:**
```powershell
.\.venv\Scripts\python.exe scripts/test_cnn_on_new_data.py --source-dirs USV_1 USV_3 USV_5
```

### Expected Output

**Console output:**
```
======================================================================
CNN Testing on New USV Data
======================================================================
Source directories: USV_1, USV_2, USV_3, USV_4, USV_5
Samples per directory: 50
Detection threshold: 0.90
...
Total WAV files found: 6593
Total sampled: 250 files
...
Total detections: 1847
Detections per file (avg): 7.4
Probability distribution:
  Mean: 0.954
  Median: 0.965
  Min: 0.900
  Max: 0.999
Duration distribution (ms):
  Mean: 55.3
  Median: 48.2
  Min: 10.1
  Max: 289.4
======================================================================
```

**Manual Review Decision:**
- ✅ **Proceed:** If >80% of `test_spectrograms/` look like real USVs
- 🤔 **Adjust:** If 50-80% real, increase threshold or proceed with caution
- ❌ **Stop:** If <50% real, CNN may need retraining on new data

---

## Script 1: Batch Detection for Clustering

**File:** `scripts/batch_detect_for_clustering.py`

**Purpose:** Auto-detect USVs from unlabeled WAV files using trained CNN at high confidence threshold. Generates clean training-mode spectrograms for clustering analysis.

### Command-Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--wav-dir` | Path | **REQUIRED** | Directory containing WAV files to process |
| `--model` | Path | `checkpoints/best_model.pt` | Path to trained CNN model |
| `--output-dir` | Path | `analysis/clustering/auto_detected` | Output directory for spectrograms and metadata |
| `--threshold` | float | `0.90` | High probability threshold for detection |
| `--low-threshold` | float | `0.80` | Low probability threshold for hysteresis |
| `--device` | str | `cpu` | Device for CNN inference (`cpu` or `cuda`) |
| `--batch-size` | int | `32` | Batch size for CNN inference |

### Input Requirements

**Required:**
- Directory with WAV files at `--wav-dir`
- Trained CNN model
- WAV files: 300 kHz sample rate, 16-bit PCM

### Output Files

| File | Location | Description |
|------|----------|-------------|
| `detections.csv` | `{output-dir}/` | Metadata for all detected USVs |
| `spectrograms/*.png` | `{output-dir}/spectrograms/` | Clean training-mode spectrograms (magma colormap, no axes) |

**detections.csv Schema:**
```
candidate_id,source_file,start_time_sec,end_time_sec,duration_ms,max_prob,mean_prob,spectrogram_path
2024-09-30_18-57-01_0002402_det000,2024-09-30_18-57-01_0002402.wav,0.450,0.520,70.0,0.985,0.972,C:\...\spectrograms\...
```

### Usage Examples

**Basic usage:**
```powershell
.\.venv\Scripts\python.exe scripts/batch_detect_for_clustering.py --wav-dir USV_5
```

**Use test results (skip this script if test_cnn_on_new_data.py already ran):**
```powershell
# If you already ran test_cnn_on_new_data.py and results look good,
# you can skip this script and use those results directly in Script 2
```

**Custom threshold:**
```powershell
.\.venv\Scripts\python.exe scripts/batch_detect_for_clustering.py \
  --wav-dir USV_5 \
  --threshold 0.95
```

**Process from manifest (after test script):**
```powershell
# Create directory with symlinks/copies from manifest, then run batch detection
# (Manual step - copy files listed in sampled_files_manifest.csv)
```

### Expected Output

**Console:**
```
Batch USV Detection for Clustering
==============================================================
Found 739 WAV files
Processing WAV files: 100%|███████████████| 739/739
==============================================================
Detection complete!
Total detections: 5234
Spectrograms saved to: analysis/clustering/auto_detected/spectrograms
Metadata saved to: analysis/clustering/auto_detected/detections.csv
==============================================================
```

---

## Script 2: Extract CNN Embeddings

**File:** `scripts/clustering_extract_features.py`

**Purpose:** Extract 128-dimensional embeddings from CNN's global_pool layer for both labeled USVs (from training splits) and auto-detected USVs. Combines into single dataset for clustering.

### Command-Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--model` | Path | `checkpoints/best_model.pt` | Path to trained CNN model |
| `--labeled-csv-dir` | Path | `splits` | Directory containing train/val/test split CSVs |
| `--auto-detected-csv` | Path | `analysis/clustering/auto_detected/detections.csv` | CSV from batch detection or test script |
| `--output-dir` | Path | `analysis/clustering` | Output directory for embeddings |
| `--device` | str | `cpu` | Device for CNN inference (`cpu` or `cuda`) |
| `--batch-size` | int | `32` | Batch size for embedding extraction |
| `--labeled-only` | flag | `False` | Extract only labeled USVs (skip auto-detected) |

### Input Requirements

**Required:**
- Trained CNN model
- Labeled USV splits: `splits/train.csv`, `splits/val.csv`, `splits/test.csv`
- Spectrogram images referenced in CSVs must exist

**Optional:**
- Auto-detected USVs CSV from Script 0 or Script 1

### Output Files

| File | Location | Description |
|------|----------|-------------|
| `labeled_usvs.csv` | `{output-dir}/` | Temporary combined labeled USVs (filtered to label='USV') |
| `embeddings_all.csv` | `{output-dir}/` | **Main output:** All embeddings with metadata |

**embeddings_all.csv Schema:**
```
candidate_id,source_file,data_source,label,embedding_0,embedding_1,...,embedding_127
2024-09-30_11-18-17_0000001_00000788,2024-09-30_11-18-17_0000001.wav,labeled,USV,0.234,-0.456,...,0.789
2024-09-30_18-57-01_0002402_det000,2024-09-30_18-57-01_0002402.wav,auto_detected,,0.123,-0.345,...,0.678
...
```

**Columns:**
- `candidate_id`: Unique identifier for USV sample
- `source_file`: Original WAV filename
- `data_source`: `'labeled'` or `'auto_detected'`
- `label`: `'USV'` for labeled data, empty for auto-detected
- `embedding_0` to `embedding_127`: 128-dimensional embedding vector

### Usage Examples

**Extract from both labeled and auto-detected:**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_extract_features.py
```

**Labeled only (for testing with smaller dataset):**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_extract_features.py --labeled-only
```

**Use test script results:**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_extract_features.py \
  --auto-detected-csv analysis/clustering_test/all_detections.csv
```

**GPU acceleration:**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_extract_features.py --device cuda
```

### Expected Output

**Console:**
```
======================================================================
CNN Embedding Extraction for Clustering
======================================================================
Loading model from checkpoints/best_model.pt...
Model loaded on cpu
Embedding dimension: 128

Preparing labeled USV dataset...
  train: 361 USVs
  val: 108 USVs
  test: 68 USVs
Combined labeled USVs: 537 samples

Auto-detected USVs: analysis/clustering/auto_detected/detections.csv
  Found 1847 auto-detected USVs

Extracting features from labeled_usvs.csv
Data source: labeled
Total samples: 537
Extracting embeddings: 100%|████████████| 17/17
Embeddings shape: (537, 128)

Extracting features from auto_detected/detections.csv
Data source: auto_detected
Total samples: 1847
Extracting embeddings: 100%|████████████| 58/58
Embeddings shape: (1847, 128)

Combined dataset:
Total samples: 2384
Data sources:
labeled           537
auto_detected    1847
Name: data_source, dtype: int64

======================================================================
Validation:
======================================================================
[PASS] No NaN values in embeddings
[PASS] Embedding dimension: 128

Total samples: 2384
Data sources:
  labeled: 537
  auto_detected: 1847
======================================================================
```

---

## Script 3: Visualize Embeddings

**File:** `scripts/clustering_visualize.py`

**Purpose:** Generate 2D visualizations of 128D embeddings using t-SNE and/or UMAP dimensionality reduction. Used to explore structure before clustering.

### Command-Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--embeddings` | Path | `analysis/clustering/embeddings_all.csv` | Path to embeddings CSV from Script 2 |
| `--method` | list[str] | `['tsne', 'umap']` | Dimensionality reduction method(s) to use |
| `--color-by` | str | `data_source` | Column name to use for coloring points |
| `--output-dir` | Path | `analysis/clustering` | Output directory for plots |
| `--perplexity` | int | `30` | t-SNE perplexity parameter (5-50) |
| `--n-iter` | int | `1000` | t-SNE number of iterations |
| `--n-neighbors` | int | `15` | UMAP n_neighbors parameter (2-200) |
| `--min-dist` | float | `0.1` | UMAP min_dist parameter (0.0-0.99) |

### Input Requirements

**Required:**
- Embeddings CSV from Script 2 (`embeddings_all.csv`)
- Must contain `embedding_0` through `embedding_127` columns
- Must contain column specified in `--color-by` (default: `data_source`)

### Output Files

| File | Location | Description |
|------|----------|-------------|
| `tsne_plot.png` | `{output-dir}/` | 2D t-SNE scatter plot (if `--method` includes `tsne`) |
| `umap_plot.png` | `{output-dir}/` | 2D UMAP scatter plot (if `--method` includes `umap`) |

**Image specifications:**
- Format: PNG, 150 DPI
- Size: 10" × 8" (1500 × 1200 pixels)
- Points colored by `--color-by` column
- Includes legend and axis labels

### Usage Examples

**Basic usage (both t-SNE and UMAP):**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_visualize.py
```

**t-SNE only:**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_visualize.py --method tsne
```

**Color by cluster labels (after clustering):**
```powershell
# First, merge embeddings with cluster assignments (manual step), then:
.\.venv\Scripts\python.exe scripts/clustering_visualize.py --color-by cluster_label
```

**Adjust t-SNE perplexity for dataset size:**
```powershell
# For smaller datasets (<500 samples): use perplexity=10-20
.\.venv\Scripts\python.exe scripts/clustering_visualize.py --perplexity 15

# For larger datasets (>5000 samples): use perplexity=50-100
.\.venv\Scripts\python.exe scripts/clustering_visualize.py --perplexity 50
```

### Expected Output

**Console:**
```
======================================================================
USV Embedding Visualization
======================================================================
Embeddings: analysis/clustering/embeddings_all.csv
Methods: tsne, umap
Color by: data_source
Output directory: analysis/clustering
======================================================================

======================================================================
Running TSNE
======================================================================
Loaded 2384 samples
Embedding shape: (2384, 128)

Running T-SNE dimensionality reduction...
Input shape: (2384, 128)
[t-SNE] Computing 91 nearest neighbors...
[t-SNE] Indexed 2384 samples in 0.123s...
[t-SNE] Computed neighbors for 2384 samples in 0.456s...
[t-SNE] Iteration 250: error = 1.234
[t-SNE] Iteration 500: error = 0.987
[t-SNE] Iteration 750: error = 0.856
[t-SNE] Iteration 1000: error = 0.789
Output shape: (2384, 2)

Plot saved to: analysis/clustering/tsne_plot.png

t-SNE complete!
======================================================================
```

**Visual validation:**
- ✅ Good: 5-10 separable clusters visible
- ✅ Good: Labeled and auto-detected samples intermixed (not separated)
- ❌ Bad: Random scatter with no structure
- ❌ Bad: Labeled and auto-detected form two separate clusters

---

## Script 4: Cluster Embeddings

**File:** `scripts/clustering_cluster.py`

**Purpose:** Apply clustering algorithms (K-means or HDBSCAN) to 128D embeddings to discover acoustic subtypes.

### Command-Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--embeddings` | Path | `analysis/clustering/embeddings_all.csv` | Path to embeddings CSV |
| `--method` | str | `hdbscan` | Clustering method (`kmeans` or `hdbscan`) |
| `--output-dir` | Path | `analysis/clustering` | Output directory |
| `--k` | int | `5` | Number of clusters for K-means |
| `--min-cluster-size` | int | `50` | HDBSCAN minimum cluster size |
| `--min-samples` | int | `5` | HDBSCAN minimum samples parameter |

### Input Requirements

**Required:**
- Embeddings CSV from Script 2
- Must contain `embedding_0` through `embedding_127` columns

### Output Files

| File | Location | Description |
|------|----------|-------------|
| `cluster_assignments.csv` | `{output-dir}/{method}/` | Cluster labels for each sample |
| `cluster_metrics.txt` | `{output-dir}/{method}/` | Quality metrics and cluster statistics |

**Subdirectory naming:**
- HDBSCAN: `{output-dir}/hdbscan/`
- K-means: `{output-dir}/kmeans_k{k}/` (e.g., `kmeans_k5/`)

**cluster_assignments.csv Schema:**
```
candidate_id,source_file,data_source,label,cluster_label
2024-09-30_11-18-17_0000001_00000788,2024-09-30_11-18-17_0000001.wav,labeled,USV,2
2024-09-30_18-57-01_0002402_det000,2024-09-30_18-57-01_0002402.wav,auto_detected,,0
...
```

**cluster_metrics.txt Contents:**
```
Clustering Method: HDBSCAN
Parameters: {'min_cluster_size': 50, 'min_samples': 5}

Total samples: 2384

Number of clusters: 7
Noise/outliers: 142 (6.0%)

Cluster sizes:
  Cluster 0: 412 (17.3%)
  Cluster 1: 387 (16.2%)
  Cluster 2: 356 (14.9%)
  Cluster 3: 298 (12.5%)
  Cluster 4: 267 (11.2%)
  Cluster 5: 234 (9.8%)
  Cluster 6: 188 (7.9%)
  Noise: 142 (6.0%)

Quality Metrics:
  silhouette_score: 0.427
  calinski_harabasz_score: 1843.2

Interpretation:
  Silhouette score (0.427) indicates ACCEPTABLE cluster structure
```

### Usage Examples

**HDBSCAN (recommended - automatic cluster count):**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_cluster.py --method hdbscan
```

**K-means with k=5:**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_cluster.py --method kmeans --k 5
```

**Try different k values:**
```powershell
# Test multiple k values to find optimal
for k in 3 5 8; do
  .\.venv\Scripts\python.exe scripts/clustering_cluster.py --method kmeans --k $k
done

# Compare silhouette scores in each kmeans_k*/cluster_metrics.txt
```

**Adjust HDBSCAN sensitivity:**
```powershell
# Smaller clusters (more fine-grained)
.\.venv\Scripts\python.exe scripts/clustering_cluster.py \
  --method hdbscan \
  --min-cluster-size 30

# Larger clusters (more conservative)
.\.venv\Scripts\python.exe scripts/clustering_cluster.py \
  --method hdbscan \
  --min-cluster-size 100
```

### Expected Output

**Console:**
```
======================================================================
USV Clustering Analysis
======================================================================
Embeddings: analysis/clustering/embeddings_all.csv
Method: hdbscan
Output directory: analysis/clustering
======================================================================

Loaded 2384 samples
Embedding shape: (2384, 128)
Parameters: {'min_cluster_size': 50, 'min_samples': 5}

Running HDBSCAN clustering...
Input shape: (2384, 128)

Clustering complete!
Number of clusters: 7
Noise/outliers: 142 (6.0%)

Cluster sizes:
  Cluster 0: 412 (17.3%)
  Cluster 1: 387 (16.2%)
  [...]

Computing clustering metrics...
Silhouette score: 0.427 (Acceptable)
Calinski-Harabasz score: 1843.2

Cluster assignments saved to: analysis/clustering/hdbscan/cluster_assignments.csv
Metrics saved to: analysis/clustering/hdbscan/cluster_metrics.txt

======================================================================
Clustering complete!
Results saved to: analysis/clustering/hdbscan
======================================================================
```

**Quality assessment:**
- **Silhouette score:**
  - > 0.5: Excellent separation
  - 0.3 - 0.5: Acceptable structure
  - < 0.3: Weak/poor structure
- **Cluster count:** 5-8 clusters is ideal (interpretable range)
- **Noise percentage:** <10% is good (HDBSCAN)

---

## Script 5: Analyze Clusters

**File:** `scripts/clustering_analyze.py`

**Purpose:** Extract cluster exemplars, generate visualizations, compute diversity metrics, and create quality report for Tier 2 manual validation.

### Command-Line Arguments

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `--embeddings` | Path | `analysis/clustering/embeddings_all.csv` | Path to embeddings CSV |
| `--clusters` | Path | `analysis/clustering/hdbscan/cluster_assignments.csv` | Path to cluster assignments CSV |
| `--spectrograms-labeled` | Path | `spectrograms_training` | Directory with labeled USV spectrograms |
| `--spectrograms-auto` | Path | `analysis/clustering/auto_detected/spectrograms` | Directory with auto-detected spectrograms |
| `--output-dir` | Path | `None` (uses cluster assignments directory) | Output directory for analysis results |
| `--n-exemplars` | int | `5` | Number of exemplar spectrograms per cluster |

### Input Requirements

**Required:**
- Embeddings CSV from Script 2
- Cluster assignments CSV from Script 4
- Spectrogram images (both labeled and auto-detected)

**File structure:**
```
spectrograms_training/
  2024-09-30_11-18-17_0000001_00000788.png
  ...
analysis/clustering/auto_detected/spectrograms/
  2024-09-30_18-57-01_0002402_det000.png
  ...
```

### Output Files

| File | Location | Description |
|------|----------|-------------|
| `exemplars_cluster_0.png` | `{output-dir}/` | Exemplar grid for cluster 0 (1 row × 5 cols) |
| `exemplars_cluster_1.png` | `{output-dir}/` | Exemplar grid for cluster 1 |
| `...` | `{output-dir}/` | One file per cluster |
| `cluster_noise.png` | `{output-dir}/` | HDBSCAN outliers (if noise exists) |
| `recording_diversity.csv` | `{output-dir}/` | Per-recording diversity metrics |
| `cluster_quality_report.txt` | `{output-dir}/` | Comprehensive QC report with checklist |

**recording_diversity.csv Schema:**
```
source_file,n_usvs,n_usvs_no_noise,n_clusters,entropy,normalized_entropy,dominant_cluster,dominant_fraction
2024-09-30_18-57-01_0002402.wav,23,21,5,1.847,0.844,2,0.381
...
```

**Columns:**
- `source_file`: WAV filename
- `n_usvs`: Total USVs detected in this file
- `n_usvs_no_noise`: USVs excluding noise cluster
- `n_clusters`: Number of unique clusters in this file
- `entropy`: Shannon entropy of cluster distribution
- `normalized_entropy`: Entropy / log2(n_clusters) (0-1 scale)
- `dominant_cluster`: Most common cluster in this file
- `dominant_fraction`: Fraction of USVs in dominant cluster

### Usage Examples

**Basic usage (HDBSCAN results):**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_analyze.py
```

**K-means results:**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_analyze.py \
  --clusters analysis/clustering/kmeans_k5/cluster_assignments.csv
```

**More exemplars per cluster:**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_analyze.py --n-exemplars 10
```

**Custom spectrogram directories:**
```powershell
.\.venv\Scripts\python.exe scripts/clustering_analyze.py \
  --spectrograms-labeled path/to/labeled \
  --spectrograms-auto path/to/auto
```

### Expected Output

**Console:**
```
======================================================================
USV Clustering Analysis
======================================================================
Embeddings: analysis/clustering/embeddings_all.csv
Cluster assignments: analysis/clustering/hdbscan/cluster_assignments.csv
Labeled spectrograms: spectrograms_training
Auto-detected spectrograms: analysis/clustering/auto_detected/spectrograms
Output directory: analysis/clustering/hdbscan
Exemplars per cluster: 5
======================================================================

Loading embeddings and cluster assignments...
Total samples: 2384
Unique clusters: 8

Generating exemplar plots...
Exemplars plot saved: analysis/clustering/hdbscan/exemplars_cluster_0.png
Exemplars plot saved: analysis/clustering/hdbscan/exemplars_cluster_1.png
[...]
Exemplars plot saved: analysis/clustering/hdbscan/cluster_noise.png

Computing recording diversity...
Recording diversity saved to: analysis/clustering/hdbscan/recording_diversity.csv

Top 10 Most Diverse Recordings:
  2024-09-30_19-17-49_0002504.wav: 6 clusters, entropy=0.94
  2024-09-30_18-57-01_0002402.wav: 5 clusters, entropy=0.89
  [...]

Generating cluster quality report...
Quality report saved to: analysis/clustering/hdbscan/cluster_quality_report.txt

======================================================================
Analysis complete!
======================================================================

Next steps:
1. Review exemplar images in: analysis/clustering/hdbscan
2. Open quality report: analysis/clustering/hdbscan/cluster_quality_report.txt
3. Perform Tier 2 QC (~5 min manual review)
4. Mark valid clusters in the quality report
======================================================================
```

**cluster_quality_report.txt sample:**
```
======================================================================
USV Clustering Quality Report
======================================================================

Overall Statistics:
  Total samples: 2384
  Data sources:
    labeled: 537 (22.5%)
    auto_detected: 1847 (77.5%)

Cluster Summary:
  Cluster 0: 412 (17.3%)
    labeled: 89 (21.6%)
    auto_detected: 323 (78.4%)
  Cluster 1: 387 (16.2%)
    labeled: 92 (23.8%)
    auto_detected: 295 (76.2%)
  [...]
  Noise/Outliers: 142 (6.0%)

Tier 2 QC Validation (Manual Review):
----------------------------------------------------------------------
Review exemplar images for each cluster and mark:
  [ ] Valid cluster - Exemplars show consistent acoustic pattern
  [ ] Noise cluster - Exemplars are inconsistent or artifacts
  [ ] Mixed cluster - Some exemplars good, some bad

Cluster 0: [ ] Valid  [ ] Noise  [ ] Mixed
  Notes: _______________________________________________

Cluster 1: [ ] Valid  [ ] Noise  [ ] Mixed
  Notes: _______________________________________________

[...]
----------------------------------------------------------------------

Top 10 Most Diverse Recordings (by entropy):
  2024-09-30_19-17-49_0002504.wav: 6 clusters, entropy=0.94
  2024-09-30_18-57-01_0002402.wav: 5 clusters, entropy=0.89
  [...]
```

### Tier 2 QC Process (~5 minutes)

1. **Open exemplar images** in `{output-dir}/`
2. **Visual inspection** for each cluster:
   - ✅ Valid: Exemplars show consistent pattern (similar duration, frequency, structure)
   - ❌ Noise: Exemplars are random/inconsistent
   - 🤔 Mixed: Some good, some bad (needs closer inspection)
3. **Fill out checklist** in `cluster_quality_report.txt`
4. **Final validation:**
   - At least 5 valid clusters identified
   - At least 80% of samples in valid clusters

---

## Complete Workflow Example

### Scenario: First-time clustering with new USV data

**Step 0: Validate CNN performance (CRITICAL FIRST STEP)**

```powershell
# Sample 50 files per directory (250 total)
# Review top 30 detections in test_spectrograms/
.\.venv\Scripts\python.exe scripts/test_cnn_on_new_data.py

# Manual review: Open test_spectrograms/, verify detections look good
# If >80% look like real USVs, proceed. Otherwise, adjust threshold or retrain.
```

**Step 1: Use test results for clustering (skip batch detection)**

```powershell
# The test script already generated detections, skip to feature extraction
# (No need to run batch_detect_for_clustering.py)
```

**Step 2: Extract embeddings**

```powershell
# Use test results instead of batch detection
.\.venv\Scripts\python.exe scripts/clustering_extract_features.py \
  --auto-detected-csv analysis/clustering_test/all_detections.csv \
  --output-dir analysis/clustering
```

**Step 3: Visualize**

```powershell
# Generate t-SNE and UMAP plots
.\.venv\Scripts\python.exe scripts/clustering_visualize.py

# Manual review: Open tsne_plot.png and umap_plot.png
# Verify 5-10 separable clusters visible
```

**Step 4: Cluster with HDBSCAN**

```powershell
# Automatic cluster detection
.\.venv\Scripts\python.exe scripts/clustering_cluster.py --method hdbscan

# Check metrics: Open analysis/clustering/hdbscan/cluster_metrics.txt
# Verify silhouette score >0.3
```

**Step 5: Analyze and QC**

```powershell
# Extract exemplars and generate QC report
.\.venv\Scripts\python.exe scripts/clustering_analyze.py \
  --spectrograms-auto analysis/clustering_test/spectrograms

# Tier 2 QC (5 min):
# 1. Open analysis/clustering/hdbscan/exemplars_cluster_*.png
# 2. Visual inspection - mark valid/noise/mixed
# 3. Fill out cluster_quality_report.txt checklist
```

**Total time:** ~45-60 minutes (plus 5 min manual QC)

---

## Troubleshooting Guide

### Script 0: Test CNN on New Data

**Problem:** No detections found
- **Solution:** Lower threshold (`--threshold 0.85`)
- **Solution:** Check WAV files are valid (300 kHz, readable)
- **Solution:** Verify model path is correct

**Problem:** All spectrograms look like noise
- **Solution:** CNN may not generalize to new data
- **Solution:** Check if new recordings have different characteristics
- **Solution:** May need to retrain CNN with some new data

**Problem:** ImportError for torch/sklearn/etc.
- **Solution:** Install dependencies: `pip install torch scikit-learn pandas pillow`

---

### Script 1: Batch Detection

**Problem:** Memory error during processing
- **Solution:** Reduce batch size (`--batch-size 16`)
- **Solution:** Process fewer files at once

**Problem:** Very slow processing
- **Solution:** Use GPU if available (`--device cuda`)
- **Solution:** Process smaller subset of files

---

### Script 2: Feature Extraction

**Problem:** CUDA out of memory
- **Solution:** Use CPU (`--device cpu`)
- **Solution:** Reduce batch size (`--batch-size 16`)

**Problem:** NaN values in embeddings
- **Solution:** Check input spectrograms are valid
- **Solution:** Verify model loaded correctly

**Problem:** Spectrogram files not found
- **Solution:** Check paths in CSVs match actual file locations
- **Solution:** Use absolute paths in CSVs

---

### Script 3: Visualization

**Problem:** ImportError: No module named 'umap'
- **Solution:** Install UMAP: `pip install umap-learn`

**Problem:** Plots show random scatter (no structure)
- **Solution:** May need more samples (run Script 0/1 on more files)
- **Solution:** Try different perplexity/n_neighbors

**Problem:** t-SNE very slow
- **Solution:** Use UMAP instead (`--method umap`)
- **Solution:** Reduce n_iter (`--n-iter 500`)

---

### Script 4: Clustering

**Problem:** ImportError: No module named 'hdbscan'
- **Solution:** Install HDBSCAN: `pip install hdbscan`

**Problem:** Silhouette score <0.3 (weak structure)
- **Solution:** Try K-means with different k values
- **Solution:** May need more diverse data
- **Solution:** Check visualizations - if random scatter, embeddings don't separate

**Problem:** All samples assigned to noise (HDBSCAN)
- **Solution:** Decrease `--min-cluster-size` (try 30 or 20)
- **Solution:** Try K-means instead

**Problem:** Too many clusters (>15)
- **Solution:** Increase `--min-cluster-size` for HDBSCAN
- **Solution:** Use K-means with fixed k

---

### Script 5: Analysis

**Problem:** Spectrogram images not found for exemplars
- **Solution:** Verify `--spectrograms-labeled` and `--spectrograms-auto` paths
- **Solution:** Check that spectrogram paths in CSVs are absolute

**Problem:** Exemplars show inconsistent patterns
- **Solution:** This may indicate a noise cluster - mark as such in QC
- **Solution:** Try different clustering parameters (Script 4)

**Problem:** Very low recording diversity
- **Solution:** Normal for homogeneous recordings
- **Solution:** Collect more diverse data for future analysis

---

## Performance Benchmarks

**Hardware:** Intel i7, 16GB RAM, CPU inference

| Script | Input Size | Runtime | Output Size |
|--------|------------|---------|-------------|
| Script 0 | 250 WAV files | ~20 min | ~1800 detections |
| Script 1 | 750 WAV files | ~60 min | ~5200 detections |
| Script 2 | 2400 samples | ~5 min | 2400×132 CSV |
| Script 3 (t-SNE) | 2400 samples | ~8 min | 2 PNG images |
| Script 3 (UMAP) | 2400 samples | ~3 min | 2 PNG images |
| Script 4 (HDBSCAN) | 2400 samples | ~2 min | CSV + metrics |
| Script 4 (K-means) | 2400 samples | ~30 sec | CSV + metrics |
| Script 5 | 7 clusters | ~3 min | 8 images + CSVs |

**Total pipeline:** ~45-90 minutes (depending on dataset size)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-01-31 | Initial documentation for clustering pipeline |

---

## Contact & Support

For issues or questions:
1. Check troubleshooting guide above
2. Review CLUSTERING_QUICK_START.md for workflow overview
3. Check IMPLEMENTATION_PROGRESS.md Session 15 for technical details

---

**End of Documentation**

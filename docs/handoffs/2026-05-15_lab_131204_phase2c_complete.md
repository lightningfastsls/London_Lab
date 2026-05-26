# Handoff — Lab 131204 Phase 2C complete (down-sample verdict + Mickey HTML report)

**Date:** 2026-05-15
**Status:** Phase 2C COMPLETE. Both pending tasks from the resume handoff
delivered: (1) UMAP+HDBSCAN down-sample test, (2) self-contained HTML report
for Mickey.
**Previous handoffs:**
1. `docs/handoffs/2026-05-14_lab_131204_post_labeling.md`
2. `docs/handoffs/2026-05-14_lab_131204_phase2a_deepsqueak_handoff.md`
3. `docs/handoffs/2026-05-15_lab_131204_phase2b_complete.md`
4. `docs/handoffs/2026-05-15_lab_131204_phase2c_resume.md` (the anchor for this phase)

---

## Task 1 — Down-sample test of UMAP unimodality: **VERDICT = ROBUST**

### What was run

1. **Stratified down-sample** of `classified_detections_lab_131204_clean.csv`
   (40,787 rows) to **n = 7,920** preserving per-couple proportions
   (`random_state=42`). Output:
   `classified_detections_lab_131204_downsampled_n7921.csv`. Per-couple
   percentages match the source within ±0.01 pp across all 17 couples.

2. **UMAP+HDBSCAN re-cluster** on the down-sample with the **default
   parameters** (`--min-cluster-size 50 --min-samples 10`) — the same
   parameters that produced 5+ separable clusters on wild 3452. Output
   directory: `results/recluster_umap_hdbscan_lab_131204_downsampled/`.

### Result vs gate

The resume handoff defined two gates:
- **≤3 clusters with one >70% mega** → unimodality robust
- **5+ clusters with no single >50% mega** → dataset-size artifact

Actual result on the down-sample:

| hdbscan_label | count | share | role |
|---|---:|---:|---|
| -1 (noise) | 14 | 0.2% | unclustered points |
| 0 | 695 | 8.8% | outlier cluster |
| **1** | **7,211** | **91.1%** | **mega-cluster** |

**Two clusters and a tiny noise tail; the mega-cluster grew from 71% (at
full n) to 91% (at matched n).** Unimodality is not a dataset-size artifact —
if anything, removing data sharpened it.

### Interpretation

The lab 131204 cohort's USV acoustic space is genuinely more continuous
than wild 3452's. Phase 2B finding #2 stands. The Scattoni-7 labels remain
useful as bookkeeping coordinates but should not be read as evidence for
discrete call-types in this cohort.

### Gotcha

The down-sample landed at n=7,920 (not 7,921) because the
proportional rounding under-allocated one event. The 1-event discrepancy
is within Monte-Carlo noise for HDBSCAN at this scale and does not affect
the verdict.

---

## Task 2 — Mickey HTML report: **DELIVERED**

### Artifact

- **Path:** `reports/lab_131204_phase2b_mickey.html`
- **Size:** ~3.97 MB
- **Embedded images:** 9 (all base64 data URIs)
- **External references:** 0 — opens offline in Chrome/Firefox/Safari
- **Builder script:** `scripts/build_mickey_report_lab_131204.py`
  (idempotent; re-running overwrites the report from current artifacts)

### Acceptance criteria (from resume handoff)

| Criterion | Status |
|---|---|
| HTML opens cleanly without internet connection | ✅ 0 external refs |
| All figures render inline (no broken-image icons) | ✅ 9 base64 PNGs |
| Down-sample-test result is incorporated | ✅ Finding 2 callout |
| No raw script names or file paths in body text | ✅ (methodology footer only) |
| Numbers in body match source CSVs exactly | ✅ verified pre-write |

### Structure

| Section | Coverage |
|---|---|
| Header banner | dataset summary (17 couples, ~14 h, 41,061 calls) |
| At a glance | 5-finding overview table with `new`/`confirmed`/`descriptive` tags |
| Finding 1 | Scattoni-7 distribution (Flat 30%, Chevron 22%, …) — descriptive |
| Finding 2 | Continuous repertoire structure + down-sample verdict — **new (robust)** |
| Finding 3 | Tier signal (V = 0.25) — confirmed |
| Finding 4 | Couple keep-set signal (V = 0.165) — confirmed |
| Finding 5 | Two independent noise mechanisms — **new** |
| Open questions | 3 questions for Mickey |
| Methodology footer | source, detection, cleaning, features, classifiers, down-sample method |

### Re-running the build

```bash
cd /home/shachar/projects/mickey_london_lab
.venv/bin/python scripts/build_mickey_report_lab_131204.py
# or with explicit paths:
.venv/bin/python scripts/build_mickey_report_lab_131204.py \
    --root /home/shachar/projects/mickey_london_lab \
    --output reports/lab_131204_phase2b_mickey.html
```

The script's `fig_paths()` enumerates the required input PNGs; missing
inputs raise `FileNotFoundError` with the full list.

---

## File inventory (new in Phase 2C)

**New script (1):**
- `scripts/build_mickey_report_lab_131204.py` — self-contained HTML report builder

**New result directory (1):**
- `results/recluster_umap_hdbscan_lab_131204_downsampled/` — n=7,920 down-sample
  HDBSCAN output (`reclassified_detections.csv`, `cluster_summary.csv`,
  `umap_hdbscan_scatter.png`, `umap_kmeans_scatter.png`,
  `contingency_matrix.png`)

**New top-level data file (1):**
- `classified_detections_lab_131204_downsampled_n7921.csv` — stratified
  down-sample of the clean Phase 2B input (7,920 rows × 35 cols)

**New report (1):**
- `reports/lab_131204_phase2b_mickey.html` — the Mickey deliverable

**New handoff (this file):**
- `docs/handoffs/2026-05-15_lab_131204_phase2c_complete.md`

---

## Updates to Phase 2B findings

Finding #2 (lab repertoire unimodality) was tagged "NEW — needs down-sample
test" in the Phase 2B handoff. With the down-sample verdict in, the tag in
the Mickey HTML and any forward references should be updated to:

> **Finding #2 (CONFIRMED).** Lab repertoire is structurally unimodal under
> default HDBSCAN settings, robust to a size-controlled comparison with the
> wild-mouse 3452 cohort. At n=7,920 the mega-cluster contains 91% of all
> calls (vs 71% at full n=40,787); wild 3452 at n=7,921 produces 5+
> separable clusters with identical parameters.

---

## What was deliberately NOT done

Per the resume handoff's "What NOT to do" list:

- **No modifications to Phase 2A or Phase 2B output files.** All new
  artifacts go in new directories.
- **No regeneration of `events_clean.parquet`.**
- **No fixes to the three filed bugs** (peak_freq column rename, PERMANOVA
  off-by-one, PYTHONPATH requirement for export_raven_tables.py). They
  remain filed for a separate cleanup pass.
- **No external CSS/JS frameworks in the HTML.** Single self-contained file.

---

## Suggested next steps

1. **Send the HTML to Mickey.** Path:
   `reports/lab_131204_phase2b_mickey.html` (~4 MB; should attach to email
   without trouble, or share via OneDrive/Dropbox).
2. **Wait for Mickey's reply** on the three open questions — particularly
   #2 (is the unimodality biologically expected for inbred lab strains?)
   and #3 (do m5fm5/m4fm4/m3fm1/m4fm2 share an environmental factor?).
3. **Phase 3 anchor:** wild-vs-lab statistical comparison. Per Finding 3's
   implication, primary statistics should use `auto_accept`-tier calls
   only. Per Finding 5's implication, a second couple-aware noise guard
   targeting m5fm5/m4fm4/m3fm1/m4fm2 may be needed before Phase 3.

---

## Pipeline diagram (full Phase 2A + 2B + 2C)

```
events_clean.parquet (41,061)
        |
        v  PHASE 2A
filter -> Raven -> MATLAB DeepSqueak (k=26)
        |
        v  PHASE 2B
import (75ms) -> clean -> 3 classifiers -> 3 repertoire variants
        |
        v  PHASE 2C
        +-- stratified down-sample n=7,920 by couple
        |       -> recluster_umap_hdbscan (default params)
        |       -> verdict: 2 clusters, mega 91% -> unimodality ROBUST
        |
        +-- build_mickey_report_lab_131204.py
                -> reports/lab_131204_phase2b_mickey.html
                   (self-contained, 9 base64 PNGs, 0 external refs)
```

---

## TL;DR

1. **Down-sample verdict:** unimodality is biology, not a sample-size
   artifact. Mega-cluster grew from 71% (full n) to 91% (matched n=7,920).
   Wild 3452 at matched n still produces 5+ clusters.
2. **Mickey HTML report:** delivered at
   `reports/lab_131204_phase2b_mickey.html` (3.97 MB, self-contained,
   covers all 5 findings + down-sample verdict).
3. **Three open questions** logged for Mickey at the end of the report.
4. **No regressions** — Phase 2A/2B outputs untouched, three filed bugs
   not touched.
5. **Phase 3 anchor:** wild-vs-lab repertoire comparison with `auto_accept`
   tier filter + second couple-aware noise guard for m5fm5/m4fm4/m3fm1/m4fm2.

# Active Projects Tracker

> **Last updated:** 2026-02-25
> **Purpose:** Single place to see all projects, their status, and what's next for each.

---

## Quick Status Dashboard

| # | Project | Domain | Status | Next Action |
|---|---------|--------|--------|-------------|
| 1 | USV Detection Pipeline | Neuroscience | Phase 11 (blocked on GPU) | Unblock 11.2 with HPC/cloud access |
| 2 | USV Vacation Workstreams | Neuroscience | Ready (3 phases) | `/implement Information Theory Metrics` |
| 3 | Parts Finder (Tevel) | Auto parts / Web | **Phases 1–7.1 DONE** (349+ tests) | Resolve review findings, then 7.2 (needs Tevel data) |
| 4 | USV Detection Desktop App | Neuroscience | **DONE and operational** | — |
| 5 | Syllable Classification | Neuroscience | Plan exists, not started | Review plan, decide priority |
| 6 | DeepSqueak Classification Bridge | Neuroscience | **Phase 2 DONE**, Phase 3 in progress | Import Raven tables in MATLAB DeepSqueak |
| 7 | Knowledge Graph (arscontexta) | Knowledge mgmt | Active (maintenance mode) | `/reduce` inbox (3 items pending) |

---

## 1. USV Detection & Analysis Pipeline

**ROADMAP:** `ROADMAP.md` (main)
**Progress:** `IMPLEMENTATION_PROGRESS.md`
**Domain:** Ultrasonic vocalization analysis for mouse neuroscience research

### What It Is
End-to-end pipeline: WAV recording (300 kHz) -> spectrogram -> CNN sliding-window detection (PyQt6 desktop app) -> syllable classification -> transformer sequence modeling -> VQ-VAE discrete codes -> linguistic analysis. The goal is to discover whether mouse USV sequences contain language-like structure. (Note: the energy detector was used historically to generate CNN training data but is not part of the current production detection workflow.)

### Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1-7 | Core pipeline (detection, extraction, labeling, dataset prep, CNN, desktop app, clustering) | DONE |
| 8.1 | Bout preprocessing (data preparation for transformer) | DONE |
| 8.2 | Causal transformer architecture | DONE |
| 8.3 | Hidden-state VQ-VAE | DONE |
| 8.4 | Analysis & interpretation tools | DONE |
| 9.1 | Training data assembly pipeline | DONE |
| 10.1 | Active learning cycle runner | DONE |
| **11.1** | **Bout extraction on real data** | **DONE** (latest: 2026-02-22) |
| 11.2 | Transformer training on real data | BLOCKED — needs HPC/cloud GPU |
| 11.3 | VQ-VAE training on real hidden states | BLOCKED — needs 11.2 |
| 11.4 | Analysis on real codes | BLOCKED — needs 11.3 |
| 12 | Cross-population USV comparison | FUTURE |
| 13 | Batch detection pipeline | FUTURE |
| 14 | DeepSqueak classification bridge | **14.1 DONE** (Raven export), 14.2 IN PROGRESS |

### Blocker
**Phase 11.2 needs GPU access.** The AMD RX 5700 is insufficient for the ~25-30M param transformer model. Need HPC cluster or cloud GPU (Colab Pro, Lambda, etc.).

### Next Action
Unblock GPU access, then run `/implement` for Phase 11.2.

---

## 2. USV Vacation Workstreams

**ROADMAP:** `ROADMAP_VACATION_DRAFT.md`
**Domain:** Advanced statistical analysis of USV sequences

### What It Is
Three additional analysis phases designed for vacation deep-work: rigorous information theory (replacing the basic Phase 8.4 metrics with statistically proper methods), null model generators (to prove USV structure is real, not artifact), and acoustic probing of the transformer's hidden representations.

### Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 14.1 | Information theory metrics (MLE Zipf, bias-corrected entropy, burstiness) | READY |
| 14.2 | Null model generators (shuffle, Markov, frequency-matched) | READY |
| 14.3 | Statistical comparison framework | READY |
| 15.1 | Acoustic property extractors | READY |
| 15.2 | Probing framework & analysis pipeline | READY |
| 16.1 | LMT data access layer | BLOCKED — needs LMT SQLite files from Prof. London |
| 16.2 | Event-triggered USV rate analysis | BLOCKED — needs 16.1 |

> **Note:** Phase numbering overlaps with main ROADMAP Phase 14 (DeepSqueak). These will need renumbering when integrated.

### Next Action
Can start Phase 14.1 anytime — no dependencies. Run `/implement Information Theory Metrics`.

---

## 3. Parts Finder (Tevel Group)

**ROADMAP:** `ROADMAP_PARTS_FINDER.md`
**Code:** `parts-finder/` (portable — move to target repo when available)
**Domain:** Israeli vehicle spare parts lookup via license plate

### What It Is
Customer-facing tool: enter Israeli license plate -> query free government API (data.gov.il) -> get vehicle ID -> look up correct oil, filters, brakes, bulbs, coolant -> show matching Tevel products. Seven product categories, ~90% Israeli vehicle fleet coverage target.

### Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1.1 | Project scaffold & config | **DONE** — `config.py` (AppConfig), 16 tests |
| 1.2 | data.gov.il API client | **DONE** — `plate_client.py` (async), 24 tests |
| 1.3 | Hebrew-English name mapper | **DONE** — `name_mapper.py` + `hebrew_names.json`, 21 tests |
| 2.1 | Database schema & models | **DONE** — `db.py` + `models.py` (7 dataclasses, 35-col schema), 65+ tests |
| 2.2 | Data import CLI | **DONE** — `seed_database.py` (JSX→CSV→SQLite), 55 tests |
| 3.1 | Oil specification lookup | **DONE** — `oil_lookup.py` (3-tier cascade), 32 tests |
| 3.2 | Bulb type lookup | **DONE** — `lookup/bulbs.py` (8 lamp positions, LED detection), 21 tests |
| 4.1 | Filtron XLS parser | **DONE** — `parsers/filtron_parser.py` (auto header detection, Polish support), 49 tests |
| 4.2 | Filter lookup module | **DONE** — `lookup/filters.py` (4 filter types + cross-refs), 18 tests |
| 5.1 | Coolant lookup | **DONE** — `lookup/coolant.py` (9 specs, compatibility matrix, mixing warnings), 20 tests |
| 5.2 | Brake parts lookup | **DONE** — `lookup/brakes.py` (pad/disc OEM + cross-refs), 18 tests |
| 6.1 | FastAPI backend | **DONE** — `api/app.py` + routes (async, caching, error mapping), 39 tests |
| 6.2 | Claude AI fallback | **DONE** — `api/fallback.py` (selective prompting, JSONL logging), 18 tests |
| 7.1 | React frontend | **DONE** — `VehicleInfo.jsx` + `CategoryCard.jsx` (7 categories, AI badges, Hebrew BiDi) |
| 7.2 | Tevel product catalog mapping | BLOCKED — needs Tevel inventory data |
| 7.3 | Admin dashboard | FUTURE |

### Architecture
- **23 Python modules** in `parts-finder/src/parts_finder/`
- **17 test files**, **349+ tests** all passing
- **3 React components** in `parts-finder/frontend/`
- Pattern: all lookups follow two/three-tier cascade with `from_vehicle_specs()` factory methods

### Known Review Findings
- `_resolve_names` method duplicated across 4 lookup classes → extract to shared module
- Missing DB-layer tests for `find_specs_by_model_year_for_X()` methods
- JSX parser regex fragile for values containing bare `word:` patterns

### Next Action
Address review findings (code quality), then wait for Tevel inventory data for Phase 7.2. When target repo is available, move entire `parts-finder/` directory there.

---

## 4. USV Detection Desktop App

**Plan:** `docs/plans/USV_DETECTION_APP_IMPLEMENTATION.md`
**Domain:** Desktop GUI for USV detection workflow

### What It Is
PyQt6 desktop application (like Audacity/DeepSqueak) that loads WAV files, displays spectrograms, runs CNN inference with sliding window detection, and allows interactive threshold adjustment. Replaces the current Streamlit labeling tool for production use.

### Status
**DONE and operational.** Fully built PyQt6 app in `src/usv_spectrogram/app/` (main_window.py, core/, widgets/). Used for CNN-based sliding window detection with interactive threshold adjustment. This is the current production detection tool — the 93 detections exported as Raven tables were generated here.

### Next Action
No immediate action needed. App is in active use.

---

## 5. USV Syllable Classification

**Plan:** `docs/plans/syllable_classification_roadmap.md`
**Domain:** USV call type taxonomy

### What It Is
Dual-approach syllable classification: supervised (Scattoni taxonomy — flat, step, chevron, etc.) AND unsupervised (VAE continuous manifold). Addresses the open question of whether USVs form discrete categories or a continuous space. Complements Phase 8's VQ-VAE discrete codes with traditional classification.

### Status
Detailed roadmap with its own phase numbering. **Not yet started.** Has scientific motivation doc.

### Next Action
Decide priority. This could run in parallel with Phase 11 (different data dependency path).

---

## 6. DeepSqueak Classification Bridge

**Plan:** `PLAN_raven_export_adapter.md`
**Code:** `src/usv_spectrogram/classification/`
**Domain:** Syllable classification via DeepSqueak (MATLAB)

### What It Is
Bridge between our Python detection pipeline and DeepSqueak's MATLAB classification tools. Exports our CNN detections as Raven selection tables, runs DeepSqueak syllable clustering in MATLAB, then imports the classified results back into Python for statistical analysis.

### Phase Status

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Discovery — WAV↔detection mapping | DONE |
| 2 | Raven export module (`raven_export.py`) | **DONE** — 33 tests, reviewed |
| 2.1 | Export real data | **DONE** — 93 detections across 5 WAVs (31 dirs awaiting WAV files from other machine) |
| 3 | DeepSqueak import & classification (MATLAB) | **IN PROGRESS** — DeepSqueak v3.1.0 installed, Raven tables ready to import. **RESUME HERE** (see steps below) |
| 4 | DeepSqueak results ingestion (`deepsqueak_import.py`) | NOT STARTED — needs Excel output from Step 3 |
| 5 | Statistical analysis (`repertoire_stats.py`) | NOT STARTED — needs classified data |

### Key Files
- `src/usv_spectrogram/classification/raven_export.py` — core export module
- `scripts/export_raven_tables.py` — CLI entry point
- `raven_tables/` — 5 exported Raven selection tables (ready for DeepSqueak)
- `raven_tables/export_summary.json` — export manifest

### Known Issues
- 31 detection directories have no matching WAV files in `5970 USV/` (WAVs are on another machine)
- 1 malformed detection JSON (`0000004/detection_003_1.025s-1.063s.json`) was skipped

### Resume Steps (MATLAB — do these manually)
1. Open MATLAB, type `DeepSqueak` in Command Window
2. **File → Import from Raven** — for each of 5 files, pick the `.txt` from `raven_tables/` then the matching `.wav` from `5970 USV/`
3. **File → Load Calls/Detections** — load one of the saved `.mat` files
4. **Tools → Call Classification → Unsupervised Clustering** — name the syllable clusters
5. **File → Export to Excel** — produces `_Stats.xlsx` with 18 columns

### Next Action (after MATLAB)
Implement Phase 4 (`deepsqueak_import.py`) to ingest the Excel output back into Python. Also: retrieve remaining WAV files from other machine to export the other 31 detection directories.

---

## 7. Knowledge Graph (arscontexta)

> *Renumbered from #6 to accommodate the new DeepSqueak Bridge project above.*

**Location:** `notes/`, `ops/`, `inbox/`
**Domain:** Knowledge management system for all projects

### What It Is
Personal knowledge graph with 171 atomic notes, topic maps, and operational tracking. Powered by wiki-links, semantic search, and a processing pipeline (`/seed` -> `/reduce` -> `/reflect` -> `/reweave`). Captures insights from all projects above.

### Current State
- **171 notes** across multiple topic maps
- **3 inbox items** pending processing (at trigger threshold)
- **1 pending observation**, **4 pending tensions**
- Maintenance mode — no active expansion, just processing incoming material

### Next Action
Run `/reduce` on inbox items (3 pending = at threshold). Optionally `/reduce` the Parts Finder plan to capture auto-parts domain knowledge.

---

## External Dependencies & Blockers

| Blocker | Affects | Action Needed |
|---------|---------|---------------|
| GPU/HPC access | USV Project 11.2+ | Get cloud GPU or HPC cluster access |
| LMT SQLite files | Vacation Phase 16 | Ask Prof. London for data files |
| Tevel inventory data | Parts Finder 7.2 | Get product catalog from Tevel Group |
| Target repo access | Parts Finder deployment | Move `parts-finder/` when repo available |
| WAV files on other machine | DeepSqueak Bridge (31 dirs) | Transfer WAVs to this machine |

---

## Suggested Priority Order

Based on blockers and momentum:

1. **DeepSqueak Classification Bridge** — Raven tables exported, MATLAB step ready now
2. **Parts Finder cleanup** — Review findings to address, then 7.2 awaits Tevel data
3. **Vacation Workstreams 14.1-14.3** — READY, no dependencies, pure analysis
4. **USV Pipeline 11.2+** — When GPU access is resolved
5. **KG maintenance** — Ongoing, process inbox when threshold triggers
6. **Syllable Classification / Desktop App** — Plan when bandwidth allows

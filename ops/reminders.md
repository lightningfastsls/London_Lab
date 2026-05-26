# Reminders

<!-- Checked at session start. Due items surface in orientation. -->
<!-- Format: - [ ] YYYY-MM-DD: Description -->
<!-- Completed: - [x] YYYY-MM-DD: Description (done YYYY-MM-DD) -->

- [x] 2026-04-05: Weekly maintenance routine -- /health run 2026-04-05; follow-through (/reflect, /rethink, dangling link fixes, /reweave) done 2026-04-06
- [ ] 2026-04-12: Weekly maintenance routine -- /health, /reflect, /reweave, /stats
<!-- 2026-05-26 housekeeping reconciliation: A1/A2/A3/B1/B2 and the Mickey check-in
     are complete (project advanced well past Phase B — now on Module 18.x lab CNN +
     shape-representation v2). A4 and Phase C remain genuinely open; left unchecked. -->
- [x] 2026-04-11: Analysis Phase A1 — Temporal dynamics. (done 2026-04 — `results/temporal_dynamics/`; see [[project_analysis_stage]])
- [x] 2026-04-11: Analysis Phase A2 — Sequential structure. (done 2026-04, commit 2828539e — `results/sequential_structure/`, within-bout MI lag-1 = 0.092 bits canonical)
- [x] 2026-04-14: Analysis Phase A3 — Acoustic feature deep-dive. (done 2026-04, commit 8cbe136d — `results/acoustic_feature_analysis/`, continuum confirmed PC1+PC2 = 60.4 %)
- [ ] 2026-04-14: Analysis Phase A4 — Detection quality audit: CNN confidence (det_prob_max) by syllable type, match quality cross-tab, inspect HDBSCAN noise points (N=37) and ultra-short cluster 0 (N=98) — real USVs or artifacts? **STILL OPEN** — Stream 3 handoff `docs/handoffs/lab_parallel/03_5970_detection_audit_a4.md` was READY but not confirmed run as of 2026-05-26.
- [x] 2026-04-18: Analysis Phase B1 — traditional taxonomy + UMAP/HDBSCAN on 3452. (done 2026-04-25, Stream 1 — 401 classified calls, `results/acoustic_feature_analysis_3452/`, corpus facts `data/corpus_facts/3452.json`)
- [x] 2026-04-18: Analysis Phase B2 — Cross-animal comparison. (done — Stream 4 `cross_population.py` commit 8def6b98 + Q1/Q2/Q3 lab-vs-wild 2026-05-16; headline 3452 vs 5970 JSD = 0.37, geometry replicates / usage diverges; see [[project_q1q2q3_lab_vs_wild]])
- [ ] 2026-04-25: Analysis Phase C — LMT behavioral correlation: PETH analysis (event_triggered.py), type-specific PETHs, USV-behavior temporal coupling. **BLOCKED** — needs `.sqlite` behavioral DB (not yet located as of 2026-05-26).
- [ ] 2026-04-25: Open questions to investigate — (1) density ridges within the UMAP manifold? (2) bout-onset signatures (3) frequency drift over hours/days (4) are Short calls real USVs? (5) Flat as default state vs communicative signal (6) sinuosity as arousal marker. See docs/analysis-roadmap.md §Open Questions. (ongoing research agenda; partially addressed by clustering work — see [[project_clustering_analysis_5970]])
- [x] 2026-04-29: Check in with Mickey — lab strain recording data status. (done — lab data arrived; `USV_lab_131204` detect+classify+cluster COMPLETE, now on Module 18.x lab CNN classifier; see [[project_lab_data_pipeline]])
- [x] 2026-04-20: SIS Benchmark 17.1 follow-up — add `syllable_type` column to real `classified_detections_full.csv` (mapping from DeepSqueak cluster names to Scattoni-7), rerun `scripts/run_sis_baselines.py` on real data, verify Scattoni-7 MI ≈ 0.093 bits (Phase A2 reproducibility check). Module code is approved (review 2026-04-17); this is the deferred exit criterion from ROADMAP_SIS_BENCHMARK.md module 17.1. (done 2026-04-18 — investigation showed `syllable_type` already present via `classified_traditional.csv`; Scattoni-7 MI is 0.0758 bits raw-consecutive vs 0.0921 bits bout-aware — not a regression, methodology difference. Reconciled in `docs/modules/sis-baselines.md` §Reproducibility Check and promoted to `data/corpus_facts/5970.json:sequential_structure_mi`; see `docs/handoffs/sis-baselines-mi-reconciliation.md`.)

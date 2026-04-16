# Reminders

<!-- Checked at session start. Due items surface in orientation. -->
<!-- Format: - [ ] YYYY-MM-DD: Description -->
<!-- Completed: - [x] YYYY-MM-DD: Description (done YYYY-MM-DD) -->

- [x] 2026-04-05: Weekly maintenance routine -- /health run 2026-04-05; follow-through (/reflect, /rethink, dangling link fixes, /reweave) done 2026-04-06
- [ ] 2026-04-12: Weekly maintenance routine -- /health, /reflect, /reweave, /stats
- [ ] 2026-04-11: Analysis Phase A1 — Temporal dynamics: call rate over time, type composition shifts across USV1–USV5 sessions, bout detection (inter-call intervals). Use begin_time_s + filename timestamps.
- [ ] 2026-04-11: Analysis Phase A2 — Sequential structure: transition matrix (P(type_B|type_A)), bigram/trigram entropy rates, idiom detection via shuffle surrogates, Zipf distribution of types. Tools: `information_theory.py`, `repertoire_stats.py`.
- [ ] 2026-04-14: Analysis Phase A3 — Acoustic feature deep-dive: feature correlations, PCA loadings, UMAP colored by individual features, within-type variability (violin plots), low-confidence boundary case inspection.
- [ ] 2026-04-14: Analysis Phase A4 — Detection quality audit: CNN confidence (det_prob_max) by syllable type, match quality cross-tab, inspect HDBSCAN noise points (N=37) and ultra-short cluster 0 (N=98) — real USVs or artifacts?
- [ ] 2026-04-18: Analysis Phase B1 — Run traditional taxonomy + UMAP/HDBSCAN on 3452 dataset (usv_lmt_035). Prerequisite: extract acoustic features from batch detections.
- [ ] 2026-04-18: Analysis Phase B2 — Cross-animal comparison: repertoire proportions (chi-squared, JSD), Shannon entropy, transition matrix differences, UMAP space overlap. Note: N=2 individuals, use descriptive not inferential stats.
- [ ] 2026-04-25: Analysis Phase C — LMT behavioral correlation: PETH analysis (event_triggered.py), type-specific PETHs, USV-behavior temporal coupling. Prerequisite: locate .sqlite behavioral database.
- [ ] 2026-04-25: Open questions to investigate — (1) density ridges within the UMAP manifold? (2) bout-onset signatures (3) frequency drift over hours/days (4) are Short calls real USVs? (5) Flat as default state vs communicative signal (6) sinuosity as arousal marker. See docs/analysis-roadmap.md §Open Questions.
- [ ] 2026-04-29: Check in with Mickey — lab strain recording data status. When it arrives, follow `HANDOFF_05_LAB_DATA_PIPELINE.md` run order. Prerequisite: Phases A+B should be complete first.

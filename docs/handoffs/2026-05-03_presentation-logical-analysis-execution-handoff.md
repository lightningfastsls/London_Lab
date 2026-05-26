# Handoff: Presentation Logical Analysis Execution
Date: 2026-05-03

## Task

Execute a logical analysis audit of presentation figure families and their implied slide claims.

This is not a PNG provenance audit. The prior provenance work established where the image files came from and whether many artifacts can be reproduced or hash-matched. This audit asks whether the analysis logic supports the scientific and presentation claims.

Do not treat a byte-identical PNG, copied result artifact, or reproducible generator as evidence that the figure is logically valid. Provenance answers "where did this image come from?" Logical analysis answers "does this figure compute the right thing, with the right assumptions, for the claim the presentation makes?"

## Read First

1. `AGENTS.md`
2. `docs/handoffs/2026-05-03_presentation-png-full-verification-results.md`
3. `docs/handoffs/2026-05-03_presentation-png-provenance-audit.md`
4. `presentation/FIGURE_GUIDE_FOR_WEB_SESSION.md`
5. `presentation/HANDOFF_FROM_CLAUDE_CODE.md`
6. `docs/handoffs/a3-acoustic-feature-deep-dive.md`
7. `docs/handoffs/a2-sequential-structure-handoff.md`
8. `docs/handoffs/sis-baselines-mi-reconciliation.md`
9. `docs/modules/sis-baselines.md`
10. `results/cross_population/wild_5970_vs_wild_3452.md`
11. `data/corpus_facts/5970.json`

Prefer current code, current result files, and `data/corpus_facts/5970.json` over older prose when they disagree. For example, the SIS reconciliation handoff records an earlier raw-consecutive vs bout-aware mismatch, while the current SIS module docs and corpus facts record the 2026-04-19 bout-aware rerun.

## Core Distinctions

Keep these four validation layers separate in every family verdict:

| Layer | Question | Evidence type |
| --- | --- | --- |
| Artifact reproducibility | Can the PNG be traced, copied, or regenerated? | PNG hashes, generator scripts, source data paths |
| Code implementation correctness | Does the script compute what it claims to compute? | Script logic, filters, joins, thresholds, grouping, row counts |
| Statistical / methodological validity | Are sample size, independence, uncertainty, and estimator assumptions appropriate? | Result summaries, confidence intervals, tests, permutation/bootstrap design |
| Presentation framing validity | Would the slide/caption lead an audience to infer only what the analysis supports? | Figure guide text, slide narrative, captions, caveats, claim scope |

The executor should cite which layer failed. A figure can be reproducible but still need a caption fix, analysis fix, or removal.

## Evaluation Rubric

Score each figure family on the dimensions below. Use `pass`, `caveat`, `fail`, or `insufficient evidence` for each dimension, then assign one overall risk classification.

### 1. Claim Clarity

- What claim does the slide or guide imply?
- Is the claim descriptive, causal, comparative, methodological, or speculative?
- Is the scope correct: window, call, bout, file, animal/couple, dataset, population, or future lab-vs-wild comparison?

### 2. Implementation Match

- Does the generating script compute the quantity needed by the claim?
- Are the right source rows, joins, filters, labels, thresholds, time windows, sample rate assumptions, and grouping variables used?
- Does the plotted quantity match the caption, labels, and narrative text?

### 3. Statistical Validity

- Are sample sizes and units of independence explicit?
- Are inferential tests used only where justified?
- Are descriptive comparisons framed as descriptive where N is too small?
- Are uncertainty intervals, bootstraps, shuffles, permutation tests, or sensitivity analyses interpreted in their valid scope?

### 4. Biological / Experimental Framing

- Does the figure distinguish animal, cage/couple, recording session, file, and population correctly?
- Does it avoid overclaiming wild-vs-lab when only wild-vs-wild data are present?
- Does it preserve the caveat that cross-population figures are a dry run until lab data arrives?

### 5. Methodological Consistency

- Are band limits, `sr=300000`, STFT/rendering assumptions, labels, and bout definitions consistent with code truth?
- Are raw consecutive, bout-aware, file-aware, no-file-aware, and within-file analyses distinguished?
- Are UMAP, HDBSCAN, k-means, and rule-taxonomy methods described without implying more certainty than they support?

### 6. Presentation Claim Support

- Does the figure actually support the slide message?
- Would a lab audience likely infer something stronger or different than the code supports?
- Should the slide be relabeled, caveated, replaced, or kept?

### 7. Risk Classification

Assign exactly one:

- `sound`
- `sound with caveat`
- `needs caption fix`
- `needs analysis fix`
- `do not use as currently framed`
- `insufficient evidence`

## Figure Families To Audit

Audit these in priority order.

### 1. `presentation/figures/03_training_data/`

Known issue: `presentation/figures/03_training_data/training_set_composition.png` is generated from `spectrograms_training/{train,val,test}.csv`, not the production corpus in `data/training/matched_windows_v2/{train,val,test}.csv`.

Evidence to inspect:

- `presentation/figures/_generate.py`
- `spectrograms_training/train.csv`
- `spectrograms_training/val.csv`
- `spectrograms_training/test.csv`
- `data/training/matched_windows_v2/train.csv`
- `data/training/matched_windows_v2/val.csv`
- `data/training/matched_windows_v2/test.csv`
- `presentation/HANDOFF_FROM_CLAUDE_CODE.md`
- `presentation/FIGURE_GUIDE_FOR_WEB_SESSION.md`

Questions:

- Is the slide claiming the production CNN training corpus, the labeling-app working corpus, or both?
- Does the chart answer the slide's intended question?
- If the figure is kept, what exact caption prevents the audience from hearing "15,444 production windows" when the chart shows 1,379 working-corpus rows?
- Should the slide use a regenerated production-corpus chart or a text/table summary of the v1 to v2 hard-negative mining delta?

### 2. `presentation/figures/09_umap/`

Includes UMAP by type/feature, HDBSCAN, k-means, contingency matrix, boundary cases, and galleries.

Evidence to inspect:

- `scripts/analyze_acoustic_features.py`
- `scripts/recluster_umap_hdbscan.py`
- `scripts/generate_cluster_gallery.py`
- `results/traditional_taxonomy/classified_traditional.csv`
- `results/acoustic_feature_analysis/analysis_summary.md`
- `results/acoustic_feature_analysis/umap_coordinates.csv`
- `results/recluster_umap_hdbscan/cluster_summary.csv`
- `results/recluster_umap_hdbscan/reclassified_detections.csv`
- `docs/handoffs/a3-acoustic-feature-deep-dive.md`

Questions:

- Are continuum claims supported by UMAP on 10 standardized DeepSqueak acoustic features, not CNN embeddings?
- Does HDBSCAN actually support "essentially one cluster," and are noise/outlier clusters framed correctly?
- Does k-means framing make clear that a forced partition can create apparent clusters?
- Are low-confidence boundary cases logically tied to classifier threshold proximity rather than independent biological ambiguity?
- Are gallery examples treated as illustrative examples, not statistical summaries?
- Are 20-125 kHz gallery render bands distinguished from 20-120 kHz CNN input and broader 30-110 kHz prose?

### 3. `presentation/figures/10_temporal_dynamics/`

Includes call rate, hourly type composition, ICI distribution, bout structure, and raster.

Evidence to inspect:

- `scripts/analyze_temporal_dynamics.py`
- `results/traditional_taxonomy/classified_traditional.csv`
- `results/temporal_dynamics/temporal_summary.csv`
- `data/corpus_facts/5970.json`
- `docs/handoffs/a2-sequential-structure-handoff.md`

Questions:

- Does timestamp parsing use filename datetime plus `begin_time_s`, rather than USV folder structure?
- Are USV1-USV5 folders avoided as session units?
- Is the 0.6 s bout threshold derived and described accurately?
- Are claims about burstiness, stability of type composition, and silent periods descriptive rather than overgeneralized?
- Are file boundary and cross-file interval effects handled or caveated?

### 4. `presentation/figures/10_sequential_structure/`

Includes transition matrix, entropy convergence, MI lag, Zipf, and bout threshold sensitivity.

Evidence to inspect:

- `scripts/analyze_sequential_structure.py`
- `scripts/bout_threshold_sensitivity.py`
- `src/usv_spectrogram/classification/repertoire_stats.py`
- `usv_language/analysis/sequence_analysis.py`
- `usv_language/analysis/information_theory.py`
- `results/sequential_structure/sequential_structure_summary.csv`
- `results/sequential_structure/idiom_report.csv`
- `results/sequential_structure/ici_gap.npy`
- `results/sequential_structure/ici_onset.npy`
- `data/corpus_facts/5970.json`
- `docs/handoffs/a2-sequential-structure-handoff.md`

Questions:

- Does the transition matrix use within-bout pairs and row-stochastic P(B|A)?
- Are self-repetition and "sticky state" claims compared against marginal expectations, not just diagonal visual salience?
- Is MI interpreted as small but nonzero uncertainty reduction, not strong syntax?
- Are lag-10 upticks or Zipf plots caveated as possible artifacts or underpowered where appropriate?
- Do sensitivity plots distinguish file-aware/no-file-aware and within-file analyses?
- Is `n_within_bout_pairs` consistent across scripts, results, and corpus facts or explained when it differs?

### 5. `presentation/figures/11_sis_baselines/`

Evidence to inspect:

- `scripts/run_sis_baselines.py`
- `src/usv_spectrogram/classification/sis_baselines.py`
- `results/sis_baselines/baselines.csv`
- `results/sis_baselines/parameters.json`
- `data/corpus_facts/5970.json`
- `docs/modules/sis-baselines.md`
- `docs/handoffs/sis-baselines-mi-reconciliation.md`
- `docs/handoffs/sis-baselines-17.1-bout-filter-rerun.md`

Questions:

- Does the current chart use bout-aware or raw-consecutive MI?
- Is Scattoni-7 MI aligned with Phase A2 and `data/corpus_facts/5970.json`?
- Are Hertz reference lines methodologically comparable under the current bout-filtered run?
- Is "DeepSqueak-27 wins" framed fairly, given alphabet-size effects and forced k-means labels?
- Is HDBSCAN low MI framed as expected from near-single-cluster labeling, not as proof that acoustic continuity implies no behavioral structure?

### 6. `presentation/figures/12_cross_population/`

Evidence to inspect:

- `src/usv_spectrogram/classification/cross_population.py`
- `results/cross_population/wild_5970_vs_wild_3452.json`
- `results/cross_population/wild_5970_vs_wild_3452.md`
- `results/traditional_taxonomy/classified_traditional.csv`
- `results/traditional_taxonomy_3452/classified_traditional.csv`
- `data/corpus_facts/5970.json`

Questions:

- Is every slide and caption explicit that this is wild 5970 vs wild 3452, not lab-vs-wild?
- Are N=2 couple-level limitations stated plainly?
- Are chi-square, Cramer's V, Cohen's h, JSD, feature effects, transition distances, and IOI differences interpreted as between-couple descriptive/diagnostic comparisons, not population generalization?
- Are small-count transition rows, especially rare labels in 3452, caveated?
- Is the dry-run value for the future lab comparison clear?

### 7. `presentation/figures/05_signal_detection/` and `presentation/figures/06_deepsqueak_validation/`

Evidence to inspect:

- `scripts/evaluate_model.py`
- `src/usv_spectrogram/models/evaluate.py`
- `models/hard_neg_retrain/evaluation/test_metrics.json`
- `presentation/figures/05_signal_detection/test_metrics.json`
- `presentation/figures/_generate.py`
- `results/validation_new_filter/comparison_report.csv`
- `results/validation_new_filter/comparison_summary.json`
- `results/validation_old_filter/comparison_summary.json`
- `presentation/FIGURE_GUIDE_FOR_WEB_SESSION.md`

Questions:

- Are CNN test-set claims implemented on the intended held-out split and production checkpoint?
- Are PR-AUC and ROC-AUC interpreted appropriately under class imbalance?
- Is DeepSqueak agreement treated as independent corroboration, not ground truth?
- Are IoU definitions, agreement counts, CNN-only, and DeepSqueak-only events interpreted correctly?
- Does the FP-filter v1/v2 figure distinguish model, post-processing, and validation-system effects?

### 8. `presentation/figures/01_raw_signal_pipeline/`, `presentation/figures/00_raw_waveform.png`, and `presentation/figures/00_raw_waveform_zoomed.png`

Evidence to inspect:

- `presentation/HANDOFF_FROM_CLAUDE_CODE.md`
- `presentation/FIGURE_GUIDE_FOR_WEB_SESSION.md`
- `results/pipeline_comparison/1_raw_cnn/2024-10-01_18-51-28_0006209.json`
- `results/pipeline_comparison/2_hysteresis_only/2024-10-01_18-51-28_0006209.json`
- `results/pipeline_comparison/3_full_pipeline/2024-10-01_18-51-28_0006209.json`
- `results/pipeline_comparison/1_raw_cnn/2024-10-01_17-46-45_0005839.json`
- `results/pipeline_comparison/2_hysteresis_only/2024-10-01_17-46-45_0005839.json`
- `results/pipeline_comparison/3_full_pipeline/2024-10-01_17-46-45_0005839.json`
- `src/usv_spectrogram/corpus.py`
- `src/usv_spectrogram/detection/extraction_config.py`
- `src/usv_spectrogram/app/core/sliding_inference.py`
- `scripts/run_batch_detection.py`

Questions:

- Does the raw waveform figure logically support "USVs are hidden in waveform space" for the chosen example?
- Does the spectrogram/pipeline sequence show the right pipeline stages and event counts?
- Are raw CNN, hysteresis, FP filter, and full pipeline described as separate stages?
- Are production thresholds and hysteresis parameters stated correctly, without implying a single bare CNN threshold is the deployed detector?
- Are claims limited to these illustrative examples unless aggregate performance evidence is also cited?

## Output Format

Write a new handoff under `docs/handoffs/` with this title pattern:

`docs/handoffs/2026-05-03_presentation-logical-analysis-results.md`

Start with a compact summary table:

| Family | Intended claim | Code computes | Logic verdict | Risk | Required slide/caption change | Evidence checked |
| --- | --- | --- | --- | --- | --- | --- |

For every `needs analysis fix`, `do not use as currently framed`, or `insufficient evidence` result, include a detail block:

```markdown
### Figure / Family

Claim:

Implementation:

Logical assessment:

Overclaim risk:

Recommended fix:

Evidence:
```

Also include:

- A "Do Not Infer" section listing claims the current presentation should not make.
- A "Caption Fix Queue" section with exact replacement caption language where possible.
- A "Needs Analysis Fix" section for any code/result problems that should be fixed separately.
- A "Keep As-Is" section for families that are logically sound.
- A "Validation" section recording commands and readbacks performed.

## Validation Requirements For Executor

For docs-only audit output:

1. Re-read the created results handoff.
2. Verify every referenced file/path in the results handoff exists unless explicitly marked optional or missing.
3. Confirm the results handoff does not use the PNG provenance audit as evidence of logical correctness.
4. Confirm each finding separates artifact reproducibility, implementation correctness, statistical/methodological validity, and presentation framing validity.
5. Where a result file is CSV/JSON, inspect the actual columns/keys used for the claim.
6. Where code and prose disagree, cite the code/result file as the controlling evidence and mention the stale prose.

If code bugs are found, report them in the results handoff. Do not patch analysis scripts in the audit chat unless the user explicitly asks for fixes.

## Priority Order

1. `03_training_data/` because a known corpus mismatch can directly mislead the audience.
2. `12_cross_population/` because wild-vs-lab overclaim risk is scientifically high.
3. `11_sis_baselines/` because methodology changed from historical raw-consecutive to current bout-aware comparison.
4. `09_umap/` because the continuum claim is central and easy to overstate.
5. `10_sequential_structure/` because syntax/MI claims require careful estimator scope.
6. `10_temporal_dynamics/` because timestamp and bout framing affect all downstream sequence claims.
7. `05_signal_detection/` and `06_deepsqueak_validation/` because validation claims must not imply ground truth beyond their evidence.
8. `01_raw_signal_pipeline/` and raw waveform figures because they are illustrative, but still need careful stage and threshold language.

## Open Questions / Known Risks

- The exact slide deck text may differ from `presentation/FIGURE_GUIDE_FOR_WEB_SESSION.md`. Treat the guide as the best available proxy unless a deck file is provided.
- Some handoffs are historical and may be stale. Prefer code and current result files.
- Several figures are copied artifacts with strong provenance; that does not imply the slide claim is sound.
- Logical analysis may identify follow-up implementation work. Keep audit and repair work separate unless directed otherwise.

## Worth Remembering For Claude

- The user asked for logical analysis criteria and execution, not another PNG hash pass.
- The highest-risk presentation failures are scope errors: working corpus vs production corpus, wild-vs-wild vs wild-vs-lab, and raw-consecutive vs bout-aware sequence metrics.
- A useful audit should make every recommended slide change concrete enough that Mickey can edit the deck without redoing the analysis.

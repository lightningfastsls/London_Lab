# Handoff: Presentation Logical Analysis Results
Date: 2026-05-03

## Task

Execute a logical analysis audit of presentation figure families and implied slide claims, separating artifact reproducibility from code implementation correctness, statistical/methodological validity, and presentation framing validity.

## Files Changed

- `docs/handoffs/2026-05-03_presentation-logical-analysis-results.md` - new audit handoff with verdicts, caption fixes, and recommended analysis fixes.

## Summary Table

| Family | Intended claim | Code computes | Logic verdict | Risk | Required slide/caption change | Evidence checked |
| --- | --- | --- | --- | --- | --- | --- |
| `03_training_data/` | Production CNN training corpus and hard-negative mining story | `presentation/figures/_generate.py` plots `spectrograms_training/{train,val,test}.csv` = 1,379 working-corpus rows | Implementation is correct for the wrong corpus; method/statistics are simple counts; presentation framing fails if audience hears "15,444 production windows" | do not use as currently framed | Replace with production `matched_windows_v2` chart/table, or explicitly label as "labeling-app working corpus" | `_generate.py`; `spectrograms_training/*.csv`; `data/training/matched_windows*.csv`; presentation guide/handoff |
| `12_cross_population/` | Detect between-animal differences; dry run for future lab-vs-wild | Wild 5970 vs wild 3452 descriptive comparison with bootstrap/permutation summaries | Code/report are explicit wild-vs-wild; stats are useful diagnostics but not population generalization | needs caption fix | Every slide title/caption must say "wild 5970 vs wild 3452; N=2 couples; dry run" | `cross_population.py`; `wild_5970_vs_wild_3452.{json,md}`; classified CSVs |
| `11_sis_baselines/` | Existing labelings differ in how much sequential structure they capture | Current SIS chart uses bout-aware MI at 0.6 s, 6,350 within-bout pairs | Current code/results are methodology-aligned with Phase A2; guide prose is stale where it says raw-consecutive | needs caption fix | State "bout-filtered depth-1 MI"; caveat alphabet-size effects for DeepSqueak-27 | `run_sis_baselines.py`; `sis_baselines.py`; `baselines.csv`; `parameters.json`; corpus facts; SIS docs |
| `09_umap/` | USV acoustic feature space is continuous rather than naturally split into 7/27 categories | UMAP on 10 standardized DeepSqueak acoustic features; HDBSCAN on 8D UMAP; k-means labels shown as forced partition | Continuum claim is supported as an acoustic-feature/manifold claim, not CNN-embedding or biological-kind proof | sound with caveat | Say "one dominant acoustic continuum with small outlier clusters"; galleries illustrative only | `analyze_acoustic_features.py`; `recluster_umap_hdbscan.py`; `generate_cluster_gallery.py`; A3 summary; UMAP/HDBSCAN CSVs |
| `10_sequential_structure/` | Calls have shallow sequential structure: sticky states, small MI, no strong syntax claim | Within-bout transition matrix and MI curves using 0.6 s silent-gap bout filter | Implementation matches the claim; overclaim risk is in syntax/memory/Zipf/idiom framing | needs caption fix | Use "small but nonzero predictability"; avoid "syntax" unless heavily qualified | `analyze_sequential_structure.py`; `bout_threshold_sensitivity.py`; summary CSV; `idiom_report.csv`; corpus facts |
| `10_temporal_dynamics/` | Calling is bursty; type composition is descriptive over time; bouts motivate sequence filtering | Timestamp = filename datetime + `begin_time_s`; A1 bout chart uses auto 0.579 s onset-to-onset threshold | Timeline logic is sound; bout methodology is inconsistent with canonical A2/SIS silent-gap 0.6 s facts | needs analysis fix | Either rerun/replace bout panels with canonical silent-gap 0.6 s or caption A1 as separate onset-ICI exploratory analysis | `analyze_temporal_dynamics.py`; `temporal_summary.csv`; corpus facts; A2 handoff |
| `05_signal_detection/` and `06_deepsqueak_validation/` | CNN test performance is strong; DeepSqueak independently corroborates detections | Held-out test metrics for hard-neg retrain; DeepSqueak overlap categories on 197 files | Code and counts are coherent; DS agreement is corroboration, not ground truth | sound with caveat | Prefer confusion matrix + PR curve; caption DS as independent detector agreement | `evaluate_model.py`; `models/evaluate.py`; metrics JSONs; validation summaries |
| `01_raw_signal_pipeline/` and raw waveform | USVs are hard to see in waveform; deployed detection is staged | Example WAV/pipeline JSONs show raw CNN, hysteresis, and full FP-filter stages | Good illustrative examples, not aggregate performance evidence | sound with caveat | State "illustrative example" and separate raw CNN, hysteresis, FP filter, full pipeline | pipeline JSONs; production configs; `run_batch_detection.py`; corpus/extraction constants |

## Detail Blocks

### `03_training_data/`

Claim:

The slide appears intended to describe the production CNN corpus and hard-negative mining evolution.

Implementation:

`presentation/figures/_generate.py` reads `spectrograms_training/{train,val,test}.csv` and counts `label == "USV"`. Those files contain train 376/476, val 155/210, test 65/97 USV/non-USV rows, total 1,379. The production hard-negative-retrain corpus is `data/training/matched_windows_v2/{train,val,test}.csv`, total 15,444 rows: train 4,377/7,099, val 872/1,267, test 541/1,288. The v1-to-v2 train delta is +620 noise negatives and +144 USV positives.

Logical assessment:

- Artifact reproducibility: the PNG is reproducible/copyable, but that does not support the production-corpus claim.
- Code implementation correctness: correct for `spectrograms_training`; wrong source for the intended production slide.
- Statistical/methodological validity: simple row counts are valid; no inference issue.
- Presentation framing validity: fails unless the slide explicitly says this is the labeling-app working corpus.

Overclaim risk:

High. A lab audience will likely hear "the CNN saw 1,379 windows" or "this is the production training set," both misleading for the current model.

Recommended fix:

Replace the figure with either a production-corpus table/chart or a v1-to-v2 delta table. Exact caption if the existing figure is kept: "Labeling-app working corpus only: 1,379 manually labeled candidate windows in `spectrograms_training/`; the production hard-negative-retrain CNN used 15,444 windows from `data/training/matched_windows_v2/`."

Evidence:

`presentation/figures/_generate.py`; `spectrograms_training/train.csv`; `spectrograms_training/val.csv`; `spectrograms_training/test.csv`; `data/training/matched_windows/train.csv`; `data/training/matched_windows_v2/train.csv`; `presentation/HANDOFF_FROM_CLAUDE_CODE.md`.

### `12_cross_population/`

Claim:

The figure family should show that the comparison machinery detects differences between two wild-mouse couples and is ready for the future lab-vs-wild comparison.

Implementation:

`src/usv_spectrogram/classification/cross_population.py` labels the comparison `wild_5970` vs `wild_3452`, records `strata_note = "wild-vs-wild between-couple"`, and writes that caveat into the Markdown/JSON. It computes type proportions, JSD, entropy, bout-aware transitions, bout-aware MI, Zipf sentinel values, burstiness, IOI, and acoustic-feature effects.

Logical assessment:

- Artifact reproducibility: figure PNGs are traceable; irrelevant to claim strength.
- Code implementation correctness: strong. Inputs are the intended classified CSVs and report metadata makes the stratum explicit.
- Statistical/methodological validity: descriptive diagnostics are fine; chi-square p-values and KS p-values should not be framed as wild-population inference because the unit is two couples.
- Presentation framing validity: fragile if any slide says or implies "lab vs wild" or "population difference."

Overclaim risk:

High if the deck uses "cross-population" without "wild-vs-wild." Rare 3452 rows are especially noisy: Frequency_Jump has only 9 calls and transition matrix B has 252 within-bout pairs total.

Recommended fix:

Exact caption: "Wild 5970 vs wild 3452, two wild-mouse couples. These are descriptive between-couple differences and a dry run for the future lab-vs-wild analysis; they are not evidence of lab-vs-wild divergence."

Evidence:

`results/cross_population/wild_5970_vs_wild_3452.json`; `results/cross_population/wild_5970_vs_wild_3452.md`; `src/usv_spectrogram/classification/cross_population.py`; `results/traditional_taxonomy/classified_traditional.csv`; `results/traditional_taxonomy_3452/classified_traditional.csv`.

### `11_sis_baselines/`

Claim:

Different labelings capture different amounts of one-step sequential structure, benchmarked against Hertz-style SIS references.

Implementation:

Current `results/sis_baselines/parameters.json` records bout detection applied at threshold 0.6 s from corpus facts, with cross-file pairs excluded via `+inf` gaps. All three labelings use 6,350 within-bout pairs and 1,513 excluded pairs. `baselines.csv` reports Scattoni-7 = 0.092051 bits, DeepSqueak-27 = 0.351802 bits, HDBSCAN-3 = 0.037021 bits. This matches `data/corpus_facts/5970.json` and `docs/modules/sis-baselines.md`, not the older raw-consecutive handoff/guide prose.

Logical assessment:

- Artifact reproducibility: the chart has provenance; not the basis for the verdict.
- Code implementation correctness: current script/result sidecar compute the intended bout-aware comparison.
- Statistical/methodological validity: Hertz reference lines are now method-family aligned, but exact threshold differs from Hertz's 160 ms and should not be sold as identical methodology.
- Presentation framing validity: stale guide text saying "raw consecutive" must be removed. "DeepSqueak-27 wins" needs an alphabet-size caveat.

Overclaim risk:

Moderate. DeepSqueak-27 has higher MI partly because a 27-label alphabet can encode finer local persistence; HDBSCAN low MI follows from near-single-cluster labels and does not prove that acoustic continuity has no behavioral structure.

Recommended fix:

Exact caption: "Bout-filtered depth-1 SIS / MI on within-bout pairs only (0.6 s silent-gap threshold; 6,350 pairs). Scattoni-7 matches Phase A2 at 0.092 bits. DeepSqueak-27 is higher partly because it uses a finer 27-label alphabet; HDBSCAN is low because it collapses most calls into one dominant acoustic cluster."

Evidence:

`results/sis_baselines/baselines.csv`; `results/sis_baselines/parameters.json`; `data/corpus_facts/5970.json`; `docs/modules/sis-baselines.md`; `scripts/run_sis_baselines.py`; `src/usv_spectrogram/classification/sis_baselines.py`; stale contrast in `docs/handoffs/sis-baselines-mi-reconciliation.md` and `presentation/FIGURE_GUIDE_FOR_WEB_SESSION.md`.

### `10_temporal_dynamics/`

Claim:

Calling is bursty, has silent periods, and type composition is fairly stable over active hours; the ICI/bout view motivates within-bout sequence analysis.

Implementation:

Timestamp parsing is correct for the stated claim: both A1 and A2 parse the datetime from `file` names and add `begin_time_s`, avoiding USV1-USV5 folder structure as session units. The issue is bout methodology. `results/temporal_dynamics/temporal_summary.csv` reports auto threshold 0.578959 s and 1,579 bouts from onset-to-onset intervals. The canonical A2/SIS/corpus path uses a 0.6 s silent-gap threshold, 1,238 bouts, and 6,350 within-bout pairs.

Logical assessment:

- Artifact reproducibility: temporal PNGs are traceable to the script/results.
- Code implementation correctness: call-rate/raster/type-composition logic matches the descriptive claims; A1 bout detection computes a different quantity from A2/SIS.
- Statistical/methodological validity: descriptive temporal claims are valid for 5970; bout-derived claims require explicit threshold/gap semantics.
- Presentation framing validity: if the slide says the ICI figure directly justifies the canonical 0.6 s silent-gap sequence filter, it conflates onset IOI with silent gap.

Overclaim risk:

Moderate. The burstiness claim is fine; the "0.6 s bout threshold is data-driven" claim needs a precise statement of which interval is plotted and which interval is used downstream.

Recommended fix:

Either regenerate A1 bout/ICI panels using the canonical silent-gap 0.6 s method, or caption the current plots exactly: "Temporal overview uses onset-to-onset ICI with an auto threshold of 0.579 s; sequence/SIS analyses use the canonical 0.6 s silent-gap filter."

Evidence:

`scripts/analyze_temporal_dynamics.py`; `results/temporal_dynamics/temporal_summary.csv`; `scripts/analyze_sequential_structure.py`; `results/sequential_structure/sequential_structure_summary.csv`; `data/corpus_facts/5970.json`; `docs/handoffs/a2-sequential-structure-handoff.md`.

### `10_sequential_structure/`

Claim:

Within-bout call order is not independent: self-repetition/sticky states exist, but the effect is shallow.

Implementation:

`scripts/analyze_sequential_structure.py` sorts by filename-derived absolute time, builds within-bout sequences at `BOUT_THRESHOLD_S = 0.6`, computes a row-stochastic transition matrix, computes MI at lags 1-10, compares mean self-transition against an independence baseline `sum(p_i^2)`, and writes summary metrics. `sequential_structure_summary.csv` reports lag-1 MI 0.0921 bits, marginal entropy 2.5436, conditional entropy 2.4499, entropy reduction 3.69%, mean self-transition 0.2573 vs chance 0.2004, and Zipf sentinel alpha 0.0.

Logical assessment:

- Artifact reproducibility: strong provenance, not used as logic proof.
- Code implementation correctness: matches transition and MI claims; self-transition baseline is present.
- Statistical/methodological validity: MI is small but nonzero; idiom counts and high-order entropy are sensitive to sparse n-grams and shuffle design; Zipf with 7 types is not meaningful.
- Presentation framing validity: "syntax" and "memory depth 10" are too strong without caveat. Lag-10 uptick should be called exploratory/artifact-prone.

Overclaim risk:

Moderate. The data supports shallow local predictability, not a strong syntax system.

Recommended fix:

Exact caption for transition/MI slides: "Within-bout transitions only (0.6 s silent-gap threshold). Previous-call identity reduces uncertainty by 0.092 bits, about 3.7%; this is shallow local predictability, not evidence for strong syntax."

Evidence:

`scripts/analyze_sequential_structure.py`; `scripts/bout_threshold_sensitivity.py`; `results/sequential_structure/sequential_structure_summary.csv`; `results/sequential_structure/idiom_report.csv`; `data/corpus_facts/5970.json`.

## Do Not Infer

- Do not infer that `training_set_composition.png` shows the production CNN training corpus.
- Do not infer lab-vs-wild differences from `wild_5970_vs_wild_3452`; both cohorts are wild couples.
- Do not infer that UMAP/HDBSCAN proves mouse USVs have no biologically meaningful categories; it supports a continuous acoustic-feature manifold for these features.
- Do not infer that low-confidence boundary cases are independent biological ambiguity; they are proximity to rule thresholds by construction.
- Do not infer strong syntax from 0.092 bits MI or from visually salient transition-matrix diagonals.
- Do not infer that Zipf fits are meaningful with only 7 syllable types.
- Do not infer DeepSqueak is ground truth; use it as independent detector corroboration.
- Do not infer the deployed detector is a single bare CNN threshold; production is CNN probabilities plus temperature scaling, hysteresis, FP filter, and triage/output logic.

## Caption Fix Queue

- Training existing chart: "Labeling-app working corpus only: 1,379 manually labeled candidate windows in `spectrograms_training/`; the production hard-negative-retrain CNN used 15,444 windows from `data/training/matched_windows_v2/`."
- Training replacement chart: "Production hard-negative-retrain corpus: 15,444 windows (5,790 USV, 9,654 non-USV). The v2 train split added 620 hard negatives and 144 recovered hard positives; validation/test stayed fixed."
- Cross-population: "Wild 5970 vs wild 3452, two wild-mouse couples. Descriptive between-couple differences and dry-run diagnostics for future lab-vs-wild analysis, not population-level inference."
- SIS: "Bout-filtered depth-1 SIS / MI on within-bout pairs only (0.6 s silent-gap threshold; 6,350 pairs). DeepSqueak-27 has a finer alphabet; HDBSCAN collapses most calls into one dominant acoustic cluster."
- UMAP: "UMAP of 10 standardized DeepSqueak acoustic features, not CNN embeddings. HDBSCAN finds one dominant acoustic cluster plus small outlier/noise groups; k-means partitions are forced labels on the same continuum."
- Boundary cases: "Low-confidence calls are threshold-proximity cases from the rule classifier; they localize arbitrary rule boundaries, not independent proof of biological ambiguity."
- Temporal ICI/bouts: "Temporal overview uses onset-to-onset ICI with an auto threshold of 0.579 s; sequence/SIS analyses use the canonical 0.6 s silent-gap filter."
- Sequential structure: "Within-bout transitions only. Previous-call identity reduces next-call uncertainty by 0.092 bits, about 3.7%; this is shallow local predictability, not strong syntax."
- Signal detection: "Held-out window-level CNN test performance on `data/training/matched_windows_v2/test.csv`; PR-AUC is the preferred curve under class imbalance."
- DeepSqueak validation: "Independent detector agreement on 197 files. DeepSqueak corroborates event reality but is not treated as ground truth."
- Raw pipeline: "Illustrative example of production stages: raw CNN probabilities, hysteresis event formation, then FP-filtered full pipeline."

## Needs Analysis Fix

- Replace or regenerate `03_training_data/training_set_composition.png` if the production training corpus is the slide claim. The current generator reads the wrong directory for that claim.
- Align temporal bout figures with canonical A2/SIS semantics or keep them explicitly separate. The current temporal output reports 0.579 s onset-ICI threshold and 1,579 bouts; corpus/A2/SIS reports 0.6 s silent-gap threshold and 1,238 bouts / 6,350 within-bout pairs.
- Consider adding small figure annotations or sidecar notes for `12_cross_population/transitions.png` warning that rare 3452 transition rows, especially Frequency_Jump with 9 calls, are noisy.
- Do not patch these during the audit unless requested; the repairs should be a separate implementation pass.

## Keep As-Is

- `05_signal_detection/` is logically usable if framed as held-out window-level model evaluation and if PR-AUC is prioritized over ROC-AUC under imbalance.
- `06_deepsqueak_validation/` is logically usable if framed as independent detector corroboration, not ground truth.
- `01_raw_signal_pipeline/` and `00_raw_waveform*.png` are logically usable as illustrative examples, with captions that separate raw CNN, hysteresis, FP filter, and full pipeline.
- `09_umap/` is logically usable for a continuum claim when scoped to 10 standardized DeepSqueak acoustic features and when HDBSCAN/k-means/galleries are caveated as above.

## Reasoning

The main failures are scope/framing failures, not PNG provenance failures. Current code and result files support many descriptive claims, but the deck must be precise about units: working corpus vs production corpus, wild couple vs population, raw consecutive vs bout-aware pairs, onset-to-onset ICI vs silent gap, and acoustic-feature UMAP vs CNN embeddings.

Where old handoff prose disagreed with current code/results, this audit treats current result artifacts and `data/corpus_facts/5970.json` as controlling evidence. The clearest example is SIS: older prose says raw-consecutive 0.0758 bits, while current `parameters.json`, module docs, and corpus facts record bout-filtered 0.0921 bits as the current run.

## Validation

Docs-only validation performed:

- Re-read the created handoff after writing.
- Inspected actual CSV/JSON columns and keys for:
  - `spectrograms_training/{train,val,test}.csv`
  - `data/training/matched_windows/{train,val,test}.csv`
  - `data/training/matched_windows_v2/{train,val,test}.csv`
  - `results/acoustic_feature_analysis/umap_coordinates.csv`
  - `results/recluster_umap_hdbscan/cluster_summary.csv`
  - `results/recluster_umap_hdbscan/reclassified_detections.csv`
  - `results/traditional_taxonomy/classified_traditional.csv`
  - `results/sis_baselines/baselines.csv`
  - `results/sis_baselines/parameters.json`
  - `results/sequential_structure/sequential_structure_summary.csv`
  - `results/temporal_dynamics/temporal_summary.csv`
  - `results/cross_population/wild_5970_vs_wild_3452.{json,md}`
  - `models/hard_neg_retrain/evaluation/test_metrics.json`
  - `results/validation_new_filter/comparison_{report.csv,summary.json}`
  - `results/validation_old_filter/comparison_{report.csv,summary.json}`
- Read implementation evidence from:
  - `presentation/figures/_generate.py`
  - `scripts/analyze_acoustic_features.py`
  - `scripts/recluster_umap_hdbscan.py`
  - `scripts/generate_cluster_gallery.py`
  - `scripts/analyze_temporal_dynamics.py`
  - `scripts/analyze_sequential_structure.py`
  - `scripts/bout_threshold_sensitivity.py`
  - `scripts/run_sis_baselines.py`
  - `src/usv_spectrogram/classification/sis_baselines.py`
  - `src/usv_spectrogram/classification/cross_population.py`
  - `scripts/evaluate_model.py`
  - `src/usv_spectrogram/models/evaluate.py`
  - `scripts/run_batch_detection.py`
  - `src/usv_spectrogram/corpus.py`
  - `src/usv_spectrogram/detection/extraction_config.py`
  - `src/usv_spectrogram/app/core/sliding_inference.py`
- Verified every path named in this handoff exists.
- Confirmed this audit does not use PNG hash/provenance as evidence of logical correctness; provenance is mentioned only as a separate layer.
- Confirmed each high-risk finding separates artifact reproducibility, code implementation correctness, statistical/methodological validity, and presentation framing validity.

## Open Questions / Known Risks

- The actual slide deck text may differ from `presentation/FIGURE_GUIDE_FOR_WEB_SESSION.md`; this audit treats the guide as the best available proxy.
- Cross-population inferential language depends heavily on slide wording. The code/report are careful, but the plotted figures themselves have terse titles.
- The temporal bout-methodology mismatch may already be acceptable if Mickey intentionally treats A1 and A2 as separate exploratory views; the deck must then say that explicitly.

## Worth Remembering For Claude

- The highest-priority deck repair is the training-data slide: current chart is the working corpus, not the production corpus.
- The cross-population section is scientifically usable only as wild-vs-wild dry-run framing.
- Current SIS results are bout-aware and aligned with Phase A2; older raw-consecutive prose is stale.
- The safest scientific phrasing is "continuous acoustic-feature manifold with shallow local sequence predictability," not "no categories" or "syntax."

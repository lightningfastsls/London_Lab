# PI Briefing — 2026-04-27

**Audience:** Mickey London
**Last formal status:** `docs/PI_PRESENTATION_SUMMARY.md` (2026-02-28)
**Window covered:** Feb 28 – Apr 27 (8 weeks)

---

## TL;DR (60 seconds)

1. **Three wild-mouse cohorts now in the pipeline.** 5970 and 3452 fully analyzed (Phases A1–A3); 9252 detection done, classification + A3 in progress.
2. **First cross-cohort comparison done — 3452 vs 5970.** Acoustic *geometry* replicates cleanly (same correlations, same PCA, same continuum). Repertoire *usage* diverges sharply: 3452 is Short-dominated (43%), 5970 is Flat-dominated (32%). JSD = 0.37 bits, χ² p ≈ 2e-79.
3. **Critical framing:** these are two *wild-mouse couples* — N=1 couple per cohort, both wild. So the divergence we see is the **between-couple noise floor for the wild stratum**, not yet a wild-vs-lab signal. The actual research question is blocked on lab-strain data.
4. **9252 surprise:** the animal is genuinely *quiet* — 7.6× lower file-yield than 5970, but session USV3 alone produces 48% of its events. We've ruled out noise-floor and seasonality as explanations.
5. **Production CNN swapped** to `hard_neg_retrain` model: precision 90.55% (+3.35 pp over previous), 16/18 known noise files eliminated, 98.7% USV rate in manual review.

---

## What's been delivered since last PI meeting (Feb 28 → Apr 27)

### Pipeline & infrastructure
- **Production CNN retrained** with 620 hard negatives + 144 hard positives. New default: `models/hard_neg_retrain/best_model.pt`. Old `matched_windows` and `production` models now deprecated. Full report: `docs/handoffs/v2-full-pipeline-results.md`.
- **DeepSqueak classification bridge** complete for 5970: 7,518 calls classified into 27 DeepSqueak clusters, merged with CNN detection metadata. Cross-validates our pipeline against an independent YOLO v2 detector — 95.6% agreement. Output: `classified_detections_full.csv`.
- **Corpus-constants unification (Phase 2)**: single source of truth for sample rate, USV band, STFT params at `src/usv_spectrogram/corpus.py`. Empirical-fact registry at `data/corpus_facts/{5970,3452,9252}.json`. Hook-enforced — prevents silent parameter drift in inference.

### Analysis (5970 — wild couple, animal `usv_lmt_034`)
- **Phase A1 (temporal dynamics)** DONE — call rate over time, type composition shifts across USV1–USV5 sessions, bout structure.
- **Phase A2 (sequential structure)** DONE — transition matrix, MI = 0.092 bits at lag 1 (within-bout), self-repetition enrichment 1.28× over independence (modest). 653 significant idioms after audit (down from inflated 1,843).
- **Phase A3 (acoustic feature deep-dive)** DONE — PCA captures 60% variance in 2 axes; UMAP/HDBSCAN confirms USVs form a *continuum* not discrete clusters. Mean power ↔ tonality r = 0.94.

### Analysis (3452 — second wild couple, animal `usv_lmt_035`)
- **Phase A2 + A3** DONE (Stream 1).
- **Phase B2 cross-animal comparison** done — see headline below.

### Analysis (9252 — third wild couple, animal `usv_lmt_036`)
- **Batch detection** complete: 597 events across 11,580 WAVs.
- **DeepSqueak Raven export** done; classification + A3 in progress.
- **Rate-anomaly investigation** complete — see "9252 surprise" below.

### Five parallel research streams closed in last 2 weeks
- Stream 1: 3452 vs 5970 cross-population A3 memo
- Stream 2: 9252 full Python + MATLAB pipeline
- Stream 4: cross-population reporting framework
- Stream 5: bout-threshold sensitivity sweep (MI plateau over [0.143, 1.0]s)
- SIS-baselines: drift-detection benchmark across 7 classifiers

---

## Headline finding: 3452 vs 5970

| | 5970 | 3452 |
|---|---|---|
| Calls (after QC) | 7,864 | 401 |
| Dominant type | **Flat (32.0%)** | **Short (42.9%)** |
| Down share | 17.6% | 4.0% |
| Frequency_Jump share | 7.4% | 2.2% |
| MI lag-1 (within-bout) | 0.092 b | **0.197 b** |
| PC1 + PC2 variance | 60.4% | 61.5% |
| Mean power ↔ tonality r | 0.94 | 0.917 |

**What replicates:** feature correlations, PCA structure, continuum geometry.
**What diverges:** type proportions (JSD = 0.37 b, p ≈ 2e-79), sequential MI (2.14× higher in 3452).

**The interpretation we need to be careful about:** N=1 couple per cohort. Both wild. So this is *between-couple variability within the wild stratum* — the noise floor a future wild-vs-lab signal must clear to be biologically interpretable as a strain effect.

Memo: `docs/handoffs/lab_parallel/01_RESULTS_3452_vs_5970.md`

---

## 9252 surprise — the quiet animal

Investigation outputs: `results/rate_anomaly_9252/`, memo: `docs/handoffs/lab_parallel/02_RESULTS_9252_rate_anomaly.md`

| metric | 5970 | 9252 | ratio |
|---|---:|---:|---:|
| File-yield | 20.75% | **2.75%** | **7.56× lower** |
| Events/file (mean, 95% CI) | 1.184 [1.107, 1.254] | **0.0516 [0.044, 0.060]** | **23× lower** |

**Hypotheses tested:**
- H1 recording length — DOES NOT explain the gap
- H2 animal silence — **SUPPORTED** (primary explanation)
- H3 noise floor — **FALSIFIED** (9252 is *quieter* in noise, KS p < 1e-6)
- H4 season / date — WEAK (only 5 days separates the cohorts)

**Within-9252 there is huge session heterogeneity:** USV3 alone contributes 48% of events with file-yield 7.12% (close to 5970), while USV4 sits at 1.02%, ratio 16.6×. **This is the scientifically interesting bit** — what makes USV3 different? Phase C (LMT behavioral correlation) is the natural follow-up.

---

## Where each cohort sits

| | Detection | Classification | A1 temporal | A2 sequential | A3 features | Cross-comp |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **5970** | done | done | done | done | done | reference |
| **3452** | done | done | (covered by A2) | done | done | done vs 5970 |
| **9252** | done | in progress | — | — | in progress | pending |

---

## What's blocking the central research question

The actual research axis is **wild vs lab-strain**. We have three wild couples and zero lab-strain recordings. Without lab-strain data, every cross-cohort number we produce is a *baseline*, not a *signal*.

**Status:** waiting on Mickey for lab-strain recording data (reminder due in 2 days).
**Pipeline ready:** `docs/handoffs/HANDOFF_05_LAB_DATA_PIPELINE.md` describes the run order once data arrives — but Phases A+B should be complete first. We're on track.

---

## Questions I'd like Mickey's input on

These are blocking decisions that need domain expertise. Full doc: `docs/questions-for-mickey.md`.

1. **Bout definition.** Should WAV-file boundaries always be bout breaks? The trigger-based recorder produces ≥2s silence between files. Within-file ICIs are bimodal at 74ms / 184ms (crossover 0.143s); MI plateau is flat over [0.143, 1.0]s. Does "bout" mean (a) continuous calling, ~150ms threshold, or (b) behavioral episode that may span brief recorder restarts?

2. **Cage 5970 — single animal or pair?** All sequential analysis assumes one vocalizer. If two animals are alternating, self-repetition signal is mechanically inflated. *Same question applies to 3452 and 9252.*

3. **Merged-call detections.** During dense bouts, our pipeline produces 200–1000ms events containing 2–6 calls with ~50ms gaps. Should we split these into individual calls, or is bout-level detection acceptable?

4. **FP filter swap.** A retrained filter without `duration_windows` performs strictly better in cross-validation (F2 0.833 vs 0.823) and recovers more long-merged real bouts. Adopt for production?

5. **Is 1.28× self-repetition enrichment biologically meaningful for wild mice?** Lab strains might show stronger structure. Any prior we should benchmark against?

6. **Lab-strain data ETA?** Pipeline is ready and waiting.

---

## Risks & known gaps

- **N=1 couple per cohort** — every cross-population number is descriptive, not inferential. Cannot do statistics across cohorts; can only describe.
- **9252 classification still underway** — A3 cross-comparison with 5970/3452 will land in the next 1–2 sessions.
- **Phase C (LMT behavioral correlation) not started** — depends on locating .sqlite behavioral databases for each cohort.
- **Open questions in `questions-for-mickey.md`** are blockers, not nice-to-haves — bout definition in particular gates re-running A2 with corrected segmentation.

---

## Pointers (for follow-up reading)

- Production CNN: `docs/handoffs/v2-full-pipeline-results.md`
- 3452 vs 5970: `docs/handoffs/lab_parallel/01_RESULTS_3452_vs_5970.md`
- 9252 anomaly: `docs/handoffs/lab_parallel/02_RESULTS_9252_rate_anomaly.md`
- Open questions: `docs/questions-for-mickey.md`
- Analysis roadmap: `docs/analysis-roadmap.md`
- Lab-strain pipeline: `docs/handoffs/HANDOFF_05_LAB_DATA_PIPELINE.md`
- Bout-threshold sensitivity: Stream 5 commit `916f4ace`

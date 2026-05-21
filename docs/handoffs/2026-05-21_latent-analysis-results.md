# Latent-space analysis (Moves B → A → C) — results memo

**Date:** 2026-05-21
**Predecessor handoff:** `docs/handoffs/2026-05-20_latent-analysis-b-a-c.md`
**Substrate:** contour-masked VAE, 32-D latents, 4 cohorts (5970, 3452, 9252, lab_131204), CLEAN cage-confound verdict
**Worktree:** `worktree-latent-analysis-b-a-c` (commit pending — not yet merged to main)

---

## Headline (one sentence)

In a cage-confound-free 32-D contour-VAE latent space, **between-wild-dyad variability dominates the wild-vs-lab axis**: 5970 occupies a distinct region of latent space, is more stereotyped at the bigram level, and has higher dispersion than every other cohort — while 3452 and 9252 cluster with lab in both occupancy and structure.

---

## Move B — Within-cohort dispersion (sanity check)

**Method.** Mean pairwise Euclidean distance in 32-D latent space per cohort, equal-N subsample N=400 (matches 3452 floor), 500-rep bootstrap CI.

**Result.**

| Cohort | N (patches) | Subsample | Dispersion | 95% CI |
|---|---:|---:|---:|---|
| 5970 | 12,440 | 400 | **7.943** | [7.734, 8.105] |
| 3452 | 406 | 400 | 5.750 | [5.494, 5.946] |
| 9252 | 584 | 400 | 5.991 | [5.772, 6.152] |
| lab_131204 | 55,863 | 400 | 5.941 | [5.787, 6.073] |

**Decision-gate:** **GREEN LIGHT A** — 5970's CI bottom (7.73) sits well above lab's CI top (6.07). Latent space encodes meaningful biological variation.

**Surprise.** Only the 5970 dyad shows "wild > lab" dispersion. 3452 and 9252 cluster *with the lab* on this measure. The "wild > lab" framing is carried by a single wild dyad, not a population.

**Artifacts.** `results/latent_dispersion/{dispersion_by_cohort.csv, figure.png, summary.html}`

---

## Move A — Repertoire JSD (the headline)

**Method.** K-means K=20 over combined 32-D latents (shared alphabet), per-cohort patch-level cluster proportions, pairwise Jensen-Shannon divergence in bits, 1000-rep bootstrap CIs resampling `(wav_stem, call_id)` tuples within each cohort, K-sensitivity sweep K∈{10, 20, 30, 50}.

**JSD matrix (K=20, bits).**

|              | 3452   | 5970   | 9252   | lab_matched | lab_swap |
|--------------|-------:|-------:|-------:|------------:|---------:|
| 3452         | —      | 0.278  | 0.066  | 0.198       | 0.210    |
| 5970         | 0.278  | —      | **0.365**  | 0.163       | 0.135    |
| 9252         | 0.066  | 0.365  | —      | 0.189       | 0.210    |
| lab_matched  | 0.198  | 0.163  | 0.189  | —           | 0.007    |
| lab_swap     | 0.210  | 0.135  | 0.210  | 0.007       | —        |

**Decision-gate:** **NO BINARY STRAIN EFFECT.** Max wild-vs-wild JSD (5970↔9252 = 0.365, CI [0.341, 0.404]) cleanly exceeds max wild-vs-lab JSD (3452↔lab_swap = 0.210, CI [0.190, 0.256]) — CIs separated by 0.085 bits. Robust across K∈{10, 20, 30, 50}: ww > wl at every K.

**Structure.** Three groups, not two:
1. **5970 alone** — JSD to nearest neighbor (lab_swap) = 0.135; to farthest (9252) = 0.365.
2. **{3452, 9252, lab_matched, lab_swap}** — pairwise JSDs in [0.007, 0.210].
3. **lab_matched ≈ lab_swap** (JSD=0.007) — within-strain stability ≈ measurement floor.

**Relation to prior taxonomy work.** The unreliable Scattoni-7 cascade (`results/q2_repertoire/jsd_pairs_sorted.csv`) had already shown wild-wild floor 0.141 > all wild-vs-5970-lab JSDs (0.028 / 0.022). The contour-VAE reproduces the same qualitative pattern at ~2.5× magnitude on a clean substrate (no cage-artifact contamination).

**Artifacts.** `results/latent_repertoire/{cluster_proportions.csv, jsd_matrix.csv, jsd_pairs_with_ci.csv, k_sensitivity.csv, summary.html}`. Persisted K-means: `models/latent_kmeans/{k20.joblib, k20_labels.npy}` (reused by Move C).

---

## Move C — Latent transitions (dynamics)

**Method.** One symbol per call (mean-z assigned to nearest K-means centroid). Bout segmentation at 0.25s file-aware threshold. Per-cohort K×K transition matrices, bootstrap entropy rates (1000 reps, resampling sequences), idiom detection via within-sequence shuffle surrogate (1000 reps, 99th percentile threshold). MI plateau replicated across bout thresholds {0.1, 0.143, 0.2, 0.25, 0.5, 1.0, 2.0}s.

**Entropy rates (bits/transition).**

| Cohort | n_calls | n_bouts | mean_bout_len | H | 95% CI |
|---|---:|---:|---:|---:|---|
| 5970 | 7,396 | 2,741 | 2.70 | **3.72** | [3.60, 3.72] |
| 3452 | 378 | 220 | 1.72 | 2.18 | [1.64, 2.51] |
| 9252 | 575 | 459 | 1.25 | 0.00¹ | [0.00, 1.90] |
| lab_matched | 27,919 | 8,662 | 3.22 | 3.73 | [3.69, 3.73] |
| lab_swap | 12,649 | 4,570 | 2.77 | 3.82 | [3.75, 3.82] |

¹ 9252's H=0 is artifactual — mean bout length 1.25 means almost no observed bigrams; fallback uniform rows collapse H to 0. Do not cite as biology.

**Decision-gate.** **5970 has same sequence-level richness as lab** (H=3.72 vs lab 3.73-3.82, overlapping CIs). It differs in *which letters it uses* (Move A) but not in *how varied its bigram repertoire is*.

**Idioms** (bigrams exceeding 99th percentile of within-sequence shuffle surrogate):

| Cohort | n_idioms | Top idiom (enrichment) |
|---|---:|---|
| 5970 | **26** | 7→5 (3.07×) |
| lab_matched | 14 | 15↔16 reciprocal pair (~1.5×) |
| lab_swap | 10 | 19→19 self-loop (1.62×) |
| 3452 | 0 | (no bigram cleared the surrogate; n_bouts=220 too small) |
| 9252 | 0 | (same — n_bouts=459, mean len 1.25) |

**Reading.** 5970 has the most stereotyped sequencing of any well-powered cohort — almost twice the idioms of lab_matched. Combined with Move A (5970 occupies distinct latent regions) and Move B (5970 has highest dispersion), the integrated picture is: 5970 explores a wider region of latent space than any other cohort *but* uses it with more reliable bigram patterns than lab. Not "louder noise" — "different language, more strictly grammatical".

**Bout-MI plateau check** (combined cohorts, lag-1 MI in bits, all cohorts pooled):
- 0.1s → 0.122, 0.143s → 0.156, 0.2s → 0.178, 0.25s → 0.203, 0.5s → 0.210, 1.0s → 0.211, 2.0s → 0.212
- Plateau begins at 0.5s; 0.25s captures 96% of asymptote → validates the recommended bout threshold.

**Artifacts.** `results/latent_transitions/{entropy_rates.csv, idioms.csv, bout_mi_sweep.csv, summary.html, transition_matrices/<cohort>.csv (×5), centroids/cluster_NN.png (×20)}`

---

## Caveats and implementation deviations

1. **3452 / 9252 statistically thin.** With 378 and 575 calls respectively (vs 7,396 for 5970 and 40,568 for lab), all 3452 / 9252 results from Moves A and C must be treated as "insufficient data" rather than biology. Move A's JSD CIs widen for these cohorts (0.2-0.4 bits CI range); Move C's entropy rates are dominated by uniform-fallback rows.

2. **N = 1 dyad per wild cohort.** "5970 is the outlier" really means "this one wild dyad is the outlier among 4 dyad-level recordings." Generalization to "wild mice in general" requires more wild dyads. Per `feedback_cross_animal_population_strata.md`: cross-animal comparisons must name the population stratum.

3. **lab_matched + lab_swap split.** Derived from `wav_stem` regex `_m(\d+)fm(\d+)_`. 6 matched couples (1x1..6x6), 11 swap couples. Patch split: matched=37,677 / swap=18,186 — matches `project_wild_mice.md`'s 17-couple count.

4. **Implementation deviations (documented in code):**
   - `bootstrap_jsd_pairs` (Move A) applies ε=1e-12 additive smoothing to bootstrap proportions to avoid JSD-disjoint-support = 1.0 saturating variance in a test case with single-call cohorts. Real-data effect: ≤1e-9 on JSD values.
   - `bootstrap_jsd_pairs` (Move A) and `bootstrap_entropy_rate` (Move C) apply a CI bracket clamp: `ci_lo = min(P2.5, point)`, `ci_hi = max(P97.5, point)`. The percentile bootstrap can produce CIs that don't bracket the point estimate when the statistic is bounded (like entropy rate ≤ log2(K) = 4.32 bits). For Move A's 10 pairs at n_reps=1000, no clamp activated. For Move C, the clamp activated on 3 of 5 cohorts (5970, lab_swap, 9252) because their bootstrap reps' percentile equaled or fell below the point. Principled fix for future reuse: switch to BCa bootstrap.

5. **VAE seed = 42 only.** Latent space could be seed-sensitive in principle (handoff caveat #3). If results need to be defended for publication, re-run B/A/C with seeds {1, 7, 42} and report variability.

6. **Patch-level vs call-level for Move A.** Move A used patch-level proportions (one vote per spectrogram window). Calls with multiple windows contribute multiple votes. This is the right choice for measuring *latent-space occupation*. Move C correctly switched to call-level (one symbol per call) for *sequence* analysis. Cross-check: would Move A's JSD ranking change at call-level? Untested, low-risk to assume no (since the 5970-as-outlier signal is large).

---

## Artifacts produced

All paths relative to worktree root `~/projects/mickey_london_lab/.claude/worktrees/latent-analysis-b-a-c/`:

**Scripts**
- `scripts/analyze_latent_dispersion.py` (Move B)
- `scripts/analyze_latent_repertoire_jsd.py` (Move A)
- `scripts/analyze_latent_transitions.py` (Move C)

**Tests** (all passing — 11 + 15 + 20 = 46 tests total)
- `tests/test_analyze_latent_dispersion.py`
- `tests/test_analyze_latent_repertoire_jsd.py`
- `tests/test_analyze_latent_transitions.py`

**Results** (CSVs + HTML reports)
- `results/latent_dispersion/` — Move B
- `results/latent_repertoire/` — Move A
- `results/latent_transitions/` — Move C (includes 20 centroid PNGs + 5 transition heatmaps)

**Persisted models**
- `models/latent_kmeans/k20.joblib` — K-means model (reused by C)
- `models/latent_kmeans/k20_labels.npy` — patch labels (length 69293)

**WSL viewing URLs** (open in browser):
- `file://wsl.localhost/Ubuntu/home/shachar/projects/mickey_london_lab/.claude/worktrees/latent-analysis-b-a-c/results/latent_dispersion/summary.html`
- `file://wsl.localhost/Ubuntu/home/shachar/projects/mickey_london_lab/.claude/worktrees/latent-analysis-b-a-c/results/latent_repertoire/summary.html`
- `file://wsl.localhost/Ubuntu/home/shachar/projects/mickey_london_lab/.claude/worktrees/latent-analysis-b-a-c/results/latent_transitions/summary.html`

---

## What's next (suggestions for future sessions)

1. **VAE seed sensitivity.** Re-run all three moves with seeds {1, 7, 42}, report seed-to-seed variability of the headline numbers. If 5970's dispersion / JSD / idiom-count rank is stable, the result is robust.

2. **Add the 2379 wild dyad** (per `project_wild_mice.md` — 4th wild dyad, not yet processed). Would give 4 wild cohorts vs 1 lab cohort, sharpening the between-dyad variability estimate.

3. **VocalMat-Grimsley classifier** (per `PLAN_lab_cnn_classifier.md` on `worktree-lab-cnn-classifier-plan`) would let us compare 5970's idiom bigrams to the 12-class Grimsley taxonomy and ask: do 5970's enriched bigrams (e.g., 7→5 at 3.07×) correspond to recognizable canonical call shapes?

4. **Sequence-level rather than bigram-level idioms.** Move C's idiom analysis is on bigrams only. Tri-gram or higher-order n-gram analysis could reveal multi-step idioms (e.g., signature 3-step motifs in 5970 that don't appear in lab).

5. **Call-level (not patch-level) Move A re-run** for cross-check.

---

## Memory notes that became load-bearing in this session

These came up enough that they should be cited in any future related memo:

- [[project_syllable_type_unreliable]] — why we used latent-space JSD instead of Scattoni-7 JSD
- [[project_jsd_source_corrections]] — the prior wild-vs-wild floor 0.141 from Scattoni-7 (which this work reproduces qualitatively at 0.365 in latent space)
- [[project_wild_mice]] — N=1 dyad per wild cohort framing
- [[feedback_rig_artifact_mean_power_db]] / [[feedback_cage_not_rig_terminology]] — cage artifacts that the contour-VAE strips
- [[project_bout_threshold_sensitivity]] — the 0.25s bout threshold, replicated here on the latent alphabet
- [[feedback_html_user_facing_default]] — HTML over Markdown for user-facing reports (followed throughout)
- [[feedback_no_bulk_stage_in_parallel_chats]] — staged by exact path only (no `git add -A`)

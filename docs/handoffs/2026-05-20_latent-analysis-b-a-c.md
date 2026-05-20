# Latent-space analysis (Moves B → A → C) — wild vs lab courtship USVs

**Date:** 2026-05-20
**Status:** Ready to execute. Move B first (risk management), then A, then C, with decision gates between.
**Predecessors:**
- `docs/handoffs/2026-05-19_path-c-cluster-execution.md` — produced the Phase 5 verdict
- `docs/handoffs/2026-05-20_path-c-cleanup.md` — source-bug fixes + regression tests + artifact pull-back
- `inbox/path_c_phase5_cross_cohort_verdict.md` — the CLEAN-verdict capture

**Successor handoff for:** future session(s) running the latent-space analyses.

---

## TL;DR

The contour-masked VAE trained on all 4 cohorts (5970, 3452, 9252, lab_131204) produced a **CLEAN cage-confound verdict**: 0 of 32 latent dims fire, max |Cohen's d| = 0.77, max |Pearson r| = 0.36. This certifies the combined 32-D latent space as **cage-signature-free** — cohort identity is not encoded via narrowband artifacts at typical USV frequencies.

We can therefore use the latents as the substrate for a clean wild-vs-lab comparison. **Three moves**, in this order:

| Order | Move | Cost | Purpose |
|---|---|---|---|
| 1 | **B** — within-cohort dispersion | ~2 hours | Sanity-check that the latents still encode known biology (wild > lab heterogeneity). Risk-management before A/C. |
| 2 | **A** — repertoire JSD in latent space | ~half a day | The headline "do wild and lab differ in repertoire?" result, with an internally-consistent wild-vs-wild floor. |
| 3 | **C** — latent transition matrices | ~1 day | A2 rebuilt on a clean alphabet (K-means cells over the latent space). Dynamics deep-dive. |

**Decision gates** are between moves — see below.

---

## Critical context for a fresh session

These are the load-bearing facts you must internalize before writing any analysis code.

### 1. The `syllable_type` column is NOT trustworthy.

User established 2026-05-20: the 7-class Scattoni / Holy & Guo cascade in `classified_detections_*.csv` is a rule-based pipeline the lab built to *approximate* the published taxonomy. It does not reliably identify the actual canonical shapes. Replacement in flight: VocalMat 12-class Grimsley taxonomy + ResNet-18 + DANN (`PLAN_lab_cnn_classifier.md`).

**Implication for B/A/C:** Do NOT use `syllable_type` for sanity-checks, color-coded plots, or JSD computation. Use continuous latent-space measurements only.

See memory: `project_syllable_type_unreliable.md`.

### 2. Past JSD figures were on `syllable_type` and are suspect.

`results/q2_repertoire/jsd_pairs_sorted.csv` reports pairwise JSDs (in bits) on the Scattoni-7 cascade:

| Pair | JSD (bits) |
|---|---|
| 5970 vs 3452 (wild-wild) | 0.141 |
| 5970 vs 9252 (wild-wild) | 0.108 |
| 5970 vs lab_matched | 0.028 |
| 5970 vs lab_swap | 0.022 |
| lab_matched vs lab_swap (lab-lab) | 0.004 |

The biggest JSD is wild-vs-wild (0.141), exceeding every wild-vs-lab JSD involving 5970. But these are all on the unreliable cascade — cite as historical context only, not as a reference floor.

A "JSD=0.37 wild-vs-wild floor" appears in some older session notes — that number was never in the data. Ignore it.

See memory: `project_jsd_source_corrections.md`.

### 3. N=1 dyad per wild cohort.

Each of 5970, 3452, 9252 is **one wild-mouse couple** (male + female), not one individual. The male is the vocalizer.

| Cohort | n_calls | n_dyads |
|---|---|---|
| 5970 (`usv_lmt_034`) | 12,440 | 1 wild dyad |
| 3452 (`usv_lmt_035`) | 406 | 1 wild dyad — statistically thin |
| 9252 (`usv_lmt_036`) | 584 | 1 wild dyad — statistically thin |
| lab_131204 | 55,863 | many lab dyads — dominates 80.6% |

**Implication:** Lab dominance biases any "global" latent analysis toward lab. Equalize via subsampling for B/A. For C, use per-cohort sequences.

See memory: `project_wild_mice.md`, `feedback_cross_animal_population_strata.md`.

### 4. mean_power_db and tonality are cage artifacts — already removed.

The contour mask + per-patch [0,1] normalization in `train_contour_vae_v2.py` strip both. CLEAN verdict shows the strip worked.

See memory: `feedback_rig_artifact_mean_power_db.md`.

### 5. Bout threshold for sequential analysis (Move C only).

`project_bout_threshold_sensitivity.md`: MI plateau over [0.143, 1.0] s for 5970. Recommended bout threshold = **0.25 s, file-aware**.

---

## Inputs

All paths relative to the main repo root:

| Input | Path | Size |
|---|---|---|
| Combined latents | `results/contour_vae_combined/latents.parquet` | 14 MB |
| VAE checkpoint | `models/contour_vae_combined/best.pt` | 31 MB |
| VAE hyperparams | `models/contour_vae_combined/hyperparams.json` | 2 KB |
| Reconstructions | `results/contour_vae_combined/reconstructions/` | 20 PNGs |
| Phase 5 verdict | `results/phase5_cross_cohort/diagnostic_result.json` | 242 B |
| Per-dim diagnostic | `results/phase5_cross_cohort/per_dim_diagnostic.csv` | 2.4 KB |
| Locked diagnostic | `scripts/cage_confound_diagnostic.py` | — |

Cohort identifiers in `latents.parquet["cohort"]`: `"5970"`, `"3452"`, `"9252"`, `"lab_131204"`.

---

## Move B — within-cohort dispersion (sanity check)

**Goal:** Replicate the prior finding "wild > lab heterogeneity" in the certified-clean latent space.

### Deliverables

- `scripts/analyze_latent_dispersion.py`
- `tests/test_analyze_latent_dispersion.py` — 3–4 unit tests, written before implementation
- `results/latent_dispersion/dispersion_by_cohort.csv` — point + 95% CI per cohort
- `results/latent_dispersion/figure.png` — forest plot
- `results/latent_dispersion/summary.html` — HTML report

### Method

Mean pairwise Euclidean distance in 32-D latent space per cohort. Subsample each cohort to N=400 (matches 3452 floor) for equal-N comparison. Bootstrap with replacement on the subsample, 500 reps, for 95% CI.

### Why these choices

- **Subsample to N=400** (matches 3452's 406): pairwise-distance estimates inflate with N due to tail sampling. Equal-N is the only honest cross-cohort comparison.
- **Bootstrap with replacement** on subsample: gives CI for the dispersion U-statistic.
- **Euclidean in raw 32-D**: VAE latents have ~N(0, ~1) prior from KL, so raw distance is meaningful. Don't z-score per dim.

### Decision gate (B → A)

| Outcome | Action |
|---|---|
| **5970 > lab_131204** with CIs separated | Green light A. Substrate is biologically informative. |
| 5970 ≈ lab_131204 (CIs overlap) | Pause. Investigate latent degeneracy via reconstruction PNGs and z-distribution. |
| 5970 < lab_131204 | Strong signal but opposite of prior expectation. Document and proceed with explicit framing. |

3452/9252 are statistical sentinels (CIs wide). If their point estimates land near 5970, confirmatory of wild > lab.

---

## Move A — repertoire JSD in latent space (the headline)

**Goal:** Replace suspect `syllable_type`-based Q2 numbers with JSD on cluster occupancy in certified latent space.

### Deliverables

- `scripts/analyze_latent_repertoire_jsd.py`
- `tests/test_analyze_latent_repertoire_jsd.py`
- `results/latent_repertoire/cluster_proportions.csv`
- `results/latent_repertoire/jsd_matrix.csv` — pairwise JSD (5×5 with lab_matched/lab_swap split)
- `results/latent_repertoire/jsd_pairs_with_ci.csv` — bootstrap CIs
- `results/latent_repertoire/summary.html`
- `models/latent_kmeans/k20.joblib` — persisted K-means (reused by Move C)

### Method

1. Cluster the combined latent space with K-means K=20 (NOT per-cohort — need shared alphabet)
2. Per-cohort cluster proportions over the K=20 alphabet
3. Pairwise JSDs (bits) between cohort proportions
4. Bootstrap CI: resample calls within each cohort, recompute proportions, recompute JSD. 1000 reps.

### Why K=20 K-means and not HDBSCAN

Prior VAE work (`docs/handoffs/2026-05-18_vae_comparison_memo.md`) used HDBSCAN on 32-D and got high noise-fraction outcomes. K-means produces a *partition* (required for JSD over cluster proportions). HDBSCAN noise points have no cluster, would need special bin.

K=20: smaller than the prior k-means count (27 in `classified_detections`), big enough for morphological diversity. Sensitivity: re-run with K ∈ {10, 20, 30, 50} as robustness check.

### Decision gate (A → C)

| Outcome | Reading |
|---|---|
| **Max wild-vs-lab JSD > max wild-vs-wild JSD** (CIs separated) | Strong evidence of strain effect. Headline result. |
| Wild-vs-lab JSDs inside the wild-vs-wild range | "Lab no more different from wild than wild dyads from each other." |
| K-sensitivity: result flips for some K | Fragile finding; report range. |

### lab_matched vs lab_swap

Derive labels from `wav_stem` (matched-couples set `{m1fm1, ..., m6fm6}` per Q2 summary). Include both as separate "cohorts" so the lab-vs-lab JSD provides within-strain control.

---

## Move C — latent transition matrices (dynamics)

**Goal:** Phase A2 rebuilt on the K-means alphabet from Move A.

### Deliverables

- `scripts/analyze_latent_transitions.py`
- `tests/test_analyze_latent_transitions.py`
- `results/latent_transitions/transition_matrices/<cohort>.csv` — K×K per cohort
- `results/latent_transitions/entropy_rates.csv` — per-cohort with CI
- `results/latent_transitions/idioms.csv` — over-represented n-grams per cohort
- `results/latent_transitions/summary.html`

### Method

Reuses `src/usv_spectrogram/analysis/information_theory.py`. Pipeline identical to A2 with K-means cluster replacing `syllable_type`. Per-cohort:
- Per-bout sequences (bout threshold 0.25s file-aware)
- Bigram transition matrix P (K×K)
- Entropy rate H(P)
- Idioms via 1000-rep within-session shuffle surrogate; bigrams exceeding 99th percentile = idioms

### Key details

- **Bout segmentation**: 0.25 s file-aware. Replicate MI plateau on combined cohorts first; per-cohort threshold if needed.
- **Shuffle scope**: within session, not across sessions/cohorts.
- **Symbol alphabet**: use *same* K-means from Move A.

### Reporting

Per-cluster centroid spectrograms: closest 9 latents per centroid, decode through `best.pt`, tile 9-up. Gives each "letter" a visual identity.

---

## Caveats that apply to all three moves

1. **Lab dominance (80.6%)**: equalize via subsampling in B; A's K-means centroids will sit in lab-dense regions. Sensitivity: re-cluster on per-cohort-balanced subsample.
2. **N=1 dyad per wild cohort**: "wild-specific" findings are really "this one dyad shows X". Generalization needs more wild dyads (don't have).
3. **Single VAE seed**: latent space could be seed-sensitive in principle. Re-run borderline results with seeds {1, 7, 42}.
4. **Patch-level not call-level**: long calls produce multiple latents auto-correlated within. Move C must count one symbol per call.
5. **Euclidean in 32-D**: less discriminating as dims grow. Try cosine for clustering if A is borderline.

---

## Working preferences (from session memory)

- **HTML over Markdown** for user-facing reports.
- **Print parameters/thresholds/row counts** in every analysis run.
- **WSL file viewing**: `file://wsl.localhost/Ubuntu/<path>` URLs.
- **No bulk `git add -A`/`.`** — stage by exact path.
- **Orchestrator mode** — delegate non-trivial work to subagents.

---

## Definition of done

Each move:
- Script + tests passing
- Output CSVs validate against schema
- Summary HTML with parameters logged and figures embedded
- Decision gate result documented in HTML

Package:
- All three moves with consistent K (Move A's K-means model reused in C)
- Memo at `docs/handoffs/2026-05-XX_latent-analysis-results.md` with B/A/C and the headline result

---

## How to start

1. Read this entire handoff.
2. Read the four memories (syllable_type unreliability, JSD corrections, wild mice N=1 dyad, cage artifacts).
3. Open `results/phase5_cross_cohort/index.html` to ground in CLEAN.
4. TaskCreate for each move (3 tops; sub-tasks for tests-before-implementation per CLAUDE.md).
5. Start with Move B. Pre-write tests via `test-architect`. Implement, validate, summary HTML, surface results before A.
6. Decision gate at end of each move — surface before starting next.

User approved this plan in concept (2026-05-20). Sub-decisions (K-sensitivity, distance metric, idiom percentile) can proceed without re-approval unless surprising.

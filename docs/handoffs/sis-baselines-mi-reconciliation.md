# Handoff: Reconcile 0.093 vs 0.0758 Scattoni-7 MI in SIS Baselines

**Date:** 2026-04-18
**From:** Claude Code (session 20260418)
**To:** Claude Code (future chat)
**Status:** Ready — all investigation complete, execution outstanding

---

## TL;DR

Phase A2 and SIS Benchmark module 17.1 both computed MI at lag 1 on Scattoni-7
labels for dataset 5970 and got **different numbers (0.0921 vs 0.0758 bits)**.
This is **not a regression** — it's a methodology difference (bout-aware vs raw
consecutive pairs). The investigation is complete; what's left is three small
documentation / registry updates to prevent future confusion. ~20–30 min of
work.

## Why this matters

- `docs/modules/sis-baselines.md:85-90` currently claims the SIS 17.1 module
  should **exactly reproduce** the Phase A2 value of 0.093 bits. That claim is
  false (it's off by 18% due to bout-filter methodology). A future reader will
  see the mismatch, think something's broken, and waste time chasing it.
- `data/corpus_facts/5970.json` captures Phase A2's *methodology*
  (`bout_detection_a2`) but not the resulting *MI values*. Future SIS
  comparisons (modules 17.8 SIM, 17.9 benchmark report) need a canonical
  registered baseline to compare against — not two numbers scattered across
  module docs and result CSVs.
- `scripts/audit_corpus.py` has hardcoded JSON section keys — if we want the
  registry to regenerate correctly for 3452/9252 when they land, we need to
  teach the audit script to harvest MI values too.

## The root cause — already investigated

Phase A2 (`scripts/analyze_sequential_structure.py:57`) uses
`BOUT_THRESHOLD_S = 0.6` and computes MI **only within bouts**. From
`results/sequential_structure/sequential_structure_summary.csv`:

```
n_within_bout_pairs          = 6350
n_cross_bout_gaps_excluded   = 1513
mi_lag1_bits                 = 0.0921
```

SIS 17.1 (`scripts/run_sis_baselines.py`) explicitly sets
`bout detection = NONE` per the ROADMAP 17.1 spec. From
`results/sis_baselines/baselines.csv`:

```
scattoni-7,7864,7,0.07575191408625362,...
```

Adding ~1,513 cross-bout pairs (near-random transitions) to the ~6,350
within-bout pairs (real sequential structure) dilutes the MI downward. **This
is expected behavior; neither number is wrong.**

Both runs used the same source data (`classified_traditional.csv` — which has
the `syllable_type` column, so the stale session-orient reminder claiming the
column is missing is wrong and can be dismissed).

## The three edits

### 1. Fix `docs/modules/sis-baselines.md` §Reproducibility Check (lines 85-90)

Current text (WRONG):

> Phase A2 previously computed MI at lag 1 on Scattoni-7 labels for the 5970
> dataset and obtained **0.093 bits**. Running this module over the same
> dataset should reproduce that value exactly — it reuses the same underlying
> estimator (`usv_language.analysis.sequence_analysis.mutual_information_at_lag`).

Replace with something like:

> Both this module and Phase A2 use the same estimator
> (`usv_language.analysis.sequence_analysis.mutual_information_at_lag`) on the
> same Scattoni-7 labels for dataset 5970, but they produce **different MI
> values** due to bout-filter methodology:
>
> | Source | Pairs | Method | Scattoni-7 MI |
> |---|---|---|---|
> | Phase A2 (`results/sequential_structure/`) | 6,350 within-bout | bout_detection_a2 (0.6 s threshold) | **0.0921 bits** |
> | SIS 17.1 (`results/sis_baselines/`) | 7,863 raw consecutive | none (per ROADMAP 17.1 spec) | **0.0758 bits** |
>
> The 18% difference comes from including ~1,513 cross-bout pairs whose
> transitions are near-random (no sequential relationship across bout gaps).
> Neither value is "wrong"; they measure different things. The canonical
> baseline for downstream 17.8/17.9 comparisons is the **raw-consecutive**
> value (0.0758 bits), since downstream modules use the same no-filter
> methodology. Both values are recorded under `sequential_structure_mi` in
> `data/corpus_facts/{dataset}.json`.

Match the existing Markdown style of the doc (no extra emojis, align tables).

### 2. Add `sequential_structure_mi` key to `data/corpus_facts/5970.json`

Schema to add at the top level (after `bout_detection_a2`, before
`labeling_distributions` — keep alphabetical/logical ordering consistent with
the file):

```json
"sequential_structure_mi": {
  "scattoni_7_bout_aware": {
    "mi_lag1_bits": 0.0921,
    "marginal_entropy_bits": 2.5436,
    "conditional_entropy_bits": 2.4499,
    "n_pairs": 6350,
    "method": "bout_detection_a2 (0.6 s threshold, cross-bout pairs excluded)",
    "source": "results/sequential_structure/sequential_structure_summary.csv"
  },
  "scattoni_7_raw_consecutive": {
    "mi_lag1_bits": 0.0758,
    "marginal_entropy_bits": 2.5481,
    "conditional_entropy_bits": 2.4723,
    "n_pairs": 7863,
    "method": "no bout filter, cross-file pairs included (ROADMAP 17.1 spec)",
    "source": "results/sis_baselines/baselines.csv",
    "canonical_for_downstream": true
  }
}
```

All numbers verified in the investigation session — source artifacts:
- Phase A2: `results/sequential_structure/sequential_structure_summary.csv`
  columns `marginal_entropy_bits`, `conditional_entropy_bits`, `mi_lag1_bits`
- SIS 17.1: `results/sis_baselines/baselines.csv` columns `marginal_entropy`,
  `conditional_entropy`, `mi_at_lag_1`

The `canonical_for_downstream: true` marker signals to 17.8/17.9 readers which
number to use as the reference when comparing other labelings.

### 3. Teach `scripts/audit_corpus.py` to harvest MI values

Add a new computation helper (e.g. `_compute_sequential_structure_mi`) that
reads both:
- `results/sequential_structure/sequential_structure_summary.csv` (for
  bout-aware values — columns `mi_lag1_bits`, `marginal_entropy_bits`,
  `conditional_entropy_bits`, `n_within_bout_pairs`)
- `results/sis_baselines/baselines.csv` (for raw-consecutive values — filter
  to `name == "scattoni-7"`, read `mi_at_lag_1`, `marginal_entropy`,
  `conditional_entropy`, `n_calls`)

Wire this into the `DATASET_REGISTRY` entry for 5970 (and leave 3452/9252 as
"not yet available" with a warning, matching the existing pattern for other
optional sources) and into the main JSON assembly around line 244-254.

The function should return a dict in the exact shape from step 2 above so the
output of `audit_corpus.py --dataset 5970` matches the hand-edited JSON byte
for byte (modulo `generated_at_utc` timestamp).

**Verification note:** After step 3 is done, run
`python scripts/audit_corpus.py --dataset 5970` and `git diff
data/corpus_facts/5970.json` should show only a timestamp change.

## Validation

1. **Markdown check:** `docs/modules/sis-baselines.md` renders correctly
   (table alignment, no broken links).
2. **JSON schema check:** `python3 -c "import json;
   json.load(open('data/corpus_facts/5970.json'))"` returns without error.
3. **Round-trip check:** `python scripts/audit_corpus.py --dataset 5970
   --output /tmp/5970-regen.json && diff data/corpus_facts/5970.json
   /tmp/5970-regen.json` — only `generated_at_utc` should differ.
4. **No test regressions:** `pytest tests/test_corpus.py tests/test_sis_baselines.py -v`
5. **Dismiss the stale reminder:** The session-orient reminder "SIS Benchmark
   17.1 follow-up — add `syllable_type` column to real
   `classified_detections_full.csv`, rerun baselines..." can be removed from
   `ops/reminders.md`. The underlying investigation showed the column is
   already present via `classified_traditional.csv`, and the "reproducibility
   check" exit criterion is now satisfied by the documentation clarifying that
   0.0758 ≠ 0.093 by design.

## Heads-up: new hook will fire on this work

A corpus-canary hook (added in the same session as this handoff —
`.claude/hooks/corpus_canary.py`) will fire on two of these edits:

- **Edit to `data/corpus_facts/5970.json`** — the file path is in the canonical
  registry; also, the values you're adding (0.0921, 0.0758) will themselves
  become canonical after the edit. Expect the decision-tree warning. Answer
  **(B)** — "Using EXISTING empirical facts, extending the registry with
  measured values that were previously scattered across result CSVs." This is
  not case (C) because the *concept* (MI of Scattoni-7 at lag 1) is already
  measured; you're just promoting it into the registry.
- **Edit to `scripts/audit_corpus.py`** — the file is already the canonical
  harvester. Answer **(B)** or **(C)** depending on how you frame it; the
  decision tree is prompting, not gating.

The `docs/modules/sis-baselines.md` edit should NOT trigger the hook (no
canonical values or names in the edited prose, modulo the table's literal
`0.0921`/`0.0758` which aren't in the registry *yet*).

## Relevant constraints

(Vault access is available — this handoff is for a Claude Code chat, not
Codex. Run `/kcheck "corpus sequential structure MI"` if you want a richer
constraint pull, but the key items are inline below.)

1. **ROADMAP 17.1 exit criterion** says Scattoni-7 MI should be "≈ 0.093
   bits". After step 1, that criterion is satisfied by the documentation
   clearly stating the two values + explaining the methodology gap. The 0.093
   target is the bout-aware number; 0.0758 is the raw-consecutive number.
   Both now live in `corpus_facts/5970.json`.

2. **`scripts/audit_corpus.py` pattern** — sections are hardcoded, not
   auto-discovered. Adding a new section means editing the script. Follow
   the pattern set by `_compute_bout_stats` and `_compute_labeling_distributions`:
   each helper reads one or two source CSVs, returns a dict, and is called
   from the main JSON-assembly block.

3. **CNN freeze is unaffected** — this work does not touch any Layer 1 physical
   constant or Layer 3 config. No drift assertions apply.

## Open questions for the future chat to decide

- **Should the Phase A2 bout-aware value (0.0921) be `canonical_for_downstream: false`?**
  I think yes — downstream modules 17.8/17.9 use no-bout-filter methodology,
  so the raw-consecutive number is the apples-to-apples reference. But if
  someone later argues "bout-aware is more biologically meaningful and should
  be the canonical comparison point," the flag can flip. This is a judgment
  call; I made it provisionally.

- **Should `audit_corpus.py` fail or warn when result CSVs are missing?**
  Current pattern (other sections) treats missing sources as warnings + empty
  output. Follow that pattern; don't hard-fail.

## Source artifacts (already read in investigation — paths for your reference)

- `scripts/analyze_sequential_structure.py:57` — `BOUT_THRESHOLD_S = 0.6`
- `scripts/run_sis_baselines.py:264-265` — "bout detection = NONE"
  declaration
- `results/sequential_structure/sequential_structure_summary.csv` —
  `mi_lag1_bits=0.0921`
- `results/sis_baselines/baselines.csv` — scattoni-7 MI=0.0758, n_calls=7864
- `results/sis_baselines/parameters.json` — SIS 17.1 methodology record
- `data/corpus_facts/5970.json` — current registry (before this work)
- `docs/modules/sis-baselines.md:85-90` — the prose to fix
- `docs/modules/corpus-constants.md` — overall corpus architecture

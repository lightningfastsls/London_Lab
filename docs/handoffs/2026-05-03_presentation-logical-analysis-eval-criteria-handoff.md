# Handoff: Presentation Logical Analysis Eval Criteria
Date: 2026-05-03

## Task

Create evaluation criteria for a logical analysis audit of the presentation figures and claims, then write a follow-up handoff for a separate chat to execute that audit.

This is not a provenance audit. The prior PNG audit established where figures came from and whether artifacts can be reproduced. This task is about whether the analysis logic supports the scientific and presentation claims.

Read first:

1. `AGENTS.md`
2. `docs/handoffs/2026-05-03_presentation-png-full-verification-results.md`
3. `docs/handoffs/2026-05-03_presentation-png-provenance-audit.md`
4. `presentation/FIGURE_GUIDE_FOR_WEB_SESSION.md`
5. `presentation/HANDOFF_FROM_CLAUDE_CODE.md`

Then read task-specific source docs as needed:

- `docs/handoffs/a3-acoustic-feature-deep-dive.md`
- `docs/handoffs/a2-sequential-structure-handoff.md`
- `docs/handoffs/sis-baselines-mi-reconciliation.md`
- `docs/modules/sis-baselines.md`
- `results/cross_population/wild_5970_vs_wild_3452.md`
- `data/corpus_facts/5970.json`
- relevant generator scripts named in the PNG audit

## Files Changed

- `docs/handoffs/2026-05-03_presentation-logical-analysis-eval-criteria-handoff.md` - this handoff.

## Required Output From This Task

Create a new handoff under `docs/handoffs/`, suggested filename:

`docs/handoffs/2026-05-03_presentation-logical-analysis-execution-handoff.md`

That execution handoff must include:

- the evaluation rubric
- the exact figure families to audit
- the evidence sources to inspect
- the per-family questions to answer
- the output format for the executor
- validation requirements
- priority order
- explicit distinction between reproducibility, implementation correctness, statistical validity, and presentation-claim validity

Do not execute the full logical audit in this criteria-creation chat unless the user explicitly asks. The goal is to design the audit so a later chat can execute it without ambiguity.

## Core Evaluation Dimensions To Define

The rubric should score each figure family on these dimensions:

1. **Claim Clarity**
   - What claim does the slide/guide imply?
   - Is the claim descriptive, causal, comparative, methodological, or speculative?
   - Is the claim stated with the correct scope?

2. **Implementation Match**
   - Does the generating script actually compute the quantity the claim needs?
   - Are the right source rows, filters, labels, thresholds, time windows, and grouping variables used?
   - Does the plotted quantity match the caption/narrative?

3. **Statistical Validity**
   - Are sample sizes, independence assumptions, and uncertainty handled appropriately?
   - Are inferential tests used only where justified?
   - Are descriptive comparisons framed as descriptive when N is too small?

4. **Biological / Experimental Framing**
   - Does the figure distinguish animal, cage/couple, recording session, and population correctly?
   - Does it avoid overclaiming wild-vs-lab when only wild-vs-wild data are present?
   - Does it preserve the key caveat that current cross-population figures are a dry run until lab data arrives?

5. **Methodological Consistency**
   - Are band limits, `sr=300000`, STFT/rendering assumptions, labels, and bout definitions consistent with code truth?
   - Are differences between raw consecutive, bout-aware, file-aware, and within-file analyses explicit?
   - Are UMAP/HDBSCAN/k-means methods described without implying more certainty than they support?

6. **Presentation Claim Support**
   - Does the figure actually support the slide message?
   - Would a lab audience likely infer something stronger or different than the code supports?
   - Should the slide be relabeled, caveated, replaced, or kept?

7. **Risk Classification**
   - Assign one of:
     - `sound`
     - `sound with caveat`
     - `needs caption fix`
     - `needs analysis fix`
     - `do not use as currently framed`
     - `insufficient evidence`

## Required Figure Families For Execution Handoff

The execution handoff should prioritize:

1. `03_training_data/`
   - Known issue: `training_set_composition.png` plots `spectrograms_training/`, not production `data/training/matched_windows_v2/`.
   - Evaluate whether current slide framing is misleading.

2. `09_umap/`
   - UMAP by type/feature, HDBSCAN/k-means, boundary cases, galleries.
   - Evaluate continuum claims, clustering claims, low-confidence boundary logic, and whether 10 DeepSqueak acoustic features support the narrative.

3. `10_temporal_dynamics/`
   - Call rate, hourly type composition, ICI distribution, bout structure, raster.
   - Evaluate timestamp parsing, session aggregation, bout threshold derivation, and claims about burstiness/stability over time.

4. `10_sequential_structure/`
   - Transition matrix, entropy, MI lag, Zipf, bout threshold sensitivity.
   - Evaluate within-bout filtering, self-repetition claims, entropy/MI interpretation, and robustness claims.

5. `11_sis_baselines/`
   - Evaluate raw-consecutive vs bout-aware MI, label vocabulary effects, literature comparison, and whether “DeepSqueak-27 wins” is a fair interpretation.

6. `12_cross_population/`
   - Evaluate wild 5970 vs wild 3452 framing, N=2 limitation, chi-square/Cramer's V use, feature and transition differences, and lab-vs-wild caveats.

7. `05_signal_detection/` and `06_deepsqueak_validation/`
   - Evaluate whether performance/validation claims are implemented correctly and whether DeepSqueak agreement is being interpreted appropriately.

8. `01_raw_signal_pipeline/` and `00_raw_waveform*.png`
   - Evaluate whether the “waveform hides USVs / spectrogram reveals them / pipeline improves detections” narrative is logically supported.

## Suggested Scoring Format

Define a compact table format for the executor:

| Family | Intended claim | Code computes | Logic verdict | Risk | Required slide/caption change | Evidence checked |
| --- | --- | --- | --- | --- | --- | --- |

For high-risk families, require detail blocks:

```markdown
### Figure / Family

Claim:

Implementation:

Logical assessment:

Overclaim risk:

Recommended fix:

Evidence:
```

## Required Validation For Criteria-Creation Chat

Before final answer:

- Re-read the execution handoff created by this task.
- Verify every referenced repo file/path in the execution handoff exists, unless explicitly marked as optional or to be discovered.
- Confirm the execution handoff tells the next chat not to rely on the PNG provenance audit as evidence of logical correctness.
- Confirm the handoff distinguishes:
  - artifact reproducibility
  - code implementation correctness
  - statistical/methodological validity
  - presentation framing validity

## Reasoning

The next useful step is not another hash or provenance pass. The project needs a rubric that asks whether each figure answers the question the presentation wants it to answer.

The most likely failure modes are:

- a real figure answering a different question than the slide implies
- a correct statistic framed too broadly
- a descriptive result presented as inferential
- a methodologically valid figure with missing caveats
- a copied artifact being treated as logically validated only because provenance is strong

The execution handoff should make those distinctions unavoidable.

## Validation

This handoff was written as a task specification. It has not yet been used to create the execution rubric handoff.

## Open Questions / Known Risks

- The exact slide deck text may differ from `presentation/FIGURE_GUIDE_FOR_WEB_SESSION.md`; the executor should treat the guide as the best available proxy unless a deck file is provided.
- Some source docs may be stale. The execution handoff should instruct the auditor to prefer code and result files over prose when they disagree.
- Logical analysis may identify code bugs or analysis bugs; the execution chat should report them, not silently patch them, unless the user explicitly asks for fixes.

## Worth Remembering For Claude

- The user explicitly wants logical analysis, not proof that images can be reproduced.
- The prior PNG audit is useful context but should not be mistaken for scientific validation.
- The next chat should produce a strong audit rubric first, then hand off execution to a separate chat.

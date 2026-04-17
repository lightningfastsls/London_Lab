# Handoff: /reduce extraction for SIS benchmark design rationale

**Date:** 2026-04-17
**Purpose:** Run `/reduce` on `inbox/sis-benchmark-design-2026-04-17.md` to extract atomic notes into the vault. A previous session identified 13 candidate extractions but ran out of context before writing them.

---

## What you're doing

Extract atomic notes from the design rationale captured for Phase 17 (`ROADMAP_SIS_BENCHMARK.md`). The design rationale is 14 numbered sections of reasoning behind a 9-module SIS benchmark that tests 4 hypothesis classes (rule-based iMSA / handcrafted Oren / learned AMVOC / direct SIM optimization).

## Read first

1. `inbox/sis-benchmark-design-2026-04-17.md` — the source (~200 lines). **This is comprehensive — the prior session wrote it specifically to be extractable.** Each numbered section (1-14) is the basis for at least one atomic claim.
2. `ROADMAP_SIS_BENCHMARK.md` — the ROADMAP the rationale supports. Useful for cross-referencing module numbers.
3. `docs/handoffs/three-paper-deep-reads-2026-04-15.md` — the upstream paper ingestion these claims build on.

## Execute

Run `/reduce` on the inbox file. Standard workflow applies:

```
/reduce inbox/sis-benchmark-design-2026-04-17.md
```

Or if handoff mode is needed for queue integration:

```
/reduce inbox/sis-benchmark-design-2026-04-17.md --handoff
```

## Prior session already identified 13 candidate extractions

The prior session ran a dedup scan and classified candidates. Use these as a starting point — validate against current vault state, then extract:

### NEW CLAIMS (9 notes to create)

| # | Proposed title (claim form) | Category | Primary connections |
|---|---|---|---|
| 1 | decision-gate methodology requires computing free SIS baselines before committing to feature engineering | methodology | `[[Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable]]` |
| 2 | four-hypothesis framing organizes SIS maximization into rules plus handcrafted features plus learned features plus direct optimization | methodology | `[[unsupervised-usv-discovery]]`, `[[classification-methodology]]` |
| 3 | autoencoder bottleneck plus PCA extracts concepts because reconstruction forces the model to preserve axes of variation that matter | claim | `[[AMVOC convolutional autoencoder provides the best open-source Python tool for unsupervised USV feature extraction and clustering]]` |
| 4 | low-dimensional intrinsic manifold argues for learned features rather than against them because bottleneck compression is how you find low-dim structure | claim | `[[Goffinet VAE found Gaussian mixture model clustering only supported k of 2 or fewer clusters for mouse USVs]]`, `[[HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold]]` |
| 5 | SIM optimization is structurally feature-independent so if it wins the finding is that labels matter more than features for sequential prediction | claim | `[[Hertz et al 2020 Syntax Information Score ranks classification schemes by how well syllable labels predict next syllable]]` |
| 6 | Oren marmoset ridge vectorization requires re-engineering not parameter tuning when adapted to mouse USVs because duration frequency band harmonics SNR and absolute-pitch relevance all differ | claim | `[[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]]` |
| 7 | pre-filtering layers each address a distinct ridge-extraction failure mode so removing any one layer likely reintroduces the failure it was blocking | methodology | `[[signal-processing]]` |
| 8 | separating deterministic vectorization from stochastic clustering into distinct modules lowers iteration cost when two stages have different costs or randomness properties | methodology | `[[classification-tools]]` |
| 9 | DSP modules need Tier 3 review because tests can pass on synthetic inputs while failing on real recordings for specific call types | methodology | project-specific review practice |

### ENRICHMENT TASKS (4 existing notes to update)

| # | Target note | What to add from source |
|---|---|---|
| 10 | `[[iMSA rule-based pitch-jump classification produces the highest SIS among compared methods despite lower label entropy]]` | Phase 17 promotes iMSA to first-implementation priority explicitly because it's the published mouse-USV top scorer AND continuity with Mickey's own lab methodology (Hertz 2020). |
| 11 | `[[ridge extraction finds the dominant frequency bin with maximum energy at each time step creating a pitch contour trajectory]]` | Concrete enumeration of 4 naive-argmax failure modes on mouse USVs (silent columns, harmonic jumps, broadband transients, low-SNR onset/offset columns) and the 4-layer defense (noise floor threshold, median filter, frequency band mask, DP continuity constraint). |
| 12 | `[[per-caller normalization of AM and FM features to 0-1 prevents individual acoustic idiosyncrasies from dominating classification]]` | Caveat: per-caller normalization may be *wrong* for mouse USV *type* classification because absolute frequency (50 kHz vs 90 kHz) distinguishes syllable types in the Scattoni taxonomy. Oren's marmoset-identity use case and Scattoni's type-classification use case have opposite absolute-frequency preferences. |
| 13 | `[[time-axis resampling to a fixed number of steps normalizes variable-duration vocalizations without discarding frequency information]]` | Duration must be added back as an explicit scalar feature when used for type classification, because time-resampling discards absolute duration and duration is primary in Scattoni's taxonomy (Short vs longer types). |

### SKIPPED (intentionally)

- The "starting mistake and course-correction" section (pure narrative) — its extractable lesson is already captured in claim #1.

## Section-to-claim mapping (if you want to re-derive)

| Inbox section | Produces claim(s) |
|---|---|
| 1 starting mistake | (skipped — narrative) |
| 2 four-hypothesis framing | #2 |
| 3 why iMSA deserves priority | #10 (enrich) |
| 4 why AMVOC belongs | #3, #4 |
| 5 why SIM could dominate | #5 |
| 6 decision-gate methodology | #1 |
| 7 domain adaptation | #6 |
| 8 pre-filtering rationale | #7, #11 (enrich) |
| 9 per-caller normalization wrong | #12 (enrich) |
| 10 duration as scalar | #13 (enrich) |
| 11 clustering separated | #8 |
| 12 Tier 3 DSP review | #9 |
| 13 effort budget | (skipped — project-tactical, not a general claim) |
| 14 what framework can/cannot tell | (skipped — scoped to Phase 17 only) |

## Dedup already checked

Prior session ran these dedup searches and found no existing overlap:

```
rg -il "decision gate|SIS baseline|free baseline" notes/        → no results
rg -il "four hypothesis|hypothesis framing" notes/               → no results
rg -il "low-dim manifold|autoencoder bottleneck PCA" notes/      → no results
rg -il "naive argmax|ridge harmonic|harmonic jumping" notes/     → 3 existing notes (confirmed as enrichment targets, not duplicates)
```

Re-run these before writing if you want fresh validation, but the classification should still hold.

## Post-extraction verification

After writing all 13 outputs:

1. Verify each new note has: frontmatter description passing the subject-echo check, at least one `Relevant Notes:` link, at least one `Topics:` link, and `Source: [[sis-benchmark-design-2026-04-17]]` footer.
2. Verify each enrichment updates the target note's body without duplicating existing content.
3. **Archive the source file:**
   ```bash
   mkdir -p archive/inbox
   mv inbox/sis-benchmark-design-2026-04-17.md archive/inbox/
   ```
4. If handoff mode was used, queue.json should have 9 claim entries + 4 enrichment entries.
5. Mark task #5 ("Extract design rationale to KG") as completed.

## Pipeline continuation

After /reduce completes, the standard pipeline continues:
- `/reflect` — find cross-note connections for the 9 new claims
- `/reweave` — backward pass: revisit older notes that should now link to these
- `/verify` — quality gate

These can be done in the same session as /reduce or deferred.

## Context note from prior session

The prior session wrote the inbox file specifically for extraction. Section 2 (four-hypothesis framing) and section 4 (why AMVOC belongs) are the most semantically dense — the rest are elaborations. If you have to triage due to context, prioritize claims #2, #3, #4, and #7 (the novel methodological framings that don't exist in the vault yet).

The four-hypothesis framing (claim #2) is especially important because it's the conceptual scaffold the whole ROADMAP hangs on — future sessions looking at Phase 17 will need to understand *why* 9 modules, and claim #2 is that explanation.

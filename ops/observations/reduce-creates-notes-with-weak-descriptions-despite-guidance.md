---
description: "/reduce produces descriptions that restate the title instead of adding mechanism, implication, or context — despite explicit quality rules in the skill spec"
category: friction
trigger: "Repeated observation across 3+ /reduce runs; confirmed by skill eval data, DeepSqueak batch output, and direct user report (2026-02-28)"
frequency: recurring
status: archived
archived: 2026-03-20
archived_by: rethink-2026-03-20
resolution: "Added Description Self-Check Gate to /reduce SKILL.md step 7c (2026-03-07) — enforces no-restatement, no-subject-echo, and 'so what' checks with batch failure flagging"
---

# /reduce creates notes with weak descriptions despite having quality guidance in the skill

**Occurrences:** 3+ (skill eval round, DeepSqueak batch reduction 2026-02-27, user-reported 2026-02-28). This observation was independently captured as two separate files before consolidation, confirming persistence.

The reduce skill (SKILL.md) explicitly instructs:

> Bad (restates title): "quality is important in knowledge work"
> Good (adds mechanism + implication): "when creation becomes trivial, maintaining signal-to-noise becomes the primary challenge — selection IS the work"
> The description is progressive disclosure: title says WHAT the claim is, description says WHY it matters or HOW it works.

Despite this, /reduce consistently produces descriptions that fail in three modes:

1. **Title restatement** — Description paraphrases the claim in different words, adding zero new information. Example: title "DeepSqueak v3 switched from Faster R-CNN to YOLO v2 improving speed and accuracy" -> description "DeepSqueak versions 1-2 used Faster-RCNN; v3.1 adopted YOLO v2 for faster and more accurate USV bounding box detection" (adds version detail but no implication).
2. **Subject echo** — Description begins with the same subject as the title. In the DeepSqueak batch, 8 of 11 descriptions start with "DeepSqueak..." — this primes the agent to restate rather than reframe from the reader's perspective.
3. **Missing the 'so what'** — Describes WHAT the claim is but not WHY it matters for our pipeline or WHEN it applies. Pass schema validation (non-empty, under 200 chars) but fail the cold-read prediction test.

This matters because descriptions are the primary filter when scanning notes — they determine whether an agent reads the full note or skips it. Weak descriptions degrade the value of `rg "^description:"` scans during /reduce's own dedup phase.

## Evidence

**Skill evals** (iteration-1 grading.json): Both with-skill and without-skill runs graded "Mixed quality" on titles/descriptions. Research-synthesis eval: "existing items use descriptive/topic titles [...] These are descriptions, not propositions." Planning-doc eval: "Some are method descriptions rather than claims."

**DeepSqueak batch** (2026-02-27): 11 notes created. ~4 have strong descriptions (add implication, e.g., "makes it unusable in Python-first pipelines without a bridge strategy"), ~4 borderline (add detail but not implication), ~3 restate the title.

**Second-order friction**: The observations themselves accumulated without triggering change — two separate observation files existed covering the same pattern. This suggests the observation-to-action pipeline has a gap for recurring single-issue friction that stays below the /rethink threshold (10 observations).

## Root Cause Hypotheses

1. **Throughput vs quality tension**: The skip rate < 10% mandate + comprehensive coverage pushes agents to satisfy the description field formally (filled in) but not substantively (adds information).
2. **No enforcement gate**: Schema validation checks presence, not quality. The cold-read test is advisory, not blocking.
3. **Missing self-check step**: The extraction flow (read -> plan -> write) has no "review descriptions before committing" phase.
4. **Subject echo pattern**: Starting descriptions with the same noun as the title primes restating. A simple heuristic could break this.

## Proposed Action

1. **Add a description self-check to /reduce**: After drafting all notes, explicitly review each description against the progressive disclosure test. If a description restates the title, rewrite to answer "so what?" or "when does this matter?"
2. **Add the subject-echo heuristic**: Instruct /reduce: "Never start a description with the same noun/subject that begins the title. Reframe from the reader's perspective or start with the implication."
3. **Add negative examples to the skill spec**: 2-3 "bad -> good" transformation examples using real notes from this vault, placed directly in the extraction template.
4. **Consider a description rubric**: Score on a 3-point scale (restates title / adds some info / adds actionable context). Only accept 2+. Makes the quality bar concrete.
5. **Strengthen /verify recite check**: Move cold-read prediction from post-hoc /verify to inline in /reduce itself — at least for the description field.
6. **Flag batch failures**: If 3+ weak descriptions found in a single /reduce run, flag as a process issue in session summary — don't silently pass.

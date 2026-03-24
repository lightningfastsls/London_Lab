# Vocabulary Transformation Reference

## Purpose

This document provides the complete lookup tables for transforming universal arscontexta terms into domain-native equivalents during system derivation. Vocabulary transformation is not cosmetic — it changes how the system feels to use. "Surface patterns in reflections" is therapy work. "Extract claims from sources" is research work. Same structural operation, different cognitive framing. The derivation engine consults these tables at Step 5b (vocabulary derivation) of the init wizard, and the vocabulary transformation test (Testing Milestone 3) uses them as ground truth for verifying zero cross-domain term leakage.

## Derivation Questions

- What is the domain-native equivalent of universal term X for use case Y?
- What template/folder/skill names should be generated for a given domain?
- How should the init wizard prompt for custom vocabulary in Experimental/mixed use cases?
- What is the quality check for vocabulary transformation completeness?
- What quality check ensures zero instances of cross-domain terms remain in generated files?

## Curated Claims

### Vocabulary Mapping Tables

The following tables function as lookup claims — each row maps a universal structural concept to its domain-native equivalent across seven domains. These are kept as tables rather than individual per-claim entries because their utility depends on cross-domain comparison and quick lookup during derivation. Four mapping dimensions must be transformed consistently: terms, template names, folder names, and skill names.

#### Universal → Domain Mapping

| Universal Term | Research | Therapy | Learning | Relationships | Creative | PM | Companion |
|---------------|----------|---------|----------|---------------|----------|-----|-----------|
| note | claim | reflection | concept note | observation | idea | decision | memory |
| extract / reduce | reduce | surface | break down | notice | discover | document | remember |
| connect / reflect | reflect | find patterns | relate concepts | trace connections | combine ideas | link decisions | recall together |
| verify | verify | check resonance | test understanding | confirm accuracy | evaluate draft | review quality | verify memory |
| MOC | topic map | theme | study guide | relationship map | project hub | decision register | memory collection |
| description field | claim context | reflection summary | concept explanation | observation context | idea sketch | decision rationale | memory context |
| topics footer | research areas | themes | study areas | relationship facets | creative projects | project areas | life areas |
| inbox | capture | journaling | study notes | encounters | inspiration | action items | moments |
| processing / pipeline | pipeline | reflection cycle | study cycle | relationship review | creative process | review cycle | reminiscence |
| wiki link | connection | pattern link | concept link | connection | inspiration thread | decision trail | memory link |
| thinking notes | claims | reflections | concepts | observations | ideas | decisions | memories |
| archive | processed sources | past reflections | mastered material | past encounters | completed works | closed decisions | past events |
| self/ space | research identity | reflection partner | study companion | relationship tracker | creative identity | project mind | companion memory |
| orient | orient | center | review progress | check in | survey ideas | status check | remember |
| persist | persist | journal | log progress | update records | save state | update status | save memories |

---

#### Template Name Mapping

| Universal Template | Research | Therapy | Learning | Relationships | Creative | PM | Companion |
|-------------------|----------|---------|----------|---------------|----------|-----|-----------|
| base-note.md | thinking-note.md | reflection-note.md | concept-note.md | observation-note.md | idea-note.md | decision-note.md | memory-note.md |
| moc.md | topic-map.md | theme.md | study-guide.md | relationship-map.md | project-hub.md | decision-register.md | collection.md |

---

#### Folder Name Mapping

| Universal Folder | Research | Therapy | Learning | Relationships | Creative | PM | Companion |
|-----------------|----------|---------|----------|---------------|----------|-----|-----------|
| notes/ | notes/ | reflections/ | concepts/ | observations/ | ideas/ | decisions/ | memories/ |
| inbox/ | inbox/ | journal/ | study-inbox/ | encounters/ | inspiration/ | action-items/ | moments/ |
| archive/ | archive/ | past/ | mastered/ | history/ | completed/ | closed/ | past/ |
| templates/ | templates/ | templates/ | templates/ | templates/ | templates/ | templates/ | templates/ |

---

#### Skill Name Mapping

| Universal Skill | Research | Therapy | Learning | Relationships | Creative | PM | Companion |
|----------------|----------|---------|----------|---------------|----------|-----|-----------|
| /reduce | /reduce | /surface | /break-down | /notice | /discover | /document | /capture |
| /reflect | /reflect | /find-patterns | /relate | /trace | /combine | /link | /recall |
| /verify | /verify | /check-resonance | /test | /confirm | /evaluate | /review | /verify |
| /reweave | /reweave | /revisit | /revise | /reconnect | /rework | /update | /revisit |
| /remember | /remember | /note-friction | /flag | /flag | /flag | /flag | /remember |
| /next | /next | /next | /next | /next | /next | /next | /next |
| /stats | /stats | /stats | /stats | /stats | /stats | /stats | /stats |
| /graph | /graph | /graph | /graph | /graph | /graph | /graph | /graph |

**Note:** /remember (formerly /friction) captures operational friction with automatic detection in session transcripts. /next (formerly /work + /next) surfaces the next recommended action from the task stack. /stats provides vault metrics and progress visualization. /graph enables graph query generation. These commands use universal names across all domains.

---

### Applying Transformations

### In the init wizard (Step 5b):

1. Determine the user's use case
2. Look up all universal terms in the mapping table above
3. Replace every instance in the generated context file
4. Replace template names and folder names
5. Replace skill names if generating skills
6. **Verify:** Read the generated output. Does it feel natural for the domain? Would a therapy user ever see the word "claim"? Would a PM user see "reduce"?

### Quality check:

The vocabulary test: read the generated context file as if you were the domain user. Every technical term should feel native to the domain. If any term feels imported from a different discipline, transform it.

### Extending the table:

For "Custom / Mixed" use cases, the init wizard should ask the user for their preferred vocabulary. Populate a custom column using the universal terms as prompts: "What do you call a single knowledge unit?" → their answer becomes the "note" equivalent.

---

## Exclusion Notes

- **Legal domain vocabulary:** Not yet mapped — legal knowledge systems (case briefs, precedent tracking, statute notes) have specialized terminology that requires domain expert input. Candidate for future column addition.
- **Medical/clinical domain:** Not yet mapped — clinical documentation vocabulary (SOAP notes, differential diagnoses, treatment plans) overlaps with but diverges from Therapy. Requires separate column, not a Therapy variant.
- **Software engineering domain:** Not yet mapped — while Developer/Engineering was excluded as a separate use-case preset, its vocabulary (ADR, runbook, postmortem, spike) is distinct enough to warrant a mapping column when the domain is supported.
- **Multi-language vocabulary:** Current tables are English-only. Localization of domain-native terms (e.g., Spanish therapy vocabulary) is out of scope but may be needed for international deployments.
- **Verb tense/conjugation transforms:** Evaluated whether mapping tables should include verb conjugations (e.g., "reducing" -> "surfacing") — excluded because the init wizard handles these programmatically from the base verb mapping.

---

## Version

- **Date:** 2026-03-20
- **Source claim count:** 4 mapping tables (terms, templates, folders, skills) across 7 domains + application procedure = 5
- **Included count:** 4 mapping tables + 1 application procedure (all original content preserved)
- **Excluded count:** 5 (Legal, Medical, Software Engineering, Multi-language, Verb conjugation transforms)

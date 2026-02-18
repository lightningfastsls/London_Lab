---
description: How this knowledge system was derived -- enables architect and reseed commands
created: 2026-02-18
engine_version: "1.0.0"
---

# System Derivation

## Configuration Dimensions
| Dimension | Position | Conversation Signal | Confidence |
|-----------|----------|--------------------|--------------------|
| Granularity | Atomic | "USV research with findings, methods, hypotheses" | High |
| Organization | Flat | Research preset default — flat-associative with semantic search | Inferred |
| Linking | Explicit+Implicit | "Yes — configure qmd" — wiki links + semantic discovery | High |
| Processing | Heavy | Research domain — full pipeline from day one | High |
| Navigation | 3-tier | Research preset default — hub/domain/topic | Inferred |
| Maintenance | Condition-based | Research preset default — event-driven | Inferred |
| Schema | Moderate | Research domain with confidence tracking and experimental conditions | High |
| Automation | Full | Claude Code platform — full automation from day one | High |

## Personality Dimensions
| Dimension | Position | Signal |
|-----------|----------|--------|
| Warmth | neutral | default — no personality signals |
| Opinionatedness | neutral | default |
| Formality | professional | default |
| Emotional Awareness | task-focused | default — research domain |

## Vocabulary Mapping
| Universal Term | Domain Term | Category |
|---------------|-------------|----------|
| notes | notes | folder |
| inbox | inbox | folder |
| archive | archive | folder |
| note (type) | note | note type |
| reduce | reduce | process phase |
| reflect | reflect | process phase |
| reweave | reweave | process phase |
| verify | verify | process phase |
| validate | validate | process phase |
| rethink | rethink | process phase |
| MOC | topic map | navigation |
| description | description | schema field |
| topics | topics | schema field |

## Platform
- Tier: Claude Code
- Automation level: full
- Automation: full (default)

## Active Feature Blocks
- [x] wiki-links -- always included (kernel)
- [x] atomic-notes -- granularity = atomic
- [x] mocs -- navigation = 3-tier
- [x] processing-pipeline -- always included
- [x] semantic-search -- qmd opted in
- [x] schema -- always included
- [x] maintenance -- always included
- [x] self-evolution -- always included
- [x] methodology-knowledge -- always included
- [x] session-rhythm -- always included
- [x] templates -- always included
- [x] ethical-guardrails -- always included
- [x] helper-functions -- always included
- [x] graph-analysis -- always included
- [ ] personality -- no personality signals (neutral-helpful default)
- [ ] multi-domain -- single domain
- [ ] self-space -- disabled (research preset default, goals route to ops/)

## Coherence Validation Results
- Hard constraints checked: 3. Violations: none
- Soft constraints checked: 5. Auto-adjusted: none. User-confirmed: none
- Compensating mechanisms active: semantic search compensates for implicit linking discovery

## Failure Mode Risks
- Collector's Fallacy (HIGH) — USV research has abundant source material (papers, recordings, experiment logs)
- Orphan Drift (HIGH) — high creation volume during active research without mandatory connection
- Verbatim Risk (HIGH) — signal processing literature tempts reproduction over transformation
- MOC Sprawl (HIGH) — topics proliferate across detection, classification, DSP, and training pipeline

## Generation Parameters
- Folder names: notes/, inbox/, archive/, templates/, ops/, manual/
- Skills to generate: all 16 (vocabulary-transformed with universal terms — research domain uses universal vocabulary)
- Hooks to generate: session-orient.sh, session-capture.sh, validate-note.sh, auto-commit.sh
- Templates to create: note.md, topic-map.md, source-capture.md, observation-note.md
- Self-space: disabled (goals route to ops/goals.md)
- Extraction categories: findings, decisions, methods, hypotheses, baselines, open-questions, patterns
- Domain schema: confidence (proven|likely|experimental|speculative), conditions [], meta_state (current|outdated|superseded)

---
description: Why each configuration dimension was chosen -- the reasoning behind initial system setup
category: derivation-rationale
created: 2026-02-18
status: active
---
# derivation rationale for USV research

This vault was derived for a USV (ultrasonic vocalization) research pipeline project. The system captures knowledge about mouse USV detection, classification, signal processing, CNN-based models, and experimental methods at 300 kHz sampling rate.

## Preset Selection

The Research preset was selected with high confidence based on strong domain signals: structured knowledge work with papers, experimental findings, DSP methods, and detection algorithms. The user works with a codebase for spectrogram generation, tiled PNG rendering, Zarr storage, USV detection pipelines, and CNN classifiers.

## Dimension Choices

**Granularity: Atomic** (High confidence) -- One insight per note maximizes composability. USV research spans detection, classification, DSP, training pipeline, and experimental methods. Atomic notes enable cross-cutting connections (e.g., an STFT parameter choice affects both detection quality and training data generation).

**Organization: Flat** (Inferred from preset) -- Flat-associative structure with semantic search. Research topics cross-cut: a finding about frequency resolution is relevant to detection, classification, AND spectrogram rendering. Folders would force artificial hierarchies.

**Linking: Explicit+Implicit** (High confidence, user opted into qmd) -- Wiki links for deliberate connections, qmd semantic search for discovery. The combination supports both known connections and serendipitous finds across the growing knowledge graph.

**Processing: Heavy** (High confidence) -- Full pipeline with all quality gates. Research content deserves careful extraction, connection-finding, and verification. The pipeline ensures notes earn their place in the graph.

**Navigation: 3-tier** (Inferred) -- Hub (index) -> domain topic maps (detection, classification, DSP, training) -> individual notes. Research domains naturally organize into topic clusters.

**Maintenance: Condition-based** (Inferred) -- Event-driven maintenance triggers rather than scheduled. Orphan accumulation, stale notes, and MOC overflow trigger specific actions.

**Schema: Moderate** (High confidence) -- Domain-specific fields for confidence tracking, experimental conditions, and meta-state. Not dense enough to create maintenance burden, but structured enough to support meaningful queries.

**Automation: Full** (High confidence) -- Claude Code platform enables full automation. All skills, hooks, and processing available from day one.

## Self-Space Decision

Self-space disabled (research preset default). Agent identity is encoded in CLAUDE.md. Goals and orientation route to ops/goals.md. This keeps the vault focused on domain knowledge rather than agent self-reflection.

## Personality Decision

Neutral-helpful (default). No personality signals detected during conversation. Research domain defaults to professional, task-focused communication.

## Platform

Claude Code with full automation. Existing hooks (check_agents_tag.ps1, check_plan_mode.ps1) preserved via additive merge. New hooks added for session orient, capture, note validation, and auto-commit.

---

Topics:
- [[methodology]]

---
description: "Institut Pasteur's standalone toolbox — tuned for same 300 kHz AviSoft recordings, includes WAV-behavior sync and burst analysis scripts directly relevant to Phase 16"
type: finding
confidence: likely
conditions: []
meta_state: current
topics:
  - "[[experimental-methods]]"
---

# LMT USV Toolbox provides Python-based offline USV processing as a reference implementation

The Live Mouse Tracker project provides a standalone Python toolbox for offline USV processing, available on GitHub. This is a natural comparison point for our custom detection pipeline — the LMT toolbox is developed by the same group (Institut Pasteur) that builds the behavioral tracking system we use. Comparing detection results, feature extraction approaches, and classification methods between our pipeline and the LMT toolbox could identify strengths and weaknesses in each approach. The toolbox is likely tuned for the same type of recordings (AviSoft Recorder output at 300 kHz from shared lab environments), making it a particularly relevant benchmark.

Critical implementation guidance: the toolbox's `LMT.USV.importer` already handles synchronization of WAV files with LMT behavioral data. Integration code should adapt this existing synchronization rather than rebuilding from scratch. The toolbox also provides scripts for acoustic analyses per behavioral context, burst analysis relating USV timing to behavioral events, and speed/duration/events with USV linking movement data to vocalizations. These are directly relevant to Phase 16 (LMT Integration).

---

Source:
- Researcher brain-dump on lab conventions (2026-02-19)
- https://micecraft.org/lmt/
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[Live Mouse Tracker from Institut Pasteur synchronizes vocalization recordings with social behavior events]] -- the parent system
- [[AviSoft Recorder captures synchronized USV recordings within the LMT behavioral tracking system]] -- same recording pipeline
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- our pipeline to compare against
- [[event-triggered USV rate via PETH in plus-minus 2 second windows per event type serves as LMT integration sanity check]] -- the specific analysis method that builds on this toolbox

Topics:
- [[experimental-methods]]

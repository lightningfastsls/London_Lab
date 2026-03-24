---
description: "SQLite file location, AviSoft recording method, wild-vs-lab labels, and LMT version are prerequisites that determine whether synchronization code can be adapted from LMT-USV-Toolbox"
type: open-question
confidence: speculative
conditions: []
meta_state: current
topics:
  - "[[experimental-methods]]"
---

# Pre-code questions for LMT integration must be resolved before implementation

Four concrete questions must be answered before writing any LMT integration code. These are not aspirational research questions — they are hard prerequisites that determine the shape (and feasibility) of the implementation.

**1. Where are the LMT SQLite files for your recordings?** The entire LMT integration workstream depends on having access to the behavioral tracking databases that correspond to the USV WAV files. Without these SQLite files, there is no behavioral data to integrate. The files should contain frame-by-frame animal positions, annotated behavioral events (approach, contact, follow, mount, etc.), and session metadata. If they are on a lab server, a network share, or a specific machine, the path needs to be known. If they do not exist for the current recordings, the LMT workstream should be deprioritized entirely — no point writing integration code for data that is not available.

**2. Were the WAV files recorded via LMT's AviSoft integration?** This is the most consequential technical question. If recordings were made through LMT's integrated AviSoft Recorder, then synchronization metadata (mapping video frames to audio samples) may already exist in the SQLite database. This would make the synchronizer module straightforward — just read the existing alignment. If recordings were made independently, synchronization must be reconstructed from recording start times, which introduces alignment uncertainty and requires careful timestamp calibration.

**3. Which mice in your dataset are wild-derived vs. lab strains?** The core scientific question involves comparing wild and lab mouse vocalizations. Population labels are needed for any comparative analysis. This metadata may live in the SQLite files, in a separate spreadsheet, or only in the researcher's head. It needs to be made explicit and machine-readable before the comparison pipeline can run.

**4. Which LMT version was used for tracking?** Different LMT versions may use different SQLite schemas. The db_loader module must target the correct schema version. If multiple LMT versions were used across recording sessions, the loader may need version-detection logic.

The resolution path is simple: ask these questions and wait for answers. Writing speculative code before knowing the answers risks building against wrong assumptions — for example, writing a complex timestamp reconstruction module when synchronization metadata already exists, or targeting the wrong SQLite schema version.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[Live Mouse Tracker from Institut Pasteur synchronizes vocalization recordings with social behavior events]] -- the system these questions are about
- [[LMT USV Toolbox provides Python-based offline USV processing as a reference implementation]] -- may provide schema documentation
- [[whether LMT SQLite schema supports the required temporal resolution for USV-behavior synchronization]] -- temporal resolution is one of the pre-code questions requiring real SQLite inspection
- [[LMT integration code belongs in dedicated src-usv_spectrogram-lmt subpackage]] -- the subpackage architecture these questions must inform

Topics:
- [[experimental-methods]]

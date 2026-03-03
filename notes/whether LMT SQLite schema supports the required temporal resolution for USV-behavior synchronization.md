---
description: "LMT video tracking runs at 30 fps while USVs last 10-500 ms — the SQLite schema's temporal granularity must be verified against USV timescales"
type: open-question
confidence: speculative
conditions: []
meta_state: current
topics:
  - "[[experimental-methods]]"
---

# Whether LMT SQLite schema supports the required temporal resolution for USV-behavior synchronization

LMT tracks animal behavior from video at approximately 30 fps, giving a temporal resolution of roughly 33 ms per frame. USVs, by contrast, can be as short as 10 ms — and per ADR-002, the STFT hop size of 128 samples at 300 kHz gives 0.427 ms temporal precision for detection timestamps. This creates a fundamental temporal resolution mismatch between the behavioral and acoustic data streams: behavioral events are localized to ~33 ms windows, while USV timing is known to sub-millisecond precision.

The practical impact depends on the analysis granularity. For peri-event time histogram (PETH) analysis with 100 ms bins — the standard approach for correlating USVs with behavioral events — 33 ms behavioral resolution is adequate. A behavioral event landing in the correct 100 ms bin requires only ~3x oversampling, which 30 fps provides. The USV rate within each bin will be computed from many calls, so per-call timing precision matters less than bin assignment accuracy.

However, for finer-grained temporal analyses, the mismatch becomes problematic. Consider the question "does USV onset precede or follow behavioral event onset?" If both events happen within a 33 ms window, the behavioral timestamp cannot resolve the ordering. This means any analysis claiming sub-33 ms temporal relationships between USVs and behavior is limited by the video frame rate, not by acoustic detection precision. The 33 ms jitter in behavioral timestamps could wash out real temporal structure at fine timescales.

Several questions need empirical investigation against the actual SQLite data:

1. Does LMT store behavioral events at frame resolution (integer frame indices) or does it interpolate to sub-frame timestamps? Some tracking systems use motion models to estimate inter-frame positions.
2. Are behavioral event annotations (approach, contact, mount) recorded as point events (single frame) or intervals (start frame, end frame)? Interval representation partially mitigates the resolution issue for sustained behaviors.
3. Does the SQLite schema include any higher-resolution timing metadata, such as hardware trigger timestamps from the AviSoft integration?

Until these questions are answered by examining real SQLite files, the synchronization module should be designed conservatively — assuming frame-locked 33 ms resolution and warning users when analyses require finer temporal precision. Since [[pre-code questions for LMT integration must be resolved before implementation]], this temporal resolution question is one more reason to inspect the actual data before writing integration code. The [[LMT USV Toolbox provides Python-based offline USV processing as a reference implementation]] and may document the actual schema fields and their temporal granularity, providing answers before manual SQLite inspection.

The resolution constraint has asymmetric impact across analysis tiers. For [[event-triggered USV rate via PETH in plus-minus 2 second windows per event type serves as LMT integration sanity check]] with 100 ms bins, 33 ms behavioral resolution is adequate. But finer-grained temporal analyses would be limited by the video frame rate. Whether [[AviSoft Recorder captures synchronized USV recordings within the LMT behavioral tracking system]] embeds higher-resolution hardware timestamps in the SQLite database could partially resolve this mismatch. The synchronizer.py module within the [[LMT integration code belongs in dedicated src-usv_spectrogram-lmt subpackage]] must encode this resolution constraint so downstream analyses know their temporal precision limits.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[Live Mouse Tracker from Institut Pasteur synchronizes vocalization recordings with social behavior events]] -- the system whose temporal resolution is in question
- [[temporal alignment between USV detections and LMT behavioral events enables USV-behavior correlation analysis]] -- the analysis this resolution constraint affects
- [[pre-code questions for LMT integration must be resolved before implementation]] -- the prerequisite questions that include schema inspection
- [[AviSoft Recorder captures synchronized USV recordings within the LMT behavioral tracking system]] -- may embed higher-resolution hardware timestamps in SQLite
- [[event-triggered USV rate via PETH in plus-minus 2 second windows per event type serves as LMT integration sanity check]] -- PETH 100ms bins tolerate 33ms resolution; finer analyses do not
- [[LMT integration code belongs in dedicated src-usv_spectrogram-lmt subpackage]] -- synchronizer.py must encode this resolution constraint
- [[LMT USV Toolbox provides Python-based offline USV processing as a reference implementation]] -- may document actual schema fields and temporal granularity

Topics:
- [[experimental-methods]]

---
description: The app maintains three distinct detection concepts — editable current, matched saved-current, and historical gray ghost — each serving a different UI and persistence role
type: decision
confidence: proven
meta_state: current
topics:
  - "[[detection]]"
---

# saved-previous ghost detections current editable and saved-current form three aligned detection state tiers in the app

The PyQt6 detection app distinguishes three conceptually separate detection states that must stay aligned:

1. **Current detections** — fresh results from the CNN sliding window for the loaded file segment; editable and savable by the user.
2. **Saved-current detections** — current detections whose boundaries match an existing tracker record within tolerance; rendered in blue to indicate they are already on disk.
3. **Saved-previous ghost detections** — historical records from the `SavedDetectionTracker` for time ranges that do NOT match any current detection; rendered in gray in the spectrogram view and probability view for context.

Ghost detections are loaded from the tracker and combined with current detections for rendering, but they are NOT editable and NOT included in the current-detection list that delete/save actions operate on. Keeping this distinction prevents delete actions from accidentally targeting ghost records.

The key invariant: tracker matching is boundary-identity (same start and end within 1ms), not overlap. This is what allows nearby distinct events to each have their own save state. Since [[JSON label files provide human-readable version-controllable persistence for detection labels and metadata]], the tracker's on-disk format (`_saved_tracking.json`) follows the same JSON persistence pattern used for labels, enabling git-friendly inspection and recovery.

---

Source:
- Codex handoff `docs/handoffs/2026-03-07_app-save-and-ghost-detection-fixes.md`

Relevant Notes:
- [[JSON label files provide human-readable version-controllable persistence for detection labels and metadata]] — ghost detection state persists through the same JSON-per-WAV pattern
- [[noise-interrupted long USVs get split into two detections by the CNN sliding window]] — ghost detections help the user see previously saved segments that the current CNN pass may have split differently

Topics:
- [[detection]]

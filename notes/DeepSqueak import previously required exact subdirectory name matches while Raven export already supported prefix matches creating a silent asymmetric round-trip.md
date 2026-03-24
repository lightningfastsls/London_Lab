---
description: "Export side matched detection dirs by prefix (longest stem wins); import side required exact name match, breaking round-trips for suffixed directories like rec_001_retry/"
type: finding
confidence: proven
meta_state: current
topics:
  - "[[classification-tools]]"
---

# DeepSqueak import previously required exact subdirectory name matches while Raven export already supported prefix matches creating a silent asymmetric round-trip

`raven_export.py` has always supported a prefix-matching rule: a detection directory whose name starts with the WAV stem is treated as belonging to that WAV file, with longest-stem-wins disambiguation for ambiguous cases. This means detection directories named `rec_001_retry/` could be exported correctly for a WAV stem `rec_001`.

`deepsqueak_import.py`, before the 2026-03-07 fix, grouped detections only by exact subdirectory name. The result was a silent asymmetry: the same suffixed directory that exported correctly would fail to re-associate during import, producing `unmatched_ds=1` and `unmatched_det=1` rows without any error message. The two quantities canceling out made the failure easy to overlook in aggregate statistics.

The fix added DS-to-detection stem resolution at merge time rather than changing the raw detection loader. This keeps `load_detections_for_merge()` simple and localizes the compatibility rule to the place where DeepSqueak stems are actually available for comparison.

---

Source:
- docs/handoffs/2026-03-07_deepsqueak-import-prefix-match-fix.md (Codex, 2026-03-07)

Relevant Notes:
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] -- the export format whose naming conventions triggered this asymmetry
- [[timestamp proximity matching with configurable tolerance bridges detection systems that use different internal time representations]] -- the other merge-time coordination mechanism in the same pipeline
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- the end goal this bridge serves; silent mismatches here produce undetected data loss

Topics:
- [[classification-tools]]

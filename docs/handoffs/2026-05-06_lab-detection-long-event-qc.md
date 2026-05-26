# Handoff: Lab Detection Long-Event QC
Date: 2026-05-06

## Task

Continue the lab mouse detection investigation by adding production batch safeguards for long, stable ultrasonic noise bands that the CNN pipeline can mis-detect as USVs.

## Files Changed

- `src/usv_spectrogram/postprocessing/hysteresis.py` - added `HysteresisConfig.max_duration_ms`, defaulting to 600 ms, and filters regions longer than that after min-duration filtering.
- `src/usv_spectrogram/postprocessing/triage.py` - added QC thresholds and flags for long events, high total detected duration, high event count, high noise floor, and events spanning most of a chunk/recording; structural artifact flags now route event-bearing recordings to `manual_review`.
- `src/usv_spectrogram/postprocessing/batch_output.py` - added `qc_flags` to `summary.parquet` output so review reasons survive batch export.
- `scripts/run_batch_detection.py` - loads optional `max_duration_ms` from hysteresis optimization JSON, defaulting old configs to 600 ms.
- `tests/test_hysteresis.py` - added max-duration validation and long-event filter coverage.
- `tests/test_triage.py` - added QC flag and manual-review routing coverage.
- `docs/modules/hysteresis-detection.md` - documented the 600 ms lab noise guard.
- `docs/modules/recording-triage.md` - documented the new QC thresholds, flags, and tier behavior.

## Reasoning

The inspected lab false positives are long, stationary ultrasonic bands, while the repo reference notes true mouse USVs are usually 10-300 ms. A batch max-duration gate is the lowest-risk production fix because it catches the exact failure mode before downstream exports treat it as science-ready. The production threshold is 600 ms to allow occasional merged detections while still rejecting multi-second bands. Triage still separately flags long or overloaded chunks; only structural artifact flags force manual review so normal high-activity chunks are not broadly downgraded.

The default max duration is 600 ms rather than the single-call 300 ms reference because some legitimate neighboring calls can merge in postprocessing. `max_duration_ms=None` remains available for reruns where long detections should be retained for validation set construction.

## Validation

- `.venv/bin/python -m py_compile src/usv_spectrogram/postprocessing/hysteresis.py src/usv_spectrogram/postprocessing/triage.py src/usv_spectrogram/postprocessing/batch_output.py scripts/run_batch_detection.py`
- `.venv/bin/python -m pytest tests/test_hysteresis.py tests/test_triage.py -q`
- 100-file lab regression sample from existing detections, selected from `results/batch_lab_131204_full/detections` with all old events <= 600 ms, rerun into `results/codex_detection_compare_100/output_600ms_tight_triage`.

Result: 48 passed. The 100-file rerun produced 100/100 matching detection JSONs for event count, boundaries, columns, and durations. Max probability delta was `1.19e-07`; max mean-probability delta was `4.47e-08`. Triage tiers also matched after limiting tier-blocking QC to structural artifact flags.

## Open Questions / Known Risks

- Existing optimized hysteresis JSON files do not contain `max_duration_ms`; the loader now defaults them to 600 ms. Historical reruns that require exact old behavior should pass a config with `"max_duration_ms": null`.
- `duration_ms` is center-to-center between window centers, so the physical event span is approximately one inference step longer than reported. This preserves existing semantics but means the 600 ms gate is slightly permissive in physical duration.
- The current QC flags are rule-based. They should be calibrated against labeled lab hard negatives/positives before treating auto-accepted lab detections as final.

## Worth Remembering For Claude

The production batch pipeline now rejects long CNN-hysteresis events by default and exports QC flags. This is a conservative guard against lab stationary-band false positives, not a substitute for lab-specific hard-negative retraining.

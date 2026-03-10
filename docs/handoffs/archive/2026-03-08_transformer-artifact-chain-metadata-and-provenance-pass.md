# Handoff: Transformer Artifact Chain Metadata And Provenance Pass
Date: 2026-03-08

## Task

Continue the transformer-stack bug hunt under `usv_language/`, staying focused on the hidden-state artifact chain:
- extractor output
- layer comparison
- VQ-VAE checkpoint/config
- analysis input

Delivered:
- `run_analysis.py` now auto-loads adjacent extractor `metadata.json` when `--metadata` is omitted
- `run_analysis.py` now fails fast when an explicit metadata path is wrong instead of silently weakening validation
- `analysis_summary.json` now records the resolved artifact provenance used for the run
- `compare_layers.py` now surfaces extractor and per-layer VQ-VAE provenance in its markdown report
- synthetic integration-style coverage for the comparison and analysis hops of the chain
- a real tiny on-disk artifact-chain regression that executes extraction -> compare layers -> analysis
- `compare_layers.py` now forwards VQ-VAE windowing/split controls so tiny extracted datasets can participate in real comparison runs

## Files Changed

- `usv_language/analysis/run_analysis.py`
  Added `load_analysis_metadata()`, auto-discovery of adjacent extractor metadata, explicit missing-metadata failure, and artifact provenance capture in `analysis_summary.json`.
- `usv_language/training/compare_layers.py`
  Added `load_vqvae_run_config()`, expanded `comparison_report.md` to include extractor provenance plus per-layer VQ-VAE artifact provenance pulled from each layer run's `config.json`, and now forwards `window_size`, `stride`, and `val_fraction` to `train_vqvae.py`.
- `usv_language/tests/test_analysis.py`
  Added regressions for metadata auto-discovery, explicit missing metadata failure, summary-level artifact provenance capture during a partial-failure analysis run, and a real tiny on-disk chain regression covering extraction -> compare -> analysis.
- `usv_language/tests/test_hidden_state_vqvae.py`
  Added report-provenance coverage and a synthetic `compare_layers()` regression that writes a layer config and verifies the generated report surfaces that layer's artifact provenance.
- `docs/handoffs/current_bug_hunt.md`
  Updated the rolling thread handoff with this pass, the new validation baseline, and the remaining next target.

## Reasoning

The main robustness gap was not a missing low-level validation check anymore. It was that the normal CLI path for `run_analysis.py` still depended on the operator remembering to pass `--metadata`, even when the authoritative extractor metadata already lived adjacent to the hidden-state file. If the operator forgot that flag, analysis silently ran with weaker provenance checks and less useful context grouping.

That is an artifact-chain bug because it creates a difference between the safest path and the most likely path. The fix was to auto-discover adjacent metadata by default and reserve silent absence only for the case where no metadata file actually exists. Conversely, once the user explicitly supplies `--metadata`, a bad path should be treated as an error instead of being ignored.

I also surfaced provenance in the human-facing outputs because strict validation alone is not enough for diagnosis. If a comparison report or analysis summary does not say which checkpoint, hidden-state file, and metadata file were actually used, a human still has to reconstruct the chain manually when something looks wrong.

After that, the new real chain regression exposed a second integration issue: `compare_layers.py` wrapped `train_vqvae.py` but did not expose the trainer's smaller-window controls. That meant the comparison stage could fail on tiny extracted datasets for avoidable reasons even though the underlying trainer already supported them. Passing those controls through makes the comparison CLI behave more like a real chain component and less like a fixed-profile wrapper.

## Validation

- `.\.venv\Scripts\python.exe -m py_compile usv_language/analysis/run_analysis.py` : PASS
- `.\.venv\Scripts\python.exe -m py_compile usv_language/training/compare_layers.py` : PASS
- `.\.venv\Scripts\python.exe -m py_compile usv_language/tests/test_analysis.py` : PASS
- `.\.venv\Scripts\python.exe -m py_compile usv_language/tests/test_hidden_state_vqvae.py` : PASS
- `.\.venv\Scripts\python.exe -m pytest usv_language/tests/test_analysis.py -q -k tiny_artifact_chain_extract_compare_analyze` : PASS (`1 passed`)
- `.\.venv\Scripts\python.exe -m pytest usv_language/tests/test_analysis.py usv_language/tests/test_hidden_state_vqvae.py -q` : PASS (`76 passed`)
- `.\.venv\Scripts\python.exe -m pytest usv_language/tests -q` : PASS (`307 passed, 1 skipped`)

## Open Questions / Known Risks

- The tiny chain regression now executes extraction/compare/analysis on real artifacts, but it still patches a few heavy downstream analysis routines for runtime. The artifact validation path is real; the full statistical and plotting workload is not exercised end-to-end in one test.
- `run_analysis.py` still encodes section dependencies procedurally inside one entrypoint rather than through a small explicit dependency graph.
- Older VQ-VAE checkpoints without provenance still rely on filename and metadata compatibility checks rather than exact checkpoint-to-artifact provenance.

## Worth Remembering For Claude

- `run_analysis.py` now auto-discovers adjacent `metadata.json`, so the safest analysis path is also the default path.
- An explicit bad `--metadata` path is now a hard failure, which closes a silent-validation downgrade.
- `analysis_summary.json` now includes an `artifacts` block with resolved checkpoint/hidden-state/metadata provenance plus the checkpoint's stored VQ-VAE provenance.
- `comparison_report.md` now includes both extractor provenance and per-layer VQ-VAE artifact provenance, which makes mixed-artifact diagnosis much faster.
- `compare_layers.py` now passes VQ-VAE window/split controls through to the trainer, which matters for tiny artifact-chain coverage and small real datasets.
- The next target is no longer basic chain connectivity; it is whichever stale-artifact or mixed-current-state mismatches still survive despite the new real chain regression.

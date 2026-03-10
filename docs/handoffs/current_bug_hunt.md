# Handoff: Current Bug Hunt
Date: 2026-03-07

## Task

Maintain a rolling handoff for the active bug-hunt thread so either Codex or Claude can resume from the same state without relying on chat history.

Current state for this thread:
- Focused the pass on the transformer path under `usv_language/`, not the app workflow thread running in parallel.
- Fixed duplicated tail chunks during spectrogram chunking, which could silently duplicate frames during no-overlap extraction and distort hidden-state datasets.
- Fixed small-recording split behavior so validation remains available when possible without breaking normal split ratios on larger datasets.
- Hardened transformer checkpoint/analysis loading against config-format drift, stale resume epochs, missing artifacts, empty hidden-state arrays, and unavailable CUDA requests.
- Extended the hunt into `train_vqvae.py` and `compare_layers.py`: added VQ-VAE checkpoint config coercion, resume guards, hidden-state dataset shape/emptiness validation, and graceful per-layer failure reporting during layer comparisons.
- Stabilized the VQ-VAE overfit regression by seeding model initialization before construction; the assertion stayed strict.
- Tightened extractor/comparison invariants: `extract_hidden_states.py` now rejects a `primary_layer` that is not being extracted, and `compare_layers.py` validates each hidden-state file against shared `metadata.json` before training on it.
- Made `run_analysis.py` degrade gracefully after downstream failures: section-level failures are recorded in `analysis_summary.json`, independent sections still run, and dependent sections are skipped explicitly instead of aborting the whole suite.
- Added hidden-state provenance to VQ-VAE training outputs and checkpoints, and `run_analysis.py` now rejects semantically wrong pairings between a VQ-VAE checkpoint, hidden-state file, metadata, and `source_layer` even when tensor shapes happen to match.
- Closed the remaining normal-CLI metadata gap: `run_analysis.py` now auto-loads adjacent extractor `metadata.json` when `--metadata` is omitted, fails fast when an explicit metadata path is wrong, records resolved artifact provenance in `analysis_summary.json`, and `compare_layers.py` now surfaces extractor/VQ-VAE provenance in `comparison_report.md`.
- Added a real tiny synthetic artifact-chain regression that executes extraction -> compare layers -> VQ-VAE checkpoint -> analysis on disk artifacts, and fixed `compare_layers.py` so VQ-VAE window/stride/val-fraction settings can be passed through instead of being silently pinned to trainer defaults.

## Files Changed

- `usv_language/data/dataset.py`
  Fixed terminal chunk duplication and made recording-level split allocation robust for both tiny and normal datasets.
- `usv_language/training/train_transformer.py`
  Added checkpoint config coercion and explicit resume guards for config mismatch and already-finished epoch ranges.
- `usv_language/training/extract_hidden_states.py`
  Added missing-checkpoint failure, checkpoint config coercion for older/newer payloads, and validation that `primary_layer` is included in the requested extraction layers.
- `usv_language/analysis/run_analysis.py`
  Added config coercion, required-artifact checks, hidden-state shape/dimension validation, metadata layer compatibility checks, CPU fallback when CUDA is unavailable, section-level failure isolation with summary reporting, and explicit compatibility checks for filename layer, metadata frame/width, and VQ-VAE checkpoint provenance.
- `usv_language/analysis/run_analysis.py`
  Now auto-discovers adjacent extractor metadata, rejects missing explicit metadata paths, and writes resolved transformer/VQ-VAE/hidden-state/metadata provenance into `analysis_summary.json`.
- `usv_language/tests/test_spectrogram_transformer.py`
  Added regressions for terminal chunk duplication, 2-recording split behavior, and invalid extraction-layer requests.
- `usv_language/tests/test_analysis.py`
  Added regressions for VQ-VAE config coercion, device fallback, empty hidden-state sample rejection, partial analysis failure isolation, VQ-VAE provenance mismatch rejection, filename-layer mismatch rejection, metadata frame-count mismatch rejection, and out-of-range source-layer rejection.
- `usv_language/tests/test_analysis.py`
  Added regressions for metadata auto-discovery, missing explicit metadata paths, and artifact provenance capture in `analysis_summary.json`.
- `usv_language/tests/test_analysis.py`
  Added a real tiny on-disk chain regression that creates spectrogram bouts, runs hidden-state extraction, trains a one-layer comparison VQ-VAE, and runs analysis against the produced artifacts.
- `usv_language/training/train_vqvae.py`
  Added hidden-state file existence checks, hidden-state width validation, dataset/val-fraction parameter validation, VQ-VAE config coercion, resume guards matching the transformer trainer, hidden-state provenance capture, and resume-time provenance mismatch rejection.
- `usv_language/training/compare_layers.py`
  Layer comparison now records skipped/failed layers, validates hidden-state files against extractor metadata (`layers_extracted`, `total_frames`, `d_model`), and still emits a report instead of aborting the whole comparison on the first failure.
- `usv_language/training/compare_layers.py`
  Comparison reports now include extractor provenance plus per-layer VQ-VAE artifact provenance loaded from each layer run's `config.json`.
- `usv_language/training/compare_layers.py`
  Now forwards `window_size`, `stride`, and `val_fraction` to `train_vqvae.py`, which unblocks tiny extracted datasets from participating in real comparison runs.
- `usv_language/tests/test_hidden_state_vqvae.py`
  Added regressions for malformed hidden-state arrays, invalid dataloader fractions, dict-backed checkpoint configs, compare-layer failure reporting, metadata compatibility checks, checkpoint provenance persistence, hidden-state provenance extraction, and deterministic single-batch overfit setup.
- `usv_language/tests/test_hidden_state_vqvae.py`
  Added provenance-report coverage plus a synthetic `compare_layers()` run that verifies the generated report surfaces trained layer artifact provenance.
- `docs/handoffs/2026-03-08_transformer-artifact-chain-metadata-and-provenance-pass.md`
  Permanent handoff for the metadata auto-discovery and provenance-reporting pass.
- `docs/handoffs/archive/2026-03-07_transformer-robustness-bug-hunt-pass.md`
  Permanent handoff for the earlier transformer-path robustness pass.
- `docs/handoffs/archive/2026-03-07_vqvae-training-robustness-bug-hunt-pass.md`
  Permanent handoff for the earlier VQ-VAE/comparison-layer follow-up pass.
- `docs/handoffs/archive/2026-03-07_transformer-analysis-partial-failure-bug-hunt-pass.md`
  Permanent handoff for the metadata-compatibility and partial-analysis failure-isolation pass.
- `docs/handoffs/archive/2026-03-07_transformer-hidden-state-provenance-bug-hunt-pass.md`
  Permanent handoff for the hidden-state provenance compatibility pass.

## Reasoning

This thread stayed focused on concrete robustness bugs rather than speculative refactors.

Carry-forward details:
- `compare_layers.py` previously trusted file naming alone. That meant a stale or mismatched `hidden_states_layer{N}.npy` could be trained as if it matched the current extraction run even when `metadata.json` disagreed about available layers, frame count, or hidden width.
- `run_analysis.py` previously remained monolithic after upfront validation: one downstream plotting/statistics error aborted the whole CLI and lost independent outputs. The new behavior preserves what can still be produced and records the failure boundary in `analysis_summary.json`.
- `extract_hidden_states.py` could previously write self-contradictory metadata if `--primary-layer` was not included in `--layers`. That was an invariant bug because downstream tools treat `primary_layer` as provenance, not a loose hint.
- VQ-VAE checkpoints previously carried no hidden-state provenance, so a later analysis run could pair the checkpoint with the wrong hidden-state file or wrong layer if shapes matched. The checkpoint/config now records the source artifact, and analysis refuses mismatched pairings.
- `run_analysis.py` also now rejects mismatches between `source_layer` and the hidden-state filename or metadata dimensions, which closes remaining wrong-artifact cases for older checkpoints without provenance.
- `run_analysis.py` also previously required the operator to remember `--metadata` even when extractor metadata already lived adjacent to the hidden-state file. That weakened the normal CLI path unnecessarily and made explicit typos degrade silently instead of failing fast.
- The real end-to-end regression exposed a second integration issue: `compare_layers.py` already wrapped `train_vqvae.py`, but it did not expose the trainer's windowing and split controls. That made small extracted datasets fail inside the chain for avoidable reasons even though the underlying trainer supported smaller settings.

## Validation

- `.\.venv\Scripts\python.exe -m py_compile usv_language/analysis/run_analysis.py usv_language/training/compare_layers.py usv_language/tests/test_analysis.py usv_language/tests/test_hidden_state_vqvae.py` : PASS
- `.\.venv\Scripts\python.exe -m pytest usv_language/tests/test_analysis.py -q -k tiny_artifact_chain_extract_compare_analyze` : PASS (`1 passed`)
- `.\.venv\Scripts\python.exe -m pytest usv_language/tests/test_analysis.py usv_language/tests/test_hidden_state_vqvae.py -q` : PASS (`76 passed`)
- `.\.venv\Scripts\python.exe -m pytest usv_language/tests -q` : PASS (`307 passed, 1 skipped`)

## Open Questions / Known Risks

- `run_analysis.py` now isolates section failures and checks provenance/metadata compatibility, but the section dependency graph is still hand-written inside one CLI entrypoint rather than expressed declaratively.
- The tiny chain regression now executes extraction/compare/analysis on real artifacts, but it still patches a few heavy downstream analysis routines for runtime. The artifact validation path is real; the full statistical and plotting workload is not exercised end-to-end in one test.

## Worth Remembering For Claude

- This thread is the transformer-stack bug-hunt thread; app workflow bug hunting is happening separately.
- The earlier next-targets from the rolling handoff (`compare_layers.py` metadata compatibility and `run_analysis.py` partial-output behavior) are now addressed, and the remaining hidden-state provenance gap is also addressed.
- The next justified transformer-thread targets are whichever remaining artifact-chain mismatches still require either monkeypatching or manual diagnosis; the highest-value remaining gap is likely around stale artifact reuse or partial-current / partial-stale directories rather than basic chain connectivity.
- Hidden-state extraction depends on chunking not duplicating tail frames; the no-overlap extraction path is now safe.
- Current `usv_language` suite baseline after this pass is `307 passed, 1 skipped`.

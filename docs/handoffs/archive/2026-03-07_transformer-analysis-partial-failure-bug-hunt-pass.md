# Handoff: Transformer Analysis Partial-Failure Bug Hunt Pass
Date: 2026-03-07

## Task

Continue the transformer-stack bug hunt under `usv_language/`, focusing on the two next justified targets called out by the rolling handoff:
- `usv_language/training/compare_layers.py` metadata compatibility checks for `hidden_states_layer{N}.npy` artifacts
- `usv_language/analysis/run_analysis.py` behavior when some downstream analysis artifacts or modules fail

Secondarily inspect hidden-state extraction invariants when a concrete issue appears.

Delivered:
- metadata-backed hidden-state validation in `compare_layers.py`
- section-level failure isolation in `run_analysis.py`
- a concrete extractor invariant fix in `extract_hidden_states.py`
- regression coverage and widened suite validation

## Files Changed

- `usv_language/training/compare_layers.py`
  Added extractor metadata loading and per-layer artifact validation. Each layer file is now checked for existence, 2D shape, non-empty frame count, non-zero width, `layers_extracted` membership, `total_frames` match, and `d_model` match before VQ-VAE training starts.
- `usv_language/analysis/run_analysis.py`
  Reworked the CLI into section-level failure boundaries. Codebook visualization, sequence analysis, concept manipulation, context analysis, compositionality, and information theory now record `completed` / `failed` / `skipped` states in `analysis_summary.json`. Independent sections continue after a failure; dependent sections skip explicitly when prerequisites were not produced.
- `usv_language/training/extract_hidden_states.py`
  Added validation that `primary_layer` must be included in `--layers`, preventing self-contradictory extractor metadata.
- `usv_language/tests/test_hidden_state_vqvae.py`
  Added regressions for metadata-layer mismatch and metadata shape mismatch in `compare_layers.py`.
- `usv_language/tests/test_analysis.py`
  Added an integration-style regression proving that a failed sequence-analysis step does not abort the whole analysis CLI and that section statuses are recorded in `analysis_summary.json`.
- `usv_language/tests/test_spectrogram_transformer.py`
  Added a regression for invalid extractor layer requests (`primary_layer` not in `layers`).
- `docs/handoffs/current_bug_hunt.md`
  Updated the rolling thread handoff with the new fixes, validation baseline, and carry-forward targets.

## Reasoning

`compare_layers.py` previously trusted naming convention alone. That made it easy to train on stale or incompatible hidden-state files if `metadata.json` no longer matched what was in the directory. Since the extractor already writes authoritative provenance, the comparison script should reject those mismatches up front instead of quietly producing misleading per-layer results.

`run_analysis.py` had already been hardened at the artifact-validation boundary, but once execution started it was still effectively all-or-nothing. That is a real robustness bug for long analysis runs because one plotting/statistics failure can destroy unrelated outputs that are still valid to generate. The new section-level isolation keeps the CLI useful under partial failure without hiding the failure: the status and reason are written into `analysis_summary.json`.

The `extract_hidden_states.py` fix is small but important: `primary_layer` is recorded as provenance and used downstream. Allowing it to point at a layer that was never extracted turns valid-looking metadata into a lie.

## Validation

- `python -m py_compile usv_language/training/compare_layers.py usv_language/analysis/run_analysis.py usv_language/training/extract_hidden_states.py usv_language/tests/test_hidden_state_vqvae.py usv_language/tests/test_analysis.py usv_language/tests/test_spectrogram_transformer.py` : PASS
- `python -m pytest usv_language/tests/test_hidden_state_vqvae.py usv_language/tests/test_analysis.py usv_language/tests/test_spectrogram_transformer.py -q` : PASS (`77 passed`)
- `python -m pytest usv_language/tests -q` : PASS (`293 passed, 1 skipped`)

## Open Questions / Known Risks

- `run_analysis.py` now isolates failures, but the section ordering/dependencies are still encoded manually. If the analysis surface keeps growing, this should likely become a small dependency graph or separate subcommands.
- `compare_layers.py` now validates extractor metadata, but downstream VQ-VAE checkpoints still do not carry an explicit provenance field tying them back to the exact hidden-state artifact used for training.
- There is still no tiny end-to-end CLI regression that covers extraction -> comparison -> analysis as a single artifact chain.

## Worth Remembering For Claude

- The previously called-out next targets from the rolling handoff are now closed.
- `analysis_summary.json` is now the machine-readable place to inspect partial-analysis outcomes.
- The next justified transformer-thread work is likely end-to-end CLI provenance/integration coverage rather than more local defensive checks in these same files.

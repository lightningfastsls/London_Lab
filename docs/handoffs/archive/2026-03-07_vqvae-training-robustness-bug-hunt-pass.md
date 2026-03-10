# Handoff: VQ-VAE Training Robustness Bug Hunt Pass
Date: 2026-03-07

## Task

Continue the transformer-thread bug hunt into the next justified layer after the transformer/data/analysis pass, focusing on:
- VQ-VAE resume-from-checkpoint consistency
- malformed or degenerate hidden-state datasets
- comparison-layer behavior when some layer artifacts are missing or broken
- targeted regression stability

Delivered:
- `train_vqvae.py` hardening
- `compare_layers.py` failure isolation
- new regression coverage
- full `usv_language` validation after the pass

## Files Changed

- `usv_language/training/train_vqvae.py`
  Added `coerce_vqvae_config()`, early hidden-state file checks, hidden-state width validation against `d_model`, dataset parameter validation (`window_size`, `stride`, `batch_size`, `val_fraction`), malformed-array rejection, and fail-fast resume guards for config mismatch and already-completed epoch ranges.
- `usv_language/training/compare_layers.py`
  Comparison now records missing/failed layers and still writes a report, instead of aborting the whole sweep on the first exception.
- `usv_language/tests/test_hidden_state_vqvae.py`
  Added regressions for empty/non-2D hidden-state arrays, invalid dataset parameters, invalid validation fractions, dict-backed checkpoint config loading, compare-layer failure reporting, and seeded the single-batch overfit test before model construction to remove a real initialization flake.
- `docs/handoffs/current_bug_hunt.md`
  Updated the rolling transformer-thread handoff with the new pass and baseline.

## Reasoning

`train_vqvae.py` had the same core checkpoint drift problem that `train_transformer.py` previously had: checkpoints could reasonably store configs as dicts, resumed runs could be launched with a different architecture, and `--epochs` could point at an already-finished run without any clear failure. Those are consistency bugs, not just polish issues.

Its dataset path also accepted malformed `.npy` arrays too late. An empty array or non-2D array would only fail implicitly downstream, which is expensive and opaque during long runs. The new checks make those failures immediate and explicit.

`compare_layers.py` was brittle in exactly the way multi-artifact orchestration usually breaks: one missing hidden-state file or one failed VQ-VAE training job would abort the entire layer comparison. That is now isolated per layer, and the report records failures so the run still produces something actionable.

The VQ-VAE overfit regression started flaking once the test order shifted. The fix was to seed before model construction so the assertion remains strict while the initialization is deterministic.

## Validation

- `python -m py_compile usv_language/training/train_vqvae.py usv_language/training/compare_layers.py usv_language/tests/test_hidden_state_vqvae.py` : PASS
- `python -m pytest usv_language/tests/test_hidden_state_vqvae.py -q` : PASS (`36 passed`)
- `python -m pytest usv_language/tests -q` : PASS (`289 passed, 1 skipped`)

## Open Questions / Known Risks

- `compare_layers.py` still trusts file naming (`hidden_states_layer{N}.npy`) rather than checking `metadata.json` for `layers_extracted` compatibility.
- `run_analysis.py` remains all-or-nothing after argument validation; a future pass could allow modules to skip independently after partial upstream success.
- There is still no explicit CLI-level integration test that runs `train_vqvae.train()` end-to-end on a tiny synthetic `.npy` file.

## Worth Remembering For Claude

- The VQ-VAE training path now matches the transformer trainer’s checkpoint-drift safeguards.
- Comparison-layer runs no longer lose all progress when one layer file is missing or malformed.
- The `usv_language` baseline after this pass is `289 passed, 1 skipped`.

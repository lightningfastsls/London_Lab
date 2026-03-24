# Handoff: Transformer Robustness Bug Hunt Pass
Date: 2026-03-07

## Task

Continue the long autonomous transformer-stack bug hunt under `usv_language/`, with emphasis on:
- resume-from-checkpoint consistency
- empty/small dataset behavior
- sequence-length truncation and padding invariants
- hidden-state extraction shape/device mismatches
- analysis assumptions after partial artifact failure
- checkpoint/config compatibility drift

Delivered:
- concrete fixes in the transformer data/training/analysis path
- new regression coverage
- full `usv_language` validation

## Files Changed

- `usv_language/data/dataset.py`
  Fixed a tail-duplication bug in `chunk_spectrogram()` and reworked `split_recordings()` so very small datasets still produce usable validation/test splits without breaking large-dataset ratios.
- `usv_language/training/train_transformer.py`
  Added `coerce_transformer_config()` so checkpoints with dict-backed configs still load, and resume now rejects architecture drift and non-advancing epoch requests.
- `usv_language/training/extract_hidden_states.py`
  Added missing-checkpoint checks and reused the transformer config coercion path for extraction compatibility.
- `usv_language/analysis/run_analysis.py`
  Added `coerce_vqvae_config()`, `resolve_device()`, required-artifact checks, hidden-state dimensionality validation, empty-array rejection, and `layers_extracted` metadata checks.
- `usv_language/tests/test_spectrogram_transformer.py`
  Added regression tests for terminal partial-chunk duplication and 2-recording split behavior.
- `usv_language/tests/test_analysis.py`
  Added regression tests for CPU fallback when CUDA is unavailable, VQ-VAE config coercion from dict checkpoints, and empty hidden-state sample rejection.
- `docs/handoffs/current_bug_hunt.md`
  Updated the rolling bug-hunt continuity file for this transformer-thread pass.

## Reasoning

The most concrete bug was in `chunk_spectrogram()`: when a final partial chunk already reached the end of a bout, the "tiny remainder" fallback could still append an extra overlapping chunk anchored to the tail. That quietly duplicated frames and undermined the intended invariant behind hidden-state extraction with `overlap_ratio=0.0`.

The small-dataset split fix was necessary because the previous ratio rounding could easily yield zero validation recordings on 2-recording datasets and other pathological allocations. The replacement allocator preserves validation/test minima where feasible, then greedily assigns remaining recordings to whichever split is furthest below its requested ratio.

The checkpoint/analysis hardening is about real compatibility drift:
- checkpoints may store configs as dataclasses or plain dicts
- resumed runs should not silently proceed with a different architecture
- resuming should fail if `--epochs` would not advance training
- analysis should fail early and explicitly when upstream training/extraction produced missing, empty, or incompatible artifacts
- explicit CPU fallback is safer than letting a requested-but-unavailable CUDA device fail later and less clearly

## Validation

- `python -m py_compile usv_language/data/dataset.py usv_language/training/train_transformer.py usv_language/training/extract_hidden_states.py usv_language/analysis/run_analysis.py usv_language/tests/test_spectrogram_transformer.py usv_language/tests/test_analysis.py` : PASS
- `python -m pytest usv_language/tests/test_spectrogram_transformer.py usv_language/tests/test_bout_dataset.py usv_language/tests/test_analysis.py -q` : PASS (`58 passed`)
- `python -m pytest usv_language/tests -q` : PASS (`282 passed, 1 skipped`)

## Open Questions / Known Risks

- `usv_language/training/train_vqvae.py` likely needs the same config-coercion and resume guard treatment as `train_transformer.py`.
- `HiddenStateDataset` in `train_vqvae.py` still allows zero-frame inputs to become padded training windows; that may or may not be desirable, but it is the next obvious degenerate-data policy decision.
- `run_analysis.py` remains all-or-nothing. If one downstream module fails, the script still aborts instead of salvaging partial outputs.

## Worth Remembering For Claude

- The duplicated-tail-chunk bug was subtle but important because it directly contaminated hidden-state extraction assumptions downstream.
- The updated split allocator was validated against both tiny datasets and the existing large-dataset split-ratio test.
- The next best transformer-thread target is `train_vqvae.py`, not the app code.

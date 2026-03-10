# Handoff: Transformer Hidden-State Provenance Bug Hunt Pass
Date: 2026-03-07

## Task

Continue the transformer-stack bug hunt under `usv_language/` after the metadata-compatibility and partial-analysis pass, focusing on the next concrete remaining gap:
- provenance drift between extracted hidden states, trained VQ-VAE checkpoints, and later analysis inputs

Delivered:
- hidden-state provenance capture in VQ-VAE training outputs/checkpoints
- provenance validation in `run_analysis.py`
- extra compatibility checks for hidden-state filename layer and metadata frame/width
- regression coverage and widened suite validation

## Files Changed

- `usv_language/training/train_vqvae.py`
  Added `infer_hidden_state_layer()` and `build_hidden_state_provenance()`. Training now records the resolved hidden-state path, filename, inferred layer, and adjacent extractor metadata (when available) into `config.json` and every saved checkpoint. Resume now rejects provenance mismatches instead of silently resuming on a different hidden-state artifact.
- `usv_language/analysis/run_analysis.py`
  Added `validate_vqvae_provenance()` and `validate_analysis_inputs()`. Analysis now rejects:
  - a VQ-VAE checkpoint trained on a different hidden-state file
  - a `source_layer` that disagrees with checkpoint provenance
  - a `source_layer` outside the transformer's layer range
  - a hidden-state filename whose `layerN` suffix disagrees with `source_layer`
  - a metadata file whose `total_frames` or `d_model` disagrees with the hidden-state array
- `usv_language/tests/test_hidden_state_vqvae.py`
  Added regressions for provenance extraction from adjacent metadata and checkpoint provenance persistence.
- `usv_language/tests/test_analysis.py`
  Added regressions for provenance-path mismatch, provenance-layer mismatch, filename-layer mismatch, metadata frame-count mismatch, and out-of-range source-layer rejection.
- `docs/handoffs/current_bug_hunt.md`
  Updated the rolling transformer-thread handoff with the new provenance fixes and validation baseline.

## Reasoning

The previous passes hardened artifact validation and partial-failure handling, but there was still a realistic silent-mismatch case: if a VQ-VAE checkpoint and a hidden-state file happened to share width and roughly compatible assumptions, `run_analysis.py` could still run on the wrong pairing and produce plausible-looking but semantically invalid outputs.

That is a provenance bug, not just a convenience issue. The training stage is the only place that definitively knows which hidden-state artifact a VQ-VAE was trained on, so that provenance needs to be persisted there and enforced later during analysis and resume.

I also added filename-layer and metadata frame/width checks in analysis because they catch remaining mismatch cases for older checkpoints that predate the new provenance fields.

## Validation

- `python -m py_compile usv_language/training/train_vqvae.py usv_language/analysis/run_analysis.py usv_language/tests/test_hidden_state_vqvae.py usv_language/tests/test_analysis.py` : PASS
- `python -m pytest usv_language/tests/test_hidden_state_vqvae.py usv_language/tests/test_analysis.py -q` : PASS (`70 passed`)
- `python -m pytest usv_language/tests/test_analysis.py -q` : PASS (`29 passed`)
- `python -m pytest usv_language/tests -q` : PASS (`300 passed, 1 skipped`)

## Open Questions / Known Risks

- The comparison/reporting layer still does not surface provenance fields directly to the human reader; provenance is enforced in checkpoints/config, not yet highlighted in markdown reports.
- There is still no end-to-end CLI regression that exercises the full artifact chain from extraction through comparison/training into analysis.
- Older checkpoints without provenance still rely on filename/metadata compatibility checks rather than exact checkpoint-to-artifact provenance.

## Worth Remembering For Claude

- This closes the remaining silent-mismatch class where a VQ-VAE checkpoint could be paired with the wrong hidden-state artifact as long as tensor widths matched.
- `config.json` and saved VQ-VAE checkpoints now carry provenance that downstream tools can validate.
- The next justified work in this thread is probably end-to-end CLI integration coverage rather than more local defensive checks in the same modules.

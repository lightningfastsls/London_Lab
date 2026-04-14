# Handoff: Fix 3 Training Pipeline Bugs

**Date:** 2026-04-14
**Context:** Pre-deployment testing of `usv_language` transformer training pipeline found 3 bugs. Tests already written and confirmed to fail. Fix the bugs, make the tests pass.

## Bug #1 — BLOCKING: File naming mismatch between prepare_data and load_bout_spectrograms

**File:** `usv_language/training/train_transformer.py`, line 206
**Test:** `usv_language/tests/test_prepare_data.py::test_npy_naming_vs_load_compatibility`

`prepare_data.py` saves spectrograms as `spec_00000.npy` (line 210 of `usv_language/data/prepare_data.py`), but `load_bout_spectrograms()` globs for `*_bout*.npy` (line 206 of `train_transformer.py`). The training script cannot find files produced by the data prep script.

**Fix:** In `load_bout_spectrograms()`, change the npy glob from `*_bout*.npy` to `*.npy` (to match any .npy file). Also update the recording ID parsing to handle both naming conventions — files matching `*_bout*` use the existing `rsplit("_bout", 1)[0]` logic, others use the full stem.

```python
# Current (line 206):
npy_files = sorted(data_dir.glob("*_bout*.npy"))

# Fix: accept any .npy file
npy_files = sorted(data_dir.glob("*.npy"))
```

And the recording ID extraction (line 215):
```python
# Current:
rec_id = p.stem.rsplit("_bout", 1)[0]

# Fix: handle both naming conventions
if "_bout" in p.stem:
    rec_id = p.stem.rsplit("_bout", 1)[0]
else:
    rec_id = p.stem  # e.g., "spec_00001" as-is
```

**Note:** This is a minimal fix. A better long-term solution would also save HDF5 from `prepare_data.py` (which `load_bout_spectrograms` checks first), but the npy path fix unblocks training now.

## Bug #2 — MINOR: LayerNorm weight decay in build_optimizer

**File:** `usv_language/training/train_transformer.py`, lines 299-309
**Test:** `usv_language/tests/test_train_pipeline.py::test_build_optimizer_param_groups`

`build_optimizer()` assigns parameters to decay/no-decay groups by checking if the parameter *name* contains "bias", "norm", or "embed". But LayerNorm weights inside `nn.Sequential` get names like `input_proj.2.weight` and `output_head.0.weight` — these don't contain "norm" so they incorrectly receive weight decay.

**Fix:** Replace the name-based check with a module-type check:

```python
# Current approach (lines 302-308):
for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if "bias" in name or "norm" in name or "embed" in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)

# Fix: use module types instead of name substrings
no_decay_set = set()
for module in model.modules():
    if isinstance(module, (nn.LayerNorm, nn.Embedding)):
        for param in module.parameters():
            no_decay_set.add(id(param))

for name, param in model.named_parameters():
    if not param.requires_grad:
        continue
    if id(param) in no_decay_set or "bias" in name:
        no_decay_params.append(param)
    else:
        decay_params.append(param)
```

## Bug #3 — DEFENSIVE: masked_mse_loss backward crash on all-padding batch

**File:** `usv_language/training/train_transformer.py`, line 168
**Test:** `usv_language/tests/test_train_pipeline.py::test_masked_mse_loss_all_padding_backward`

When `n_valid == 0` (all-padding batch), the function returns `torch.tensor(0.0, device=...)` which has `requires_grad=False`. Calling `.backward()` on this crashes.

**Fix:** Return a graph-connected zero instead:

```python
# Current (line 168):
if n_valid == 0:
    return torch.tensor(0.0, device=predictions.device)

# Fix:
if n_valid == 0:
    return (predictions * 0).sum()
```

## Verification

After fixing all 3 bugs, run:

```bash
.venv/bin/python -m pytest usv_language/tests/test_train_pipeline.py usv_language/tests/test_prepare_data.py -v
```

Expected: all 22 tests pass (0 failures). Then run the full suite:

```bash
.venv/bin/python -m pytest usv_language/tests/ -v
```

Expected: 341 passed, 0 failed, 1 skipped.

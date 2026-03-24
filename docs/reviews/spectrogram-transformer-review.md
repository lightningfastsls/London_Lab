# Spectrogram Transformer (v2 Phase 1) Module Review

**Reviewed by:** Master Reviewer (Sonnet 4.6)
**Date:** 2026-02-20
**Module:** `usv_language/models/` + `usv_language/training/`
**Tier:** 3 (Critical — ML model + training pipeline)
**Verdict:** CHANGES NEEDED

---

## BLOCKER (must fix before Phase 8.3)

### B1. DataParallel + resume checkpoint key mismatch

**Files:** `usv_language/training/train_transformer.py:385-393` (load_checkpoint), `usv_language/training/train_transformer.py:465-485` (call site)
**Problem:** `save_checkpoint` correctly unwraps DataParallel before saving (`model.module.state_dict()` — keys have no `module.` prefix). But at resume time, `load_checkpoint` is called AFTER DataParallel wrapping (line 465 wraps, line 485 loads). `DataParallel.load_state_dict()` expects `module.`-prefixed keys but the saved dict has none. Result: `RuntimeError: unexpected key(s) in state_dict` — blocks any DataParallel + resume workflow.
**Fix:** Add DataParallel awareness to `load_checkpoint`:

```python
def load_checkpoint(path, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(str(path), map_location="cpu", weights_only=False)
    # Unwrap DataParallel for loading
    target = model.module if hasattr(model, "module") else model
    target.load_state_dict(checkpoint["model_state_dict"])
    ...
```

### B2. CosineWarmupScheduler state_dict incomplete — LR corruption on resume

**Files:** `usv_language/training/train_transformer.py:95-101`
**Problem:** `state_dict()` saves only `step_count`. It does NOT save `warmup_steps`, `max_steps`, `min_lr`, or `base_lr`. `max_steps` is computed as `args.epochs * len(train_loader)`. If the user resumes with different `--epochs` (common practice), `max_steps` changes, which silently corrupts the cosine decay rate — the LR curve stretches or compresses relative to the original run. No error raised; training regresses silently.
**Fix:** Save and restore all schedule-defining fields:

```python
def state_dict(self) -> dict:
    return {
        "step_count": self.step_count,
        "warmup_steps": self.warmup_steps,
        "max_steps": self.max_steps,
        "min_lr": self.min_lr,
        "base_lr": self.base_lr,
    }

def load_state_dict(self, state: dict) -> None:
    self.step_count = state["step_count"]
    if "warmup_steps" in state:
        self.warmup_steps = state["warmup_steps"]
    if "max_steps" in state:
        self.max_steps = state["max_steps"]
    if "min_lr" in state:
        self.min_lr = state["min_lr"]
    if "base_lr" in state:
        self.base_lr = state["base_lr"]
```

---

## WARNINGS (fix before HPC training — Phase 11)

### W1. Dead code: `real_len` fetched but unused in extract_hidden_states

**File:** `usv_language/training/extract_hidden_states.py:146-147`
**Problem:** `chunk_info = dataset._chunks[chunk_idx]` / `_, real_len = chunk_info` is fetched but `real_len` is never used — `valid_len` from `mask.sum()` is used instead. Accesses private `_chunks` attribute unnecessarily. Confusing for future readers.
**Fix:** Delete lines 146-147. `valid_len` from `mask.sum()` is sufficient.

### W2. `patience_counter` not persisted in checkpoints

**File:** `usv_language/training/train_transformer.py:480` (init), `:369-376` (checkpoint dict)
**Problem:** Early stopping patience counter resets to 0 on every resume. A model at 15/20 epochs of no improvement restarts at 0/20 after resume, delaying early stopping by up to `patience` additional epochs.
**Fix:** Add `patience_counter` and `best_val_loss` to the checkpoint dict:

```python
# In save_checkpoint:
"patience_counter": patience_counter,
"best_val_loss": best_val_loss,

# In load_checkpoint return / resume handling:
patience_counter = ckpt.get("patience_counter", 0)
best_val_loss = ckpt.get("best_val_loss", float("inf"))
```

### W3. `--resume` with nonexistent path silently starts fresh training

**File:** `usv_language/training/train_transformer.py:482-489`
**Problem:** `if resume_path.exists(): load_checkpoint(...)` silently skips if the file doesn't exist. A typo'd checkpoint path starts fresh training and potentially overwrites a previous best checkpoint. Dangerous in HPC jobs.
**Fix:** Emit a warning (or error) when path doesn't exist:

```python
if not resume_path.exists():
    logger.warning("Resume checkpoint not found: %s -- starting from scratch", resume_path)
else:
    ckpt = load_checkpoint(...)
```

### W4. Test docstring says "8 test cases" but there are 11

**File:** `usv_language/tests/test_spectrogram_transformer.py:1-12`
**Problem:** Module docstring enumerates 8 tests but the file contains 11 (8 standalone + 3 config validation).
**Fix:** Update docstring to reflect 11 tests.

### W5. Missing module documentation

**File:** `docs/modules/spectrogram-transformer.md` — DOES NOT EXIST
**Problem:** All modules require a module doc per CLAUDE.md workflow. This module has none.
**Fix:** Create `docs/modules/spectrogram-transformer.md` with purpose, public interface, usage examples, and key decisions. Can reference the handoff for architecture details.

---

## SUGGESTIONS (nice to have)

| # | Issue | File | Fix |
|---|-------|------|-----|
| S1 | `weights_only=False` in `torch.load` — security concern for shared checkpoints | `train_transformer.py:388`, `extract_hidden_states.py:56` | Save config as plain dict instead of pickle'd dataclass, enabling `weights_only=True` |
| S2 | `masked_mse_loss` returns graph-detached tensor for `n_valid==0` edge case | `train_transformer.py:137` | Return `predictions.sum() * 0.0` to preserve computation graph |
| S3 | `return_hidden_states=True` during training multiplies memory (no guard) | `usv_language/models/transformer.py:239` | Add docstring warning or `if self.training: raise ValueError(...)` guard |
| S4 | NameError if `--epochs` equals `start_epoch` (loop never runs, `epoch` undefined) | `train_transformer.py:507-561` | Add `epoch = start_epoch - 1` before loop or early-exit guard |

---

## Summary

| Severity | Count | Items |
|----------|-------|-------|
| BLOCKER | 2 | B1 (DataParallel resume), B2 (scheduler state) |
| WARNING | 5 | W1 (dead code), W2 (patience reset), W3 (silent resume fail), W4 (docstring), W5 (missing module doc) |
| SUGGESTION | 4 | S1 (weights_only), S2 (masked_mse edge), S3 (hidden state guard), S4 (NameError) |

---

## What Passed

| Area | Verdict | Notes |
|------|---------|-------|
| DSP correctness | PASS | No STFT computations; downstream of spectrogram pipeline |
| Spec compliance | PASS | Pre-norm, causal masking, learned positional embeddings, ~25.6M params, masked MSE, AdamW decay groups all match ROADMAP |
| ML rigor | PASS | No data leakage, eval/train mode gates correct, seeds set, splits by recording upstream |
| Integration | PASS | Mask convention inversion (`~mask.bool()`) correct in train + validate |
| Test quality | PASS | 11/11 tests meaningful; overfit test, causal mask test, checkpoint round-trip all sound |

---

## Documentation Status

| Doc | Status | Issues |
|-----|--------|--------|
| Module doc (`docs/modules/spectrogram-transformer.md`) | MISSING | W5 — needs creation |
| `docs/architecture/patterns.md` | UP TO DATE | No new patterns; existing patterns followed |
| `DECISIONS.md` | UP TO DATE | ADR-007 covers this architecture |
| `IMPLEMENTATION_PROGRESS.md` | UPDATED | Correctly updated 2026-02-20 |

---

## Verdict

**CHANGES NEEDED**

Priority order:
1. **B1** — Fix DataParallel + resume key mismatch (single-GPU unaffected, but blocks multi-GPU resume)
2. **B2** — Fix CosineWarmupScheduler state_dict (silent LR corruption on resume with different epochs)
3. **W2** — Persist patience_counter in checkpoints (important for long HPC runs)
4. **W3** — Warn on missing resume path (prevents silent data loss)
5. **W1, W4, W5** — Cleanup items (dead code, docstring, module doc)

After fixing B1 and B2, a **Tier 1 spot-check re-review** is sufficient to verify the fixes.

---

## Fix Log

| Item | Status | Fixed in | Date | Notes |
|------|--------|----------|------|-------|
| B1 | FIXED | train_transformer.py | 2026-02-20 | load_checkpoint unwraps DataParallel via model.module |
| B2 | FIXED | train_transformer.py | 2026-02-20 | state_dict saves warmup_steps, max_steps, min_lr, base_lr |
| W1 | FIXED | extract_hidden_states.py | 2026-02-20 | Removed dead _chunks access, use get_lengths() + close mmaps before trim |
| W2 | FIXED | train_transformer.py | 2026-02-20 | patience_counter persisted in checkpoint, restored on resume |
| W3 | FIXED | train_transformer.py | 2026-02-20 | Warn on missing resume path |
| W4 | FIXED | test_spectrogram_transformer.py | 2026-02-20 | Docstring updated to 11 tests |
| W5 | DEFERRED | | | Module doc — deferred to next session |
| S1 | DEFERRED | | | weights_only=True migration |
| S2 | DEFERRED | | | masked_mse zero-valid edge case |
| S3 | DEFERRED | | | Hidden state training guard |
| S4 | FIXED | train_transformer.py | 2026-02-20 | epoch = max(0, start_epoch - 1) before loop |

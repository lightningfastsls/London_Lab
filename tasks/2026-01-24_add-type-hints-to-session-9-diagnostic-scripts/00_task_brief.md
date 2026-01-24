# Task Brief

Title: Add Type Hints to Session 9 Diagnostic Scripts
Date: 2026-01-24

## Goal
Add complete type annotations to Session 9 diagnostic scripts without changing runtime behavior.

## Context
Assumptions:
- The four scripts are already present and runnable.
- Existing codebase accepts type hints and `from __future__ import annotations`.
Uncertainties:
- Exact types for some values (use `Any` when unclear).

## Scope
In scope:
- Add `from __future__ import annotations` where appropriate.
- Annotate all function and method signatures (params + return types).
- Use `Path` from `pathlib` and typing helpers (`Dict`, `List`, `Tuple`, `Optional`, `Union`, `Any`).
Out of scope:
- Any logic changes or refactors beyond enabling imports for typing.

## Constraints
Dependencies: No new packages.
Performance: No effect.
File ownership: Modify only the four scripts listed below.
API stability: Do not change behavior or CLI interfaces.
Style: Follow existing conventions; be explicit with `-> None` and library types (`pd.DataFrame`, `np.ndarray`, `torch.Tensor`).

## Acceptance criteria
- All functions/methods in the four scripts have type hints for parameters and returns.
- `from __future__ import annotations` is added if needed for forward references.
- Scripts compile with `python -m py_compile` for each file.

## File touch list
New files: None.
Modified files:
- `scripts/threshold_sweep.py`
- `scripts/compare_probability_distributions.py`
- `scripts/analyze_recording_performance.py`
- `scripts/extract_visual_samples.py`

## Plan (small diffs)
1) Inspect each script to inventory functions and current imports.
2) Add typing imports and annotate signatures.
3) Run `py_compile` for each file to validate syntax.

## Implementer instructions
Do:
- Use `Optional[X]` for nullable values and `Union[str, Path]` for flexible path inputs.
- Use `Any` when a precise type is unclear or too complex.
Do not:
- Modify logic or data handling.
- Add type-checker configuration or new dependencies.

## Verifier checklist
- Read `00_task_brief.md` and `10_impl_notes.md`.
- Run `python -m py_compile` for each touched script (use `.venv` python if present).
- Record commands and results in `20_verification.md`.

# Handoff: Windows PyQt Torch Pytest Bootstrap Fix
Date: 2026-03-07

## Task

Investigate the intermittent full-suite failure on Windows where `tests -q` could not be used as a reliable closeout signal because collection sometimes died while importing `torch`.

Delivered:

- reproduced the failure outside pytest as an import-order bug
- confirmed that importing `PyQt6` before `torch` triggers the same `c10.dll` / `WinError 1114` failure in this environment
- fixed pytest collection stability by preloading `torch` before any Qt-backed test modules are imported

## Files Changed

- `conftest.py`
  Added a repo-wide pytest bootstrap that imports `torch` first on Windows so collection order no longer determines whether PyTorch DLL initialization succeeds.

## Reasoning

This was not an app regression and not random process instability.

The key reproducer was:

- `python -c "from PyQt6.QtWidgets import QApplication; import torch"` -> FAIL
- `python -c "import torch; from PyQt6.QtWidgets import QApplication"` -> PASS

The same pattern held for `PyQt6.QtCore` and `PyQt6.QtGui`, which means the problem is effectively "any PyQt6 import before torch" in this environment.

That explains the earlier full-suite behavior:

- app-focused Qt tests import PyQt6 during collection
- later `tests/test_cnn_model.py` imports `torch`
- `torch` DLL initialization then fails with `WinError 1114`

The most pragmatic repo-local fix is a root `conftest.py` that imports `torch` first on Windows. That keeps the workaround in the test harness rather than application code, makes collection order-independent, and restores `pytest tests -q` as a valid closeout signal.

## Validation

- `python -m py_compile conftest.py` : PASS
- `python -m pytest tests/test_app_qt_integration.py tests/test_cnn_model.py -q` : PASS (`43 passed`)
- `python -m pytest tests -q` : PASS (`632 passed, 1 skipped`)

Reproduction notes before the fix:

- `python -c "import torch; print(torch.__version__)"` : PASS
- `python -c "from PyQt6.QtWidgets import QApplication; import torch"` : FAIL (`WinError 1114`, `c10.dll`)
- `python -c "import torch; from PyQt6.QtWidgets import QApplication"` : PASS

## Open Questions / Known Risks

The underlying native-runtime conflict between this Windows PyQt6 environment and PyTorch is still not explained at the DLL level; the fix only makes pytest robust against it.

If this environment is rebuilt later, it would still be worth checking whether a different PyQt / torch / Python combination removes the import-order sensitivity entirely.

## Worth Remembering For Claude

- The previous "full suite is not a reliable closeout signal" note is no longer true in this environment after the pytest bootstrap fix.
- On this machine, PyQt6 imported before torch is sufficient to trigger the torch DLL initialization failure.
- The workaround is intentionally test-only and lives in root `conftest.py`, not in app startup code.

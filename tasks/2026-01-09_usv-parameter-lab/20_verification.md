# Verification Notes

## Environment
OS: Windows (PowerShell)
Python version: 3.12.0
Virtual environment: Not activated (none detected)

## Commands executed (verbatim)
python -m py_compile scripts/usv_parameter_lab.py src/usv_spectrogram/param_lab/app.py src/usv_spectrogram/param_lab/metrics.py src/usv_spectrogram/param_lab/heuristic_detect.py src/usv_spectrogram/param_lab/sweep.py src/usv_spectrogram/param_lab/explain.py
python -m unittest tests/test_param_lab_segment.py
python -m unittest tests/test_param_lab_heuristic.py
python -m pip install -r requirements.txt
python -m pip install -r requirements.txt
python -m pip show soundfile scipy
python -m unittest tests/test_param_lab_segment.py
python -m unittest tests/test_param_lab_heuristic.py

## Results (pass/fail)
pass - py_compile for touched scripts
fail - tests/test_param_lab_segment.py (missing dependency: soundfile)
fail - tests/test_param_lab_heuristic.py (missing dependency: scipy)
fail - python -m pip install -r requirements.txt (timed out; stdout pipe error)
fail - python -m pip install -r requirements.txt (timed out, but packages appear installed)
pass - python -m pip show soundfile scipy (deps present after install)
pass - tests/test_param_lab_segment.py (after test fix + deps)
pass - tests/test_param_lab_heuristic.py (after deps)

## Output validation
Files produced:
Sizes:
Sanity checks:
None - no runtime outputs expected from compile/tests

## Failures (if any)
Root cause: Missing Python dependencies (`soundfile`, `scipy`) in the current environment; test wrote WAV without float subtype leading to quantization mismatch.
Fix summary: Installed dependencies; updated test WAV write to use float subtype; re-ran tests.
Rerun transcript: `python -m unittest tests/test_param_lab_segment.py` -> pass; `python -m unittest tests/test_param_lab_heuristic.py` -> pass.

## Rerun (verifier)
Commands:
python -m py_compile scripts/usv_parameter_lab.py src/usv_spectrogram/param_lab/app.py src/usv_spectrogram/param_lab/metrics.py src/usv_spectrogram/param_lab/heuristic_detect.py src/usv_spectrogram/param_lab/sweep.py src/usv_spectrogram/param_lab/explain.py
python -m unittest tests/test_param_lab_segment.py
python -m unittest tests/test_param_lab_heuristic.py
Results:
pass - py_compile for touched scripts
pass - tests/test_param_lab_segment.py
pass - tests/test_param_lab_heuristic.py

## 2026-01-09 follow-up (WAV path guard)

### Environment
OS: Windows (PowerShell)
Python version: 3.12.0
Virtual environment: `.venv` (used `.venv\\Scripts\\python.exe`)

### Commands executed (verbatim)
.venv\Scripts\python.exe -m py_compile src\usv_spectrogram\param_lab\app.py

### Results (pass/fail)
pass - py_compile for touched script

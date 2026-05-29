"""Benchmark the two baseline_mode options on a representative lab WAV.

Compares 'median_envelope' (current default, slow) vs 'percentile' on the
same input, to put hard numbers on the algorithmic-fix claim. No GPU
required.

Diagnostic-only -- imports prep + cleaning internals, does not modify
either.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from usv_spectrogram.classifier import CleaningConfig  # noqa: E402
from usv_spectrogram.classifier.cleaning_pipeline import (  # noqa: E402
    _apply_baseline_subtraction,
)
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "cnn_prepare_training_data", SCRIPTS / "cnn_prepare_training_data.py"
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def bench(spec_db: np.ndarray, mode: str, label: str) -> float:
    cfg = CleaningConfig(baseline_mode=mode)
    t0 = time.perf_counter()
    out = _apply_baseline_subtraction(spec_db, cfg, "bench")
    dt = time.perf_counter() - t0
    print(f"  [{dt:7.2f}s] {label}  shape={spec_db.shape}  out_dtype={out.dtype}  out_mean={float(out.mean()):+.3f}  out_std={float(out.std()):.3f}")
    return dt


def main() -> int:
    lab_wav = Path("/home/shachar/projects/mickey_london_lab/USV_lab_131204/131204_1400_m1fm1.wav")
    print(f"loading {lab_wav.name} ...")
    samples, sr = sf.read(str(lab_wav), dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]

    for trunc_s in (30, 60, 120):
        print(f"\n=== {trunc_s}s of audio ===")
        chunk = samples[: int(trunc_s * sr)]
        spec_db = prep._spectrogram_db(chunk, sr)
        # Run percentile first (cheap) then median_envelope (expensive).
        t_pct = bench(spec_db.copy(), "percentile", "percentile (per-bin 10th)")
        t_med = bench(spec_db.copy(), "median_envelope", "median_envelope (kernel 977)")
        print(f"  -> speedup percentile vs median_envelope: {t_med / t_pct:.1f}x")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

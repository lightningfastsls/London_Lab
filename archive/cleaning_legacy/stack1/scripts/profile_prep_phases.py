"""One-shot profiler for cnn_prepare_training_data internals.

Imports the prep's internal functions and times each phase on:
  1. A short wild WAV (5970, ~5s, 300 kHz) -- sanity check
  2. A truncated lab WAV (first 30s of a 600s 250 kHz lab recording) --
     extrapolatable estimate of per-recording cost

Does NOT modify the prep script or the cleaning pipeline; this is a
side-channel profile to inform a fix decision.
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

from usv_spectrogram.classifier import CleaningConfig, clean_spectrogram  # noqa: E402
from usv_spectrogram.classifier.cleaning_pipeline import (  # noqa: E402
    _apply_baseline_subtraction,
    _apply_global_mad,
    _apply_per_recording_zscore,
    _apply_soft_notch,
)
from usv_spectrogram.classifier.resample import (  # noqa: E402
    SOURCE_SAMPLE_RATE_HZ,
    TARGET_SAMPLE_RATE_HZ,
    resample_to_vocalmat,
)
# Import internals from prep script via importlib (it has hyphens? no, underscores)
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "cnn_prepare_training_data", SCRIPTS / "cnn_prepare_training_data.py"
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def _tic() -> float:
    return time.perf_counter()


def _toc(label: str, t0: float, extra: str = "") -> float:
    dt = time.perf_counter() - t0
    print(f"  [{dt:7.2f}s] {label} {extra}")
    return dt


def profile_wav(wav_path: Path, truncate_s: float | None, label: str) -> None:
    print(f"\n=== {label}: {wav_path.name} ===")
    cfg = CleaningConfig()  # match prep defaults

    t0 = _tic()
    samples, sr_in = sf.read(str(wav_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]
    if truncate_s is not None:
        samples = samples[: int(truncate_s * sr_in)]
    _toc("read", t0, f"sr={sr_in} samples={samples.shape[0]:,}")

    t0 = _tic()
    if sr_in == SOURCE_SAMPLE_RATE_HZ:
        samples_250 = resample_to_vocalmat(samples)
        sr_eff = TARGET_SAMPLE_RATE_HZ
    else:
        samples_250 = samples.astype(np.float32, copy=False)
        sr_eff = int(sr_in)
    _toc("resample", t0, f"-> sr={sr_eff} samples={samples_250.shape[0]:,}")

    t0 = _tic()
    spec_db = prep._spectrogram_db(samples_250, sr_eff)
    _toc("stft", t0, f"shape={spec_db.shape} dtype={spec_db.dtype}")

    # Profile each cleaning layer in isolation (input is the raw STFT-dB).
    print("  -- per-layer cleaning --")
    out = spec_db
    t0 = _tic()
    out = _apply_soft_notch(out, cfg, wav_path.stem)
    _toc("soft_notch (no-op without tonal lib)", t0)

    t0 = _tic()
    out = _apply_baseline_subtraction(out, cfg, wav_path.stem)
    _toc("baseline_subtraction (median_envelope)", t0)

    t0 = _tic()
    out = _apply_global_mad(out, cfg, wav_path.stem)
    _toc("global_mad", t0)

    t0 = _tic()
    out = _apply_per_recording_zscore(out, cfg, wav_path.stem)
    _toc("per_recording_zscore", t0)

    # Patch loop -- compute only, no PIL save (separately timed).
    print("  -- patch loop --")
    patch_duration_s = 0.22
    frames_per_patch = max(
        1, int(round(patch_duration_s * sr_eff / prep._VOCALMAT_STFT_HOP))
    )
    n_time = out.shape[1]
    n_patches = max(1, n_time // frames_per_patch)
    print(f"  expected patches: {n_patches:,}")

    # Time the per-patch compute (resize+uint8) without writing PNGs.
    t0 = _tic()
    n_sample = min(n_patches, 50)
    for i in range(n_sample):
        start = i * frames_per_patch
        end = min(start + frames_per_patch, n_time)
        slab = out[:, start:end]
        if slab.size == 0:
            continue
        _ = prep._spec_to_uint8_patch(slab)
    sample_dt = time.perf_counter() - t0
    per_patch = sample_dt / max(1, n_sample)
    extrapolated = per_patch * n_patches
    print(
        f"  [{sample_dt:7.2f}s] patch_compute on {n_sample} samples "
        f"-> {per_patch * 1000:.2f} ms/patch "
        f"=> extrapolated {extrapolated:.1f}s for {n_patches} patches"
    )


def main() -> int:
    # Wild 5970 short WAV
    wild_dir = Path("/home/shachar/projects/mickey_london_lab/5970 USV/")
    wild_wavs = sorted(wild_dir.glob("*.wav"))
    if wild_wavs:
        profile_wav(wild_wavs[0], None, "WILD 5970 (full duration)")

    # Lab 131204 long WAV -- truncate to 30s to keep the profile tractable.
    lab_dir = Path("/home/shachar/projects/mickey_london_lab/USV_lab_131204/")
    lab_wavs = sorted(lab_dir.glob("*.wav"))
    if lab_wavs:
        profile_wav(lab_wavs[0], 30.0, "LAB 131204 (first 30s only)")
        profile_wav(lab_wavs[0], 60.0, "LAB 131204 (first 60s -- doubling check)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

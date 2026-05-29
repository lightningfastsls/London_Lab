"""Module 18.2b: End-to-end training-data preparation for the lab CNN classifier.

Walks the VocalMat OSF download (snake_case class folders of 227x227 RGB PNGs),
processes our own lab/wild 300 kHz WAV recordings (resample → cleaning stack →
STFT → 0.22 s patches), assembles a unified per-call manifest, emits 50 random
sanity patches per cohort for human inspection, and writes 80/10/10
stratified train/val/test splits with strict recording-level grouping.

ROADMAP §18.2b drives every choice here. Constraints:

- C1: 300 kHz source → 250 kHz target via ``resample_to_vocalmat`` (5/6
  polyphase + Kaiser FIR). Never modify ``corpus.py``.
- C2: Global MAD normalization happens on the WHOLE spectrogram once, then
  patches are cropped from the normalized image. Per-window MAD is a known
  trap (silent USV loss).
- D1: Patch duration 0.22 s primary; the 0.08 s variant is deferred to 18.5.
- D5: Keep all 12 classes; class-weighted CE + focal loss + oversampling are
  produced as side effects of :func:`build_stratified_split`.

Usage::

    .venv/bin/python scripts/cnn_prepare_training_data.py \\
        --vocalmat-source data/vocalmat_full/ \\
        --lab-wav-dirs USV_lab_131204/ \\
        --wild-wav-dirs '5970 USV/' USV_3452_sample_reviewed/ \\
        --output-dir data/lab_cnn_training/ \\
        --patch-duration-s 0.22 \\
        --workers 4

The script is also importable: ``from cnn_prepare_training_data import main``;
calling ``main(argv)`` returns ``int`` exit code. The end-to-end smoke test
exercises this in-process entry point.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Path bootstrap: this script lives in scripts/; the package is at src/.
REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from usv_spectrogram.classifier import (  # noqa: E402
    TARGET_SAMPLE_RATE_HZ,
    CleaningConfig,
    clean_spectrogram,
)
from usv_spectrogram.classifier.dataset import (  # noqa: E402
    GRIMSLEY_12_CLASSES,
    build_stratified_split,
)
from usv_spectrogram.classifier.resample import (  # noqa: E402
    SOURCE_SAMPLE_RATE_HZ,
    resample_to_vocalmat,
)


# ---------------------------------------------------------------------------
# VocalMat folder-name → display-name mapping (matches OSF layout)
# ---------------------------------------------------------------------------

_FOLDER_TO_DISPLAY: dict[str, str] = {
    "noise": "Noise",
    "step_up": "Step up",
    "down_fm": "Down-FM",
    "short": "Short",
    "chevron": "Chevron",
    "up_fm": "Up-FM",
    "flat": "Flat",
    "two_steps": "Two steps",
    "step_down": "Step down",
    "complex": "Complex",
    "rev_chevron": "Reverse Chevron",
    "mult_steps": "Multi-steps",
}
assert set(_FOLDER_TO_DISPLAY.values()) == set(GRIMSLEY_12_CLASSES), (
    "Folder→display mapping diverges from GRIMSLEY_12_CLASSES — keep in sync."
)

# VocalMat STFT parameters from the Grimsley 2011 / Romoli VocalMat paper:
# Hamming-256 window, hop 128, NFFT 1024 at 250 kHz. Numerically the hop value
# matches ``corpus.STFT_HOP=128`` by coincidence — these are anchored to the
# VocalMat reference, NOT the corpus. Keep separate so a future corpus tweak
# does not silently change the classifier patch geometry (CNN training-grid
# invariant).
_VOCALMAT_STFT_WINDOW_LEN: int = 256
_VOCALMAT_STFT_HOP: int = 128
_VOCALMAT_STFT_NFFT: int = 1024

# AlexNet / VocalMat patch dimensions
_PATCH_SIDE: int = 227

# Sanity patches per cohort (ROADMAP target; clamped if cohort is smaller).
_SANITY_PATCHES_PER_COHORT: int = 50

# Lab and wild WAV patches are UNLABELED at prep time. They are emitted to
# ``domain_unlabeled.csv`` for use by Module 18.4 (DANN cage-invariance
# training) but kept OUT of the supervised train/val/test manifests so
# Module 18.3's VocalMat-anchored ResNet-18 is not poisoned by a
# placeholder label. Master-reviewer flagged this in WARNING 2 of
# ``docs/reviews/cnn-data-preparation-review.md``.


# ---------------------------------------------------------------------------
# VocalMat walk
# ---------------------------------------------------------------------------


def _walk_vocalmat(vm_root: Path) -> list[dict]:
    """Walk VocalMat OSF directory tree, return one manifest row per PNG.

    Each row's ``source_recording`` is unique
    (``vocalmat/<folder>/<png-stem>``) so the stratified-split allocator
    treats every PNG as its own recording-level unit. This is conservative
    (no inter-PNG correlation assumed) and matches the OSF layout where
    individual PNGs do not carry recording-session metadata.
    """
    rows: list[dict] = []
    for folder_name, display_name in _FOLDER_TO_DISPLAY.items():
        cls_dir = vm_root / folder_name
        if not cls_dir.is_dir():
            continue
        for png in sorted(cls_dir.glob("*.png")):
            rows.append(
                {
                    "path": str(png),
                    "class": display_name,
                    "source_recording": f"vocalmat/{folder_name}/{png.stem}",
                    "duration_ms": 220.0,  # VocalMat PNGs lack ground truth; placeholder
                }
            )
    return rows


# ---------------------------------------------------------------------------
# WAV → patches
# ---------------------------------------------------------------------------


def _spectrogram_db(samples: np.ndarray, sr: int) -> np.ndarray:
    """STFT magnitude in dB. Matches VocalMat conventions at 250 kHz."""
    from scipy.signal import stft as _stft

    _, _, spec = _stft(
        samples,
        fs=sr,
        window="hamming",
        nperseg=_VOCALMAT_STFT_WINDOW_LEN,
        noverlap=_VOCALMAT_STFT_WINDOW_LEN - _VOCALMAT_STFT_HOP,
        nfft=_VOCALMAT_STFT_NFFT,
        return_onesided=True,
        padded=False,
        boundary=None,
    )
    return 20.0 * np.log10(np.abs(spec) + 1e-10)


def _spec_to_uint8_patch(spec_2d: np.ndarray) -> np.ndarray:
    """Convert a cleaned (already MAD-normalized) sub-spectrogram to uint8.

    Output shape is exactly ``(_PATCH_SIDE, _PATCH_SIDE)`` greyscale. The
    caller stacks 3 channels for VocalMat compatibility.
    """
    from PIL import Image

    if spec_2d.size == 0:
        return np.zeros((_PATCH_SIDE, _PATCH_SIDE), dtype=np.uint8)
    lo, hi = float(spec_2d.min()), float(spec_2d.max())
    if hi - lo < 1e-9:
        scaled = np.zeros_like(spec_2d, dtype=np.uint8)
    else:
        scaled = ((spec_2d - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return np.asarray(
        Image.fromarray(scaled).resize(
            (_PATCH_SIDE, _PATCH_SIDE), resample=Image.Resampling.BILINEAR
        )
    )


def _wav_to_patches(
    wav_path: Path,
    patch_duration_s: float,
    out_dir: Path,
    cohort: str,
    cfg: CleaningConfig,
) -> list[dict]:
    """Load one WAV → resample → clean → STFT → emit patches and manifest rows."""
    from PIL import Image
    import soundfile as sf

    samples, sr_in = sf.read(str(wav_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]

    if sr_in == SOURCE_SAMPLE_RATE_HZ:
        samples_250 = resample_to_vocalmat(samples)
        sr_eff = TARGET_SAMPLE_RATE_HZ
    elif sr_in == TARGET_SAMPLE_RATE_HZ:
        samples_250 = samples.astype(np.float32, copy=False)
        sr_eff = TARGET_SAMPLE_RATE_HZ
    else:
        # Unknown rate: bypass resampling, let the cleaner reject if needed.
        samples_250 = samples.astype(np.float32, copy=False)
        sr_eff = int(sr_in)

    spec_db = _spectrogram_db(samples_250, sr_eff)
    cleaned = clean_spectrogram(spec_db, cfg, recording_id=wav_path.stem)

    frames_per_patch = max(
        1, int(round(patch_duration_s * sr_eff / _VOCALMAT_STFT_HOP))
    )
    n_time = cleaned.shape[1]
    n_patches = max(1, n_time // frames_per_patch)

    cohort_dir = out_dir / "patches" / cohort
    cohort_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for i in range(n_patches):
        start = i * frames_per_patch
        end = min(start + frames_per_patch, n_time)
        slab = cleaned[:, start:end]
        if slab.size == 0:
            continue
        gray = _spec_to_uint8_patch(slab)
        rgb = np.stack([gray, gray, gray], axis=-1)
        patch_path = cohort_dir / f"{wav_path.stem}_p{i:04d}.png"
        Image.fromarray(rgb, mode="RGB").save(patch_path)
        rows.append(
            {
                "path": str(patch_path),
                "cohort": cohort,
                "source_recording": wav_path.stem,
                "duration_ms": patch_duration_s * 1000.0,
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Sanity patches
# ---------------------------------------------------------------------------


def _write_sanity_patches(
    cohort_rows: dict[str, list[dict]],
    sanity_dir: Path,
    rng: np.random.Generator,
) -> int:
    """Copy up to ``_SANITY_PATCHES_PER_COHORT`` random patches per cohort.

    Returns the total file count written (sum across cohorts).
    """
    from PIL import Image

    sanity_dir.mkdir(parents=True, exist_ok=True)
    written = 0
    for cohort, rows in cohort_rows.items():
        if not rows:
            continue
        n_sample = min(_SANITY_PATCHES_PER_COHORT, len(rows))
        idxs = rng.choice(len(rows), size=n_sample, replace=False)
        for i, idx in enumerate(idxs):
            src = Path(rows[int(idx)]["path"])
            if not src.exists():
                continue
            dst = sanity_dir / f"{cohort}_{i:02d}_{src.name}"
            Image.open(src).save(dst)
            written += 1
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Module 18.2b: prepare unified training-data manifest + 80/10/10 "
            "stratified split with recording-level grouping for the lab CNN "
            "classifier."
        ),
    )
    p.add_argument(
        "--vocalmat-source",
        type=Path,
        required=True,
        help="Root of VocalMat OSF download (snake_case class folders).",
    )
    p.add_argument(
        "--lab-wav-dirs",
        type=Path,
        nargs="+",
        required=True,
        help="One or more lab WAV directories (300 kHz).",
    )
    p.add_argument(
        "--wild-wav-dirs",
        type=Path,
        nargs="*",
        default=[],
        help="Optional wild WAV directories (300 kHz).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Where to write {train,val,test}/manifest.csv + sanity_patches/.",
    )
    p.add_argument(
        "--patch-duration-s",
        type=float,
        default=0.22,
        help="Patch duration in seconds (ROADMAP D1: 0.22 primary).",
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Currently informational; WAV loop is sequential to keep "
        "memory bounded for the smoke test.",
    )
    p.add_argument(
        "--skip-checksum-verify",
        action="store_true",
        help="Skip VocalMat checksum verification (not yet implemented).",
    )
    p.add_argument("--seed", type=int, default=1729)
    return p


def _collect_wav_rows(
    dirs: Iterable[Path],
    cohort: str,
    out_dir: Path,
    patch_duration_s: float,
    cfg: CleaningConfig,
) -> list[dict]:
    rows: list[dict] = []
    for root in dirs:
        if not root.is_dir():
            continue
        for wav in sorted(root.glob("*.wav")):
            rows.extend(
                _wav_to_patches(wav, patch_duration_s, out_dir, cohort, cfg)
            )
    return rows


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    if not args.vocalmat_source.exists():
        print(
            f"ERROR: --vocalmat-source does not exist: {args.vocalmat_source}",
            file=sys.stderr,
        )
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # --- Step 1: VocalMat manifest -----------------------------------------
    vm_rows = _walk_vocalmat(args.vocalmat_source)
    print(f"[vocalmat] {len(vm_rows)} PNG rows from {args.vocalmat_source}")

    # --- Step 2: lab + wild WAVs -> patches --------------------------------
    # baseline_mode='percentile' is required (not just preferred):
    # 'median_envelope' on 10-min lab WAVs floors most cells to _DB_TO_LINEAR_EPS;
    # global_mad then collapses MAD->0 and emits all-zero patches. Witnessed
    # on the 2026-05-22 prep attempt (2,734 all-black PNGs on the first lab
    # WAV). 'percentile' is also ~280x faster on CPU (3 s vs 250 s per 30 s
    # of audio). See docs/handoffs/2026-05-22_post-18.2b-download-followup.md
    # closure entry. The latent cleaning_pipeline.py integration bug
    # (degenerate global_mad after aggressive baseline subtraction) is
    # tracked as a future Tier-2 ticket.
    cfg = CleaningConfig(baseline_mode="percentile")
    lab_rows = _collect_wav_rows(
        args.lab_wav_dirs, "lab", args.output_dir, args.patch_duration_s, cfg
    )
    print(f"[lab]      {len(lab_rows)} patch rows from {len(list(args.lab_wav_dirs))} dir(s)")

    wild_rows = _collect_wav_rows(
        args.wild_wav_dirs, "wild", args.output_dir, args.patch_duration_s, cfg
    )
    print(f"[wild]     {len(wild_rows)} patch rows")

    # --- Step 3: sanity patches (human-review checkpoint) ------------------
    rng = np.random.default_rng(args.seed)
    sanity_dir = args.output_dir / "sanity_patches"
    written = _write_sanity_patches(
        {"vocalmat": vm_rows, "lab": lab_rows, "wild": wild_rows},
        sanity_dir,
        rng,
    )
    print(f"[sanity]   {written} patches in {sanity_dir}")

    # --- Step 4: supervised manifest (VocalMat only) + stratified split ----
    # Lab/wild WAV patches are UNLABELED and routed to a separate
    # domain_unlabeled.csv for Module 18.4 (DANN). They do NOT enter the
    # supervised manifest because the placeholder-label approach would
    # silently corrupt Module 18.3's training signal (master-reviewer
    # WARNING 2, 2026-05-22).
    supervised_manifest = pd.DataFrame(vm_rows)
    if len(supervised_manifest) == 0:
        print(
            "ERROR: VocalMat manifest is empty — check --vocalmat-source.",
            file=sys.stderr,
        )
        return 3

    (args.output_dir / "manifest_all.csv").write_text(
        supervised_manifest.to_csv(index=False), encoding="utf-8"
    )

    # Domain-adversarial unlabeled manifest (lab + wild). Empty is OK
    # if the user did not supply WAV dirs.
    domain_rows = lab_rows + wild_rows
    if domain_rows:
        domain_manifest = pd.DataFrame(domain_rows)
        domain_csv = args.output_dir / "domain_unlabeled.csv"
        domain_manifest.to_csv(domain_csv, index=False)
        print(
            f"[domain]   {len(domain_rows)} unlabeled lab+wild patches "
            f"-> {domain_csv} (for Module 18.4 DANN)"
        )

    split_scratch = args.output_dir / "_splits"
    split = build_stratified_split(
        supervised_manifest,
        train_frac=0.80,
        val_frac=0.10,
        seed=args.seed,
        out_dir=split_scratch,
    )

    # Reorganize into the conventional per-split subdirectory layout the
    # test harness (and Module 18.3 training script) expects.
    for src_csv, split_name in (
        (split.train_csv, "train"),
        (split.val_csv, "val"),
        (split.test_csv, "test"),
    ):
        dst_dir = args.output_dir / split_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        pd.read_csv(src_csv).to_csv(dst_dir / "manifest.csv", index=False)

    print(
        f"[done]     train/val/test manifests + sanity patches written to "
        f"{args.output_dir}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

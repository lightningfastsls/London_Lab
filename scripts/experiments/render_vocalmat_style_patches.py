"""α₃-C Phase A3 — render VocalMat-style spectrogram patches for the 131204 lab set.

For every detected call in ``classified_detections_lab_131204_clean.csv`` this
script renders a single 227×227 (3-channel) grayscale PNG patch and writes a
``manifest.csv`` that the A4 oracle (``scripts/experiments/label_patches_v1.py``)
can consume directly via ``--path-column path`` (+ optional ``--id-column call_id``).

----------------------------------------------------------------------------
WHY THIS EXISTS / WHERE IT SITS
----------------------------------------------------------------------------
α₃-C uses our in-house VocalMat reproduction (``results/lab_classifier_v1/best.pt``)
as a substrate-independent labeling oracle for shape-VAE evaluation. A4
(label_patches_v1.py) needs PNG patches to label. This A3 stage produces them
from the Stack-4 ("our cleaning pipeline") contour-masked focus spectrogram so
the patches the oracle sees are the cleaned representation, not raw STFT.

----------------------------------------------------------------------------
call_id CONVENTION
----------------------------------------------------------------------------
The CSV has no literal ``call_id`` column. We derive::

    call_id = f"{wav_stem}__det{det_index}"

Verified UNIQUE and STABLE: 40,787/40,787 rows yield distinct keys with zero
collisions and zero NaN ``det_index`` on the production CSV (checked 2026-05-29).
``det_index`` is the per-WAV hysteresis-event index, so the pair (wav_stem,
det_index) uniquely names a detection event. The existing held_out_844 manifest
used a ``<wav_stem>_<ms>ms.png`` filename convention but carried NO ``call_id``
column, so there was no prior convention to inherit — this is the first.

----------------------------------------------------------------------------
RENDER PIPELINE (per call), in order
----------------------------------------------------------------------------
1. Load the 2-second 300 kHz WAV chunk ``<wav-root>/<wav_stem>.wav`` and crop
   the DETECTION EVENT window [det_start_s, det_end_s] (NOT call_length_s — that
   is the DeepSqueak tonal sweep and differs by up to 10×; the patch must show
   the hysteresis event the PNG viewer shows). A small symmetric pad is added
   (--pad-ms, default 5 ms) before cropping, matching the focus-STFT's need for
   context around the event edges.
2. Q2: resample 300 → 250 kHz via classifier.resample.resample_to_vocalmat
   (5/6 polyphase + Kaiser FIR). --sample-rate 300000 is the documented fallback
   (skips resampling, runs the focus STFT at 300 kHz).
3. Stack 4: per-call adaptive focus STFT (scripts/deepsqueak_focus_stft.py) +
   contour/bandwidth mask (scripts/contour_mask_utils.py). Public functions only.
4. Render VocalMat-style PNG: 10·log10(P) → frequency-crop to > 45 kHz →
   mat2gray (min-max to [0,1]) → flipud → resize to 227×227 bicubic → 3-channel,
   matching cnn_prepare_training_data._spec_to_uint8_patch's min-max/flipud/
   3-channel structure (the spec overrides BILINEAR→BICUBIC at resize time).

----------------------------------------------------------------------------
GLOBAL (NOT PER-WINDOW) NORMALIZATION — CRITICAL FOOTGUN
----------------------------------------------------------------------------
Per repo rule ``feedback_cnn_inference_global_mad`` and CLAUDE.md: never apply
per-window MAD normalization (it silently kills high-confidence USVs). The
reference ``_spec_to_uint8_patch`` (archive/cleaning_legacy/stack1/scripts/
cnn_prepare_training_data.py:165-184) min-max scales over the WHOLE patch it is
handed in ONE pass::

    lo, hi = float(spec_2d.min()), float(spec_2d.max())
    scaled = ((spec_2d - lo) / (hi - lo) * 255.0)

We do exactly that: a single min-max over the entire (frequency-cropped) focus
spectrogram of this call — one normalization, then resize. No per-column / per-
window statistic is ever computed. Because Stack 4 here is a per-call focus STFT
(one spectrogram == one call, by construction), the "whole spectrogram" IS this
call's spectrogram; there is no larger image to crop windows out of, so the
per-window trap cannot arise.

----------------------------------------------------------------------------
NOT-TO-TOUCH
----------------------------------------------------------------------------
This script only READS/IMPORTS deepsqueak_focus_stft.py, contour_mask_utils.py
(Stack 4 canonical 🔒), classifier.resample, and corpus.py. It never writes to
any of them, to the model dir, or to the production detection pipeline. It only
writes PNGs + manifest.csv under --out-dir.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/experiments/render_vocalmat_style_patches.py \\
        --csv classified_detections_lab_131204_clean.csv \\
        --wav-root USV_lab_131204_chunked_2s_full/ \\
        --out-dir data/alpha3_patches \\
        --sample-rate 250000 \\
        --workers 8
"""
from __future__ import annotations

import argparse
import csv
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Path bootstrap: this script lives in scripts/experiments/; package is at src/,
# and the Stack-4 modules live in scripts/.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
SCRIPTS_ROOT = REPO_ROOT / "scripts"
for _p in (str(SRC_ROOT), str(SCRIPTS_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import soundfile as sf  # noqa: E402
from PIL import Image  # noqa: E402

from usv_spectrogram import corpus  # noqa: E402
from usv_spectrogram.classifier.resample import (  # noqa: E402
    SOURCE_SAMPLE_RATE_HZ,
    TARGET_SAMPLE_RATE_HZ,
    resample_to_vocalmat,
)

# Stack 4 canonical (🔒 read-only — import/call only).
from deepsqueak_focus_stft import (  # noqa: E402
    CallBox,
    create_focus_spectrogram,
    extract_contour_for_call,
)
from contour_mask_utils import apply_hard_bandwidth_mask  # noqa: E402

# ---------------------------------------------------------------------------
# Render constants
# ---------------------------------------------------------------------------
_PATCH_SIDE: int = 227                 # VocalMat / AlexNet patch side
_FREQ_CROP_MIN_KHZ: float = 45.0       # spec: frequency-crop to > 45 kHz
_DEFAULT_PAD_MS: float = 5.0           # symmetric context pad around the event
_LOG10_EPS: float = 1e-10              # matches _spectrogram_db's +1e-10

# Stack-4 mask defaults (match scripts/mass_apply_contour_mask.py canonical run:
# bandwidth ±5 kHz, tonality_threshold 0.0 — keep every contour bin).
_DEFAULT_BANDWIDTH_KHZ: float = 5.0
_DEFAULT_TONALITY_THRESHOLD: float = 0.0


def _make_call_id(wav_stem: str, det_index) -> str:
    """Stable, unique key: ``{wav_stem}__det{det_index}`` (det_index as int)."""
    return f"{wav_stem}__det{int(det_index)}"


def _spec_to_uint8_patch(spec_2d: np.ndarray) -> np.ndarray:
    """mat2gray (whole-patch min-max → [0,1] → uint8) + bicubic resize to 227×227.

    Transcribed from cnn_prepare_training_data._spec_to_uint8_patch; the ONLY
    deviation is the resampler (BICUBIC per the A3 spec, vs the reference's
    BILINEAR). Normalization is a SINGLE whole-patch min-max — never per-window.
    """
    if spec_2d.size == 0:
        return np.zeros((_PATCH_SIDE, _PATCH_SIDE), dtype=np.uint8)
    lo, hi = float(spec_2d.min()), float(spec_2d.max())
    if hi - lo < 1e-9:
        scaled = np.zeros_like(spec_2d, dtype=np.uint8)
    else:
        scaled = ((spec_2d - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return np.asarray(
        Image.fromarray(scaled).resize(
            (_PATCH_SIDE, _PATCH_SIDE), resample=Image.Resampling.BICUBIC
        )
    )


def _render_one(
    wav_path: Path,
    call_box: CallBox,
    sample_rate: int,
    bandwidth_kHz: float,
    tonality_threshold: float,
) -> np.ndarray:
    """Render a single call to a (227, 227, 3) uint8 RGB array.

    Steps: load WAV window → optional 300→250 resample → focus STFT (Stack 4) →
    contour mask (Stack 4) → 10·log10(P) → >45 kHz crop → mat2gray → flipud →
    bicubic 227×227 → 3-channel.
    """
    samples, sr_in = sf.read(str(wav_path), dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]
    samples = np.ascontiguousarray(samples, dtype=np.float32)

    # Q2: resample to the requested working rate.
    if sample_rate == TARGET_SAMPLE_RATE_HZ and sr_in == SOURCE_SAMPLE_RATE_HZ:
        work = resample_to_vocalmat(samples)
        sr_eff = TARGET_SAMPLE_RATE_HZ
    elif sample_rate == TARGET_SAMPLE_RATE_HZ and sr_in == TARGET_SAMPLE_RATE_HZ:
        work = samples
        sr_eff = TARGET_SAMPLE_RATE_HZ
    else:
        # --sample-rate 300000 fallback (or unexpected source rate): run the
        # focus STFT at the source rate, no resampling.
        work = samples
        sr_eff = int(sr_in)

    # Stack 4: per-call adaptive focus STFT (magnitude, freq-cropped to the band).
    focus = create_focus_spectrogram(work, sr_eff, call_box)
    I = focus.I  # (F_band, T), magnitude
    if I.size == 0 or I.shape[1] == 0:
        return np.zeros((_PATCH_SIDE, _PATCH_SIDE, 3), dtype=np.uint8)

    # Frequency axis (kHz) for exactly the rows in I (post freq-crop).
    freqs_kHz_axis = (
        focus.fr_hz[focus.freq_lo_idx : focus.freq_hi_idx + 1] / 1000.0
    )

    # Stack 4: contour ridge in physical units, then map onto this focus grid.
    contour = extract_contour_for_call(work, sr_eff, call_box)
    if contour.time_s.size > 0:
        # Time → focus-STFT column index: column k spans ti_s[k]; map each
        # contour time to the nearest focus column. ti_s is relative to the
        # cropped segment start, contour.time_s is absolute → shift by start.
        rel_t = contour.time_s - call_box.time_start_s
        t_cols = np.searchsorted(focus.ti_s, rel_t)
        t_cols = np.clip(t_cols, 0, I.shape[1] - 1).astype(np.int64)
        S_masked = apply_hard_bandwidth_mask(
            S_pow=I,
            contour_t_bins=t_cols,
            contour_freqs_kHz=contour.freq_kHz,
            contour_tonalities=contour.tonality,
            freqs_kHz_axis=freqs_kHz_axis,
            bandwidth_kHz=bandwidth_kHz,
            tonality_threshold=tonality_threshold,
        )
    else:
        # No contour survived → fully masked (all-zero), consistent with the
        # mask's behavior for columns lacking a qualifying contour bin.
        S_masked = np.zeros_like(I)

    # VocalMat render: power → dB.  I is magnitude, P = |I|^2.
    P = S_masked.astype(np.float64) ** 2
    spec_db = 10.0 * np.log10(P + _LOG10_EPS)

    # Frequency-crop to > 45 kHz (rows of the focus band above 45 kHz).
    keep = freqs_kHz_axis > _FREQ_CROP_MIN_KHZ
    if keep.any():
        spec_db = spec_db[keep, :]
    # else: band lies entirely <=45 kHz (rare) — keep as-is so we still emit a patch.

    # flipud (low freq at bottom, as VocalMat renders) THEN mat2gray + resize.
    spec_db = np.flipud(spec_db)
    gray = _spec_to_uint8_patch(spec_db)
    return np.stack([gray, gray, gray], axis=-1)


def _worker(task: dict) -> dict:
    """Process-pool worker. Returns a manifest row dict (with 'ok'/'error')."""
    out_path = Path(task["out_path"])
    row = {
        "call_id": task["call_id"],
        "path": task["rel_path"],
        "wav_stem": task["wav_stem"],
        "cohort": task["cohort"],
        "ok": False,
        "error": "",
    }
    try:
        if out_path.exists() and not task["overwrite"]:
            row["ok"] = True
            row["error"] = "skipped-exists"
            return row
        call_box = CallBox(
            time_start_s=task["det_start_s"],
            freq_start_kHz=task["freq_start_kHz"],
            duration_s=task["duration_s"],
            freq_range_kHz=task["freq_range_kHz"],
        )
        rgb = _render_one(
            wav_path=Path(task["wav_path"]),
            call_box=call_box,
            sample_rate=task["sample_rate"],
            bandwidth_kHz=task["bandwidth_kHz"],
            tonality_threshold=task["tonality_threshold"],
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgb, mode="RGB").save(out_path)
        row["ok"] = True
    except Exception as exc:  # noqa: BLE001 — record per-call failure, keep going
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def _build_tasks(args: argparse.Namespace) -> tuple[list[dict], int, int]:
    """Read the CSV and assemble one task dict per renderable call.

    Returns (tasks, n_total_rows, n_missing_wav).
    """
    df = pd.read_csv(args.csv)
    n_total = len(df)
    if args.limit and args.limit > 0:
        df = df.head(args.limit)

    pad_s = args.pad_ms / 1000.0
    out_dir = Path(args.out_dir)
    wav_root = Path(args.wav_root)

    tasks: list[dict] = []
    n_missing_wav = 0
    for r in df.itertuples(index=False):
        wav_stem = str(getattr(r, "wav_stem"))
        det_index = getattr(r, "det_index")
        call_id = _make_call_id(wav_stem, det_index)

        wav_path = wav_root / f"{wav_stem}.wav"
        if not wav_path.exists():
            n_missing_wav += 1
            continue

        det_start = float(getattr(r, "det_start_s"))
        det_end = float(getattr(r, "det_end_s"))
        # Symmetric pad; clamp start at 0 (the focus STFT itself re-clamps to the
        # WAV bounds, so the 2 s chunk edge is handled safely by Stack 4).
        start_padded = max(0.0, det_start - pad_s)
        end_padded = det_end + pad_s
        duration_s = max(end_padded - start_padded, 1e-4)

        # Frequency band for the focus STFT: use the call's measured low/high
        # frequency. Fall back to the full corpus USV band if either is missing.
        low_hz = getattr(r, "low_freq_hz", None)
        high_hz = getattr(r, "high_freq_hz", None)
        try:
            low_khz = float(low_hz) / 1000.0
            high_khz = float(high_hz) / 1000.0
        except (TypeError, ValueError):
            low_khz = corpus.USV_FREQ_MIN_HZ / 1000.0
            high_khz = corpus.USV_FREQ_MAX_HZ / 1000.0
        if not np.isfinite(low_khz) or not np.isfinite(high_khz) or high_khz <= low_khz:
            low_khz = corpus.USV_FREQ_MIN_HZ / 1000.0
            high_khz = corpus.USV_FREQ_MAX_HZ / 1000.0
        freq_range_khz = high_khz - low_khz

        cohort = str(getattr(r, "couple", ""))

        tasks.append(
            {
                "call_id": call_id,
                "wav_stem": wav_stem,
                "cohort": cohort,
                "wav_path": str(wav_path),
                "out_path": str(out_dir / f"{call_id}.png"),
                "rel_path": str((out_dir / f"{call_id}.png").as_posix()),
                "det_start_s": start_padded,
                "duration_s": duration_s,
                "freq_start_kHz": low_khz,
                "freq_range_kHz": freq_range_khz,
                "sample_rate": args.sample_rate,
                "bandwidth_kHz": args.bandwidth_kHz,
                "tonality_threshold": args.tonality_threshold,
                "overwrite": args.overwrite,
            }
        )
    return tasks, n_total, n_missing_wav


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--csv",
        type=Path,
        default=REPO_ROOT / "classified_detections_lab_131204_clean.csv",
        help="Detections CSV (default: classified_detections_lab_131204_clean.csv).",
    )
    p.add_argument(
        "--wav-root",
        type=Path,
        default=REPO_ROOT / "USV_lab_131204_chunked_2s_full",
        help="Directory of 2 s 300 kHz WAV chunks named <wav_stem>.wav.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "data" / "alpha3_patches",
        help="Output dir for <call_id>.png + manifest.csv.",
    )
    p.add_argument(
        "--sample-rate",
        type=int,
        choices=(250000, 300000),
        default=250000,
        help="Working rate. 250000 = VocalMat-aligned (default); 300000 = fallback.",
    )
    p.add_argument(
        "--pad-ms",
        type=float,
        default=_DEFAULT_PAD_MS,
        help="Symmetric pad (ms) added around [det_start_s, det_end_s] before crop.",
    )
    p.add_argument(
        "--bandwidth-kHz",
        type=float,
        default=_DEFAULT_BANDWIDTH_KHZ,
        help="Stack-4 hard-mask half-bandwidth around the contour ridge (kHz).",
    )
    p.add_argument(
        "--tonality-threshold",
        type=float,
        default=_DEFAULT_TONALITY_THRESHOLD,
        help="Stack-4 mask tonality threshold (0.0 = keep every contour bin).",
    )
    p.add_argument("--workers", type=int, default=4, help="Process-pool workers.")
    p.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Render only the first N CSV rows (0 = all). For dry-runs.",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-render PNGs even if the output file already exists.",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = out_dir / "manifest.csv"

    print("=" * 72)
    print("α₃-C Phase A3 — render VocalMat-style contour-masked patches")
    print("=" * 72)
    print(f"  csv                 : {args.csv}")
    print(f"  wav-root            : {args.wav_root}")
    print(f"  out-dir             : {out_dir}")
    print(f"  manifest            : {manifest_path}")
    print(f"  sample-rate         : {args.sample_rate} Hz "
          f"(source {SOURCE_SAMPLE_RATE_HZ} → target {TARGET_SAMPLE_RATE_HZ})")
    print(f"  pad-ms              : {args.pad_ms}")
    print(f"  bandwidth-kHz       : {args.bandwidth_kHz}")
    print(f"  tonality-threshold  : {args.tonality_threshold}")
    print(f"  freq-crop           : > {_FREQ_CROP_MIN_KHZ} kHz")
    print(f"  patch side          : {_PATCH_SIDE} (bicubic, 3-channel)")
    print(f"  normalization       : WHOLE-patch min-max (mat2gray), single pass")
    print(f"  window              : DETECTION EVENT [det_start_s, det_end_s] "
          f"(+/- pad); NOT call_length_s")
    print(f"  call_id format      : {{wav_stem}}__det{{det_index}}")
    print(f"  workers             : {args.workers}")
    print(f"  limit               : {args.limit if args.limit else 'ALL'}")
    print(f"  overwrite           : {args.overwrite}")

    tasks, n_total, n_missing_wav = _build_tasks(args)
    print(f"\n  CSV rows total      : {n_total}")
    print(f"  rows after --limit  : {n_total if not args.limit else min(args.limit, n_total)}")
    print(f"  missing WAV (skip)  : {n_missing_wav}")
    print(f"  tasks to render     : {len(tasks)}")
    if not tasks:
        print("  nothing to render — exiting.")
        return 0

    rows: list[dict] = []
    n_ok = n_skip = n_err = 0
    if args.workers <= 1:
        for t in tasks:
            row = _worker(t)
            rows.append(row)
            if row["error"] == "skipped-exists":
                n_skip += 1
            elif row["ok"]:
                n_ok += 1
            else:
                n_err += 1
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_worker, t) for t in tasks]
            done = 0
            for fut in as_completed(futs):
                row = fut.result()
                rows.append(row)
                done += 1
                if row["error"] == "skipped-exists":
                    n_skip += 1
                elif row["ok"]:
                    n_ok += 1
                else:
                    n_err += 1
                if done % 2000 == 0:
                    print(f"    ... {done}/{len(tasks)} processed "
                          f"(ok={n_ok}, skip={n_skip}, err={n_err})")

    # Manifest: only successfully-rendered (or skip-existing) calls. Sorted by
    # call_id for determinism. Columns: call_id, path, wav_stem, cohort.
    good = [r for r in rows if r["ok"]]
    good.sort(key=lambda r: r["call_id"])
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["call_id", "path", "wav_stem", "cohort"])
        for r in good:
            w.writerow([r["call_id"], r["path"], r["wav_stem"], r["cohort"]])

    print(f"\n  rendered ok         : {n_ok}")
    print(f"  skipped (existing)  : {n_skip}")
    print(f"  errors              : {n_err}")
    if n_err:
        first_errs = [r for r in rows if not r["ok"]][:5]
        print("  first errors:")
        for r in first_errs:
            print(f"    {r['call_id']}: {r['error']}")
    print(f"  manifest rows       : {len(good)}  → {manifest_path}")
    return 0 if n_err == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

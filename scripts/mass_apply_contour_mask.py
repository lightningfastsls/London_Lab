"""Phase 3 sub-task 3.2 — mass-apply the chosen contour mask.

Applies the user-selected hard-bandwidth contour mask (bandwidth = +/- 5 kHz,
tonality_threshold = 0.0) to every accepted window in
``window_index.parquet``. Saves the masked power-spectrogram patches as a
single ``.npz`` plus a typed manifest parquet linking each patch row to its
source (wav_stem, call_id, window_idx) and diagnostics.

Refinement D: patches are stored as raw power (float32). Per-patch
normalization happens at VAE training time, NOT here.

Reuses data-loading helpers from ``sweep_contour_mask`` and the masking
primitive from ``contour_mask_utils`` — no duplication of STFT, WAV index,
or window-cutting code.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make scripts/ and src/ importable regardless of cwd.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _p in (_REPO_ROOT, _SRC_ROOT, _SCRIPTS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from usv_spectrogram import corpus  # noqa: E402

from contour_mask_utils import apply_hard_bandwidth_mask  # noqa: E402
from sweep_contour_mask import (  # noqa: E402
    build_wav_index,
    cut_patch,
    get_contour_rows_for_window,
    load_full_power_spec,
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Mass-apply chosen contour mask (hard +/- 5 kHz, tonality "
            "threshold 0.0) to all accepted windows; emit patches.npz + "
            "manifest.parquet."
        )
    )
    p.add_argument(
        "--contours-parquet",
        type=Path,
        default=_REPO_ROOT / "results/contour_extraction/5970/contours.parquet",
    )
    p.add_argument(
        "--window-index-parquet",
        type=Path,
        default=_REPO_ROOT / "results/masked_patches/5970/window_index.parquet",
    )
    p.add_argument("--wav-search-dirs", type=Path, nargs="+", required=True)
    p.add_argument(
        "--output-patches-npz",
        type=Path,
        default=_REPO_ROOT / "results/masked_patches/5970/patches.npz",
    )
    p.add_argument(
        "--output-manifest-parquet",
        type=Path,
        default=_REPO_ROOT / "results/masked_patches/5970/patches_manifest.parquet",
    )
    p.add_argument("--bandwidth-kHz", type=float, default=5.0)
    p.add_argument("--tonality-threshold", type=float, default=0.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    mask_kind = "hard"

    # Load inputs.
    contours = pd.read_parquet(args.contours_parquet)
    window_index = pd.read_parquet(args.window_index_parquet)

    # Filter to windows whose (wav_stem, call_id) is accepted.
    accepted_pairs = (
        contours.loc[contours["accepted"], ["wav_stem", "call_id"]]
        .drop_duplicates()
    )
    # PERFORMANCE: set a MultiIndex on (wav_stem, call_id) so per-window
    # contour lookups in get_contour_rows_for_window() are O(log N) instead of
    # O(N). For the lab cohort (2.5M rows x 55K windows), this is the
    # difference between ~3 hours and ~30 seconds.
    contours = contours.sort_values(["wav_stem", "call_id", "time_bin_index"]) \
                        .set_index(["wav_stem", "call_id"], drop=False)
    windows = window_index.merge(
        accepted_pairs, on=["wav_stem", "call_id"], how="inner"
    )
    if len(windows) == 0:
        raise RuntimeError(
            "No accepted windows after merging window_index with the "
            "accepted-pairs from contours.parquet. Check that the upstream "
            "contour extractor populated the 'accepted' flag."
        )
    print(f"  accepted windows: {len(windows)}")

    # Deterministic order.
    windows = windows.sort_values(
        ["wav_stem", "call_id", "window_idx"], kind="mergesort"
    ).reset_index(drop=True)

    # Resolve WAV paths.
    wav_index = build_wav_index([Path(d) for d in args.wav_search_dirs])
    required_stems = sorted(windows["wav_stem"].unique().tolist())
    missing = [s for s in required_stems if s not in wav_index]
    if missing:
        raise FileNotFoundError(
            f"Could not locate WAV file(s) for {len(missing)} stem(s): "
            f"{missing}"
        )

    # Verify expected patch geometry up-front (F derived from corpus).
    expected_F = corpus.STFT_N_FFT // 2 + 1
    widths = (windows["end_bin_index"] - windows["start_bin_index"]).unique()
    if len(widths) != 1:
        raise AssertionError(
            f"window_index has non-uniform window widths: {widths}. "
            f"This pipeline assumes a single fixed T per patch."
        )
    expected_T = int(widths[0])

    # Pre-allocate output array. dtype float32 per spec.
    n_patches = len(windows)
    patches = np.zeros((n_patches, expected_F, expected_T), dtype=np.float32)

    # Manifest accumulator columns.
    n_nonzero_freqs_arr = np.zeros(n_patches, dtype=np.int32)
    patch_max_power_arr = np.zeros(n_patches, dtype=np.float32)

    # Iterate by wav_stem so we hold one STFT in memory at a time.
    freqs_axis_ref: np.ndarray | None = None
    for stem, sub in windows.groupby("wav_stem", sort=False):
        wav_path = wav_index[stem]
        S_pow_full, freqs_axis = load_full_power_spec(wav_path)
        if freqs_axis_ref is None:
            freqs_axis_ref = freqs_axis.astype(np.float64)
        elif not np.allclose(freqs_axis_ref, freqs_axis):
            raise AssertionError(
                f"Frequency axis drift across WAVs at stem {stem!r}. "
                f"Canonical STFT params should yield identical axes."
            )
        if S_pow_full.shape[0] != expected_F:
            raise AssertionError(
                f"STFT F={S_pow_full.shape[0]} != expected F={expected_F} "
                f"for {stem!r}."
            )

        for row in sub.itertuples(index=True):
            patch_idx = row.Index
            start_bin = int(row.start_bin_index)
            end_bin = int(row.end_bin_index)
            call_id = int(row.call_id)

            patch = cut_patch(S_pow_full, start_bin, end_bin)

            crows = get_contour_rows_for_window(
                contours, stem, call_id, start_bin, end_bin
            )
            t_bins_local = (
                crows["time_bin_index"].to_numpy(dtype=np.int64) - start_bin
            )
            f_ridge_kHz = crows["frequency_kHz"].to_numpy(dtype=np.float64)
            tonality = crows["tonality"].to_numpy(dtype=np.float64)

            masked = apply_hard_bandwidth_mask(
                S_pow=patch,
                contour_t_bins=t_bins_local,
                contour_freqs_kHz=f_ridge_kHz,
                contour_tonalities=tonality,
                freqs_kHz_axis=freqs_axis,
                bandwidth_kHz=args.bandwidth_kHz,
                tonality_threshold=args.tonality_threshold,
            )
            masked_f32 = masked.astype(np.float32, copy=False)
            patches[patch_idx, :, :] = masked_f32

            # Diagnostics.
            row_any = np.any(masked_f32 > 0, axis=1)  # (F,) — any-nonzero per row
            n_nonzero_freqs_arr[patch_idx] = int(row_any.sum())
            patch_max_power_arr[patch_idx] = (
                float(masked_f32.max()) if masked_f32.size else 0.0
            )

        # Drop cache.
        del S_pow_full

    assert freqs_axis_ref is not None  # populated by loop

    # Save patches + freq axis.
    args.output_patches_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_patches_npz,
        patches=patches,
        freqs_kHz=freqs_axis_ref.astype(np.float32),
    )

    # Build manifest.
    manifest = pd.DataFrame(
        {
            "patch_idx": np.arange(n_patches, dtype=np.int32),
            "wav_stem": windows["wav_stem"].astype("string").to_numpy(),
            "call_id": windows["call_id"].to_numpy(dtype=np.int64),
            "window_idx": windows["window_idx"].to_numpy(dtype=np.int32),
            "start_bin_index": windows["start_bin_index"].to_numpy(dtype=np.int32),
            "end_bin_index": windows["end_bin_index"].to_numpy(dtype=np.int32),
            "abs_time_start_s": windows["abs_time_start_s"].to_numpy(dtype=np.float32),
            "abs_time_end_s": windows["abs_time_end_s"].to_numpy(dtype=np.float32),
            "num_contour_bins_in_window": windows["num_contour_bins_in_window"].to_numpy(
                dtype=np.int32
            ),
            "mask_kind": np.full(n_patches, mask_kind, dtype=object),
            "bandwidth_kHz": np.full(n_patches, args.bandwidth_kHz, dtype=np.float32),
            "tonality_threshold": np.full(
                n_patches, args.tonality_threshold, dtype=np.float32
            ),
            "n_nonzero_freqs": n_nonzero_freqs_arr,
            "patch_max_power": patch_max_power_arr,
        }
    )
    # Enforce typed string columns and exact column order.
    manifest = manifest.astype(
        {
            "patch_idx": "int32",
            "wav_stem": "string",
            "call_id": "int64",
            "window_idx": "int32",
            "start_bin_index": "int32",
            "end_bin_index": "int32",
            "abs_time_start_s": "float32",
            "abs_time_end_s": "float32",
            "num_contour_bins_in_window": "int32",
            "mask_kind": "string",
            "bandwidth_kHz": "float32",
            "tonality_threshold": "float32",
            "n_nonzero_freqs": "int32",
            "patch_max_power": "float32",
        }
    )
    manifest = manifest[
        [
            "patch_idx",
            "wav_stem",
            "call_id",
            "window_idx",
            "start_bin_index",
            "end_bin_index",
            "abs_time_start_s",
            "abs_time_end_s",
            "num_contour_bins_in_window",
            "mask_kind",
            "bandwidth_kHz",
            "tonality_threshold",
            "n_nonzero_freqs",
            "patch_max_power",
        ]
    ]
    args.output_manifest_parquet.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_parquet(args.output_manifest_parquet, index=False)

    # Summary.
    n_all_zero = int((manifest["n_nonzero_freqs"] == 0).sum())
    avg_cov_pct = float(
        100.0 * manifest["n_nonzero_freqs"].to_numpy().mean() / expected_F
    )
    print(f"windows_processed: {n_patches}")
    print(f"wav_stems_used: {len(required_stems)}")
    print(f"patches shape: ({n_patches}, {expected_F}, {expected_T})")
    print(f"all-zero patches: {n_all_zero}")
    print(f"manifest: {args.output_manifest_parquet}")
    print(f"patches:  {args.output_patches_npz}")
    print(f"average mask coverage (n_nonzero_freqs / F): {avg_cov_pct:.1f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

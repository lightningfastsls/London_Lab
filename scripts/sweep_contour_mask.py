"""Phase 3 sub-task 3.1 — contour-mask visual sweep.

Produces a 12-cell (4 bandwidth variants x 3 tonality thresholds) figure
set + 1 reference PNG of the raw unmasked patches for the same 20 calls.
The user reviews these visually and picks (bandwidth, threshold) before
Phase 3 sub-task 3.2 (mass-apply chosen mask to all 291 windows).

Cells (12):
    bandwidth in {bw_2kHz, bw_5kHz, bw_10kHz, bw_gauss3kHz}
    threshold in {thr_p25 (94.4), thr_p50 (314.1), thr_p75 (1254.7)}

Threshold values are percentile anchors on our peak-to-median tonality
distribution (rather than DeepSqueak's internal 0.3 — that lived on a
different scale). p25 / p50 / p75 keep the top 75 / 50 / 25 percent of
contour bins respectively.

Per-patch normalization is the project default for VAE training
(refinement D), but for THIS visual sweep we render in absolute dB power
with a shared vmin/vmax across all panels so cells are honestly
comparable. db_floor = -60 dB.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

from contour_mask_utils import (  # noqa: E402
    apply_hard_bandwidth_mask,
    apply_soft_gaussian_mask,
)


# ---------------------------------------------------------------------------
# Sweep matrix definition
# ---------------------------------------------------------------------------

# Bandwidth variants — each is a callable (S_pow, t_bins, f_kHz, ton, axis, thr) -> masked_S.
# We use lambdas with captured bandwidth/sigma so the dispatch loop is clean.
def _mk_hard(bw_kHz: float):
    def _apply(S, t_bins, f_kHz, ton, axis, thr):
        return apply_hard_bandwidth_mask(
            S_pow=S,
            contour_t_bins=t_bins,
            contour_freqs_kHz=f_kHz,
            contour_tonalities=ton,
            freqs_kHz_axis=axis,
            bandwidth_kHz=bw_kHz,
            tonality_threshold=thr,
        )
    return _apply


def _mk_gauss(sigma_kHz: float):
    def _apply(S, t_bins, f_kHz, ton, axis, thr):
        return apply_soft_gaussian_mask(
            S_pow=S,
            contour_t_bins=t_bins,
            contour_freqs_kHz=f_kHz,
            contour_tonalities=ton,
            freqs_kHz_axis=axis,
            sigma_kHz=sigma_kHz,
            tonality_threshold=thr,
        )
    return _apply


BANDWIDTH_VARIANTS: list[tuple[str, str, callable]] = [
    ("bw_2kHz",     "+/- 2 kHz",          _mk_hard(2.0)),
    ("bw_5kHz",     "+/- 5 kHz",          _mk_hard(5.0)),
    ("bw_10kHz",    "+/- 10 kHz",         _mk_hard(10.0)),
    ("bw_gauss3kHz","Gaussian sigma=3 kHz", _mk_gauss(3.0)),
]

THRESHOLD_VARIANTS: list[tuple[str, str, float]] = [
    ("thr_p25", "p25", 94.4),
    ("thr_p50", "p50", 314.1),
    ("thr_p75", "p75", 1254.7),
]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Visual sweep of contour-mask bandwidth x tonality threshold."
    )
    p.add_argument("--contours-parquet", type=Path, required=True)
    p.add_argument("--window-index-parquet", type=Path, required=True)
    p.add_argument("--wav-search-dirs", type=Path, nargs="+", required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--n-example-calls", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--db-floor",
        type=float,
        default=-60.0,
        help="vmin (dB power) for the magma colormap. Cells share this floor.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------

def build_wav_index(wav_search_dirs: list[Path]) -> dict[str, Path]:
    """Return {wav_stem: Path} honoring search-dir order (first wins)."""
    index: dict[str, Path] = {}
    for root in wav_search_dirs:
        if not root.exists():
            continue
        for wav in root.rglob("*.wav"):
            stem = wav.stem
            if stem not in index:
                index[stem] = wav
    return index


def pick_example_calls(
    window_index: pd.DataFrame,
    contours: pd.DataFrame,
    n: int,
    seed: int,
) -> pd.DataFrame:
    """Return ``n`` (wav_stem, call_id, first-window-row) records.

    Selection pool: windows that have ``window_idx == 0`` AND whose call
    is ``accepted == True`` in the contour parquet. We then sample ``n``
    rows deterministically with the given seed.
    """
    accepted_pairs = (
        contours.loc[contours["accepted"], ["wav_stem", "call_id"]]
        .drop_duplicates()
    )
    first_windows = window_index[window_index["window_idx"] == 0].merge(
        accepted_pairs, on=["wav_stem", "call_id"], how="inner"
    )
    if len(first_windows) < n:
        raise ValueError(
            f"Not enough accepted first-windows: have {len(first_windows)}, "
            f"need {n}"
        )
    sampled = first_windows.sample(n=n, random_state=seed).reset_index(drop=True)
    return sampled


def load_full_power_spec(wav_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load WAV and compute |STFT|^2 with canonical params.

    Returns (S_pow [F, T], freqs_kHz_axis [F]).
    """
    y, _ = librosa.load(str(wav_path), sr=corpus.SAMPLE_RATE_HZ, mono=True)
    S = np.abs(
        librosa.stft(
            y,
            n_fft=corpus.STFT_N_FFT,
            hop_length=corpus.STFT_HOP,
            window="hann",
            center=True,
        )
    )
    S_pow = S.astype(np.float64) ** 2
    freqs_kHz = (
        librosa.fft_frequencies(sr=corpus.SAMPLE_RATE_HZ, n_fft=corpus.STFT_N_FFT)
        / 1000.0
    )
    return S_pow, freqs_kHz


def cut_patch(
    S_pow: np.ndarray,
    start_bin: int,
    end_bin: int,
) -> np.ndarray:
    """Cut the [start_bin:end_bin] columns from ``S_pow``.

    Zero-pads on the right if ``end_bin`` extends past the recording's
    total bin count (defensive — window_index.parquet was built to keep
    windows inside the recording, but we don't trust that downstream).
    """
    F, T_total = S_pow.shape
    width = end_bin - start_bin
    left = max(0, start_bin)
    right = min(T_total, end_bin)
    pad_left = left - start_bin       # >= 0
    pad_right = end_bin - right       # >= 0

    out = np.zeros((F, width), dtype=S_pow.dtype)
    if right > left:
        out[:, pad_left:width - pad_right] = S_pow[:, left:right]
    return out


def get_contour_rows_for_window(
    contours: pd.DataFrame,
    wav_stem: str,
    call_id: int,
    start_bin: int,
    end_bin: int,
) -> pd.DataFrame:
    """Return contour rows for this (wav_stem, call_id) that fall in [start_bin, end_bin).

    Performance: when ``contours`` has a MultiIndex on ('wav_stem', 'call_id'),
    the per-call lookup is O(log N) instead of the O(N) linear scan that the
    boolean-mask version did. For the lab cohort (2.5M rows × 55K windows)
    this is the difference between ~3 hours and ~30 seconds for the masking
    step. Callers should set the index once at load time.
    """
    if isinstance(contours.index, pd.MultiIndex) and contours.index.names[:2] == ["wav_stem", "call_id"]:
        # Fast path: indexed lookup. KeyError if the (wav_stem, call_id) is
        # absent — but that should not happen because the windowing step
        # already restricted to accepted contours.
        try:
            sub = contours.loc[(wav_stem, call_id)]
        except KeyError:
            return contours.iloc[0:0]
        if isinstance(sub, pd.Series):
            sub = sub.to_frame().T
        tb = sub["time_bin_index"].to_numpy()
        keep = (tb >= start_bin) & (tb < end_bin)
        return sub.iloc[keep]
    # Fallback: linear-scan path (used in unit tests / older callers).
    mask = (
        (contours["wav_stem"] == wav_stem)
        & (contours["call_id"] == call_id)
        & (contours["time_bin_index"] >= start_bin)
        & (contours["time_bin_index"] < end_bin)
    )
    return contours.loc[mask]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def power_to_db(patch: np.ndarray, db_floor: float) -> np.ndarray:
    """Convert linear power to dB, clipping at ``db_floor``.

    Uses librosa.power_to_db with ref=1.0 so all panels share a common
    reference. Output is then clipped so masked-out zeros become
    ``db_floor`` (instead of -inf), preserving honest visual comparison.
    """
    # Tiny eps so log of zero -> very negative, then clipped.
    db = librosa.power_to_db(patch + 1e-20, ref=1.0)
    return np.clip(db, db_floor, None)


def render_cell(
    fig_path: Path,
    suptitle: str,
    panels: list[dict],
    db_floor: float,
    rows: int = 5,
    cols: int = 4,
    dpi: int = 110,
) -> None:
    """Render one PNG (5x4 grid of panels) with a shared vmin/vmax.

    Each panel dict has: ``patch_db`` (F, T), ``title`` (str), ``freqs_kHz`` (axis).
    All panels share the same vmin (= db_floor) and vmax (= max across cells).
    """
    vmin = db_floor
    vmax = max(p["patch_db"].max() for p in panels)

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.5, rows * 1.7), dpi=dpi)
    axes = np.atleast_2d(axes)
    for k in range(rows * cols):
        ax = axes[k // cols, k % cols]
        if k >= len(panels):
            ax.set_axis_off()
            continue
        p = panels[k]
        patch_db = p["patch_db"]
        freqs_kHz = p["freqs_kHz"]
        # Restrict to USV band rows.
        f_lo = corpus.USV_FREQ_MIN_HZ / 1000.0
        f_hi = corpus.USV_FREQ_MAX_HZ / 1000.0
        band_mask = (freqs_kHz >= f_lo) & (freqs_kHz <= f_hi)
        ax.imshow(
            patch_db[band_mask, :],
            origin="lower",
            aspect="auto",
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            extent=[0, patch_db.shape[1], f_lo, f_hi],
        )
        ax.set_title(p["title"], fontsize=6)
        ax.tick_params(labelsize=5)
        ax.set_ylabel("kHz", fontsize=5)
        ax.set_xlabel("t bins", fontsize=5)
    fig.suptitle(suptitle, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_path, dpi=dpi)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    contours = pd.read_parquet(args.contours_parquet)
    window_index = pd.read_parquet(args.window_index_parquet)

    sampled = pick_example_calls(
        window_index=window_index,
        contours=contours,
        n=args.n_example_calls,
        seed=args.seed,
    )

    # Build wav index once, load each unique WAV once.
    wav_index = build_wav_index([Path(d) for d in args.wav_search_dirs])
    unique_wavs = sorted(set(sampled["wav_stem"].tolist()))
    full_spec_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    missing_wavs: list[str] = []
    for stem in unique_wavs:
        wav_path = wav_index.get(stem)
        if wav_path is None:
            missing_wavs.append(stem)
            continue
        full_spec_cache[stem] = load_full_power_spec(wav_path)
    if missing_wavs:
        raise FileNotFoundError(
            f"Could not locate WAVs for stems: {missing_wavs[:5]}{'...' if len(missing_wavs) > 5 else ''}"
        )

    # Pre-compute the raw 100 ms patches + contour-row lookups once per call.
    raw_patches: list[dict] = []
    truncated_warnings: list[str] = []
    for _, row in sampled.iterrows():
        wav_stem = str(row["wav_stem"])
        call_id = int(row["call_id"])
        start_bin = int(row["start_bin_index"])
        end_bin = int(row["end_bin_index"])
        num_contour = int(row["num_contour_bins_in_window"])

        S_pow_full, freqs_axis = full_spec_cache[wav_stem]
        if end_bin > S_pow_full.shape[1]:
            truncated_warnings.append(
                f"{wav_stem} call {call_id}: end_bin {end_bin} > "
                f"recording_total_bins {S_pow_full.shape[1]} — right-padding with zeros"
            )
        patch = cut_patch(S_pow_full, start_bin, end_bin)

        # Window-local contour rows (re-index t_bins to [0, width)).
        crows = get_contour_rows_for_window(
            contours, wav_stem, call_id, start_bin, end_bin
        )
        t_bins_local = (crows["time_bin_index"].to_numpy(dtype=np.int64)
                        - start_bin)
        f_ridge_kHz = crows["frequency_kHz"].to_numpy(dtype=np.float64)
        tonality = crows["tonality"].to_numpy(dtype=np.float64)

        # Short label for titles: last 8 chars of wav_stem.
        short_stem = wav_stem[-8:]

        raw_patches.append(
            dict(
                wav_stem=wav_stem,
                short_stem=short_stem,
                call_id=call_id,
                num_contour=num_contour,
                patch=patch,
                freqs_axis=freqs_axis,
                t_bins_local=t_bins_local,
                f_ridge_kHz=f_ridge_kHz,
                tonality=tonality,
            )
        )

    # -------------------------------------------------------------------
    # Reference PNG: raw unmasked patches.
    # -------------------------------------------------------------------
    unmasked_panels = []
    for rp in raw_patches:
        title = (
            f"{rp['short_stem']} c{rp['call_id']} | "
            f"n_c={rp['num_contour']}"
        )
        unmasked_panels.append(
            dict(
                patch_db=power_to_db(rp["patch"], args.db_floor),
                title=title,
                freqs_kHz=rp["freqs_axis"],
            )
        )
    raw_path = args.output_dir / "cell_raw_unmasked.png"
    render_cell(
        fig_path=raw_path,
        suptitle="Raw 100 ms patches (no mask) — reference baseline",
        panels=unmasked_panels,
        db_floor=args.db_floor,
    )

    # -------------------------------------------------------------------
    # 12-cell sweep.
    # -------------------------------------------------------------------
    written_paths: list[Path] = [raw_path]
    zeroed_counts: dict[str, int] = {}

    for bw_label, bw_human, mask_fn in BANDWIDTH_VARIANTS:
        for thr_label, thr_human, thr_value in THRESHOLD_VARIANTS:
            cell_panels = []
            zeroed = 0
            for rp in raw_patches:
                masked = mask_fn(
                    rp["patch"],
                    rp["t_bins_local"],
                    rp["f_ridge_kHz"],
                    rp["tonality"],
                    rp["freqs_axis"],
                    thr_value,
                )
                if not np.any(masked > 0):
                    zeroed += 1
                title = (
                    f"{rp['short_stem']} c{rp['call_id']} | "
                    f"n_c={rp['num_contour']}"
                )
                cell_panels.append(
                    dict(
                        patch_db=power_to_db(masked, args.db_floor),
                        title=title,
                        freqs_kHz=rp["freqs_axis"],
                    )
                )

            cell_name = f"cell_{bw_label}_{thr_label}.png"
            cell_path = args.output_dir / cell_name
            suptitle = (
                f"Bandwidth: {bw_human}, "
                f"Tonality threshold: tonality >= {thr_value:g} ({thr_human})"
            )
            render_cell(
                fig_path=cell_path,
                suptitle=suptitle,
                panels=cell_panels,
                db_floor=args.db_floor,
            )
            written_paths.append(cell_path)
            zeroed_counts[f"{bw_label}/{thr_label}"] = zeroed

    # -------------------------------------------------------------------
    # Summary print.
    # -------------------------------------------------------------------
    print(f"n_example_calls: {len(raw_patches)} (seed={args.seed})")
    print(f"PNGs written: {len(written_paths)}")
    for p in written_paths:
        print(f"  {p}")
    print("Zeroed-out patches per cell (no contour bin passed threshold):")
    for k, v in zeroed_counts.items():
        print(f"  {k}: {v}/{len(raw_patches)}")
    if truncated_warnings:
        print(f"\nRight-padded windows ({len(truncated_warnings)}):")
        for w in truncated_warnings[:5]:
            print(f"  {w}")
        if len(truncated_warnings) > 5:
            print(f"  ... ({len(truncated_warnings) - 5} more)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

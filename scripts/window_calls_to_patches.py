"""Window per-call DeepSqueak contour rows into fixed-size training patches.

Phase 2 of the contour-masked VAE pipeline. Consumes the canonical contour
parquet produced by ``scripts/load_deepsqueak_contours.py`` (one row per
STFT time-bin, columns ``wav_stem, call_id, time_bin_index, time_s,
frequency_kHz, tonality, accepted``) and emits a *window index* parquet —
one row per VAE training window (patch) — with provenance fields linking
each window back to the recording, the source call, and the bin range it
covers.

Refinement B (encoded in tests/test_window_calls_to_patches.py): the step
between consecutive sliding windows is **50 ms (117 bins)**, NOT the 10 ms
(23 bins) that an early draft of the plan proposed. Highly-overlapping
windows (10 ms step) produce strongly-correlated training examples which
hurt VAE generalisation.

Windowing rules (refinement B):

  (a) Short calls — call_duration_bins < WINDOW_BINS
        Exactly one window, centred on the midpoint of the call. The
        window's start is then clipped to ``[0, recording_total_bins -
        WINDOW_BINS]`` so the window always lies fully inside the
        recording (modulo recordings shorter than WINDOW_BINS, which we
        handle gracefully).

  (b) Exactly-100 ms calls — call_duration_bins == WINDOW_BINS
        Exactly one window, aligned to the call's first bin. ``window_idx
        = 0``.

  (c) Long calls — call_duration_bins > WINDOW_BINS
        A regular sliding sequence with step ``STEP_BINS`` from
        ``call_start_bin``, generating window ``k`` while ``call_start_bin
        + k * STEP_BINS + WINDOW_BINS <= call_end_bin``. If the last
        regular window's end is strictly less than ``call_end_bin``, one
        additional **tail-aligned** window is appended at
        ``start = call_end_bin - WINDOW_BINS`` so the entire call is
        covered.

All canonical constants (sample rate, STFT hop) come from
``usv_spectrogram.corpus``. WINDOW_BINS and STEP_BINS are derived from
those constants — never hardcoded — so any future change to the STFT grid
propagates automatically.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Add repo root and src/ so ``from usv_spectrogram import corpus`` resolves
# regardless of how the script is invoked (pytest, CLI, IDE).
_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC_ROOT):
    p_str = str(_p)
    if p_str not in sys.path:
        sys.path.insert(0, p_str)

from usv_spectrogram import corpus  # noqa: E402


# ---------------------------------------------------------------------------
# Canonical constants (derived from corpus — never hardcode 234 / 117)
# ---------------------------------------------------------------------------

# 100 ms of STFT hop-bins at the canonical sample rate.
WINDOW_BINS: int = round(0.100 * corpus.SAMPLE_RATE_HZ / corpus.STFT_HOP)

# 50 ms of STFT hop-bins — the sliding step for long calls (refinement B).
STEP_BINS: int = round(0.050 * corpus.SAMPLE_RATE_HZ / corpus.STFT_HOP)


# Provenance columns of the window index (output schema).
OUTPUT_COLUMNS = [
    "wav_stem",
    "call_id",
    "window_idx",
    "abs_time_start_s",
    "abs_time_end_s",
    "start_bin_index",
    "end_bin_index",
    "num_contour_bins_in_window",
]


def _empty_window_frame() -> pd.DataFrame:
    """Return an empty window-index DataFrame with the correct dtypes.

    Used both for empty inputs and as a sentinel when no accepted calls
    are present.
    """
    return pd.DataFrame(
        {
            "wav_stem": pd.array([], dtype="string"),
            "call_id": pd.array([], dtype="int64"),
            "window_idx": np.array([], dtype=np.int32),
            "abs_time_start_s": np.array([], dtype=np.float32),
            "abs_time_end_s": np.array([], dtype=np.float32),
            "start_bin_index": np.array([], dtype=np.int32),
            "end_bin_index": np.array([], dtype=np.int32),
            "num_contour_bins_in_window": np.array([], dtype=np.int32),
        }
    )


def _clip_window(
    start_uncliped: int,
    recording_total_bins: int,
) -> tuple[int, int]:
    """Clip a candidate window's [start, end) to lie inside the recording.

    Implements the boundary rules:
      - If ``start_uncliped < 0`` → start clamped to 0, end = WINDOW_BINS.
      - If end would exceed ``recording_total_bins`` → end clamped to
        ``recording_total_bins``, start = ``recording_total_bins -
        WINDOW_BINS``.
      - Otherwise use ``[start_uncliped, start_uncliped + WINDOW_BINS)``.

    Both clips are expressed by the single composed formula
    ``start = max(0, min(start_uncliped, recording_total_bins -
    WINDOW_BINS))``. When ``recording_total_bins < WINDOW_BINS`` (a
    degenerate but legal case, e.g. the fallback test with a 130-bin
    "recording") the start clamps to 0 and end defaults to WINDOW_BINS.
    """
    max_start = recording_total_bins - WINDOW_BINS
    start_bin = max(0, min(start_uncliped, max_start))
    end_bin = start_bin + WINDOW_BINS
    return start_bin, end_bin


def _candidate_starts_for_call(
    call_start_bin: int,
    call_end_bin: int,
) -> list[int]:
    """Generate the *uncliped* window start indices for a single call.

    Returns a list of integer start bin indices following rules (a), (b),
    (c). Clipping against the recording bounds is applied separately by
    ``_clip_window``.

    For long calls, regular windows step by ``STEP_BINS`` starting from
    ``call_start_bin``; a final tail-aligned window is appended when the
    last regular window's end is strictly less than ``call_end_bin``.
    """
    call_duration_bins = call_end_bin - call_start_bin

    # Rule (a): short call — single centred window.
    if call_duration_bins < WINDOW_BINS:
        center_bin = (call_start_bin + call_end_bin) // 2
        return [center_bin - (WINDOW_BINS // 2)]

    # Rule (b): exactly-100 ms call — single window aligned to start.
    if call_duration_bins == WINDOW_BINS:
        return [call_start_bin]

    # Rule (c): long call — regular sliding sequence + optional tail.
    starts: list[int] = []
    k = 0
    while call_start_bin + k * STEP_BINS + WINDOW_BINS <= call_end_bin:
        starts.append(call_start_bin + k * STEP_BINS)
        k += 1

    # Tail-aligned window if the call's tail isn't yet covered.
    last_end = starts[-1] + WINDOW_BINS if starts else call_start_bin
    if last_end < call_end_bin:
        starts.append(call_end_bin - WINDOW_BINS)

    return starts


def compute_windows_for_call(
    contour_df: pd.DataFrame,
    wav_stem: str,
    call_id: int,
    recording_total_bins: int,
) -> pd.DataFrame:
    """Compute the window index for a single call.

    Parameters
    ----------
    contour_df
        Rows of the canonical contour parquet belonging to this single
        call. Must contain a ``time_bin_index`` column (int-castable).
        ``wav_stem`` / ``call_id`` filtering is the caller's
        responsibility — this function does NOT re-filter.
    wav_stem
        The recording stem to stamp on the output rows.
    call_id
        The DeepSqueak call id to stamp on the output rows.
    recording_total_bins
        Total number of STFT time-bins in the recording. Used for boundary
        clipping so windows never extend past the recording end (or before
        bin 0).

    Returns
    -------
    pd.DataFrame
        One row per window. Columns match ``OUTPUT_COLUMNS`` with the
        canonical dtypes documented in the test suite.
    """
    if len(contour_df) == 0:
        return _empty_window_frame()

    # Call extent on the STFT bin grid. call_end_bin is exclusive: it is
    # the index of the bin one past the last contour bin.
    bin_indices = contour_df["time_bin_index"].to_numpy(dtype=np.int64)
    call_start_bin = int(bin_indices.min())
    call_end_bin = int(bin_indices.max()) + 1

    candidate_starts = _candidate_starts_for_call(call_start_bin, call_end_bin)

    hop_s = corpus.STFT_HOP / corpus.SAMPLE_RATE_HZ

    rows: list[dict] = []
    for window_idx, start_uncliped in enumerate(candidate_starts):
        start_bin, end_bin = _clip_window(start_uncliped, recording_total_bins)

        # Count contour rows that fall inside [start_bin, end_bin).
        in_window = (bin_indices >= start_bin) & (bin_indices < end_bin)
        num_in_window = int(in_window.sum())

        rows.append(
            {
                "wav_stem": wav_stem,
                "call_id": int(call_id),
                "window_idx": int(window_idx),
                "abs_time_start_s": float(start_bin) * hop_s,
                "abs_time_end_s": float(end_bin) * hop_s,
                "start_bin_index": int(start_bin),
                "end_bin_index": int(end_bin),
                "num_contour_bins_in_window": num_in_window,
            }
        )

    result = pd.DataFrame(rows)

    # Enforce the canonical dtypes. Doing it column-by-column avoids any
    # surprises with pandas inferring object/Int64 from the row-dict path.
    result["wav_stem"] = result["wav_stem"].astype("string")
    result["call_id"] = result["call_id"].astype(np.int64)
    result["window_idx"] = result["window_idx"].astype(np.int32)
    result["abs_time_start_s"] = result["abs_time_start_s"].astype(np.float32)
    result["abs_time_end_s"] = result["abs_time_end_s"].astype(np.float32)
    result["start_bin_index"] = result["start_bin_index"].astype(np.int32)
    result["end_bin_index"] = result["end_bin_index"].astype(np.int32)
    result["num_contour_bins_in_window"] = result["num_contour_bins_in_window"].astype(np.int32)

    return result[OUTPUT_COLUMNS]


def generate_window_index(
    contour_df: pd.DataFrame,
    recording_bins_map: Optional[dict[str, int]] = None,
) -> pd.DataFrame:
    """Dispatch ``compute_windows_for_call`` over every accepted call.

    Parameters
    ----------
    contour_df
        The full canonical contour parquet (multiple wavs / calls). Rows
        with ``accepted == False`` are dropped before grouping.
    recording_bins_map
        Optional ``wav_stem -> total_bins`` lookup. When a wav_stem is
        absent from the map (or the map is ``None``), ``recording_total_bins``
        falls back to ``max(time_bin_index) + 1`` over that wav's contour
        rows.

    Returns
    -------
    pd.DataFrame
        The concatenated window index across all accepted calls. Empty
        input produces an empty DataFrame with the canonical dtypes (no
        exception).
    """
    if recording_bins_map is None:
        recording_bins_map = {}

    if len(contour_df) == 0:
        return _empty_window_frame()

    # Spec rule (d): drop rejected calls before any windowing.
    if "accepted" in contour_df.columns:
        accepted_df = contour_df[contour_df["accepted"].astype(bool)].copy()
    else:
        accepted_df = contour_df.copy()

    if len(accepted_df) == 0:
        return _empty_window_frame()

    # Precompute per-wav fallback total_bins (max time_bin_index + 1).
    fallback_bins_per_wav = (
        accepted_df.groupby("wav_stem")["time_bin_index"].max() + 1
    ).astype(np.int64).to_dict()

    frames: list[pd.DataFrame] = []
    for (wav_stem, call_id), call_df in accepted_df.groupby(
        ["wav_stem", "call_id"], sort=False
    ):
        wav_stem_str = str(wav_stem)
        if wav_stem_str in recording_bins_map:
            recording_total_bins = int(recording_bins_map[wav_stem_str])
        else:
            recording_total_bins = int(fallback_bins_per_wav[wav_stem])

        frames.append(
            compute_windows_for_call(
                contour_df=call_df,
                wav_stem=wav_stem_str,
                call_id=int(call_id),
                recording_total_bins=recording_total_bins,
            )
        )

    if not frames:
        return _empty_window_frame()

    out = pd.concat(frames, ignore_index=True)
    # Re-assert dtypes after concat (pd.concat can up-cast).
    out["wav_stem"] = out["wav_stem"].astype("string")
    out["call_id"] = out["call_id"].astype(np.int64)
    out["window_idx"] = out["window_idx"].astype(np.int32)
    out["abs_time_start_s"] = out["abs_time_start_s"].astype(np.float32)
    out["abs_time_end_s"] = out["abs_time_end_s"].astype(np.float32)
    out["start_bin_index"] = out["start_bin_index"].astype(np.int32)
    out["end_bin_index"] = out["end_bin_index"].astype(np.int32)
    out["num_contour_bins_in_window"] = out["num_contour_bins_in_window"].astype(np.int32)
    return out[OUTPUT_COLUMNS]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    """Construct the argparse parser. Kept separate from main() so tests
    (and any future callers) can introspect the CLI surface."""
    p = argparse.ArgumentParser(
        description=(
            "Window per-call DeepSqueak contour rows into fixed-size VAE "
            "training patches (refinement B: 100 ms windows, 50 ms step)."
        ),
    )
    p.add_argument(
        "--contours-parquet",
        type=Path,
        required=True,
        help="Path to the canonical contour parquet produced by "
             "scripts/load_deepsqueak_contours.py.",
    )
    p.add_argument(
        "--wav-search-dirs",
        nargs="+",
        type=Path,
        required=False,
        default=None,
        help="One or more directories to search for the WAV files referenced "
             "in the contour parquet. Used to compute recording_total_bins "
             "(librosa.get_duration * sample_rate / hop). If omitted, the "
             "fallback path is taken: recording_total_bins = "
             "max(time_bin_index) + 1 over each wav's contour rows.",
    )
    p.add_argument(
        "--output-parquet",
        type=Path,
        required=True,
        help="Where to write the resulting window-index parquet.",
    )
    p.add_argument(
        "--dataset",
        type=str,
        required=False,
        default=None,
        help="Optional dataset tag (e.g. '5970', '3452'). Documentation only "
             "— not currently stamped onto the output rows.",
    )
    p.add_argument(
        "--limit-calls",
        type=int,
        required=False,
        default=None,
        help="If set, process only the first N accepted calls (for previews).",
    )
    return p


def parse_args() -> argparse.Namespace:
    return build_parser().parse_args()


def _build_recording_bins_map(
    wav_stems: list[str],
    wav_search_dirs: list[Path],
) -> dict[str, int]:
    """Locate each wav under ``wav_search_dirs`` (first match wins) and
    compute its total STFT bin count.

    A wav that cannot be found is silently omitted — ``generate_window_index``
    will fall back to ``max(time_bin_index) + 1`` for that wav.
    """
    # Imported lazily so that the unit test (which never executes main())
    # does not require librosa.
    import librosa  # noqa: WPS433

    bins_map: dict[str, int] = {}
    hop_s = corpus.STFT_HOP / corpus.SAMPLE_RATE_HZ
    for stem in wav_stems:
        wav_path: Optional[Path] = None
        for d in wav_search_dirs:
            candidate = d / f"{stem}.wav"
            if candidate.exists():
                wav_path = candidate
                break
        if wav_path is None:
            continue
        duration_s = librosa.get_duration(path=str(wav_path))
        total_bins = int(round(duration_s / hop_s))
        bins_map[stem] = total_bins
    return bins_map


def main() -> int:
    args = parse_args()

    contour_df = pd.read_parquet(args.contours_parquet)

    if args.limit_calls is not None:
        accepted_mask = contour_df["accepted"].astype(bool)
        accepted_pairs = (
            contour_df.loc[accepted_mask, ["wav_stem", "call_id"]]
            .drop_duplicates()
            .head(args.limit_calls)
        )
        keep = contour_df.merge(
            accepted_pairs,
            on=["wav_stem", "call_id"],
            how="inner",
        )
        contour_df = keep

    recording_bins_map: Optional[dict[str, int]] = None
    if args.wav_search_dirs:
        wav_stems = sorted(contour_df["wav_stem"].astype(str).unique().tolist())
        recording_bins_map = _build_recording_bins_map(
            wav_stems=wav_stems,
            wav_search_dirs=list(args.wav_search_dirs),
        )

    window_index = generate_window_index(
        contour_df=contour_df,
        recording_bins_map=recording_bins_map,
    )

    args.output_parquet.parent.mkdir(parents=True, exist_ok=True)
    window_index.to_parquet(args.output_parquet, engine="pyarrow", index=False)
    print(
        f"Wrote {len(window_index):,} windows to {args.output_parquet}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

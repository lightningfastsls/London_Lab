"""Peri-event time histogram (PETH) analysis for USV-behavior correlation.

Computes event-triggered USV rate histograms: for each behavioral event type,
bins USV count in a configurable window around event onset and normalizes to
rate (Hz). Statistical significance is assessed via circular-shift permutation
test (preserves USV temporal autocorrelation while destroying cross-correlation
with events). Confidence intervals use bootstrap resampling over events.

This is the first cross-modal analysis bridging LMT behavioral annotations
with USV detection output. If USVs correlate with social behavior events,
PETHs will show rate modulation around event onset.

Depends on:
    - usv_spectrogram.lmt.db_loader (BehavioralEvent)
    - numpy (computation)
    - matplotlib (plotting, lazy import)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_VALID_BASELINE_METHODS = {"whole_recording", "pre_event"}


@dataclass(frozen=True)
class PETHConfig:
    """Configuration for peri-event time histogram computation.

    Parameters
    ----------
    window_before_s : float
        Time before event onset to include (seconds). Default 2.0.
    window_after_s : float
        Time after event onset to include (seconds). Default 2.0.
    bin_size_s : float
        Histogram bin width (seconds). Default 0.1 (100 ms).
    n_permutations : int
        Number of circular-shift permutations for significance test.
    min_events : int
        Minimum number of events required to compute PETH.
    baseline_method : str
        How to compute baseline rate. "whole_recording" uses total USV count
        divided by recording duration. "pre_event" uses mean rate in pre-onset
        bins.
    """

    window_before_s: float = 2.0
    window_after_s: float = 2.0
    bin_size_s: float = 0.1
    n_permutations: int = 1000
    min_events: int = 5
    baseline_method: str = "whole_recording"

    def __post_init__(self) -> None:
        if self.window_before_s <= 0:
            raise ValueError("window_before_s must be positive")
        if self.window_after_s <= 0:
            raise ValueError("window_after_s must be positive")
        if self.bin_size_s <= 0:
            raise ValueError("bin_size_s must be positive")
        if self.n_permutations < 1:
            raise ValueError("n_permutations must be >= 1")
        if self.min_events < 1:
            raise ValueError("min_events must be >= 1")
        if self.baseline_method not in _VALID_BASELINE_METHODS:
            raise ValueError(
                f"baseline_method must be one of {_VALID_BASELINE_METHODS}, "
                f"got '{self.baseline_method}'"
            )

    @property
    def n_bins(self) -> int:
        """Number of histogram bins in the window."""
        return int(round((self.window_before_s + self.window_after_s) / self.bin_size_s))


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PETHResult:
    """Result of a peri-event time histogram computation for one event type.

    All array-like fields are stored as tuples for frozen-dataclass
    immutability and JSON serializability.

    Parameters
    ----------
    event_type : str
        Behavioral event name (e.g., "Oral-oral Contact").
    time_bins_s : tuple[float, ...]
        Bin centers relative to event onset (seconds). Negative = before.
    rate_hz : tuple[float, ...]
        USV rate in each bin (events per second).
    rate_ci_lower_hz : tuple[float, ...]
        Lower bound of 95% bootstrap confidence interval.
    rate_ci_upper_hz : tuple[float, ...]
        Upper bound of 95% bootstrap confidence interval.
    baseline_rate_hz : float
        Baseline USV rate for comparison.
    n_events : int
        Number of behavioral events used.
    p_value : float
        Permutation test p-value (circular shift).
    significant : bool
        Whether p_value < 0.05.
    peak_time_s : float
        Time of maximum rate relative to event onset.
    peak_rate_hz : float
        Maximum rate across all bins.
    """

    event_type: str
    time_bins_s: tuple
    rate_hz: tuple
    rate_ci_lower_hz: tuple
    rate_ci_upper_hz: tuple
    baseline_rate_hz: float
    n_events: int
    p_value: float
    significant: bool
    peak_time_s: float
    peak_rate_hz: float


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _count_peri_event(
    usv_times: np.ndarray,
    event_times: np.ndarray,
    bin_edges: np.ndarray,
    window_before: float,
    window_after: float,
) -> np.ndarray:
    """Count USVs in peri-event bins across all events.

    Uses binary search (np.searchsorted) per event for efficiency —
    O(E * log(U)) where E = events, U = USVs.

    Parameters
    ----------
    usv_times : ndarray
        Sorted USV onset times in seconds.
    event_times : ndarray
        Event onset times in seconds.
    bin_edges : ndarray
        Bin edges relative to event onset (seconds).
    window_before : float
        Time before event onset (positive value).
    window_after : float
        Time after event onset (positive value).

    Returns
    -------
    ndarray
        Total USV count per bin (shape: n_bins).
    """
    n_bins = len(bin_edges) - 1
    total_counts = np.zeros(n_bins, dtype=np.float64)

    for event_t in event_times:
        # Find USVs within the full window around this event
        win_start = event_t - window_before
        win_end = event_t + window_after
        i_lo = np.searchsorted(usv_times, win_start, side="left")
        i_hi = np.searchsorted(usv_times, win_end, side="right")

        if i_lo >= i_hi:
            continue

        # Compute relative times and histogram into bins
        relative_times = usv_times[i_lo:i_hi] - event_t
        counts, _ = np.histogram(relative_times, bins=bin_edges)
        total_counts += counts

    return total_counts


def compute_peth(
    usv_times_s: list[float] | np.ndarray,
    event_times_s: list[float] | np.ndarray,
    recording_duration_s: float,
    config: Optional[PETHConfig] = None,
    event_type: str = "unknown",
    rng: Optional[np.random.Generator] = None,
) -> Optional[PETHResult]:
    """Compute a peri-event time histogram for one event type.

    Parameters
    ----------
    usv_times_s : array-like
        USV onset times in seconds (need not be sorted).
    event_times_s : array-like
        Behavioral event onset times in seconds.
    recording_duration_s : float
        Total recording duration in seconds (for baseline and permutation).
    config : PETHConfig, optional
        Configuration. Uses defaults if None.
    event_type : str
        Name of the event type (for labeling the result).
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    PETHResult or None
        None if fewer than min_events events are provided.
    """
    if config is None:
        config = PETHConfig()
    if rng is None:
        rng = np.random.default_rng()

    usv_arr = np.asarray(usv_times_s, dtype=np.float64)
    event_arr = np.asarray(event_times_s, dtype=np.float64)

    # Validate: need enough events
    if len(event_arr) < config.min_events:
        return None

    # Sort USV times for binary search
    usv_arr = np.sort(usv_arr)

    # Build bin edges relative to event onset
    n_bins = config.n_bins
    bin_edges = np.linspace(
        -config.window_before_s, config.window_after_s, n_bins + 1
    )
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    # --- Count USVs per bin ---
    counts = _count_peri_event(
        usv_arr, event_arr, bin_edges,
        config.window_before_s, config.window_after_s,
    )

    # Rate = counts / (n_events * actual_bin_width)
    # Use actual bin width from linspace (not config.bin_size_s) to handle
    # cases where bin_size doesn't divide evenly into the window.
    actual_bin_width = (config.window_before_s + config.window_after_s) / n_bins
    rate = counts / (len(event_arr) * actual_bin_width)

    # --- Baseline rate ---
    if config.baseline_method == "whole_recording":
        if recording_duration_s > 0:
            baseline = len(usv_arr) / recording_duration_s
        else:
            baseline = 0.0
    else:  # pre_event
        pre_mask = bin_centers < 0
        if np.any(pre_mask):
            baseline = float(np.mean(rate[pre_mask]))
        else:
            baseline = 0.0

    # --- Bootstrap CI (200 resamples over event indices) ---
    n_bootstrap = 200
    n_events = len(event_arr)
    bootstrap_rates = np.zeros((n_bootstrap, n_bins), dtype=np.float64)

    for i in range(n_bootstrap):
        resample_idx = rng.integers(0, n_events, size=n_events)
        resampled_events = event_arr[resample_idx]
        boot_counts = _count_peri_event(
            usv_arr, resampled_events, bin_edges,
            config.window_before_s, config.window_after_s,
        )
        bootstrap_rates[i] = boot_counts / (n_events * actual_bin_width)

    ci_lower = np.percentile(bootstrap_rates, 2.5, axis=0)
    ci_upper = np.percentile(bootstrap_rates, 97.5, axis=0)

    # --- Permutation test (circular shift) ---
    # Observed peak rate
    observed_peak = float(np.max(rate))

    n_exceed = 0
    for _ in range(config.n_permutations):
        # Shift all USV times by a random offset, wrapping around
        shift = rng.uniform(0, recording_duration_s)
        shifted_usvs = np.mod(usv_arr + shift, recording_duration_s)
        shifted_usvs = np.sort(shifted_usvs)

        perm_counts = _count_peri_event(
            shifted_usvs, event_arr, bin_edges,
            config.window_before_s, config.window_after_s,
        )
        perm_rate = perm_counts / (n_events * actual_bin_width)
        perm_peak = float(np.max(perm_rate))

        if perm_peak >= observed_peak:
            n_exceed += 1

    # Conservative p-value: avoids zero p-values
    p_value = (n_exceed + 1) / (config.n_permutations + 1)

    # --- Peak extraction ---
    peak_idx = int(np.argmax(rate))
    peak_time = float(bin_centers[peak_idx])
    peak_rate = float(rate[peak_idx])

    return PETHResult(
        event_type=event_type,
        time_bins_s=tuple(float(x) for x in bin_centers),
        rate_hz=tuple(float(x) for x in rate),
        rate_ci_lower_hz=tuple(float(x) for x in ci_lower),
        rate_ci_upper_hz=tuple(float(x) for x in ci_upper),
        baseline_rate_hz=float(baseline),
        n_events=n_events,
        p_value=float(p_value),
        significant=p_value < 0.05,
        peak_time_s=peak_time,
        peak_rate_hz=peak_rate,
    )


def compute_all_peths(
    usv_times_s: list[float] | np.ndarray,
    events_by_type: dict[str, list[float]],
    recording_duration_s: float,
    config: Optional[PETHConfig] = None,
    rng: Optional[np.random.Generator] = None,
) -> dict[str, PETHResult]:
    """Compute PETHs for multiple event types.

    Parameters
    ----------
    usv_times_s : array-like
        USV onset times in seconds.
    events_by_type : dict
        Mapping from event type name to list of onset times (seconds).
    recording_duration_s : float
        Total recording duration in seconds.
    config : PETHConfig, optional
        Configuration (shared across all event types).
    rng : numpy Generator, optional
        Random number generator for reproducibility.

    Returns
    -------
    dict[str, PETHResult]
        Results keyed by event type. Types with too few events are omitted.
    """
    if config is None:
        config = PETHConfig()
    if rng is None:
        rng = np.random.default_rng()

    results = {}
    for event_type, event_times in events_by_type.items():
        result = compute_peth(
            usv_times_s=usv_times_s,
            event_times_s=event_times,
            recording_duration_s=recording_duration_s,
            config=config,
            event_type=event_type,
            rng=rng,
        )
        if result is not None:
            results[event_type] = result

    return results


# ---------------------------------------------------------------------------
# Population comparison (descriptive)
# ---------------------------------------------------------------------------


def compare_populations(
    group_a: dict[str, PETHResult],
    group_b: dict[str, PETHResult],
    group_a_name: str = "Group A",
    group_b_name: str = "Group B",
) -> dict[str, dict]:
    """Descriptive comparison of PETHs between two populations.

    For each event type present in both groups, computes:
    - Peak rate and peak-to-baseline ratio for each group
    - Pointwise rate difference (A - B)

    Note: Formal statistical comparison (e.g., Mann-Whitney U) requires
    multiple recordings per group (one PETH per recording). This function
    provides descriptive comparison for a single recording per group.

    Parameters
    ----------
    group_a, group_b : dict[str, PETHResult]
        PETH results keyed by event type, one per group.
    group_a_name, group_b_name : str
        Labels for the groups.

    Returns
    -------
    dict[str, dict]
        Comparison dict keyed by event type. Each value contains:
        - group_a_name, group_b_name
        - peak_rate_a, peak_rate_b
        - peak_baseline_ratio_a, peak_baseline_ratio_b
        - rate_difference (tuple: A rate - B rate per bin)
        - time_bins_s
    """
    common_types = set(group_a.keys()) & set(group_b.keys())
    comparisons = {}

    for event_type in sorted(common_types):
        a = group_a[event_type]
        b = group_b[event_type]

        rate_a = np.array(a.rate_hz)
        rate_b = np.array(b.rate_hz)

        # Truncate to common length if bin counts differ
        min_len = min(len(rate_a), len(rate_b))
        rate_diff = rate_a[:min_len] - rate_b[:min_len]

        # Peak-to-baseline ratio (guard division by zero)
        ratio_a = (a.peak_rate_hz / a.baseline_rate_hz) if a.baseline_rate_hz > 0 else float("inf")
        ratio_b = (b.peak_rate_hz / b.baseline_rate_hz) if b.baseline_rate_hz > 0 else float("inf")

        comparisons[event_type] = {
            "group_a_name": group_a_name,
            "group_b_name": group_b_name,
            "peak_rate_a": a.peak_rate_hz,
            "peak_rate_b": b.peak_rate_hz,
            "peak_baseline_ratio_a": ratio_a,
            "peak_baseline_ratio_b": ratio_b,
            "rate_difference": tuple(float(x) for x in rate_diff),
            "time_bins_s": a.time_bins_s[:min_len],
        }

    return comparisons


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_peth(
    result: PETHResult,
    ax=None,
    show_ci: bool = True,
    show_baseline: bool = True,
) -> None:
    """Plot a single peri-event time histogram.

    Renders a step histogram of USV rate with optional shaded confidence
    interval and baseline reference line.

    Parameters
    ----------
    result : PETHResult
        PETH result to plot.
    ax : matplotlib Axes, optional
        Axes to draw on. Creates a new figure if None.
    show_ci : bool
        Whether to show the 95% bootstrap confidence interval.
    show_baseline : bool
        Whether to show the baseline rate as a dashed line.
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 4))

    times = np.array(result.time_bins_s)
    rate = np.array(result.rate_hz)

    # Step histogram of rate
    ax.step(times, rate, where="mid", color="steelblue", linewidth=1.5, label="USV rate")

    # Confidence interval
    if show_ci:
        ci_lo = np.array(result.rate_ci_lower_hz)
        ci_hi = np.array(result.rate_ci_upper_hz)
        ax.fill_between(
            times, ci_lo, ci_hi,
            step="mid", alpha=0.2, color="steelblue", label="95% CI",
        )

    # Baseline
    if show_baseline:
        ax.axhline(
            result.baseline_rate_hz, color="gray", linestyle="--",
            linewidth=1, label=f"Baseline ({result.baseline_rate_hz:.2f} Hz)",
        )

    # Event onset line
    ax.axvline(0, color="red", linestyle="-", linewidth=1, alpha=0.7, label="Event onset")

    sig_str = f"p={result.p_value:.3f}" + (" *" if result.significant else "")
    ax.set_title(f"{result.event_type} (n={result.n_events}, {sig_str})")
    ax.set_xlabel("Time relative to event onset (s)")
    ax.set_ylabel("USV rate (Hz)")
    ax.legend(fontsize=8)


def plot_all_peths(
    results: dict[str, PETHResult],
    n_cols: int = 3,
    figsize_per_panel: tuple[float, float] = (5, 3.5),
) -> Optional[object]:
    """Plot all PETHs in a grid layout.

    Parameters
    ----------
    results : dict[str, PETHResult]
        PETH results keyed by event type.
    n_cols : int
        Number of columns in the grid. Default 3.
    figsize_per_panel : tuple
        (width, height) per panel in inches.

    Returns
    -------
    matplotlib.figure.Figure or None
        The figure, or None if results is empty.
    """
    import math

    import matplotlib.pyplot as plt

    n = len(results)
    if n == 0:
        return None

    n_rows = math.ceil(n / n_cols)
    fig_w = figsize_per_panel[0] * min(n, n_cols)
    fig_h = figsize_per_panel[1] * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h), squeeze=False)

    for idx, (event_type, result) in enumerate(sorted(results.items())):
        row, col = divmod(idx, n_cols)
        plot_peth(result, ax=axes[row][col])

    # Hide unused axes
    for idx in range(n, n_rows * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row][col].set_visible(False)

    fig.tight_layout()
    return fig

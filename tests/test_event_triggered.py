"""Tests for the event-triggered USV rate analysis (PETH) module.

All tests use synthetic USV and event times — no real data required.
Statistical tests use fixed random seeds for deterministic results.
"""

from __future__ import annotations

import numpy as np
import pytest

from usv_spectrogram.lmt.event_triggered import (
    PETHConfig,
    PETHResult,
    compare_populations,
    compute_all_peths,
    compute_peth,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> PETHConfig:
    """Default PETH configuration."""
    return PETHConfig()


@pytest.fixture
def fast_config() -> PETHConfig:
    """Fast config with fewer permutations for speed."""
    return PETHConfig(n_permutations=50, min_events=3)


@pytest.fixture
def uniform_usv_times() -> list[float]:
    """USV times at ~1 Hz over 100 seconds with random jitter.

    Jitter prevents aliasing artifacts when events are also periodic.
    Mean rate is ~1 Hz, but times don't align to integer boundaries.
    """
    rng = np.random.default_rng(99)
    # ~1 Hz Poisson process: 100 events in 100 seconds
    times = sorted(rng.uniform(0, 100, size=100).tolist())
    return times


@pytest.fixture
def clustered_usv_times() -> list[float]:
    """USVs that consistently occur 0.1-0.4s after events at t=10,20,...,90.

    Creates a strong temporal correlation: USVs cluster in the 0.1-0.4s
    post-event window. Should produce a visible peak in PETH.
    """
    rng = np.random.default_rng(42)
    event_onsets = list(range(10, 100, 10))  # 9 events
    usv_times = []
    for t in event_onsets:
        # 3-5 USVs per event, clustered 0.1-0.4s after onset
        n_usvs = rng.integers(3, 6)
        for _ in range(n_usvs):
            usv_times.append(t + rng.uniform(0.1, 0.4))
    return sorted(usv_times)


@pytest.fixture
def event_times_regular() -> list[float]:
    """Regular event times at t=10, 20, ..., 90 seconds."""
    return [float(t) for t in range(10, 100, 10)]


# ---------------------------------------------------------------------------
# PETHConfig tests
# ---------------------------------------------------------------------------


class TestPETHConfig:
    """Tests for PETHConfig dataclass."""

    def test_default_config(self, default_config: PETHConfig):
        """Default config should match the spec."""
        assert default_config.window_before_s == 2.0
        assert default_config.window_after_s == 2.0
        assert default_config.bin_size_s == 0.1
        assert default_config.n_permutations == 1000
        assert default_config.min_events == 5
        assert default_config.baseline_method == "whole_recording"

    def test_n_bins_property(self, default_config: PETHConfig):
        """n_bins should be (window_before + window_after) / bin_size."""
        # (2.0 + 2.0) / 0.1 = 40
        assert default_config.n_bins == 40

    def test_invalid_config_window_before(self):
        """window_before_s <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="window_before_s must be positive"):
            PETHConfig(window_before_s=0.0)

    def test_invalid_config_window_after(self):
        """window_after_s <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="window_after_s must be positive"):
            PETHConfig(window_after_s=-1.0)

    def test_invalid_config_bin_size(self):
        """bin_size_s <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="bin_size_s must be positive"):
            PETHConfig(bin_size_s=0.0)

    def test_invalid_config_n_permutations(self):
        """n_permutations < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="n_permutations must be >= 1"):
            PETHConfig(n_permutations=0)

    def test_invalid_config_min_events(self):
        """min_events < 1 should raise ValueError."""
        with pytest.raises(ValueError, match="min_events must be >= 1"):
            PETHConfig(min_events=0)

    def test_invalid_config_baseline_method(self):
        """Invalid baseline_method should raise ValueError."""
        with pytest.raises(ValueError, match="baseline_method must be one of"):
            PETHConfig(baseline_method="invalid")

    def test_frozen(self, default_config: PETHConfig):
        """Config should be immutable."""
        with pytest.raises(AttributeError):
            default_config.window_before_s = 5.0


# ---------------------------------------------------------------------------
# compute_peth tests
# ---------------------------------------------------------------------------


class TestComputePETH:
    """Tests for the core PETH computation."""

    def test_uniform_rate_flat_peth(
        self, fast_config: PETHConfig, uniform_usv_times: list[float],
        event_times_regular: list[float],
    ):
        """Uniform 1 Hz USVs with random events should produce flat ~1 Hz PETH.

        Since USVs are at exactly 1 Hz and events are regular, the PETH
        should be approximately flat at ~1 Hz (within statistical noise).
        """
        result = compute_peth(
            usv_times_s=uniform_usv_times,
            event_times_s=event_times_regular,
            recording_duration_s=100.0,
            config=fast_config,
            event_type="test_uniform",
            rng=np.random.default_rng(123),
        )

        assert result is not None
        assert result.event_type == "test_uniform"

        rates = np.array(result.rate_hz)
        # Mean rate should be close to 1.0 Hz (within tolerance for discrete bins)
        assert np.mean(rates) == pytest.approx(1.0, abs=0.5)
        # Standard deviation should be small (flat-ish)
        assert np.std(rates) < 1.0

    def test_clustered_usvs_show_peak(
        self, fast_config: PETHConfig, clustered_usv_times: list[float],
        event_times_regular: list[float],
    ):
        """USVs clustered 0.1-0.4s after events should produce a peak there.

        The peak should be in the first few positive bins (0 to 0.5s),
        and the peak rate should exceed the baseline significantly.
        """
        result = compute_peth(
            usv_times_s=clustered_usv_times,
            event_times_s=event_times_regular,
            recording_duration_s=100.0,
            config=fast_config,
            event_type="clustered",
            rng=np.random.default_rng(42),
        )

        assert result is not None
        # Peak should be in the post-onset window (positive time)
        assert result.peak_time_s > 0
        assert result.peak_time_s < 1.0
        # Peak rate should be well above baseline
        assert result.peak_rate_hz > result.baseline_rate_hz

    def test_too_few_events_returns_none(self, default_config: PETHConfig):
        """Fewer events than min_events should return None."""
        result = compute_peth(
            usv_times_s=[1.0, 2.0, 3.0],
            event_times_s=[10.0, 20.0],  # Only 2, default min is 5
            recording_duration_s=100.0,
            config=default_config,
            event_type="too_few",
        )
        assert result is None

    def test_correlated_significant(
        self, clustered_usv_times: list[float],
        event_times_regular: list[float],
    ):
        """Correlated USVs should yield p < 0.05 in permutation test.

        Uses more permutations for reliable statistical testing.
        """
        config = PETHConfig(n_permutations=199, min_events=3)
        result = compute_peth(
            usv_times_s=clustered_usv_times,
            event_times_s=event_times_regular,
            recording_duration_s=100.0,
            config=config,
            event_type="correlated",
            rng=np.random.default_rng(42),
        )

        assert result is not None
        assert result.significant is True
        assert result.p_value < 0.05

    def test_uniform_not_significant(
        self, uniform_usv_times: list[float],
        event_times_regular: list[float],
    ):
        """Uniform (uncorrelated) USVs should yield p > 0.05.

        Since USVs are ~1 Hz with random jitter, there's no temporal
        correlation with events beyond what chance would produce.
        Uses 499 permutations (min p = 0.002) for robust testing.
        """
        config = PETHConfig(n_permutations=499, min_events=3)
        result = compute_peth(
            usv_times_s=uniform_usv_times,
            event_times_s=event_times_regular,
            recording_duration_s=100.0,
            config=config,
            event_type="uniform",
            rng=np.random.default_rng(42),
        )

        assert result is not None
        assert result.p_value > 0.05

    def test_ci_bounds_contain_rate(
        self, fast_config: PETHConfig, uniform_usv_times: list[float],
        event_times_regular: list[float],
    ):
        """CI lower should be <= rate and CI upper should be >= rate.

        The point estimate (rate) should generally fall within its own
        bootstrap confidence interval.
        """
        result = compute_peth(
            usv_times_s=uniform_usv_times,
            event_times_s=event_times_regular,
            recording_duration_s=100.0,
            config=fast_config,
            event_type="ci_test",
            rng=np.random.default_rng(42),
        )

        assert result is not None
        ci_lo = np.array(result.rate_ci_lower_hz)
        ci_hi = np.array(result.rate_ci_upper_hz)
        rate = np.array(result.rate_hz)

        # Bootstrap percentile CI: ci_lower should be <= rate in all bins
        # (the point estimate is included in the bootstrap distribution),
        # with small tolerance for floating-point noise.
        eps = 0.01
        assert np.all(ci_lo <= rate + eps), "CI lower exceeds rate in some bins"
        assert np.all(ci_hi >= rate - eps), "CI upper below rate in some bins"

    def test_no_usvs_zero_rate(
        self, fast_config: PETHConfig, event_times_regular: list[float],
    ):
        """Empty USV list should produce all-zero rate."""
        result = compute_peth(
            usv_times_s=[],
            event_times_s=event_times_regular,
            recording_duration_s=100.0,
            config=fast_config,
            event_type="no_usvs",
            rng=np.random.default_rng(42),
        )

        assert result is not None
        assert all(r == 0.0 for r in result.rate_hz)
        assert result.peak_rate_hz == 0.0
        assert result.baseline_rate_hz == 0.0

    def test_short_recording_handled(self, fast_config: PETHConfig):
        """Recording shorter than the window should not crash.

        The window extends beyond the recording boundaries, but events
        at the boundaries may have fewer USVs. The function should
        handle this gracefully without errors.
        """
        # 5s recording, 2s window on each side, events at boundaries
        usv_times = [0.5, 1.0, 1.5, 2.5, 3.0, 4.0]
        event_times = [1.0, 2.0, 3.0, 4.0]
        config = PETHConfig(
            window_before_s=2.0, window_after_s=2.0,
            bin_size_s=0.5, n_permutations=10, min_events=3,
        )

        result = compute_peth(
            usv_times_s=usv_times,
            event_times_s=event_times,
            recording_duration_s=5.0,
            config=config,
            event_type="short",
            rng=np.random.default_rng(42),
        )

        assert result is not None
        assert len(result.rate_hz) == config.n_bins
        assert result.n_events == 4

    def test_result_is_frozen(
        self, fast_config: PETHConfig, uniform_usv_times: list[float],
        event_times_regular: list[float],
    ):
        """PETHResult should be immutable."""
        result = compute_peth(
            usv_times_s=uniform_usv_times,
            event_times_s=event_times_regular,
            recording_duration_s=100.0,
            config=fast_config,
            event_type="frozen_test",
            rng=np.random.default_rng(42),
        )
        assert result is not None
        with pytest.raises(AttributeError):
            result.p_value = 0.0

    def test_pre_event_baseline(
        self, uniform_usv_times: list[float],
        event_times_regular: list[float],
    ):
        """pre_event baseline should use mean of pre-onset bins."""
        config = PETHConfig(
            n_permutations=10, min_events=3,
            baseline_method="pre_event",
        )
        result = compute_peth(
            usv_times_s=uniform_usv_times,
            event_times_s=event_times_regular,
            recording_duration_s=100.0,
            config=config,
            event_type="pre_baseline",
            rng=np.random.default_rng(42),
        )

        assert result is not None
        # For uniform 1 Hz USVs, baseline should be approximately 1.0
        assert result.baseline_rate_hz == pytest.approx(1.0, abs=0.5)


# ---------------------------------------------------------------------------
# compute_all_peths tests
# ---------------------------------------------------------------------------


class TestComputeAllPETHs:
    """Tests for computing PETHs across multiple event types."""

    def test_multiple_event_types(
        self, fast_config: PETHConfig, uniform_usv_times: list[float],
    ):
        """Should return results for each event type with enough events."""
        events_by_type = {
            "TypeA": [10.0, 20.0, 30.0, 40.0, 50.0],
            "TypeB": [15.0, 25.0, 35.0, 45.0, 55.0],
            "TypeC": [5.0],  # Too few events (min_events=3 in fast_config)
        }

        results = compute_all_peths(
            usv_times_s=uniform_usv_times,
            events_by_type=events_by_type,
            recording_duration_s=100.0,
            config=fast_config,
            rng=np.random.default_rng(42),
        )

        assert "TypeA" in results
        assert "TypeB" in results
        assert "TypeC" not in results  # Filtered out (only 1 event < min_events=3)

    def test_empty_events(self, fast_config: PETHConfig):
        """No event types should return empty dict."""
        results = compute_all_peths(
            usv_times_s=[1.0, 2.0],
            events_by_type={},
            recording_duration_s=100.0,
            config=fast_config,
        )
        assert results == {}


# ---------------------------------------------------------------------------
# compare_populations tests
# ---------------------------------------------------------------------------


class TestComparePopulations:
    """Tests for descriptive population comparison."""

    def test_compare_output_keys(
        self, fast_config: PETHConfig,
        uniform_usv_times: list[float],
        clustered_usv_times: list[float],
        event_times_regular: list[float],
    ):
        """Comparison dict should have expected keys for common event types."""
        rng = np.random.default_rng(42)

        group_a = compute_all_peths(
            usv_times_s=uniform_usv_times,
            events_by_type={"EventX": event_times_regular},
            recording_duration_s=100.0,
            config=fast_config,
            rng=rng,
        )
        group_b = compute_all_peths(
            usv_times_s=clustered_usv_times,
            events_by_type={"EventX": event_times_regular},
            recording_duration_s=100.0,
            config=fast_config,
            rng=rng,
        )

        comparison = compare_populations(group_a, group_b, "Wild", "Lab")

        assert "EventX" in comparison
        c = comparison["EventX"]
        assert c["group_a_name"] == "Wild"
        assert c["group_b_name"] == "Lab"
        assert "peak_rate_a" in c
        assert "peak_rate_b" in c
        assert "peak_baseline_ratio_a" in c
        assert "peak_baseline_ratio_b" in c
        assert "rate_difference" in c
        assert "time_bins_s" in c

    def test_compare_no_common_types(self):
        """Groups with no common event types should return empty dict."""
        # Create minimal mock results
        result_a = PETHResult(
            event_type="TypeA", time_bins_s=(0.0,), rate_hz=(1.0,),
            rate_ci_lower_hz=(0.5,), rate_ci_upper_hz=(1.5,),
            baseline_rate_hz=1.0, n_events=10, p_value=0.5,
            significant=False, peak_time_s=0.0, peak_rate_hz=1.0,
        )
        result_b = PETHResult(
            event_type="TypeB", time_bins_s=(0.0,), rate_hz=(2.0,),
            rate_ci_lower_hz=(1.5,), rate_ci_upper_hz=(2.5,),
            baseline_rate_hz=2.0, n_events=10, p_value=0.5,
            significant=False, peak_time_s=0.0, peak_rate_hz=2.0,
        )

        comparison = compare_populations(
            {"TypeA": result_a}, {"TypeB": result_b}
        )
        assert comparison == {}

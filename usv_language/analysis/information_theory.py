"""Statistically rigorous information theory metrics for VQ-VAE code sequences.

Extends Phase 8.4 sequence_analysis with:
- MLE-based Zipf exponent estimation (Clauset et al. 2009 discrete MLE)
- Shannon entropy inversion for PLC alpha estimation
- Bias-corrected entropy rates (Miller-Madow)
- Shuffle-surrogate idiom detection with Benjamini-Hochberg FDR
- Bootstrap confidence intervals for n-gram productivity
- Burstiness coefficient (CV of inter-event intervals)

All functions accept 1D integer arrays (VQ-VAE code indices) or float arrays
(event timestamps).  No matplotlib at module level -- computation only.

Dependencies: numpy, scipy (already in requirements).  ``powerlaw`` optional.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import optimize, stats

from usv_language.analysis.sequence_analysis import (
    extract_ngrams,
    mutual_information_at_lag,
)

# Optional powerlaw package (Clauset et al. reference implementation)
try:
    import powerlaw as _powerlaw

    _HAS_POWERLAW = True
except ImportError:
    _HAS_POWERLAW = False


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ZipfResult:
    """Result from discrete MLE Zipf exponent estimation.

    **Important:** ``alpha`` is the power-law exponent of the **count-value
    distribution** (Clauset et al. 2009 applied to frequency counts), NOT the
    standard rank-frequency Zipf exponent.  The two are related by::

        alpha_rank = 1.0 / (alpha - 1.0)   # when alpha > 1

    Use ``rank_alpha`` for comparison with the standard Zipf reference
    (natural language alpha_rank ~ 1.0) and with ``ZipfEntropyResult.alpha_estimate``
    (which directly estimates the rank-frequency exponent).

    Attributes
    ----------
    alpha : float
        Count-distribution power-law exponent (> 0 for valid fit).
    xmin : float
        Lower bound of the power-law region.
    p_value : float
        Goodness-of-fit p-value (> 0.1 suggests power-law plausible).
    n_tail : int
        Number of data points in the tail (x >= xmin).
    log_likelihood_ratio : float
        Log-likelihood ratio vs exponential alternative (> 0 favours power-law).
    rank_alpha : float
        Derived rank-frequency exponent: 1/(alpha - 1).  Comparable to
        ``ZipfEntropyResult.alpha_estimate`` and the standard Zipf reference.
        ``inf`` when alpha <= 1 (no valid transform).
    """

    alpha: float
    xmin: float
    p_value: float
    n_tail: int
    log_likelihood_ratio: float
    rank_alpha: float = 0.0

    def __post_init__(self) -> None:
        if self.rank_alpha == 0.0 and self.alpha > 1.0:
            # Frozen dataclass: use object.__setattr__
            object.__setattr__(
                self, "rank_alpha",
                1.0 / (self.alpha - 1.0),
            )
        elif self.rank_alpha == 0.0 and self.alpha > 0.0:
            object.__setattr__(self, "rank_alpha", float("inf"))


@dataclass(frozen=True)
class ZipfEntropyResult:
    """Result from Shannon entropy inversion for PLC alpha.

    Attributes
    ----------
    alpha_estimate : float
        Power-law exponent recovered by inverting H(alpha, K) = H_obs.
    entropy_observed : float
        Observed Shannon entropy (bits).
    entropy_ci : tuple[float, float]
        Bootstrap 95% CI on observed entropy.
    method : str
        Inversion method used ("brentq" or "grid_search").
    """

    alpha_estimate: float
    entropy_observed: float
    entropy_ci: tuple[float, float]
    method: str


@dataclass(frozen=True)
class EntropyRateResult:
    """Result from bias-corrected entropy rate estimation.

    Attributes
    ----------
    orders : list[int]
        N-gram orders computed (1 .. max_order).
    rates_plugin : list[float]
        Plugin (naive) entropy rates h_n = H(n-gram) / n.
    rates_corrected : list[float]
        Bias-corrected rates (Miller-Madow).
    convergence_order : int
        Smallest order k where |h_k - h_{k-1}| < 0.01 for 2 consecutive.
        0 if convergence not detected.
    """

    orders: list[int]
    rates_plugin: list[float]
    rates_corrected: list[float]
    convergence_order: int


@dataclass(frozen=True)
class IdiomResult:
    """A single statistically significant n-gram idiom.

    Attributes
    ----------
    ngram : tuple[int, ...]
        The n-gram sequence.
    observed_count : int
        Count in the original sequence.
    expected_count : float
        Mean count across shuffled surrogates.
    z_score : float
        (observed - expected) / std_shuffled.
    p_value : float
        One-sided p-value from normal CDF.
    n : int
        N-gram order.
    """

    ngram: tuple[int, ...]
    observed_count: int
    expected_count: float
    z_score: float
    p_value: float
    n: int


@dataclass(frozen=True)
class BurstinessResult:
    """Burstiness analysis of event timestamps.

    Attributes
    ----------
    cv : float
        Coefficient of variation of inter-event intervals.
    mean_iei : float
        Mean inter-event interval.
    std_iei : float
        Standard deviation of inter-event intervals.
    n_bursts : int
        Number of detected burst clusters (simplified detection).
    mean_burst_duration : float
        Mean duration of burst clusters.
    mean_inter_burst_interval : float
        Mean gap between burst clusters.
    interpretation : str
        One of "periodic", "poisson", "bursty", "insufficient_data".
    """

    cv: float
    mean_iei: float
    std_iei: float
    n_bursts: int
    mean_burst_duration: float
    mean_inter_burst_interval: float
    interpretation: str


@dataclass(frozen=True)
class ProductivityResult:
    """N-gram productivity with null-hypothesis reference interval.

    The bootstrap resamples sequence elements i.i.d. (destroying sequential
    structure), so ``null_ci_lower``/``null_ci_upper`` represent the
    productivity distribution under the null hypothesis of token independence,
    NOT a confidence interval around the observed ratio.

    Attributes
    ----------
    observed : int
        Number of distinct n-grams observed.
    possible : int
        Maximum possible n-grams (K^n, capped at 2^53).
    productivity_ratio : float
        observed / possible.
    null_ci_lower : float
        2.5th percentile of productivity under token-independence null.
    null_ci_upper : float
        97.5th percentile of productivity under token-independence null.
    n : int
        N-gram order.
    """

    observed: int
    possible: int
    productivity_ratio: float
    null_ci_lower: float
    null_ci_upper: float
    n: int


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _generalized_harmonic(alpha: float, K: int) -> float:
    """Compute generalized harmonic number Z(alpha, K) = sum(k^{-alpha}, k=1..K).

    This is the normalization constant for a truncated power-law (PLC)
    distribution over K categories.  We compute it by direct summation
    (not scipy.special.zeta, which sums to infinity).

    Parameters
    ----------
    alpha : float
        Power-law exponent.
    K : int
        Number of categories.
    """
    ks = np.arange(1, K + 1, dtype=np.float64)
    return float(np.sum(ks ** (-alpha)))


def _plc_entropy(alpha: float, K: int) -> float:
    """Shannon entropy of a truncated power-law (PLC) distribution.

    For P(k) = k^{-alpha} / Z(alpha, K):
        H(alpha) = log2(Z) + alpha * (1/Z) * sum(k^{-alpha} * ln(k)) / ln(2)

    Parameters
    ----------
    alpha : float
        Power-law exponent.
    K : int
        Number of categories.
    """
    ks = np.arange(1, K + 1, dtype=np.float64)
    weights = ks ** (-alpha)
    Z = weights.sum()
    if Z <= 0:
        return 0.0
    # H = log2(Z) + alpha * sum(k^{-alpha} * ln(k)) / (Z * ln(2))
    log_ks = np.log(ks)  # natural log
    entropy = np.log2(Z) + alpha * np.sum(weights * log_ks) / (Z * np.log(2))
    return float(entropy)


def _benjamini_hochberg(p_values: np.ndarray, alpha: float = 0.01) -> np.ndarray:
    """Benjamini-Hochberg FDR correction.

    Parameters
    ----------
    p_values : ndarray
        Raw p-values.
    alpha : float
        FDR significance level.

    Returns
    -------
    Boolean array: True where the test is significant after correction.
    """
    m = len(p_values)
    if m == 0:
        return np.array([], dtype=bool)

    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # BH thresholds: (rank / m) * alpha
    thresholds = (np.arange(1, m + 1) / m) * alpha

    # Find largest k where p_(k) <= threshold_(k)
    significant = sorted_p <= thresholds
    if not significant.any():
        return np.zeros(m, dtype=bool)

    max_k = np.max(np.where(significant)[0])

    # All tests with rank <= max_k are significant
    result = np.zeros(m, dtype=bool)
    result[sorted_indices[: max_k + 1]] = True
    return result


# ---------------------------------------------------------------------------
# Zipf analysis (dual approach)
# ---------------------------------------------------------------------------


def zipf_exponent_mle(sequence: np.ndarray) -> ZipfResult:
    """Estimate Zipf exponent via discrete MLE (Clauset et al. 2009).

    Operates on the frequency distribution: each unique code's count is
    treated as a data point.  Tries the ``powerlaw`` package first; falls
    back to scipy-based discrete MLE if unavailable.

    Parameters
    ----------
    sequence : ndarray
        1D integer array of VQ-VAE code indices.

    Returns
    -------
    ZipfResult
        MLE alpha, xmin, p-value, tail count, log-likelihood ratio.
    """
    if len(sequence) == 0:
        return ZipfResult(alpha=0.0, xmin=0.0, p_value=1.0, n_tail=0,
                          log_likelihood_ratio=0.0)

    # Compute frequency distribution
    counts = Counter(sequence.tolist())
    freqs = np.array(sorted(counts.values(), reverse=True), dtype=np.float64)

    if len(freqs) < 10:
        return ZipfResult(alpha=0.0, xmin=0.0, p_value=1.0, n_tail=len(freqs),
                          log_likelihood_ratio=0.0)

    if _HAS_POWERLAW:
        return _zipf_mle_powerlaw(freqs)
    return _zipf_mle_scipy(freqs)


def _zipf_mle_powerlaw(freqs: np.ndarray) -> ZipfResult:
    """Zipf MLE using the ``powerlaw`` package."""
    fit = _powerlaw.Fit(freqs, discrete=True, verbose=False)
    alpha = fit.power_law.alpha
    xmin = fit.power_law.xmin

    # p-value via KS test
    p_value = 0.0
    try:
        # Distribution comparison gives (R, p)
        R, p = fit.distribution_compare("power_law", "exponential")
        log_likelihood_ratio = float(R)
    except Exception:
        log_likelihood_ratio = 0.0

    n_tail = int(np.sum(freqs >= xmin))

    return ZipfResult(
        alpha=float(alpha),
        xmin=float(xmin),
        p_value=float(p_value),
        n_tail=n_tail,
        log_likelihood_ratio=log_likelihood_ratio,
    )


def _zipf_mle_scipy(freqs: np.ndarray) -> ZipfResult:
    """Zipf MLE fallback using scipy.

    For each candidate xmin, computes discrete MLE:
        alpha = 1 + n * [sum(ln(x_i / (xmin - 0.5)))]^{-1}
    Picks xmin minimizing KS distance.  Bootstrap p-value (100 samples).
    """
    # Candidate xmin values: up to 20 values from unique frequencies
    unique_freqs = np.unique(freqs)
    if len(unique_freqs) > 20:
        candidates = unique_freqs[np.linspace(0, len(unique_freqs) - 1, 20, dtype=int)]
    else:
        candidates = unique_freqs

    best_alpha = 0.0
    best_xmin = float(candidates[0])
    best_ks = np.inf
    best_n_tail = 0

    for xmin in candidates:
        tail = freqs[freqs >= xmin]
        n = len(tail)
        if n < 5:
            continue

        # Discrete MLE formula (Clauset eq. 3.1)
        denom = np.sum(np.log(tail / (xmin - 0.5)))
        if denom <= 0:
            continue
        alpha = 1.0 + n / denom

        # KS distance between empirical and fitted CDF
        sorted_tail = np.sort(tail)
        # Theoretical CDF: F(x) = sum(k^{-alpha} for k=xmin..x) / Z
        ks_vals = np.arange(xmin, sorted_tail[-1] + 1)
        if len(ks_vals) == 0 or xmin <= 0:
            continue
        theoretical_pmf = ks_vals ** (-alpha)
        Z = theoretical_pmf.sum()
        if Z <= 0:
            continue
        theoretical_cdf = np.cumsum(theoretical_pmf) / Z

        # Map each data point to its CDF position
        empirical_cdf = np.arange(1, n + 1) / n
        # Find theoretical CDF at each data point
        data_cdf_indices = np.searchsorted(ks_vals, sorted_tail, side="right") - 1
        data_cdf_indices = np.clip(data_cdf_indices, 0, len(theoretical_cdf) - 1)
        theoretical_at_data = theoretical_cdf[data_cdf_indices]

        ks_stat = np.max(np.abs(empirical_cdf - theoretical_at_data))

        if ks_stat < best_ks:
            best_ks = ks_stat
            best_alpha = alpha
            best_xmin = float(xmin)
            best_n_tail = n

    # Bootstrap p-value (100 synthetic samples)
    rng = np.random.RandomState(42)
    n_boot = 100
    n_better = 0
    tail = freqs[freqs >= best_xmin]

    if len(tail) >= 5 and best_alpha > 1.0:
        for _ in range(n_boot):
            # Generate synthetic power-law sample (match actual data support)
            ks = np.arange(best_xmin, int(freqs.max()) + 1)
            if len(ks) == 0:
                break
            pmf = ks ** (-best_alpha)
            Z = pmf.sum()
            if Z <= 0:
                break
            pmf = pmf / Z
            synth = rng.choice(ks, size=len(tail), p=pmf)

            # Fit MLE to synthetic
            denom_s = np.sum(np.log(synth / (best_xmin - 0.5)))
            if denom_s <= 0:
                continue
            alpha_s = 1.0 + len(synth) / denom_s

            # KS for synthetic
            sorted_s = np.sort(synth)
            empirical_s = np.arange(1, len(synth) + 1) / len(synth)
            ks_range = np.arange(best_xmin, sorted_s[-1] + 1)
            if len(ks_range) == 0:
                continue
            theor_pmf = ks_range ** (-alpha_s)
            Z_s = theor_pmf.sum()
            if Z_s <= 0:
                continue
            theor_cdf = np.cumsum(theor_pmf) / Z_s
            idx = np.searchsorted(ks_range, sorted_s, side="right") - 1
            idx = np.clip(idx, 0, len(theor_cdf) - 1)
            ks_synth = np.max(np.abs(empirical_s - theor_cdf[idx]))

            if ks_synth >= best_ks:
                n_better += 1

        p_value = n_better / n_boot
    else:
        p_value = 1.0

    # Log-likelihood ratio vs exponential
    llr = 0.0
    if len(tail) >= 5 and best_alpha > 1.0:
        # Power-law log-likelihood (normalization over [xmin, xmax])
        ks_tail_range = np.arange(int(best_xmin), int(tail.max()) + 1,
                                  dtype=np.float64)
        Z_tail = np.sum(ks_tail_range ** (-best_alpha))
        ll_pl = -best_alpha * np.sum(np.log(tail)) - len(tail) * np.log(Z_tail)
        # Exponential fit
        rate = 1.0 / np.mean(tail)
        ll_exp = len(tail) * np.log(rate) - rate * np.sum(tail)
        llr = ll_pl - ll_exp

    return ZipfResult(
        alpha=float(best_alpha),
        xmin=float(best_xmin),
        p_value=float(p_value),
        n_tail=int(best_n_tail),
        log_likelihood_ratio=float(llr),
    )


def zipf_via_shannon_entropy(
    sequence: np.ndarray,
    K: Optional[int] = None,
) -> ZipfEntropyResult:
    """Estimate Zipf exponent by inverting the PLC entropy function.

    Computes observed Shannon entropy H_obs, then finds alpha such that
    H(alpha, K) = H_obs for the truncated power-law P(k) = k^{-alpha}/Z.

    Parameters
    ----------
    sequence : ndarray
        1D integer array of VQ-VAE code indices.
    K : int, optional
        Alphabet size.  If None, uses max(sequence) + 1.

    Returns
    -------
    ZipfEntropyResult
        Alpha estimate, observed entropy, bootstrap CI, method.
    """
    if len(sequence) == 0:
        return ZipfEntropyResult(
            alpha_estimate=0.0, entropy_observed=0.0,
            entropy_ci=(0.0, 0.0), method="none",
        )

    if K is None:
        K = int(np.max(sequence)) + 1

    # Observed entropy
    counts = Counter(sequence.tolist())
    total = sum(counts.values())
    probs = np.array([counts.get(k, 0) / total for k in range(K)])
    probs = probs[probs > 0]
    H_obs = float(-np.sum(probs * np.log2(probs)))

    # Bootstrap CI on entropy (200 resamples)
    rng = np.random.RandomState(42)
    boot_entropies = []
    for _ in range(200):
        resample = rng.choice(sequence, size=len(sequence), replace=True)
        c = Counter(resample.tolist())
        t = sum(c.values())
        p = np.array([c.get(k, 0) / t for k in range(K)])
        p = p[p > 0]
        boot_entropies.append(float(-np.sum(p * np.log2(p))))
    boot_entropies = np.array(boot_entropies)
    ci = (float(np.percentile(boot_entropies, 2.5)),
          float(np.percentile(boot_entropies, 97.5)))

    # Invert: find alpha where _plc_entropy(alpha, K) == H_obs
    # H(alpha=0, K) = log2(K) (uniform), H(alpha->inf, K) -> 0
    # So H is monotonically decreasing in alpha

    # Check if H_obs is in range
    H_uniform = _plc_entropy(0.01, K)
    H_steep = _plc_entropy(20.0, K)

    method = "brentq"
    alpha_estimate = 0.0

    if H_obs >= H_uniform:
        # Entropy too high for any PLC -- essentially uniform
        alpha_estimate = 0.01
        method = "clamped_high"
    elif H_obs <= H_steep:
        # Entropy too low -- very steep distribution
        alpha_estimate = 20.0
        method = "clamped_low"
    else:
        # Brentq inversion
        try:
            alpha_estimate = optimize.brentq(
                lambda a: _plc_entropy(a, K) - H_obs,
                0.01, 20.0,
                xtol=1e-6,
            )
        except (ValueError, RuntimeError):
            # Grid search fallback
            method = "grid_search"
            best_err = np.inf
            for a in np.linspace(0.01, 20.0, 2000):
                err = abs(_plc_entropy(a, K) - H_obs)
                if err < best_err:
                    best_err = err
                    alpha_estimate = a

    return ZipfEntropyResult(
        alpha_estimate=float(alpha_estimate),
        entropy_observed=float(H_obs),
        entropy_ci=ci,
        method=method,
    )


# ---------------------------------------------------------------------------
# Sequential structure
# ---------------------------------------------------------------------------


def entropy_rate(
    sequence: np.ndarray,
    max_order: int = 8,
    bias_correction: str = "miller_madow",
) -> EntropyRateResult:
    """Compute bias-corrected entropy rates at multiple n-gram orders.

    Extends the existing ``sequence_analysis.entropy_rate`` with Miller-Madow
    bias correction and convergence detection.

    Parameters
    ----------
    sequence : ndarray
        1D integer array of code indices.
    max_order : int
        Maximum n-gram order.
    bias_correction : str
        "miller_madow" or "none".

    Returns
    -------
    EntropyRateResult
        Plugin rates, corrected rates, convergence order.
    """
    if len(sequence) == 0:
        return EntropyRateResult(
            orders=[], rates_plugin=[], rates_corrected=[],
            convergence_order=0,
        )

    # Clamp max_order to sequence length
    max_order = min(max_order, len(sequence))

    orders = list(range(1, max_order + 1))
    rates_plugin = []
    rates_corrected = []

    for n in orders:
        ngrams = extract_ngrams(sequence, n)
        total = sum(ngrams.values())
        if total == 0:
            rates_plugin.append(0.0)
            rates_corrected.append(0.0)
            continue

        # Plugin entropy H_n = -sum(p * log2(p))
        h_plugin = 0.0
        m_nonzero = 0  # number of nonzero bins
        for count in ngrams.values():
            p = count / total
            if p > 0:
                h_plugin -= p * np.log2(p)
                m_nonzero += 1

        rate_plugin = h_plugin / n
        rates_plugin.append(float(rate_plugin))

        # Miller-Madow correction: H_corrected = H_plugin + (m-1)/(2*N*ln(2))
        if bias_correction == "miller_madow" and total > 0:
            correction = (m_nonzero - 1) / (2 * total * np.log(2))
            h_corrected = h_plugin + correction
            rates_corrected.append(float(h_corrected / n))
        else:
            rates_corrected.append(float(rate_plugin))

    # Convergence detection: |h_k - h_{k-1}| < 0.01 for 2 consecutive orders
    # When consecutive reaches 2 at index i, the two small deltas are at
    # (i-1, i) and (i-2, i-1).  The first order is orders[i-2].
    convergence_order = 0
    consecutive = 0
    for i in range(1, len(rates_corrected)):
        if abs(rates_corrected[i] - rates_corrected[i - 1]) < 0.01:
            consecutive += 1
            if consecutive >= 2 and convergence_order == 0:
                convergence_order = orders[i - 2]  # first order of the pair
        else:
            consecutive = 0

    return EntropyRateResult(
        orders=orders,
        rates_plugin=rates_plugin,
        rates_corrected=rates_corrected,
        convergence_order=convergence_order,
    )


def conditional_entropy_by_lag(
    sequence: np.ndarray,
    K: int,
    max_lag: int = 10,
) -> list[float]:
    """Compute H(X_t | X_{t-lag}) for each lag from 1 to max_lag.

    Uses the identity H(X|Y) = H(X) - I(X;Y).

    Parameters
    ----------
    sequence : ndarray
        1D integer array of code indices.
    K : int
        Alphabet size.
    max_lag : int
        Maximum lag to compute.

    Returns
    -------
    List of conditional entropies (one per lag, length = max_lag).
    """
    if len(sequence) == 0:
        return [0.0] * max_lag

    # Marginal entropy H(X_t)
    counts = Counter(sequence.tolist())
    total = sum(counts.values())
    probs = np.array([counts.get(k, 0) / total for k in range(K)])
    probs = probs[probs > 0]
    H_x = float(-np.sum(probs * np.log2(probs)))

    result = []
    for lag in range(1, max_lag + 1):
        mi = mutual_information_at_lag(sequence, K, lag)
        result.append(H_x - mi)
    return result


def mutual_information_rate(
    sequence: np.ndarray,
    K: int,
    max_lag: int = 20,
) -> list[float]:
    """Compute MI decay curve: I(X_t; X_{t+lag}) for lags 1..max_lag.

    Parameters
    ----------
    sequence : ndarray
        1D integer array of code indices.
    K : int
        Alphabet size.
    max_lag : int
        Maximum lag.

    Returns
    -------
    List of MI values (one per lag, length = max_lag).
    """
    if len(sequence) == 0:
        return [0.0] * max_lag
    return [mutual_information_at_lag(sequence, K, lag)
            for lag in range(1, max_lag + 1)]


# ---------------------------------------------------------------------------
# Compositionality
# ---------------------------------------------------------------------------


def ngram_productivity(
    sequence: np.ndarray,
    K: int,
    n: int = 2,
    n_bootstrap: int = 1000,
) -> ProductivityResult:
    """Measure n-gram productivity with bootstrap confidence interval.

    Parameters
    ----------
    sequence : ndarray
        1D integer array of code indices.
    K : int
        Alphabet size.
    n : int
        N-gram order.
    n_bootstrap : int
        Number of bootstrap resamples.

    Returns
    -------
    ProductivityResult
        Observed count, possible, ratio, CI bounds.
    """
    # Cap K^n to avoid overflow
    possible = min(K ** n, 2**53)

    if len(sequence) < n:
        return ProductivityResult(
            observed=0, possible=possible, productivity_ratio=0.0,
            null_ci_lower=0.0, null_ci_upper=0.0, n=n,
        )

    ngrams = extract_ngrams(sequence, n)
    observed = len(ngrams)
    ratio = observed / possible if possible > 0 else 0.0

    # Bootstrap CI
    rng = np.random.RandomState(42)
    boot_ratios = []
    for _ in range(n_bootstrap):
        resample = rng.choice(sequence, size=len(sequence), replace=True)
        boot_ngrams = extract_ngrams(resample, n)
        boot_ratios.append(len(boot_ngrams) / possible if possible > 0 else 0.0)

    boot_ratios = np.array(boot_ratios)
    null_ci_lower = float(np.percentile(boot_ratios, 2.5))
    null_ci_upper = float(np.percentile(boot_ratios, 97.5))

    return ProductivityResult(
        observed=observed,
        possible=possible,
        productivity_ratio=float(ratio),
        null_ci_lower=null_ci_lower,
        null_ci_upper=null_ci_upper,
        n=n,
    )


def ngram_idioms(
    sequence: np.ndarray,
    K: int,
    max_n: int = 5,
    n_shuffles: int = 100,
    fdr_alpha: float = 0.01,
) -> list[IdiomResult]:
    """Detect statistically over-represented n-grams via shuffle surrogates.

    For each n=2..max_n, counts n-grams in the original sequence and in
    shuffled versions.  Uses z-scores and Benjamini-Hochberg FDR correction
    to identify significant "idioms".

    Parameters
    ----------
    sequence : ndarray
        1D integer array of code indices.
    K : int
        Alphabet size.
    max_n : int
        Maximum n-gram length to test.
    n_shuffles : int
        Number of shuffle surrogates (default 100 for speed).
    fdr_alpha : float
        FDR significance threshold.

    Returns
    -------
    List of IdiomResult for significant n-grams, sorted by z-score descending.

    Note: Only tests n-grams present in the original sequence (one-sided test
    for over-representation).  Suppressed n-grams (below-chance frequency) are
    not reported.
    """
    if len(sequence) < 2:
        return []

    rng = np.random.RandomState(42)
    all_results: list[IdiomResult] = []

    for n in range(2, max_n + 1):
        if len(sequence) < n:
            continue

        # Original n-gram counts
        original_counts = extract_ngrams(sequence, n)
        if not original_counts:
            continue

        # Only track n-grams observed in original (performance optimization)
        observed_ngrams = set(original_counts.keys())

        # Shuffle surrogates
        shuffle_counts: dict[tuple, list[int]] = {
            ng: [] for ng in observed_ngrams
        }

        for _ in range(n_shuffles):
            shuffled = rng.permutation(sequence)
            shuf_ngrams = extract_ngrams(shuffled, n)
            for ng in observed_ngrams:
                shuffle_counts[ng].append(shuf_ngrams.get(ng, 0))

        # Compute z-scores and p-values
        ngram_list = []
        p_values = []

        for ng in observed_ngrams:
            obs = original_counts[ng]
            shuf_vals = np.array(shuffle_counts[ng], dtype=np.float64)
            mean_shuf = shuf_vals.mean()
            std_shuf = shuf_vals.std()

            if std_shuf == 0:
                z = 0.0
                p = 1.0
            else:
                z = (obs - mean_shuf) / std_shuf
                p = float(1.0 - stats.norm.cdf(z))  # one-sided

            ngram_list.append(IdiomResult(
                ngram=ng,
                observed_count=obs,
                expected_count=float(mean_shuf),
                z_score=float(z),
                p_value=p,
                n=n,
            ))
            p_values.append(p)

        # Benjamini-Hochberg FDR correction
        if not p_values:
            continue

        p_arr = np.array(p_values)
        significant = _benjamini_hochberg(p_arr, alpha=fdr_alpha)

        for i, is_sig in enumerate(significant):
            if is_sig:
                all_results.append(ngram_list[i])

    # Sort by z-score descending
    all_results.sort(key=lambda r: r.z_score, reverse=True)
    return all_results


# ---------------------------------------------------------------------------
# Temporal dynamics
# ---------------------------------------------------------------------------


def burstiness_coefficient(event_times: np.ndarray) -> BurstinessResult:
    """Compute burstiness from inter-event intervals.

    CV (coefficient of variation) of IEIs determines temporal pattern:
    - CV < 0.8: periodic (more regular than Poisson)
    - 0.8 <= CV <= 1.2: Poisson-like (memoryless)
    - CV > 1.2: bursty (clustered events)

    Simplified burst detection: events within mean_iei/2 of each other
    are grouped into bursts.

    Note: This is NOT the full Kleinberg (2003) dynamic programming
    approach.  It correctly identifies IEI clusters but does not model
    hierarchical burst structure.

    Parameters
    ----------
    event_times : ndarray
        1D float array of event timestamps (need not be sorted).

    Returns
    -------
    BurstinessResult
        CV, IEI statistics, burst count, interpretation.
    """
    if len(event_times) < 2:
        return BurstinessResult(
            cv=0.0, mean_iei=0.0, std_iei=0.0,
            n_bursts=0, mean_burst_duration=0.0,
            mean_inter_burst_interval=0.0,
            interpretation="insufficient_data",
        )

    sorted_times = np.sort(event_times)
    iei = np.diff(sorted_times)

    if len(iei) == 0:
        return BurstinessResult(
            cv=0.0, mean_iei=0.0, std_iei=0.0,
            n_bursts=0, mean_burst_duration=0.0,
            mean_inter_burst_interval=0.0,
            interpretation="insufficient_data",
        )

    mean_iei = float(np.mean(iei))
    std_iei = float(np.std(iei))
    cv = std_iei / mean_iei if mean_iei > 0 else 0.0

    # Interpretation
    if cv < 0.8:
        interpretation = "periodic"
    elif cv <= 1.2:
        interpretation = "poisson"
    else:
        interpretation = "bursty"

    # Simplified burst detection: threshold = mean_iei / 2
    threshold = mean_iei / 2.0
    if threshold <= 0:
        return BurstinessResult(
            cv=float(cv), mean_iei=mean_iei, std_iei=std_iei,
            n_bursts=0, mean_burst_duration=0.0,
            mean_inter_burst_interval=0.0,
            interpretation=interpretation,
        )

    # Find runs of short IEIs (burst regions)
    is_short = iei < threshold
    bursts: list[tuple[int, int]] = []  # (start_idx, end_idx) in sorted_times
    i = 0
    while i < len(is_short):
        if is_short[i]:
            start = i
            while i < len(is_short) and is_short[i]:
                i += 1
            # Burst spans sorted_times[start] to sorted_times[i]
            bursts.append((start, i))
        else:
            i += 1

    n_bursts = len(bursts)
    if n_bursts == 0:
        return BurstinessResult(
            cv=float(cv), mean_iei=mean_iei, std_iei=std_iei,
            n_bursts=0, mean_burst_duration=0.0,
            mean_inter_burst_interval=0.0,
            interpretation=interpretation,
        )

    burst_durations = [
        sorted_times[end] - sorted_times[start]
        for start, end in bursts
    ]
    mean_burst_duration = float(np.mean(burst_durations))

    # Inter-burst intervals
    if n_bursts > 1:
        inter_burst = [
            sorted_times[bursts[j + 1][0]] - sorted_times[bursts[j][1]]
            for j in range(n_bursts - 1)
        ]
        mean_inter_burst = float(np.mean(inter_burst))
    else:
        mean_inter_burst = 0.0

    return BurstinessResult(
        cv=float(cv),
        mean_iei=mean_iei,
        std_iei=std_iei,
        n_bursts=n_bursts,
        mean_burst_duration=mean_burst_duration,
        mean_inter_burst_interval=mean_inter_burst,
        interpretation=interpretation,
    )


def burstiness_by_context(
    event_times: np.ndarray,
    context_labels: np.ndarray,
) -> dict[str, BurstinessResult]:
    """Compute burstiness for each context group.

    Parameters
    ----------
    event_times : ndarray
        1D float array of event timestamps.
    context_labels : ndarray
        1D array of context labels (same length as event_times).

    Returns
    -------
    Dict mapping context label to BurstinessResult.
    """
    results = {}
    unique_labels = np.unique(context_labels)
    for label in unique_labels:
        mask = context_labels == label
        subset = event_times[mask]
        key = str(label)
        results[key] = burstiness_coefficient(subset)
    return results

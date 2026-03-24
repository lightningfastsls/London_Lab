"""Statistical comparison framework: real USV sequences vs null model surrogates.

Bridges information_theory.py (metric functions) and null_models.py (surrogate
generators) to produce the core publishable result: a table showing whether
real VQ-VAE code sequences have significantly more structure than expected
under each null hypothesis.

Design:
    - NullModelComparison.compare() computes z-score and rank-based p-value
      for one (metric, null_model) pair.
    - NullModelComparison.full_analysis() iterates all metrics x all null models,
      builds a FullAnalysisResult with plain-language summary.
    - FullAnalysisResult.to_markdown() and .to_dataframe() produce publication-
      ready output (to_dataframe requires pandas, to_markdown does not).

Metrics tested (7 core):
    zipf_alpha, entropy_rate, excess_entropy, mutual_information,
    mi_decay_half_life, bigram_productivity, n_significant_idioms.

burstiness_cv is excluded from the comparison matrix because null models
operate on code sequences, not timestamps.  It is reported as a standalone
metric in the summary when event_times are provided.

Dependencies: numpy (all computation), information_theory, null_models,
sequence_analysis (all existing, no new packages).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from usv_language.analysis.null_models import NullModelConfig, NullModelGenerator
from usv_language.analysis import information_theory
from usv_language.analysis import sequence_analysis


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetricComparison:
    """Result of comparing one metric between real data and null surrogates.

    Attributes
    ----------
    metric_name : str
        Which information-theoretic metric was compared.
    null_model : str
        Which null model surrogates were used (e.g. "shuffled", "markov_1").
    real_value : float
        Metric value on the real sequence.
    null_mean : float
        Mean metric value across surrogates.
    null_std : float
        Sample standard deviation across surrogates (ddof=1).
    z_score : float
        (real - null_mean) / null_std; 0.0 when null_std == 0.
    rank_p_value : float
        Fraction of surrogates with metric >= real value (one-sided,
        conservative: includes equals).
    effect_size : float
        Standardized distance from null (identical to z_score in the
        single-observation design; analogous to Cohen's d).
    n_surrogates : int
        Number of surrogates used.
    significant : bool
        True when |z_score| > 2 OR rank_p_value < 0.05.
    """

    metric_name: str
    null_model: str
    real_value: float
    null_mean: float
    null_std: float
    z_score: float
    rank_p_value: float
    effect_size: float
    n_surrogates: int
    significant: bool


@dataclass(frozen=True)
class FullAnalysisResult:
    """Complete null model x metric comparison table.

    Attributes
    ----------
    comparisons : list[MetricComparison]
        All (metric, null_model) comparisons.
    metrics_used : list[str]
        Names of metrics in the comparison.
    null_models_used : list[str]
        Names of null models in the comparison.
    sequence_length : int
        Length of the real sequence.
    codebook_size : int
        Number of discrete codes (K).
    summary : str
        Plain-language interpretation of results.
    """

    comparisons: list[MetricComparison]
    metrics_used: list[str]
    null_models_used: list[str]
    sequence_length: int
    codebook_size: int
    summary: str

    def to_dataframe(self) -> "pandas.DataFrame":
        """Pivot to pandas DataFrame (rows=null models, cols=metrics, cells=z-scores).

        Raises
        ------
        ImportError
            If pandas is not installed.
        """
        try:
            import pandas as pd
        except ImportError:
            raise ImportError(
                "pandas is required for to_dataframe(). "
                "Install with: pip install pandas"
            )

        rows: list[dict[str, object]] = []
        for c in self.comparisons:
            rows.append({
                "null_model": c.null_model,
                "metric": c.metric_name,
                "z_score": c.z_score,
                "rank_p": c.rank_p_value,
                "significant": c.significant,
            })

        df = pd.DataFrame(rows)
        if df.empty:
            return df

        # Pivot: rows = null_model, columns = metric, values = z_score
        pivot = df.pivot_table(
            index="null_model",
            columns="metric",
            values="z_score",
            aggfunc="first",
        )
        return pivot

    def to_markdown(self) -> str:
        """Format as publishable markdown table (no pandas dependency).

        Layout: rows = null models, columns = metrics.
        Each cell shows z-score with significance marker (*).
        """
        if not self.comparisons:
            return "No comparisons to display."

        metrics = self.metrics_used
        models = self.null_models_used

        # Build lookup: (null_model, metric_name) -> MetricComparison
        lookup: dict[tuple[str, str], MetricComparison] = {}
        for c in self.comparisons:
            lookup[(c.null_model, c.metric_name)] = c

        # Header row
        header = "| Null Model | " + " | ".join(metrics) + " |"
        separator = "|" + "---|" * (len(metrics) + 1)

        rows = [header, separator]
        for model in models:
            cells = [model]
            for metric in metrics:
                key = (model, metric)
                if key in lookup:
                    c = lookup[key]
                    marker = "*" if c.significant else ""
                    cells.append(f"{c.z_score:+.2f}{marker}")
                else:
                    cells.append("--")
            rows.append("| " + " | ".join(cells) + " |")

        # Legend
        rows.append("")
        rows.append("\\* significant (|z| > 2 or rank p < 0.05)")

        return "\n".join(rows)


# ---------------------------------------------------------------------------
# Core comparison engine
# ---------------------------------------------------------------------------

# The 7 metrics we always compare in the null model framework.
CORE_METRICS = (
    "zipf_alpha",
    "entropy_rate",
    "excess_entropy",
    "mutual_information",
    "mi_decay_half_life",
    "bigram_productivity",
    "n_significant_idioms",
)

# Scientific hierarchy from weakest to strongest null hypothesis.
# Used to order rows in the comparison table for interpretability.
_NULL_MODEL_ORDER = [
    "shuffled", "markov_1", "markov_2", "markov_3",
    "renewal", "hmm", "phase_randomized",
]


class NullModelComparison:
    """Compare information-theoretic metrics between real and null sequences.

    Usage
    -----
    >>> comp = NullModelComparison()
    >>> result = comp.full_analysis(real_sequence, K=64)
    >>> print(result.to_markdown())
    """

    # -----------------------------------------------------------------
    # Single comparison
    # -----------------------------------------------------------------

    @staticmethod
    def compare(
        real_value: float,
        null_values: np.ndarray | list[float],
        metric_name: str,
        null_model_name: str,
    ) -> MetricComparison:
        """Compare a single metric between real data and null surrogates.

        Parameters
        ----------
        real_value : float
            Metric computed on the real sequence.
        null_values : array-like
            Metric computed on each surrogate.
        metric_name : str
            Name of the metric.
        null_model_name : str
            Name of the null model.

        Returns
        -------
        MetricComparison

        Raises
        ------
        ValueError
            If null_values is empty.
        """
        null_values = np.asarray(null_values, dtype=np.float64)
        if len(null_values) == 0:
            raise ValueError("null_values must not be empty")

        # Filter NaNs
        valid = null_values[~np.isnan(null_values)]
        n_surrogates = len(valid)

        if n_surrogates == 0:
            # All surrogates produced NaN — can't compare
            return MetricComparison(
                metric_name=metric_name,
                null_model=null_model_name,
                real_value=real_value,
                null_mean=float("nan"),
                null_std=float("nan"),
                z_score=float("nan"),
                rank_p_value=float("nan"),
                effect_size=float("nan"),
                n_surrogates=0,
                significant=False,
            )

        null_mean = float(np.mean(valid))
        # Bessel's correction (ddof=1) for sample std
        null_std = float(np.std(valid, ddof=1)) if n_surrogates > 1 else 0.0

        # z-score
        if null_std > 0:
            z_score = (real_value - null_mean) / null_std
        else:
            z_score = 0.0

        # NaN real_value propagation
        if np.isnan(real_value):
            z_score = float("nan")

        # Rank-based p-value: fraction of surrogates >= real value
        # (one-sided, conservative: includes equals)
        if np.isnan(real_value):
            rank_p_value = float("nan")
        else:
            rank_p_value = float(np.sum(valid >= real_value) / n_surrogates)

        # Cohen's d = z_score for this design (single observation vs population)
        effect_size = z_score

        # Significance
        if np.isnan(z_score) or np.isnan(rank_p_value):
            significant = False
        else:
            significant = abs(z_score) > 2 or rank_p_value < 0.05

        return MetricComparison(
            metric_name=metric_name,
            null_model=null_model_name,
            real_value=real_value,
            null_mean=null_mean,
            null_std=null_std,
            z_score=z_score,
            rank_p_value=rank_p_value,
            effect_size=effect_size,
            n_surrogates=n_surrogates,
            significant=significant,
        )

    # -----------------------------------------------------------------
    # Metric computation helper
    # -----------------------------------------------------------------

    @staticmethod
    def _compute_metric(
        metric_name: str,
        sequence: np.ndarray,
        K: int,
        event_times: Optional[np.ndarray] = None,
    ) -> float:
        """Compute a single metric on a code sequence.

        Parameters
        ----------
        metric_name : str
            One of the CORE_METRICS or "burstiness_cv".
        sequence : np.ndarray
            1D integer array of VQ-VAE code indices.
        K : int
            Codebook size (alphabet size).
        event_times : np.ndarray, optional
            Event timestamps (only used for burstiness_cv).

        Returns
        -------
        float
            The metric value.

        Raises
        ------
        ValueError
            If metric_name is unknown.
        """
        if metric_name == "zipf_alpha":
            result = information_theory.zipf_exponent_mle(sequence)
            return result.rank_alpha

        elif metric_name == "entropy_rate":
            result = information_theory.entropy_rate(sequence)
            if result.rates_corrected:
                return result.rates_corrected[-1]
            return 0.0

        elif metric_name == "excess_entropy":
            return sequence_analysis.excess_entropy(sequence, K)

        elif metric_name == "mutual_information":
            return sequence_analysis.mutual_information_at_lag(sequence, K, 1)

        elif metric_name == "mi_decay_half_life":
            mi_values = information_theory.mutual_information_rate(sequence, K, 20)
            if not mi_values or mi_values[0] <= 0:
                return 0.0
            threshold = 0.5 * mi_values[0]
            for lag_idx, mi_val in enumerate(mi_values):
                if mi_val <= threshold:
                    return float(lag_idx + 1)  # 1-indexed lag
            return float(len(mi_values))  # never decayed below half

        elif metric_name == "bigram_productivity":
            result = information_theory.ngram_productivity(
                sequence, K, n=2, n_bootstrap=1,
            )
            return result.productivity_ratio

        elif metric_name == "n_significant_idioms":
            result = information_theory.ngram_idioms(
                sequence, K, max_n=4, n_shuffles=50,
            )
            return float(len(result))

        elif metric_name == "burstiness_cv":
            if event_times is None:
                return float("nan")
            result = information_theory.burstiness_coefficient(event_times)
            return result.cv

        else:
            raise ValueError(f"Unknown metric: {metric_name!r}")

    # -----------------------------------------------------------------
    # Full analysis
    # -----------------------------------------------------------------

    def full_analysis(
        self,
        real_sequence: np.ndarray,
        K: int,
        event_times: Optional[np.ndarray] = None,
        null_config: Optional[NullModelConfig] = None,
    ) -> FullAnalysisResult:
        """Run all metrics against all null models.

        Parameters
        ----------
        real_sequence : np.ndarray
            1D integer array of VQ-VAE code indices.
        K : int
            Codebook size (alphabet size).
        event_times : np.ndarray, optional
            Event timestamps for burstiness (reported standalone, not compared).
        null_config : NullModelConfig, optional
            Surrogate generation configuration.  Defaults to NullModelConfig().

        Returns
        -------
        FullAnalysisResult

        Raises
        ------
        ValueError
            If real_sequence is empty or K <= 0.
        """
        # Validate inputs
        real_sequence = np.asarray(real_sequence)
        if real_sequence.ndim != 1 or len(real_sequence) == 0:
            raise ValueError(
                "real_sequence must be a non-empty 1D integer array"
            )
        if K <= 0:
            raise ValueError(f"K must be positive, got {K}")

        if null_config is None:
            null_config = NullModelConfig()

        metrics = list(CORE_METRICS)

        # 1. Compute real metric values
        real_values: dict[str, float] = {}
        for metric in metrics:
            try:
                real_values[metric] = self._compute_metric(
                    metric, real_sequence, K,
                )
            except Exception as e:
                warnings.warn(
                    f"Metric {metric!r} failed on real sequence: {e}",
                    RuntimeWarning,
                    stacklevel=2,
                )
                real_values[metric] = float("nan")

        # 2. Generate all surrogates
        generator = NullModelGenerator(null_config)
        all_surrogates = generator.generate_all(real_sequence)

        # 3. Compute metrics on surrogates and compare
        comparisons: list[MetricComparison] = []
        null_models_used: list[str] = sorted(
            all_surrogates.keys(),
            key=lambda m: (
                _NULL_MODEL_ORDER.index(m) if m in _NULL_MODEL_ORDER else 99
            ),
        )

        for null_model_name in null_models_used:
            surrogate_list = all_surrogates[null_model_name]

            for metric in metrics:
                null_metric_values: list[float] = []

                for surr in surrogate_list:
                    try:
                        val = self._compute_metric(metric, surr, K)
                        null_metric_values.append(val)
                    except Exception:
                        null_metric_values.append(float("nan"))

                if not null_metric_values:
                    continue

                comparison = self.compare(
                    real_value=real_values[metric],
                    null_values=null_metric_values,
                    metric_name=metric,
                    null_model_name=null_model_name,
                )
                if comparison.n_surrogates == 0:
                    warnings.warn(
                        f"All surrogates produced NaN for "
                        f"({metric!r}, {null_model_name!r}). "
                        f"Check sequence length and K.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                comparisons.append(comparison)

        # 4. Build summary
        summary = self._build_summary(
            comparisons=comparisons,
            metrics=metrics,
            null_models=null_models_used,
            sequence_length=len(real_sequence),
            K=K,
            n_surrogates=null_config.n_surrogates,
            real_values=real_values,
            event_times=event_times,
        )

        return FullAnalysisResult(
            comparisons=comparisons,
            metrics_used=metrics,
            null_models_used=null_models_used,
            sequence_length=len(real_sequence),
            codebook_size=K,
            summary=summary,
        )

    # -----------------------------------------------------------------
    # Summary generation
    # -----------------------------------------------------------------

    def _build_summary(
        self,
        comparisons: list[MetricComparison],
        metrics: list[str],
        null_models: list[str],
        sequence_length: int,
        K: int,
        n_surrogates: int,
        real_values: dict[str, float],
        event_times: Optional[np.ndarray] = None,
    ) -> str:
        """Build a plain-language interpretation of the results."""
        lines: list[str] = []
        lines.append("=== Null Model Comparison Summary ===")
        lines.append(f"Sequence: {sequence_length} tokens, codebook size {K}")
        lines.append(f"Surrogates: {n_surrogates} per null model")
        lines.append("")

        # Build lookup
        lookup: dict[tuple[str, str], MetricComparison] = {}
        for c in comparisons:
            lookup[(c.metric_name, c.null_model)] = c

        # Per-metric interpretation
        sig_metrics: list[str] = []
        nonsig_metrics: list[str] = []

        for metric in metrics:
            sig_models: list[str] = []
            nonsig_models: list[str] = []
            for model in null_models:
                key = (metric, model)
                if key in lookup:
                    c = lookup[key]
                    if c.significant:
                        sig_models.append(
                            f"{model} (z={c.z_score:+.1f}, p={c.rank_p_value:.2f})"
                        )
                    else:
                        nonsig_models.append(
                            f"{model} (z={c.z_score:+.1f}, p={c.rank_p_value:.2f})"
                        )

            if sig_models:
                sig_metrics.append(metric)
                lines.append(f"- {metric} (real={real_values.get(metric, float('nan')):.4f}):")
                lines.append(f"    significant vs: {', '.join(sig_models)}")
                if nonsig_models:
                    lines.append(f"    NOT significant vs: {', '.join(nonsig_models)}")
                lines.append(f"    {self._interpret_metric(metric, sig_models, nonsig_models)}")
                lines.append("")
            else:
                nonsig_metrics.append(metric)

        if nonsig_metrics:
            lines.append("Non-significant metrics (no comparison significant):")
            for metric in nonsig_metrics:
                note = ""
                if metric == "mi_decay_half_life":
                    # Check if all comparisons had null_std == 0 (degenerate)
                    all_degenerate = all(
                        lookup.get((metric, m), None) is not None
                        and lookup[(metric, m)].null_std == 0.0
                        for m in null_models
                        if (metric, m) in lookup
                    )
                    if all_degenerate:
                        note = (
                            " (NOTE: degenerate -- plug-in MI bias floor "
                            "may prevent half-life discrimination at large K)"
                        )
                lines.append(f"  - {metric}: matches all null models{note}")
            lines.append("")

        # Burstiness standalone report
        if event_times is not None:
            try:
                burst = information_theory.burstiness_coefficient(event_times)
                lines.append(f"Burstiness (standalone, not compared to null models):")
                lines.append(
                    f"  CV={burst.cv:.3f} ({burst.interpretation}), "
                    f"n_bursts={burst.n_bursts}"
                )
                lines.append(
                    "  Note: null models operate on code sequences, not timestamps, "
                    "so burstiness is excluded from the comparison matrix."
                )
                lines.append("")
            except Exception:
                pass

        return "\n".join(lines)

    @staticmethod
    def _interpret_metric(
        metric: str,
        sig_models: list[str],
        nonsig_models: list[str],
    ) -> str:
        """Generate a one-line interpretation for a metric's comparison pattern."""
        sig_names = {m.split(" (")[0] for m in sig_models}
        nonsig_names = {m.split(" (")[0] for m in nonsig_models}

        # Common interpretation patterns
        if metric == "entropy_rate":
            if "shuffled" in sig_names and "markov_1" not in sig_names:
                return ("-> Sequential structure exists but is fully captured "
                        "by first-order transitions.")
            elif "shuffled" in sig_names and "markov_1" in sig_names:
                highest_nonsig = None
                for order in [3, 2, 1]:
                    if f"markov_{order}" in nonsig_names:
                        highest_nonsig = order
                        break
                if highest_nonsig:
                    return (f"-> Structure exceeds first-order memory but is "
                            f"captured by order-{highest_nonsig} transitions.")
                return "-> Structure exceeds all tested Markov orders."
            return "-> No sequential structure detected beyond chance."

        elif metric == "excess_entropy":
            if "shuffled" in sig_names:
                for order in [3, 2, 1]:
                    if f"markov_{order}" in nonsig_names:
                        return (f"-> Long-range structure exists but is captured "
                                f"by order-{order} memory.")
                return "-> Long-range dependencies exceed all tested null models."
            return "-> No significant long-range dependencies."

        elif metric == "zipf_alpha":
            if sig_names:
                return "-> Code frequency distribution differs from null models."
            return "-> Code frequency distribution matches null models."

        elif metric == "mutual_information":
            if "shuffled" in sig_names:
                return "-> Adjacent codes are statistically dependent."
            return "-> No adjacent-code dependency detected."

        elif metric == "mi_decay_half_life":
            if sig_names:
                return "-> Temporal correlations persist longer than null models predict."
            return "-> Temporal correlation decay matches null models."

        elif metric == "bigram_productivity":
            if sig_names:
                return "-> Combinatorial diversity differs from null expectation."
            return "-> Combinatorial diversity matches null models."

        elif metric == "n_significant_idioms":
            if sig_names:
                return "-> More recurrent motifs than null models produce."
            return "-> Recurrent motif count matches null models."

        return ""

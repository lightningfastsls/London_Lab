"""Probing framework for transformer hidden state interpretability.

Linear and MLP probes test whether acoustic properties are *linearly
accessible* or merely *decodable* from each transformer layer's hidden
states.  This directly answers **which layer encodes the richest
acoustic information** — the primary input for VQ-VAE layer selection.

Technique reference: Belinkov (2022), "Probing Classifiers: Promises,
Shortcomings, and Advances."

Data flow::

    SpectrogramTransformer  -->  hidden_states[layer]  (N, d_model)
    acoustic_properties.py  -->  labels[property]      (N,)
    ProbingExperiment       -->  cross-validated score
    ProbingAnalysisPipeline -->  (layers x properties) heatmap

Dependencies: numpy, scikit-learn, matplotlib, seaborn (all in
usv_language/requirements.txt).
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

# Matplotlib backend set before pyplot import (matches existing pattern)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_VALID_PROBE_TYPES = frozenset({"linear", "mlp"})

_DEFAULT_REGRESSION = (
    "peak_frequency",
    "spectral_centroid",
    "energy",
    "bout_position",
    "time_since_last_usv",
)

_DEFAULT_CLASSIFICATION = (
    "is_voiced",
    "frequency_direction",
)


@dataclass(frozen=True)
class ProbingConfig:
    """Configuration for the probing experiment suite.

    Parameters
    ----------
    probe_types:
        Which probe architectures to evaluate.
    mlp_hidden_size:
        Hidden layer width for MLP probes.
    n_folds:
        Number of cross-validation folds.
    max_samples:
        Subsample cap (0 = use all data).
    random_seed:
        Seed for reproducibility.
    regression_properties:
        Property names treated as regression targets.
    classification_properties:
        Property names treated as classification targets.
    ridge_alpha:
        Regularization strength for Ridge regression.
    logistic_max_iter:
        Maximum iterations for LogisticRegression.
    mlp_max_iter:
        Maximum iterations for MLP probes.
    """

    probe_types: tuple[str, ...] = ("linear", "mlp")
    mlp_hidden_size: int = 64
    n_folds: int = 5
    max_samples: int = 0
    random_seed: int = 42
    regression_properties: tuple[str, ...] = _DEFAULT_REGRESSION
    classification_properties: tuple[str, ...] = _DEFAULT_CLASSIFICATION
    ridge_alpha: float = 1.0
    logistic_max_iter: int = 1000
    mlp_max_iter: int = 500

    def __post_init__(self) -> None:
        for pt in self.probe_types:
            if pt not in _VALID_PROBE_TYPES:
                raise ValueError(
                    f"Invalid probe_type '{pt}'. "
                    f"Must be one of {sorted(_VALID_PROBE_TYPES)}"
                )
        if self.n_folds < 2:
            raise ValueError(
                f"n_folds must be >= 2, got {self.n_folds}"
            )
        if self.mlp_hidden_size <= 0:
            raise ValueError(
                f"mlp_hidden_size must be > 0, got {self.mlp_hidden_size}"
            )
        if self.ridge_alpha <= 0:
            raise ValueError(
                f"ridge_alpha must be > 0, got {self.ridge_alpha}"
            )


# ---------------------------------------------------------------------------
# Result dataclasses
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ProbingResult:
    """Result of a single probing experiment.

    Attributes
    ----------
    property_name:
        Name of the probed acoustic property.
    layer:
        0-indexed transformer layer.
    probe_type:
        ``'linear'`` or ``'mlp'``.
    task_type:
        ``'regression'`` or ``'classification'``.
    score:
        Mean cross-validated score (R² or accuracy).
    score_std:
        Standard deviation across folds.
    n_samples:
        Number of valid samples used.
    fold_scores:
        Per-fold scores.
    """

    property_name: str
    layer: int
    probe_type: str
    task_type: str
    score: float
    score_std: float
    n_samples: int
    fold_scores: tuple[float, ...]


@dataclass
class ProbingAnalysisResult:
    """Aggregated results from the full probing pipeline.

    Mutable because computed properties depend on the full result set.
    """

    results: List[ProbingResult]
    n_layers: int
    property_names: List[str]
    config: ProbingConfig

    def heatmap_data(self, probe_type: str) -> np.ndarray:
        """Build a (n_layers, n_properties) score matrix.

        Scores are clamped to [0, 1] for heatmap display (R² can be
        negative for terrible fits, but that's visually misleading).

        Parameters
        ----------
        probe_type:
            ``'linear'`` or ``'mlp'``.

        Returns
        -------
        ndarray of shape ``(n_layers, len(property_names))``.
        """
        mat = np.zeros((self.n_layers, len(self.property_names)))
        prop_idx = {name: i for i, name in enumerate(self.property_names)}
        for r in self.results:
            if r.probe_type == probe_type and r.property_name in prop_idx:
                mat[r.layer, prop_idx[r.property_name]] = max(0.0, r.score)
        return mat

    @property
    def best_layer_by_property(self) -> Dict[str, Tuple[int, float]]:
        """For each property, find the layer with the highest score.

        Returns dict mapping property_name -> (layer, score).  Uses the
        first probe_type in config.probe_types for consistency.
        """
        best: Dict[str, Tuple[int, float]] = {}
        target_probe = self.config.probe_types[0]
        for r in self.results:
            if r.probe_type != target_probe:
                continue
            key = r.property_name
            if key not in best or r.score > best[key][1]:
                best[key] = (r.layer, r.score)
        return best

    @property
    def best_overall_layer(self) -> int:
        """Layer with the highest mean score across all properties.

        Uses the first probe_type in config.probe_types.
        """
        mat = self.heatmap_data(self.config.probe_types[0])
        mean_per_layer = mat.mean(axis=1)
        return int(np.argmax(mean_per_layer))

    @property
    def summary(self) -> str:
        """Human-readable summary of probing results."""
        lines = ["Probing Analysis Summary", "=" * 40]
        lines.append(f"Layers: {self.n_layers}")
        lines.append(f"Properties: {len(self.property_names)}")
        lines.append(f"Total experiments: {len(self.results)}")
        lines.append("")

        for pt in self.config.probe_types:
            lines.append(f"--- {pt} probe ---")
            mat = self.heatmap_data(pt)
            mean_per_layer = mat.mean(axis=1)
            best_layer = int(np.argmax(mean_per_layer))
            lines.append(
                f"Best overall layer: {best_layer} "
                f"(mean score: {mean_per_layer[best_layer]:.3f})"
            )
            for j, prop in enumerate(self.property_names):
                best_l = int(np.argmax(mat[:, j]))
                lines.append(
                    f"  {prop}: best at layer {best_l} "
                    f"(score={mat[best_l, j]:.3f})"
                )
            lines.append("")

        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Core experiment runner
# ---------------------------------------------------------------------------

class ProbingExperiment:
    """Run a single probing experiment (one layer, one property, one probe).

    Parameters
    ----------
    config:
        Probing configuration.
    """

    def __init__(self, config: ProbingConfig) -> None:
        self.config = config

    def run(
        self,
        X: np.ndarray,
        y: np.ndarray,
        property_name: str,
        layer: int,
        probe_type: str,
        task_type: str,
    ) -> ProbingResult:
        """Fit and cross-validate a probe.

        Parameters
        ----------
        X:
            Hidden states, shape ``(N, d_model)``.
        y:
            Labels, shape ``(N,)``.
        property_name:
            Name of the target property.
        layer:
            0-indexed layer number.
        probe_type:
            ``'linear'`` or ``'mlp'``.
        task_type:
            ``'regression'`` or ``'classification'``.

        Returns
        -------
        ProbingResult with cross-validated scores.
        """
        # --- Filter invalid labels ---
        X_clean, y_clean = self._filter_labels(X, y, task_type)

        if len(y_clean) < self.config.n_folds:
            return ProbingResult(
                property_name=property_name,
                layer=layer,
                probe_type=probe_type,
                task_type=task_type,
                score=0.0,
                score_std=0.0,
                n_samples=len(y_clean),
                fold_scores=(),
            )

        # --- Encode classification labels ---
        if task_type == "classification":
            y_clean = self._encode_labels(y_clean)

            # Need at least 2 classes for classification
            if len(np.unique(y_clean)) < 2:
                return ProbingResult(
                    property_name=property_name,
                    layer=layer,
                    probe_type=probe_type,
                    task_type=task_type,
                    score=0.0,
                    score_std=0.0,
                    n_samples=len(y_clean),
                    fold_scores=(),
                )

        # --- Subsample ---
        if self.config.max_samples > 0 and len(y_clean) > self.config.max_samples:
            rng = np.random.RandomState(self.config.random_seed)
            idx = rng.choice(len(y_clean), self.config.max_samples, replace=False)
            X_clean = X_clean[idx]
            y_clean = y_clean[idx]

        # --- Build sklearn pipeline ---
        pipeline = self._build_pipeline(probe_type, task_type)

        # --- Cross-validate ---
        from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold

        if task_type == "regression":
            cv = KFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.random_seed,
            )
            scoring = "r2"
        else:
            cv = StratifiedKFold(
                n_splits=self.config.n_folds,
                shuffle=True,
                random_state=self.config.random_seed,
            )
            scoring = "accuracy"

        # Suppress convergence warnings from MLP probes
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=UserWarning)
            try:
                from sklearn.exceptions import ConvergenceWarning
                warnings.filterwarnings("ignore", category=ConvergenceWarning)
            except ImportError:
                pass

            fold_scores = cross_val_score(
                pipeline, X_clean, y_clean, cv=cv, scoring=scoring,
            )

        return ProbingResult(
            property_name=property_name,
            layer=layer,
            probe_type=probe_type,
            task_type=task_type,
            score=float(np.mean(fold_scores)),
            score_std=float(np.std(fold_scores)),
            n_samples=len(y_clean),
            fold_scores=tuple(float(s) for s in fold_scores),
        )

    def _filter_labels(
        self, X: np.ndarray, y: np.ndarray, task_type: str,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Remove invalid labels (NaN, inf, sentinel -1.0 for TSLU)."""
        if task_type == "classification":
            # String/bool labels — no NaN filtering needed
            if y.dtype.kind in ("U", "S", "O", "b"):
                return X, y

        # Numeric: filter NaN, inf, and -1.0 sentinels
        y_float = y.astype(np.float64)
        valid = np.isfinite(y_float) & (y_float != -1.0)
        return X[valid], y[valid]

    def _encode_labels(self, y: np.ndarray) -> np.ndarray:
        """Encode string/bool labels to integers."""
        if y.dtype == bool or y.dtype.kind == "b":
            return y.astype(np.int64)
        if y.dtype.kind in ("U", "S", "O"):
            _, encoded = np.unique(y, return_inverse=True)
            return encoded.astype(np.int64)
        return y.astype(np.int64)

    def _build_pipeline(self, probe_type: str, task_type: str):
        """Construct an sklearn Pipeline with StandardScaler + probe.

        The scaler is inside the pipeline to prevent data leakage:
        it fits only on training folds, never on test data.
        """
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        from sklearn.linear_model import Ridge, LogisticRegression
        from sklearn.neural_network import MLPRegressor, MLPClassifier

        if probe_type == "linear":
            if task_type == "regression":
                estimator = Ridge(alpha=self.config.ridge_alpha)
            else:
                estimator = LogisticRegression(
                    max_iter=self.config.logistic_max_iter,
                    solver="lbfgs",
                    random_state=self.config.random_seed,
                )
        else:  # mlp
            hidden = (self.config.mlp_hidden_size,)
            if task_type == "regression":
                estimator = MLPRegressor(
                    hidden_layer_sizes=hidden,
                    max_iter=self.config.mlp_max_iter,
                    random_state=self.config.random_seed,
                )
            else:
                estimator = MLPClassifier(
                    hidden_layer_sizes=hidden,
                    max_iter=self.config.mlp_max_iter,
                    random_state=self.config.random_seed,
                )

        return Pipeline([
            ("scaler", StandardScaler()),
            ("probe", estimator),
        ])


# ---------------------------------------------------------------------------
# Orchestration pipeline
# ---------------------------------------------------------------------------

class ProbingAnalysisPipeline:
    """Run probing experiments across all layers, properties, and probe types.

    Parameters
    ----------
    config:
        Probing configuration.
    """

    def __init__(self, config: Optional[ProbingConfig] = None) -> None:
        self.config = config or ProbingConfig()

    def run(
        self,
        hidden_states: List[np.ndarray],
        properties: Dict[str, np.ndarray],
    ) -> ProbingAnalysisResult:
        """Execute the full probing analysis.

        This is **frame-level probing** — each element of ``hidden_states``
        and each value in ``properties`` must have the same first dimension
        N (total frame count).  No temporal pooling is applied; every
        spectrogram frame is an independent data point.

        Parameters
        ----------
        hidden_states:
            List of arrays, one per layer.  Each has shape ``(N, d_model)``.
        properties:
            Dict mapping property names to label arrays of length ``N``.

        Returns
        -------
        ProbingAnalysisResult aggregating all experiments.
        """
        n_layers = len(hidden_states)
        experiment = ProbingExperiment(self.config)

        # Intersect requested properties with available ones
        all_requested = set(
            self.config.regression_properties
            + self.config.classification_properties
        )
        available = set(properties.keys())
        missing = all_requested - available
        if missing:
            warnings.warn(
                f"Missing properties (will be skipped): {sorted(missing)}",
                stacklevel=2,
            )

        # Build ordered list of (property_name, task_type)
        targets: List[Tuple[str, str]] = []
        for prop in self.config.regression_properties:
            if prop in available:
                targets.append((prop, "regression"))
        for prop in self.config.classification_properties:
            if prop in available:
                targets.append((prop, "classification"))

        property_names = [t[0] for t in targets]

        results: List[ProbingResult] = []
        for layer_idx in range(n_layers):
            X = hidden_states[layer_idx]
            for prop_name, task_type in targets:
                y = properties[prop_name]
                for probe_type in self.config.probe_types:
                    result = experiment.run(
                        X, y, prop_name, layer_idx, probe_type, task_type,
                    )
                    results.append(result)

        return ProbingAnalysisResult(
            results=results,
            n_layers=n_layers,
            property_names=property_names,
            config=self.config,
        )


# ---------------------------------------------------------------------------
# Visualization
# ---------------------------------------------------------------------------

def plot_probing_heatmap(
    analysis: ProbingAnalysisResult,
    probe_type: str,
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot a layers x properties heatmap of probing scores.

    Parameters
    ----------
    analysis:
        Completed probing analysis.
    probe_type:
        ``'linear'`` or ``'mlp'``.
    output_path:
        If provided, save figure to this path.
    dpi:
        Figure resolution.

    Returns
    -------
    The matplotlib Figure object.
    """
    mat = analysis.heatmap_data(probe_type)
    fig, ax = plt.subplots(
        figsize=(max(8, len(analysis.property_names) * 1.2), max(4, analysis.n_layers * 0.6)),
    )

    try:
        import seaborn as sns
        sns.heatmap(
            mat,
            ax=ax,
            annot=True,
            fmt=".2f",
            cmap="YlOrRd",
            vmin=0,
            vmax=1,
            xticklabels=analysis.property_names,
            yticklabels=[f"Layer {i}" for i in range(analysis.n_layers)],
            cbar_kws={"label": "Score (R² / Accuracy)"},
        )
    except ImportError:
        # Fallback to plain matplotlib if seaborn is not installed
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
        ax.set_xticks(range(len(analysis.property_names)))
        ax.set_xticklabels(analysis.property_names, rotation=45, ha="right")
        ax.set_yticks(range(analysis.n_layers))
        ax.set_yticklabels([f"Layer {i}" for i in range(analysis.n_layers)])
        # Annotate cells
        for i in range(analysis.n_layers):
            for j in range(len(analysis.property_names)):
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, label="Score (R² / Accuracy)")

    ax.set_title(f"Probing Scores — {probe_type} probe")
    ax.set_xlabel("Acoustic Property")
    ax.set_ylabel("Transformer Layer")
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig


def plot_layer_comparison(
    analysis: ProbingAnalysisResult,
    output_path: Optional[str] = None,
    dpi: int = 150,
) -> plt.Figure:
    """Plot mean probe score per layer with error bars.

    One line per probe type, x-axis = layer, y-axis = mean score
    across all properties.

    Parameters
    ----------
    analysis:
        Completed probing analysis.
    output_path:
        If provided, save figure to this path.
    dpi:
        Figure resolution.

    Returns
    -------
    The matplotlib Figure object.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    layers = np.arange(analysis.n_layers)

    for probe_type in analysis.config.probe_types:
        mat = analysis.heatmap_data(probe_type)
        means = mat.mean(axis=1)
        stds = mat.std(axis=1)
        ax.errorbar(
            layers, means, yerr=stds,
            marker="o", capsize=4, label=f"{probe_type} probe",
        )

    ax.set_xlabel("Transformer Layer")
    ax.set_ylabel("Mean Score (R² / Accuracy)")
    ax.set_title("Layer Comparison — Mean Probing Score")
    ax.set_xticks(layers)
    ax.set_xticklabels([str(i) for i in layers])
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")

    return fig

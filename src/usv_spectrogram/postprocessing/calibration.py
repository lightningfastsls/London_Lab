"""Temperature scaling for CNN probability calibration.

Implements Guo et al. (2017) temperature scaling: a single learned
parameter T that divides logits before sigmoid, improving calibration
without changing ROC AUC (the transform is monotonic).

Fit T by minimizing negative log-likelihood on a held-out validation set.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


@dataclass
class TemperatureScaler:
    """Post-hoc temperature scaling for binary classification.

    Divides logits by temperature T before sigmoid to improve calibration.
    T > 1 softens predictions (toward 0.5), T < 1 sharpens them.
    """

    temperature: float = 1.5
    fitted: bool = False
    nll_before: float | None = None
    nll_after: float | None = None

    def __post_init__(self) -> None:
        if self.temperature <= 0:
            raise ValueError(
                f"Temperature must be positive, got {self.temperature}"
            )

    def fit(self, logits: np.ndarray, labels: np.ndarray) -> float:
        """Fit temperature by minimizing NLL on validation data.

        Args:
            logits: Raw model outputs before sigmoid, shape (n,)
            labels: Binary labels (0 or 1), shape (n,)

        Returns:
            Optimal temperature value.

        Raises:
            ValueError: If logits and labels have different shapes.
        """
        if logits.shape != labels.shape:
            raise ValueError(
                f"logits and labels must have the same shape, "
                f"got {logits.shape} and {labels.shape}"
            )

        self.nll_before = _binary_nll(logits, labels, temperature=1.0)

        result = minimize(
            fun=lambda t: _binary_nll(logits, labels, temperature=t[0]),
            x0=[self.temperature],
            method="L-BFGS-B",
            bounds=[(0.01, 50.0)],
        )

        self.temperature = float(result.x[0])
        self.nll_after = _binary_nll(logits, labels, temperature=self.temperature)
        self.fitted = True
        return self.temperature

    def calibrate(self, logits: np.ndarray) -> np.ndarray:
        """Apply temperature scaling and return calibrated probabilities.

        Args:
            logits: Raw model outputs before sigmoid, shape (n,)

        Returns:
            Calibrated probabilities in [0, 1], shape (n,).
        """
        if not self.fitted:
            import warnings
            warnings.warn(
                "calibrate() called before fit() — using default temperature "
                f"({self.temperature}). Call fit() first for calibrated results.",
                UserWarning,
                stacklevel=2,
            )
        scaled = logits / self.temperature
        # Numerically stable sigmoid: avoid overflow in exp() for extreme values
        result = np.empty_like(scaled, dtype=np.float64)
        pos = scaled >= 0
        neg = ~pos
        result[pos] = 1.0 / (1.0 + np.exp(-scaled[pos]))
        result[neg] = np.exp(scaled[neg]) / (1.0 + np.exp(scaled[neg]))
        return result

    def save(self, path: Path) -> None:
        """Save scaler parameters to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "temperature": self.temperature,
            "fitted": self.fitted,
            "nll_before": self.nll_before,
            "nll_after": self.nll_after,
        }
        path.write_text(json.dumps(data, indent=2))

    @classmethod
    def load(cls, path: Path) -> TemperatureScaler:
        """Load scaler from JSON file."""
        data = json.loads(Path(path).read_text())
        scaler = cls(temperature=data["temperature"])
        scaler.fitted = data["fitted"]
        scaler.nll_before = data["nll_before"]
        scaler.nll_after = data["nll_after"]
        return scaler


def compute_ece(
    probabilities: np.ndarray,
    labels: np.ndarray,
    n_bins: int = 15,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Partitions predictions into equal-width confidence bins and computes
    the weighted average of |accuracy - confidence| per bin.

    Args:
        probabilities: Predicted probabilities, shape (n,)
        labels: Binary labels (0 or 1), shape (n,)
        n_bins: Number of equal-width bins.

    Returns:
        ECE value in [0, 1]. Lower is better.
    """
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n_total = len(labels)

    for lo, hi in zip(bin_edges[:-1], bin_edges[1:]):
        mask = (probabilities > lo) & (probabilities <= hi)
        # Include 0.0 in the first bin
        if lo == 0.0:
            mask = mask | (probabilities == 0.0)

        n_bin = mask.sum()
        if n_bin == 0:
            continue

        avg_confidence = probabilities[mask].mean()
        avg_accuracy = labels[mask].mean()
        ece += (n_bin / n_total) * abs(avg_accuracy - avg_confidence)

    return float(ece)


def _binary_nll(
    logits: np.ndarray,
    labels: np.ndarray,
    temperature: float,
) -> float:
    """Numerically stable binary negative log-likelihood.

    Uses the identity: -log σ(z) = log(1 + exp(-z))
    Combined NLL: mean(log(1 + exp(z/T)) - y * z/T)

    Args:
        logits: Raw model outputs, shape (n,)
        labels: Binary labels (0 or 1), shape (n,)
        temperature: Temperature parameter (> 0).

    Returns:
        Mean NLL (scalar).
    """
    z = logits / temperature
    # Stable log(1 + exp(z)): use log-sum-exp trick
    # log(1 + exp(z)) = max(0, z) + log(1 + exp(-|z|))
    nll = np.maximum(z, 0) + np.log1p(np.exp(-np.abs(z))) - labels * z
    return float(nll.mean())

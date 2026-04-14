"""Second-stage logistic regression filter for USV false positive rejection.

Takes EventFeatures extracted from hysteresis-detected events and classifies
them as true USV (True) or false positive (False). Uses a StandardScaler +
LogisticRegression pipeline with balanced class weights to handle the typical
imbalance where noise artifacts outnumber real USVs.

Reference: Clarfeld et al. (2025) — second-stage classifiers catch spectral
artifacts that sustain above hysteresis thresholds with ~85-90% accuracy.
"""

from __future__ import annotations

import dataclasses
import pickle
from pathlib import Path
from typing import List

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .event_features import EventFeatures

# Feature names in the order they appear in the dataclass — used for
# feature importance mapping and array construction.
_FEATURE_NAMES: list[str] = [f.name for f in dataclasses.fields(EventFeatures)]


def _features_to_array(features: List[EventFeatures]) -> np.ndarray:
    """Convert a list of EventFeatures to a (n_samples, n_features) array.

    Uses dataclasses.astuple for speed — field order matches _FEATURE_NAMES.
    """
    if not features:
        return np.empty((0, len(_FEATURE_NAMES)), dtype=np.float64)
    return np.array(
        [dataclasses.astuple(f) for f in features], dtype=np.float64
    )


class FalsePositiveFilter:
    """Second-stage logistic regression filter for USV events.

    Wraps an sklearn Pipeline (StandardScaler → LogisticRegression) and
    provides a typed interface over EventFeatures dataclasses.

    Parameters
    ----------
    excluded_features : set[str] | None
        Feature names to drop before training and prediction. For example,
        ``{"duration_windows"}`` trains a 10-feature model that ignores
        event duration. Default ``None`` uses all 11 features.
    """

    def __init__(self, excluded_features: set[str] | None = None) -> None:
        self.pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(
                class_weight="balanced",
                C=1.0,
                max_iter=1000,
            )),
        ])
        self._fitted = False
        self._constant_label: bool | None = None
        self.excluded_features: set[str] = excluded_features or set()
        # Precompute column mask for feature array slicing
        self._feature_mask: list[int] = [
            i for i, name in enumerate(_FEATURE_NAMES)
            if name not in self.excluded_features
        ]
        self._active_feature_names: list[str] = [
            _FEATURE_NAMES[i] for i in self._feature_mask
        ]

    def _apply_mask(self, X: np.ndarray) -> np.ndarray:
        """Drop columns for excluded features."""
        if not self.excluded_features:
            return X
        return X[:, self._feature_mask]

    def fit(self, features: List[EventFeatures], labels: List[bool]) -> None:
        """Train the filter on labeled event features.

        Args:
            features: List of EventFeatures, one per detected event.
            labels: True = real USV, False = false positive.

        Raises:
            ValueError: If features and labels have different lengths.
        """
        if len(features) != len(labels):
            raise ValueError(
                f"features and labels must have the same length, "
                f"got {len(features)} and {len(labels)}"
            )
        X = self._apply_mask(_features_to_array(features))
        y = np.array(labels, dtype=bool)

        n_classes = len(np.unique(y))
        if n_classes < 2:
            # Degenerate case: all labels are the same class.
            # sklearn requires 2+ classes, so we store the constant label
            # and skip fitting the pipeline.
            self._constant_label = bool(y[0])
            # Still fit the scaler so feature_importances has valid shape
            self.pipeline["scaler"].fit(X)
            # Set dummy coefficients so feature_importances() works
            clf = self.pipeline["classifier"]
            clf.classes_ = np.array([False, True])
            clf.coef_ = np.zeros((1, X.shape[1]))
            clf.intercept_ = np.array([0.0])
        else:
            self._constant_label = None
            self.pipeline.fit(X, y)

        self._fitted = True

    def predict(self, features: List[EventFeatures]) -> List[bool]:
        """Classify events as USV (True) or false positive (False).

        Returns:
            List of Python bool values (not np.bool_).
        """
        if not self._fitted:
            raise RuntimeError("Filter has not been fitted yet — call fit() first")
        if not features:
            return []
        if self._constant_label is not None:
            return [self._constant_label] * len(features)
        X = self._apply_mask(_features_to_array(features))
        raw = self.pipeline.predict(X)
        return [bool(v) for v in raw]

    def predict_proba(self, features: List[EventFeatures]) -> np.ndarray:
        """Return class probabilities, shape (n_samples, 2).

        Column 0 = P(false positive), column 1 = P(real USV).
        sklearn orders classes_ = [False, True] for bool labels.
        """
        if not self._fitted:
            raise RuntimeError("Filter has not been fitted yet — call fit() first")
        if not features:
            return np.empty((0, 2))
        if self._constant_label is not None:
            n = len(features)
            proba = np.zeros((n, 2))
            col = 1 if self._constant_label else 0
            proba[:, col] = 1.0
            return proba
        X = self._apply_mask(_features_to_array(features))
        return self.pipeline.predict_proba(X)

    def feature_importances(self) -> dict[str, float]:
        """Return absolute logistic regression coefficients keyed by feature name.

        Higher absolute value = more influence on the classification decision.
        """
        if not self._fitted:
            raise RuntimeError("Filter has not been fitted yet — call fit() first")
        coefs = self.pipeline["classifier"].coef_[0]
        return {
            name: float(abs(c))
            for name, c in zip(self._active_feature_names, coefs)
        }

    def save(self, path: Path) -> None:
        """Serialize the filter to a pickle file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path) -> FalsePositiveFilter:
        """Deserialize a filter from a pickle file.

        Backward compatible: old pickles without ``excluded_features``
        get an empty set (all features active).
        """
        with open(path, "rb") as f:
            obj = pickle.load(f)
        if not isinstance(obj, cls):
            raise TypeError(
                f"Expected FalsePositiveFilter, got {type(obj).__name__}"
            )
        # Backfill attributes missing from older pickles
        if not hasattr(obj, "excluded_features"):
            obj.excluded_features = set()
            obj._feature_mask = list(range(len(_FEATURE_NAMES)))
            obj._active_feature_names = list(_FEATURE_NAMES)
        return obj

"""Pure-NumPy 2DPCA and (2D)^2PCA for USV spectrogram classification.

This module implements two-dimensional principal component analysis as
introduced by Yang et al. (2004, *IEEE TPAMI* 26(1):131-137) and its
two-directional extension (2D)^2PCA by Zhang & Zhou (2005, *Pattern
Recognition Letters* 26:1131-1137). Everything here is plain NumPy; the
only sklearn usage is confined to the optional SVM/LDA classifier paths.

Why 2DPCA instead of flattened (vector) PCA
--------------------------------------------
Classical PCA (eigenfaces-style) first *vectorizes* each m x n image into
a single length-(m*n) vector and then estimates an (m*n) x (m*n)
covariance matrix. For a spectrogram patch that is, say, 257 freq bins by
234 time frames, that is a ~60,000-dimensional vector and a covariance
matrix with ~3.6 billion entries — both poorly estimated from a few
thousand samples (the small-sample-size problem) and expensive to
eigendecompose.

2DPCA never vectorizes. It treats each image as a *matrix* and builds an
image covariance ("scatter") matrix directly:

    Gt = (1/M) Σ_k (A_k - Abar)^T (A_k - Abar)        # n x n

which is only n x n (here ~234 x 234). Projecting the image onto the top-d
eigenvectors X (n x d) yields a *feature matrix* Y = A·X of shape m x d
rather than a feature vector. This keeps the row structure of the image
intact and is statistically far better conditioned.

Why this suits spectrograms specifically
-----------------------------------------
For our USV patches the convention is **rows = frequency, columns = time**.
The right/column projection Gt = Σ (A-Abar)^T (A-Abar) is an n x n matrix
indexed by *time frames*, so its eigenvectors X capture covariation across
the time axis — i.e. characteristic temporal pitch trajectories — while
each row (a frequency band) is projected through the same temporal basis.
Y = A·X is therefore a per-frequency-band time-signature, which is exactly
the kind of structure that distinguishes USV call shapes.

The row variant (left projection) Gt' = Σ (A-Abar)(A-Abar)^T is m x m,
indexed by *frequency bins*, and compresses the frequency axis. (2D)^2PCA
applies both: C = Z^T · A · X compresses time and frequency simultaneously
into a compact q x d feature matrix.

Public API (frozen)
-------------------
- fit_2dpca(images, *, energy=0.95, n_components=None) -> TwoDPCAModel
- fit_2d2dpca(images, *, energy=0.95, n_components=None,
              n_components_row=None) -> TwoDTwoDPCAModel
- project(A, model) -> Y
- project_bilateral(A, model) -> C
- feature_matrix_distance(Yi, Yj) -> float
- TwoDPCAClassifier(...)  with .fit / .predict
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

__all__ = [
    "TwoDPCAModel",
    "TwoDTwoDPCAModel",
    "fit_2dpca",
    "fit_2d2dpca",
    "project",
    "project_bilateral",
    "feature_matrix_distance",
    "TwoDPCAClassifier",
]


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #
@dataclass
class TwoDPCAModel:
    """Single-direction (column / right) 2DPCA model.

    Attributes
    ----------
    X : (n, d) projection matrix (top-d eigenvectors of the n x n covariance).
    eigenvalues : (n,) eigenvalues in descending order (clipped at 0).
    mean_image : (m, n) training mean image.
    energy_ratio : cumulative eigenvalue energy retained by the d components.
    n_components : number of retained column components, d.
    """

    X: np.ndarray
    eigenvalues: np.ndarray
    mean_image: np.ndarray
    energy_ratio: float
    n_components: int


@dataclass
class TwoDTwoDPCAModel:
    """Two-directional (2D)^2PCA model.

    Attributes
    ----------
    X : (n, d) column projection matrix.
    Z : (m, q) row projection matrix.
    eigenvalues_col : (n,) descending column-covariance eigenvalues.
    eigenvalues_row : (m,) descending row-covariance eigenvalues.
    mean_image : (m, n) training mean image.
    n_components : number of retained column components, d.
    n_components_row : number of retained row components, q.
    energy_ratio_col : cumulative column energy retained.
    energy_ratio_row : cumulative row energy retained.
    """

    X: np.ndarray
    Z: np.ndarray
    eigenvalues_col: np.ndarray
    eigenvalues_row: np.ndarray
    mean_image: np.ndarray
    n_components: int
    n_components_row: int
    energy_ratio_col: float = field(default=0.0)
    energy_ratio_row: float = field(default=0.0)


# --------------------------------------------------------------------------- #
# Internal helpers
# --------------------------------------------------------------------------- #
def _validate_images(images: np.ndarray) -> np.ndarray:
    arr = np.asarray(images, dtype=np.float64)
    if arr.ndim != 3:
        raise ValueError(
            f"images must be a 3-D array (M, m, n); got shape {arr.shape}"
        )
    if arr.shape[0] < 1:
        raise ValueError("images must contain at least one image")
    return arr


def _symmetric_eig(cov: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Eigendecompose a symmetric PSD covariance, sorted descending.

    Uses ``np.linalg.eigh`` (symmetric solver) and clips tiny negative
    eigenvalues (numerical noise) to 0 so cumulative energy is well-defined.

    Returns
    -------
    eigvals : (k,) eigenvalues, descending, clipped at 0.
    eigvecs : (k, k) corresponding eigenvectors as columns, same order.
    """
    # eigh returns ascending eigenvalues for a symmetric matrix.
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, a_min=0.0, a_max=None)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    return eigvals, eigvecs


def _select_n_components(
    eigvals: np.ndarray,
    energy: float,
    n_components: int | None,
) -> tuple[int, float]:
    """Pick d either from an explicit count or by cumulative energy.

    Returns (d, energy_ratio_retained).
    """
    total = float(eigvals.sum())
    n_max = eigvals.shape[0]

    if n_components is not None:
        d = int(n_components)
        if d < 1 or d > n_max:
            raise ValueError(
                f"n_components={d} out of range [1, {n_max}]"
            )
    else:
        if not (0.0 < energy <= 1.0):
            raise ValueError(f"energy must be in (0, 1]; got {energy}")
        if total <= 0.0:
            # Degenerate (all-zero variance) -> keep one component.
            return 1, 0.0
        cum = np.cumsum(eigvals) / total
        # smallest d with cumulative energy >= energy
        d = int(np.searchsorted(cum, energy) + 1)
        d = min(d, n_max)

    retained = float(eigvals[:d].sum() / total) if total > 0.0 else 0.0
    return d, retained


# --------------------------------------------------------------------------- #
# Fitting
# --------------------------------------------------------------------------- #
def fit_2dpca(
    images: np.ndarray,
    *,
    energy: float = 0.95,
    n_components: int | None = None,
) -> TwoDPCAModel:
    """Fit single-direction (column/right) 2DPCA.

    Builds Gt = (1/M) Σ_k (A_k - Abar)^T (A_k - Abar) (n x n) and takes the
    top-d eigenvectors as the projection matrix X (n x d).
    """
    arr = _validate_images(images)
    m_count = arr.shape[0]
    mean_image = arr.mean(axis=0)  # (m, n)

    centered = arr - mean_image  # (M, m, n)
    # Gt = (1/M) Σ C_k^T C_k. Vectorized via einsum over the M axis.
    gt = np.einsum("kij,kil->jl", centered, centered) / m_count  # (n, n)
    gt = (gt + gt.T) / 2.0  # enforce exact symmetry

    eigvals, eigvecs = _symmetric_eig(gt)
    d, retained = _select_n_components(eigvals, energy, n_components)
    X = eigvecs[:, :d]  # (n, d)

    return TwoDPCAModel(
        X=X,
        eigenvalues=eigvals,
        mean_image=mean_image,
        energy_ratio=retained,
        n_components=d,
    )


def fit_2d2dpca(
    images: np.ndarray,
    *,
    energy: float = 0.95,
    n_components: int | None = None,
    n_components_row: int | None = None,
) -> TwoDTwoDPCAModel:
    """Fit two-directional (2D)^2PCA.

    Column covariance Gt = (1/M) Σ C_k^T C_k (n x n) -> X (n x d).
    Row covariance    Gt' = (1/M) Σ C_k C_k^T (m x m) -> Z (m x q).
    """
    arr = _validate_images(images)
    m_count = arr.shape[0]
    mean_image = arr.mean(axis=0)  # (m, n)
    centered = arr - mean_image  # (M, m, n)

    # Column (right) covariance, n x n.
    gt_col = np.einsum("kij,kil->jl", centered, centered) / m_count
    gt_col = (gt_col + gt_col.T) / 2.0
    eigvals_col, eigvecs_col = _symmetric_eig(gt_col)
    d, retained_col = _select_n_components(eigvals_col, energy, n_components)
    X = eigvecs_col[:, :d]  # (n, d)

    # Row (left) covariance, m x m.
    gt_row = np.einsum("kij,klj->il", centered, centered) / m_count
    gt_row = (gt_row + gt_row.T) / 2.0
    eigvals_row, eigvecs_row = _symmetric_eig(gt_row)
    q, retained_row = _select_n_components(eigvals_row, energy, n_components_row)
    Z = eigvecs_row[:, :q]  # (m, q)

    return TwoDTwoDPCAModel(
        X=X,
        Z=Z,
        eigenvalues_col=eigvals_col,
        eigenvalues_row=eigvals_row,
        mean_image=mean_image,
        n_components=d,
        n_components_row=q,
        energy_ratio_col=retained_col,
        energy_ratio_row=retained_row,
    )


# --------------------------------------------------------------------------- #
# Projection
# --------------------------------------------------------------------------- #
def project(A: np.ndarray, model: TwoDPCAModel) -> np.ndarray:
    """Project one image A (m x n) -> feature matrix Y (m x d) = A · X.

    Note: following Yang et al., projection uses the raw image A·X (the
    mean is centered only when forming the covariance). Both conventions
    yield the same nearest-neighbour ranking because the mean term cancels
    in pairwise differences.
    """
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D (m, n); got shape {A.shape}")
    if A.shape[1] != model.X.shape[0]:
        raise ValueError(
            f"A has n={A.shape[1]} columns but X expects n={model.X.shape[0]}"
        )
    return A @ model.X  # (m, d)


def project_bilateral(A: np.ndarray, model: TwoDTwoDPCAModel) -> np.ndarray:
    """Project one image A (m x n) -> C (q x d) = Z^T · A · X."""
    A = np.asarray(A, dtype=np.float64)
    if A.ndim != 2:
        raise ValueError(f"A must be 2-D (m, n); got shape {A.shape}")
    if A.shape[0] != model.Z.shape[0]:
        raise ValueError(
            f"A has m={A.shape[0]} rows but Z expects m={model.Z.shape[0]}"
        )
    if A.shape[1] != model.X.shape[0]:
        raise ValueError(
            f"A has n={A.shape[1]} columns but X expects n={model.X.shape[0]}"
        )
    return model.Z.T @ A @ model.X  # (q, d)


# --------------------------------------------------------------------------- #
# Feature-matrix distance (Yang 2004)
# --------------------------------------------------------------------------- #
def feature_matrix_distance(Yi: np.ndarray, Yj: np.ndarray) -> float:
    """d(Yi, Yj) = Σ_k || Yi[:,k] - Yj[:,k] ||_2 over feature columns k.

    The Yang et al. (2004) distance between two feature matrices: for each
    column (principal component) take the Euclidean norm of the difference
    vector, then sum over columns.
    """
    Yi = np.asarray(Yi, dtype=np.float64)
    Yj = np.asarray(Yj, dtype=np.float64)
    if Yi.shape != Yj.shape:
        raise ValueError(
            f"feature matrices must match in shape; got {Yi.shape} vs {Yj.shape}"
        )
    diff = Yi - Yj  # (m, d)
    col_norms = np.sqrt((diff ** 2).sum(axis=0))  # (d,)
    return float(col_norms.sum())


# --------------------------------------------------------------------------- #
# Classifier
# --------------------------------------------------------------------------- #
class TwoDPCAClassifier:
    """2DPCA / (2D)^2PCA classifier with NN, SVM, or LDA back-ends.

    Parameters
    ----------
    variant : {"2dpca", "2d2dpca"}
        Single-direction column 2DPCA, or two-directional (2D)^2PCA.
    classifier : {"nn", "svm", "lda"}
        - "nn": faithful Yang 2004 1-NN using the feature-matrix distance.
        - "svm": flatten feature matrix -> StandardScaler -> LinearSVC.
        - "lda": flatten feature matrix -> StandardScaler -> LDA.
    energy : float, default 0.95
        Cumulative eigenvalue energy threshold for component selection.
    n_components : int | None
        Explicit number of column components (overrides energy if set).
    n_components_row : int | None
        Explicit number of row components for the "2d2dpca" variant.
    """

    def __init__(
        self,
        *,
        variant: str = "2dpca",
        classifier: str = "nn",
        energy: float = 0.95,
        n_components: int | None = None,
        n_components_row: int | None = None,
    ) -> None:
        if variant not in ("2dpca", "2d2dpca"):
            raise ValueError(
                f"variant must be '2dpca' or '2d2dpca'; got {variant!r}"
            )
        if classifier not in ("nn", "svm", "lda"):
            raise ValueError(
                f"classifier must be 'nn', 'svm', or 'lda'; got {classifier!r}"
            )
        self.variant = variant
        self.classifier = classifier
        self.energy = energy
        self.n_components = n_components
        self.n_components_row = n_components_row

        # Populated by .fit()
        self.model_: TwoDPCAModel | TwoDTwoDPCAModel | None = None
        self.train_features_: np.ndarray | None = None  # (N, p, d) stacked
        self.train_labels_: np.ndarray | None = None
        self._scaler = None
        self._estimator = None

    # -- internal projection dispatch -------------------------------------- #
    def _project_one(self, A: np.ndarray) -> np.ndarray:
        if self.variant == "2dpca":
            return project(A, self.model_)
        return project_bilateral(A, self.model_)

    def _project_stack(self, images: np.ndarray) -> np.ndarray:
        """Project a stack (N, m, n) -> features (N, p, d) using einsum.

        p = m for "2dpca", p = q for "2d2dpca".
        """
        images = np.asarray(images, dtype=np.float64)
        if self.variant == "2dpca":
            # Y_k = A_k · X  -> (N, m, d)
            return np.einsum("kij,jd->kid", images, self.model_.X)
        # C_k = Z^T · A_k · X -> (N, q, d)
        zt_a = np.einsum("pi,kij->kpj", self.model_.Z.T, images)  # (N,q,n)
        return np.einsum("kpj,jd->kpd", zt_a, self.model_.X)  # (N,q,d)

    # -- fit ---------------------------------------------------------------- #
    def fit(self, images: np.ndarray, labels: np.ndarray) -> "TwoDPCAClassifier":
        arr = _validate_images(images)
        labels = np.asarray(labels)
        if labels.shape[0] != arr.shape[0]:
            raise ValueError(
                f"labels length {labels.shape[0]} != number of images "
                f"{arr.shape[0]}"
            )

        if self.variant == "2dpca":
            self.model_ = fit_2dpca(
                arr, energy=self.energy, n_components=self.n_components
            )
        else:
            self.model_ = fit_2d2dpca(
                arr,
                energy=self.energy,
                n_components=self.n_components,
                n_components_row=self.n_components_row,
            )

        features = self._project_stack(arr)  # (N, p, d)
        self.train_labels_ = labels

        if self.classifier == "nn":
            # Store feature matrices for the Yang-distance 1-NN search.
            self.train_features_ = features
        else:
            # Flatten -> standardize -> sklearn estimator.
            from sklearn.preprocessing import StandardScaler

            flat = features.reshape(features.shape[0], -1)  # (N, p*d)
            self._scaler = StandardScaler()
            flat_std = self._scaler.fit_transform(flat)

            if self.classifier == "svm":
                from sklearn.svm import LinearSVC

                self._estimator = LinearSVC()
            else:  # "lda"
                from sklearn.discriminant_analysis import (
                    LinearDiscriminantAnalysis,
                )

                self._estimator = LinearDiscriminantAnalysis()
            self._estimator.fit(flat_std, labels)

        return self

    # -- predict ------------------------------------------------------------ #
    def predict(self, images: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("classifier is not fitted; call .fit() first")
        arr = _validate_images(images)
        feats = self._project_stack(arr)  # (K, p, d)

        if self.classifier == "nn":
            return self._predict_nn(feats)

        flat = feats.reshape(feats.shape[0], -1)
        flat_std = self._scaler.transform(flat)
        return self._estimator.predict(flat_std)

    def _predict_nn(self, test_feats: np.ndarray) -> np.ndarray:
        """Vectorized Yang-2004 1-NN.

        For each test feature matrix te (p, d) and the stacked train
        features Tr (N, p, d), the Yang distance is

            d = Σ_k sqrt(Σ_i (Tr[:,i,k] - te[i,k])^2)
              = sqrt(((Tr - te)**2).sum(axis=1)).sum(axis=1)

        which is computed without any python loop over train pairs. We loop
        only over the (typically far fewer) test images.
        """
        Tr = self.train_features_  # (N, p, d)
        labels = self.train_labels_
        preds = np.empty(test_feats.shape[0], dtype=labels.dtype)
        for t in range(test_feats.shape[0]):
            te = test_feats[t]  # (p, d)
            diff = Tr - te  # (N, p, d) via broadcasting
            # per-column Euclidean norm, then sum over columns -> (N,)
            dists = np.sqrt((diff ** 2).sum(axis=1)).sum(axis=1)
            preds[t] = labels[int(np.argmin(dists))]
        return preds

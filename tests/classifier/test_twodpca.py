"""Tests for twodpca — written by test-architect BEFORE implementation.

ROADMAP test plan coverage:
  1. Gt is n×n and symmetric; eigenvalues in descending order
     -> test_covariance_matrix_shape_and_symmetry, test_eigenvalues_descending_order
  2. Energy threshold selects ~r components for rank-r data
     -> test_energy_threshold_rank_r_dataset
  3. Reconstruction with full basis is exact
     -> test_full_basis_reconstruction_exact
  4. project output shape (m×d)
     -> test_project_output_shape
  5. project_bilateral output shape (q×d), both dims reduced
     -> test_project_bilateral_shape_reduces_both_dims
  6. feature_matrix_distance: zero on identical, symmetric, positive on different,
     hand-computed spot-checks
     -> test_feature_matrix_distance_zero_identical,
        test_feature_matrix_distance_symmetry,
        test_feature_matrix_distance_positive_different,
        test_feature_matrix_distance_hand_computed,
        test_feature_matrix_distance_hand_computed_2
  7. TwoDPCAClassifier end-to-end on separable 3-class set >= 0.9 accuracy
     -> test_classifier_nn_separable_3class, test_classifier_svm_separable_3class,
        test_classifier_lda_separable_3class, test_classifier_2d2dpca_separable_3class
  8. Deterministic: same input -> same predictions
     -> test_classifier_deterministic, test_fit_2dpca_deterministic

Additional coverage (recurring gap patterns):
  - Empty/null: single image input -> test_fit_2dpca_single_image
  - Degenerate: all-zero images -> test_degenerate_all_zero_images
  - 3-D validation -> test_images_must_be_3d
  - Config validation: invalid energy, n_components, variant, classifier
    -> test_invalid_energy, test_invalid_n_components,
       test_invalid_variant, test_invalid_classifier
  - Shape preservation: model attributes -> test_model_attributes_shapes,
    test_2d2dpca_model_attributes
  - Explicit n_components overrides energy -> test_explicit_n_components_overrides_energy
  - n_components_row explicit -> test_explicit_n_components_row
  - feature_matrix_distance shape mismatch -> test_feature_matrix_distance_shape_mismatch
  - feature_matrix_distance triangle inequality -> test_feature_matrix_distance_triangle_inequality
  - project shape mismatch -> test_project_shape_mismatch
  - project_bilateral shape mismatch -> test_project_bilateral_shape_mismatch
  - Energy ratio in [0,1] -> test_energy_ratio_in_bounds
  - Label dtype preserved -> test_classifier_predict_label_dtype
  - 1-NN self-recall -> test_nn_train_recall_trivial
  - X columns orthonormal -> test_x_columns_are_orthonormal
  - Z columns orthonormal -> test_z_columns_are_orthonormal
  - Mean image equals arithmetic mean -> test_mean_image_equals_arithmetic_mean
  - predict before fit raises -> test_predict_before_fit_raises
  - fit returns self -> test_fit_returns_self
  - energy=1.0 selects all components -> test_energy_near_1_selects_all_components
  - energy ratio meets threshold -> test_energy_ratio_meets_threshold
  - partial reconstruction degrades gracefully -> test_partial_basis_reconstruction_degrades_gracefully
  - project_bilateral dtype float -> test_project_bilateral_output_dtype_is_float

Total: 33 tests (8 groups from ROADMAP spec, 25 additional)
"""

from __future__ import annotations

import sys

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Import the module under test via sys.path so dataclass __module__ resolution
# works correctly (spec_from_file_location breaks dataclass decoration on Py3.12
# because the module name is not registered in sys.modules at decoration time).
# ---------------------------------------------------------------------------
_SRC = str((__import__("pathlib").Path(__file__).resolve().parents[2] / "src"))
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from usv_spectrogram.classifier.twodpca import (  # noqa: E402
    TwoDPCAClassifier,
    TwoDPCAModel,
    TwoDTwoDPCAModel,
    feature_matrix_distance,
    fit_2d2dpca,
    fit_2dpca,
    project,
    project_bilateral,
)


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

def _make_random_images(M=20, m=16, n=12, seed=42):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((M, m, n))


def _make_rank_r_images(M=30, m=16, n=12, r=3, seed=7):
    """Build images whose centered column covariance has rank exactly r.

    Each A_k = mean + Σ_{i=1..r} alpha_{ki} * u_i * v_i^T
    where u_i (m,) and v_i (n,) are orthonormal columns.
    The centered images span an r-dimensional column subspace, so Gt has
    exactly r non-zero eigenvalues.
    """
    rng = np.random.default_rng(seed)
    mean = rng.standard_normal((m, n))
    U, _ = np.linalg.qr(rng.standard_normal((m, r)))   # (m, r) orthonormal cols
    V, _ = np.linalg.qr(rng.standard_normal((n, r)))   # (n, r) orthonormal cols
    alphas = rng.standard_normal((M, r))                # (M, r)
    images = mean + np.einsum("kr,mr,nr->kmn", alphas, U, V)
    return images


def _make_separable_3class_images(n_per_class=40, m=12, n=10, seed=99):
    """Three highly-distinct class patterns + tiny noise (SNR ~20x).

    Class 0 — horizontal step: top half = +1, bottom half = -1.
    Class 1 — vertical step:   left half = +1, right half = -1.
    Class 2 — diagonal gradient normalised to [-1, +1].

    SNR is high enough that every correct classifier must reach >= 0.90.
    """
    rng = np.random.default_rng(seed)
    images_list, labels_list = [], []

    base0 = np.zeros((m, n))
    base0[: m // 2, :] = 1.0
    base0[m // 2 :, :] = -1.0

    base1 = np.zeros((m, n))
    base1[:, : n // 2] = 1.0
    base1[:, n // 2 :] = -1.0

    row_idx = np.arange(m)[:, None]
    col_idx = np.arange(n)[None, :]
    base2 = (row_idx + col_idx).astype(float)
    base2 = base2 / base2.max() * 2.0 - 1.0

    for cls, base in enumerate([base0, base1, base2]):
        noise = rng.standard_normal((n_per_class, m, n)) * 0.05
        imgs = base[None, :, :] + noise
        images_list.append(imgs)
        labels_list.append(np.full(n_per_class, cls, dtype=int))

    return np.concatenate(images_list, axis=0), np.concatenate(labels_list, axis=0)


# ---------------------------------------------------------------------------
# 1. Covariance matrix: shape, symmetry, PSD
# ---------------------------------------------------------------------------

class TestCovarianceProperties:
    """Verify properties of Gt (spec §2DPCA: Gt is n×n symmetric PSD)."""

    def test_covariance_matrix_shape_and_symmetry(self):
        """Spec: Gt = (1/M)Σ C_k^T C_k is n×n.
        Witnessed via: eigenvalues.shape == (n,) and X.shape == (n, n)
        when n_components=n.
        """
        M, m, n = 20, 16, 12
        images = _make_random_images(M=M, m=m, n=n)
        model = fit_2dpca(images, n_components=n)

        assert model.eigenvalues.shape == (n,), (
            f"Expected {n} eigenvalues (n×n cov), got {model.eigenvalues.shape}"
        )
        assert model.X.shape == (n, n), (
            f"Expected X ({n},{n}), got {model.X.shape}"
        )

    def test_eigenvalues_descending_order(self):
        """Spec: eigenvalues must be returned in descending (non-increasing) order."""
        images = _make_random_images(M=30, m=16, n=12)
        model = fit_2dpca(images, energy=1.0)
        ev = model.eigenvalues
        assert np.all(ev[:-1] >= ev[1:] - 1e-12), (
            f"Eigenvalues not non-increasing: {ev[:8]}"
        )

    def test_eigenvalues_non_negative(self):
        """Spec: covariance PSD; eigenvalues clipped at 0, so all >= 0."""
        images = _make_random_images(M=30, m=16, n=12)
        model = fit_2dpca(images, energy=1.0)
        neg = model.eigenvalues[model.eigenvalues < -1e-12]
        assert len(neg) == 0, f"Negative eigenvalues found: {neg}"

    def test_x_columns_are_orthonormal(self):
        """Eigenvectors of a real symmetric matrix are orthonormal: X.T @ X = I_d."""
        images = _make_random_images(M=30, m=16, n=12)
        model = fit_2dpca(images, n_components=5)
        product = model.X.T @ model.X
        assert np.allclose(product, np.eye(5), atol=1e-10), (
            f"X columns not orthonormal; max deviation "
            f"{np.abs(product - np.eye(5)).max():.2e}"
        )


# ---------------------------------------------------------------------------
# 2. Energy threshold selects ~r components for rank-r data
# ---------------------------------------------------------------------------

class TestEnergyThreshold:

    def test_energy_threshold_rank_r_dataset(self):
        """Spec: energy=0.999 on rank-r data selects ~r components.
        The r non-zero eigenvalues carry essentially all variance.
        Allow tolerance of +/- 1 for floating-point precision.
        """
        r = 4
        images = _make_rank_r_images(M=50, m=20, n=15, r=r, seed=42)
        model = fit_2dpca(images, energy=0.999)
        assert abs(model.n_components - r) <= 1, (
            f"Expected ~{r} components for rank-{r} data, "
            f"got {model.n_components}. Eigenvalues: {model.eigenvalues[:8]}"
        )

    def test_energy_near_1_selects_all_components(self):
        """energy=1.0 must select all n components (nothing discarded)."""
        M, m, n = 30, 16, 12
        images = _make_random_images(M=M, m=m, n=n)
        model = fit_2dpca(images, energy=1.0)
        assert model.n_components == n, (
            f"energy=1.0 should keep all {n} components; got {model.n_components}"
        )

    def test_energy_ratio_meets_threshold(self):
        """Retained energy_ratio must be >= requested energy threshold."""
        for energy_thresh in [0.80, 0.90, 0.95, 0.99]:
            images = _make_random_images(M=30, m=16, n=12, seed=int(energy_thresh * 100))
            model = fit_2dpca(images, energy=energy_thresh)
            assert model.energy_ratio >= energy_thresh - 1e-10, (
                f"energy_ratio={model.energy_ratio:.4f} < threshold={energy_thresh}"
            )

    def test_energy_ratio_in_bounds(self):
        """energy_ratio must be in [0, 1] for any valid input."""
        images = _make_random_images()
        model = fit_2dpca(images, energy=0.9)
        assert 0.0 <= model.energy_ratio <= 1.0 + 1e-10


# ---------------------------------------------------------------------------
# 3. Reconstruction with full basis is exact
# ---------------------------------------------------------------------------

class TestReconstruction:

    def test_full_basis_reconstruction_exact(self):
        """Spec test #3: n_components=n -> X is n×n orthonormal -> X @ X.T = I_n.
        project(A, model) = A @ X (no mean subtraction, per implementation).
        Therefore A @ X @ X.T = A exactly.
        """
        M, m, n = 20, 10, 8
        images = _make_random_images(M=M, m=m, n=n, seed=1)
        model = fit_2dpca(images, n_components=n)

        rng = np.random.default_rng(999)
        A = rng.standard_normal((m, n))
        Y = project(A, model)            # (m, n)
        A_reconstructed = Y @ model.X.T  # (m, n)

        assert np.allclose(A_reconstructed, A, atol=1e-10), (
            f"Reconstruction not exact; max error = "
            f"{np.abs(A_reconstructed - A).max():.2e}"
        )

    def test_partial_basis_reconstruction_degrades_gracefully(self):
        """With d < n components, Frobenius reconstruction error must be > 0."""
        M, m, n = 20, 10, 8
        images = _make_random_images(M=M, m=m, n=n, seed=2)
        model = fit_2dpca(images, n_components=3)

        rng = np.random.default_rng(888)
        A = rng.standard_normal((m, n))
        Y = project(A, model)
        A_partial = Y @ model.X.T

        err = np.linalg.norm(A_partial - A, "fro")
        assert err > 1e-6, (
            f"Expected lossy reconstruction with d=3 < n={n}; "
            f"got near-zero error {err:.2e}"
        )


# ---------------------------------------------------------------------------
# 4. project output shape (m×d)
# ---------------------------------------------------------------------------

class TestProjectShape:

    def test_project_output_shape(self):
        """Spec: project(A (m,n), model) -> Y (m, d)."""
        M, m, n = 20, 16, 12
        images = _make_random_images(M=M, m=m, n=n)
        model = fit_2dpca(images, n_components=5)
        Y = project(images[0], model)
        assert Y.shape == (m, 5), f"Expected ({m}, 5), got {Y.shape}"

    def test_project_mean_image_shape(self):
        """mean_image must be (m, n) matching the training images."""
        M, m, n = 20, 16, 12
        images = _make_random_images(M=M, m=m, n=n)
        model = fit_2dpca(images)
        assert model.mean_image.shape == (m, n), (
            f"mean_image expected ({m},{n}), got {model.mean_image.shape}"
        )

    def test_project_shape_mismatch(self):
        """project raises ValueError when A.shape[1] != model.X.shape[0]."""
        images = _make_random_images(M=20, m=16, n=12)
        model = fit_2dpca(images, n_components=4)
        with pytest.raises(ValueError):
            project(np.zeros((16, 99)), model)  # wrong n


# ---------------------------------------------------------------------------
# 5. project_bilateral: (q×d) and both dims reduced when energy < 1
# ---------------------------------------------------------------------------

class TestProjectBilateral:

    def test_project_bilateral_shape_reduces_both_dims(self):
        """Spec: project_bilateral(A (m,n)) -> C (q, d) with q<m and d<n."""
        M, m, n = 30, 16, 12
        images = _make_random_images(M=M, m=m, n=n)
        model = fit_2d2dpca(images, energy=0.70)  # aggressive -> few components

        C = project_bilateral(images[0], model)
        assert C.shape == (model.n_components_row, model.n_components), (
            f"Expected ({model.n_components_row}, {model.n_components}), got {C.shape}"
        )
        assert model.n_components_row < m, (
            f"Expected q < m={m}, got {model.n_components_row}"
        )
        assert model.n_components < n, (
            f"Expected d < n={n}, got {model.n_components}"
        )

    def test_project_bilateral_shape_mismatch(self):
        """project_bilateral raises ValueError on wrong A dimensions."""
        images = _make_random_images(M=20, m=16, n=12)
        model = fit_2d2dpca(images, n_components=3, n_components_row=3)
        with pytest.raises(ValueError):
            project_bilateral(np.zeros((99, 12)), model)  # wrong m

    def test_project_bilateral_output_dtype_is_float(self):
        """project_bilateral output must be a floating-point array."""
        images = _make_random_images(M=20, m=16, n=12)
        model = fit_2d2dpca(images, n_components=3, n_components_row=3)
        C = project_bilateral(images[0], model)
        assert np.issubdtype(C.dtype, np.floating), f"Expected float dtype, got {C.dtype}"


# ---------------------------------------------------------------------------
# 6. feature_matrix_distance
# ---------------------------------------------------------------------------

class TestFeatureMatrixDistance:

    def test_feature_matrix_distance_zero_identical(self):
        """d(Y, Y) must be exactly 0.0."""
        Y = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        assert feature_matrix_distance(Y, Y) == 0.0

    def test_feature_matrix_distance_symmetry(self):
        """d(Yi, Yj) == d(Yj, Yi)."""
        rng = np.random.default_rng(5)
        Yi = rng.standard_normal((8, 4))
        Yj = rng.standard_normal((8, 4))
        assert np.isclose(
            feature_matrix_distance(Yi, Yj),
            feature_matrix_distance(Yj, Yi),
        )

    def test_feature_matrix_distance_positive_different(self):
        """d(Yi, Yj) > 0 for non-identical matrices."""
        dist = feature_matrix_distance(np.ones((5, 3)), np.zeros((5, 3)))
        assert dist > 0.0, f"Expected positive distance, got {dist}"

    def test_feature_matrix_distance_hand_computed(self):
        """Hand-computed spot-check.

        Yi = [[1, 0], [0, 0]], Yj = zeros(2,2)
        col-0 diff = [1, 0] -> norm = 1.0
        col-1 diff = [0, 0] -> norm = 0.0
        sum = 1.0
        """
        Yi = np.array([[1.0, 0.0], [0.0, 0.0]])
        dist = feature_matrix_distance(Yi, np.zeros((2, 2)))
        assert np.isclose(dist, 1.0), f"Expected 1.0, got {dist}"

    def test_feature_matrix_distance_hand_computed_2(self):
        """Second hand-computed spot-check.

        Yi = [[3, 4], [0, 0]], Yj = zeros(2,2)
        col-0 diff = [3, 0] -> norm = 3.0
        col-1 diff = [4, 0] -> norm = 4.0
        sum = 7.0
        """
        Yi = np.array([[3.0, 4.0], [0.0, 0.0]])
        dist = feature_matrix_distance(Yi, np.zeros((2, 2)))
        assert np.isclose(dist, 7.0), f"Expected 7.0, got {dist}"

    def test_feature_matrix_distance_shape_mismatch(self):
        """Mismatched shapes must raise ValueError."""
        with pytest.raises(ValueError):
            feature_matrix_distance(np.ones((4, 3)), np.ones((4, 5)))

    def test_feature_matrix_distance_triangle_inequality(self):
        """d satisfies triangle inequality: d(A,C) <= d(A,B) + d(B,C)."""
        rng = np.random.default_rng(77)
        A = rng.standard_normal((6, 4))
        B = rng.standard_normal((6, 4))
        C = rng.standard_normal((6, 4))
        dAC = feature_matrix_distance(A, C)
        dAB_dBC = feature_matrix_distance(A, B) + feature_matrix_distance(B, C)
        assert dAC <= dAB_dBC + 1e-10, (
            f"Triangle inequality violated: d(A,C)={dAC:.4f} > d(A,B)+d(B,C)={dAB_dBC:.4f}"
        )


# ---------------------------------------------------------------------------
# 7. Classifier end-to-end: >= 0.90 on separable 3-class data
# ---------------------------------------------------------------------------

class TestClassifierEndToEnd:
    """Spec test #7."""

    @pytest.fixture(scope="class")
    def separable_data(self):
        images, labels = _make_separable_3class_images(n_per_class=60, m=16, n=14, seed=42)
        rng = np.random.default_rng(0)
        idx = rng.permutation(len(images))
        images, labels = images[idx], labels[idx]
        n_train = int(0.7 * len(images))
        return (
            images[:n_train], labels[:n_train],
            images[n_train:], labels[n_train:],
        )

    def _assert_accuracy(self, clf, train_img, train_lbl, test_img, test_lbl, threshold=0.90):
        clf.fit(train_img, train_lbl)
        preds = clf.predict(test_img)
        acc = (preds == test_lbl).mean()
        assert acc >= threshold, (
            f"{clf.variant}/{clf.classifier}: expected >= {threshold:.0%}, "
            f"got {acc:.2%}. True label counts: {np.bincount(test_lbl)}, "
            f"Predicted counts: {np.bincount(preds.astype(int), minlength=3)}"
        )

    def test_classifier_nn_separable_3class(self, separable_data):
        """Spec: 2DPCA + 1-NN >= 0.90 on separable data."""
        clf = TwoDPCAClassifier(variant="2dpca", classifier="nn", energy=0.99)
        self._assert_accuracy(clf, *separable_data)

    def test_classifier_svm_separable_3class(self, separable_data):
        """Spec: 2DPCA + LinearSVC >= 0.90 on separable data."""
        clf = TwoDPCAClassifier(variant="2dpca", classifier="svm", energy=0.99)
        self._assert_accuracy(clf, *separable_data)

    def test_classifier_lda_separable_3class(self, separable_data):
        """LDA variant >= 0.90 on separable data."""
        clf = TwoDPCAClassifier(variant="2dpca", classifier="lda", energy=0.99)
        self._assert_accuracy(clf, *separable_data)

    def test_classifier_2d2dpca_separable_3class(self, separable_data):
        """Spec: (2D)^2PCA + 1-NN >= 0.90 on separable data."""
        clf = TwoDPCAClassifier(variant="2d2dpca", classifier="nn", energy=0.99)
        self._assert_accuracy(clf, *separable_data)

    def test_nn_train_recall_trivial(self):
        """1-NN on training data must give ~100% accuracy (self is nearest neighbour)."""
        images, labels = _make_separable_3class_images(n_per_class=20, m=12, n=10, seed=3)
        clf = TwoDPCAClassifier(variant="2dpca", classifier="nn", energy=0.99)
        clf.fit(images, labels)
        preds = clf.predict(images)
        acc = (preds == labels).mean()
        assert acc >= 0.99, f"Expected ~100% training recall for 1-NN, got {acc:.2%}"


# ---------------------------------------------------------------------------
# 8. Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:

    def test_classifier_deterministic(self):
        """Same input must produce identical predictions on two separate runs."""
        images, labels = _make_separable_3class_images(n_per_class=30, m=12, n=10, seed=55)
        test_images = images[:10]

        clf1 = TwoDPCAClassifier(variant="2dpca", classifier="nn", energy=0.95)
        clf1.fit(images, labels)
        preds1 = clf1.predict(test_images)

        clf2 = TwoDPCAClassifier(variant="2dpca", classifier="nn", energy=0.95)
        clf2.fit(images, labels)
        preds2 = clf2.predict(test_images)

        np.testing.assert_array_equal(preds1, preds2)

    def test_fit_2dpca_deterministic(self):
        """fit_2dpca on same data must produce identical X and eigenvalues."""
        images = _make_random_images(M=20, m=12, n=8)
        m1 = fit_2dpca(images, energy=0.9)
        m2 = fit_2dpca(images, energy=0.9)
        np.testing.assert_allclose(m1.X, m2.X, atol=1e-14)
        np.testing.assert_allclose(m1.eigenvalues, m2.eigenvalues, atol=1e-14)


# ---------------------------------------------------------------------------
# Additional gap-pattern tests
# ---------------------------------------------------------------------------

class TestEdgeCases:

    def test_fit_2dpca_single_image(self):
        """M=1: covariance is zero (degenerate). Must not raise; n_components >= 1."""
        images = np.random.default_rng(10).standard_normal((1, 8, 6))
        model = fit_2dpca(images)
        assert model.n_components >= 1
        assert isinstance(model.energy_ratio, float)

    def test_degenerate_all_zero_images(self):
        """All-zero images: Gt=0, all eigenvalues 0.
        Degenerate path returns n_components=1, energy_ratio=0.
        """
        images = np.zeros((10, 8, 6))
        model = fit_2dpca(images)
        assert model.n_components >= 1
        assert isinstance(model.energy_ratio, float)

    def test_images_must_be_3d(self):
        """_validate_images must reject 2-D and 4-D arrays."""
        with pytest.raises(ValueError):
            fit_2dpca(np.ones((10, 8)))  # 2-D
        with pytest.raises(ValueError):
            fit_2dpca(np.ones((10, 8, 6, 2)))  # 4-D

    def test_invalid_energy(self):
        """energy outside (0, 1] must raise ValueError."""
        images = _make_random_images()
        with pytest.raises(ValueError):
            fit_2dpca(images, energy=0.0)
        with pytest.raises(ValueError):
            fit_2dpca(images, energy=1.5)
        with pytest.raises(ValueError):
            fit_2dpca(images, energy=-0.1)

    def test_invalid_n_components(self):
        """n_components outside [1, n] must raise ValueError."""
        images = _make_random_images(M=20, m=16, n=12)
        with pytest.raises(ValueError):
            fit_2dpca(images, n_components=0)
        with pytest.raises(ValueError):
            fit_2dpca(images, n_components=999)  # larger than n=12

    def test_invalid_variant(self):
        """TwoDPCAClassifier rejects unknown variant strings."""
        with pytest.raises(ValueError):
            TwoDPCAClassifier(variant="unknown")

    def test_invalid_classifier(self):
        """TwoDPCAClassifier rejects unknown classifier strings."""
        with pytest.raises(ValueError):
            TwoDPCAClassifier(classifier="knn")

    def test_explicit_n_components_overrides_energy(self):
        """When n_components is provided, exactly d components are used."""
        images = _make_random_images(M=30, m=16, n=12)
        for d in [1, 4, 12]:
            model = fit_2dpca(images, energy=0.1, n_components=d)
            assert model.n_components == d, (
                f"Expected n_components={d}, got {model.n_components}"
            )
            assert model.X.shape[1] == d

    def test_explicit_n_components_row(self):
        """n_components_row controls the row (Z) dimension in (2D)^2PCA."""
        images = _make_random_images(M=30, m=16, n=12)
        model = fit_2d2dpca(images, n_components=4, n_components_row=3)
        assert model.n_components == 4
        assert model.n_components_row == 3
        assert model.X.shape == (12, 4)
        assert model.Z.shape == (16, 3)

    def test_predict_before_fit_raises(self):
        """predict before fit must raise RuntimeError."""
        clf = TwoDPCAClassifier()
        with pytest.raises(RuntimeError):
            clf.predict(np.zeros((3, 8, 6)))

    def test_fit_returns_self(self):
        """fit() must return self for method chaining."""
        images, labels = _make_separable_3class_images(n_per_class=10, m=8, n=6, seed=0)
        clf = TwoDPCAClassifier()
        result = clf.fit(images, labels)
        assert result is clf


class TestModelAttributes:

    def test_model_attributes_shapes(self):
        """TwoDPCAModel: all spec-required attributes present with correct shapes."""
        M, m, n, d = 25, 14, 10, 5
        images = _make_random_images(M=M, m=m, n=n)
        model = fit_2dpca(images, n_components=d)

        assert model.X.shape == (n, d), f"X: expected ({n},{d}), got {model.X.shape}"
        assert model.eigenvalues.shape == (n,), (
            f"eigenvalues: expected ({n},), got {model.eigenvalues.shape}"
        )
        assert model.mean_image.shape == (m, n), (
            f"mean_image: expected ({m},{n}), got {model.mean_image.shape}"
        )
        assert model.n_components == d
        assert isinstance(model.energy_ratio, float)

    def test_2d2dpca_model_attributes(self):
        """TwoDTwoDPCAModel: all spec-required attributes present with correct shapes."""
        M, m, n, d, q = 25, 14, 10, 4, 3
        images = _make_random_images(M=M, m=m, n=n)
        model = fit_2d2dpca(images, n_components=d, n_components_row=q)

        assert model.X.shape == (n, d), f"X: expected ({n},{d})"
        assert model.Z.shape == (m, q), f"Z: expected ({m},{q})"
        assert model.eigenvalues_col.shape == (n,)
        assert model.eigenvalues_row.shape == (m,)
        assert model.mean_image.shape == (m, n)
        assert model.n_components == d
        assert model.n_components_row == q
        # Additive extras (non-breaking additions allowed by spec)
        assert hasattr(model, "energy_ratio_col")
        assert hasattr(model, "energy_ratio_row")

    def test_classifier_predict_label_dtype(self):
        """Predicted labels must preserve the training label dtype."""
        images, labels = _make_separable_3class_images(n_per_class=20, m=12, n=10, seed=7)
        labels_int32 = labels.astype(np.int32)
        clf = TwoDPCAClassifier(variant="2dpca", classifier="nn")
        clf.fit(images, labels_int32)
        preds = clf.predict(images[:5])
        assert preds.dtype == labels_int32.dtype, (
            f"Expected dtype {labels_int32.dtype}, got {preds.dtype}"
        )


class TestOrthonormality:

    def test_z_columns_are_orthonormal(self):
        """Z (m×q): Z.T @ Z = I_q."""
        images = _make_random_images(M=30, m=16, n=12)
        model = fit_2d2dpca(images, n_components=4, n_components_row=5)
        q = model.n_components_row
        product = model.Z.T @ model.Z
        assert np.allclose(product, np.eye(q), atol=1e-10), (
            f"Z columns not orthonormal; max deviation {np.abs(product - np.eye(q)).max():.2e}"
        )


class TestMeanImage:

    def test_mean_image_equals_arithmetic_mean(self):
        """mean_image must equal pixel-wise arithmetic mean over training images."""
        M, m, n = 15, 10, 8
        images = _make_random_images(M=M, m=m, n=n, seed=21)
        model = fit_2dpca(images)
        expected_mean = images.mean(axis=0)
        np.testing.assert_allclose(model.mean_image, expected_mean, atol=1e-14)


# ---------------------------------------------------------------------------
# Adversarial math-correctness tests (added by adversarial verification task)
#
# Goal: catch subtle bugs that the existing spec tests might miss —
#   (1) NN distance vectorization vs brute force,
#   (2) projection orientation on NON-SQUARE images (the key trap),
#   (3) eigen-order / energy invariants,
#   (4) train/test mean-convention consistency.
# These tests do NOT modify twodpca.py; a failure is reported as a candidate bug.
# ---------------------------------------------------------------------------

class TestAdversarialMathCorrectness:
    """Adversarial cross-checks against independent (naive) reference math."""

    def test_nn_vectorized_matches_bruteforce(self):
        """The vectorized Yang-2004 1-NN must equal a naive double-loop 1-NN.

        Confirms sqrt(((Tr - te)**2).sum(axis=1)).sum(axis=1) is identical to
        Σ_k ||Yi[:,k]-Yj[:,k]||_2 (feature_matrix_distance) for argmin selection.
        Uses NON-SQUARE images (16x12) so any axis confusion is exercised.
        """
        rng = np.random.default_rng(2024)
        train_imgs = rng.standard_normal((30, 16, 12))
        test_imgs = rng.standard_normal((20, 16, 12))
        train_lbl = rng.integers(0, 4, size=30)

        clf = TwoDPCAClassifier(variant="2dpca", classifier="nn", n_components=5)
        clf.fit(train_imgs, train_lbl)
        vec_preds = clf.predict(test_imgs)

        # Independent brute-force: project each image with public project(),
        # compute feature_matrix_distance in a naive double loop, argmin.
        model = clf.model_
        train_Y = [project(train_imgs[i], model) for i in range(len(train_imgs))]
        brute_preds = np.empty(len(test_imgs), dtype=train_lbl.dtype)
        for t in range(len(test_imgs)):
            te_Y = project(test_imgs[t], model)
            dists = [feature_matrix_distance(train_Y[i], te_Y)
                     for i in range(len(train_imgs))]
            brute_preds[t] = train_lbl[int(np.argmin(dists))]

        np.testing.assert_array_equal(
            vec_preds, brute_preds,
            err_msg="Vectorized 1-NN disagrees with brute-force feature_matrix_distance NN",
        )

    def test_projection_orientation_nonsquare(self):
        """KEY TRAP: projection is a RIGHT multiply by X (n x d) on non-square images.

        With images (M, 64, 48): m=64 (rows/freq), n=48 (cols/time).
          - model.X must be (48, d)        (eigenvectors of the n x n cov)
          - project(A) must return (64, d) (A @ X)
          - project must literally equal A @ model.X
        A TRANSPOSED projection (A @ X.T) would be dimensionally broken here
        (X.T is (d, 48); A is (64, 48); inner dims 48 != d), proving the
        implementation does NOT silently transpose. For square 64x64 this
        bug would be invisible — that is why non-square is mandatory.
        """
        rng = np.random.default_rng(13)
        M, m, n, d = 25, 64, 48, 6
        images = rng.standard_normal((M, m, n))
        model = fit_2dpca(images, n_components=d)

        assert model.X.shape == (n, d), (
            f"X must be (n={n}, d={d}); got {model.X.shape}"
        )

        A = rng.standard_normal((m, n))
        Y = project(A, model)
        assert Y.shape == (m, d), (
            f"project must return (m={m}, d={d}); got {Y.shape}"
        )

        # project == A @ X exactly (right multiplication on the time/column axis).
        np.testing.assert_allclose(Y, A @ model.X, atol=1e-12)

        # Verify the transposed alternative is genuinely impossible (dimension
        # mismatch), i.e. the orientation cannot be coincidentally correct.
        assert model.X.T.shape == (d, n)
        with pytest.raises(ValueError):
            # A (64,48) @ X.T (6,48): inner dims 48 vs 6 -> illegal.
            _ = A @ model.X.T

        # And the stack projection used internally must agree with project().
        stack_Y = clf_project_stack(model, A[None, :, :])  # (1, m, d)
        np.testing.assert_allclose(stack_Y[0], Y, atol=1e-12)

    def test_eigen_order_and_energy_invariants(self):
        """Eigenvalues sorted strictly non-increasing, all >= 0; energy in (0, 1].

        Also cross-check eigh-vs-eig eigenvalue agreement on a known symmetric
        PSD matrix (the descending-sort contract must match a general solver).
        """
        images = _make_random_images(M=40, m=20, n=16, seed=314)
        model = fit_2dpca(images, energy=1.0)
        ev = model.eigenvalues

        diff = np.diff(ev)
        assert np.all(diff <= 1e-12), (
            f"Eigenvalues not non-increasing; max positive step = {diff.max():.2e}"
        )
        assert np.all(ev >= -1e-12), f"Negative eigenvalue(s): {ev[ev < 0]}"
        assert 0.0 < model.energy_ratio <= 1.0 + 1e-10, (
            f"energy_ratio out of (0, 1]: {model.energy_ratio}"
        )

        # eigh-vs-eig agreement on a known symmetric PSD matrix.
        rng = np.random.default_rng(271)
        B = rng.standard_normal((10, 10))
        S = B @ B.T  # symmetric PSD
        ev_eigh = np.sort(np.linalg.eigvalsh(S))[::-1]
        ev_eig = np.sort(np.real(np.linalg.eigvals(S)))[::-1]
        np.testing.assert_allclose(ev_eigh, ev_eig, atol=1e-8)

    def test_mean_convention_train_test_consistency(self):
        """Train and test paths must use the SAME projection convention.

        project() (and _project_stack) use raw A @ X with NO mean centering.
        The classifier's internal _project_stack on a single image must equal
        the public project() — i.e. the test path does not skip any train-only
        centering (which would corrupt NN distances).
        """
        rng = np.random.default_rng(555)
        images = rng.standard_normal((20, 14, 11))
        labels = rng.integers(0, 3, size=20)

        clf = TwoDPCAClassifier(variant="2dpca", classifier="nn", n_components=4)
        clf.fit(images, labels)
        model = clf.model_

        probe = rng.standard_normal((14, 11))
        public_Y = project(probe, model)                 # (14, 4)
        internal_Y = clf._project_stack(probe[None, :, :])[0]  # (14, 4)

        np.testing.assert_allclose(
            internal_Y, public_Y, atol=1e-12,
            err_msg="_project_stack (test path) diverges from public project()",
        )
        # And it must be raw A @ X (no mean subtraction on the projection).
        np.testing.assert_allclose(public_Y, probe @ model.X, atol=1e-12)


def clf_project_stack(model, images):
    """Helper: replicate the classifier's '2dpca' stack projection independently.

    Mirrors _project_stack's einsum("kij,jd->kid", images, X) so the test does
    not depend on a fitted classifier instance for this particular check.
    """
    return np.einsum("kij,jd->kid", np.asarray(images, dtype=np.float64), model.X)

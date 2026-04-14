"""Tests for A3 acoustic feature deep-dive analysis.

Uses synthetic data to validate analysis functions without the full dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Ensure scripts/ is importable
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_acoustic_features as afa


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    """Create a synthetic classified DataFrame for testing."""
    rng = np.random.RandomState(42)
    n = 200
    types = rng.choice(afa.TYPE_ORDER, size=n)
    confs = rng.choice(["high", "medium", "low"], size=n, p=[0.5, 0.3, 0.2])

    data = {
        "id": range(n),
        "syllable_type": types,
        "classification_confidence": confs,
        "call_length_s": rng.uniform(0.001, 0.5, n),
        "principal_freq_hz": rng.uniform(30, 110, n),
        "low_freq_hz": rng.uniform(20, 80, n),
        "high_freq_hz": rng.uniform(50, 150, n),
        "bandwidth_hz": rng.uniform(0, 100, n),
        "freq_std_dev_hz": rng.uniform(0, 30, n),
        "slope": rng.uniform(-5000, 5000, n),
        "sinuosity": rng.uniform(1, 10, n),
        "mean_power_db": rng.uniform(-80, -50, n),
        "tonality": rng.uniform(0.15, 0.65, n),
    }
    return pd.DataFrame(data)


@pytest.fixture
def features_and_scaled(synthetic_df):
    """Extract and standardize features from synthetic data."""
    from sklearn.preprocessing import StandardScaler

    features_raw = synthetic_df[afa.ACOUSTIC_FEATURES].copy()
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_raw)
    return features_raw, features_scaled


class TestDataLoading:
    def test_load_and_prepare_from_csv(self, synthetic_df, tmp_path):
        """load_and_prepare reads CSV, drops NaN rows, returns scaled features."""
        csv_path = tmp_path / "test.csv"
        synthetic_df.to_csv(csv_path, index=False)
        df, features_raw, features_scaled = afa.load_and_prepare(csv_path)

        assert len(df) == len(synthetic_df)
        assert features_raw.shape == (len(df), len(afa.ACOUSTIC_FEATURES))
        assert features_scaled.shape == features_raw.shape

    def test_nan_rows_dropped(self, synthetic_df, tmp_path):
        """Rows with NaN in acoustic features are dropped."""
        synthetic_df.loc[0, "slope"] = np.nan
        synthetic_df.loc[1, "tonality"] = np.nan
        csv_path = tmp_path / "test.csv"
        synthetic_df.to_csv(csv_path, index=False)
        df, features_raw, features_scaled = afa.load_and_prepare(csv_path)

        assert len(df) == len(synthetic_df) - 2
        assert not features_raw.isna().any().any()

    def test_unclassified_excluded(self, synthetic_df, tmp_path):
        """Unclassified calls are excluded from analysis."""
        synthetic_df.loc[0, "syllable_type"] = "unclassified"
        csv_path = tmp_path / "test.csv"
        synthetic_df.to_csv(csv_path, index=False)
        df, _, _ = afa.load_and_prepare(csv_path)

        assert "unclassified" not in df["syllable_type"].values


class TestStandardization:
    def test_zero_mean_unit_variance(self, features_and_scaled):
        """Standardized features have zero mean and unit variance."""
        _, features_scaled = features_and_scaled
        means = features_scaled.mean(axis=0)
        stds = features_scaled.std(axis=0)
        np.testing.assert_allclose(means, 0, atol=1e-10)
        np.testing.assert_allclose(stds, 1, atol=0.05)


class TestCorrelation:
    def test_symmetric_with_diagonal_ones(self, features_and_scaled, tmp_path):
        """Correlation matrix is symmetric with 1.0 on diagonal."""
        features_raw, _ = features_and_scaled
        corr = afa.plot_correlation_matrix(features_raw, tmp_path)
        np.testing.assert_allclose(np.diag(corr.values), 1.0)
        np.testing.assert_allclose(corr.values, corr.values.T)

    def test_output_file_created(self, features_and_scaled, tmp_path):
        """Clustermap PNG is saved."""
        features_raw, _ = features_and_scaled
        afa.plot_correlation_matrix(features_raw, tmp_path)
        assert (tmp_path / "correlation_matrix.png").exists()


class TestPCA:
    def test_loadings_shape(self, synthetic_df, features_and_scaled, tmp_path):
        """PCA loadings have shape (n_features, n_components)."""
        _, features_scaled = features_and_scaled
        _, loadings_df = afa.run_pca(synthetic_df, features_scaled, tmp_path)
        assert loadings_df.shape[0] == len(afa.ACOUSTIC_FEATURES)
        # n_components columns + feature_label column
        assert "PC1" in loadings_df.columns
        assert "PC10" in loadings_df.columns

    def test_explained_variance_sums_to_one(self, synthetic_df, features_and_scaled, tmp_path):
        """Total explained variance sums to approximately 1.0."""
        _, features_scaled = features_and_scaled
        explained_var, _ = afa.run_pca(synthetic_df, features_scaled, tmp_path)
        np.testing.assert_allclose(explained_var.sum(), 1.0, atol=1e-6)

    def test_output_files_created(self, synthetic_df, features_and_scaled, tmp_path):
        """PCA produces scree, biplot, and loadings CSV."""
        _, features_scaled = features_and_scaled
        afa.run_pca(synthetic_df, features_scaled, tmp_path)
        assert (tmp_path / "pca_scree.png").exists()
        assert (tmp_path / "pca_biplot.png").exists()
        assert (tmp_path / "pca_loadings.csv").exists()


class TestUMAP:
    def test_embedding_shape(self, features_and_scaled):
        """UMAP/t-SNE output has shape (N, 2)."""
        _, features_scaled = features_and_scaled
        embedding = afa.compute_umap(features_scaled)
        assert embedding.shape == (features_scaled.shape[0], 2)

    def test_coordinates_saved(self, synthetic_df, features_and_scaled, tmp_path):
        """UMAP coordinates CSV is saved with correct row count."""
        _, features_scaled = features_and_scaled
        embedding = afa.compute_umap(features_scaled)
        afa.save_umap_coordinates(embedding, synthetic_df, tmp_path)
        saved = pd.read_csv(tmp_path / "umap_coordinates.csv")
        assert len(saved) == len(synthetic_df)
        assert "syllable_type" in saved.columns


class TestBoundaryCases:
    def test_identifies_low_confidence(self, synthetic_df, features_and_scaled, tmp_path):
        """Boundary analysis correctly counts low-confidence calls."""
        features_raw, features_scaled = features_and_scaled
        embedding = afa.compute_umap(features_scaled)
        stats = afa.analyze_boundary_cases(synthetic_df, features_raw, embedding, tmp_path)

        expected_low = (synthetic_df["classification_confidence"] == "low").sum()
        assert stats["n_low"] == expected_low
        assert stats["n_total"] == len(synthetic_df)
        assert 0 <= stats["pct_low"] <= 100

    def test_output_file_created(self, synthetic_df, features_and_scaled, tmp_path):
        """Boundary cases PNG is saved."""
        features_raw, features_scaled = features_and_scaled
        embedding = afa.compute_umap(features_scaled)
        afa.analyze_boundary_cases(synthetic_df, features_raw, embedding, tmp_path)
        assert (tmp_path / "boundary_cases.png").exists()


class TestViolinPlots:
    def test_returns_cv_dict(self, synthetic_df, features_and_scaled, tmp_path):
        """Within-type violin plot returns coefficient of variation dict."""
        features_raw, _ = features_and_scaled
        type_cv = afa.plot_within_type_violins(synthetic_df, features_raw, tmp_path)
        assert isinstance(type_cv, dict)
        assert len(type_cv) > 0
        # Keys are "feature|type" format
        for key in type_cv:
            assert "|" in key

    def test_output_file_created(self, synthetic_df, features_and_scaled, tmp_path):
        """Violin plot PNG is saved."""
        features_raw, _ = features_and_scaled
        afa.plot_within_type_violins(synthetic_df, features_raw, tmp_path)
        assert (tmp_path / "within_type_violins.png").exists()


class TestSummary:
    def test_summary_written(self, synthetic_df, features_and_scaled, tmp_path):
        """Analysis summary markdown is written to output dir."""
        features_raw, features_scaled = features_and_scaled
        corr = features_raw.corr()
        corr.index = [afa.FEATURE_LABELS.get(f, f) for f in corr.index]
        corr.columns = corr.index

        from sklearn.decomposition import PCA

        pca = PCA(n_components=len(afa.ACOUSTIC_FEATURES))
        pca.fit(features_scaled)
        loadings_df = pd.DataFrame(
            pca.components_.T,
            index=afa.ACOUSTIC_FEATURES,
            columns=[f"PC{i+1}" for i in range(len(afa.ACOUSTIC_FEATURES))],
        )

        afa.write_summary(
            tmp_path,
            corr,
            pca.explained_variance_ratio_,
            loadings_df,
            {"slope|Flat": 0.5, "sinuosity|Complex": 0.3},
            {"n_low": 10, "n_total": 100, "pct_low": 10.0, "low_by_type": {"Flat": 5, "Down": 3, "Short": 2}},
            n_calls=100,
        )
        summary_path = tmp_path / "analysis_summary.md"
        assert summary_path.exists()
        text = summary_path.read_text()
        assert "Acoustic Feature Deep-Dive" in text
        assert "Variance explained" in text

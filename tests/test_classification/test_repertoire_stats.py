"""Tests for syllable repertoire statistical analysis.

Covers: config validation, syllable proportions, Shannon entropy, transition
matrices, PERMANOVA, Jensen-Shannon divergence, chi-squared, transition
comparison, edge cases, and plot generation.

Uses synthetic data with known statistical properties to validate correctness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Bootstrap sys.path — tests live in tests/test_classification/
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.classification.repertoire_stats import (
    RepertoireConfig,
    _validate_required_columns,
    analyze_repertoire,
    compare_repertoires,
    compare_transition_matrices,
    generate_report,
    plot_repertoire_comparison,
    syllable_diversity,
    syllable_proportions,
    transition_matrix,
)


# ---------------------------------------------------------------------------
# Fixtures: synthetic classified data
# ---------------------------------------------------------------------------

def _make_classified_df(
    n_animals_per_pop: int = 5,
    n_calls_per_animal: int = 100,
    syllable_types: list[str] | None = None,
    weights_pop1: list[float] | None = None,
    weights_pop2: list[float] | None = None,
    seed: int = 42,
) -> pd.DataFrame:
    """Build a synthetic classified DataFrame with two populations.

    Parameters
    ----------
    n_animals_per_pop : int
        Animals per population.
    n_calls_per_animal : int
        Calls per animal.
    syllable_types : list[str]
        Available syllable types.
    weights_pop1, weights_pop2 : list[float]
        Probability weights for syllable type sampling per population.
    seed : int
        Random seed for reproducibility.
    """
    if syllable_types is None:
        syllable_types = ["chirp", "trill", "step_up", "step_down", "flat"]
    if weights_pop1 is None:
        weights_pop1 = [0.4, 0.3, 0.15, 0.1, 0.05]
    if weights_pop2 is None:
        weights_pop2 = weights_pop1  # Identical by default.

    rng = np.random.default_rng(seed)
    records: list[dict] = []

    for pop_idx, (pop, weights) in enumerate(
        [("wild", weights_pop1), ("lab", weights_pop2)]
    ):
        w = np.array(weights, dtype=np.float64)
        w = w / w.sum()

        for animal_idx in range(n_animals_per_pop):
            animal_id = f"{pop}_{animal_idx:02d}"
            for call_idx in range(n_calls_per_animal):
                syl = rng.choice(syllable_types, p=w)
                records.append({
                    "animal_id": animal_id,
                    "population": pop,
                    "label": syl,
                    "begin_time_s": float(call_idx) * 0.5 + pop_idx * 1000,
                    "wav_stem": f"rec_{animal_id}",
                })

    return pd.DataFrame(records)


@pytest.fixture
def classified_df_identical() -> pd.DataFrame:
    """Two populations with IDENTICAL syllable distributions (5 animals × 100 calls)."""
    return _make_classified_df(
        weights_pop1=[0.3, 0.3, 0.2, 0.1, 0.1],
        weights_pop2=[0.3, 0.3, 0.2, 0.1, 0.1],
    )


@pytest.fixture
def classified_df_different() -> pd.DataFrame:
    """Two populations with VERY DIFFERENT syllable distributions."""
    return _make_classified_df(
        weights_pop1=[0.6, 0.3, 0.05, 0.03, 0.02],
        weights_pop2=[0.02, 0.03, 0.05, 0.3, 0.6],
    )


@pytest.fixture
def classified_df_single_animal() -> pd.DataFrame:
    """Single animal per population — edge case."""
    return _make_classified_df(n_animals_per_pop=1, n_calls_per_animal=50)


# ===================================================================
# Config validation
# ===================================================================


class TestRepertoireConfig:
    def test_valid_defaults(self, tmp_path: Path) -> None:
        cfg = RepertoireConfig(classified_data_path=tmp_path / "data.csv")
        assert cfg.n_permutations == 1000
        assert cfg.random_seed == 42

    def test_string_paths_converted(self, tmp_path: Path) -> None:
        cfg = RepertoireConfig(
            classified_data_path=str(tmp_path / "data.csv"),
            output_dir=str(tmp_path / "output"),
        )
        assert isinstance(cfg.classified_data_path, Path)
        assert isinstance(cfg.output_dir, Path)

    def test_zero_permutations_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="positive"):
            RepertoireConfig(
                classified_data_path=tmp_path / "data.csv",
                n_permutations=0,
            )

    def test_negative_permutations_raises(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="positive"):
            RepertoireConfig(
                classified_data_path=tmp_path / "data.csv",
                n_permutations=-100,
            )


class TestValidation:
    def test_missing_columns_raises(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_required_columns(df, ["a", "c", "d"], "test")

    def test_all_columns_present_passes(self) -> None:
        df = pd.DataFrame({"a": [1], "b": [2]})
        _validate_required_columns(df, ["a", "b"], "test")  # No error.


# ===================================================================
# 1. Syllable proportions sum to 1.0 per animal
# ===================================================================


class TestSyllableProportions:
    def test_proportions_sum_to_one(self, classified_df_identical: pd.DataFrame) -> None:
        """Proportions sum to 1.0 per animal."""
        props = syllable_proportions(classified_df_identical, "animal_id", "label")

        for animal, group in props.groupby("animal_id"):
            total = group["proportion"].sum()
            assert total == pytest.approx(1.0), (
                f"Animal {animal} proportions sum to {total}"
            )

    def test_proportions_match_counts(self, classified_df_identical: pd.DataFrame) -> None:
        """Proportions are consistent with raw counts."""
        props = syllable_proportions(classified_df_identical, "animal_id", "label")

        for animal, group in props.groupby("animal_id"):
            total_count = group["count"].sum()
            for _, row in group.iterrows():
                expected_prop = row["count"] / total_count
                assert row["proportion"] == pytest.approx(expected_prop)


# ===================================================================
# 2. Shannon entropy: 0 for single-type, log2(K) for uniform
# ===================================================================


class TestSyllableDiversity:
    def test_entropy_zero_for_single_type(self) -> None:
        """Shannon entropy is 0 when only one syllable type is used."""
        df = pd.DataFrame({
            "animal_id": ["a"] * 50,
            "label": ["chirp"] * 50,
        })
        diversity = syllable_diversity(df, "animal_id", "label")
        assert diversity["entropy_h"].iloc[0] == pytest.approx(0.0)
        assert diversity["n_types_used"].iloc[0] == 1

    def test_entropy_log2k_for_uniform(self) -> None:
        """Shannon entropy is log2(K) for a uniform distribution over K types."""
        K = 4
        types = ["A", "B", "C", "D"]
        n_each = 100  # Exactly equal counts.
        df = pd.DataFrame({
            "animal_id": ["a"] * (K * n_each),
            "label": types * n_each,
        })
        diversity = syllable_diversity(df, "animal_id", "label")
        expected_h = np.log2(K)
        assert diversity["entropy_h"].iloc[0] == pytest.approx(expected_h)
        assert diversity["n_types_used"].iloc[0] == K

    def test_entropy_increases_with_evenness(self) -> None:
        """More evenly distributed repertoires have higher entropy."""
        # Skewed: 90% one type.
        df_skewed = pd.DataFrame({
            "animal_id": ["a"] * 100,
            "label": ["chirp"] * 90 + ["trill"] * 10,
        })
        # Even: 50/50.
        df_even = pd.DataFrame({
            "animal_id": ["b"] * 100,
            "label": ["chirp"] * 50 + ["trill"] * 50,
        })
        h_skewed = syllable_diversity(df_skewed, "animal_id")["entropy_h"].iloc[0]
        h_even = syllable_diversity(df_even, "animal_id")["entropy_h"].iloc[0]
        assert h_even > h_skewed


# ===================================================================
# 3. Transition matrix rows sum to 1.0 (row-stochastic)
# ===================================================================


class TestTransitionMatrix:
    def test_row_stochastic(self, classified_df_identical: pd.DataFrame) -> None:
        """All non-zero rows in the transition matrix sum to 1.0."""
        animals = classified_df_identical["animal_id"].unique()
        mat, labels = transition_matrix(classified_df_identical, animals[0])

        for i in range(mat.shape[0]):
            row_sum = mat[i].sum()
            if row_sum > 0:
                assert row_sum == pytest.approx(1.0), (
                    f"Row {i} ({labels[i]}) sums to {row_sum}"
                )

    def test_zero_row_for_unseen_predecessor(self) -> None:
        """A label that never appears as a predecessor has a zero row."""
        df = pd.DataFrame({
            "animal_id": ["a", "a", "a"],
            "label": ["X", "Y", "X"],
            "begin_time_s": [0.0, 0.5, 1.0],
        })
        mat, labels = transition_matrix(
            df, "a", all_labels=["X", "Y", "Z"]
        )
        z_idx = labels.index("Z")
        assert mat[z_idx].sum() == 0.0

    def test_all_labels_parameter(self) -> None:
        """The all_labels parameter controls matrix dimensions."""
        df = pd.DataFrame({
            "animal_id": ["a", "a", "a"],
            "label": ["X", "Y", "X"],
            "begin_time_s": [0.0, 0.5, 1.0],
        })
        mat, labels = transition_matrix(
            df, "a", all_labels=["X", "Y", "Z", "W"]
        )
        assert mat.shape == (4, 4)
        assert labels == ["X", "Y", "Z", "W"]

    def test_deterministic_sequence(self) -> None:
        """A perfectly cyclic sequence yields known transition probabilities."""
        # Cycle: A -> B -> C -> A -> B -> C
        df = pd.DataFrame({
            "animal_id": ["a"] * 6,
            "label": ["A", "B", "C", "A", "B", "C"],
            "begin_time_s": [0.0, 0.5, 1.0, 1.5, 2.0, 2.5],
        })
        mat, labels = transition_matrix(df, "a")
        a, b, c = labels.index("A"), labels.index("B"), labels.index("C")

        # A always goes to B.
        assert mat[a, b] == pytest.approx(1.0)
        # B always goes to C.
        assert mat[b, c] == pytest.approx(1.0)
        # C always goes to A.
        assert mat[c, a] == pytest.approx(1.0)


# ===================================================================
# 4. PERMANOVA: identical distributions → not significant
# ===================================================================


class TestPermanovaIdentical:
    def test_identical_not_significant(
        self, classified_df_identical: pd.DataFrame
    ) -> None:
        """PERMANOVA with identical distributions yields p > 0.05."""
        result = compare_repertoires(
            classified_df_identical,
            method="permanova",
            n_permutations=999,
            seed=42,
        )
        assert result["method"] == "permanova"
        assert result["p_value"] > 0.05, (
            f"Expected p > 0.05 for identical distributions, got {result['p_value']}"
        )


# ===================================================================
# 5. PERMANOVA: different distributions → significant
# ===================================================================


class TestPermanovaDifferent:
    def test_different_significant(
        self, classified_df_different: pd.DataFrame
    ) -> None:
        """PERMANOVA with very different distributions yields p < 0.05."""
        result = compare_repertoires(
            classified_df_different,
            method="permanova",
            n_permutations=999,
            seed=42,
        )
        assert result["method"] == "permanova"
        assert result["p_value"] < 0.05, (
            f"Expected p < 0.05 for different distributions, got {result['p_value']}"
        )
        assert result["effect_size"] > 0.0


# ===================================================================
# 6. JSD ≈ 0 for identical, > 0 for different
# ===================================================================


class TestJSD:
    def test_jsd_near_zero_for_identical(
        self, classified_df_identical: pd.DataFrame
    ) -> None:
        """JSD is near 0 for identical distributions."""
        result = compare_repertoires(
            classified_df_identical,
            method="jsd",
            n_permutations=999,
            seed=42,
        )
        assert result["method"] == "jsd"
        assert result["statistic"] < 0.1, (
            f"Expected JSD < 0.1 for identical distributions, got {result['statistic']}"
        )

    def test_jsd_positive_for_different(
        self, classified_df_different: pd.DataFrame
    ) -> None:
        """JSD is clearly positive for different distributions."""
        result = compare_repertoires(
            classified_df_different,
            method="jsd",
            n_permutations=999,
            seed=42,
        )
        assert result["statistic"] > 0.1
        assert result["p_value"] < 0.05


# ===================================================================
# 7. Chi-squared detects known differences
# ===================================================================


class TestChiSquared:
    def test_chi_squared_detects_differences(
        self, classified_df_different: pd.DataFrame
    ) -> None:
        """Chi-squared detects known frequency differences in synthetic data."""
        result = compare_repertoires(
            classified_df_different,
            method="chi_square",
        )
        assert result["method"] == "chi_square"
        assert result["p_value"] < 0.05
        assert result["effect_size"] > 0.0  # Cramer's V

    def test_chi_squared_not_significant_for_identical(
        self, classified_df_identical: pd.DataFrame
    ) -> None:
        """Chi-squared should not be significant for identical distributions."""
        result = compare_repertoires(
            classified_df_identical,
            method="chi_square",
        )
        assert result["p_value"] > 0.01  # Lenient — pooled counts may still vary


# ===================================================================
# 8. Transition comparison detects structural differences
# ===================================================================


class TestTransitionComparison:
    def test_detects_structural_differences(self) -> None:
        """Transition comparison detects cyclic vs self-loop structure."""
        rng = np.random.default_rng(42)
        records: list[dict] = []

        # Wild: A -> B -> C -> A (cyclic).
        for a_idx in range(5):
            seq = ["A", "B", "C"] * 34  # 102 calls
            for i, syl in enumerate(seq):
                records.append({
                    "animal_id": f"wild_{a_idx:02d}",
                    "population": "wild",
                    "label": syl,
                    "begin_time_s": float(i) * 0.5,
                })

        # Lab: self-loops (A -> A, B -> B, C -> C).
        for a_idx in range(5):
            for i in range(102):
                syl = rng.choice(["A", "B", "C"])
                records.append({
                    "animal_id": f"lab_{a_idx:02d}",
                    "population": "lab",
                    "label": syl,
                    "begin_time_s": float(i) * 0.5,
                })

        df = pd.DataFrame(records)

        result = compare_transition_matrices(
            df, n_permutations=999, seed=42,
        )
        assert result["frobenius_norm"] > 0.0
        assert result["p_value"] < 0.05
        assert len(result["differentially_used_transitions"]) > 0

    def test_identical_transition_not_significant(
        self, classified_df_identical: pd.DataFrame
    ) -> None:
        """Identical distributions yield non-significant transition comparison."""
        result = compare_transition_matrices(
            classified_df_identical,
            n_permutations=999,
            seed=42,
        )
        assert result["p_value"] > 0.05


# ===================================================================
# 9. Single-animal data works (no div-by-zero)
# ===================================================================


class TestEdgeCases:
    def test_single_animal_per_pop(
        self, classified_df_single_animal: pd.DataFrame
    ) -> None:
        """Single animal per population produces valid results (no crash)."""
        result = compare_repertoires(
            classified_df_single_animal,
            method="permanova",
            n_permutations=99,
            seed=42,
        )
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0

    def test_single_syllable_type(self) -> None:
        """All calls are the same type — entropy is 0, no crash."""
        df = pd.DataFrame({
            "animal_id": ["a"] * 50 + ["b"] * 50,
            "population": ["wild"] * 50 + ["lab"] * 50,
            "label": ["chirp"] * 100,
            "begin_time_s": list(range(100)),
        })
        diversity = syllable_diversity(df, "animal_id")
        assert all(diversity["entropy_h"] == 0.0)

        # Chi-squared should still run (though test may not be meaningful).
        result = compare_repertoires(df, method="chi_square")
        assert "p_value" in result


# ===================================================================
# 10. Missing population column → ValueError
# ===================================================================


class TestMissingColumns:
    def test_missing_population_column(self) -> None:
        """Missing population column raises ValueError with helpful message."""
        df = pd.DataFrame({
            "animal_id": ["a", "b"],
            "label": ["chirp", "trill"],
        })
        with pytest.raises(ValueError, match="missing required columns"):
            compare_repertoires(df, population_column="population")

    def test_missing_syllable_column(self) -> None:
        df = pd.DataFrame({
            "animal_id": ["a"],
            "population": ["wild"],
        })
        with pytest.raises(ValueError, match="missing required columns"):
            syllable_proportions(df, "animal_id", "label")

    def test_unknown_method_raises(self, classified_df_identical: pd.DataFrame) -> None:
        with pytest.raises(ValueError, match="Unknown method"):
            compare_repertoires(classified_df_identical, method="magic")

    def test_jsd_rejects_three_populations(self) -> None:
        """JSD requires exactly 2 populations — 3 raises ValueError."""
        df = pd.DataFrame({
            "animal_id": ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
            "population": ["wild"] * 30 + ["lab"] * 30 + ["hybrid"] * 30,
            "label": ["chirp", "trill", "flat"] * 30,
            "begin_time_s": list(range(90)),
        })
        with pytest.raises(ValueError, match="exactly 2 populations"):
            compare_repertoires(df, method="jsd", n_permutations=99)

    def test_transition_comparison_rejects_three_populations(self) -> None:
        """Transition comparison requires exactly 2 populations."""
        df = pd.DataFrame({
            "animal_id": ["a"] * 30 + ["b"] * 30 + ["c"] * 30,
            "population": ["wild"] * 30 + ["lab"] * 30 + ["hybrid"] * 30,
            "label": ["chirp", "trill", "flat"] * 30,
            "begin_time_s": list(range(90)),
        })
        with pytest.raises(ValueError, match="exactly 2 populations"):
            compare_transition_matrices(df, n_permutations=99)


# ===================================================================
# Plot generation
# ===================================================================


class TestPlots:
    def test_plot_files_created(
        self, classified_df_identical: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Plot function creates all four expected figure files."""
        paths = plot_repertoire_comparison(
            classified_df_identical,
            output_dir=tmp_path / "plots",
        )
        assert len(paths) == 4

        expected_names = {
            "syllable_proportions.png",
            "diversity_boxplot.png",
            "transition_heatmaps.png",
            "animal_scatter.png",
        }
        actual_names = {p.name for p in paths}
        assert actual_names == expected_names

        for p in paths:
            assert p.exists()
            assert p.stat().st_size > 0


# ===================================================================
# Report generation
# ===================================================================


class TestReport:
    def test_report_generated(self, tmp_path: Path) -> None:
        """generate_report writes a markdown file."""
        results = {
            "diversity": pd.DataFrame({
                "animal_id": ["a", "b"],
                "entropy_h": [1.5, 2.0],
                "n_types_used": [3, 4],
                "n_calls": [100, 100],
            }),
            "permanova": {
                "method": "permanova",
                "statistic": 3.5,
                "p_value": 0.03,
                "effect_size": 0.25,
                "interpretation": "Significant difference.",
            },
        }
        path = generate_report(results, tmp_path)
        assert path.exists()
        content = path.read_text(encoding="utf-8")
        assert "Syllable Repertoire Analysis Report" in content
        assert "PERMANOVA" in content
        assert "Significant" in content


# ===================================================================
# End-to-end integration
# ===================================================================


class TestIntegration:
    def test_full_pipeline(
        self, classified_df_different: pd.DataFrame, tmp_path: Path
    ) -> None:
        """Full pipeline runs end-to-end on synthetic data."""
        # Write classified data to CSV.
        csv_path = tmp_path / "classified.csv"
        classified_df_different.to_csv(csv_path, index=False)

        config = RepertoireConfig(
            classified_data_path=csv_path,
            n_permutations=99,
            random_seed=42,
            output_dir=tmp_path / "results",
        )
        results = analyze_repertoire(config)

        # Check outputs exist.
        output_dir = tmp_path / "results"
        assert (output_dir / "syllable_proportions.csv").exists()
        assert (output_dir / "diversity_comparison.csv").exists()
        assert (output_dir / "statistical_tests.json").exists()
        assert (output_dir / "repertoire_report.md").exists()
        assert (output_dir / "figures" / "syllable_proportions.png").exists()

        # Statistical tests should detect differences.
        assert results["permanova"]["p_value"] < 0.05
        assert results["chi_square"]["p_value"] < 0.05
        # JSD and transition comparison should also be present and significant.
        assert "jsd" in results
        assert results["jsd"]["p_value"] < 0.05
        assert "transition_comparison" in results
        assert results["transition_comparison"]["p_value"] < 0.05

    def test_three_population_orchestrator(self, tmp_path: Path) -> None:
        """Orchestrator gracefully skips JSD/transitions for 3+ populations."""
        rng = np.random.default_rng(42)
        types = ["chirp", "trill", "flat", "step_up"]
        rows = []
        for pop, w in [
            ("wild", [0.5, 0.2, 0.2, 0.1]),
            ("lab", [0.1, 0.5, 0.2, 0.2]),
            ("hybrid", [0.3, 0.3, 0.2, 0.2]),
        ]:
            for animal_idx in range(3):
                aid = f"{pop}_{animal_idx}"
                labels = rng.choice(types, size=50, p=w)
                for i, label in enumerate(labels):
                    rows.append({
                        "animal_id": aid,
                        "population": pop,
                        "label": label,
                        "begin_time_s": float(i) * 0.1,
                    })
        df = pd.DataFrame(rows)
        csv_path = tmp_path / "three_pop.csv"
        df.to_csv(csv_path, index=False)

        config = RepertoireConfig(
            classified_data_path=csv_path,
            n_permutations=99,
            random_seed=42,
            output_dir=tmp_path / "results",
        )
        results = analyze_repertoire(config)

        # PERMANOVA and chi-square should run and succeed.
        assert "permanova" in results
        assert "chi_square" in results
        # JSD and transition_comparison should be skipped (not crash).
        assert "jsd" not in results
        assert "transition_comparison" not in results
        # Outputs should still be created.
        assert (tmp_path / "results" / "statistical_tests.json").exists()
        assert (tmp_path / "results" / "repertoire_report.md").exists()

    def test_pca_single_animal_no_crash(self, tmp_path: Path) -> None:
        """PCA scatter plot degrades gracefully with only 1 total animal."""
        df = pd.DataFrame({
            "animal_id": ["a"] * 20,
            "population": ["wild"] * 20,
            "label": ["chirp"] * 10 + ["trill"] * 10,
            "begin_time_s": [float(i) * 0.1 for i in range(20)],
        })
        # Should not raise — PCA is skipped, fallback text rendered.
        paths = plot_repertoire_comparison(
            df,
            population_column="population",
            animal_id_column="animal_id",
            syllable_column="label",
            time_column="begin_time_s",
            output_dir=tmp_path,
        )
        assert any("animal_scatter" in str(p) for p in paths)

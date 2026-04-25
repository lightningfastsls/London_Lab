"""Unit tests for usv_spectrogram.classification.cross_population.

Tests use synthetic CSV fixtures (not the real 7,921-call wild datasets) so the
suite runs in a few seconds. The smoke test against the real 5970/3452 data
is separately runnable via the handoff's bash block.

Canary tests:
    - identical populations -> chi2 p > 0.05, JSD ~= 0
    - deliberately different populations -> chi2 p < 0.001
    - single-type edge -> no crash, sensible sentinels
    - same random_state -> identical bootstrap CIs
    - JSON round-trip preserves schema
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from usv_spectrogram.classification.cross_population import (
    CANONICAL_BOUT_THRESHOLD_S,
    SCHEMA_VERSION,
    ComparisonReport,
    CrossPopulationComparison,
)


# ---------------------------------------------------------------------------
# Synthetic-data helpers
# ---------------------------------------------------------------------------


SYLLABLE_TYPES = ["Flat", "Down", "Chevron", "Short", "Complex", "Frequency_Jump", "Up"]


def _make_synthetic_csv(
    path: Path,
    n_calls: int,
    type_probs: dict[str, float],
    n_files: int = 4,
    seed: int = 0,
    mean_call_length_s: float = 0.05,
    mean_ici_within_s: float = 0.1,
    file_prefix: str = "synth",
    feature_shift: float = 0.0,
) -> None:
    """Create a minimal classified-detection CSV matching the real schema.

    Generates ``n_calls`` calls distributed across ``n_files`` recordings,
    with per-type probabilities from ``type_probs``. Call timing is
    arranged so that within a file all gaps are < ``CANONICAL_BOUT_THRESHOLD_S``
    (i.e. every file is one bout). Acoustic features are drawn from
    type-specific Gaussians (optionally shifted by ``feature_shift``).
    """
    rng = np.random.default_rng(seed)
    types_arr = np.array(list(type_probs.keys()))
    probs = np.array(list(type_probs.values()), dtype=float)
    probs = probs / probs.sum()

    chosen = rng.choice(types_arr, size=n_calls, p=probs)

    # Split across files roughly evenly.
    file_split = np.array_split(np.arange(n_calls), n_files)

    rows: list[dict] = []
    for f_idx, call_idxs in enumerate(file_split):
        if len(call_idxs) == 0:
            continue
        fname = f"{file_prefix}_file_{f_idx:03d}.wav"
        t = 0.0
        for call_idx in call_idxs:
            t_begin = t
            dur = max(0.005, mean_call_length_s + rng.normal(0, 0.005))
            t_end = t_begin + dur
            # Next call's ici_gap stays within-bout (< 0.6 s).
            ici = max(0.01, mean_ici_within_s + rng.normal(0, 0.02))
            t = t_end + ici

            typ = str(chosen[call_idx])
            # Type-specific feature means (shifted per pop to test KS).
            feature_means = {
                "Flat": 60_000 + feature_shift * 1000,
                "Down": 70_000 + feature_shift * 1000,
                "Chevron": 80_000 + feature_shift * 1000,
                "Short": 65_000 + feature_shift * 1000,
                "Complex": 75_000 + feature_shift * 1000,
                "Frequency_Jump": 85_000 + feature_shift * 1000,
                "Up": 72_000 + feature_shift * 1000,
            }
            pf = feature_means.get(typ, 70_000) + rng.normal(0, 3000)
            rows.append({
                "file": fname,
                "id": int(call_idx),
                "label": typ,
                "accepted": True,
                "score": 0.9,
                "begin_time_s": t_begin,
                "end_time_s": t_end,
                "call_length_s": dur,
                "principal_freq_hz": pf,
                "low_freq_hz": pf - 5000,
                "high_freq_hz": pf + 5000,
                "bandwidth_hz": 10_000 + rng.normal(0, 1000),
                "freq_std_dev_hz": 800 + rng.normal(0, 50),
                "slope": rng.normal(0 + feature_shift * 0.1, 1),
                "sinuosity": 1.2 + rng.normal(0, 0.05),
                "mean_power_db": -60 + rng.normal(0, 3),
                "tonality": 0.8 + rng.normal(0, 0.05),
                "peak_freq_khz": pf / 1000.0,
                "source_file": fname,
                "wav_stem": fname.replace(".wav", ""),
                "det_start_s": t_begin,
                "det_end_s": t_end,
                "det_duration_ms": dur * 1000.0,
                "det_index": int(call_idx),
                "det_prob_max": 0.95,
                "det_prob_mean": 0.9,
                "det_user_action": "accepted",
                "det_json_path": "",
                "match_quality": "good",
                "match_distance_ms": 0.0,
                "syllable_type": typ,
                "classification_confidence": 0.9 + rng.uniform(-0.05, 0.05),
            })
    df = pd.DataFrame(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def identical_populations(tmp_path: Path):
    probs = {t: 1.0 for t in SYLLABLE_TYPES}  # uniform; will be normalized
    p_a = tmp_path / "pop_a.csv"
    p_b = tmp_path / "pop_b.csv"
    # Same seed + same probs -> distributions are identical up to sampling noise;
    # larger n_calls reduces that noise so chi2 p is high.
    _make_synthetic_csv(p_a, n_calls=2000, type_probs=probs, seed=1, file_prefix="a")
    _make_synthetic_csv(p_b, n_calls=2000, type_probs=probs, seed=2, file_prefix="b")
    return p_a, p_b


@pytest.fixture
def different_populations(tmp_path: Path):
    probs_a = {
        "Flat": 0.5, "Down": 0.1, "Chevron": 0.1, "Short": 0.1,
        "Complex": 0.1, "Frequency_Jump": 0.05, "Up": 0.05,
    }
    probs_b = {
        "Flat": 0.05, "Down": 0.05, "Chevron": 0.1, "Short": 0.5,
        "Complex": 0.1, "Frequency_Jump": 0.1, "Up": 0.1,
    }
    p_a = tmp_path / "pop_a.csv"
    p_b = tmp_path / "pop_b.csv"
    _make_synthetic_csv(
        p_a, n_calls=1500, type_probs=probs_a, seed=11,
        file_prefix="a", feature_shift=0.0,
    )
    _make_synthetic_csv(
        p_b, n_calls=1500, type_probs=probs_b, seed=22,
        file_prefix="b", feature_shift=5.0,
    )
    return p_a, p_b


@pytest.fixture
def single_type_population(tmp_path: Path):
    probs_a = {t: 1.0 for t in SYLLABLE_TYPES}
    probs_b = {"Flat": 1.0}  # only one type present
    p_a = tmp_path / "pop_a.csv"
    p_b = tmp_path / "pop_b.csv"
    _make_synthetic_csv(p_a, n_calls=400, type_probs=probs_a, seed=31, file_prefix="a")
    _make_synthetic_csv(p_b, n_calls=400, type_probs=probs_b, seed=32, file_prefix="b")
    return p_a, p_b


# ---------------------------------------------------------------------------
# Canonical tests from the handoff
# ---------------------------------------------------------------------------


def test_synthetic_identical_chi2_not_significant(identical_populations):
    p_a, p_b = identical_populations
    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=100, n_permutations=200, random_state=7,
    )
    r = cmp.compare_type_proportions()
    assert r.chi2_p_value > 0.05, (
        f"identical pops expected chi2 p > 0.05, got {r.chi2_p_value}"
    )
    assert r.max_abs_cohens_h < 0.2, (
        f"identical pops expected small Cohen's h, got {r.max_abs_cohens_h}"
    )


def test_synthetic_identical_jsd_near_zero(identical_populations):
    p_a, p_b = identical_populations
    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=50, random_state=7,
    )
    r = cmp.compare_repertoires_jsd()
    assert r.jsd_bits < 0.01, f"identical pops expected JSD ~0, got {r.jsd_bits}"


def test_synthetic_identical_entropy_close(identical_populations):
    p_a, p_b = identical_populations
    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=50, n_permutations=100, random_state=7,
    )
    r = cmp.compare_entropy()
    assert abs(r.difference_bits) < 0.1, (
        f"identical pops expected entropy diff ~0, got {r.difference_bits}"
    )


def test_synthetic_different_chi2_significant(different_populations):
    p_a, p_b = different_populations
    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=50, n_permutations=100, random_state=13,
    )
    r = cmp.compare_type_proportions()
    assert r.chi2_p_value < 0.001, (
        f"different pops expected chi2 p<0.001, got {r.chi2_p_value}"
    )
    assert r.max_abs_cohens_h > 0.3, (
        f"different pops expected Cohen's h > 0.3, got {r.max_abs_cohens_h}"
    )


def test_single_type_no_crash(single_type_population):
    p_a, p_b = single_type_population
    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=30, n_permutations=50, random_state=1,
    )
    # Should not crash and should produce a sensible structured result.
    report = cmp.run_all(skip=["umap_overlap"])
    assert report.type_proportions is not None
    assert report.jsd is not None
    # Zipf should flag insufficient types (only 1 type in pop_b).
    assert report.zipf is not None
    assert report.zipf.pop_b.insufficient_types is True


def test_bootstrap_reproducibility(identical_populations):
    p_a, p_b = identical_populations
    args = dict(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=50, random_state=99,
    )
    c1 = CrossPopulationComparison(**args)
    c2 = CrossPopulationComparison(**args)
    r1 = c1.compare_repertoires_jsd()
    r2 = c2.compare_repertoires_jsd()
    assert r1.bootstrap_ci_95 == r2.bootstrap_ci_95, (
        "same random_state must yield identical CIs"
    )


def test_json_round_trip(different_populations, tmp_path: Path):
    p_a, p_b = different_populations
    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=20, n_permutations=30, random_state=2,
    )
    report = cmp.run_all(skip=["umap_overlap"])
    json_path = report.write_json(tmp_path / "report.json")
    blob = json.loads(json_path.read_text())
    assert blob["metadata"]["schema_version"] == SCHEMA_VERSION
    assert blob["metadata"]["pop_a_label"] == "a"
    assert blob["metadata"]["bout_threshold_s"] == CANONICAL_BOUT_THRESHOLD_S
    assert "type_proportions" in blob
    assert "jsd" in blob
    assert "entropy" in blob
    assert "mi_lag1" in blob
    assert "transitions" in blob


def test_markdown_writer_produces_nonempty(different_populations, tmp_path: Path):
    p_a, p_b = different_populations
    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=10, n_permutations=20, random_state=3,
    )
    report = cmp.run_all(skip=["umap_overlap"])
    md_path = report.write_markdown(tmp_path / "report.md")
    text = md_path.read_text()
    assert "Cross-Population Comparison" in text
    assert "a" in text and "b" in text
    assert len(text) > 500


def test_transitions_bout_aware_counts_match_expectation(tmp_path: Path):
    probs = {"Flat": 0.5, "Down": 0.5}
    p_a = tmp_path / "a.csv"
    p_b = tmp_path / "b.csv"
    _make_synthetic_csv(p_a, n_calls=200, type_probs=probs, seed=41, file_prefix="a")
    _make_synthetic_csv(p_b, n_calls=200, type_probs=probs, seed=42, file_prefix="b")

    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=10, random_state=5,
    )
    r = cmp.compare_transitions()
    # Per-file, every call within a bout -> total within-bout pairs =
    # sum over files of (n_file - 1).
    def expected_pairs(df):
        per_file = df.groupby("file").size() - 1
        return int(per_file[per_file > 0].sum())
    assert r.pop_a_n_within_pairs == expected_pairs(cmp.pop_a_df)
    assert r.pop_b_n_within_pairs == expected_pairs(cmp.pop_b_df)


def test_mismatched_labels_raise(tmp_path: Path):
    probs = {"Flat": 1.0}
    p_a = tmp_path / "a.csv"
    p_b = tmp_path / "b.csv"
    _make_synthetic_csv(p_a, n_calls=50, type_probs=probs, seed=1, file_prefix="a")
    _make_synthetic_csv(p_b, n_calls=50, type_probs=probs, seed=2, file_prefix="b")

    with pytest.raises(ValueError, match="must differ"):
        CrossPopulationComparison(
            pop_a_csv=p_a, pop_a_label="same",
            pop_b_csv=p_b, pop_b_label="same",
            strata_note="synthetic-test",
        )


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        CrossPopulationComparison(
            pop_a_csv=tmp_path / "nope.csv", pop_a_label="a",
            pop_b_csv=tmp_path / "also_nope.csv", pop_b_label="b",
            strata_note="synthetic-test",
        )


def test_missing_strata_note_raises(tmp_path: Path):
    """Schema 1.1 makes strata_note required. Omitting it should TypeError
    (Python's missing-arg) before any other validation runs."""
    probs = {"Flat": 1.0}
    p_a = tmp_path / "a.csv"
    p_b = tmp_path / "b.csv"
    _make_synthetic_csv(p_a, n_calls=20, type_probs=probs, seed=1, file_prefix="a")
    _make_synthetic_csv(p_b, n_calls=20, type_probs=probs, seed=2, file_prefix="b")
    with pytest.raises(TypeError, match="strata_note"):
        CrossPopulationComparison(
            pop_a_csv=p_a, pop_a_label="a",
            pop_b_csv=p_b, pop_b_label="b",
        )


def test_empty_strata_note_raises(tmp_path: Path):
    """Empty / whitespace-only strata_note must be rejected with a clear message
    pointing the caller at the project framing docs."""
    probs = {"Flat": 1.0}
    p_a = tmp_path / "a.csv"
    p_b = tmp_path / "b.csv"
    _make_synthetic_csv(p_a, n_calls=20, type_probs=probs, seed=1, file_prefix="a")
    _make_synthetic_csv(p_b, n_calls=20, type_probs=probs, seed=2, file_prefix="b")
    for bad in ("", "   ", "\n\t"):
        with pytest.raises(ValueError, match="strata_note is required"):
            CrossPopulationComparison(
                pop_a_csv=p_a, pop_a_label="a",
                pop_b_csv=p_b, pop_b_label="b",
                strata_note=bad,
            )


def test_strata_note_flows_into_metadata(different_populations):
    """strata_note + strata_note_extra should appear in metadata, JSON, and
    the rendered markdown header."""
    p_a, p_b = different_populations
    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="wild-vs-wild between-couple",
        strata_note_extra="N=1 couple per cohort, both wild-caught.",
        n_bootstrap=10, n_permutations=20, random_state=1,
    )
    report = cmp.run_all(skip=["umap_overlap"])
    assert report.metadata.strata_note == "wild-vs-wild between-couple"
    assert "N=1 couple per cohort" in report.metadata.strata_note_extra
    summary = report.summary()
    assert "wild-vs-wild between-couple" in summary
    assert "STRATA" in summary


def test_feature_comparison_detects_shift(different_populations):
    p_a, p_b = different_populations
    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=10, random_state=17,
    )
    r = cmp.compare_features()
    # We shifted principal_freq_hz by ~5000 Hz in pop_b (via feature_shift=5.0
    # which multiplies by 1000 in the fixture). That should produce a large
    # Cohen's d on principal_freq_hz.
    assert "principal_freq_hz" in r.per_feature
    assert abs(r.per_feature["principal_freq_hz"].cohens_d) > 0.3
    assert r.per_feature["principal_freq_hz"].ks_p_value < 0.01


def test_ioi_distributions_kstest(different_populations):
    p_a, p_b = different_populations
    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=10, random_state=19,
    )
    r = cmp.compare_ioi_distributions()
    assert r.n_a > 0 and r.n_b > 0
    assert not np.isnan(r.ks_statistic)
    assert 0.0 <= r.ks_statistic <= 1.0


def test_mi_lag1_reproduces_known_canary(identical_populations):
    """When both pops are the same, MI(A) and MI(B) should be very close."""
    p_a, p_b = identical_populations
    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=10, random_state=23,
    )
    r = cmp.compare_mi_lag1()
    # Both generated from uniform independent draws, so MI should be near
    # zero for both. This also exercises the bout-aware MI path.
    assert abs(r.pop_a_mi_bits) < 0.1
    assert abs(r.pop_b_mi_bits) < 0.1


def test_zipf_on_uniform_flags_insufficient(tmp_path: Path):
    """Uniform over 7 types has <10 unique values → zipf flags insufficient."""
    probs = {t: 1.0 for t in SYLLABLE_TYPES}
    p_a = tmp_path / "a.csv"
    p_b = tmp_path / "b.csv"
    _make_synthetic_csv(p_a, n_calls=300, type_probs=probs, seed=61, file_prefix="a")
    _make_synthetic_csv(p_b, n_calls=300, type_probs=probs, seed=62, file_prefix="b")
    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=5, random_state=29,
    )
    r = cmp.compare_zipf()
    # Both pops have 7 types → alpha should be sentinel 0.0 and flagged.
    assert r.pop_a.insufficient_types is True
    assert r.pop_b.insufficient_types is True


def test_run_all_with_umap_skipped_produces_all_other_metrics(different_populations):
    p_a, p_b = different_populations
    cmp = CrossPopulationComparison(
        pop_a_csv=p_a, pop_a_label="a",
        pop_b_csv=p_b, pop_b_label="b",
        strata_note="synthetic-test",
        n_bootstrap=10, n_permutations=20, random_state=31,
    )
    report = cmp.run_all(skip=["umap_overlap"])
    assert isinstance(report, ComparisonReport)
    for name in (
        "type_proportions", "jsd", "entropy", "transitions", "mi_lag1",
        "zipf", "burstiness", "ioi", "features",
    ):
        assert getattr(report, name) is not None, f"{name} missing"
    assert report.umap_overlap is None
    # summary() must not crash and must mention both labels.
    s = report.summary()
    assert "a" in s and "b" in s

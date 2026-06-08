"""Tests for WS-D pure archetype helpers (scripts/experiments/shape_archetypes.py).

These cover the reusable, deterministic functions only — not the PCHA fit or the
HTML report (those are integration-exercised by running the script).
"""
from __future__ import annotations

import numpy as np
import pytest

from scripts.experiments.shape_archetypes import (
    bootstrap_mean_ci,
    cohort_simplex_means,
    match_archetypes,
    matched_archetype_similarity,
    project_to_simplex,
)


def test_project_to_simplex_rows_are_valid_simplex():
    rng = np.random.default_rng(0)
    Z = rng.normal(size=(4, 6))      # 4 archetypes in 6-D
    X = rng.normal(size=(20, 6))
    S = project_to_simplex(X, Z)
    assert S.shape == (20, 4)
    np.testing.assert_allclose(S.sum(axis=1), 1.0, atol=1e-6)
    assert (S >= -1e-9).all()


def test_project_to_simplex_recovers_vertices():
    # A point sitting exactly on an archetype should map to that vertex.
    Z = np.array([[5.0, 0.0], [0.0, 5.0], [-5.0, -5.0]])
    S = project_to_simplex(Z.copy(), Z)
    np.testing.assert_allclose(S, np.eye(3), atol=1e-3)


def test_project_to_simplex_recovers_known_mixture():
    Z = np.array([[10.0, 0.0], [0.0, 10.0], [-10.0, 0.0]])
    true = np.array([0.5, 0.3, 0.2])
    x = (true[:, None] * Z).sum(axis=0)  # convex combo of archetypes
    S = project_to_simplex(x[None, :], Z)[0]
    np.testing.assert_allclose(S, true, atol=1e-2)


def test_match_archetypes_identity():
    a = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, -1.0]])
    perm, cost = match_archetypes(a, a)
    assert list(perm) == [0, 1, 2]
    assert cost == pytest.approx(0.0)


def test_match_archetypes_recovers_permutation():
    a = np.array([[0.0, 0.0], [10.0, 0.0], [0.0, 10.0]])
    shuffle = [2, 0, 1]
    b = a[shuffle]
    perm, cost = match_archetypes(a, b)
    # b[perm[i]] should equal a[i]: perm must invert `shuffle`.
    np.testing.assert_array_equal(b[perm], a)
    assert cost == pytest.approx(0.0)


def test_match_archetypes_shape_mismatch_raises():
    a = np.zeros((3, 2))
    b = np.zeros((4, 2))
    with pytest.raises(ValueError):
        match_archetypes(a, b)


def test_matched_similarity_identical_is_one():
    a = np.array([[1.0, 2.0], [-3.0, 1.0], [0.5, 0.5]])
    assert matched_archetype_similarity(a, a) == pytest.approx(1.0)


def test_matched_similarity_below_one_for_rotated_set():
    # A genuinely different archetype set (rotated 45°, distinct directions)
    # must score below 1: stability is not trivially saturated.
    a = np.array([[1.0, 0.0], [0.0, 1.0]])
    theta = np.pi / 4
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    b = a @ rot.T
    s = matched_archetype_similarity(a, b)
    assert s < 0.999

def test_matched_similarity_separated_sets_match_by_proximity():
    # Far-separated archetypes get matched to their nearest counterpart (same
    # direction), giving similarity 1 — confirms matching is by proximity.
    a = np.array([[10.0, 0.0], [-10.0, 0.0]])
    b = np.array([[9.0, 0.0], [-9.0, 0.0]])
    assert matched_archetype_similarity(a, b) == pytest.approx(1.0)


def test_matched_similarity_invariant_to_archetype_order():
    a = np.array([[1.0, 0.0], [0.0, 2.0], [3.0, 3.0]])
    b = a[[1, 2, 0]] * 1.0001  # tiny perturbation, reordered
    s1 = matched_archetype_similarity(a, b)
    s2 = matched_archetype_similarity(a, a[[2, 0, 1]] * 1.0001)
    assert s1 == pytest.approx(s2, abs=1e-6)
    assert s1 > 0.999


def test_cohort_simplex_means_basic():
    mem = np.array([
        [1.0, 0.0],
        [0.0, 1.0],   # cohort A mean -> [0.5, 0.5]
        [0.2, 0.8],
        [0.4, 0.6],   # cohort B mean -> [0.3, 0.7]
    ])
    cohorts = np.array(["A", "A", "B", "B"])
    out = cohort_simplex_means(mem, cohorts)
    np.testing.assert_allclose(out["A"], [0.5, 0.5])
    np.testing.assert_allclose(out["B"], [0.3, 0.7])


def test_cohort_simplex_means_rows_sum_preserved():
    rng = np.random.default_rng(1)
    raw = rng.random((50, 4))
    mem = raw / raw.sum(axis=1, keepdims=True)  # simplex rows
    cohorts = np.array(["x"] * 25 + ["y"] * 25)
    out = cohort_simplex_means(mem, cohorts)
    for v in out.values():
        assert v.sum() == pytest.approx(1.0)


def test_bootstrap_mean_ci_contains_mean_and_orders():
    rng = np.random.default_rng(2)
    x = rng.normal(loc=[1.0, -2.0], scale=0.5, size=(2000, 2))
    mean, lo, hi = bootstrap_mean_ci(x, n_boot=500, alpha=0.05, seed=3)
    assert np.all(lo <= mean) and np.all(mean <= hi)
    # CI should bracket the true means for this large, well-behaved sample.
    assert lo[0] < 1.0 < hi[0]
    assert lo[1] < -2.0 < hi[1]


def test_bootstrap_mean_ci_deterministic_with_seed():
    x = np.random.default_rng(4).random((200, 3))
    a = bootstrap_mean_ci(x, n_boot=200, seed=7)
    b = bootstrap_mean_ci(x, n_boot=200, seed=7)
    for u, v in zip(a, b):
        np.testing.assert_array_equal(u, v)

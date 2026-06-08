"""Reversal (time-direction) unit test — cross-cutting rule (1).

An up-sweep and its mirror are biologically distinct; no method may be
time-reversal invariant. For a method's `encode_fn`, we measure the
feature-space distance between encode(x) and encode(reverse(x)) for a sample of
contours, and compare it to the pairwise encode-distance distribution.

PASS iff median self-reverse distance >= 90th percentile of pairwise distances
(i.e. reversing a curve moves it into the top decile of how far apart distinct
curves are). A reversal-blind encode (turning function, persistence) FAILS:
self-reverse distance ~ 0. The remedy (handoff) is to append a signed direction
feature (net slope) and re-test.
"""
from __future__ import annotations

import numpy as np


def reversal_test(encode_fn, contours50, n_pairs=2000, seed=42):
    """Compute the reversal verdict for `encode_fn`.

    Parameters
    ----------
    encode_fn : callable(contour:(L,)) -> feature vector (1-D). Must be
                deterministic.
    contours50 : (N,L) array of contours to sample from.
    n_pairs : number of self-reverse samples and pairwise samples.

    Returns
    -------
    dict {passed, self_reverse_median, decile_threshold, note,
          self_reverse_distances(summary), pairwise(summary)}
    """
    contours50 = np.asarray(contours50, dtype=np.float64)
    n = len(contours50)
    rng = np.random.default_rng(seed)

    # encode a working sample once
    m = min(n_pairs, n)
    samp = rng.choice(n, size=m, replace=False)
    feats = np.array([np.asarray(encode_fn(contours50[i]), dtype=np.float64).ravel() for i in samp])
    feats_rev = np.array([np.asarray(encode_fn(contours50[i][::-1]), dtype=np.float64).ravel()
                          for i in samp])

    # self-reverse distances: dist(encode(x), encode(reverse(x)))
    self_rev = np.linalg.norm(feats - feats_rev, axis=1)

    # pairwise encode-distances among the sample (random pairs)
    a = rng.integers(0, m, n_pairs)
    b = rng.integers(0, m, n_pairs)
    ok = a != b
    a, b = a[ok], b[ok]
    pair = np.linalg.norm(feats[a] - feats[b], axis=1)

    self_med = float(np.median(self_rev))
    decile = float(np.percentile(pair, 90))
    passed = bool(self_med >= decile)
    note = ("PASS: reversing a curve moves it into the top decile of pairwise distance "
            "-> encode is direction-sensitive."
            if passed else
            "FAIL: encode is (near-)reversal-blind -> append a signed direction feature "
            "(net slope) and re-test.")
    return {
        "passed": passed,
        "self_reverse_median": self_med,
        "decile_threshold": decile,
        "note": note,
        "self_reverse_summary": {
            "median": self_med,
            "p10": float(np.percentile(self_rev, 10)),
            "p90": float(np.percentile(self_rev, 90)),
        },
        "pairwise_summary": {
            "median": float(np.median(pair)),
            "p90": decile,
            "max": float(pair.max()),
        },
    }

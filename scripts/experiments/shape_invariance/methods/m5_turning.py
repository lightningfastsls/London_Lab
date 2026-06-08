"""M5 — Classical curve signatures (turning function + curvature scale space).

The simplest, dependency-free method; used FIRST to shake out the harness, the
reversal test, and the side-channel logic (handoff "M5 ... DO THIS FIRST").

TURNING FUNCTION (primary descriptor)
-------------------------------------
A registered contour is the curve C(t) = (t, f(t)) with t the normalized
[0,1] time index and f the mean-pitch-subtracted frequency. The turning
function Theta(s) is the tangent-direction angle as a function of normalized
arc length s in [0,1]:
    seg vectors  dP_i = (dt_i, df_i)
    tangent angle theta_i = atan2(df_i, dt_i)   (unwrapped)
    arc length   s_i = cumsum(|dP_i|) / total
    Theta = interp(theta onto a uniform s-grid of n_out points)
L2 distance on Theta is translation-invariant by construction (differences) and
arc-length-normalized. Reversing the curve flips Theta -> direction-SENSITIVE
in principle, BUT for a contour the start point is pinned to the registered
onset and the curve is a function-graph; the turning function is dominated by
the unsigned bend sequence, so it is near-reversal-blind in practice -> the
reversal test is expected to FAIL and we append a signed net-slope feature.

AXIS-SCALE HANDLING (the real subtlety for a (time,freq) graph)
---------------------------------------------------------------
atan2(df, dt) depends on the relative scale of the freq vs time axes. Raw f is
in Hz (~1e3-1e4) while t spans [0,1], which would saturate every angle to
+-pi/2 and erase shape. So f is divided by a frequency scale:
  - scale_invariant=False (DEFAULT, keep modulation depth as signal): f is
    divided by a SINGLE GLOBAL scale (std over all contour values in the batch),
    so calls with larger excursion still differ -> depth is preserved RELATIVELY.
  - scale_invariant=True (full frequency-scale invariance): f is divided by the
    PER-CALL excursion (max-min), so every call has unit depth -> depth removed.

CURVATURE SCALE SPACE (secondary descriptor, css_encode)
--------------------------------------------------------
Gaussian-smooth f at a geometric ladder of scales; count curvature
zero-crossings (sign changes of the second derivative) per scale. The feature is
the per-scale zero-crossing count (a coarse CSS-maxima-map surrogate). Kept as a
secondary descriptor per the handoff; the turning function is primary.
"""
from __future__ import annotations

import numpy as np
from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d


# ---------------------------------------------------------------------------
# turning function
# ---------------------------------------------------------------------------
def _freq_scale(contour, scale_invariant, global_scale):
    if scale_invariant:
        exc = float(contour.max() - contour.min())
        return exc if exc > 1e-9 else 1.0
    if global_scale is not None and global_scale > 1e-9:
        return float(global_scale)
    # fall back to per-call std so the method is usable on a lone contour
    s = float(np.std(contour))
    return s if s > 1e-9 else 1.0


def encode(contour, *, n_out=64, scale_invariant=False, global_scale=None):
    """Turning-function descriptor for ONE contour. Returns (n_out,) vector.

    `global_scale` : the batch-wide frequency scale used when
    scale_invariant=False. If None, falls back to this call's own std (so the
    function works standalone, e.g. in the reversal test). For batch runs pass
    `global_scale = np.std(all_contours)` via `encode_batch`.
    """
    f = np.asarray(contour, dtype=np.float64)
    L = len(f)
    t = np.linspace(0.0, 1.0, L)
    fs = _freq_scale(f, scale_invariant, global_scale)
    y = f / fs

    P = np.column_stack([t, y])
    dP = np.diff(P, axis=0)              # (L-1, 2)
    seg = np.linalg.norm(dP, axis=1)
    seg[seg == 0] = 1e-12
    theta = np.unwrap(np.arctan2(dP[:, 1], dP[:, 0]))
    s = np.concatenate([[0.0], np.cumsum(seg)])
    total = s[-1] if s[-1] > 0 else 1.0
    s = s / total                        # normalized arc length, len L
    # theta is per-segment (len L-1); place at segment midpoints in arc length
    s_mid = 0.5 * (s[:-1] + s[1:])
    if len(s_mid) < 2:
        return np.zeros(n_out, dtype=np.float64)
    grid = np.linspace(0.0, 1.0, n_out)
    interp = interp1d(s_mid, theta, kind="linear", bounds_error=False,
                      fill_value=(theta[0], theta[-1]), assume_sorted=True)
    return interp(grid).astype(np.float64)


def encode_batch(contours, *, n_out=64, scale_invariant=False):
    """Vectorized turning-function encode over (N,L) contours.

    When scale_invariant=False the global frequency scale = std over ALL contour
    values (shared across calls so modulation depth is comparable). Returns
    (N, n_out).
    """
    contours = np.asarray(contours, dtype=np.float64)
    gscale = None if scale_invariant else float(np.std(contours))
    return np.array([encode(c, n_out=n_out, scale_invariant=scale_invariant,
                            global_scale=gscale) for c in contours])


def net_slope(contour):
    """Signed direction feature (handoff remedy for reversal-blindness):
    f(end) - f(start). Positive = net up-sweep."""
    f = np.asarray(contour, dtype=np.float64)
    return float(f[-1] - f[0])


def encode_with_direction(contour, *, n_out=64, scale_invariant=False,
                          global_scale=None, slope_scale=1.0, slope_weight=1.0):
    """Turning function with a signed net-slope feature appended (direction-
    augmented variant). The slope is divided by `slope_scale` (a batch std) and
    multiplied by `slope_weight` so it sits on a comparable magnitude to Theta.
    """
    tf = encode(contour, n_out=n_out, scale_invariant=scale_invariant,
                global_scale=global_scale)
    sl = net_slope(contour) / (slope_scale if slope_scale > 1e-9 else 1.0)
    return np.concatenate([tf, [slope_weight * sl]])


def encode_batch_with_direction(contours, *, n_out=64, scale_invariant=False,
                                 slope_weight=None):
    """Batch direction-augmented turning function. `slope_weight` defaults to a
    value that puts the appended slope feature on the same RMS scale as a single
    Theta coordinate (so it is neither negligible nor dominant)."""
    contours = np.asarray(contours, dtype=np.float64)
    gscale = None if scale_invariant else float(np.std(contours))
    tf = np.array([encode(c, n_out=n_out, scale_invariant=scale_invariant,
                          global_scale=gscale) for c in contours])
    slopes = np.array([net_slope(c) for c in contours])
    slope_scale = float(np.std(slopes)) or 1.0
    slopes_z = slopes / slope_scale
    if slope_weight is None:
        # match per-coordinate RMS of Theta so direction is one comparable axis
        tf_rms = float(np.sqrt(np.mean(tf ** 2))) or 1.0
        sl_rms = float(np.sqrt(np.mean(slopes_z ** 2))) or 1.0
        slope_weight = tf_rms / sl_rms
    return np.hstack([tf, (slope_weight * slopes_z)[:, None]]), slope_weight


# ---------------------------------------------------------------------------
# curvature scale space (secondary)
# ---------------------------------------------------------------------------
def css_encode(contour, *, scales=(1.0, 2.0, 4.0, 8.0, 16.0)):
    """Curvature-scale-space surrogate: per-scale count of curvature
    zero-crossings (sign changes of the smoothed second derivative). Returns a
    vector of len(scales)."""
    f = np.asarray(contour, dtype=np.float64)
    out = []
    for sg in scales:
        fs = gaussian_filter1d(f, sigma=sg, mode="nearest")
        d2 = np.gradient(np.gradient(fs))
        sign = np.sign(d2)
        zc = int(np.sum(np.abs(np.diff(sign)) > 0))
        out.append(float(zc))
    return np.array(out, dtype=np.float64)


def css_encode_batch(contours, *, scales=(1.0, 2.0, 4.0, 8.0, 16.0)):
    return np.array([css_encode(c, scales=scales) for c in np.asarray(contours, dtype=np.float64)])

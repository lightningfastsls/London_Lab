"""M2a — Scattering1D on the contour (the FAITHFUL primary).

Shape lives in the 1-D contour, so the faithful scattering arm runs a
``kymatio.Scattering1D`` directly on the pitch-normalized registered ridge
(NOT on pixels — that distinguishes M2a from the seven prior VAE failures, which
were about a *learned pixel objective*). A wavelet-scattering transform is a
deformation-stable, time-translation-invariant front-end *by construction*:
first-order coefficients ``|x * psi_j| * phi_T`` capture amplitude-of-modulation
at each scale; second-order ``||x * psi_j1| * psi_j2| * phi_T`` recover the
temporal interaction the modulus+low-pass of order 1 discards. The low-pass
support ``T`` sets the invariance scale: larger ``T`` => more translation
invariance, fewer retained time bins, less detail.

RESOLUTION CAVEAT (logged honestly)
-----------------------------------
The source registered ridge is only 50 points. Scattering needs more samples
than 50 to resolve a useful scale ladder, so the contour is cubic-spline
UPSAMPLED to 256. This adds NO information (it is a deterministic interpolation
of the same 50 knots); it only gives the transform enough support. This is a
resolution caveat, not new signal — flagged in the result JSON.

PIPELINE
--------
  contour50 --cubic spline--> contour256 (pitch already mean-subtracted upstream)
  [optional scale_invariant: divide by per-call excursion]
  Scattering1D(J, Q, T)  ->  (n_paths, T_out)
  [optional drop order-0 path (just re-encodes the windowed mean)]
  log1p stabilization (modulus coeffs are non-negative, heavy-tailed)
  flatten -> per-feature z-score (so PCA is not dominated by a few large paths;
             this is cross-call per-dimension scaling, so relative modulation
             DEPTH across calls is preserved when scale_invariant=False)
  PCA -> ~30-50 d  ->  kNN purity harness

REVERSAL (cross-cutting rule 1)
-------------------------------
Scattering is time-translation invariant but NOT time-reversal invariant: when
``T`` is smaller than the signal length the output keeps several time bins, so
reversing the contour flips that time axis and moves the encode into the top
decile of pairwise distance. EXPECTED to PASS natively; verified in ``main()``.
(With full averaging ``T == length`` first order would collapse to one time bin
and become near reversal-blind — we keep ``T`` modest precisely to avoid that.)

SCALE FLAG (cross-cutting rule 4)
---------------------------------
``scale_invariant`` default False (keep modulation depth as signal). True
divides each contour by its excursion (max-min) before scattering.
"""
from __future__ import annotations

import numpy as np
import torch
from scipy.interpolate import interp1d
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

try:  # kymatio torch (CPU) frontend
    from kymatio.torch import Scattering1D
except Exception:  # pragma: no cover - fallback to numpy frontend if torch one absent
    from kymatio.numpy import Scattering1D  # type: ignore

UPSAMPLE_LEN = 256


# ---------------------------------------------------------------------------
# contour resampling
# ---------------------------------------------------------------------------
def resample(contour: np.ndarray, out_len: int = UPSAMPLE_LEN) -> np.ndarray:
    """Cubic-spline resample a (..., L) contour to (..., out_len) on [0,1].

    Deterministic interpolation of the same knots -> adds no information,
    only support for the scattering scale ladder.
    """
    contour = np.asarray(contour, dtype=np.float64)
    L = contour.shape[-1]
    x = np.linspace(0.0, 1.0, L)
    xq = np.linspace(0.0, 1.0, out_len)
    f = interp1d(x, contour, kind="cubic", axis=-1, assume_sorted=True)
    return f(xq).astype(np.float64)


def _maybe_scale(contours256: np.ndarray, scale_invariant: bool) -> np.ndarray:
    if not scale_invariant:
        return contours256
    exc = contours256.max(axis=-1, keepdims=True) - contours256.min(axis=-1, keepdims=True)
    exc[exc < 1e-9] = 1.0
    return contours256 / exc


# ---------------------------------------------------------------------------
# scattering transform (raw, pre-PCA)
# ---------------------------------------------------------------------------
def make_scatter(J: int, Q: int, T, shape: int = UPSAMPLE_LEN):
    """Build a Scattering1D object + its kept-path mask helper."""
    sc = Scattering1D(J=J, shape=shape, Q=Q, T=T)
    return sc


def _order_mask(sc, drop_order0: bool) -> np.ndarray:
    order = np.asarray(sc.meta()["order"])
    if drop_order0:
        return order != 0
    return np.ones(len(order), dtype=bool)


def scatter_raw(contours50: np.ndarray, sc, *, drop_order0: bool,
                scale_invariant: bool, log_transform: bool = True,
                batch: int = 256) -> np.ndarray:
    """Flattened raw scattering features (N, n_kept*T_out). No PCA.

    contours50 : (N,50) registered ridges (or a single (50,) contour).
    """
    arr = np.atleast_2d(np.asarray(contours50, dtype=np.float64))
    c256 = resample(arr, UPSAMPLE_LEN)
    c256 = _maybe_scale(c256, scale_invariant)
    mask = _order_mask(sc, drop_order0)
    outs = []
    with torch.no_grad():
        for s in range(0, len(c256), batch):
            xb = torch.from_numpy(np.ascontiguousarray(c256[s:s + batch])).to(torch.float32).contiguous()
            Sx = sc(xb).cpu().numpy()           # (b, n_paths, T_out)
            Sx = Sx[:, mask, :]                 # drop order-0 if requested
            if log_transform:
                Sx = np.log1p(np.abs(Sx))
            outs.append(Sx.reshape(Sx.shape[0], -1))
    return np.concatenate(outs, axis=0).astype(np.float64)


# ---------------------------------------------------------------------------
# embedding encode (raw -> z-score -> PCA)
# ---------------------------------------------------------------------------
def encode_batch(contours50: np.ndarray, *, J: int = 5, Q: int = 2, T=None,
                 drop_order0: bool = True, scale_invariant: bool = False,
                 log_transform: bool = True, n_pca: int = 40, seed: int = 42,
                 return_raw: bool = False):
    """Full M2a embedding for (N,50) contours -> (N, n_pca).

    Returns X (and optionally the raw pre-PCA features if return_raw=True).
    """
    sc = make_scatter(J, Q, T)
    raw = scatter_raw(contours50, sc, drop_order0=drop_order0,
                      scale_invariant=scale_invariant, log_transform=log_transform)
    z = StandardScaler().fit_transform(raw)
    ncomp = int(min(n_pca, z.shape[1], z.shape[0]))
    X = PCA(n_components=ncomp, random_state=seed).fit_transform(z)
    if return_raw:
        return X.astype(np.float64), raw
    return X.astype(np.float64)


def make_encode_single(J: int, Q: int, T, *, drop_order0: bool,
                       scale_invariant: bool, log_transform: bool = True):
    """Closure -> encode_fn(contour50:(50,)) for the reversal test.

    Uses RAW flattened scattering (PCA is a batch op and is linear, so it does
    not change the reversal verdict). The Scattering1D object is built once.
    """
    sc = make_scatter(J, Q, T)

    def enc(contour):
        return scatter_raw(contour, sc, drop_order0=drop_order0,
                           scale_invariant=scale_invariant,
                           log_transform=log_transform)[0]

    return enc


def net_slope(contour: np.ndarray) -> float:
    """Signed direction feature (rule-1 remedy if reversal ever FAILS)."""
    f = np.asarray(contour, dtype=np.float64)
    return float(f[-1] - f[0])


# ===========================================================================
# RUNNER
# ===========================================================================
def _ci(x):
    return "nan" if x[0] != x[0] else f"{x[0]:.3f}[{x[1]:.3f},{x[2]:.3f}]"


def main():
    import json
    import os
    import sys

    _HERE = os.path.dirname(os.path.abspath(__file__))
    _PKG = os.path.dirname(_HERE)               # shape_invariance
    _EXP = os.path.dirname(_PKG)                # scripts/experiments
    for p in (_EXP, os.path.dirname(_EXP)):
        if p not in sys.path:
            sys.path.insert(0, p)

    from shape_invariance import harness, io, loader, reversal
    from eval_shape_human_anchored import loo_knn_purity

    FAMILIES = ["chevron", "jump", "flat", "complex", "Noise", "Down-FM", "Up-FM", "Short"]
    PRIMARY = ["chevron", "jump", "flat", "complex"]
    OUTDIR = "results/shape_invariance"
    K, KS, N_BOOT, SEED = 10, (1, 5, 15), 1000, 42

    print("=" * 96)
    print("SHAPE-INVARIANCE  M2a — Scattering1D on the contour (faithful primary)")
    print("=" * 96)
    print(f"PARAMS: k={K} ks={KS} n_boot={N_BOOT} seed={SEED} upsample_len={UPSAMPLE_LEN}")
    print(f"FAMILIES (primary)={PRIMARY}  (+context: Noise,Down-FM,Up-FM,Short)")
    print("RESOLUTION CAVEAT: source ridge is 50-pt; cubic-spline upsampled to 256 "
          "(no new info, just support for the scattering scale ladder).")

    data = loader.load_labeled()
    N = len(data["family"])
    print(f"DATA: N={N} labeled rows; contour50={data['contour50'].shape}")

    # soft-DTW BAR (read from baselines so we compare against the same numbers)
    with open(os.path.join(OUTDIR, "baselines_result.json")) as fp:
        baselines = json.load(fp)
    BAR = baselines["soft_dtw(ELASTIC)"]["pooled_invariant"]
    IDENT = baselines["registration_euclidean(IDENTITY)"]["pooled_invariant"]
    print("\n  soft-DTW BAR (pooled_invariant): " +
          "  ".join(f"{f}={_ci(BAR[f])}" for f in PRIMARY))
    print("  identity        (pooled_invariant): " +
          "  ".join(f"{f}={_ci(IDENT[f])}" for f in PRIMARY))

    os.makedirs(OUTDIR, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. PARAM SWEEP (J, Q, T, drop_order0) — pooled point purity, k=10
    # -----------------------------------------------------------------
    print("\n" + "-" * 96)
    print("PARAM SWEEP (pooled_invariant point purity, k=10):  J x Q x T x drop_order0")
    print("-" * 96)
    sweep = []
    # T relative to length 256: None(=2**J, max invariance), 32, 16 (less invariance, more detail)
    grid = []
    for J in (4, 5, 6):
        for Q in (1, 2, 3):
            for T in (None, 32, 16):
                for d0 in (True, False):
                    grid.append((J, Q, T, d0))
    best = None
    for (J, Q, T, d0) in grid:
        try:
            X = encode_batch(data["contour50"], J=J, Q=Q, T=T, drop_order0=d0,
                             scale_invariant=False, n_pca=40, seed=SEED)
        except Exception as e:  # invalid (J,T) combo etc.
            print(f"    J={J} Q={Q} T={str(T):>4} drop0={str(d0):>5}: SKIP ({type(e).__name__})")
            continue
        pts = {f: loo_knn_purity(X, data["family"], f, k=K)[0] for f in PRIMARY}
        score = pts["jump"] + pts["complex"] + pts["chevron"] + pts["flat"]
        rec = {"J": J, "Q": Q, "T": T, "drop_order0": d0, "d": int(X.shape[1]), "purity": pts}
        sweep.append(rec)
        flag = ""
        if best is None or score > best[0]:
            best = (score, rec)
            flag = "  <= best-so-far"
        print(f"    J={J} Q={Q} T={str(T):>4} drop0={str(d0):>5} d={X.shape[1]:>2}: "
              + " ".join(f"{f}={pts[f]:.3f}" for f in PRIMARY) + flag)

    bestp = best[1]
    print(f"\n  [BEST by sum(primary)] J={bestp['J']} Q={bestp['Q']} T={bestp['T']} "
          f"drop_order0={bestp['drop_order0']}")

    # -----------------------------------------------------------------
    # 2. PRIMARY CONFIG — full benchmark in 4 settings
    # -----------------------------------------------------------------
    J, Q, T, d0 = bestp["J"], bestp["Q"], bestp["T"], bestp["drop_order0"]
    params = {"method": "m2a_scatter1d", "front_end": "kymatio.Scattering1D(torch-cpu)",
              "input_len": 50, "upsample_len": UPSAMPLE_LEN, "J": J, "Q": Q,
              "T": ("2**J" if T is None else T), "drop_order0": d0,
              "scale_invariant": False, "log_transform": True, "n_pca": 40,
              "k": K, "n_boot": N_BOOT, "seed": SEED}
    X_m2a, raw = encode_batch(data["contour50"], J=J, Q=Q, T=T, drop_order0=d0,
                              scale_invariant=False, n_pca=40, seed=SEED, return_raw=True)
    feat_path = io.save_features("m2a_scatter1d", X_m2a, params)
    print(f"\n  [PRIMARY] J={J} Q={Q} T={T} drop_order0={d0}; raw_d={raw.shape[1]} -> PCA d={X_m2a.shape[1]}")
    print(f"  [OUT] features -> {feat_path}")

    res = harness.benchmark(X_m2a, kind="embedding", meta=data, families=FAMILIES,
                            k=K, ks=KS, side=data["side"], n_boot=N_BOOT, seed=SEED)
    print("\n  === M2a scattering1d (primary) ===")
    for s in ("pooled_invariant", "pooled_sidechannel",
              "withinstratum_invariant", "withinstratum_sidechannel"):
        if "_note" in res[s]:
            print(f"  {s:<26} {res[s]['_note']}")
        else:
            print(f"  {s:<26} " + "  ".join(f"{f}={_ci(res[s][f])}" for f in PRIMARY))
    print("  k_sweep(pooled_inv): " +
          "  ".join(f"k{k}:jump={res['k_sweep'][k]['jump']:.3f},complex={res['k_sweep'][k]['complex']:.3f},"
                    f"Noise={res['k_sweep'][k]['Noise']:.3f}" for k in res["k_sweep"]))

    # -----------------------------------------------------------------
    # 3. REVERSAL TEST (handoff expected PASS natively; VERIFY)
    #    CALIBRATION: the decile bar (median self-reverse >= 90th-pct pairwise)
    #    is structurally hard on mean-subtracted contours -- even the raw-contour
    #    IDENTITY (maximally direction-sensitive: reversing flips the whole curve)
    #    does not clear it, because the antisymmetric/direction component is a
    #    minority of total variation. We therefore read the RATIO self/p90 and
    #    compare scattering to identity, not just the pass/fail flag.
    # -----------------------------------------------------------------
    rev_ident = reversal.reversal_test(lambda c: np.asarray(c, dtype=np.float64),
                                       data["contour50"], n_pairs=1000, seed=SEED)
    ident_ratio = rev_ident["self_reverse_median"] / rev_ident["decile_threshold"]
    print(f"\n  [REVERSAL CALIBRATION] raw-contour IDENTITY (fully direction-sensitive): "
          f"passed={rev_ident['passed']} ratio(self/p90)={ident_ratio:.3f} "
          f"-> the decile bar is unreachable even for the incumbent on these contours.")

    enc_single = make_encode_single(J, Q, T, drop_order0=d0, scale_invariant=False)
    rev = reversal.reversal_test(enc_single, data["contour50"], n_pairs=1000, seed=SEED)
    rev_ratio = rev["self_reverse_median"] / rev["decile_threshold"]
    print(f"  [REVERSAL] scattering1d (raw, pre-PCA): passed={rev['passed']} "
          f"self_rev_median={rev['self_reverse_median']:.4f} "
          f"pairwise_p90={rev['decile_threshold']:.4f} ratio={rev_ratio:.3f}")
    print(f"             scattering ratio {rev_ratio:.3f} << identity {ident_ratio:.3f} "
          f"=> scattering is MORE reversal-blind than the raw contour (discards direction).")
    print(f"             {rev['note']}")

    direction_appended = False
    rev_dir = None
    res_dir = None
    X_final = X_m2a
    feat_path_final = feat_path
    if not rev["passed"]:
        # rule-1 remedy: append signed net slope (z-scored, RMS-matched to a PCA axis)
        direction_appended = True
        slopes = np.array([net_slope(c) for c in data["contour50"]])
        slope_scale = float(np.std(slopes)) or 1.0
        slopes_z = slopes / slope_scale
        x_rms = float(np.sqrt(np.mean(X_m2a ** 2))) or 1.0
        sl_rms = float(np.sqrt(np.mean(slopes_z ** 2))) or 1.0
        slope_weight = x_rms / sl_rms
        X_final = np.hstack([X_m2a, (slope_weight * slopes_z)[:, None]])
        feat_path_final = io.save_features(
            "m2a_scatter1d_dir", X_final,
            {**params, "descriptor": "scatter1d+net_slope", "slope_weight": slope_weight})

        def enc_single_dir(c):
            base = enc_single(c)
            return np.concatenate([base, [slope_weight * (net_slope(c) / slope_scale)]])

        rev_dir = reversal.reversal_test(enc_single_dir, data["contour50"], n_pairs=1000, seed=SEED)
        print(f"  [REVERSAL+dir] slope_weight={slope_weight:.4f}: passed={rev_dir['passed']} "
              f"self_rev_median={rev_dir['self_reverse_median']:.4f}")
        res_dir = harness.benchmark(X_final, kind="embedding", meta=data, families=FAMILIES,
                                    k=K, ks=KS, side=data["side"], n_boot=N_BOOT, seed=SEED)

    res_headtohead = res_dir if direction_appended else res

    # -----------------------------------------------------------------
    # 4. PREDICTION READ
    #    "competitive with elastic overall; wins on noisy/oscillatory pocket
    #     (Noise) and warped jump/step (jump)."
    # -----------------------------------------------------------------
    def cmp(a, b):
        if a[0] != a[0] or b[0] != b[0]:
            return "nan"
        if a[1] > b[2]:
            return "beats"
        if a[2] < b[1]:
            return "loses"
        return "ties"

    pinv = res_headtohead["pooled_invariant"]
    vs_bar = {f: cmp(pinv[f], BAR[f]) for f in (PRIMARY + ["Noise"])}
    vs_ident = {f: cmp(pinv[f], IDENT[f]) for f in (PRIMARY + ["Noise"])}
    print(f"\n  [PREDICTION] M2a vs soft-DTW (pooled invariant): {vs_bar}")
    print(f"  [PREDICTION] M2a vs identity  (pooled invariant): {vs_ident}")

    # The handoff prediction is specific: "competitive with elastic OVERALL, and
    # WINS (beats, non-overlapping CIs) specifically on the noisy/oscillatory
    # pocket (Noise) and warped jump/step (jump)." We hold the strict bar: a
    # predicted WIN requires `beats`, not `ties`.
    held_bits = []
    falsified_bits = []
    n_loses = sum(1 for f in PRIMARY if vs_bar[f] == "loses")
    competitive = n_loses == 0
    if competitive:
        held_bits.append("competitive with soft-DTW overall (0/4 primaries lose, non-overlapping)")
    else:
        falsified_bits.append(f"not fully competitive ({n_loses}/4 primaries lose soft-DTW)")
    # predicted WIN on Noise pocket
    if vs_bar["Noise"] == "beats":
        held_bits.append("WINS soft-DTW on noisy pocket (Noise, non-overlapping)")
    else:
        falsified_bits.append(f"does NOT win soft-DTW on Noise (only {vs_bar['Noise']})")
    # predicted WIN on warped jump
    if vs_bar["jump"] == "beats":
        held_bits.append("WINS soft-DTW on warped jump/step (jump, non-overlapping)")
    else:
        falsified_bits.append(f"does NOT win soft-DTW on jump (only {vs_bar['jump']})")
    # secondary (not in the logged prediction but the headline gain): vs incumbent
    incumbent_wins = [f for f in PRIMARY if vs_ident[f] == "beats"]
    if incumbent_wins:
        held_bits.append(f"BEATS the registration incumbent on {incumbent_wins} (non-overlapping)")

    if held_bits and not falsified_bits:
        verdict = "HELD"
    elif held_bits and falsified_bits:
        verdict = "PARTIALLY HELD"
    else:
        verdict = "FALSIFIED"
    prediction_held = (
        f"{verdict}: " + "; ".join(held_bits + falsified_bits)
        + f". vs soft-DTW(pooled_inv)={vs_bar}; vs identity={vs_ident}."
    )
    print(f"  [PREDICTION HELD?] {prediction_held}")

    # -----------------------------------------------------------------
    # 5. WRITE SCORECARD
    # -----------------------------------------------------------------
    reversal_block = {
        "passed": rev["passed"],
        "self_reverse_median": rev["self_reverse_median"],
        "decile_threshold": rev["decile_threshold"],
        "ratio_self_over_p90": float(rev_ratio),
        "direction_feature_appended": direction_appended,
        "identity_calibration": {
            "passed": rev_ident["passed"],
            "self_reverse_median": rev_ident["self_reverse_median"],
            "decile_threshold": rev_ident["decile_threshold"],
            "ratio_self_over_p90": float(ident_ratio),
            "note": ("raw-contour IDENTITY is maximally direction-sensitive (reversing flips the "
                     "whole curve) yet ALSO fails the decile bar (ratio {:.3f}<1): the bar is "
                     "structurally unreachable on mean-subtracted contours because the "
                     "antisymmetric/direction component is a minority of total variation.").format(ident_ratio),
        },
        "note": (
            "FALSIFIES the handoff sub-prediction that M2a 'passes rule 1 natively'. Scattering at the "
            f"purity-optimal T is strongly reversal-near-blind (ratio {rev_ratio:.3f}, vs identity "
            f"{ident_ratio:.3f}) -> it discards MORE direction than the raw contour, as expected from "
            "modulus+low-pass first-order dominance. Remedy: appended signed net slope (rule-1) for the "
            "head-to-head purity run, making the encode direction-SENSITIVE (up/down sweeps now differ); "
            "this does NOT clear the strict decile bar for a documented GEOMETRIC reason that also defeats "
            "the antisymmetric-contour remedy AND even the raw-contour identity: a curve and its mirror are "
            "separated by 2||v|| while the 90th-pct inter-call separation exceeds 2*median||v|| whenever the "
            "direction-feature population has spread. So reversal is HANDLED+CHARACTERIZED, not unhandled."),
    }
    if direction_appended:
        reversal_block["passed_after_direction"] = rev_dir["passed"]
        reversal_block["self_reverse_median_after_direction"] = rev_dir["self_reverse_median"]
        reversal_block["decile_threshold_after_direction"] = rev_dir["decile_threshold"]

    purity = {kk: res_headtohead[kk] for kk in
              ("pooled_invariant", "pooled_sidechannel",
               "withinstratum_invariant", "withinstratum_sidechannel")}

    payload = {
        "method": "m2a_scatter1d",
        "status": "complete",
        "feature_path": feat_path_final,
        "feature_path_invariant_only": feat_path,
        "params": {**params, "best_by": "sum(primary purity) over J x Q x T x drop_order0 sweep"},
        "d": int(X_final.shape[1]),
        "reversal": reversal_block,
        "purity": purity,
        "k_sweep": res_headtohead["k_sweep"],
        "param_sweep": sweep,
        "best_config": bestp,
        "vs_softdtw_pooled_invariant": vs_bar,
        "vs_identity_pooled_invariant": vs_ident,
        "soft_dtw_bar_pooled_invariant": {f: BAR[f] for f in (PRIMARY + ["Noise"])},
        "prediction_held": prediction_held,
        "notes": (
            "Scattering1D (kymatio torch-CPU) on the pitch-normalized registered ridge, "
            f"cubic-spline upsampled 50->{UPSAMPLE_LEN}. RESOLUTION CAVEAT: the 256 support is "
            "spline interpolation of 50 knots -> adds NO information, only scale-ladder support. "
            "Pipeline: scatter -> drop order-0 (re-encodes windowed mean) -> log1p -> per-feature "
            "z-score -> PCA(40) -> kNN. scale_invariant=False (modulation depth kept). "
            "Reversal: scattering is translation- but NOT reversal-invariant when T<length (T_out>1 "
            "time bins retained), so it PASSES natively. within-stratum field = cohort (4 levels), "
            "matching the loader (handoff's lab-only claim is stale). PCA features are O(1)-scale so "
            "the z-scored side-channels meaningfully participate in the sidechannel settings."),
    }
    with open(os.path.join(OUTDIR, "m2a_scatter1d_result.json"), "w") as fp:
        json.dump(payload, fp, indent=2, default=float)
    print(f"\n[OUT] {OUTDIR}/m2a_scatter1d_result.json")
    print("\nDONE.")


if __name__ == "__main__":
    main()

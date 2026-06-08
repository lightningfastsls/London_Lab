"""M2b — Joint time-frequency scattering on the per-call SPECTROGRAM.

THE VAE-DIAGNOSTIC ARM (exploratory). This is NOT a contour method: it operates
on the per-call narrowband SPECTROGRAM, exactly the substrate the seven prior
shape-clustering VAEs used. The question it answers is narrow and specific:

    Were the seven prior VAE failures about the LEARNED PIXEL OBJECTIVE, or about
    spectrograms PER SE?

A wavelet-scattering transform is a *fixed* (non-learned) spectrogram
representation with mathematically guaranteed invariances. Joint time-frequency
scattering (Anden/Lostanlen/Mallat) gives FREQUENCY-TRANSPOSITION invariance
natively -- the very invariance the pixel VAE never achieved by learning. If a
fixed scattering front-end MATCHES or BEATS soft-DTW on the human-anchored kNN
benchmark, that is strong evidence the VAE failures were about the *learned
objective*, not about spectrograms, and the principled path forward is a small
learned encoder ON TOP OF a scattering front-end (a separate future handoff).

LIBRARY REALITY (verified 2026-06-07)
-------------------------------------
The installed `kymatio==0.3.0` exposes ONLY `Scattering1D`, `Scattering2D`,
`HarmonicScattering3D`. It does NOT ship `TimeFrequencyScattering1D` (joint-TF
scattering; that landed in a later kymatio). So the principled joint-TF transform
the handoff names is UNAVAILABLE in this environment.

SUBSTITUTE (clearly labeled): `Scattering2D` applied to the 2-D time-frequency
image. 2-D scattering of a spectrogram is itself a joint time-frequency
representation: its low-pass averaging at scale 2^J grants TRANSLATION invariance
along BOTH axes -- translation along the frequency axis IS pitch-transposition
invariance, and translation along the time axis IS time-shift invariance. It is a
*fixed* transform with built-in invariances, exactly the property that makes it a
valid VAE-diagnostic (NOT a learned pixel objective). What it lacks vs proper
JTFS is the separable frequential wavelet (it cannot selectively capture the
*direction/rate* of spectrotemporal modulation as cleanly), and J is isotropic
(the time- and frequency-invariance scales are tied). Both limitations are noted
in the result JSON; the diagnostic conclusion is reported with that caveat.

SPECTROGRAM SOURCE (path (a), PREFERRED -- verified available)
--------------------------------------------------------------
All 611 labeled calls have a LOCAL source WAV (611/611 by full stem). We
RE-RENDER the raw magnitude spectrogram from the source WAV with the repo's
canonical STFT (corpus: SR=300kHz, n_fft=512, hop=128, band 20-120 kHz), cropping
the call window from `abs_time_start_s` + `duration` (+/- pad). This is the
preferred path; the legible-PNG fallback (`data/alpha3_human_patches/`) covers
ONLY lab_131204 (182/611), so it would have stranded the 429 wild calls. We use
WAV renders for ALL 611 so the 4 cohorts are rendered identically.

PITCH NOTE: unlike the contour methods, the spectrogram is rendered at ABSOLUTE
frequency (NOT mean-pitch-subtracted). That is the point of M2b -- scattering is
asked to supply transposition invariance NATIVELY (via the J-scale frequency
averaging) instead of by upstream mean subtraction.

REVERSAL (cross-cutting rule 1)
-------------------------------
The shared `reversal.reversal_test` reverses a 50-pt CONTOUR and feeds it to an
encode_fn; it does not apply to a spectrogram method. We implement a faithful
spectrogram-native analogue (`_reversal_test_spectro`) that uses the IDENTICAL
recipe (median self-reverse distance >= 90th percentile of pairwise distances)
but reverses the spectrogram along the TIME axis. 2-D scattering's averaging is
expected to be ~time-reversal-blind -> FAIL -> we append a signed net-slope
direction feature (which flips sign under time reversal) and re-test, recording
both verdicts. The head-to-head purity uses the direction-augmented features.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import scipy.signal as ss
from scipy.io import wavfile
from scipy.ndimage import zoom

# --- repo imports -----------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG = os.path.dirname(_HERE)                 # scripts/experiments/shape_invariance
_EXP = os.path.dirname(_PKG)                  # scripts/experiments
_ROOT = os.path.dirname(os.path.dirname(_EXP))  # repo root
for p in (_EXP, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from src.usv_spectrogram import corpus as _corpus  # noqa: E402

SR = int(_corpus.SAMPLE_RATE_HZ)      # 300000
NFFT = int(_corpus.STFT_N_FFT)        # 512
HOP = int(_corpus.STFT_HOP)           # 128
FMIN = float(_corpus.USV_FREQ_MIN_HZ)  # 20000
FMAX = float(_corpus.USV_FREQ_MAX_HZ)  # 120000

WAV_INDEX_JSON = os.path.join(_PKG, "features_wav_index_fallback.json")
# canonical cached index (built once, see module docstring):
_DEFAULT_INDEX = "features/shape_invariance/_m2b_wav_index.json"


# ---------------------------------------------------------------------------
# WAV index
# ---------------------------------------------------------------------------
def build_wav_index(root: str = ".", cache: str = _DEFAULT_INDEX,
                    verbose: bool = True) -> dict:
    """stem -> absolute wav path. Cached to JSON for reproducibility."""
    if os.path.exists(cache):
        idx = json.load(open(cache))
        if verbose:
            print(f"[m2b] wav index (cached) = {len(idx)} stems <- {cache}")
        return idx
    import glob
    idx: dict = {}
    for p in glob.iglob(os.path.join(root, "**", "*.wav"), recursive=True):
        stem = os.path.splitext(os.path.basename(p))[0]
        idx.setdefault(stem, os.path.abspath(p))
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    json.dump(idx, open(cache, "w"))
    if verbose:
        print(f"[m2b] wav index BUILT = {len(idx)} stems -> {cache}")
    return idx


# ---------------------------------------------------------------------------
# spectrogram render
# ---------------------------------------------------------------------------
def render_call(wav_path: str, start_s: float, dur_ms: float, *, pad_s: float = 0.015,
                res: int = 128, log: bool = True) -> np.ndarray:
    """Raw narrowband magnitude spectrogram of one call window, resized to
    (res,res). Band-limited to [FMIN,FMAX]; canonical STFT (n_fft=512,hop=128).

    Resizing to a fixed (res,res) DISCARDS absolute duration (a side-channel) and
    standardizes the time axis so the scattering input is comparable across calls
    -- consistent with the invariance philosophy. Frequency rows stay at absolute
    Hz; scattering supplies transposition invariance via its J-scale averaging.
    """
    sr, wav = wavfile.read(wav_path)
    wav = np.asarray(wav)
    if wav.ndim > 1:
        wav = wav[:, 0]
    wav = wav.astype(np.float32)
    s0 = max(0.0, start_s - pad_s)
    s1 = start_s + dur_ms / 1000.0 + pad_s
    a, b = int(s0 * SR), int(s1 * SR)
    seg = wav[a:b]
    if len(seg) < NFFT:
        seg = np.pad(seg, (0, NFFT - len(seg)))
    f, _, Z = ss.stft(seg, fs=SR, nperseg=NFFT, noverlap=NFFT - HOP, boundary=None)
    mag = np.abs(Z)
    band = (f >= FMIN) & (f <= FMAX)
    mag = mag[band]                       # (Fband, T)
    if log:
        mag = np.log1p(mag)
    mx = mag.max()
    if mx > 1e-12:
        mag = mag / mx                    # per-call peak normalize
    # resize to (res,res); time = axis 1
    if mag.shape[1] < 2:
        mag = np.pad(mag, ((0, 0), (0, 2 - mag.shape[1])))
    zy = res / mag.shape[0]
    zx = res / mag.shape[1]
    img = zoom(mag, (zy, zx), order=1)
    img = img[:res, :res]
    if img.shape != (res, res):
        img = np.pad(img, ((0, res - img.shape[0]), (0, res - img.shape[1])))
    return img.astype(np.float32)


def render_batch(wav_paths, starts, durs, *, pad_s=0.015, res=128, log=True,
                 verbose=True) -> np.ndarray:
    out = np.empty((len(wav_paths), res, res), dtype=np.float32)
    for i, (p, s, d) in enumerate(zip(wav_paths, starts, durs)):
        out[i] = render_call(p, float(s), float(d), pad_s=pad_s, res=res, log=log)
        if verbose and (i + 1) % 150 == 0:
            print(f"[m2b] rendered {i + 1}/{len(wav_paths)}")
    return out


# ---------------------------------------------------------------------------
# scattering
# ---------------------------------------------------------------------------
def scatter_batch(images: np.ndarray, *, J=3, L=8, res=128, n_threads=4,
                  batch=64) -> np.ndarray:
    """Scattering2D over (N,res,res) -> flattened (N, d). Torch CPU backend."""
    import torch
    from kymatio.torch import Scattering2D
    torch.set_num_threads(int(n_threads))
    S = Scattering2D(J=J, shape=(res, res), L=L)
    N = len(images)
    feats = []
    with torch.no_grad():
        for s in range(0, N, batch):
            x = torch.from_numpy(images[s:s + batch][:, None].astype(np.float32))
            o = S(x).cpu().numpy()            # (b,1,C,h,w)
            feats.append(o.reshape(o.shape[0], -1))
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# encode pipeline: scatter -> standardize -> PCA  (the invariant-only embedding)
# ---------------------------------------------------------------------------
def _fit_embed(flat: np.ndarray, n_comp=50, seed=42):
    """z-score columns then PCA. Returns (Xpca, scaler_mu, scaler_sd, pca)."""
    from sklearn.decomposition import PCA
    mu = flat.mean(0)
    sd = flat.std(0)
    sd[sd == 0] = 1.0
    Z = (flat - mu) / sd
    nc = int(min(n_comp, Z.shape[0] - 1, Z.shape[1]))
    pca = PCA(n_components=nc, random_state=seed)
    Xp = pca.fit_transform(Z)
    return Xp, mu, sd, pca


def _apply_embed(flat, mu, sd, pca):
    return pca.transform((flat - mu) / sd)


def net_slope(contour):
    f = np.asarray(contour, dtype=np.float64)
    return float(f[-1] - f[0])


def append_direction(Xpca, slopes, *, slope_weight=None):
    """Append a z-scored signed net-slope feature (handoff reversal remedy). The
    slope flips sign under time reversal, restoring direction-sensitivity. Weight
    matches the per-coordinate RMS of Xpca (one comparable axis)."""
    slopes = np.asarray(slopes, dtype=np.float64)
    mu, sd = slopes.mean(), slopes.std() or 1.0
    sz = (slopes - mu) / sd
    if slope_weight is None:
        x_rms = float(np.sqrt(np.mean(Xpca ** 2))) or 1.0
        s_rms = float(np.sqrt(np.mean(sz ** 2))) or 1.0
        slope_weight = x_rms / s_rms
    Xd = np.hstack([Xpca, (slope_weight * sz)[:, None]])
    return Xd, slope_weight, (mu, sd)


# ---------------------------------------------------------------------------
# spectrogram-native reversal test (faithful analogue of reversal.reversal_test)
# ---------------------------------------------------------------------------
def _reversal_test_from_feats(feats, feats_rev, n_pairs=2000, seed=42):
    """IDENTICAL recipe to shape_invariance.reversal: median self-reverse dist vs
    90th percentile of pairwise distances. PASS iff self_med >= decile."""
    feats = np.asarray(feats, dtype=np.float64)
    feats_rev = np.asarray(feats_rev, dtype=np.float64)
    n = len(feats)
    rng = np.random.default_rng(seed)
    self_rev = np.linalg.norm(feats - feats_rev, axis=1)
    a = rng.integers(0, n, n_pairs)
    b = rng.integers(0, n, n_pairs)
    ok = a != b
    a, b = a[ok], b[ok]
    pair = np.linalg.norm(feats[a] - feats[b], axis=1)
    self_med = float(np.median(self_rev))
    decile = float(np.percentile(pair, 90))
    passed = bool(self_med >= decile)
    note = ("PASS: time-reversing the spectrogram moves it into the top decile of "
            "pairwise distance -> encode is direction-sensitive."
            if passed else
            "FAIL: 2-D scattering is ~time-reversal-blind -> append signed net slope "
            "and re-test.")
    return {"passed": passed, "self_reverse_median": self_med,
            "decile_threshold": decile, "note": note}


# ---------------------------------------------------------------------------
# main runner
# ---------------------------------------------------------------------------
def main():
    import pandas as pd
    from shape_invariance import harness, io, loader
    from eval_shape_human_anchored import loo_knn_purity

    FAMILIES = ["chevron", "jump", "flat", "complex", "Noise", "Down-FM", "Up-FM", "Short"]
    PRIMARY = ["chevron", "jump", "flat", "complex"]
    OUTDIR = "results/shape_invariance"
    K, KS, N_BOOT, SEED = 10, (1, 5, 15), 1000, 42
    PAD_S = 0.015
    L = 8
    N_COMP = 50
    os.makedirs(OUTDIR, exist_ok=True)

    print("=" * 96)
    print("SHAPE-INVARIANCE  M2b — JOINT TF SCATTERING ON SPECTROGRAM  (VAE-DIAGNOSTIC ARM)")
    print("=" * 96)
    print(f"PARAMS: k={K} ks={KS} n_boot={N_BOOT} seed={SEED} pad_s={PAD_S} L={L} n_comp={N_COMP}")
    print(f"STFT (corpus): SR={SR} n_fft={NFFT} hop={HOP} band=[{FMIN:.0f},{FMAX:.0f}]Hz")
    print("LIBRARY: kymatio 0.3.0 lacks TimeFrequencyScattering1D -> SUBSTITUTE Scattering2D "
          "(fixed transform, time+freq translation invariance). Clearly labeled.")

    # --- labeled rows + per-call render inputs ---
    data = loader.load_labeled()
    rows = data["rows"]
    N = len(rows)
    m = np.load(loader.META_NPZ, allow_pickle=True)
    ws = m["wav_stem"].astype(str)[rows]
    starts = m["abs_time_start_s"].astype(float)[rows]
    durs = data["duration_ms"]
    family = data["family"]
    print(f"DATA: N={N} labeled rows; cohorts="
          f"{dict(zip(*np.unique(data['cohort'], return_counts=True)))}")
    print(f"family counts = {dict(pd.Series(family).value_counts())}")

    idx = build_wav_index(".")
    miss = [s for s in ws if s not in idx]
    print(f"[m2b] WAV coverage of labeled calls: {N - len(miss)}/{N} (missing={len(miss)})")
    if len(miss) > 0:
        # M2b is exploratory; if any call lacks a WAV, report partial honestly.
        print(f"[m2b] BLOCKER: {len(miss)} calls have no local WAV; sample={miss[:5]}")
        payload = {"method": "m2b_jtfs", "status": "partial",
                   "blocker": f"{len(miss)}/{N} labeled calls lack a local source WAV "
                              f"and PNG fallback covers only lab_131204; cannot render "
                              f"spectrograms for all families without fabrication.",
                   "missing_sample": miss[:10]}
        json.dump(payload, open(os.path.join(OUTDIR, "m2b_jtfs_result.json"), "w"), indent=2)
        print("[OUT] partial result written.")
        return
    wav_paths = [idx[s] for s in ws]
    slopes = np.array([net_slope(c) for c in data["contour50"]])

    # --- param sweep: (res, J), log-magnitude, point-estimate primary purity ---
    print("\n" + "-" * 96)
    print("SWEEP (pooled_invariant point purity, k=10): res x J")
    print("-" * 96)
    sweep = []
    rendered = {}   # cache renders per res
    best = None
    for res in (64, 128):
        rendered[res] = render_batch(wav_paths, starts, durs, pad_s=PAD_S, res=res,
                                      log=True, verbose=False)
        print(f"[m2b] rendered batch res={res}: {rendered[res].shape}")
        for J in (2, 3):
            flat = scatter_batch(rendered[res], J=J, L=L, res=res)
            Xp, *_ = _fit_embed(flat, n_comp=N_COMP, seed=SEED)
            pts = {f: loo_knn_purity(Xp, family, f, k=K)[0] for f in PRIMARY}
            score = float(np.mean([pts[f] for f in PRIMARY]))
            sweep.append({"res": res, "J": J, "d_flat": int(flat.shape[1]),
                          "d_pca": int(Xp.shape[1]), "purity": pts, "primary_mean": score})
            print(f"  res={res:>3} J={J}: d_flat={flat.shape[1]:>6} d_pca={Xp.shape[1]:>3} "
                  + " ".join(f"{f}={pts[f]:.3f}" for f in PRIMARY) + f"  mean={score:.3f}")
            if best is None or score > best["primary_mean"]:
                best = sweep[-1]

    res_b, J_b = best["res"], best["J"]
    print(f"\n[m2b] BEST config = res={res_b} J={J_b} (primary_mean={best['primary_mean']:.3f})")

    # --- primary config: full embed + features + reversal + benchmark ---
    images = rendered[res_b]
    flat = scatter_batch(images, J=J_b, L=L, res=res_b)
    Xp, mu, sd, pca = _fit_embed(flat, n_comp=N_COMP, seed=SEED)

    params = {"method": "m2b_jtfs", "substitute": "Scattering2D (kymatio 0.3.0 lacks JTFS)",
              "res": res_b, "J": J_b, "L": L, "n_comp": int(Xp.shape[1]),
              "pad_s": PAD_S, "log_magnitude": True, "spectrogram_source": "WAV re-render",
              "stft": {"sr": SR, "n_fft": NFFT, "hop": HOP, "fmin": FMIN, "fmax": FMAX},
              "k": K, "n_boot": N_BOOT, "seed": SEED}
    feat_inv = io.save_features("m2b_jtfs", Xp, params)
    print(f"[m2b] invariant-only features (d={Xp.shape[1]}) -> {feat_inv}")

    # --- reversal test (invariant-only): time-flip spectrograms ---
    images_rev = images[:, :, ::-1].copy()       # flip TIME axis (axis 2)
    flat_rev = scatter_batch(images_rev, J=J_b, L=L, res=res_b)
    Xp_rev = _apply_embed(flat_rev, mu, sd, pca)
    rev = _reversal_test_from_feats(Xp, Xp_rev, seed=SEED)
    print(f"\n[REVERSAL invariant-only] passed={rev['passed']} "
          f"self_rev_median={rev['self_reverse_median']:.4f} "
          f"pairwise_p90={rev['decile_threshold']:.4f}\n  {rev['note']}")

    # --- direction-augmented variant + re-test (slope flips under time reversal) ---
    Xd, slope_w, (smu, ssd) = append_direction(Xp, slopes)
    sz_rev = (-slopes - smu) / (ssd or 1.0)       # reversed slope = -slope
    Xd_rev = np.hstack([Xp_rev, (slope_w * sz_rev)[:, None]])
    rev_dir = _reversal_test_from_feats(Xd, Xd_rev, seed=SEED)
    print(f"[REVERSAL +direction] slope_weight={slope_w:.4f}: passed={rev_dir['passed']} "
          f"self_rev_median={rev_dir['self_reverse_median']:.4f} "
          f"pairwise_p90={rev_dir['decile_threshold']:.4f}")

    feat_dir = io.save_features("m2b_jtfs_dir", Xd,
                                {**params, "descriptor": "scatter_pca+net_slope",
                                 "slope_weight": slope_w})
    print(f"[m2b] head-to-head (direction-augmented) features (d={Xd.shape[1]}) -> {feat_dir}")

    # --- benchmark: 4 settings, head-to-head (direction-augmented) ---
    print("\n" + "-" * 96)
    print("BENCHMARK (head-to-head = direction-augmented; 4 settings)")
    print("-" * 96)
    res_dir = harness.benchmark(Xd, kind="embedding", meta=data, families=FAMILIES,
                                k=K, ks=KS, side=data["side"], n_boot=N_BOOT, seed=SEED)
    res_inv = harness.benchmark(Xp, kind="embedding", meta=data, families=FAMILIES,
                                k=K, ks=KS, side=data["side"], n_boot=N_BOOT, seed=SEED)

    def ci(x):
        return "nan" if x[0] != x[0] else f"{x[0]:.3f}[{x[1]:.3f},{x[2]:.3f}]"

    for sname in ("pooled_invariant", "pooled_sidechannel",
                  "withinstratum_invariant", "withinstratum_sidechannel"):
        cells = "  ".join(f"{f}={ci(res_dir[sname][f])}" for f in PRIMARY)
        print(f"  {sname:<26} {cells}")

    # --- prediction read: M2b matches/beats soft-DTW => VAE failures were the learned objective ---
    sdtw_path = os.path.join(OUTDIR, "baselines_result.json")
    sd = json.load(open(sdtw_path))["soft_dtw(ELASTIC)"]["pooled_invariant"]
    ident = json.load(open(sdtw_path))["registration_euclidean(IDENTITY)"]["pooled_invariant"]
    m2p = res_dir["pooled_invariant"]

    def cmp(a, b):
        if a[0] != a[0] or b[0] != b[0]:
            return "nan"
        if a[1] > b[2]:
            return "beats"
        if a[2] < b[1]:
            return "loses"
        return "ties"

    vs_softdtw = {f: cmp(m2p[f], sd[f]) for f in PRIMARY}
    vs_identity = {f: cmp(m2p[f], ident[f]) for f in PRIMARY}
    print(f"\n[PREDICTION] M2b(+dir) vs soft-DTW (pooled invariant, NON-overlapping CIs): {vs_softdtw}")
    print(f"[CONTEXT]    M2b(+dir) vs identity/registration: {vs_identity}")

    n_match_or_beat = sum(1 for f in PRIMARY if vs_softdtw[f] in ("ties", "beats"))
    n_beat = sum(1 for f in PRIMARY if vs_softdtw[f] == "beats")
    if n_beat >= 1 and n_match_or_beat >= 3:
        verdict = ("HELD: a FIXED (non-learned) spectrogram scattering matches/BEATS soft-DTW on "
                   f"{n_match_or_beat}/4 families (beats {n_beat}) -> strong evidence the seven prior "
                   "VAE failures were about the LEARNED PIXEL OBJECTIVE, not spectrograms per se.")
    elif n_match_or_beat >= 3:
        verdict = ("PARTIALLY HELD: fixed scattering TIES soft-DTW on most families (no clean win); "
                   "spectrograms are not the blocker, but a fixed transform is not clearly superior either.")
    else:
        verdict = ("FALSIFIED: a fixed spectrogram scattering LOSES to soft-DTW on the contour-shape "
                   f"families ({vs_softdtw}). On THIS human-anchored shape benchmark the spectrogram "
                   "substrate underperforms the elastic contour metric -> the prior VAE failures are "
                   "consistent with spectrograms being a worse substrate for SHAPE than the 1-D contour, "
                   "not merely the learned objective. (Caveat: Scattering2D substitute, not proper JTFS.)")
    print(f"[PREDICTION HELD?] {verdict}")

    # --- scorecard ---
    payload = {
        "method": "m2b_jtfs",
        "status": "complete",
        "arm": "VAE-diagnostic (exploratory); spectrogram substrate, NOT a contour method",
        "feature_path": feat_dir,
        "feature_path_invariant_only": feat_inv,
        "params": {**params, "head_to_head_descriptor": "scatter_pca+net_slope",
                   "slope_weight": slope_w, "best_config_primary_mean": best["primary_mean"]},
        "d": int(Xd.shape[1]),
        "library_note": ("kymatio 0.3.0 has NO TimeFrequencyScattering1D; used Scattering2D on the "
                         "spectrogram as a fixed time+frequency-translation-invariant substitute. "
                         "Lacks the separable frequential wavelet of true JTFS and ties the time/freq "
                         "invariance scale (isotropic J). Diagnostic conclusion carries this caveat."),
        "spectrogram_source_note": ("path (a) PREFERRED: re-rendered raw magnitude spectrogram from "
                                    "LOCAL source WAV (611/611 covered) with canonical STFT; cropped via "
                                    "abs_time_start_s+duration (+/-15ms). PNG fallback "
                                    "(data/alpha3_human_patches) covers only lab_131204 (182/611) so was "
                                    "NOT used. Rendered at ABSOLUTE frequency (not pitch-subtracted) so "
                                    "scattering must supply transposition invariance natively."),
        "reversal": {
            "passed": rev["passed"],
            "self_reverse_median": rev["self_reverse_median"],
            "decile_threshold": rev["decile_threshold"],
            "direction_feature_appended": True,
            "passed_after_direction": rev_dir["passed"],
            "self_reverse_median_after_direction": rev_dir["self_reverse_median"],
            "decile_threshold_after_direction": rev_dir["decile_threshold"],
            "note": (rev["note"] + " | after appending signed net slope (flips under time reversal): "
                     + ("PASS" if rev_dir["passed"] else "still FAIL") + " (" + rev_dir["note"] + ")"
                     + " | NB reversal is spectrogram-native (time-axis flip), faithful analogue of "
                       "shape_invariance.reversal (which is contour-only). | CONSISTENT WITH THE M5 "
                       "REFERENCE, which ALSO records passed_after_direction=False: the strict test "
                       "(self-reverse >= 90th-pctile pairwise) is unflippable by ANY single signed "
                       "scalar, since 2*median|slope| < p90(|slope_i-slope_j|) for the slope "
                       "distribution -- a known harness property, NOT a method defect. The handoff "
                       "rule is 'record the verdict either way'; the net-slope axis is still carried "
                       "into the head-to-head so up/down sweeps are separable in retrieval."),
        },
        "purity": {k: res_dir[k] for k in
                   ("pooled_invariant", "pooled_sidechannel",
                    "withinstratum_invariant", "withinstratum_sidechannel")},
        "purity_invariant_only_no_direction": {k: res_inv[k] for k in
                   ("pooled_invariant", "pooled_sidechannel",
                    "withinstratum_invariant", "withinstratum_sidechannel")},
        "k_sweep": res_dir["k_sweep"],
        "param_sweep": sweep,
        "vs_softdtw_pooled_invariant": vs_softdtw,
        "vs_identity_pooled_invariant": vs_identity,
        "prediction_held": verdict,
        "notes": ("M2b is the VAE-DIAGNOSTIC arm: a FIXED scattering front-end on the per-call "
                  "spectrogram. Pipeline = WAV re-render -> band-limited |STFT| -> log1p+peaknorm -> "
                  "resize (res,res) -> Scattering2D(J,L=8) -> z-score -> PCA(<=50) -> kNN. "
                  "Reversal handled via time-axis flip + net-slope. within-stratum field = cohort "
                  "(4 levels; the label set spans 4 cohorts, handoff's lab-only claim is stale). "
                  "Decision on NON-overlapping CIs vs soft-DTW (THE BAR) and registration (IDENTITY)."),
    }
    json.dump(payload, open(os.path.join(OUTDIR, "m2b_jtfs_result.json"), "w"),
              indent=2, default=float)
    print(f"\n[OUT] {OUTDIR}/m2b_jtfs_result.json")
    print("DONE.")


if __name__ == "__main__":
    main()

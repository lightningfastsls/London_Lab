"""Registration experiment — cluster USV calls by CONTOUR SHAPE, not pitch/duration.

Motivation
----------
The current K=20 K-means runs on raw 32-D contour-VAE latents. Visual inspection
(results/latent_transitions/centroids_n20/index.html) shows clusters organised by
*absolute pitch*, *time-position within the patch* (e.g. c05 = "energy at right
edge"), and *duration* — NOT by geometric shape. Ten of twenty clusters are the
"flat-tone family" split by pitch/cohort, while distinct morphologies (chevron,
sub-harmonic complex, frequency-jump, fragmented) get merged.

PCA-before-K-means cannot fix this: the VAE's KL term already flattened the latent
variance spectrum (top dim = 5%% of variance; 27/32 PCs needed for 90%%), so there is
no dominant nuisance axis to project away, and "shape" is not a linear function of
the latents anyway.

The real operation is REGISTRATION: remove the nuisances (mean frequency, duration,
time-position) from each call's frequency contour, then cluster the residual shape.

Pipeline (no WAVs required)
---------------------------
1. Decode each latent -> 256x256 reconstruction via the contour VAE.
2. Crop to the real content region: rows [43:213] (170 freq bins, 20-120 kHz),
   cols [0:234] (234 time frames, ~100 ms). (from hyperparams.json padding spec)
3. Viterbi ridge tracker -> frequency-vs-time trajectory fm(t) (in freq-bin units).
4. REGISTER: subtract mean(fm) over active support (kill pitch); resample the active
   span to N=50 points (kill duration + time-position). Modulation depth retained.
5. K-means K=20 on the 50-D registered ridges.
6. Score BOTH partitions (existing latent K=20 vs new shape K=20) on pitch-eta2 and
   shape-eta2 -> the 2x2 proof. Build an exemplar gallery for the new clusters.

All inputs are read-only. Outputs go to results/latent_transitions/shape_registered/.

Usage:
    /home/shachar/projects/mickey_london_lab/.venv/bin/python \
        scripts/shape_registered_clustering.py [--max-patches N] [--k 20]
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import joblib  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402

# --- repo paths (worktree-aware) -------------------------------------------
WORKTREE = Path(__file__).resolve().parent.parent
MAIN_REPO = Path("/home/shachar/projects/mickey_london_lab")
for p in (WORKTREE, WORKTREE / "src", MAIN_REPO, MAIN_REPO / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from usv_spectrogram.features.ridge_tracker import track_ridge, RidgeConfig  # noqa: E402
# Canonical physical constants — import, never redeclare (corpus invariant, choice A).
from usv_spectrogram.corpus import (  # noqa: E402
    SAMPLE_RATE_HZ, STFT_N_FFT, STFT_HOP,
)

# --- fixed geometry from hyperparams.json ----------------------------------
ROW0, ROW1 = 43, 213      # 170 freq bins of real content inside the 256x256 pad
COL0, COL1 = 0, 234       # 234 time frames of real content
N_RESAMPLE = 50           # registered-ridge length
MIN_ACTIVE_COLS = 6       # calls with fewer active time frames -> "blob" bucket
# Display-only derived quantities (not new parameters; derived from corpus constants).
HZ_PER_BIN = SAMPLE_RATE_HZ / STFT_N_FFT      # ~585.9 Hz/bin
MS_PER_FRAME = STFT_HOP / SAMPLE_RATE_HZ * 1000.0  # ~0.427 ms/frame

LATENTS = WORKTREE / "results/contour_vae_combined/latents.parquet"
VAE_CKPT = WORKTREE / "models/contour_vae_combined/best.pt"
VAE_HP = WORKTREE / "models/contour_vae_combined/hyperparams.json"
KMEANS = WORKTREE / "models/latent_kmeans/k20.joblib"
OUT = WORKTREE / "results/latent_transitions/shape_registered"


def load_vae():
    import torch
    from scripts.train_contour_vae_v2 import ImageVAE, ImageVAEConfig
    hp = json.loads(VAE_HP.read_text())
    cfg = ImageVAEConfig(**hp["image_vae_config"])
    vae = ImageVAE(cfg=cfg)
    state = torch.load(VAE_CKPT, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    vae.load_state_dict(state)
    vae.eval()
    return vae


def register_one(crop: np.ndarray):
    """crop: (170 freq, 234 time) decoded reconstruction.

    Returns (shape50, pitch_bin, duration_frames, n_gaps) or None if too short.
    """
    thr = max(1e-6, 0.20 * float(crop.max()))
    cfg = RidgeConfig(transition_penalty=0.1, max_jump_bins=10, silence_threshold=thr)
    fm, am = track_ridge(crop, np.arange(crop.shape[0], dtype=float), cfg)
    active = np.isfinite(fm)
    if active.sum() < MIN_ACTIVE_COLS:
        return None
    idx = np.where(active)[0]
    lo, hi = idx[0], idx[-1]
    span = fm[lo:hi + 1].copy()
    # interpolate internal NaN gaps (fragmented calls) for a continuous shape
    n_gaps = int(np.sum(np.diff(idx) > 1))
    nan = ~np.isfinite(span)
    if nan.any():
        good = np.where(~nan)[0]
        span[nan] = np.interp(np.where(nan)[0], good, span[good])
    pitch_bin = float(span.mean())
    span_c = span - pitch_bin                      # kill absolute pitch
    # resample active span to fixed length -> kill duration + time-position
    xs = np.linspace(0, 1, len(span_c))
    shape50 = np.interp(np.linspace(0, 1, N_RESAMPLE), xs, span_c)
    return shape50.astype(np.float32), pitch_bin, float(hi - lo + 1), n_gaps


def eta2(values: np.ndarray, labels: np.ndarray) -> float:
    """Between-cluster fraction of variance (eta^2) for a 1-D or N-D attribute.

    values: (n,) or (n, d).  Returns 1 - within_SS/total_SS in [0,1].
    Higher = the partition separates this attribute well.
    """
    v = values if values.ndim == 2 else values[:, None]
    grand = v.mean(0)
    tot = float(((v - grand) ** 2).sum())
    within = 0.0
    for lab in np.unique(labels):
        m = labels == lab
        within += float(((v[m] - v[m].mean(0)) ** 2).sum())
    return 1.0 - within / tot if tot > 0 else 0.0


def main() -> int:
    import torch
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max-patches", type=int, default=0, help="0 = all")
    ap.add_argument("--k", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    print("[PARAM] latents       =", LATENTS)
    print("[PARAM] vae_ckpt      =", VAE_CKPT)
    print("[PARAM] k             =", args.k)
    print("[PARAM] N_RESAMPLE    =", N_RESAMPLE, "(registered-ridge dims)")
    print("[PARAM] crop rows     =", (ROW0, ROW1), " cols =", (COL0, COL1))
    print("[PARAM] HZ_PER_BIN    = %.1f   MS_PER_FRAME = %.3f" % (HZ_PER_BIN, MS_PER_FRAME))
    print("[PARAM] max_patches   =", args.max_patches or "ALL")
    OUT.mkdir(parents=True, exist_ok=True)

    lat = pd.read_parquet(LATENTS)
    z_cols = [f"z_{i}" for i in range(32)]
    if args.max_patches and args.max_patches < len(lat):
        # random sample (cohort-representative), NOT iloc[:N] which is cohort-skewed
        lat = lat.sample(n=args.max_patches, random_state=args.seed).reset_index(drop=True)
    Z = lat[z_cols].to_numpy(np.float32)
    n = len(Z)
    print(f"[INFO] {n} patches loaded")

    vae = load_vae()

    shapes = np.full((n, N_RESAMPLE), np.nan, np.float32)
    pitch = np.full(n, np.nan, np.float32)
    dur = np.full(n, np.nan, np.float32)
    gaps = np.zeros(n, np.int32)

    t0 = time.time()
    with torch.no_grad():
        for s in range(0, n, args.batch):
            e = min(s + args.batch, n)
            zб = torch.from_numpy(Z[s:e])
            recon = vae.decode(zб).cpu().numpy()[:, 0]  # (b, 256, 256)
            crop = recon[:, ROW0:ROW1, COL0:COL1]       # (b, 170, 234)
            for j in range(crop.shape[0]):
                r = register_one(crop[j])
                if r is not None:
                    shapes[s + j], pitch[s + j], dur[s + j], gaps[s + j] = r
            if s % (args.batch * 20) == 0:
                el = time.time() - t0
                rate = e / el if el else 0
                print(f"[INFO] decoded {e}/{n}  ({rate:.0f}/s, eta {((n-e)/rate)/60:.1f} min)")

    ok = np.isfinite(shapes[:, 0])
    print(f"[INFO] {ok.sum()}/{n} patches yielded a trackable ridge "
          f"({100*ok.mean():.1f}%); {n-ok.sum()} too short/blob")

    Sh = shapes[ok]
    Zok = Z[ok]
    P = pitch[ok]
    D = dur[ok]
    G = gaps[ok]
    cohorts = lat["cohort"].to_numpy()[ok] if "cohort" in lat.columns else np.array(["?"] * ok.sum())

    # --- three representations --------------------------------------------
    # 1) existing latent K=20 (raw 32-D VAE latents) on the SAME patches
    latkm = joblib.load(KMEANS)
    old_lab = latkm.predict(Zok)
    # 2) registered SHAPE: cluster the centered, time-normalised ridge directly
    shape_km = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit(Sh)
    shape_lab = shape_km.labels_
    # 3) DERIVATIVE (the user's dF/dt idea): cluster the slope sequence.
    #    Differencing makes pitch-invariance exact and emphasises local slope
    #    changes (kinks/jumps) over the slow overall arc.
    D1 = np.diff(Sh, axis=1)                # slope  (n, N-1)
    D2 = np.diff(Sh, n=2, axis=1)           # curvature (n, N-2)
    deriv_km = KMeans(n_clusters=args.k, n_init=10, random_state=args.seed).fit(D1)
    deriv_lab = deriv_km.labels_

    # --- attributes scored per partition ----------------------------------
    jump = np.abs(D2).sum(1)                # total |curvature| = "kinkiness"
    attrs = {"pitch": P[:, None], "duration": D[:, None],
             "shape": Sh, "curvature": jump[:, None]}
    parts = {"existing_latent": old_lab, "shape": shape_lab, "derivative": deriv_lab}

    print("\n===== eta^2 (between-cluster variance fraction; higher = partition separates it) =====")
    print("partition           |  pitch  | duration|  shape  |curvature")
    res = {"n_clustered": int(ok.sum()), "k": args.k, "eta2": {}}
    for pname, lab in parts.items():
        row = {a: eta2(v, lab) for a, v in attrs.items()}
        res["eta2"][pname] = row
        print("  %-18s| %7.3f | %7.3f | %7.3f | %7.3f" %
              (pname, row["pitch"], row["duration"], row["shape"], row["curvature"]))
    print("======================================================================================")

    # --- chevron vs valley separation (sanity diagnostic) -----------------
    cv = chevron_valley_labels(Sh)
    sel = cv != "other"
    if sel.sum() > 50:
        from sklearn.metrics import normalized_mutual_info_score as nmi
        res["chevron_valley_nmi"] = {
            pname: float(nmi(cv[sel], lab[sel])) for pname, lab in parts.items()}
        print("[chevron/valley NMI] " + "  ".join(
            f"{p}={v:.3f}" for p, v in res["chevron_valley_nmi"].items()),
            f"  (n_chevron={int((cv=='chevron').sum())}, n_valley={int((cv=='valley').sum())})")

    # --- persist arrays for any follow-on (no re-decode needed) -----------
    np.savez_compressed(
        OUT / "registered_ridges.npz",
        shapes=Sh, pitch=P, duration=D, gaps=G, jump=jump,
        cohort=cohorts.astype(str),
        wav_stem=lat["wav_stem"].to_numpy().astype(str)[ok],
        call_id=lat["call_id"].to_numpy()[ok],
        latent_label=old_lab, shape_label=shape_lab, deriv_label=deriv_lab,
        chevron_valley=cv.astype(str))

    # --- galleries: derivative (the proposed idea) + plain shape ----------
    deriv_cards = render_gallery("deriv", deriv_km.cluster_centers_, deriv_lab, D1, Sh,
                                 P, D, G, jump, cohorts, Zok, vae)
    shape_cards = render_gallery("shape", shape_km.cluster_centers_, shape_lab, Sh, Sh,
                                 P, D, G, jump, cohorts, Zok, vae)

    (OUT / "shape_clustering_summary.json").write_text(json.dumps({
        **res,
        "params": {"N_RESAMPLE": N_RESAMPLE, "crop_rows": [ROW0, ROW1],
                   "crop_cols": [COL0, COL1], "hz_per_bin": HZ_PER_BIN,
                   "ms_per_frame": MS_PER_FRAME, "seed": args.seed,
                   "ridge_cfg": "penalty=0.1, max_jump=10, silence=0.2*max"},
    }, indent=2))
    write_html_compare(res, {"derivative (dF/dt)": deriv_cards,
                             "registered shape": shape_cards})
    print("[DONE] wrote", OUT)
    return 0


def chevron_valley_labels(Sh: np.ndarray, margin_bins: float = 5.0) -> np.ndarray:
    """Label each centered ridge chevron (∧) / valley (∨) / other.

    margin_bins ~5 bins ≈ 3 kHz prominence required to call a hump.
    """
    n, N = Sh.shape
    lo, hi = int(0.2 * N), int(0.8 * N)
    out = np.array(["other"] * n, dtype=object)
    peak_i = Sh.argmax(1); peak_v = Sh.max(1)
    trough_i = Sh.argmin(1); trough_v = Sh.min(1)
    ends_max = np.maximum(Sh[:, 0], Sh[:, -1])
    ends_min = np.minimum(Sh[:, 0], Sh[:, -1])
    is_chev = (peak_i >= lo) & (peak_i <= hi) & (peak_v - ends_max > margin_bins)
    is_val = (trough_i >= lo) & (trough_i <= hi) & (ends_min - trough_v > margin_bins)
    out[is_chev & ~is_val] = "chevron"
    out[is_val & ~is_chev] = "valley"
    return np.asarray(out)


def render_gallery(name, centers, lab, feats, Sh, pitch, dur, gaps, jump,
                   cohorts, Zok, vae, n_show=24):
    """Render one gallery; return the list of card dicts (no HTML write here).

    name    : "deriv" | "shape" (filename + title prefix)
    centers : cluster centers in the FEATURE space used for clustering
    feats   : the feature vectors that were clustered (for nearest-exemplar pick)
    Sh      : registered ridges — always plotted as the visible shape overlay
    """
    import torch
    K = centers.shape[0]
    cards = []
    for k in range(K):
        m = np.where(lab == k)[0]
        if len(m) == 0:
            continue
        d = np.linalg.norm(feats[m] - centers[k], axis=1)
        near = m[np.argsort(d)[:n_show]]
        cvals, ccnts = np.unique(cohorts[m], return_counts=True)
        cohort_str = "  ".join(f"{c}={n}" for c, n in
                               sorted(zip(cvals, ccnts), key=lambda x: -x[1]))
        pit_khz = pitch[m] * HZ_PER_BIN / 1000.0   # relative within band crop
        dur_ms = dur[m] * MS_PER_FRAME

        fig, (axc, axr) = plt.subplots(1, 2, figsize=(7.2, 3.0),
                                       gridspec_kw={"width_ratios": [1.1, 1]})
        for i in near:
            axc.plot(Sh[i], color="#7c2d12", alpha=0.18, lw=0.8)
        axc.plot(Sh[m].mean(0), color="black", lw=2.2)   # cluster-mean SHAPE
        axc.set_title(f"{name} c{k:02d} — n={len(m)}", fontsize=10, loc="left")
        axc.set_xlabel("normalised time"); axc.set_ylabel("freq − mean (bins)")
        axc.axhline(0, color="#999", lw=0.6, ls=":")
        axc.set_ylim(-40, 40)
        with torch.no_grad():
            rec = vae.decode(torch.from_numpy(Zok[near[0]][None]))[0, 0].cpu().numpy()
        axr.imshow(rec[ROW0:ROW1, COL0:COL1], origin="lower", cmap="magma",
                   aspect="auto", vmax=max(np.percentile(rec, 99.5), 1e-4))
        axr.set_xticks([]); axr.set_yticks([]); axr.set_title("nearest patch", fontsize=9)
        fig.tight_layout(pad=0.3)
        png = OUT / f"{name}_c{k:02d}.png"
        fig.savefig(png, dpi=110); plt.close(fig)

        cards.append({
            "k": k, "n": int(len(m)), "png": png.name, "cohort": cohort_str,
            "pitch_spread_khz": f"{pit_khz.std():.1f}",
            "dur_med_ms": f"{np.median(dur_ms):.0f}",
            "dur_spread_ms": f"{dur_ms.std():.0f}",
            "curv_med": f"{np.median(jump[m]):.0f}",
            "frag_frac": f"{(gaps[m] > 0).mean():.0%}",
        })
    return cards


def write_html_compare(res, galleries):
    """galleries: {section_title: [card, ...]}."""
    def b64(p):
        return base64.b64encode((OUT / p).read_bytes()).decode()

    # eta^2 comparison table
    attrs = ["pitch", "duration", "shape", "curvature"]
    head = "".join(f"<th>{a}</th>" for a in attrs)
    trows = ""
    for pname, row in res["eta2"].items():
        cells = "".join(f"<td>{row[a]:.3f}</td>" for a in attrs)
        trows += f"<tr><th>{pname}</th>{cells}</tr>"
    nmi = res.get("chevron_valley_nmi", {})
    nmi_str = ("chevron/valley NMI — " +
               "  ".join(f"{k}: {v:.3f}" for k, v in nmi.items())) if nmi else ""

    sections = ""
    for title, cards in galleries.items():
        figs = "\n".join(
            f'<figure><img src="data:image/png;base64,{b64(c["png"])}">'
            f'<figcaption><b>{title.split()[0]} c{c["k"]:02d}</b> — {c["n"]}<br>'
            f'pitch σ={c["pitch_spread_khz"]}kHz · dur {c["dur_med_ms"]}±{c["dur_spread_ms"]}ms '
            f'· curv {c["curv_med"]} · frag {c["frag_frac"]}<br>'
            f'<span style="font-size:10px">{c["cohort"]}</span></figcaption></figure>'
            for c in cards)
        sections += f'<h2>{title}</h2><div class="grid">{figs}</div>'

    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Shape vs derivative clustering — pitch/position/duration removed</title>
<style>
body{{font:14px/1.5 -apple-system,Segoe UI,Arial;margin:0;padding:24px;max-width:1400px;
margin-left:auto;margin-right:auto;background:#fafafa;color:#1a1a1f}}
h1{{font-size:21px;margin:0 0 4px}} h2{{font-size:17px;margin:24px 0 8px}}
.sub{{color:#5a5a64;font-size:13px;margin-bottom:16px}}
.grid{{display:grid;grid-template-columns:repeat(2,1fr);gap:10px}}
figure{{margin:0;background:#fff;border:1px solid #d8d8e0;border-radius:6px;padding:6px}}
figure img{{width:100%;display:block}}
figcaption{{font-size:11.5px;color:#5a5a64;margin-top:4px;line-height:1.35}}
table{{border-collapse:collapse;margin:8px 0 4px;font-size:13px}}
th,td{{border:1px solid #d8d8e0;padding:4px 10px;text-align:center}}
th{{background:#f0f0f4}}
.note{{background:#fff8e6;border-left:3px solid #d97706;padding:8px 12px;margin:8px 0 16px;
border-radius:0 4px 4px 0;font-size:13px}}
</style></head><body>
<h1>Clustering by contour shape &amp; its derivative — pitch, position, duration removed</h1>
<div class="sub">Each call's frequency ridge was registered (mean subtracted → pitch out;
active span resampled to 50 pts → position + duration out), then clustered three ways.</div>
<h2>η² — between-cluster variance fraction (higher = the partition organises by that attribute)</h2>
<table><tr><th>partition</th>{head}</tr>{trows}</table>
<div class="note"><b>Read the table:</b> the <i>existing latent</i> row should be high on
<b>pitch</b>/<b>duration</b> and low on <b>shape</b>; the <i>shape</i> and <i>derivative</i>
rows should flip that — low pitch/duration, high shape — with <b>derivative</b> expected to
score highest on <b>curvature</b> (it isolates kinks/jumps). {nmi_str}</div>
{sections}
</body></html>"""
    (OUT / "comparison.html").write_text(html)


if __name__ == "__main__":
    raise SystemExit(main())

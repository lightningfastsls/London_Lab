"""R1 (rig) — extract TRUE registered ridges from real contour-masked patches,
then re-run the Tier-0 bake-off on ground truth (non-circular; no VAE decode).

Real patches live per-cohort at /data/shachar/contour_vae/results/masked_patches/
{5970,3452,9252,lab_131204}_focus/patches.npz, each holding:
    patches  (N, 257, 234) float32 linear power   (contour-masked: off-ridge = 0)
    freqs_kHz (257,)        real frequency axis
USV band = rows [35:205] (170 bins, 20-120 kHz).

Per call: band-crop -> Viterbi ridge (real kHz) -> register (subtract mean kHz =
kill pitch; resample active span to 50 pts = kill duration + time-position).
Then cluster 7 ways and score eta2(pitch/duration/shape/curvature) + chevron/valley
NMI. Outputs true_registered_ridges.npz (feeds M8 1-D VAE) + a scorecard + HTML.

Memory: processed PER COHORT (peak ~13 GB for lab) to stay under the 31 GB box.
"""
from __future__ import annotations
import base64, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi

sys.path.insert(0, "/data/mickey_london_lab/src")
from usv_spectrogram.features.ridge_tracker import track_ridge, RidgeConfig

R = Path("/data/shachar/contour_vae")
MP = R / "results/masked_patches"
OUT = R / "results/latent_transitions/shape_registered_TRUE"
OUT.mkdir(parents=True, exist_ok=True)
COHORTS = ["5970", "3452", "9252", "lab_131204"]
BAND0, BAND1 = 35, 205          # USV band rows (170 bins), from hyperparams.json
N_RESAMPLE = 50
MIN_ACTIVE_COLS = 6


def register_one(crop, freqs_khz):
    """crop: (170 freq, T) real linear-power; freqs_khz: (170,). -> (shape50,pitch,dur,gaps) or None."""
    thr = max(1e-9, 0.02 * float(crop.max()))
    cfg = RidgeConfig(transition_penalty=0.1, max_jump_bins=10, silence_threshold=thr)
    fm, am = track_ridge(crop, freqs_khz.astype(float), cfg)
    active = np.isfinite(fm)
    if active.sum() < MIN_ACTIVE_COLS:
        return None
    idx = np.where(active)[0]; lo, hi = idx[0], idx[-1]
    span = fm[lo:hi + 1].copy()
    n_gaps = int(np.sum(np.diff(idx) > 1))
    nanm = ~np.isfinite(span)
    if nanm.any():
        good = np.where(~nanm)[0]
        span[nanm] = np.interp(np.where(nanm)[0], good, span[good])
    pitch = float(span.mean())
    sc = span - pitch
    shape = np.interp(np.linspace(0, 1, N_RESAMPLE), np.linspace(0, 1, len(sc)), sc)
    return shape.astype(np.float32), pitch, float(hi - lo + 1), n_gaps


def chevron_valley_labels(Sh, margin=2.0):   # margin in kHz
    n, N = Sh.shape; lo, hi = int(0.2 * N), int(0.8 * N)
    out = np.array(["other"] * n, dtype=object)
    pk_i, pk_v = Sh.argmax(1), Sh.max(1)
    tr_i, tr_v = Sh.argmin(1), Sh.min(1)
    emax = np.maximum(Sh[:, 0], Sh[:, -1]); emin = np.minimum(Sh[:, 0], Sh[:, -1])
    chev = (pk_i >= lo) & (pk_i <= hi) & (pk_v - emax > margin)
    val = (tr_i >= lo) & (tr_i <= hi) & (emin - tr_v > margin)
    out[chev & ~val] = "chevron"; out[val & ~chev] = "valley"
    return np.asarray(out)


def eta2(values, labels):
    v = values if values.ndim == 2 else values[:, None]
    keep = labels >= 0; v, labels = v[keep], labels[keep]
    g = v.mean(0); tot = float(((v - g) ** 2).sum())
    within = sum(float(((v[labels == l] - v[labels == l].mean(0)) ** 2).sum())
                 for l in np.unique(labels))
    return 1.0 - within / tot if tot > 0 else 0.0


def main():
    t0 = time.time()
    shapes, pitch, dur, gaps, cohort = [], [], [], [], []
    for c in COHORTS:
        p = MP / f"{c}_focus/patches.npz"
        if not p.exists():
            print(f"[WARN] missing {p}"); continue
        z = np.load(p)
        arr = z["patches"]; freqs = z["freqs_kHz"][BAND0:BAND1]
        print(f"[INFO] {c}: {arr.shape[0]} patches", flush=True)
        for i in range(arr.shape[0]):
            r = register_one(arr[i, BAND0:BAND1, :], freqs)
            if r is not None:
                shapes.append(r[0]); pitch.append(r[1]); dur.append(r[2])
                gaps.append(r[3]); cohort.append(c)
            if i % 5000 == 0 and i:
                el = time.time() - t0
                print(f"  {c} {i}/{arr.shape[0]} ({len(shapes)/el:.0f}/s total)", flush=True)
        del arr, z
    Sh = np.array(shapes, np.float32)
    pitch = np.array(pitch, np.float32); dur = np.array(dur, np.float32)
    gaps = np.array(gaps, np.int32); cohort = np.array(cohort)
    print(f"[INFO] total trackable ridges: {len(Sh)} in {(time.time()-t0)/60:.1f} min", flush=True)

    D1 = np.diff(Sh, 1, 1); D2 = np.diff(Sh, 2, 1)
    jump = np.abs(D2).sum(1)
    cv = chevron_valley_labels(Sh); sel = cv != "other"
    lam = float(Sh.std() / (D1.std() + 1e-9))
    feats = {"shape": Sh, "deriv": D1, "curv": D2,
             "combined": np.concatenate([Sh, lam * D1], 1),
             "shape_scalenorm": Sh / (Sh.std(1, keepdims=True) + 1e-6), "srvf": np.sign(D1) * np.sqrt(np.abs(D1))}
    parts = {n: KMeans(20, n_init=10, random_state=42).fit_predict(X) for n, X in feats.items()}
    try:
        import umap, hdbscan
        emb = umap.UMAP(n_neighbors=30, min_dist=0.0, n_components=10, random_state=42).fit_transform(Sh)
        parts["umap_hdbscan"] = hdbscan.HDBSCAN(min_cluster_size=max(50, len(Sh)//200)).fit_predict(emb)
    except Exception as ex:
        print("[WARN] umap/hdbscan skipped:", repr(ex))

    P, Dr, J = pitch[:, None], dur[:, None], jump[:, None]
    rows = []
    print("\n===== TRUE-RIDGE SCORECARD (eta^2; want low pitch/dur, high shape) =====")
    print("method            |  pitch | dur   | shape | curv  | CV-NMI | nclu")
    for n, lab in parts.items():
        cvnmi = float(nmi(cv[sel], lab[sel])) if sel.sum() > 50 else float("nan")
        r = dict(method=n, pitch=eta2(P, lab), duration=eta2(Dr, lab), shape=eta2(Sh, lab),
                 curvature=eta2(J, lab), cv_nmi=cvnmi, n_clusters=int(len(np.unique(lab[lab >= 0]))))
        rows.append(r)
        print("  %-16s| %.3f | %.3f | %.3f | %.3f | %.3f | %4d" %
              (n, r["pitch"], r["duration"], r["shape"], r["curvature"], cvnmi, r["n_clusters"]), flush=True)
    print("=======================================================================")

    np.savez_compressed(OUT / "true_registered_ridges.npz", shapes=Sh, pitch=pitch,
                        duration=dur, gaps=gaps, jump=jump, cohort=cohort.astype(str),
                        chevron_valley=cv.astype(str), **{f"lab_{n}": l for n, l in parts.items()})
    render(parts, Sh, pitch)
    write_html(rows)
    (OUT / "true_scorecard.json").write_text(json.dumps({"n": len(Sh), "rows": rows, "lambda": lam}, indent=2))
    print("[DONE]", OUT, flush=True)


def render(parts, Sh, pitch):
    rng = np.random.default_rng(0)
    for name, lab in parts.items():
        uniq = sorted([c for c in np.unique(lab) if c >= 0], key=lambda c: -(lab == c).sum())[:20]
        nrow = int(np.ceil(len(uniq) / 5)) or 1
        fig, axes = plt.subplots(nrow, 5, figsize=(10, nrow * 1.7), squeeze=False)
        for ax in axes.ravel():
            ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-40, 40); ax.axhline(0, color="#ccc", lw=.5)
        for i, c in enumerate(uniq):
            ax = axes[i // 5][i % 5]; m = np.where(lab == c)[0]
            for j in m[rng.choice(len(m), min(40, len(m)), replace=False)]:
                ax.plot(Sh[j], color="#7c2d12", alpha=.10, lw=.6)
            ax.plot(Sh[m].mean(0), color="black", lw=1.8)
            ax.set_title(f"c{c} n={len(m)} σ{pitch[m].std():.0f}k", fontsize=7, loc="left")
        fig.suptitle(f"{name} (TRUE ridges) — cluster-mean shapes", fontsize=10)
        fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(OUT / f"sweep_{name}.png", dpi=110); plt.close(fig)


def write_html(rows):
    b64 = lambda p: base64.b64encode((OUT / p).read_bytes()).decode()
    th = "".join(f"<th>{a}</th>" for a in ["pitch", "duration", "shape", "curvature", "CV-NMI", "#clu"])
    tr = "".join(f"<tr><th>{r['method']}</th><td>{r['pitch']:.3f}</td><td>{r['duration']:.3f}</td>"
                 f"<td><b>{r['shape']:.3f}</b></td><td>{r['curvature']:.3f}</td><td>{r['cv_nmi']:.3f}</td>"
                 f"<td>{r['n_clusters']}</td></tr>" for r in rows)
    parts_html = []
    for r in rows:
        png = OUT / f"sweep_{r['method']}.png"
        if png.exists():
            parts_html.append(f"<h2>{r['method']}</h2><img style='width:100%;border:1px solid #ccc' "
                              f"src='data:image/png;base64,{b64(png.name)}'>")
    imgs = "".join(parts_html)
    html = (f"<!DOCTYPE html><html><head><meta charset='utf-8'><title>TRUE-ridge bake-off</title>"
            f"<style>body{{font:14px -apple-system,Arial;max-width:1200px;margin:auto;padding:24px}}"
            f"table{{border-collapse:collapse}}th,td{{border:1px solid #ccc;padding:4px 10px;text-align:center}}"
            f"img{{margin:6px 0}}</style></head><body><h1>TRUE-ridge Tier-0 bake-off (real patches, non-circular)</h1>"
            f"<table><tr><th>method</th>{th}</tr>{tr}</table>{imgs}</body></html>")
    (OUT / "true_method_sweep.html").write_text(html)


if __name__ == "__main__":
    main()

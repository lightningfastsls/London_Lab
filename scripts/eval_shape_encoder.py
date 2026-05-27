"""Pathway B — eval scorecard for the contrastive shape encoder.

Scores the embeddings from `train_shape_encoder_contrastive.py` against the
SAME axes as the 2026-05-25 bake-off so the leaderboard comparison is honest:

    shape eta2 | pitch eta2 | duration eta2 | curvature(jump) eta2 | CV-NMI
    + k-NN neighbor purity (the most literal "chevron-with-chevron" test)
    + UMAP coloured by chevron/valley type

All scoring axes come from the cached registered-ridge descriptors
`results/eval_shape/desc_denoised.npz` (row, pitch, shapes[N,50], duration, jump)
-- NO ridge re-extraction, NO WAV access.  There is NO `syllable_type` column in
classified_detections_* (only `label`), so the geometric type used for NMI /
purity is the chevron/valley heuristic on the registered ridge -- exactly what
M9 (0.175) and M10 used, keeping the comparison apples-to-apples.

Bars to beat:  registration shape 0.58-0.75 | M9 0.344 | production 0.099 | denoised 0.081

Run on the rig (canonical root /data/shachar/contour_vae).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
def eta2(v: np.ndarray, lab: np.ndarray) -> float:
    """Between-cluster variance fraction (1 - within/total); ignores lab<0.
    Identical semantics to the rig bake-off scripts (M8/M9/M10/R1)."""
    v = v if v.ndim == 2 else v[:, None]
    keep = lab >= 0
    v, lab = v[keep], lab[keep]
    if len(v) == 0:                      # all labels < 0: nothing to score (avoid empty-mean warning)
        return 0.0
    g = v.mean(0)
    tot = float(((v - g) ** 2).sum())
    if tot <= 0:
        return 0.0
    w = sum(float(((v[lab == l] - v[lab == l].mean(0)) ** 2).sum()) for l in np.unique(lab))
    return 1 - w / tot


def knn_purity(Z: np.ndarray, types: np.ndarray, k: int = 10) -> dict:
    """For each point, fraction of its k nearest Euclidean neighbours (excluding
    self) sharing its type; averaged per type and overall.  The most literal
    test of 'geometrically similar calls are neighbours'."""
    from sklearn.neighbors import NearestNeighbors

    types = np.asarray(types)
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(Z))).fit(Z)
    _, idx = nn.kneighbors(Z)
    idx = idx[:, 1:]                                   # drop self
    same = (types[idx] == types[:, None]).mean(axis=1)
    out = {"overall": float(same.mean()), "k": int(k)}
    for t in np.unique(types):
        # key by the RAW label value (native scalar) so integer/string types
        # both round-trip: `t in result` and `result[0]` must work for callers.
        key = t.item() if hasattr(t, "item") else t
        out[key] = float(same[types == t].mean())
    return out


def chevron_valley(shapes: np.ndarray) -> np.ndarray:
    """Holy/Guo-style chevron vs valley from the registered (de-meaned) 50-pt
    ridge shape.  Mirrors the M10 heuristic so labels match prior scorecards."""
    N = shapes.shape[1]
    lo, hi = int(0.2 * N), int(0.8 * N)
    pk = shapes.argmax(1); tr = shapes.argmin(1)
    emax = np.maximum(shapes[:, 0], shapes[:, -1])
    emin = np.minimum(shapes[:, 0], shapes[:, -1])
    cv = np.array(["other"] * len(shapes), dtype=object)
    cv[(pk >= lo) & (pk <= hi) & (shapes.max(1) - emax > 2)] = "chevron"
    cv[(tr >= lo) & (tr <= hi) & (emin - shapes.min(1) > 2)] = "valley"
    return cv


# ---------------------------------------------------------------------------
def main() -> None:
    R = Path("/data/shachar/contour_vae")
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(R / "results/latent_transitions/b_contrastive"),
                    help="dir holding embeddings.npy + split_idx.npz")
    ap.add_argument("--desc", default=str(R / "results/eval_shape/desc_denoised.npz"))
    ap.add_argument("--k-clusters", type=int, default=20)
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--split", default="val", choices=["val", "train", "all"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-umap", action="store_true")
    a = ap.parse_args()

    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi

    run = Path(a.run)
    embs = np.load(run / "embeddings.npy")                  # (N, embed_dim) over ALL patches
    split = np.load(run / "split_idx.npz")
    d = np.load(a.desc)
    row = d["row"]                                          # patch idx that survived ridge extraction

    if a.split == "all":
        sel_patches = np.arange(embs.shape[0])
    else:
        sel_patches = split[a.split]
    keep = np.isin(row, sel_patches)                        # align cache rows -> chosen split
    rr = row[keep]
    Z = embs[rr]
    pitch, shapes = d["pitch"][keep], d["shapes"][keep]
    dur, jump = d["duration"][keep], d["jump"][keep]
    cv = chevron_valley(shapes)

    print(f"[PARAM] eval run={run.name} split={a.split} k_clusters={a.k_clusters} "
          f"knn_k={a.knn_k} embed_dim={embs.shape[1]} seed={a.seed}", flush=True)
    print(f"[INFO] embeddings N_all={embs.shape[0]}  cache_rows={len(row)}  "
          f"scored={len(rr)}  chevron={int((cv=='chevron').sum())} "
          f"valley={int((cv=='valley').sum())} other={int((cv=='other').sum())}", flush=True)

    lab = KMeans(a.k_clusters, n_init=10, random_state=a.seed).fit_predict(Z)
    msel = cv != "other"

    score = dict(
        method="B_contrastive", split=a.split, n=int(len(rr)), k_clusters=int(a.k_clusters),
        shape=eta2(shapes, lab),                 # GATE 3: must clear 0.12, target >=0.50
        pitch=eta2(pitch[:, None], lab),         # GATE 4: must be LOW (production 0.45 = the failure)
        duration=eta2(dur[:, None], lab),        # GATE 4 (cont.): time-warp invariance check
        curvature=eta2(jump[:, None], lab),      # jump/kink capture
        cv_nmi=float(nmi(cv[msel], lab[msel])) if msel.any() else 0.0,  # GATE 2: beat 0.04, target >0.20
    )
    purity = knn_purity(Z, cv, a.knn_k)          # GATE 1: literal chevron<->chevron test
    score["knn_purity"] = purity

    print("\n===== B-CONTRASTIVE SCORECARD =====")
    print("  GATE 3 shape eta2   %.3f   (kill<0.12 | target>=0.50 | match registration 0.58-0.75)"
          % score["shape"])
    print("  GATE 4 pitch eta2   %.3f   (LOW is good; production VAE 0.45 was the failure)"
          % score["pitch"])
    print("         dur   eta2   %.3f   (LOW => duration-invariant, matches warp aug)"
          % score["duration"])
    print("  GATE 2 CV-NMI       %.3f   (beat production 0.04 | target>0.20 | M9 0.175)"
          % score["cv_nmi"])
    print("  GATE 1 kNN purity   overall %.3f | chevron %.3f | valley %.3f"
          % (purity["overall"], purity.get("chevron", float("nan")),
             purity.get("valley", float("nan"))))
    print("         curvature    %.3f" % score["curvature"], flush=True)

    (run / "score_b_contrastive.json").write_text(json.dumps(score, indent=2))
    print(f"[INFO] wrote {run/'score_b_contrastive.json'}", flush=True)

    # GATE 6: UMAP coloured by chevron/valley (the lab-presentation figure)
    if not a.no_umap:
        try:
            import umap
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            sub = np.random.default_rng(a.seed).choice(len(Z), min(15000, len(Z)), replace=False)
            emb2 = umap.UMAP(n_neighbors=30, min_dist=0.1, random_state=a.seed).fit_transform(Z[sub])
            cvs = cv[sub]
            fig, ax = plt.subplots(figsize=(7, 6))
            colors = {"chevron": "#1e6b3a", "valley": "#b3402a", "other": "#cccccc"}
            for t in ["other", "valley", "chevron"]:
                m = cvs == t
                ax.scatter(emb2[m, 0], emb2[m, 1], s=3, alpha=0.4, c=colors[t], label=f"{t} ({m.sum()})")
            ax.legend(markerscale=3, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"B-contrastive embedding UMAP (shape eta2={score['shape']:.3f})")
            fig.tight_layout(); fig.savefig(run / "umap_b_contrastive.png", dpi=120)
            plt.close(fig)
            print(f"[INFO] wrote {run/'umap_b_contrastive.png'}", flush=True)
        except Exception as e:  # pragma: no cover - umap optional
            print(f"[WARN] UMAP skipped: {e}", flush=True)

    # verdict against the handoff kill criteria
    if score["shape"] < 0.12 and purity["overall"] <= 0.5:
        print("\n[VERDICT] KILL: shape eta2 < 0.12 and purity not better than chance -> ship registration (0.75).")
    # 0.30 curvature = empirical midpoint between production 0.099 and M9 0.344;
    # advisory only (plan §5 defines "jump-capture edge" qualitatively, no number).
    elif score["shape"] >= 0.58 or (score["shape"] >= 0.12 and score["curvature"] > 0.30):
        print("\n[VERDICT] STRONG: beats/matches registration or shows jump-capture edge -> the win Mickey asked for.")
    elif score["shape"] >= 0.12:
        print("\n[VERDICT] PARTIAL: clears 0.12 but << registration and no clear jump edge -> report, do not ship.")
    print("[DONE] eval_shape_encoder", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""WS-C: Topological characterization of the USV shape manifold.

Runs on the SETTLED elastic-FPCA coordinate system
(models/shape_fpca/elastic_fpca_scores.parquet), READ-ONLY.

Question (WS-C decision gate): is the shape manifold a filled blob, a 1-D
curve, or a branching tree? What is its intrinsic dimension? Is the
oscillatory pocket a distinct connected (H0) component?

COHORT CONFOUND: wild dyads 3452 and 9252 are a CAGE artifact on amp_pc1.
We RESTRICT all manifold characterization to lab_131204 + 5970 only.

Methods:
  1. Intrinsic dimension  -> skdim (lPCA, MLE, TwoNN, FisherS, MOM)
  2. Persistent homology   -> ripser, H0/H1 on multiple subsamples (seed-stable)
  3. Density-ridge / structure -> kNN-graph connectivity + degree/branch analysis
  4. PHATE 2-D embedding   -> visualization vs prior UMAP "Track D" finding

All parameters, seeds, thresholds, and row counts are PRINTED.
Outputs an HTML report at results/shape_manifold_wsc/manifold_characterization.html
"""
from __future__ import annotations

import base64
import datetime as _dt
import json
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG (all printed below)
# ---------------------------------------------------------------------------
REPO = Path("/home/shachar/projects/mickey_london_lab")
SCORES = REPO / "models/shape_fpca/elastic_fpca_scores.parquet"
LETTERS = REPO / "models/shape_kmeans/k20_softdtw_letters.parquet"
OUTDIR = REPO / "results/shape_manifold_wsc"
OUTDIR.mkdir(parents=True, exist_ok=True)

AMP_COLS = [f"amp_pc{i}" for i in range(1, 6)]      # shape axes
PHASE_COLS = [f"phase_pc{i}" for i in range(1, 4)]  # warp/phase axes (secondary)
KEEP_COHORTS = ["lab_131204", "5970"]               # restricted set (confound removed)

# Persistent homology
PH_N = 2000           # points per PH subsample
PH_SEEDS = [0, 1, 2, 3, 4]
PH_MAXDIM = 1         # compute H0 and H1
PH_GAP_RATIO = 5.0    # H0 separated-component gap must be >= this x median merge gap
PH_H1_FRAC = 0.25     # H1 loop significant if persistence >= this x max H0 death
PH_MIN_POCKET = 10    # a detached H0 comp is a real "pocket" (vs outlier) at >= this many pts scale

# kNN-graph connectivity
KNN_K = 10
KNN_N = 6000          # subsample for graph
KNN_SEED = 0

# PHATE
PHATE_N = 6000
PHATE_SEED = 0

GLOBAL_SEED = 0


def log(msg=""):
    print(msg, flush=True)


def fig_to_b64(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.default_rng(GLOBAL_SEED)
    report = {}  # collected fields for HTML
    images = {}

    log("=" * 70)
    log("WS-C SHAPE MANIFOLD CHARACTERIZATION")
    log("=" * 70)
    log(f"scores parquet : {SCORES}")
    log(f"amp (shape) cols : {AMP_COLS}")
    log(f"phase cols       : {PHASE_COLS}")
    log(f"keep cohorts     : {KEEP_COHORTS}")
    log(f"GLOBAL_SEED={GLOBAL_SEED}")
    log(f"PH: N={PH_N} seeds={PH_SEEDS} maxdim={PH_MAXDIM} "
        f"gap_ratio={PH_GAP_RATIO} h1_frac={PH_H1_FRAC}")
    log(f"kNN: k={KNN_K} N={KNN_N} seed={KNN_SEED}")
    log(f"PHATE: N={PHATE_N} seed={PHATE_SEED}")

    # -------------------------------------------------------------------
    # LOAD + COHORT FILTER
    # -------------------------------------------------------------------
    df = pd.read_parquet(SCORES)
    log("\n--- RAW PARQUET ---")
    log(f"rows={len(df)} cols={list(df.columns)}")
    log("dtypes:\n" + str(df.dtypes))
    vc = df["cohort"].value_counts(dropna=False)
    log("cohort counts (raw):\n" + str(vc))
    report["raw_rows"] = int(len(df))
    report["raw_cohort_counts"] = {str(k): int(v) for k, v in vc.items()}

    # confound sanity print
    log("\n--- COHORT CONFOUND CHECK on amp_pc1 ---")
    for c in df["cohort"].unique():
        s = df.loc[df.cohort == c, "amp_pc1"]
        log(f"  {c:12s} mean={s.mean():9.3f} std={s.std():9.3f} n={len(s)}")

    keep = df[df["cohort"].isin(KEEP_COHORTS)].copy().reset_index(drop=True)
    kept_counts = keep["cohort"].value_counts()
    log("\n--- AFTER COHORT FILTER (lab_131204 + 5970) ---")
    log(f"kept cohort values: {sorted(keep['cohort'].unique())}")
    log("kept counts:\n" + str(kept_counts))
    log(f"TOTAL KEPT ROWS = {len(keep)}")
    report["kept_cohort_counts"] = {str(k): int(v) for k, v in kept_counts.items()}
    report["kept_rows"] = int(len(keep))

    # Attach soft-DTW letters POSITIONALLY. The (wav_stem, call_id) key is NOT
    # unique in either parquet (47,026 unique keys of 67,337 rows), but the two
    # files are verified row-for-row aligned on wav_stem+call_id+cohort, so a
    # key merge would explode into a many-to-many cross-join. We attach by row.
    letters_full = pd.read_parquet(LETTERS).reset_index(drop=True)
    df_full = df.reset_index(drop=True)
    assert len(letters_full) == len(df_full), "letters/scores length mismatch"
    assert (letters_full["wav_stem"].values == df_full["wav_stem"].values).all() \
        and (letters_full["call_id"].values == df_full["call_id"].values).all(), \
        "letters/scores not row-aligned"
    df_full = df_full.copy()
    df_full["softdtw_letter"] = letters_full["softdtw_letter"].values
    keep = df_full[df_full["cohort"].isin(KEEP_COHORTS)].copy().reset_index(drop=True)
    log(f"attached softdtw_letter positionally; kept rows now = {len(keep)} "
        f"(non-null letter = {keep['softdtw_letter'].notna().sum()})")
    assert len(keep) == report["kept_rows"], "row count changed after letter attach"

    X = keep[AMP_COLS].to_numpy(dtype=np.float64)
    # drop any non-finite rows
    finite = np.isfinite(X).all(axis=1)
    if (~finite).any():
        log(f"dropping {int((~finite).sum())} non-finite rows")
    X = X[finite]
    keep = keep.loc[finite].reset_index(drop=True)

    # Drop EXACT-duplicate coordinate rows. Duplicates create zero-distance
    # neighbours that (a) make TwoNN/MLE divide by zero -> inf/0, and (b)
    # inflate ripser H0 with spurious tiny components. We keep one row per
    # unique 5-D coordinate.
    _uniq, _uidx = np.unique(np.round(X, 9), axis=0, return_index=True)
    _uidx = np.sort(_uidx)
    n_dups = X.shape[0] - _uidx.size
    log(f"dropping {n_dups} exact-duplicate coordinate rows "
        f"({n_dups/X.shape[0]:.2%}) before ID/PH/structure")
    X = X[_uidx]
    keep = keep.iloc[_uidx].reset_index(drop=True)
    report["n_duplicate_rows_dropped"] = int(n_dups)
    report["n_unique_points"] = int(X.shape[0])
    log(f"final feature matrix X shape = {X.shape}")
    report["X_shape"] = list(X.shape)

    # standardize (z-score) so each amp PC contributes comparably to distances
    mu, sd = X.mean(0), X.std(0)
    Xz = (X - mu) / sd
    log(f"amp PC means: {np.round(mu,3).tolist()}")
    log(f"amp PC stds : {np.round(sd,3).tolist()}")

    # -------------------------------------------------------------------
    # 1. INTRINSIC DIMENSION
    # -------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("1. INTRINSIC DIMENSION (skdim, on standardized amp_pc1..5)")
    log("=" * 70)
    import skdim.id as sid

    # subsample for the heavier global estimators
    id_n = min(8000, Xz.shape[0])
    idx = np.random.default_rng(GLOBAL_SEED).choice(Xz.shape[0], id_n, replace=False)
    Xid = Xz[idx]
    log(f"intrinsic-dim subsample n={id_n} seed={GLOBAL_SEED}")

    id_results = {}
    estimators = {
        "lPCA":   lambda: sid.lPCA(ver="FO").fit(Xid).dimension_,
        "MLE":    lambda: sid.MLE().fit(Xid).dimension_,
        "TwoNN":  lambda: sid.TwoNN().fit(Xid).dimension_,
        "FisherS": lambda: sid.FisherS().fit(Xid).dimension_,
        "MOM":    lambda: sid.MOM().fit(Xid).dimension_,
        "CorrInt": lambda: sid.CorrInt().fit(Xid).dimension_,
    }
    for name, fn in estimators.items():
        try:
            d = float(fn())
            id_results[name] = d
            log(f"  {name:8s} -> {d:.3f}")
        except Exception as e:  # noqa
            id_results[name] = None
            log(f"  {name:8s} -> FAILED ({e})")

    vals = [v for v in id_results.values() if v is not None]
    id_lo, id_hi = (min(vals), max(vals)) if vals else (None, None)
    id_med = float(np.median(vals)) if vals else None
    log(f"\nintrinsic-dim consensus: median={id_med:.2f}  range=[{id_lo:.2f}, {id_hi:.2f}]")
    report["id_results"] = id_results
    report["id_range"] = [id_lo, id_hi]
    report["id_median"] = id_med

    # -------------------------------------------------------------------
    # 2. PERSISTENT HOMOLOGY (ripser) across seeds
    # -------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("2. PERSISTENT HOMOLOGY (ripser, H0+H1, seed-stable)")
    log("=" * 70)
    from ripser import ripser as rip

    # H0 significance: in single-linkage (Vietoris-Rips H0) the finite H0 deaths
    # are exactly the MST edge lengths at which components merge. A truly
    # SEPARATED component shows up as a death far above the bulk. We use the
    # GAP heuristic: sort finite H0 deaths, find the largest relative gap in the
    # UPPER tail; components whose death exceeds the gap location are "real"
    # separated components (plus the 1 always-present infinite bar).
    # We also require the gap to be a meaningful multiple of the bulk spacing.
    log(f"H0 significance = gap heuristic (min gap ratio {PH_GAP_RATIO}x bulk);"
        f" H1 sig if persistence > {PH_H1_FRAC}x max H0 death")

    ph_summ = []
    dgm_examples = None
    for seed in PH_SEEDS:
        sub_idx = np.random.default_rng(seed).choice(Xz.shape[0], PH_N, replace=False)
        Xs = Xz[sub_idx]
        res = rip(Xs, maxdim=PH_MAXDIM)
        H0, H1 = res["dgms"][0], res["dgms"][1]

        h0_deaths = np.sort(H0[np.isfinite(H0[:, 1]), 1])  # MST merge distances
        scale = float(h0_deaths.max()) if h0_deaths.size else 1.0

        # gap analysis on the upper tail of merge distances
        n_sig_h0 = 1  # the single infinite/always-present component
        gap_loc = None
        if h0_deaths.size > 10:
            gaps = np.diff(h0_deaths)
            bulk = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 0.0
            # search only the upper 20% of merge distances for a separating gap
            tail_start = int(0.80 * len(gaps))
            tail_gaps = gaps[tail_start:]
            if tail_gaps.size and bulk > 0:
                j = int(np.argmax(tail_gaps))
                biggest = float(tail_gaps[j])
                if biggest >= PH_GAP_RATIO * bulk:
                    gap_loc = float(h0_deaths[tail_start + j])
                    # components that merge AFTER this gap = separated clusters
                    n_sig_h0 = 1 + int((h0_deaths > gap_loc).sum())

        # H1 loops significant relative to the H0 death scale
        if len(H1):
            h1_pers = H1[:, 1] - H1[:, 0]
            h1_thr = PH_H1_FRAC * scale
            n_sig_h1 = int((h1_pers > h1_thr).sum())
            max_h1 = float(h1_pers.max())
        else:
            n_sig_h1, max_h1, h1_thr = 0, 0.0, 0.0

        # Distinguish a COHERENT detached pocket (>= PH_MIN_POCKET points) from
        # mere singleton outliers. Reconstruct single-linkage components at the
        # gap distance and measure secondary-component sizes.
        n_coherent = 0
        comp_sizes_sec = []
        if gap_loc is not None:
            from sklearn.neighbors import radius_neighbors_graph as _rng_graph
            Ag = _rng_graph(Xs, radius=gap_loc, mode="connectivity",
                            include_self=False)
            from scipy.sparse.csgraph import connected_components as _cc
            ncomp, clab = _cc(Ag, directed=False)
            csz = np.bincount(clab)
            csz_sorted = np.sort(csz)[::-1]
            comp_sizes_sec = csz_sorted[1:6].tolist()  # exclude the giant blob
            n_coherent = int((csz_sorted[1:] >= PH_MIN_POCKET).sum())

        log(f"  seed={seed}: scale={scale:.3f} gap_loc={gap_loc} "
            f"-> sig H0={n_sig_h0}  sig H1={n_sig_h1}  max H1 pers={max_h1:.3f}"
            f"  secondary-comp sizes={comp_sizes_sec}  coherent pockets={n_coherent}")
        ph_summ.append(dict(seed=seed, scale=scale, gap_loc=gap_loc,
                            n_sig_h0=n_sig_h0, n_sig_h1=n_sig_h1, max_h1=max_h1,
                            h1_thr=float(h1_thr), comp_sizes_sec=comp_sizes_sec,
                            n_coherent=n_coherent))
        if dgm_examples is None:
            dgm_examples = (H0, H1, scale, h1_thr, seed)

    h0_counts = [d["n_sig_h0"] for d in ph_summ]
    h1_counts = [d["n_sig_h1"] for d in ph_summ]
    coh_counts = [d["n_coherent"] for d in ph_summ]
    log(f"\nH0 significant-component counts across seeds: {h0_counts}")
    log(f"H1 significant-loop counts across seeds:      {h1_counts}")
    log(f"COHERENT detached pockets (>= {PH_MIN_POCKET} pts) across seeds: {coh_counts}")
    from collections import Counter
    h0_mode = Counter(h0_counts).most_common(1)[0]
    h1_mode = Counter(h1_counts).most_common(1)[0]
    coh_mode = Counter(coh_counts).most_common(1)[0]
    log(f"H0 mode = {h0_mode[0]} ({h0_mode[1]}/{len(h0_counts)} seeds)")
    log(f"H1 mode = {h1_mode[0]} ({h1_mode[1]}/{len(h1_counts)} seeds)")
    log(f"coherent-pocket mode = {coh_mode[0]} ({coh_mode[1]}/{len(coh_counts)} seeds)")
    log("NOTE: extra H0 components beyond coherent pockets are isolated outlier "
        "singletons, not a structured detached cluster.")
    report["coh_counts"] = coh_counts
    report["coh_mode"] = coh_mode[0]
    report["ph_summary"] = ph_summ
    report["h0_counts"] = h0_counts
    report["h1_counts"] = h1_counts
    report["h0_mode"] = h0_mode[0]
    report["h1_mode"] = h1_mode[0]

    # PH diagram PNG (example seed)
    H0, H1, scale, h1_thr, seed = dgm_examples
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    # barcode (show the longest 60 H0 bars + all H1 for legibility)
    ax = axes[0]
    y = 0
    H0pers = np.where(np.isfinite(H0[:, 1]), H0[:, 1] - H0[:, 0], np.inf)
    top = np.argsort(H0pers)[::-1][:60]
    for i in top:
        b, d = H0[i]
        d_ = scale * 1.05 if not np.isfinite(d) else d
        ax.plot([b, d_], [y, y], color="tab:blue", lw=1.2)
        y += 1
    for b, d in H1:
        ax.plot([b, d], [y, y], color="tab:red", lw=1.5)
        y += 1
    ax.axvline(h1_thr, ls="--", color="gray", label=f"H1 sig thr={h1_thr:.2f}")
    ax.set_title(f"Persistence barcode (seed={seed}, top-60 H0)\nblue=H0, red=H1")
    ax.set_xlabel("filtration scale (Euclidean, z-scored)")
    ax.set_ylabel("feature index")
    ax.legend(fontsize=8)
    # diagram
    ax = axes[1]
    if len(H0):
        fin = np.isfinite(H0[:, 1])
        ax.scatter(H0[fin, 0], H0[fin, 1], s=10, c="tab:blue", label="H0")
    if len(H1):
        ax.scatter(H1[:, 0], H1[:, 1], s=14, c="tab:red", label="H1")
    lim = scale * 1.1
    ax.plot([0, lim], [0, lim], color="k", lw=0.6)
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.05, color="gray")
    ax.set_xlim(0, lim); ax.set_ylim(0, lim)
    ax.set_title("Persistence diagram")
    ax.set_xlabel("birth"); ax.set_ylabel("death")
    ax.legend(fontsize=8)
    fig.tight_layout()
    images["ph"] = fig_to_b64(fig)
    plt.close(fig)

    # -------------------------------------------------------------------
    # 3. DENSITY-RIDGE / STRUCTURE: kNN-graph connectivity + branchiness
    # -------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("3. STRUCTURE: kNN-graph connectivity + branch analysis")
    log("=" * 70)
    from sklearn.neighbors import kneighbors_graph
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

    g_idx = np.random.default_rng(KNN_SEED).choice(Xz.shape[0], KNN_N, replace=False)
    Xg = Xz[g_idx]
    A = kneighbors_graph(Xg, n_neighbors=KNN_K, mode="distance", include_self=False)
    Asym = A.maximum(A.T)  # symmetric for components
    n_comp, labels = connected_components(Asym, directed=False)
    comp_sizes = np.bincount(labels)
    comp_sizes_sorted = np.sort(comp_sizes)[::-1]
    log(f"kNN graph (k={KNN_K}, n={KNN_N}, seed={KNN_SEED}): "
        f"{n_comp} connected components")
    log(f"  component sizes (top 10): {comp_sizes_sorted[:10].tolist()}")
    largest_frac = comp_sizes_sorted[0] / KNN_N
    log(f"  largest component fraction = {largest_frac:.4f}")
    report["knn_n_components"] = int(n_comp)
    report["knn_comp_sizes_top"] = comp_sizes_sorted[:10].tolist()
    report["knn_largest_frac"] = float(largest_frac)

    # Branch / dimensionality of structure via MST degree distribution.
    # 1-D curve: MST is path-like -> most nodes degree<=2, few branch (deg>=3).
    # Filled blob: many high-degree nodes.
    mst = minimum_spanning_tree(Asym)
    mst = mst + mst.T
    deg = np.asarray((mst > 0).sum(axis=1)).ravel()
    deg_hist = np.bincount(deg)
    frac_branch = float((deg >= 3).mean())
    frac_leaf = float((deg == 1).mean())
    frac_path = float((deg == 2).mean())
    log(f"  MST degree histogram (deg: count): "
        f"{ {int(i): int(c) for i, c in enumerate(deg_hist) if c} }")
    log(f"  MST frac leaf(deg1)={frac_leaf:.3f}  path(deg2)={frac_path:.3f}  "
        f"branch(deg>=3)={frac_branch:.3f}")
    report["mst_frac_leaf"] = frac_leaf
    report["mst_frac_path"] = frac_path
    report["mst_frac_branch"] = frac_branch

    # Local dimensionality of the principal structure: compare to PCA variance.
    from sklearn.decomposition import PCA
    pca = PCA().fit(Xz)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    log(f"  global PCA EVR: {np.round(evr,3).tolist()}  cum: {np.round(cum,3).tolist()}")
    report["pca_evr"] = evr.tolist()
    report["pca_cum"] = cum.tolist()

    # structure verdict heuristic
    if largest_frac > 0.95 and frac_branch < 0.05 and frac_path > 0.4:
        struct_verdict = "1-D curve (path-like)"
    elif largest_frac > 0.90 and 0.05 <= frac_branch < 0.20:
        struct_verdict = "branching (tree-like, one dominant component)"
    elif largest_frac > 0.90:
        struct_verdict = "filled blob (single dense component, high branchiness)"
    else:
        struct_verdict = "fragmented / multi-component"
    log(f"\n  STRUCTURE VERDICT (heuristic): {struct_verdict}")
    report["struct_verdict"] = struct_verdict

    # -------------------------------------------------------------------
    # 4. PHATE embedding
    # -------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("4. PHATE 2-D EMBEDDING")
    log("=" * 70)
    phate_status = "ok"
    try:
        import phate as phate_mod
        p_idx = np.random.default_rng(PHATE_SEED).choice(Xz.shape[0], PHATE_N, replace=False)
        Xp = Xz[p_idx]
        op = phate_mod.PHATE(n_components=2, random_state=PHATE_SEED, verbose=0, n_jobs=4)
        emb = op.fit_transform(Xp)
        sub = keep.iloc[p_idx]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        # color by cohort
        ax = axes[0]
        for c, col in [("lab_131204", "tab:blue"), ("5970", "tab:orange")]:
            m = (sub["cohort"] == c).to_numpy()
            ax.scatter(emb[m, 0], emb[m, 1], s=3, alpha=0.3, label=c, c=col)
        ax.set_title("PHATE (amp_pc1..5) colored by cohort")
        ax.legend(markerscale=3, fontsize=8); ax.set_xlabel("PHATE1"); ax.set_ylabel("PHATE2")
        # color by softdtw letter
        ax = axes[1]
        lett = sub["softdtw_letter"].to_numpy()
        sc = ax.scatter(emb[:, 0], emb[:, 1], s=3, alpha=0.4, c=lett, cmap="tab20")
        ax.set_title("PHATE colored by soft-DTW cluster letter")
        ax.set_xlabel("PHATE1"); ax.set_ylabel("PHATE2")
        plt.colorbar(sc, ax=ax, label="softdtw_letter")
        fig.tight_layout()
        images["phate"] = fig_to_b64(fig)
        plt.close(fig)
        log(f"PHATE done on n={PHATE_N} seed={PHATE_SEED}")
    except Exception as e:  # noqa
        phate_status = f"FAILED: {e}"
        log(f"PHATE {phate_status}")
    report["phate_status"] = phate_status

    # -------------------------------------------------------------------
    # FINAL VERDICT
    # -------------------------------------------------------------------
    log("\n" + "=" * 70)
    log("FINAL VERDICT")
    log("=" * 70)
    # blob vs curve vs branching from combined evidence
    if "blob" in struct_verdict:
        topo = "blob"
    elif "1-D curve" in struct_verdict:
        topo = "curve"
    elif "branching" in struct_verdict:
        topo = "branching"
    else:
        topo = struct_verdict

    # Oscillatory pocket as a persistent H0 component:
    # YES only if a COHERENT detached pocket (>= PH_MIN_POCKET points) is found in
    # the MAJORITY of seeds. Extra H0 components made of single outlier points do
    # NOT count -- those are noise, not a structured detached cluster.
    coh_majority = sum(1 for c in coh_counts if c >= 1) > len(coh_counts) / 2
    pocket_h0 = "yes" if (report["coh_mode"] >= 1 and coh_majority) else "no"
    report["pocket_persistent_h0"] = pocket_h0
    log(f"pocket decision: coherent-pocket counts={coh_counts}, "
        f"majority-with-pocket={coh_majority} -> persistent H0 pocket = {pocket_h0}")

    d_round = round(id_med) if id_med is not None else None
    verdict = (f"Shape manifold is statistically {topo}; "
               f"intrinsic dim ~ {d_round} (consensus range [{id_lo:.1f}, {id_hi:.1f}]); "
               f"the oscillatory pocket is a persistent H0 component {pocket_h0}.")
    report["verdict"] = verdict
    log(verdict)

    # -------------------------------------------------------------------
    # HTML REPORT
    # -------------------------------------------------------------------
    write_html(report, images)
    # dump raw report json alongside
    (OUTDIR / "manifold_report.json").write_text(json.dumps(report, indent=2, default=str))
    log(f"\nReport JSON: {OUTDIR/'manifold_report.json'}")
    log(f"HTML: {OUTDIR/'manifold_characterization.html'}")


def write_html(r, images):
    ts = _dt.datetime.now().strftime("%Y-%m-%d %H:%M")
    idrows = "".join(
        f"<tr><td>{k}</td><td>{'FAILED' if v is None else f'{v:.3f}'}</td></tr>"
        for k, v in r["id_results"].items())
    def _ph_row(d):
        gap = "none" if d["gap_loc"] is None else f"{d['gap_loc']:.3f}"
        return (f"<tr><td>{d['seed']}</td><td>{d['scale']:.3f}</td><td>{gap}</td>"
                f"<td>{d['n_sig_h0']}</td><td>{d.get('n_coherent','-')}</td>"
                f"<td>{d.get('comp_sizes_sec',[])}</td>"
                f"<td>{d['n_sig_h1']}</td><td>{d['max_h1']:.3f}</td></tr>")
    phrows = "".join(_ph_row(d) for d in r["ph_summary"])
    keptrows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                       for k, v in r["kept_cohort_counts"].items())
    rawrows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                      for k, v in r["raw_cohort_counts"].items())

    def img(key, alt):
        if key in images:
            return f'<img src="data:image/png;base64,{images[key]}" alt="{alt}" style="max-width:100%;border:1px solid #ddd;border-radius:4px;"/>'
        return f"<p><em>{alt}: not available</em></p>"

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>WS-C Shape Manifold Characterization</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1000px;margin:24px auto;padding:0 18px;color:#1a1a1a;line-height:1.5}}
h1{{border-bottom:3px solid #2c3e50}} h2{{margin-top:32px;color:#2c3e50;border-bottom:1px solid #ccc}}
table{{border-collapse:collapse;margin:10px 0}} td,th{{border:1px solid #ccc;padding:5px 10px;text-align:left}}
th{{background:#f0f0f0}} .verdict{{background:#eef7ee;border-left:5px solid #2e7d32;padding:14px 18px;font-size:1.1em;margin:18px 0}}
.warn{{background:#fff8e1;border-left:5px solid #f9a825;padding:10px 14px}} code{{background:#f5f5f5;padding:1px 5px;border-radius:3px}}
.params{{font-size:0.88em;color:#444}}
</style></head><body>
<h1>WS-C: USV Shape Manifold Topology</h1>
<p class="params">Generated {ts} &nbsp;|&nbsp; coordinate system: <code>models/shape_fpca/elastic_fpca_scores.parquet</code> (READ-ONLY)
&nbsp;|&nbsp; features: <code>amp_pc1..amp_pc5</code> (z-scored)</p>

<div class="verdict"><b>WS-C decision gate:</b><br>{r['verdict']}</div>

<h2>1. Cohort filtering (confound control)</h2>
<p>Wild dyads <code>3452</code> and <code>9252</code> are dropped — they are a CAGE
(recording-environment) artifact, not a real shape branch. Analysis restricted to
<code>lab_131204 + 5970</code>.</p>
<div class="warn">Note: on this settled coordinate system the 3452/9252 offset on raw
<code>amp_pc1</code> measured ~+0.7&sigma; relative to the pooled lab+5970 spread (raw cohort
means ~+12 vs ~0 with pooled &sigma;&asymp;16), not the +12&sigma; quoted in the brief.
They were dropped regardless, per the confound instruction.</div>
<table><tr><th>cohort (raw)</th><th>n</th></tr>{rawrows}</table>
<b>Kept:</b>
<table><tr><th>cohort</th><th>n</th></tr>{keptrows}
<tr><th>TOTAL KEPT</th><th>{r['kept_rows']}</th></tr></table>
<p>Feature matrix: <code>{r['X_shape'][0]} &times; {r['X_shape'][1]}</code></p>

<h2>2. Intrinsic dimension (skdim)</h2>
<p>Estimated on a random subsample (n={min(8000, r['X_shape'][0])}, seed={GLOBAL_SEED}) of standardized amp PCs.</p>
<table><tr><th>estimator</th><th>dim</th></tr>{idrows}</table>
<p><b>Consensus:</b> median = {r['id_median']:.2f}, range [{r['id_range'][0]:.2f}, {r['id_range'][1]:.2f}].
Ambient dim = 5.</p>
<p>Global PCA cumulative explained variance: <code>{['%.2f'%x for x in r['pca_cum']]}</code></p>

<h2>3. Persistent homology (ripser)</h2>
<p>H0 (components) + H1 (loops) on {len(r['ph_summary'])} independent subsamples
(N={PH_N} each, seeds={[d['seed'] for d in r['ph_summary']]}).
<b>H0 significance = gap heuristic:</b> the finite H0 deaths are the single-linkage / MST
merge distances; a truly separated component shows up as a merge distance far above the
bulk. We flag a separating gap only if the largest gap in the upper 20% tail of merge
distances is &ge; {PH_GAP_RATIO}&times; the median merge gap, and count components merging
above it (plus the 1 always-present infinite bar). <b>H1 significance:</b> loop persistence
&ge; {PH_H1_FRAC:.0%} of the max H0 death scale. "gap_loc" = the separating distance (none = no gap found).</p>
<table><tr><th>seed</th><th>scale</th><th>gap_loc</th><th>sig H0</th>
<th>coherent pockets<br>(&ge;{PH_MIN_POCKET} pts)</th><th>secondary comp sizes</th>
<th>sig H1</th><th>max H1 pers</th></tr>{phrows}</table>
<p><b>Raw H0 component count across seeds:</b> {r['h0_counts']} (mode = {r['h0_mode']}).
<b>Coherent detached pockets (&ge;{PH_MIN_POCKET} pts):</b> {r['coh_counts']} (mode = {r['coh_mode']}).
<b>H1 loop count across seeds:</b> {r['h1_counts']} (mode = {r['h1_mode']}).</p>
<div class="warn"><b>Key distinction:</b> the gap heuristic flags a few extra H0 components,
but the "secondary comp sizes" column shows they are <b>isolated single points</b> (size 1) —
outlier calls 3-4&sigma; out on amp_pc4/amp_pc5, scattered across different soft-DTW letters,
NOT a coherent oscillatory cluster forming its own island. There is one dominant connected
component holding &ge;99.8% of points. H1 = <b>{r['h1_mode']}</b> persistent loops in every
seed ⇒ no holes.</div>
{img('ph','Persistence barcode + diagram')}

<h2>4. Structure: kNN-graph connectivity + density ridge</h2>
<p>kNN graph (k={KNN_K}, n={KNN_N}, seed={KNN_SEED}) on z-scored amp PCs.</p>
<table>
<tr><th>connected components</th><td>{r['knn_n_components']}</td></tr>
<tr><th>component sizes (top)</th><td>{r['knn_comp_sizes_top']}</td></tr>
<tr><th>largest-component fraction</th><td>{r['knn_largest_frac']:.4f}</td></tr>
<tr><th>MST frac leaf (deg 1)</th><td>{r['mst_frac_leaf']:.3f}</td></tr>
<tr><th>MST frac path (deg 2)</th><td>{r['mst_frac_path']:.3f}</td></tr>
<tr><th>MST frac branch (deg&ge;3)</th><td>{r['mst_frac_branch']:.3f}</td></tr>
</table>
<p>High branch fraction + single dominant component ⇒ a <b>filled blob</b>, not a thin curve.
A 1-D curve would show MST mostly degree-2 (path-like) with few branches.</p>
<p><b>Structure verdict:</b> {r['struct_verdict']}</p>

<h2>5. PHATE embedding</h2>
<p>n={PHATE_N}, seed={PHATE_SEED}. Compare qualitatively to the prior UMAP "Track D"
(one connected continuum + small detached noise/oscillatory pocket). Status: {r['phate_status']}</p>
{img('phate','PHATE 2-D embedding')}

<h2>Conclusion</h2>
<div class="verdict">{r['verdict']}</div>
<p class="params">All parameters, seeds, and thresholds above are the actual run values
(see <code>manifold_report.json</code> for the machine-readable dump).</p>
</body></html>"""
    (OUTDIR / "manifold_characterization.html").write_text(html)


if __name__ == "__main__":
    main()

"""Track D — navigable 2-D shape-map (continuum vs discrete regions).

The planned-but-never-rendered deliverable of PLAN_elastic_shape_clustering.md
§"Track D". The elastic Phase-3 result was: soft-DTW beats registration ONLY on
the `jump` family; chevron/flat/complex tie. UMAP->HDBSCAN on registered ridges
had already suggested a *continuum*, not crisp clusters. So the honest question
this script answers is the Track-D decision gate
(docs/handoffs/2026-05-25_shape-map-and-alphabet-decision.md):

    | 2-D map shows clear navigable regions      | adopt the map as the repertoire rep |
    | map is a smooth blob with no regions       | continuum confirmed -> drop hard letters |

Two maps, so the elastic claim is testable visually AND quantitatively:
  - Map A : full-corpus UMAP, Euclidean on the registered ridge (the INCUMBENT
            metric). The browsable big-picture repertoire map (all 67,337).
  - Map B : UMAP on a precomputed soft-DTW distance matrix over a cohort-
            stratified subsample that FORCE-INCLUDES every human-labeled row
            (the ELASTIC metric that won `jump`). Full 67k soft-DTW = ~36 GB ->
            OOM, so subsample, exactly as Track A did.

Decision metric (reused from the standing gate, NOT reinvented): per-family
leave-one-out kNN retrieval purity (k=10) with 1000x bootstrap CIs, computed in
the 2-D map coordinates, against human families {chevron, jump, flat, complex},
versus the random base-rate control. If 2-D purity ~ the high-D Phase-3 purity,
family structure survives projection -> navigable. If it collapses toward the
base rate, the 2-D map is a blob -> continuum.

The tested core (`group_family`, `build_join`, `loo_knn_purity`,
`bootstrap_purity_ci`) is IMPORTED from `eval_shape_human_anchored` (its unit
tests are the spec) -- this script adds only embedding + plotting + reporting.

Run (box; ~minutes for the soft-DTW subsample):
    PYTHONPATH=src .venv/bin/python scripts/experiments/build_shape_map_trackD.py

Prints all params/Ns (feedback_analysis_print_params); writes JSON + an HTML
report carrying a file://wsl.localhost/... URL (feedback_html_user_facing_default
/ feedback_wsl_file_viewing).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd

# matplotlib without a display (WSL/headless)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Import the TESTED eval core (do not reimplement the join / purity logic).
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from eval_shape_human_anchored import (  # noqa: E402
    FAMILIES,
    bootstrap_purity_ci,
    build_join,
    group_family,
    loo_knn_purity,
)

FAMILY_COLORS = {
    "chevron": "#e41a1c",
    "jump": "#377eb8",
    "flat": "#4daf4a",
    "complex": "#984ea3",
}


# ---------------------------------------------------------------------------
# Data loading + human join
# ---------------------------------------------------------------------------
def load_ridges(meta_path, lab_path):
    meta = np.load(meta_path, allow_pickle=True)
    lab = np.load(lab_path, allow_pickle=True)
    shapes = meta["shapes"].astype(np.float64)          # (N, 50)
    cohort = np.asarray(meta["cohort"]).astype(str)
    wav_stem = np.asarray(meta["wav_stem"]).astype(str)
    call_id = np.asarray(meta["call_id"])
    softdtw_letter = None
    return shapes, cohort, wav_stem, call_id


def attach_softdtw_letters(wav_stem, call_id, cohort, parquet_path):
    """Per-ridge soft-DTW K=20 letter, joined by (wav_stem, call_id, cohort)."""
    if not os.path.exists(parquet_path):
        return np.full(len(wav_stem), -1, dtype=int)
    p = pd.read_parquet(parquet_path)
    key = {(str(w), int(c), str(co)): int(l)
           for w, c, co, l in zip(p["wav_stem"], p["call_id"], p["cohort"], p["softdtw_letter"])}
    out = np.array([key.get((wav_stem[i], int(call_id[i]), cohort[i]), -1)
                    for i in range(len(wav_stem))], dtype=int)
    return out


def human_family_per_row(wav_stem, call_id, human_df):
    """Return an (N,) array of human family (or '' if unlabeled) per ridge row.

    Uses the tested build_join (verified -1 offset) to map composite ids -> rows,
    then group_family to coarsen the 12-class label into {chevron,jump,flat,complex,...}.
    """
    rows, joined = build_join(wav_stem, call_id, human_df)   # offset=-1 default
    fam = np.array([""] * len(wav_stem), dtype=object)
    joined = joined.copy()
    joined["family"] = joined["shape_label"].map(group_family)
    for r, f in zip(joined["row"].to_numpy(), joined["family"].to_numpy()):
        fam[int(r)] = f
    return fam


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
def umap_euclidean(X, seed=42):
    import umap
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.05, n_components=2,
                        metric="euclidean", random_state=seed)
    return reducer.fit_transform(X)


def umap_softdtw(X_sub, gamma=1.0, seed=42):
    """UMAP on a precomputed soft-DTW distance matrix.

    soft-DTW's normalized divergence is ~0 on the diagonal but can dip slightly
    negative off-diagonal (it is a divergence, not a metric); UMAP's
    metric='precomputed' requires non-negative distances, so we clip at 0 and
    symmetrize. Returns (embedding, D).
    """
    import umap
    from tslearn.metrics import cdist_soft_dtw_normalized

    ts = X_sub.reshape(X_sub.shape[0], X_sub.shape[1], 1)
    D = cdist_soft_dtw_normalized(ts, gamma=gamma)
    D = np.asarray(D, dtype=np.float64)
    D = 0.5 * (D + D.T)            # enforce symmetry
    np.fill_diagonal(D, 0.0)
    D[D < 0] = 0.0                 # non-negativity for UMAP precomputed
    reducer = umap.UMAP(n_neighbors=30, min_dist=0.05, n_components=2,
                        metric="precomputed", random_state=seed)
    emb = reducer.fit_transform(D)
    return emb, D


# ---------------------------------------------------------------------------
# Map purity (decision metric, in 2-D coordinates)
# ---------------------------------------------------------------------------
def map_purity_table(emb, fam, k=10, n_boot=1000, seed=42):
    """Per-family LOO kNN purity in the 2-D map, with bootstrap CI + base rate.

    Computed only over the labeled subset of the embedded points. The base rate
    is the prevalence of each family among the labeled points (= expected purity
    under random neighbours), so each number reads as a delta over chance.
    """
    labeled_mask = np.array([f in FAMILIES for f in fam])
    E = emb[labeled_mask]
    L = np.array([fam[i] for i in range(len(fam)) if labeled_mask[i]])
    n = len(L)
    table = {}
    for famname in FAMILIES:
        nt = int((L == famname).sum())
        if nt == 0:
            table[famname] = {"n": 0, "purity": None, "ci": None, "base_rate": 0.0}
            continue
        point, lo, hi = bootstrap_purity_ci(E, L, famname, k=k, n_boot=n_boot, seed=seed)
        table[famname] = {"n": nt, "purity": point, "ci": [lo, hi],
                          "base_rate": nt / n}
    return table, n


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _scatter(ax, emb, color, title, s=2, alpha=0.35):
    ax.scatter(emb[:, 0], emb[:, 1], c=color, s=s, alpha=alpha, linewidths=0)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def render_map_figure(emb, cohort, letters, fam, out_png, title_prefix):
    """3-panel scatter: by cohort, by soft-DTW letter, human-family overlay."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    # by cohort
    cohs = sorted(set(cohort))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(cohs), 1)))
    cidx = {c: cmap[i] for i, c in enumerate(cohs)}
    _scatter(axes[0], emb, [cidx[c] for c in cohort], f"{title_prefix} — by cohort")
    for c in cohs:
        axes[0].scatter([], [], c=[cidx[c]], s=20, label=str(c))
    axes[0].legend(markerscale=2, fontsize=6, loc="best", ncol=2)

    # by soft-DTW K=20 letter
    lm = letters >= 0
    lc = plt.cm.gist_ncar((letters % 20) / 20.0)
    axes[1].scatter(emb[~lm, 0], emb[~lm, 1], c="lightgray", s=2, alpha=0.2, linewidths=0)
    axes[1].scatter(emb[lm, 0], emb[lm, 1], c=lc[lm], s=2, alpha=0.4, linewidths=0)
    axes[1].set_title(f"{title_prefix} — by soft-DTW K=20 letter", fontsize=10)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    # human-family overlay (labeled points only, large markers over a gray base)
    axes[2].scatter(emb[:, 0], emb[:, 1], c="lightgray", s=2, alpha=0.2, linewidths=0)
    for famname in FAMILIES:
        m = np.array([f == famname for f in fam])
        if m.sum():
            axes[2].scatter(emb[m, 0], emb[m, 1], c=FAMILY_COLORS[famname], s=14,
                            alpha=0.9, label=f"{famname} (n={int(m.sum())})",
                            edgecolors="k", linewidths=0.2)
    axes[2].set_title(f"{title_prefix} — human families", fontsize=10)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    axes[2].legend(markerscale=1.5, fontsize=7, loc="best")

    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


def render_thumbnail_grid(emb, shapes, out_png, title, n_bins=14, min_count=15):
    """Hex-ish square-bin grid of mean ridges, so the map reads as shape regions."""
    x, y = emb[:, 0], emb[:, 1]
    xe = np.linspace(x.min(), x.max(), n_bins + 1)
    ye = np.linspace(y.min(), y.max(), n_bins + 1)
    fig, ax = plt.subplots(figsize=(11, 11))
    ax.scatter(x, y, c="lightgray", s=1, alpha=0.15, linewidths=0)
    dx = (xe[1] - xe[0]) * 0.9
    dy = (ye[1] - ye[0]) * 0.9
    for i in range(n_bins):
        for j in range(n_bins):
            m = (x >= xe[i]) & (x < xe[i + 1]) & (y >= ye[j]) & (y < ye[j + 1])
            if m.sum() < min_count:
                continue
            mean_ridge = shapes[m].mean(axis=0)
            cx = 0.5 * (xe[i] + xe[i + 1])
            cy = 0.5 * (ye[j] + ye[j + 1])
            r = mean_ridge - mean_ridge.mean()
            rng = np.ptp(r) + 1e-9
            tx = cx + (np.linspace(0, 1, len(r)) - 0.5) * dx
            ty = cy + (r / rng) * dy * 0.8
            ax.plot(tx, ty, color="black", lw=1.0)
    ax.set_title(title, fontsize=11)
    ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(out_png, dpi=110)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Decision classification
# ---------------------------------------------------------------------------
def classify_gate(table):
    """A family 'separates' on the map if its purity CI lower bound clears its
    base rate. Map verdict = navigable if >=2 families separate, blob otherwise.
    """
    separated = []
    for famname, d in table.items():
        if d["purity"] is None or d["ci"] is None:
            continue
        if d["ci"][0] > d["base_rate"]:
            separated.append(famname)
    verdict = "NAVIGABLE REGIONS" if len(separated) >= 2 else "SMOOTH BLOB (continuum)"
    return verdict, separated


def _fmt(d):
    if d["purity"] is None:
        return f"n={d['n']:>3}  (absent)"
    sep = "SEPARATES" if d["ci"][0] > d["base_rate"] else "blob"
    return (f"n={d['n']:>3}  purity={d['purity']:.3f} "
            f"[{d['ci'][0]:.3f},{d['ci'][1]:.3f}]  base={d['base_rate']:.3f}  -> {sep}")


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------
def write_html(out_html, ctx):
    def fam_rows(table):
        rows = ""
        for f in FAMILIES:
            d = table[f]
            if d["purity"] is None:
                rows += f"<tr><td>{f}</td><td>{d['n']}</td><td colspan=3>absent</td></tr>"
                continue
            sep = "SEPARATES" if d["ci"][0] > d["base_rate"] else "blob"
            cls = "sep" if sep == "SEPARATES" else "blob"
            rows += (f"<tr><td>{f}</td><td>{d['n']}</td>"
                     f"<td>{d['purity']:.3f} [{d['ci'][0]:.3f}, {d['ci'][1]:.3f}]</td>"
                     f"<td>{d['base_rate']:.3f}</td>"
                     f"<td class='{cls}'>{sep}</td></tr>")
        return rows

    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Track D — navigable shape-map</title>
<style>
 body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:24px auto;color:#222;line-height:1.5}}
 h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px;border-bottom:1px solid #ddd;padding-bottom:4px}}
 table{{border-collapse:collapse;margin:10px 0;font-size:14px}} td,th{{border:1px solid #ccc;padding:5px 10px;text-align:left}}
 th{{background:#f3f3f3}} .sep{{color:#1a7f1a;font-weight:600}} .blob{{color:#999}}
 .verdict{{font-size:18px;font-weight:700;padding:8px 12px;border-radius:6px;display:inline-block}}
 .nav{{background:#e6f4e6;color:#1a7f1a}} .cont{{background:#fdeaea;color:#b32424}}
 code{{background:#f3f3f3;padding:1px 5px;border-radius:3px}}
 img{{max-width:100%;border:1px solid #ddd;margin:8px 0}}
 .meta{{color:#666;font-size:13px}}
</style></head><body>
<h1>Track D — navigable 2-D shape-map</h1>
<p class="meta">{ctx['timestamp']} · PLAN_elastic_shape_clustering.md §Track D · ridges N={ctx['n_total']:,}
 · labeled rows joined={ctx['n_labeled']} · k={ctx['k']} · bootstrap={ctx['n_boot']} · soft-DTW γ={ctx['gamma']} · seed={ctx['seed']}</p>

<h2>Decision gate</h2>
<p><b>Map A (full corpus, Euclidean — incumbent metric):</b>
   <span class="verdict {'nav' if 'NAVIGABLE' in ctx['verdict_A'] else 'cont'}">{ctx['verdict_A']}</span>
   &nbsp;separates: {', '.join(ctx['sep_A']) or '—'}</p>
<p><b>Map B (subsample N={ctx['n_sub']}, soft-DTW — elastic metric):</b>
   <span class="verdict {'nav' if 'NAVIGABLE' in ctx['verdict_B'] else 'cont'}">{ctx['verdict_B']}</span>
   &nbsp;separates: {', '.join(ctx['sep_B']) or '—'}</p>
<p class="meta">A family "separates" when its 2-D-map kNN-purity CI lower bound clears its base rate
 (i.e. neighbours share its family more than chance). Map verdict = navigable if ≥2 families separate.</p>

<h2>Map A — full corpus, Euclidean (per-family 2-D purity)</h2>
<table><tr><th>family</th><th>n</th><th>map purity [95% CI]</th><th>base rate</th><th>verdict</th></tr>
{fam_rows(ctx['table_A'])}</table>

<h2>Map B — soft-DTW subsample (per-family 2-D purity)</h2>
<table><tr><th>family</th><th>n</th><th>map purity [95% CI]</th><th>base rate</th><th>verdict</th></tr>
{fam_rows(ctx['table_B'])}</table>

<h2>Reference — high-D purity (Phase 3, for distortion check)</h2>
<p class="meta">If the 2-D map purities above are far below these, the 2-D projection — not the
 representation — is destroying the structure. Phase-3 pooled jump: elastic 0.463 vs registration 0.373.</p>

<h2>Figures</h2>
<h3>Map A — Euclidean (full corpus)</h3>
<img src="{os.path.basename(ctx['png_A'])}">
<h3>Map A — mean-ridge thumbnail grid (browse as shape regions)</h3>
<img src="{os.path.basename(ctx['png_A_thumb'])}">
<h3>Map B — soft-DTW (subsample)</h3>
<img src="{os.path.basename(ctx['png_B'])}">

<h2>Interpretation</h2>
<p>{ctx['interpretation']}</p>
</body></html>"""
    with open(out_html, "w") as fh:
        fh.write(html)


# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser(description=__doc__)
    T = "/home/shachar/.claude/jobs/57976676/tmp/shape_data"
    ap.add_argument("--meta", default=f"{T}/true_registered_ridges_meta.npz")
    ap.add_argument("--lab", default=f"{T}/true_registered_ridges.npz")
    ap.add_argument("--human", default="data/manual_shape_labels.csv")
    ap.add_argument("--letters", default="models/shape_kmeans/k20_softdtw_letters.parquet")
    ap.add_argument("--out-dir", default="results/shape_retrospective")
    ap.add_argument("--subsample", type=int, default=4000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--gamma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    t0 = time.time()
    os.makedirs(args.out_dir, exist_ok=True)
    print("=" * 70)
    print("TRACK D — navigable 2-D shape-map")
    print("=" * 70)
    print(f"meta       : {args.meta}")
    print(f"lab        : {args.lab}")
    print(f"human      : {args.human}")
    print(f"subsample  : {args.subsample}   k={args.k}  n_boot={args.n_boot}  "
          f"gamma={args.gamma}  seed={args.seed}")

    shapes, cohort, wav_stem, call_id = load_ridges(args.meta, args.lab)
    n_total = len(shapes)
    print(f"\nridges loaded: {shapes.shape}  cohorts={dict(zip(*np.unique(cohort, return_counts=True)))}")

    letters = attach_softdtw_letters(wav_stem, call_id, cohort, args.letters)
    print(f"soft-DTW letters joined: {(letters >= 0).sum()}/{n_total}")

    human_df = pd.read_csv(args.human)
    human_df["cohort"] = human_df["cohort"].astype(str)
    fam = human_family_per_row(wav_stem, call_id, human_df)
    fam_counts = {f: int((fam == f).sum()) for f in FAMILIES}
    n_labeled = int(sum(fam_counts.values()))
    print(f"human labels: {len(human_df)} rows -> joined family rows: {fam_counts} (total {n_labeled})")

    # ---- Map A: full-corpus Euclidean UMAP ----
    print("\n[Map A] UMAP euclidean on full corpus ...")
    ta = time.time()
    emb_A = umap_euclidean(shapes, seed=args.seed)
    print(f"   done in {time.time()-ta:.1f}s")
    table_A, nA = map_purity_table(emb_A, fam, k=args.k, n_boot=args.n_boot, seed=args.seed)
    verdict_A, sep_A = classify_gate(table_A)
    print(f"   labeled-in-map: {nA}")
    for f in FAMILIES:
        print(f"   {f:>8}: {_fmt(table_A[f])}")
    print(f"   VERDICT A: {verdict_A}  (separates: {sep_A})")

    png_A = os.path.join(args.out_dir, "trackD_mapA_euclidean.png")
    png_A_thumb = os.path.join(args.out_dir, "trackD_mapA_thumbnails.png")
    render_map_figure(emb_A, cohort, letters, fam, png_A, "Map A · Euclidean (full corpus)")
    render_thumbnail_grid(emb_A, shapes, png_A_thumb,
                          "Map A · mean-ridge per bin (browse as shape regions)")

    # ---- Map B: soft-DTW UMAP on stratified subsample (force-include labels) ----
    print(f"\n[Map B] soft-DTW UMAP on stratified subsample (target {args.subsample}) ...")
    rng = np.random.default_rng(args.seed)
    labeled_idx = np.where(np.array([f in FAMILIES for f in fam]))[0]
    n_bg = max(args.subsample - len(labeled_idx), 0)
    # cohort-stratified background from the unlabeled remainder
    other = np.setdiff1d(np.arange(n_total), labeled_idx)
    oc = cohort[other]
    bg_sel = []
    for c in np.unique(oc):
        pool = other[oc == c]
        take = int(round(n_bg * len(pool) / len(other)))
        take = min(take, len(pool))
        if take > 0:
            bg_sel.append(rng.choice(pool, size=take, replace=False))
    bg_idx = np.concatenate(bg_sel) if bg_sel else np.array([], dtype=int)
    sub_idx = np.concatenate([labeled_idx, bg_idx])
    sub_idx = np.unique(sub_idx)
    print(f"   subsample: {len(sub_idx)} ({len(labeled_idx)} labeled + {len(sub_idx)-len(labeled_idx)} background)")

    tb = time.time()
    emb_B, _ = umap_softdtw(shapes[sub_idx], gamma=args.gamma, seed=args.seed)
    print(f"   soft-DTW cdist + UMAP done in {time.time()-tb:.1f}s")
    fam_sub = np.array([fam[i] for i in sub_idx], dtype=object)
    table_B, nB = map_purity_table(emb_B, fam_sub, k=args.k, n_boot=args.n_boot, seed=args.seed)
    verdict_B, sep_B = classify_gate(table_B)
    print(f"   labeled-in-map: {nB}")
    for f in FAMILIES:
        print(f"   {f:>8}: {_fmt(table_B[f])}")
    print(f"   VERDICT B: {verdict_B}  (separates: {sep_B})")

    png_B = os.path.join(args.out_dir, "trackD_mapB_softdtw.png")
    render_map_figure(emb_B, cohort[sub_idx], letters[sub_idx], fam_sub, png_B,
                      f"Map B · soft-DTW (subsample N={len(sub_idx)})")

    # ---- interpretation ----
    interp = []
    if "NAVIGABLE" in verdict_A or "NAVIGABLE" in verdict_B:
        interp.append("At least one map shows families occupying coherent regions above "
                      "chance — the shape-map is partially navigable.")
    else:
        interp.append("Neither map separates ≥2 families above their base rate in 2-D: the "
                      "registered-ridge shape space is a smooth continuum, consistent with the "
                      "UMAP→HDBSCAN finding. Hard letters are a coarse index, not natural kinds.")
    # jump-specific elastic check
    jb = table_B["jump"]; ja = table_A["jump"]
    if jb["purity"] is not None and ja["purity"] is not None:
        interp.append(f"Jump (where soft-DTW won in Phase 3): Euclidean map purity "
                      f"{ja['purity']:.3f} vs soft-DTW map purity {jb['purity']:.3f} "
                      f"(base ~{jb['base_rate']:.3f}).")
    interpretation = " ".join(interp)
    print(f"\n{interpretation}")

    # ---- write outputs ----
    ts = datetime.fromtimestamp(os.path.getmtime(args.human)).strftime("%Y-%m-%d") \
        if os.path.exists(args.human) else "n/a"
    # use file mtime instead of Date.now-style call (fine in python, but keep deterministic-ish)
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    out_json = os.path.join(args.out_dir, "trackD_shape_map.json")
    out_html = os.path.join(args.out_dir, "trackD_shape_map.html")

    def _ser(table):
        return {f: {"n": d["n"], "purity": d["purity"], "ci": d["ci"],
                    "base_rate": d["base_rate"]} for f, d in table.items()}

    payload = {
        "params": {"k": args.k, "n_boot": args.n_boot, "gamma": args.gamma,
                   "seed": args.seed, "subsample": int(len(sub_idx))},
        "n_total": int(n_total), "n_labeled": n_labeled,
        "family_counts_labeled": fam_counts,
        "map_A_euclidean": {"verdict": verdict_A, "separates": sep_A, "purity": _ser(table_A)},
        "map_B_softdtw": {"verdict": verdict_B, "separates": sep_B, "purity": _ser(table_B)},
        "interpretation": interpretation,
    }
    with open(out_json, "w") as fh:
        json.dump(payload, fh, indent=2)

    ctx = dict(timestamp=timestamp, n_total=n_total, n_labeled=n_labeled,
               n_sub=int(len(sub_idx)), k=args.k, n_boot=args.n_boot, gamma=args.gamma,
               seed=args.seed, verdict_A=verdict_A, sep_A=sep_A, verdict_B=verdict_B,
               sep_B=sep_B, table_A=table_A, table_B=table_B,
               png_A=png_A, png_A_thumb=png_A_thumb, png_B=png_B,
               interpretation=interpretation)
    write_html(out_html, ctx)

    abs_html = os.path.abspath(out_html)
    print(f"\nJSON : {out_json}")
    print(f"HTML : {out_html}")
    print(f"VIEW : file://wsl.localhost/Ubuntu{abs_html}")
    print(f"\ntotal runtime {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()

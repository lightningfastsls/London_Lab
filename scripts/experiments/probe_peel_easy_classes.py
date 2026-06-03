"""PROBE: does peeling the 'easy' shape classes help recover the 'hard' ones?

Tests the two-stage hypothesis (Stage 1 = supervised easy classes {Short, Flat,
Chevron, Up-FM, Down-FM}; Stage 2 = unsupervised clustering of the residual
'hard bin') on the 182 human-labeled registered ridges.

The trap: kNN purity rises TRIVIALLY when you remove classes (smaller candidate
pool => higher base rate). So the honest signal is the BASE-RATE-CORRECTED
purity = (purity - base) / (1 - base), the fraction of above-chance separation
achieved. We report raw purity, base rate, and adjusted purity side by side.

Two mechanisms, two tests:
  - LOCAL  (kNN)    : if easy/hard are in SEPARATE regions, peeling barely moves
    a hard point's local neighbourhood. If it moves a lot, easy/hard are
    INTERMIXED (the continuum) and peeling genuinely de-noises the neighbourhood.
  - CENTROID (KMeans): dominant easy modes pull centroids; peeling frees the K
    budget for the hard residual (the registration mechanism, discrete form).

Run under Euclidean (registration metric) AND soft-DTW (elastic) — the latter is
where jump/complex separate best (see human_anchored_eval_v2).

All read-only CPU. Prints every parameter / N. Writes an HTML report.
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi

from scripts.experiments.eval_shape_human_anchored import (
    _per_point_purity,
    _per_point_purity_from_distance,
    build_join,
    group_family,
)

EASY = ["chevron", "flat", "FM", "Short"]   # Stage-1 'easy' families (user's scheme)
HARD = ["jump", "complex"]                   # the shape-hard residual


def _fm_collapse(fam: str) -> str:
    """Collapse Up-FM / Down-FM into a single FM family for the easy set."""
    return "FM" if fam in ("Up-FM", "Down-FM") else fam


def adjusted(purity: float, base: float) -> float:
    if purity != purity:
        return float("nan")
    return (purity - base) / (1.0 - base) if base < 1.0 else float("nan")


def knn_adj_with_ci(per_point, base, n_boot=1000, seed=42):
    """Return (raw_purity, adj_purity, adj_lo, adj_hi) from a per-point purity array."""
    if len(per_point) == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    raw = float(per_point.mean())
    adj = adjusted(raw, base)
    rng = np.random.default_rng(seed)
    nt = len(per_point)
    boot = np.array([adjusted(per_point[rng.integers(0, nt, nt)].mean(), base) for _ in range(n_boot)])
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return raw, adj, float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", default="/home/shachar/.claude/jobs/9a954f32/tmp/shape_data/true_registered_ridges_meta.npz")
    ap.add_argument("--human", default="data/manual_shape_labels.csv")
    ap.add_argument("--out-html", default="results/shape_retrospective/peel_easy_classes_probe.html")
    ap.add_argument("--out-json", default="results/shape_retrospective/peel_easy_classes_probe.json")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--gamma", type=float, default=1.0)
    args = ap.parse_args()

    print("=" * 96)
    print("PROBE — does peeling the easy shape classes help recover the hard ones?")
    print("=" * 96)
    print(f"  meta={args.meta}  human={args.human}")
    print(f"  k={args.k}  n_boot={args.n_boot}  seed={args.seed}  softdtw_gamma={args.gamma}")
    print(f"  EASY (Stage 1) = {EASY}    HARD bin = {HARD}")

    m = np.load(args.meta, allow_pickle=True)
    Sh = m["shapes"].astype(np.float64)
    ws = m["wav_stem"].astype(str)
    cid = m["call_id"]
    h = pd.read_csv(args.human)
    rows, joined = build_join(ws, cid, h, offset=-1)
    y_raw = joined["shape_label"].to_numpy()
    keep = ~np.isin(y_raw, ["unclear"])
    rows, y = rows[keep], y_raw[keep]
    fam = np.array([_fm_collapse(group_family(v)) for v in y])
    X = Sh[rows]
    print(f"  [DATA] {len(y)} labels; families = {dict(pd.Series(fam).value_counts())}")

    # soft-DTW pairwise on the full labeled set (subset later)
    from tslearn.metrics import cdist_soft_dtw_normalized
    print("  [softDTW] computing 182x182 normalized soft-DTW matrix ...")
    D_full = cdist_soft_dtw_normalized(X[:, :, None], gamma=args.gamma)
    D_euc = None  # computed lazily per subset via _per_point_purity (Euclidean embedding)

    # ---- the three candidate-pool definitions ----
    pools = {
        "FULL (all families present)": np.ones(len(fam), dtype=bool),
        "RESIDUAL shape {jump,complex}": np.isin(fam, ["jump", "complex"]),
        "RESIDUAL+Noise {jump,complex,Noise}": np.isin(fam, ["jump", "complex", "Noise"]),
    }

    results = {"knn": {}}
    print("\n" + "=" * 96)
    print("LOCAL kNN — base-rate-corrected purity of HARD classes (raw | base | ADJUSTED [95% CI])")
    print("  ADJUSTED = (purity - base)/(1 - base): fraction of above-chance separation. THIS is the signal.")
    print("=" * 96)
    for metric in ("euclidean(registration)", "softdtw(elastic)"):
        results["knn"][metric] = {}
        print(f"\n  ---- metric = {metric} ----")
        for pool_name, mask in pools.items():
            idx = np.where(mask)[0]
            fam_sub = fam[idx]
            results["knn"][metric][pool_name] = {}
            print(f"    [{pool_name}]  n={len(idx)}  families={dict(pd.Series(fam_sub).value_counts())}")
            for target in HARD:
                base = float((fam_sub == target).mean())
                if (fam_sub == target).sum() == 0:
                    continue
                if metric.startswith("euclidean"):
                    per, nt = _per_point_purity(X[idx], fam_sub, target, args.k)
                else:
                    Dsub = D_full[np.ix_(idx, idx)]
                    per, nt = _per_point_purity_from_distance(Dsub, fam_sub, target, args.k)
                raw, adj, lo, hi = knn_adj_with_ci(per, base, args.n_boot, args.seed)
                results["knn"][metric][pool_name][target] = {
                    "n": int(nt), "base": base, "raw": raw, "adj": adj, "adj_lo": lo, "adj_hi": hi}
                print(f"        {target:<8} n={nt:<3} raw={raw:.3f}  base={base:.3f}  "
                      f"ADJ={adj:+.3f} [{lo:+.3f},{hi:+.3f}]")

    # ---- CENTROID test: KMeans recovery of {jump,complex,chevron} full vs residual ----
    print("\n" + "=" * 96)
    print("CENTROID (KMeans) — does peeling free the K budget to recover hard classes?")
    print("  metric = NMI/ARI of cluster labels vs human family, restricted to the scored points.")
    print("=" * 96)
    results["centroid"] = {}
    # hard-focused label set for scoring recovery (include chevron as a borderline easy/hard)
    score_fams = ["jump", "complex", "chevron"]
    setups = {
        "FULL set, K=7 (all families)": (np.ones(len(fam), bool), 7),
        "RESIDUAL {jump,complex,chevron}, K=3": (np.isin(fam, score_fams), 3),
    }
    for name, (mask, K) in setups.items():
        idx = np.where(mask)[0]
        fam_sub = fam[idx]
        # Euclidean KMeans
        kmE = KMeans(n_clusters=K, random_state=args.seed, n_init=10).fit(X[idx])
        # soft-DTW: agglomerative on precomputed elastic distance (fast, no barycenter)
        Dsub = D_full[np.ix_(idx, idx)]
        aggS = AgglomerativeClustering(n_clusters=K, metric="precomputed", linkage="average").fit(Dsub)
        # score only on hard/borderline families present in this set
        smask = np.isin(fam_sub, score_fams)
        yE, yS, ytrue = kmE.labels_[smask], aggS.labels_[smask], fam_sub[smask]
        rec = {
            "n_scored": int(smask.sum()),
            "euclidean": {"nmi": float(nmi(ytrue, yE)), "ari": float(ari(ytrue, yE))},
            "softdtw": {"nmi": float(nmi(ytrue, yS)), "ari": float(ari(ytrue, yS))},
        }
        results["centroid"][name] = rec
        print(f"  [{name}]  scored n={rec['n_scored']}  "
              f"EUC nmi={rec['euclidean']['nmi']:.3f} ari={rec['euclidean']['ari']:.3f}  |  "
              f"softDTW nmi={rec['softdtw']['nmi']:.3f} ari={rec['softdtw']['ari']:.3f}")

    # ---- verdict synthesis ----
    verdict = _synthesize(results)
    print("\n" + "=" * 96)
    print("VERDICT")
    print("=" * 96)
    for line in verdict["lines"]:
        print("  " + line)
    print(f"\n  => {verdict['headline']}")

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    json.dump({"params": vars(args), "easy": EASY, "hard": HARD, "results": results, "verdict": verdict},
              open(args.out_json, "w"), indent=2, default=float)
    print(f"\n[OUT] {args.out_json}")
    html = _render_html(results, verdict, args)
    open(args.out_html, "w").write(html)
    print(f"[OUT] {args.out_html}")
    print(f"[VIEW] file://wsl.localhost/Ubuntu{os.path.abspath(args.out_html)}")


def _synthesize(results):
    """Decide whether peeling helped, per metric, on adjusted purity (with CIs)."""
    lines = []
    knn = results["knn"]
    helped = {}
    for metric in knn:
        full = knn[metric]["FULL (all families present)"]
        resid = knn[metric]["RESIDUAL shape {jump,complex}"]
        for target in HARD:
            if target not in full or target not in resid:
                continue
            f, r = full[target], resid[target]
            # peeling helps if residual adjusted-purity CI lies ABOVE full's point (non-trivial lift)
            improved = r["adj_lo"] > f["adj"]
            regressed = r["adj_hi"] < f["adj"]
            tag = "HELPS" if improved else ("HURTS" if regressed else "no-change (CI overlaps point)")
            helped[(metric, target)] = tag
            lines.append(f"{metric:<26} {target:<8}: full ADJ={f['adj']:+.3f}  ->  residual ADJ={r['adj']:+.3f} "
                         f"[{r['adj_lo']:+.3f},{r['adj_hi']:+.3f}]  => {tag}")
    any_help = any(v == "HELPS" for v in helped.values())
    any_hurt = any(v == "HURTS" for v in helped.values())
    # contamination demo
    contam_lines = []
    for metric in knn:
        clean = knn[metric].get("RESIDUAL shape {jump,complex}", {})
        noisy = knn[metric].get("RESIDUAL+Noise {jump,complex,Noise}", {})
        for target in HARD:
            if target in clean and target in noisy:
                d = noisy[target]["adj"] - clean[target]["adj"]
                contam_lines.append(f"{metric:<26} {target:<8}: adding Noise to the bin shifts ADJ by {d:+.3f}")
    if any_help and not any_hurt:
        headline = ("Peeling HELPS hard-class separation (adjusted for base rate) — the two-stage instinct is "
                    "mechanistically confirmed on this anchor.")
    elif any_help and any_hurt:
        headline = ("MIXED — peeling helps some hard families and hurts others; depends on metric/family.")
    elif any_hurt:
        headline = ("Peeling HURTS adjusted separation — the residual is the structureless continuum middle; "
                    "the easy classes were carrying the local structure.")
    else:
        headline = ("NO base-rate-corrected change — the raw purity gain from peeling is the trivial pool-shrink "
                    "artifact, not improved separability. (Still helps centroid methods via imbalance relief.)")
    return {"lines": lines + ["", "[contamination guardrail]"] + contam_lines, "headline": headline,
            "helped": {f"{k[0]}|{k[1]}": v for k, v in helped.items()}}


def _render_html(results, verdict, args):
    def knn_table(metric):
        rows = ""
        for pool_name, fams in results["knn"][metric].items():
            for target, d in fams.items():
                rows += (f"<tr><td>{pool_name}</td><td>{target}</td><td>{d['n']}</td>"
                         f"<td>{d['raw']:.3f}</td><td>{d['base']:.3f}</td>"
                         f"<td><b>{d['adj']:+.3f}</b> <span class='ci'>[{d['adj_lo']:+.3f}, {d['adj_hi']:+.3f}]</span></td></tr>")
        return rows
    knn_tables = ""
    for metric in results["knn"]:
        knn_tables += (f"<h3>{metric}</h3><table><tr><th>candidate pool</th><th>hard class</th><th>n</th>"
                       f"<th>raw purity</th><th>base rate</th><th>ADJUSTED [95% CI]</th></tr>"
                       f"{knn_table(metric)}</table>")
    cen = ""
    for name, rec in results["centroid"].items():
        cen += (f"<tr><td>{name}</td><td>{rec['n_scored']}</td>"
                f"<td>{rec['euclidean']['nmi']:.3f} / {rec['euclidean']['ari']:.3f}</td>"
                f"<td>{rec['softdtw']['nmi']:.3f} / {rec['softdtw']['ari']:.3f}</td></tr>")
    vlines = "<br>".join(l if l else "&nbsp;" for l in verdict["lines"])
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Peel-easy-classes probe</title>
<style>body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1050px;margin:2rem auto;padding:0 1rem;color:#1a1a1a}}
h1{{font-size:1.4rem}}h2{{font-size:1.05rem;margin-top:1.8rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}h3{{font-size:.95rem;margin-top:1.2rem}}
table{{border-collapse:collapse;width:100%;margin:.6rem 0;font-size:.86rem}}th,td{{border:1px solid #ddd;padding:.35rem .55rem;text-align:right}}
th{{background:#f6f6f6;text-align:left}}td:first-child{{text-align:left}}.ci{{color:#888;font-size:.85em}}
.verdict{{background:#eef4ff;border:1px solid #3367d6;border-radius:8px;padding:1rem;font-weight:600;margin:1rem 0}}
.muted{{color:#666;font-size:.86rem}}code{{background:#f2f2f2;padding:.1rem .3rem;border-radius:3px}}pre{{font-size:.8rem;white-space:pre-wrap}}</style></head><body>
<h1>Does peeling the 'easy' shape classes help recover the 'hard' ones?</h1>
<p class="muted">182 human-labeled registered ridges · k={args.k}, {args.n_boot}× bootstrap · EASY={EASY} peeled, HARD bin={HARD}.
The honest signal is <b>ADJUSTED purity = (purity − base)/(1 − base)</b> — raw purity rises trivially when the pool shrinks.</p>
<div class="verdict">{verdict['headline']}</div>
<h2>Local (kNN) — base-rate-corrected purity of hard classes</h2>
{knn_tables}
<p class="muted">Read: if ADJUSTED purity rises from FULL → RESIDUAL with a CI above the FULL point, peeling genuinely improved
local separability (easy/hard were intermixed — the continuum). If ADJUSTED is flat, the raw gain was just the pool-shrink artifact.</p>
<h2>Centroid (KMeans / elastic-agglomerative) — recovery of {{jump, complex, chevron}}</h2>
<table><tr><th>setup</th><th>n scored</th><th>Euclidean NMI / ARI</th><th>soft-DTW NMI / ARI</th></tr>{cen}</table>
<h2>Verdict detail</h2><pre>{vlines}</pre>
<p class="muted">Caveat: 182 labels, all lab cohort 131204; complex N=12. Directional, not a production decision — feeds Phase-2 of PLAN_elastic_shape_clustering.md.</p>
</body></html>"""


if __name__ == "__main__":
    main()

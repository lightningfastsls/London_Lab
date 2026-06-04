"""Human-anchored shape-representation eval harness — the STANDING GATE.

Replaces the circular `shape η²` / `chevron_valley` metric (which was computed
on the same ridge KMeans it graded — doubly circular). Instead, score 1-D
registered-ridge shape representations against the ~200 human shape labels in
`data/manual_shape_labels.csv`, using leave-one-out kNN retrieval purity with
1000× bootstrap confidence intervals.

Decision metric (per PLAN_elastic_shape_clustering.md §"The decision metric"):
  - PRIMARY  : per-family leave-one-out kNN retrieval purity (k=10) vs human labels
               for {chevron, jump, flat, complex} (+ FM/Noise for context).
  - SECONDARY: NMI of the incumbent K=20 alphabet (`lab_shape`) vs human labels.
  - CONTROLS reported alongside EVERY method:
        (a) random label assignment  (= base rate),
        (b) registration-Euclidean ridge (the incumbent / IDENTITY).
  - UNCERTAINTY: 1000× bootstrap 95% CIs on every purity. Decisions are made on
    NON-overlapping CIs, never point estimates.
  - `shape η²` is intentionally NOT computed here — it is the circular metric
    this harness exists to retire.

Representations compared:
  - registered_shape  (Euclidean on the registered ridge)  -> the INCUMBENT
  - soft-DTW          (per-pair elastic warp alignment)     -> the CANDIDATE
  - srvf              (sign(g)·sqrt|g|, a Fisher-Rao surrogate)
  - derivative        (np.diff)

The five functions below (`group_family`, `build_join`, `loo_knn_purity`,
`knn_purity_from_distance`, `bootstrap_purity_ci`) are the unit-tested SPEC
(tests/experiments/test_eval_shape_human_anchored.py, written BEFORE this
module). Do not change their signatures.

CLI: `.venv/bin/python scripts/experiments/eval_shape_human_anchored.py`
prints all parameters/Ns (per feedback_analysis_print_params) and writes a
JSON scorecard + an HTML report with a file://wsl.localhost/... URL
(per feedback_html_user_facing_default / feedback_wsl_file_viewing).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

# ---------------------------------------------------------------------------
# 1. Label taxonomy
# ---------------------------------------------------------------------------
_FAMILY = {
    "Chevron": "chevron",
    "Reverse Chevron": "chevron",
    "Step up": "jump",
    "Step down": "jump",
    "Two steps": "jump",
    "Multi-steps": "jump",
    "Flat": "flat",
    "Complex": "complex",
}


def group_family(label: str) -> str:
    """Map a Grimsley display label to a coarse shape family.

    chevron <- {Chevron, Reverse Chevron}; jump <- {Step up/down, Two steps,
    Multi-steps}; flat <- Flat; complex <- Complex. Any other label (Noise, FM,
    Short, ...) is returned UNCHANGED. Case-sensitive: keys are title-case
    display names, so "chevron" (lowercase) falls through to identity.
    """
    return _FAMILY.get(label, label)


# ---------------------------------------------------------------------------
# 2. Ridge <-> human-label join
# ---------------------------------------------------------------------------
def build_join(wav_stem, call_id, human_df, offset: int = -1):
    """Join ridge rows to human-labeled call ids.

    For ridge row i the composite id is ``f"{wav_stem[i]}__det{call_id[i]+offset}"``.
    The default ``offset=-1`` corrects a 0-indexed-detection / 1-indexed-call_id
    mismatch (verified: offset -1 lands 200/204 unique hits on the real data).

    Dedup rule: each composite id maps to the FIRST ridge row index it appears
    at. Returns ``(rows, joined_df)`` where ``joined_df`` is the subset of
    ``human_df`` whose ``call_id`` matched (preserving human_df order) with an
    added integer ``row`` column, and ``rows`` is that column as an ndarray.
    """
    wav_stem = np.asarray(wav_stem).astype(str)
    call_id = np.asarray(call_id)
    comp = np.array(
        [f"{wav_stem[i]}__det{int(call_id[i]) + offset}" for i in range(len(wav_stem))]
    )
    hset = set(human_df["call_id"])

    id2row: dict[str, int] = {}
    for i, c in enumerate(comp):
        if c in hset and c not in id2row:
            id2row[c] = i

    mask = human_df["call_id"].isin(id2row)
    joined = human_df[mask].copy()
    # map -> first-occurrence ridge row; force integer dtype even when empty.
    joined["row"] = joined["call_id"].map(id2row).astype(np.int64)
    rows = joined["row"].to_numpy()
    return rows, joined


# ---------------------------------------------------------------------------
# 3. Leave-one-out kNN retrieval purity (shared per-point core)
# ---------------------------------------------------------------------------
def _per_point_purity(X, labels, target, k):
    """Per-target-point purity = fraction of each target point's k LOO
    neighbours that share its family. Returns (per_point_array, n_target).

    Self is excluded by requesting k+1 neighbours and dropping column 0 (the
    point itself, distance 0) — the convention the original probe used.
    """
    X = np.asarray(X, dtype=np.float64)
    labels = np.asarray(labels)
    n = len(X)
    tmask = labels == target
    nt = int(tmask.sum())
    if nt == 0:
        return np.array([], dtype=np.float64), 0
    k = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=k + 1).fit(X)
    _, nbr = nn.kneighbors(X)
    nbr = nbr[:, 1:]  # drop self (column 0)
    per = (labels[nbr[tmask]] == target).mean(axis=1)
    return per.astype(np.float64), nt


def loo_knn_purity(X, labels, target, k: int = 10):
    """Leave-one-out kNN retrieval purity for ``target`` family.

    Returns ``(purity, n_target)``. Purity is the mean over target points of the
    fraction of each point's k Euclidean neighbours (self excluded) that are also
    ``target``. If no point has ``label==target``, returns ``(nan, 0)``.
    """
    per, nt = _per_point_purity(X, labels, target, k)
    if nt == 0:
        return float("nan"), 0
    return float(per.mean()), nt


def _per_point_purity_from_distance(D, labels, target, k):
    """Per-point purity from a precomputed pairwise distance matrix.

    The diagonal is set to +inf (self-exclusion) before sorting. Used by the
    soft-DTW / elastic path, which only has a distance matrix, not an embedding.
    """
    D = np.array(D, dtype=np.float64, copy=True)
    labels = np.asarray(labels)
    n = D.shape[0]
    tmask = labels == target
    nt = int(tmask.sum())
    if nt == 0:
        return np.array([], dtype=np.float64), 0
    np.fill_diagonal(D, np.inf)
    k = min(k, n - 1)
    nbr = np.argsort(D, axis=1)[:, :k]
    per = (labels[nbr[tmask]] == target).mean(axis=1)
    return per.astype(np.float64), nt


def knn_purity_from_distance(D, labels, target, k: int = 10) -> float:
    """kNN retrieval purity from a precomputed (n,n) distance matrix.

    Same definition as :func:`loo_knn_purity` but neighbours come from sorting
    ``D`` rows with the diagonal treated as +inf. Returns a float (nan if the
    target family is absent).
    """
    per, nt = _per_point_purity_from_distance(D, labels, target, k)
    if nt == 0:
        return float("nan")
    return float(per.mean())


# ---------------------------------------------------------------------------
# 4. Bootstrap CIs
# ---------------------------------------------------------------------------
def bootstrap_purity_ci(X, labels, target, k: int = 10, n_boot: int = 1000, seed: int = 42):
    """Bootstrap 95% CI for LOO kNN purity of ``target``.

    Point estimate == ``loo_knn_purity(X, labels, target, k)[0]`` exactly. The
    bootstrap resamples (with replacement, seeded) the SET of target-class points
    and re-averages their per-point purities; CI = 2.5/97.5 percentiles of that
    distribution. Returns ``(point, ci_lo, ci_hi)`` ((nan,nan,nan) if absent).
    """
    per, nt = _per_point_purity(X, labels, target, k)
    if nt == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(per.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, nt, nt)
        boot[b] = per[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


def bootstrap_purity_ci_from_distance(D, labels, target, k: int = 10, n_boot: int = 1000, seed: int = 42):
    """Distance-matrix analogue of :func:`bootstrap_purity_ci` (soft-DTW path)."""
    per, nt = _per_point_purity_from_distance(D, labels, target, k)
    if nt == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(per.mean())
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, nt, nt)
        boot[b] = per[idx].mean()
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return point, float(lo), float(hi)


# ===========================================================================
# CLI: run the full human-anchored comparison and emit JSON + HTML
# ===========================================================================
FAMILIES = ["chevron", "jump", "flat", "complex"]


def _srvf(x: np.ndarray) -> np.ndarray:
    """Square-Root Velocity Function transform (a Fisher-Rao surrogate)."""
    g = np.gradient(x, axis=1)
    return np.sign(g) * np.sqrt(np.abs(g))


def _ci_str(point, lo, hi):
    if point != point:  # nan
        return "   nan"
    return f"{point:.3f} [{lo:.3f},{hi:.3f}]"


def main():
    import argparse
    import json
    import os
    from datetime import datetime

    from sklearn.metrics import normalized_mutual_info_score as nmi

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--meta", default="/home/shachar/.claude/jobs/9a954f32/tmp/shape_data/true_registered_ridges_meta.npz")
    ap.add_argument("--lab", default="/home/shachar/.claude/jobs/9a954f32/tmp/shape_data/true_registered_ridges.npz")
    ap.add_argument("--human", default="data/manual_shape_labels.csv")
    ap.add_argument("--out-json", default="results/shape_retrospective/human_anchored_eval_v2.json")
    ap.add_argument("--out-html", default="results/shape_retrospective/human_anchored_eval_v2.html")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--softdtw-gamma", type=float, default=1.0)
    ap.add_argument("--no-softdtw", action="store_true", help="skip the elastic metric (debug)")
    ap.add_argument("--fpca-lambda", type=float, default=0.0,
                    help="elasticity penalty lam for the SRVF elastic-FPCA amplitude distance")
    ap.add_argument("--no-elasticfpca", action="store_true",
                    help="skip the SRVF elastic-FPCA (warp-aligned) amplitude-distance method")
    args = ap.parse_args()

    print("=" * 96)
    print("HUMAN-ANCHORED SHAPE EVAL  (replaces circular shape-eta-squared / chevron_valley)")
    print("=" * 96)
    print("PARAMETERS")
    print(f"  meta   = {args.meta}")
    print(f"  lab    = {args.lab}")
    print(f"  human  = {args.human}")
    print(f"  k      = {args.k}   n_boot = {args.n_boot}   seed = {args.seed}   softdtw_gamma = {args.softdtw_gamma}")
    print(f"  families = {FAMILIES}   metric = LOO kNN purity (non-circular)")

    m = np.load(args.meta, allow_pickle=True)
    L = np.load(args.lab, allow_pickle=True)
    Sh = m["shapes"].astype(np.float64)
    ws = m["wav_stem"].astype(str)
    cid = m["call_id"]
    coh = m["cohort"].astype(str)
    lab_shape = L["lab_shape"]
    chev_val = L["chevron_valley"].astype(str)
    assert L["shapes"].shape == Sh.shape, "meta/lab row mismatch"
    print(f"  ridges = {Sh.shape}   cohorts = {dict(zip(*np.unique(coh, return_counts=True)))}")

    h = pd.read_csv(args.human)
    hset = set(h["call_id"])
    # join-offset diagnostic
    for off in (0, -1):
        comp = np.array([f"{ws[i]}__det{int(cid[i]) + off}" for i in range(len(ws))])
        print(f"  [JOIN] offset {off:>2}: unique human ids hit = {len(set(comp) & hset)}/{len(h)}")

    rows, joined = build_join(ws, cid, h, offset=-1)
    print(f"  [JOIN] matched {len(joined)}/{len(h)} human labels (offset -1)")

    # drop 'unclear'; keep everything else (Noise/FM kept as own family for context)
    y_raw = joined["shape_label"].to_numpy()
    keep = ~np.isin(y_raw, ["unclear"])
    rows = rows[keep]
    y = y_raw[keep]
    yf = np.array([group_family(v) for v in y])
    row_coh = coh[rows]
    print(f"  [DATA] {len(y)} labels after dropping 'unclear'")
    print(f"  [FAMILY] counts = {dict(pd.Series(yf).value_counts())}")
    print(f"  [STRATUM] cohorts in labeled set = {dict(zip(*np.unique(row_coh, return_counts=True)))}")
    print("  [STRATUM] NOTE: all human labels are lab cohort 131204; wild (5970/3452/9252) UNLABELED.")
    print("            (per feedback_cross_animal_population_strata)")

    # ---- representations restricted to the labeled rows ----
    X_reg = Sh[rows]                       # registration-Euclidean = INCUMBENT / IDENTITY
    X_srvf = _srvf(Sh)[rows]
    X_deriv = np.diff(Sh, axis=1)[rows]
    base = {f: float((yf == f).mean()) for f in FAMILIES}

    results = {}

    def eval_embedding(name, X):
        row = {}
        for f in FAMILIES:
            row[f] = bootstrap_purity_ci(X, yf, f, k=args.k, n_boot=args.n_boot, seed=args.seed)
        results[name] = row

    eval_embedding("registration_euclidean(IDENTITY)", X_reg)
    eval_embedding("srvf", X_srvf)
    eval_embedding("derivative", X_deriv)

    # ---- soft-DTW (the elastic candidate) via pairwise distance matrix ----
    if not args.no_softdtw:
        from tslearn.metrics import cdist_soft_dtw_normalized
        print("\n  [softDTW] computing normalized soft-DTW pairwise matrix on labeled ridges...")
        D_sdtw = cdist_soft_dtw_normalized(X_reg[:, :, None], gamma=args.softdtw_gamma)
        sd_row = {}
        for f in FAMILIES:
            sd_row[f] = bootstrap_purity_ci_from_distance(D_sdtw, yf, f, k=args.k, n_boot=args.n_boot, seed=args.seed)
        results["soft_dtw(ELASTIC)"] = sd_row

    # ---- elastic FPCA (SRVF + warp alignment) via pairwise amplitude-distance matrix ----
    # The principled generalization of soft-DTW: Fisher-Rao elastic metric with the
    # `min over gamma` warp step (the active ingredient our pointwise SRVF lacked).
    if not args.no_elasticfpca:
        import os as _os, sys as _sys
        if _os.path.dirname(__file__) not in _sys.path:
            _sys.path.insert(0, _os.path.dirname(__file__))
        from build_elastic_fpca import elastic_amplitude_distance_matrix as _elastic_amplitude_distance_matrix
        print(f"\n  [elasticFPCA] computing SRVF elastic (amplitude) distance matrix on "
              f"labeled ridges (lam={args.fpca_lambda})...")
        D_efpca = _elastic_amplitude_distance_matrix(X_reg, lam=args.fpca_lambda)   # (n_labeled, n_labeled)
        ef_row = {f: bootstrap_purity_ci_from_distance(D_efpca, yf, f,
                                                       k=args.k, n_boot=args.n_boot, seed=args.seed)
                  for f in FAMILIES}
        results["elastic_fpca(SRVF-WARP)"] = ef_row

    # ---- random-label control (= base rate) ----
    rng = np.random.default_rng(args.seed)
    yf_rand = yf.copy()
    rng.shuffle(yf_rand)
    rand_row = {f: bootstrap_purity_ci(X_reg, yf_rand, f, k=args.k, n_boot=args.n_boot, seed=args.seed) for f in FAMILIES}
    results["random_control(BASE RATE)"] = rand_row

    # ---- secondary: NMI of incumbent K=20 alphabet vs human family ----
    km = lab_shape[rows]
    nmi_inc = float(nmi(yf, km))

    # ---- chevron_valley heuristic vs human (audit the retired proxy) ----
    cv = chev_val[rows]
    human_chev = np.isin(y, ["Chevron", "Reverse Chevron"])
    heur_chev = cv == "chevron"
    tp = int((heur_chev & human_chev).sum())
    fp = int((heur_chev & ~human_chev).sum())
    fn = int((~heur_chev & human_chev).sum())
    cv_prec = tp / (tp + fp) if (tp + fp) else float("nan")
    cv_rec = tp / (tp + fn) if (tp + fn) else float("nan")

    # ---- print scorecard ----
    print("\n" + "=" * 96)
    print("HUMAN-ANCHORED kNN PURITY  (point [95% bootstrap CI])  — decide on NON-overlapping CIs")
    print("=" * 96)
    print(f"{'representation':<34}" + "".join(f"{f:>22}" for f in FAMILIES))
    print(f"{'BASE RATE':<34}" + "".join(f"{base[f]:>22.3f}" for f in FAMILIES))
    for name, row in results.items():
        cells = "".join(f"{_ci_str(*row[f]):>22}" for f in FAMILIES)
        print(f"{name:<34}{cells}")
    print(f"\n[SECONDARY] NMI(incumbent K=20 alphabet 'lab_shape' vs human family) = {nmi_inc:.3f}")
    print("\n[PROXY AUDIT] chevron_valley heuristic vs human chevron-family:")
    print(f"   precision={cv_prec:.3f}  recall={cv_rec:.3f}  (tp={tp} fp={fp} fn={fn})")
    print("   -> the metric the whole bake-off optimized; low values = it was a poor proxy.")

    # ---- GATE 1 read ----
    gate = gate1_read(results, base)
    print("\n" + "=" * 96)
    print("GATE 1  (plan §Phase 1):  proceed if soft-DTW beats IDENTITY on jump OR complex with")
    print("        NON-overlapping CIs, AND chevron/flat do not regress beyond CI overlap.")
    print("=" * 96)
    for line in gate["lines"]:
        print("   " + line)
    print(f"\n   VERDICT: {gate['verdict']}")

    # ---- persist JSON ----
    payload = {
        "params": {"k": args.k, "n_boot": args.n_boot, "seed": args.seed, "softdtw_gamma": args.softdtw_gamma},
        "n_labels": int(len(y)),
        "family_counts": {k: int(v) for k, v in pd.Series(yf).value_counts().items()},
        "cohorts_in_labeled": {str(k): int(v) for k, v in zip(*np.unique(row_coh, return_counts=True))},
        "base_rate": base,
        "purity": {name: {f: list(row[f]) for f in FAMILIES} for name, row in results.items()},
        "nmi_incumbent_k20_vs_human": nmi_inc,
        "chevron_valley_proxy": {"precision": cv_prec, "recall": cv_rec, "tp": tp, "fp": fp, "fn": fn},
        "gate1": gate,
        "caveat": "All 204 human labels are lab cohort 131204; wild cohorts UNLABELED. "
                  "Per-class N tiny (esp. chevron ~25) — read CIs, not point estimates.",
    }
    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as fp:
        json.dump(payload, fp, indent=2)
    print(f"\n[OUT] {args.out_json}")

    # ---- HTML report ----
    html = _render_html(payload, results, base, args, generated=datetime.now().isoformat(timespec="seconds"))
    with open(args.out_html, "w") as fp:
        fp.write(html)
    abspath = os.path.abspath(args.out_html)
    print(f"[OUT] {args.out_html}")
    print(f"[VIEW] file://wsl.localhost/Ubuntu{abspath}")


def gate1_read(results, base):
    """Encode the Phase-1 GATE 1 logic on non-overlapping bootstrap CIs.

    soft-DTW beats IDENTITY on a family iff soft-DTW ci_lo > identity ci_hi.
    soft-DTW regresses on a family iff soft-DTW ci_hi < identity ci_lo.
    """
    ident = results.get("registration_euclidean(IDENTITY)")
    elastic = results.get("soft_dtw(ELASTIC)")
    lines = []
    if elastic is None or ident is None:
        return {"verdict": "INCONCLUSIVE — soft-DTW not computed", "lines": ["soft-DTW absent"],
                "beats": [], "regresses": []}
    beats, regresses = [], []
    for f in FAMILIES:
        ep, elo, ehi = elastic[f]
        ip, ilo, ihi = ident[f]
        if ep == ep and ip == ip:
            if elo > ihi:
                beats.append(f)
                tag = "BEATS (non-overlapping)"
            elif ehi < ilo:
                regresses.append(f)
                tag = "REGRESSES (non-overlapping)"
            else:
                tag = "ties (CIs overlap)"
            lines.append(f"{f:<9} elastic {ep:.3f}[{elo:.3f},{ehi:.3f}]  vs  identity {ip:.3f}[{ilo:.3f},{ihi:.3f}]  -> {tag}")
    win = any(f in beats for f in ("jump", "complex"))
    no_regress = not any(f in regresses for f in ("chevron", "flat"))
    if win and no_regress:
        verdict = "PROCEED to Phase 2 — elastic beats identity on jump/complex; chevron/flat hold."
    elif win and not no_regress:
        verdict = "MIXED — elastic wins jump/complex BUT regresses chevron/flat. Inspect before proceeding."
    else:
        verdict = "KILL — elastic within CI of identity on jump & complex at this N; keep registration."
    return {"verdict": verdict, "lines": lines, "beats": beats, "regresses": regresses}


def _render_html(payload, results, base, args, generated):
    def row_html(name, row, highlight=False):
        cells = ""
        for f in FAMILIES:
            p, lo, hi = row[f]
            txt = "nan" if p != p else f"{p:.3f}<span class='ci'> [{lo:.3f}, {hi:.3f}]</span>"
            cells += f"<td>{txt}</td>"
        cls = " class='ident'" if "IDENTITY" in name else (" class='elastic'" if "ELASTIC" in name else "")
        return f"<tr{cls}><th>{name}</th>{cells}</tr>"

    gate = payload["gate1"]
    rows_html = "".join(row_html(n, r) for n, r in results.items())
    base_cells = "".join(f"<td>{base[f]:.3f}</td>" for f in FAMILIES)
    gate_lines = "<br>".join(gate["lines"])
    cohorts = ", ".join(f"{k}={v}" for k, v in payload["cohorts_in_labeled"].items())
    fam_counts = ", ".join(f"{k}={v}" for k, v in payload["family_counts"].items())
    cv = payload["chevron_valley_proxy"]
    verdict_cls = "proceed" if gate["verdict"].startswith("PROCEED") else ("kill" if gate["verdict"].startswith("KILL") else "mixed")
    return f"""<!doctype html><html><head><meta charset="utf-8">
<title>Human-anchored shape eval (Phase 1, GATE 1)</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;margin:2rem auto;color:#1a1a1a;padding:0 1rem}}
h1{{font-size:1.5rem}} h2{{font-size:1.1rem;margin-top:2rem;border-bottom:1px solid #ddd;padding-bottom:.3rem}}
table{{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.9rem}}
th,td{{border:1px solid #ddd;padding:.4rem .6rem;text-align:right}} th{{text-align:left;background:#f6f6f6}}
.ci{{color:#888;font-size:.8em}} tr.ident th{{background:#eef4ff}} tr.elastic th{{background:#fff4e6}}
.verdict{{padding:1rem;border-radius:8px;font-weight:600;margin:1rem 0}}
.proceed{{background:#e6f7ed;border:1px solid #34a853}} .kill{{background:#fdeeee;border:1px solid #d33}}
.mixed{{background:#fff8e1;border:1px solid #f4b400}}
.muted{{color:#666;font-size:.88rem}} code{{background:#f2f2f2;padding:.1rem .3rem;border-radius:3px}}
</style></head><body>
<h1>Human-anchored shape-representation eval — Phase 1 / GATE 1</h1>
<p class="muted">Generated {generated} · <code>PLAN_elastic_shape_clustering.md</code> ·
de-circularized: scored against {payload['n_labels']} human labels, not the <code>chevron_valley</code> heuristic.</p>

<div class="verdict {verdict_cls}">{gate['verdict']}</div>

<h2>Decision metric — LOO kNN purity (k={args.k}), point [95% bootstrap CI, {args.n_boot}×]</h2>
<table><tr><th>representation</th>{"".join(f"<th>{f}</th>" for f in FAMILIES)}</tr>
<tr><th>BASE RATE</th>{base_cells}</tr>
{rows_html}</table>
<p class="muted">Blue = incumbent (registration-Euclidean, IDENTITY control). Orange = elastic candidate (soft-DTW).
Decisions are made on <b>non-overlapping CIs</b>, never point estimates.</p>

<h2>GATE 1 read (per family)</h2>
<p>{gate_lines}</p>

<h2>Secondary &amp; controls</h2>
<ul>
<li><b>NMI</b>(incumbent K=20 alphabet <code>lab_shape</code> vs human family) = {payload['nmi_incumbent_k20_vs_human']:.3f}</li>
<li><b>chevron_valley proxy audit</b> (the retired circular metric): precision={cv['precision']:.3f},
recall={cv['recall']:.3f} (tp={cv['tp']} fp={cv['fp']} fn={cv['fn']}) — low values confirm it was a poor proxy.</li>
</ul>

<h2>Provenance &amp; caveats</h2>
<ul>
<li>Labels: <code>{args.human}</code> · families: {fam_counts}</li>
<li>Cohort stratum in labeled set: {cohorts} — <b>all lab cohort 131204; wild cohorts UNLABELED</b>
(per <code>feedback_cross_animal_population_strata</code>). A production decision (Phase 3) requires the expanded, wild-covering gold set.</li>
<li>{payload['caveat']}</li>
<li><code>shape η²</code> deliberately NOT computed — it is the circular metric this harness retires.</li>
</ul>
</body></html>"""


if __name__ == "__main__":
    main()

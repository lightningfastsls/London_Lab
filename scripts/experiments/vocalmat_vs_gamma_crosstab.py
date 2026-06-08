#!/usr/bin/env python3
"""vocalmat_vs_gamma_crosstab.py — α₃-C Phase A8 cross-tab (oracle taxonomy vs γ).

PURPOSE
-------
Cross-validate the v1 VocalMat *oracle* taxonomy against the USER's γ
hand-labels on the ~200 γ-labeled patches. The single question:

    Does the 12-class Grimsley/VocalMat taxonomy agree with what the user
    means by SHAPE?

    - If γ (8 shape categories) and the oracle's ``top1_class`` (12 Grimsley
      classes) agree strongly (Cramér's V > 0.5) → the taxonomy IS geometric
      in the user's sense, so α₃ is a credible eval anchor.
    - If they disagree (V < 0.3) → the taxonomy keys on something OTHER than
      shape, γ is the only honest shape anchor, and any α₃ VAE eval that scores
      against the oracle is *contaminated*.
    - 0.3 ≤ V ≤ 0.5 → ambiguous.

INPUTS (exact schemas — confirmed by reading the producing scripts)
-------------------------------------------------------------------
γ labels  (``scripts/labeling/hand_label_200.py`` output)
    columns: call_id, cohort, shape_label, labeled_at_index
    shape_label vocabulary (8): chevron, jump, flat, complex, up_fm, down_fm,
                                multi_component, unclear
    MAY NOT EXIST YET (user labels later) → graceful message + exit 0.

oracle labels (``scripts/experiments/label_patches_v1.py`` output)
    columns: call_id, top1_class, top1_idx, top1_prob, high_confidence, softmax_12
    top1_class vocabulary (12 GRIMSLEY classes) imported from
        usv_spectrogram.classifier.dataset.GRIMSLEY_12_CLASSES
    NOTE: ``call_id`` is only present if label_patches_v1.py was run with an
    ``--id-column``. If absent we cannot join → clear message + exit 0.
    --oracle-labels accepts a COMMA-SEPARATED list of CSVs (γ patches may span
    cohorts whose oracle labels live in different files); they are concatenated
    then deduped on call_id.

ANALYSIS
--------
- Contingency table: ROWS = γ shape_label (8), COLS = oracle top1_class (12)
  via pandas.crosstab.
- Cramér's V (uncorrected) from the chi-square statistic:
      V = sqrt( chi2 / (n * min(r-1, c-1)) )
  computed on the matched, joined rows.
- NMI: sklearn.metrics.normalized_mutual_info_score on the two label vectors.
- --exclude-unclear: re-runs the primary stat with γ=="unclear" rows dropped;
  BOTH (with and without) are always reported.

OUTPUT
------
- stdout: params/row-counts, contingency table, V, NMI, verdict.
- HTML report → $CLAUDE_JOB_DIR/alpha3_oracle_vs_gamma.html (--output override):
  styled contingency table + embedded heatmap PNG + V + NMI + n + verdict.
- CSV of the contingency table → results/alpha3/oracle_vs_gamma_crosstab.csv.
- Prints the WSL file:// URL for the HTML (repo rule feedback_wsl_file_viewing).

CONSTRAINTS
-----------
- Reads only; writes only to --output (HTML) and results/alpha3/ (CSV).
- Does not touch corpus.py / models/ / the production pipeline.

VERIFY (py_compile)
-------------------
    cd /home/shachar/projects/mickey_london_lab && \\
      PYTHONPATH=src .venv/bin/python -m py_compile \\
        scripts/experiments/vocalmat_vs_gamma_crosstab.py

USER COMMAND (after γ labeling + A4 oracle labeling are done)
-------------------------------------------------------------
    cd /home/shachar/projects/mickey_london_lab && \\
      PYTHONPATH=src .venv/bin/python scripts/experiments/vocalmat_vs_gamma_crosstab.py \\
        --gamma data/manual_shape_labels.csv \\
        --oracle-labels data/labels_vocalmat_v1_on_131204.csv
"""

from __future__ import annotations

import argparse
import base64
import html as html_mod
import io
import os
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# ----------------------------------------------------------------------------
# Repo root (script lives at scripts/experiments/vocalmat_vs_gamma_crosstab.py)
# ----------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

# Import the canonical 12-class column order. Add src/ to path so this runs
# even without PYTHONPATH=src (though the documented invocation sets it).
sys.path.insert(0, str(REPO_ROOT / "src"))
try:
    from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES
except Exception as exc:  # pragma: no cover - defensive
    print(f"FATAL: could not import GRIMSLEY_12_CLASSES: {exc}", file=sys.stderr)
    sys.exit(2)

# γ vocabulary. 2026-05-30: γ switched from the old 8-family scheme to the FULL
# 12-class VocalMat taxonomy (user wants a direct human-vs-machine comparison),
# so the γ label strings are byte-identical to the oracle's top1_class. Derive
# the γ row axis from the oracle's class order → the contingency table is a
# SQUARE 12×12 confusion matrix (diagonal = agreement). 'unclear' is the human
# escape hatch (∉ oracle space; dropped before κ / per-class precision-recall).
GAMMA_SHAPE_VOCAB: Tuple[str, ...] = tuple(GRIMSLEY_12_CLASSES) + ("unclear",)

DEFAULT_JOB_DIR = os.environ.get("CLAUDE_JOB_DIR", "/home/shachar/.claude/jobs/6ddc7ce2")
DEFAULT_OUTPUT = str(Path(DEFAULT_JOB_DIR) / "alpha3_oracle_vs_gamma.html")
DEFAULT_CSV_OUT = str(REPO_ROOT / "results" / "alpha3" / "oracle_vs_gamma_crosstab.csv")
def _wsl_url(path: Path) -> str:
    """file:// URL for WSL (repo rule feedback_wsl_file_viewing). Derived from
    the ACTUAL output path so it never drifts from the hardcoded job dir."""
    return "file://wsl.localhost/Ubuntu" + str(Path(path).resolve())


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------
def _resolve(path_like: str) -> Path:
    p = Path(path_like)
    return p if p.is_absolute() else (REPO_ROOT / p)


def graceful_exit(msg: str) -> None:
    """Print a clear message and exit 0 (missing-input is not an error)."""
    print("\n" + "=" * 70)
    print("  α₃-C A8 cross-tab — NOT RUN")
    print("=" * 70)
    print(f"  {msg}")
    print("  This is expected before the user has hand-labeled (γ) and/or")
    print("  before the A4 oracle-labeling step has produced its CSV.")
    print("  Exiting 0 (no error).")
    print("=" * 70)
    sys.exit(0)


def cramers_v(contingency: pd.DataFrame) -> Tuple[float, float, int, int, int]:
    """Uncorrected Cramér's V from the chi-square statistic.

    V = sqrt( chi2 / (n * min(r-1, c-1)) )

    Returns (V, chi2, n, r, c) where r/c are the number of non-empty rows/cols
    actually present in the contingency table (so a single observed shape or
    class does not silently inflate the denominator's min term).
    """
    from scipy.stats import chi2_contingency

    # Drop all-zero rows/cols so r,c reflect realised categories.
    obs = contingency.loc[(contingency.sum(axis=1) > 0), (contingency.sum(axis=0) > 0)]
    arr = obs.to_numpy(dtype=float)
    n = int(arr.sum())
    r, c = arr.shape
    if n == 0 or r < 2 or c < 2:
        # V undefined when fewer than 2 realised categories on either axis.
        return (float("nan"), float("nan"), n, r, c)
    chi2, _, _, _ = chi2_contingency(arr, correction=False)
    denom = n * (min(r - 1, c - 1))
    v = float(np.sqrt(chi2 / denom)) if denom > 0 else float("nan")
    return (v, float(chi2), n, r, c)


def nmi_score(labels_a: pd.Series, labels_b: pd.Series) -> float:
    from sklearn.metrics import normalized_mutual_info_score

    return float(normalized_mutual_info_score(labels_a.astype(str), labels_b.astype(str)))


def agreement_stats(truth: pd.Series, pred: pd.Series):
    """Human-vs-machine agreement on the SQUARE 12-class space.

    After the 2026-05-30 γ switch both label sets share GRIMSLEY_12_CLASSES, so
    we can treat γ (human) as TRUTH and the v1 oracle as PREDICTION and report:
      - Cohen's κ  : agreement corrected for chance (the headline metric the
                     handoff asks for — V/NMI measure association, not whether
                     the *same* label is chosen).
      - overall exact-agreement accuracy (diagonal fraction).
      - per-class precision/recall/F1 (sklearn classification_report):
          recall[X]    = of γ-said-X, fraction the oracle also said X
          precision[X] = of oracle-said-X, fraction γ also said X
    Caller MUST drop γ=='unclear' first (unclear ∉ oracle space)."""
    from sklearn.metrics import (
        cohen_kappa_score,
        accuracy_score,
        classification_report,
    )

    labels = list(GRIMSLEY_12_CLASSES)
    t = truth.astype(str)
    p = pred.astype(str)
    kappa = float(cohen_kappa_score(t, p, labels=labels))
    acc = float(accuracy_score(t, p))
    report = classification_report(
        t, p, labels=labels, output_dict=True, zero_division=0
    )
    return kappa, acc, report


def verdict_for(v: float) -> Tuple[str, str]:
    """Return (verdict_text, css_class) for a Cramér's V value."""
    if np.isnan(v):
        return (
            "UNDEFINED — too few realised categories to compute V "
            "(need ≥2 distinct γ shapes AND ≥2 distinct oracle classes).",
            "ambiguous",
        )
    if v > 0.5:
        return (
            "TAXONOMY IS GEOMETRIC in the user's sense — α₃ is a credible "
            "shape anchor (V > 0.5).",
            "good",
        )
    if v < 0.3:
        return (
            "TAXONOMY KEYS ON SOMETHING OTHER THAN SHAPE — γ is the only "
            "honest shape anchor; the α₃ VAE's oracle-based eval is "
            "CONTAMINATED (V < 0.3).",
            "bad",
        )
    return (
        "AMBIGUOUS — partial shape signal in the taxonomy (0.3 ≤ V ≤ 0.5). "
        "Treat oracle-based eval with caution; prefer γ.",
        "ambiguous",
    )


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def load_gamma(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"call_id", "shape_label"}
    missing = required - set(df.columns)
    if missing:
        graceful_exit(
            f"γ file {path} is missing required column(s) {sorted(missing)}; "
            f"found {list(df.columns)}."
        )
    df = df[df["call_id"].notna() & df["shape_label"].notna()].copy()
    df["call_id"] = df["call_id"].astype(str)
    df["shape_label"] = df["shape_label"].astype(str).str.strip()
    # Surface any out-of-vocabulary shape labels (don't drop — just warn).
    oov = sorted(set(df["shape_label"]) - set(GAMMA_SHAPE_VOCAB))
    if oov:
        print(f"  WARNING: γ shape_label values not in the known 8-vocab: {oov}")
    return df


def load_oracle(paths: List[Path]) -> pd.DataFrame:
    frames: List[pd.DataFrame] = []
    for p in paths:
        if not p.exists():
            graceful_exit(f"oracle-labels file does not exist: {p}")
        d = pd.read_csv(p)
        if "call_id" not in d.columns:
            graceful_exit(
                f"oracle file {p} has no 'call_id' column (columns={list(d.columns)}). "
                "label_patches_v1.py only writes call_id when run with --id-column. "
                "Re-run the A4 oracle-labeling step with --id-column set to the "
                "manifest's call_id column so the join key exists."
            )
        if "top1_class" not in d.columns:
            graceful_exit(
                f"oracle file {p} has no 'top1_class' column (columns={list(d.columns)})."
            )
        d["__src__"] = str(p)
        frames.append(d)
    cat = pd.concat(frames, ignore_index=True)
    n_before = len(cat)
    cat["call_id"] = cat["call_id"].astype(str)
    cat = cat.drop_duplicates(subset=["call_id"], keep="first").copy()
    n_after = len(cat)
    if n_after < n_before:
        print(
            f"  deduped oracle rows on call_id: {n_before} → {n_after} "
            f"({n_before - n_after} duplicate call_id(s) dropped, kept first)."
        )
    cat["top1_class"] = cat["top1_class"].astype(str).str.strip()
    return cat


# ----------------------------------------------------------------------------
# Heatmap (base64 PNG)
# ----------------------------------------------------------------------------
def heatmap_b64(ct: pd.DataFrame, title: str) -> Optional[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover
        print(f"  (heatmap skipped — matplotlib unavailable: {exc})")
        return None

    arr = ct.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, 0.7 * ct.shape[1] + 3),
                                    max(4, 0.5 * ct.shape[0] + 2)))
    im = ax.imshow(arr, aspect="auto", cmap="viridis")
    ax.set_xticks(range(ct.shape[1]))
    ax.set_xticklabels(ct.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(ct.shape[0]))
    ax.set_yticklabels(ct.index, fontsize=8)
    ax.set_xlabel("oracle top1_class (12 Grimsley)")
    ax.set_ylabel("γ shape_label")
    ax.set_title(title, fontsize=10)
    # annotate counts
    vmax = arr.max() if arr.size else 0
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            val = int(arr[i, j])
            if val:
                ax.text(j, i, str(val), ha="center", va="center",
                        color="white" if val < 0.6 * vmax else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("ascii")


# ----------------------------------------------------------------------------
# HTML report
# ----------------------------------------------------------------------------
def write_html(
    out_path: Path,
    ct_full: pd.DataFrame,
    v_full: float,
    nmi_full: float,
    n_full: int,
    verdict_full: Tuple[str, str],
    ct_excl: Optional[pd.DataFrame],
    v_excl: Optional[float],
    nmi_excl: Optional[float],
    n_excl: Optional[int],
    verdict_excl: Optional[Tuple[str, str]],
    n_gamma: int,
    n_oracle: int,
    n_matched: int,
    gamma_path: Path,
    oracle_paths: List[Path],
    heatmap_png: Optional[str],
    kappa_excl: Optional[float] = None,
    acc_excl: Optional[float] = None,
    report_excl: Optional[dict] = None,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    def style_table(ct: pd.DataFrame) -> str:
        # Add row totals / col totals for readability.
        disp = ct.copy()
        disp["TOTAL"] = disp.sum(axis=1)
        totals = disp.sum(axis=0)
        totals.name = "TOTAL"
        disp = pd.concat([disp, totals.to_frame().T])
        rows_html = []
        header = "<tr><th>γ \\ oracle</th>" + "".join(
            f"<th>{html_mod.escape(str(c))}</th>" for c in disp.columns
        ) + "</tr>"
        for idx, row in disp.iterrows():
            cells = []
            for col in disp.columns:
                val = int(row[col])
                cls = ' class="total"' if (idx == "TOTAL" or col == "TOTAL") else ""
                shade = ""
                if cls == "" and val > 0:
                    shade = ' style="background:rgba(46,134,193,0.18);font-weight:600;"'
                cells.append(f"<td{cls or shade}>{val}</td>")
            rows_html.append(
                f"<tr><th>{html_mod.escape(str(idx))}</th>" + "".join(cells) + "</tr>"
            )
        return f"<table class='ct'>{header}{''.join(rows_html)}</table>"

    vc = verdict_full[1]
    verdict_block = (
        f"<div class='verdict {vc}'><span class='vlabel'>VERDICT "
        f"(all γ rows)</span><br>Cramér's V = "
        f"<b>{v_full:.4f}</b> &nbsp;|&nbsp; NMI = <b>{nmi_full:.4f}</b> "
        f"&nbsp;|&nbsp; n matched = <b>{n_full}</b><br><br>"
        f"{html_mod.escape(verdict_full[0])}</div>"
    )

    excl_block = ""
    if ct_excl is not None and v_excl is not None and verdict_excl is not None:
        excl_block = (
            f"<h2>Excluding γ == 'unclear'</h2>"
            f"<div class='verdict {verdict_excl[1]}'><span class='vlabel'>"
            f"VERDICT (unclear excluded)</span><br>Cramér's V = "
            f"<b>{v_excl:.4f}</b> &nbsp;|&nbsp; NMI = <b>{nmi_excl:.4f}</b> "
            f"&nbsp;|&nbsp; n matched = <b>{n_excl}</b><br><br>"
            f"{html_mod.escape(verdict_excl[0])}</div>"
            f"{style_table(ct_excl)}"
        )
        # human-vs-machine agreement on the square 12-class space
        if kappa_excl is not None and report_excl is not None:
            pr_rows = []
            for cls in GRIMSLEY_12_CLASSES:
                r = report_excl.get(cls, {})
                sup = int(r.get("support", 0))
                if sup == 0:
                    continue
                pr_rows.append(
                    f"<tr><th>{html_mod.escape(cls)}</th>"
                    f"<td>{r['precision']:.2f}</td><td>{r['recall']:.2f}</td>"
                    f"<td>{r['f1-score']:.2f}</td><td>{sup}</td></tr>"
                )
            macro_f1 = report_excl.get("macro avg", {}).get("f1-score", float("nan"))
            excl_block += (
                "<h2>Human(γ)-vs-machine(oracle) agreement — square 12-class</h2>"
                "<div class='verdict ambiguous'><span class='vlabel'>AGREEMENT</span>"
                f"<br>Cohen's κ = <b>{kappa_excl:.4f}</b> &nbsp;|&nbsp; "
                f"exact-agreement acc = <b>{acc_excl:.4f}</b> &nbsp;|&nbsp; "
                f"macro-F1 = <b>{macro_f1:.4f}</b><br>"
                "<span class='interp'>κ corrects agreement for chance "
                "(κ&lt;0.2 poor, 0.2–0.4 fair, 0.4–0.6 moderate, &gt;0.6 substantial). "
                "γ = human truth, oracle = machine prediction: recall = of γ-said-X "
                "fraction oracle agreed; precision = of oracle-said-X fraction γ "
                "agreed.</span></div>"
                "<table class='ct'><tr><th>class</th><th>precision</th>"
                "<th>recall</th><th>f1</th><th>support</th></tr>"
                + "".join(pr_rows) + "</table>"
            )

    heatmap_html = ""
    if heatmap_png:
        heatmap_html = (
            "<h2>Heatmap (all γ rows)</h2>"
            f"<img class='hm' src='data:image/png;base64,{heatmap_png}' "
            "alt='contingency heatmap'/>"
        )

    oracle_list = "<br>".join(html_mod.escape(str(p)) for p in oracle_paths)

    doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<title>α₃-C A8 — oracle taxonomy vs γ hand-labels</title>
<style>
 body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 2rem;
        color:#1a1a1a; max-width: 1100px; }}
 h1 {{ font-size: 1.5rem; }}
 h2 {{ margin-top: 1.8rem; border-bottom: 2px solid #eee; padding-bottom: .3rem; }}
 .meta {{ background:#f6f8fa; border:1px solid #e1e4e8; border-radius:8px;
          padding:.8rem 1rem; font-size:.9rem; }}
 .meta code {{ background:#eef; padding:0 .25rem; border-radius:3px; }}
 table.ct {{ border-collapse: collapse; margin: 1rem 0; font-size:.85rem; }}
 table.ct th, table.ct td {{ border:1px solid #d0d7de; padding:.3rem .55rem;
        text-align:center; }}
 table.ct th {{ background:#f0f3f6; }}
 table.ct td.total, table.ct th {{ }}
 table.ct td.total {{ background:#fff3cd; font-weight:700; }}
 .verdict {{ padding:1rem 1.2rem; border-radius:10px; margin:1rem 0;
        font-size:1.02rem; line-height:1.45; }}
 .verdict .vlabel {{ font-size:.75rem; letter-spacing:.08em; opacity:.7;
        text-transform:uppercase; }}
 .verdict.good {{ background:#d4edda; border:2px solid #28a745; }}
 .verdict.bad  {{ background:#f8d7da; border:2px solid #dc3545; }}
 .verdict.ambiguous {{ background:#fff3cd; border:2px solid #ffc107; }}
 img.hm {{ max-width:100%; border:1px solid #ddd; border-radius:6px; }}
 .interp {{ font-size:.85rem; color:#555; }}
</style></head><body>
<h1>α₃-C Phase A8 — oracle (VocalMat v1) taxonomy vs γ hand-labels</h1>
<div class="meta">
 <b>γ file:</b> <code>{html_mod.escape(str(gamma_path))}</code> ({n_gamma} rows)<br>
 <b>oracle file(s):</b> {oracle_list} ({n_oracle} unique call_id)<br>
 <b>γ rows matched on call_id:</b> <b>{n_matched}</b> / {n_gamma}<br>
 <b>γ shape vocab (8):</b> {", ".join(GAMMA_SHAPE_VOCAB)}<br>
 <b>oracle classes (12):</b> {", ".join(GRIMSLEY_12_CLASSES)}<br>
 <b>Cramér's V:</b> <code>V = sqrt(chi2 / (n * min(r-1, c-1)))</code>, uncorrected,
   chi2 from scipy.stats.chi2_contingency(correction=False).<br>
 <b>NMI:</b> sklearn.metrics.normalized_mutual_info_score.
</div>
{verdict_block}
<h2>Contingency table (all γ rows) — rows = γ shape_label, cols = oracle top1_class</h2>
{style_table(ct_full)}
<p class="interp">Interpretation thresholds: V &gt; 0.5 ⇒ taxonomy IS geometric
 (α₃ credible); V &lt; 0.3 ⇒ taxonomy keys on non-shape (γ is the only honest
 anchor; α₃ oracle eval contaminated); 0.3 ≤ V ≤ 0.5 ⇒ ambiguous.</p>
{heatmap_html}
{excl_block}
</body></html>"""
    out_path.write_text(doc, encoding="utf-8")


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="α₃-C A8: cross-tab oracle (VocalMat v1) taxonomy vs γ hand-labels."
    )
    ap.add_argument("--gamma", default="data/manual_shape_labels.csv",
                    help="γ hand-labels CSV (call_id, cohort, shape_label, labeled_at_index).")
    ap.add_argument("--oracle-labels", default="data/labels_vocalmat_v1_on_131204.csv",
                    help="Oracle CSV(s); comma-separated list allowed (concatenated, "
                         "deduped on call_id).")
    ap.add_argument("--output", default=DEFAULT_OUTPUT,
                    help="HTML report output path.")
    ap.add_argument("--csv-out", default=DEFAULT_CSV_OUT,
                    help="Contingency-table CSV output path.")
    ap.add_argument("--exclude-unclear", action="store_true",
                    help="Also report the primary stat with γ=='unclear' rows dropped "
                         "(both with/without are always shown).")
    args = ap.parse_args(argv)

    gamma_path = _resolve(args.gamma)
    oracle_paths = [_resolve(s.strip()) for s in args.oracle_labels.split(",") if s.strip()]
    output_path = _resolve(args.output) if not Path(args.output).is_absolute() else Path(args.output)
    csv_out_path = _resolve(args.csv_out)

    # ----- startup param echo (repo rule) -----
    print("=" * 70)
    print("  α₃-C Phase A8 — oracle taxonomy vs γ hand-labels (cross-tab)")
    print("=" * 70)
    print("  PARAMETERS")
    print(f"    --gamma          : {gamma_path}")
    print(f"    --oracle-labels  : {[str(p) for p in oracle_paths]}")
    print(f"    --output (HTML)  : {output_path}")
    print(f"    --csv-out        : {csv_out_path}")
    print(f"    --exclude-unclear: {args.exclude_unclear}")
    print(f"    γ shape vocab (8): {list(GAMMA_SHAPE_VOCAB)}")
    print(f"    oracle classes(12): {list(GRIMSLEY_12_CLASSES)}")
    print(f"    Cramér's V       : V = sqrt(chi2 / (n * min(r-1, c-1)))  [uncorrected]")
    print("=" * 70)

    # ----- graceful absence handling -----
    if not gamma_path.exists():
        graceful_exit(f"γ file does not exist yet: {gamma_path}")
    for p in oracle_paths:
        if not p.exists():
            graceful_exit(f"oracle-labels file does not exist yet: {p}")

    # ----- load -----
    gamma = load_gamma(gamma_path)
    oracle = load_oracle(oracle_paths)
    n_gamma = len(gamma)
    n_oracle = len(oracle)
    print(f"\n  loaded γ rows         : {n_gamma}")
    print(f"  loaded oracle rows    : {n_oracle} (unique call_id after dedup)")

    # ----- inner join on call_id -----
    merged = gamma.merge(
        oracle[["call_id", "top1_class"]], on="call_id", how="inner"
    )
    n_matched = len(merged)
    print(f"  γ rows matched on call_id: {n_matched} / {n_gamma} "
          f"({n_matched / n_gamma:.1%})" if n_gamma else "  no γ rows")

    if n_matched == 0:
        graceful_exit(
            "0 γ rows matched any oracle call_id. Verify both files use the SAME "
            "call_id convention (hand_label_200.py builds '<wav_stem>__<det_index>'; "
            "the oracle's call_id must come from the same manifest column via "
            "label_patches_v1.py --id-column)."
        )

    # ----- categorical ordering for a stable full 8x12 grid -----
    gamma_cats = [c for c in GAMMA_SHAPE_VOCAB] + sorted(
        set(merged["shape_label"]) - set(GAMMA_SHAPE_VOCAB)
    )
    merged["shape_label"] = pd.Categorical(merged["shape_label"], categories=gamma_cats)
    merged["top1_class"] = pd.Categorical(
        merged["top1_class"],
        categories=list(GRIMSLEY_12_CLASSES)
        + sorted(set(merged["top1_class"].astype(str)) - set(GRIMSLEY_12_CLASSES)),
    )

    # ----- FULL contingency table -----
    ct_full = pd.crosstab(merged["shape_label"], merged["top1_class"], dropna=False)
    v_full, chi2_full, n_full, r_full, c_full = cramers_v(ct_full)
    nmi_full = nmi_score(merged["shape_label"].astype(str), merged["top1_class"].astype(str))
    verdict_full = verdict_for(v_full)

    # ----- print to stdout -----
    print("\n" + "-" * 70)
    print("  CONTINGENCY TABLE (all γ rows)  rows=γ shape_label  cols=oracle top1_class")
    print("-" * 70)
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(ct_full.to_string())
    print("-" * 70)
    print(f"  chi2 = {chi2_full:.4f}  | realised r×c = {r_full}×{c_full}  | n = {n_full}")
    print(f"  Cramér's V = {v_full:.4f}")
    print(f"  NMI        = {nmi_full:.4f}")
    print(f"  VERDICT    : {verdict_full[0]}")

    # ----- EXCLUDE-UNCLEAR variant (always computed, also reported) -----
    ct_excl = v_excl = nmi_excl = n_excl = verdict_excl = None
    sub = merged[merged["shape_label"].astype(str) != "unclear"].copy()
    if len(sub) > 0:
        sub["shape_label"] = sub["shape_label"].cat.remove_unused_categories()
        ct_excl = pd.crosstab(sub["shape_label"], sub["top1_class"], dropna=False)
        v_excl, chi2_excl, n_excl, r_excl, c_excl = cramers_v(ct_excl)
        nmi_excl = nmi_score(sub["shape_label"].astype(str), sub["top1_class"].astype(str))
        verdict_excl = verdict_for(v_excl)
        print("\n" + "-" * 70)
        print("  EXCLUDING γ == 'unclear'")
        print("-" * 70)
        print(f"  n = {n_excl}  | chi2 = {chi2_excl:.4f}  | realised r×c = {r_excl}×{c_excl}")
        print(f"  Cramér's V = {v_excl:.4f}")
        print(f"  NMI        = {nmi_excl:.4f}")
        print(f"  VERDICT    : {verdict_excl[0]}")

        # ----- human-vs-machine agreement on the square 12-class space -----
        # (2026-05-30: γ shares GRIMSLEY_12_CLASSES with the oracle, so κ +
        # per-class P/R are meaningful — γ is TRUTH, oracle is PREDICTION.)
        kappa_excl, acc_excl, report_excl = agreement_stats(
            sub["shape_label"], sub["top1_class"]
        )
        print("\n  --- human(γ)-vs-machine(oracle) agreement, 12-class square ---")
        print(f"  Cohen's κ              = {kappa_excl:.4f}")
        print(f"  exact-agreement acc    = {acc_excl:.4f}  (diagonal fraction)")
        print(f"  macro-F1 (oracle vs γ) = {report_excl['macro avg']['f1-score']:.4f}")
        print("  per-class  (γ=truth, oracle=pred):")
        print(f"    {'class':18s} {'prec':>6s} {'recall':>7s} {'f1':>6s} {'support':>8s}")
        for cls in GRIMSLEY_12_CLASSES:
            r = report_excl.get(cls, {})
            sup = int(r.get("support", 0))
            if sup == 0:
                continue
            print(f"    {cls:18s} {r['precision']:6.2f} {r['recall']:7.2f} "
                  f"{r['f1-score']:6.2f} {sup:8d}")
    else:
        kappa_excl = acc_excl = report_excl = None
        print("\n  (all matched γ rows were 'unclear' — exclude-unclear variant skipped)")

    # ----- write CSV -----
    csv_out_path.parent.mkdir(parents=True, exist_ok=True)
    ct_full.to_csv(csv_out_path)
    print(f"\n  wrote contingency CSV → {csv_out_path}")

    # ----- heatmap + HTML -----
    heatmap_png = heatmap_b64(ct_full, "γ shape_label vs oracle top1_class (counts)")
    write_html(
        out_path=output_path,
        ct_full=ct_full,
        v_full=v_full,
        nmi_full=nmi_full,
        n_full=n_full,
        verdict_full=verdict_full,
        ct_excl=ct_excl,
        v_excl=v_excl,
        nmi_excl=nmi_excl,
        n_excl=n_excl,
        verdict_excl=verdict_excl,
        n_gamma=n_gamma,
        n_oracle=n_oracle,
        n_matched=n_matched,
        gamma_path=gamma_path,
        oracle_paths=oracle_paths,
        heatmap_png=heatmap_png,
        kappa_excl=kappa_excl,
        acc_excl=acc_excl,
        report_excl=report_excl,
    )
    print(f"  wrote HTML report     → {output_path}")

    # ----- MANDATORY WSL URL (repo rule feedback_wsl_file_viewing) -----
    print("\n  View the report:")
    print(f"    {_wsl_url(output_path)}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Move B — within-cohort latent dispersion.

Computes the mean pairwise Euclidean distance in the 32-D VAE latent space
per cohort, with a bootstrap 95% CI on an equal-N (N=400) subsample.

Why these choices (from the source handoff
``docs/handoffs/2026-05-20_latent-analysis-b-a-c.md`` Move B section):

- Mean pairwise Euclidean is a U-statistic and is comparable across cohorts
  ONLY if N is equalized — pairwise distances inflate with N due to tail
  sampling. We subsample each cohort to N=400 (3452 floor is 406).
- Bootstrap with replacement is performed on the equal-N subsample (NOT the
  full cohort) to give a CI for the dispersion statistic at that N.
- Latents come from a VAE with KL prior ~N(0, I) so raw Euclidean is
  meaningful — no per-dim z-scoring.

CLI usage::

    .venv/bin/python scripts/analyze_latent_dispersion.py \\
        --latents-path results/contour_vae_combined/latents.parquet \\
        --out-dir results/latent_dispersion \\
        --n-per-cohort 400 --n-boot 500 --seed 42
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.spatial.distance import pdist  # noqa: E402

# Top-to-bottom forest-plot order (per spec).
_FOREST_ORDER = ["5970", "3452", "9252", "lab_131204"]


# ---------------------------------------------------------------------------
# Core math
# ---------------------------------------------------------------------------

def mean_pairwise_euclidean(Z: np.ndarray) -> float:
    """Mean pairwise Euclidean distance over the unique (i<j) pairs of rows.

    Implemented as ``pdist(Z, 'euclidean').mean()`` — a U-statistic with no
    self-pairs and no double-counting. Raises ``ValueError`` if fewer than
    two rows are supplied (no pairs exist).
    """
    Z = np.asarray(Z)
    if Z.ndim != 2:
        raise ValueError(f"Z must be 2-D, got shape {Z.shape}")
    n = Z.shape[0]
    if n < 2:
        raise ValueError(
            f"mean_pairwise_euclidean requires N >= 2 rows; got N={n}"
        )
    d = pdist(Z, metric="euclidean")
    return float(d.mean())


def bootstrap_dispersion_ci(
    Z: np.ndarray,
    n_reps: int,
    seed: int,
    ci_pct: float = 95.0,
) -> Dict[str, Any]:
    """Bootstrap CI for ``mean_pairwise_euclidean(Z)``.

    Each rep: ``rng.choice(N, N, replace=True)`` -> sub-Z -> dispersion.
    CI is the central ``ci_pct`` interval via ``np.percentile``.

    Returns
    -------
    dict
        Keys: ``'point'`` (float, dispersion on the full input Z),
        ``'ci_lo'``, ``'ci_hi'`` (CI bounds),
        ``'reps'`` (np.ndarray, length n_reps, the bootstrap replicates).
    """
    Z = np.asarray(Z)
    n = Z.shape[0]
    point = mean_pairwise_euclidean(Z)

    rng = np.random.default_rng(seed)
    reps = np.empty(n_reps, dtype=np.float64)
    for r in range(n_reps):
        idx = rng.choice(n, n, replace=True)
        reps[r] = mean_pairwise_euclidean(Z[idx])

    tail = (100.0 - ci_pct) / 2.0
    ci_lo, ci_hi = np.percentile(reps, [tail, 100.0 - tail])
    return {
        "point": float(point),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "reps": reps,
    }


# ---------------------------------------------------------------------------
# Data plumbing
# ---------------------------------------------------------------------------

def load_latents(path: str) -> pd.DataFrame:
    """Load the combined VAE latents parquet into a DataFrame.

    Returns columns: ``z_0..z_31`` (float), ``cohort``, ``wav_stem``,
    ``call_id`` (and any other columns present in the parquet).
    """
    df = pd.read_parquet(path)
    return df


def subsample_per_cohort(
    df: pd.DataFrame,
    n_per_cohort: int,
    seed: int,
) -> pd.DataFrame:
    """Equal-N subsample without replacement, per cohort.

    Iterates cohorts in sorted (lexicographic) name order with a SINGLE
    ``rng`` instance so the call sequence is fully deterministic given
    ``seed``. Raises ``ValueError`` (mentioning the offending cohort) if any
    cohort has fewer than ``n_per_cohort`` rows.
    """
    if "cohort" not in df.columns:
        raise ValueError("DataFrame missing required 'cohort' column")

    rng = np.random.default_rng(seed)
    pieces = []
    # Deterministic cohort order: sorted by name. Critical for seed
    # reproducibility — see test_subsample_per_cohort_exact_n_and_seed_reproducibility.
    for coh in sorted(df["cohort"].unique()):
        sub = df[df["cohort"] == coh]
        n_avail = len(sub)
        if n_avail < n_per_cohort:
            raise ValueError(
                f"Cohort {coh!r} has only {n_avail} rows, requested "
                f"n_per_cohort={n_per_cohort} (cannot sample without replacement)"
            )
        idx = rng.choice(n_avail, n_per_cohort, replace=False)
        pieces.append(sub.iloc[idx])
    return pd.concat(pieces, axis=0)


def _latent_matrix(df: pd.DataFrame) -> np.ndarray:
    """Extract the z_0..z_31 columns into a contiguous float64 matrix."""
    cols = [f"z_{i}" for i in range(32)]
    Z = df[cols].to_numpy(dtype=np.float64, copy=False)
    return Z


def compute_cohort_dispersion(
    df: pd.DataFrame,
    n_per_cohort: int,
    n_boot: int,
    seed: int,
) -> pd.DataFrame:
    """Per-cohort dispersion with bootstrap CI on the equal-N subsample.

    Returns a DataFrame sorted by cohort name with columns:
    ``['cohort', 'n_in_cohort', 'n_subsampled', 'point', 'ci_lo', 'ci_hi',
       'n_boot', 'seed']``.
    """
    # Snapshot full-cohort sizes BEFORE subsampling.
    cohort_sizes = df["cohort"].value_counts().to_dict()

    sub = subsample_per_cohort(df, n_per_cohort=n_per_cohort, seed=seed)

    rows = []
    # Deterministic order: sort cohorts. Per-cohort bootstrap seed derived
    # from the master seed so changing one cohort doesn't perturb others.
    for offset, coh in enumerate(sorted(sub["cohort"].unique())):
        sub_coh = sub[sub["cohort"] == coh]
        Z = _latent_matrix(sub_coh)
        ci = bootstrap_dispersion_ci(
            Z, n_reps=n_boot, seed=seed + offset, ci_pct=95.0
        )
        rows.append({
            "cohort": coh,
            "n_in_cohort": int(cohort_sizes[coh]),
            "n_subsampled": int(len(sub_coh)),
            "point": ci["point"],
            "ci_lo": ci["ci_lo"],
            "ci_hi": ci["ci_hi"],
            "n_boot": int(n_boot),
            "seed": int(seed),
        })
    return pd.DataFrame(rows).sort_values("cohort").reset_index(drop=True)


# ---------------------------------------------------------------------------
# Output artifacts (figure + HTML)
# ---------------------------------------------------------------------------

def _make_forest_plot(out: pd.DataFrame, png_path: Path) -> None:
    """Forest plot: cohorts on Y, dispersion on X, horizontal CI bars."""
    order = [c for c in _FOREST_ORDER if c in set(out["cohort"])]
    # Append any cohort not in the canonical order (defensive).
    for c in out["cohort"]:
        if c not in order:
            order.append(c)

    rows = out.set_index("cohort").loc[order]
    # Y axis: row 0 at the TOP (so 5970 is at the top per spec).
    y = np.arange(len(rows))[::-1]

    point = rows["point"].to_numpy()
    ci_lo = rows["ci_lo"].to_numpy()
    ci_hi = rows["ci_hi"].to_numpy()
    err_lo = point - ci_lo
    err_hi = ci_hi - point

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.errorbar(
        point,
        y,
        xerr=[err_lo, err_hi],
        fmt="o",
        capsize=4,
        color="black",
        ecolor="black",
        markersize=7,
        linewidth=1.5,
    )
    ax.set_yticks(y)
    ax.set_yticklabels(order)
    ax.set_xlabel("Mean pairwise Euclidean distance (32-D latent)")
    ax.set_ylabel("Cohort")
    ax.set_title(
        "Mean pairwise Euclidean distance in 32-D latent\n"
        "(N=400 subsample, 500-rep bootstrap)"
    )
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)


def _decision_gate(out: pd.DataFrame) -> str:
    """Return the one-line decision-gate verdict string per the spec."""
    idx = out.set_index("cohort")
    if "5970" not in idx.index or "lab_131204" not in idx.index:
        return "INDETERMINATE — missing 5970 and/or lab_131204 from results"

    wild = idx.loc["5970"]
    lab = idx.loc["lab_131204"]

    if wild["ci_lo"] > lab["ci_hi"]:
        return (
            f"GREEN LIGHT A — 5970 > lab_131204, CIs separated "
            f"(5970 ci_lo={wild['ci_lo']:.4f} > lab ci_hi={lab['ci_hi']:.4f})"
        )
    if lab["ci_lo"] > wild["ci_hi"]:
        return (
            f"REVERSE SIGNAL — lab_131204 > 5970; document and reframe "
            f"(lab ci_lo={lab['ci_lo']:.4f} > 5970 ci_hi={wild['ci_hi']:.4f})"
        )
    return (
        f"OVERLAP — pause and investigate latent degeneracy "
        f"(5970 CI=[{wild['ci_lo']:.4f},{wild['ci_hi']:.4f}], "
        f"lab CI=[{lab['ci_lo']:.4f},{lab['ci_hi']:.4f}])"
    )


def _write_summary_html(
    out: pd.DataFrame,
    html_path: Path,
    png_path: Path,
    params: Dict[str, Any],
) -> None:
    """Self-contained HTML report (figure inlined as base64 PNG)."""
    img_b64 = base64.b64encode(png_path.read_bytes()).decode("ascii")
    img_tag = f'<img alt="forest plot" src="data:image/png;base64,{img_b64}" />'

    # Result table
    table_html = out.to_html(index=False, float_format=lambda v: f"{v:.6f}")

    # Run parameters dl
    dl_items = []
    for k, v in params.items():
        dl_items.append(f"<dt>{k}</dt><dd>{v}</dd>")
    dl_html = "<dl>" + "".join(dl_items) + "</dl>"

    decision = _decision_gate(out)
    timestamp = _dt.datetime.now().isoformat(timespec="seconds")

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Latent dispersion — Move B</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 980px; margin: 2em auto; padding: 0 1em; color: #222; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
    h2 {{ margin-top: 1.6em; color: #333; }}
    dl {{ display: grid; grid-template-columns: max-content 1fr;
          column-gap: 1em; row-gap: 0.25em; }}
    dt {{ font-weight: 600; color: #555; }}
    table {{ border-collapse: collapse; margin: 0.5em 0; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: right; }}
    th {{ background: #f0f0f0; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    .decision {{ padding: 0.8em 1em; background: #f7f7f0;
                 border-left: 4px solid #888; margin: 1em 0; }}
    footer {{ margin-top: 2em; color: #888; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>Latent dispersion — Move B</h1>

  <h2>Run parameters</h2>
  {dl_html}

  <h2>Result</h2>
  {table_html}

  <h2>Forest plot</h2>
  {img_tag}

  <h2>Decision-gate read</h2>
  <p class="decision">{decision}</p>

  <footer>Generated on {timestamp}</footer>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Move B — within-cohort latent dispersion (mean pairwise "
                    "Euclidean) with bootstrap 95% CI on equal-N subsample."
    )
    p.add_argument("--latents-path", required=True, type=str)
    p.add_argument("--out-dir", required=True, type=str)
    p.add_argument("--n-per-cohort", type=int, default=400)
    p.add_argument("--n-boot", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Echo every parameter — per project feedback rule
    # (feedback_analysis_print_params).
    print(f"[PARAM] latents_path     = {args.latents_path}")
    print(f"[PARAM] out_dir          = {args.out_dir}")
    print(f"[PARAM] n_per_cohort     = {args.n_per_cohort}")
    print(f"[PARAM] n_boot           = {args.n_boot}")
    print(f"[PARAM] seed             = {args.seed}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading latents...")
    df = load_latents(args.latents_path)
    print(f"[INFO] Loaded {len(df):,} rows, columns: "
          f"{[c for c in df.columns if not c.startswith('z_')]} (+ 32 z_*)")

    pre_counts = df["cohort"].value_counts().to_dict()
    print("[INFO] Per-cohort counts BEFORE subsample:")
    for coh in sorted(pre_counts):
        print(f"[INFO]   {coh:>14s}: {pre_counts[coh]:>7d}")

    print(f"[INFO] Computing dispersion (n_per_cohort={args.n_per_cohort}, "
          f"n_boot={args.n_boot}, seed={args.seed})...")
    out = compute_cohort_dispersion(
        df,
        n_per_cohort=args.n_per_cohort,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    csv_path = out_dir / "dispersion_by_cohort.csv"
    out.to_csv(csv_path, index=False)
    print(f"[INFO] Wrote CSV  -> {csv_path}")

    png_path = out_dir / "figure.png"
    _make_forest_plot(out, png_path)
    print(f"[INFO] Wrote PNG  -> {png_path}")

    html_path = out_dir / "summary.html"
    params = {
        "n_per_cohort": args.n_per_cohort,
        "n_boot": args.n_boot,
        "seed": args.seed,
        "latents_path": args.latents_path,
        "total_patches": int(len(df)),
        **{f"n_in_cohort[{c}]": int(pre_counts[c]) for c in sorted(pre_counts)},
    }
    _write_summary_html(out, html_path, png_path, params)
    print(f"[INFO] Wrote HTML -> {html_path}")

    print("[INFO] Decision-gate read:")
    print(f"[INFO]   {_decision_gate(out)}")


if __name__ == "__main__":
    main()

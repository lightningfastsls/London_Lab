"""Exploratory comparison of two 'normal-rate' wild dyads: 3452 vs 9252.

Motivation: Lab cohorts are selected for high vocalization rate. 5970 is also
a high-rate wild dyad. The clean "is wild different from lab?" comparison
should hold rate constant — and the cleanest matched-rate comparison we have
is 3452 vs 9252 (both normal-rate wild). If they're essentially identical,
"wild" has a coherent repertoire at this rate. If they're distinct, individual
variability dominates even at matched rate.

This is exploratory analysis. No tests; reuses tested helpers from Move A.
"""
from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import sys
from io import BytesIO
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from analyze_latent_repertoire_jsd import (  # noqa: E402
    js_divergence_bits,
    cluster_proportions,
)


COHORT_A = "3452"
COHORT_B = "9252"
N_BOOT = 1000
SEED = 42
DETECTION_CSV_PATHS = {
    "3452": "/home/shachar/projects/mickey_london_lab/classified_detections_3452.csv",
    "9252": "/home/shachar/projects/mickey_london_lab/classified_detections_9252.csv",
}
# Shape-only acoustic features (skip mean_power_db, tonality — cage-confounded
# per [[feedback-rig-artifact-mean-power-db]])
SHAPE_FEATURES = [
    "call_length_s",
    "principal_freq_hz",
    "bandwidth_hz",
    "freq_std_dev_hz",
    "slope",
    "sinuosity",
    "peak_freq_khz",
]


def cohen_d(x: np.ndarray, y: np.ndarray) -> float:
    """Cohen's d with pooled SD. Positive = x > y."""
    nx, ny = len(x), len(y)
    if nx < 2 or ny < 2:
        return float("nan")
    mx, my = np.mean(x), np.mean(y)
    vx, vy = np.var(x, ddof=1), np.var(y, ddof=1)
    pooled_sd = np.sqrt(((nx - 1) * vx + (ny - 1) * vy) / (nx + ny - 2))
    if pooled_sd == 0:
        return float("nan")
    return (mx - my) / pooled_sd


def fig_to_b64(fig) -> str:
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--latents-path",
                        default="results/contour_vae_combined/latents.parquet")
    parser.add_argument("--kmeans-path", default="models/latent_kmeans/k20.joblib")
    parser.add_argument("--out-dir", default="results/compare_3452_9252")
    parser.add_argument("--n-boot", type=int, default=N_BOOT)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[PARAM] latents_path = {args.latents_path}")
    print(f"[PARAM] kmeans_path  = {args.kmeans_path}")
    print(f"[PARAM] cohorts      = {COHORT_A} vs {COHORT_B}")
    print(f"[PARAM] n_boot       = {args.n_boot}")
    print(f"[PARAM] seed         = {args.seed}")

    # ---- 1. Load and filter ----
    lat = pd.read_parquet(args.latents_path)
    z_cols = [f"z_{i}" for i in range(32)]
    a = lat[lat["cohort"] == COHORT_A].copy()
    b = lat[lat["cohort"] == COHORT_B].copy()
    print(f"[INFO] {COHORT_A}: {len(a)} patches, {a['call_id'].nunique()} unique call_ids "
          f"({a.groupby(['wav_stem','call_id']).ngroups} (wav_stem,call_id))")
    print(f"[INFO] {COHORT_B}: {len(b)} patches, {b['call_id'].nunique()} unique call_ids "
          f"({b.groupby(['wav_stem','call_id']).ngroups} (wav_stem,call_id))")

    Z_a = a[z_cols].to_numpy(dtype=np.float32)
    Z_b = b[z_cols].to_numpy(dtype=np.float32)

    # ---- 2. Per-cluster occupancy with bootstrap CI ----
    kmeans = joblib.load(args.kmeans_path)
    k = kmeans.n_clusters
    labels_all = kmeans.predict(np.vstack([Z_a, Z_b]).astype(np.float32))
    labels_a = labels_all[: len(Z_a)]
    labels_b = labels_all[len(Z_a):]

    def counts_to_props(counts, k):
        t = counts.sum()
        return counts / t if t > 0 else np.ones(k) / k

    def occupancy(labels, k):
        cnts = np.bincount(labels, minlength=k).astype(float)
        return counts_to_props(cnts, k), cnts

    p_a, c_a = occupancy(labels_a, k)
    p_b, c_b = occupancy(labels_b, k)

    # Bootstrap by (wav_stem, call_id) tuples per cohort
    def bootstrap_props(df, labels, k, n_reps, rng):
        # group by (wav_stem, call_id): list of cluster-count vectors per call
        df = df.copy()
        df["_lab"] = labels
        groups = df.groupby(["wav_stem", "call_id"])["_lab"].apply(
            lambda s: np.bincount(s.to_numpy(), minlength=k).astype(float)
        )
        call_counts = np.stack(groups.values)
        n_calls = len(call_counts)
        reps = np.empty((n_reps, k))
        for r in range(n_reps):
            idx = rng.integers(0, n_calls, n_calls)
            total = call_counts[idx].sum(axis=0)
            reps[r] = counts_to_props(total, k)
        return reps

    print(f"[INFO] bootstrap {args.n_boot} reps for occupancy CIs")
    reps_a = bootstrap_props(a, labels_a, k, args.n_boot, np.random.default_rng(args.seed))
    reps_b = bootstrap_props(b, labels_b, k, args.n_boot,
                              np.random.default_rng(args.seed + 1))

    # ---- 3. JSD with bootstrap CI ----
    jsd_point = js_divergence_bits(p_a, p_b)
    jsd_reps = np.array([js_divergence_bits(reps_a[r], reps_b[r])
                         for r in range(args.n_boot)])
    jsd_lo, jsd_hi = np.percentile(jsd_reps, [2.5, 97.5])
    jsd_lo = min(jsd_lo, jsd_point)
    jsd_hi = max(jsd_hi, jsd_point)
    print(f"[RESULT] JSD({COHORT_A}, {COHORT_B}) = {jsd_point:.4f} "
          f"[{jsd_lo:.4f}, {jsd_hi:.4f}]")

    # ---- 4. Per-cluster log-ratio + significance via bootstrap ----
    eps = 1e-6
    cluster_rows = []
    for c in range(k):
        # bootstrap log-ratio
        lr_reps = np.log2((reps_a[:, c] + eps) / (reps_b[:, c] + eps))
        lr_point = float(np.log2((p_a[c] + eps) / (p_b[c] + eps)))
        lr_lo, lr_hi = np.percentile(lr_reps, [2.5, 97.5])
        sig = (lr_lo > 0) or (lr_hi < 0)
        cluster_rows.append({
            "cluster": c,
            f"prop_{COHORT_A}": p_a[c],
            f"prop_{COHORT_B}": p_b[c],
            f"n_{COHORT_A}": int(c_a[c]),
            f"n_{COHORT_B}": int(c_b[c]),
            "log2_ratio": lr_point,
            "lr_ci_lo": lr_lo,
            "lr_ci_hi": lr_hi,
            "differs_at_95": sig,
        })
    cluster_df = pd.DataFrame(cluster_rows)
    cluster_df.to_csv(out_dir / "cluster_occupancy.csv", index=False)

    n_differ = int(cluster_df["differs_at_95"].sum())
    print(f"[RESULT] {n_differ} of {k} clusters differ at 95% (CI excludes 0)")

    # ---- 5. Per-latent-dim Cohen's d ----
    dim_rows = []
    for d in range(32):
        x = Z_a[:, d]
        y = Z_b[:, d]
        d_val = cohen_d(x, y)
        dim_rows.append({
            "dim": d,
            f"mean_{COHORT_A}": float(np.mean(x)),
            f"mean_{COHORT_B}": float(np.mean(y)),
            f"sd_{COHORT_A}": float(np.std(x, ddof=1)),
            f"sd_{COHORT_B}": float(np.std(y, ddof=1)),
            "cohen_d": d_val,
            "abs_d": abs(d_val),
        })
    dim_df = pd.DataFrame(dim_rows).sort_values("abs_d", ascending=False)
    dim_df.to_csv(out_dir / "per_dim_cohen_d.csv", index=False)
    n_d04 = int((dim_df["abs_d"] > 0.4).sum())
    n_d08 = int((dim_df["abs_d"] > 0.8).sum())
    print(f"[RESULT] {n_d04} of 32 latent dims have |Cohen's d| > 0.4")
    print(f"[RESULT] {n_d08} of 32 latent dims have |Cohen's d| > 0.8")

    # ---- 6. PCA scatter ----
    Z_all = np.vstack([Z_a, Z_b])
    pca = PCA(n_components=2, random_state=args.seed)
    pc = pca.fit_transform(Z_all)
    pc_a = pc[: len(Z_a)]
    pc_b = pc[len(Z_a):]
    print(f"[INFO] PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, "
          f"PC2={pca.explained_variance_ratio_[1]:.3f}")

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pc_b[:, 0], pc_b[:, 1], s=8, alpha=0.4, label=f"{COHORT_B} (n={len(pc_b)})",
               color="#1f77b4")
    ax.scatter(pc_a[:, 0], pc_a[:, 1], s=8, alpha=0.4, label=f"{COHORT_A} (n={len(pc_a)})",
               color="#d62728")
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    ax.set_title(f"PCA of contour-VAE latents: {COHORT_A} vs {COHORT_B}")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.2)
    pca_b64 = fig_to_b64(fig)

    # ---- 7. Acoustic shape features ----
    df_a = pd.read_csv(DETECTION_CSV_PATHS[COHORT_A])
    df_b = pd.read_csv(DETECTION_CSV_PATHS[COHORT_B])
    feat_rows = []
    for feat in SHAPE_FEATURES:
        if feat not in df_a.columns or feat not in df_b.columns:
            continue
        x = df_a[feat].dropna().to_numpy()
        y = df_b[feat].dropna().to_numpy()
        feat_rows.append({
            "feature": feat,
            f"mean_{COHORT_A}": float(np.mean(x)),
            f"mean_{COHORT_B}": float(np.mean(y)),
            f"sd_{COHORT_A}": float(np.std(x, ddof=1)),
            f"sd_{COHORT_B}": float(np.std(y, ddof=1)),
            f"n_{COHORT_A}": len(x),
            f"n_{COHORT_B}": len(y),
            "cohen_d": cohen_d(x, y),
        })
    feat_df = pd.DataFrame(feat_rows)
    feat_df["abs_d"] = feat_df["cohen_d"].abs()
    feat_df = feat_df.sort_values("abs_d", ascending=False)
    feat_df.to_csv(out_dir / "acoustic_features_comparison.csv", index=False)

    # ---- 8. Cluster occupancy bar plot ----
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(k)
    width = 0.4
    ax.bar(x - width/2, p_a, width, label=COHORT_A, color="#d62728", alpha=0.8)
    ax.bar(x + width/2, p_b, width, label=COHORT_B, color="#1f77b4", alpha=0.8)
    # Mark significant clusters
    for i, row in cluster_df.iterrows():
        if row["differs_at_95"]:
            ax.text(row["cluster"], max(p_a[int(row["cluster"])],
                                         p_b[int(row["cluster"])]) + 0.005,
                    "*", ha="center", fontsize=14, color="black")
    ax.set_xlabel("K-means cluster (0..19)")
    ax.set_ylabel("Fraction of patches")
    ax.set_title(f"Per-cluster occupancy: {COHORT_A} vs {COHORT_B}  (* = 95% CI excludes 0)")
    ax.set_xticks(x)
    ax.legend()
    ax.grid(True, axis="y", alpha=0.2)
    occ_b64 = fig_to_b64(fig)

    # ---- 9. Per-dim Cohen's d plot ----
    fig, ax = plt.subplots(figsize=(10, 4))
    dim_ordered = dim_df.sort_values("dim")
    colors = ["#888"] * 32
    for i, row in dim_ordered.iterrows():
        if abs(row["cohen_d"]) > 0.4:
            colors[int(row["dim"])] = "#d62728" if row["cohen_d"] > 0 else "#1f77b4"
    ax.bar(dim_ordered["dim"], dim_ordered["cohen_d"], color=colors)
    ax.axhline(0.4, color="#888", linestyle="--", alpha=0.5, label="|d| = 0.4 (small effect)")
    ax.axhline(-0.4, color="#888", linestyle="--", alpha=0.5)
    ax.axhline(0.8, color="#444", linestyle=":", alpha=0.5, label="|d| = 0.8 (large effect)")
    ax.axhline(-0.8, color="#444", linestyle=":", alpha=0.5)
    ax.set_xlabel("Latent dim (0..31)")
    ax.set_ylabel(f"Cohen's d  ({COHORT_A} - {COHORT_B})")
    ax.set_title("Per-latent-dim effect size")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, axis="y", alpha=0.2)
    cohen_b64 = fig_to_b64(fig)

    # ---- 10. HTML report ----
    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    html = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8" />
<title>3452 vs 9252 — normal-rate wild dyad comparison</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
          max-width: 1100px; margin: 2em auto; padding: 0 1em; color: #222; }}
  h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
  h2 {{ margin-top: 1.6em; color: #333; }}
  dl {{ display: grid; grid-template-columns: max-content 1fr;
        column-gap: 1em; row-gap: 0.25em; }}
  dt {{ font-weight: 600; color: #555; }}
  table {{ border-collapse: collapse; margin: 0.5em 0; font-size: 0.92em; }}
  th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: right; }}
  th {{ background: #f0f0f0; }}
  td:first-child, th:first-child {{ text-align: left; }}
  img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
  .verdict {{ padding: 0.8em 1em; background: #f7f7f0;
              border-left: 4px solid #888; margin: 1em 0; }}
  .differ {{ background: #fff5e6; }}
  footer {{ margin-top: 2em; color: #888; font-size: 0.9em; }}
</style></head><body>

<h1>3452 vs 9252 — two normal-rate wild dyads</h1>
<p><em>Exploratory comparison. Motivation: rate-stratification hypothesis —
lab USV research is biased toward high-singers, so the cleanest matched-rate
comparison we have is between two normal-rate wild dyads.</em></p>

<h2>Run parameters</h2>
<dl>
  <dt>latents</dt><dd>{args.latents_path}</dd>
  <dt>K-means</dt><dd>{args.kmeans_path}  (K = {k})</dd>
  <dt>n_boot</dt><dd>{args.n_boot}</dd>
  <dt>seed</dt><dd>{args.seed}</dd>
  <dt>{COHORT_A}</dt><dd>{len(a)} patches  /  {a.groupby(['wav_stem','call_id']).ngroups} unique calls</dd>
  <dt>{COHORT_B}</dt><dd>{len(b)} patches  /  {b.groupby(['wav_stem','call_id']).ngroups} unique calls</dd>
</dl>

<h2>Headline JSD</h2>
<p class="verdict">JSD({COHORT_A}, {COHORT_B}) = <b>{jsd_point:.4f} bits</b>
[{jsd_lo:.4f}, {jsd_hi:.4f}] (95% bootstrap CI, resampling calls).
<br>Comparison floors for reference: lab_matched ↔ lab_swap = 0.007 (alphabet-identity floor);
3452 ↔ lab_swap = 0.210 (cross-strain reference).</p>

<h2>Per-cluster occupancy</h2>
<p>{n_differ} of {k} clusters differ at 95% (bootstrap CI of log-ratio excludes 0).
Bars: red = {COHORT_A}, blue = {COHORT_B}. Stars = clusters where the two cohorts
differ. Stars over a tall red bar mean "{COHORT_A} uses this letter much more
than {COHORT_B}".</p>
<img src="data:image/png;base64,{occ_b64}" />

{cluster_df.sort_values("abs_d" if "abs_d" in cluster_df.columns else "log2_ratio",
    key=lambda s: s.abs() if "log2_ratio" in s.name else s, ascending=False).round(4).to_html(index=False, classes="dataframe")}

<h2>Per-latent-dim effect size (Cohen's d)</h2>
<p>{n_d04} of 32 dims have |d| &gt; 0.4 (small effect);
{n_d08} of 32 dims have |d| &gt; 0.8 (large effect).
Red bars = {COHORT_A} higher; blue bars = {COHORT_B} higher.</p>
<img src="data:image/png;base64,{cohen_b64}" />

<h3>Top-10 most-differentiating latent dims</h3>
{dim_df.head(10).round(4).to_html(index=False)}

<h2>PCA 2D projection</h2>
<p>Top 2 principal components of just these two cohorts' latents
({pca.explained_variance_ratio_[0]:.1%} + {pca.explained_variance_ratio_[1]:.1%}
= {(pca.explained_variance_ratio_[0]+pca.explained_variance_ratio_[1]):.1%} variance).
If the two cohorts occupy distinct regions, the clouds will separate visually.</p>
<img src="data:image/png;base64,{pca_b64}" />

<h2>Acoustic shape features (from detection CSVs)</h2>
<p>Sorted by |Cohen's d|. Cage-confounded features (mean_power_db, tonality) are
deliberately excluded. Positive d = {COHORT_A} higher; negative d = {COHORT_B} higher.</p>
{feat_df.round(4).to_html(index=False)}

<h2>Reading guide</h2>
<ul>
<li>JSD &lt; 0.05 → distributions essentially identical (alphabets shared).</li>
<li>JSD 0.05–0.15 → small but detectable difference.</li>
<li>JSD &gt; 0.15 → substantial difference.</li>
<li>Cohen's d &lt; 0.2 → negligible per-dim effect; 0.2–0.5 = small; 0.5–0.8 = medium; &gt; 0.8 = large.</li>
<li>Sample sizes are small (3452: 406 patches; 9252: 584 patches). Effect sizes
that depend on means are reliable; effect sizes that depend on tails are not.</li>
</ul>

<footer>Generated {timestamp}.  Worktree: latent-analysis-b-a-c.</footer>
</body></html>"""

    out_html = out_dir / "summary.html"
    out_html.write_text(html)
    print(f"[OK] wrote {out_html}")
    print(f"[OK] wrote {out_dir / 'cluster_occupancy.csv'}")
    print(f"[OK] wrote {out_dir / 'per_dim_cohen_d.csv'}")
    print(f"[OK] wrote {out_dir / 'acoustic_features_comparison.csv'}")


if __name__ == "__main__":
    main()

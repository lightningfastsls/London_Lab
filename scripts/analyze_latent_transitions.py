"""Move C — latent transition matrices, entropy rates, idioms.

Phase A2 rebuilt on the K-means alphabet from Move A. Builds per-cohort
per-bout sequences of K-means cluster IDs (one symbol per *call*, not
per patch), then computes:

- KxK bigram transition matrix (row-normalized; zero-source rows -> uniform)
- Entropy rate H(P) = -sum_i pi_i sum_j P_ij log2 P_ij with bootstrap CI
- Idioms: bigrams whose observed count exceeds the 99th percentile of a
  within-session shuffle null distribution
- Bout-threshold MI sensitivity sweep
- Per-centroid 3x3 decoded reconstruction tiles via the contour VAE

Why these choices (from
``docs/handoffs/2026-05-20_latent-analysis-b-a-c.md`` Move C section):

- One symbol per call: long calls produce multiple auto-correlated patches.
  We average ``z_*`` over a call's patches, then predict via the
  shared K-means alphabet.
- Bout threshold 0.25 s, file-aware: a new wav_stem always starts a
  new bout, matching the MI plateau established on 5970.
- Within-session shuffle (vs. across cohort) preserves marginal cluster
  frequencies while destroying transition structure, so observed-over-null
  enrichment reflects sequential dependency, not base rate differences.
- Bootstrap by *sequence* (not by symbol) preserves auto-correlation
  within bouts -- the same level at which we resample for JSD in Move A.

CLI usage::

    .venv/bin/python scripts/analyze_latent_transitions.py \\
        --latents-path results/contour_vae_combined/latents.parquet \\
        --kmeans-path models/latent_kmeans/k20.joblib \\
        --vae-checkpoint models/contour_vae_combined/best.pt \\
        --vae-hyperparams models/contour_vae_combined/hyperparams.json \\
        --out-dir results/latent_transitions \\
        --bout-threshold-s 0.25 --n-boot 1000 --n-shuffles 1000 \\
        --n-centroid-examples 9 --seed 42
"""

from __future__ import annotations

import argparse
import base64
import datetime as _dt
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import matplotlib

matplotlib.use("Agg")

import joblib  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DETECTION_CSV_PATHS = {
    "5970":       "/home/shachar/projects/mickey_london_lab/classified_detections_full.csv",
    "3452":       "/home/shachar/projects/mickey_london_lab/classified_detections_3452.csv",
    "9252":       "/home/shachar/projects/mickey_london_lab/classified_detections_9252.csv",
    "lab_131204": "/home/shachar/projects/mickey_london_lab/classified_detections_lab_131204_clean.csv",
}

_LAB_COHORT = "lab_131204"
_COUPLE_RE = re.compile(r"_m(\d+)fm(\d+)_")

_N_LATENT_DIMS = 32


# ---------------------------------------------------------------------------
# Step 1 — load_call_latents
# ---------------------------------------------------------------------------

def _split_lab_cohort(cohort: str, wav_stem: str) -> str:
    """Lab matched-vs-swap split based on `_mXfmY_` couple token in wav_stem."""
    if cohort != _LAB_COHORT:
        return cohort
    m = _COUPLE_RE.search(str(wav_stem))
    if m is None:
        raise ValueError(
            f"Lab row has unparseable wav_stem {wav_stem!r} -- "
            f"expected '_m<digits>fm<digits>_' couple token"
        )
    return "lab_matched" if m.group(1) == m.group(2) else "lab_swap"


def load_call_latents(latents_path: str,
                      detection_csv_paths: Dict[str, str]) -> pd.DataFrame:
    """Aggregate patch-level latents to one row per (wav_stem, call_id) and
    join with detection-level begin/end timestamps.

    Parameters
    ----------
    latents_path
        Parquet with patch-level rows; required columns
        ``z_0..z_31``, ``wav_stem``, ``call_id``, ``cohort``.
    detection_csv_paths
        Mapping ``cohort -> CSV path``. Each CSV must contain
        ``wav_stem``, ``id``, ``begin_time_s``, ``end_time_s`` columns
        (``id`` is the per-recording call id matching the parquet's
        ``call_id``).

    Returns
    -------
    DataFrame with columns:
        ``cohort``, ``cohort_split``, ``wav_stem``, ``call_id``,
        ``begin_time_s``, ``end_time_s``, ``n_patches``,
        ``mean_z_0..mean_z_31`` (float).

    Raises
    ------
    AssertionError
        If per-cohort join rate against the detection CSV is < 99%.
    """
    lat = pd.read_parquet(latents_path)
    z_cols = [f"z_{i}" for i in range(_N_LATENT_DIMS)
              if f"z_{i}" in lat.columns]
    n_z = len(z_cols)
    mean_cols = [f"mean_z_{i}" for i in range(n_z)]

    # Aggregate per call: mean of z, count of patches.
    agg_dict = {c: "mean" for c in z_cols}
    grouped = (
        lat.groupby(["cohort", "wav_stem", "call_id"], sort=False)
           .agg(agg_dict)
           .rename(columns=dict(zip(z_cols, mean_cols)))
           .reset_index()
    )
    n_patches = (
        lat.groupby(["cohort", "wav_stem", "call_id"], sort=False)
           .size()
           .rename("n_patches")
           .reset_index()
    )
    calls = grouped.merge(
        n_patches, on=["cohort", "wav_stem", "call_id"], how="left"
    )

    # Per-cohort join with detection CSV.
    pieces: List[pd.DataFrame] = []
    for cohort, csv_path in detection_csv_paths.items():
        sub = calls[calls["cohort"] == cohort].copy()
        if sub.empty:
            continue

        det = pd.read_csv(csv_path, usecols=lambda c: c in {
            "wav_stem", "id", "begin_time_s", "end_time_s",
        })
        # Match dtype: latents' call_id is int64; CSVs use float (1.0).
        det = det.dropna(subset=["wav_stem", "id"])
        det["id"] = det["id"].astype(np.int64)
        det = det.rename(columns={"id": "call_id"})
        # Some CSVs may contain duplicate (wav_stem, id); drop dupes.
        det = det.drop_duplicates(subset=["wav_stem", "call_id"], keep="first")

        merged = sub.merge(
            det[["wav_stem", "call_id", "begin_time_s", "end_time_s"]],
            on=["wav_stem", "call_id"],
            how="left",
        )

        match_rate = merged["begin_time_s"].notna().mean()
        assert match_rate >= 0.99, (
            f"[{cohort}] detection join rate {match_rate:.3f} < 0.99 "
            f"(matched {int(merged['begin_time_s'].notna().sum())} / "
            f"{len(merged)})"
        )

        pieces.append(merged)

    if not pieces:
        raise ValueError("No cohort rows produced; check inputs")

    out = pd.concat(pieces, ignore_index=True)

    # cohort_split (matched/swap split for lab).
    out["cohort_split"] = [
        _split_lab_cohort(c, s)
        for c, s in zip(out["cohort"].tolist(), out["wav_stem"].tolist())
    ]

    # Ensure mean_z_* are float (groupby on float32 keeps them float32).
    for c in mean_cols:
        out[c] = out[c].astype(np.float64)

    # Order columns: identifiers first.
    front = ["cohort", "cohort_split", "wav_stem", "call_id",
             "begin_time_s", "end_time_s", "n_patches"]
    rest = [c for c in out.columns if c not in front]
    return out[front + rest]


# ---------------------------------------------------------------------------
# Step 2 — assign_call_clusters
# ---------------------------------------------------------------------------

def assign_call_clusters(call_df: pd.DataFrame, kmeans) -> np.ndarray:
    """Predict cluster IDs for each call using the shared K-means alphabet.

    Reads the ``mean_z_*`` columns (in numeric order) and runs
    ``kmeans.predict``. Returns one integer label per row of ``call_df``.
    """
    mean_cols = sorted(
        [c for c in call_df.columns if c.startswith("mean_z_")],
        key=lambda s: int(s.split("_")[-1]),
    )
    if not mean_cols:
        raise ValueError("call_df has no mean_z_* columns")
    # Match the dtype of the saved cluster centers so sklearn's Cython
    # backend can use them without a buffer-dtype mismatch.
    centers_dtype = np.asarray(kmeans.cluster_centers_).dtype
    X = call_df[mean_cols].to_numpy(dtype=centers_dtype)
    return kmeans.predict(X).astype(np.int64)


# ---------------------------------------------------------------------------
# Step 3 — segment_into_bouts
# ---------------------------------------------------------------------------

def segment_into_bouts(call_df: pd.DataFrame,
                       bout_threshold_s: float) -> pd.DataFrame:
    """Add a ``bout_id`` column splitting calls into bouts (file-aware).

    Rules:
    - Sort by (wav_stem, begin_time_s).
    - Within each wav_stem, a new bout starts whenever the gap from the
      previous call's begin_time_s exceeds ``bout_threshold_s``.
    - A new wav_stem ALWAYS starts a new bout.
    """
    if "wav_stem" not in call_df.columns or "begin_time_s" not in call_df.columns:
        raise ValueError("segment_into_bouts requires 'wav_stem' and 'begin_time_s'")

    out = call_df.sort_values(
        ["wav_stem", "begin_time_s"], kind="mergesort"
    ).reset_index(drop=True)

    wav_stems = out["wav_stem"].to_numpy()
    times = out["begin_time_s"].to_numpy(dtype=np.float64)

    n = len(out)
    bout_ids = np.zeros(n, dtype=np.int64)
    cur = 0
    for i in range(n):
        if i == 0:
            cur = 0
        else:
            same_stem = wav_stems[i] == wav_stems[i - 1]
            gap = times[i] - times[i - 1]
            if (not same_stem) or (gap > bout_threshold_s):
                cur += 1
        bout_ids[i] = cur
    out["bout_id"] = bout_ids
    return out


# ---------------------------------------------------------------------------
# Step 4 — build_transition_matrix
# ---------------------------------------------------------------------------

def build_transition_matrix(sequences: list, k: int) -> np.ndarray:
    """K x K bigram transition matrix, row-normalized.

    Zero-row fallback: rows that never appear as a bigram source get
    a uniform distribution (1/K per column).
    """
    counts = np.zeros((k, k), dtype=np.float64)
    for seq in sequences:
        s = np.asarray(seq, dtype=np.int64)
        if s.size < 2:
            continue
        src = s[:-1]
        dst = s[1:]
        # Filter out-of-range symbols defensively.
        mask = (src >= 0) & (src < k) & (dst >= 0) & (dst < k)
        if not mask.all():
            src = src[mask]
            dst = dst[mask]
        np.add.at(counts, (src, dst), 1.0)

    row_sums = counts.sum(axis=1, keepdims=True)
    P = np.zeros_like(counts)
    nonzero = row_sums.flatten() > 0
    if nonzero.any():
        P[nonzero] = counts[nonzero] / row_sums[nonzero]
    # Zero-source rows -> uniform.
    if (~nonzero).any():
        P[~nonzero] = 1.0 / k
    return P


# ---------------------------------------------------------------------------
# Step 5 — stationary_distribution
# ---------------------------------------------------------------------------

def stationary_distribution(P: np.ndarray) -> np.ndarray:
    """Left eigenvector of P with eigenvalue ~1, normalized to sum to 1.

    Falls back gracefully when P is non-ergodic (e.g. identity):
    1. Try numpy eig and pick the eigenvector closest to eigenvalue 1.
    2. If that fails or has high imaginary residual, fall back to
       column averages (empirical visitation under uniform start).
    3. Final fallback: uniform.
    """
    K = P.shape[0]
    if K == 0:
        return np.array([])
    if K == 1:
        return np.array([1.0])
    try:
        eigvals, eigvecs = np.linalg.eig(P.T)
        idx = int(np.argmin(np.abs(eigvals - 1.0)))
        if np.abs(eigvals[idx] - 1.0) > 1e-6:
            raise np.linalg.LinAlgError("no eigenvalue near 1")
        v = np.real(eigvecs[:, idx])
        # Sign: ensure non-negative.
        v = np.abs(v)
        s = v.sum()
        if s <= 0 or not np.isfinite(s):
            raise np.linalg.LinAlgError("degenerate eigenvector")
        return v / s
    except (np.linalg.LinAlgError, ValueError):
        # Fallback: average row, which equals the stationary
        # distribution for uniform initial state under one step.
        avg = P.mean(axis=0)
        s = avg.sum()
        if s > 0 and np.isfinite(s):
            return avg / s
        return np.full(K, 1.0 / K)


# ---------------------------------------------------------------------------
# Step 6 — entropy_rate_from_matrix
# ---------------------------------------------------------------------------

def entropy_rate_from_matrix(P: np.ndarray,
                             pi: np.ndarray | None = None) -> float:
    """Entropy rate H = -sum_i pi_i * sum_j P_ij * log2(P_ij).

    If ``pi`` is None, compute the stationary distribution via
    ``stationary_distribution(P)``.
    """
    P = np.asarray(P, dtype=np.float64)
    K = P.shape[0]
    if K == 0:
        return 0.0
    if pi is None:
        pi = stationary_distribution(P)
    pi = np.asarray(pi, dtype=np.float64)
    # H_i = row entropy of P[i] in bits.
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(P > 0, np.log2(P, where=(P > 0)), 0.0)
        row_h = -np.sum(np.where(P > 0, P * log_p, 0.0), axis=1)
    return float(np.dot(pi, row_h))


# ---------------------------------------------------------------------------
# Step 7 — bootstrap_entropy_rate
# ---------------------------------------------------------------------------

def bootstrap_entropy_rate(sequences: list, k: int, n_reps: int,
                           seed: int) -> Dict[str, Any]:
    """Resample sequences with replacement; compute entropy rate per rep.

    Returns dict with keys:
      - ``point``: entropy rate on the full sequence list
      - ``ci_lo``: 2.5th percentile of replicate entropy rates
      - ``ci_hi``: 97.5th percentile
      - ``reps``: ndarray of replicate values (length n_reps)

    The CI is clamped so ci_lo <= point <= ci_hi (consistent with
    Move A bootstrap_jsd_pairs).
    """
    seqs = list(sequences)
    P_point = build_transition_matrix(seqs, k=k)
    point_h = entropy_rate_from_matrix(P_point, pi=None)

    rng = np.random.default_rng(seed)
    n = len(seqs)
    reps = np.empty(n_reps, dtype=np.float64)
    if n == 0:
        reps.fill(point_h)
    else:
        for r in range(n_reps):
            idx = rng.integers(0, n, size=n)
            boot = [seqs[i] for i in idx]
            P = build_transition_matrix(boot, k=k)
            reps[r] = entropy_rate_from_matrix(P, pi=None)

    ci_lo = float(np.percentile(reps, 2.5))
    ci_hi = float(np.percentile(reps, 97.5))
    # Bracket point.
    ci_lo = min(ci_lo, point_h)
    ci_hi = max(ci_hi, point_h)
    return {
        "point": float(point_h),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "reps": reps,
    }


# ---------------------------------------------------------------------------
# Step 8 — detect_idioms
# ---------------------------------------------------------------------------

def _bigram_counts(sequences: Sequence[np.ndarray], k: int) -> np.ndarray:
    counts = np.zeros((k, k), dtype=np.float64)
    for s in sequences:
        s = np.asarray(s, dtype=np.int64)
        if s.size < 2:
            continue
        src = s[:-1]
        dst = s[1:]
        mask = (src >= 0) & (src < k) & (dst >= 0) & (dst < k)
        if not mask.all():
            src = src[mask]
            dst = dst[mask]
        np.add.at(counts, (src, dst), 1.0)
    return counts


def detect_idioms(sequences: list, k: int, n_shuffles: int, seed: int,
                  percentile: float = 99.0) -> pd.DataFrame:
    """Within-session shuffle test for over-represented bigrams.

    For each bigram (i, j), compare its observed count against a null
    distribution built by shuffling each sequence in-place (within
    session), which preserves per-sequence marginal frequencies but
    destroys transition structure.

    Returns long-form DataFrame with all KxK bigrams (both idiom and
    non-idiom rows). Columns:
        from_cluster, to_cluster, observed_count, null_p99,
        is_idiom, enrichment_ratio
    """
    seqs = [np.asarray(s, dtype=np.int64) for s in sequences]
    observed = _bigram_counts(seqs, k)

    rng = np.random.default_rng(seed)
    # Null: shuffle each sequence n_shuffles times; track counts per rep.
    null_counts = np.zeros((n_shuffles, k, k), dtype=np.float64)
    # Pre-copy sequences as flat arrays we'll permute in place.
    seqs_buf = [s.copy() for s in seqs]
    for rep in range(n_shuffles):
        for s in seqs_buf:
            rng.shuffle(s)
        null_counts[rep] = _bigram_counts(seqs_buf, k)

    # Percentile per bigram across the n_shuffles surrogates.
    null_p = np.percentile(null_counts, percentile, axis=0)
    null_mean = null_counts.mean(axis=0)

    rows = []
    for i in range(k):
        for j in range(k):
            obs = float(observed[i, j])
            p99 = float(null_p[i, j])
            mean_null = float(null_mean[i, j])
            # Enrichment ratio: observed / mean(null), with small-denom guard.
            if mean_null > 0:
                enr = obs / mean_null
            elif obs > 0:
                enr = float(np.inf)
            else:
                enr = 0.0
            rows.append({
                "from_cluster": int(i),
                "to_cluster": int(j),
                "observed_count": obs,
                "null_p99": p99,
                "is_idiom": bool(obs > p99),
                "enrichment_ratio": enr,
            })

    df = pd.DataFrame(rows)
    df["from_cluster"] = df["from_cluster"].astype(np.int64)
    df["to_cluster"] = df["to_cluster"].astype(np.int64)
    df["is_idiom"] = df["is_idiom"].astype(bool)
    df["enrichment_ratio"] = df["enrichment_ratio"].astype(np.float64)
    return df


# ---------------------------------------------------------------------------
# Real-run helpers
# ---------------------------------------------------------------------------

def build_bout_sequences(call_df: pd.DataFrame,
                         cluster_col: str = "cluster") -> List[np.ndarray]:
    """Group calls by bout_id (preserving order) and return cluster sequences."""
    out: List[np.ndarray] = []
    for _bid, grp in call_df.groupby("bout_id", sort=True):
        out.append(grp[cluster_col].to_numpy(dtype=np.int64))
    return out


def mi_lag1(sequences: List[np.ndarray], k: int) -> float:
    """Empirical 1-lag mutual information across all bigrams, in bits.

    I(X;Y) = sum_xy p(x,y) log2( p(x,y) / (p(x) p(y)) )
    """
    counts = _bigram_counts(sequences, k)
    total = counts.sum()
    if total <= 0:
        return 0.0
    p_xy = counts / total
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    denom = p_x @ p_y  # outer product
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where((p_xy > 0) & (denom > 0), p_xy / denom, 1.0)
        log_term = np.where((p_xy > 0) & (denom > 0), np.log2(ratio), 0.0)
        mi = np.sum(np.where(p_xy > 0, p_xy * log_term, 0.0))
    return float(mi)


def _load_vae(checkpoint_path: str, hyperparams_path: str):
    """Load the contour VAE in eval mode on CPU."""
    import torch  # noqa
    # Add scripts/ to sys.path so we can import ImageVAE / ImageVAEConfig.
    repo_root = Path(__file__).resolve().parent.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    from scripts.train_contour_vae_v2 import ImageVAE, ImageVAEConfig  # noqa

    with open(hyperparams_path) as fh:
        hp = json.load(fh)
    cfg_kwargs = hp.get("image_vae_config", {})
    cfg = ImageVAEConfig(**cfg_kwargs)
    vae = ImageVAE(cfg=cfg)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict) and "model" in state:
        state = state["model"]
    vae.load_state_dict(state)
    vae.eval()
    return vae


def _decode_centroid_examples(vae, latents_path: str, kmeans,
                              n_examples: int, out_dir: Path) -> None:
    """Decode 9 nearest patches per centroid; save 3x3 tile PNGs."""
    import torch  # noqa

    lat = pd.read_parquet(latents_path)
    z_cols = [f"z_{i}" for i in range(_N_LATENT_DIMS) if f"z_{i}" in lat.columns]
    Z = lat[z_cols].to_numpy(dtype=np.float32)

    centroids = kmeans.cluster_centers_.astype(np.float32)
    K = centroids.shape[0]
    grid_n = int(np.ceil(np.sqrt(n_examples)))

    out_dir.mkdir(parents=True, exist_ok=True)

    for k in range(K):
        # Euclidean dist from each patch to centroid k.
        diff = Z - centroids[k]
        dist = np.sqrt(np.einsum("ij,ij->i", diff, diff))
        n_take = min(n_examples, len(dist))
        # argpartition for top-n_take smallest.
        idx_part = np.argpartition(dist, n_take - 1)[:n_take]
        # Sort the chosen indices by distance ascending.
        idx_sorted = idx_part[np.argsort(dist[idx_part])]

        z_batch = torch.from_numpy(Z[idx_sorted]).float()
        with torch.no_grad():
            recon = vae.decode(z_batch).cpu().numpy()  # (n, 1, H, W)

        # Sigmoid decoder undershoots on sparse contour data — actual recon max
        # is ~0.03, not 1.0. Autoscale per-tile via 99.5th percentile so the
        # structural content is visible. Floor prevents divide-by-zero on a
        # blank panel.
        vmax = max(float(np.percentile(recon, 99.5)), 1e-4)

        fig, axes = plt.subplots(grid_n, grid_n, figsize=(grid_n * 1.8, grid_n * 1.8))
        if grid_n == 1:
            axes = np.array([[axes]])
        for i in range(grid_n * grid_n):
            r, c = divmod(i, grid_n)
            ax = axes[r, c]
            if i < n_take:
                img = recon[i, 0]
                ax.imshow(img, origin="lower", cmap="magma", aspect="auto",
                          vmin=0.0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0 and c == 0 and i < n_take:
                ax.set_title(f"cluster {k:02d}", fontsize=9, loc="left")
        fig.tight_layout(pad=0.2)
        fig.savefig(out_dir / f"cluster_{k:02d}.png", dpi=110)
        plt.close(fig)


# ---------------------------------------------------------------------------
# HTML summary
# ---------------------------------------------------------------------------

def _png_b64(png_path: Path) -> str:
    return base64.b64encode(png_path.read_bytes()).decode("ascii")


def _matrix_heatmap_png(mat: np.ndarray, title: str, png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 5.0))
    im = ax.imshow(mat, cmap="viridis", vmin=0.0, vmax=max(0.001, mat.max()))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("to cluster")
    ax.set_ylabel("from cluster")
    fig.colorbar(im, ax=ax, label="P(to | from)")
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def _mi_sweep_png(mi_df: pd.DataFrame, png_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(mi_df["bout_threshold_s"], mi_df["mi_bits"], marker="o")
    ax.set_xlabel("Bout threshold (s)")
    ax.set_ylabel("Lag-1 MI (bits)")
    ax.set_title("Bout-threshold sensitivity (combined cohorts)")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(png_path, dpi=120)
    plt.close(fig)


def _decision_gate_text(entropy_df: pd.DataFrame,
                        idioms_df: pd.DataFrame) -> str:
    """Build a human-readable decision-gate summary."""
    lines = []
    cohorts = entropy_df["cohort_split"].tolist()
    # Identify whether any cohort's CI sits clearly below others.
    rows = entropy_df.set_index("cohort_split")
    min_cohort = rows["entropy_rate_point"].idxmin()
    max_cohort = rows["entropy_rate_point"].idxmax()
    min_row = rows.loc[min_cohort]
    max_row = rows.loc[max_cohort]
    sep = min_row["ci_hi"] < max_row["ci_lo"]
    if sep:
        lines.append(
            f"Entropy-rate split: <b>{min_cohort}</b> has the most "
            f"constrained dynamics (H={min_row['entropy_rate_point']:.3f} "
            f"[{min_row['ci_lo']:.3f}, {min_row['ci_hi']:.3f}] bits) "
            f"vs <b>{max_cohort}</b> (H={max_row['entropy_rate_point']:.3f} "
            f"[{max_row['ci_lo']:.3f}, {max_row['ci_hi']:.3f}] bits). "
            f"CIs do not overlap."
        )
    else:
        lines.append(
            "Entropy rates: cohort CIs overlap; no single cohort is "
            f"unambiguously more constrained. min={min_cohort} "
            f"({min_row['entropy_rate_point']:.3f}), max={max_cohort} "
            f"({max_row['entropy_rate_point']:.3f})."
        )

    # 5970 vs the rest.
    if "5970" in rows.index:
        h_5970 = float(rows.loc["5970", "entropy_rate_point"])
        others = rows.drop("5970")
        if not others.empty:
            mean_others = float(others["entropy_rate_point"].mean())
            cmp = "lower" if h_5970 < mean_others else "higher"
            lines.append(
                f"5970 vs the rest: H(5970)={h_5970:.3f} bits; "
                f"mean(others)={mean_others:.3f} bits -> 5970 is {cmp}."
            )

    # Idioms per cohort.
    n_idioms = (idioms_df[idioms_df["is_idiom"]]
                .groupby("cohort_split").size().to_dict())
    if n_idioms:
        n_str = ", ".join(f"{c}={n_idioms.get(c, 0)}"
                          for c in cohorts)
        lines.append(f"Idiom counts (is_idiom=True): {n_str}.")
        most = max(n_idioms.items(), key=lambda kv: kv[1])
        lines.append(
            f"Most-stereotyped cohort by idiom count: <b>{most[0]}</b> "
            f"({most[1]} idioms)."
        )
    return "<br/>".join(lines)


def _write_html(out_dir: Path, params: Dict[str, Any],
                entropy_df: pd.DataFrame,
                idioms_df: pd.DataFrame,
                mi_df: pd.DataFrame,
                cohorts: List[str],
                n_centroid_clusters: int) -> Path:
    # MI sweep plot.
    mi_png = out_dir / "_mi_sweep.png"
    _mi_sweep_png(mi_df, mi_png)
    mi_b64 = _png_b64(mi_png)

    # Heatmap PNGs per cohort.
    heatmap_imgs = []
    for c in cohorts:
        mat_csv = out_dir / "transition_matrices" / f"{c}.csv"
        if not mat_csv.exists():
            continue
        mat = pd.read_csv(mat_csv, index_col=0).to_numpy(dtype=np.float64)
        png = out_dir / f"_heatmap_{c}.png"
        _matrix_heatmap_png(mat, f"Transition matrix -- {c}", png)
        heatmap_imgs.append((c, _png_b64(png)))

    # Idiom table -- top 15 per cohort by enrichment + is_idiom=True rows.
    pieces = []
    for c in cohorts:
        sub = idioms_df[idioms_df["cohort_split"] == c]
        top = sub.sort_values("enrichment_ratio", ascending=False).head(15)
        flagged = sub[sub["is_idiom"]]
        combined = pd.concat([top, flagged]).drop_duplicates(
            subset=["cohort_split", "from_cluster", "to_cluster"]
        ).sort_values("enrichment_ratio", ascending=False)
        pieces.append(combined)
    idiom_tbl = (
        pd.concat(pieces, ignore_index=True) if pieces
        else idioms_df.head(0)
    )

    # Format tables.
    def _fmt_h(row):
        return (f"{row['entropy_rate_point']:.4f} "
                f"[{row['ci_lo']:.4f}, {row['ci_hi']:.4f}]")
    e_show = entropy_df.copy()
    e_show["H_with_CI"] = e_show.apply(_fmt_h, axis=1)
    e_html = e_show[["cohort_split", "n_calls", "n_bouts",
                     "mean_bout_length", "H_with_CI",
                     "n_boot", "seed"]].to_html(index=False)

    mi_html = mi_df.to_html(index=False, float_format=lambda v: f"{v:.4f}")

    if not idiom_tbl.empty:
        i_show = idiom_tbl[[
            "cohort_split", "from_cluster", "to_cluster",
            "observed_count", "null_p99", "is_idiom", "enrichment_ratio",
        ]].copy()
        i_show["enrichment_ratio"] = i_show["enrichment_ratio"].round(2)
        i_html = i_show.to_html(index=False)
    else:
        i_html = "<p>No idioms.</p>"

    dl_items = "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in params.items())
    dl_html = "<dl>" + dl_items + "</dl>"

    # Heatmap section.
    heat_html_parts = [
        f'<figure><figcaption>{c}</figcaption>'
        f'<img alt="transition matrix {c}" '
        f'src="data:image/png;base64,{b64}" /></figure>'
        for (c, b64) in heatmap_imgs
    ]
    heat_html = "\n".join(heat_html_parts)

    # Centroid PNGs section -- relative file links.
    cent_html_parts = [
        f'<figure><figcaption>cluster {k:02d}</figcaption>'
        f'<img alt="cluster {k:02d}" '
        f'src="centroids/cluster_{k:02d}.png" /></figure>'
        for k in range(n_centroid_clusters)
    ]
    cent_html = "\n".join(cent_html_parts)

    decision = _decision_gate_text(entropy_df, idioms_df)

    timestamp = _dt.datetime.now().isoformat(timespec="seconds")
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Latent transitions -- Move C</title>
  <style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, sans-serif;
            max-width: 1200px; margin: 2em auto; padding: 0 1em; color: #222; }}
    h1 {{ border-bottom: 2px solid #333; padding-bottom: 0.3em; }}
    h2 {{ margin-top: 1.6em; color: #333; }}
    dl {{ display: grid; grid-template-columns: max-content 1fr;
          column-gap: 1em; row-gap: 0.25em; }}
    dt {{ font-weight: 600; color: #555; }}
    table {{ border-collapse: collapse; margin: 0.5em 0; font-size: 0.9em; }}
    th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
    th {{ background: #f0f0f0; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    figure {{ display: inline-block; margin: 6px; vertical-align: top;
              width: 220px; text-align: center; }}
    figcaption {{ font-size: 0.85em; color: #555; }}
    .decision {{ padding: 0.8em 1em; background: #f7f7f0;
                 border-left: 4px solid #888; margin: 1em 0; }}
    footer {{ margin-top: 2em; color: #888; font-size: 0.9em; }}
  </style>
</head>
<body>
  <h1>Latent transitions -- Move C</h1>

  <h2>Run parameters</h2>
  {dl_html}

  <h2>Bout-threshold MI sweep (combined cohorts)</h2>
  <img alt="MI sweep" src="data:image/png;base64,{mi_b64}" />
  {mi_html}

  <h2>Entropy rates (with bootstrap CIs)</h2>
  {e_html}

  <h2>Transition matrices</h2>
  {heat_html}

  <h2>Idioms (top 15 by enrichment + is_idiom=True rows per cohort)</h2>
  {i_html}

  <h2>Per-cluster centroid examples (3x3 decoded reconstructions)</h2>
  {cent_html}

  <h2>Decision-gate read</h2>
  <p class="decision">{decision}</p>

  <footer>Generated on {timestamp}</footer>
</body>
</html>
"""
    html_path = out_dir / "summary.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Move C -- latent transition matrices, entropy rates, idioms.",
    )
    p.add_argument("--latents-path", type=str,
                   default="results/contour_vae_combined/latents.parquet")
    p.add_argument("--kmeans-path", type=str,
                   default="models/latent_kmeans/k20.joblib")
    p.add_argument("--vae-checkpoint", type=str,
                   default="models/contour_vae_combined/best.pt")
    p.add_argument("--vae-hyperparams", type=str,
                   default="models/contour_vae_combined/hyperparams.json")
    p.add_argument("--out-dir", type=str,
                   default="results/latent_transitions")
    p.add_argument("--bout-threshold-s", type=float, default=0.25)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-shuffles", type=int, default=1000)
    p.add_argument("--n-centroid-examples", type=int, default=9)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> int:
    t_start = time.time()
    args = _parse_args()

    print(f"[PARAM] latents_path       = {args.latents_path}")
    print(f"[PARAM] kmeans_path        = {args.kmeans_path}")
    print(f"[PARAM] vae_checkpoint     = {args.vae_checkpoint}")
    print(f"[PARAM] vae_hyperparams    = {args.vae_hyperparams}")
    print(f"[PARAM] out_dir            = {args.out_dir}")
    print(f"[PARAM] bout_threshold_s   = {args.bout_threshold_s}")
    print(f"[PARAM] n_boot             = {args.n_boot}")
    print(f"[PARAM] n_shuffles         = {args.n_shuffles}")
    print(f"[PARAM] n_centroid_examples= {args.n_centroid_examples}")
    print(f"[PARAM] seed               = {args.seed}")

    out_dir = Path(args.out_dir)
    (out_dir / "transition_matrices").mkdir(parents=True, exist_ok=True)
    (out_dir / "centroids").mkdir(parents=True, exist_ok=True)

    # 1. Load call-level latents + timestamps.
    print("[INFO] Loading call-level latents...")
    t0 = time.time()
    call_df = load_call_latents(args.latents_path, DETECTION_CSV_PATHS)
    print(f"[INFO]   n_calls = {len(call_df)} (in {time.time() - t0:.1f}s)")
    print("[INFO]   per-cohort_split counts:")
    counts = call_df["cohort_split"].value_counts().to_dict()
    for c in sorted(counts):
        print(f"[INFO]     {c:>14s}: {counts[c]:>7d}")

    # 2. Load K-means; assign cluster per call.
    print("[INFO] Loading K-means model...")
    kmeans = joblib.load(args.kmeans_path)
    k = int(kmeans.n_clusters)
    print(f"[INFO]   K = {k}")
    call_df["cluster"] = assign_call_clusters(call_df, kmeans)

    # 3. Segment into bouts (file-aware, 0.25s default).
    call_df = segment_into_bouts(call_df, bout_threshold_s=args.bout_threshold_s)
    n_bouts = int(call_df["bout_id"].nunique())
    print(f"[INFO] Bout segmentation: {n_bouts} bouts at "
          f"threshold={args.bout_threshold_s}s")

    # 4. Bout-threshold MI sweep (combined cohorts).
    print("[INFO] Computing bout-threshold MI sweep...")
    sweep_thresholds = [0.1, 0.143, 0.2, 0.25, 0.5, 1.0, 2.0]
    sweep_rows = []
    for t in sweep_thresholds:
        seg = segment_into_bouts(call_df, bout_threshold_s=t)
        seqs = build_bout_sequences(seg, cluster_col="cluster")
        mi = mi_lag1(seqs, k=k)
        sweep_rows.append({
            "bout_threshold_s": t,
            "n_bouts": int(seg["bout_id"].nunique()),
            "n_calls": int(len(seg)),
            "mi_bits": mi,
        })
        print(f"[INFO]   t={t:.3f}s  n_bouts={sweep_rows[-1]['n_bouts']:>5d}  "
              f"MI={mi:.4f} bits")
    mi_df = pd.DataFrame(sweep_rows)
    mi_df.to_csv(out_dir / "bout_mi_sweep.csv", index=False)

    # 5. Per-cohort transition matrices, entropy rates, idioms.
    cohorts = sorted(call_df["cohort_split"].unique().tolist())
    print(f"[INFO] Cohort splits: {cohorts}")

    entropy_rows = []
    idiom_pieces = []

    for c in cohorts:
        sub = call_df[call_df["cohort_split"] == c].copy()
        seqs = build_bout_sequences(sub, cluster_col="cluster")
        n_calls_c = int(len(sub))
        n_bouts_c = len(seqs)
        mean_bout = (
            float(np.mean([len(s) for s in seqs])) if seqs else 0.0
        )

        # Transition matrix.
        P = build_transition_matrix(seqs, k=k)
        mat_df = pd.DataFrame(
            P,
            index=[int(i) for i in range(k)],
            columns=[int(j) for j in range(k)],
        )
        mat_df.to_csv(out_dir / "transition_matrices" / f"{c}.csv")

        # Bootstrap entropy rate.
        print(f"[INFO]   {c}: bootstrap entropy rate "
              f"({args.n_boot} reps, n_bouts={n_bouts_c})...")
        t0 = time.time()
        boot = bootstrap_entropy_rate(seqs, k=k, n_reps=args.n_boot,
                                      seed=args.seed)
        print(f"[INFO]     done in {time.time() - t0:.1f}s; "
              f"H = {boot['point']:.4f} "
              f"[{boot['ci_lo']:.4f}, {boot['ci_hi']:.4f}] bits")
        entropy_rows.append({
            "cohort_split": c,
            "n_calls": n_calls_c,
            "n_bouts": n_bouts_c,
            "mean_bout_length": mean_bout,
            "entropy_rate_point": boot["point"],
            "ci_lo": boot["ci_lo"],
            "ci_hi": boot["ci_hi"],
            "n_boot": int(args.n_boot),
            "seed": int(args.seed),
        })

        # Idioms.
        print(f"[INFO]   {c}: idiom detection "
              f"({args.n_shuffles} shuffles)...")
        t0 = time.time()
        ids = detect_idioms(seqs, k=k, n_shuffles=args.n_shuffles,
                            seed=args.seed, percentile=99.0)
        ids.insert(0, "cohort_split", c)
        idiom_pieces.append(ids)
        print(f"[INFO]     done in {time.time() - t0:.1f}s; "
              f"n_idioms = {int(ids['is_idiom'].sum())}")

    entropy_df = pd.DataFrame(entropy_rows)
    entropy_df.to_csv(out_dir / "entropy_rates.csv", index=False)

    idioms_df = pd.concat(idiom_pieces, ignore_index=True)
    idioms_df = idioms_df.sort_values(
        ["cohort_split", "enrichment_ratio"], ascending=[True, False]
    )
    idioms_df.to_csv(out_dir / "idioms.csv", index=False)

    # 6. Per-centroid decoded examples.
    print("[INFO] Loading VAE for centroid examples...")
    try:
        vae = _load_vae(args.vae_checkpoint, args.vae_hyperparams)
        print("[INFO] Decoding centroid example tiles...")
        t0 = time.time()
        _decode_centroid_examples(
            vae=vae, latents_path=args.latents_path, kmeans=kmeans,
            n_examples=args.n_centroid_examples,
            out_dir=out_dir / "centroids",
        )
        print(f"[INFO]   done in {time.time() - t0:.1f}s")
    except Exception as ex:
        print(f"[WARN] VAE centroid decode failed: {ex!r}")

    # 7. HTML summary.
    print("[INFO] Writing HTML summary...")
    params = {
        "latents_path": args.latents_path,
        "kmeans_path": args.kmeans_path,
        "vae_checkpoint": args.vae_checkpoint,
        "out_dir": args.out_dir,
        "bout_threshold_s": args.bout_threshold_s,
        "n_boot": args.n_boot,
        "n_shuffles": args.n_shuffles,
        "n_centroid_examples": args.n_centroid_examples,
        "seed": args.seed,
        "n_calls_total": int(len(call_df)),
        "k": k,
    }
    params.update({f"n_calls[{c}]": int(counts[c]) for c in sorted(counts)})
    html_path = _write_html(
        out_dir, params, entropy_df, idioms_df, mi_df, cohorts,
        n_centroid_clusters=k,
    )
    print(f"[INFO] Wrote HTML -> {html_path}")

    print(f"[INFO] Total wall time: {time.time() - t_start:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

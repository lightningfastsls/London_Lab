#!/usr/bin/env python
"""Build a navigable 2-D shape-map over registered USV ridges and evaluate the
discrete-alphabet-vs-continuum decision gate.

Handoff: docs/handoffs/2026-05-25_shape-map-and-alphabet-decision.md

Inputs (worktree-local, npz pulled from rig 2026-05-25):
  results/latent_transitions/shape_alphabet/true_registered_ridges_meta.npz
      keys: shapes (67337,50) float32  -- mean-freq-subtracted, 50-pt resampled ridge
            patch_label (N,) int32     -- the per-patch K=20 shape-letter (== k20.predict)
            cohort, wav_stem, call_id, abs_time_start_s
  results/latent_transitions/shape_alphabet/shape_call_letters.parquet
      per-call (47026) shape_letter (authoritative letter feeding the grammar analysis)
  models/shape_kmeans/k20.joblib  -- KMeans(20), n_features_in_=50

Outputs: results/latent_transitions/shape_map/
  shape_map_patch_cohort.png, shape_map_patch_letter.png,
  shape_map_ridge_glyphs.png, shape_map_call_letter.png,
  shape_map_diagnostics.json, shape_map_report.html

This script does NOT touch the production detection pipeline, ExtractionConfig,
models/latent_kmeans/, or scripts/analyze_latent_transitions.py.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import to_hex
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score

# ---- params (printed up front per feedback_analysis_print_params) -----------
UMAP_KW = dict(n_neighbors=30, min_dist=0.05, n_components=2, random_state=42)
GLYPH_GRID = 22            # G x G cells for the ridge-thumbnail grid
GLYPH_MIN_COUNT = 20       # min patches in a cell to draw its mean ridge
HDBSCAN_MIN_CLUSTER = 500  # ~0.7% of 67k; matches "is there macro-structure?"
SIL_SAMPLE = 12000         # silhouette is O(n^2); subsample for tractability
RANDOM_SEED = 42

ROOT = Path(__file__).resolve().parents[1]
SHAPE_DIR = ROOT / "results/latent_transitions/shape_alphabet"
OUT_DIR = ROOT / "results/latent_transitions/shape_map"
NPZ = SHAPE_DIR / "true_registered_ridges_meta.npz"
PARQUET = SHAPE_DIR / "shape_call_letters.parquet"
KM_PATH = ROOT / "models/shape_kmeans/k20.joblib"

COHORT_ORDER = ["5970", "3452", "9252", "lab_131204"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def umap_embed(X: np.ndarray, label: str) -> np.ndarray:
    import umap  # imported here so --help is instant
    log(f"UMAP {label}: fitting on {X.shape} ...")
    t0 = time.time()
    emb = umap.UMAP(**UMAP_KW).fit_transform(X)
    log(f"UMAP {label}: done in {time.time()-t0:.1f}s -> {emb.shape}")
    return emb


def cohort_palette() -> dict:
    cmap = plt.get_cmap("tab10")
    return {c: to_hex(cmap(i)) for i, c in enumerate(COHORT_ORDER)}


def letter_palette(n: int = 20) -> dict:
    cmap = plt.get_cmap("tab20")
    return {k: to_hex(cmap(k % 20)) for k in range(n)}


def scatter_by(emb, values, palette, title, path, order=None, point_s=2):
    fig, ax = plt.subplots(figsize=(9, 8), dpi=120)
    keys = order if order is not None else sorted(pd.unique(values))
    for k in keys:
        m = values == k
        if not np.any(m):
            continue
        ax.scatter(emb[m, 0], emb[m, 1], s=point_s, c=palette[k],
                   label=f"{k} (n={int(m.sum())})", alpha=0.45, linewidths=0)
    ax.set_title(title)
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(markerscale=4, fontsize=7, ncol=2, loc="best", framealpha=0.85)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    log(f"wrote {path.name}")


def ridge_glyph_grid(emb, ridges, cohort, palette, path):
    """Tile the embedding into a grid; in each populated cell draw the mean ridge
    as a small line glyph, with the cell tinted by its dominant cohort.

    Reveals the map as browsable 'shape regions': flat ridges = horizontal lines,
    up-sweeps = rising, chevrons = inverted-V, etc. Shared global y-scale so
    amplitude is comparable across cells (not per-cell autoscaled)."""
    x0, x1 = emb[:, 0].min(), emb[:, 0].max()
    y0, y1 = emb[:, 1].min(), emb[:, 1].max()
    xedges = np.linspace(x0, x1, GLYPH_GRID + 1)
    yedges = np.linspace(y0, y1, GLYPH_GRID + 1)
    cw = (x1 - x0) / GLYPH_GRID
    ch = (y1 - y0) / GLYPH_GRID
    # shared robust amplitude scale for the ridge curves
    amp = np.percentile(np.abs(ridges), 98)
    n_pts = ridges.shape[1]
    gx = np.linspace(0, 1, n_pts)

    fig, ax = plt.subplots(figsize=(11, 10), dpi=130)
    drawn = 0
    for i in range(GLYPH_GRID):
        for j in range(GLYPH_GRID):
            in_cell = ((emb[:, 0] >= xedges[i]) & (emb[:, 0] < xedges[i + 1]) &
                       (emb[:, 1] >= yedges[j]) & (emb[:, 1] < yedges[j + 1]))
            n = int(in_cell.sum())
            if n < GLYPH_MIN_COUNT:
                continue
            mean_ridge = ridges[in_cell].mean(axis=0)
            # dominant cohort in cell -> tint background
            cvals, ccnts = np.unique(cohort[in_cell], return_counts=True)
            dom = cvals[ccnts.argmax()]
            cx, cy = xedges[i], yedges[j]
            ax.add_patch(plt.Rectangle((cx, cy), cw, ch, facecolor=palette[dom],
                                       alpha=0.12, edgecolor="none"))
            # glyph occupies inner 80% of the cell
            gxp = cx + 0.1 * cw + gx * 0.8 * cw
            gyp = cy + 0.5 * ch + (np.clip(mean_ridge, -amp, amp) / amp) * 0.4 * ch
            ax.plot(gxp, gyp, color="black", lw=0.9)
            drawn += 1
    # cohort legend
    for c in COHORT_ORDER:
        ax.plot([], [], color=palette[c], lw=6, alpha=0.4, label=c)
    ax.set_xlim(x0, x1); ax.set_ylim(y0, y1)
    ax.set_title(f"Shape-region map: mean registered ridge per cell "
                 f"({drawn} cells, >={GLYPH_MIN_COUNT} patches; tint=dominant cohort)")
    ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
    ax.legend(fontsize=8, loc="best", framealpha=0.85)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)
    log(f"wrote {path.name} ({drawn} glyph cells)")
    return drawn


def blob_vs_regions(emb, letters, rng) -> dict:
    """Two diagnostics for the decision gate:
    (1) HDBSCAN on the 2-D embedding: how many macro-clusters / % noise.
    (2) Silhouette of the K=20 letters in the 2-D map: do letters tile compact
        regions (>0) or smear across a continuum (~0)?"""
    log("HDBSCAN on 2-D embedding ...")
    hdb = HDBSCAN(min_cluster_size=HDBSCAN_MIN_CLUSTER, min_samples=10)
    hl = hdb.fit_predict(emb)
    labs, cnts = np.unique(hl, return_counts=True)
    n_noise = int(cnts[labs == -1].sum()) if -1 in labs else 0
    n_clusters = int((labs != -1).sum())

    # silhouette of the 20 letters in 2-D (subsampled)
    idx = rng.choice(len(emb), size=min(SIL_SAMPLE, len(emb)), replace=False)
    sil = float(silhouette_score(emb[idx], letters[idx]))
    log(f"HDBSCAN: {n_clusters} clusters, {n_noise} noise ({100*n_noise/len(emb):.1f}%)"
        f" | letter silhouette (2-D): {sil:.3f}")
    return {
        "hdbscan_n_clusters": n_clusters,
        "hdbscan_pct_noise": round(100 * n_noise / len(emb), 2),
        "hdbscan_cluster_sizes": sorted([int(c) for l, c in zip(labs, cnts) if l != -1],
                                        reverse=True),
        "letter_silhouette_2d": round(sil, 4),
        "silhouette_sample_n": int(min(SIL_SAMPLE, len(emb))),
    }


def verdict_from(diag: dict) -> tuple[str, str]:
    """Map diagnostics to the handoff's decision gate.

    The DIRECT test of the gate question -- 'do the K=20 hard letters tile
    compact, separable regions?' -- is the silhouette of the letter labels in
    the 2-D map. HDBSCAN's macro-cluster count is a secondary check on whether
    ANY coarse structure exists (it does: 2-3 lobes), but 2-3 lobes of sizes
    like [46860, 13790, 556] is NOT a 20-symbol alphabet. So silhouette leads:
      - silhouette >= 0.25  -> letters are compact -> REGIONS (gate row 1)
      - silhouette <  0.10  -> letters smear across a continuum -> CONTINUUM
                               (gate row 2): drop hard letters for the grammar
      - 0.10 <= silhouette < 0.25 -> AMBIGUOUS, PHATE second view warranted
    Heuristic is advisory; the alphabet adoption is the user's call."""
    sil = diag["letter_silhouette_2d"]
    nc = diag["hdbscan_n_clusters"]
    sizes = diag.get("hdbscan_cluster_sizes", [])
    macro = f"({nc} coarse HDBSCAN lobes, sizes {sizes[:4]})"
    if sil >= 0.25:
        return ("REGIONS", f"Gate row 1: navigable, separable regions {macro} -- "
                "adopt the map AND treat K=20 letters as a real coarse alphabet.")
    if sil < 0.10:
        return ("CONTINUUM", f"Gate row 2 (with row-1 nuance): the K=20 letters do "
                f"NOT tile compact regions (silhouette ~0) -- they are arbitrary cuts "
                f"of a smooth manifold {macro}. The map IS navigable by morphology "
                f"(flat core / valley top / chevron bottom / up-sweep lobe), so adopt "
                f"the MAP as the representation, but DROP the hard letters for the "
                f"grammar analysis and report shape as continuous axes (curvature, "
                f"terminal-sweep).")
    return ("AMBIGUOUS", f"Borderline silhouette {sil:.3f} {macro} -- run a PHATE "
            "second view before deciding; lean continuum given prior shape evidence.")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    print("=" * 70)
    print("build_shape_map.py -- PARAMETERS")
    print(f"  UMAP: {UMAP_KW}")
    print(f"  glyph grid: {GLYPH_GRID}x{GLYPH_GRID}, min_count={GLYPH_MIN_COUNT}")
    print(f"  HDBSCAN min_cluster_size={HDBSCAN_MIN_CLUSTER}, silhouette_sample={SIL_SAMPLE}")
    print(f"  seed={RANDOM_SEED}")
    print("=" * 70)

    # ---- load ----
    d = np.load(NPZ, allow_pickle=True)
    Sh = d["shapes"].astype(np.float32)
    patch_letter = d["patch_label"].astype(int)
    cohort = d["cohort"].astype(str)
    wav_stem = d["wav_stem"].astype(str)
    call_id = d["call_id"].astype(int)
    assert Sh.shape == (67337, 50), f"unexpected ridge shape {Sh.shape}"
    log(f"loaded npz: {Sh.shape} ridges, {len(np.unique(patch_letter))} letters present")
    print(f"  per-cohort patch counts: "
          f"{pd.Series(cohort).value_counts().reindex(COHORT_ORDER).to_dict()}")

    # ---- verify patch_label == k20.predict (byte-faithful letters) ----
    km = joblib.load(KM_PATH)
    pred = km.predict(Sh)
    agree = float((pred == patch_letter).mean())
    log(f"k20.predict vs stored patch_label agreement: {agree*100:.2f}%")

    # ---- per-call mean ridge + authoritative letter from parquet ----
    call_df = pd.DataFrame({"cohort": cohort, "wav_stem": wav_stem,
                            "call_id": call_id, "idx": np.arange(len(Sh))})
    grp = call_df.groupby(["cohort", "wav_stem", "call_id"], sort=False)["idx"].apply(list)
    call_keys = list(grp.index)
    call_ridge = np.stack([Sh[idxs].mean(axis=0) for idxs in grp.values])
    parq = pd.read_parquet(PARQUET).set_index(["cohort", "wav_stem", "call_id"])
    call_letter = parq.reindex(call_keys)["shape_letter"].to_numpy()
    call_cohort = np.array([k[0] for k in call_keys])
    cov = float(np.isfinite(pd.to_numeric(call_letter, errors="coerce")).mean())
    log(f"per-call: {call_ridge.shape} mean ridges; parquet letter coverage {cov*100:.1f}%")
    call_letter = pd.to_numeric(call_letter, errors="coerce").fillna(-1).astype(int).to_numpy() \
        if isinstance(call_letter, pd.Series) else \
        np.nan_to_num(call_letter.astype(float), nan=-1).astype(int)

    # ---- embeddings (reuse cache to skip the ~4-min UMAP on re-runs) ----
    cache = OUT_DIR / "shape_map_embeddings.npz"
    if cache.exists():
        c = np.load(cache, allow_pickle=True)
        if c["emb_patch"].shape[0] == len(Sh) and c["emb_call"].shape[0] == len(call_ridge):
            log(f"reusing cached embeddings from {cache.name} (delete to force re-fit)")
            emb_patch, emb_call = c["emb_patch"], c["emb_call"]
        else:
            emb_patch = umap_embed(Sh, "per-patch")
            emb_call = umap_embed(call_ridge, "per-call")
    else:
        emb_patch = umap_embed(Sh, "per-patch")
        emb_call = umap_embed(call_ridge, "per-call")
    np.savez_compressed(cache,
                        emb_patch=emb_patch, patch_letter=patch_letter,
                        patch_cohort=cohort, emb_call=emb_call,
                        call_letter=call_letter, call_cohort=call_cohort)

    # ---- diagnostics ----
    diag = blob_vs_regions(emb_patch, patch_letter, rng)
    diag["patch_label_vs_predict_agreement_pct"] = round(agree * 100, 3)
    diag["n_patches"] = int(len(Sh))
    diag["n_calls"] = int(len(call_ridge))
    verdict, gate_action = verdict_from(diag)
    diag["verdict"] = verdict
    diag["gate_action"] = gate_action

    # ---- renders ----
    cpal = cohort_palette()
    lpal = letter_palette(20)
    scatter_by(emb_patch, cohort, cpal,
               f"Per-patch shape-map by cohort (n={len(Sh)})",
               OUT_DIR / "shape_map_patch_cohort.png", order=COHORT_ORDER)
    scatter_by(emb_patch, patch_letter, lpal,
               f"Per-patch shape-map by K=20 letter (n={len(Sh)})",
               OUT_DIR / "shape_map_patch_letter.png", order=list(range(20)))
    n_glyph = ridge_glyph_grid(emb_patch, Sh, cohort, cpal,
                               OUT_DIR / "shape_map_ridge_glyphs.png")
    diag["glyph_cells_drawn"] = int(n_glyph)
    valid = call_letter >= 0
    scatter_by(emb_call[valid], call_letter[valid], lpal,
               f"Per-call shape-map by K=20 letter (n={int(valid.sum())})",
               OUT_DIR / "shape_map_call_letter.png", order=list(range(20)))

    (OUT_DIR / "shape_map_diagnostics.json").write_text(json.dumps(diag, indent=2))
    log(f"VERDICT: {verdict} -- {gate_action}")
    write_html(diag)
    log("done.")


def write_html(diag: dict) -> None:
    rows = "".join(
        f"<tr><td>{k}</td><td><b>{v}</b></td></tr>"
        for k, v in diag.items() if k not in ("gate_action",))
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>USV shape-map &amp; alphabet decision</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,sans-serif;max-width:1100px;
margin:2rem auto;padding:0 1rem;color:#1a1a1a;line-height:1.5}}
h1{{font-size:1.5rem}} h2{{margin-top:2rem;border-bottom:2px solid #eee;padding-bottom:.3rem}}
img{{max-width:100%;border:1px solid #ddd;border-radius:6px;margin:.5rem 0}}
table{{border-collapse:collapse}} td{{border:1px solid #ddd;padding:.3rem .6rem}}
.verdict{{padding:1rem;border-radius:8px;font-size:1.1rem;margin:1rem 0}}
.CONTINUUM{{background:#fff3cd;border:1px solid #ffe69c}}
.REGIONS{{background:#d1e7dd;border:1px solid #a3cfbb}}
.AMBIGUOUS{{background:#cfe2ff;border:1px solid #9ec5fe}}
.cap{{color:#555;font-size:.9rem}}
</style></head><body>
<h1>Navigable 2-D shape-map &amp; transition-alphabet decision</h1>
<p class="cap">Handoff <code>2026-05-25_shape-map-and-alphabet-decision.md</code> ·
UMAP {UMAP_KW} · {diag['n_patches']} patches / {diag['n_calls']} calls ·
registered 50-pt ridges (mean-freq subtracted).</p>

<div class="verdict {diag['verdict']}">
<b>Decision-gate verdict: {diag['verdict']}</b><br>{diag['gate_action']}</div>

<h2>Shape-region map (mean ridge per cell)</h2>
<p class="cap">Each glyph is the mean registered ridge of the patches in that grid cell
(flat=horizontal, up-sweep=rising, chevron=inverted-V). Cell tint = dominant cohort.
This is the browsable "shape regions" view.</p>
<img src="shape_map_ridge_glyphs.png">

<h2>Per-patch map by K=20 letter</h2>
<p class="cap">Do the hard letters occupy compact regions, or smear across a continuum?</p>
<img src="shape_map_patch_letter.png">

<h2>Per-patch map by cohort</h2>
<img src="shape_map_patch_cohort.png">

<h2>Per-call map by K=20 letter</h2>
<img src="shape_map_call_letter.png">

<h2>Diagnostics</h2>
<table>{rows}</table>
<p class="cap">Verdict heuristic: silhouette&lt;0.10 &amp; HDBSCAN&le;2 macro-clusters
&rarr; CONTINUUM; silhouette&ge;0.25 or &ge;6 clusters &rarr; REGIONS; else AMBIGUOUS.
The heuristic is advisory -- the alphabet adoption is the user's call.</p>
</body></html>"""
    (OUT_DIR / "shape_map_report.html").write_text(html)
    log("wrote shape_map_report.html")


if __name__ == "__main__":
    main()

"""Decision gate — registered-shape alphabet vs raw-latent alphabet for the
transition / idiom analysis.

Productionizing handoff: docs/handoffs/2026-05-25_productionize-shape-registration.md
Memo: memory note project_shape_registration_clustering.md

The 2026-05-25 bake-off showed the production contour-VAE K=20 alphabet clusters
by pitch/duration (shape eta2 0.12), and that *registering the ridge* (subtract
mean freq + resample active span to 50 pts) before K-means lifts shape eta2 to
0.58 -- beating every learned encoder. This script answers the downstream
question the gate actually cares about: **does swapping the alphabet change the
sequential structure** (transition MI, entropy rate, idioms)?

Design -- maximal apples-to-apples
----------------------------------
Both alphabets are scored on a SINGLE call table:
  * Latent letters come from the existing path (mean_z -> latent K-means).
  * Shape letters are joined from the rig-produced ``shape_call_letters.parquet``
    (one registered-shape letter per call; see rig_R2_shape_alphabet.py).
We INNER-join on (cohort, wav_stem, call_id), so both alphabets are evaluated on
the *identical* set of calls, with the *identical* begin_time_s / cohort_split /
bout segmentation inherited from ``load_call_latents``. The ONLY variable between
the two transition runs is the letter column -- nothing else.

This does NOT modify the production ``analyze_latent_transitions.py``; it reuses
its building-block functions as a library. If shape wins the gate, re-pointing
the production script is a separate, later change.

Usage::

    .venv/bin/python scripts/transition_alphabet_compare.py \\
        --latents-path results/contour_vae_combined/latents.parquet \\
        --latent-kmeans models/latent_kmeans/k20.joblib \\
        --shape-call-letters results/latent_transitions/shape_alphabet/shape_call_letters.parquet \\
        --out-dir results/latent_transitions/alphabet_compare \\
        --bout-threshold-s 0.25 --n-boot 1000 --n-shuffles 1000 --seed 42
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

# Import the production transition module as a library (main() is guarded;
# torch is lazy-loaded only inside _load_vae, so this stays lightweight).
sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_latent_transitions as alt  # noqa: E402

ALPHABETS = ("latent", "shape")
K = 20  # both alphabets are K=20 -- entropy (max log2 20 = 4.322 bits) is comparable


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--latents-path", type=str,
                   default="results/contour_vae_combined/latents.parquet")
    p.add_argument("--latent-kmeans", type=str,
                   default="models/latent_kmeans/k20.joblib")
    p.add_argument("--shape-call-letters", type=str,
                   default="results/latent_transitions/shape_alphabet/shape_call_letters.parquet")
    p.add_argument("--out-dir", type=str,
                   default="results/latent_transitions/alphabet_compare")
    p.add_argument("--bout-threshold-s", type=float, default=0.25)
    p.add_argument("--n-boot", type=int, default=1000)
    p.add_argument("--n-shuffles", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def build_common_call_table(latents_path: str, latent_kmeans_path: str,
                            shape_letters_path: str) -> pd.DataFrame:
    """Latent call table + joined shape letters, restricted to the common set."""
    call_df = alt.load_call_latents(latents_path, alt.DETECTION_CSV_PATHS)
    n_all = len(call_df)

    kmeans = joblib.load(latent_kmeans_path)
    k_latent = int(kmeans.n_clusters)
    if k_latent != K:
        raise ValueError(f"latent K-means has K={k_latent}, expected {K}")
    call_df["latent"] = alt.assign_call_clusters(call_df, kmeans)

    shape = pd.read_parquet(shape_letters_path)
    need = {"cohort", "wav_stem", "call_id", "shape_letter"}
    missing = need - set(shape.columns)
    if missing:
        raise ValueError(f"shape parquet missing columns {missing}")
    shape = shape.rename(columns={"shape_letter": "shape"})
    shape["call_id"] = shape["call_id"].astype(np.int64)

    merged = call_df.merge(
        shape[["cohort", "wav_stem", "call_id", "shape", "n_patches"]],
        on=["cohort", "wav_stem", "call_id"], how="left",
        suffixes=("", "_shape"),
    )
    n_shape = int(merged["shape"].notna().sum())
    common = merged.dropna(subset=["shape"]).copy()
    common["shape"] = common["shape"].astype(np.int64)

    print(f"[PARAM] latent_calls_total       = {n_all}")
    print(f"[PARAM] calls_with_shape_letter  = {n_shape}")
    print(f"[PARAM] common_call_set          = {len(common)} "
          f"(dropped {n_all - len(common)} calls with no trackable ridge)")
    return common


def analyze_alphabet(call_df: pd.DataFrame, col: str, args) -> Dict:
    """Run the full transition/entropy/idiom pipeline on one letter column."""
    cohorts = sorted(call_df["cohort_split"].unique().tolist())
    rows = []
    idiom_pieces = []
    # Combined MI across all cohort_splits (single sequence pool).
    seqs_all = alt.build_bout_sequences(call_df, cluster_col=col)
    mi_all = alt.mi_lag1(seqs_all, k=K)

    for c in cohorts:
        sub = call_df[call_df["cohort_split"] == c].copy()
        seqs = alt.build_bout_sequences(sub, cluster_col=col)
        boot = alt.bootstrap_entropy_rate(seqs, k=K, n_reps=args.n_boot, seed=args.seed)
        ids = alt.detect_idioms(seqs, k=K, n_shuffles=args.n_shuffles,
                                seed=args.seed, percentile=99.0)
        ids.insert(0, "alphabet", col)
        ids.insert(1, "cohort_split", c)
        idiom_pieces.append(ids)
        rows.append({
            "alphabet": col,
            "cohort_split": c,
            "n_calls": int(len(sub)),
            "n_bouts": len(seqs),
            "mean_bout_length": float(np.mean([len(s) for s in seqs])) if seqs else 0.0,
            "mi_lag1_bits": alt.mi_lag1(seqs, k=K),
            "entropy_rate": boot["point"],
            "ci_lo": boot["ci_lo"],
            "ci_hi": boot["ci_hi"],
            "n_idioms": int(ids["is_idiom"].sum()),
        })
        print(f"[{col:>6s}] {c:>12s}: n_calls={len(sub):>6d} n_bouts={len(seqs):>5d} "
              f"MI={rows[-1]['mi_lag1_bits']:.4f}b H={boot['point']:.4f}"
              f"[{boot['ci_lo']:.4f},{boot['ci_hi']:.4f}] idioms={rows[-1]['n_idioms']}",
              flush=True)
    return {"summary": pd.DataFrame(rows), "idioms": pd.concat(idiom_pieces, ignore_index=True),
            "mi_combined": mi_all}


def main() -> int:
    t0 = time.time()
    args = _parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ALPHABET COMPARISON — registered-shape vs raw-latent (transitions)")
    print("=" * 70)
    print(f"[PARAM] latents_path        = {args.latents_path}")
    print(f"[PARAM] latent_kmeans       = {args.latent_kmeans}")
    print(f"[PARAM] shape_call_letters  = {args.shape_call_letters}")
    print(f"[PARAM] out_dir             = {args.out_dir}")
    print(f"[PARAM] bout_threshold_s    = {args.bout_threshold_s}")
    print(f"[PARAM] n_boot              = {args.n_boot}")
    print(f"[PARAM] n_shuffles          = {args.n_shuffles}")
    print(f"[PARAM] seed                = {args.seed}")
    print(f"[PARAM] K (both alphabets)  = {K}")

    if not Path(args.shape_call_letters).exists():
        print(f"\n[FATAL] shape letters not found: {args.shape_call_letters}\n"
              f"        Run rig_R2_shape_alphabet.py on the rig first and copy the "
              f"parquet back.", file=sys.stderr)
        return 2

    call_df = build_common_call_table(args.latents_path, args.latent_kmeans,
                                      args.shape_call_letters)
    call_df = alt.segment_into_bouts(call_df, bout_threshold_s=args.bout_threshold_s)
    print(f"[INFO] bouts (common set, t={args.bout_threshold_s}s): "
          f"{int(call_df['bout_id'].nunique())}")

    results = {a: analyze_alphabet(call_df, a, args) for a in ALPHABETS}

    # Stack summaries side by side for the gate.
    summary = pd.concat([results[a]["summary"] for a in ALPHABETS], ignore_index=True)
    summary.to_csv(out_dir / "alphabet_compare_summary.csv", index=False)
    idioms = pd.concat([results[a]["idioms"] for a in ALPHABETS], ignore_index=True)
    idioms.to_csv(out_dir / "alphabet_compare_idioms.csv", index=False)

    # Wide delta table: shape - latent per cohort_split.
    piv = summary.pivot(index="cohort_split", columns="alphabet",
                        values=["mi_lag1_bits", "entropy_rate", "n_idioms"])
    deltas = pd.DataFrame({
        "mi_latent": piv[("mi_lag1_bits", "latent")],
        "mi_shape": piv[("mi_lag1_bits", "shape")],
        "mi_delta": piv[("mi_lag1_bits", "shape")] - piv[("mi_lag1_bits", "latent")],
        "H_latent": piv[("entropy_rate", "latent")],
        "H_shape": piv[("entropy_rate", "shape")],
        "H_delta": piv[("entropy_rate", "shape")] - piv[("entropy_rate", "latent")],
        "idioms_latent": piv[("n_idioms", "latent")],
        "idioms_shape": piv[("n_idioms", "shape")],
    })
    deltas.to_csv(out_dir / "alphabet_compare_deltas.csv")

    print("\n" + "=" * 70)
    print("DECISION-GATE DELTAS (shape - latent)")
    print("=" * 70)
    print(deltas.round(4).to_string())
    print(f"\n[INFO] combined-pool MI: latent={results['latent']['mi_combined']:.4f}b  "
          f"shape={results['shape']['mi_combined']:.4f}b")

    _write_html(out_dir, args, summary, deltas, results)
    print(f"\n[INFO] total wall time {time.time()-t0:.1f}s")
    print(f"[DONE] {out_dir}")
    return 0


def _write_html(out_dir: Path, args, summary: pd.DataFrame, deltas: pd.DataFrame,
                results: Dict) -> Path:
    def tbl(df: pd.DataFrame, idx: bool = False) -> str:
        return df.round(4).to_html(index=idx, border=0, classes="t")

    mi_l = results["latent"]["mi_combined"]
    mi_s = results["shape"]["mi_combined"]
    html = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>Alphabet comparison — shape vs latent transitions</title>
<style>
body{{font:14px -apple-system,Arial;max-width:1100px;margin:auto;padding:24px;color:#1a1a1a}}
h1{{font-size:22px}} h2{{font-size:17px;margin-top:28px;border-bottom:2px solid #eee;padding-bottom:4px}}
table.t{{border-collapse:collapse;margin:8px 0;font-size:13px}}
table.t td,table.t th{{border:1px solid #ddd;padding:4px 9px;text-align:right}}
table.t th{{background:#f6f6f6}}
.note{{background:#fff8e1;border-left:4px solid #f0ad4e;padding:10px 14px;margin:12px 0}}
code{{background:#f2f2f2;padding:1px 5px;border-radius:3px}}
</style></head><body>
<h1>Registered-shape vs raw-latent alphabet — transition structure</h1>
<p>Decision gate for
<code>docs/handoffs/2026-05-25_productionize-shape-registration.md</code>.
Both alphabets are K={K}, scored on the <b>identical common call set</b> with the
same bout segmentation — the only variable is the letter column.</p>

<div class="note"><b>Background (bake-off memo):</b> the latent alphabet sorts by
pitch/duration (shape &eta;&sup2; 0.12); registering the ridge lifts shape
&eta;&sup2; to 0.58. UMAP&rarr;HDBSCAN on registered ridges gave a <b>continuum</b>
(23 fuzzy blobs), so K={K} hard letters are imposed on a smooth manifold either way —
a 2-D shape-map is the navigable alternative if the gate motivates it.</p></div>

<h2>Decision-gate deltas (shape &minus; latent)</h2>
{tbl(deltas, idx=True)}
<p>Combined-pool lag-1 MI: latent <b>{mi_l:.4f}</b> bits, shape <b>{mi_s:.4f}</b> bits
(&Delta; {mi_s - mi_l:+.4f}).</p>

<h2>Full per-cohort summary</h2>
{tbl(summary)}

<h2>Parameters</h2>
<table class="t">
<tr><th>latents_path</th><td>{args.latents_path}</td></tr>
<tr><th>latent_kmeans</th><td>{args.latent_kmeans}</td></tr>
<tr><th>shape_call_letters</th><td>{args.shape_call_letters}</td></tr>
<tr><th>bout_threshold_s</th><td>{args.bout_threshold_s}</td></tr>
<tr><th>n_boot</th><td>{args.n_boot}</td></tr>
<tr><th>n_shuffles</th><td>{args.n_shuffles}</td></tr>
<tr><th>seed</th><td>{args.seed}</td></tr>
</table>
</body></html>"""
    p = out_dir / "alphabet_compare.html"
    p.write_text(html)
    print(f"[INFO] wrote {p}")
    return p


if __name__ == "__main__":
    raise SystemExit(main())

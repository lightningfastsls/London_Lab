"""Evaluation driver for Pathway A — derivative-loss contour-VAE.

Spec: docs/handoffs/2026-05-27_shape-vae-A-derivative-loss.md  (Eval gates §)

Scores a trained shape VAE's latents against the gates the handoff defines, on a
held-out split, with full PRINT DISCIPLINE (params, thresholds, sort keys, row
counts). The head-to-head reference points:

    registration ceiling   shape η² ≈ 0.58–0.75   (the baseline to beat)
    production contour-VAE  shape η² ≈ 0.12        (a pitch/duration sorter)
    denoised dead-end       shape η² ≈ 0.081       (no derivative term)

GATES
  1. shape η²            — must clear 0.12 DECISIVELY; target ≥ 0.50; stretch ≥ 0.58.
                           Below 0.12 ⇒ KILL (registration is the answer).
  2. pitch & duration η² — must DROP vs the dead-end's 0.527 / 0.404 (the direct
                           test the derivative term moved the latent off pitch).
  3. geometric-type NMI  — vs syllable_type; must beat production VAE's 0.04
                           (target > 0.20). Verify the label column exists first.
  4. jump capture        — do frequency-jump / multi-component calls form their
                           own latent neighbourhood? (the registration-can't payoff)

METRIC LINEAGE
  ``eta2`` and ``register_one`` are reproduced VERBATIM from the rig scripts
  (rig_M8_contour_vae.py:77 and rig_R2_shape_alphabet.py:47). They are copied —
  not imported — because those scripts run ``OUT.mkdir('/data/shachar/…')`` at
  module import time, which raises PermissionError off the rig and would make
  this driver unimportable (and untestable). The copies are byte-faithful to the
  rig behaviour; only the import-time side effect is dropped.

Designed to run on the rig (needs the trained model + denoised patches).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
for _p in (_REPO_ROOT, _SRC_ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


# ===========================================================================
# Metric helpers — VERBATIM copies (see MODULE DOCSTRING for why not imported)
# ===========================================================================

# rig_R2_shape_alphabet.py constants (register_one resample / active-col floor).
N_RESAMPLE = 50
MIN_ACTIVE_COLS = 6


def eta2(v, lab):
    """Between-group variance fraction (1 - within/total).

    VERBATIM from scripts/experiments/rig_M8_contour_vae.py:77-82, with an added
    empty-input guard (returns 0.0 when every row is filtered out as noise).
    """
    v = v if v.ndim == 2 else v[:, None]
    keep = lab >= 0
    v, lab = v[keep], lab[keep]
    if len(v) == 0:
        return 0.0
    g = v.mean(0)
    tot = float(((v - g) ** 2).sum())
    w = sum(
        float(((v[lab == l] - v[lab == l].mean(0)) ** 2).sum())
        for l in np.unique(lab)
    )
    return 1 - w / tot if tot > 0 else 0.0


def register_one(crop, freqs_khz):
    """Registered shape vector (kill pitch via mean-subtract; kill duration via
    resample to N_RESAMPLE points) or None if too few active columns.

    VERBATIM from scripts/experiments/rig_R2_shape_alphabet.py:47-64. Imported
    lazily so this module loads even where ridge_tracker is unavailable.
    """
    from usv_spectrogram.features.ridge_tracker import track_ridge, RidgeConfig

    thr = max(1e-9, 0.02 * float(crop.max()))
    cfg = RidgeConfig(transition_penalty=0.1, max_jump_bins=10, silence_threshold=thr)
    fm, am = track_ridge(crop, freqs_khz.astype(float), cfg)
    active = np.isfinite(fm)
    if active.sum() < MIN_ACTIVE_COLS:
        return None
    idx = np.where(active)[0]
    lo, hi = idx[0], idx[-1]
    span = fm[lo:hi + 1].copy()
    nanm = ~np.isfinite(span)
    if nanm.any():
        good = np.where(~nanm)[0]
        span[nanm] = np.interp(np.where(nanm)[0], good, span[good])
    pitch = float(span.mean())
    sc = span - pitch
    shape = np.interp(
        np.linspace(0, 1, N_RESAMPLE), np.linspace(0, 1, len(sc)), sc
    )
    return shape.astype(np.float32)


# ===========================================================================
# Scoring
# ===========================================================================


def score_gates(
    latents: np.ndarray,
    shape_labels: np.ndarray,
    pitch: np.ndarray,
    duration: np.ndarray,
    type_labels: np.ndarray | None,
    n_clusters: int = 20,
    seed: int = 0,
) -> dict:
    """Compute the handoff's eval gates on a held-out split.

    Returns a dict scorecard. Prints params/thresholds/row counts per the lab
    convention. ``shape_labels`` is the k-means partition over REGISTERED ridges
    (the shape ground truth); pitch/duration are the nuisance axes the latent
    must stop sorting by.
    """
    from sklearn.cluster import KMeans
    from sklearn.metrics import normalized_mutual_info_score as nmi

    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=seed)
    latent_part = km.fit_predict(latents)

    shape_eta2 = eta2(latents, shape_labels)
    pitch_eta2 = eta2(latents, _bin_continuous(pitch))
    dur_eta2 = eta2(latents, _bin_continuous(duration))
    type_nmi = (
        float(nmi(type_labels, latent_part))
        if type_labels is not None else None
    )

    score = {
        "n_rows": int(latents.shape[0]),
        "latent_dim": int(latents.shape[1]),
        "n_clusters": n_clusters,
        "seed": seed,
        "shape_eta2": float(shape_eta2),
        "pitch_eta2": float(pitch_eta2),
        "duration_eta2": float(dur_eta2),
        "type_nmi": type_nmi,
        "gate1_shape_clears_0.12": bool(shape_eta2 >= 0.12),
        "gate1_target_0.50": bool(shape_eta2 >= 0.50),
        "gate2_pitch_below_deadend_0.527": bool(pitch_eta2 < 0.527),
        "gate2_dur_below_deadend_0.404": bool(dur_eta2 < 0.404),
        "gate3_nmi_beats_prod_0.04": (
            bool(type_nmi > 0.04) if type_nmi is not None else None
        ),
        "kill": bool(shape_eta2 < 0.12),
    }
    print("[scorecard] " + json.dumps(score, indent=2), flush=True)
    print(
        "[reference] registration≈0.58-0.75  production≈0.12  dead-end≈0.081  "
        f"|  THIS shape_eta2={shape_eta2:.3f}",
        flush=True,
    )
    if score["kill"]:
        print("[VERDICT] KILL — shape η² < 0.12; the derivative term cannot rescue "
              "the image-VAE objective. Registration (0.75) is the answer.", flush=True)
    return score


def _bin_continuous(x: np.ndarray, n_bins: int = 20) -> np.ndarray:
    """Quantile-bin a continuous nuisance variable into integer labels for eta2."""
    x = np.asarray(x, dtype=float)
    edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    edges = np.unique(edges)
    if len(edges) < 3:
        return np.zeros(len(x), dtype=int)
    return np.clip(np.digitize(x, edges[1:-1]), 0, len(edges) - 2)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Score a Pathway-A shape VAE against the gates.")
    ap.add_argument("--latents", required=True, help="latents.npy from training")
    ap.add_argument("--shape-labels", required=True, help="registered-ridge k-means labels (.npy)")
    ap.add_argument("--pitch", required=True, help="per-call mean frequency (.npy)")
    ap.add_argument("--duration", required=True, help="per-call duration (.npy)")
    ap.add_argument("--type-labels", default=None, help="syllable_type labels (.npy), optional")
    ap.add_argument("--n-clusters", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="write scorecard JSON here")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    latents = np.load(args.latents)
    shape_labels = np.load(args.shape_labels)
    pitch = np.load(args.pitch)
    duration = np.load(args.duration)
    type_labels = np.load(args.type_labels) if args.type_labels else None
    print("[params] " + json.dumps({
        "latents": args.latents, "shape_labels": args.shape_labels,
        "pitch": args.pitch, "duration": args.duration,
        "type_labels": args.type_labels, "n_clusters": args.n_clusters,
        "seed": args.seed,
    }), flush=True)
    score = score_gates(
        latents, shape_labels, pitch, duration, type_labels,
        n_clusters=args.n_clusters, seed=args.seed,
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(score, indent=2))
        print(f"[done] scorecard -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Phase 0a -- Linear-probe precursor for ROADMAP_SHAPE_INVARIANT_LATENT.

HARD KILL-GATE (roadmap Phase 0a). No training, no new model. Reuses the FROZEN
Pathway B contrastive encoder (shape eta2 0.044 on the prior eval) and asks one
narrow question:

    Does the frozen-B representation contain *linearly decodable* chevron-shape
    signal, beyond what a RANDOM-INIT encoder of the same architecture yields?

If a frozen-encoder linear probe is at chance (or no better than a random-init
projection), no substrate swap + retrain on this architecture can rescue shape
clustering -> ship registration permanently (models/shape_kmeans/k20.joblib) and
close the VAE family. ~50% of the roadmap's probability mass is expected to die
here, for ~30-60 min of CPU.

Eval gate (verbatim from the roadmap):
  Probe A (chevron-vs-non-chevron, 5-fold stratified CV accuracy):
    PASS  if  acc >= 0.65  AND  acc >= random_init_acc + 0.10
    KILL  if  acc <  0.65  OR   acc <= random_init_acc + 0.05
    (the gap between 0.05 and 0.10 is a MARGINAL band -> report, do not auto-kill)
  Probe B (manual syllable_type): N/A. classified_detections_* has only the
    DeepSqueak `Cluster_NN` `label` column, no `syllable_type`. Reported
    unavailable per the roadmap's "if available" hedge -- NOT fabricated.

Why the AND-clause matters: chevron is a minority class, so raw accuracy is
inflated by the majority "non-chevron" prior. The random-init control absorbs
that prior (it sees the same imbalance), so the trained encoder must beat random
by a margin to be credited -- this defends against substrate-eval circularity.
Balanced accuracy is also reported as the imbalance-honest companion metric.

Data + compute (all on the rig, canonical root /data/shachar/contour_vae):
  - embeddings.npy   : frozen-B forward over all N patches (CACHED -> reused as-is)
  - desc_denoised.npz: row (surviving patch idx) + shapes[N,50] -> chevron labels
  - patches.npz      : 16.6 GB (N,257,234) float32 -- touched ONLY by the
                       random-init control forward, BATCHED over a memmap. Never
                       full-loaded (the box OOM'd once doing exactly that).

CPU is sufficient (--device cpu default; pass --device cuda:0 only if explicitly
clearing it with Pathway A). Mirrors the train script's inference preprocessing
EXACTLY (per-patch max-normalize, model.eval(), batch=1024) so the control is fair.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# chevron/valley heuristic -- INLINED verbatim from the archived
# scripts/eval_shape_encoder.py::chevron_valley (rig: scripts/eval_shape_encoder.py).
# Kept inline so this precursor has no fragile cross-module import for the label
# function; the architecture (ContrastiveEncoder) IS imported, because the
# random-init control must match it bit-for-bit.
# ---------------------------------------------------------------------------
def chevron_valley(shapes: np.ndarray) -> np.ndarray:
    """Holy/Guo-style chevron vs valley from the registered (de-meaned) 50-pt
    ridge shape. Mirrors the M10 heuristic so labels match prior scorecards."""
    N = shapes.shape[1]
    lo, hi = int(0.2 * N), int(0.8 * N)
    pk = shapes.argmax(1)
    tr = shapes.argmin(1)
    emax = np.maximum(shapes[:, 0], shapes[:, -1])
    emin = np.minimum(shapes[:, 0], shapes[:, -1])
    cv = np.array(["other"] * len(shapes), dtype=object)
    cv[(pk >= lo) & (pk <= hi) & (shapes.max(1) - emax > 2)] = "chevron"
    cv[(tr >= lo) & (tr <= hi) & (emin - shapes.min(1) > 2)] = "valley"
    return cv


def _import_contrastive_encoder(train_module_dir: Path):
    """Import ContrastiveEncoder from the (rig) train script so the random-init
    control is the *same architecture*. Re-implementing it here would risk
    silent drift that makes the control invalid."""
    d = str(train_module_dir)
    if d not in sys.path:
        sys.path.insert(0, d)
    from train_shape_encoder_contrastive import ContrastiveEncoder  # noqa: E402
    return ContrastiveEncoder


def cv_probe(Z: np.ndarray, y: np.ndarray, folds: int, seed: int) -> dict:
    """5-fold stratified CV logistic-regression probe. StandardScaler in-pipeline
    (applied identically to trained and random embeddings). Returns mean/std of
    accuracy and balanced_accuracy."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    skf = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    pipe = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    acc = cross_val_score(pipe, Z, y, cv=skf, scoring="accuracy")
    bal = cross_val_score(pipe, Z, y, cv=skf, scoring="balanced_accuracy")
    return {
        "acc_mean": float(acc.mean()), "acc_std": float(acc.std()),
        "bal_acc_mean": float(bal.mean()), "bal_acc_std": float(bal.std()),
        "acc_folds": [float(x) for x in acc],
    }


def random_init_embeddings(
    ContrastiveEncoder, patches, embed_dim: int, proj_dim: int,
    seed: int, batch: int, device: str,
) -> np.ndarray:
    """Forward ALL patches through a random-init encoder of the SAME architecture.
    The one step that touches the 16.6 GB patches.npz -- batched over the memmap,
    eval mode (fresh BatchNorm running stats = near-identity). Preprocessing
    mirrors train_shape_encoder_contrastive.load_batch EXACTLY."""
    import torch

    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = device if (device == "cpu" or torch.cuda.is_available()) else "cpu"
    model = ContrastiveEncoder(embed_dim, proj_dim).to(dev)
    model.eval()

    N = patches.shape[0]
    out = np.zeros((N, embed_dim), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for s0 in range(0, N, batch):
            idx = np.arange(s0, min(s0 + batch, N))            # already sorted
            b = np.asarray(patches[idx], dtype=np.float32)     # (B,H,W) from memmap
            t = torch.from_numpy(b).unsqueeze(1)               # (B,1,H,W)
            mx = t.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
            e, _ = model((t / mx).to(dev))                     # per-patch [0,1]
            out[idx] = e.cpu().numpy()
            if s0 % (batch * 20) == 0:
                print(f"    random-init forward {s0:>6d}/{N}  "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)
    print(f"[INFO] random-init forward done in {(time.time()-t0)/60:.1f} min", flush=True)
    return out


def main() -> None:
    R = Path("/data/shachar/contour_vae")
    ap = argparse.ArgumentParser(description="Phase 0a linear-probe precursor")
    ap.add_argument("--run", default=str(R / "results/latent_transitions/b_contrastive"),
                    help="dir holding the frozen-B embeddings.npy + encoder.pt")
    ap.add_argument("--desc", default=str(R / "results/eval_shape/desc_denoised.npz"))
    ap.add_argument("--patches",
                    default=str(R / "results/denoised_patches/combined_denoised/patches.npz"))
    ap.add_argument("--train-module-dir", default="scripts/experiments",
                    help="dir containing train_shape_encoder_contrastive.py (for ContrastiveEncoder)")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda:0")
    ap.add_argument("--out", default="", help="JSON output path (default: <run>/score_phase0a_linear_probe.json)")
    a = ap.parse_args()

    run = Path(a.run)
    out_json = Path(a.out) if a.out else run / "score_phase0a_linear_probe.json"

    # ---- load cached frozen-B embeddings + chevron labels -------------------
    embs = np.load(run / "embeddings.npy")                     # (N, embed_dim) frozen-B, ALL patches
    d = np.load(a.desc)
    row = d["row"]                                             # patch idx that survived ridge extraction
    shapes = d["shapes"]
    cv = chevron_valley(shapes)
    y = (cv == "chevron").astype(int)                          # Probe A: chevron(1) vs non-chevron(0)

    Z_trained = embs[row]                                      # align cache rows -> labelled patches
    n_chev = int(y.sum()); n_tot = int(len(y))
    chev_frac = n_chev / max(n_tot, 1)
    majority = max(chev_frac, 1 - chev_frac)                   # trivial-classifier accuracy

    print(f"[PARAM] phase0a run={run.name} embed_dim={embs.shape[1]} folds={a.folds} "
          f"seed={a.seed} batch={a.batch} device={a.device}", flush=True)
    print(f"[PARAM] probe=chevron-vs-non-chevron  scaler=StandardScaler  "
          f"clf=LogisticRegression(max_iter=2000)  scoring=accuracy(gate)+balanced_accuracy",
          flush=True)
    print(f"[INFO] embeddings N_all={embs.shape[0]}  labelled(row)={n_tot}  "
          f"chevron={n_chev} ({chev_frac:.3f})  non-chevron={n_tot - n_chev}  "
          f"majority-baseline-acc={majority:.3f}", flush=True)
    print(f"[INFO] cv breakdown: chevron={int((cv=='chevron').sum())} "
          f"valley={int((cv=='valley').sum())} other={int((cv=='other').sum())}", flush=True)

    # ---- Probe A on FROZEN-B embeddings -------------------------------------
    print("\n[RUN] Probe A on frozen-B embeddings (5-fold CV) ...", flush=True)
    frozen = cv_probe(Z_trained, y, a.folds, a.seed)
    print(f"  frozen-B   acc={frozen['acc_mean']:.3f}+/-{frozen['acc_std']:.3f}  "
          f"bal_acc={frozen['bal_acc_mean']:.3f}+/-{frozen['bal_acc_std']:.3f}", flush=True)

    # ---- random-init control (same architecture, untrained) -----------------
    print("\n[RUN] random-init control: forwarding patches through untrained encoder ...", flush=True)
    ContrastiveEncoder = _import_contrastive_encoder(Path(a.train_module_dir))
    z = np.load(a.patches, mmap_mode="r")
    patches = z["patches"]                                     # (N,257,234) float32 memmap
    print(f"[INFO] patches {patches.shape} (memmap, batched -- never full-loaded)", flush=True)
    rand_embs = random_init_embeddings(
        ContrastiveEncoder, patches, a.embed_dim, a.proj_dim, a.seed, a.batch, a.device)
    Z_random = rand_embs[row]
    print("\n[RUN] Probe A on random-init embeddings (5-fold CV) ...", flush=True)
    rnd = cv_probe(Z_random, y, a.folds, a.seed)
    print(f"  random-init acc={rnd['acc_mean']:.3f}+/-{rnd['acc_std']:.3f}  "
          f"bal_acc={rnd['bal_acc_mean']:.3f}+/-{rnd['bal_acc_std']:.3f}", flush=True)

    # ---- gate evaluation ----------------------------------------------------
    acc = frozen["acc_mean"]
    rnd_acc = rnd["acc_mean"]
    gap = acc - rnd_acc
    passed = (acc >= 0.65) and (gap >= 0.10)
    killed = (acc < 0.65) or (gap <= 0.05)
    verdict = "PASS" if passed else ("KILL" if killed else "MARGINAL")

    score = dict(
        phase="0a_linear_probe",
        probe_A=dict(
            label="chevron-vs-non-chevron (ridge heuristic; substrate-independent)",
            n=n_tot, n_chevron=n_chev, chevron_frac=chev_frac,
            majority_baseline_acc=majority,
            frozen_B=frozen, random_init=rnd,
            acc=acc, random_init_acc=rnd_acc, gap=gap,
        ),
        probe_B=dict(available=False,
                     reason="no syllable_type column in classified_detections_* "
                            "(only DeepSqueak Cluster_NN); per roadmap 'if available'"),
        gate=dict(pass_rule="acc>=0.65 AND acc>=random+0.10",
                  kill_rule="acc<0.65 OR acc<=random+0.05"),
        verdict=verdict,
    )
    out_json.write_text(json.dumps(score, indent=2))

    print("\n===== PHASE 0a LINEAR-PROBE SCORECARD =====")
    print(f"  Probe A  frozen-B acc   {acc:.3f}   (gate: >=0.65)")
    print(f"           random-init    {rnd_acc:.3f}")
    print(f"           gap            {gap:+.3f}   (gate: PASS>=+0.10 | KILL<=+0.05)")
    print(f"           majority base  {majority:.3f}   (trivial non-chevron classifier)")
    print(f"           balanced acc   frozen={frozen['bal_acc_mean']:.3f}  "
          f"random={rnd['bal_acc_mean']:.3f}  (imbalance-honest)")
    print(f"  Probe B  N/A (no manual syllable_type labels)")
    print(f"\n[VERDICT] {verdict}")
    if verdict == "PASS":
        print("  -> frozen-B has linearly-decodable shape signal beyond random. Proceed to Phase 0b.")
    elif verdict == "KILL":
        print("  -> architecture cannot represent shape regardless of substrate/loss.")
        print("     SHIP models/shape_kmeans/k20.joblib permanently; write family-CLOSED memo.")
    else:
        print("  -> MARGINAL (0.05 < gap < 0.10 with acc>=0.65). Report; user decides Phase 0b.")
    print(f"[INFO] wrote {out_json}", flush=True)
    print("[DONE] probe_shape_existing_encoder", flush=True)


if __name__ == "__main__":
    main()

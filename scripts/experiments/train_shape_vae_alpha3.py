"""α₃-C Phase A6 — train a shape β-VAE with externally-anchored evaluation.

Trains an unsupervised β-VAE on the Stack-4 contour-masked patches produced by
Phase A3 (``data/alpha3_patches/<call_id>.png``) and evaluates whether its
latent space clusters by SHAPE — judged against the **v1 oracle labels** from
Phase A4 (``data/labels_vocalmat_v1_on_131204.csv``), NOT against the F(t) ridge
that built the encoder input. That substrate independence is the whole point:
the production registration→shape result (η²≈0.75) is eval-circular because the
labels and the features both come from the same registered ridge. Here the
labels come from an entirely different oracle (our in-house VocalMat-anchored
ResNet-18, ``results/lab_classifier_v1/best.pt``), so a positive result would
break that circularity.

Architecture family: matches the production / bake-off conv β-VAE
(``archive/cleaning_legacy/stack3/scripts/experiments/rig_M10_image_vae.py``
``ImgVAE``) — stride-2 4×4 Conv blocks (BatchNorm + LeakyReLU 0.1) down to a
small spatial grid, linear μ/logvar heads at z=32, a mirror ConvTranspose
decoder with sigmoid output, KL = -0.5·mean(1 + logvar - μ² - exp(logvar)),
Adam @ 1e-3, β=0.5. The encoder depth auto-adapts to --img-size (each block
halves the spatial dims), so 64→4×4 (4 blocks) and 128→4×4 (5 blocks).

Outputs go to ``results/alpha3/shape_vae/`` — NEVER to ``models/`` (production).

NOT-to-touch: this script only READS the A3 patches and A4 labels. It never
writes to ``models/``, ``corpus.py``, Stack 4, or the production detection
pipeline.

Usage (rig)::

    PYTHONPATH=src .venv/bin/python scripts/experiments/train_shape_vae_alpha3.py \\
        --manifest data/alpha3_patches/manifest.csv \\
        --labels data/labels_vocalmat_v1_on_131204.csv \\
        --img-size 128 --z-dim 32 --beta 0.5 \\
        --epochs 120 --batch-size 256 --lr 1e-3 --device cuda

CPU smoke (harness proof, garbage numbers expected)::

    PYTHONPATH=src .venv/bin/python scripts/experiments/train_shape_vae_alpha3.py \\
        --manifest <fixture>/manifest.csv --labels <fixture>/labels.csv \\
        --epochs 1 --batch-size 8 --img-size 64 --device cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

# GRIMSLEY_12_CLASSES from the production classifier dataset (substrate-independent
# oracle taxonomy). Imported so the geometric-class mapping is anchored to the
# canonical names, not hand-typed strings.
from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES

# ---------------------------------------------------------------------------
# Geometric-class mapping: GRIMSLEY 12-class taxonomy -> coarse SHAPE families.
# Documented per the A6 spec. GRIMSLEY_12_CLASSES (verified at write time) is:
#   ('Noise','Step up','Down-FM','Short','Chevron','Up-FM','Flat','Two steps',
#    'Step down','Complex','Reverse Chevron','Multi-steps')
# Mapping rationale:
#   chevron : peaked / inverted-V pitch contours -> "Chevron", "Reverse Chevron"
#   jump    : discrete frequency-step calls       -> "Step up","Step down",
#             "Two steps","Multi-steps"
#   complex : non-decomposable multi-component     -> "Complex"
#   flat    : near-constant-pitch tones            -> "Flat"
# (Down-FM, Up-FM, Short, Noise are intentionally NOT in a geometric family;
#  they fall in the "others" pool for the chevron-vs-others purity test.)
# ---------------------------------------------------------------------------
GEOM_FAMILY = {
    "Chevron": "chevron",
    "Reverse Chevron": "chevron",
    "Step up": "jump",
    "Step down": "jump",
    "Two steps": "jump",
    "Multi-steps": "jump",
    "Complex": "complex",
    "Flat": "flat",
}


# ===========================================================================
# Model — conv β-VAE matching the rig ImgVAE family.
# ===========================================================================
class ShapeVAE(nn.Module):
    """Minimal conv β-VAE. Depth adapts so img_size halves down to a 4x4 grid."""

    def __init__(self, img_size: int = 128, z_dim: int = 32):
        super().__init__()
        if img_size % 4 != 0 or img_size < 16:
            raise ValueError(f"--img-size must be a multiple of 4 and >=16, got {img_size}")
        # number of stride-2 blocks to reach a 4x4 spatial grid
        n_blocks = 0
        s = img_size
        while s > 4:
            s //= 2
            n_blocks += 1
        if s != 4:
            raise ValueError(f"--img-size {img_size} must reduce to a 4x4 grid (power-of-2 * 4)")
        self.spatial = 4
        chans = [1, 32, 64, 128, 256, 512]
        chans = chans[: n_blocks + 1]
        self.enc_chans = chans

        def cb(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1))

        self.enc = nn.Sequential(*[cb(chans[i], chans[i + 1]) for i in range(n_blocks)])
        self.flat_dim = chans[-1] * self.spatial * self.spatial
        self.mu = nn.Linear(self.flat_dim, z_dim)
        self.lv = nn.Linear(self.flat_dim, z_dim)
        self.fd = nn.Linear(z_dim, self.flat_dim)

        def db(i, o):
            return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1))

        rev = list(reversed(chans))  # e.g. [256,128,64,32,1]
        dec_blocks = [db(rev[i], rev[i + 1]) for i in range(len(rev) - 2)]
        dec_blocks.append(nn.ConvTranspose2d(rev[-2], rev[-1], 4, 2, 1))
        self.dec = nn.Sequential(*dec_blocks)
        self._top_chan = chans[-1]

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.mu(h), self.lv(h)

    def decode(self, z):
        h = self.fd(z).view(-1, self._top_chan, self.spatial, self.spatial)
        return torch.sigmoid(self.dec(h))

    def forward(self, x):
        mu, lv = self.encode(x)
        zz = mu + torch.randn_like(lv) * (0.5 * lv).exp()
        return self.decode(zz), mu, lv


def vae_loss(x, xh, mu, lv, beta: float):
    """β-VAE objective: MSE reconstruction on [0,1] pixels + β·KL."""
    recon = F.mse_loss(xh, x)
    kl = -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp())
    return recon + beta * kl, float(recon), float(kl)


# ===========================================================================
# Data
# ===========================================================================
def _resolve_path(repo_root: Path, manifest_path: Path, raw: str) -> Path:
    p = Path(raw)
    if p.is_absolute() and p.exists():
        return p
    # try relative to repo root, then relative to the manifest's directory
    for base in (repo_root, manifest_path.parent, Path.cwd()):
        cand = base / raw
        if cand.exists():
            return cand
    return repo_root / raw  # best effort; load will surface the error


def load_dataset(manifest_csv: Path, labels_csv: Path, img_size: int,
                 min_prob: float, use_high_conf: bool, repo_root: Path):
    """Join A3 patches to A4 oracle labels; return (X float32 [N,1,H,W] in [0,1],
    class_idx int [N], class_name str [N], call_ids [N])."""
    man = pd.read_csv(manifest_csv)
    lab = pd.read_csv(labels_csv)
    if "call_id" not in man.columns or "path" not in man.columns:
        raise ValueError(f"manifest must have call_id,path columns; got {list(man.columns)}")
    needed = {"call_id", "top1_class"}
    if not needed.issubset(lab.columns):
        raise ValueError(f"labels must have {needed}; got {list(lab.columns)}")

    df = man.merge(lab, on="call_id", suffixes=("_manifest", "_label"))
    n_join = len(df)

    # Label gate: high_confidence flag (preferred) OR min-prob threshold.
    if use_high_conf and "high_confidence" in df.columns:
        hc = df["high_confidence"]
        if hc.dtype == object:
            hc = hc.astype(str).str.lower().isin(["true", "1", "yes"])
        df = df[hc.astype(bool)]
        gate = "high_confidence==True"
    elif "top1_prob" in df.columns:
        df = df[df["top1_prob"].astype(float) >= min_prob]
        gate = f"top1_prob>={min_prob}"
    else:
        gate = "none"

    df = df.reset_index(drop=True)
    if len(df) == 0:
        raise RuntimeError(f"0 patches after gate '{gate}' (joined {n_join} rows)")

    # path column: manifest's path (label CSV may also carry one)
    path_col = "path_manifest" if "path_manifest" in df.columns else "path"

    imgs = np.zeros((len(df), img_size, img_size), dtype=np.float32)
    for i, raw in enumerate(df[path_col].tolist()):
        fp = _resolve_path(repo_root, manifest_csv, str(raw))
        im = Image.open(fp).convert("L").resize((img_size, img_size), Image.BILINEAR)
        imgs[i] = np.asarray(im, dtype=np.float32) / 255.0

    X = torch.from_numpy(imgs).unsqueeze(1)  # [N,1,H,W]
    class_name = df["top1_class"].astype(str).to_numpy()
    # stable class index over the GRIMSLEY taxonomy (fallback to factorize if unseen)
    name_to_idx = {c: i for i, c in enumerate(GRIMSLEY_12_CLASSES)}
    class_idx = np.array([name_to_idx.get(c, -1) for c in class_name], dtype=int)
    if (class_idx < 0).any():
        # unseen names (e.g. synthetic fixture) -> factorize the unseen ones
        codes, _ = pd.factorize(class_name)
        class_idx = codes.astype(int)
    return X, class_idx, class_name, df["call_id"].to_numpy(), gate, n_join


# ===========================================================================
# Evaluation
# ===========================================================================
def knn_purity_chevron(latent: np.ndarray, geom_label: np.ndarray, target: str, k: int = 10) -> float:
    """For each point whose geom family == target, fraction of its k nearest
    neighbours (in latent space, excluding self) that share the target family.
    'chevron-vs-others' purity = same-family rate among the target's neighbours."""
    n = len(latent)
    if n <= 1:
        return 0.0
    is_target = geom_label == target
    if is_target.sum() == 0:
        return float("nan")
    kk = min(k, n - 1)
    nn = NearestNeighbors(n_neighbors=kk + 1).fit(latent)
    _, idx = nn.kneighbors(latent)
    purities = []
    for i in np.where(is_target)[0]:
        neigh = idx[i, 1:]  # drop self
        purities.append(float(np.mean(geom_label[neigh] == target)))
    return float(np.mean(purities)) if purities else float("nan")


def linear_probe(z_tr, y_tr, z_va, y_va) -> float:
    """Logistic-regression accuracy: latent -> oracle class. Returns val accuracy."""
    if len(np.unique(y_tr)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=1000)
    clf.fit(z_tr, y_tr)
    return float(clf.score(z_va, y_va))


def column_mean_features(X: np.ndarray) -> np.ndarray:
    """Identity baseline: collapse each [1,H,W] image to its per-column mean
    (length-W vector). No learning — pure pixel statistics."""
    # X: [N,1,H,W] -> mean over H (rows) -> [N,W]
    return X[:, 0, :, :].mean(axis=1)


def run_metrics(latent: np.ndarray, class_idx: np.ndarray, geom_label: np.ndarray,
                val_mask: np.ndarray, k: int) -> dict:
    """All 3 metrics on a given latent representation."""
    km = KMeans(n_clusters=20, n_init=10, random_state=42).fit_predict(latent)
    nmi = float(normalized_mutual_info_score(class_idx, km))
    out = {"nmi_kmeans20": nmi}
    for fam in ("chevron", "jump", "complex", "flat"):
        out[f"knn_purity_{fam}"] = knn_purity_chevron(latent, geom_label, fam, k=k)
    tr = ~val_mask
    out["linear_probe_val_acc"] = linear_probe(latent[tr], class_idx[tr],
                                               latent[val_mask], class_idx[val_mask])
    return out


# ===========================================================================
# Train
# ===========================================================================
def train_vae(model, X_tr, beta, epochs, batch_size, lr, device):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    Xtr = X_tr.to(device)
    n = len(Xtr)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        last = (0.0, 0.0, 0.0)
        for s in range(0, n, batch_size):
            b = Xtr[perm[s:s + batch_size]]
            xh, mu, lv = model(b)
            loss, recon, kl = vae_loss(b, xh, mu, lv, beta)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last = (float(loss), recon, kl)
        if ep % 10 == 0 or ep == epochs - 1:
            print(f"  ep{ep:3d} loss={last[0]:.4f} recon={last[1]:.4f} kl={last[2]:.4f}", flush=True)
    return float(last[0])


def encode_all(model, X, device, bs=512) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        chunks = [model.encode(X[s:s + bs].to(device))[0].cpu() for s in range(0, len(X), bs)]
    return torch.cat(chunks).numpy()


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description="α₃-C A6 shape β-VAE with externally-anchored eval")
    ap.add_argument("--manifest", default="data/alpha3_patches/manifest.csv")
    ap.add_argument("--labels", default="data/labels_vocalmat_v1_on_131204.csv")
    ap.add_argument("--out-dir", default="results/alpha3/shape_vae")
    ap.add_argument("--img-size", type=int, default=128)
    ap.add_argument("--z-dim", type=int, default=32)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--min-prob", type=float, default=0.5,
                    help="alternative label gate when --no-high-conf is set")
    ap.add_argument("--no-high-conf", action="store_true",
                    help="gate on --min-prob instead of the high_confidence flag")
    ap.add_argument("--knn-k", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    a = ap.parse_args()

    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    repo_root = Path(__file__).resolve().parents[2]
    device = a.device if (a.device == "cpu" or torch.cuda.is_available()) else "cpu"
    out_dir = Path(a.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("α₃-C Phase A6 — shape β-VAE, externally-anchored (v1 oracle) eval")
    print("=" * 72)
    print("PARAMS:")
    for k, v in vars(a).items():
        print(f"  {k:18s} = {v}")
    print(f"  resolved_device    = {device}  (cuda_available={torch.cuda.is_available()})")
    print(f"  out_dir            = {out_dir}")
    print(f"  GRIMSLEY_12_CLASSES= {GRIMSLEY_12_CLASSES}")
    print("  GEOM mapping:")
    for fam in ("chevron", "jump", "complex", "flat"):
        members = [c for c, f in GEOM_FAMILY.items() if f == fam]
        print(f"    {fam:8s} <- {members}")
    print("=" * 72, flush=True)

    # ---- data ----
    man_path = Path(a.manifest)
    lab_path = Path(a.labels)
    if not man_path.is_absolute():
        man_path = repo_root / a.manifest if (repo_root / a.manifest).exists() else man_path
    if not lab_path.is_absolute():
        lab_path = repo_root / a.labels if (repo_root / a.labels).exists() else lab_path

    X, class_idx, class_name, call_ids, gate, n_join = load_dataset(
        man_path, lab_path, a.img_size, a.min_prob,
        use_high_conf=not a.no_high_conf, repo_root=repo_root)
    geom_label = np.array([GEOM_FAMILY.get(c, "others") for c in class_name], dtype=object)
    n = len(X)
    print(f"[DATA] joined {n_join} manifest+label rows; {n} kept after gate '{gate}'")
    uniq, cnts = np.unique(class_name, return_counts=True)
    print(f"[DATA] class distribution: {dict(zip(uniq.tolist(), cnts.tolist()))}")
    guniq, gcnts = np.unique(geom_label.astype(str), return_counts=True)
    print(f"[DATA] geom-family distribution: {dict(zip(guniq.tolist(), gcnts.tolist()))}", flush=True)

    # ---- deterministic 10% val split ----
    rng = np.random.RandomState(a.seed)
    perm = rng.permutation(n)
    n_val = max(1, int(round(a.val_frac * n)))
    val_idx = perm[:n_val]
    val_mask = np.zeros(n, dtype=bool)
    val_mask[val_idx] = True
    tr_mask = ~val_mask
    print(f"[SPLIT] train={tr_mask.sum()} val={val_mask.sum()} (val_frac={a.val_frac}, seed={a.seed})", flush=True)

    # ---- BASELINE (a): random-init encoder (same arch, untrained) ----
    rand_model = ShapeVAE(img_size=a.img_size, z_dim=a.z_dim).to(device)
    z_rand = encode_all(rand_model, X, device)
    print("[BASELINE-a] random-init encoder latent computed", flush=True)
    metrics_rand = run_metrics(z_rand, class_idx, geom_label, val_mask, a.knn_k)

    # ---- BASELINE (b): identity = per-column-mean of the patch ----
    z_identity = column_mean_features(X.numpy())
    print(f"[BASELINE-b] identity (per-column mean) features computed: dim={z_identity.shape[1]}", flush=True)
    metrics_identity = run_metrics(z_identity, class_idx, geom_label, val_mask, a.knn_k)

    # ---- LEARNED encoder: train the β-VAE ----
    model = ShapeVAE(img_size=a.img_size, z_dim=a.z_dim).to(device)
    print(f"[TRAIN] {sum(p.numel() for p in model.parameters()):,} params; "
          f"enc_chans={model.enc_chans}; training {a.epochs} epochs on {tr_mask.sum()} patches", flush=True)
    final_loss = train_vae(model, X[tr_mask], a.beta, a.epochs, a.batch_size, a.lr, device)
    ckpt_path = out_dir / "best_model.pt"
    torch.save({"state_dict": model.state_dict(),
                "img_size": a.img_size, "z_dim": a.z_dim, "beta": a.beta,
                "final_train_loss": final_loss}, ckpt_path)
    print(f"[TRAIN] saved checkpoint -> {ckpt_path}", flush=True)
    z_learned = encode_all(model, X, device)
    metrics_learned = run_metrics(z_learned, class_idx, geom_label, val_mask, a.knn_k)

    # ---- verdict ----
    def fmt(x):
        return "nan" if (x is None or (isinstance(x, float) and np.isnan(x))) else f"{x:.4f}"

    rows = [
        ("nmi_kmeans20", "NMI(KMeans20, oracle)"),
        ("knn_purity_chevron", "kNN purity chevron"),
        ("knn_purity_jump", "kNN purity jump"),
        ("knn_purity_complex", "kNN purity complex"),
        ("knn_purity_flat", "kNN purity flat"),
        ("linear_probe_val_acc", "linear probe val acc"),
    ]
    print("\n" + "=" * 72)
    print("SCORECARD  (learned vs random-init vs identity baseline)")
    print("=" * 72)
    print(f"{'metric':28s} {'learned':>10s} {'rand-init':>10s} {'identity':>10s}")
    for key, label in rows:
        print(f"{label:28s} {fmt(metrics_learned.get(key)):>10s} "
              f"{fmt(metrics_rand.get(key)):>10s} {fmt(metrics_identity.get(key)):>10s}")

    nmi_l = metrics_learned["nmi_kmeans20"]
    chev_l = metrics_learned["knn_purity_chevron"]
    nmi_best_base = max([m["nmi_kmeans20"] for m in (metrics_rand, metrics_identity)])
    chev_bases = [m["knn_purity_chevron"] for m in (metrics_rand, metrics_identity)]
    chev_best_base = np.nanmax(chev_bases) if not all(np.isnan(chev_bases)) else float("nan")

    beats_nmi = nmi_l >= nmi_best_base + 0.10
    beats_chev = (not np.isnan(chev_l)) and (np.isnan(chev_best_base) or chev_l >= chev_best_base + 0.10)
    beats_baselines = beats_nmi and beats_chev

    ship = (nmi_l >= 0.25) and (not np.isnan(chev_l) and chev_l >= 0.50) and beats_baselines
    kill = (nmi_l < 0.15) or (not beats_baselines)

    if ship:
        verdict = "SHIP"
    elif kill:
        verdict = "KILL"
    else:
        verdict = "INCONCLUSIVE"

    print("-" * 72)
    print(f"  NMI learned={fmt(nmi_l)} (gate >=0.25; best baseline {fmt(nmi_best_base)}, "
          f"beat-by-0.10={beats_nmi})")
    print(f"  chevron kNN purity learned={fmt(chev_l)} (gate >=0.50; best baseline "
          f"{fmt(chev_best_base)}, beat-by-0.10={beats_chev})")
    print(f"  beats BOTH baselines by >=0.10 on NMI+chevron: {beats_baselines}")
    print(f"\n  >>> VERDICT: {verdict} <<<")
    print("=" * 72, flush=True)

    eval_out = {
        "phase": "alpha3-C-A6",
        "params": vars(a),
        "resolved_device": device,
        "n_patches": int(n),
        "n_joined": int(n_join),
        "label_gate": gate,
        "train_n": int(tr_mask.sum()),
        "val_n": int(val_mask.sum()),
        "grimsley_classes": list(GRIMSLEY_12_CLASSES),
        "geom_family_map": GEOM_FAMILY,
        "final_train_loss": final_loss,
        "metrics": {
            "learned": metrics_learned,
            "baseline_random_init": metrics_rand,
            "baseline_identity_colmean": metrics_identity,
        },
        "gates": {
            "nmi_ship_threshold": 0.25,
            "nmi_kill_threshold": 0.15,
            "chevron_purity_ship_threshold": 0.50,
            "beat_margin": 0.10,
            "beats_nmi_baseline": bool(beats_nmi),
            "beats_chevron_baseline": bool(beats_chev),
            "beats_both_baselines": bool(beats_baselines),
        },
        "verdict": verdict,
        "checkpoint": str(ckpt_path),
    }
    eval_path = out_dir / "eval.json"
    eval_path.write_text(json.dumps(eval_out, indent=2, default=str))
    print(f"[DONE] wrote eval -> {eval_path}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())

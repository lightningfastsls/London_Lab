"""M9 (rig, GPU) — contrastive 1-D shape encoder (NT-Xent) on TRUE ridges.

The fairer learned test than M8: no reconstruction bottleneck, no KL — trained
directly to make augmentation-invariant shape embeddings. Input is the registered
ridge (pitch/position/duration already removed), so the residual invariance to
learn is local TIME-WARP (tempo wobble) + noise. Two augmented views per ridge ->
NT-Xent. Cluster the pre-projection embedding -> same eta2 scorecard as M8/R1.

Bar to beat: hand-crafted registered `shape` eta2 = 0.577 (R1, true ridges).
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi

R = Path("/data/shachar/contour_vae")
DATA = R / "results/latent_transitions/shape_registered_TRUE/true_registered_ridges.npz"
OUT = R / "results/latent_transitions/m9_contrastive"; OUT.mkdir(parents=True, exist_ok=True)
L = 50


class Encoder(nn.Module):
    def __init__(self, edim=32, pdim=64):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.LeakyReLU(0.1),
            nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.LeakyReLU(0.1))
        self.emb = nn.Linear(128 * 7, edim)
        self.proj = nn.Sequential(nn.Linear(edim, edim), nn.ReLU(), nn.Linear(edim, pdim))

    def forward(self, x):
        h = self.emb(self.enc(x).flatten(1))
        return h, self.proj(h)


def augment(x):
    """x: (B,1,L) -> warped+noised view. Vectorized monotonic time-warp + amp + noise."""
    B = x.shape[0]; dev = x.device
    xf = x[:, 0, :]                                          # (B,L)
    inc = torch.rand(B, L, device=dev) * 0.5 + 0.75          # per-step speed 0.75..1.25
    cum = torch.cumsum(inc, 1)
    cum = (cum - cum[:, :1]) / (cum[:, -1:] - cum[:, :1] + 1e-9)   # monotonic [0,1]
    pos = cum * (L - 1)                                      # warped sample positions
    f = pos.floor().long().clamp(0, L - 2); frac = pos - f
    warped = torch.gather(xf, 1, f) * (1 - frac) + torch.gather(xf, 1, f + 1) * frac
    warped = warped * (0.9 + 0.2 * torch.rand(B, 1, device=dev)) + 0.05 * torch.randn_like(warped)
    return warped.unsqueeze(1)                              # (B,1,L)


def ntxent(z1, z2, t=0.2):
    z = F.normalize(torch.cat([z1, z2], 0), dim=1); N = z1.shape[0]
    sim = z @ z.T / t
    sim.fill_diagonal_(-1e9)
    tgt = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, tgt)


def eta2(v, lab):
    v = v if v.ndim == 2 else v[:, None]; keep = lab >= 0; v, lab = v[keep], lab[keep]
    g = v.mean(0); tot = float(((v - g) ** 2).sum())
    w = sum(float(((v[lab == l] - v[lab == l].mean(0)) ** 2).sum()) for l in np.unique(lab))
    return 1 - w / tot if tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=200); ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--edim", type=int, default=32); ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    dev = a.device if torch.cuda.is_available() else "cpu"
    print(f"[PARAM] M9 contrastive edim={a.edim} temp={a.temp} epochs={a.epochs} batch={a.batch} dev={dev}", flush=True)
    d = np.load(DATA, allow_pickle=True)
    Sh = d["shapes"].astype(np.float32); pitch, dur, jump, cv = d["pitch"], d["duration"], d["jump"], d["chevron_valley"]
    gstd = float(Sh.std()) or 1.0
    X = torch.from_numpy(Sh / gstd).unsqueeze(1).to(dev)
    n = len(X); print(f"[INFO] {n} ridges gstd={gstd:.2f}", flush=True)
    m = Encoder(a.edim).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(n, device=dev); tot = 0.0; nb = 0
        for s in range(0, n, a.batch):
            b = X[perm[s:s + a.batch]]
            if b.shape[0] < 8: continue
            _, p1 = m(augment(b)); _, p2 = m(augment(b))
            loss = ntxent(p1, p2, a.temp)
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss); nb += 1
        if ep % 40 == 0 or ep == a.epochs - 1:
            print(f"  ep{ep:3d} ntxent={tot/max(nb,1):.4f}", flush=True)
    m.eval()
    with torch.no_grad():
        Z = torch.cat([m(X[s:s + 2048])[0] for s in range(0, n, 2048)]).cpu().numpy()
    lab = KMeans(20, n_init=10, random_state=42).fit_predict(Z)
    sel = cv != "other"
    row = dict(method="M9_contrastive", pitch=eta2(pitch[:, None], lab), duration=eta2(dur[:, None], lab),
               shape=eta2(Sh, lab), curvature=eta2(jump[:, None], lab), cv_nmi=float(nmi(cv[sel], lab[sel])), n=int(n))
    print("\n===== M9 CONTRASTIVE SCORECARD =====")
    print("  pitch %.3f | dur %.3f | shape %.3f | curv %.3f | CV-NMI %.3f  (bar: shape 0.577)"
          % (row["pitch"], row["duration"], row["shape"], row["curvature"], row["cv_nmi"]), flush=True)
    np.save(OUT / "latent_m9.npy", Z); np.save(OUT / "labels_m9.npy", lab)
    (OUT / "score_m9.json").write_text(json.dumps(row, indent=2))
    rng = np.random.default_rng(0)
    uniq = sorted([c for c in np.unique(lab) if c >= 0], key=lambda c: -(lab == c).sum())[:20]
    nrow = int(np.ceil(len(uniq) / 5)) or 1
    fig, axes = plt.subplots(nrow, 5, figsize=(10, nrow * 1.7), squeeze=False)
    for ax in axes.ravel(): ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-40, 40); ax.axhline(0, color="#ccc", lw=.5)
    for i, c in enumerate(uniq):
        ax = axes[i // 5][i % 5]; mm = np.where(lab == c)[0]
        for j in mm[rng.choice(len(mm), min(40, len(mm)), replace=False)]: ax.plot(Sh[j], color="#1e6b3a", alpha=.10, lw=.6)
        ax.plot(Sh[mm].mean(0), color="black", lw=1.8); ax.set_title(f"c{c} n={len(mm)}", fontsize=7, loc="left")
    fig.suptitle("M9 contrastive — cluster-mean shapes", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(OUT / "m9.png", dpi=110); plt.close(fig)
    print("[DONE] M9", flush=True)


if __name__ == "__main__":
    main()

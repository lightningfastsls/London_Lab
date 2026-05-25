"""M8 (rig, GPU) — 1-D contour VAE with derivative-weighted loss on TRUE ridges.

Tests the user's idea: a learned shape-latent whose loss "cares about dF/dt".
Input = registered ridge (50-pt centered kHz; pitch/position/duration already
removed by R1). Encoder = Conv1d stack -> latent; decoder mirrors. Loss:

    L = w0·MSE(x,x̂) + w1·MSE(Δx,Δx̂) + w2·MSE(Δ²x,Δ²x̂) + β·KL

Run twice to isolate the derivative effect:
    --tag deriv  --w "1,3,1.5"   (emphasise slope + curvature)
    --tag plain  --w "1,0,0"     (vanilla VAE — value only)

Then encode all ridges -> KMeans(20) -> score eta2(pitch/dur/shape/curv) +
chevron/valley NMI, and compare to the hand-crafted registered features (R1).
Question: does the learned latent BEAT the fixed shape feature, or just recover it?

Data: /data/shachar/contour_vae/results/latent_transitions/shape_registered_TRUE/
      true_registered_ridges.npz  (key 'shapes' = (N,50) centered kHz).
"""
from __future__ import annotations
import argparse, base64, json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi

R = Path("/data/shachar/contour_vae")
DATA = R / "results/latent_transitions/shape_registered_TRUE/true_registered_ridges.npz"
OUT = R / "results/latent_transitions/m8_contour_vae"
OUT.mkdir(parents=True, exist_ok=True)
L = 50


class ContourVAE(nn.Module):
    def __init__(self, zdim=16):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv1d(1, 32, 5, 2, 2), nn.BatchNorm1d(32), nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, 5, 2, 2), nn.BatchNorm1d(64), nn.LeakyReLU(0.1),
            nn.Conv1d(64, 128, 5, 2, 2), nn.BatchNorm1d(128), nn.LeakyReLU(0.1))
        self.flat = 128 * 7  # 50->25->13->7
        self.fc_mu = nn.Linear(self.flat, zdim)
        self.fc_lv = nn.Linear(self.flat, zdim)
        self.fc_dec = nn.Linear(zdim, self.flat)
        self.dec = nn.Sequential(
            nn.Upsample(scale_factor=2), nn.Conv1d(128, 64, 5, 1, 2), nn.BatchNorm1d(64), nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2), nn.Conv1d(64, 32, 5, 1, 2), nn.BatchNorm1d(32), nn.LeakyReLU(0.1),
            nn.Upsample(scale_factor=2), nn.Conv1d(32, 16, 5, 1, 2), nn.BatchNorm1d(16), nn.LeakyReLU(0.1),
            nn.Conv1d(16, 1, 5, 1, 2))

    def encode(self, x):
        h = self.enc(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z):
        h = self.fc_dec(z).view(-1, 128, 7)
        h = self.dec(h)                       # (B,1,56)
        return F.interpolate(h, size=L, mode="linear", align_corners=False)

    def forward(self, x):
        mu, lv = self.encode(x)
        z = mu + torch.randn_like(lv) * (0.5 * lv).exp()
        return self.decode(z), mu, lv


def dloss(x, xh, w):
    w0, w1, w2 = w
    l = w0 * F.mse_loss(xh, x)
    if w1: l = l + w1 * F.mse_loss(torch.diff(xh, 1, -1), torch.diff(x, 1, -1))
    if w2: l = l + w2 * F.mse_loss(torch.diff(xh, 2, -1), torch.diff(x, 2, -1))
    return l


def eta2(v, lab):
    v = v if v.ndim == 2 else v[:, None]
    keep = lab >= 0; v, lab = v[keep], lab[keep]
    g = v.mean(0); tot = float(((v - g) ** 2).sum())
    w = sum(float(((v[lab == l] - v[lab == l].mean(0)) ** 2).sum()) for l in np.unique(lab))
    return 1 - w / tot if tot > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="deriv")
    ap.add_argument("--w", default="1,3,1.5", help="w0,w1,w2 recon weights")
    ap.add_argument("--zdim", type=int, default=16)
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    w = tuple(float(x) for x in a.w.split(","))
    torch.manual_seed(42); np.random.seed(42)
    dev = a.device if torch.cuda.is_available() else "cpu"
    print(f"[PARAM] tag={a.tag} w={w} zdim={a.zdim} beta={a.beta} epochs={a.epochs} dev={dev}", flush=True)

    d = np.load(DATA, allow_pickle=True)
    Sh = d["shapes"].astype(np.float32)
    pitch, dur, jump, cv = d["pitch"], d["duration"], d["jump"], d["chevron_valley"]
    gstd = float(Sh.std()) or 1.0
    X = torch.from_numpy(Sh / gstd).unsqueeze(1)          # (N,1,50), O(1)-scaled
    n = len(X); idx = np.random.permutation(n); ntr = int(0.9 * n)
    tr, va = idx[:ntr], idx[ntr:]
    print(f"[INFO] {n} ridges, gstd={gstd:.2f} kHz", flush=True)

    m = ContourVAE(a.zdim).to(dev)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    Xtr = X[tr].to(dev)
    best = 1e9
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(len(Xtr), device=dev)
        for s in range(0, len(Xtr), a.batch):
            b = Xtr[perm[s:s + a.batch]]
            xh, mu, lv = m(b)
            kl = -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp())
            loss = dloss(b, xh, w) + a.beta * kl
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 20 == 0 or ep == a.epochs - 1:
            m.eval()
            with torch.no_grad():
                xv = X[va].to(dev); xh, mu, lv = m(xv)
                vr = float(dloss(xv, xh, w))
            print(f"  ep{ep:3d} val_recon={vr:.4f}", flush=True)
            best = min(best, vr)

    # encode all (mu) -> cluster -> score
    m.eval()
    with torch.no_grad():
        Z = m.encode(X.to(dev))[0].cpu().numpy()
    lab = KMeans(20, n_init=10, random_state=42).fit_predict(Z)
    sel = cv != "other"
    row = dict(tag=a.tag, w=list(w), pitch=eta2(pitch[:, None], lab), duration=eta2(dur[:, None], lab),
               shape=eta2(Sh, lab), curvature=eta2(jump[:, None], lab),
               cv_nmi=float(nmi(cv[sel], lab[sel])), best_val_recon=best, n=int(n))
    print("\n===== M8 LEARNED-LATENT SCORECARD (tag=%s) =====" % a.tag)
    print("  pitch %.3f | dur %.3f | shape %.3f | curv %.3f | CV-NMI %.3f"
          % (row["pitch"], row["duration"], row["shape"], row["curvature"], row["cv_nmi"]), flush=True)

    np.save(OUT / f"latent_{a.tag}.npy", Z)
    np.save(OUT / f"labels_{a.tag}.npy", lab)
    (OUT / f"score_{a.tag}.json").write_text(json.dumps(row, indent=2))
    render(a.tag, lab, Sh, pitch)
    print("[DONE]", a.tag, flush=True)


def render(tag, lab, Sh, pitch):
    rng = np.random.default_rng(0)
    uniq = sorted([c for c in np.unique(lab) if c >= 0], key=lambda c: -(lab == c).sum())[:20]
    nrow = int(np.ceil(len(uniq) / 5)) or 1
    fig, axes = plt.subplots(nrow, 5, figsize=(10, nrow * 1.7), squeeze=False)
    for ax in axes.ravel():
        ax.set_xticks([]); ax.set_yticks([]); ax.set_ylim(-40, 40); ax.axhline(0, color="#ccc", lw=.5)
    for i, c in enumerate(uniq):
        ax = axes[i // 5][i % 5]; mm = np.where(lab == c)[0]
        for j in mm[rng.choice(len(mm), min(40, len(mm)), replace=False)]:
            ax.plot(Sh[j], color="#1d3b8a", alpha=.10, lw=.6)
        ax.plot(Sh[mm].mean(0), color="black", lw=1.8)
        ax.set_title(f"c{c} n={len(mm)} σ{pitch[mm].std():.0f}k", fontsize=7, loc="left")
    fig.suptitle(f"M8 contour-VAE [{tag}] — cluster-mean shapes", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, .97)); fig.savefig(OUT / f"m8_{tag}.png", dpi=110); plt.close(fig)


if __name__ == "__main__":
    main()

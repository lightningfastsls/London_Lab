"""M10 (rig, GPU) — 2-D image VAE on REGISTERED real patches w/ masked edge loss.

The user's literal choice: stay a 2-D image VAE, but (a) feed PURE-SHAPE images
(each patch vertically centered to kill pitch, active span cropped+resized to kill
position/duration) and (b) make the loss derivative/edge-aware and BACKGROUND-
EXCLUDED (masked) — not "reconstruct the black correctly".

Build pass (CPU, per-cohort to bound RAM): real patch (257,234) -> band-crop [35:205]
-> roll so energy-centroid row -> center -> crop active cols -> resize to 64x64.
In the SAME pass, extract the ridge (pitch/shape/dur/curv) so cluster scoring stays
on the same axes as R1/M8/M9. Train pass (GPU): conv VAE, loss =
  0.2*MSE_full + 1.0*MSE_masked + 1.0*edge(grad_h,grad_w) + beta*KL .
Then encode -> KMeans(20) -> scorecard vs the 0.577 registration bar.

Honest expectation (per M8): a VAE bottleneck likely won't beat free registration
for arc-shape; the image VAE's possible edge is capturing 2-D sub-harmonic /
multi-component structure a 1-D ridge discards.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.cluster import KMeans
from sklearn.metrics import normalized_mutual_info_score as nmi

sys.path.insert(0, "/data/mickey_london_lab/src")
from usv_spectrogram.features.ridge_tracker import track_ridge, RidgeConfig

R = Path("/data/shachar/contour_vae")
MP = R / "results/masked_patches"
OUT = R / "results/latent_transitions/m10_image_vae"; OUT.mkdir(parents=True, exist_ok=True)
CACHE = OUT / "registered_images.npz"
COHORTS = ["5970", "3452", "9252", "lab_131204"]
BAND0, BAND1 = 35, 205
CANVAS = 64
N_RESAMPLE = 50


def register_one(crop, freqs_khz):
    """crop (170,T) real power -> (img64, pitch, shape50, dur, jump) or None."""
    thr = max(1e-9, 0.02 * float(crop.max()))
    colmax = crop.max(0)
    act = np.where(colmax > thr)[0]
    if len(act) < 6:
        return None
    c0, c1 = act[0], act[-1]
    sub = crop[:, c0:c1 + 1]                                  # (170, W)
    # vertical center via energy-weighted centroid row
    e = sub.sum(1) + 1e-9
    crow = int(round(float((np.arange(len(e)) * e).sum() / e.sum())))
    shift = CANVAS // 2 - crow  # not used directly; we crop a window around crow
    lo = max(0, crow - 64); hi = min(sub.shape[0], crow + 64)
    win = sub[lo:hi, :]
    # resize to CANVAS x CANVAS
    t = torch.from_numpy(win[None, None].astype(np.float32))
    img = F.interpolate(t, size=(CANVAS, CANVAS), mode="bilinear", align_corners=False)[0, 0].numpy()
    mx = img.max()
    img = img / mx if mx > 0 else img                          # per-image [0,1]
    # ridge for scoring axes (in real kHz)
    cfg = RidgeConfig(transition_penalty=0.1, max_jump_bins=10, silence_threshold=thr)
    fm, am = track_ridge(crop, freqs_khz.astype(float), cfg)
    a2 = np.isfinite(fm); idx = np.where(a2)[0]
    if len(idx) < 6:
        return None
    lo2, hi2 = idx[0], idx[-1]; span = fm[lo2:hi2 + 1].copy()
    nanm = ~np.isfinite(span)
    if nanm.any():
        g = np.where(~nanm)[0]; span[nanm] = np.interp(np.where(nanm)[0], g, span[g])
    pitch = float(span.mean()); sc = span - pitch
    shape = np.interp(np.linspace(0, 1, N_RESAMPLE), np.linspace(0, 1, len(sc)), sc).astype(np.float32)
    jump = float(np.abs(np.diff(sc, 2)).sum()) if len(sc) > 2 else 0.0
    return img.astype(np.float32), pitch, shape, float(hi2 - lo2 + 1), jump


def build():
    imgs, pitch, shapes, dur, jump, cohort = [], [], [], [], [], []
    t0 = time.time()
    for c in COHORTS:
        p = MP / f"{c}_focus/patches.npz"
        if not p.exists():
            print("[WARN] missing", p); continue
        z = np.load(p); arr = z["patches"]; freqs = z["freqs_kHz"][BAND0:BAND1]
        print(f"[INFO] build {c}: {arr.shape[0]}", flush=True)
        for i in range(arr.shape[0]):
            r = register_one(arr[i, BAND0:BAND1, :], freqs)
            if r is not None:
                imgs.append(r[0]); pitch.append(r[1]); shapes.append(r[2]); dur.append(r[3]); jump.append(r[4]); cohort.append(c)
        del arr, z
    imgs = np.stack(imgs).astype(np.float32)
    np.savez_compressed(CACHE, images=imgs, pitch=np.array(pitch, np.float32),
                        shapes=np.stack(shapes).astype(np.float32), duration=np.array(dur, np.float32),
                        jump=np.array(jump, np.float32), cohort=np.array(cohort).astype(str))
    print(f"[INFO] built {len(imgs)} registered images in {(time.time()-t0)/60:.1f} min -> {CACHE}", flush=True)


class ImgVAE(nn.Module):
    def __init__(self, z=32):
        super().__init__()
        def cb(i, o): return nn.Sequential(nn.Conv2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1))
        self.enc = nn.Sequential(cb(1, 32), cb(32, 64), cb(64, 128), cb(128, 256))  # 64->4
        self.mu = nn.Linear(256 * 16, z); self.lv = nn.Linear(256 * 16, z); self.fd = nn.Linear(z, 256 * 16)
        def db(i, o): return nn.Sequential(nn.ConvTranspose2d(i, o, 4, 2, 1), nn.BatchNorm2d(o), nn.LeakyReLU(0.1))
        self.dec = nn.Sequential(db(256, 128), db(128, 64), db(64, 32), nn.ConvTranspose2d(32, 1, 4, 2, 1))

    def encode(self, x): h = self.enc(x).flatten(1); return self.mu(h), self.lv(h)
    def decode(self, z): return torch.sigmoid(self.dec(self.fd(z).view(-1, 256, 4, 4)))

    def forward(self, x):
        mu, lv = self.encode(x); zz = mu + torch.randn_like(lv) * (0.5 * lv).exp()
        return self.decode(zz), mu, lv


def grad(x):
    return torch.diff(x, 1, -1)[..., :-1, :], torch.diff(x, 1, -2)[..., :, :-1]


def loss_fn(x, xh, beta, mu, lv):
    mask = (x > 0.05).float()
    full = F.mse_loss(xh, x)
    masked = ((xh - x) ** 2 * mask).sum() / (mask.sum() + 1e-6)
    gxh, gyh = grad(xh); gx, gy = grad(x)
    edge = F.mse_loss(gxh, gx) + F.mse_loss(gyh, gy)
    kl = -0.5 * torch.mean(1 + lv - mu ** 2 - lv.exp())
    return 0.2 * full + 1.0 * masked + 1.0 * edge + beta * kl


def eta2(v, lab):
    v = v if v.ndim == 2 else v[:, None]; keep = lab >= 0; v, lab = v[keep], lab[keep]
    g = v.mean(0); tot = float(((v - g) ** 2).sum())
    w = sum(float(((v[lab == l] - v[lab == l].mean(0)) ** 2).sum()) for l in np.unique(lab))
    return 1 - w / tot if tot > 0 else 0.0


def train(a):
    d = np.load(CACHE, allow_pickle=True)
    imgs = d["images"]; pitch, dur, jump, cv_shapes, cohort = d["pitch"], d["duration"], d["jump"], d["shapes"], d["cohort"]
    dev = a.device if torch.cuda.is_available() else "cpu"
    X = torch.from_numpy(imgs).unsqueeze(1)
    n = len(X); print(f"[INFO] train on {n} images {imgs.shape[1:]} dev={dev}", flush=True)
    m = ImgVAE(a.zdim).to(dev); opt = torch.optim.Adam(m.parameters(), 1e-3)
    idx = np.random.permutation(n); ntr = int(0.9 * n); tr = idx[:ntr]
    Xtr = X[tr].to(dev)
    for ep in range(a.epochs):
        m.train(); perm = torch.randperm(len(Xtr), device=dev)
        for s in range(0, len(Xtr), a.batch):
            b = Xtr[perm[s:s + a.batch]]; xh, mu, lv = m(b)
            loss = loss_fn(b, xh, a.beta, mu, lv)
            opt.zero_grad(); loss.backward(); opt.step()
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  ep{ep:3d} loss={float(loss):.4f}", flush=True)
    m.eval()
    with torch.no_grad():
        Z = torch.cat([m.encode(X[s:s + 512].to(dev))[0].cpu() for s in range(0, n, 512)]).numpy()
    lab = KMeans(20, n_init=10, random_state=42).fit_predict(Z)
    # chevron/valley from the aligned ridge shapes
    Sh = cv_shapes; N = Sh.shape[1]; lo, hi = int(.2 * N), int(.8 * N)
    pk = Sh.argmax(1); tr_i = Sh.argmin(1); emax = np.maximum(Sh[:, 0], Sh[:, -1]); emin = np.minimum(Sh[:, 0], Sh[:, -1])
    cv = np.array(["other"] * len(Sh), object)
    cv[(pk >= lo) & (pk <= hi) & (Sh.max(1) - emax > 2)] = "chevron"
    cv[(tr_i >= lo) & (tr_i <= hi) & (emin - Sh.min(1) > 2)] = "valley"
    sel = cv != "other"
    row = dict(method="M10_image_vae", pitch=eta2(pitch[:, None], lab), duration=eta2(dur[:, None], lab),
               shape=eta2(Sh, lab), curvature=eta2(jump[:, None], lab), cv_nmi=float(nmi(cv[sel], lab[sel])), n=int(n))
    print("\n===== M10 IMAGE-VAE SCORECARD =====")
    print("  pitch %.3f | dur %.3f | shape %.3f | curv %.3f | CV-NMI %.3f  (bar: shape 0.577)"
          % (row["pitch"], row["duration"], row["shape"], row["curvature"], row["cv_nmi"]), flush=True)
    np.save(OUT / "latent_m10.npy", Z); (OUT / "score_m10.json").write_text(json.dumps(row, indent=2))
    # gallery: mean registered IMAGE per cluster (shows what 2-D structure each holds)
    uniq = sorted([c for c in np.unique(lab)], key=lambda c: -(lab == c).sum())[:20]
    fig, axes = plt.subplots(4, 5, figsize=(10, 8))
    for ax in axes.ravel(): ax.axis("off")
    for i, c in enumerate(uniq):
        mm = np.where(lab == c)[0]; axes[i // 5][i % 5].imshow(imgs[mm].mean(0), origin="lower", cmap="magma", aspect="auto")
        axes[i // 5][i % 5].set_title(f"c{c} n={len(mm)}", fontsize=7)
    fig.suptitle("M10 image-VAE — mean registered patch per cluster", fontsize=10)
    fig.tight_layout(); fig.savefig(OUT / "m10.png", dpi=110); plt.close(fig)
    print("[DONE] M10", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zdim", type=int, default=32); ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--device", default="cuda:0"); ap.add_argument("--rebuild", action="store_true")
    a = ap.parse_args()
    torch.manual_seed(42); np.random.seed(42)
    if a.rebuild or not CACHE.exists():
        build()
    train(a)


if __name__ == "__main__":
    main()

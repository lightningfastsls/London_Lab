"""Pathway B (rig, GPU) — contrastive 2-D shape encoder with pitch/time-shift
invariance augmentation.  "M9 done right."

The one un-falsified VAE-family idea for clustering USVs by *geometric shape*.
Every reconstruction objective tried so far loses to free registration because
it spends capacity on pitch/position pixel-variance:
    production contour-VAE  shape eta2 0.099 (a pitch/duration sorter)
    denoised retrain        shape eta2 0.081 (input sparsity was NOT the cause)
    M10 image-VAE + edge    shape eta2 0.009 (pixel-grad != dF/dt; destructive crop)
    M8  1-D VAE + deriv      shape eta2 0.42  (deriv term did nothing post-register)
    registration + k-means   shape eta2 0.58-0.75  (the ceiling to beat)
    M9  1-D contrastive      shape eta2 0.344 (best LEARNED; but 1-D ridge, warp/noise aug)

Pathway B = M9's two fixes:
  (1) input is the 2-D DENOISED image (not the 1-D ridge) -> jumps/sub-harmonics survive,
  (2) the positive pair differs by PITCH-SHIFT + TIME-SHIFT (+ optional time-warp),
      not time-warp/noise -> the encoder is *forced* to learn that absolute pitch and
      position do not matter.  This is dF/dt's invariance realised through the data.

A contrastive objective has NO reconstruction term, so it never has to represent
pitch/position at all -- the structural reason it can succeed where recon VAEs failed.

Encoder-only by design: no decoder, no KL, no reconstruction (the hybrid is the
sibling B+A handoff).  Scoring axes (pitch/shape/dur/curvature/CV-NMI) match
M9/M10/R1 so the leaderboard comparison is apples-to-apples.

Run on the rig (`/data/shachar/contour_vae` is canonical; NOT /data/mickey_london_lab).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Corpus band is used only for a sanity print; the in-band clamp is derived
# per-patch from energy span (the patch rows already map to the corpus band via
# freqs_kHz, see manifest freq_min/max_kHz).  Import the canonical constants --
# never redeclare them (a missing src path just drops the print, not a fallback
# literal that could drift from corpus.py).
try:  # keep importable for tests on the box even if src is not on sys.path
    from usv_spectrogram.corpus import USV_FREQ_MIN_HZ, USV_FREQ_MAX_HZ
except Exception:  # pragma: no cover - import convenience only
    USV_FREQ_MIN_HZ = USV_FREQ_MAX_HZ = None


# ---------------------------------------------------------------------------
# Model: 2-D conv encoder (reuses the ImageVAE backbone 1->32->64->128->256)
# + a SimCLR projection head.  AdaptiveAvgPool makes it input-size agnostic so
# we feed the native (257, 234) patch without M10's destructive resize/crop.
# ---------------------------------------------------------------------------
class ContrastiveEncoder(nn.Module):
    def __init__(self, embed_dim: int = 128, proj_dim: int = 64) -> None:
        super().__init__()

        def block(i: int, o: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Conv2d(i, o, kernel_size=4, stride=2, padding=1),
                nn.BatchNorm2d(o),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.backbone = nn.Sequential(
            block(1, 32), block(32, 64), block(64, 128), block(128, 256),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)            # size-agnostic -> (B,256,1,1)
        self.embed = nn.Linear(256, embed_dim)
        self.proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(inplace=True),
            nn.Linear(embed_dim, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.pool(self.backbone(x)).flatten(1)     # (B, 256)
        emb = self.embed(h)                            # (B, embed_dim)
        return emb, self.proj(emb)                     # cluster on emb, contrast on proj


# ---------------------------------------------------------------------------
# NT-Xent (SimCLR).  Mirrors M9's `ntxent` so scores are directly comparable.
# ---------------------------------------------------------------------------
def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, tau: float = 0.2) -> torch.Tensor:
    z = F.normalize(torch.cat([z1, z2], dim=0), dim=1)
    n = z1.shape[0]
    sim = z @ z.T / tau
    sim.fill_diagonal_(-1e9)
    tgt = torch.cat([torch.arange(n, 2 * n), torch.arange(0, n)]).to(z.device)
    return F.cross_entropy(sim, tgt)


# ---------------------------------------------------------------------------
# Augmentation = the invariance.  pitch-shift (vertical) + time-shift
# (horizontal) + time-warp (horizontal scale), via a single grid_sample with
# zero padding (no wraparound).  Pitch/time shifts are INTEGER pixels (exact,
# no interpolation blur); only the warp is fractional.  The per-sample vertical
# shift is CLAMPED to the call's own energy span so a call can never be pushed
# off the band edge -- this is the "ridge stays inside 20-120 kHz" guarantee.
# ---------------------------------------------------------------------------
def augment(
    x: torch.Tensor,
    freqs_khz: torch.Tensor,
    *,
    max_df_khz: float = 15.0,
    max_dt_frames: int = 20,
    warp_lo: float = 0.9,
    warp_hi: float = 1.1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """x: (B,1,H,W) >=0 spectrogram patch.  freqs_khz: (H,) ascending kHz/row."""
    B, _, H, W = x.shape
    dev = x.device

    def rand(*shape: int) -> torch.Tensor:
        return torch.rand(*shape, device=dev, generator=generator)

    # row freq-resolution (kHz per row) -> max pitch shift in integer rows
    freqs_khz = freqs_khz.to(dev)
    df_per_row = float((freqs_khz[-1] - freqs_khz[0]) / max(H - 1, 1))
    max_drows = int(round(max_df_khz / df_per_row)) if df_per_row > 0 else 0

    # raw per-sample draws (rounded to integer pixels for pitch/time)
    dy = torch.round((rand(B) * 2 - 1) * max_drows).long()        # +dy moves content DOWN in row idx
    dx = torch.round((rand(B) * 2 - 1) * max_dt_frames).long()    # +dx moves content RIGHT in col idx
    s = warp_lo + rand(B) * (warp_hi - warp_lo)                   # time-warp scale (>1 = wider)

    # USV band rows (corpus 20-120 kHz) within this patch's frequency axis.  The
    # denoised patches are the FULL 0-150 kHz STFT (257 bins), so the USV band is
    # a sub-range (rows ~35-205); a pitch-shift must keep the call inside it --
    # "never push a call out of band" (handoff).  Verified: real call energy lives
    # in 20-120 kHz.  Fall back to the full frame [0,H) if the band can't be
    # resolved or a call already spills out of band (preserves the no-clip guarantee).
    lo_khz = USV_FREQ_MIN_HZ / 1e3 if USV_FREQ_MIN_HZ else float(freqs_khz.min())
    hi_khz = USV_FREQ_MAX_HZ / 1e3 if USV_FREQ_MAX_HZ else float(freqs_khz.max())
    band = ((freqs_khz >= lo_khz) & (freqs_khz <= hi_khz)).nonzero().flatten()
    b_lo = int(band.min()) if band.numel() else 0
    b_hi = int(band.max()) if band.numel() else H - 1

    # content rows [r0, r1] must satisfy floor <= r0+dy and r1+dy <= ceil.
    # Sentinels are SCALARS so torch.where broadcasts cleanly to (B,H) for any H,B
    # (a (B,)-shaped sentinel breaks broadcasting whenever H != B).
    lit = (x[:, 0] > 1e-6).any(dim=2)                             # (B,H) rows holding energy
    rows = torch.arange(H, device=dev)[None, :]                   # (1,H)
    has = lit.any(dim=1)                                          # (B,)
    sent_hi = torch.tensor(H, device=dev, dtype=torch.long)       # scalar: empty-row min sentinel
    sent_lo = torch.tensor(-1, device=dev, dtype=torch.long)      # scalar: empty-row max sentinel
    r0 = torch.where(lit, rows, sent_hi).min(dim=1).values        # first lit row (H if none) (B,)
    r1 = torch.where(lit, rows, sent_lo).max(dim=1).values        # last lit row (-1 if none) (B,)
    inband = has & (r0 >= b_lo) & (r1 <= b_hi)                    # in-band call -> clamp to band
    floor = torch.where(inband, torch.full_like(r0, b_lo), torch.zeros_like(r0))
    ceil = torch.where(inband, torch.full_like(r1, b_hi), torch.full_like(r1, H - 1))
    lo = torch.where(has, floor - r0, torch.zeros_like(r0))       # dy >= lo
    hi = torch.where(has, ceil - r1, torch.zeros_like(r1))        # dy <= hi
    dy = torch.maximum(torch.minimum(dy, hi), lo)

    # build per-sample sampling grid (align_corners=True): for output (i,j) sample
    # source (src_row, src_col); shift content => sample from (i - dy, (j - dx)/s).
    ii = torch.arange(H, device=dev).float()
    jj = torch.arange(W, device=dev).float()
    gy = (ii[None, :, None] - dy[:, None, None].float())          # (B,H,1) source rows
    gx = ((jj[None, None, :] - dx[:, None, None].float())         # (B,1,W) source cols
          / s[:, None, None])
    # normalise to [-1, 1]
    ny = (2.0 * gy / max(H - 1, 1) - 1.0).expand(B, H, W)
    nx = (2.0 * gx / max(W - 1, 1) - 1.0).expand(B, H, W)
    grid = torch.stack([nx, ny], dim=-1)                          # (B,H,W,2): last dim (x,y)
    return F.grid_sample(x, grid, mode="bilinear",
                         padding_mode="zeros", align_corners=True)


# ---------------------------------------------------------------------------
# eta2: between-cluster variance fraction (mirror of M9/M10 — for the in-script
# quick scorecard; the full gates live in eval_shape_encoder.py).
# ---------------------------------------------------------------------------
def eta2(v: np.ndarray, lab: np.ndarray) -> float:
    v = v if v.ndim == 2 else v[:, None]
    keep = lab >= 0
    v, lab = v[keep], lab[keep]
    if len(v) == 0:                      # all labels < 0: nothing to score (avoid empty-mean warning)
        return 0.0
    g = v.mean(0)
    tot = float(((v - g) ** 2).sum())
    if tot <= 0:
        return 0.0
    w = sum(float(((v[lab == l] - v[lab == l].mean(0)) ** 2).sum()) for l in np.unique(lab))
    return 1 - w / tot


# ---------------------------------------------------------------------------
def main() -> None:
    R = Path("/data/shachar/contour_vae")
    ap = argparse.ArgumentParser()
    ap.add_argument("--patches", default=str(R / "results/denoised_patches/combined_denoised/patches.npz"))
    ap.add_argument("--out", default=str(R / "results/latent_transitions/b_contrastive"))
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--proj-dim", type=int, default=64)
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--max-df-khz", type=float, default=15.0)
    ap.add_argument("--max-dt-frames", type=int, default=20)
    ap.add_argument("--warp-lo", type=float, default=0.9)
    ap.add_argument("--warp-hi", type=float, default=1.1)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="debug: cap #patches (0=all)")
    a = ap.parse_args()

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); np.random.seed(a.seed)
    dev = a.device if torch.cuda.is_available() else "cpu"

    band_str = (f"{USV_FREQ_MIN_HZ/1e3:.0f}-{USV_FREQ_MAX_HZ/1e3:.0f}kHz"
                if USV_FREQ_MIN_HZ is not None and USV_FREQ_MAX_HZ is not None
                else "from-patch-freqs (corpus import unavailable)")
    print(f"[PARAM] B-contrastive embed={a.embed_dim} proj={a.proj_dim} temp={a.temp} "
          f"epochs={a.epochs} batch={a.batch} lr={a.lr} dev={dev}", flush=True)
    print(f"[PARAM] aug: pitch +/-{a.max_df_khz}kHz  time +/-{a.max_dt_frames}fr  "
          f"warp [{a.warp_lo},{a.warp_hi}]  band={band_str}", flush=True)

    z = np.load(a.patches, mmap_mode="r")
    patches = z["patches"]                       # (N,257,234) float32, memmap
    freqs = torch.from_numpy(np.asarray(z["freqs_kHz"], dtype=np.float32))
    N = patches.shape[0] if a.limit == 0 else min(a.limit, patches.shape[0])
    H, W = patches.shape[1], patches.shape[2]
    print(f"[INFO] patches {patches.shape} using N={N}  freqs {float(freqs[0]):.1f}-{float(freqs[-1]):.1f}kHz",
          flush=True)

    # train/val split (seeded, saved so eval scores the SAME held-out rows)
    rng = np.random.default_rng(a.seed)
    perm = rng.permutation(N)
    n_val = int(a.val_frac * N)
    val_idx = np.sort(perm[:n_val]); tr_idx = np.sort(perm[n_val:])
    np.savez(out / "split_idx.npz", train=tr_idx, val=val_idx, N=N)
    print(f"[INFO] split train={len(tr_idx)} val={len(val_idx)}", flush=True)

    def load_batch(idx: np.ndarray) -> torch.Tensor:
        # sort for contiguous memmap access; pair identity is preserved because
        # both augmented views are built from this same loaded batch (NT-Xent
        # pairs by position i<->i+B, independent of the original shuffle order).
        b = np.asarray(patches[np.sort(idx)], dtype=np.float32)          # (B,H,W)
        t = torch.from_numpy(b).unsqueeze(1)                             # (B,1,H,W)
        mx = t.amax(dim=(2, 3), keepdim=True).clamp_min(1e-6)
        return (t / mx).to(dev)                                          # per-patch [0,1]

    model = ContrastiveEncoder(a.embed_dim, a.proj_dim).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)
    gen = torch.Generator(device=dev); gen.manual_seed(a.seed)
    aug_kw = dict(max_df_khz=a.max_df_khz, max_dt_frames=a.max_dt_frames,
                  warp_lo=a.warp_lo, warp_hi=a.warp_hi)
    fr = freqs.to(dev)

    t0 = time.time()
    for ep in range(a.epochs):
        model.train()
        ep_perm = tr_idx[rng.permutation(len(tr_idx))]
        tot, nb = 0.0, 0
        for s0 in range(0, len(ep_perm), a.batch):
            xb = load_batch(ep_perm[s0:s0 + a.batch])
            if xb.shape[0] < 8:
                continue
            _, p1 = model(augment(xb, fr, generator=gen, **aug_kw))
            _, p2 = model(augment(xb, fr, generator=gen, **aug_kw))
            loss = nt_xent_loss(p1, p2, a.temp)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  ep{ep:3d} ntxent={tot/max(nb,1):.4f}  ({(time.time()-t0)/60:.1f} min)", flush=True)

    # encode ALL patches (deterministic embedding, no augmentation)
    model.eval()
    embs = np.zeros((N, a.embed_dim), dtype=np.float32)
    with torch.no_grad():
        for s0 in range(0, N, 1024):
            idx = np.arange(s0, min(s0 + 1024, N))
            e, _ = model(load_batch(idx))
            embs[idx] = e.cpu().numpy()
    np.save(out / "embeddings.npy", embs)
    torch.save(model.state_dict(), out / "encoder.pt")

    # quick in-script scorecard on the registered-ridge cache, val split only
    desc_p = R / "results/eval_shape/desc_denoised.npz"
    if desc_p.exists():
        from sklearn.cluster import KMeans
        d = np.load(desc_p)
        row = d["row"]                                    # patch indices that survived ridge extraction
        keep = np.isin(row, val_idx)
        rr = row[keep]
        Zval = embs[rr]
        lab = KMeans(20, n_init=10, random_state=a.seed).fit_predict(Zval)
        sc = dict(
            method="B_contrastive", n=int(len(rr)),
            pitch=eta2(d["pitch"][keep], lab),
            duration=eta2(d["duration"][keep], lab),
            shape=eta2(d["shapes"][keep], lab),
            curvature=eta2(d["jump"][keep], lab),
        )
        print("\n===== B-CONTRASTIVE QUICK SCORECARD (val split) =====")
        print("  pitch %.3f | dur %.3f | shape %.3f | curv %.3f   "
              "(bars: shape registration 0.58-0.75, M9 0.344, production 0.099)"
              % (sc["pitch"], sc["duration"], sc["shape"], sc["curvature"]), flush=True)
        (out / "score_b_quick.json").write_text(json.dumps(sc, indent=2))
    else:
        print(f"[WARN] {desc_p} absent -> skipping quick scorecard; run eval_shape_encoder.py", flush=True)

    print(f"[DONE] B-contrastive -> {out}", flush=True)


if __name__ == "__main__":
    sys.exit(main())

"""Train the global-code 1-D VQ-VAE on registered USV ridges, dump codes+latents.

Usage:
    .venv/bin/python scripts/experiments/vqvae_shape/train_vq_shape.py \
        --ridges data/shape_substrate/true_registered_ridges.npz \
        --meta   data/shape_substrate/true_registered_ridges_meta.npz \
        --num-codes 20 --dim 64 --epochs 120 \
        --out results/vqvae_shape/k20

Prints every parameter, the codebook-health trajectory (perplexity + dead
codes per epoch), and writes:
    codes.parquet   -- per-call: wav_stem, call_id, cohort, code index, all
                       bundled shape descriptors (for eta^2/NMI scoring)
    latents.npy     -- (N, dim) continuous pre-quantization latents (for kNN)
    model.pt        -- weights + config + input normalization stats
    train_log.json  -- per-epoch loss/perplexity/dead-code counts
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd
import torch

from vq_shape_model import VQVAE1D


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--ridges", required=True)
    p.add_argument("--meta", required=True)
    p.add_argument("--num-codes", type=int, default=20)
    p.add_argument("--dim", type=int, default=64)
    p.add_argument("--epochs", type=int, default=120)
    p.add_argument("--batch", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--commitment", type=float, default=0.25)
    p.add_argument("--decay", type=float, default=0.99)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    return p.parse_args()


def main():
    a = parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(a.seed)
    np.random.seed(a.seed)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ridges = np.load(a.ridges, allow_pickle=True)
    meta = np.load(a.meta, allow_pickle=True)
    X = ridges["shapes"].astype(np.float32)                 # (N, 50)
    N, L = X.shape

    # Global standardization (preserves shape; the raw curves span ~+-75 kHz).
    mu, sd = float(X.mean()), float(X.std())
    Xn = (X - mu) / sd

    print("=" * 66)
    print("GLOBAL-CODE 1-D VQ-VAE  --  USV shape clustering (Mickey test)")
    print("=" * 66)
    print(f"  device         : {dev}")
    print(f"  ridges         : {a.ridges}  -> X {X.shape}")
    print(f"  input norm     : global z-score  mu={mu:.4f} sd={sd:.4f}")
    print(f"  num_codes (K)  : {a.num_codes}")
    print(f"  latent dim     : {a.dim}")
    print(f"  epochs/batch   : {a.epochs} / {a.batch}   lr={a.lr}")
    print(f"  commitment/dec : {a.commitment} / {a.decay}   seed={a.seed}")
    print("=" * 66)

    Xt = torch.from_numpy(Xn).to(dev)
    model = VQVAE1D(num_codes=a.num_codes, dim=a.dim, length=L,
                    commitment=a.commitment, decay=a.decay).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr)

    log = []
    order = np.arange(N)
    for ep in range(a.epochs):
        model.train()
        np.random.shuffle(order)
        tot, rec, com, ppl = 0.0, 0.0, 0.0, 0.0
        nb = 0
        for i in range(0, N, a.batch):
            b = torch.as_tensor(order[i:i + a.batch], device=dev)
            xb = Xt[b]
            out = model(xb)
            opt.zero_grad()
            out["loss"].backward()
            opt.step()
            tot += out["loss"].item(); rec += out["recon_loss"].item()
            com += out["commit_loss"].item(); ppl += out["perplexity"].item(); nb += 1
        # Codebook health: reset dead codes from a fresh encoder pass.
        model.eval()
        with torch.no_grad():
            z_pool = model.enc(Xt[torch.as_tensor(order[:4096], device=dev)])
        n_dead = model.vq.reset_dead_codes(z_pool)
        row = {"epoch": ep, "loss": tot / nb, "recon": rec / nb,
               "commit": com / nb, "perplexity": ppl / nb, "dead_codes": n_dead}
        log.append(row)
        if ep % 10 == 0 or ep == a.epochs - 1:
            print(f"  ep {ep:3d} | loss {row['loss']:.4f} | recon {row['recon']:.4f} "
                  f"| commit {row['commit']:.4f} | perplexity {row['perplexity']:5.2f}"
                  f"/{a.num_codes} | dead {n_dead}")

    # Final assignment pass.
    model.eval()
    idx_all, z_all = [], []
    with torch.no_grad():
        for i in range(0, N, a.batch):
            xb = Xt[i:i + a.batch]
            idx, z_e = model.encode(xb)
            idx_all.append(idx.cpu().numpy()); z_all.append(z_e.cpu().numpy())
    codes = np.concatenate(idx_all); latents = np.concatenate(z_all)

    used = len(np.unique(codes))
    counts = np.bincount(codes, minlength=a.num_codes)
    print("-" * 66)
    print(f"  codes used     : {used}/{a.num_codes}")
    print(f"  cluster sizes  : min {counts.min()} / median {int(np.median(counts))} "
          f"/ max {counts.max()}")

    # Assemble per-call table with all bundled descriptors for scoring.
    df = pd.DataFrame({
        "wav_stem": meta["wav_stem"].astype(str),
        "call_id": meta["call_id"].astype(int),
        "cohort": meta["cohort"].astype(str),
        "vq_code": codes.astype(int),
        "pitch": ridges["pitch"], "duration": ridges["duration"],
        "jump": ridges["jump"], "chevron_valley": ridges["chevron_valley"].astype(str),
        "lab_shape_kmeans": ridges["lab_shape"].astype(int),   # registration k20 baseline
    })
    df.to_parquet(os.path.join(a.out, "codes.parquet"))
    np.save(os.path.join(a.out, "latents.npy"), latents)
    torch.save({"state_dict": model.state_dict(),
                "config": vars(a), "norm": {"mu": mu, "sd": sd}, "length": L},
               os.path.join(a.out, "model.pt"))
    with open(os.path.join(a.out, "train_log.json"), "w") as f:
        json.dump({"args": vars(a), "log": log,
                   "codes_used": int(used), "cluster_counts": counts.tolist()}, f, indent=2)
    print(f"  wrote          : {a.out}/{{codes.parquet, latents.npy, model.pt, train_log.json}}")
    print("=" * 66)


if __name__ == "__main__":
    main()

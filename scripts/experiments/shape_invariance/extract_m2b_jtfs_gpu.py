"""STAGE B (rig, GPU 0) — proper Joint Time-Frequency Scattering on call waveforms.

This is the principled M2b arm the original handoff named: kymatio
`TimeFrequencyScattering` (JTFS), which only exists in kymatio >=0.4 (shipped to
the rig as a pure-Python package dir; the box-pinned 0.3.0 lacks it). It applies a
frequential wavelet + a 2-D (time x log-frequency) wavelet transform to the raw
1-D waveform, giving FREQUENCY-TRANSPOSITION invariance NATIVELY via the F-lowpass
(the very invariance the 7 prior learned-pixel VAEs never achieved). T=N collapses
the time axis to one frame -> a fixed-length per-call descriptor.

Runs on cuda:0 (the free GPU). Batched to fit 8 GB. Extracts forward AND
time-reversed (seg[:, ::-1]) features for the reversal unit test, for each config
in a small F-sweep (F = transposition-invariance scale, the knob the handoff names).

Run on rig:
  PYTHONPATH=<kymatio_main_dir> /data/mickey_london_lab/.venv/bin/python \
      extract_m2b_jtfs_gpu.py --seg-dir <dir> --out <dir> --device cuda:0
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np


# JTFS configs. J/Q/J_fr/Q_fr fixed at principled defaults; F swept (the
# frequency-transposition-invariance scale the handoff calls out). T=N -> 1 frame.
CONFIGS = [
    {"name": "F2", "J": 13, "Q": 8, "J_fr": 3, "Q_fr": 1, "F": 2},
    {"name": "F4", "J": 13, "Q": 8, "J_fr": 3, "Q_fr": 1, "F": 4},
]


def _import_jtfs():
    """Import TimeFrequencyScattering, shimming the scipy `sph_harm` removal.

    kymatio.torch eagerly imports the scattering3d frontend, which references the
    deprecated `scipy.special.sph_harm` (removed in newer scipy on the rig; renamed
    sph_harm_y). We never use 3-D harmonic scattering, so a dummy alias satisfies the
    dead import without affecting any JTFS math.
    """
    import scipy.special as _sp
    if not hasattr(_sp, "sph_harm"):
        _sp.sph_harm = getattr(_sp, "sph_harm_y", None)
    from kymatio.torch import TimeFrequencyScattering
    return TimeFrequencyScattering


def _jtfs_batch(sig, cfg, N, device, batch, log):
    import torch
    TimeFrequencyScattering = _import_jtfs()

    jtfs = TimeFrequencyScattering(
        J=cfg["J"], J_fr=cfg["J_fr"], Q=cfg["Q"], Q_fr=cfg["Q_fr"],
        shape=N, T=N, F=cfg["F"], format="time", backend="torch",
    ).to(device)

    outs = []
    n = sig.shape[0]
    b = batch
    i = 0
    while i < n:
        while True:
            try:
                xb = torch.from_numpy(sig[i:i + b]).float().to(device)
                with torch.no_grad():
                    Sb = jtfs(xb)                      # (b, C, 1)
                Sb = Sb.reshape(Sb.shape[0], -1).cpu().numpy()
                outs.append(Sb)
                i += b
                break
            except RuntimeError as e:
                if "out of memory" in str(e).lower() and b > 1:
                    torch.cuda.empty_cache()
                    b = max(1, b // 2)
                    log(f"    OOM -> batch={b}")
                    continue
                raise
        if log and (i % (b * 8) == 0 or i >= n):
            log(f"    {cfg['name']}: {min(i, n)}/{n} (batch={b})")
    return np.concatenate(outs, 0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seg-dir", required=True, help="dir with segments_fwd.npy + jtfs_meta.npz")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    def log(*a):
        print(*a, flush=True)

    import torch
    log("=" * 90)
    log("STAGE B — proper JTFS on call waveforms (rig GPU)")
    log("=" * 90)
    log(f"device={args.device} cuda_name={torch.cuda.get_device_name(0)} "
        f"free={torch.cuda.mem_get_info()[0] / 1e9:.2f}GB")

    fwd = np.load(os.path.join(args.seg_dir, "segments_fwd.npy"))
    N = fwd.shape[1]
    rev = fwd[:, ::-1].copy()
    # SR is read from the box-rendered meta (canonical SAMPLE_RATE_HZ); used only
    # for a human-readable ms annotation here -- no DSP is done at this stage.
    _meta = np.load(os.path.join(args.seg_dir, "jtfs_meta.npz"), allow_pickle=True)
    _sr = int(_meta["SR"])
    log(f"segments fwd={fwd.shape} N={N} ({N / _sr * 1000:.1f} ms @ SR={_sr})")

    manifest = {"N": int(N), "device": args.device, "configs": []}
    for cfg in CONFIGS:
        log(f"\n--- config {cfg} ---")
        t = time.time()
        Ff = _jtfs_batch(fwd, cfg, N, args.device, args.batch, log)
        log(f"  fwd features {Ff.shape} in {time.time() - t:.1f}s")
        t = time.time()
        Fr = _jtfs_batch(rev, cfg, N, args.device, args.batch, log)
        log(f"  rev features {Fr.shape} in {time.time() - t:.1f}s")
        np.save(os.path.join(args.out, f"jtfs_{cfg['name']}_fwd.npy"), Ff)
        np.save(os.path.join(args.out, f"jtfs_{cfg['name']}_rev.npy"), Fr)
        manifest["configs"].append({**cfg, "d": int(Ff.shape[1])})

    json.dump(manifest, open(os.path.join(args.out, "jtfs_manifest.json"), "w"), indent=2)
    log(f"\n[OUT] {args.out}  (features + jtfs_manifest.json)")
    log("STAGE B done.")


if __name__ == "__main__":
    main()

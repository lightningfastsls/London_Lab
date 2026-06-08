"""STAGE A (box, CPU) — render the 611 labeled-call WAVEFORM segments for M2b JTFS.

The proper joint time-frequency scattering transform (`TimeFrequencyScattering`,
kymatio >=0.4) operates on a 1-D SIGNAL, not a spectrogram image. So unlike the
Scattering2D substitute (`m2b_jtfs.py`, which rendered a (res,res) spectrogram),
the principled arm feeds JTFS the raw call WAVEFORM and lets the transform's own
frequential wavelet + F-lowpass supply frequency-transposition invariance NATIVELY.

This stage does ONLY the cheap, I/O-bound part on the box:
  - identical 611-row selection + call windows as `m2b_jtfs.main()`
    (loader.load_labeled join offset -1, drop 'unclear'; start = META abs_time_start_s),
  - crop a FIXED-LENGTH window N=2**17 samples (437 ms @ 300 kHz) starting at
    (start - pad). N covers the longest labeled call window (max ~79.2k samples
    incl. 2x15 ms pad) with zero truncation -> no call is clipped.
  - native SR kept (NO resampling): absolute Hz preserved across calls, which is
    what makes JTFS's transposition invariance meaningful.
  - per-call peak-amplitude normalization (loudness removed; shape kept).

Output (shipped to the rig for the GPU JTFS pass):
  $OUT/segments_fwd.npy   float32 (611, 131072)   ~320 MB
  $OUT/jtfs_meta.npz      rows, family, cohort, stratum, side, duration_ms,
                          slopes (net dF over the registered 50-pt contour), wav_stem
The time-reversed segments are derived on the rig (seg[:, ::-1]) to halve transfer.

Run:
  .venv/bin/python scripts/experiments/shape_invariance/extract_m2b_jtfs_segments.py \
      --out /home/shachar/.claude/jobs/60a0acf7/tmp/jtfs_stageA
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
from scipy.io import wavfile

_HERE = os.path.dirname(os.path.abspath(__file__))
_EXP = os.path.dirname(_HERE)               # scripts/experiments
_ROOT = os.path.dirname(os.path.dirname(_EXP))
for p in (_EXP, _ROOT):
    if p not in sys.path:
        sys.path.insert(0, p)

from shape_invariance import loader                      # noqa: E402
from shape_invariance.methods import m2b_jtfs as m2b     # noqa: E402  (no kymatio at import)
from src.usv_spectrogram import corpus as C              # noqa: E402

SR = int(C.SAMPLE_RATE_HZ)        # 300000
N = 2 ** 17                       # 131072 samples = 436.9 ms ; covers max call window
PAD_S = 0.015                     # identical to m2b render window left edge


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=N)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    Nfix = int(args.n)

    print("=" * 90)
    print("STAGE A — M2b JTFS waveform segment render (box, CPU)")
    print("=" * 90)
    print(f"PARAMS: N={Nfix} samples ({Nfix / SR * 1000:.1f} ms) SR={SR} pad_s={PAD_S}")

    data = loader.load_labeled()
    rows = data["rows"]
    Nrows = len(rows)
    m = np.load(loader.META_NPZ, allow_pickle=True)
    ws = m["wav_stem"].astype(str)[rows]
    starts = m["abs_time_start_s"].astype(float)[rows]
    durs = data["duration_ms"]
    family = data["family"]
    slopes = np.array([m2b.net_slope(c) for c in data["contour50"]], dtype=np.float64)
    print(f"DATA: N={Nrows} labeled rows; cohorts="
          f"{dict(zip(*np.unique(data['cohort'], return_counts=True)))}")
    print(f"family counts = {dict(zip(*np.unique(family, return_counts=True)))}")

    idx = m2b.build_wav_index(".")
    miss = sorted({s for s in ws if s not in idx})
    print(f"WAV coverage: {Nrows - len([s for s in ws if s in miss])}/{Nrows} "
          f"(missing stems={len(miss)})")
    if miss:
        print(f"BLOCKER: {len(miss)} labeled calls lack a local WAV; sample={miss[:5]}")
        sys.exit(2)

    # Group calls by unique WAV so each (possibly large) recording is read once.
    segs = np.zeros((Nrows, Nfix), dtype=np.float32)
    by_wav: dict[str, list[int]] = {}
    for i, s in enumerate(ws):
        by_wav.setdefault(s, []).append(i)
    print(f"unique WAV files = {len(by_wav)} (avg {Nrows / len(by_wav):.1f} calls/file)")

    n_clipped = 0
    for w_i, (stem, members) in enumerate(by_wav.items()):
        sr, wav = wavfile.read(idx[stem])
        wav = np.asarray(wav)
        if wav.ndim > 1:
            wav = wav[:, 0]
        wav = wav.astype(np.float32)
        assert sr == SR, f"{stem}: sr={sr} != {SR}"
        for i in members:
            s0 = max(0.0, float(starts[i]) - PAD_S)
            a = int(round(s0 * SR))
            seg = wav[a:a + Nfix]
            call_end = a + int(round((durs[i] / 1000.0 + 2 * PAD_S) * SR))
            if call_end > a + Nfix:
                n_clipped += 1
            if len(seg) < Nfix:
                seg = np.pad(seg, (0, Nfix - len(seg)))
            peak = float(np.abs(seg).max())
            if peak > 1e-9:
                seg = seg / peak
            segs[i] = seg.astype(np.float32)
        if (w_i + 1) % 50 == 0:
            print(f"  read {w_i + 1}/{len(by_wav)} wavs")

    print(f"clipped calls (window > N): {n_clipped} (expect 0)")
    fwd_path = os.path.join(args.out, "segments_fwd.npy")
    np.save(fwd_path, segs)
    meta_path = os.path.join(args.out, "jtfs_meta.npz")
    np.savez(
        meta_path,
        rows=rows,
        family=family.astype(str),
        cohort=data["cohort"].astype(str),
        stratum=data["stratum"].astype(str),
        side=data["side"].astype(np.float64),
        duration_ms=durs.astype(np.float64),
        slopes=slopes,
        wav_stem=ws.astype(str),
        N=np.int64(Nfix),
        SR=np.int64(SR),
        pad_s=np.float64(PAD_S),
    )
    mb = os.path.getsize(fwd_path) / 1e6
    print(f"[OUT] {fwd_path}  shape={segs.shape}  {mb:.1f} MB")
    print(f"[OUT] {meta_path}")
    print("STAGE A done.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Download one VocalMat OSF recording + its ground-truth, render a sample of
GT-annotated calls RAW vs DENOISED (SIS prefilter) into an HTML gallery.

Purpose: let the user (a) see VocalMat calls rendered in OUR denoised domain,
(b) judge whether the noise/vocalization GT labels look trustworthy. This is a
viewing/validation step before committing to download all 7 recordings for a
matched-domain noise-gate training set.

Audio + GT live at https://osf.io/bk2uj/ (Audios/ + Audios/Ground truth/).
The SIS denoiser is the archived `prefilter_spectrogram` (Stack 3) — the same
pre-filter used before the production / denoised-retrain VAE.
"""
from __future__ import annotations
import io
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.io import wavfile
from scipy.signal import stft

# --- archived SIS denoiser ---
ARCHIVE_SRC = Path("archive/cleaning_legacy/stack3/src")
sys.path.insert(0, str(ARCHIVE_SRC))
from features.spectrogram_filter import FilterConfig, prefilter_spectrogram  # noqa: E402

WAV_DL = "https://osf.io/download/s864e/"          # 1303.WAV
GT_DL_NAME = "1303_GT.xlsx"
REC = "1303"
CACHE = Path("/home/shachar/.claude/jobs/6f04ea1f/tmp/vm")
OUT = Path("results/vocalmat_gt_sample")

# STFT for VocalMat's native ~250 kHz audio (viewing resolution).
N_FFT = 1024
HOP = 256
WIN_PRE_S = 0.015
WIN_POST_S = 0.185          # 0.2 s window around each GT onset
N_VOCAL = 40
N_NOISE = 20
SEED = 1729
BAND_MIN_KHZ, BAND_MAX_KHZ = 0, 125


def fetch(url: str, dst: Path) -> None:
    if dst.exists() and dst.stat().st_size > 0:
        print(f"  cached: {dst} ({dst.stat().st_size/1e6:.0f} MB)")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"  downloading {url} -> {dst} ...")
    urllib.request.urlretrieve(url, dst)
    print(f"  done ({dst.stat().st_size/1e6:.0f} MB)")


def get_gt_download_link() -> str:
    import json
    def get(u):
        with urllib.request.urlopen(u, timeout=40) as r:
            return json.load(r)
    root = get("https://api.osf.io/v2/nodes/bk2uj/files/osfstorage/")
    hrefs = {f['attributes']['name']: f['relationships']['files']['links']['related']['href']
             for f in root['data'] if f['attributes']['kind'] == 'folder'}
    aud = get(hrefs['Audios'])
    gt = [c for c in aud['data'] if c['attributes']['name'] == 'Ground truth'][0]
    page = get(gt['relationships']['files']['links']['related']['href'])
    for c in page['data']:
        if c['attributes']['name'] == GT_DL_NAME:
            return c['links']['download']
    raise RuntimeError("GT xlsx not found")


def render_panel(ax, mag, freqs_hz, title):
    band = (freqs_hz >= BAND_MIN_KHZ * 1000) & (freqs_hz <= BAND_MAX_KHZ * 1000)
    m = mag[band, :]
    disp = np.log1p(m)
    ax.imshow(disp, origin="lower", aspect="auto", cmap="magma",
              extent=[0, disp.shape[1], freqs_hz[band][0] / 1000, freqs_hz[band][-1] / 1000])
    ax.set_title(title, fontsize=8)
    ax.set_ylabel("kHz", fontsize=7)
    ax.tick_params(labelsize=6)


def main() -> None:
    wav_path = CACHE / f"{REC}.WAV"
    gt_path = CACHE / GT_DL_NAME
    print("=== fetch audio + GT ===")
    fetch(WAV_DL, wav_path)
    fetch(get_gt_download_link(), gt_path)

    sr, audio = wavfile.read(wav_path)
    if audio.ndim > 1:
        audio = audio[:, 0]
    print(f"\nWAV {REC}: sr={sr} Hz  samples={len(audio)}  dur={len(audio)/sr:.1f}s  dtype={audio.dtype}")

    gt = pd.read_excel(gt_path)
    counts = gt["GT"].value_counts().to_dict()
    print(f"GT rows={len(gt)}  label counts={counts}")
    print(f"STFT: n_fft={N_FFT} hop={HOP}  window={WIN_PRE_S+WIN_POST_S:.3f}s  "
          f"sample: {N_VOCAL} vocalization + {N_NOISE} noise  seed={SEED}")

    rng = np.random.default_rng(SEED)
    picks = []
    for label, n in [("vocalization", N_VOCAL), ("noise", N_NOISE)]:
        sub = gt[gt["GT"] == label]
        take = min(n, len(sub))
        idx = rng.choice(sub.index.to_numpy(), size=take, replace=False)
        picks.extend((label, gt.loc[i, "Start_time"]) for i in idx)
    print(f"rendering {len(picks)} calls ...")

    OUT.mkdir(parents=True, exist_ok=True)
    img_dir = OUT / "img"
    img_dir.mkdir(exist_ok=True)
    cfg = FilterConfig(sample_rate=int(sr), freq_min_hz=25_000.0, freq_max_hz=120_000.0)

    rows = []
    for k, (label, t0) in enumerate(picks):
        a = int(max(0, (t0 - WIN_PRE_S) * sr))
        b = int(min(len(audio), (t0 + WIN_POST_S) * sr))
        seg = audio[a:b].astype(np.float64)
        if seg.size < N_FFT:
            continue
        f, _t, Z = stft(seg, fs=sr, nperseg=N_FFT, noverlap=N_FFT - HOP)
        mag = np.abs(Z)
        cleaned, _mask = prefilter_spectrogram(mag, f, cfg)

        fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.2))
        render_panel(axes[0], mag, f, "RAW")
        render_panel(axes[1], cleaned, f, "DENOISED")
        fig.suptitle(f"{label}  |  t0={t0:.3f}s", fontsize=8)
        fig.tight_layout(rect=[0, 0, 1, 0.92])
        fn = f"{label}__{k:03d}__t{t0:.3f}.png"
        fig.savefig(img_dir / fn, dpi=90)
        plt.close(fig)
        rows.append((label, fn))

    # HTML
    css = ("body{font-family:system-ui,Arial;background:#111;color:#eee;margin:0;padding:18px}"
           "h2{border-bottom:2px solid #444;padding-bottom:5px;margin-top:30px}"
           ".grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:8px}"
           ".cell{background:#1b1b1b;border:1px solid #333;border-radius:4px;padding:3px}"
           ".cell img{width:100%;display:block}.cap{font-size:10px;color:#8c8;margin-top:2px}")
    parts = [f"<!doctype html><meta charset='utf-8'><title>VocalMat {REC} GT sample</title>",
             f"<style>{css}</style>",
             f"<h1>VocalMat recording {REC} &mdash; GT calls, RAW vs DENOISED (SIS prefilter)</h1>",
             f"<p>sr={sr} Hz, {len(picks)} sampled calls. Left=raw STFT, right=our denoiser. "
             "Judge: (1) does denoising make calls easier to read? (2) are the noise/USV GT labels right?</p>"]
    for label in ("vocalization", "noise"):
        sel = [fn for lb, fn in rows if lb == label]
        parts.append(f"<h2>{label} <span style='color:#fc6'>({len(sel)})</span></h2><div class='grid'>")
        for fn in sel:
            parts.append(f"<div class='cell'><img loading='lazy' src='img/{fn}'>"
                         f"<div class='cap'>{fn}</div></div>")
        parts.append("</div>")
    (OUT / "index.html").write_text("\n".join(parts))
    print(f"\nGallery: {OUT/'index.html'}  ({len(rows)} calls rendered)")


if __name__ == "__main__":
    main()

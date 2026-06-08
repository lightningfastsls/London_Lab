#!/usr/bin/env python
"""Patch-duration sweep on the held-out-844 (lab) set, scored with the v1
VocalMat-anchored ResNet-18 (results/lab_classifier_v1/best.pt).

Hypothesis under test (vault note "patch-duration mismatch ... kappa 0.13"):
shrinking the analysis window so a short call FILLS more of the 227x227 patch
should improve the VocalMat-trained classifier's transfer to our data.

Design
------
- Re-extract each of the 844 held-out calls CALL-CENTERED at several window
  durations, reusing the EXACT production rendering chain from
  cnn_prepare_training_data.py (resample 300->250k -> Hamming/256/128/1024 STFT
  -> clean_spectrogram(baseline_mode='percentile') -> min-max uint8 -> 227^2
  bilinear -> 3ch). Cleaning config matches the 18.2b training patches.
- Score with v1 (plain timm ResNet-18, num_classes=12). Collapse argmax==Noise
  (index 0) -> noise else usv, compare to the manifest usv_verdict.
- VALIDATION GATE: also score the ORIGINAL pre-extracted 844 patches with v1
  (BASELINE-A). My 0.22 s re-extraction should land close to BASELINE-A; if it
  does not, my extraction is unfaithful and the sweep is not trustworthy.
"""
from __future__ import annotations
import os, sys, glob, json
from pathlib import Path
import numpy as np, pandas as pd
import torch
from PIL import Image
from torchvision import transforms

REPO = Path("/home/shachar/projects/mickey_london_lab")
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))

from usv_spectrogram.classifier import CleaningConfig, clean_spectrogram  # noqa
from usv_spectrogram.classifier.resample import (  # noqa
    SOURCE_SAMPLE_RATE_HZ, TARGET_SAMPLE_RATE_HZ, resample_to_vocalmat)
from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES  # noqa
from usv_spectrogram.classifier.model import build_resnet18_classifier  # noqa
# reuse production rendering helpers verbatim:
from cnn_prepare_training_data import (  # noqa
    _spectrogram_db, _spec_to_uint8_patch, _VOCALMAT_STFT_HOP)

NOISE_IDX = GRIMSLEY_12_CLASSES.index("Noise")
CKPT = REPO / "results/lab_classifier_v1/best.pt"
MANIFEST = REPO / "data/lab_cnn_training/held_out_844/manifest.csv"
WAV_ROOT = REPO / "USV_lab_131204_chunked_2s_full"
DURATIONS = [0.22, 0.14, 0.08, 0.05]
OUT = Path(os.environ["CLAUDE_JOB_DIR"]) / "sweep_out"
OUT.mkdir(parents=True, exist_ok=True)

_TF = transforms.Compose([
    transforms.Resize((227, 227)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
])


def load_model():
    blob = torch.load(CKPT, map_location="cpu")
    sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    m = build_resnet18_classifier(num_classes=12, pretrained=False)
    missing, unexpected = m.load_state_dict(sd, strict=False)
    if missing:   print(f"  [load] missing keys: {len(missing)} (first: {list(missing)[:3]})")
    if unexpected:print(f"  [load] unexpected keys: {len(unexpected)} (first: {list(unexpected)[:3]})")
    return m.eval()


@torch.no_grad()
def score_pils(model, pils):
    """Score a list of PIL images, return argmax-12 array."""
    out = []
    for i in range(0, len(pils), 64):
        batch = torch.stack([_TF(p.convert("L")) for p in pils[i:i+64]])
        logits = model(batch)
        out.append(logits.argmax(-1).numpy())
    return np.concatenate(out)


def metrics(argmax12, gt_is_noise):
    pred_noise = (argmax12 == NOISE_IDX)
    gtn = gt_is_noise.astype(bool); gtu = ~gtn; prn = pred_noise; pru = ~prn
    tn = int((gtn & prn).sum()); tu = int((gtu & pru).sum())
    nrec = tn / max(1, gtn.sum()); urec = tu / max(1, gtu.sum())
    nprec = tn / max(1, prn.sum()); uprec = tu / max(1, pru.sum())
    f1n = (2*nprec*nrec/(nprec+nrec)) if (nprec+nrec) else 0.0
    f1u = (2*uprec*urec/(uprec+urec)) if (uprec+urec) else 0.0
    return dict(noise_recall=nrec, usv_recall=urec, noise_prec=nprec, usv_prec=uprec,
                balanced_acc=(nrec+urec)/2, macro_f1=(f1n+f1u)/2,
                pct_pred_usv=float(pru.mean()),
                conf=dict(noise_noise=tn, noise_usv=int((gtn&pru).sum()),
                          usv_noise=int((gtu&prn).sum()), usv_usv=tu))


def reextract_pils(df, duration_s, cleaned_cache):
    """Call-centered re-extraction at duration_s for every row. Returns list[PIL]."""
    fpp = max(1, int(round(duration_s * TARGET_SAMPLE_RATE_HZ / _VOCALMAT_STFT_HOP)))
    half = fpp // 2
    import soundfile as sf
    pils = []
    for _, r in df.iterrows():
        stem = r["wav_stem"]
        if stem not in cleaned_cache:
            wav = WAV_PATHS[stem]
            s, sr = sf.read(str(wav), dtype="float32")
            if s.ndim > 1: s = s[:, 0]
            s250 = resample_to_vocalmat(s) if sr == SOURCE_SAMPLE_RATE_HZ else s.astype(np.float32)
            spec = _spectrogram_db(s250, TARGET_SAMPLE_RATE_HZ)
            cleaned_cache[stem] = clean_spectrogram(
                spec, CleaningConfig(baseline_mode="percentile"), recording_id=stem)
        cleaned = cleaned_cache[stem]
        nt = cleaned.shape[1]
        center = int(round((r["det_start_s"] + r["det_end_s"]) / 2 * TARGET_SAMPLE_RATE_HZ / _VOCALMAT_STFT_HOP))
        slab = np.full((cleaned.shape[0], fpp), float(cleaned.min()), dtype=cleaned.dtype)
        lo, hi = center - half, center - half + fpp
        vlo, vhi = max(0, lo), min(nt, hi)
        if vhi > vlo:
            slab[:, (vlo - lo):(vhi - lo)] = cleaned[:, vlo:vhi]
        gray = _spec_to_uint8_patch(slab)
        pils.append(Image.fromarray(np.stack([gray]*3, -1), mode="RGB"))
    return pils


print("=" * 72)
print("PATCH-DURATION SWEEP — held-out-844 (lab), v1 VocalMat-anchored ResNet-18")
print(f"  checkpoint    : {CKPT.relative_to(REPO)}")
print(f"  cleaning cfg  : CleaningConfig(baseline_mode='percentile')  [soft-notch no-op]")
print(f"  durations (s) : {DURATIONS}")
print(f"  collapse      : argmax==Noise(idx {NOISE_IDX}) -> noise, else usv")
print("=" * 72)

df = pd.read_csv(MANIFEST)
gt_noise = (df["usv_verdict"].astype(str).str.lower() == "noise").to_numpy().astype(int)
print(f"set: {len(df)} calls | {int((gt_noise==0).sum())} usv / {int(gt_noise.sum())} noise "
      f"({(gt_noise==0).mean():.1%} usv) | unique chunks: {df['wav_stem'].nunique()}")

WAV_PATHS = {os.path.splitext(os.path.basename(p))[0]: p
             for p in glob.glob(str(WAV_ROOT / "**/*.wav"), recursive=True)}

model = load_model()

# ---- BASELINE-A: score the ORIGINAL pre-extracted patches with v1 ----
orig_pils = []
for _, r in df.iterrows():
    p = REPO / r["path"]
    orig_pils.append(Image.open(p).copy())
mA = metrics(score_pils(model, orig_pils), gt_noise)
print(f"\n[BASELINE-A] original pre-extracted patches (v1): "
      f"noise_rec={mA['noise_recall']:.3f} usv_rec={mA['usv_recall']:.3f} "
      f"bal_acc={mA['balanced_acc']:.3f} macroF1={mA['macro_f1']:.3f} pct_usv={mA['pct_pred_usv']:.3f}")

# ---- SWEEP ----
results = {"baseline_A_original_patches": mA, "sweep": {}}
cache = {}
for D in DURATIONS:
    pils = reextract_pils(df, D, cache)
    m = metrics(score_pils(model, pils), gt_noise)
    results["sweep"][f"{D:.2f}s"] = m
    flag = "  <-- matches BASELINE-A (extraction faithful)" if abs(D-0.22) < 1e-9 else ""
    print(f"[{D:.2f}s] noise_rec={m['noise_recall']:.3f} usv_rec={m['usv_recall']:.3f} "
          f"bal_acc={m['balanced_acc']:.3f} macroF1={m['macro_f1']:.3f} "
          f"pct_usv={m['pct_pred_usv']:.3f}{flag}")

(OUT / "sweep_results.json").write_text(json.dumps(results, indent=2))
print(f"\nWrote {OUT/'sweep_results.json'}")

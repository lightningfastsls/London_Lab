"""α₃-C Phase A4 — label spectrogram patches with the lab_classifier_v1 oracle.

The α₃ pivot (docs/handoffs/2026-05-28_alpha3-vocalmat-blocker-and-pivot.md)
selected variant **α₃-C**: instead of the published VocalMat AlexNet (the
``Mdl_categorical_DL.mat`` blocker — undecodable MATLAB v7.0 MCOS opaque object),
use our in-house reproduction ``results/lab_classifier_v1/best.pt`` (plain timm
ResNet-18, 12-class VocalMat-anchored, val 0.77 macro F1) as the substrate-
independent labeling oracle for shape-VAE evaluation.

This script is the reusable A4 stage:
  - now:   run it on data/lab_cnn_training/held_out_844/manifest.csv to reproduce
           the known ~0.81 usv/noise balanced accuracy → proves the oracle loads
           and infers correctly BEFORE we spend rig time rendering ~40k patches.
  - later: run it on the A3 render output (data/alpha3_patches/manifest.csv) to
           assign VocalMat-taxonomy labels to our Stack-4-cleaned 131204 patches.

Inference convention is transcribed verbatim from the v1 reproduction harness
(archive/cleaning_legacy/stack1/scripts/experiments/patch_duration_sweep.py):
PIL→grayscale→Resize(227,227)→Grayscale(3ch)→ToTensor, argmax over 12 classes,
index-0 ("Noise") collapses to a noise verdict.

Usage::

    PYTHONPATH=src .venv/bin/python scripts/experiments/label_patches_v1.py \\
        --manifest data/lab_cnn_training/held_out_844/manifest.csv \\
        --output results/alpha3/labels_v1_held_out_844.csv

NOT-to-touch: this script only READS results/lab_classifier_v1/best.pt; it never
writes to the model dir, corpus.py, Stack 4, or the production detection pipeline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES  # noqa: E402
from usv_spectrogram.classifier.model import build_resnet18_classifier  # noqa: E402

NOISE_IDX = GRIMSLEY_12_CLASSES.index("Noise")
HIGH_CONF_THRESHOLD = 0.85  # roadmap A4: eval-gold filter

# Verbatim from patch_duration_sweep.py (the v1 reproduction harness).
_TF = transforms.Compose(
    [
        transforms.Resize((227, 227)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
    ]
)


def load_oracle(ckpt: Path, device: str) -> torch.nn.Module:
    """Load lab_classifier_v1 (plain ResNet-18, num_classes=12)."""
    blob = torch.load(ckpt, map_location="cpu")
    state_dict = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    model = build_resnet18_classifier(num_classes=12, pretrained=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [load] missing keys: {len(missing)} (first: {list(missing)[:3]})")
    if unexpected:
        print(f"  [load] unexpected keys: {len(unexpected)} (first: {list(unexpected)[:3]})")
    return model.eval().to(device)


@torch.no_grad()
def score_paths(model: torch.nn.Module, paths: list[Path], device: str, batch_size: int) -> np.ndarray:
    """Return an (N, 12) softmax-probability matrix for the given PNG paths."""
    probs = []
    for i in range(0, len(paths), batch_size):
        batch_paths = paths[i : i + batch_size]
        batch = torch.stack([_TF(Image.open(p).convert("L")) for p in batch_paths]).to(device)
        logits = model(batch)
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    return np.concatenate(probs, axis=0)


def usv_noise_smoke(argmax12: np.ndarray, verdict: pd.Series) -> dict:
    """2×2 usv/noise smoke metrics when the manifest carries a usv_verdict column.

    Collapse argmax==Noise(idx 0) → noise, else usv; compare to the human verdict.
    Used only to confirm the oracle is sound (expected balanced acc ~0.81 on the
    844-row held-out set). NOT part of the α₃ eval gold itself.
    """
    gt_noise = (verdict.astype(str).str.lower() == "noise").to_numpy().astype(bool)
    pred_noise = argmax12 == NOISE_IDX
    gtn, gtu = gt_noise, ~gt_noise
    prn, pru = pred_noise, ~pred_noise
    tn = int((gtn & prn).sum())
    tu = int((gtu & pru).sum())
    noise_recall = tn / max(1, int(gtn.sum()))
    usv_recall = tu / max(1, int(gtu.sum()))
    return {
        "n": int(len(argmax12)),
        "n_usv_gt": int(gtu.sum()),
        "n_noise_gt": int(gtn.sum()),
        "noise_recall": noise_recall,
        "usv_recall": usv_recall,
        "balanced_accuracy": (noise_recall + usv_recall) / 2.0,
        "pct_pred_usv": float(pru.mean()),
        "confusion": {
            "noise_noise": tn,
            "noise_usv": int((gtn & pru).sum()),
            "usv_noise": int((gtu & prn).sum()),
            "usv_usv": tu,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", type=Path, required=True,
                    help="CSV with a 'path' column of PNG paths (relative to repo root or absolute).")
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "results/lab_classifier_v1/best.pt")
    ap.add_argument("--output", type=Path, required=True,
                    help="Output labels CSV.")
    ap.add_argument("--path-column", default="path")
    ap.add_argument("--id-column", default=None,
                    help="Optional manifest column to carry through as call_id (e.g. wav_stem).")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--high-conf-threshold", type=float, default=HIGH_CONF_THRESHOLD)
    args = ap.parse_args()

    print("=" * 72)
    print("α₃-C Phase A4 — label patches with lab_classifier_v1 oracle")
    print("=" * 72)
    print(f"  manifest        : {args.manifest}")
    print(f"  checkpoint      : {args.checkpoint}")
    print(f"  device          : {args.device}")
    print(f"  high-conf thresh: {args.high_conf_threshold}")
    print(f"  classes (12)    : {list(GRIMSLEY_12_CLASSES)}")
    print(f"  Noise index     : {NOISE_IDX}")

    df = pd.read_csv(args.manifest)
    if args.path_column not in df.columns:
        raise SystemExit(f"manifest has no '{args.path_column}' column; columns={list(df.columns)}")

    def resolve(p: str) -> Path:
        pp = Path(p)
        return pp if pp.is_absolute() else (REPO_ROOT / pp)

    paths = [resolve(p) for p in df[args.path_column].tolist()]
    missing_files = [p for p in paths if not p.exists()]
    if missing_files:
        raise SystemExit(f"{len(missing_files)} PNG paths do not exist (first: {missing_files[:3]})")
    print(f"  patches         : {len(paths)}")

    model = load_oracle(args.checkpoint, args.device)
    probs = score_paths(model, paths, args.device, args.batch_size)
    argmax12 = probs.argmax(axis=1)
    top1_prob = probs.max(axis=1)

    out = pd.DataFrame(
        {
            "path": df[args.path_column].tolist(),
            "top1_class": [GRIMSLEY_12_CLASSES[i] for i in argmax12],
            "top1_idx": argmax12,
            "top1_prob": top1_prob,
            "high_confidence": top1_prob >= args.high_conf_threshold,
            "softmax_12": [json.dumps([round(float(x), 5) for x in row]) for row in probs],
        }
    )
    if args.id_column and args.id_column in df.columns:
        out.insert(0, "call_id", df[args.id_column].tolist())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)
    print(f"\n  wrote {len(out)} rows → {args.output}")

    # 12-class distribution
    print("\n  top-1 class distribution:")
    dist = out["top1_class"].value_counts()
    for cls in GRIMSLEY_12_CLASSES:
        n = int(dist.get(cls, 0))
        print(f"    {cls:<16} {n:>5}  ({n / len(out):.1%})")
    n_hc = int(out["high_confidence"].sum())
    print(f"\n  high-confidence (≥{args.high_conf_threshold}): {n_hc} ({n_hc / len(out):.1%})")

    # Oracle smoke test if a usv_verdict column is present.
    if "usv_verdict" in df.columns:
        m = usv_noise_smoke(argmax12, df["usv_verdict"])
        print("\n  --- ORACLE SMOKE (usv/noise collapse vs human verdict) ---")
        print(f"  set: {m['n']} | {m['n_usv_gt']} usv / {m['n_noise_gt']} noise")
        print(f"  noise_recall = {m['noise_recall']:.3f}   usv_recall = {m['usv_recall']:.3f}")
        print(f"  BALANCED ACCURACY = {m['balanced_accuracy']:.3f}   (expected ~0.81 on held_out_844)")
        print(f"  pct predicted usv = {m['pct_pred_usv']:.3f}")
        print(f"  confusion (gt→pred): {m['confusion']}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""Real held-out-844 evaluator for the Module 18.4 DANN classifier (v2).

Why this exists
---------------
``src/usv_spectrogram/classifier/training.py::_evaluate_held_out_845`` is a
PLACEHOLDER — it computes a label-vs-verdict proxy and never loads patches or
runs the model. ``training.py`` is on the 18.4 "do NOT touch" list (it is the
18.3 non-DANN reference), so the real patch-loading evaluator is built here as
an additive script rather than by editing that file.

What it does
------------
1. Loads the 844 held-out lab patches via ``manifest.csv`` (column
   ``usv_verdict`` ∈ {usv, noise}; 705 usv / 139 noise = 83.5 % usv).
2. Runs v2 inference with the EXACT v2 preprocessing
   (Resize(227) → Grayscale(3ch) → ToTensor, no Normalize) and the EXACT v2
   model (``ResNet18DANN(num_classes=12, num_domains=2)``).
3. Collapses the 12-way syllable output to binary usv/noise:
   ``argmax == GRIMSLEY_12_CLASSES.index("Noise")`` (index 0) → noise, else usv.
4. Reports **per-class** metrics (noise recall / specificity / precision,
   usv recall / precision, balanced accuracy, macro-F1) + a 2×2 confusion
   matrix + the 12-class argmax breakdown split by ground-truth verdict.

Why per-class, not pooled accuracy
-----------------------------------
The set is 83.5 % usv. A trivial "always usv" classifier scores 0.835 pooled
accuracy — *above* the old 0.80 gate. Pooled accuracy is therefore meaningless
here; the signal is whether the model can actually identify the 139 noise
patches (noise recall) without sacrificing usv recall.

Usage (rig)
-----------
    PYTHONPATH=src python scripts/evaluate_held_out_844.py \\
        --manifest data/lab_cnn_training/held_out_844/manifest.csv \\
        --checkpoint results/lab_classifier_v2/best.pt \\
        --output-dir results/lab_classifier_v2/ \\
        --device cuda
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
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

REPO_ROOT = Path(__file__).resolve().parents[1]
# Make the package importable whether or not it is pip-installed.
sys.path.insert(0, str(REPO_ROOT / "src"))

from usv_spectrogram.classifier.dann import ResNet18DANN  # noqa: E402
from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES  # noqa: E402

NUM_CLASSES = len(GRIMSLEY_12_CLASSES)
NOISE_INDEX = GRIMSLEY_12_CLASSES.index("Noise")  # 0 — the collapse pivot
DEFAULT_IMAGE_SIZE = 227  # matches train_lab_classifier._DEFAULT_IMAGE_SIZE


def _resolve_patch_path(raw: str, roots: list[Path]) -> Path:
    """Resolve a manifest patch path: absolute first, then each candidate root."""
    p = Path(raw)
    if p.is_absolute():
        return p
    for root in roots:
        cand = root / p
        if cand.exists():
            return cand
    return roots[0] / p


class HeldOutDataset(Dataset):
    """Loads the 844 held-out patches + binary usv/noise ground truth.

    Reproduces the v2 ``ManifestDataset`` transform exactly so inference matches
    training-time preprocessing. Returns ``(image_tensor, gt_is_noise:int)``.
    """

    def __init__(self, manifest: Path, image_size: int = DEFAULT_IMAGE_SIZE):
        df = pd.read_csv(manifest)
        for col in ("path", "usv_verdict"):
            if col not in df.columns:
                raise ValueError(f"{manifest} missing required column {col!r}; got {list(df.columns)}")
        verdict = df["usv_verdict"].astype(str).str.lower()
        bad = set(verdict.unique()) - {"usv", "noise"}
        if bad:
            raise ValueError(f"usv_verdict has unexpected values {bad}; expected usv/noise")
        self._df = df.reset_index(drop=True)
        self._gt_is_noise = (verdict == "noise").to_numpy().astype(int)
        self._roots = [REPO_ROOT, manifest.resolve().parent]
        # EXACT v2 preprocessing (train_lab_classifier_v2.ManifestDataset).
        self._transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int):
        row = self._df.iloc[idx]
        path = _resolve_patch_path(str(row["path"]), self._roots)
        with Image.open(path) as im:
            tensor = self._transform(im.convert("L"))
        return tensor, int(self._gt_is_noise[idx])


def _load_v2_model(ckpt_path: Path, device: torch.device) -> ResNet18DANN:
    """Build ResNet18DANN and load the v2 ``best.pt`` state_dict."""
    blob = torch.load(ckpt_path, map_location="cpu")
    state_dict = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob
    # pretrained=False: the trained weights come entirely from the checkpoint.
    model = ResNet18DANN(num_classes=NUM_CLASSES, num_domains=2, pretrained=False)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"WARNING: {len(missing)} missing keys when loading v2 checkpoint "
              f"(first few: {list(missing)[:5]})", file=sys.stderr)
    if unexpected:
        print(f"WARNING: {len(unexpected)} unexpected keys "
              f"(first few: {list(unexpected)[:5]})", file=sys.stderr)
    return model.to(device).eval()


@torch.no_grad()
def _run_inference(model, loader, device):
    argmax12, gt_noise = [], []
    for images, gt in loader:
        images = images.to(device)
        class_logits, _, _ = model(images, lambda_=0.0)
        argmax12.append(class_logits.argmax(dim=-1).cpu().numpy())
        gt_noise.append(gt.numpy())
    return np.concatenate(argmax12), np.concatenate(gt_noise)


def _safe_div(num: float, den: float) -> float:
    return float(num) / float(den) if den else 0.0


def _binary_metrics(pred_is_noise: np.ndarray, gt_is_noise: np.ndarray) -> dict:
    """Per-class metrics treating {usv, noise} as two classes. NOT pooled-only."""
    gt_noise = gt_is_noise.astype(bool)
    gt_usv = ~gt_noise
    pr_noise = pred_is_noise.astype(bool)
    pr_usv = ~pr_noise

    # 2x2 confusion (rows = true, cols = pred)
    tn_noise = int(np.sum(gt_noise & pr_noise))   # noise → noise (correct noise)
    noise_as_usv = int(np.sum(gt_noise & pr_usv))  # noise → usv (missed noise)
    usv_as_noise = int(np.sum(gt_usv & pr_noise))  # usv → noise (false noise)
    tp_usv = int(np.sum(gt_usv & pr_usv))          # usv → usv (correct usv)

    n_noise = int(gt_noise.sum())
    n_usv = int(gt_usv.sum())
    pred_noise_total = int(pr_noise.sum())
    pred_usv_total = int(pr_usv.sum())

    noise_recall = _safe_div(tn_noise, n_noise)      # = specificity for the usv task
    usv_recall = _safe_div(tp_usv, n_usv)            # = sensitivity for the usv task
    noise_precision = _safe_div(tn_noise, pred_noise_total)
    usv_precision = _safe_div(tp_usv, pred_usv_total)

    def _f1(p, r):
        return _safe_div(2 * p * r, p + r)

    f1_noise = _f1(noise_precision, noise_recall)
    f1_usv = _f1(usv_precision, usv_recall)

    total = n_noise + n_usv
    return {
        "n_total": total,
        "n_usv": n_usv,
        "n_noise": n_noise,
        "frac_usv": _safe_div(n_usv, total),
        "confusion": {
            "true_noise_pred_noise": tn_noise,
            "true_noise_pred_usv": noise_as_usv,
            "true_usv_pred_noise": usv_as_noise,
            "true_usv_pred_usv": tp_usv,
        },
        "noise_recall_specificity": noise_recall,
        "noise_precision": noise_precision,
        "noise_f1": f1_noise,
        "usv_recall_sensitivity": usv_recall,
        "usv_precision": usv_precision,
        "usv_f1": f1_usv,
        "balanced_accuracy": (noise_recall + usv_recall) / 2.0,
        "macro_f1_binary": (f1_noise + f1_usv) / 2.0,
        "pooled_accuracy_MISLEADING": _safe_div(tp_usv + tn_noise, total),
        "always_usv_baseline_pooled": _safe_div(n_usv, total),
    }


def _twelve_class_breakdown(argmax12: np.ndarray, gt_is_noise: np.ndarray) -> dict:
    """Argmax distribution over the 12 syllable classes, split by GT verdict."""
    out = {}
    for label, mask in (("true_usv", gt_is_noise == 0), ("true_noise", gt_is_noise == 1)):
        counts = np.bincount(argmax12[mask], minlength=NUM_CLASSES)
        out[label] = {GRIMSLEY_12_CLASSES[i]: int(counts[i]) for i in range(NUM_CLASSES)}
    return out


def _write_report(metrics: dict, breakdown: dict, ckpt: Path, out_md: Path) -> None:
    c = metrics["confusion"]
    lines = [
        "# Held-out 844 — real patch-loading evaluation (Module 18.4 bonus)",
        "",
        f"**Checkpoint:** `{ckpt}`",
        f"**Set:** {metrics['n_total']} patches "
        f"({metrics['n_usv']} usv / {metrics['n_noise']} noise = "
        f"{metrics['frac_usv']:.1%} usv)",
        f"**Collapse rule:** 12-class argmax == 'Noise' (index {NOISE_INDEX}) → noise, else usv.",
        "",
        "> Pooled accuracy is reported but **misleading** on an 83.5%-usv set: a "
        "trivial 'always usv' classifier already scores "
        f"{metrics['always_usv_baseline_pooled']:.4f}. Read the **per-class** rows.",
        "",
        "## Per-class metrics",
        "",
        "| Class | Recall | Precision | F1 |",
        "|---|---|---|---|",
        f"| **noise** (recall = specificity) | {metrics['noise_recall_specificity']:.4f} "
        f"| {metrics['noise_precision']:.4f} | {metrics['noise_f1']:.4f} |",
        f"| **usv** (recall = sensitivity) | {metrics['usv_recall_sensitivity']:.4f} "
        f"| {metrics['usv_precision']:.4f} | {metrics['usv_f1']:.4f} |",
        "",
        f"- **Balanced accuracy:** {metrics['balanced_accuracy']:.4f}",
        f"- **Macro-F1 (binary):** {metrics['macro_f1_binary']:.4f}",
        f"- Pooled accuracy (misleading): {metrics['pooled_accuracy_MISLEADING']:.4f} "
        f"vs always-usv baseline {metrics['always_usv_baseline_pooled']:.4f}",
        "",
        "## Confusion (rows = true, cols = predicted)",
        "",
        "| true \\ pred | noise | usv |",
        "|---|---|---|",
        f"| **noise** | {c['true_noise_pred_noise']} | {c['true_noise_pred_usv']} |",
        f"| **usv** | {c['true_usv_pred_noise']} | {c['true_usv_pred_usv']} |",
        "",
        "## 12-class argmax breakdown by ground-truth verdict",
        "",
        "| Syllable class | true_usv | true_noise |",
        "|---|---|---|",
    ]
    for cls in GRIMSLEY_12_CLASSES:
        lines.append(f"| {cls} | {breakdown['true_usv'][cls]} | {breakdown['true_noise'][cls]} |")
    lines.append("")
    out_md.write_text("\n".join(lines))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path,
                    default=REPO_ROOT / "data/lab_cnn_training/held_out_844/manifest.csv")
    ap.add_argument("--checkpoint", type=Path,
                    default=REPO_ROOT / "results/lab_classifier_v2/best.pt")
    ap.add_argument("--output-dir", type=Path,
                    default=REPO_ROOT / "results/lab_classifier_v2")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    args = ap.parse_args()

    for pth, what in ((args.manifest, "manifest"), (args.checkpoint, "checkpoint")):
        if not pth.exists():
            print(f"ERROR: --{what} not found: {pth}", file=sys.stderr)
            return 2

    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: --device cuda requested but no GPU; using cpu", file=sys.stderr)
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print("=" * 70)
    print("Held-out 844 evaluation — parameters")
    print(f"  manifest    : {args.manifest}")
    print(f"  checkpoint  : {args.checkpoint}")
    print(f"  device      : {device}")
    print(f"  batch_size  : {args.batch_size}   image_size: {args.image_size}")
    print(f"  classes     : {NUM_CLASSES}  (Noise index = {NOISE_INDEX})")
    print(f"  collapse    : argmax == {NOISE_INDEX} ('Noise') -> noise, else usv")
    print("=" * 70)

    ds = HeldOutDataset(args.manifest, image_size=args.image_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers)
    model = _load_v2_model(args.checkpoint, device)

    argmax12, gt_is_noise = _run_inference(model, loader, device)
    pred_is_noise = (argmax12 == NOISE_INDEX).astype(int)

    metrics = _binary_metrics(pred_is_noise, gt_is_noise)
    breakdown = _twelve_class_breakdown(argmax12, gt_is_noise)

    print("\n--- PER-CLASS (the meaningful numbers) ---")
    print(f"  noise recall (specificity): {metrics['noise_recall_specificity']:.4f}  "
          f"precision: {metrics['noise_precision']:.4f}  f1: {metrics['noise_f1']:.4f}")
    print(f"  usv   recall (sensitivity): {metrics['usv_recall_sensitivity']:.4f}  "
          f"precision: {metrics['usv_precision']:.4f}  f1: {metrics['usv_f1']:.4f}")
    print(f"  balanced accuracy: {metrics['balanced_accuracy']:.4f}   "
          f"macro-F1: {metrics['macro_f1_binary']:.4f}")
    print(f"  pooled acc (MISLEADING): {metrics['pooled_accuracy_MISLEADING']:.4f}   "
          f"always-usv baseline: {metrics['always_usv_baseline_pooled']:.4f}")
    c = metrics["confusion"]
    print(f"  confusion: noise->noise={c['true_noise_pred_noise']} "
          f"noise->usv={c['true_noise_pred_usv']} "
          f"usv->noise={c['true_usv_pred_noise']} "
          f"usv->usv={c['true_usv_pred_usv']}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_json = args.output_dir / "held_out_844_eval.json"
    out_md = args.output_dir / "held_out_844_eval.md"
    with open(out_json, "w") as f:
        json.dump({"metrics": metrics, "twelve_class_breakdown": breakdown,
                   "checkpoint": str(args.checkpoint)}, f, indent=2)
    _write_report(metrics, breakdown, args.checkpoint, out_md)
    print(f"\nWrote {out_json}")
    print(f"Wrote {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Run 0 / per-variant evaluator: model performance on labeled events.

Loads a CNN checkpoint, runs inference on a CSV of labeled events (one row
per PNG), and reports overall + per-class metrics. Used to compare lab
fine-tune variants on the 99 held-out lab events without needing to run
full batch detection.

Usage:
    .venv/bin/python scripts/eval_model_on_lab_holdout.py \
        --model models/lab_finetune_v1/best_model.pt \
        --val-csv data/training/lab_finetune_v1/csv/val.csv \
        --filter-dataset lab \
        --output results/lab_finetune_v1_run2_holdout_eval.json

The --filter-dataset flag picks the lab-only subset of the merged val csv;
omit to evaluate on all rows.
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

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.models.cnn_classifier import USVClassifierCNN

LABEL_MAP = {"Not USV": 0, "USV": 1}


def load_checkpoint(path: Path) -> tuple[USVClassifierCNN, dict]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    num_filters = ckpt.get("num_filters", [32, 96, 192])
    dense_units = ckpt.get("dense_units", 64)
    model = USVClassifierCNN(num_filters=num_filters, dense_units=dense_units)
    state_dict = ckpt.get("model_state_dict", ckpt)
    model.load_state_dict(state_dict)
    model.eval()
    return model, ckpt


def load_spec(png_path: Path) -> np.ndarray:
    img = Image.open(png_path).convert("L")
    arr = np.array(img, dtype=np.float32)
    mn, mx = arr.min(), arr.max()
    if mx > mn:
        arr = (arr - mn) / (mx - mn)
    else:
        arr = np.zeros_like(arr)
    return arr  # (H, W) in [0, 1]


def metrics(y_true: np.ndarray, y_pred: np.ndarray, p: np.ndarray) -> dict:
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = tp + fp + tn + fn
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / total if total else 0.0
    return {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "accuracy": acc, "precision": prec, "recall": rec, "f1": f1,
        "n": int(total),
        "n_pos": int((y_true == 1).sum()),
        "n_neg": int((y_true == 0).sum()),
        "prob_mean": float(p.mean()),
        "prob_p10": float(np.percentile(p, 10)),
        "prob_p50": float(np.percentile(p, 50)),
        "prob_p90": float(np.percentile(p, 90)),
        "logit_abs_p90": float(np.percentile(np.abs(np.log(p / (1 - p + 1e-12) + 1e-12)), 90)),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--val-csv", type=Path, required=True)
    ap.add_argument("--filter-dataset", default=None,
                    help="If set, keep only rows where dataset==<value> (e.g. 'lab').")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="Classification threshold on probability (default 0.5).")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = pd.read_csv(args.val_csv)
    if args.filter_dataset:
        if "dataset" not in df.columns:
            raise ValueError(f"--filter-dataset given but no 'dataset' column in {args.val_csv}")
        df = df[df["dataset"] == args.filter_dataset].copy()
    df = df[df["label"].isin(LABEL_MAP)].copy()
    print(f"Evaluating on {len(df)} rows (filter_dataset={args.filter_dataset})")
    print(f"Class counts: {df['label'].value_counts().to_dict()}")

    model, ckpt = load_checkpoint(args.model)
    model = model.to(device)

    probs, y_true, y_pred = [], [], []
    per_row = []

    with torch.no_grad():
        for _, row in df.iterrows():
            spec = load_spec(REPO_ROOT / row["spectrogram_path"])
            x = torch.from_numpy(spec).unsqueeze(0).unsqueeze(0).to(device)
            logit = model(x).item()
            p = float(1.0 / (1.0 + np.exp(-logit)))
            label = LABEL_MAP[row["label"]]
            pred = int(p >= args.threshold)
            probs.append(p)
            y_true.append(label)
            y_pred.append(pred)
            per_row.append({
                "candidate_id": row.get("candidate_id", ""),
                "label": row["label"],
                "label_idx": label,
                "prob": p,
                "logit": float(logit),
                "pred": pred,
                "correct": int(pred == label),
            })

    probs = np.array(probs)
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    result = {
        "model": str(args.model),
        "val_csv": str(args.val_csv),
        "filter_dataset": args.filter_dataset,
        "threshold": args.threshold,
        "checkpoint_epoch": ckpt.get("epoch"),
        "checkpoint_metrics": {k: float(v) for k, v in (ckpt.get("metrics") or {}).items()},
        "eval": metrics(y_true, y_pred, probs),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({**result, "per_row": per_row}, f, indent=2, default=float)

    e = result["eval"]
    print("=" * 64)
    print(f"Model:       {args.model}")
    print(f"Filter:      dataset={args.filter_dataset}, threshold={args.threshold}")
    print(f"Rows:        {e['n']} ({e['n_pos']} USV, {e['n_neg']} Not USV)")
    print(f"Accuracy:    {e['accuracy']:.4f}")
    print(f"Precision:   {e['precision']:.4f}")
    print(f"Recall:      {e['recall']:.4f}")
    print(f"F1:          {e['f1']:.4f}")
    print(f"Confusion:   tp={e['tp']} fp={e['fp']} tn={e['tn']} fn={e['fn']}")
    print(f"Prob mean:   {e['prob_mean']:.4f}")
    print(f"Prob p10/50/90: {e['prob_p10']:.4f} / {e['prob_p50']:.4f} / {e['prob_p90']:.4f}")
    print(f"|logit| p90: {e['logit_abs_p90']:.4f}  (>>5 means saturated)")
    print("=" * 64)
    print(f"Saved: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

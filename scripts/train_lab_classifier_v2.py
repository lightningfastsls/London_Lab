"""Module 18.4 — DANN training CLI for the cage-invariant lab USV classifier.

ROADMAP §18.4 file 3. Extends the Module 18.3 v1 trainer
(``scripts/train_lab_classifier.py``) with a Domain-Adversarial Neural Network
head (Ganin & Lempitsky 2015). The shared ResNet-18 encoder is trained to be
**good** at the 12-class Grimsley syllable task (on the labeled VocalMat
*source* domain) and **bad** at telling cages apart (VocalMat vs lab_131204),
via a gradient-reversal domain head.

Domain setup (D4, 2-cage granularity)
-------------------------------------
- **Source** (domain label 0): VocalMat patches — labeled with Grimsley
  classes. Both the class head and the domain head train on these.
- **Target** (domain label 1): lab_131204 patches — *unlabeled* (this is
  UNSUPERVISED domain adaptation; lab data was never syllable-classified).
  Only the domain head trains on these.

The class loss is the Module 18.3 focal loss; the domain loss is plain
cross-entropy on the 2-way domain head, whose gradient is reversed (×−λ) before
reaching the encoder. λ ramps 0 → 1 on the Ganin schedule.

Warm-start
----------
``--v1-checkpoint`` loads the 18.3 ``best.pt``. The timm ResNet-18 final layer
is named ``fc.*``; those weights seed the v2 ``class_head`` and the remaining
backbone keys seed the v2 ``encoder``. This starts v2 from a competent v1
classifier so the adversarial objective only has to remove the *residual* cage
signal, not relearn syllables.

Collapse detection
-------------------
If the adversarial pressure is too strong the encoder collapses (cage-invariant
but syllable-blind). The exit gate is ``syllable macro-F1 ≥ v1 − 0.05``; the
written ``comparison_v1_vs_v2.md`` flags a collapse if the drop exceeds 0.05 or
the cage probe stays ≥ 0.65.

This script is additive — Module 18.3 files are imported, never modified.

Usage
-----
    python scripts/train_lab_classifier_v2.py \\
        --train-csv data/lab_cnn_training/train/manifest.csv \\
        --val-csv   data/lab_cnn_training/val/manifest.csv \\
        --test-csv  data/lab_cnn_training/test/manifest.csv \\
        --domain-unlabeled-csv data/lab_cnn_training/domain_unlabeled.csv \\
        --v1-checkpoint results/lab_classifier_v1/best.pt \\
        --output-dir results/lab_classifier_v2/ \\
        --domain-granularity 2cage --epochs 50 --batch-size 64

Reference: Ganin & Lempitsky 2015 ICML (arXiv:1409.7495).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from itertools import cycle
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
_SCRIPTS = Path(__file__).resolve().parent
for _p in (str(_SRC), str(_SCRIPTS)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from sklearn.metrics import confusion_matrix as sk_confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

# Module 18.3 reuse (imported, never modified).
from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES
from usv_spectrogram.classifier.losses import focal_loss
from usv_spectrogram.classifier.training import _compute_class_weights, _cosine_warmup_lr, TrainingConfig
# Module 18.4 components.
from usv_spectrogram.classifier.dann import LambdaSchedule, ResNet18DANN
from usv_spectrogram.classifier.cage_probe import linear_cage_probe
# v1 dataset + plotting reuse (these live in the v1 script).
from train_lab_classifier import ManifestDataset, _render_confusion_matrix_png, _DEFAULT_IMAGE_SIZE

NUM_CLASSES = len(GRIMSLEY_12_CLASSES)
CAGE_PROBE_THRESHOLD = 0.65          # ROADMAP §18.4 pass gate (lower = more invariant)
COLLAPSE_F1_DROP = 0.05              # ROADMAP §18.4 collapse tripwire


# ---------------------------------------------------------------------------
# Target (unlabeled) dataset
# ---------------------------------------------------------------------------

def _resolve_patch_path(raw: str, candidate_roots: list[Path]) -> Path:
    """Resolve a manifest patch path, trying absolute then each candidate root."""
    p = Path(raw)
    if p.is_absolute():
        return p
    for root in candidate_roots:
        cand = root / p
        if cand.exists():
            return cand
    # Fall back to the first root (lets the caller surface a clear FileNotFound).
    return candidate_roots[0] / p


class UnlabeledPatchDataset(Dataset):
    """Loads unlabeled target-domain patches from ``domain_unlabeled.csv``.

    The CSV has columns ``path, cohort, source_recording, duration_ms``. We
    filter to ``cohort == cohort_filter`` (default 'lab' — the DANN target).
    Returns ``(image_tensor, domain_label)`` so it can be concatenated with the
    source domain in the adversarial loop.
    """

    def __init__(
        self,
        domain_csv: Path,
        domain_label: int,
        cohort_filter: str = "lab",
        image_size: int = _DEFAULT_IMAGE_SIZE,
        candidate_roots: list[Path] | None = None,
        max_samples: int | None = None,
    ) -> None:
        df = pd.read_csv(domain_csv)
        if "path" not in df.columns:
            raise ValueError(f"{domain_csv} must have a 'path' column; got {list(df.columns)}")
        if "cohort" in df.columns and cohort_filter is not None:
            df = df[df["cohort"].astype(str) == cohort_filter].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"{domain_csv} has no rows with cohort=={cohort_filter!r}"
            )
        if max_samples is not None and len(df) > max_samples:
            df = df.sample(n=max_samples, random_state=0).reset_index(drop=True)
        self._df = df
        self._domain_label = int(domain_label)
        self._roots = candidate_roots or [REPO_ROOT, domain_csv.resolve().parent]
        self._transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.Grayscale(num_output_channels=3),
                transforms.ToTensor(),
            ]
        )

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self._df.iloc[idx]
        path = _resolve_patch_path(str(row["path"]), self._roots)
        with Image.open(path) as im:
            tensor = self._transform(im.convert("L"))
        return tensor, self._domain_label


# ---------------------------------------------------------------------------
# Adversarial training loop
# ---------------------------------------------------------------------------

def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("Warning: --device cuda requested but no GPU; falling back to cpu", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(device)


def _warm_start_from_v1(model: ResNet18DANN, ckpt_path: Path) -> dict[str, int]:
    """Seed the v2 encoder + class head from a v1 ``best.pt`` checkpoint.

    The v1 model is a timm ResNet-18 (num_classes=12) whose head is ``fc.*``.
    Backbone keys → ``model.encoder``; ``fc.*`` → ``model.class_head``. The
    domain head is left at its random init. Returns a small report dict.
    """
    blob = torch.load(ckpt_path, map_location="cpu")
    v1_sd = blob["state_dict"] if isinstance(blob, dict) and "state_dict" in blob else blob

    encoder_sd = {k: v for k, v in v1_sd.items() if not k.startswith("fc.")}
    missing, unexpected = model.encoder.load_state_dict(encoder_sd, strict=False)

    head_loaded = 0
    if "fc.weight" in v1_sd and "fc.bias" in v1_sd:
        with torch.no_grad():
            if model.class_head.weight.shape == v1_sd["fc.weight"].shape:
                model.class_head.weight.copy_(v1_sd["fc.weight"])
                model.class_head.bias.copy_(v1_sd["fc.bias"])
                head_loaded = 1
    return {
        "encoder_keys_loaded": len(encoder_sd) - len(unexpected),
        "encoder_keys_missing": len(missing),
        "encoder_keys_unexpected": len(unexpected),
        "class_head_loaded": head_loaded,
    }


@torch.no_grad()
def _collect_class_predictions(model, loader, device):
    model.eval()
    preds, trues = [], []
    for batch in loader:
        images, targets = batch[0].to(device), batch[1].to(device)
        class_logits, _, _ = model(images, lambda_=0.0)
        preds.append(class_logits.argmax(dim=-1).cpu().numpy())
        trues.append(targets.cpu().numpy())
    if not preds:
        return np.array([], dtype=int), np.array([], dtype=int)
    return np.concatenate(preds), np.concatenate(trues)


def _split_metrics(y_true, y_pred):
    if y_true.size == 0:
        return {"macro_f1": 0.0,
                "per_class_precision": [0.0] * NUM_CLASSES,
                "per_class_recall": [0.0] * NUM_CLASSES,
                "confusion_matrix": np.zeros((NUM_CLASSES, NUM_CLASSES), int).tolist()}
    labels = list(range(NUM_CLASSES))
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0)
    cm = sk_confusion_matrix(y_true, y_pred, labels=labels)
    return {"macro_f1": float(np.mean(f1)),
            "per_class_precision": [float(x) for x in precision],
            "per_class_recall": [float(x) for x in recall],
            "confusion_matrix": cm.tolist()}


def train_dann(
    source_train_loader: DataLoader,
    source_val_loader: DataLoader,
    source_test_loader: DataLoader,
    target_train_loader: DataLoader,
    cfg: TrainingConfig,
    lambda_gamma: float,
    output_dir: Path,
    v1_checkpoint: Path | None,
    num_domains: int,
) -> dict[str, Any]:
    """Run the DANN training loop; return a metrics dict + write best.pt.

    Source batches drive both the class (focal) loss and the source side of the
    domain loss; target batches drive only the target side of the domain loss.
    The domain head's gradient is reversed by the GRL inside ``ResNet18DANN``.
    """
    device = _resolve_device(cfg.device)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model = ResNet18DANN(num_classes=NUM_CLASSES, num_domains=num_domains,
                         pretrained=cfg.pretrained).to(device)
    warm_report = {}
    if v1_checkpoint is not None:
        warm_report = _warm_start_from_v1(model, Path(v1_checkpoint))
        print(f"Warm-start from {v1_checkpoint}: {warm_report}")

    optimizer = optim.AdamW(model.parameters(), lr=cfg.learning_rate,
                            weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda e: _cosine_warmup_lr(e, cfg))
    class_weights = _compute_class_weights(source_train_loader, NUM_CLASSES).to(device)
    lam_schedule = LambdaSchedule(total_epochs=cfg.epochs, gamma=lambda_gamma)
    domain_criterion = nn.CrossEntropyLoss()

    best_f1, epochs_no_improve, best_state = -math.inf, 0, None
    history: list[dict[str, float]] = []

    for epoch in range(cfg.epochs):
        lam = lam_schedule.lambda_at(epoch)
        model.train()
        target_iter = cycle(target_train_loader)
        run_cls = run_dom = 0.0
        n = 0
        for src_imgs, src_cls in source_train_loader:
            src_imgs, src_cls = src_imgs.to(device), src_cls.to(device)
            tgt_imgs, _ = next(target_iter)
            tgt_imgs = tgt_imgs.to(device)

            optimizer.zero_grad()
            # Source: class + domain(0)
            cls_logits, dom_logits_s, _ = model(src_imgs, lambda_=lam)
            loss_cls = focal_loss(cls_logits, src_cls, class_weights, gamma=cfg.focal_gamma)
            dom_tgt_s = torch.zeros(src_imgs.size(0), dtype=torch.long, device=device)
            loss_dom_s = domain_criterion(dom_logits_s, dom_tgt_s)
            # Target: domain(1) only (unlabeled — no class loss)
            _, dom_logits_t, _ = model(tgt_imgs, lambda_=lam)
            dom_tgt_t = torch.ones(tgt_imgs.size(0), dtype=torch.long, device=device)
            loss_dom_t = domain_criterion(dom_logits_t, dom_tgt_t)

            loss = loss_cls + loss_dom_s + loss_dom_t
            loss.backward()
            optimizer.step()
            run_cls += float(loss_cls.item()); run_dom += float((loss_dom_s + loss_dom_t).item()); n += 1
        scheduler.step()

        val_pred, val_true = _collect_class_predictions(model, source_val_loader, device)
        val_f1 = _split_metrics(val_true, val_pred)["macro_f1"]
        history.append({"epoch": epoch, "lambda": lam,
                        "class_loss": run_cls / max(1, n),
                        "domain_loss": run_dom / max(1, n),
                        "macro_f1_val": val_f1})
        print(f"epoch {epoch:02d} λ={lam:.3f} class_loss={run_cls/max(1,n):.4f} "
              f"domain_loss={run_dom/max(1,n):.4f} val_macroF1={val_f1:.4f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                print(f"Early stop at epoch {epoch} (no val-F1 improvement for "
                      f"{cfg.early_stop_patience} epochs)")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    torch.save({"state_dict": model.state_dict(), "history": history,
                "warm_start": warm_report,
                "lambda_schedule": {"total_epochs": cfg.epochs, "gamma": lambda_gamma}},
               output_dir / "best.pt")

    val_pred, val_true = _collect_class_predictions(model, source_val_loader, device)
    test_pred, test_true = _collect_class_predictions(model, source_test_loader, device)
    val_metrics = _split_metrics(val_true, val_pred)
    test_metrics = _split_metrics(test_true, test_pred)

    return {
        "model": model,
        "macro_f1_val": float(val_metrics["macro_f1"]),
        "macro_f1_test": float(test_metrics["macro_f1"]),
        "per_class_precision": test_metrics["per_class_precision"],
        "per_class_recall": test_metrics["per_class_recall"],
        "confusion_matrix": test_metrics["confusion_matrix"],
        "history": history,
        "warm_start": warm_report,
    }


# ---------------------------------------------------------------------------
# Cage probe + comparison report
# ---------------------------------------------------------------------------

def _build_cage_probe_loaders(source_val_ds, target_ds, device, batch_size, max_per_cage=None):
    """Build train/val DataLoaders labeling source patches cage 0, target cage 1.

    The probe measures whether cage is *linearly decodable* from the v2
    encoder's features. We pull a balanced sample from each cage and split
    80/20 into train/val.
    """
    import torch as _torch
    from torch.utils.data import TensorDataset

    def _stack(ds, label, cap):
        idx = list(range(len(ds)))
        if cap is not None and len(idx) > cap:
            rng = np.random.default_rng(0)
            idx = list(rng.choice(idx, size=cap, replace=False))
        imgs = _torch.stack([ds[i][0] for i in idx])
        labels = _torch.full((len(idx),), label, dtype=_torch.long)
        return imgs, labels

    n_each = max_per_cage if max_per_cage is not None else min(len(source_val_ds), len(target_ds))
    s_imgs, s_lbl = _stack(source_val_ds, 0, n_each)
    t_imgs, t_lbl = _stack(target_ds, 1, n_each)
    imgs = _torch.cat([s_imgs, t_imgs]); lbls = _torch.cat([s_lbl, t_lbl])

    g = _torch.Generator().manual_seed(0)
    perm = _torch.randperm(len(imgs), generator=g)
    imgs, lbls = imgs[perm], lbls[perm]
    n_val = max(1, int(0.2 * len(imgs)))
    val_ds = TensorDataset(imgs[:n_val], lbls[:n_val])
    train_ds = TensorDataset(imgs[n_val:], lbls[n_val:])
    return (DataLoader(train_ds, batch_size=batch_size, shuffle=False),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False))


def _write_comparison(output_dir: Path, v2: dict, v1_metrics_path: Path | None,
                      cage_probe_acc: float) -> None:
    """Write comparison_v1_vs_v2.md with the collapse / cage gates evaluated."""
    v1 = {}
    if v1_metrics_path is not None and Path(v1_metrics_path).exists():
        with open(v1_metrics_path) as f:
            v1 = json.load(f)
    v1_f1 = v1.get("macro_f1_test")
    v2_f1 = v2["macro_f1_test"]

    f1_drop = (v1_f1 - v2_f1) if v1_f1 is not None else None
    collapse = (f1_drop is not None and f1_drop > COLLAPSE_F1_DROP)
    cage_pass = cage_probe_acc < CAGE_PROBE_THRESHOLD

    lines = [
        "# Module 18.4 — v1 vs v2 (DANN) comparison",
        "",
        "| Metric | v1 (18.3) | v2 (DANN) | Gate |",
        "|---|---|---|---|",
        f"| Syllable macro-F1 (test) | {v1_f1 if v1_f1 is not None else 'n/a'} | {v2_f1:.4f} | "
        f"≥ v1−{COLLAPSE_F1_DROP} → {'FAIL (collapse)' if collapse else 'PASS'} |",
        f"| Syllable macro-F1 (val) | {v1.get('macro_f1_val', 'n/a')} | {v2['macro_f1_val']:.4f} | — |",
        f"| Linear cage probe acc | n/a | {cage_probe_acc:.4f} | "
        f"< {CAGE_PROBE_THRESHOLD} → {'PASS' if cage_pass else 'FAIL'} |",
        "",
        f"- **F1 drop v1→v2:** {f'{f1_drop:.4f}' if f1_drop is not None else 'n/a (no v1 metrics)'}",
        f"- **Collapse tripwire (drop > {COLLAPSE_F1_DROP}):** {'TRIGGERED — STOP' if collapse else 'clear'}",
        f"- **Cage gate (probe < {CAGE_PROBE_THRESHOLD}):** {'PASS' if cage_pass else 'FAIL — encoder still cage-decodable'}",
        "",
        "## Verdict",
        ("**SHIP candidate** — cage invariance achieved without syllable collapse."
         if (cage_pass and not collapse) else
         "**DO NOT SHIP** — " + ("encoder collapsed (F1 drop > 0.05). " if collapse else "")
         + ("cage probe ≥ 0.65 (cage still decodable). " if not cage_pass else "")
         + "Surface λ-schedule alternatives per ROADMAP §18.4 exit criteria."),
        "",
        "_Note: the VAE falsifiable re-run on encoder features "
        "(`run_vae_diagnostic_on_encoder.py` → `cage_invariance_probe.md`) is the "
        "third gate and must also pass; it is computed separately._",
    ]
    (output_dir / "comparison_v1_vs_v2.md").write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train the cage-invariant DANN lab classifier (Module 18.4)")
    p.add_argument("--train-csv", required=True, type=Path, help="VocalMat (source) train manifest")
    p.add_argument("--val-csv", required=True, type=Path, help="VocalMat (source) val manifest")
    p.add_argument("--test-csv", required=True, type=Path, help="VocalMat (source) test manifest")
    p.add_argument("--domain-unlabeled-csv", required=True, type=Path,
                   help="Unlabeled domain CSV (target = lab patches)")
    p.add_argument("--target-cohort", default="lab", help="cohort value to use as DANN target")
    p.add_argument("--v1-checkpoint", type=Path, default=None, help="18.3 best.pt for warm-start")
    p.add_argument("--v1-metrics", type=Path, default=None,
                   help="18.3 metrics.json for the comparison report (defaults next to --v1-checkpoint)")
    p.add_argument("--output-dir", required=True, type=Path)
    p.add_argument("--domain-granularity", default="2cage", choices=("2cage", "per_recording"),
                   help="D4: 2cage (default) or per_recording")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=0)
    p.add_argument("--focal-gamma", type=float, default=2.0)
    p.add_argument("--lambda-gamma", type=float, default=10.0, help="γ for the Ganin λ schedule")
    p.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--max-source-samples", type=int, default=None, help="cap source rows (smoke)")
    p.add_argument("--max-target-samples", type=int, default=None, help="cap target rows (smoke)")
    p.add_argument("--cage-probe-samples", type=int, default=500,
                   help="patches per cage for the linear cage probe")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    for flag, path in [("--train-csv", args.train_csv), ("--val-csv", args.val_csv),
                       ("--test-csv", args.test_csv),
                       ("--domain-unlabeled-csv", args.domain_unlabeled_csv)]:
        if not path.exists():
            print(f"Error: {flag} does not exist: {path}", file=sys.stderr)
            return 1
    if args.v1_checkpoint is not None and not args.v1_checkpoint.exists():
        print(f"Error: --v1-checkpoint does not exist: {args.v1_checkpoint}", file=sys.stderr)
        return 1

    if args.domain_granularity == "per_recording":
        # D4 commits to 2cage; per_recording (50-100 domains) is documented as a
        # more aggressive option but is not wired (num_domains would need to equal
        # the unique recording count in the target domain). Fail loudly rather than
        # silently training a wrong 2-domain model under a per_recording flag.
        raise NotImplementedError(
            "domain-granularity 'per_recording' is not implemented (D4 uses 2cage). "
            "TODO(18.5+): set num_domains = len(unique recordings in target domain).")
    num_domains = 2

    cfg = TrainingConfig(epochs=args.epochs, batch_size=args.batch_size,
                         learning_rate=args.lr, weight_decay=args.weight_decay,
                         warmup_epochs=args.warmup_epochs, focal_gamma=args.focal_gamma,
                         device=args.device, pretrained=not args.no_pretrained)

    # Source (labeled VocalMat) — optionally capped for smoke via head().
    src_train_ds = ManifestDataset(args.train_csv)
    src_val_ds = ManifestDataset(args.val_csv)
    src_test_ds = ManifestDataset(args.test_csv)
    if args.max_source_samples is not None:
        # Trim the underlying frames in place for a fast smoke run.
        for ds in (src_train_ds, src_val_ds, src_test_ds):
            ds._df = ds._df.head(args.max_source_samples).reset_index(drop=True)

    tgt_ds = UnlabeledPatchDataset(
        args.domain_unlabeled_csv, domain_label=1, cohort_filter=args.target_cohort,
        candidate_roots=[REPO_ROOT, args.domain_unlabeled_csv.resolve().parent,
                         args.domain_unlabeled_csv.resolve().parent.parent],
        max_samples=args.max_target_samples)

    dl = lambda ds, sh: DataLoader(ds, batch_size=args.batch_size, shuffle=sh,
                                   num_workers=args.num_workers, pin_memory=False)
    result = train_dann(
        source_train_loader=dl(src_train_ds, True), source_val_loader=dl(src_val_ds, False),
        source_test_loader=dl(src_test_ds, False), target_train_loader=dl(tgt_ds, True),
        cfg=cfg, lambda_gamma=args.lambda_gamma, output_dir=args.output_dir,
        v1_checkpoint=args.v1_checkpoint, num_domains=num_domains)

    model = result.pop("model")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    _render_confusion_matrix_png(result["confusion_matrix"], GRIMSLEY_12_CLASSES,
                                 args.output_dir / "confusion_matrix.png",
                                 title="Lab classifier v2 (DANN) — test confusion matrix")

    # Linear cage probe on the trained encoder. Use the source TEST split (not
    # val): the val split selected the early-stopping checkpoint, so probing on
    # it would carry a mild optimism bias (MINOR-3). The test split is
    # independent of checkpoint selection.
    device = _resolve_device(cfg.device)
    probe_train, probe_val = _build_cage_probe_loaders(
        src_test_ds, tgt_ds, device, args.batch_size, max_per_cage=args.cage_probe_samples)
    cage_acc = linear_cage_probe(model.encoder, probe_train, probe_val,
                                 num_cages=2, device=str(device))
    print(f"Linear cage probe accuracy: {cage_acc:.4f} "
          f"({'PASS' if cage_acc < CAGE_PROBE_THRESHOLD else 'FAIL'} vs <{CAGE_PROBE_THRESHOLD})")

    metrics_out = {k: v for k, v in result.items() if k != "history"}
    metrics_out["cage_probe_acc"] = cage_acc
    with open(args.output_dir / "metrics.json", "w") as f:
        json.dump(metrics_out, f, indent=2)

    v1_metrics = args.v1_metrics
    if v1_metrics is None and args.v1_checkpoint is not None:
        cand = args.v1_checkpoint.parent / "metrics.json"
        v1_metrics = cand if cand.exists() else None
    _write_comparison(args.output_dir, result, v1_metrics, cage_acc)

    print(f"v2 training complete. macro_f1_val={result['macro_f1_val']:.4f} "
          f"macro_f1_test={result['macro_f1_test']:.4f} cage_probe_acc={cage_acc:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

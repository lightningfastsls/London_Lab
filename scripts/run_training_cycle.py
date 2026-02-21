"""Run a complete active learning training cycle.

Chains 7 steps into a single reproducible command:
  1. ASSEMBLE  - Build training data from labels + WAV files
  2. TRAIN     - Train CNN classifier
  3. EVALUATE  - Evaluate on held-out test set
  4. OPTIMIZE  - Find optimal classification threshold
  5. MINE      - Mine hard negatives from false positives
  6. COMPARE   - Compare with previous model (optional)
  7. REPORT    - Generate markdown summary

Usage:
    python scripts/run_training_cycle.py \\
        --labels-dir data/labeled_detections \\
        --wav-dir "5970 USV" \\
        --cycle-name milestone_1 \\
        --output-dir runs/milestone_1

    # Quick smoke test (2 epochs, skip mining):
    python scripts/run_training_cycle.py \\
        --labels-dir data/labeled_detections \\
        --wav-dir "5970 USV" \\
        --cycle-name smoke_test \\
        --output-dir runs/smoke_test \\
        --model-size small --epochs 2 --skip-mining
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

# Add src to path (Pattern 8: guarded with .resolve())
REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from usv_spectrogram.dataset import AssemblyConfig, DatasetAssembler
from usv_spectrogram.models import USVClassifierCNN, Trainer, create_data_loaders
from usv_spectrogram.models.evaluate import (
    evaluate_model,
    plot_confusion_matrix,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from usv_spectrogram.training.cycle_report import (
    CycleMetrics,
    generate_cycle_report,
)

logger = logging.getLogger(__name__)

# Model size configurations (from train_cnn.py:34-53)
MODEL_CONFIGS = {
    "small": {"filters": [32, 64, 128], "dense_units": 64},
    "medium": {"filters": [64, 128, 256], "dense_units": 128},
    "large": {"filters": [128, 256, 512], "dense_units": 256},
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments.

    Parameters
    ----------
    argv : list[str] | None
        Argument list (defaults to sys.argv[1:]).
    """
    parser = argparse.ArgumentParser(
        description="Run a complete active learning training cycle",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Required
    parser.add_argument(
        "--labels-dir",
        type=Path,
        required=True,
        help="Directory with LabelStorage JSON files",
    )
    parser.add_argument(
        "--wav-dir",
        type=Path,
        required=True,
        help="Directory containing WAV files",
    )
    parser.add_argument(
        "--cycle-name",
        type=str,
        required=True,
        help="Cycle name (e.g. 'milestone_1')",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output root directory for all cycle artifacts",
    )

    # Optional
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Model size configuration (default: small)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Maximum training epochs (default: 50)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience (default: 10)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--skip-mining",
        action="store_true",
        help="Skip hard negative mining step",
    )
    parser.add_argument(
        "--previous-model",
        type=Path,
        default=None,
        help="Path to previous model .pt file for comparison",
    )
    parser.add_argument(
        "--previous-model-size",
        type=str,
        default="small",
        choices=["small", "medium", "large"],
        help="Architecture of previous model (default: small)",
    )
    parser.add_argument(
        "--target-recall",
        type=float,
        default=0.90,
        help="Target recall for threshold optimization (default: 0.90)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["cpu", "cuda", "auto"],
        help="Device for training and inference (default: auto)",
    )

    return parser.parse_args(argv)


def resolve_device(device_str: str) -> torch.device:
    """Resolve device string to torch.device."""
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def create_model(model_size: str) -> USVClassifierCNN:
    """Create a CNN model from the size configuration."""
    config = MODEL_CONFIGS[model_size]
    return USVClassifierCNN(
        num_filters=config["filters"],
        dense_units=config["dense_units"],
    )


def load_model_with_architecture(
    model_path: Path, model_size: str, device: torch.device
) -> USVClassifierCNN:
    """Load a model checkpoint with the correct architecture.

    Reads architecture metadata (num_filters, dense_units) from the checkpoint
    if available, falling back to MODEL_CONFIGS[model_size] otherwise.
    This ensures backward compatibility with older checkpoints that lack metadata.
    """
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)

    # Prefer architecture from checkpoint metadata (saved by Trainer)
    num_filters = checkpoint.get("num_filters")
    dense_units = checkpoint.get("dense_units")

    if num_filters is not None and dense_units is not None:
        model = USVClassifierCNN(num_filters=num_filters, dense_units=dense_units)
    else:
        model = create_model(model_size)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model


# ── Step functions ────────────────────────────────────────────────────


def step_assemble(
    args: argparse.Namespace, metrics: CycleMetrics
) -> Path:
    """Step 1: Assemble training data from labels."""
    print("=" * 60)
    print("STEP 1/7: ASSEMBLE")
    print("=" * 60)

    data_dir = args.output_dir / "data"

    config = AssemblyConfig(
        labels_dir=args.labels_dir,
        wav_dir=args.wav_dir,
        output_dir=data_dir,
        seed=args.seed,
    )
    assembler = DatasetAssembler(config)
    report = assembler.assemble()

    # Approximate unique label count (pre-jitter)
    jitter_n = config.jitter_n_samples
    # Each original label produces 1 original + up to jitter_n jittered copies
    # total_positives / (1 + jitter_n) is approximate (some USVs may not jitter)
    metrics.label_count = report.total_positives // (1 + jitter_n)
    metrics.train_samples = report.train_count
    metrics.val_samples = report.val_count
    metrics.test_samples = report.test_count

    if report.val_count == 0 or report.test_count == 0:
        raise RuntimeError(
            f"Insufficient data for splits: val={report.val_count}, "
            f"test={report.test_count}. Need more recordings."
        )

    print(f"  Labels: ~{metrics.label_count} unique")
    print(f"  Train/Val/Test: {report.train_count}/{report.val_count}/{report.test_count}")
    print()

    return data_dir


def step_train(
    args: argparse.Namespace, metrics: CycleMetrics, data_dir: Path, device: torch.device
) -> None:
    """Step 2: Train CNN classifier.

    Best model is saved to output_dir/model/best_model.pt for downstream steps.
    """
    print("=" * 60)
    print("STEP 2/7: TRAIN")
    print("=" * 60)

    # Seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    model_dir = args.output_dir / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    model = create_model(args.model_size)
    metrics.model_size = args.model_size

    train_loader, val_loader, class_weights = create_data_loaders(
        train_csv=data_dir / "train.csv",
        val_csv=data_dir / "val.csv",
        num_workers=0,  # Windows compatibility
    )

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        class_weights=class_weights,
        patience=args.patience,
        checkpoint_dir=model_dir,
        device=str(device),
    )

    t0 = time.time()
    history = trainer.train(num_epochs=args.epochs, verbose=True)
    training_time = time.time() - t0

    metrics.epochs_trained = len(history.train_loss)
    metrics.best_val_loss = min(history.val_loss)
    metrics.training_time_s = training_time

    print(f"  Trained {metrics.epochs_trained} epochs in {training_time:.0f}s")
    print(f"  Best val loss: {metrics.best_val_loss:.4f}")
    print()


def step_evaluate(
    args: argparse.Namespace,
    metrics: CycleMetrics,
    data_dir: Path,
    device: torch.device,
) -> None:
    """Step 3: Evaluate on test set."""
    print("=" * 60)
    print("STEP 3/7: EVALUATE")
    print("=" * 60)

    eval_dir = args.output_dir / "eval"
    eval_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"

    # Load best model with correct architecture
    model = load_model_with_architecture(
        model_dir / "best_model.pt", args.model_size, device
    )

    # Create test DataLoader
    from usv_spectrogram.models.data_loader import USVDataset, pad_collate_fn
    from torch.utils.data import DataLoader

    test_dataset = USVDataset(data_dir / "test.csv")
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_collate_fn,
    )

    eval_metrics, y_pred, y_proba, y_true = evaluate_model(
        model, test_loader, device=str(device), verbose=True
    )

    # Save metrics
    with open(eval_dir / "metrics.json", "w") as f:
        json.dump(
            {k: float(v) if isinstance(v, (np.floating, float)) else v
             for k, v in eval_metrics.items()},
            f,
            indent=2,
        )

    # Generate plots
    plot_confusion_matrix(y_true, y_pred, output_path=eval_dir / "confusion_matrix.png")
    plot_roc_curve(y_true, y_proba, output_path=eval_dir / "roc_curve.png")
    plot_precision_recall_curve(
        y_true, y_proba, output_path=eval_dir / "precision_recall_curve.png"
    )

    metrics.precision = float(eval_metrics["precision"])
    metrics.recall = float(eval_metrics["recall"])
    metrics.f1 = float(eval_metrics["f1"])

    print(f"  Precision: {metrics.precision:.4f}")
    print(f"  Recall:    {metrics.recall:.4f}")
    print(f"  F1:        {metrics.f1:.4f}")
    print()


def step_optimize_threshold(
    args: argparse.Namespace,
    metrics: CycleMetrics,
    data_dir: Path,
    device: torch.device,
) -> None:
    """Step 4: Optimize classification threshold on validation set."""
    print("=" * 60)
    print("STEP 4/7: OPTIMIZE THRESHOLD")
    print("=" * 60)

    threshold_dir = args.output_dir / "threshold"
    threshold_dir.mkdir(parents=True, exist_ok=True)
    model_dir = args.output_dir / "model"

    # Load best model
    model = load_model_with_architecture(
        model_dir / "best_model.pt", args.model_size, device
    )

    # Import threshold functions from sibling script (guarded)
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    from optimize_threshold import compute_metrics_at_threshold, get_predictions

    # Run on VALIDATION set (not test — avoids information leakage)
    val_csv = data_dir / "val.csv"
    probs, labels = get_predictions(model, device, val_csv)

    # Sweep thresholds
    thresholds = np.arange(0.05, 0.96, 0.05)
    best_f1 = 0.0
    best_thresh = 0.5
    best_recall_thresh = 0.5

    results = []
    for thresh in thresholds:
        m = compute_metrics_at_threshold(probs, labels, thresh)
        results.append(m)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_thresh = thresh

    # Find high-recall threshold
    for m in results:
        if m["recall"] >= args.target_recall:
            best_recall_thresh = m["threshold"]
            break

    metrics.optimal_threshold = float(best_thresh)
    metrics.optimal_f1 = float(best_f1)
    metrics.high_recall_threshold = float(best_recall_thresh)

    # Save results
    import pandas as pd

    results_df = pd.DataFrame(results)
    results_df.to_csv(threshold_dir / "threshold_results.csv", index=False)

    optimal_data = {
        "optimal_threshold": float(best_thresh),
        "optimal_f1": float(best_f1),
        "high_recall_threshold": float(best_recall_thresh),
        "target_recall": args.target_recall,
    }
    with open(threshold_dir / "optimal_threshold.json", "w") as f:
        json.dump(optimal_data, f, indent=2)

    # Generate threshold sweep plot
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(results_df["threshold"], results_df["precision"], "b-", label="Precision", linewidth=2)
    ax.plot(results_df["threshold"], results_df["recall"], "g-", label="Recall", linewidth=2)
    ax.plot(results_df["threshold"], results_df["f1"], "r-", label="F1", linewidth=2)
    ax.axvline(x=best_thresh, color="r", linestyle="--", alpha=0.5, label=f"Best F1 ({best_thresh:.2f})")
    ax.set_xlabel("Threshold")
    ax.set_ylabel("Score")
    ax.set_title("Threshold Sweep (Validation Set)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 1])
    plt.tight_layout()
    plt.savefig(threshold_dir / "threshold_sweep.png", dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Optimal threshold: {best_thresh:.2f} (F1={best_f1:.4f})")
    print(f"  High-recall threshold: {best_recall_thresh:.2f}")
    print()


def step_mine_hard_negatives(
    args: argparse.Namespace, metrics: CycleMetrics
) -> None:
    """Step 5: Mine hard negatives via subprocess."""
    print("=" * 60)
    print("STEP 5/7: MINE HARD NEGATIVES")
    print("=" * 60)

    if args.skip_mining:
        metrics.mining_skipped = True
        print("  Skipped (--skip-mining flag)")
        print()
        return

    hn_dir = args.output_dir / "hard_negatives"
    hn_dir.mkdir(parents=True, exist_ok=True)

    mine_script = Path(__file__).parent / "mine_hard_negatives.py"
    model_path = args.output_dir / "model" / "best_model.pt"

    cmd = [
        sys.executable,
        str(mine_script),
        "--model", str(model_path),
        "--wav-dir", str(args.wav_dir),
        "--labeled-detections-dir", str(args.labels_dir),
        "--output-dir", str(hn_dir),
        "--device", str(args.device) if args.device != "auto" else (
            "cuda" if torch.cuda.is_available() else "cpu"
        ),
    ]

    result = subprocess.run(cmd, check=False)

    if result.returncode != 0:
        logger.warning("Hard negative mining exited with code %d", result.returncode)

    # Count results from summary CSV
    summary_csv = hn_dir / "hard_negative_candidates_summary.csv"
    if summary_csv.exists():
        import pandas as pd

        df = pd.read_csv(summary_csv)
        metrics.hard_negatives_found = len(df)
    else:
        metrics.hard_negatives_found = 0

    print(f"  Found {metrics.hard_negatives_found} hard negative candidates")
    print()


def step_compare(
    args: argparse.Namespace,
    metrics: CycleMetrics,
    data_dir: Path,
    device: torch.device,
) -> None:
    """Step 6: Compare with previous model."""
    print("=" * 60)
    print("STEP 6/7: COMPARE")
    print("=" * 60)

    if args.previous_model is None:
        print("  Skipped (no --previous-model provided)")
        print()
        return

    comp_dir = args.output_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    # Load previous model
    prev_model = load_model_with_architecture(
        args.previous_model, args.previous_model_size, device
    )

    # Create test DataLoader
    from usv_spectrogram.models.data_loader import USVDataset, pad_collate_fn
    from torch.utils.data import DataLoader

    test_dataset = USVDataset(data_dir / "test.csv")
    test_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        collate_fn=pad_collate_fn,
    )

    prev_metrics, _, _, _ = evaluate_model(
        prev_model, test_loader, device=str(device), verbose=False
    )

    metrics.previous_f1 = float(prev_metrics["f1"])
    metrics.f1_delta = metrics.f1 - metrics.previous_f1

    comparison = {
        "current_f1": metrics.f1,
        "previous_f1": metrics.previous_f1,
        "f1_delta": metrics.f1_delta,
        "current_precision": metrics.precision,
        "previous_precision": float(prev_metrics["precision"]),
        "current_recall": metrics.recall,
        "previous_recall": float(prev_metrics["recall"]),
    }
    with open(comp_dir / "model_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    sign = "+" if metrics.f1_delta >= 0 else ""
    print(f"  Previous F1: {metrics.previous_f1:.4f}")
    print(f"  Current F1:  {metrics.f1:.4f}")
    print(f"  Delta:       {sign}{metrics.f1_delta:.4f}")
    print()


def step_report(
    args: argparse.Namespace, metrics: CycleMetrics
) -> None:
    """Step 7: Generate markdown report."""
    print("=" * 60)
    print("STEP 7/7: REPORT")
    print("=" * 60)

    report_path = args.output_dir / "cycle_report.md"
    content = generate_cycle_report(metrics, report_path)

    # Also save metrics as JSON
    metrics.to_json(args.output_dir / "cycle_metrics.json")

    print(f"  Report: {report_path}")
    print(f"  Metrics: {args.output_dir / 'cycle_metrics.json'}")
    print()

    # Print summary
    print("=" * 60)
    print("CYCLE COMPLETE")
    print("=" * 60)
    print(f"  Cycle:     {metrics.cycle_name}")
    print(f"  Labels:    ~{metrics.label_count}")
    print(f"  Model:     {metrics.model_size}")
    print(f"  F1:        {metrics.f1:.4f}")
    print(f"  Threshold: {metrics.optimal_threshold:.2f}")
    if metrics.f1_delta is not None:
        sign = "+" if metrics.f1_delta >= 0 else ""
        print(f"  F1 delta:  {sign}{metrics.f1_delta:.4f}")
    if metrics.failed_step:
        print(f"  WARNING: Failed at step {metrics.failed_step}")
    print("=" * 60)


# ── Main ──────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    """Run the full training cycle pipeline."""
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    metrics = CycleMetrics(
        cycle_name=args.cycle_name,
        timestamp=datetime.now(timezone.utc).isoformat(),
        model_size=args.model_size,
    )

    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print()
    print("=" * 60)
    print(f"TRAINING CYCLE: {args.cycle_name}")
    print("=" * 60)
    print(f"  Labels dir:  {args.labels_dir}")
    print(f"  WAV dir:     {args.wav_dir}")
    print(f"  Output:      {args.output_dir}")
    print(f"  Model size:  {args.model_size}")
    print(f"  Epochs:      {args.epochs}")
    print(f"  Device:      {device}")
    print(f"  Seed:        {args.seed}")
    print()

    # Step definitions: (name, callable, extra_args)
    # Each step takes (args, metrics, ...) and may raise on failure.
    data_dir: Path | None = None

    steps = [
        "ASSEMBLE",
        "TRAIN",
        "EVALUATE",
        "OPTIMIZE",
        "MINE",
        "COMPARE",
        "REPORT",
    ]

    for step_name in steps:
        try:
            if step_name == "ASSEMBLE":
                data_dir = step_assemble(args, metrics)
            elif step_name == "TRAIN":
                step_train(args, metrics, data_dir, device)
            elif step_name == "EVALUATE":
                step_evaluate(args, metrics, data_dir, device)
            elif step_name == "OPTIMIZE":
                step_optimize_threshold(args, metrics, data_dir, device)
            elif step_name == "MINE":
                step_mine_hard_negatives(args, metrics)
            elif step_name == "COMPARE":
                step_compare(args, metrics, data_dir, device)
            elif step_name == "REPORT":
                step_report(args, metrics)

            metrics.completed_steps.append(step_name)

        except Exception:
            logger.error("Step %s failed:\n%s", step_name, traceback.format_exc())
            print(f"\n  ERROR in step {step_name}:")
            print(f"  {traceback.format_exc()}")
            metrics.failed_step = step_name

            # Skip to report step to preserve partial results
            if step_name != "REPORT":
                print("  Skipping to REPORT with partial results...")
                try:
                    step_report(args, metrics)
                    metrics.completed_steps.append("REPORT")
                except Exception:
                    logger.error("Report generation also failed")

            return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())

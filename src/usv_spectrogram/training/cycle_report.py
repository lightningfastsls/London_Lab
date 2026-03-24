"""Cycle metrics and markdown report generation for training cycles.

Each active learning cycle (assemble -> train -> evaluate -> optimize ->
mine -> compare -> report) produces a CycleMetrics record and a markdown
report summarizing the results.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


@dataclass
class CycleMetrics:
    """Metrics collected across all steps of a training cycle.

    Non-frozen because fields are populated incrementally as each pipeline
    step completes (same pattern as AssemblyReport in dataset/assembler.py).
    """

    # Identity
    cycle_name: str = ""
    timestamp: str = ""

    # Step 1: Assembly
    label_count: int = 0
    train_samples: int = 0
    val_samples: int = 0
    test_samples: int = 0

    # Step 2: Training
    model_size: str = ""
    epochs_trained: int = 0
    best_val_loss: float = float("inf")
    training_time_s: float = 0.0

    # Step 3: Evaluation
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Step 4: Threshold optimization
    optimal_threshold: float = 0.5
    optimal_f1: float = 0.0
    high_recall_threshold: float = 0.5

    # Step 5: Hard negative mining
    hard_negatives_found: int = 0
    mining_skipped: bool = False

    # Step 6: Comparison
    previous_f1: Optional[float] = None
    f1_delta: Optional[float] = None

    # Pipeline state
    failed_step: Optional[str] = None
    completed_steps: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Serialize to JSON-compatible dictionary."""
        return asdict(self)

    def to_json(self, path: Path) -> None:
        """Write metrics to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_json(cls, path: Path) -> CycleMetrics:
        """Load metrics from a JSON file."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Filter to known fields only
        known = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in data.items() if k in known})


def generate_cycle_report(metrics: CycleMetrics, output_path: Path) -> str:
    """Generate a markdown report from cycle metrics.

    Parameters
    ----------
    metrics : CycleMetrics
        Populated metrics from a training cycle.
    output_path : Path
        Where to write the markdown file.

    Returns
    -------
    str
        The markdown content (also written to output_path).
    """
    lines: list[str] = []

    lines.append(f"# Training Cycle Report: {metrics.cycle_name}")
    lines.append("")
    lines.append(f"**Generated:** {metrics.timestamp or datetime.now(timezone.utc).isoformat()}")
    lines.append(f"**Model size:** {metrics.model_size}")
    lines.append("")

    # Failure banner
    if metrics.failed_step:
        lines.append(f"> **WARNING:** Pipeline failed at step: {metrics.failed_step}")
        lines.append(f"> Completed steps: {', '.join(metrics.completed_steps)}")
        lines.append("")

    # Data summary
    lines.append("## Data Summary")
    lines.append("")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Unique labels (approx) | ~{metrics.label_count} |")
    lines.append(f"| Training samples | {metrics.train_samples} |")
    lines.append(f"| Validation samples | {metrics.val_samples} |")
    lines.append(f"| Test samples | {metrics.test_samples} |")
    total = metrics.train_samples + metrics.val_samples + metrics.test_samples
    lines.append(f"| Total samples | {total} |")
    lines.append("")

    # Training summary
    if metrics.epochs_trained > 0:
        lines.append("## Training Summary")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Epochs trained | {metrics.epochs_trained} |")
        lines.append(f"| Best validation loss | {metrics.best_val_loss:.4f} |")
        minutes = metrics.training_time_s / 60.0
        lines.append(f"| Training time | {minutes:.1f} min |")
        lines.append("")

    # Evaluation results
    if metrics.f1 > 0 or "EVALUATE" in metrics.completed_steps:
        lines.append("## Evaluation Results (Test Set)")
        lines.append("")
        lines.append(f"| Metric | Value |")
        lines.append(f"|--------|-------|")
        lines.append(f"| Precision | {metrics.precision:.4f} |")
        lines.append(f"| Recall | {metrics.recall:.4f} |")
        lines.append(f"| F1 Score | {metrics.f1:.4f} |")
        lines.append("")

    # Threshold optimization
    if "OPTIMIZE" in metrics.completed_steps:
        lines.append("## Threshold Optimization")
        lines.append("")
        lines.append(f"| Threshold | Value | F1 at Threshold |")
        lines.append(f"|-----------|-------|-----------------|")
        lines.append(
            f"| Optimal (best F1) | {metrics.optimal_threshold:.2f} "
            f"| {metrics.optimal_f1:.4f} |"
        )
        lines.append(
            f"| High-recall | {metrics.high_recall_threshold:.2f} | — |"
        )
        lines.append("")

    # Hard negative mining
    if metrics.mining_skipped:
        lines.append("## Hard Negative Mining")
        lines.append("")
        lines.append("Mining was **skipped** (--skip-mining flag).")
        lines.append("")
    elif "MINE" in metrics.completed_steps:
        lines.append("## Hard Negative Mining")
        lines.append("")
        lines.append(f"Found **{metrics.hard_negatives_found}** hard negative candidates.")
        lines.append("")

    # Model comparison
    if metrics.previous_f1 is not None:
        lines.append("## Model Comparison")
        lines.append("")
        lines.append(
            "> Note: Previous model evaluated on current cycle's test set. "
            "Delta reflects both model improvement and test set distribution shift."
        )
        lines.append("")
        lines.append(f"| Metric | Previous | Current | Delta |")
        lines.append(f"|--------|----------|---------|-------|")
        delta_str = f"{metrics.f1_delta:+.4f}" if metrics.f1_delta is not None else "—"
        lines.append(
            f"| F1 Score | {metrics.previous_f1:.4f} "
            f"| {metrics.f1:.4f} | {delta_str} |"
        )
        lines.append("")

    # Output artifacts
    lines.append("## Output Artifacts")
    lines.append("")
    lines.append("```")
    lines.append(f"{metrics.cycle_name}/")
    lines.append("  data/           # Assembly output (CSVs, spectrograms)")
    lines.append("  model/          # Trained model checkpoints")
    lines.append("  eval/           # Evaluation metrics and plots")
    lines.append("  threshold/      # Threshold optimization results")
    if not metrics.mining_skipped:
        lines.append("  hard_negatives/ # Mined hard negative candidates")
    if metrics.previous_f1 is not None:
        lines.append("  comparison/     # Model comparison results")
    lines.append("  cycle_report.md # This report")
    lines.append("```")
    lines.append("")

    # Recommended next steps (heuristic)
    next_steps: list[str] = []
    if metrics.hard_negatives_found > 0 and not metrics.mining_skipped:
        next_steps.append(
            f"Review {metrics.hard_negatives_found} hard negative candidates "
            "in `hard_negatives/` and add confirmed ones to training data"
        )
    if metrics.f1_delta is not None and metrics.f1_delta < 0.01:
        next_steps.append(
            "Consider increasing model size or adding more labeled data"
        )
    if metrics.f1 > 0 and metrics.f1 < 0.85:
        next_steps.append("F1 below 0.85 — prioritize labeling more data")
    if next_steps:
        lines.append("## Recommended Next Steps")
        lines.append("")
        for i, step in enumerate(next_steps, 1):
            lines.append(f"{i}. {step}")
        lines.append("")

    content = "\n".join(lines)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(content)

    return content

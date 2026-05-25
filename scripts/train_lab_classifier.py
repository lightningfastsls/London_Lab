"""Module 18.3 — training CLI for the lab USV syllable classifier.

ROADMAP §18.3 file 5. Wires manifest CSVs (from Module 18.2b) into a
PyTorch DataLoader, constructs a TrainingConfig, and invokes the training
loop.

Usage
-----
    python scripts/train_lab_classifier.py \\
        --train-csv   data/lab_cnn_training/train/manifest.csv \\
        --val-csv     data/lab_cnn_training/val/manifest.csv \\
        --test-csv    data/lab_cnn_training/test/manifest.csv \\
        --held-out-845 classified_detections_lab_131204_clean.csv \\
        --output-dir  results/lab_classifier_v1/ \\
        --epochs 50 --batch-size 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC = REPO_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from usv_spectrogram.classifier.dataset import GRIMSLEY_12_CLASSES
from usv_spectrogram.classifier.training import TrainingConfig, train_classifier


_DEFAULT_IMAGE_SIZE = 227


class ManifestDataset(Dataset):
    """PyTorch Dataset that loads patch images from a Module 18.2b manifest.

    The manifest CSV must have at least these columns:
    - ``path`` — absolute or repo-relative path to a PNG/JPEG patch.
    - ``class`` — one of the Grimsley 12 display names (or the snake-case
      folder name; both are accepted).
    """

    def __init__(
        self,
        manifest_csv: Path,
        image_size: int = _DEFAULT_IMAGE_SIZE,
        repo_root: Path = REPO_ROOT,
    ) -> None:
        df = pd.read_csv(manifest_csv)
        if "path" not in df.columns or "class" not in df.columns:
            raise ValueError(
                f"Manifest {manifest_csv} must have 'path' and 'class' columns; "
                f"got {list(df.columns)}"
            )
        snake = {
            "Noise": "noise", "Step up": "step_up", "Down-FM": "down_fm",
            "Short": "short", "Chevron": "chevron", "Up-FM": "up_fm",
            "Flat": "flat", "Two steps": "two_steps", "Step down": "step_down",
            "Complex": "complex", "Reverse Chevron": "rev_chevron",
            "Multi-steps": "mult_steps",
        }
        class_to_idx_display = {c: i for i, c in enumerate(GRIMSLEY_12_CLASSES)}
        class_to_idx_snake = {snake[c]: i for c, i in class_to_idx_display.items()}

        def _to_idx(label: str) -> int:
            if label in class_to_idx_display:
                return class_to_idx_display[label]
            return class_to_idx_snake.get(label, -1)

        df = df.assign(class_idx=df["class"].astype(str).map(_to_idx))
        df = df[df["class_idx"] >= 0].reset_index(drop=True)
        if df.empty:
            raise ValueError(
                f"Manifest {manifest_csv} has no rows with known Grimsley classes"
            )

        self._df = df
        self._repo_root = Path(repo_root)
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
        path = Path(row["path"])
        if not path.is_absolute():
            path = self._repo_root / path
        with Image.open(path) as im:
            tensor = self._transform(im.convert("L"))
        return tensor, int(row["class_idx"])


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the 12-class lab USV syllable classifier (Module 18.3)",
    )
    parser.add_argument("--train-csv", required=True, type=Path,
                        help="Path to the training manifest CSV (from Module 18.2b)")
    parser.add_argument("--val-csv", required=True, type=Path,
                        help="Path to the validation manifest CSV")
    parser.add_argument("--test-csv", required=True, type=Path,
                        help="Path to the test manifest CSV")
    parser.add_argument("--held-out-845", required=True, type=Path,
                        help="Path to classified_detections_lab_131204_clean.csv")
    parser.add_argument("--output-dir", required=True, type=Path,
                        help="Directory to write best.pt and metrics.json")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (must be > 0)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for train/val/test DataLoaders")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Initial learning rate (AdamW)")
    parser.add_argument("--warmup-epochs", type=int, default=0,
                        help="Warmup epochs before cosine LR decay starts "
                             "(default 0 to avoid masking data-path errors on "
                             "short --epochs runs; production 50-epoch runs "
                             "should pass --warmup-epochs 3)")
    parser.add_argument("--focal-gamma", type=float, default=2.0,
                        help="Focal-loss gamma parameter (Lin et al. 2017)")
    parser.add_argument("--device", default="auto",
                        choices=("auto", "cpu", "cuda"),
                        help="Compute device — 'auto' picks cuda if available, else cpu")
    parser.add_argument("--num-workers", type=int, default=2,
                        help="DataLoader num_workers (set 0 on small CI hosts)")
    parser.add_argument("--no-pretrained", action="store_true",
                        help="Skip ImageNet pretrained weights (smoke / offline runs)")
    return parser


def _validate_paths(args: argparse.Namespace) -> None:
    for flag, path in [
        ("--train-csv", args.train_csv),
        ("--val-csv", args.val_csv),
        ("--test-csv", args.test_csv),
        ("--held-out-845", args.held_out_845),
    ]:
        if not path.exists():
            print(
                f"Error: {flag} path does not exist: {path}",
                file=sys.stderr,
            )
            sys.exit(1)


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    _validate_paths(args)

    try:
        cfg = TrainingConfig(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            warmup_epochs=args.warmup_epochs,
            focal_gamma=args.focal_gamma,
            device=args.device,
            pretrained=not args.no_pretrained,
        )
    except ValueError as exc:
        print(f"Error: invalid training config (epochs/batch_size/lr): {exc}",
              file=sys.stderr)
        return 2

    train_ds = ManifestDataset(args.train_csv)
    val_ds = ManifestDataset(args.val_csv)
    test_ds = ManifestDataset(args.test_csv)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=False,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = train_classifier(
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        cfg=cfg,
        output_dir=args.output_dir,
        held_out_845_csv=args.held_out_845,
    )
    print(
        "Training complete. "
        f"macro_f1_val={metrics['macro_f1_val']:.4f} "
        f"macro_f1_test={metrics['macro_f1_test']:.4f} "
        f"usv_noise_acc={metrics['usv_noise_acc']:.4f} "
        f"syllable_entropy_mean={metrics['syllable_entropy_mean']:.4f}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

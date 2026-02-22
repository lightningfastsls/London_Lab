"""Phase 11.1 validation: Check prepared USV bout data for correctness.

Verifies directory structure, spectrogram shapes/dtypes, normalization stats,
and optionally plots samples and tests the DataLoader.

Usage:
    python validate_preprocessing.py --data-dir usv_language/prepared_data
    python validate_preprocessing.py --data-dir usv_language/prepared_data --plot --check-dataloader
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

# Bootstrap: ensure project root is on sys.path
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

EXPECTED_N_FREQ = 170
SPLITS = ("train", "val", "test")


def check_directory_structure(data_dir: Path) -> tuple[bool, dict[str, int]]:
    """Check that train/val/test dirs exist with manifest.json and .npy files."""
    ok = True
    counts: dict[str, int] = {}
    for split in SPLITS:
        split_dir = data_dir / split
        if not split_dir.is_dir():
            print(f"  FAIL: {split}/ directory missing")
            ok = False
            counts[split] = 0
            continue

        manifest_path = split_dir / "manifest.json"
        if not manifest_path.exists():
            print(f"  FAIL: {split}/manifest.json missing")
            ok = False

        npy_files = sorted(split_dir.glob("spec_*.npy"))
        counts[split] = len(npy_files)
        if not npy_files:
            print(f"  WARN: {split}/ has 0 .npy files")
        else:
            print(f"  OK: {split}/ has {len(npy_files)} spectrograms")

    return ok, counts


def check_random_samples(
    data_dir: Path, n_samples: int = 5,
) -> tuple[bool, list[tuple[str, np.ndarray]]]:
    """Check random spectrogram samples for shape, dtype, and value range."""
    ok = True
    samples: list[tuple[str, np.ndarray]] = []
    rng = np.random.RandomState(42)

    for split in SPLITS:
        split_dir = data_dir / split
        npy_files = sorted(split_dir.glob("spec_*.npy"))
        if not npy_files:
            continue

        indices = rng.choice(len(npy_files), min(n_samples, len(npy_files)), replace=False)
        for idx in indices:
            path = npy_files[idx]
            spec = np.load(str(path))
            label = f"{split}/{path.name}"
            samples.append((label, spec))

            # Shape check: (n_freq, T)
            if spec.ndim != 2:
                print(f"  FAIL: {label} has {spec.ndim}D shape {spec.shape}, expected 2D")
                ok = False
                continue

            if spec.shape[0] != EXPECTED_N_FREQ:
                print(f"  FAIL: {label} has {spec.shape[0]} freq bins, expected {EXPECTED_N_FREQ}")
                ok = False

            if spec.shape[1] == 0:
                print(f"  FAIL: {label} has 0 time frames")
                ok = False

            # Dtype check
            if spec.dtype != np.float32:
                print(f"  WARN: {label} dtype={spec.dtype}, expected float32")

            # Finite check
            if not np.all(np.isfinite(spec)):
                n_bad = np.sum(~np.isfinite(spec))
                print(f"  FAIL: {label} has {n_bad} non-finite values")
                ok = False

            # Value range (dB scale — typically -80 to +20 range)
            vmin, vmax = spec.min(), spec.max()
            if vmax - vmin < 1e-6:
                print(f"  WARN: {label} has near-constant values (range={vmax - vmin:.2e})")

    if ok:
        print(f"  OK: All sampled spectrograms have valid shape ({EXPECTED_N_FREQ}, T), float32, finite")
    return ok, samples


def check_normalization_stats(data_dir: Path) -> bool:
    """Check normalization_stats.npz for correctness."""
    stats_path = data_dir / "normalization_stats.npz"
    if not stats_path.exists():
        print("  FAIL: normalization_stats.npz missing")
        return False

    data = np.load(str(stats_path))
    ok = True

    for key in ("mean", "std", "count"):
        if key not in data:
            print(f"  FAIL: normalization_stats.npz missing '{key}'")
            ok = False

    if not ok:
        return False

    mean = data["mean"]
    std = data["std"]
    count = int(data["count"])

    if mean.shape != (EXPECTED_N_FREQ,):
        print(f"  FAIL: mean shape {mean.shape}, expected ({EXPECTED_N_FREQ},)")
        ok = False

    if std.shape != (EXPECTED_N_FREQ,):
        print(f"  FAIL: std shape {std.shape}, expected ({EXPECTED_N_FREQ},)")
        ok = False

    if count <= 0:
        print(f"  FAIL: count={count}, expected > 0")
        ok = False

    if np.any(std <= 0):
        n_zero = np.sum(std <= 0)
        print(f"  FAIL: {n_zero} frequency bins have std <= 0")
        ok = False

    if ok:
        print(f"  OK: normalization stats — mean/std shape ({EXPECTED_N_FREQ},), "
              f"count={count:,} frames, all std > 0")

    return ok


def print_dataset_summary(data_dir: Path) -> None:
    """Print a summary of the dataset across all splits."""
    print("\n=== Dataset Summary ===")
    total_bouts = 0
    total_frames = 0

    for split in SPLITS:
        split_dir = data_dir / split
        npy_files = sorted(split_dir.glob("spec_*.npy"))
        if not npy_files:
            print(f"  {split}: 0 bouts")
            continue

        frame_counts = []
        for path in npy_files:
            spec = np.load(str(path))
            frame_counts.append(spec.shape[1])

        n_bouts = len(npy_files)
        total_bouts += n_bouts
        total_frames += sum(frame_counts)

        arr = np.array(frame_counts)
        print(f"  {split}: {n_bouts} bouts, "
              f"frames: min={arr.min()}, median={int(np.median(arr))}, "
              f"max={arr.max()}, total={arr.sum():,}")

        # Read manifest for recording count
        manifest_path = split_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = json.load(f)
            rec_ids = manifest.get("recording_ids", [])
            unique_recs = set(rec_ids)
            print(f"         {len(unique_recs)} unique recordings")

    print(f"  TOTAL: {total_bouts} bouts, {total_frames:,} frames")

    # Pipeline config
    config_path = data_dir / "pipeline_config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
        print(f"\n  Pipeline config:")
        print(f"    Bout gap: {config['bout_extraction']['bout_gap_threshold_ms']}ms")
        print(f"    Padding: {config['bout_extraction']['context_padding_ms']}ms")
        print(f"    n_fft: {config['spectrogram']['n_fft']}")
        print(f"    hop: {config['spectrogram']['hop_length']}")
        print(f"    freq range: {config['spectrogram']['freq_min_hz']}-{config['spectrogram']['freq_max_hz']} Hz")


def plot_samples(
    samples: list[tuple[str, np.ndarray]], output_dir: Path,
) -> None:
    """Plot spectrogram samples as PNG files."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  SKIP: matplotlib not installed, cannot plot")
        return

    plot_dir = output_dir / "validation_plots"
    plot_dir.mkdir(exist_ok=True)

    for label, spec in samples[:10]:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.imshow(
            spec, aspect="auto", origin="lower",
            extent=[0, spec.shape[1], 20, 120],
        )
        ax.set_xlabel("Time frames")
        ax.set_ylabel("Frequency (kHz)")
        ax.set_title(label)
        plt.colorbar(ax.images[0], ax=ax, label="dB")

        safe_name = label.replace("/", "_").replace("\\", "_")
        fig.savefig(str(plot_dir / f"{safe_name}.png"), dpi=100, bbox_inches="tight")
        plt.close(fig)

    print(f"  OK: Saved {min(len(samples), 10)} plots to {plot_dir}")


def check_dataloader(data_dir: Path) -> bool:
    """Create a USVBoutDataset and fetch one batch to verify shapes."""
    try:
        from torch.utils.data import DataLoader

        from usv_language.data.dataset import TransformerDataConfig, USVBoutDataset
    except ImportError as e:
        print(f"  SKIP: {e}")
        return True

    train_dir = data_dir / "train"
    npy_files = sorted(train_dir.glob("spec_*.npy"))
    if len(npy_files) < 2:
        print("  SKIP: Not enough training spectrograms for DataLoader test")
        return True

    specs = [np.load(str(p)) for p in npy_files[:20]]
    rec_ids = [p.stem for p in npy_files[:20]]

    config = TransformerDataConfig(max_seq_len=256, batch_size=4)
    dataset = USVBoutDataset(specs, rec_ids, config)

    if len(dataset) == 0:
        print("  FAIL: USVBoutDataset has 0 items")
        return False

    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    batch = next(iter(loader))

    ok = True
    for key in ("input", "target", "mask"):
        if key not in batch:
            print(f"  FAIL: batch missing '{key}'")
            ok = False

    if ok:
        inp = batch["input"]
        print(f"  OK: DataLoader batch — input {tuple(inp.shape)}, "
              f"dtype={inp.dtype}, dataset size={len(dataset)}")

    return ok


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate prepared USV bout data",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        required=True,
        help="Directory containing prepared data (train/, val/, test/)",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=5,
        help="Number of random samples to check per split",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Save spectrogram sample plots",
    )
    parser.add_argument(
        "--check-dataloader",
        action="store_true",
        help="Test PyTorch DataLoader on training data",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_dir = Path(args.data_dir)

    if not data_dir.is_dir():
        print(f"FAIL: {data_dir} does not exist or is not a directory")
        return 1

    all_ok = True

    print("\n=== Check 1: Directory Structure ===")
    ok, counts = check_directory_structure(data_dir)
    all_ok &= ok

    print("\n=== Check 2: Random Sample Shapes ===")
    ok, samples = check_random_samples(data_dir, n_samples=args.n_samples)
    all_ok &= ok

    print("\n=== Check 3: Normalization Stats ===")
    ok = check_normalization_stats(data_dir)
    all_ok &= ok

    print_dataset_summary(data_dir)

    if args.plot and samples:
        print("\n=== Plotting Samples ===")
        plot_samples(samples, data_dir)

    if args.check_dataloader:
        print("\n=== Check 4: DataLoader ===")
        ok = check_dataloader(data_dir)
        all_ok &= ok

    print("\n" + "=" * 50)
    if all_ok:
        print("ALL CHECKS PASSED")
        return 0
    else:
        print("SOME CHECKS FAILED — see above")
        return 1


if __name__ == "__main__":
    sys.exit(main())

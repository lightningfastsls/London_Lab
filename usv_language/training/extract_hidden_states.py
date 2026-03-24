"""Extract hidden states from a trained spectrogram transformer.

Runs inference on all sequences with return_hidden_states=True and saves
the hidden representations as memory-mapped numpy arrays for downstream
analysis (Phase 2 VQ-VAE).

Usage:
    python -m usv_language.training.extract_hidden_states \
        --checkpoint /path/to/best.pt \
        --data-dir /path/to/bout_spectrograms \
        --output-dir /path/to/hidden_states \
        --layers 2 4 6 8
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from usv_language.data.dataset import TransformerDataConfig, USVBoutDataset
from usv_language.data.normalization import load_normalization_stats, normalize
from usv_language.models.transformer import SpectrogramTransformer, TransformerConfig
from usv_language.training.train_transformer import (
    coerce_transformer_config,
    load_bout_spectrograms,
)

logger = logging.getLogger(__name__)


def count_total_frames(dataset: USVBoutDataset) -> int:
    """Count total valid (non-padding) frames across all chunks."""
    return sum(max(0, length - 1) for length in dataset.get_lengths())


def validate_extraction_layers(
    target_layers: list[int],
    primary_layer: int,
    n_layers: int,
) -> list[int]:
    """Validate requested extraction layers and return zero-indexed layer ids."""
    layer_indices = [int(layer) - 1 for layer in target_layers]
    for idx in layer_indices:
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer {idx + 1} out of range [1, {n_layers}]")
    if primary_layer not in target_layers:
        raise ValueError(
            f"primary_layer={primary_layer} must be included in extracted layers "
            f"{target_layers}"
        )
    return layer_indices


def extract_hidden_states(args: argparse.Namespace) -> None:
    """Main extraction entry point."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)
    config = coerce_transformer_config(checkpoint.get("config"))

    logger.info("Loaded checkpoint from %s (epoch %d)", checkpoint_path, checkpoint["epoch"])

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model
    model = SpectrogramTransformer(config).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    # Parse target layers (1-indexed in CLI, 0-indexed internally)
    target_layers = [int(l) for l in args.layers]
    layer_indices = validate_extraction_layers(
        target_layers, args.primary_layer, config.n_layers,
    )

    # Load data
    data_dir = Path(args.data_dir)
    spectrograms, recording_ids = load_bout_spectrograms(data_dir)

    stats_path = data_dir / "normalization_stats.npz"
    if stats_path.exists():
        stats = load_normalization_stats(stats_path)
        spectrograms = [normalize(s, stats) for s in spectrograms]

    data_config = TransformerDataConfig(
        max_seq_len=config.max_seq_len,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        overlap_ratio=0.0,  # No overlap for extraction (no duplicated frames)
    )

    dataset = USVBoutDataset(
        spectrograms, recording_ids, data_config,
        training=False,
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size,
        shuffle=False, num_workers=args.num_workers, pin_memory=True,
    )

    total_frames = count_total_frames(dataset)
    logger.info("Dataset: %d chunks, ~%d valid frames", len(dataset), total_frames)

    # Pre-allocate memory-mapped arrays
    mmap_files: dict[int, np.memmap] = {}
    for layer_num in target_layers:
        path = output_dir / f"hidden_states_layer{layer_num}.npy"
        mmap = np.lib.format.open_memmap(
            str(path), mode="w+", dtype=np.float32,
            shape=(total_frames, config.d_model),
        )
        mmap_files[layer_num] = mmap

    # Metadata: maps frame index to source info and recording-level spans
    metadata_entries: list[dict] = []
    recording_entries: list[dict] = []
    current_recording_id: str | None = None
    current_recording_start = 0
    current_recording_frames = 0
    frame_offset = 0

    # Extract
    logger.info("Extracting hidden states from layers %s...", target_layers)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            inputs = batch["input"].to(device)
            mask = batch["mask"].to(device)
            padding_mask = ~mask.bool()

            _, hidden_states = model(
                inputs, attention_mask=padding_mask, return_hidden_states=True,
            )

            # Process each item in the batch
            batch_size = inputs.shape[0]
            for i in range(batch_size):
                chunk_idx = batch_idx * args.batch_size + i
                if chunk_idx >= len(dataset):
                    break

                item_mask = mask[i]  # (seq_len,)
                valid_len = int(item_mask.sum().item())

                if valid_len == 0:
                    continue

                rec_id = batch["recording_id"][i]

                # Extract valid frames from each target layer
                for layer_num, layer_idx in zip(target_layers, layer_indices):
                    h = hidden_states[layer_idx][i, :valid_len, :]  # (valid_len, d_model)
                    end = frame_offset + valid_len
                    mmap_files[layer_num][frame_offset:end] = h.cpu().numpy()

                rec_id = str(rec_id)

                if current_recording_id is None:
                    current_recording_id = rec_id
                    current_recording_start = frame_offset
                elif rec_id != current_recording_id:
                    recording_entries.append({
                        "recording_id": current_recording_id,
                        "start_frame": current_recording_start,
                        "end_frame": current_recording_start + current_recording_frames,
                        "n_frames": current_recording_frames,
                    })
                    current_recording_id = rec_id
                    current_recording_start = frame_offset
                    current_recording_frames = 0

                # Record metadata for each frame
                for f in range(valid_len):
                    metadata_entries.append({
                        "frame_index": frame_offset + f,
                        "chunk_index": chunk_idx,
                        "recording_id": rec_id,
                        "frame_within_chunk": f,
                    })

                current_recording_frames += valid_len
                frame_offset += valid_len

            if (batch_idx + 1) % 10 == 0:
                logger.info(
                    "  Batch %d/%d, frames extracted: %d/%d",
                    batch_idx + 1, len(loader), frame_offset, total_frames,
                )

    # Flush and close mmaps (must close before trim on Windows)
    for mmap in mmap_files.values():
        mmap.flush()
        del mmap
    mmap_files.clear()

    # Trim if we extracted fewer frames than estimated
    if frame_offset < total_frames:
        logger.info("Trimming arrays from %d to %d frames", total_frames, frame_offset)
        for layer_num in target_layers:
            path = output_dir / f"hidden_states_layer{layer_num}.npy"
            full = np.load(str(path), mmap_mode="r")
            trimmed = full[:frame_offset].copy()
            np.save(str(path), trimmed)

    if current_recording_id is not None:
        recording_entries.append({
            "recording_id": current_recording_id,
            "start_frame": current_recording_start,
            "end_frame": current_recording_start + current_recording_frames,
            "n_frames": current_recording_frames,
        })

    # Save metadata
    metadata = {
        "total_frames": frame_offset,
        "d_model": config.d_model,
        "layers_extracted": target_layers,
        "primary_layer": args.primary_layer,
        "checkpoint": str(checkpoint_path),
        "checkpoint_epoch": checkpoint["epoch"],
        "recordings": recording_entries,
        "frames": metadata_entries,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    logger.info(
        "Extraction complete. %d frames from %d layers saved to %s",
        frame_offset, len(target_layers), output_dir,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    p = argparse.ArgumentParser(
        description="Extract hidden states from trained spectrogram transformer",
    )
    p.add_argument("--checkpoint", type=str, required=True,
                    help="Path to model checkpoint (.pt)")
    p.add_argument("--data-dir", type=str, required=True,
                    help="Directory with bout spectrograms")
    p.add_argument("--output-dir", type=str, required=True,
                    help="Directory for output hidden state arrays")
    p.add_argument("--layers", type=int, nargs="+", default=[2, 4, 6, 8],
                    help="Layer numbers to extract (1-indexed)")
    p.add_argument("--primary-layer", type=int, default=4,
                    help="Primary layer for VQ-VAE (1-indexed)")
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=4)

    return p.parse_args(argv)


if __name__ == "__main__":
    extract_hidden_states(parse_args())

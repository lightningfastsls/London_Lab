"""Streaming STFT utilities for large WAV files."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator, Tuple

import numpy as np
from scipy import signal
import soundfile as sf
from tqdm import tqdm

from .config import SpectrogramConfig
from ._stft_core import compute_stft_frames_db, extract_frames


def stream_wav_spectrogram_db(
    path: str | Path,
    cfg: SpectrogramConfig,
    block_size_samples: int | None = None,
    progress: bool = False,
) -> Iterator[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Stream a WAV file and yield spectrogram chunks in dB.

    Parameters
    ----------
    path:
        Path to the input WAV file.
    cfg:
        Spectrogram configuration.
    block_size_samples:
        Number of samples to read per block. Defaults to config stream size.
    progress:
        Whether to show a tqdm progress bar.

    Yields
    ------
    spec_db:
        Spectrogram chunk in dB, shape (freqs, frames).
    freqs_hz:
        Frequency bins in Hz for the chunk.
    times_s:
        Time bins in seconds for the chunk.
    """
    block_size = block_size_samples or cfg.stream_block_size_samples
    if block_size <= 0:
        raise ValueError("block_size_samples must be a positive integer.")

    with sf.SoundFile(str(path)) as wav:
        sample_rate_hz = int(wav.samplerate)
        if cfg.enforce_sample_rate and sample_rate_hz != cfg.expected_sample_rate_hz:
            raise ValueError(
                f"Expected {cfg.expected_sample_rate_hz} Hz, got {sample_rate_hz} Hz."
            )

        hop_length = cfg.hop_length(sample_rate_hz)
        if hop_length >= cfg.window_length:
            raise ValueError("Hop length must be smaller than window length.")

        n_fft = cfg.n_fft()
        window = signal.get_window(cfg.window, cfg.window_length, fftbins=True)
        freqs_hz = np.fft.rfftfreq(n_fft, d=1.0 / sample_rate_hz)
        band_mask = (freqs_hz >= cfg.f_min_hz) & (freqs_hz <= cfg.f_max_hz)

        # Use list for O(1) append instead of np.concatenate (O(n²))
        buffer_chunks: list[np.ndarray] = []
        buffer_len = 0
        frame_offset = 0
        total_frames = wav.frames

        progress_bar = (
            tqdm(total=total_frames, unit="frames") if progress else None
        )
        try:
            while True:
                block = wav.read(block_size, dtype="float32", always_2d=True)
                if block.size == 0:
                    break
                if progress_bar:
                    progress_bar.update(block.shape[0])

                # Convert to mono
                if block.shape[1] > 1:
                    block = block.mean(axis=1)
                else:
                    block = block[:, 0]

                # O(1) append to list
                buffer_chunks.append(block)
                buffer_len += len(block)

                while buffer_len >= cfg.window_length:
                    # Convert to contiguous array only when needed
                    buffer = np.concatenate(buffer_chunks)

                    n_frames = 1 + (len(buffer) - cfg.window_length) // hop_length
                    if n_frames <= 0:
                        break

                    # Extract frames and compute STFT using shared helper
                    frames = extract_frames(buffer, cfg.window_length, hop_length)
                    spec_db = compute_stft_frames_db(frames, window, n_fft, band_mask, cfg.eps)

                    times_s = (
                        (frame_offset + np.arange(n_frames)) * hop_length
                        + cfg.window_length / 2.0
                    ) / sample_rate_hz
                    frame_offset += n_frames
                    yield spec_db, freqs_hz[band_mask], times_s

                    # Keep only the remainder after processed frames
                    frame_start = n_frames * hop_length
                    remainder = buffer[frame_start:]
                    buffer_chunks = [remainder] if len(remainder) > 0 else []
                    buffer_len = len(remainder)
        finally:
            if progress_bar:
                progress_bar.close()

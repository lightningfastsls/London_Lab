"""Configuration for Phase 8.4 analysis and interpretation tools.

AnalysisConfig holds parameters spanning all five analysis modules:
codebook visualization, sequence analysis, concept manipulation,
context-dependent analysis, and compositionality tests.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AnalysisConfig:
    """Configuration for the VQ-VAE analysis suite.

    Parameters
    ----------
    n_exemplars:
        Number of exemplar frames to find per codebook entry.
    context_frames:
        Context window around exemplar frames for gallery plots.
    zipf_min_count:
        Minimum code occurrence count for Zipf analysis.
    max_ngram_order:
        Maximum n-gram order for entropy rate computation.
    manipulation_n_future_steps:
        Number of autoregressive future steps for concept injection.
    freq_min_hz:
        Minimum frequency for spectrogram axis labels.
    freq_max_hz:
        Maximum frequency for spectrogram axis labels.
    n_freq:
        Number of frequency bins in spectrogram output.
    source_layer:
        1-indexed transformer layer where hidden states were captured.
    tsne_perplexity:
        Perplexity for t-SNE codebook projection.
    umap_n_neighbors:
        Number of neighbors for UMAP codebook projection.
    output_dpi:
        DPI for saved figures.
    """

    n_exemplars: int = 10
    context_frames: int = 50
    zipf_min_count: int = 5
    max_ngram_order: int = 8
    manipulation_n_future_steps: int = 50
    freq_min_hz: int = 20_000
    freq_max_hz: int = 120_000
    n_freq: int = 170
    source_layer: int = 4
    tsne_perplexity: float = 5.0
    umap_n_neighbors: int = 10
    output_dpi: int = 150

    def __post_init__(self) -> None:
        if self.n_exemplars <= 0:
            raise ValueError(
                f"n_exemplars must be positive, got {self.n_exemplars}"
            )
        if self.max_ngram_order <= 0:
            raise ValueError(
                f"max_ngram_order must be positive, got {self.max_ngram_order}"
            )
        if self.source_layer <= 0:
            raise ValueError(
                f"source_layer must be positive (1-indexed), got {self.source_layer}"
            )
        if self.freq_min_hz >= self.freq_max_hz:
            raise ValueError(
                f"freq_min_hz ({self.freq_min_hz}) must be < "
                f"freq_max_hz ({self.freq_max_hz})"
            )
        if self.n_freq <= 0:
            raise ValueError(f"n_freq must be positive, got {self.n_freq}")

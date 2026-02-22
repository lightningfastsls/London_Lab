"""VQ-VAE analysis and interpretation tools (Phase 8.4).

Scientific analysis suite for the two-phase architecture (ADR-007):
SpectrogramTransformer (frozen) + HiddenStateVQVAE (frozen).

Modules:
    config              -- AnalysisConfig dataclass
    transformer_suffix  -- Decode hidden states through remaining transformer layers
    codebook_viz        -- Codebook visualization and exemplar galleries
    sequence_analysis   -- Zipf, transitions, entropy, mutual information
    concept_manipulation -- Concept injection experiments
    context_analysis    -- Metadata-based group comparison
    compositionality    -- Bigram productivity, positional independence
    run_analysis        -- CLI entry point
"""

from usv_language.analysis.config import AnalysisConfig

__all__ = ["AnalysisConfig"]

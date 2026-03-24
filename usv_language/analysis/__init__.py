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
    information_theory  -- Statistically rigorous info theory (MLE Zipf, bias-corrected
                           entropy, idiom detection, burstiness)
    null_models         -- Surrogate generators for hierarchical hypothesis testing
    statistical_tests   -- Null model comparison framework (NullModelComparison)
    acoustic_properties -- Ground-truth acoustic property extractors (probing targets)
    probing             -- Linear/MLP probing framework for hidden state interpretability
    run_analysis        -- CLI entry point
"""

from usv_language.analysis.acoustic_properties import (
    AcousticPropertyConfig,
    extract_all_properties,
)
from usv_language.analysis.config import AnalysisConfig
from usv_language.analysis.null_models import NullModelConfig, NullModelGenerator
from usv_language.analysis.statistical_tests import (
    MetricComparison,
    FullAnalysisResult,
    NullModelComparison,
)
from usv_language.analysis.probing import (
    ProbingConfig,
    ProbingResult,
    ProbingAnalysisResult,
    ProbingExperiment,
    ProbingAnalysisPipeline,
    plot_probing_heatmap,
    plot_layer_comparison,
)

__all__ = [
    "AcousticPropertyConfig",
    "extract_all_properties",
    "AnalysisConfig",
    "NullModelConfig",
    "NullModelGenerator",
    "MetricComparison",
    "FullAnalysisResult",
    "NullModelComparison",
    "ProbingConfig",
    "ProbingResult",
    "ProbingAnalysisResult",
    "ProbingExperiment",
    "ProbingAnalysisPipeline",
    "plot_probing_heatmap",
    "plot_layer_comparison",
]

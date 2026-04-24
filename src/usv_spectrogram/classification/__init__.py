"""USV classification utilities — format adapters for interop with external tools."""

from __future__ import annotations

from .raven_export import (
    RavenExportConfig,
    ExportSummary,
    load_detection_json,
    discover_wav_detection_mapping,
    detections_to_raven_table,
    export_raven_tables,
)

from .deepsqueak_import import (
    DeepSqueakImportConfig,
    ImportSummary,
    load_deepsqueak_excel,
    load_all_deepsqueak_results,
    load_detections_for_merge,
    merge_with_detections,
    export_classified_detections,
    import_deepsqueak_results,
)

from .repertoire_stats import (
    RepertoireConfig,
    syllable_proportions,
    syllable_diversity,
    transition_matrix,
    compare_repertoires,
    compare_transition_matrices,
    plot_repertoire_comparison,
    generate_report,
    analyze_repertoire,
)

from .sis_baselines import (
    SISResult,
    compute_sis_depth_1,
)

from .cross_population import (
    CrossPopulationComparison,
    ComparisonReport,
    ComparisonMetadata,
    TypeProportionResult,
    JSDResult,
    EntropyResult,
    TransitionResult,
    MILag1Result,
    ZipfPopResult,
    ZipfComparisonResult,
    BurstinessResult,
    IOIDistributionResult,
    FeatureComparison,
    FeatureComparisonResult,
    UMAPOverlapResult,
    SCHEMA_VERSION as CROSS_POPULATION_SCHEMA_VERSION,
    CANONICAL_BOUT_THRESHOLD_S,
    DEFAULT_FEATURE_COLUMNS,
)

__all__ = [
    # Raven export
    "RavenExportConfig",
    "ExportSummary",
    "load_detection_json",
    "discover_wav_detection_mapping",
    "detections_to_raven_table",
    "export_raven_tables",
    # DeepSqueak import
    "DeepSqueakImportConfig",
    "ImportSummary",
    "load_deepsqueak_excel",
    "load_all_deepsqueak_results",
    "load_detections_for_merge",
    "merge_with_detections",
    "export_classified_detections",
    "import_deepsqueak_results",
    # Repertoire statistics
    "RepertoireConfig",
    "syllable_proportions",
    "syllable_diversity",
    "transition_matrix",
    "compare_repertoires",
    "compare_transition_matrices",
    "plot_repertoire_comparison",
    "generate_report",
    "analyze_repertoire",
    # SIS baselines
    "SISResult",
    "compute_sis_depth_1",
    # Cross-population comparison
    "CrossPopulationComparison",
    "ComparisonReport",
    "ComparisonMetadata",
    "TypeProportionResult",
    "JSDResult",
    "EntropyResult",
    "TransitionResult",
    "MILag1Result",
    "ZipfPopResult",
    "ZipfComparisonResult",
    "BurstinessResult",
    "IOIDistributionResult",
    "FeatureComparison",
    "FeatureComparisonResult",
    "UMAPOverlapResult",
    "CROSS_POPULATION_SCHEMA_VERSION",
    "CANONICAL_BOUT_THRESHOLD_S",
    "DEFAULT_FEATURE_COLUMNS",
]

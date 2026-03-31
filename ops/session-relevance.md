# Session Relevance Brief
<!-- Generated: 2026-03-31 13:30 -->
<!-- Method: topic-map-traversal + ripgrep -->

## DeepSqueak Classification Bridge
- "DeepSqueak import previously required exact subdirectory name matches while Raven export already supported prefix matches creating a silent asymmetric round-trip" (finding) -- the 2026-03-07 bug: export supported prefix match, import required exact name, breaking round-trips for suffixed dirs
- "DeepSqueak regenerates its own spectrograms from raw audio so exported bounding boxes serve as regions of interest not precise frequency boundaries" (finding) -- frequency bounds need only be approximate
- "Raven selection table format is the standard interchange format between bioacoustic analysis tools" (method) -- tab-separated .txt format used by Raven Pro, DeepSqueak, Audacity

## Phase 5.3
- "DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method" (finding) -- k-means finds k=20 but GMM finds k<=2 on learned representations
- "SqueakOut autoencoder segmentation achieves Dice 90.2 designed to feed downstream unsupervised clustering pipelines" (finding) -- upstream segmentation improves downstream clustering quality
- "supervised bioacoustic foundation models vastly outperform self-supervised for species-level clustering" (finding) -- Muenster 2025: supervised 0.418 AMI vs self-supervised 0.256



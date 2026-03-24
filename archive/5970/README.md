# 5970 Archive

## Status: Labeling complete (partial)

Labeled USV_1 and USV_5 sessions only. USV_2, USV_3, USV_4 were not labeled.
Decision: sufficient detections (714) from this group for CNN training with augmentation.
Moving to other mice groups for dataset diversity.

## Stats
- **Files in unreviewed/**: 1,029 WAVs (sampled subset from full recordings)
- **Files in reviewed/**: 270 WAVs (USV_1: 210, USV_5: 60)
- **Detection folders**: 92 (in USV_Detections/5970/)
- **Noise labels**: 284 JSONs (in USV_Detections/5970/noise_labeled_files/)
- **Rejected detections**: 33 folders (in USV_Detections/5970/rejected_detections/)
- **Total detections**: 714 (USV_1: 270, USV_5: 444)
- **Sessions labeled**: USV_1, USV_5
- **Sessions not labeled**: USV_2, USV_3, USV_4

## Label Generation Estimate
- Base labels (40ms window, 10ms hop): ~2,855
- With constrained jittering (5x): ~14,275
- With half-masking augmentation (2x): ~5,710
- With both (10x): ~28,550

## Original Source
Full recordings (6,494 WAVs across USV_1-5) exist on external drive.
Only the sampled subset is archived here.

## Date archived: 2026-02-16

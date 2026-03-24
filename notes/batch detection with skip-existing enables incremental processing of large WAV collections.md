---
description: Headless CNN sliding-window inference over ~6,500 WAV files with skip-existing logic, error recovery, per-file JSON output, and summary CSV; targets 5+ files/minute on CPU.
type: method
confidence: likely
topics:
  - "[[detection]]"
---

# batch detection with skip-existing enables incremental processing of large WAV collections

Processing a collection of approximately 6,500 WAV recordings requires an architecture that can be interrupted and resumed without reprocessing completed files. A batch detection script that tracks which files have already been processed — by checking for the existence of an output JSON before starting inference on each WAV — enables incremental operation. New recordings can be added to the input directory and the script re-run; previously processed files are skipped, making re-runs proportional in cost to the number of new files.

The detection pipeline applies a CNN sliding window across each recording, implementing [[two-stage detection uses permissive energy detector followed by CNN precision filter]]. The energy detector rapidly identifies candidate windows, and the CNN evaluates each candidate to reject false positives. Per-file output is written as a detection JSON in ADR-010 format, recording the timing and confidence of each detected call. A summary CSV aggregates counts and metadata across all files in the collection, enabling downstream analysis without parsing individual JSONs.

Error recovery is a design requirement for batch processing at this scale. Individual WAV files may be corrupt, truncated, or have unexpected sample rates. The script catches per-file exceptions, logs the error with the filename and traceback to a separate error log, and continues to the next file. This ensures a single bad recording does not halt a multi-hour batch run. See [[auto sample rate reading from WAV headers prevents silent frequency miscalculation]] for the mechanism that handles sample rate variation across files.

The processing rate target of more than 5 files per minute on CPU is a practical constraint given the hardware environment. At this rate, the full 6,500-file collection completes in approximately 22 hours. GPU inference would substantially exceed this target, but CPU compatibility ensures the script can run on any lab machine without GPU configuration. The skip-existing logic makes it feasible to run the script across multiple sessions, pausing and resuming as needed. Batch detection generates the raw candidate pool that feeds the [[active learning cycle automates the label-train-evaluate-mine loop for iterative CNN improvement]], and the detection JSONs it produces can be exported to [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] for cross-tool classification via DeepSqueak -- each new batch run surfaces additional unlabeled candidates that can be prioritized for human review via hard negative mining.

For format interoperability, [[Crowsetta standardizes annotation format interoperability across bioacoustic tools via a unified Python API]] could serve as the translation layer between our JSON output format and tools like Raven and DeepSqueak. Alternatively, [[OpenSoundscape provides a full ML training pipeline with native Raven format support for bioacoustic classification]] offers built-in batch processing with Raven integration that could complement our pipeline for classification tasks.

---

Source: [ROADMAP](../ROADMAP.md), Phase 3

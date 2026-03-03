---
description: "BoxedAnnotations class, CNN transfer learning, active learning workflows, and batch prediction — a Python framework that reads Raven tables and trains classifiers without custom data loading code"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[classification]]"
  - "[[detection-landscape]]"
---

# OpenSoundscape provides a full ML training pipeline with native Raven format support for bioacoustic classification

OpenSoundscape (Lapp et al., 2023, Methods in Ecology and Evolution) provides a complete ML pipeline for bioacoustic classification: data preparation from Raven selection tables via the BoxedAnnotations class, CNN-based classification with transfer learning, active learning workflows, and batch prediction on large audio collections.

Unlike building a custom training pipeline, OpenSoundscape handles the data loading, augmentation, and evaluation plumbing that typically requires significant engineering effort. For our pipeline, the most relevant capabilities are: (1) native Raven format reading — our existing Raven export adapter produces exactly the format OpenSoundscape consumes, (2) active learning support — iteratively refining classifiers with focused human annotation, (3) CNN transfer learning — using pretrained features rather than training from scratch.

The tool is focused on birds and general bioacoustics with no USV-specific features, but the architecture is species-agnostic. Since [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]], our Raven export adapter creates a direct integration path to OpenSoundscape's training pipeline. This could serve as an alternative training framework if we move beyond our custom CNN, particularly for experimenting with transfer learning from general audio models.

The combination of OpenSoundscape's Raven import with [[Crowsetta standardizes annotation format interoperability across bioacoustic tools via a unified Python API]] means our detected USV candidates could flow through multiple classification frameworks without custom format adapters for each one.

---

Source:
- bioacoustic-annotation-tools-landscape-2025-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] — OpenSoundscape reads this natively
- [[active learning annotation workflows are the frontier in bioacoustic tools]] — OpenSoundscape implements this paradigm
- [[Crowsetta standardizes annotation format interoperability across bioacoustic tools via a unified Python API]] — complementary interop layer

Topics:
- [[classification]]
- [[detection]]

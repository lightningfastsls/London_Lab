---
description: "Python/FastAPI backend + React frontend supports project management, multi-annotator workflows, model prediction import, and spectrogram visualization — published Dec 2024 in Methods in Ecology"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[detection]]"
---

# Whombat is the first web-based platform for collaborative bioacoustic annotation with ML-assisted review

Whombat (Martinez Balvanera et al., 2025) addresses two capabilities most lacking in USV-specific tools: collaborative annotation and ML-assisted review. Built on a Python/FastAPI backend with React frontend, it provides browser-based spectrogram annotation accessible to non-programmers.

Key features include project management (organize recordings, assign annotators), ML-assisted labeling (import model predictions for human correction/confirmation), and multi-annotator review workflows. Unlike Raven (desktop-only, manual-only) or DeepSqueak (MATLAB, single-user), Whombat supports the iterative ML workflow that has become best practice: annotate, train, predict, review, repeat.

The frequency range is not limited, so it should handle ultrasonic spectrograms if configured correctly. This makes it potentially applicable to USV research, though this has not been tested. As a December 2024 publication, it has a smaller community than established tools, and its custom JSON export format (COCO-inspired) means interop with [[Raven selection table format is the standard interchange format between bioacoustic analysis tools]] requires conversion — which [[Crowsetta standardizes annotation format interoperability across bioacoustic tools via a unified Python API]] could handle.

The significance of Whombat is that it fills the [[no single bioacoustic tool covers the full detection-annotation-review-export pipeline]] gap more completely than any other tool by combining annotation, ML-assisted review, and collaborative workflows in a single platform.

---

Source:
- bioacoustic-annotation-tools-landscape-2025-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[no single bioacoustic tool covers the full detection-annotation-review-export pipeline]] — Whombat comes closest to filling this gap
- [[active learning annotation workflows are the frontier in bioacoustic tools]] — Whombat implements this paradigm

Topics:
- [[detection]]

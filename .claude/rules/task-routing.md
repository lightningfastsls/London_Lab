---
paths:
  - "src/**"
  - "usv_language/**"
  - "scripts/**"
---
## Task Routing

| Task Type | Start With | Reference Doc |
|-----------|-----------|---------------|
| Spectrogram / STFT changes | `spectrogram.py`, `_stft_core.py`, `config.py` | `docs/reference/usv_signal_processing_reference.md` |
| Detection pipeline (production) | `scripts/run_batch_detection.py`, `app/core/sliding_inference.py`, `app/core/audio_loader.py`, `postprocessing/` | `docs/modules/cnn-classifier.md`, `docs/handoffs/v2-full-pipeline-results.md` |
| Legacy energy detector (tuning/tests only) | `detection/energy_detector.py`, `detection/config.py` | `docs/modules/energy-detector.md` |
| CNN training / evaluation | `models/cnn_classifier.py`, `models/trainer.py` | `docs/modules/cnn-classifier.md` |
| Training data assembly | `dataset/assembler.py`, `scripts/assemble_training_data.py` | `docs/modules/dataset-assembler.md` |
| PyQt6 desktop app | `app/main_window.py`, `app/core/`, `app/widgets/` | `docs/plans/USV_DETECTION_APP_IMPLEMENTATION.md` |
| Labeling tool (Streamlit) | `labeling/labeling_app.py` | `docs/LABELING_TOOL_QUICKSTART.md` |
| Parameter Lab (Streamlit) | `param_lab/app.py`, `param_lab/ui/` | -- |
| Clustering / repertoire | `clustering/`, `classification/repertoire_stats.py` | `docs/modules/repertoire-stats.md` |
| DeepSqueak / Raven bridge | `classification/raven_export.py`, `classification/deepsqueak_import.py` | `docs/modules/raven-export.md`, `docs/modules/deepsqueak-import.md` |
| VQ-VAE / Transformer | `usv_language/models/`, `usv_language/training/` | `docs/plans/vq_vae_transformer_plan.md` |
| LMT behavioral integration | `lmt/`, `scripts/run_event_triggered_analysis.py` | `docs/modules/event-triggered-analysis.md` |
| Script index (all ~76) | -- | `docs/scripts-index.md` |

All `src/` paths above are relative to `src/usv_spectrogram/` unless they start with `usv_language/`.

## Key Reference Documents

| Document | When to Read |
|----------|--------------|
| `ops/goals.md` | **Start of every session** (session state, active threads) |
| `notes/index.md` + topic maps | **Before any architectural/design choice** (domain knowledge) |
| `ROADMAP*.md` / plan files | Before implementing -- check relevant plan (no single master ROADMAP) |
| `docs/architecture/patterns.md` | Before implementing (follow established patterns) |
| `docs/workflow/completion-sequence.md` | When implementing 2+ file changes (includes handoff rules) |
| `docs/reviews/REVIEW-TEMPLATE.md` | When writing handoff or requesting review (includes tier system) |
| `docs/workflow/approval-request-template.md` | Full approval request + struggle protocol templates |
| `docs/workflow/knowledge-graph-reference.md` | Full verbose KG section details |
| `docs/plans/USV_TRAINING_PIPELINE_PLAN.md` | Building training data generation pipeline |
| `docs/plans/USV_DETECTION_APP_IMPLEMENTATION.md` | Building PyQt6 desktop app for detection |
| `docs/reference/usv_signal_processing_reference.md` | Any signal processing work |
| `IMPLEMENTATION_PROGRESS.md` | **Append after implementation** (session archive, never modify existing entries) |

**After implementing a module**, also update: module doc (`docs/modules/<module>.md`), `docs/architecture/patterns.md` (if new pattern), create decision note in `notes/` + run `/reflect` (if non-obvious architectural decision).

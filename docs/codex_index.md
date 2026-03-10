# Codex Index

Compact navigation map for Codex sessions in `mickey_london_lab`.

## Start Here

Minimum startup read order:
1. `AGENTS.md`
2. `ops/goals.md`
3. `ops/reminders.md`
4. task-specific docs below

Use this file as a router, not as a substitute for the source documents.

## Repo Boundaries

Claude-owned by default:
- `.claude/`
- `ops/`
- `notes/`
- `methodology/`
- `reference/`
- `templates/`
- `inbox/`

Codex-writable by default:
- `src/`
- `tests/`
- `scripts/`
- `usv_language/`
- `docs/handoffs/`
- targeted files under `docs/`

If something should survive the session, prefer a handoff in `docs/handoffs/` instead of writing directly into Claude-managed memory systems.

## Common Task Routing

| Task | Start In Code | Read Before Editing |
|------|---------------|---------------------|
| STFT / spectrogram | `src/usv_spectrogram/spectrogram.py`, `src/usv_spectrogram/_stft_core.py` | `docs/reference/usv_signal_processing_reference.md` |
| Detection pipeline | `src/usv_spectrogram/detection/` | `docs/modules/energy-detector.md` |
| PyQt desktop app | `src/usv_spectrogram/app/` | `docs/plans/USV_DETECTION_APP_IMPLEMENTATION.md` |
| Streamlit tools | `src/usv_spectrogram/param_lab/`, `src/usv_spectrogram/labeling/` | `docs/LABELING_TOOL_QUICKSTART.md` when relevant |
| CNN training / evaluation | `src/usv_spectrogram/models/` | `docs/modules/cnn-classifier.md` |
| Training data assembly | `src/usv_spectrogram/dataset/` | `docs/modules/dataset-assembler.md` |
| Classification bridge | `src/usv_spectrogram/classification/` | `docs/modules/raven-export.md`, `docs/modules/deepsqueak-import.md` |
| VQ-VAE / Transformer | `usv_language/models/`, `usv_language/training/`, `usv_language/analysis/` | `docs/plans/vq_vae_transformer_plan.md` |
| LMT integration | `src/usv_spectrogram/lmt/` | `docs/modules/event-triggered-analysis.md` |
| Scripts landscape | `scripts/` | `docs/scripts-index.md` |

## Architecture Rules To Preserve

Confirmed in `docs/architecture/patterns.md`:
- config-heavy modules use frozen dataclasses with `__post_init__` validation
- scripts bootstrap `src/` into `sys.path`
- PyQt keeps business logic in `app/core/` and rendering in `app/widgets/`
- tests use synthetic fixtures, not real recordings
- shared STFT behavior lives in `_stft_core.py`

Hard constraints that matter often:
- always specify `sr=300000`
- do not casually change STFT or detection behavior without reading the relevant docs
- do not change test expectations to force a pass

## Scripts And Tests

Standard commands:

```powershell
.\.venv\Scripts\python.exe <script>
.\.venv\Scripts\python.exe -m pytest tests/ -v
.\.venv\Scripts\python.exe -m py_compile <file.py>
```

Use `docs/scripts-index.md` to locate entry points by workflow area.

Test locations:
- `tests/` for the main suite
- script-specific checks sometimes live under `scripts/` as diagnostic helpers

For code changes, run targeted validation first, then widen only if needed.

## Current State

High-signal active items from `ops/goals.md`:
- DeepSqueak Classification Bridge is in progress, currently Phase 3: MATLAB import and clustering
- Phase 5.2 two-week validation checkpoint is active as of 2026-03-06

Before starting adjacent work, check whether your task overlaps either thread.

## Durable Output

`docs/handoffs/` is the Codex handoff area.

Use it for:
- non-trivial implementation summaries
- architectural reasoning worth preserving
- unresolved risks or follow-up work
- context Claude may want to ingest into its own systems later
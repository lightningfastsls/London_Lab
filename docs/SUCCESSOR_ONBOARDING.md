# Successor Onboarding — USV Analysis Toolkit

A getting-started guide for whoever inherits this repo. Goal: clone → install →
run the detection pipeline and the desktop app, without prior context.

> Environment reality: this repo is developed on **WSL/Linux (POSIX)**, Python
> **3.12**. Commands use `.venv/bin/python`, not the Windows `.\.venv\Scripts\python.exe`
> you may see in older docs. There is also a **GPU rig** (`cloudyclaude`) used for
> training — see "The box-vs-rig split" below.

---

## 1. Install

```bash
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements_frozen.txt   # pinned, reproduces the production model env
.venv/bin/python -m pip install -e .                         # makes `import usv_spectrogram` work everywhere
```

`requirements_frozen.txt` is the pinned set (use it, not the unpinned
`requirements.txt`). The editable install (`-e .`) replaces the legacy
`sys.path.insert(".../src")` hacks scattered through scripts.

Smoke-test the install:

```bash
.venv/bin/python -m pytest -q          # expect ~1584 passing, 0 collection errors
.venv/bin/python -c "import usv_spectrogram; print('ok')"
```

## 2. Run the production detection pipeline

The **only** supported batch command (all five flags are required — omitting the
fp-filter or hysteresis-config silently produces incomplete triage):

```bash
.venv/bin/python scripts/run_batch_detection.py \
    --wav-dir <WAV_FOLDER>/ \
    --model models/hard_neg_retrain/best_model.pt \
    --output-dir results/batch_<NAME>/ \
    --temperature models/hard_neg_retrain/temperature.json \
    --fp-filter models/hard_neg_retrain/fp_filter.pkl \
    --hysteresis-config models/hard_neg_retrain/hysteresis_optimization_v2.json \
    --workers 4
```

The production CNN is `models/hard_neg_retrain/best_model.pt`. Do **not** use
`models/matched_windows/` or `models/production/` (deprecated baselines).

## 3. Launch the desktop app (PyQt6)

```bash
.venv/bin/python scripts/run_app.py
```

Interactive detection: synchronized spectrogram + probability views, sliding-window
inference, label management. Note: the app's **Detect** = CNN + hysteresis only (no
fp-filter/temperature/soft-notch) — those are batch-only. Reproduce the app's
behavior, not the batch pipeline, when debugging "what the app shows".

## 4. Where the data lives

WAV recordings are **not** in one directory — they span `5970/`, `USV_3452_sample/`,
`USV_9252/`, `USV_lab_131204*/`, etc. See **`docs/DATA_LOCATIONS.md`** for the
authoritative map (and for which large dirs are regenerable vs irreplaceable).

Large regenerable caches (`data/alpha3_*patches/`, `features/`, render galleries)
are gitignored — see `.gitignore`. They must be rsync-backed-up to the rig, not
committed.

## 5. The box-vs-rig split

- **"box"** = this WSL dev machine (analysis, app, batch inference on CPU/small GPU).
- **"rig"** = `cloudyclaude` GPU server (`shachar@100.113.224.57`, 3× RTX 3060 Ti),
  repo mirror at `/data/mickey_london_lab` (non-git rsync copy). Used for CNN/VAE
  training. The box pushes to the rig via SSH; the rig can't reach GitHub.
- Some artifacts (VAE checkpoints, contour-VAE patches) live **only on the rig** —
  `docs/DATA_LOCATIONS.md` marks these.

## 6. Which doc do I read?

| Doc | Audience | Purpose |
|-----|----------|---------|
| **this file** | successor (human) | install + run |
| `README.md` | everyone | project overview + core commands |
| `docs/DATA_LOCATIONS.md` | everyone | where every dataset/artifact lives |
| `CLAUDE.md`, `AGENTS.md` | AI agents | operating rules, agent roster |
| `DECISIONS.md` | everyone | ADRs (sample rate, STFT params) |
| `docs/handoffs/` | continuation | per-session work handoffs (dated) |
| `docs/plans/`, root `PLAN_*.md`/`ROADMAP_*.md` | researcher | design docs (current + historical) |
| `notes/`, `ops/`, `methodology/`, `reference/` | the knowledge-graph system | research memory — separate from the USV tools |

> The `notes/` + `ops/` + `methodology/` tree is an arscontexta knowledge-graph
> vault, not part of the USV toolchain. Ignore it unless you're maintaining the
> research memory.

## 7. Known handoff gaps (as of 2026-06-08 cleanup)

- `scripts/train_lab_classifier.py` (training CLI for the production lab classifier
  `results/lab_classifier_v1/best.pt`) survives only in
  `archive/cleaning_legacy/stack1/scripts/`. Restore from there to retrain.
- `requirements.txt` is unpinned — prefer `requirements_frozen.txt`.
- A pre-existing failing test, `tests/.../test_audit_corpus.py::test_3452_skip_is_graceful`,
  needs triage (is `audit_corpus` wrongly returning success on missing input, or is the
  expectation stale?). Not touched by the cleanup.

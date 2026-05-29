# Cleaning Legacy Archive

**Archived:** 2026-05-28
**Reason:** Stacks 1 and 3 (per `docs/modules/cleaning-subsystems.md`) were retired because their respective analysis families hit dead ends. Their source code, callers, and tests are preserved here (not deleted) so historical work can be inspected, but they are no longer imported by any live code path.

## What's here

| Subdir | What it was | Why archived |
|---|---|---|
| `stack1/` | The lab CNN classifier's **cleaning-validation gate** (Module 18.1: `cleaning_pipeline.py` + `diagnostics.py`) and its 12 caller scripts (Module 18.2b/18.3 training/eval prep) | DANN cage-invariance work (Module 18.4) shelved as a dead end 2026-05-27; VocalMat re-render verified dead (OSF audio mismatch); `lab_classifier_v1` shipped as production lab classifier without needing a re-derive path. The trained model weights live on the rig at `/data/shachar/contour_vae/.../lab_classifier_v1/best.pt` and are unaffected. |
| `stack3/` | The **SIS prefilter** (`features/spectrogram_filter.py::prefilter_spectrogram`) + **ridge tracker** (`features/ridge_tracker.py`) + 19 downstream shape-VAE experiment scripts (M8/M9/M10, Pathway A/B/B+A, registration alphabet driver) + 10 test files | All 6 attempts to build a learned shape representation on top of this stack failed (shape η² 0.009–0.105 vs registration ceiling 0.58–0.75). VAE family formally CLOSED for shape clustering on 2026-05-28 (`docs/handoffs/2026-05-28_pathway-B-kill-and-canonical.md`). The productionized registration alphabet `models/shape_kmeans/k20.joblib` was built by these scripts and lives on the rig; it does not need to be re-derivable. |

## What is NOT archived

| Kept on main | Why |
|---|---|
| Stack 2a (`src/usv_spectrogram/app/core/notch.py`) | Production PyQt6 + `run_batch_detection.py` |
| Stack 2b (`src/usv_spectrogram/app/core/denoise.py`) | Same |
| **Stack 4** = the **canonical "our cleaning pipeline"** (`scripts/deepsqueak_focus_stft.py` + `scripts/contour_mask_utils.py`) | Feeds the contour-masked VAE; user-designated canonical cleaning |
| The rest of `src/usv_spectrogram/classifier/` (model.py, training.py, augmentation.py, losses.py, dataset.py, resample.py) | Module 18.3 classifier *inference* — `lab_classifier_v1` weights are loaded via `build_resnet18_classifier`; this code stays live |

## How to use the archive

- **Inspect history**: `git log --follow archive/cleaning_legacy/stack3/src/features/spectrogram_filter.py` walks the file's evolution before and after the archive move.
- **Resurrect a file**: `git mv archive/cleaning_legacy/<path> <original_path>` brings it back; the `__init__.py` re-export still needs adding back in `src/usv_spectrogram/<package>/__init__.py`.
- **Run tests in archive**: the archived tests under `archive/cleaning_legacy/*/tests/` will not be auto-collected by pytest from project root (they're outside the conventional `tests/` tree); to run them, pass the file path explicitly.

## Why this is "archive", not "delete"

The user authorization (2026-05-28) was: *"1 and 3 are the ones that we don't really need anymore, maybe don't delete them but we can archive them since their family of analysis hit a dead end."* Archive (preserve in tree) was chosen over deletion (irreversible without git checkout) because:
- The shape-clustering negative result is itself a publishable finding; the code may need to be cited or re-inspected in writeups.
- Stack 1's Module 18.1 cleaning-validation gate documented 4 falsifiable diagnostics with a real methodology; future work on a different classifier family may want to reuse the gate.
- Storage cost is negligible (<5 MB).

If a future cleanup wants to harden this archive further (e.g. tarball + add to `.gitignore` to slim working-tree size), that's a separate decision.

## Cross-references

- Canonical cleaning pipeline → `docs/modules/cleaning-subsystems.md` (Stack 4 section).
- Why the shape VAE family closed → `docs/handoffs/2026-05-28_pathway-B-kill-and-canonical.md`, `docs/handoffs/2026-05-28_shape-vae-BA-hybrid-KILL.md`.
- Why Module 18.4 DANN shelved → memory note `project_lab_cnn_classifier_scope`, `docs/handoffs/2026-05-27_lab-classifier-transfer-solve.md`.
- Production registration alphabet → memory note `project_shape_registration_clustering`, `docs/handoffs/2026-05-25_productionize-shape-registration.md`.

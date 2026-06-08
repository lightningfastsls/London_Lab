# Archived tests (not collected)

These test modules errored at collection because the code they target was
**removed or archived**. They are kept here as spec/provenance and are excluded
from `pytest tests/` via `conftest.py` (`collect_ignore_glob`). None are run.

Moved during the 2026-06-08 handoff cleanup (tag `pre-cleanup-2026-06-08`).
Verified absent in the live tree before moving.

| Test | Target it expected | Status of target |
|------|--------------------|------------------|
| `test_amvoc_autoencoder.py` | `src/usv_spectrogram/features/amvoc_autoencoder.py` | `features/` package never existed (TDD spec, unbuilt) |
| `test_omer_vectorize.py` | `src/usv_spectrogram/features/omer_vectorize.py` | unbuilt — file says "Will pass after … is implemented" |
| `test_imsa_classifier.py` | `imsa_classifier` module | absent everywhere except this test |
| `test_cluster_sweep.py` | `src/usv_spectrogram/classification/cluster_sweep.py` | removed |
| `test_cnn_download_vocalmat_sample.py` | `scripts/cnn_download_vocalmat_sample.py` | removed |
| `test_cnn_prepare_training_data.py` | `scripts/cnn_prepare_training_data.py` | removed |
| `test_cleaning_real_data_loader.py` | `scripts/cnn_cleaning_validation.py` | removed |
| `test_analyze_latent_dispersion.py` | `scripts/analyze_latent_dispersion.py` | archived → `archive/cleaning_legacy/stack3/scripts/` |
| `test_analyze_latent_repertoire_jsd.py` | `scripts/analyze_latent_repertoire_jsd.py` | archived → `archive/cleaning_legacy/stack3/scripts/` |
| `test_analyze_latent_transitions.py` | `scripts/analyze_latent_transitions.py` | archived → `archive/cleaning_legacy/stack3/scripts/` |
| `test_render_confusion_matrix.py` | `scripts/train_lab_classifier.py` (`_render_confusion_matrix_png`) | absent from live tree — see note below |

> **Handoff note:** `scripts/train_lab_classifier.py` — the training CLI for the
> *production* lab classifier v1 (`results/lab_classifier_v1/best.pt`) — survives
> only in `archive/cleaning_legacy/stack1/scripts/` and a git worktree. The model
> ships but its training script is not in the live tree. Restore it from archive
> if a successor needs to retrain; then this test and `test_train_lab_classifier.py`
> become live again.

**To revive a test:** restore its target, move the test back up to `tests/`,
confirm it collects.

# PLAN — 2DPCA classifier for the 12 VocalMat USV syllable types

**Created:** 2026-06-02 (autonomous overnight session — user pre-authorized implementation via agent wave)
**Status:** implementation orchestrated via Workflow
**Comparator:** ResNet-18 lab classifier v1 (`results/lab_classifier_v1/`), VocalMat test macro-F1 **0.7669**.

---

## 0. Goal

Implement **2DPCA** (Two-Dimensional Principal Component Analysis, Yang et al. 2004) and its
two-directional extension **(2D)²PCA** (Zhang & Zhou 2005) as a *classical, interpretable*
baseline classifier for the 12 Grimsley USV syllable types, and benchmark it head-to-head
against the production ResNet-18 on the **same recording-grouped split** of the VocalMat data.

This is a research baseline, NOT a production-pipeline change. It touches **no** locked files
(`corpus.py`, `run_batch_detection.py`, `sliding_inference.py`, `training.py`).

## 1. Data (verified present locally 2026-06-02)

- `data/vocalmat_full/manifest.csv` — 12,178 rows, columns `path,class,source_recording,osf_path,size_bytes`.
- `data/vocalmat_full/<class>/*.png` — 227×227 RGB (grayscale content, 3 identical channels).
- 12 classes (`GRIMSLEY_12_CLASSES`, src/usv_spectrogram/classifier/dataset.py:46): Noise, Step up,
  Down-FM, Short, Chevron, Up-FM, Flat, Two steps, Step down, Complex, Reverse Chevron, Multi-steps.
  (manifest class strings are lowercase/underscored: `noise, step_up, down_fm, short, chevron, up_fm,
  flat, two_steps, step_down, complex, rev_chevron, mult_steps` — map to display names.)
- Class imbalance is severe (mult_steps 74, rev_chevron 136 … step_up 1806). Same imbalance v1 faced.

## 2. The algorithm (frozen spec — all implementers use exactly this)

Images are m×n matrices `A_k` (after grayscale + resize). `M` = number of training images.
Mean image `Ā = (1/M) Σ_k A_k`.

**2DPCA (right / column projection):**
- Image covariance (scatter) matrix: `Gt = (1/M) Σ_k (A_k − Ā)ᵀ (A_k − Ā)`  → shape **n×n**, symmetric PSD.
- Projection axes `X = [x_1 … x_d]` = top-`d` eigenvectors of `Gt` (descending eigenvalue), shape **n×d**.
- Feature matrix: `Y_k = A_k · X`  → shape **m×d**.

**Row variant (left projection):**
- `Gt' = (1/M) Σ_k (A_k − Ā)(A_k − Ā)ᵀ`  → **m×m**. Top-`q` eigenvectors `Z` (m×q).
- Feature: `B_k = Zᵀ · A_k`  → shape **q×n**.

**(2D)²PCA (two-directional):** apply both: `C_k = Zᵀ · A_k · X`  → shape **q×d** (compresses both axes).

**Component selection:** smallest `d` (and `q`) with cumulative eigenvalue energy
`Σ_{i≤d} λ_i / Σ_i λ_i ≥ energy` (default 0.95). Also allow explicit integer `d`.

**Classification — two modes:**
1. `nn` (faithful Yang 2004): 1-NN using **feature-matrix distance**
   `d(Y_i, Y_j) = Σ_{k=1}^{d} ‖ Y_i[:,k] − Y_j[:,k] ‖₂`  (sum of Euclidean norms of column-difference vectors).
2. `svm` / `lda`: flatten the feature matrix to a vector → standardize → `LinearSVC` (or `LDA`).
   Stronger, fast, and the fair "what can these features do with a real classifier" comparison.

## 3. Deliverables

| # | File | What |
|---|------|------|
| A | `src/usv_spectrogram/classifier/twodpca.py` | Pure-NumPy library: `fit_2dpca`, `fit_2d2dpca`, `project`, `project_bilateral`, `feature_matrix_distance`, `TwoDPCAClassifier`. |
| B | `tests/classifier/test_twodpca.py` | Real runnable pytest: math properties + synthetic end-to-end. Treated as spec. |
| C | `scripts/experiments/train_2dpca_classifier.py` | CLI: load manifest → grouped split → fit → eval; writes metrics.json + confusion_matrix.png + eval_report.md in the **same schema** as `results/lab_classifier_v1/metrics.json`. |
| D | `results/twodpca_vocalmat/` | Real run outputs for ≥3 configs; comparison table vs ResNet-18. |
| E | `docs/handoffs/2026-06-02_2dpca_results.html` | User-facing HTML report (per html-default convention) + `file://wsl.localhost/...` URL. |

### Frozen public API (B and C code against this exactly)

```python
# src/usv_spectrogram/classifier/twodpca.py

def fit_2dpca(images: np.ndarray, *, energy: float = 0.95, n_components: int | None = None
              ) -> "TwoDPCAModel": ...
    # images: (M, m, n) float. Returns model with .X (n×d), .eigenvalues, .mean_image (m×n),
    # .energy_ratio, .n_components.

def fit_2d2dpca(images, *, energy=0.95, n_components=None, n_components_row=None
                ) -> "TwoDTwoDPCAModel": ...
    # Returns model with .X (n×d), .Z (m×q), .eigenvalues_col, .eigenvalues_row, .mean_image.

def project(A: np.ndarray, model: "TwoDPCAModel") -> np.ndarray: ...      # A (m×n) -> Y (m×d)
def project_bilateral(A, model: "TwoDTwoDPCAModel") -> np.ndarray: ...    # A (m×n) -> C (q×d)

def feature_matrix_distance(Yi: np.ndarray, Yj: np.ndarray) -> float: ...
    # sum over columns k of ||Yi[:,k]-Yj[:,k]||_2

class TwoDPCAClassifier:
    def __init__(self, *, variant: str = "2dpca", classifier: str = "nn",
                 energy: float = 0.95, n_components: int | None = None,
                 n_components_row: int | None = None): ...
    def fit(self, images: np.ndarray, labels: np.ndarray) -> "TwoDPCAClassifier": ...
    def predict(self, images: np.ndarray) -> np.ndarray: ...
    # variant in {"2dpca","2d2dpca"}; classifier in {"nn","svm","lda"}.
```

## 4. Split protocol (apples-to-apples with v1)

Reuse `src/usv_spectrogram/classifier/dataset.py::build_stratified_split` — recording-level
grouping (no cage/recording leakage), 80/10/10. The manifest needs a `duration_ms` column; the
script adds a constant dummy (split logic only uses it for nothing critical — it groups by
`source_recording` and stratifies by `class`). Same split seed (1729) as v1 for comparability.

## 5. Default experiment matrix (script runs all, report compares)

Images: grayscale, resize **64×64** (default; `--resize` knob). 64×64 keeps the NN matrix-distance
tractable and Gt at 64×64. A full-227 run is a documented follow-up, not tonight's gate.

| Config | variant | classifier | energy |
|--------|---------|-----------|--------|
| 1 | 2dpca | nn | 0.95 |
| 2 | 2dpca | svm | 0.95 |
| 3 | 2d2dpca | svm | 0.95 |
| 4 | 2d2dpca | lda | 0.95 |

Report: per-config macro-F1 (val+test), per-class precision/recall, confusion matrix, feature
dimension, fit time. Headline = best 2DPCA config vs ResNet-18 0.7669.

## 6. Validation gates (definition of done)

1. `.venv/bin/python -m py_compile` clean on A, C.
2. `pytest tests/classifier/test_twodpca.py -v` all green. **Test expectations are spec — do not weaken to pass.**
3. Script runs end-to-end on real `data/vocalmat_full` producing the 4 configs' metrics.
4. `results/twodpca_vocalmat/metrics_*.json` schema matches v1's keys (`macro_f1_val`,
   `macro_f1_test`, `per_class_precision`, `per_class_recall`, `confusion_matrix`).
5. HTML report exists with the comparison table + the `file://` URL surfaced to the user.

## 7. Non-goals / boundaries

- Do NOT modify `corpus.py`, `training.py`, `run_batch_detection.py`, `sliding_inference.py`, or any test under `tests/` other than the new `test_twodpca.py`.
- No claim that 2DPCA replaces the CNN — this is a baseline/interpretability study.
- mypy not configured — no type-safety claims.

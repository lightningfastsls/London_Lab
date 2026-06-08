# HANDOFF — Adversarial verification of the 2DPCA implementation

**Created:** 2026-06-02 (overnight autonomous session)
**For:** an independent follow-up session (`/execute docs/handoffs/2026-06-02_2dpca_verification_handoff.md`)
**Predecessor work:** `PLAN_2dpca_usv_classifier.md`, results in `results/twodpca_vocalmat/`, report `docs/handoffs/2026-06-02_2dpca_results.html`.

---

## 0. Intent (the one job)

A 2DPCA / (2D)²PCA classifier was implemented and benchmarked against the production ResNet-18
on the 12-class VocalMat USV corpus. The classical method scored **far** below the CNN:

| Config | test macro-F1 |
|---|---|
| 2dpca / nn | 0.191 |
| 2dpca / svm | 0.208 |
| 2d2dpca / svm | 0.209 |
| **2d2dpca / lda (best)** | **0.240** |
| ResNet-18 v1 | **0.767** |

This was reported as a *clean negative result* (linear method can't capture nonlinear syllable shape).
**Your job is to falsify the competing hypothesis: that the low score is an IMPLEMENTATION ARTIFACT
(a bug in `twodpca.py` or the driver), not a true property of the method.** We do not believe it is
a bug — but we want it ruled out with evidence, not assertion.

You are NOT trying to improve the score. You are trying to *break* the conclusion, and report whether
it survives. A correct implementation that still scores 0.24 is the success outcome here.

---

## 1. Artifacts under test (READ these first)

| File | Role | Size |
|---|---|---|
| `src/usv_spectrogram/classifier/twodpca.py` | Library under test | 485 lines — read fully |
| `tests/classifier/test_twodpca.py` | Existing 46-test suite (treat as spec — do NOT weaken) | 667 lines — read in 2 chunks |
| `scripts/experiments/train_2dpca_classifier.py` | Driver (load → split → fit → eval) | 442 lines — read fully |
| `results/twodpca_vocalmat/metrics_*.json` | The numbers to explain | — |
| `src/usv_spectrogram/classifier/dataset.py` | `build_stratified_split` (recording-grouped) | already reviewed |

### Frozen API (the contract the math must satisfy)
```
fit_2dpca(images (M,m,n), *, energy=0.95, n_components=None) -> TwoDPCAModel(.X (n×d), .eigenvalues, .mean_image, .energy_ratio, .n_components)
fit_2d2dpca(...) -> TwoDTwoDPCAModel(.X (n×d), .Z (m×q), .eigenvalues_col, .eigenvalues_row, .mean_image, ...)
project(A, model) -> Y (m×d);  project_bilateral(A, model) -> C (q×d)
feature_matrix_distance(Yi, Yj) -> sum over columns k of ||Yi[:,k]-Yj[:,k]||_2
TwoDPCAClassifier(variant∈{2dpca,2d2dpca}, classifier∈{nn,svm,lda}, energy, n_components, n_components_row).fit(images,labels).predict(images)
```
**Algorithm (the truth the code is checked against):**
`Gt = (1/M) Σ (A_k−Ā)ᵀ(A_k−Ā)` (n×n); `X` = top-d eigvecs of `Gt`; `Y=A·X`. Row variant `Gt'=(1/M)Σ(A−Ā)(A−Ā)ᵀ` (m×m). Bilateral `C=ZᵀAX`. Component count = smallest d with cumulative eigenvalue-energy ≥ `energy`.

---

## 2. The decisive test — reproduce a PUBLISHED benchmark (do this FIRST)

If the implementation is correct, it must reproduce Yang et al. (2004)'s headline face-recognition
result. This is the single highest-value check: a published number with a known answer, no USV data involved.

**Task:** Write `scripts/experiments/verify_2dpca_olivetti.py`:
- Load ORL/Olivetti faces via `sklearn.datasets.fetch_olivetti_faces()` (40 people × 10 images, 64×64, already grayscale [0,1]). No external download needed beyond sklearn's cache.
- Split: first 5 images/person train, last 5 test (the standard Yang-2004 protocol). 200 train / 200 test.
- Run `TwoDPCAClassifier(variant="2dpca", classifier="nn", n_components=d)` for d ∈ {2,4,6,8,10,15,20}.
- **Expected (Yang 2004, Table on ORL):** rank-1 accuracy climbs to ~**0.92–0.96** by d≈5–10 and plateaus. Anything ≥ 0.90 confirms the core 2DPCA math + NN distance are correct.
- Also run `variant="2d2dpca"` and the `svm`/`lda` heads — all should be high (≥0.90) on faces.

**Decision:**
- Olivetti ≥ 0.90 → **the 2DPCA math is correct.** The low USV score is NOT a core-algorithm bug. Proceed to §3–§5 to rule out *pipeline* bugs.
- Olivetti < 0.85 → **likely a real bug.** Stop and debug the library (eigenvector selection, distance, projection orientation) before trusting any USV conclusion. See §6 micro-checks.

---

## 3. Parity control — does ordinary PCA agree?

If flattened sklearn PCA + the same downstream classifier *also* scores ~0.24 on the USV data, the
ceiling is the data/linear-model, not anything specific to the 2DPCA code.

**Task:** small script (or notebook cell): load the SAME split the driver used
(`results/twodpca_vocalmat/split/{train,val,test}.csv`), 64×64 grayscale, flatten to 4096-vectors,
`sklearn.decomposition.PCA(n_components=…)` → `LinearDiscriminantAnalysis` / `LinearSVC`. Sweep components.
- **Expected:** test macro-F1 in the **same ballpark (~0.20–0.30)** as 2DPCA. PCA and 2DPCA are different
  factorizations of the same linear-subspace idea; on this data they should agree to within a few points.
- If PCA scores *much* higher (say > 0.45) → 2DPCA is leaving signal on the table → investigate the 2DPCA feature construction. If PCA ≈ 2DPCA → conclusion holds firmly.

---

## 4. Label-shuffle / leakage controls

1. **Chance floor:** re-run the best config with **training labels shuffled** (`rng.permutation`). Test
   macro-F1 must collapse to ≈ 1/12 ≈ 0.083. If a shuffled-label model scores well above chance, there is
   leakage or a label-alignment bug — find it.
2. **Real > shuffled:** the unshuffled 0.24 must be clearly above the shuffled control. (It is signal, just weak.)
3. **Split integrity:** confirm `source_recording` sets are disjoint across train/val/test (no recording leakage), and that label→image alignment in the driver's loader is correct (spot-check 10 rows: open the PNG, confirm the class folder in `path` matches the `class` column and the loaded array's label).

---

## 5. Ablations that distinguish "bug" from "ceiling"

Run via the existing driver flags (`--resize`, `--energy`, and add `--n-components` if not present — see note).
A *bug* tends to produce flat/non-monotonic nonsense; a *real ceiling* produces sensible, saturating curves.

1. **Component sweep:** energy ∈ {0.80, 0.90, 0.95, 0.99} and/or explicit d ∈ {5,9,20,40}. Expected: macro-F1
   rises then plateaus/saturates below the CNN. (At energy 0.95 the selector chose only d=9 — check whether
   forcing d=20–40 helps; if it does, the 0.95 default was the limiter, not the method. Report either way.)
2. **Resize sweep:** `--resize` ∈ {32, 64, 96, 128} and a full **227** (no downscale) run on the best config.
   Expected: modest gains with resolution, still far below 0.767. A *large* jump at 227 would mean the 64×64
   default — not 2DPCA — depressed the headline; report it prominently if so.
3. **Per-class vs imbalance:** the LDA head had no class weighting and Multi-steps (n≈7 test) scored 0/0.
   Re-run with `class_weight='balanced'` on the SVM/LDA head. Does macro-F1 move materially? (Tests whether the
   number is suppressed by imbalance handling vs genuine inseparability.)

> Note: if the driver lacks an explicit `--n-components` flag, add one (additive, backward-compatible). This is the ONE place you may extend the driver. Do not change its default behavior.

---

## 6. Micro-checks on the math (only if §2 fails, or for completeness)

- **NN distance vs brute force:** on 50 random images, assert the vectorized
  `TwoDPCAClassifier(classifier="nn")` predictions equal a naive double-loop using
  `feature_matrix_distance`. (Confirms the `sqrt(((Tr-te)**2).sum(1)).sum(1)` vectorization == sum-of-column-L2-norms.)
- **Projection orientation:** verify `Y = A·X` uses `X` (n×d) on the right (time/freq axis correct), not `Xᵀ`.
  A transposed projection silently still "works" dimensionally if m==n (64==64!) — this is a real trap with square
  images. Test with a **non-square** synthetic case (e.g. 64×48) to catch an m/n swap that 64×64 would hide.
- **Eigen-order & energy:** eigenvalues strictly descending; `energy_ratio` ∈ (0,1]; `np.linalg.eigh` (symmetric), not `eig`.
- **Mean convention:** `Gt` centers by `Ā`, but `project` uses raw `A·X` (per the existing tests). Confirm train and
  test are projected with the *same* convention (no train-only centering that test skips).

---

## 7. Binding constraints (flattened — obey)

- **Files NOT to touch:** `src/usv_spectrogram/corpus.py`, `src/usv_spectrogram/classifier/training.py`,
  `scripts/run_batch_detection.py`, `app/core/sliding_inference.py`. None are needed for this task.
- **`tests/classifier/test_twodpca.py` is spec.** You MAY add adversarial tests (§6) — do NOT weaken or delete
  existing assertions to make anything pass (CLAUDE.md Test Protocol).
- **`twodpca.py` is the artifact under test.** If you find a real bug, fixing it is in scope — but log every change
  with file:line and rationale, and re-run the full 46-test suite + the Olivetti benchmark after each fix.
- **Corpus invariants:** the VocalMat patches are 227×227, rendered at the locked STFT (Hamming-256/hop-128/nfft-1024
  @ 250 kHz, global-MAD). Do not re-render. Downscaling at load time is a study knob, not a pipeline change.
- **Recording-grouped split (seed 1729)** must be preserved for any USV comparison — splitting a recording across
  train/test leaks cage-acoustic features (`docs/handoffs/2026-05-18_vae_comparison_memo.md`).
- **Print parameters** at the top of every run (resize, energy, d, per-class split counts) — repo convention.

## 8. Data & commands

```bash
# Data (already local): data/vocalmat_full/ (12,178 PNGs) + manifest.csv. Comparator: results/lab_classifier_v1/metrics.json (0.7669).
cd /home/shachar/projects/mickey_london_lab

# existing suite (must stay green)
.venv/bin/python -m pytest tests/classifier/test_twodpca.py -q

# §2 decisive benchmark (write this script)
.venv/bin/python scripts/experiments/verify_2dpca_olivetti.py

# fast USV iteration (smoke), then full
.venv/bin/python scripts/experiments/train_2dpca_classifier.py --limit-per-class 80 --resize 64 --out .claude/jobs/56144c19/tmp/verify_out
.venv/bin/python scripts/experiments/train_2dpca_classifier.py --resize 64 --out results/twodpca_verify/   # full ~12k; the nn config is the slow one
```
Use `.venv/bin/python` for everything. The full `nn` config over 12k images is the slow path — favor `svm`/`lda`
and `--limit-per-class` while iterating; reserve the full `nn` run for the final confirmation.

## 9. Definition of done

A short report (HTML preferred per repo convention, with a `file://wsl.localhost/...` URL) answering:

1. **Olivetti benchmark:** what rank-1 accuracy did our 2DPCA reach? (≥0.90 = math correct.)
2. **PCA parity:** does flattened PCA+LDA score ~0.24 on the same USV split? (≈ = ceiling confirmed.)
3. **Shuffle control:** does shuffled-label collapse to ~0.083? (yes = no leakage.)
4. **Ablation curves:** do component/resize sweeps rise-and-saturate sensibly below 0.767? (yes = ceiling, not bug.)
5. **Bugs found:** list any, with fix + re-run evidence. (We expect none material.)
6. **VERDICT:** one of —
   - `CONFIRMED — implementation correct, 0.24 is the genuine linear-method ceiling` (expected), or
   - `BUG FOUND — corrected score is X, conclusion revised`.

### Suggested orchestration
This parallelizes cleanly — consider a Workflow: §2 (Olivetti) ∥ §3 (PCA parity) ∥ §6 (math micro-checks) run
concurrently (independent), barrier, then §4–§5 (controls + ablations) on the surviving hypothesis, then a
report+review phase. Keep the existing 46 tests as the regression gate throughout.
```
```

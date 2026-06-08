# Handoff — α₃-C: A8 + binding A6 (post-labeling, 2026-05-30)

**Continuation of:** `docs/handoffs/2026-05-29_alpha3-C-A6-A8-gamma.md`. That
handoff's **label-independent prep is now DONE** (this session). What remains is
gated on (1) the user's γ hand-labels and (2) one rig compute OK.

**One-sentence job:** once `data/manual_shape_labels.csv` exists, run A8, run the
γ-anchored A6, compute the real A6 baselines on the rig → ship/kill per the gates.

---

## What's already done (do NOT redo)

| Item | State |
|---|---|
| A6 ID-bridge (latents → oracle/γ call_ids) | DONE — `scripts/experiments/build_a6_latent_bridge.py`; **100% match, 98.5% oracle coverage, 0 invented call_ids** |
| Bridge table on disk | `data/alpha3_a6/lab_131204_latent_bridge.parquet` (55,863 rows, z_* + matched_call_id + overlap_frac) |
| A6 eval driver | DONE — `scripts/experiments/eval_a6_existing_latents.py` (reuses `train_shape_vae_alpha3.run_metrics`; `--label-kind oracle|gamma`; `--baseline-features`) |
| Preliminary oracle A6 read | NMI **0.184** (borderline), chevron-kNN 0.526 (Reverse-Chevron-dominated), margin over placeholder +0.177 — **NON-BINDING** (placeholder baseline) |
| Rig latents + manifest pulled to box | `data/alpha3_a6/{contour_vae_latents_combined,lab_131204_patches_manifest}.parquet` |
| A8 script verified | `vocalmat_vs_gamma_crosstab.py` — clean `call_id` join, no bridge needed |

**Bridge design (so you don't re-derive it):** latents ⋈ manifest on
`(wav_stem, call_id, window_idx)` — **NOT** `patch_idx` (global vs local index →
false join, wav_stem agreement 0.0). Then latents-with-time ⋈
`classified_detections_lab_131204_clean.csv` on `wav_stem` + max time-overlap of
`abs_time_start/end_s` vs `det_start_s/det_end_s` → `__det{det_index}`.

---

## Input expected (from the user)

`data/manual_shape_labels.csv` — from:
```bash
PYTHONPATH=src .venv/bin/python scripts/labeling/hand_label_200.py \
    --manifest data/alpha3_gamma_manifest.csv --per-cohort 12
```
Columns: `call_id, cohort, shape_label, labeled_at_index`. 204 rows (17 cohorts ×
12).

**2026-05-30: γ now uses the FULL 12-class VocalMat taxonomy** (user wants a direct
human-vs-machine system comparison, not the old coarse 8-family scheme). shape_label
∈ GRIMSLEY_12_CLASSES = {Noise, Step up, Down-FM, Short, Chevron, Up-FM, Flat, Two
steps, Step down, Complex, Reverse Chevron, Multi-steps} + `unclear` (human escape
hatch, excluded via `--exclude-unclear`). Keys: C/R/U/D = Chevron/Reverse-Chevron/
Up-FM/Down-FM, 1–4 = the step family, F/X/S/N = Flat/Complex/Short/Noise. The label
strings are byte-identical to the oracle's `top1_class`, so A8 is a square 12×12
confusion matrix (diagonal = agreement).

---

## γ label QA — multi-USV contamination fixed (2026-05-30)

The first labeling pass had a crop problem: **50/204 (25%)** human-view patches
contained >1 detection in the crop window (neighbours clipping in), and those
systematically inflated **Complex (13) + Multi-steps (13)** — the eye reads 2–3
crammed USVs as "complex/multi-step". Audit: `gamma_crop_audit.csv` (per-patch
`n_dets_in_crop`). Fix applied:
- The 154 single-USV labels are kept as-is.
- The 50 contaminated patches were re-rendered with **target isolation**
  (`render_human_view_patches.py --isolate --isolate-dim 0.04`: neighbours outside
  the target's time span are masked to near-black, so only the target USV is visible)
  → `data/alpha3_gamma_relabel/`, manifest `data/alpha3_gamma_relabel_manifest.csv`.
  Tiering (neighbour in-frame extent): 33 clearly-multi (>40 ms) / 16 partial / 1
  edge-sliver. User-confirmed true positive: `131212_1000_m1fm2_chunk_016__det8`
  was mislabeled Reverse Chevron but is 3 separate USVs.
- `data/manual_shape_labels.csv` was trimmed to the 154 clean rows (full 204 backed
  up at `data/manual_shape_labels.backup_204_pre_relabel.csv`); the user re-labels
  the 50 isolated patches (`--manifest data/alpha3_gamma_relabel_manifest.csv --n 50
  --per-cohort 50`), appending back to 204.

When A8/A6 run, the γ set should again be 204 rows. If only 154 are present, the
re-label pass wasn't finished — check before trusting Complex/Multi-steps counts.

## Step 1 — A8 oracle ↔ γ cross-tab (box, ~5 min)

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/vocalmat_vs_gamma_crosstab.py \
    --gamma data/manual_shape_labels.csv \
    --oracle-labels data/labels_vocalmat_v1_on_131204.csv \
    --exclude-unclear
```
Surface the `file://wsl.localhost/...` URL.

**Reinterpretation under the 12-class switch:** γ and oracle now share the SAME
12-class axis, so the contingency table is a square **human-vs-machine confusion
matrix** — the diagonal is direct agreement, off-diagonal shows *which* classes the
v1 oracle confuses on lab data (it has never had human-verified lab labels before).
Cramér's V / NMI still summarize association, but the user's goal ("compare the
systems") is better served by **Cohen's κ + per-class precision/recall of the oracle
vs the γ human truth**. The crosstab script currently emits V + NMI + the table;
**check whether it reports κ — if not, add it** (it's a few lines on the same
contingency table; do NOT change the existing V/NMI outputs). Read the result → it
still sets the A6 anchor (strong agreement → oracle is a trustworthy anchor).

## Step 2 — γ-anchored A6 (box, fast — existing latents)

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/eval_a6_existing_latents.py \
    --bridge data/alpha3_a6/lab_131204_latent_bridge.parquet \
    --labels data/manual_shape_labels.csv --label-kind gamma \
    --out data/alpha3_a6/a6_gamma_existing.json
```
This is the cheap read. It is **non-binding until Step 3** supplies real baselines.

## Step 3 — real A6 baselines (RIG, gated — get a per-session OK)

The eval-validity rule (`learned must beat random-init-encoder AND
column-mean-identity by ≥0.10 NMI`) needs both baselines computed from the **same
substrate the production VAE saw** = the masked `patches.npz` on the rig
(`/data/shachar/contour_vae/results/masked_patches/lab_131204_focus/`, 13 GB).

**Do NOT guess the preprocessing.** Read the production VAE's training/encode
script on the rig first to replicate its exact patch→tensor pipeline (resize,
normalization), then for each labeled call_id's max-overlap patch emit:
- `identity_*` = per-column mean of the patch (the `column_mean_features` recipe).
- `z_random_*` = `ShapeVAE(img_size, z_dim)` untrained forward `mu` (fixed seed).

Write a small `call_id, identity_*, z_random_*` parquet, rsync to box, then:
```bash
... eval_a6_existing_latents.py ... --baseline-features data/alpha3_a6/a6_baselines.parquet
```
Per-class floor (handoff caveat): for the chevron-kNN gate keep **top-N chevrons
by prob** regardless of the 0.85 high-conf bar — only 1 true Chevron clears 0.85.

---

## ✅ RESULT (2026-06-02) — A8 + binding A6 COMPLETE → **KILL**

All three steps ran. γ labels finalized at 204 rows (186 clean single-shape +
18 `unclear`); the 18 unclear are **hysteresis-merged multi-USV events** (median
det-span 172.8 ms vs 85.3 ms corpus median; 17/18 from the contaminated-50 set) —
excluding them is correct, not a labeling failure.

**A8 (human γ vs machine oracle, 12-class, n=186 excl-unclear):** Cramér's V
**0.508** (just clears the 0.5 "geometric" gate), NMI 0.425, **Cohen's κ 0.392**
(fair), exact-agreement **45.2%**. Strong agreement on Noise (P0.96/R0.82),
Reverse-Chevron, Flat, Up/Down-FM; the **step/multi-step family diverges** (oracle
under-segments: γ Multi-steps R0.08 → machine calls them "Step up"; γ Chevron 0/4
→ "Complex"). Read: oracle is *moderately* credible, NOT a gold standard → run A6
on BOTH anchors, weight γ. Report: `$CLAUDE_JOB_DIR/alpha3_oracle_vs_gamma.html`.
A8 script now emits κ + per-class P/R (additive; V/NMI unchanged).

**A6 binding** (real baselines from the rig — random-init `ImageVAE` +
column-mean identity on the SAME `combined_all_cohorts/patches.npz` substrate the
production VAE saw; `rig_extract_a6_baselines.py`, 40,191 call_ids):

| anchor | NMI learned | NMI random | NMI **identity** | margin vs best | chevron-kNN | verdict |
|---|---|---|---|---|---|---|
| γ (n=185, honest) | 0.287 | 0.173 | **0.245** | **+0.042** | 0.124 | **KILL** |
| oracle (n=5788) | 0.184 | 0.033 | **0.165** | **+0.019** | 0.526* | **KILL** |

Both fail the ≥0.10 baseline margin. **Kill shot = the column-mean identity
baseline** (no learning, per-time-column pixel average) lands within 0.04 / 0.02
NMI of the trained 32-dim VAE → the latent bottleneck adds ~nothing over pixel
statistics. *Oracle chevron-kNN 0.526 is the A8-predicted circularity (κ=0.39);
even there identity beats the VAE on complex-kNN 0.574 vs 0.542.*

**Decision: VAE family CLOSED for shape clustering (7/7 falsification now).**
Production shape rep stays **registration** (`models/shape_kmeans/k20.joblib`);
the **186 γ labels are the human eval gold**. The "escalate to a fresh β-VAE"
branch is NOT taken — the honest anchor isn't borderline (chevron below chance,
identity ≈ learned), so a new train would re-litigate a settled question.
Artifacts: `data/alpha3_a6/{a6_gamma_binding,a6_oracle_binding}.json`,
`data/alpha3_a6/a6_baselines.parquet`.

## Decision gates (from the roadmap — do not re-derive)

| Outcome | Action |
|---|---|
| A8 Cramér's V > 0.5 | taxonomy IS geometric — α₃ is a credible anchor; oracle-A6 is meaningful |
| A8 V < 0.3 | taxonomy not shape — **γ is the only honest anchor**; oracle-A6 is contaminated |
| A6 SHIP: NMI≥0.25 ∧ chevron kNN≥0.50 ∧ beats both baselines by ≥0.10 | ship the shape rep |
| A6 KILL: NMI<0.15 OR fails baselines | VAE family CLOSED permanently; ship registration (`models/shape_kmeans/k20.joblib`); γ becomes the human eval gold |
| A6 BORDERLINE (existing latents 0.15–0.25) | escalate to a fresh α₃ β-VAE (`train_shape_vae_alpha3.py`, the genuine ~25% bet) — needs gated rig train |

**Modal prior:** P(2-D substrate adds nothing) ≥ 0.5. The preliminary 0.184 is
consistent with borderline/kill — a clean kill is an acceptable, informative
outcome against the 6/6 falsification record.

---

## 🔒 Files NOT to touch
`models/shape_kmeans/k20.joblib`, `results/lab_classifier_v1/best.pt`,
`scripts/deepsqueak_focus_stft.py`, `scripts/contour_mask_utils.py`,
`scripts/mass_apply_contour_mask.py` + canonical patch-gen scripts,
`src/usv_spectrogram/corpus.py` + `ExtractionConfig`, the production detection
pipeline, `archive/cleaning_legacy/` (read/import only).

## Reference artifacts (do not edit — validation gold)
`data/labels_vocalmat_v1_on_131204.csv`, `data/alpha3_a6/lab_131204_latent_bridge.parquet`,
`data/manual_shape_labels.csv` (once written), `data/lab_cnn_training/held_out_844/patches/`.

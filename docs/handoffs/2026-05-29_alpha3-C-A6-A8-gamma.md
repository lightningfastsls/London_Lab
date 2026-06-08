# Handoff — α₃-C: A6 + A8 + γ (2026-05-29, evening)

**Continuation of:** `docs/handoffs/2026-05-29_alpha3-C-execution-A3-A8.md` (→ roadmap
`docs/plans/ROADMAP_ALPHA3_VOCALMAT_LABELED.md`). That handoff's A3→A5 are **done**;
this picks up at γ labeling + A6 + A8.

**One-sentence job:** the user hand-labels 204 γ patches → run A8 (oracle↔γ cross-tab)
and A6 (shape-rep eval against the anchor) → ship/kill per the roadmap gates.

---

## What's already done (do NOT redo)

| Item | State |
|---|---|
| Substrate-mismatch diagnosis | DONE — masked patches are OOD for the v1 oracle (75% Noise collapse) |
| Two-render architecture | DONE — see below |
| A3b v1-faithful oracle render | `scripts/experiments/render_v1_faithful_patches.py` — **byte-identical to held_out_844 (844/844)** |
| A4 oracle labels (40,787) | `data/labels_vocalmat_v1_on_131204.csv` — healthy 12-class dist, 5,875 high-conf |
| A5 audit (tight-crop) | `$CLAUDE_JOB_DIR/alpha3_label_audit.html` — regenerated, judgeable |
| Tight-crop human view | `scripts/experiments/render_human_view_patches.py` → `data/alpha3_human_patches/` (40,787) |
| γ tool + sample | `scripts/labeling/hand_label_200.py` + `data/alpha3_gamma_manifest.csv` (204 patches, tight-crop) |
| A6/A8 scripts | `train_shape_vae_alpha3.py`, `vocalmat_vs_gamma_crosstab.py` — built + smoke-verified |

**Two renders, one per role (eval-validity rule #1):**
- **VAE substrate** = canonical contour-masked `patches.npz` on the rig
  (`/data/shachar/contour_vae/results/masked_patches/lab_131204_focus/`). DO NOT
  reinvent — see `feedback_cleaning_pipeline_impl_on_rig` + `docs/DATA_LOCATIONS.md`.
- **Oracle-label render** = v1-faithful UNMASKED patches (`render_v1_faithful_patches.py`).
- `scripts/experiments/render_vocalmat_style_patches.py` + `data/alpha3_patches/` are
  **DEPRECATED** (my wrong-substrate reconstruction) — do not use.

---

## Input expected (from the user)

`data/manual_shape_labels.csv` — produced by:
```bash
PYTHONPATH=src .venv/bin/python scripts/labeling/hand_label_200.py \
    --manifest data/alpha3_gamma_manifest.csv --per-cohort 12
```
Columns: `call_id, cohort, shape_label, labeled_at_index`. shape_label ∈
{chevron, jump, flat, complex, up_fm, down_fm, multi_component, unclear} (underscore forms).

---

## A8 — oracle ↔ γ cross-tab (box, ~5 min)

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/vocalmat_vs_gamma_crosstab.py \
    --gamma data/manual_shape_labels.csv \
    --oracle-labels data/labels_vocalmat_v1_on_131204.csv \
    --exclude-unclear
```
Join on `call_id` (γ patches are lab → all have oracle labels). Outputs Cramér's V + NMI
+ HTML. **Surface the `file://wsl.localhost/...` URL.**

## A6 — shape-rep eval (DECISION NEEDED first)

**Open decision:** evaluate the **existing production VAE latents**
(`/data/shachar/contour_vae/results/contour_vae_combined/latents.parquet`, rig — fast,
but the known pitch/duration sorter, `project_shape_registration_clustering`) **vs**
train a **fresh α₃ β-VAE** (`train_shape_vae_alpha3.py`, the roadmap's genuine ~25% bet).

Either path needs:
- **ID bridge:** latents are keyed by **DeepSqueak call_id**; oracle labels + γ are keyed by
  `wav_stem__det{det_index}`. No direct join — bridge by `(wav_stem, time-overlap)` of
  `abs_time_start/end` (latents/patches_manifest) vs `det_start_s/det_end_s` (CSV). Many-to-one.
- **Mandatory baselines:** random-init encoder + per-column-average identity (beat both by ≥0.10).
- **Per-class floor:** only 1 high-conf chevron at 0.85 → for the chevron-kNN gate keep top-N
  chevrons by prob regardless of the 0.85 bar (handoff caveat).

---

## Decision gates (from the roadmap — do not re-derive)

| Outcome | Action |
|---|---|
| A8 Cramér's V > 0.5 | taxonomy IS geometric — α₃ is a credible anchor |
| A8 V < 0.3 | taxonomy not shape — γ is the only honest anchor; VAE-vs-oracle eval is contaminated |
| A6 SHIP: NMI≥0.25 ∧ chevron kNN≥0.50 ∧ beats both baselines by ≥0.10 | ship the shape rep |
| A6 KILL: NMI<0.15 OR fails baselines | VAE family CLOSED permanently; ship registration (`models/shape_kmeans/k20.joblib`); γ becomes the human eval gold |

**Modal prior:** P(2-D substrate adds nothing) ≥ 0.5 — a clean kill is an acceptable, informative outcome.

---

## 🔒 Files NOT to touch
`models/shape_kmeans/k20.joblib`, `results/lab_classifier_v1/best.pt`,
`scripts/deepsqueak_focus_stft.py`, `scripts/contour_mask_utils.py`,
`scripts/mass_apply_contour_mask.py` + the canonical patch-gen scripts,
`src/usv_spectrogram/corpus.py` + `ExtractionConfig`, the production detection pipeline,
`archive/cleaning_legacy/` (read/import only). The v1-faithful render imports archived
`_spectrogram_db`/`_spec_to_uint8_patch` — do not "fix" those imports.

## Reference artifacts (do not edit — they are validation gold)
`data/lab_cnn_training/held_out_844/patches/` (the byte-identical reference),
`data/labels_vocalmat_v1_on_131204.csv`, `data/manual_shape_labels.csv` (once written).

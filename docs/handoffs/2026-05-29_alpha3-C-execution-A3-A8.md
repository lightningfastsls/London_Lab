# Handoff — α₃-C execution, Phases A3 → A8 (2026-05-29)

**Continuation of:** `docs/handoffs/2026-05-28_alpha3-vocalmat-blocker-and-pivot.md`
(the blocker + variant menu). That decision is now **made**: the user chose
**α₃-C + γ in parallel** on 2026-05-29.

**Binding spec:** `docs/plans/ROADMAP_ALPHA3_VOCALMAT_LABELED.md` (canonical
Markdown of the prior `alpha3_roadmap.html`, adapted for α₃-C). Read it first —
this handoff is the execution pointer; the roadmap holds the full gates/kills.

**Next chat's job (one sentence):** write the A3 render script, get a per-session
rig-launch OK, render → label → spot-check → train the shape VAE (A3→A6) while the
γ tool (A7) runs in parallel, then cross-tab oracle ↔ γ (A8).

---

## What's already done (do NOT redo)

| Item | State |
|---|---|
| Variant decision | **α₃-C + γ** (user, 2026-05-29) |
| Stacks 1+3 archive | **committed** `67deb0c5` (45 renames, 0 deletions) |
| Stack-4 canonicalization | committed (banners + trimmed `classifier/__init__.py`) |
| `data/vocalmat/` gitignored | yes (212 MB `.mat` excluded) |
| A4 oracle script | `scripts/experiments/label_patches_v1.py` — **written + verified** |
| A4 oracle smoke | **balanced acc 0.812** reproduced on `held_out_844` (matches v1) |

**Uncommitted (this session's new work — user to review/commit):**
`docs/plans/ROADMAP_ALPHA3_VOCALMAT_LABELED.md`, `scripts/experiments/label_patches_v1.py`,
`results/alpha3/labels_v1_held_out_844.csv`, this handoff. Plus an orphan
`--ckpt-every` edit on `archive/cleaning_legacy/stack3/scripts/experiments/train_shape_encoder_contrastive.py`
(decide: keep / revert / discard — it's a frozen baseline).

---

## A4 oracle — ready to reuse as-is

`label_patches_v1.py` is generic: point it at any manifest with a `path` column.

```bash
PYTHONPATH=src .venv/bin/python scripts/experiments/label_patches_v1.py \
    --manifest data/alpha3_patches/manifest.csv \
    --id-column call_id \
    --output data/labels_vocalmat_v1_on_131204.csv \
    --device cuda          # or cpu on the box
```

**Calibration caveat discovered at smoke time:** on `held_out_844`, only **9.7%**
of patches cleared the 0.85 high-confidence bar and **Chevron got 2/844**
predictions. That is the lab→our-data domain gap (under-confidence + chevron-shy),
not a bug. The roadmap's `high_confidence ≥ 0.85` eval-gold filter may be too
strict on our 40k render — **decide the threshold from the real A4 distribution**,
not a priori. A per-class floor (keep top-N per class regardless of prob) is a
sensible fallback so rare geometric classes aren't starved.

---

## A3 — the next thing to build

`scripts/experiments/render_vocalmat_style_patches.py` (not yet written). Per the
roadmap, for each of ~40,787 calls in `classified_detections_lab_131204_clean.csv`:

1. Load the WAV window centered on the call.
   - WAV root for lab 131204: `USV_lab_131204_chunked_2s_full/` (per the archived
     `patch_duration_sweep.py` — confirm it's present on the rig, not just the box).
2. **Q2 = resample 300→250 kHz** via `usv_spectrogram.classifier.resample.resample_to_vocalmat`
   (5/6 polyphase). Keeping 300 kHz is the documented fallback if A5 label quality
   looks off.
3. Apply **Stack 4**: `scripts/deepsqueak_focus_stft.py` + `scripts/contour_mask_utils.py`
   (per-call adaptive STFT + contour + bandwidth mask). 🔒 Do not edit these — Stack 4
   is canonical.
4. Render VocalMat-style PNG: `10·log10(P)` → freq-crop > 45 kHz → `mat2gray` min-max
   → `flipud` → 227×227 bicubic → 3-channel.
5. Write `data/alpha3_patches/<call_id>.png` + `data/alpha3_patches/manifest.csv`
   (`call_id, path` minimum).

**⚠ Footgun (CLAUDE.md + `feedback_cnn_inference_global_mad`):** normalize the
*whole* spectrogram once, then crop — never per-window MAD. Match the v1 training
render convention exactly (the archived `cnn_prepare_training_data._spec_to_uint8_patch`
+ `_VOCALMAT_STFT_HOP` is the reference; it lives in
`archive/cleaning_legacy/stack1/scripts/cnn_prepare_training_data.py`, read-only).

---

## Decision gate (from the roadmap — do not re-derive)

| Outcome | Action |
|---|---|
| A5 spot-check: >50% wrong/class | retry A3 with 300 kHz; still bad → γ-only |
| A6 SHIP: NMI≥0.25 ∧ chevron kNN≥0.50 ∧ beats baselines by ≥0.10 | ship α₃ VAE as shape rep |
| A6 SHIP + A8 Cramér's V > 0.5 | ship VAE + γ as concordant anchors |
| A6 SHIP + A8 V < 0.3 | ship VAE only as "matches taxonomy", NOT "matches user shape sense" |
| A6 KILL: NMI < 0.15 OR fails baselines | **VAE family CLOSED permanently**; ship registration (`models/shape_kmeans/k20.joblib`); γ becomes the human eval gold |

**Modal prior:** P(shape lives in 1-D, 2-D adds nothing) ≥ 0.5. A clean kill is an
acceptable, informative outcome against a 6/6 falsification record.

---

## Eval-validity rules (load-bearing — do not regress)

1. All shape labels MUST be **substrate-independent** (NOT from the same `F(t)`
   ridge that built the encoder input). Oracle labels + γ replace the old
   `chevron_valley`/`syllable_type` heuristics.
2. Mandatory baselines: random-init encoder + per-column-average identity; learned
   must beat **both** by ≥0.10 on NMI/η²/k-NN.
3. Primary metrics: NMI vs labels, chevron-vs-non k-NN purity (k=10), linear probe.
   shape η² is **secondary**.

---

## 🔒 Files NOT to touch

`models/shape_kmeans/k20.joblib`, `results/lab_classifier_v1/best.pt`,
`scripts/deepsqueak_focus_stft.py`, `scripts/contour_mask_utils.py`,
`src/usv_spectrogram/corpus.py` + `ExtractionConfig`,
`scripts/run_batch_detection.py`, `app/core/sliding_inference.py`,
`app/core/{audio_loader,notch,denoise}.py`, `postprocessing/`,
`archive/cleaning_legacy/` (read-only history).

---

## Rig coordination

A3 (render) / A4-at-scale / A6 (train) land on `cloudyclaude`
(`shachar@100.113.224.57`, repo `/data/mickey_london_lab`, non-git rsync copy;
box pushes via SSH-443). Per `feedback_rig_claude_mediation`, read-only inspection
is free but **compute launches are gated by the user per-session** — get an explicit
OK before launching the 40k render. Move code+data by rsync (rig can't reach GitHub).

A5/A7/A8 are box CPU + user review. A7 (γ tool) is box-only and load-bearing — a
good thing to build and hand the user early so labeling can proceed in parallel.

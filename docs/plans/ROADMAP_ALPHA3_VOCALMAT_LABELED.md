# ROADMAP — α₃ shape representation via externally-anchored labels (α₃-C variant)

**Status:** ACTIVE — 2026-05-29. Canonical Markdown of the prior session's
`alpha3_roadmap.html`, adapted for the **α₃-C** variant chosen by the user
(2026-05-29). Supersedes the HTML's Phases A1–A2.

**Predecessor handoff:** `docs/handoffs/2026-05-28_alpha3-vocalmat-blocker-and-pivot.md`
**Superseded plan (record only):** `docs/plans/ROADMAP_SHAPE_INVARIANT_LATENT.md`
(keep its eval-validity controls + modal-prior framing).

---

## Goal (one paragraph)

Build an unsupervised learned shape representation of mouse USVs that clusters by
**geometry** (chevron-with-chevron, jump-with-jump), invariant to cage / mean
pitch / onset+duration. Six prior 2-D image-VAE attempts were formally CLOSED as a
family on 2026-05-28 (`docs/handoffs/2026-05-28_pathway-B-kill-and-canonical.md`).
α₃ breaks the **eval-circularity** problem those attempts were judged by: all prior
"shape labels" (`chevron_valley`, `syllable_type`) derived from the same `F(t)`
ridge that built the encoder input, making the eval self-fulfilling. α₃ uses
**external** taxonomy labels as the eval anchor, paired with **γ** (~200 user
hand-labels) for human-judgment ground truth.

---

## Decision record — why α₃-C

The original α₃ plan loaded the published VocalMat AlexNet weights
(`Mdl_categorical_DL.mat`, 212 MB, OSF `bk2uj/Extras/`) directly as the oracle.
**Blocker:** that `.mat` is a MATLAB v7.0 file with the network serialized as a
228 MB MCOS *opaque* object inside `__function_workspace__`. `scipy.io.loadmat`
and `pymatreader 1.2.3` both fail ("Complex objects … not supported"); the file
is **not** v7.3 HDF5, so the roadmap HTML's "h5py + weight transcription" A1 is
falsified. No community PyTorch/ONNX port exists. The `.mat` is preserved as a
provenance artifact at `data/vocalmat/Mdl_categorical_DL.mat` (gitignored).

**Chosen variant (user, 2026-05-29): α₃-C + γ in parallel.**

| Variant | Oracle | Cost | Quality | Chosen |
|---|---|---|---|---|
| α₃-A | MATLAB Engine on rig → literal author weights | license + ~½ day | author-exact | no |
| α₃-B | re-train AlexNet on 12,221 VocalMat PNGs | ~3–6 hr rig GPU | same arch, our weights | no |
| **α₃-C** | `results/lab_classifier_v1/best.pt` (ResNet-18, val 0.77 F1) | **$0, local** | ~3–5% arch gap; same lab→wild domain gap | **✅** |
| γ-only | skip α; 200 hand-labels only | ~1–2 hr user | honest small-N human | (γ runs anyway) |

**Quantitative justification:** α₃-B and α₃-C label outputs are predicted to agree
on ≥85% of top-1 calls — the bottleneck on label quality is the lab→our-data
**domain gap**, not the AlexNet-vs-ResNet-18 architecture difference. So α₃-C
captures essentially all of α₃-B's value at $0; α₃-B's only marginal gain is
"architectural fidelity for a methods writeup," addable later as a sanity pass.

**A4 oracle verified sound (2026-05-29):** `scripts/experiments/label_patches_v1.py`
reproduces the documented v1 numbers exactly on `held_out_844` —
**balanced acc 0.812** (noise-recall 0.640, usv-recall 0.984). The oracle loads
and infers correctly before any rig render is spent.

---

## Phase plan (A3 → A8; A1/A2 collapsed for α₃-C)

α₃-C collapses A1 (load .mat) and A2 (verify on VocalMat PNGs): the oracle already
exists with a known val/test 0.77 macro F1 and a verified 0.812 held-out balanced
accuracy. No `.mat` loading, no AlexNet PNG sanity gate.

### A3 — render 131204 patches in VocalMat PNG style  *(rig CPU, ~2–4 hr)*
For each detected call in `classified_detections_lab_131204_clean.csv` (~40,787):
- Load WAV window centered on the call.
- **Q2 = resample 300→250 kHz** (`classifier.resample.resample_to_vocalmat`,
  5/6 polyphase) — matches VocalMat's training distribution. *Revisit at the A3
  gate if A4 label quality looks off; keeping 300 kHz is the fallback.*
- Apply **Stack 4** (`scripts/deepsqueak_focus_stft.py` + `contour_mask_utils.py`)
  — per-call adaptive STFT + contour + bandwidth mask.
- Render VocalMat-style PNG: `10·log10(P)` → freq-crop > 45 kHz → `mat2gray`
  min-max → `flipud` → 227×227 bicubic → 3-channel.
- **Output:** `data/alpha3_patches/<call_id>.png` + `data/alpha3_patches/manifest.csv`.
- Script: `scripts/experiments/render_vocalmat_style_patches.py` (to write).

### A4 — label our patches via the v1 oracle  *(rig GPU ~30 min, or box CPU)*
- Run `scripts/experiments/label_patches_v1.py --manifest data/alpha3_patches/manifest.csv`.
- Output: `data/labels_vocalmat_v1_on_131204.csv` (`call_id, top1_class, top1_idx,
  top1_prob, high_confidence, softmax_12`).
- High-confidence filter ≥ 0.85 for eval gold. **⚠ calibration caveat:** on
  held_out_844 only 9.7% clear 0.85 and Chevron is near-zero — if the 40k render
  yields too little high-conf gold, lower the threshold or use top-1 regardless of
  confidence with a per-class floor. Decide at A4 from the real distribution.

### A5 — user spot-check  *(box CPU render + ~10 min user review)*
- Render `$CLAUDE_JOB_DIR/alpha3_label_audit.html`: 12 sections, one per class,
  20 random high-confidence patches each.
- Script: `scripts/experiments/render_label_audit_html.py` (to write).
- **GATE:** ≥ ~80% of patches per class "look right." Heavy mislabeling ⇒ retry A3
  with the 300 kHz choice, or fall back to γ-only.

### A6 — train shape VAE on Stack-4 patches with v1 labels  *(rig GPU ~2–4 hr)*
- Substrate: the Stack-4 contour-masked patches from A3.
- Architecture: minimal β-VAE, z=32, β=0.5 (same family as production), trained on
  the high-confidence-labeled subset.
- Script: `scripts/experiments/train_shape_vae_alpha3.py` (to write).
- **Per-round eval** (held-out 10% split):
  - NMI(latent k-means-20, oracle top-1 class)
  - k-NN purity (k=10) on chevron, jump, complex, flat
  - linear probe accuracy (latent → oracle class)
  - **Random-init encoder baseline + per-column-average identity baseline**
    (learned must beat **both** by ≥ 0.10 — eval-validity rule).
- **SHIP:** NMI ≥ 0.25 **AND** chevron-vs-others k-NN purity ≥ 0.50 **AND** beats
  random-init + identity by ≥ 0.10.
- **KILL:** NMI < 0.15 **OR** fails to beat the baselines. Combined with an A8
  γ-disagreement, the VAE family closes permanently for shape clustering.

### A7 — γ: 200 hand-labels  *(box CPU tool build ~30 min + ~1–2 hr user)*  **[PARALLEL]**
- Sample 200 patches stratified by cohort (50 each: 5970, 3452, 9252, lab_131204)
  and dominant slope-sign (so chevrons + jumps + flats + complex all appear).
- PyQt6 single-screen tool: one patch + 8 keypress labels
  `C`=chevron `J`=jump `F`=flat `X`=complex `U`=up-FM `D`=down-FM `M`=multi-component `?`=unclear.
- Output: `data/manual_shape_labels.csv` keyed by `call_id`. **Permanent eval anchor.**
- Script: `scripts/labeling/hand_label_200.py` (to write).
- γ ships independently of α₃'s success/failure — it is the load-bearing external
  anchor either way.

### A8 — cross-validate oracle ↔ γ taxonomy  *(box CPU ~30 min + ~5 min user)*
- On the 200 γ-labeled patches, look up their A4 oracle labels.
- Confusion matrix (rows = γ, cols = oracle classes) + Cramér's V / NMI.
- Script: `scripts/experiments/vocalmat_vs_gamma_crosstab.py` (to write).
- **Interpretation:** V > 0.5 ⇒ the 12-class taxonomy IS geometric in the user's
  sense (α₃ is a credible anchor). V < 0.3 ⇒ the taxonomy keys on something other
  than "shape"; γ is the only honest anchor and α₃'s VAE eval is contaminated.

---

## Decision tree (post-execution)

```
A5 fail (>50% wrong/class)  → retry A3 with 300 kHz; still bad → γ-only.
A6 ship + A8 high V         → ship α₃ VAE as production shape rep (γ supporting).
A6 ship + A8 low V          → ship VAE only as "agrees with VocalMat taxonomy",
                              explicitly NOT "agrees with user γ shape judgment".
A6 marginal + A8 high V     → ship registration; α₃ confirms registration ≈ taxonomy.
A6 kill (NMI < 0.15)        → VAE family CLOSED permanently; ship registration;
                              γ becomes the human-anchored eval gold.
γ-only fallback             → γ adjudicates registration + frozen-B-encoder +
                              production contour-VAE in a 3-way comparison.
```

**Modal prior (committed):** P(shape lives in 1-D, 2-D substrate adds nothing) ≥ 0.5.
The 6/6 falsification record is strong; α₃-C + γ is a ~25% bet on a worthwhile
shipping artifact. A clean kill closes the family permanently — that is an
acceptable, informative outcome, not a failure of the plan.

---

## Eval-validity rules (load-bearing — do not regress)

1. **All shape labels MUST be substrate-independent** — NOT derived from the same
   `F(t)` ridge that built the encoder input. (The user caught `chevron_valley` /
   `syllable_type` as substrate-derived; the oracle labels + γ replace them.)
2. **Mandatory baselines for any shape-clustering claim:** random-init encoder
   (untrained same arch) + identity (per-column average → KMeans-20). Learned must
   beat **both** by ≥ 0.10 on NMI / η² / k-NN.
3. **Primary metrics (in order):** NMI vs labels, chevron-vs-non k-NN purity (k=10),
   linear-probe acc. **shape η² is secondary**, interpreted with baselines.

---

## Files this roadmap creates

| Path | Purpose | Phase | Status |
|---|---|---|---|
| `scripts/experiments/label_patches_v1.py` | v1 oracle inference (A4) | A4 | **DONE 2026-05-29** |
| `results/alpha3/labels_v1_held_out_844.csv` | oracle smoke output | A4 | DONE |
| `scripts/experiments/render_vocalmat_style_patches.py` | Stack 4 → PNG | A3 | to write |
| `data/alpha3_patches/` | rendered PNGs + manifest | A3 | pending rig |
| `data/labels_vocalmat_v1_on_131204.csv` | oracle labels on 40k | A4 | pending A3 |
| `scripts/experiments/render_label_audit_html.py` | contact sheet | A5 | to write |
| `scripts/experiments/train_shape_vae_alpha3.py` | VAE training | A6 | to write |
| `scripts/labeling/hand_label_200.py` | PyQt6 γ tool | A7 | to write |
| `data/manual_shape_labels.csv` | γ labels (permanent anchor) | A7 | pending user |
| `scripts/experiments/vocalmat_vs_gamma_crosstab.py` | oracle ↔ γ cross-tab | A8 | to write |

---

## Files / paths NOT to touch (🔒 binding)

- `models/shape_kmeans/k20.joblib`, `results/lab_classifier_v1/best.pt` — production artifacts (read-only).
- `scripts/deepsqueak_focus_stft.py`, `scripts/contour_mask_utils.py` — Stack 4 canonical cleaning.
- `src/usv_spectrogram/corpus.py`, `ExtractionConfig` — frozen by the CNN training grid.
- `scripts/run_batch_detection.py`, `app/core/sliding_inference.py`,
  `app/core/audio_loader.py`, `app/core/notch.py`, `app/core/denoise.py`,
  `postprocessing/` — production detection pipeline.
- `archive/cleaning_legacy/` — read-only history (Stacks 1 & 3).

**Cleaning canonical (locked 2026-05-28):** "our cleaning pipeline" = **Stack 4**
only, in any conversation. Stacks 2a/2b = production-detection cleaning. Stacks 1/3
= archived.

---

## Compute / wall-clock

| Phase | Wall-clock | Where | User action |
|---|---|---|---|
| A3 | ~2–4 hr | rig CPU | — |
| A4 | ~30 min | rig GPU / box CPU | — |
| A5 | ~15 min render | box CPU | review contact sheet (~10 min) |
| A6 | ~2–4 hr | rig GPU | — |
| A7 | ~30 min build | box CPU | hand-label 200 (~1–2 hr) |
| A8 | ~30 min | box CPU | review cross-tab (~5 min) |
| **Total** | **~1–1.5 days active + ~1–2 hr user** | | |

**Rig coordination:** A3/A4/A6 land on `cloudyclaude` (`shachar@100.113.224.57`,
repo `/data/mickey_london_lab`, rsync copy). Per `feedback_rig_claude_mediation`,
compute launches are gated by the user per-session — confirm before launching A3.

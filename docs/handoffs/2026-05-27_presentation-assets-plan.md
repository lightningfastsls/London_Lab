# Handoff: Presentation Asset Plan
Date: 2026-05-27

## Task

The user asked for a repo-only planning pass for the USV presentation images and graphs. No deck edits were requested in this chat. This handoff tells the next presentation chat where the deck, current assets, model-output figures, and missing/rework items live.

Current proposed target: 18 slides for about 12 minutes, expanding the 13-slide v4 deck. The high-level slide list is usable, but the CNN iteration slides need naming/claim corrections before assets are inserted.

## Files Changed

- `docs/handoffs/2026-05-27_presentation-assets-plan.md` -- this planning handoff only.

No `.pptx`, source code, figures, or result artifacts were modified.

## Reasoning

### Deck source of truth

The `.pptx` is the source of truth, not `build_deck_v3.js`.

Read first:
- `C:/Users/shach/OneDrive/מסמכים/HANDOFF_PRESENTATION_v4.md`
- `/mnt/c/Users/shach/OneDrive/מסמכים/v4_workspace/v4_workspace/USV_presentation_v3.pptx`
- `/mnt/c/Users/shach/OneDrive/מסמכים/v4_workspace/v4_workspace/slide_renders/`

Do not re-run `build_deck_v3.js`; use it only as design-token/layout reference. For real deck edits, unpack/pack the existing PPTX and verify by rendering to PDF/JPEG.

### Slide-list corrections

The user's 18-slide sequence is broadly right. I recommend these corrections:

1. Slide 4 title/question should be explicit: **"How do you measure courtship when you can only hear it?"** Visual should be "wanted synchronized audio + video" vs "available: audio only." Keep the bird-song analogy verbal; do not spend a slide explaining bird song.
2. Avoid saying "v1/v2/production" without defining them. In this repo, `models/production/` is the oldest CNN, not current production. Current production detection model is `models/hard_neg_retrain/best_model.pt`.
3. The better three-step CNN story is:
   - **First CNN**: `models/production/`, partial reconstruction only, illustrative whole-file AUC 0.821 when scored with native inferno/25-110 extraction.
   - **Matched-window CNN**: `models/matched_windows/`, fixed train/inference window mismatch, still noise-prone; whole-file AUC 0.885 on the 33-file illustrative run and 0.877 on the larger matched-vs-production run.
   - **Hard-negative retrain**: `models/hard_neg_retrain/`, current production; whole-file AUC 0.920 on the 33-file illustrative run and 0.914 on the larger matched-vs-production run.
4. Do not use the per-patch AUC plateau as the slide headline. It gives about 0.964/0.989/0.989 and is explicitly called misleading in the CNN handoffs because the negatives are curated patches, not real whole-recording noise.

### Asset plan by slide

1. **Title** -- current slide 1/title can stay.
2. **Question: "Have lab mice forgotten how to court?"** -- unhide/polish current slide 2. Needs real lab/wild mouse photos. Browse for open-license or user-owned images in the next chat; do not use random pet/stock mice without source provenance. The wild photo should be actual *Mus musculus* or a defensible wild/house-mouse proxy.
3. **Why there should be a difference** -- build as a simple native PPT timeline/diagram: about 100 years of inbreeding, mate choice removed, sexual selection abolished. No repo figure needed.
4. **How to measure** -- use the title above. Native diagram: video icon + audio icon on "ideal", video crossed/greyed on "available." No repo figure needed unless the user wants a bird-song spectrogram.
5. **What's a USV?** -- needs a new clean spectrogram intro. Best candidate source:
   - `5970 USV/2024-09-30_11-20-14_0000022.wav`
   - flat-call exemplar at about `5.727 s`, documented in `presentation/figures/thumbs/SLIDE14_AUDIT_REPORT.md`
   - inspection PNG: `presentation/figures/thumbs/candidates/Flat/1_inspect.png`
   Render a new deck-ready PNG without title clutter, vertical guide lines, or CNN probability overlays. Use canonical `sr=300000` and import corpus constants.
6. **...and they're noisy** -- current slide 6 exists, but the render shows black side bars/clipping. Re-render or replace with a clean noisy spectrogram crop. Existing diagnostic examples:
   - `presentation/figures/noise_full_files/2024-09-30_21-18-09_0003091__noise_with_cnn_curve.png`
   - `presentation/figures/noise_full_files/131204_1400_m1fm1_chunk_003__noise_with_cnn_curve.png`
   These are useful references, but likely need a deck-native crop.
7. **How we record** -- current slide 3 video is already embedded in the v4 deck. Verify in PowerPoint; PDF renders show a black rectangle due to LibreOffice.
8. **Why I built my own labeling tool** -- current slide 7 still has a placeholder. If this is a local Codex session, run/screenshot the PyQt app; if it is a web-only presentation chat, ask the user for a screenshot. Do not fabricate the app UI.
9. **First CNN / early CNN** -- decide whether this means the true first model (`models/production/`) or the first solid baseline (`models/matched_windows/`). If using the true first model, label the graph as illustrative and include the native-extraction caveat.
10. **Matched-window CNN** -- use `presentation/figures/cnn_improvement/cnn_matched_vs_prod_scaled.png` for the validated matched-vs-production quantitative story, or crop/redraw it into the deck style.
11. **Hard-negative retrain** -- current production. Use the same validated quantitative story plus a qualitative noise-rejection grid. Existing qualitative composite is `presentation/figures/cnn_improvement/cnn_noise_vs_usv_composite.png`, but it has a dead USV column and should be reworked into a multi-noise-file before/after grid.
12. **How does a CNN actually work?** -- YouTube embed placeholder is fine.
13. **CNN architecture** -- CNN feature figures already exist:
   - `/mnt/c/Users/shach/OneDrive/מסמכים/cnn_feature_figures/cnn_activations_demo.png`
   - `/mnt/c/Users/shach/OneDrive/מסמכים/cnn_feature_figures/cnn_gradcam_positive.png`
   - `/mnt/c/Users/shach/OneDrive/מסמכים/cnn_feature_figures/cnn_gradcam_negative.png`
   Prefer activation stack + positive Grad-CAM. Negative Grad-CAM is optional and visually noisier. Notes are in `/mnt/c/Users/shach/OneDrive/מסמכים/cnn_feature_figures/NOTES.md`.
14. **What mistakes can the model make?** -- current slide 11 can stay unless the user wants a redesign.
15. **Do these calls actually exist? DeepSqueak** -- current slide 12 can stay. Source figure: `presentation/figures/06_deepsqueak_validation/deepsqueak_agreement.png`.
16. **Generalization to lab audio** -- needs a deck-ready lab detection example. Strong candidate source:
   - WAV: `USV_lab_131204_chunked_2s_full/131204_1800_m2fm2_chunk_010.wav`
   - Detection JSON: `results/batch_lab_full_softnotch_20260513_1538/detections/131204_1800_m2fm2_chunk_010.json`
   - Existing diagnostic PNG: `results/batch_lab_full_softnotch_20260513_1538/eyeball_inspection/clean_control/131204_1800_m2fm2_chunk_010_132ms.png`
   Re-render as a clean slide figure without diagnostic title/colorbar clutter.
17. **Now: the actual question...** -- transition slide. No result figure unless the user decides to show early lab-vs-wild stats.
18. **Thank you** -- current slide 13 can stay.

### CNN slide assets and commands

Validated/available CNN figures:
- `presentation/figures/cnn_improvement/cnn_wholefile_roc_nativefirst.png` -- three-model illustrative ROC, AUC 0.821 -> 0.885 -> 0.920.
- `presentation/figures/cnn_improvement/cnn_matched_vs_prod_scaled.png` -- validated 248-file matched-vs-production slide, AUC 0.877 -> 0.914, known-noise flagged windows 4.9% -> 1.3%.
- `presentation/figures/cnn_improvement/_predictions/scaled_window_scores.csv` -- saved scores for fast redraw.

Qualitative hard-negative candidates:
- `5970/USV4/usv_lmt_034/2024-10-01_17-18-59_0005656.wav` -- old 9 false boxes, production 0.
- `5970/USV4/usv_lmt_034/2024-09-30_22-36-23_0003502.wav` -- old 5 false boxes, production 0.
- `5970/USV1/usv_lmt_034/2024-09-30_17-11-57_0001700.wav`
- `5970/USV2/usv_lmt_034/2024-09-30_22-35-36_0003493.wav`

Detection JSONs for those stems exist in both:
- `results/batch_5970/detections/`
- `results/batch_5970_v2_full/detections/`

Suggested lab-detection render command:

```bash
.venv/bin/python scripts/make_cnn_progression_slide.py \
  --wav USV_lab_131204_chunked_2s_full/131204_1800_m2fm2_chunk_010.wav \
  --stage "Production CNN on lab audio=results/batch_lab_full_softnotch_20260513_1538/detections/131204_1800_m2fm2_chunk_010.json" \
  --out presentation/figures/lab_generalization_131204_1800_m2fm2_chunk_010.png \
  --title "Production CNN on lab-strain audio"
```

This command is a starting point; the result may still need presentation styling.

## Validation

Repo orientation files were read: `AGENTS.md`, `ops/goals.md`, `ops/reminders.md`, `docs/codex_index.md`.

Task-specific files read or inspected include:
- `C:/Users/shach/OneDrive/מסמכים/HANDOFF_PRESENTATION_v4.md`
- `presentation/FIGURE_GUIDE_FOR_WEB_SESSION.md`
- `presentation/HANDOFF_FROM_CLAUDE_CODE.md`
- `docs/handoffs/HANDOFF_01_PRESENTATION_STRUCTURE.md`
- `docs/handoffs/2026-05-26_cnn-improvement-slide-handoff.md`
- `docs/handoffs/2026-05-27_cnn-iteration-comparison-followup.md`
- `docs/handoffs/2026-05-27_cnn-iteration-eval-redo.md`
- `docs/handoffs/v2-full-pipeline-results.md`
- `docs/handoffs/hard-neg-retrain-results.md`
- `docs/handoffs/deepsqueak-full-pipeline-results.md`
- `docs/handoffs/2026-05-14_lab_131204_post_labeling.md`
- `docs/handoffs/2026-05-15_lab_131204_phase2c_complete.md`

Several candidate PNGs were visually inspected, including the CNN ROC figures, CNN feature figures, current slide renders, and lab detection examples.

This is docs-only work. After writing, re-read this handoff and verify referenced paths that are meant to exist.

## Open Questions / Known Risks

- The exact meaning of "v1" and "v2" in the proposed slide list is ambiguous. Resolve this before making slides 9-11.
- Web photo sourcing was not done in this repo-only pass. The next chat should browse and record image provenance/license for lab/wild mouse photos.
- Existing CNN qualitative composite is not final quality for the new narrative; it should be redesigned or regenerated.
- The lab-audio candidate is visually strong but currently only exists as a diagnostic rendering; it needs a cleaner slide figure.
- Current v4 slide footers still say `/27`; the v4 handoff says not to fix this until the deck stabilizes.

## Worth Remembering For Claude

- Current production detection model is `models/hard_neg_retrain/best_model.pt`; the directory `models/production/` is the oldest CNN, not current production.
- Use whole-file sliding ROC for the "CNN got better" story. Do not use the per-patch AUC plateau as the headline.
- If reproducing the first CNN, pass legacy inferno/25-110 extraction explicitly; never change `ExtractionConfig` defaults or `corpus.py`.
- Any WAV/spectrogram rendering must specify or import canonical `sr=300000`.
- The deck should be edited directly as PPTX. Do not regenerate from `build_deck_v3.js`.

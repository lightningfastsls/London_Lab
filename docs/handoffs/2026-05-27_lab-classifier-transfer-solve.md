# Handoff — Solve the lab/wild USV classifier transfer problem

**Date:** 2026-05-27
**Resume with:** `/execute docs/handoffs/2026-05-27_lab-classifier-transfer-solve.md`
**Predecessor context:** interactive diagnostic session (no production code changed). All
findings below were measured this session and are reproducible.

---

## Mission

Produce a classifier that reliably types **our** USVs (lab 131204 + wild 5970/3452/9252).
The blocker has been transfer: VocalMat-trained → our 300 kHz recordings. This session
**reframed the problem** and several previously-assumed paths turned out to be dead ends
or already-decent. Read "Corrected picture" before choosing an approach — the obvious
moves (DANN, clean-and-retrain, re-render VocalMat) are mostly closed.

---

## Corrected picture (what changed this session)

1. **The base VocalMat model already transfers to lab usv/noise reasonably — and DANN BROKE it.**
   Scored on the *same* 844-row lab held-out set (`scripts/evaluate_held_out_844.py` logic,
   argmax==Noise→noise collapse):

   | Model | balanced acc | noise recall | usv recall |
   |---|---|---|---|
   | **v1 VocalMat base** (`results/lab_classifier_v1/best.pt`) | **0.81** | **0.64** | 0.98 |
   | v2 DANN (`results/lab_classifier_v2/best.pt`) | 0.58 | 0.16 | 0.99 |

   The "16% noise recall / poor lab transfer" in `results/lab_classifier_v2/held_out_844_eval.md`
   is the **DANN-corrupted** model. **DANN did not just fail cage-invariance (all 3 gates failed,
   see `comparison_v1_vs_v2.md`); it destroyed the base model's noise rejection.**
   → **Action implied: shelve v2/DANN. v1 is the working model.** Update `ops/goals.md`
   (currently implies 18.4/DANN is the live path).

2. **Patch-duration is an operating-point knob, NOT a discrimination fix (for lab usv/noise).**
   Call-centered re-extraction sweep at 0.22/0.14/0.08/0.05 s, v1 model
   (`scripts/experiments/patch_duration_sweep.py`): **balanced accuracy is FLAT ~0.80–0.81**
   across all durations. Shrinking the window trades usv_recall (0.99→0.71) for
   noise_recall (0.61→0.90). So a shorter window is a *free threshold shift* (e.g. 0.08 s →
   0.74/0.88) but does not raise the ceiling. The vault hypothesis
   `notes/patch-duration mismatch was the mechanistic cause of VocalMat-on-wild kappa 0.13 failure.md`
   is **NOT confirmed for lab** — but its actual claim is about **wild 12-class κ=0.13**, which
   is **still untested** (see Path A).

3. **Re-rendering VocalMat through our pipeline is DEAD.** OSF `bk2uj` audio (7 GT WAVs,
   ~2.1 GB) is a *different recording set* than the 12,954 labeled images (disjoint ID spaces;
   verified via OSF API by a sub-agent). Source audio for the labeled corpus is **not
   distributed**. The PNGs are usable only as-is (pre-rendered 227×227, VocalMat-cleaned;
   our pipeline never touches them — only our own WAVs go through `clean_spectrogram`).

4. **Pseudo-labeling / self-training is now MORE viable than the record implies.** Prior
   `project_vocalmat_transfer_test` logged auto-labels at κ=0.08, but that pessimism is
   entangled with the weak/DANN-era setup. v1 at balanced-acc 0.81 on lab is a decent enough
   source model to bootstrap from — *especially with human review in the loop* (the user has
   said they will manually correct where needed).

---

## Candidate solution paths (ranked; pick with the user)

**Path A — Test the patch-duration hypothesis on WILD (cheap, closes an open question).**
Run the same sweep on wild 5970 (the note's actual claim). Needs a wild labeled set:
5970 has human usv/noise GT (see `project_cnn_iteration_eval_5970`); 12-class wild GT does
NOT exist (only rule-based `syllable_type`). So this can cleanly test usv/noise transfer on
wild, and at best a rule-label proxy for 12-class. Reuse `patch_duration_sweep.py` pointed at
`5970 USV/` WAVs + a wild verdict manifest. ~30 min. **Recommended first.**

**Path B — Human-in-the-loop active-learning fine-tune of v1 (the real "solve").**
v1 proposes labels on our data → human reviews (prioritise low-confidence + the minority
classes) → fine-tune v1 on the verified set → iterate. This is the sanctioned fallback
(`project_vocalmat_transfer_test`: "revisit VocalMat as human-supervised assist") and the
infrastructure partly exists (`notes/active learning cycle automates the label-train-evaluate-mine loop`,
`training/cycle_report.py`, `run_training_cycle.py` from Phase 10.1). Decisions to make with
user: how many labels to start (active-learning note scales 2K→30K), full fine-tune vs LoRA
(`notes/LoRA exploits low intrinsic rank...` — parameter-efficient, fits the 300 kHz domain-shift
framing), and which cohort first (lab has the 844 verdicts already).

**Path C — Cleaning as a *discrimination* lever (lower priority).** The flat 0.81 ceiling
is what a cleaned retrain would have to beat. Worth a small ablation (does a harder denoise
of our data raise balanced acc above 0.81?) but expectation is modest — cleaning addresses
the pixel-level cage, and the 18.1 gate already showed the *substantive* cage metrics pass
under the current stack. Do NOT re-enable soft-notch without a calibrated tonal library
(it no-ops / over-aggressive — `project_clustering_analysis_lab`).

**Path D — Ship an immediate usable detector now.** If the user wants something usable today:
pick a v1 operating point on the held-out sweep (0.08 s ≈ 0.74 noise / 0.88 usv) and run it +
manual review. No training. Fastest value.

**Recommended sequence: A → B**, with D available as a stopgap and C as an ablation inside B.

---

## Verified artifacts & paths (all confirmed present this session)

- **WORKING model:** `results/lab_classifier_v1/best.pt` (44.8 MB; plain timm ResNet-18 via
  `build_resnet18_classifier(num_classes=12, pretrained=False)`; `metrics.json`, `eval_report.md`).
- **DEPRECATED:** `results/lab_classifier_v2/best.pt` (DANN; do-not-ship; keep as cautionary baseline).
- **Labeled OUR-data set:** `data/lab_cnn_training/held_out_844/manifest.csv` — 844 lab calls
  (705 usv / 139 noise), cols `path, usv_verdict, wav_stem, det_start_s, det_end_s, traditional_label, couple`.
  This is usv/noise only — **no 12-class Grimsley GT exists for our data** (gate withdrawn,
  `docs/handoffs/2026-05-25_ERRATA_held-out-845.md`).
- **Lab source WAVs:** `USV_lab_131204_chunked_2s_full/*.wav` (300 kHz, 2 s, mono; all 752
  held-out stems present).
- **Wild source WAVs:** `5970 USV/` (+ `5970_reviewed/`, `5970_manual_review*`).
- **Sweep script (reproduces all of Finding 2 + the v1 vs v2 table via BASELINE-A):**
  `scripts/experiments/patch_duration_sweep.py`. Run: `.venv/bin/python scripts/experiments/patch_duration_sweep.py`.
- **Production rendering chain to reuse:** `scripts/cnn_prepare_training_data.py`
  (`_spectrogram_db`, `_spec_to_uint8_patch`, `_wav_to_patches`, `--patch-duration-s`).
- **Scorer:** `scripts/evaluate_held_out_844.py` (NOTE: it loads `ResNet18DANN`; for v1 use
  `build_resnet18_classifier` — the sweep script already does this correctly).
- **Memory note:** `project_lab_transfer_v1_vs_dann_patchsweep`.

---

## Binding constraints (flattened from vault — assume the new session has NO vault access)

- **CNN FREEZE:** never modify `src/usv_spectrogram/corpus.py` or `ExtractionConfig`
  (sample rate / USV band / STFT / patch geometry) — changing them silently corrupts
  inference and requires a CNN retrain. Import constants; do not redeclare.
- **`src/usv_spectrogram/classifier/training.py` is the 18.3 reference — do NOT edit it.**
  Build evaluators/fine-tuners as **additive scripts** (the held-out evaluator did this).
- **Cleaning config that the patches actually used:** `CleaningConfig(baseline_mode="percentile")`
  → baseline-subtraction + global-MAD + per-recording-Z; **soft-notch no-ops** (no tonal
  library). Do not turn soft-notch on without a calibrated tonal library — it is over-aggressive
  and erased real USVs in c2 (`project_clustering_analysis_lab`).
- **Global-MAD then crop** — never per-window MAD (silently kills high-confidence USVs;
  `feedback_cnn_inference_global_mad`).
- **VocalMat is 250 kHz, ours is 300 kHz** → `resample_to_vocalmat` (5/6 polyphase). Any
  our-data → model path must resample first.
- **Recording-level splits only** — a recording in train must not appear in val/test, or the
  cage confound leaks (`notes/training-methodology`).
- **Class imbalance ~24×** (e.g. Multi-steps 74 vs Step-up 1806) breaks naive CE → use
  class-weighted + focal + oversampling (`notes/class imbalance 24x breaks naive cross-entropy`).
- **Do NOT eyeball noise-vs-USV** — anchor "this is noise" on human verdicts, not visual reads
  (`feedback_cannot_eyeball_noise_vs_usv`). The 844 verdicts are the trusted lab anchor.
- **Two duration columns:** visual-verdict filters use `det_duration_ms` (the hysteresis event
  shown in PNGs), NOT `call_length_s` (`feedback_duration_columns_differ`).
- **GPU training runs on the rig** `shachar@100.113.224.57` (cloudyclaude, 3× RTX 3060 Ti;
  repo at `/data/mickey_london_lab`, non-git rsync copy). Read-only SSH is free; **compute
  writes/launches need per-session user OK** (`feedback_rig_claude_mediation`). Box↔rig moves
  are rsync (rig can't reach GitHub).
- **Tests-as-spec:** any pre-existing `test-architect` tests are the contract — do not weaken
  their expectations to pass (`/implement` Step 0).

---

## Validation gates (anti-greenwashing — carry these forward)

- **Faithful-preprocessing gate:** any new re-extraction must reproduce the v1 BASELINE-A on
  the 844 set at 0.22 s (balanced acc ≈ 0.81) before its other-duration / other-cohort numbers
  are trusted. The sweep script bakes this in.
- **Report per-class, not pooled:** the held-out set is 83.5 % usv; "always-usv" scores 0.835
  pooled. Read noise_recall / usv_recall / balanced_acc, never pooled accuracy.
- **For any fine-tune:** hold out by *recording* and report whether val improves over the v1
  baseline (0.81 lab balanced acc) by a margin that survives the 139-noise sample size.

---

## Files NOT to touch
`corpus.py` · `ExtractionConfig` · `src/usv_spectrogram/classifier/training.py` ·
`scripts/run_batch_detection.py` · `app/core/sliding_inference.py` · `postprocessing/` ·
`results/lab_classifier_v1/best.pt` (the working model — read-only).

---

## First moves for the new session
1. Full-read this handoff; follow the `/execute` contract. Run `/kcheck` if touching any
   HIGH-risk canary (cleaning / extraction / detection).
2. Re-run `scripts/experiments/patch_duration_sweep.py` to re-confirm the v1-vs-v2 + sweep
   numbers locally (sanity that nothing drifted).
3. Present a CLAUDE.md-shaped approval request for the chosen Path (A recommended first) BEFORE
   any code. Surface to the user the one standing decision: **shelve DANN/v2 and treat v1 as
   the production lab classifier?** (the evidence says yes).

# Handoff — α₃ blocker + path forward (2026-05-28)

**Continuation of:** session at end-of-day 2026-05-28. User explicitly requested this handoff with "doesn't matter that much, just give me a handoff so we continue from right here with another chat."

**Next chat's job (one sentence):** pick **α₃-C** (use existing `lab_classifier_v1` as labeling oracle) **OR** α₃-A (install MATLAB on rig) **OR** α₃-B (re-train AlexNet from scratch on VocalMat PNGs) **OR** γ-only (skip α entirely), then execute Phases A3 → A8 of `$CLAUDE_JOB_DIR/alpha3_roadmap.html` accordingly. **Recommendation = α₃-C + γ in parallel.**

---

## Status as of this moment

### What we're trying to do (one paragraph)
Build an unsupervised learned shape representation of mouse USVs that clusters by **geometry** (chevron-with-chevron, jump-with-jump) and is invariant to cage / mean pitch / onset+duration. Six prior 2-D image-VAE attempts were formally CLOSED as a family on 2026-05-28 (`docs/handoffs/2026-05-28_pathway-B-kill-and-canonical.md`). The current bet (α₃) tries to break the eval-circularity problem the prior 6 attempts were judged by — the user identified that all "shape labels" used in prior evals (`chevron_valley`, `syllable_type`) derive from the same `F(t)` ridge that built the substrate, making the eval self-fulfilling. α₃ uses **external** VocalMat-taxonomy labels as the eval anchor, paired with γ (~200 user-hand-labels) for human-judgment ground truth.

### The blocker that stopped this session
The original α₃ pitch was *"load `Mdl_categorical_DL.mat` (the published VocalMat AlexNet weights) directly as a Python labeling oracle."* That .mat file IS now downloaded at `data/vocalmat/Mdl_categorical_DL.mat` (212 MB, verified). **But** it's a MATLAB v7.0 file containing a `SeriesNetwork` object named `netTransfer` serialized as a 228 MB MATLAB MCOS opaque object inside `__function_workspace__`. Two pure-Python paths tried and failed: `scipy.io.loadmat` and `pymatreader 1.2.3` (both see the wrapper, emit "Complex objects (like classes) are not supported", and can't decode the payload). Octave is not installed locally. WebSearch + WebFetch of the VocalMat GitHub README confirm: **no community PyTorch/ONNX port exists**.

### Three degraded variants of α₃, plus γ-only
| Option | What it is | Cost | Quality |
|---|---|---|---|
| **α₃-A** | Install MATLAB Engine API on the rig, load the .mat there, run inference | License + ~half-day setup | Literal author weights |
| **α₃-B** | Re-train AlexNet from scratch on the 12,221 VocalMat PNGs (ImageNet-pretrained AlexNet + 12-class FC head; recipe in [VocalMat `train_model.m`](https://github.com/ahof1704/VocalMat/blob/master/vocalmat_classifier/training/train_model.m)) | ~3–6 hr rig GPU + downloading 12,221 PNGs from OSF Dataset/ | Same architecture as published; our weights, not theirs |
| **α₃-C** ⭐ | Use existing `lab_classifier_v1/best.pt` (ResNet-18 trained on the same 12,221 VocalMat PNGs, val 0.77 macro F1, held-out lab balanced acc 0.812) as A4 oracle | $0 — exists on rig | Different arch (ResNet-18 ≠ AlexNet); ~3–5% acc gap; same lab→wild domain gap |
| **γ-only** | Skip α entirely; user hand-labels 200 of our patches via PyQt6 tool; use as eval anchor for all existing reps (registration, frozen B encoder, contour-VAE) | ~1–2 hr user time + tool build | Honest small-N human ground truth |

**Quantitative prediction the next chat should respect:** α₃-B and α₃-C label outputs on our patches will agree on ≥85% of top-1 calls — because the bottleneck on label quality is the lab→our-data domain gap, NOT the AlexNet-vs-ResNet-18 architecture difference (per `lab_classifier_v1`'s known 0.64 noise-recall at the lab/wild boundary). So α₃-B's marginal value over α₃-C is ~"architectural fidelity for methods writeup" not "more accurate labels."

### Why α₃-C + γ is the recommendation
- α₃-C costs $0; the model is already on the rig at `/data/shachar/.../lab_classifier_v1/best.pt`.
- γ is the load-bearing external anchor either way; without it, neither α₃-B nor α₃-C has a substrate-independent eval and we re-inherit the circularity problem.
- α₃-B only justifies its 3–6 hr cost if architectural fidelity matters for a paper — which can be added later as a sanity-check pass.

---

## What's been done in this session (full state)

### Documents written / updated (uncommitted)
| Path | What | Status |
|---|---|---|
| `docs/plans/ROADMAP_SHAPE_INVARIANT_LATENT.md` | v2 roadmap for the substrate-hypothesis VAE family (Phase 0a linear probe + Phase 1 contrastive on registered substrate + Phase 2 post-hoc residualization) | **Superseded** by α₃ pivot — keep as record but do NOT execute |
| `docs/reviews/ROADMAP_SHAPE_INVARIANT_LATENT-adversarial.md` | 11-item adversarial review of v1 of the substrate-hypothesis roadmap (11 required revisions; 5 strongest attacks) | Reference; informs the eval-validity controls below |
| `docs/modules/cleaning-subsystems.md` | Rewritten with canonical-banner: "WHEN ANY DOC OR CONVERSATION REFERS TO 'OUR CLEANING PIPELINE', IT MEANS STACK 4 (DeepSqueak focus-STFT port) AND ONLY STACK 4" | Live |
| `scripts/deepsqueak_focus_stft.py` | Canary banner added at top declaring canonical status | Live |
| `src/usv_spectrogram/classifier/__init__.py` | Trimmed to remove archived Stack 1 re-exports (`CleaningConfig`, `clean_spectrogram`, `DiagnosticResult`, `notch_injection_test`, `per_band_cohens_d`, `knn_same_cohort_rate`, `raw_pixel_pca_d`, `train_diagnostic_vae`) | Live; `build_resnet18_classifier` etc. still importable |
| `archive/cleaning_legacy/README.md` | Explains what's archived, why, what stayed live | New |
| `$CLAUDE_JOB_DIR/alpha3_roadmap.html` | 8-phase α₃ roadmap with decisions, gates, kill conditions, file manifest | Reference; the next chat should NOT re-render this — it's still accurate for A3–A8 once A4's oracle is picked |
| `/home/shachar/.claude/projects/-home-shachar-projects-mickey-london-lab/memory/project_cleaning_stacks_three_distinct.md` | Memory rewritten to reflect canonical re-designation + archive | Live |
| `data/vocalmat/Mdl_categorical_DL.mat` | Downloaded 212 MB VocalMat AlexNet .mat (cannot decode without MATLAB; preserve as provenance artifact) | Downloaded |

### Files moved (Stack 1 + Stack 3 archive — 50 files staged, NOT YET COMMITTED)
- **Stack 1 → `archive/cleaning_legacy/stack1/`**: `src/usv_spectrogram/classifier/cleaning_pipeline.py` + `classifier/diagnostics.py` + 13 caller scripts (cnn_prepare_training_data, prepare_held_out_844, cnn_cleaning_validation, benchmark_baseline_modes, compare_baseline_modes_visual, cnn_wild_topup, profile_prep_phases, train_lab_classifier, post_prep_reconcile, regen_sanity_patches, cnn_download_vocalmat_sample, scripts/experiments/patch_duration_sweep) + 4 test files. **Reason archived:** Module 18.4 DANN dead-end + VocalMat re-render verified dead. `lab_classifier_v1` model weights remain on rig — usable for inference via `build_resnet18_classifier`.
- **Stack 3 → `archive/cleaning_legacy/stack3/`**: entire `src/usv_spectrogram/features/` (`__init__.py`, `ridge_tracker.py`, `spectrogram_filter.py`) + 19 shape-VAE consumer scripts (rig_M{8,9,10,R1,R2}, train_shape_{encoder_contrastive, vae_v3_{deriv,hybrid}}, eval_shape_{encoder,vae_v3,kill_gate_v3}, extract_ridge_targets_v3, shape_registered_clustering, build_shape_map, analyze_latent_{dispersion,repertoire_jsd,transitions}, transition_alphabet_compare, compare_3452_vs_9252) + 10 test files. **Reason archived:** VAE family CLOSED for shape clustering 2026-05-28. Productionized k20 alphabet `models/shape_kmeans/k20.joblib` lives on rig and does not need to be re-derivable.
- **Total: 44 tracked moves via `git mv` + 6 untracked-then-`mv` files = 50 files repositioned**

### Verification done (all PASS)
- `from usv_spectrogram.classifier import build_resnet18_classifier, GRIMSLEY_12_CLASSES` ✓
- `from usv_spectrogram.app.core import audio_loader, denoise, notch` ✓ (production stack at 300 kHz)
- `from deepsqueak_focus_stft import DS_ENTROPY_THRESHOLD; import contour_mask_utils` ✓ (Stack 4 canonical)
- Zero broken imports in the live tree (3 grep hits were pre-broken orphan tests for `features.omer_vectorize` / `features.amvoc_autoencoder` — modules that never existed in main; not caused by archive)

---

## Critical context the next chat MUST respect

### Hands OFF (binding)
- `models/shape_kmeans/k20.joblib` — production registration alphabet
- `scripts/deepsqueak_focus_stft.py` + `scripts/contour_mask_utils.py` — Stack 4 canonical cleaning
- `src/usv_spectrogram/corpus.py`, `ExtractionConfig` — frozen by CNN training grid (CLAUDE.md canary)
- `scripts/run_batch_detection.py`, `app/core/sliding_inference.py`, `app/core/audio_loader.py`, `app/core/notch.py`, `app/core/denoise.py`, `postprocessing/` — production detection pipeline (CLAUDE.md canary)
- `archive/cleaning_legacy/` — read-only history; do not edit files inside

### Eval-validity rules (load-bearing)
1. **All "shape labels" used in any eval MUST be substrate-independent.** This means: NOT derived from the same `F(t)` ridge that built the input to the encoder. The user explicitly caught the prior `chevron_valley` and `syllable_type` heuristics as substrate-derived; do not regress.
2. **Mandatory baselines for any shape-clustering claim:**
   - Random-init encoder baseline (untrained same architecture; learned must beat by ≥0.10 on NMI/η²/k-NN)
   - Identity baseline (per-column average of substrate → KMeans-20; learned must beat by ≥0.10)
3. **Primary eval metrics (in order):** NMI vs labels, chevron-vs-non k-NN purity (k=10), linear probe acc; shape η² is *secondary* and interpreted with baselines.

### Modal prior (committed at top of v2 roadmap)
P(shape genuinely lives in 1-D and 2-D substrate adds nothing useful) ≥ **0.5**. The 6/6 falsification track record is strong; α₃ + γ is a ~25% bet on a worthwhile shipping artifact. **If α₃-C trains and clears the eval, that's a real win against a hostile prior. If it fails, the family closes permanently.**

### Cleaning canonical (locked 2026-05-28)
- "Our cleaning pipeline" = **Stack 4** (DeepSqueak focus-STFT port via `scripts/deepsqueak_focus_stft.py` + `scripts/contour_mask_utils.py`). Always. In any conversation.
- Stacks 2a/2b = production-detection cleaning, NEVER "our cleaning" in conversation.
- Stacks 1/3 = ARCHIVED.

---

## What the next chat should do (step-by-step)

### Step 0 — Get the decision
Ask the user (or read their inline answer if they appended one to this handoff):
> **Pick one: α₃-A / α₃-B / α₃-C / γ-only.**
> If unsure, default is α₃-C + γ in parallel.

### Step 1 — Commit the archive operation (if user OKs)
The Stack 1/3 archive moves (50 files) are staged but not committed. Run:
```bash
git status --short | head -60   # confirm state matches the handoff's description
git add archive/cleaning_legacy/  # for untracked files
git diff --cached --stat | tail
# review with user before:
# git commit -m "refactor(cleaning): archive Stacks 1 and 3; canonicalize Stack 4 as 'our cleaning pipeline'"
```
**Do NOT commit without user OK.** The user has been cautious about commits in this codebase (see [[feedback_no_bulk_stage_in_parallel_chats]]).

### Step 2 — Execute α₃ Phases A3 → A8 (substituting per choice)
The 8-phase plan in `$CLAUDE_JOB_DIR/alpha3_roadmap.html` is still accurate for A3–A8. Phase A1+A2 collapse based on the user's choice:

- **If α₃-C (recommended):** A1+A2 = "use `lab_classifier_v1/best.pt` directly; no .mat loading needed; no verification PNG check (we already have val/test 0.77 macro F1 from training)." Skip to A3.
- **If α₃-A:** A1 = "install MATLAB Engine API + load .mat via MATLAB"; A2 unchanged.
- **If α₃-B:** A1 = "download all 12,221 PNGs from OSF `bk2uj/Dataset/<class>/`; train AlexNet ImageNet→12-class FC on the rig"; A2 = "eval on 844-row held-out (`data/lab_cnn_training/held_out_844/`); macro F1 ≥ 0.70 sanity gate".
- **If γ-only:** Skip all of α. Build only Phase A7 (γ tool) and A8 (cross-validation against existing representations).

### Step 3 — γ in parallel
Regardless of α choice, build the γ labeling tool early (~30 min) and start the user labeling whenever they have ~1–2 hours. This is the load-bearing external anchor.

---

## Critical references

### Plans
- `$CLAUDE_JOB_DIR/alpha3_roadmap.html` — full α₃ phase-by-phase plan with decisions, gates, kill conditions, file manifest (still current; do not re-render)
- `docs/plans/ROADMAP_SHAPE_INVARIANT_LATENT.md` (v2) — superseded but contains the eval-validity controls and modal-prior framing the next chat should keep
- `docs/reviews/ROADMAP_SHAPE_INVARIANT_LATENT-adversarial.md` — 11 required revisions that informed v2

### Handoffs that define the prior dead-ends
- `docs/handoffs/2026-05-28_pathway-B-kill-and-canonical.md` — VAE family CLOSED verdict
- `docs/handoffs/2026-05-28_shape-vae-BA-hybrid-KILL.md` — B+A mechanism dissection
- `docs/handoffs/2026-05-25_productionize-shape-registration.md` — production registration alphabet
- `docs/handoffs/2026-05-27_lab-classifier-transfer-solve.md` — lab classifier transfer status

### Memory notes the next chat should read first
1. `project_shape_registration_clustering` — the full bake-off + 6/6 falsification record
2. `project_lab_cnn_classifier_scope` — `lab_classifier_v1` status (= α₃-C oracle)
3. `project_cleaning_stacks_three_distinct` (just rewritten) — cleaning canonical
4. `project_lab_transfer_v1_vs_dann_patchsweep` — DANN failed; classifier reliable at 0.81 lab balanced acc
5. `feedback_corpus_protocol_first` — corpus reading protocol

### Files that are FROZEN baselines (do not modify)
- `train_contour_vae_v2.py` (frozen baseline, archived consumer)
- `train_shape_vae_v3_hybrid.py` (in `archive/cleaning_legacy/stack3/scripts/experiments/`)
- `train_shape_encoder_contrastive.py` (in `archive/cleaning_legacy/stack3/scripts/experiments/`)

---

## Git state at handoff time

```
Uncommitted (this session's work):
  - 50 files moved to archive/cleaning_legacy/{stack1,stack3}/
  - archive/cleaning_legacy/README.md (new)
  - docs/modules/cleaning-subsystems.md (modified)
  - docs/plans/ROADMAP_SHAPE_INVARIANT_LATENT.md (new)
  - docs/reviews/ROADMAP_SHAPE_INVARIANT_LATENT-adversarial.md (new)
  - docs/handoffs/2026-05-28_alpha3-vocalmat-blocker-and-pivot.md (this file, new)
  - scripts/deepsqueak_focus_stft.py (canary banner)
  - src/usv_spectrogram/classifier/__init__.py (trimmed)

NOT in git, but exist on disk:
  - data/vocalmat/Mdl_categorical_DL.mat (212 MB — gitignored or to be gitignored)
  - $CLAUDE_JOB_DIR/alpha3_roadmap.html (job-scoped temp)
```

The branch is `main`. Last commit: `777131f3 docs(housekeeping): cleaning-subsystems module doc + repo-housekeeping handoff & triage` (from before this session).

---

## What was deliberately NOT done

- **Did NOT commit the archive operation.** Per user's cautious-commits stance (memory `feedback_no_bulk_stage_in_parallel_chats`), the next chat should confirm with the user before committing 50 file moves + the doc updates.
- **Did NOT write the .md canonical of `alpha3_roadmap.html`.** That was promised conditional on user approval; the HTML is the source of truth for now.
- **Did NOT delete `data/vocalmat/Mdl_categorical_DL.mat`** despite the decode failure. It's useful provenance and the next chat may revisit α₃-A if MATLAB shows up.

---

## End state — exactly where we paused

User said *"it doesn't matter that much, please give me a handoff so we continue from right here with another chat."*

The "doesn't matter that much" refers to the α₃-A/B/C choice — the user is signalling they don't care strongly which variant ships, just that we ship something. **Lean toward α₃-C + γ unless the next chat picks up a strong reason otherwise.**

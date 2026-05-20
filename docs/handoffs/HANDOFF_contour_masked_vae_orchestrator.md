# Handoff — Contour-Masked VAE Pipeline (Orchestrator Session)

You are the orchestrator for implementing `PLAN_contour_masked_vae_pipeline.md`. Read this entire handoff before doing anything. The plan itself is at the repo root — read it after this handoff.

## Your role

You are a **main session in orchestrator mode** (per the standing `feedback_orchestrator_mode` rule). That means:

- Non-trivial work (Phase 1 export, Phase 3 sweep, Phase 4 training, Phase 5 figures) is delegated to subagents via the `Agent` tool or `claude -p`.
- You decompose the plan into independent tasks, dispatch them, verify their results with a fresh agent, and only do the synthesis / decision steps yourself.
- You never blindly accept a subagent's "done" — always verify with a separate agent or a direct check before marking a phase complete.

You are NOT the implementer. You are the dispatcher + reviewer + decision-maker.

## Project one-paragraph context

This is the Mickey London Lab USV (ultrasonic vocalization) analysis pipeline for wild-mouse couples recorded at 300 kHz. A hard-negative-retrained CNN does call detection (`models/hard_neg_retrain/best_model.pt`, precision ~90.55%, manual-review tier 98.7%). DeepSqueak (MATLAB) is used as a *feature extractor only* — contour-based acoustic descriptors per call. Two existing parallel classifications live in `classified_detections_*.csv`: `Cluster_NN` (DeepSqueak k-means, 27 numbered clusters) and `syllable_type` (our rule-based Holy & Guo cascade, 7 types). The 5970 cohort (animal lmt_034, 5 sessions, ~8k accepted detections) is the reference. 3452 and 9252 are comparison cohorts. Two prior VAE runs (raw-spectrogram, DS-VAE and our own architecture, both completed 2026-05-18 — see `docs/handoffs/2026-05-18_vae_comparison_memo.md`) both found cohort-disjoint latent spaces driven by cage confound. The new pipeline aims to fix that by training the VAE on **contour-masked** patches rather than raw spectrograms.

## The plan

Read `PLAN_contour_masked_vae_pipeline.md` in full before starting Phase 1. Summary:

```
CNN detection → DeepSqueak contour extraction → 100ms windowing → contour mask → masked patch → VAE
```

Five phases:
1. Extract per-time-bin contour coordinates from DeepSqueak
2. 100 ms windowing (centered for short calls, moving window for long calls)
3. Contour-mask application with a visual bandwidth sweep
4. Convolutional VAE training (ELBO, 8–16 latent dims)
5. Latent-space analysis, UMAP visualization, cross-cohort encoding

## Refinements the previous Claude added (BLOCKING — apply before Phase 1)

These came out of a design review of the plan against the cage-confound evidence in the vault. They are not optional. Push back on any of them only if you have direct evidence the rationale is wrong.

### A. Phase 1 export must include tonality per bin, not just frequency

Schema: `call_id, time_bin_index, frequency_kHz, tonality`. The tonality column is what lets Phase 3 sweep tonality threshold and Phase 5 cohort-normalize. Without it, the mask is locked to whatever threshold DeepSqueak used internally (default 0.3), and threshold drift across cohorts becomes invisible.

### B. Phase 2 windowing: call-center-aligned, 50 ms step

- Short calls (< 100 ms): center the call at the patch midpoint. Position-symmetric zero padding.
- Long calls (> 100 ms): 50 ms step (50% overlap), NOT 10 ms (90% overlap). 10 ms gives correlated training examples that inflate apparent dataset size and risk overfitting per-step structure.
- Exact 100 ms: single window.

### C. Phase 3 sweep matrix: bandwidth × tonality threshold

The plan called for a 4-cell bandwidth sweep. Make it 4 × 3 = 12 cells: bandwidth ∈ {±2 kHz, ±5 kHz, ±10 kHz, σ=3 kHz Gaussian} × tonality ∈ {0.3, 0.4, 0.5}. Same 20 example calls per cell. This catches threshold-driven coverage drift before training.

### D. Phase 4 normalization: per-patch, full stop

Per-patch normalization strips amplitude information. Given the vault memory `feedback_rig_artifact_mean_power_db.md` (mean_power_db and tonality are cage artifacts), stripping amplitude is *desired*, not a side effect. Global normalization preserves cage signature in the VAE input. The plan said "try both" — replace that with "per-patch is the default; global normalization is reserved as a single ablation if per-patch fails."

### E. Phase 5 pre-registered cage-confound diagnostic

Before Phase 4 starts, write the diagnostic into Phase 5:

> A latent dimension z_K with |Cohen's d| > 1.5 between any two cohorts AND |r| > 0.4 with `principal_freq_hz` indicates that the cage signature jumped through the mask via narrowband artifacts at typical USV frequencies.

Pre-registering this rule prevents post-hoc rationalization of cohort separation as "biological signal."

### F. Phase 5 must include head-to-head comparison with the prior VAE memo

Table form, scored on (latent quality, cohort overlap, type separability):

| Model | Input | Latent quality | Cohort overlap | Type separation |
|---|---|---|---|---|
| DS-VAE (2026-05-18) | raw spec | … | cohort-disjoint, d≈+2.89 | … |
| Our VAE (2026-05-18) | raw spec | … | cohort-disjoint, d≈−2.72 | … |
| Contour-masked VAE (this plan) | masked spec | TBD | TBD | TBD |

Without this table the new pipeline is unmoored from the prior work.

### G. Hold cross-cohort encoding until within-cohort 5970 is verified clean

The plan listed cross-cohort encoding as a Phase 5 stretch goal. Make it explicitly blocked on "5970 within-cohort latent space passes the diagnostic in (E)." Trying to compare cohorts before knowing whether one cohort's latent space is morphology-driven is misattribution waiting to happen.

## Constraints that override your judgment

These are vault/memory constraints. Do not violate without explicit user instruction.

- **Corpus invariants.** `src/usv_spectrogram/corpus.py` is canonical for SR, USV band, STFT. Never redeclare; import. See `docs/modules/corpus-constants.md`. The corpus hook will fire on every edit that mentions canonical names — answer A/B/C/D each time.
- **No bulk staging.** `git add -A` and `git add .` are banned in parallel-chat workflows (see `feedback_no_bulk_stage_in_parallel_chats.md`). Stage every file by exact path.
- **Cage vs. rig terminology.** "rig" = compute; "cage" = recording chamber. Do not conflate when discussing acoustic confounds.
- **mean_power_db and tonality are cage artifacts.** Never present these as biological signal without cross-cage calibration.
- **Duration columns differ.** `call_length_s` (DeepSqueak tonal sweep) vs `det_duration_ms` (hysteresis event) — can differ by up to 10×. Pick the right one for the analysis. PNGs in the existing galleries show the hysteresis event, so visual-verdict filters use `det_duration_ms`.
- **Detection JSONs in `5970 USV/*_detections.json` are from an older CNN.** Probabilities don't match production model. Always re-score before captioning or filtering. Use `results/batch_5970_v2_full/` outputs for current-model detections.
- **No single WAV directory.** WAVs span `5970/`, `5970_reviewed/`, `5970 USV/`, `5970_manual_review/`, `5970_manual_review_reviewed/`. For 3452: `USV_3452_sample/`, `USV_3452_sample_reviewed/`. For 9252: `USV_9252/`. Use `--wav-search-dirs` with the full list, never assume one root.
- **Streamlit app is obsolete.** PyQt6 is the production app. Don't launch the Streamlit one for any reason.
- **Production CNN inference uses global MAD normalization** of the whole spectrogram, then crops windows. Per-window MAD silently destroys high-confidence USVs (`feedback_cnn_inference_global_mad.md`).
- **HIGH-risk files require `/kcheck` before modification.** Registry: `ops/vault-canary-map.md`. Production detection files (`scripts/run_batch_detection.py`, `app/core/sliding_inference.py`, `postprocessing/*`) are in this set.

## Reference paths the new session will need

- Plan: `PLAN_contour_masked_vae_pipeline.md`
- This handoff: `HANDOFF_contour_masked_vae_orchestrator.md`
- Prior VAE memo: `docs/handoffs/2026-05-18_vae_comparison_memo.md`
- Detection outputs (current model, 5970): `results/batch_5970_v2_full/`
- DeepSqueak feature CSVs: `classified_detections_full.csv` (5970), `classified_detections_3452.csv`, `classified_detections_9252.csv`
- DeepSqueak summary export script (existing): `scripts/deepsqueak_export_stats.m`
- DeepSqueak batch classifier: `scripts/deepsqueak_batch_classify.m`
- WAV roots for 5970: `5970_reviewed/`, `5970 USV/`, `5970/`, `5970_manual_review/`, `5970_manual_review_reviewed/`
- Traditional taxonomy gallery (recent): `.claude/worktrees/traditional-taxonomy-gallery/results/traditional_gallery/`
- Cluster gallery (older): `results/cluster_gallery/`
- Corpus facts: `data/corpus_facts/5970.json`, `data/corpus_facts/3452.json`, etc.
- Python venv: `.venv/bin/python`

## Suggested decomposition for subagent dispatch

These are the units of work you should hand to subagents. Each is small enough to verify independently.

| Phase | Task | Suggested agent | Verification |
|---|---|---|---|
| 1 | Audit existing CSVs and scripts for any contour export — does anything already dump `(time_bin, frequency, tonality)`? | `Explore` (very thorough) | Read the agent's findings, spot-check one CSV header yourself |
| 1 | If absent: write MATLAB exporter `scripts/deepsqueak_export_contours.m` and Python loader `scripts/load_deepsqueak_contours.py` | `general-purpose` | Run loader on one .mat, sanity-check 5 calls visually |
| 1 | Define output schema: parquet file at `results/contour_extraction/<dataset>/contours.parquet` with columns `wav_stem, call_id, time_bin_index, time_s, frequency_kHz, tonality, accepted` | (decision, you) | — |
| 2 | Write windowing script `scripts/window_calls_to_patches.py` per refinement B | `general-purpose` | `test-architect` writes tests for short/long/exact-100ms paths before implementation |
| 3 | Sweep script + 12-cell visual grid figure | `general-purpose` | Visual review with you, then user approval |
| 4 | VAE training script — keep architecture simple, ELBO, per-patch norm per refinement D | `general-purpose` | `dsp-reviewer` for the spectrogram I/O, `master-reviewer` for the overall module |
| 4 | Training run | (you launch via Bash background) | Spot-check 10 reconstructions |
| 5 | Latent-space analysis script + UMAP + diagnostic checks per refinement E | `general-purpose` | Independent agent re-runs the diagnostic; you cross-check the table in refinement F |

## Decisions still owed to the user before you can finish

Surface these explicitly when each phase is reached. Don't decide them silently.

1. **End of Phase 1**: confirm the parquet schema with the user. Adding columns later is cheap; renaming or reshaping is expensive.
2. **End of Phase 3**: present the 12-cell visual grid. The user picks bandwidth + tonality threshold. Do not pick for them.
3. **End of Phase 4**: present 20 reconstruction pairs (input | output). The user signs off on whether reconstruction is good enough to trust the latent space.
4. **End of Phase 5 if cage diagnostic fires**: present the diagnostic result. The user decides whether to revise the mask, change cohorts, or accept the confound and document it.

## Definition of done

The pipeline is done when:

1. `results/contour_extraction/5970/contours.parquet` exists and validates against the agreed schema.
2. `results/masked_patches/5970/` contains all training patches and their metadata.
3. `models/contour_vae_5970/best.pt` exists with documented hyperparameters.
4. `results/contour_vae_analysis/5970/` contains the UMAP figure, the diagnostic result, and the comparison table from refinement F.
5. A memo at `docs/handoffs/contour_masked_vae_memo.md` summarizes the outcome with explicit pass/fail on the cage-confound diagnostic.

Cross-cohort encoding (3452, 9252) is only attempted after (1)–(5) for 5970 pass.

## How to start

1. Read `PLAN_contour_masked_vae_pipeline.md`.
2. Read `docs/handoffs/2026-05-18_vae_comparison_memo.md` (the prior VAE work — this pipeline must beat it).
3. Run `git status` to see current branch state. If you are NOT in a worktree, create one named `contour-masked-vae-pipeline` via the EnterWorktree tool before any code changes.
4. Spawn an `Explore` agent (subagent_type: Explore, breadth: very thorough) with this prompt:

> Audit the USV pipeline for any existing per-time-bin contour export. Specifically: do any CSV files in `classified_detections_*.csv` or `results/` contain columns like `contour_time`, `contour_freq`, `time_bin`, `tonality_per_bin`? Do any MATLAB scripts in `scripts/` write per-bin contour arrays to disk (look for `ContourTime`, `ContourFreq` in .m files)? Do any `.mat` files in `results/` have those fields? Report findings as: (a) confirmed present at path X (b) absent — would need to be added. Under 400 words.

5. Based on Explore's findings, decide whether Phase 1 starts with extending the MATLAB export or with a Python `.mat` reader. Surface the decision to the user before writing code.

6. Use TaskCreate to track phases — one task per phase, with refinements A–G as subtasks where they apply.

## Things the previous Claude wants you to NOT do

- **Don't skip refinement A.** The tonality column will look unimportant until Phase 3, then it'll be the difference between a usable sweep and a hardcoded mask.
- **Don't write β-VAE before standard ELBO is working.** Disentanglement losses only help when the base VAE is already converging. They are not a noise fix.
- **Don't run cross-cohort encoding "to see what happens."** Wait until 5970 within-cohort passes the diagnostic. Otherwise you'll spend a week debugging a confound that the diagnostic would have flagged in an hour.
- **Don't trust subagent "done" reports without verification.** Especially for the MATLAB exporter — run the Python loader on the output and visually confirm 5 random calls' contours align with their spectrograms.
- **Don't write planning, decision, or analysis documents unless the user asks.** Work from conversation state and the plan file. Add to `docs/handoffs/contour_masked_vae_memo.md` only at the end.

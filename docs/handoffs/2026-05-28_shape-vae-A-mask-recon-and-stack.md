# HANDOFF — Pathway A follow-up: mask-recon test + (conditional) VAE1→VAE2 stack

**Date:** 2026-05-28  **Status:** PAUSED — read "STOP: bigger picture" first; this handoff only fires if the user decides to add a 6/6 closure data point after seeing the new evidence.

## STOP: bigger picture (new context the previous chat lacked)

While Pathway A was running, the parallel chat finished **Pathway B (contrastive with shift-augmentation)** and **also killed it** (held-out shape η² = 0.044, kill < 0.12). Their field-level summary, now in `project_shape_registration_clustering` memory:

> *5/5 falsified 2-D image-objective attempts (production 0.099, denoised 0.081, A 0.028, B 0.044, M10 0.009) vs 2 successful preprocessing approaches (registration 0.58–0.75, M8/M9 on registered ridge 0.42/0.34). Shape lives in the 1-D registered ridge, not 2-D pixel space. **VAE-family is closed for shape clustering**; production = registration → KMeans.*

**This dramatically lowers the prior on mask-recon + VAE-stack succeeding.** Pathway B was the *strongest alternative mechanism* — contrastive bypasses recon entirely, which is what the user's dot-pattern hypothesis worries about — and it still failed. Mask-recon and VAE-stacking are both still 2-D image-objective attempts in the same now-falsified family. The honest assessment: running them adds a 6/6 closure data point to a question that's effectively answered.

**Default recommendation: don't execute this handoff.** Accept the kill, ship registration (`models/shape_kmeans/k20.joblib` — already productionized). The user's dot-pattern intuition is correct *and refuted at the same time* — refuted because B's failure shows that even bypassing recon doesn't help; the binding constraint is "shape lives in 1-D, not 2-D pixels."

**Only execute if** the user explicitly wants the closure: "I want the 6/6 data point on the record before we close the file." Then proceed with the experiments below.
**Predecessor chain (read once for context, then operate on this doc):**
- Original spec: `docs/handoffs/2026-05-27_shape-vae-A-derivative-loss.md`
- Rig dispatch: `docs/handoffs/2026-05-27_shape-vae-A-rig-dispatch.md`
- Code review: `docs/reviews/shape-vae-v3-deriv-review.md`
- **The result that motivates THIS handoff** is summarized below — you do not need to re-execute prior phases.

## TL;DR — what's already done and what's left

**Done (do NOT redo):** Pathway A literal test ran end-to-end on the rig — code shipped + twice-reviewed + hardened, ridge targets extracted (`ridge_targets_v3.npz`, 154 MB, 100% active), model trained 60 epochs on GPU2 (`best.pt`, `latents.npy`, `val_idx.npy`), quick scorecard computed.

**Result:** **shape η² (held-out, n=6927) = 0.028** vs the **0.12 KILL gate** — far below; **even below the no-derivative dead-end's 0.081**. The handoff's explicit kill criterion is met. The ceiling sanity check (`shape_self_η² = 0.748`) confirms the eval is methodologically sound — the latent really does not encode shape, and the registration baseline (~0.75) is recovered when you cluster the *registered shapes themselves*.

**Mechanism the loss balance reveals:** at the trained config (`λ_d=1.0, β=0.1, τ=0.05`), `recon ≈ 380`, `β·kl ≈ 9.8`, `λ_d·deriv ≈ 0.82` — the derivative term contributed **<0.3% of total loss**. Too weak to re-weight the 32-D latent off the pitch/duration axes that pixel-variance naturally rewards. The handoff's suggested sweep `{0.3, 3.0}` is in the same too-small regime.

**User's hypothesis (the reason this follow-up exists):** the denoised patches may be sparse "dot patterns" (not smooth ribbons), so the VAE's pixel-BCE objective spends its latent capacity memorizing dot positions rather than encoding the smooth curve the dots approximate. This is a refined version of the dead-end memo's "pixel-variance axes" finding: the highest-entropy pixel-level thing happens to also be pitch/duration, which is why both diagnoses point at recon dominance.

## The experiment THIS handoff asks you to run

**Phase 1 — `--mask-recon` test (cheap, ~3h, one config knob).** The `--mask-recon` CLI flag already exists in `scripts/experiments/train_shape_vae_v3_deriv.py`. It weights the BCE recon loss by a per-pixel mask so the loss concentrates on the in-call region instead of the dot-vs-zero pattern over the whole patch.

*Caveat to resolve before launch:* the current implementation expects a `recon_weight_mask` tensor in the data path, but the dataset doesn't currently yield one. **You will need to either:** (a) add a simple binary mask built from `valid_mask` (broadcast to image shape) inside `_run_epoch`, or (b) construct it from `prefilter_spectrogram` output. Option (a) is faster and adequate for the diagnostic. Spec: a `(B,1,H,W)` float tensor that's 1.0 where the call ridge is active and 0.0 elsewhere, broadcast from the band-region active columns. Add a test before re-running.

The decision number is shape η² on held-out:

| Phase 1 outcome | Action |
|---|---|
| shape η² **< 0.10** (no movement) | The bottleneck is NOT dot-pattern capacity. The pitch/duration variance-dominance is the real story. **KILL Pathway A**, document the negative result, recommend Track A (registration). Skip Phase 2. |
| shape η² **0.10–0.20** (modest movement) | Dot-pattern hypothesis is partially right. Proceed to Phase 2 (VAE1→VAE2 stack) to see if a smarter smoother amplifies the effect. |
| shape η² **≥ 0.20** (clear movement) | Dot-pattern hypothesis is the leading explanation. Definitely proceed to Phase 2. |

**Phase 2 — VAE1→VAE2 stack (~3.5h, conditional on Phase 1 ≥ 0.10).** User's exact suggestion: *"the first VAE we created is pretty good at reconstructing the dots into something smooth, why not just use these outputs as the new inputs for the new VAE?"*

Operationally:
1. Identify which "first VAE" — most likely the **dead-end denoised retrain** at `/data/shachar/contour_vae/models/contour_vae_denoised/` (rig) or the production contour-VAE. Use whichever the user confirms produces visibly smooth ribbons on these denoised patches.
2. Forward-pass all 69,293 denoised patches through that VAE in eval mode (no grad). Save `x_recon` as a parallel `.npz` (~30 min, single GPU, no special tricks).
3. Train Pathway A's `train_shape_vae_v3_deriv.py` with the SAME hyperparams as the kill run BUT pointed at the smoothed-input npz (`--patches <new>.npz`). Keep `--ridge-cache` pointed at the existing `ridge_targets_v3.npz` (the targets are unchanged — they're derived from the TRUE ridges, not the inputs).
4. Run the same `quick_shape_eta2.py` eval on the new latents.

**Hard ceiling caveat:** VAE1's latent had shape η² ≈ 0.12 (production) or 0.081 (dead-end). Its reconstructions can carry only that much shape info forward — so VAE2's ceiling under this stack is approximately VAE1's ceiling. If Phase 2 clears 0.12 decisively, it's evidence that the dot-pattern reduction *combined with* the existing shape info preserved by VAE1 is enough. If it's still <0.12, the information loss in VAE1's encoder is the binding constraint.

## Reusable rig artifacts (do NOT rebuild)

All under `/data/shachar/contour_vae/`:

| Artifact | Path | Use |
|---|---|---|
| Denoised patches (16 GB) | `results/denoised_patches/combined_denoised/patches.npz` | Input corpus. **Note the path — the original handoff incorrectly stated `/data/mickey_london_lab/...`** |
| Ridge dF/dt targets (154 MB) | `results/denoised_patches/combined_denoised/ridge_targets_v3.npz` | Pathway A's derivative target. Reuse for both phases. |
| Pathway A trained model | `results/shape_vae_v3_deriv/run_ld1_beta0p1/best.pt` | The failed run; useful as a baseline comparison and for Phase 2 if used as VAE1. |
| Pathway A latents | `results/shape_vae_v3_deriv/run_ld1_beta0p1/latents.npy` | Already evaluated; shape η² = 0.028. |
| Held-out split | `results/shape_vae_v3_deriv/run_ld1_beta0p1/val_idx.npy` | **Reuse the SAME split** in Phase 1/2 so results are directly comparable. |
| Quick scorecard | `results/shape_vae_v3_deriv/run_ld1_beta0p1/quick_scorecard.json` | The kill-gate verdict for Phase 0. |
| Dead-end model (candidate VAE1) | `models/contour_vae_denoised/` | Phase 2 candidate. Confirm with user before using. |

**Quick-eval script (rig-side, copy this for every retrain):** `scripts/experiments/quick_shape_eta2.py` (already on rig + box). Builds shape labels from `ridge_targets_v3.npz` via cumsum → mean-subtract → KMeans-20, then computes `eta2(latents[val], labels[val])`. Methodologically equivalent to `rig_R2.register_one`. Print discipline already wired.

## Run commands (verbatim, just substitute `<RUN_TAG>`)

### Phase 1 — mask-recon (after adding the mask plumbing per the caveat above)
```bash
DENO=/data/shachar/contour_vae/results/denoised_patches/combined_denoised
OUT=/data/shachar/contour_vae/results/shape_vae_v3_deriv/run_mask_recon
LOG=/data/shachar/contour_vae/results/shape_vae_v3_train_mask_recon.log
ssh shachar@100.113.224.57 "
  cd /data/mickey_london_lab && mkdir -p ${OUT} && rm -f ${LOG} ${LOG%.log}.DONE
  nohup bash -c '
    PYTHONPATH=/data/mickey_london_lab/src .venv/bin/python -u scripts/experiments/train_shape_vae_v3_deriv.py \
      --patches ${DENO}/patches.npz --ridge-cache ${DENO}/ridge_targets_v3.npz --out ${OUT} \
      --lambda-d 1.0 --beta 0.1 --tau 0.05 --latent-dim 32 \
      --device cuda:\$FREEGPU --epochs 60 --batch 128 --workers 0 \
      --mask-recon > ${LOG} 2>&1
    echo \$? > ${LOG%.log}.DONE
  ' >/dev/null 2>&1 &"
```
Pick `$FREEGPU` from `nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | sort -t, -k2 -n | head -1`.

### Phase 2 — stacked VAE (only if Phase 1 ≥ 0.10)
1. Write `scripts/experiments/precompute_vae1_recon.py` (forward-pass VAE1 over patches, save `x_recon` npz with same key layout as `patches.npz`).
2. Rerun the Phase 1 command pointing `--patches` at the new `vae1_recon.npz` (drop `--mask-recon` for the first stack run; the smoothing is in the input now).

### Eval (both phases)
```bash
ssh shachar@100.113.224.57 "cd /data/mickey_london_lab && \
  PYTHONPATH=/data/mickey_london_lab/src .venv/bin/python scripts/experiments/quick_shape_eta2.py"
```
(Edit the script's `LATENTS`/`OUT` paths to point at the new run dir, or parameterize.)

## Operational gotchas (learned the hard way last session)

- **Compute is GATED** (`feedback_rig_claude_mediation`) — read-only SSH is pre-authorized, but launches need the user's per-session OK.
- **OOM trap:** `--workers 4` causes the 16 GB patches array to COW-leak via forked DataLoader workers → OOM during epoch 1. **Always use `--workers 0`** (matches the frozen baseline). The 16 GB stays as reclaimable page cache, anonymous RSS stays small.
- **Box↔rig Tailscale relay is flaky** (NAT-traversed, ~0.5 MB/s). Drops for minutes at a time. Use `nohup` + `.DONE` sentinel + a backgrounded polling script (`while sleep 90; do ssh rig 'cat $DONE'; done`) rather than blocking SSH for hours. Pattern lives in `$CLAUDE_JOB_DIR/poll_*.sh` from last session (template available).
- **GPU contention:** other users/processes appear and disappear on GPUs 0–3. **Always query `nvidia-smi` at launch to pick the free GPU**, don't hardcode `cuda:3`.
- **Print discipline (lab convention):** every eval run must print params, thresholds, sort keys, and row counts (`quick_shape_eta2.py` already does this).

## Files NOT to touch

🔒 `scripts/train_contour_vae_v2.py` (frozen baseline — `ImageVAE` is imported from it, never edited), production detection pipeline (`scripts/run_batch_detection.py`, `app/core/sliding_inference.py`, `postprocessing/`), `ExtractionConfig`, `corpus.py` constants (CNN-frozen), committed `models/`.

**Sibling-pathway files (`*_hybrid*`, `*_contrastive*`) belong to PARALLEL chats** — stage only Pathway-A files by exact path (`feedback_no_bulk_stage_in_parallel_chats`). The 2026-05-21 incident swept a stream-5 memo into an unrelated commit.

## Tests-as-spec (do NOT modify expectations)

- `tests/test_train_shape_vae_v3_deriv.py` (32 tests, written by `test-architect` BEFORE implementation)
- `tests/test_train_shape_vae_v3_deriv_hardening.py` (15 tests, adversarial edge cases)
- `tests/test_eval_shape_vae_v3.py` (9 + 5 rig-only)

If you add the recon-mask plumbing in `_run_epoch`, run `test-architect` for the new mask-construction helper BEFORE writing the code (`/implement` Step 0 — CLAUDE.md non-negotiable). The existing `TestRunEpochForwardPath` integration tests will catch any shape-mismatch from a wrongly-shaped mask.

## Kill criteria (avoid sunk cost)

- Phase 1 shape η² < 0.10 ⇒ **KILL Pathway A entirely.** The dot-pattern hypothesis is refuted; the pitch/duration variance-dominance is the binding constraint. Document the negative result; the answer is registration (Track A) or the sibling contrastive pathways (B / BA).
- Phase 1 ≥ 0.10 but Phase 2 < 0.12 ⇒ stack confirms direction but doesn't clear the gate. Report; do not ship.
- Phase 2 ≥ 0.12 *and* shows multi-component/jump capture registration lacks ⇒ ship a write-up and consider productionizing.

## Close

After execution, `/wrap-session` → HTML head-to-head: registration 0.58–0.75 | production 0.12 | dead-end 0.081 | **Pathway A literal 0.028** | Phase 1 result | Phase 2 result (if run). Include the `file://wsl.localhost/Ubuntu/<path>` URL in the final message. Update the project memory note `project_shape_registration_clustering.md` with the final verdict.

# PLAN — Shape Representation v2 (registration + derivative-aware contour-VAE retrain)

**Date:** 2026-05-25  **Owner thread:** shape-vs-pitch clustering (with Mickey)
**Status:** PLAN — purpose = MAIN REPRESENTATION (train everything). Gating decision moved upstream: the training PATCHES are impoverished (see Track 0) — pick a mask regime + regenerate BEFORE any retrain.

## Why this plan exists
The 2026-05-25 bake-off (11 representations, 67,337 true ridges) proved:
- Production contour-VAE K=20 clusters by **pitch (0.45) / duration (0.47)**, shape η² **0.12**.
- **Registration** (remove pitch/position/duration from the ridge) → shape η² **0.58–0.75**, beating every learned encoder. The clustering win is **preprocessing, not modeling**.
- Learned encoders all lost on shape clustering: M8 1-D VAE 0.50/0.42, M9 contrastive 0.34, M10 image-VAE **0.009**.

Full memo: memory `project_shape_registration_clustering.md`; report `results/latent_transitions/shape_registered/SESSION_REPORT_shape_clustering.html`.

**Key distinction this plan rests on:** clustering ≠ the latent's downstream jobs. The contour-VAE latent also feeds transition/idiom sequence analysis and lab-vs-wild comparison, where its **pitch/cage confound** is a known liability. A retrain can serve *that* even though it won't beat registration at clustering.

---

## Track 0 — REGENERATE TRAINING PATCHES (PREREQUISITE — patches confirmed impoverished)
**Finding (2026-05-25 audit + contact sheet `results/latent_transitions/patch_audit_5970.png`):** the current patches use a HARD ±5 kHz tube mask (`mask_kind="hard"`, `bandwidth_kHz=5` for ALL 69,293). Visually: ~95% black with a thin, faint, often broken ridge trace. No call body, no sub-harmonics (masked out), no harmonic stack. This is why the image VAE never beat the 1-D ridge and why sub-harmonic capture is impossible. **Retraining on these cannot yield a rich representation — regenerate first.**

**Pipeline knobs (exist already, in `worktree-contour-masked-vae-pipeline/scripts/`):**
- `mass_apply_contour_mask.py` — `--bandwidth-kHz` (currently 5), `--tonality-threshold` (currently 0).
- `contour_mask_utils.py` — `apply_hard_bandwidth_mask` (used) AND `apply_soft_gaussian_mask` (UNUSED, attenuates by distance from ridge).

**Options for the new "pictures" (DECISION NEEDED):**
| Option | What it keeps | Tradeoff |
|---|---|---|
| Wider hard mask (±15–25 kHz) | ridge + call body + near sub-harmonics | simple knob; still cuts far harmonics |
| Soft Gaussian mask | graded structure, faint detail | smoother; partial noise/cage reentry |
| Unmasked USV-band patch | EVERYTHING (body, harmonics, noise) | richest, but reintroduces the cage confound we fight elsewhere |
| Better contour extraction first | fix the broken/faint traces | revisit focus-STFT params + tonality gate (traces are dotty) |

**RESOLVED (2026-05-25, user via the c06 three-panel `original/denoised/TRUE-VAE-input`):** the contour mask is the root problem — even a STRONG call (#23, band-max 11.4) is reduced to **0.21% nonzero** as VAE input, while the **denoised "SIS prefilter"** panel shows the call crisply. **Decision: regenerate training data from the denoised spectrogram (drop the contour mask).** Recipe: per call window → STFT → band-crop → `features/spectrogram_filter.py::prefilter_spectrogram` (the same SIS prefilter that feeds the ridge tracker) → that denoised band patch IS the new VAE input. This is the likely root-cause fix for the whole "clustering ignores shape" thread — the VAE never actually saw the calls.

**Gate:** pick the mask regime → regenerate all 4 cohorts on the rig → re-audit visually → THEN Track B retrain on good pictures. Richness-vs-cage-confound is the core tradeoff.

## Track A — Productionize registration (PROVEN; do this regardless)
The cheap, proven win. Rebuild the shape alphabet + transition/idiom analysis on registered ridges instead of raw latents.
- Input: `results/latent_transitions/shape_registered_TRUE/true_registered_ridges.npz` (rig).
- Detail handoff: `docs/handoffs/2026-05-25_productionize-shape-registration.md`.
- Eval: compare transition MI / entropy vs latent-based; expect the pitch confound to drop out.

## Track B — Derivative/edge-aware contour-VAE retrain (the "either way" idea, designed properly)
The faithful version of Mickey's proposal — NOT yet cleanly tested (M8 was 1-D-on-registered; M10 was a small from-scratch image VAE).
- **Base:** the production `scripts/train_contour_vae_v2.py` (ImageVAE, 256², z=32), retrained on the real combined `patches.npz`.
- **Loss additions (the idea):**
  1. **Derivative/edge term** — Sobel/finite-diff on the reconstruction, matching the contour's dF/dt (slope) and d²F/dt² (curvature). Weight it heavily relative to raw pixel MSE.
  2. **Background-masked reconstruction** — exclude the zeroed off-ridge region from the objective (you said: NOT about getting the black right). Weight loss by the contour mask.
  3. **Pitch + time-position invariance** — random vertical (pitch) shift + horizontal (onset) shift augmentation of input patches, so the encoder *learns* the invariance (alternative to pre-registering the input).
- **Eval (depends on PURPOSE — see decision below):**
  - shape-clustering: K-means on new latent → shape η² vs 0.12 baseline (won't beat 0.58 registration, but should clear 0.12).
  - cage-invariance: cohort η² / cage-correlation of new latent vs current (the real payoff).
  - sub-harmonic: qualitative — does the latent separate sub-harmonic/multi-component calls?
- **Learn from the failures:** M10 (edge-loss image VAE) scored 0.009 on shape — likely KL over-smoothing + aggressive image-crop registration mangling structure. Mitigations: lower β, larger z, masked loss, augmentation-instead-of-crop.

## Track C — Image VAE for sub-harmonics (the one thing ridges discard)
Only if multi-component / sub-harmonic structure (the c07.2 "biological prize") is a target. Retrain with an objective that explicitly rewards reconstructing the 2-D harmonic stack, not gross layout. Distinct from Track B's contour-derivative focus.

## Track D — Continuum vs discrete (open question, cross-cutting)
UMAP→HDBSCAN found a **continuum**, not crisp clusters, even on registered shapes. The honest end-state may be a navigable 2-D shape-map rather than N "letters." Decide before over-investing in any K=20 alphabet (affects Track A and B eval).

---

## THE DECISION THAT GATES TRACK B (and the loss/eval design)
**What is the VAE retrain primarily FOR?**
| Purpose | Loss emphasis | Eval bar | Verdict from bake-off |
|---|---|---|---|
| Shape clustering | n/a | shape η² > 0.58 | DON'T retrain — registration already wins |
| Better downstream latent (sequences, **cage-invariance**) | derivative + masked + invariance-augmentation | cohort/cage η² ↓ vs current; shape η² ≫ 0.12 | RETRAIN — payoff is the de-confounded latent |
| Sub-harmonic / multi-component capture | 2-D harmonic reconstruction | qualitative separation of sub-harmonic calls | Track C, different loss |

Mickey's "do the derivative either way" = include the derivative term in **Track B**, justified by the downstream-latent / cage-invariance purpose. Confirm that's the purpose before we design the loss + eval.

## Files to touch / NOT touch
- Touch: `latent-analysis-b-a-c` worktree + rig `/data/shachar/contour_vae`. New script (e.g. `train_contour_vae_v3_deriv.py`) — do NOT overwrite `train_contour_vae_v2.py`.
- DO NOT touch: production detection pipeline, `ExtractionConfig`, `corpus.py` constants (CNN-frozen).
- Compute: rig (GPU0 free; ~50× faster than box). Box is 12-core/11 GiB and OOM-crashed WSL under load — keep heavy work on the rig.

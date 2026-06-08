# Module: Contrastive shape encoder (Pathway B)

**Files:** `scripts/experiments/train_shape_encoder_contrastive.py`, `scripts/eval_shape_encoder.py`
**Tests:** `tests/test_shape_encoder_contrastive.py`, `tests/test_eval_shape_encoder.py`, `tests/test_shape_encoder_hardening.py` (53 total)
**Handoff:** `docs/handoffs/2026-05-27_shape-vae-B-contrastive-invariance.md` · **Plan:** `PLAN_geometric_shape_clustering_vae.md` §3 (B-contrastive) / §4 (gates)
**Review:** `docs/reviews/shape-encoder-contrastive-review.md`

## Purpose
Learn an unsupervised embedding in which USVs that are **geometrically similar** (chevron↔chevron,
jump↔jump) are neighbours, **invariant to absolute pitch and temporal position**. This is the one
VAE-family idea not yet falsified for shape clustering: a **contrastive** objective has *no
reconstruction term*, so — unlike the four prior reconstruction attempts — it never spends capacity
representing pitch/position pixels, and pitch+time-shift augmentation actively forces that invariance.

Leaderboard it must beat (shape η², registered-ridge): registration 0.58–0.75 (ceiling) ·
M9 1-D contrastive 0.344 · production contour-VAE 0.099 · denoised retrain 0.081 · M10 image-VAE 0.009.

## Design (encoder-only — NO decoder, NO KL, NO reconstruction)
- **`ContrastiveEncoder(embed_dim=128, proj_dim=64)`** — 2-D conv stack `1→32→64→128→256`
  (`Conv2d(k=4,s=2,p=1)+BN+LeakyReLU`, reusing the `ImageVAE` backbone) → `AdaptiveAvgPool2d(1)`
  (size-agnostic; avoids M10's destructive crop/resize) → `Linear→embed_dim`; SimCLR projection head
  `Linear→ReLU→Linear→proj_dim`. `forward(x) → (embedding, projection)`. **Cluster on the embedding;
  contrast on the projection** (projection discarded at inference).
- **`nt_xent_loss(z1, z2, tau=0.2)`** — standard SimCLR NT-Xent; numerically identical to M9's `ntxent`
  (keeps the leaderboard comparison apples-to-apples).
- **`augment(x, freqs_khz, *, max_df_khz=15, max_dt_frames=20, warp_lo=0.9, warp_hi=1.1, generator=None)`**
  — the invariance, applied via one `grid_sample(padding_mode="zeros")`:
  - vertical **pitch shift** (integer rows from Δf via the row freq-resolution),
  - horizontal **time shift** (integer frames),
  - fractional **time-warp** 0.9–1.1× (duration partial-invariance; the user opted in).
  - **In-band clamp:** the per-sample vertical shift is bounded by the call's own energy span so the
    call stays within the corpus **20–120 kHz** band (patches are the full 0–150 kHz STFT, so the band
    is a sub-range, rows ~35–204). Out-of-band/empty patches fall back to the full frame `[0,H)`
    (no-clip guarantee). Verified on real patches: 50×5 augmentations all stayed in band.
  - The positive pair = two independent `augment()` views of the same patch (their only systematic
    difference is pitch/position/duration → that is what the encoder learns to ignore).

## Data & compute (rig)
- Input: `results/denoised_patches/combined_denoised/patches.npz` (69,293 × 257 × 234 f32) + `freqs_kHz`.
  Per-patch max-normalised to [0,1] at load. Canonical rig root `/data/shachar/contour_vae` (NOT
  `/data/mickey_london_lab`, which only holds the code mirror + `src`).
- Train/val split saved to `split_idx.npz` (seeded) so eval scores the *same* held-out rows.
- **`--ckpt-every N`**: every N epochs, encode all patches → overwrite `embeddings.npy` + `encoder.pt`
  + quick scorecard. Added after the rig OOM-killed two runs (RAM contention with the sibling Pathway-A
  VAE); a mid-run kill now still leaves an eval-ready checkpoint. Loss plateaus by ~epoch 20.
- RAM note: the 16 GB patch corpus + a concurrent 17 GB Pathway-A job exceed the rig's 31 GB → run B
  alone (wait for A) or it gets OOM-killed.

## Evaluation — `eval_shape_encoder.py`
Reuses the cached registered-ridge descriptors `results/eval_shape/desc_denoised.npz`
(`row, pitch, shapes[N,50], duration, jump`) — **no ridge re-extraction, no WAV access**. KMeans(20)
on the held-out embeddings; gates:
1. **k-NN purity** (`knn_purity`, k=10) — most literal chevron↔chevron test.
2. **CV-NMI** vs the chevron/valley heuristic on the registered ridge (there is **no `syllable_type`
   column** in `classified_detections`, only `label`; the heuristic is what M9/M10 used → comparable).
   Beat production 0.04; target >0.20.
3. **shape η²** (`eta2` on registered ridge) — kill <0.12; target ≥0.50; stretch ≥0.58.
4. **pitch / duration η² LOW** — the direct invariance test (production VAE's 0.45 pitch η² was the failure).
5. **jump/curvature** capture (qualitative) + **UMAP** coloured by shape type.

Kill criteria (handoff): shape η²<0.12 AND purity≈chance ⇒ KILL, ship registration. Beats/matches
registration OR captures jumps registration can't ⇒ the win.

## Do-NOT-touch / conventions
Imports corpus constants (`USV_FREQ_MIN_HZ/MAX_HZ`) — never redeclares. Does not modify
`train_contour_vae_v2.py`, `ExtractionConfig`, `corpus.py`, or the detection pipeline. Prints
params/thresholds/sort keys/row counts on every eval run (lab convention).

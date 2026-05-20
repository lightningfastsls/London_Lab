# Path C — Cluster execution handoff

**Date:** 2026-05-19
**Status:** COMPLETE — Phase 5 verdict CLEAN (2026-05-20). All 5 steps executed. See `2026-05-20_path-c-cleanup.md` for residual cleanup.
**Blocked on:** Nothing.

## Status update — 2026-05-19 17:XX

- **Tailscale restored.** The rig is the preferred Step-1 target again; ELSC HUJI remains a documented fallback. Pipeline scripts and sbatch template are unchanged.
- **Step 0 labeled (round 1).** User labeled 23/24 grid panels. Clustered failure detected: 13 of 14 B-panels (~93%) showed an "ours-down" vertical offset; symmetric MATLAB-up signal on 5 of 6 A-panels. Decision gate triggered.
- **Root cause identified.** `scripts/deepsqueak_focus_stft.py` line 491 used `call_box.freq_start_kHz` as the freq offset — same value MATLAB adds — but MATLAB's `ridgeFreq` is 1-based while Python's `argmax` is 0-based. Net effect: every contour bin was reported one FFT-bin width below the corresponding MATLAB bin (~0.13–0.66 kHz depending on adaptive nfft). Measured median MATLAB-Python diff on a sample call: **0.658 kHz**, matching the predicted one-bin offset within rounding.
- **Fix applied.** Line 491 now uses `focus.fr_hz[focus.freq_lo_idx] / 1000.0` (the physical center of the first kept FFT bin). 23/23 unit tests in `tests/test_deepsqueak_focus_stft.py` still pass after the change — this is a test-coverage gap (the existing tests don't lock down absolute physical freq), to be patched with a regression test after Step 0 clears.
- **Re-extract complete.** Pre-fix 5970_focus parquet preserved at `results/contour_extraction/5970_focus/contours.parquet.prefix_bak`. Re-extraction used `--mat-root /home/shachar/projects/mickey_london_lab/results/deepsqueak_independent/mat_files` (the 55 .mat files covering the MATLAB-comparable subset). Post-fix headline stats: median per-call parity **0.980** (was 0.973 pre-fix), median MATLAB−Python freq offset **+0.376 kHz** (was ~0.65 — almost halved), density_ratio 1.000.
- **Step 0 CLEARED (round 2 labeling).** User reviewed `results/path_c_comparison/index.html` (post-fix) and reported "23 of 24 grid panels great, panel (1,1) borderline on a hard call, overall ready." Per handoff line 122 user-override branch: proceed to Step 1. Down-cluster dissolved; residual disagreements are human-eyeball noise floor, not systematic bias.

## Step 0 → Step 1 transition (this is where the next session picks up)

1. **Add regression test for the freq-offset bug** — `tests/test_deepsqueak_focus_stft.py` did not catch this (23/23 tests passed both pre- and post-fix). Add a synthetic-pure-tone test that asserts extracted freq_kHz median is within ½ FFT bin of the known tone. Without this, the bug could regress silently again.
2. **Document the fix** in either `notes/` (vault, via inbox→/reduce) or a short standalone `docs/handoffs/2026-05-19_path-c-freq-offset-fix.md`. Key facts: line 491 of `scripts/deepsqueak_focus_stft.py`; 1-based-MATLAB vs 0-based-Python convention mismatch; symmetric: MATLAB still biased ~½ bin UP relative to true ridge; our fix puts Python on the bin centers.
3. **Pivot Step 1 target from ELSC to rig.** Tailscale restored. Same pipeline scripts (`assemble_combined_patches.py`, `train_contour_vae_v2.py`, `phase5_cross_cohort.py`) — only the deployment commands change.
4. **Re-run the OTHER 3 cohorts' contour extractions with the fix** before Step 1's `assemble_combined_patches.py`. The 3452, 9252, and lab_131204 patches in `results/masked_patches/*_focus/` were generated with the buggy freq formula and would carry the down-bias into VAE training. Either re-extract each, or re-mask using the corrected contours.

The remainder of this handoff (Steps 1–5) is unchanged; only the Step 0 decision and the deployment-target preference have moved.

## TL;DR

The Python port of DeepSqueak's `CreateFocusSpectrogram` + `CalculateStats` is **complete and validated** (median per-call density ratio = 1.000, median per-call parity = 97.3% vs MATLAB reference). All 4 cohorts (5970, 3452, 9252, lab_131204) have been extracted, windowed, and masked locally — 69,293 patches across **16.3 GB**. The remaining work is:

1. Ship 16.3 GB of patches + scripts to the ELSC HUJI cluster
2. Run `assemble_combined_patches.py` to concat the 4 cohorts into one dataset
3. Train VAE from scratch (hyperparameters match the 5970 methodology-test run exactly)
4. Run `phase5_cross_cohort.py` for the cross-cohort cage-confound diagnostic
5. SCP results back to the worktree; surface HTML report

The original `result:` deliverable for this whole effort is the **Phase 5 cross-cohort verdict** (CLEAN or FIRES) — that's the answer to whether the contour-masked VAE actually removes cage signatures across cohorts.

---

## ⚠ STEP 0 — Read user's human-labeled disagreements BEFORE shipping

In the wrap-session that birthed this handoff, the user reviewed `results/path_c_comparison/index.html` and noted: **"in most pictures DS is better, there are some where we are better."** This is a real human observation that the parity statistics don't capture. The user agreed to leave panel-by-panel labels (filled in below before the next session starts).

### Decision gate based on user labels

| Pattern in user labels | Action before shipping |
|---|---|
| **Mostly C (tie) and A (Python better)** with a few scattered B's | No fix needed. Statistics tell the true story; the human eye picked up sampling noise in the worst-disagreement subset. Proceed to Step 1 (SCP). |
| **B cases cluster on one reason** (e.g. ≥ 60% of B's are `edge` or `noise` or `gap`) | **Stop. Apply a targeted fix to `scripts/deepsqueak_focus_stft.py` before shipping.** The taxonomy → fix mapping is below. Re-run `scripts/extract_contours_python.py` on 5970, regenerate the deep comparison, re-check. Only ship after the systematic pattern is gone. |
| **B cases dominate but reasons are scattered** | Investigate manually. The port may have a subtle bug not in the obvious failure modes. Don't ship until understood. |
| **Many A cases (Python better than MATLAB)** | Scientifically interesting. Document in `notes/2026-XX-XX_python-improves-on-deepsqueak.md` and proceed — preserve our improvement rather than try to match MATLAB's worse behavior. |

### Failure mode → algorithmic fix mapping

If user labels show a clustered B pattern, the fix lives in `scripts/deepsqueak_focus_stft.py`:

| Reason | Most likely root cause | Code location |
|---|---|---|
| `edge` (misses ridge at start/end) | rlowess tail effects amplified by short `ridge_time_col_idx` array; or audio crop boundary | `_rlowess_smooth()` lines 150-180 + `create_focus_spectrogram()` audio crop lines 165-175 |
| `noise` (includes spurious bins MATLAB skips) | `MAX_LOWERING_ITERS=10` too permissive; or `_spectral_flatness` epsilon too small | `calculate_stats()` lowering loop lines 240-260 |
| `gap` (holes in continuous ridge) | rlowess `span=0.025` too aggressive for medium calls; or amplitude threshold cutting columns mid-ridge | `DS_RIDGE_SMOOTH_SPAN` at line 38 |
| `jagged` (bumpy vs smooth) | rlowess `n_iter=5` insufficient; or wrong span for the call length | `DS_RLOWESS_ITERATIONS` at line 38 |
| `wrong` (latched onto noise band) | Defensive `MIN_FREQ_RANGE_KHZ=1.0` floor pushing calls into wider windows than intended | `MIN_FREQ_RANGE_KHZ` at line 60 |
| `other` | Read user's free-text note + look at the specific failing call in `worst_disagreements.png` |

---

## Panel coordinate map — for the user to fill in labels

**Legend** — for each panel, pick one letter and (optionally) a reason for `B`:
- **A** = Python better · **B** = MATLAB better · **C** = tie / both fine · **D** = both wrong
- Reasons (for B only): `edge`, `noise`, `gap`, `jagged`, `wrong`, `other: <5 words>`

### `grid.png` — 24-panel grid (4 cols × 6 rows). Top-left is row 1 col 1; reading order is left-to-right, top-to-bottom.

| row | col | wav_stem | call_id | dur(ms) | n_ref | n_py | parity | **YOUR LABEL** |
|----:|----:|----------|--------:|--------:|------:|-----:|-------:|:--------------:|
| 1   | 1   | 2024-09-30_17-32-17_0001844 | 7 | 9 | 8 | 8 | 100% | _____ |
| 1   | 2   | 2024-09-30_20-59-17_0002978 | 5 | 11 | 14 | 15 | 100% | _____ |
| 1   | 3   | 2024-09-30_20-12-53_0002767 | 9 | 22 | 30 | 27 | 100% | _____ |
| 1   | 4   | 2024-09-30_18-00-21_0002060 | 2 | 29 | 30 | 30 | 100% | _____ |
| 2   | 1   | 2024-09-30_18-14-18_0002160 | 13 | 31 | 31 | 31 | 100% | _____ |
| 2   | 2   | 2024-09-30_18-14-18_0002160 | 10 | 32 | 13 | 8 | 38% | _____ |
| 2   | 3   | 2024-09-30_11-23-03_0000067 | 1 | 36 | 10 | 12 | 100% | _____ |
| 2   | 4   | 2024-10-01_18-53-59_0006239 | 4 | 38 | 12 | 17 | 75% | _____ |
| 3   | 1   | 2024-09-30_12-33-18_0000567 | 5 | 41 | 42 | 42 | 100% | _____ |
| 3   | 2   | 2024-09-30_21-01-25_0002999 | 6 | 43 | 51 | 50 | 100% | _____ |
| 3   | 3   | 2024-09-30_19-17-23_0002502 | 3 | 51 | 55 | 50 | 91% | _____ |
| 3   | 4   | 2024-09-30_11-27-49_0000130 | 1 | 52 | 17 | 17 | 88% | _____ |
| 4   | 1   | 2024-09-30_16-03-58_0001363 | 7 | 56 | 64 | 64 | 100% | _____ |
| 4   | 2   | 2024-09-30_17-32-17_0001844 | 4 | 56 | 83 | 83 | 100% | _____ |
| 4   | 3   | 2024-09-30_17-24-04_0001792 | 9 | 65 | 71 | 71 | 99% | _____ |
| 4   | 4   | 2024-09-30_20-19-32_0002790 | 2 | 69 | 113 | 111 | 97% | _____ |
| 5   | 1   | 2024-09-30_12-33-18_0000567 | 12 | 79 | 126 | 126 | 90% | _____ |
| 5   | 2   | 2024-09-30_18-14-18_0002160 | 1 | 84 | 78 | 78 | 99% | _____ |
| 5   | 3   | 2024-09-30_20-12-53_0002767 | 7 | 97 | 82 | 82 | 94% | _____ |
| 5   | 4   | 2024-09-30_17-24-04_0001792 | 18 | 118 | 103 | 103 | 99% | _____ |
| 6   | 1   | 2024-09-30_19-43-29_0002661 | 7 | 129 | 78 | 78 | 96% | _____ |
| 6   | 2   | 2024-09-30_12-33-18_0000567 | 9 | 175 | 205 | 204 | 94% | _____ |
| 6   | 3   | 2024-09-30_20-12-53_0002767 | 3 | 186 | 159 | 159 | 96% | _____ |
| 6   | 4   | 2024-09-30_12-33-18_0000567 | 11 | 191 | 144 | 142 | 97% | _____ |

### `worst_disagreements.png` — 5 panels left-to-right (1..5). The 5 lowest-parity calls; expect these to skew B.

| col | wav_stem | call_id | dur(ms) | n_ref | n_py | parity | **YOUR LABEL** |
|----:|----------|--------:|--------:|------:|-----:|-------:|:--------------:|
| 1   | 2024-09-30_18-14-18_0002160 | 10 | 32 | 13 | 8 | 38% | _____ |
| 2   | 2024-09-30_11-32-10_0000176 | 14 | 9 | 6 | 11 | 67% | _____ |
| 3   | 2024-09-30_20-19-32_0002790 | 3 | 126 | 160 | 159 | 74% | _____ |
| 4   | 2024-10-01_18-53-59_0006239 | 4 | 38 | 12 | 17 | 75% | _____ |
| 5   | 2024-09-30_16-42-24_0001505 | 1 | 405 | 346 | 346 | 77% | _____ |

### How to leave labels (any of these formats work)

**Format 1 — table fill-in.** Edit this file, replace `_____` with your label.

**Format 2 — pasted list.** In the next chat, paste something like:

```
grid:
  (1,1)=C  (1,2)=C  (1,3)=C  (1,4)=C
  (2,1)=C  (2,2)=B(edge)  (2,3)=C  (2,4)=B(noise)
  (3,1)=A  ...etc...

worst:
  1=B(edge)
  2=B(jagged)
  3=B(noise)
  4=D
  5=A
```

**Format 3 — free text.** "Most panels are fine, the edge ones at rows 2/4 are where we miss MATLAB" — I'll parse it.

### What the next session does FIRST

1. Read these labels (whatever format).
2. Tally: count of A, B (by reason), C, D.
3. Hit the decision gate above.
4. If a fix is needed: apply it to `deepsqueak_focus_stft.py`, regenerate **5970-only** contours + deep comparison, ask user to re-validate the labeled panels, iterate.
5. Only after labels look acceptable (or user explicitly says "ship it anyway"): proceed to Step 1 (SCP bundle).

---

## Cluster: ELSC HUJI

**SSH:** `ssh shachar.levysahar@loginserver.elsc.huji.ac.il`

**What we don't know yet** (discover with the first SSH session):
- GPU partition names (run `sinfo` once logged in — look for partitions with `gres/gpu` in the GRES column)
- Module system layout (run `module avail python` and `module avail cuda` — typical ELSC has `python/3.10` or `python/3.11` plus `cuda/12.x`)
- Per-user storage quota and the path to lab/group storage (`quota -s -u` and check `/ems/elsc-labs/` or `/labs/<lab>/`)
- Whether sshd allows `ProxyJump`/`ssh-copy-id` (usually yes — needed for SCP)

**File locations to choose** (placeholders to fill in after first login):
- `$HUJI_HOME` — typically `/ems/elsc-labs/<lab>/<user>` or `~/` if not part of a lab group
- `$HUJI_DATA` — where the 16 GB patches will land. Needs ≥ 30 GB free (16 GB patches + ~16 GB combined intermediate)

---

## Step 1 — Ship the bundle to the cluster

From this worktree (`/home/shachar/projects/mickey_london_lab/.claude/worktrees/contour-masked-vae-pipeline/`):

```bash
# Pipe-stream tar over ssh (no intermediate file; halves local I/O)
{
  tar cf - \
      scripts/assemble_combined_patches.py \
      scripts/train_contour_vae_v2.py \
      scripts/phase5_cross_cohort.py \
      scripts/cage_confound_diagnostic.py \
      scripts/deepsqueak_focus_stft.py \
      src/usv_spectrogram/corpus.py \
      classified_detections_full.csv \
      results/masked_patches/5970_focus \
      results/masked_patches/3452_focus \
      results/masked_patches/9252_focus \
      results/masked_patches/lab_131204_focus
  tar cf - -C /home/shachar/projects/mickey_london_lab \
      classified_detections_3452.csv \
      classified_detections_9252.csv \
      classified_detections_lab_131204_clean.csv
} | ssh shachar.levysahar@loginserver.elsc.huji.ac.il \
      "mkdir -p \$HUJI_DATA/contour_vae && cd \$HUJI_DATA/contour_vae && tar xf -; tar xf -"
```

**Expected time:** 16 GB / Tailscale-equivalent throughput. If the cluster has 1 Gb/s WAN: ~3 min. If 100 Mb/s: ~20 min.

**Sanity verify on cluster:**

```bash
ssh shachar.levysahar@loginserver.elsc.huji.ac.il
cd $HUJI_DATA/contour_vae
ls -la results/masked_patches/{5970,3452,9252,lab_131204}_focus/
du -sh results/masked_patches/*_focus
# Expect: 2.85G  3452: 93M  9252: 133M  lab: 12.82G
```

---

## Step 2 — Cluster environment setup

ELSC clusters typically use `module load` + a per-user venv. First-time setup on the cluster:

```bash
ssh shachar.levysahar@loginserver.elsc.huji.ac.il

# Discover available toolchain
module avail 2>&1 | grep -iE "(python|cuda|cudnn)" | head -20

# Typical incantation (adjust versions to what's actually available):
module load python/3.11 cuda/12.6 cudnn

# Create a venv in $HUJI_DATA (NOT home — home has small quota)
python -m venv $HUJI_DATA/venv
source $HUJI_DATA/venv/bin/activate

# Install pinned packages. Versions match what runs locally.
pip install \
    'torch==2.11.0' \
    'numpy==2.3.5' \
    'pandas==2.3.3' \
    'scipy==1.16.3' \
    'librosa==0.11.0' \
    'pyarrow==22.0.0' \
    'tqdm==4.67.1' \
    'statsmodels==0.14.6' \
    'umap-learn' \
    'matplotlib' \
    'h5py' \
    'mat73'
```

> **Note**: `statsmodels` is only needed if re-running contour extraction (already done locally). `mat73` is only needed if re-running parity validation against `.mat` files (already done). For just `assemble + train + Phase 5` you can skip both.

---

## Step 3 — Slurm sbatch script

Write `$HUJI_DATA/contour_vae/run_full_pipeline.sbatch`:

```bash
#!/bin/bash
#SBATCH --job-name=contour_vae_combined
#SBATCH --partition=<GPU_PARTITION>       # FILL IN — find via `sinfo`
#SBATCH --gres=gpu:1                       # one GPU is plenty; batch size 32
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                          # 16 GB patches + 16 GB headroom
#SBATCH --time=04:00:00                    # 4 h: 5 min assemble + 1-2 h train + 30 min Phase 5 + slack
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
cd $HUJI_DATA/contour_vae

# Load environment
module load python/3.11 cuda/12.6 cudnn 2>/dev/null || true
source $HUJI_DATA/venv/bin/activate

# Step A — assemble combined patches.npz (CPU only, ~5 min, 16 GB output)
mkdir -p results/masked_patches/combined_all_cohorts
python scripts/assemble_combined_patches.py \
    --output-dir results/masked_patches/combined_all_cohorts

# Step B — train VAE from scratch (1 GPU, ~1-2 h)
mkdir -p models/contour_vae_combined results/contour_vae_combined
python scripts/train_contour_vae_v2.py \
    --patches-npz results/masked_patches/combined_all_cohorts/patches.npz \
    --manifest-parquet results/masked_patches/combined_all_cohorts/patches_manifest.parquet \
    --output-model-dir models/contour_vae_combined \
    --output-results-dir results/contour_vae_combined \
    --latent-dim 32 --batch-size 32 --lr 2.5e-4 \
    --max-epochs 500 --patience 50 --seed 42 --beta 1.0

# Step C — Phase 5 cross-cohort cage-confound diagnostic (CPU, ~10 min)
mkdir -p results/phase5_cross_cohort
python scripts/phase5_cross_cohort.py \
    --latents-parquet results/contour_vae_combined/latents.parquet \
    --manifest-parquet results/masked_patches/combined_all_cohorts/patches_manifest.parquet \
    --classified-csv-by-cohort \
        5970=classified_detections_full.csv \
        3452=classified_detections_3452.csv \
        9252=classified_detections_9252.csv \
        lab_131204=classified_detections_lab_131204_clean.csv \
    --output-dir results/phase5_cross_cohort

echo "DONE at $(date)"
```

Submit:

```bash
mkdir -p logs
sbatch run_full_pipeline.sbatch
# Note the job ID. Track with:
squeue -u $USER
tail -f logs/contour_vae_combined_<JOBID>.out
```

---

## Step 4 — Phase 5 verdict interpretation (decision gate)

The pre-registered diagnostic ([scripts/cage_confound_diagnostic.py](../../scripts/cage_confound_diagnostic.py)) uses the locked rule:

> A latent dimension `z_K` FIRES if BOTH:
> `|Cohen's d| > 1.5` between any pair of cohorts AND `|Pearson r vs principal_freq_hz| > 0.4`

| Outcome | Action |
|---|---|
| **CLEAN** (no dim fires) | Contour mask successfully removes cage signature even cross-cohort. Write `notes/2026-05-19_path-c-cleared-cross-cohort.md` recording the verdict. Phase 5 is the project's headline result. |
| **FIRES (1–3 dims)** | Partial cage leak. Investigate which cohort pair is driving (`per_dim_diagnostic.csv`). Likely action: tighten mask (try 2 kHz instead of 5 kHz) and retrain. Document in handoff. |
| **FIRES (4+ dims)** | Cross-cohort cage signature dominates. Path C is methodologically sound but the corpus has structural acoustic confounds the mask can't remove. Document failure mode and propose Path D (recording-environment normalization) as separate work. |

---

## Step 5 — Pull results back to worktree

After Slurm reports the job is done:

```bash
# From this worktree:
mkdir -p results/contour_vae_combined results/phase5_cross_cohort

scp -r shachar.levysahar@loginserver.elsc.huji.ac.il:$HUJI_DATA/contour_vae/results/contour_vae_combined/{latents.parquet,training_log.csv,reconstructions/index.html} results/contour_vae_combined/
scp -r shachar.levysahar@loginserver.elsc.huji.ac.il:$HUJI_DATA/contour_vae/results/phase5_cross_cohort/ results/
scp shachar.levysahar@loginserver.elsc.huji.ac.il:$HUJI_DATA/contour_vae/models/contour_vae_combined/best.pt models/contour_vae_combined/
```

Then surface `results/phase5_cross_cohort/index.html` to the user (the diagnostic verdict + UMAP).

---

## Memory + RAM math (why this fits on the cluster)

- **Combined patches.npz**: 69,293 × 257 × 234 × 4 bytes ≈ **16.0 GB** (np.savez uncompressed)
- **Training peak RAM**: ~16 GB (patches load) + ~4 GB (PyTorch + Adam state + batches) ≈ **20 GB**
- **Slurm `--mem=32G`** gives 12 GB headroom — safe.

If we hit OOM (unlikely): the fallback is to convert assembly output to a raw `.npy` and modify `train_contour_vae_v2.py:587` to use `np.load(..., mmap_mode='r')` — 3-line patch. The current code does `data = np.load(...)["patches"]` which loads eagerly.

---

## Files NOT to touch on the cluster

These are reference artifacts and must remain pristine. If they get edited mid-execution, the methodology defense breaks:

| Path | Why |
|---|---|
| `results/contour_extraction/5970/contours.parquet` | MATLAB-derived reference. The basis for all parity claims. |
| `results/contour_vae_v2_analysis/5970/` | Methodology-test (within-cohort) Phase 5 result. Refinement F comparison anchor. |
| `models/contour_vae_v2_5970/` | The 5970-only VAE that produced the within-cohort CLEAN verdict. |

---

## Quick references

- **Project context**: `CLAUDE.md` in worktree root
- **Original plan**: `PLAN_contour_masked_vae_pipeline.md`
- **Original orchestrator handoff**: `HANDOFF_contour_masked_vae_orchestrator.md`
- **Pre-registered diagnostic source of truth**: `scripts/cage_confound_diagnostic.py` (LOCKED — do not modify)
- **Local comparison report (just shipped to user)**: `results/path_c_comparison/index.html`
- **All Path C decision rationale**: `$CLAUDE_JOB_DIR/session_report.json` + `session_report.html` from the wrap-session that birthed this handoff (2026-05-19 13:26 session)

---

## Resume in next chat — paste this

```
Continue Path C from the cluster-execution handoff
at docs/handoffs/2026-05-19_path-c-cluster-execution.md.
Local state: all 4 cohort patches.npz files exist at
results/masked_patches/{cohort}_focus/, total 16.3 GB,
69,293 windows. Tailscale was down last session so we
pivoted from the rig to the ELSC HUJI cluster.

User's cluster login: ssh shachar.levysahar@loginserver.elsc.huji.ac.il
First action: SSH in, discover GPU partition name with `sinfo`,
then proceed with the handoff's Step 1 (SCP bundle).
If Tailscale is now back up, the rig is also a viable path —
the pipeline scripts are the same; only the deployment differs.
```

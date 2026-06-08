#!/usr/bin/env bash
# Tier-4 disk reclaim — handoff cleanup 2026-06-08.
#
# Deletes large REGENERABLE data dirs + cloud-backed archives that the user
# authorized for removal. Every target's provenance + regeneration recipe is in
# docs/DATA_REGENERATION_RECIPES.md; manifests preserving "how it was cut" /
# "what was curated" are in data/handoff_manifests/.
#
# SAFETY:
#   - Run from the repo root.
#   - Each dir target is re-verified UNTRACKED before deletion (skips if tracked).
#   - KEEPS: USV_lab_131204/ (24 GB source), data/alpha3_human_patches/ (human
#     shape-label substrate), data/lab_finetune_v1/{labeled,labels_audit_72.csv}
#     (human labels), everything tracked by git.
#   - This is NOT reversible from git (targets are untracked). Recoverable only
#     via the recipes in docs/DATA_REGENERATION_RECIPES.md (+ the 24 GB source).
#
# Usage:  bash scripts/reclaim_tier4_data.sh           # delete for real
#         DRY_RUN=1 bash scripts/reclaim_tier4_data.sh # print only, delete nothing
set -uo pipefail
cd "$(git rev-parse --show-toplevel)" || { echo "not in a git repo"; exit 1; }

DRY_RUN="${DRY_RUN:-0}"
freed_note() { echo "  [$1] $2"; }

rm_dir() {  # rm -rf a dir, but ONLY if it is untracked by git
  local d="$1"
  [ -e "$d" ] || { freed_note "skip" "$d (absent)"; return; }
  local tracked; tracked=$(git ls-files "$d" | wc -l)
  if [ "$tracked" -ne 0 ]; then freed_note "KEEP" "$d ($tracked TRACKED files — not deleting; use git rm by path if intended)"; return; fi
  local sz; sz=$(du -sh "$d" 2>/dev/null | cut -f1)
  if [ "$DRY_RUN" = "1" ]; then freed_note "would-rm" "$d ($sz)"; else rm -rf "$d" && freed_note "removed" "$d ($sz)"; fi
}

rm_file() {  # rm a single file (cloud-backed archives)
  local f="$1"
  [ -e "$f" ] || { freed_note "skip" "$f (absent)"; return; }
  local tracked; tracked=$(git ls-files "$f" | wc -l)
  if [ "$tracked" -ne 0 ]; then freed_note "KEEP" "$f (TRACKED — not deleting)"; return; fi
  local sz; sz=$(du -sh "$f" 2>/dev/null | cut -f1)
  if [ "$DRY_RUN" = "1" ]; then freed_note "would-rm" "$f ($sz)"; else rm -f "$f" && freed_note "removed" "$f ($sz)"; fi
}

echo "=== Tier-4 reclaim (DRY_RUN=$DRY_RUN) ==="
df -h . | tail -1

echo "--- 1. 58 GB chunked corpus (manifest preserved: data/handoff_manifests/chunk_manifest_131204.csv) ---"
rm_dir USV_lab_131204_chunked_2s_full

echo "--- 2. scratch chunk-test dirs + _hot review cascade (~2 GB) ---"
rm_dir USV_lab_131204_2chunk_test
rm_dir USV_lab_131204_10chunk_test
rm_dir USV_lab_131204_100chunk_test
rm_dir USV_lab_131204_chunked_300k_test
rm_dir USV_lab_131204_chunked_2s_test
rm_dir USV_lab_131204_chunked_2s_hot
rm_dir USV_lab_131204_chunked_2s_hot_reviewed
rm_dir USV_lab_131204_chunked_2s_hot_reviewed_reviewed

echo "--- 3. regenerable caches (~7 GB; KEEPS alpha3_human_patches + lab_finetune labeled/) ---"
rm_dir usv_language/prepared_data
rm_dir data/lab_finetune_v1/mining_candidates_500   # deterministic; labeled/ + labels_audit_72.csv KEPT
rm_dir data/alpha3_oracle_patches
rm_dir data/alpha3_patches
rm_dir data/alpha3_a6
# NOTE: data/alpha3_human_patches/ is intentionally NOT listed — human shape-label substrate.

echo "--- 4. dated result pilots (~1.5 GB; selections/reproducers tracked separately) ---"
for d in results/softnotch_* results/subtracted_pilot results/pipeline_comparison results/codex_detection_compare_*; do rm_dir "$d"; done

echo "--- 5. human-review cascades (selections preserved: data/handoff_manifests/review_cascade_selections.csv) ---"
rm_dir 5970_reviewed
rm_dir 5970_manual_review
rm_dir 5970_manual_review_reviewed
rm_dir 5970_manual_review_reviewed_reviewed
rm_dir USV_lab_131204_chunked_2s_full_reviewed
rm_dir USV_3452_sample_reviewed
rm_dir USV_3452_sample_reviewed_reviewed
rm_dir USV_2379_sample_reviewed
rm_dir detection_review_wavs
rm_dir detection_review_wavs_reviewed

echo "--- 6. cloud-backed archives (~14 GB) ---"
for f in USV1.zip USV2.zip USV3.zip USV4.zip USV5.zip USV6.zip USV7.zip USV8.zip \
         usv_lmt_031-20260515T151920Z-3-001.zip \
         mickey_meeting_20260427.tar.gz usv_teaching_20260427.tar.gz; do rm_file "$f"; done

echo "=== done ==="
df -h . | tail -1
echo "If DRY_RUN was 0: review 'git status' — extracted dirs that were also gitignored won't show; tracked ones (if any 'KEEP' above) need a separate decision."

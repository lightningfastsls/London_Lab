# Session Memory

> **Sync rule:** This file mirrors `docs/SESSION_MEMORY.md` in the repo.
> Update both whenever either changes, and include `docs/SESSION_MEMORY.md` in the next commit/push.

## arscontexta Setup (USV Research Pipeline)
- **Status**: Vault generated and committed (2026-02-18, commit aafe406)
- **Preset**: Research (atomic, flat, explicit+implicit linking, heavy processing, full automation)
- **Self-space**: Disabled (research preset) — ops/ absorbs identity/methodology
- **Semantic search**: qmd v1.0.6 installed and working (Vulkan GPU, AMD Radeon RX 5700). Config in `.mcp.json`. After adding notes: `qmd update && qmd embed`.
  - **Vulkan fix**: Patched `llm.js:253` to prefer `["vulkan", "cuda", "metal"]` — re-apply after qmd updates.
- **Skills**: 16 vocabulary-transformed skills in .claude/skills/ (reduce, reflect, reweave, verify, validate, seed, ralph, pipeline, tasks, stats, graph, next, learn, remember, rethink, refactor)
- **Hooks**: Two layers — project hooks (`.claude/hooks/*.ps1`, working) and plugin hooks (plugin cache `hooks.json`, disabled).
  - Project hooks in `settings.local.json`: session-orient.ps1 (SessionStart), session-capture.ps1 (Stop), check_agents_tag.cmd (Stop), check_plan_mode.cmd (PreToolUse), validate-note.cmd + auto-commit.cmd (PostToolUse:Write).
  - Plugin `hooks.json` emptied (2026-02-20) — bash `.sh` scripts don't work on Windows. **Re-empty after arscontexta plugin updates.**
  - Hook chain: `.cmd` wrappers -> `powershell.exe` via `cmd.exe /c` -> `.ps1` scripts.
- **Hook errors (KNOWN BUG)**: SessionStart/Stop hook errors on Windows are cosmetic (upstream bug #12671). Hooks work despite error messages. Don't try to fix.
- **Session continuity** (2026-02-20): Orient hook enhanced with overdue reminder detection, last-session bridge, pending tasks, status-filtered counts, queue.json thresholds, lifecycle archival. Capture hook writes `ops/last-session.md` and enforces State Update Rule. Dead `.sh` files deleted from project.
- **Implementation plan**: `skill-graph-implementation-plan.md` — Phases 1-4 ALL DONE. Phase 5.1 DONE (weekly routine, 2026-02-20). Remaining: Phase 5.2 (two-week validation, starts 2026-03-06)
- **USV Pipeline Phase 11.1 DONE** (2026-02-22): Bout Extraction & Preprocessing on Real Data — fixed deleted_by_user filtering in bout_extractor.py, added recursive WAV resolution, validation script `usv_language/scripts/validate_preprocessing.py`, 9 new tests (184 total usv_language). Pipeline run: 124 bouts -> 27 spectrograms (19/4/4 split), 103K frames. Tier 1 review APPROVED. Handoff at `docs/reviews/preprocessing-real-data-handoff.md`.
- **USV Pipeline Phase 10.1 DONE** (2026-02-21): Active Learning Cycle Runner — CycleMetrics + generate_cycle_report() in `src/usv_spectrogram/training/cycle_report.py`, 7-step orchestration CLI `scripts/run_training_cycle.py` (427 lines), 34 tests (461 total). Handoff at `docs/reviews/training-cycle-handoff.md`. Review at `docs/reviews/training-cycle-review.md`.
- **USV Pipeline Phase 9.1 DONE** (2026-02-21): Dataset Assembler — DatasetAssembler, AssemblyConfig, AssemblyReport in `src/usv_spectrogram/dataset/assembler.py` (~480 lines), CLI `scripts/assemble_training_data.py`, 10 tests (434 total). Key fixes: Hamilton's allocation, frame-level buffer masking, .gitignore case-sensitivity. Handoff at `docs/reviews/dataset-assembler-handoff.md`. Review at `docs/reviews/dataset-assembler-review.md`.
- **USV Pipeline Phase 8.4 DONE** (2026-02-21): Analysis & Interpretation Tools — 9 modules in `usv_language/analysis/` (config, transformer_suffix, codebook_viz, sequence_analysis, concept_manipulation, context_analysis, compositionality, run_analysis), 17 tests (599 total). Key fixes: batch decode (K,1,d_model) not (1,K,d_model), excess_entropy via entropy rate convergence. Handoff at `docs/reviews/analysis-tools-handoff.md`. Review at `docs/reviews/analysis-tools-review.md`.
- **USV Pipeline Phase 8.3 DONE** (2026-02-20): Hidden State VQ-VAE — VQVAEConfig, VectorQuantizerV2, HiddenStateVQVAE (~820K params), train_vqvae.py, compare_layers.py, 21 tests (151 total). Handoff at `docs/reviews/hidden-state-vqvae-handoff.md`.
- **Skills updated** (2026-02-20): `/implement` now includes Phase 4 REVIEW (spawns master-reviewer, writes review file). New `/roadmap-from-plan` skill converts web Claude plans into ROADMAP.md format with `/implement` blocks.
- **Methodology knowledge graph** (2026-02-23): 249 research claim .md files installed at `./methodology/` (project root). Sparse-cloned from `github.com/agenticnotetaking/arscontexta`. Git-tracked so it travels with the repo. These back arscontexta reasoning commands.
- **Plugin skills + reference** (2026-02-23): 4 plugin-level skills installed as local skills: `/ask` (query research graph), `/architect` (research-backed evolution), `/recommend` (architecture advice), `/health` (vault diagnostics — 8 categories, FAIL/WARN/PASS). Reference directory (`reference/`, 37 files) provides routing indexes, dimension maps, constraint docs, and templates that these skills depend on. All `${CLAUDE_PLUGIN_ROOT}` paths replaced with relative paths.
- **Topic map splits DONE** (2026-03-03): agent-governance split → code-review-governance (24 notes: multi-agent review architectures, cost optimization, effectiveness research, tooling). detection split → detection-landscape (28 notes: architectural taxonomy, alternative tools, source separation, annotation ecosystem). experimental-methods confirmed lean hub (0 direct notes, 3 sub-maps, no split needed). 52 note frontmatter files updated. Vault: ~497 notes, 17 topic maps. Bulk Knowledge Ingestion also moved to Completed in goals.md.

## State Update Rule (CRITICAL — every session)
- Before ending ANY session where a milestone/phase/task was completed, update ALL THREE:
  1. `ops/goals.md` — move from Active to Completed (orient hook reads this — stale = stale start)
  2. The tracking file (e.g. `skill-graph-implementation-plan.md`) — mark DONE
  3. This MEMORY.md — update summary so next session's system prompt is correct
- **This applies to work done in THIS repo too** — Phase 4.3 was completed here and goals.md wasn't updated, causing the next session to treat it as active.
- **Cross-repo**: When tevel-erp work affects files here, that session must note in its own MEMORY.md what needs syncing next USV session.
- **Parallel session hazard** (2026-02-22): When two sessions run simultaneously, the first to commit can sweep the other's files into a broad commit ("accumulated artifacts") without proper attribution. Phase 10.1 was lost this way — completed in a parallel session but committed by the Phase 9.1 session without documentation. **Mitigation**: each session should only `git add` files it created, never `git add .` or `git add -A`.

## Sync Rule: docs/SESSION_MEMORY.md
- `docs/SESSION_MEMORY.md` in the repo mirrors this file. **Update it on every push.**
- This ensures the memory is version-controlled and visible outside Claude Code.

## Bash Permissions
- settings.local.json has restrictive Bash permissions (allowlist-based)
- Use `git` commands for shell operations (always allowed)
- Use Glob/Grep/Read tools instead of bash for file operations
- `echo`, `ls`, `cd` are whitelisted but may fail due to shell quoting issues

## Notion Integration
- **Package**: `notion_notes/` — CLI toolkit for Notion KB automation (tag, atomize, link, process, move)
- **Credentials**: `.env` file at repo root (NOT in env vars — must load via `load_config(env_path=Path('.env'))`)
  - `NOTION_TOKEN` (ntn_...), `ANTHROPIC_API_KEY`, `NOTION_KB_DATABASE_ID`, `NOTION_NOTES_DATABASE_ID`
- **KB Database ID**: `30b2bc599f6e8032b337fbda2c975dda`
- **"Miki London lab" page ID**: `2ad2bc59-9f6e-80f1-8ee0-e85e0ef3d8a8` (used in "Projects" relation)
- **Upload script**: `scripts/upload_report_to_notion.py` — handles tables, batching (100-block limit), rich text splitting
- **Project State Report**: Uploaded 2026-02-20, page ID `30d2bc59-9f6e-81e4-ae8f-e97649f29014`

## Project Structure
- USV spectrogram/detection pipeline: `src/usv_spectrogram/`
- Tests: `tests/`
- Python venv: `.venv/Scripts/python.exe`
- WAV files: `5970 USV/`
- Knowledge graph vault: notes/, inbox/, ops/, templates/, manual/, archive/
- arscontexta methodology: methodology/ (249 research claim files, git-tracked)
- arscontexta reference: reference/ (37 structured reference files — routing indexes, constraints, templates)
- Plugin skills: .claude/skills/{ask,architect,recommend,health}/ (4 research-graph reasoning commands)

## Feedback: Proactive Vault Search
- [feedback_vault_search_before_tasks.md](feedback_vault_search_before_tasks.md) — Always search vault (qmd/grep) before any domain-touching task, not just code modifications. Cheap upfront cost, high-value context.

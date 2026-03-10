# Session Memory

> **Sync rule:** This file mirrors `docs/SESSION_MEMORY.md` in the repo.
> Update both whenever either changes, and include `docs/SESSION_MEMORY.md` in the next commit/push.

## arscontexta Setup (USV Research Pipeline)
- **Status**: Vault generated and committed (2026-02-18, commit aafe406)
- **Preset**: Research (atomic, flat, explicit+implicit linking, heavy processing, full automation)
- **Self-space**: Disabled (research preset) — ops/ absorbs identity/methodology
- **Semantic search**: qmd v1.0.7 installed. Config in `.mcp.json` with `INDEX_PATH` env pointing to `~/.cache/qmd/index.sqlite`. On new machine: (1) `npm i -g @tobilu/qmd`, (2) `qmd add mickey_london_lab . "**/*.md"`, (3) `qmd update && qmd embed`, (4) update `.mcp.json` `INDEX_PATH` to match new machine's home dir, (5) restart Claude Code.
  - **INDEX_PATH fix (2026-03-02)**: Claude Code's MCP server process breaks qmd's default `homedir()` resolution — MCP sees 0 docs despite CLI seeing 930. `INDEX_PATH` env var in `.mcp.json` bypasses all resolution. Machine-specific: update path when switching computers.
  - **Vulkan fix**: `llm.js:253` prefers `["vulkan", "cuda", "metal"]`. File: `C:\Users\shach\AppData\Roaming\npm\node_modules\@tobilu\qmd\dist\llm.js`. May need re-applying after qmd updates.
  - **MCP server stale index**: After running `qmd embed`, the MCP server must be restarted (restart Claude Code) for `mcp__qmd__vector_search` to see the new index. CLI `qmd vsearch` works immediately.
- **Skills**: 16 vocabulary-transformed skills in .claude/skills/ (reduce, reflect, reweave, verify, validate, seed, ralph, pipeline, tasks, stats, graph, next, learn, remember, rethink, refactor)
- **Hooks**: Two layers — project hooks (`.claude/hooks/*.ps1`, working) and plugin hooks (plugin cache `hooks.json`, disabled).
  - Project hooks in `settings.local.json`: session-orient.ps1 (SessionStart), session-capture.ps1 (Stop), check_agents_tag.cmd (Stop), check_plan_mode.cmd (PreToolUse), validate-note.cmd + auto-commit.cmd (PostToolUse:Write).
  - Plugin `hooks.json` emptied (2026-02-20) — bash `.sh` scripts don't work on Windows. **Re-empty after arscontexta plugin updates.**
  - Hook chain: `.cmd` wrappers -> `powershell.exe` via `cmd.exe /c` -> `.ps1` scripts.
- **Hook errors (KNOWN BUG)**: SessionStart/Stop hook errors on Windows are cosmetic (upstream bug #12671). Hooks work despite error messages. Don't try to fix.
- **Session continuity** (2026-02-20): Orient hook enhanced with overdue reminder detection, last-session bridge, pending tasks, status-filtered counts, queue.json thresholds, lifecycle archival. Capture hook writes `ops/last-session.md` and enforces State Update Rule. Dead `.sh` files deleted from project.
- **Implementation plan**: `skill-graph-implementation-plan.md` — Phases 1-4 ALL DONE. Phase 5.1 DONE (weekly routine, 2026-02-20). Remaining: Phase 5.2 (two-week validation, starts 2026-03-06)
- **USV Pipeline Phase 9.1 DONE** (2026-02-21): Dataset Assembler — DatasetAssembler, AssemblyConfig, AssemblyReport in `src/usv_spectrogram/dataset/assembler.py` (~480 lines), CLI `scripts/assemble_training_data.py`, 10 tests (434 total). Key fixes: Hamilton's allocation, frame-level buffer masking, .gitignore case-sensitivity. Handoff at `docs/reviews/dataset-assembler-handoff.md`. Review at `docs/reviews/dataset-assembler-review.md`.
- **USV Pipeline Phase 8.4 DONE** (2026-02-21): Analysis & Interpretation Tools — 9 modules in `usv_language/analysis/` (config, transformer_suffix, codebook_viz, sequence_analysis, concept_manipulation, context_analysis, compositionality, run_analysis), 17 tests (599 total). Key fixes: batch decode (K,1,d_model) not (1,K,d_model), excess_entropy via entropy rate convergence. Handoff at `docs/reviews/analysis-tools-handoff.md`. Review at `docs/reviews/analysis-tools-review.md`.
- **USV Pipeline Phase 8.3 DONE** (2026-02-20): Hidden State VQ-VAE — VQVAEConfig, VectorQuantizerV2, HiddenStateVQVAE (~820K params), train_vqvae.py, compare_layers.py, 21 tests (151 total). Handoff at `docs/reviews/hidden-state-vqvae-handoff.md`.
- **Skills updated** (2026-02-20): `/implement` now includes Phase 4 REVIEW (spawns master-reviewer, writes review file). New `/roadmap-from-plan` skill converts web Claude plans into ROADMAP.md format with `/implement` blocks.
- **Methodology knowledge graph** (2026-02-23): 249 research claim .md files installed at `./methodology/` (project root). Sparse-cloned from `github.com/agenticnotetaking/arscontexta`. Git-tracked so it travels with the repo. These back arscontexta reasoning commands (`/arscontexta:ask`, `/arscontexta:architect`).
- **Topic map splits DONE** (2026-03-03): agent-governance split → code-review-governance (24 notes: multi-agent review architectures, cost optimization, effectiveness research, tooling). detection split → detection-landscape (28 notes: architectural taxonomy, alternative tools, source separation, annotation ecosystem). experimental-methods confirmed lean hub (0 direct notes, 3 sub-maps, no split needed). 52 note frontmatter files updated. Vault: ~497 notes, 17 topic maps. Bulk Knowledge Ingestion also moved to Completed in goals.md.

## State Update Rule (CRITICAL — every session)
- Before ending ANY session where a milestone/phase/task was completed, update ALL THREE:
  1. `ops/goals.md` — move from Active to Completed (orient hook reads this — stale = stale start)
  2. The tracking file (e.g. `skill-graph-implementation-plan.md`) — mark DONE
  3. This MEMORY.md — update summary so next session's system prompt is correct
- **This applies to work done in THIS repo too** — Phase 4.3 was completed here and goals.md wasn't updated, causing the next session to treat it as active.
- **Cross-repo**: When tevel-erp work affects files here, that session must note in its own MEMORY.md what needs syncing next USV session.

## Sync Rule: docs/SESSION_MEMORY.md
- `docs/SESSION_MEMORY.md` in the repo mirrors this file. **Update it on every push.**
- This ensures the memory is version-controlled and visible outside Claude Code.

## Environment
- Windows (win32), Python 3.12.1
- Use `powershell -Command "& '.venv\Scripts\python.exe' ..."` for running Python
- py_compile via temp script is most reliable (quoting issues with inline commands)
- `cmd /c` often swallows output; prefer powershell

## Bash Permissions
- settings.local.json has restrictive Bash permissions (allowlist-based)
- Use `git` commands for shell operations (always allowed)
- Use Glob/Grep/Read tools instead of bash for file operations
- `echo`, `ls`, `cd` are whitelisted but may fail due to shell quoting issues

## Pre-existing Test Failures
- `test_long_continuous_tone_rejected` in `test_energy_detector.py` — known flaky, unrelated to app changes

## Architecture Notes
- Detection app: PyQt6, `main_window.py` is the main hub
- `DetectedUSV` dataclass in `detection_logic.py` — has `save_state` field (Phase 2)
- `SavedDetectionTracker` handles duplicate checking by boundary identity (1ms tolerance), not overlap
- Views: `SpectrogramView` (canvas+scroll) and `ProbabilityView` (canvas+scroll), synchronized via signals
- Ghost detections (saved_previous) are combined with current detections for view rendering

## Notion Integration
- **Package**: `notion_notes/` — CLI toolkit for Notion KB automation (tag, atomize, link, process, move)
- **Credentials**: `.env` file at repo root (NOT in env vars — must load via `load_config(env_path=Path('.env'))`)
  - `NOTION_TOKEN` (ntn_...), `ANTHROPIC_API_KEY`, `NOTION_KB_DATABASE_ID`, `NOTION_NOTES_DATABASE_ID`
- **KB Database ID**: `30b2bc599f6e8032b337fbda2c975dda`
- **"Miki London lab" page ID**: `2ad2bc59-9f6e-80f1-8ee0-e85e0ef3d8a8` (used in "Projects" relation)
- **Upload script**: `scripts/upload_report_to_notion.py` — handles tables, batching (100-block limit), rich text splitting
- **Project State Report**: Uploaded 2026-02-20, page ID `30d2bc59-9f6e-81e4-ae8f-e97649f29014`

## Skill-Creator
- **Known bugs & fixes**: See `skill-creator-known-bugs.md` — Windows compat fixes + critical cross-platform detection bugs, all fixed in `.claude/skills/skill-creator/scripts/`. If skill-creator updates upstream, re-apply fixes.
- **`/reduce` description improved** (2026-03-01): Changed to directive/intent-focused style. Not yet validated with fixed eval harness.
- **`/learn` skill rebuilt** (2026-03-01): v3 deployed. 3 iterations: v1 (pipeline-heavy, shallow research) -> v2 ("thoroughness first", 60% faster quick lookups) -> v3 (deep survey self-check loop, 40 sources matching baseline). 100% pass on 10 assertions across 3 test cases. Description optimization attempted but abandoned — `claude -p` doesn't trigger skills for research queries.

## Project Structure
- USV spectrogram/detection pipeline: `src/usv_spectrogram/`
- Tests: `tests/`
- Python venv: `.venv/Scripts/python.exe`
- WAV files: `5970 USV/`
- Knowledge graph vault: notes/, inbox/, ops/, templates/, manual/, archive/
- arscontexta methodology: methodology/ (249 research claim files, git-tracked)

## Codex Collaboration Setup (2026-03-06)
- **AGENTS.md** at repo root = Codex's behavioral contract (like CLAUDE.md for Claude Code)
- **Ownership boundary**: Codex owns `src/`, `tests/`, `scripts/`, `usv_language/`, `docs/handoffs/`. Claude Code owns `.claude/`, `ops/`, `notes/`, `methodology/`, `reference/`, `templates/`, `inbox/`.
- **Handoff protocol**: Bidirectional. Codex writes implementation handoffs, Claude Code writes review handoffs (`From/To/Re` header). Both go to `docs/handoffs/YYYY-MM-DD_task-name.md`. Resolved handoffs archive to `docs/handoffs/archive/`. **Orientation reads only top-level `docs/handoffs/*.md`, never `archive/`.** **Action rule: after reviewing a non-rolling Codex handoff, immediately `mv` it to `docs/handoffs/archive/`. The directory IS the tracking mechanism — no separate review log needed.**
- **KG capture from handoffs (2026-03-07)**: Do NOT `/seed` handoffs eagerly. Instead: (1) review all code/handoffs first, (2) accumulate candidate insights in `ops/kg-candidates.md`, (3) at end of review batch, consult `arscontexta-expert` to filter for durable knowledge vs implementation artifacts, (4) only then `/seed` or write notes for what passes. Lesson: `/seed` extracts maximally — the agent must be the filter before invoking it. Codex also instructed to batch bug fixes into single handoffs rather than stopping after each fix.
- **Navigation**: `docs/codex_index.md` is Codex's compact routing map (reads `ops/goals.md` + `ops/reminders.md` for session context)
- **Context briefing**: `docs/handoffs/codex_context_briefing.md` = the original briefing given to Codex
- **Key files**: `AGENTS.md`, `docs/codex_index.md`, `docs/handoffs/README.md`

# currentDate
Today's date is 2026-03-07.

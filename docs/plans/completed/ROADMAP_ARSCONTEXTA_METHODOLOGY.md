# arscontexta Methodology Notes — Implementation Roadmap

> This file tracks the setup task for pulling the arscontexta methodology knowledge graph
> into the local plugin installation. Separate from the main USV ROADMAP.md.
> Human: use the `/implement` command below by copy-pasting it into a Claude Code session.

---

## How to Use This File

1. This is a **single-module operational task** — no phase dependencies
2. The `/implement` command below is self-contained
3. After completion: commit, verify methodology notes are accessible, then mark DONE

## Status Key

- **DONE** — Implemented and verified
- **READY** — Can start immediately
- **BLOCKED** — Waiting on dependency or external input

---

## Module 1: Pull arscontexta Methodology Notes

### 1.1 Methodology Knowledge Graph Setup

**What:** Download the `methodology/` directory (~249 research claim files) from the arscontexta GitHub repo into the local plugin installation. These files are the backing knowledge graph that arscontexta reasoning commands (`/ask`, `/recommend`, `/architect`, `/health`) require to function.
**Status:** DONE (2026-02-23) — installed at `./methodology/` (project root), 249 .md files, git-tracked. Plugin skills (/ask, /architect, /recommend, /health) and reference/ directory (37 files) installed 2026-02-23.
**Review Tier:** 1
**Depends on:** arscontexta plugin installed (via `/plugin install arscontexta@agenticnotetaking`)

/implement Pull arscontexta Methodology Knowledge Graph

Download and install the arscontexta methodology directory (~249 markdown research claim files) into the existing plugin installation. Without these files, arscontexta reasoning commands operate without their backing research graph.

**Context:** arscontexta is a Claude Code plugin for knowledge management. Its reasoning commands reference a curated set of methodology notes — research claims about note-taking, knowledge graphs, atomic notes, etc. The plugin is already installed but the methodology directory was not included (it's large and optional). We need to pull it from the GitHub repo and place it in the correct location.

**Steps to execute:**

1. **Locate the arscontexta plugin installation**

```bash
# Check standard plugin locations (in order of likelihood)
# Store result as $ARSCONTEXTA_ROOT for subsequent steps
ls -d ~/.claude/plugins/arscontexta 2>/dev/null
ls -d ./.claude/plugins/arscontexta 2>/dev/null
```

If not found in standard locations, search more broadly:
```bash
find ~ -maxdepth 5 -type d -name "arscontexta" 2>/dev/null | head -10
```

2. **Check for existing methodology/ directory**

```bash
ls "$ARSCONTEXTA_ROOT/methodology/" 2>/dev/null
```

If it exists, count files to check completeness — expect ~249 `.md` files. If incomplete, proceed to overwrite. If complete, skip to verification.

3. **Sparse-checkout the methodology directory from GitHub**

```bash
cd /tmp
git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/agenticnotetaking/arscontexta.git \
    arscontexta-methodology
cd arscontexta-methodology
git sparse-checkout set methodology
```

Key flags:
- `--depth 1` — only latest commit (no history)
- `--filter=blob:none` — don't download blobs until needed
- `--sparse` — only checkout specified paths

4. **Copy methodology/ into the plugin installation**

```bash
cp -r /tmp/arscontexta-methodology/methodology/ "$ARSCONTEXTA_ROOT/methodology/"
```

If the plugin directory is read-only (managed by plugin manager), use alternative:
```bash
# Copy to project knowledge directory instead
cp -r /tmp/arscontexta-methodology/methodology/ ./methodology/
# Then update arscontexta config to point here (check plugin docs)
```

5. **Verify the installation**

```bash
# Count markdown files — expect ~249
find "$ARSCONTEXTA_ROOT/methodology/" -name "*.md" | wc -l

# Spot-check file contents
ls "$ARSCONTEXTA_ROOT/methodology/" | head -10
head -20 "$ARSCONTEXTA_ROOT/methodology/"/*.md | head -40
```

6. **Cleanup temporary clone**

```bash
rm -rf /tmp/arscontexta-methodology
```

**Files affected:**

- `$ARSCONTEXTA_ROOT/methodology/` (NEW directory, ~249 files) — Methodology research claims
- No project source code files modified

**Test plan:**
```
1. methodology/ directory exists at the plugin installation path
2. File count is approximately 249 markdown files
3. Files contain valid markdown with frontmatter (YAML between --- delimiters)
4. /arscontexta:ask responds with methodology-backed answers (manual test)
5. No files outside methodology/ were modified in the plugin installation
```

**Exit criteria:**
- [x] arscontexta plugin root located — N/A (no plugin install on this machine; files placed at project root `./methodology/`)
- [x] methodology/ directory contains ~249 .md files — confirmed 249
- [x] Spot-check confirms files have valid content (not empty, have frontmatter) — verified
- [x] 4 plugin skills installed as local skills: /ask, /architect, /recommend, /health (2026-02-23)
- [x] reference/ directory installed (37 files — routing indexes, constraints, templates) (2026-02-23)
- [x] All ${CLAUDE_PLUGIN_ROOT} path references replaced with relative paths (38 replacements)
- [x] `/ask` returns methodology-informed responses — tested 2026-02-23, queried "why do we use atomic notes?", returned 8-claim synthesis with citations
- [x] Temporary /tmp clone cleaned up
- [x] No unintended changes to project files

**Rollback:** If anything goes wrong, simply `rm -rf "$ARSCONTEXTA_ROOT/methodology/"` to remove the added files. The plugin will revert to operating without the methodology graph (its previous state).

---

## Notes

- The sparse checkout approach minimizes bandwidth — only the methodology directory is downloaded, not the full repo
- If the plugin was installed via `/plugin install arscontexta@agenticnotetaking`, the installation path is likely `~/.claude/plugins/arscontexta/`
- If the plugin manager enforces read-only on plugin directories, methodology notes may need to live in the project directory with a config pointer
- After a future `arscontexta` plugin update, check if methodology/ was overwritten or removed — may need to re-pull

---

## Gate

After completion:
- [x] arscontexta reasoning commands (/ask, /health) work with methodology+reference backing — both tested 2026-02-23. /ask synthesized 8 claims. /health quick ran 3 categories (PASS/WARN/FAIL), wrote report to ops/health/.
- [x] methodology/ is not accidentally gitignored in this project — git-tracked, will be committed
- [x] Document final installation path in project MEMORY.md for future sessions

**STATUS: ALL GATES PASSED — PLAN COMPLETE (2026-02-23)**

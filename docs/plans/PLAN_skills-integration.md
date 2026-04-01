# Plan: Skills Integration Improvements

**Source:** Analysis of https://github.com/anthropics/skills against WORKFLOW-AUDIT.md and PLUGIN-AUDIT.md
**Date:** 2026-02-27
**Scope:** 4 improvements ordered by impact. Each is independent — implement in any order or skip any.

---

## Improvement 1: Playwright Visual Verification Layer

**Addresses:** Friction Point #2 (master-reviewer bias as subagent)
**Impact:** High — adds objective visual verification to the review pipeline
**Effort:** Medium (new command + helper script + Playwright test patterns)

### Problem

The master-reviewer runs as a Task subagent within the implementor's session. It found real bugs (Phase 8.4: causal attention cross-contamination), but the implementor controls when to spawn the reviewer, interprets the findings, and marks own fixes without re-review. For the Streamlit Parameter Lab specifically, the reviewer can only read code and run pytest — it cannot verify that the UI actually renders correctly.

### What to Build

A `/verify-visual` command that launches the Streamlit app, takes screenshots, and runs Playwright assertions against the rendered UI. This provides objective evidence that the reviewer (and the implementor) cannot interpret away.

### Implementation Steps

#### Step 1: Install Playwright

```powershell
# In the project .venv
.\.venv\Scripts\pip.exe install playwright
.\.venv\Scripts\playwright.exe install chromium
```

Verify installation works:
```powershell
.\.venv\Scripts\python.exe -c "from playwright.sync_api import sync_playwright; print('OK')"
```

#### Step 2: Create `scripts/with_server.py`

Adapt from the `webapp-testing` skill pattern. This helper manages the Streamlit server lifecycle — starts it, waits for the port, runs the test script, then cleans up.

```python
#!/usr/bin/env python3
"""
Start one or more servers, wait for them to be ready, run a command, then clean up.

Usage:
    python scripts/with_server.py --server ".\.venv\Scripts\streamlit.exe run scripts/usv_parameter_lab.py" --port 8501 -- .\.venv\Scripts\python.exe scripts/verify_visual.py
"""

import subprocess
import socket
import time
import sys
import argparse

def is_server_ready(port, timeout=30):
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection(('localhost', port), timeout=1):
                return True
        except (socket.error, ConnectionRefusedError):
            time.sleep(0.5)
    return False

def main():
    parser = argparse.ArgumentParser(description='Run command with one or more servers')
    parser.add_argument('--server', action='append', dest='servers', required=True)
    parser.add_argument('--port', action='append', dest='ports', type=int, required=True)
    parser.add_argument('--timeout', type=int, default=30)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()

    if args.command and args.command[0] == '--':
        args.command = args.command[1:]
    if not args.command:
        print("Error: No command specified")
        sys.exit(1)
    if len(args.servers) != len(args.ports):
        print("Error: --server and --port count must match")
        sys.exit(1)

    servers = [{'cmd': cmd, 'port': port} for cmd, port in zip(args.servers, args.ports)]
    server_processes = []

    try:
        for i, server in enumerate(servers):
            print(f"Starting server {i+1}/{len(servers)}: {server['cmd']}")
            process = subprocess.Popen(server['cmd'], shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            server_processes.append(process)
            print(f"Waiting for port {server['port']}...")
            if not is_server_ready(server['port'], timeout=args.timeout):
                raise RuntimeError(f"Server failed on port {server['port']} within {args.timeout}s")
            print(f"Server ready on port {server['port']}")

        print(f"\nAll {len(servers)} server(s) ready")
        print(f"Running: {' '.join(args.command)}\n")
        result = subprocess.run(args.command)
        sys.exit(result.returncode)
    finally:
        for i, process in enumerate(server_processes):
            try:
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            print(f"Server {i+1} stopped")
```

#### Step 3: Create `scripts/verify_visual.py`

The actual Playwright verification script. This should be a TEMPLATE that gets extended per-module. Start with the Parameter Lab checks:

```python
#!/usr/bin/env python3
"""
Visual verification for the USV Parameter Lab Streamlit app.
Launched via with_server.py — Streamlit is already running on port 8501.

Checks:
1. App loads without errors
2. Main UI elements are present (sidebar, plot area)
3. No Streamlit error banners visible
4. Screenshot captured for manual review
"""

from playwright.sync_api import sync_playwright
from pathlib import Path
import sys
import json

RESULTS_DIR = Path("docs/reviews/visual")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

def run_checks():
    results = {"passed": [], "failed": [], "screenshots": []}

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(viewport={"width": 1280, "height": 900})

        # 1. Navigate and wait for full load
        page.goto("http://localhost:8501")
        page.wait_for_load_state("networkidle")

        # 2. Check no Streamlit error/exception banners
        error_elements = page.locator('[data-testid="stException"]').all()
        if len(error_elements) == 0:
            results["passed"].append("No Streamlit exceptions on load")
        else:
            results["failed"].append(f"Found {len(error_elements)} Streamlit exception(s) on load")

        # 3. Check sidebar exists (Parameter Lab always has sidebar controls)
        sidebar = page.locator('[data-testid="stSidebar"]')
        if sidebar.count() > 0:
            results["passed"].append("Sidebar present")
        else:
            results["failed"].append("Sidebar missing — Parameter Lab should have sidebar controls")

        # 4. Capture full-page screenshot
        screenshot_path = RESULTS_DIR / "parameter-lab-load.png"
        page.screenshot(path=str(screenshot_path), full_page=True)
        results["screenshots"].append(str(screenshot_path))
        results["passed"].append(f"Screenshot saved: {screenshot_path}")

        # 5. Check for key UI elements (extend per module)
        # These selectors should be updated as the Parameter Lab evolves.
        # The point is that the reviewer can READ what was checked and SEE the screenshot.

        browser.close()

    return results


def main():
    print("=" * 60)
    print("VISUAL VERIFICATION: USV Parameter Lab")
    print("=" * 60)

    results = run_checks()

    print(f"\nPassed: {len(results['passed'])}")
    for p in results["passed"]:
        print(f"  ✓ {p}")

    if results["failed"]:
        print(f"\nFailed: {len(results['failed'])}")
        for f in results["failed"]:
            print(f"  ✗ {f}")

    print(f"\nScreenshots: {len(results['screenshots'])}")
    for s in results["screenshots"]:
        print(f"  📸 {s}")

    # Write machine-readable results
    results_json = RESULTS_DIR / "results.json"
    with open(results_json, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults written to {results_json}")

    if results["failed"]:
        print("\n⚠ VISUAL VERIFICATION FAILED")
        sys.exit(1)
    else:
        print("\n✓ VISUAL VERIFICATION PASSED")
        sys.exit(0)


if __name__ == "__main__":
    main()
```

#### Step 4: Create `.claude/commands/verify-visual.md`

```markdown
Run visual verification on the Streamlit Parameter Lab.

Steps:
1. Run the visual verification suite:
   ```
   .\.venv\Scripts\python.exe scripts/with_server.py \
     --server ".\.venv\Scripts\streamlit.exe run scripts/usv_parameter_lab.py --server.headless true" \
     --port 8501 \
     -- .\.venv\Scripts\python.exe scripts/verify_visual.py
   ```
2. Read docs/reviews/visual/results.json for pass/fail
3. If screenshots were captured, examine them for visual correctness
4. Report: which checks passed, which failed, and show screenshot paths

If any checks fail, describe what's wrong and suggest fixes.
```

#### Step 5: Integrate with review workflow

In `.claude/commands/implement.md`, add to Phase 6 (Review), after master-reviewer:

```
If module has Streamlit UI components:
  Run /verify-visual before or after master-reviewer
  Include visual verification results in the review report
```

In `.claude/agents/master-reviewer.md`, add to the review checklist:

```
If docs/reviews/visual/results.json exists and is recent (< 1 hour old):
  Include visual verification results in review findings
  Reference screenshots in report
```

### Extending for New Modules

When new modules have visual components, add check functions to `verify_visual.py`:

```python
# Example: checking spectrogram rendering
def check_spectrogram_render(page):
    """Verify spectrogram plot renders with correct axes."""
    # Load a known test file
    # Check that the plot canvas exists
    # Verify axis labels show frequency (kHz) and time (s)
    # Screenshot the spectrogram area specifically
    pass
```

### What This Does NOT Replace

This is additive — it does not replace master-reviewer, pytest, or `/verify`. It adds a visual evidence layer that provides screenshots and DOM-based assertions the reviewer can cite.

---

## Improvement 2: Rename `/verify` to Resolve Command Collision

**Addresses:** Friction Point #4 (command name collision)
**Impact:** Low-medium — eliminates dispatch ambiguity
**Effort:** Very low (rename 1 file, update 2-3 references)

### Problem

Two `/verify` commands exist:
- `.claude/commands/verify.md` — py_compile + pytest + flake8 (code verification)
- `.claude/skills/verify/SKILL.md` — recite + validate + review (note quality verification)

Context-dependent dispatch works most of the time, but the ambiguity can confuse the dispatch system.

### Implementation Steps

1. Rename `.claude/commands/verify.md` → `.claude/commands/verify-code.md`
   - This creates `/verify-code` for code verification
   - The arscontexta `/verify` keeps its name (it's the more complex skill and harder to rename since it's plugin-managed)

2. Update CLAUDE.md Quick Commands section:
   - Change any reference to `/verify` (for code) to `/verify-code`

3. Update `.claude/commands/implement.md`:
   - In Phase 4 (Implementation), change `/verify` references to `/verify-code`

4. Update `.claude/commands/verify-quick.md`:
   - If it references `/verify` as the "full version", update to `/verify-code`

5. Verify: search all `.claude/` files for bare `/verify` references and update code-context ones.

### Naming After Change

| Command | Purpose | Source |
|---------|---------|--------|
| `/verify-code` | py_compile + pytest + flake8 | `.claude/commands/verify-code.md` |
| `/verify-quick` | py_compile on modified files + pytest | `.claude/commands/verify-quick.md` |
| `/verify-visual` | Playwright screenshots + DOM assertions | `.claude/commands/verify-visual.md` (from Improvement 1) |
| `/verify` | Recite + validate + review (notes) | `.claude/skills/verify/SKILL.md` (arscontexta) |
| `/validate` | Schema validation only (notes) | `.claude/skills/validate/SKILL.md` (arscontexta) |

---

## Improvement 3: Doc Staleness Check in `/implement`

**Addresses:** Friction Point #3 (documentation-code drift causing agent to distrust user)
**Impact:** Medium — prevents the specific failure mode where stale docs override user's correct claims
**Effort:** Low (add a step to existing command)

### Problem

PROJECTS.md said the PyQt6 app was "Not yet started" when it was fully built. Agent cited stale docs to contradict user's correct claims. When docs are wrong, agents trust docs over users — this is backwards.

The current `/implement` command has a documentation phase (Phase 5) that creates/updates module docs. But it only creates NEW docs — it doesn't audit EXISTING docs for staleness after the implementation changes things.

### Implementation Steps

Add a new step to `.claude/commands/implement.md` between Phase 5 (Documentation) and Phase 6 (Review).

#### New Step: Phase 5.5 — Staleness Audit

Add the following to `implement.md`:

```markdown
### Phase 5.5: Staleness Audit

After creating/updating module documentation, check whether the implementation
invalidated any existing documentation claims:

1. Grep the following files for references to the implemented module name:
   - PROJECTS.md
   - IMPLEMENTATION_PROGRESS.md
   - docs/modules/*.md (all module docs, not just the current one)
   - docs/architecture/patterns.md

2. For each reference found, check if it's still accurate given what was just implemented.
   Common drift patterns:
   - Status fields saying "Not started" or "Planned" for things that now exist
   - Dependency descriptions that reference old interfaces
   - Architecture docs showing outdated data flow

3. If stale references found:
   - Fix them immediately (status updates, interface corrections)
   - List all fixes in the handoff document under "## Staleness Fixes"

4. If no stale references found, note "Staleness audit: clean" in the handoff.

This step is mandatory — do not skip to Phase 6 without completing it.
```

### Why This Works

The staleness check runs at exactly the right moment — after implementation is done (so we know what changed) and before review (so the reviewer sees accurate docs). It catches the specific failure mode from Friction #3 without adding a new command or hook.

---

## Improvement 4: MCP Builder Patterns for Cloudy Claude ERP Integration

**Addresses:** Future Cloudy Claude architecture
**Impact:** High (for Cloudy Claude specifically, not USV)
**Effort:** Reference material — no implementation in USV repo

### What to Do

When you start building the Priority/SAP ERP connectors for Cloudy Claude, install the `mcp-builder` skill in that project's Claude Code:

```
/plugin marketplace add anthropics/skills
/plugin install example-skills@anthropic-agent-skills
```

Then when working on ERP integration, say "Use the mcp-builder skill" and it will load the full guide for building MCP servers.

### Key Patterns to Adopt from mcp-builder

These are the most relevant patterns for your ERP use case:

1. **Tool naming**: Use consistent prefixes — `priority_get_order`, `priority_list_parts`, `sap_query_inventory` etc. This helps Claude discover the right tool.

2. **Actionable error messages**: When the ERP API returns an error, the MCP tool response should say what to try next, not just "Error 500". Example: `"Part not found in Priority. Try searching with manufacturer part number instead of OEM number."`

3. **Pagination**: ERP queries can return thousands of results. Design tools with built-in pagination (`limit`, `offset` params) so Claude doesn't choke on a 10,000-row response.

4. **Eval framework**: After building the MCP server, write 10 realistic questions ("What's the current stock of brake pad X for Toyota Corolla 2019?") and test whether Claude can actually answer them using your tools. The mcp-builder skill has a full eval template for this.

### Not Needed Now

This is reference for when you start the Cloudy Claude ERP work. Don't install anything in the USV project — it's not relevant there.

---

## Implementation Order

| # | Improvement | Effort | Impact | Dependencies |
|---|------------|--------|--------|-------------|
| 2 | `/verify` rename | 15 min | Removes daily friction | None |
| 3 | Staleness audit | 30 min | Prevents trust violations | None |
| 1 | Playwright visual verification | 2-3 hours | Objective review evidence | Playwright install |
| 4 | MCP builder for Cloudy Claude | N/A | Future reference | Cloudy Claude project |

Recommended: Do #2 and #3 first (quick wins), then #1 when you have a longer session available.

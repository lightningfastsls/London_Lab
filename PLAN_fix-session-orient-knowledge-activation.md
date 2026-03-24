# Plan: Fix session-orient Knowledge Activation

> **Priority:** CRITICAL — knowledge activation has never worked on this system
> **File to modify:** `.claude/hooks/session-orient.ps1`
> **Lines affected:** 222-339 (Knowledge Activation section)
> **Risk level:** MEDIUM — modifying a SessionStart hook; must not break other sections of the script
> **Estimated scope:** ~30 lines changed in one file

---

## Background: Why This Fix Matters

The session-orient hook runs at the start of every Claude Code session. It's a PowerShell script (`.claude/hooks/session-orient.ps1`, 494 lines) that:
1. Shows active goals from `ops/goals.md`
2. Shows reminders with overdue detection
3. Shows last session summary
4. Shows pending tasks
5. **Surfaces relevant vault notes per active goal thread** (lines 222-339) ← THIS IS BROKEN
6. Archives stale completed items (goals, tasks, reminders)

Section 5 ("Knowledge Activation") is supposed to query `qmd` (a local markdown search engine) to find vault notes relevant to each active goal thread and write them to `ops/session-relevance.md`. This file is then read by the agent at session start to have domain context ready.

**The problem:** `ops/session-relevance.md` always shows "No strong matches above relevance threshold" for every thread. Investigation revealed this is NOT a search quality issue — `qmd` literally cannot execute.

---

## Root Cause Analysis (4 independent bugs found)

### Bug 1: Broken qmd npm shim on Windows (CRITICAL — total execution failure)

**Location:** Line 267 (`& qmd search ...`) and line 285 (`& qmd vsearch ...`)

The npm-installed `qmd` command has a PowerShell shim at `C:\Users\light\AppData\Roaming\npm\qmd.ps1`. This shim tries to invoke `/bin/sh.exe` — a Unix path that does not exist on Windows:

```powershell
# From C:\Users\light\AppData\Roaming\npm\qmd.ps1 line 24:
& "/bin/sh$exe"  "$basedir/node_modules/@tobilu/qmd/bin/qmd" $args
# Resolves to: & "/bin/sh.exe" ...
# ERROR: /bin/sh.exe not found
```

The `catch {}` blocks on lines 280 and 301 silently swallow this error. Result: every `qmd` call fails silently, `$threadNotes` stays empty, and "No strong matches" is written for every thread.

**Evidence:** Running `powershell.exe -Command '& qmd search "test" --json'` from WSL produces:
```
The term '/bin/sh.exe' is not recognized as the name of a cmdlet...
```

**Fix:** Call `node` directly with the qmd JavaScript entry point instead of using the broken shim. The actual qmd code exists at `C:\Users\light\AppData\Roaming\npm\node_modules\@tobilu\qmd\dist\cli\qmd.js` and works when invoked via `node`.

The qmd index database is at `~/.cache/qmd/index.sqlite` (or `C:\Users\light\.cache\qmd\index.sqlite` on Windows). The `INDEX_PATH` environment variable must be set for qmd to find it.

### Bug 2: BM25 query dilution from status words (HIGH — wrong results even if qmd works)

**Location:** Lines 256-261 (vsearch query construction)

The vsearch query concatenates the thread title + first sentence of the description:
```
"DeepSqueak Classification Bridge Phase 2 (Raven export) DONE, Phase 3 (MATLAB import+clustering) IN PROGRESS."
```

Words like "DONE", "IN PROGRESS", "Phase 2", "Phase 3" dilute BM25 scoring. Benchmark results:
- Full prose query (52 words): **0 results**
- Condensed query ("DeepSqueak Raven classification bridge"): **5 relevant results, 0.93 top score**

**Fix:** Strip status words and phase markers from the description before building the query.

### Bug 3: Period stripping breaks version numbers (MEDIUM)

**Location:** Line 264

```powershell
$titleWords = $threadTitle -replace '[^\w\s]', '' -replace '\s+', ' '
```

This regex removes ALL non-word, non-space characters, including periods. "Phase 5.3" becomes "Phase 53" — which matches nothing in BM25 search.

**Fix:** Preserve periods in the regex: `'[^\w\s.]'` instead of `'[^\w\s]'`

### Bug 4: Keyword search JSON parsing doesn't handle stderr (MEDIUM — defensive)

**Location:** Line 269

```powershell
$kwJson = ($kwRaw | Out-String).Trim()
if ($kwJson -match '^\[') {  # <-- fails if stderr warning prepends the output
```

In PowerShell, `2>&1` captures stderr as ErrorRecord objects. If qmd ever prints warnings on stderr (e.g., "no GPU acceleration"), they'd be rendered as text before the JSON array, causing the `^\[` regex to fail.

The vec search code (lines 288-289) already handles this correctly with `IndexOf('[')`. The keyword search code does not.

**Fix:** Use the same bracket-finding pattern as the vec search code.

---

## Implementation Steps

All changes are in `.claude/hooks/session-orient.ps1`.

### Step 1: Add qmd path resolution (insert after line 228)

Replace the current qmd availability check:

**Current code (line 226-228):**
```powershell
    try {
        # Check qmd is available
        $qmdPath = Get-Command qmd -ErrorAction Stop | Select-Object -ExpandProperty Source
```

**Replace with:**
```powershell
    try {
        # Resolve qmd via node directly (npm shim is broken on Windows — /bin/sh.exe not found)
        $qmdScript = Join-Path $env:APPDATA "npm\node_modules\@tobilu\qmd\dist\cli\qmd.js"
        if (-not (Test-Path $qmdScript)) {
            throw "qmd not found at $qmdScript"
        }
        $nodeExe = "node"
        $env:INDEX_PATH = Join-Path $env:USERPROFILE ".cache\qmd\index.sqlite"
```

### Step 2: Strip status words from vsearch query (modify lines 256-261)

**Current code (lines 256-261):**
```powershell
                # Extract first sentence of description for vsearch query
                $firstSentence = $threadDesc
                if ($threadDesc -match '^([^.]+\.)') { $firstSentence = $Matches[1] }
                $vsearchQuery = "$threadTitle $firstSentence"
                # Truncate vsearch query to 120 chars
                if ($vsearchQuery.Length -gt 120) { $vsearchQuery = $vsearchQuery.Substring(0, 120) }
```

**Replace with:**
```powershell
                # Extract first sentence of description, strip status noise for vsearch
                $firstSentence = $threadDesc
                if ($threadDesc -match '^([^.]+\.)') { $firstSentence = $Matches[1] }
                # Remove status words that dilute BM25 scoring
                $cleanDesc = $firstSentence -replace '\b(DONE|IN PROGRESS|COMPLETE|TODO|BLOCKED|DEFERRED)\b', '' `
                    -replace '\bPhase\s+\d+(\.\d+)?\b', '' `
                    -replace '\(\s*\)', '' `
                    -replace '\s+', ' '
                $vsearchQuery = "$threadTitle $cleanDesc".Trim()
                if ($vsearchQuery.Length -gt 120) { $vsearchQuery = $vsearchQuery.Substring(0, 120) }
```

### Step 3: Preserve periods in title word extraction (modify line 264)

**Current code (line 264):**
```powershell
                $titleWords = $threadTitle -replace '[^\w\s]', '' -replace '\s+', ' '
```

**Replace with:**
```powershell
                $titleWords = $threadTitle -replace '[^\w\s.]', '' -replace '\s+', ' '
```

### Step 4: Replace `& qmd search` with `& $nodeExe $qmdScript search` (modify line 267)

**Current code (line 267):**
```powershell
                        $kwRaw = & qmd search $titleWords --limit 3 --json 2>&1
```

**Replace with:**
```powershell
                        $kwRaw = & $nodeExe $qmdScript search $titleWords --limit 3 --json 2>&1
```

### Step 5: Fix keyword JSON parsing to handle stderr (modify lines 268-269)

**Current code (lines 268-269):**
```powershell
                        $kwJson = ($kwRaw | Out-String).Trim()
                        if ($kwJson -match '^\[') {
```

**Replace with:**
```powershell
                        $kwString = ($kwRaw | Out-String)
                        $kwBracketIdx = $kwString.IndexOf('[')
                        if ($kwBracketIdx -ge 0) {
                            $kwJson = $kwString.Substring($kwBracketIdx)
```

Note: this adds one level of nesting. The `$kwResults = $kwJson | ConvertFrom-Json` on the next line and the `foreach` loop that follows remain unchanged, but they now sit inside this `if` block. The closing `}` for the old `if ($kwJson -match '^\[')` block (which was at line 279 just before `catch`) should close this new `if` block instead.

### Step 6: Replace `& qmd vsearch` with `& $nodeExe $qmdScript vsearch` (modify line 285)

**Current code (line 285):**
```powershell
                    $vsRaw = & qmd vsearch $vsearchQuery --limit 3 --json 2>&1
```

**Replace with:**
```powershell
                    $vsRaw = & $nodeExe $qmdScript vsearch $vsearchQuery --limit 3 --json 2>&1
```

---

## Complete Replacement Block

For clarity, here is the full replacement of lines 226-301 (the knowledge activation try block and search logic). Everything before line 222 and after line 339 remains unchanged.

**Replace lines 226-301 with:**

```powershell
    try {
        # Resolve qmd via node directly (npm shim is broken on Windows — /bin/sh.exe not found)
        $qmdScript = Join-Path $env:APPDATA "npm\node_modules\@tobilu\qmd\dist\cli\qmd.js"
        if (-not (Test-Path $qmdScript)) {
            throw "qmd not found at $qmdScript"
        }
        $nodeExe = "node"
        $env:INDEX_PATH = Join-Path $env:USERPROFILE ".cache\qmd\index.sqlite"

        # Parse thread titles from goalLines
        $threads = @()
        foreach ($gl in $goalLines) {
            $trimGl = $gl.Trim()
            # Pattern: - **Title** -- description  OR  - Title -- description
            if ($trimGl -match '^\s*-\s+\*\*(.+?)\*\*\s*--\s*(.+)$') {
                $threads += @{ Title = $Matches[1].Trim(); Desc = $Matches[2].Trim() }
            } elseif ($trimGl -match '^\s*-\s+(.+?)\s*--\s*(.+)$') {
                $threads += @{ Title = $Matches[1].Trim(); Desc = $Matches[2].Trim() }
            }
            if ($threads.Count -ge 5) { break }
        }

        if ($threads.Count -eq 0) {
            $relevanceContent = "# Session Relevance Brief`n<!-- Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm') -->`n<!-- Threads: 0 active from goals.md -->`n`nNo active threads found in goals.md.`n"
            Set-Content -Path $relevanceFile -Value $relevanceContent -Encoding UTF8
            $activationSuccess = $true
        } else {
            $allThreadResults = @{}
            $totalNotes = 0

            foreach ($thread in $threads) {
                $threadTitle = $thread.Title
                $threadDesc = $thread.Desc
                $threadNotes = @{}  # key=note title, value=@{Title; Score; Source}

                # Extract first sentence of description, strip status noise for vsearch
                $firstSentence = $threadDesc
                if ($threadDesc -match '^([^.]+\.)') { $firstSentence = $Matches[1] }
                # Remove status words that dilute BM25 scoring
                $cleanDesc = $firstSentence -replace '\b(DONE|IN PROGRESS|COMPLETE|TODO|BLOCKED|DEFERRED)\b', '' `
                    -replace '\bPhase\s+\d+(\.\d+)?\b', '' `
                    -replace '\(\s*\)', '' `
                    -replace '\s+', ' '
                $vsearchQuery = "$threadTitle $cleanDesc".Trim()
                if ($vsearchQuery.Length -gt 120) { $vsearchQuery = $vsearchQuery.Substring(0, 120) }

                # --- Keyword search ---
                $titleWords = $threadTitle -replace '[^\w\s.]', '' -replace '\s+', ' '
                if ($titleWords.Trim().Split(' ').Count -ge 2) {
                    try {
                        $kwRaw = & $nodeExe $qmdScript search $titleWords --limit 3 --json 2>&1
                        $kwString = ($kwRaw | Out-String)
                        $kwBracketIdx = $kwString.IndexOf('[')
                        if ($kwBracketIdx -ge 0) {
                            $kwJson = $kwString.Substring($kwBracketIdx)
                            $kwResults = $kwJson | ConvertFrom-Json
                            foreach ($r in $kwResults) {
                                if ($r.score -ge 0.1 -and $r.title) {
                                    $noteKey = $r.title
                                    if (-not $threadNotes.ContainsKey($noteKey) -or $threadNotes[$noteKey].Score -lt $r.score) {
                                        $threadNotes[$noteKey] = @{ Title = $r.title; Score = $r.score; Source = 'keyword' }
                                    }
                                }
                            }
                        }
                    } catch {}
                }

                # --- Vector search ---
                try {
                    $vsRaw = & $nodeExe $qmdScript vsearch $vsearchQuery --limit 3 --json 2>&1
                    $vsString = ($vsRaw | Out-String)
                    # Extract JSON array: find first '[' and parse from there
                    $bracketIdx = $vsString.IndexOf('[')
                    if ($bracketIdx -ge 0) {
                        $vsJsonStr = $vsString.Substring($bracketIdx)
                        $vsResults = $vsJsonStr | ConvertFrom-Json
                        foreach ($r in $vsResults) {
                            if ($r.score -ge 0.3 -and $r.title) {
                                $noteKey = $r.title
                                if (-not $threadNotes.ContainsKey($noteKey) -or $threadNotes[$noteKey].Score -lt $r.score) {
                                    $threadNotes[$noteKey] = @{ Title = $r.title; Score = $r.score; Source = 'vector' }
                                }
                            }
                        }
                    }
                } catch {}
```

Lines 302-339 (note sorting, relevance file building, output) remain **unchanged**.

---

## Validation

After making the changes:

1. **Syntax check:** Run `powershell.exe -NoProfile -Command "& { . '.claude/hooks/session-orient.ps1' }"` — should not throw parse errors (it will produce output or fail gracefully at runtime since it needs specific paths).

2. **Functional test (manual):** Run the hook directly:
   ```powershell
   powershell.exe -NoProfile -NonInteractive -ExecutionPolicy Bypass -File .claude/hooks/session-orient.ps1
   ```
   Expected output should include `Knowledge activation: N notes surfaced for 2 threads` (where N > 0) instead of the current silent 0.

3. **Verify session-relevance.md:** After running, check `ops/session-relevance.md` — it should contain actual note titles with scores, not "No strong matches."

4. **Regression check:** Ensure the other hook sections still work:
   - Goals section shows "Current Goals:"
   - Reminders section works
   - Task section works
   - Vault counts still display
   - TRIGGER warnings still fire
   - Lifecycle archiving still works

5. **Edge case: qmd not installed.** If `$qmdScript` path doesn't exist, the `throw` on the new line 4 is caught by the outer `catch` on line 335, which writes a graceful "qmd unavailable" fallback. This is the same behavior as before.

---

## What NOT to Change

- **Lines 1-221:** Goals, reminders, last session, tasks, vault counts, triggers — all working correctly
- **Lines 302-339:** Note sorting, relevance file building, output formatting — all correct
- **Lines 341-493:** Lifecycle archiving — all working correctly
- **The `.claude/settings.local.json` hook configuration** — no changes needed
- **The `session-orient.cmd` wrapper** — no changes needed

---

## Upstream Issues to Report (informational, not part of this fix)

These are qmd bugs discovered during the investigation that should be reported to the qmd maintainer:

1. **npm shim broken on Windows** — `qmd.ps1` tries to invoke `/bin/sh.exe`
2. **Hyphen-as-negation in vec/hyde** — "VQ-VAE" parsed as "VQ minus VAE"
3. **Parallel vec query crashes** — "Object is disposed" error under concurrent load
4. **Uniform 0.93 scoring** — all queries return same top score regardless of relevance

These are NOT blockers for this fix (the fix works around bug 1; bugs 2-4 affect vec search which is a nice-to-have, not critical).

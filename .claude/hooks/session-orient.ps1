# Hook: Session orientation - show vault state at session start
# Triggered on: SessionStart (no stdin)
# Outputs high-signal context: goals, reminders (with overdue detection),
# last session summary, pending tasks, filtered observation/tension counts.

$ErrorActionPreference = 'SilentlyContinue'

try {
    $VaultRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

    # Guard: only run in arscontexta vaults
    if (-not (Test-Path (Join-Path $VaultRoot ".arscontexta"))) { [Environment]::Exit(0) }

    $today = Get-Date -Format "yyyy-MM-dd"
    $todayDate = [datetime]::ParseExact($today, "yyyy-MM-dd", $null)

    Write-Host "=== Session Orient ==="

    # --- Active Goals ---
    $goalsFile = Join-Path $VaultRoot "ops\goals.md"
    if (Test-Path $goalsFile) {
        $lines = Get-Content $goalsFile -ErrorAction SilentlyContinue
        $inSection = $false
        $goalLines = @()
        foreach ($line in $lines) {
            if ($line -match "^## Active Threads") {
                $inSection = $true
                continue
            }
            if ($inSection) {
                if ($line -match "^## " -and $line -notmatch "^## Active Threads") { break }
                if ($line.Trim()) { $goalLines += $line }
            }
        }
        if ($goalLines.Count -gt 0) {
            Write-Host ""
            Write-Host "Current Goals:"
            $goalLines | ForEach-Object { Write-Host $_ }
        }
    }

    # --- Reminders with overdue detection ---
    $remindersFile = Join-Path $VaultRoot "ops\reminders.md"
    if (Test-Path $remindersFile) {
        $content = Get-Content $remindersFile -ErrorAction SilentlyContinue
        $overdueItems = @()
        $soonItems = @()
        $futureCount = 0

        foreach ($line in $content) {
            # Match unchecked reminders: - [ ] YYYY-MM-DD: Description
            if ($line -match "^- \[ \] (\d{4}-\d{2}-\d{2}):\s*(.+)$") {
                $dateStr = $Matches[1]
                $desc = $Matches[2]
                try {
                    $reminderDate = [datetime]::ParseExact($dateStr, "yyyy-MM-dd", $null)
                    $daysUntil = ($reminderDate - $todayDate).Days

                    if ($daysUntil -lt 0) {
                        $daysAgo = [Math]::Abs($daysUntil)
                        $overdueItems += "  OVERDUE ($daysAgo days): $desc"
                    } elseif ($daysUntil -le 3) {
                        if ($daysUntil -eq 0) {
                            $soonItems += "  DUE TODAY: $desc"
                        } else {
                            $soonItems += "  DUE in $daysUntil days: $desc"
                        }
                    } else {
                        $futureCount++
                    }
                } catch {
                    # Unparseable date - count as future
                    $futureCount++
                }
            }
        }

        if ($overdueItems.Count -gt 0 -or $soonItems.Count -gt 0 -or $futureCount -gt 0) {
            Write-Host ""
            Write-Host "Reminders:"
            foreach ($item in $overdueItems) { Write-Host $item }
            foreach ($item in $soonItems) { Write-Host $item }
            if ($futureCount -gt 0) {
                Write-Host "  $futureCount more scheduled (distant)"
            }
        }
    }

    # --- Last Session Summary ---
    $lastSessionFile = Join-Path $VaultRoot "ops\last-session.md"
    if (Test-Path $lastSessionFile) {
        $lsContent = Get-Content $lastSessionFile -Raw -ErrorAction SilentlyContinue
        if ($lsContent -and $lsContent.Trim()) {
            Write-Host ""
            Write-Host "Last session:"
            # Show content lines (skip frontmatter if any, skip blank lines at top)
            $lsLines = $lsContent -split "`n"
            $inFrontmatter = $false
            $shown = 0
            foreach ($line in $lsLines) {
                if ($line.Trim() -eq "---" -and $shown -eq 0) {
                    $inFrontmatter = -not $inFrontmatter
                    continue
                }
                if (-not $inFrontmatter -and $line.Trim() -and $shown -lt 5) {
                    Write-Host "  $($line.Trim())"
                    $shown++
                }
            }
        }
    }

    # --- Pending Tasks ---
    $tasksFile = Join-Path $VaultRoot "ops\tasks.md"
    if (Test-Path $tasksFile) {
        $taskContent = Get-Content $tasksFile -ErrorAction SilentlyContinue
        $pendingTasks = @()
        $inProgressTasks = @()
        $inPending = $false
        $inInProgress = $false

        foreach ($line in $taskContent) {
            if ($line -match "^## Pending") { $inPending = $true; $inInProgress = $false; continue }
            if ($line -match "^## In Progress") { $inInProgress = $true; $inPending = $false; continue }
            if ($line -match "^## ") { $inPending = $false; $inInProgress = $false; continue }

            $trimmed = $line.Trim()
            if ($trimmed -and $trimmed -ne "(none)") {
                if ($inPending -and $trimmed -match "^- ") { $pendingTasks += $trimmed }
                if ($inInProgress -and $trimmed -match "^- ") { $inProgressTasks += $trimmed }
            }
        }

        if ($inProgressTasks.Count -gt 0 -or $pendingTasks.Count -gt 0) {
            Write-Host ""
            Write-Host "Tasks:"
            foreach ($t in $inProgressTasks) { Write-Host "  [IN PROGRESS] $($t -replace '^- ','')" }
            foreach ($t in $pendingTasks) { Write-Host "  [PENDING] $($t -replace '^- ','')" }
        }
    }

    # --- Vault Counts (with status-filtered observations/tensions) ---
    $notesDir = Join-Path $VaultRoot "notes"
    $inboxDir = Join-Path $VaultRoot "inbox"
    $obsDir = Join-Path $VaultRoot "ops\observations"
    $tensionDir = Join-Path $VaultRoot "ops\tensions"

    $notesCount = 0
    $inboxCount = 0
    $pendingObsCount = 0
    $pendingTensionCount = 0

    if (Test-Path $notesDir) {
        $notesCount = (Get-ChildItem -Path $notesDir -Recurse -Filter "*.md" -ErrorAction SilentlyContinue | Measure-Object).Count
    }
    if (Test-Path $inboxDir) {
        $inboxCount = (Get-ChildItem -Path $inboxDir -Recurse -Filter "*.md" -ErrorAction SilentlyContinue | Measure-Object).Count
    }

    # Parse frontmatter status for observations
    if (Test-Path $obsDir) {
        $obsFiles = Get-ChildItem -Path $obsDir -Recurse -Filter "*.md" -ErrorAction SilentlyContinue
        foreach ($f in $obsFiles) {
            $fc = Get-Content $f.FullName -Raw -ErrorAction SilentlyContinue
            if ($fc -match "(?ms)^---\s*\n(.+?)\n---") {
                $fm = $Matches[1]
                if ($fm -match "status:\s*pending") { $pendingObsCount++ }
            }
        }
    }

    # Parse frontmatter status for tensions
    if (Test-Path $tensionDir) {
        $tensionFiles = Get-ChildItem -Path $tensionDir -Recurse -Filter "*.md" -ErrorAction SilentlyContinue
        foreach ($f in $tensionFiles) {
            $fc = Get-Content $f.FullName -Raw -ErrorAction SilentlyContinue
            if ($fc -match "(?ms)^---\s*\n(.+?)\n---") {
                $fm = $Matches[1]
                if ($fm -match "status:\s*pending") { $pendingTensionCount++ }
            }
        }
    }

    Write-Host ""
    Write-Host "Vault: $notesCount notes | $inboxCount inbox | $pendingObsCount pending observations | $pendingTensionCount pending tensions"

    # --- Trigger warnings (thresholds from queue.json) ---
    $inboxThreshold = 3
    $obsThreshold = 7
    $obsMaxAgeDays = 14
    $tensionThreshold = 5

    $queueFile = Join-Path $VaultRoot "ops\queue\queue.json"
    if (Test-Path $queueFile) {
        try {
            $queueData = Get-Content $queueFile -Raw -ErrorAction SilentlyContinue | ConvertFrom-Json
            if ($queueData.maintenance_conditions) {
                $mc = $queueData.maintenance_conditions
                if ($mc.inbox_threshold) { $inboxThreshold = $mc.inbox_threshold }
                if ($mc.observation_threshold) { $obsThreshold = $mc.observation_threshold }
                if ($mc.observation_max_age_days) { $obsMaxAgeDays = $mc.observation_max_age_days }
                if ($mc.tension_threshold) { $tensionThreshold = $mc.tension_threshold }
            }
        } catch {}
    }

    if ($inboxCount -ge $inboxThreshold) { Write-Host "TRIGGER: Inbox has $inboxCount items (threshold: $inboxThreshold)" }
    if ($pendingObsCount -ge $obsThreshold) { Write-Host "TRIGGER: $pendingObsCount pending observations (threshold: $obsThreshold)" }
    if ($pendingTensionCount -ge $tensionThreshold) { Write-Host "TRIGGER: $pendingTensionCount pending tensions (threshold: $tensionThreshold)" }

    # Time-based observation trigger: any pending observation older than obsMaxAgeDays
    if ($obsMaxAgeDays -gt 0 -and (Test-Path $obsDir)) {
        $cutoff = (Get-Date).AddDays(-$obsMaxAgeDays)
        $staleObs = Get-ChildItem -Path $obsDir -Filter "*.md" -ErrorAction SilentlyContinue | Where-Object {
            $_.LastWriteTime -lt $cutoff -and (Select-String -Path $_.FullName -Pattern "^status:\s*pending" -Quiet -ErrorAction SilentlyContinue)
        }
        if ($staleObs -and $staleObs.Count -gt 0) {
            Write-Host "TRIGGER: $($staleObs.Count) observation(s) older than $obsMaxAgeDays days without review - run /rethink"
        }
    }

    # --- Knowledge Activation: Surface relevant vault notes per active thread ---
    $relevanceFile = Join-Path $VaultRoot "ops\session-relevance.md"
    $activationSuccess = $false

    try {
        # Resolve qmd via node directly (npm shim is broken on Windows - /bin/sh.exe not found)
        $qmdScript = Join-Path $env:APPDATA "npm\node_modules\@tobilu\qmd\dist\cli\qmd.js"
        if (-not (Test-Path $qmdScript)) {
            throw "qmd not found at $qmdScript"
        }
        $nodeExe = "node"

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

                # Cap at 4 notes per thread, sorted by score descending
                $sortedNotes = $threadNotes.Values | Sort-Object { $_.Score } -Descending | Select-Object -First 4
                $allThreadResults[$threadTitle] = $sortedNotes
                $totalNotes += $sortedNotes.Count
            }

            # Build relevance file content
            $sb = [System.Text.StringBuilder]::new()
            [void]$sb.AppendLine("# Session Relevance Brief")
            [void]$sb.AppendLine("<!-- Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm') -->")
            [void]$sb.AppendLine("<!-- Threads: $($threads.Count) active from goals.md -->")
            [void]$sb.AppendLine("")

            foreach ($thread in $threads) {
                $threadTitle = $thread.Title
                [void]$sb.AppendLine("## $threadTitle")

                $notes = $allThreadResults[$threadTitle]
                if ($notes -and $notes.Count -gt 0) {
                    foreach ($n in $notes) {
                        [void]$sb.AppendLine("- **$($n.Title)** (score: $($n.Score), $($n.Source))")
                    }
                } else {
                    [void]$sb.AppendLine("- No strong matches above relevance threshold.")
                }
                [void]$sb.AppendLine("")
            }

            Set-Content -Path $relevanceFile -Value $sb.ToString() -Encoding UTF8
            $activationSuccess = $true
            Write-Host "Knowledge activation: $totalNotes notes surfaced for $($threads.Count) threads"
        }
    } catch {
        # qmd unavailable or other error - write graceful fallback
        $relevanceContent = "# Session Relevance Brief`n<!-- Generated: $(Get-Date -Format 'yyyy-MM-dd HH:mm') -->`n`nqmd unavailable - knowledge activation skipped.`n"
        Set-Content -Path $relevanceFile -Value $relevanceContent -Encoding UTF8
    }

    # --- Lifecycle: Archive old completed goals ---
    if (Test-Path $goalsFile) {
        $gLines = Get-Content $goalsFile -ErrorAction SilentlyContinue
        $completedSection = @()
        $inCompleted = $false
        $completedStart = -1
        $completedEnd = $gLines.Count

        for ($i = 0; $i -lt $gLines.Count; $i++) {
            if ($gLines[$i] -match "^## Completed") {
                $inCompleted = $true
                $completedStart = $i + 1
                continue
            }
            if ($inCompleted -and $gLines[$i] -match "^## ") {
                $completedEnd = $i
                $inCompleted = $false
                break
            }
            if ($inCompleted -and $gLines[$i].Trim() -match "^- ") {
                $completedSection += $gLines[$i]
            }
        }

        $maxCompleted = 15
        if ($completedSection.Count -gt $maxCompleted) {
            $toArchive = $completedSection[0..($completedSection.Count - $maxCompleted - 1)]
            $toKeep = $completedSection[($completedSection.Count - $maxCompleted)..($completedSection.Count - 1)]

            # Write archived items
            $archiveFile = Join-Path $VaultRoot "ops\goals-archive.md"
            $archiveHeader = "# Archived Completed Goals`n`n"
            if (Test-Path $archiveFile) {
                $existingArchive = Get-Content $archiveFile -Raw -ErrorAction SilentlyContinue
                $archiveContent = $existingArchive.TrimEnd() + "`n" + ($toArchive -join "`n") + "`n"
            } else {
                $archiveContent = $archiveHeader + ($toArchive -join "`n") + "`n"
            }
            Set-Content -Path $archiveFile -Value $archiveContent -Encoding UTF8

            # Rewrite goals.md with trimmed completed section
            $newGoals = @()
            for ($i = 0; $i -lt $gLines.Count; $i++) {
                if ($i -ge $completedStart -and $i -lt $completedEnd) { continue }
                $newGoals += $gLines[$i]
                if ($gLines[$i] -match "^## Completed") {
                    $toKeep | ForEach-Object { $newGoals += $_ }
                }
            }
            Set-Content -Path $goalsFile -Value ($newGoals -join "`n") -Encoding UTF8
            Write-Host "LIFECYCLE: Archived $($toArchive.Count) old completed goals to ops/goals-archive.md"
        }
    }

    # --- Lifecycle: Archive old completed tasks ---
    if (Test-Path $tasksFile) {
        $tLines = Get-Content $tasksFile -ErrorAction SilentlyContinue
        $completedTasks = @()
        $inCompletedSection = $false

        foreach ($line in $tLines) {
            if ($line -match "^## Completed") { $inCompletedSection = $true; continue }
            if ($line -match "^## " -and $inCompletedSection) { $inCompletedSection = $false; continue }
            if ($inCompletedSection -and $line.Trim() -match "^- ") {
                $completedTasks += $line
            }
        }

        # Check for date-stamped completed tasks older than 7 days
        $staleCompletedTasks = @()
        $freshCompletedTasks = @()
        foreach ($task in $completedTasks) {
            if ($task -match "\(done\s+(\d{4}-\d{2}-\d{2})\)") {
                try {
                    $doneDate = [datetime]::ParseExact($Matches[1], "yyyy-MM-dd", $null)
                    if (($todayDate - $doneDate).Days -gt 7) {
                        $staleCompletedTasks += $task
                    } else {
                        $freshCompletedTasks += $task
                    }
                } catch {
                    $freshCompletedTasks += $task
                }
            } else {
                # No date stamp - keep in completed (can't determine age)
                $freshCompletedTasks += $task
            }
        }

        if ($staleCompletedTasks.Count -gt 0) {
            $taskArchiveFile = Join-Path $VaultRoot "ops\tasks-archive.md"
            $archiveHeader = "# Archived Completed Tasks`n`n"
            if (Test-Path $taskArchiveFile) {
                $existingArchive = Get-Content $taskArchiveFile -Raw -ErrorAction SilentlyContinue
                $archiveContent = $existingArchive.TrimEnd() + "`n" + ($staleCompletedTasks -join "`n") + "`n"
            } else {
                $archiveContent = $archiveHeader + ($staleCompletedTasks -join "`n") + "`n"
            }
            Set-Content -Path $taskArchiveFile -Value $archiveContent -Encoding UTF8

            # Rewrite tasks.md with only fresh completed tasks
            $newTaskLines = @()
            $skipCompleted = $false
            foreach ($line in $tLines) {
                if ($line -match "^## Completed") {
                    $newTaskLines += $line
                    $skipCompleted = $true
                    if ($freshCompletedTasks.Count -gt 0) {
                        $freshCompletedTasks | ForEach-Object { $newTaskLines += $_ }
                    } else {
                        $newTaskLines += "(none)"
                    }
                    continue
                }
                if ($skipCompleted -and $line -match "^## ") { $skipCompleted = $false }
                if (-not $skipCompleted) { $newTaskLines += $line }
            }
            Set-Content -Path $tasksFile -Value ($newTaskLines -join "`n") -Encoding UTF8
            Write-Host "LIFECYCLE: Archived $($staleCompletedTasks.Count) old completed tasks to ops/tasks-archive.md"
        }
    }

    # --- Lifecycle: Archive old completed reminders ---
    if (Test-Path $remindersFile) {
        $rLines = Get-Content $remindersFile -ErrorAction SilentlyContinue
        $staleReminders = @()
        $freshLines = @()

        foreach ($line in $rLines) {
            if ($line -match "^- \[x\].*\(done\s+(\d{4}-\d{2}-\d{2})\)") {
                try {
                    $doneDate = [datetime]::ParseExact($Matches[1], "yyyy-MM-dd", $null)
                    if (($todayDate - $doneDate).Days -gt 14) {
                        $staleReminders += $line
                        continue
                    }
                } catch {}
            }
            $freshLines += $line
        }

        if ($staleReminders.Count -gt 0) {
            Set-Content -Path $remindersFile -Value ($freshLines -join "`n") -Encoding UTF8
            Write-Host "LIFECYCLE: Removed $($staleReminders.Count) old completed reminders"
        }
    }

    Write-Host "=== End Orient ==="
} catch {
    [Console]::Error.WriteLine("[HOOK session-orient] $($_.Exception.Message)")
}

[Environment]::Exit(0)

# Hook: Session capture - save session metadata and bridging context on Stop
# Triggered on: Stop event (receives stdin JSON with transcript_path)
# Writes: ops/sessions/<id>.json (session metadata)
#         ops/last-session.md (bridging context for next session's orient hook)
# Also checks for State Update Rule compliance (goals.md update after substantial work).

$ErrorActionPreference = 'SilentlyContinue'

try {
    $raw = [Console]::In.ReadToEnd()
    if (-not $raw) { [Environment]::Exit(0) }

    $input_json = $raw | ConvertFrom-Json
    if (-not $input_json) { [Environment]::Exit(0) }

    # Only create session file if we have a transcript to reference
    if (-not $input_json.transcript_path) { [Environment]::Exit(0) }

    $VaultRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).ProviderPath
    $WslVaultRoot = "/home/shachar/projects/mickey_london_lab"

    function Invoke-RepoGit {
        param(
            [Parameter(ValueFromRemainingArguments = $true)]
            [string[]]$GitArgs
        )

        & wsl.exe -d Ubuntu --cd $WslVaultRoot git @GitArgs 2>&1
    }

    # Guard: only run in arscontexta vaults
    if (-not (Test-Path (Join-Path $VaultRoot ".arscontexta"))) { [Environment]::Exit(0) }

    $sessionDir = Join-Path $VaultRoot "ops\sessions"
    if (-not (Test-Path $sessionDir)) {
        New-Item -ItemType Directory -Path $sessionDir -Force | Out-Null
    }

    # Use conversation ID from env or generate timestamp-based ID
    $sessionId = $env:CLAUDE_CONVERSATION_ID
    if (-not $sessionId) {
        $sessionId = Get-Date -Format "yyyyMMdd-HHmmss"
    }

    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"

    # --- Save session metadata JSON ---
    $sessionFile = Join-Path $sessionDir "$sessionId.json"
    $sessionData = @{
        session_id      = $sessionId
        timestamp       = $timestamp
        transcript_path = $input_json.transcript_path
        status          = "completed"
    } | ConvertTo-Json
    Set-Content -Path $sessionFile -Value $sessionData -Encoding UTF8

    # --- Write ops/last-session.md (bridging context for orient hook) ---
    $lastSessionFile = Join-Path $VaultRoot "ops\last-session.md"

    # Collect git-based session summary
    $summary = @()
    $summary += "Session $sessionId ($timestamp)"

    # What files changed since last commit? (unstaged + staged)
    try {
        $gitStatus = Invoke-RepoGit status --porcelain
        if ($gitStatus) {
            $changedFiles = @($gitStatus)
            $srcChanges = @($changedFiles | Where-Object { $_ -match "src/" })
            $testChanges = @($changedFiles | Where-Object { $_ -match "tests/" })
            $noteChanges = @($changedFiles | Where-Object { $_ -match "notes/" })
            $opsChanges = @($changedFiles | Where-Object { $_ -match "ops/" })

            $parts = @()
            if ($srcChanges.Count -gt 0) { $parts += "$($srcChanges.Count) src" }
            if ($testChanges.Count -gt 0) { $parts += "$($testChanges.Count) tests" }
            if ($noteChanges.Count -gt 0) { $parts += "$($noteChanges.Count) notes" }
            if ($opsChanges.Count -gt 0) { $parts += "$($opsChanges.Count) ops" }
            $otherCount = $changedFiles.Count - $srcChanges.Count - $testChanges.Count - $noteChanges.Count - $opsChanges.Count
            if ($otherCount -gt 0) { $parts += "$otherCount other" }

            if ($parts.Count -gt 0) {
                $summary += "Files changed: $($parts -join ', ') ($($changedFiles.Count) total)"
            }
        } else {
            $summary += "No uncommitted changes at session end."
        }
    } catch {
        $summary += "(git status unavailable)"
    }

    # Recent commits from this session (last 3, in case auto-commit ran)
    try {
        $recentCommits = Invoke-RepoGit log --oneline -3 --since="2 hours ago"
        if ($recentCommits -and $recentCommits -notmatch "fatal:") {
            $summary += "Recent commits:"
            foreach ($c in $recentCommits) {
                $summary += "  $c"
            }
        }
    } catch {}

    # Check goals.md for any changes (did active threads shift?)
    try {
        $goalsChanged = Invoke-RepoGit diff --name-only HEAD -- "ops/goals.md"
        $goalsStagedChanged = Invoke-RepoGit diff --name-only --cached -- "ops/goals.md"
        if ($goalsChanged -or $goalsStagedChanged) {
            $summary += "Goals updated: yes"
        }
    } catch {}

    Set-Content -Path $lastSessionFile -Value ($summary -join "`n") -Encoding UTF8

    # Codex does not own ops/goals.md. Claude's copy of this hook enforces the
    # state update rule; the Codex copy only records session context.

} catch {
    [Console]::Error.WriteLine("[HOOK session-capture] $($_.Exception.Message)")
}

[Environment]::Exit(0)

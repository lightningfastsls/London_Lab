# Hook: Session orientation - show vault state at session start
# Triggered on: SessionStart (no stdin)

$ErrorActionPreference = 'SilentlyContinue'

try {
    $VaultRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

    # Guard: only run in arscontexta vaults
    if (-not (Test-Path (Join-Path $VaultRoot ".arscontexta"))) { [Environment]::Exit(0) }

    Write-Host "=== Session Orient ==="

    # Show active goals
    $goalsFile = Join-Path $VaultRoot "ops\goals.md"
    if (Test-Path $goalsFile) {
        $lines = Get-Content $goalsFile -ErrorAction SilentlyContinue
        $inSection = $false
        $count = 0
        $goalLines = @()
        foreach ($line in $lines) {
            if ($line -match "^## Active Threads") {
                $inSection = $true
                continue
            }
            if ($inSection) {
                if ($line -match "^## " -and $line -notmatch "^## Active Threads") { break }
                $goalLines += $line
                $count++
                if ($count -ge 15) { break }
            }
        }
        if ($goalLines.Count -gt 0) {
            Write-Host ""
            Write-Host "Current Goals:"
            $goalLines | ForEach-Object { Write-Host $_ }
        }
    }

    # Check reminders
    $remindersFile = Join-Path $VaultRoot "ops\reminders.md"
    if (Test-Path $remindersFile) {
        $content = Get-Content $remindersFile -ErrorAction SilentlyContinue
        $overdue = ($content | Select-String -Pattern "^- \[ \]" -ErrorAction SilentlyContinue | Measure-Object).Count
        if ($overdue -gt 0) {
            Write-Host ""
            Write-Host "Reminders: $overdue pending"
        }
    }

    # Count vault items
    $notesDir = Join-Path $VaultRoot "notes"
    $inboxDir = Join-Path $VaultRoot "inbox"
    $obsDir = Join-Path $VaultRoot "ops\observations"
    $tensionDir = Join-Path $VaultRoot "ops\tensions"

    $notesCount = 0
    $inboxCount = 0
    $obsCount = 0
    $tensionCount = 0

    if (Test-Path $notesDir) {
        $notesCount = (Get-ChildItem -Path $notesDir -Recurse -Filter "*.md" -ErrorAction SilentlyContinue | Measure-Object).Count
    }
    if (Test-Path $inboxDir) {
        $inboxCount = (Get-ChildItem -Path $inboxDir -Recurse -Filter "*.md" -ErrorAction SilentlyContinue | Measure-Object).Count
    }
    if (Test-Path $obsDir) {
        $obsCount = (Get-ChildItem -Path $obsDir -Recurse -Filter "*.md" -ErrorAction SilentlyContinue | Measure-Object).Count
    }
    if (Test-Path $tensionDir) {
        $tensionCount = (Get-ChildItem -Path $tensionDir -Recurse -Filter "*.md" -ErrorAction SilentlyContinue | Measure-Object).Count
    }

    Write-Host ""
    Write-Host "Vault: $notesCount notes | $inboxCount inbox | $obsCount observations | $tensionCount tensions"

    # Trigger warnings
    if ($inboxCount -ge 3) { Write-Host "TRIGGER: Inbox has $inboxCount items" }
    if ($obsCount -ge 10) { Write-Host "TRIGGER: $obsCount pending observations" }
    if ($tensionCount -ge 5) { Write-Host "TRIGGER: $tensionCount open tensions" }

    Write-Host "=== End Orient ==="
} catch {
    [Console]::Error.WriteLine("[HOOK session-orient] $($_.Exception.Message)")
}

[Environment]::Exit(0)

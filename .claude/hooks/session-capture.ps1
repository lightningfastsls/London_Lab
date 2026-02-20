# Hook: Session capture - save session metadata on Stop
# Triggered on: Stop event (receives stdin JSON with transcript_path)
# Only creates a file when transcript_path is present, avoiding empty stubs.

$ErrorActionPreference = 'SilentlyContinue'

try {
    $raw = [Console]::In.ReadToEnd()
    if (-not $raw) { [Environment]::Exit(0) }

    $input_json = $raw | ConvertFrom-Json
    if (-not $input_json) { [Environment]::Exit(0) }

    # Only create session file if we have a transcript to reference
    if (-not $input_json.transcript_path) { [Environment]::Exit(0) }

    $VaultRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

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

    $sessionFile = Join-Path $sessionDir "$sessionId.json"

    $timestamp = Get-Date -Format "yyyy-MM-ddTHH:mm:ss"
    $sessionData = @{
        session_id      = $sessionId
        timestamp       = $timestamp
        transcript_path = $input_json.transcript_path
        status          = "completed"
    } | ConvertTo-Json
    Set-Content -Path $sessionFile -Value $sessionData -Encoding UTF8
} catch {
    [Console]::Error.WriteLine("[HOOK session-capture] $($_.Exception.Message)")
}

[Environment]::Exit(0)

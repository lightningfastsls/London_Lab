# Hook: Check if response ends with **Agents:** tag
# Triggered on: Stop event

$ErrorActionPreference = 'SilentlyContinue'

try {
    $raw = [Console]::In.ReadToEnd()
    if (-not $raw) { [Environment]::Exit(0) }

    $input_json = $raw | ConvertFrom-Json
    if (-not $input_json -or -not $input_json.transcript_path) { [Environment]::Exit(0) }

    $transcript = Get-Content -Raw $input_json.transcript_path -ErrorAction SilentlyContinue
    if ($transcript -and $transcript -notmatch '\*\*Agents:\*\*') {
        Write-Host "[HOOK] Remember to end your response with: **Agents:** [list or None]"
    }
} catch {
    [Console]::Error.WriteLine("[HOOK check_agents_tag] $($_.Exception.Message)")
}

[Environment]::Exit(0)

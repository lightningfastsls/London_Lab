# Hook: Remind about plan mode before editing code files
# Triggered on: PreToolUse (Edit|Write)

$ErrorActionPreference = 'SilentlyContinue'

try {
    $raw = [Console]::In.ReadToEnd()
    if (-not $raw) { [Environment]::Exit(0) }

    $input_json = $raw | ConvertFrom-Json
    if (-not $input_json -or -not $input_json.tool_input -or -not $input_json.tool_input.file_path) { [Environment]::Exit(0) }

    $path = $input_json.tool_input.file_path

    # Only remind for Python files, not plan files
    if ($path -match '\.py$' -and $path -notmatch 'plans[/\\]') {
        Write-Host "[HOOK] Editing code file - did you use EnterPlanMode for non-trivial tasks?"
    }
} catch {
    [Console]::Error.WriteLine("[HOOK check_plan_mode] $($_.Exception.Message)")
}

[Environment]::Exit(0)

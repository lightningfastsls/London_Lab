# Hook: Remind about plan mode before editing code files
# Triggered on: PreToolUse (Edit|Write)

$input_json = [Console]::In.ReadToEnd() | ConvertFrom-Json -ErrorAction SilentlyContinue

if ($input_json -and $input_json.tool_input -and $input_json.tool_input.file_path) {
    $path = $input_json.tool_input.file_path

    # Only remind for Python files, not plan files
    if ($path -match '\.py$' -and $path -notmatch 'plans[/\\]') {
        Write-Host "[HOOK] Editing code file - did you use EnterPlanMode for non-trivial tasks?"
    }
}

exit 0

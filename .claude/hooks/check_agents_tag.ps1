# Hook: Check if response ends with **Agents:** tag
# Triggered on: Stop event

$input_json = [Console]::In.ReadToEnd() | ConvertFrom-Json -ErrorAction SilentlyContinue

if ($input_json -and $input_json.transcript_path) {
    $transcript = Get-Content -Raw $input_json.transcript_path -ErrorAction SilentlyContinue
    if ($transcript -and $transcript -notmatch '\*\*Agents:\*\*') {
        Write-Host "[HOOK] Remember to end your response with: **Agents:** [list or None]"
    }
}

exit 0

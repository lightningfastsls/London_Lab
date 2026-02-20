# Hook: Validate note schema - check frontmatter on notes written to notes/
# Triggered on: PostToolUse (Write) - receives stdin JSON

$ErrorActionPreference = 'SilentlyContinue'

try {
    $raw = [Console]::In.ReadToEnd()
    if (-not $raw) { [Environment]::Exit(0) }

    $input_json = $raw | ConvertFrom-Json
    if (-not $input_json -or -not $input_json.tool_input -or -not $input_json.tool_input.file_path) { [Environment]::Exit(0) }

    $path = $input_json.tool_input.file_path

    # Only validate files in notes/
    if ($path -notmatch '[/\\]notes[/\\]') { [Environment]::Exit(0) }

    # Only validate .md files
    if ($path -notmatch '\.md$') { [Environment]::Exit(0) }

    # Check file exists
    if (-not (Test-Path $path)) { [Environment]::Exit(0) }

    $content = Get-Content $path -ErrorAction SilentlyContinue

    # Check for YAML frontmatter
    if (-not $content -or $content.Count -eq 0 -or $content[0] -ne '---') {
        Write-Host "WARN: $path missing YAML frontmatter"
        [Environment]::Exit(0)
    }

    # Check required fields
    $hasDescription = $false
    $hasTopics = $false
    foreach ($line in $content) {
        if ($line -match '^description:') { $hasDescription = $true }
        if ($line -match '^topics:') { $hasTopics = $true }
    }

    if (-not $hasDescription) {
        Write-Host "WARN: $path missing required 'description' field"
    }
    if (-not $hasTopics) {
        Write-Host "WARN: $path missing required 'topics' field"
    }
} catch {
    [Console]::Error.WriteLine("[HOOK validate-note] $($_.Exception.Message)")
}

[Environment]::Exit(0)

# Hook: Validate note schema - check frontmatter on notes written to notes/
# Triggered on: PostToolUse (Write) - receives stdin JSON
# BLOCKING on required fields (description, topics, type) - exits 1 to signal the agent must fix.
# Advisory on optional fields (description length, type enum) - warns but does not block.

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

    # Parse frontmatter fields
    $hasDescription = $false
    $hasTopics = $false
    $hasType = $false
    $typeValue = ""
    $descriptionValue = ""
    $inFrontmatter = $false

    foreach ($line in $content) {
        if ($line -eq '---' -and -not $inFrontmatter) {
            $inFrontmatter = $true
            continue
        }
        if ($line -eq '---' -and $inFrontmatter) {
            break
        }
        if ($inFrontmatter) {
            if ($line -match '^description:\s*(.*)') {
                $hasDescription = $true
                $descriptionValue = $Matches[1].Trim().Trim('"').Trim("'")
            }
            if ($line -match '^topics:') { $hasTopics = $true }
            if ($line -match '^type:\s*(.*)') {
                $hasType = $true
                $typeValue = $Matches[1].Trim()
            }
        }
    }

    # Track whether any required field is missing
    $blockingFailure = $false

    # Required field checks — BLOCKING (exit 1)
    if (-not $hasDescription) {
        Write-Host "BLOCK: $path missing required 'description' field — add description to frontmatter before proceeding"
        $blockingFailure = $true
    }
    if (-not $hasTopics) {
        Write-Host "BLOCK: $path missing required 'topics' field — add topics with wiki-link to topic map before proceeding"
        $blockingFailure = $true
    }

    # Required: type field — BLOCKING (exit 1)
    if (-not $hasType) {
        Write-Host "BLOCK: $path missing required 'type' field — add type (finding|decision|method|hypothesis|baseline|open-question|pattern|moc|learning|tension) to frontmatter before proceeding"
        $blockingFailure = $true
    }

    # Soft warning: type enum validation
    $validTypes = @("finding", "decision", "method", "hypothesis", "baseline", "open-question", "pattern", "moc", "learning", "tension")
    if ($hasType -and $typeValue -and $typeValue -notin $validTypes) {
        Write-Host "WARN: $path has invalid type '$typeValue' - expected one of: $($validTypes -join ', ')"
    }

    # Soft warning: description length
    if ($hasDescription -and $descriptionValue.Length -gt 200) {
        Write-Host "WARN: $path description is $($descriptionValue.Length) chars (max 200)"
    }

    if ($blockingFailure) {
        [Environment]::Exit(1)
    }
} catch {
    Write-Host "[HOOK validate-note] $($_.Exception.Message)"
}

[Environment]::Exit(0)

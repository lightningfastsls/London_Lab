# Hook: Validate note schema - check frontmatter on notes written to notes/
# Triggered on: PostToolUse (Write) - receives stdin JSON
# All checks are advisory (exit 0) - warns but never blocks.

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

    # Required field checks
    if (-not $hasDescription) {
        Write-Host "WARN: $path missing required 'description' field"
    }
    if (-not $hasTopics) {
        Write-Host "WARN: $path missing required 'topics' field"
    }

    # Soft warning: type field missing
    if (-not $hasType) {
        Write-Host "WARN: $path missing 'type' field (finding|decision|method|hypothesis|baseline|open-question|pattern)"
    }

    # Soft warning: type enum validation
    $validTypes = @("finding", "decision", "method", "hypothesis", "baseline", "open-question", "pattern")
    if ($hasType -and $typeValue -and $typeValue -notin $validTypes) {
        Write-Host "WARN: $path has invalid type '$typeValue' - expected one of: $($validTypes -join ', ')"
    }

    # Soft warning: description length
    if ($hasDescription -and $descriptionValue.Length -gt 200) {
        Write-Host "WARN: $path description is $($descriptionValue.Length) chars (max 200)"
    }
} catch {
    [Console]::Error.WriteLine("[HOOK validate-note] $($_.Exception.Message)")
}

[Environment]::Exit(0)

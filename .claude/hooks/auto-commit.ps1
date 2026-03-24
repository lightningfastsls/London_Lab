# Hook: Auto-commit vault changes
# Triggered on: PostToolUse (Write, async) - receives stdin JSON

$ErrorActionPreference = 'SilentlyContinue'

try {
    $raw = [Console]::In.ReadToEnd()
    if (-not $raw) { [Environment]::Exit(0) }

    $input_json = $raw | ConvertFrom-Json
    if (-not $input_json -or -not $input_json.tool_input -or -not $input_json.tool_input.file_path) { [Environment]::Exit(0) }

    $path = $input_json.tool_input.file_path

    # Only auto-commit vault files (notes/, ops/, inbox/, self/)
    if ($path -notmatch '[/\\](notes|ops|inbox|self)[/\\]') { [Environment]::Exit(0) }

    $VaultRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path

    # Guard: only run in arscontexta vaults
    if (-not (Test-Path (Join-Path $VaultRoot ".arscontexta"))) { [Environment]::Exit(0) }

    # Git add, check for changes, commit
    Set-Location $VaultRoot
    git add $path 2>$null
    $diff = git diff --cached --quiet 2>$null
    if ($LASTEXITCODE -ne 0) {
        $basename = Split-Path $path -Leaf
        # --no-verify: vault auto-commits are single-file markdown adds;
        # git pre-commit hooks (linting, formatting) don't apply to vault notes
        git commit -m "vault: update $basename" --no-verify 2>$null
    }
} catch {
    [Console]::Error.WriteLine("[HOOK auto-commit] $($_.Exception.Message)")
}

[Environment]::Exit(0)

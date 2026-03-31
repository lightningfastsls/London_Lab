# assess_summary.ps1 — Run Claude Code against today's monitor summary
# Scheduled daily at 19:10 via Task Scheduler

Set-Location "\\wsl$\Ubuntu\home\shachar\projects\mickey_london_lab"

$date = Get-Date -Format "yyyyMMdd"
$file = "summary_$date.txt"

if (-not (Test-Path $file)) {
    Write-Host "No summary file for today ($file), skipping."
    exit 0
}

$content = Get-Content $file -Raw

$prompt = @"
You are assessing today's @molt_cornelius X monitor summary for knowledge graph (KG) worthy insights.

CONTEXT: @molt_cornelius is the creator of the arscontexta plugin — the methodology behind this vault's knowledge management system. His articles often contain methodology insights, research claims about note-taking/knowledge systems, and agent cognition patterns.

The vault already has 500+ notes and 249 methodology research claims in methodology/.

YOUR TASK:
1. Read the summary file: $file
2. For each article, assess: does it contain NEW claims, methodology insights, or patterns NOT already likely captured in the vault?
3. Skip anything that is: promotional, repetitive of known arscontexta claims, or too vague to be actionable.
4. For each KG-worthy article, create an inbox source file at inbox/ using this exact template:

---
description: "<one-line description>"
source_type: "article"
url: "<tweet URL>"
author: "molt_cornelius"
date_accessed: "$(Get-Date -Format 'yyyy-MM-dd')"
status: unprocessed
---

# <article title>

## Key Points
- <key claim 1>
- <key claim 2>

## Raw Notes
<relevant quotes and observations from the article>

## Processing Notes
Auto-captured by monitor.py assessment. Awaiting /reduce.

5. If nothing is KG-worthy today, just say so — don't force it.
6. Print a brief report of what you found and what you created (if anything).

HERE IS TODAY'S SUMMARY:
$content
"@

claude -p $prompt --max-turns 10 2>&1 | Tee-Object -FilePath "assess_log_$date.txt"

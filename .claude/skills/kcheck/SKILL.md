---
name: kcheck
description: "Knowledge check — search vault for constraints relevant to planned work. Use before modifying detection, export, labeling, or other constrained systems. Triggers on /kcheck, 'knowledge check', 'check vault'."
version: "1.0"
generated_from: "manual"
user-invocable: true
context: fork
model: haiku
allowed-tools: Read, Grep, Glob, Bash
argument-hint: "<what I'm about to do>"
---

## EXECUTE NOW

**Planned work: $ARGUMENTS**

If no description provided, ask the user what they are about to do.

**Execute these steps:**

### Phase 1: Parse Intent

Extract from the user's description:
1. **Key concepts** — nouns and domain terms for keyword search (e.g., "detection", "threshold", "energy", "export", "labeling")
2. **File paths** — any specific files mentioned (e.g., `energy_detector.py`, `raven_export.py`)
3. **Action type** — what kind of change (modify, add, remove, refactor)

### Phase 2: Search

Run vault search via topic map traversal + ripgrep:

1. **Structural + content search** — find notes via topic map routing and keyword matching:
   ```bash
   node ops/scripts/vault-search.mjs --query "[full description of planned work]" --limit 8
   ```
   Parse the JSON output to get note titles, descriptions, types, and scores.

2. **Supplementary keyword search** — catch any notes the structural search missed:
   ```bash
   rg -il "key-term-1|key-term-2" notes/
   ```
   Read any additional matches not already in the vault-search results.

**Deduplication:** Results from vault-search.mjs are already deduplicated. Merge any supplementary ripgrep finds by note title. Cap at 8 results total, sorted by score descending.

### Phase 3: Format & Flag

Present results in this format:

```
## 🔍 Knowledge Check: [short summary of intent]

### Relevant Notes (N found)

| Score | Note | Key Point |
|-------|------|-----------|
| 0.XX  | [[note title]] | first sentence of description field |

### ⚠️ CRITICAL — File-Specific Constraints
[Only if any result's content references files mentioned in the intent]
- **[[note title]]** references `file.py` — READ BEFORE PROCEEDING

### Summary
[1-2 sentence synthesis: what constraints or prior decisions are most relevant to the planned work]
```

**If no results pass the score threshold:** Display "No strong matches found in vault. Proceed with standard care."

**If file paths were mentioned in the intent:** After getting search results, use `Grep` to check if any of the top results' note files contain the mentioned file paths. If they do, flag them in the CRITICAL section — these notes have direct opinions about the files being modified.

---

## Scope Boundaries

- This skill is READ-ONLY. Never create, edit, or delete notes.
- Do not read `ops/derivation.md`, `ops/config.yaml`, or methodology files — this is a fast vault lookup, not a deep analysis.
- Do not follow wiki links or do multi-hop exploration. Surface what the search finds, nothing more.
- If the user needs deeper analysis, recommend `/ask` or `/learn` as follow-ups.

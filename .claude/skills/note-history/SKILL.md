---
name: note-history
description: Show how a note evolved over time using git history. Inspect changes, interpret what shifted semantically, and optionally restore a previous version. Safety net for incorrect edits. Triggers on "/note-history", "/note-history [note]", "show note history", "what changed in this note", "restore note".
version: "1.0"
generated_from: "manual"
user-invocable: true
context: fork
model: haiku
allowed-tools: Read, Grep, Glob, Bash
argument-hint: "[note title or path] [--restore N] [--full]"
---

## THE MISSION

You are the vault's time-travel engine. Given a note title or path, you reconstruct its evolution from git history and present it as a meaningful narrative — not raw diffs, but interpreted changes that show how thinking evolved.

When `--restore N` is provided, you recover the Nth historical version of the note safely.

---

## EXECUTE NOW

**Target: $ARGUMENTS**

Parse the arguments:

| Pattern | Action |
|---------|--------|
| `[note title or partial path]` | Show history for that note |
| `--restore N [note]` | Restore version N of the note (with confirmation) |
| `--full [note]` | Show full diffs for every commit, not just summaries |
| (empty) | List recently modified vault notes and prompt for selection |

---

## Step 1: Locate the Note

The user may provide:
- A full filename: `notes/some note title.md`
- Just a title: `some note title`
- A partial match: `energy threshold`

**Resolution strategy:**

1. If the argument looks like a path (contains `/` or ends in `.md`), try it directly
2. Otherwise, search for it:
   ```
   Glob pattern: notes/*{argument}*.md
   ```
3. If multiple matches, list them and ask the user to pick
4. If no matches, try a broader search with individual keywords from the argument

**Critical:** Note filenames contain spaces. Always quote paths in git commands.

---

## Step 2: Retrieve Git History

Run git log for the located file:

```bash
git -C "D:/mickey_london_lab" log --follow --format="%h|%ai|%s" -- "FILEPATH"
```

The `--follow` flag tracks the file across renames (important if notes were ever renamed via the rename script).

Parse each line into: `short_hash | date | commit_message`

Number the commits from most recent (1) to oldest (N).

---

## Step 3: Present the Timeline

Output format:

```
=== Note History: [title] ===
File: [relative path]
Commits: N total | First created: [date] | Last modified: [date]

--- Timeline ---
[1] YYYY-MM-DD HH:MM  [commit message]
    [+N -M lines] | [interpretation of what changed]
[2] YYYY-MM-DD HH:MM  [commit message]
    [+N -M lines] | [interpretation of what changed]
...
```

### Interpreting Changes

For each commit (or at minimum the most recent 5), run:

```bash
git -C "D:/mickey_london_lab" diff HASH~1 HASH -- "FILEPATH"
```

For the oldest commit (initial creation), run:

```bash
git -C "D:/mickey_london_lab" show HASH -- "FILEPATH" | head -30
```

**Interpret the diff semantically, not syntactically:**

| Raw Diff | Semantic Interpretation |
|----------|----------------------|
| Changed `description:` field | Description refined |
| Added `[[wiki link]]` | New connection added to [linked note] |
| Changed `confidence:` value | Confidence level updated (e.g., speculative -> likely) |
| Added/removed body paragraphs | Reasoning expanded/trimmed |
| Changed `meta_state:` | Note status changed (e.g., current -> outdated) |
| Added `topics:` entries | Note added to new topic map |
| Changed title (rename) | Claim reframed |
| Lines added to Relevant Notes | Cross-references expanded |

**Focus on WHAT shifted intellectually, not line numbers.** The user wants to know "the confidence was upgraded from speculative to likely" not "+1 -1 in YAML frontmatter."

---

## Step 4: Show Most Recent Change (Detail)

After the timeline, show the interpreted diff for the most recent change in detail:

```
--- Most Recent Change ([date]) ---

[Narrative interpretation of what changed and why it matters]

Raw diff (collapsed):
[the actual diff output, for reference]
```

If `--full` flag is present, show interpreted diffs for ALL commits instead of just the most recent.

---

## Step 5: Restore Mode (--restore N)

When `--restore N` is in the arguments:

1. Identify version N from the timeline (1 = most recent, 2 = one before that, etc.)
2. Get the commit hash for that version
3. Retrieve the file content at that commit:
   ```bash
   git -C "D:/mickey_london_lab" show HASH:"FILEPATH"
   ```
4. **Show the user what will be restored** — display the full content of the historical version
5. Show a diff between the CURRENT version and the version to be restored
6. **Ask for confirmation before writing** — this is a destructive operation (overwrites current file)
7. If confirmed, write the historical content to the file using the Write tool
8. The auto-commit hook will capture this as a new commit (creating a "restore point")

**Safety rules:**
- NEVER restore without showing the user what will change
- NEVER restore without explicit confirmation
- The restore creates a NEW commit (it doesn't rewrite git history)
- Output: "Restored [note] to version from [date]. The previous version is still in git history as the commit before this one."

---

## Empty Arguments: Recent Changes View

If no arguments provided, show recently modified vault notes:

```bash
git -C "D:/mickey_london_lab" log --name-only --format="%h|%ai" -20 -- "notes/*.md"
```

Group by file and show:

```
=== Recently Modified Notes ===

1. [note title] — last modified [date] ([N] total commits)
2. [note title] — last modified [date] ([N] total commits)
...

Use: /note-history [title] to see full history
```

---

## Edge Cases

| Situation | Handling |
|-----------|----------|
| Note has only 1 commit | Show creation info, note "No changes since creation" |
| Note not tracked by git | "This note has no git history. It may be new and uncommitted." |
| File doesn't exist | "No note found matching '[query]'. Check spelling or try a partial match." |
| `--restore 0` or out of range | "Version N doesn't exist. This note has M versions (1 = most recent, M = oldest)." |
| Binary or non-markdown file | Refuse — "This skill works on vault notes (.md files) only." |

---

## Output Footer

Always end with:

```
---
Tip: /note-history --restore N [title] to recover a previous version
Tip: /note-history --full [title] to see all diffs expanded
```

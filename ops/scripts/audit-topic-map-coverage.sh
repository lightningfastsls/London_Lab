#!/usr/bin/env bash
# Phase 0: Topic Map Coverage Audit
# Validates that topic-map-based retrieval can cover the vault adequately.
set -eu

VAULT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NOTES_DIR="$VAULT_ROOT/notes"

echo "=== Topic Map Coverage Audit ==="
echo "Date: $(date '+%Y-%m-%d %H:%M')"
echo ""

# --- 0.1 Coverage Analysis ---

# Collect MOC files (type: moc in frontmatter)
mapfile -t MOC_FILES < <(grep -rl "type: moc" "$NOTES_DIR" 2>/dev/null | sort)
MOC_COUNT=${#MOC_FILES[@]}
echo "## MOC Files Found: $MOC_COUNT"
for f in "${MOC_FILES[@]}"; do
    echo "  - $(basename "$f" .md)"
done
echo ""

# Collect MOC basenames for exclusion
declare -A MOC_BASENAMES
for f in "${MOC_FILES[@]}"; do
    MOC_BASENAMES["$(basename "$f" .md)"]=1
done

# Collect all non-MOC note files
mapfile -t ALL_NOTES < <(find "$NOTES_DIR" -name "*.md" -type f | sort)
NON_MOC_NOTES=()
for f in "${ALL_NOTES[@]}"; do
    bn="$(basename "$f" .md)"
    if [[ -z "${MOC_BASENAMES[$bn]+_}" ]]; then
        NON_MOC_NOTES+=("$bn")
    fi
done
TOTAL_NON_MOC=${#NON_MOC_NOTES[@]}
echo "## Total Notes: ${#ALL_NOTES[@]} (${TOTAL_NON_MOC} non-MOC)"
echo ""

# Extract all [[wiki-links]] from MOC files
declare -A LINKED_NOTES
for moc in "${MOC_FILES[@]}"; do
    while IFS= read -r link; do
        [[ -n "$link" ]] && LINKED_NOTES["$link"]=1
    done < <(grep -oP '\[\[([^\]]+)\]\]' "$moc" | sed 's/\[\[//;s/\]\]//')
done

# Find uncovered notes
COVERED=0
UNCOVERED=()
for note in "${NON_MOC_NOTES[@]}"; do
    if [[ -n "${LINKED_NOTES[$note]+_}" ]]; then
        ((COVERED++)) || true
    else
        UNCOVERED+=("$note")
    fi
done

COVERAGE_PCT=0
if [[ $TOTAL_NON_MOC -gt 0 ]]; then
    COVERAGE_PCT=$(echo "scale=1; $COVERED * 100 / $TOTAL_NON_MOC" | bc)
fi

echo "## Coverage Results"
echo "  Notes in at least one topic map: $COVERED / $TOTAL_NON_MOC ($COVERAGE_PCT%)"
echo "  Uncovered notes: ${#UNCOVERED[@]}"
echo ""

if [[ ${#UNCOVERED[@]} -gt 0 ]]; then
    echo "## Uncovered Notes"
    for note in "${UNCOVERED[@]}"; do
        # Check if note has topics frontmatter pointing to valid maps
        note_file="$NOTES_DIR/$note.md"
        topics=""
        if [[ -f "$note_file" ]]; then
            topics=$(head -20 "$note_file" | grep "^topics:" | head -1 | sed 's/^topics://' | tr -d '"')
        fi
        echo "  - $note"
        [[ -n "$topics" ]] && echo "    topics: $topics"
    done
    echo ""
fi

# --- 0.1b Reverse check: notes with topics frontmatter pointing to invalid maps ---
echo "## Reverse Check: Invalid topic references"
invalid_refs=0
for note_file in "${ALL_NOTES[@]}"; do
    bn="$(basename "$note_file" .md)"
    topics_line=$(head -20 "$note_file" | grep "^topics:" | head -1 || true)
    if [[ -n "$topics_line" ]]; then
        while IFS= read -r ref; do
            [[ -z "$ref" ]] && continue
            if [[ ! -f "$NOTES_DIR/$ref.md" ]]; then
                echo "  BROKEN: '$bn' references topic '${ref}' (file not found)"
                ((invalid_refs++)) || true
            fi
        done < <(echo "$topics_line" | grep -oP '\[\[([^\]]+)\]\]' | sed 's/\[\[//;s/\]\]//')
    fi
done
if [[ $invalid_refs -eq 0 ]]; then
    echo "  All topic references valid."
fi
echo ""

# --- 0.2 Topic Map Quality Check ---
echo "## Topic Map Quality"
for moc in "${MOC_FILES[@]}"; do
    moc_name="$(basename "$moc" .md)"
    section_count=$(grep -c "^## " "$moc" 2>/dev/null || echo 0)
    link_count=$(grep -oP '\[\[([^\]]+)\]\]' "$moc" | wc -l)
    # Count links with context phrases (-- after the link)
    context_count=$(grep -cP '\]\].*--' "$moc" 2>/dev/null || echo 0)

    # Count non-MOC note links only
    note_link_count=0
    while IFS= read -r link; do
        [[ -z "$link" ]] && continue
        if [[ -z "${MOC_BASENAMES[$link]+_}" ]]; then
            ((note_link_count++)) || true
        fi
    done < <(grep -oP '\[\[([^\]]+)\]\]' "$moc" | sed 's/\[\[//;s/\]\]//')

    flags=""
    if [[ $note_link_count -gt 40 ]]; then flags+=" [SPLIT-CANDIDATE >40]"; fi
    if [[ $note_link_count -lt 5 ]] && [[ "$moc_name" != "index" ]]; then flags+=" [SPARSE <5]"; fi
    if [[ $section_count -lt 2 ]] && [[ "$moc_name" != "index" ]]; then flags+=" [FLAT]"; fi

    echo "  $moc_name: $section_count sections, $note_link_count note-links, $context_count with context$flags"
done
echo ""

# --- 0.3 qmd Reference Catalog ---
echo "## qmd References"
echo "  Cataloging files referencing 'qmd'..."
while IFS= read -r f; do
    rel_path="${f#$VAULT_ROOT/}"
    category="unknown"
    case "$rel_path" in
        .claude/hooks/*) category="hook" ;;
        .claude/skills/*) category="skill" ;;
        docs/investigations/*) category="investigation" ;;
        docs/*) category="documentation" ;;
        ops/*) category="operational" ;;
        notes/*) category="note" ;;
        methodology/*) category="methodology" ;;
        reference/*) category="reference" ;;
        *.md) category="root-doc" ;;
    esac
    echo "  [$category] $rel_path"
done < <(rg -l "qmd" "$VAULT_ROOT" --glob "*.md" --glob "*.sh" --glob "*.mjs" --glob "*.yaml" --glob "*.json" 2>/dev/null | sort)
echo ""

echo "=== Audit Complete ==="

#!/usr/bin/env bash
# Hook: Validate note schema - check frontmatter on notes written to notes/
# Triggered on: PostToolUse (Write)

raw=$(cat)
[[ -z "$raw" ]] && exit 0

path=$(python3 -c "import json,sys; d=json.loads(sys.argv[1]); print(d.get('tool_input',{}).get('file_path',''))" "$raw" 2>/dev/null)
[[ -z "$path" ]] && exit 0

# Only validate files in notes/
[[ "$path" =~ /notes/ ]] || exit 0
[[ "$path" =~ \.md$ ]] || exit 0
[[ -f "$path" ]] || exit 0

content=$(cat "$path")

# Check for YAML frontmatter
first_line=$(echo "$content" | head -1)
if [[ "$first_line" != "---" ]]; then
    echo "WARN: $path missing YAML frontmatter"
    exit 0
fi

# Parse frontmatter
has_description=false
has_topics=false
has_type=false
type_value=""
desc_value=""
in_frontmatter=false

while IFS= read -r line; do
    if [[ "$line" == "---" ]]; then
        if $in_frontmatter; then break; fi
        in_frontmatter=true
        continue
    fi
    if $in_frontmatter; then
        if [[ "$line" =~ ^description:[[:space:]]*(.*) ]]; then
            has_description=true
            desc_value="${BASH_REMATCH[1]}"
            desc_value="${desc_value#\"}"
            desc_value="${desc_value%\"}"
            desc_value="${desc_value#\'}"
            desc_value="${desc_value%\'}"
        fi
        if [[ "$line" =~ ^topics: ]]; then has_topics=true; fi
        if [[ "$line" =~ ^type:[[:space:]]*(.*) ]]; then
            has_type=true
            type_value="${BASH_REMATCH[1]}"
            type_value="${type_value## }"
        fi
    fi
done <<< "$content"

if ! $has_description; then echo "WARN: $path missing required 'description' field"; fi
if ! $has_topics; then echo "WARN: $path missing required 'topics' field"; fi
if ! $has_type; then echo "WARN: $path missing 'type' field (finding|decision|method|hypothesis|baseline|open-question|pattern)"; fi

valid_types="finding decision method hypothesis baseline open-question pattern"
if $has_type && [[ -n "$type_value" ]] && [[ ! " $valid_types " =~ " $type_value " ]]; then
    echo "WARN: $path has invalid type '$type_value' - expected one of: $valid_types"
fi

if $has_description && [[ ${#desc_value} -gt 200 ]]; then
    echo "WARN: $path description is ${#desc_value} chars (max 200)"
fi

exit 0

# Plan: Pull arscontexta methodology notes

## Goal
Download the `methodology/` directory from the arscontexta repo into the existing plugin installation. These 249 research claim files are the knowledge graph that arscontexta commands reason against — without them, `/arscontexta:ask`, `/arscontexta:recommend`, `/arscontexta:architect`, and `/arscontexta:reseed` operate without their backing research.

## Steps

### 1. Locate existing arscontexta installation
```bash
# Check common plugin locations
ls -d ~/.claude/plugins/arscontexta 2>/dev/null || \
ls -d ./.claude/plugins/arscontexta 2>/dev/null || \
find ~ -maxdepth 5 -type d -name "arscontexta" 2>/dev/null | head -10
```
Store the path as `$ARSCONTEXTA_ROOT`.

### 2. Check if methodology/ already exists
```bash
ls "$ARSCONTEXTA_ROOT/methodology/" 2>/dev/null && echo "Already exists — check if incomplete" || echo "Not present — proceed"
```

### 3. Sparse checkout just the methodology directory
```bash
cd /tmp
git clone --depth 1 --filter=blob:none --sparse https://github.com/agenticnotetaking/arscontexta.git arscontexta-methodology
cd arscontexta-methodology
git sparse-checkout set methodology
```

### 4. Copy into existing installation
```bash
cp -r /tmp/arscontexta-methodology/methodology/ "$ARSCONTEXTA_ROOT/methodology/"
```

### 5. Verify
```bash
# Count files — expect ~249 markdown files
find "$ARSCONTEXTA_ROOT/methodology/" -name "*.md" | wc -l
# Spot check a few
ls "$ARSCONTEXTA_ROOT/methodology/" | head -5
```

### 6. Cleanup
```bash
rm -rf /tmp/arscontexta-methodology
```

## Notes
- If arscontexta was installed via `/plugin install arscontexta@agenticnotetaking`, the location is likely `~/.claude/plugins/arscontexta/`
- If the plugin manager manages files read-only, you may need to copy methodology/ into your project's knowledge directory instead and point arscontexta at it
- After pulling, test with `/arscontexta:ask` to confirm the research graph is queryable

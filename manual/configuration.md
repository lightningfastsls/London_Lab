---
description: How to adjust your system via config.yaml and /architect
type: manual
generated_from: "arscontexta-0.8.0"
---
# Configuration

## config.yaml

Your system configuration lives at ops/config.yaml. Key sections:

### Dimensions
```yaml
dimensions:
  granularity: atomic      # atomic | moderate | coarse
  processing: heavy        # light | moderate | heavy
  automation: full         # manual | convention | full
```

### Processing Settings
```yaml
processing:
  depth: standard          # deep | standard | quick
  chaining: suggested      # manual | suggested | automatic
```

### Features
```yaml
features:
  semantic-search: true    # qmd integration
  processing-pipeline: true
```

## Using /architect

For guided configuration changes backed by research:
1. Run /arscontexta:architect with your question
2. Review the proposed change with research justification
3. Approve or modify before implementation

## Feature Toggling

Most features can be enabled/disabled in config.yaml. Changes take effect next session.

## Your Research Preset

Your vault uses the Research preset:
- Atomic granularity (one claim per note)
- Flat organization (no subfolders in notes/)
- Heavy processing (full pipeline with quality gates)
- Full automation (all skills and hooks active)

See ops/derivation.md for the full reasoning behind each choice.

See [[meta-skills]] for /architect details.
See [[troubleshooting]] for configuration issues.

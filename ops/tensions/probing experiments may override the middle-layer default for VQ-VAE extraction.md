---
description: "Probing creates independent evidence for layer selection that could conflict with VQ-VAE perplexity-based comparison"
category: methodology
status: pending
created: 2026-02-23
---

# Probing experiments may override the middle-layer default for VQ-VAE extraction

The current VQ-VAE pipeline (Phase 8.3) defaults to layer 4 of 8, based on the heuristic that middle layers capture "mid-level concepts." The compare_layers.py script compares VQ-VAE metrics (perplexity, utilization, reconstruction loss) across layers 2, 4, 6, 8. But probing experiments (Phase 15.2) provide an INDEPENDENT line of evidence: which layer best encodes acoustic properties like peak frequency, spectral centroid, and voicing. These two criteria could disagree — the layer with highest VQ-VAE perplexity may NOT be the layer that best encodes acoustic information. If so, the project faces a choice: optimize for VQ-VAE interpretability (codebook diversity) or optimize for acoustic information content (probing R-squared).

The tension dissolves if both methods agree. If they disagree, the resolution depends on what the downstream analysis needs: if the goal is to create interpretable codebook entries that map to acoustic properties, probing should win. If the goal is to create the most diverse vocabulary for sequence analysis, VQ-VAE metrics should win. A third possibility is that the "best" layer differs by use case — one layer for codebook visualization and acoustic interpretation, another for sequential structure analysis. This would complicate the pipeline but might be the most honest approach.

## Conflicting Notes
- [[middle-layer hidden states capture mid-level concepts better than early or late layers for VQ-VAE input]] -- the current default assumption (layer 4)
- [[comparing VQ-VAE across transformer layers reveals which abstraction level yields the most interpretable codebook]] -- VQ-VAE metric-based layer selection
- [[layer-property heatmap is the key output showing where acoustic information lives across transformer depth]] -- probing-based layer selection

## Possible Resolution
Run both analyses and compare. If they agree, the tension was never real. If they disagree, the resolution depends on the specific downstream task. For the core research question (comparing wild vs. lab vocal repertoires), acoustic interpretability likely matters more than codebook diversity — suggesting probing results should take precedence. But this decision should be made explicitly when the data is available, not assumed in advance.

Source:
- [[vacation-master-plan-v2]]

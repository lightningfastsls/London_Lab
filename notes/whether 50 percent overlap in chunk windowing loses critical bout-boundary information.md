---
description: Open question on whether 50% stride chunking of long bouts causes positional encoding artifacts at chunk boundaries that degrade transformer learning.
type: open-question
confidence: speculative
topics:
  - "[[representation-learning]]"
---

# whether 50 percent overlap in chunk windowing loses critical bout-boundary information

Bouts longer than max_seq_len=512 frames are handled by sliding a window with 50% overlap (stride=256 frames). Each acoustic event near a chunk boundary appears in two overlapping chunks, but with different positional encodings — its position relative to the start of chunk N differs from its position relative to the start of chunk N+1. If the transformer relies on absolute positional encoding to interpret acoustic context, the same sound gets two distinct representations depending on which chunk it falls in.

Whether this constitutes meaningful information loss depends on two unknowns: how strongly the transformer uses absolute position (versus attending to relative content), and whether bout boundaries carry disproportionate structure worth preserving. Bouts often begin and end with silence, so boundary frames may be acoustically uninformative regardless. Conversely, if the transformer learns that position 0 means "bout onset" and uses that signal for long-range attention, chunked representations would break that learned prior.

A diagnostic is to compare loss curves on chunks that span bout boundaries versus chunks that fall entirely within bouts. If boundary-crossing chunks have systematically higher loss, the overlap strategy is insufficient and alternatives — such as relative positional encoding (RoPE or ALiBi) or treating each bout as a single variable-length sequence with truncation only at the HPC memory limit — should be evaluated.

This connects to [[bout-level spectrograms preserve inter-USV timing context for transformer training]] (which motivates preserving bout context) and [[length-bucketed batching minimizes padding waste when sequences vary in duration]] (which handles the variable-length consequence of not chunking). The diagnostic comparison of boundary-crossing vs intra-bout chunks should be performed during [[staged transformer training catches issues early by incrementally scaling from one bout to full dataset]], specifically at Stage C where enough bouts are present to observe the pattern. If boundary artifacts are confirmed, alternative positional encodings would affect the architecture choices documented in [[pre-norm transformer architecture improves training stability for spectrogram prediction]].

---

Source: [ROADMAP](../ROADMAP.md)

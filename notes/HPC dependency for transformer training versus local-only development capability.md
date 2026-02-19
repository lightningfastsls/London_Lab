---
description: The ~25-30M param transformer cannot train on the local AMD RX 5700; real training requires HPC or cloud A100, creating a gap between code readiness and execution.
type: finding
confidence: proven
topics:
  - "[[representation-learning]]"
---

# HPC dependency for transformer training versus local-only development capability

The autoregressive transformer specified in the roadmap (~25–30M parameters) exceeds what is practical to train on the locally available AMD RX 5700 GPU. Full training on the complete dataset is estimated at 1–2 days on an A100-class GPU; on the local card, memory constraints alone would require extreme gradient checkpointing and batch size reduction that could make training impractically slow or numerically unstable with mixed precision.

This creates a structural gap: code can be fully written, tested for correctness, and verified for convergence locally using dummy data (correct tensor shapes, single-batch overfitting test), but real experimental runs require cloud or HPC access. The local development workflow must therefore include "HPC readiness" as an explicit quality gate — the code should launch, profile, and converge on one bout before being submitted to an HPC queue.

The staged training protocol partially mitigates this gap. Stage A (single bout, ~2K frames) and Stage B (10 bouts, curriculum ordering) are small enough that they may run on the local GPU with reduced batch size, giving early feedback on whether the architecture is learning anything before committing HPC compute. Stages C and D (full dataset, fine-tuning with VQ-VAE) clearly require HPC.

This dependency connects to [[transformer-first then VQ-VAE avoids forcing premature discretization]], which motivates why the transformer is the first major investment despite its compute cost, and [[staged transformer training catches issues early by incrementally scaling from one bout to full dataset]], which is the primary mitigation for the HPC dependency. Architecture stability is critical for remote HPC runs where interactive debugging is impractical, making [[pre-norm transformer architecture improves training stability for spectrogram prediction]] an essential prerequisite for HPC deployment.

---

Source: [[ROADMAP.md]]

Relevant Notes:
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- why the transformer investment is justified despite compute cost
- [[staged transformer training catches issues early by incrementally scaling from one bout to full dataset]] -- primary mitigation for the HPC gap
- [[pre-norm transformer architecture improves training stability for spectrogram prediction]] -- stability is critical for non-interactive HPC runs
- [[bout-level spectrograms preserve inter-USV timing context for transformer training]] -- the data format that determines training compute requirements

Topics:
- [[classification]]

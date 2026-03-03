---
description: Three CNN size tiers (101K, 400K, 1.6M params) matched to label count ranges; underfitting and overfitting signals guide tier selection empirically.
type: pattern
confidence: likely
topics:
  - "[[classification]]"
  - "[[experimental-methods]]"
---

# model size should scale with labeled dataset size to balance underfitting and overfitting

Capacity mismatch is a recurring failure mode in supervised learning on small datasets. A model with too many parameters relative to the available labeled examples will memorize training data rather than learn generalizable features, producing a large gap between training and validation loss. Conversely, a model with too few parameters cannot represent the complexity of the input distribution, resulting in high loss on both training and validation sets.

For the USV CNN pipeline, three capacity tiers are defined to match expected label counts at each milestone. The Small tier (~101K parameters) targets datasets below 5,000 labels. The Medium tier (~400K parameters) is designed for the 5,000-15,000 label range. The Large tier (~1.6M parameters) is appropriate when more than 15,000 labels are available. These boundaries are empirical estimates based on the current dataset characteristics, not universal constants, and should be revisited if evidence from training curves suggests otherwise.

Diagnosis proceeds through loss curve inspection. If both training loss and validation loss plateau at a high value and fail to decrease, the model is underfitting — increase to the next tier. If training loss continues to decrease while validation loss stagnates or rises, the model is overfitting — decrease to a smaller tier or add regularization before scaling up. The three-tier structure described here relates to [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]], which defines the architectural template that each tier scales.

The confidence rating of "likely" reflects that these tier boundaries were set prospectively based on general deep learning heuristics rather than validated through systematic ablation on the USV dataset. The pattern is well-established in the field, but the specific parameter counts and label-count thresholds are approximate. See [[recording-level splits reduce effective training set size but prevent data leakage]] for context on why the effective training set is smaller than the total label count suggests.

---

Source: [ROADMAP](../ROADMAP.md), Phase 2

Relevant Notes:
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- decouples model size from trainable parameter count, enabling large pretrained models on small datasets without overfitting
- [[dataset quality exceeds quantity for LoRA fine-tuning as curated 1K LIMA matches 50K Alpaca performance]] -- LoRA's data efficiency suggests these scaling tiers may be conservative when adapting rather than training from scratch

---
description: "Splitting by recording gives honest evaluation metrics but yields fewer effective training samples per recording"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[experimental-methods]]"
---

# recording-level splits reduce effective training set size but prevent data leakage

Since [[recording-level splits prevent data leakage in USV classification]], all dataset splits are performed by recording file stem. The cost is a smaller effective training set -- chunks from the same recording cannot be distributed across train/val/test splits. If a recording contains 50 USV candidates, all 50 go to the same split. This means the model sees fewer distinct "contexts" during training compared to random splitting, but the evaluation metrics are honest -- they measure the model's ability to generalize to truly unseen recordings, not its ability to memorize recording-specific characteristics. This tradeoff becomes more favorable as the number of distinct recordings increases, since each recording contributes all its candidates to the training set. With the planned scaling from 2K to 30K samples, more recordings means less wasted data per split.

---

Source:
- DECISIONS.md (ADR-004) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[recording-level splits prevent data leakage in USV classification]] -- the decision creating this tradeoff
- [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] -- another data preparation concern
- [[3x class weight boost compensates for USV class imbalance in CNN training]] -- smaller effective training sets from recording-level splits intensify class imbalance, increasing the need for weight boosting
- [[dataset quality exceeds quantity for LoRA fine-tuning as curated 1K LIMA matches 50K Alpaca performance]] -- LoRA's data efficiency suggests the reduced effective training set from recording-level splits may be less of a limitation when adapting pre-trained models rather than training from scratch
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- parameter-efficient adaptation could make the most of the smaller honest training set by requiring fewer examples to signal task-relevant directions

Topics:
- [[experimental-methods]]

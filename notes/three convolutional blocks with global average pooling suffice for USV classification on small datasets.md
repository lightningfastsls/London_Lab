---
description: "USVClassifierCNN uses 3 conv blocks [32,64,128] with GlobalAvgPool, totaling ~101K params -- appropriate for hundreds to low thousands of examples"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
---

# Three convolutional blocks with global average pooling suffice for USV classification on small datasets

The CNN classifier (USVClassifierCNN) uses a deliberately small architecture: 3 convolutional blocks with [32, 64, 128] filters. Each block consists of Conv2d(3x3, padding=1) -> BatchNorm2d -> ReLU -> MaxPool2d(2x2). Global Average Pooling replaces the typical flatten+dense layer, enabling variable-size input spectrograms without fixed dimensions. The dense head is Linear(128->64) -> ReLU -> Dropout(0.5) -> Linear(64->1), outputting logits for BCEWithLogitsLoss. Total parameters: ~101,889 (93,568 conv+bn + 8,321 classifier). A larger variant (USVClassifierCNNLarge, 5 blocks, [32,64,128,256,512]) exists but is not recommended for datasets under 5,000 samples due to overfitting risk.

---

Source:
- DECISIONS.md (ADR-006) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[two-stage detection uses permissive energy detector followed by CNN precision filter]] -- the CNN's role in the pipeline
- [[3x class weight boost compensates for USV class imbalance in CNN training]] -- training strategy for this architecture
- [[three-source negative sampling teaches the CNN the full spectrum of non-USV audio]] -- data strategy
- [[recording-level splits prevent data leakage in USV classification]] -- the small architecture is partly motivated by the limited effective training set from recording-level splits
- [[model size should scale with labeled dataset size to balance underfitting and overfitting]] -- the three-tier scaling principle that places this 101K architecture at the small end for datasets under 5K labels
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- as an alternative to scaling this small CNN, LoRA adaptation of a pre-trained foundation model could achieve better classification with the same limited data
- [[mid-c-cnn-balances-capacity-and-inference-speed-for-14k-samples]] -- custom [32, 96, 192] variant that preserves this 3-block template while doubling capacity for the 14.7K matched-windows dataset
- [[AMVOC autoencoder encodes 64x160 spectrogram patches through three convolutional layers to an 8x8x20 bottleneck with 8x compression]] -- AMVOC also uses 3 conv layers but without BatchNorm or GlobalAvgPool; validates that 3-layer depth suffices for USV feature extraction across both classification and autoencoder tasks
- [[AMVOC lacks batch normalization dropout validation monitoring and VAE variant — all high-value improvements for our wild-mouse pipeline]] -- our CNN already implements the BatchNorm and Dropout that AMVOC identifies as gaps, confirming these are standard practice for our pipeline

Topics:
- [[classification]]

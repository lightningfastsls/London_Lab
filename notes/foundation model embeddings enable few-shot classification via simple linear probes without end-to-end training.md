---
description: "Perch 2.0, BEATs, and AVES provide off-the-shelf embeddings where a linear layer or k-NN on frozen features matches purpose-built classifiers -- no GPU-intensive fine-tuning needed"
type: finding
confidence: proven
meta_state: current
topics:
  - "[[bioacoustic-ssl]]"
  - "[[classification]]"
---

# foundation model embeddings enable few-shot classification via simple linear probes without end-to-end training

The shift from training classifiers from scratch to using pre-trained foundation models as feature extractors represents a paradigm change in bioacoustic classification. Perch 2.0 (Google DeepMind), BEATs (Microsoft), and AVES (Earth Species Project) produce embeddings where a simple linear probe or k-NN achieves competitive performance with purpose-built classifiers. Perch 2.0's "agile modeling" paradigm allows building a custom classifier from a small number of labeled examples in hours rather than days. Despite zero underwater training data, Perch 2.0 embeddings even transferred to marine mammal classification (killer whale ecotype discrimination), demonstrating remarkable cross-domain generalization.

The practical implication is transformative: researchers can classify new species or call types by collecting approximately 10 examples, embedding them through a frozen model, and training a lightweight classifier -- no deep learning expertise or GPU infrastructure required. This accessibility matters because most bioacoustics labs lack ML engineering resources. The frozen-embedding approach also means that adding a new class never requires retraining the embedding model, only computing a new prototype or retraining a small linear layer.

However, there is an important caveat for our USV pipeline: these foundation models are trained on audio at 16-48 kHz sample rates, while our USVs occupy 25-120 kHz recorded at 300 kHz. This domain shift means we cannot directly feed our audio into these models. But the approach still applies if we either frequency-shift our recordings into the audible range or treat spectrograms as images and use vision-based embeddings. The core principle -- that pre-trained representations plus simple classifiers can match or exceed custom-trained models -- holds regardless of the specific embedding source.

---

Source:
- few-shot-learning-animal-sound-classification-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[prototypical probing with frozen MAE features enables bioacoustic classification with as few as 10 labeled examples]] -- the same approach with MAE specifically
- [[BEATs self-distilled discrete tokenizer achieves the highest BEANS benchmark score among bioacoustic encoders]] -- BEATs as a top embedding source
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- LoRA bridges the gap between frozen-embedding linear probes and full fine-tuning: when linear probes are insufficient, LoRA fine-tuning adapts the embedding model with minimal labeled data
- [[LoRA acts as implicit regularizer preserving base model capabilities with strong inverse linear relationship between adaptation and forgetting]] -- LoRA's regularization preserves the general embedding quality while adapting to the target task, complementing the frozen-embedding approach
- [[dataset quality exceeds quantity for LoRA fine-tuning as curated 1K LIMA matches 50K Alpaca performance]] -- aligns with the few-shot paradigm: small curated datasets suffice for both linear probes and LoRA adaptation
- [[no few-shot learning method has been applied to USV syllable-type classification]] -- this note's approach directly addresses the gap: foundation embeddings + linear probe is the simplest path to few-shot USV syllable classification

Topics:
- [[bioacoustic-ssl]]
- [[classification-methodology]]

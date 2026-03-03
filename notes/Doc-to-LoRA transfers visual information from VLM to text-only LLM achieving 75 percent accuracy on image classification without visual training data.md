---
description: "Cross-modal weight transfer — Gemma-3-4b-it VLM context encodes image info into LoRA adapter applied to text-only Gemma-2-2b; 75.03% on Imagenette zero-shot without visual data"
type: finding
confidence: likely
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
---

# Doc-to-LoRA transfers visual information from VLM to text-only LLM achieving 75 percent accuracy on image classification without visual training data

One of the most surprising Doc-to-LoRA findings: when using a vision-language model (Gemma-3-4b-it) as the context encoder instead of the text-only base model, the hypernetwork can transfer visual information from the VLM into a text-only LLM via weight updates. The text-only model (Gemma-2-2b) achieves 75.03% accuracy on Imagenette image classification — zero-shot, without any visual training data.

The mechanism is conceptually clean: the VLM processes an image and produces internal representations (activations at each layer). The Doc-to-LoRA hypernetwork consumes these activations and produces a LoRA adapter that, when applied to the text-only model, modifies its behavior to reflect the visual information. The visual knowledge is literally transferred through weight modifications.

This finding suggests that since [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]], the text-only model may already have latent visual-linguistic associations from its text-only pre-training (e.g., learning associations between image captions and descriptions). The VLM-generated adapter amplifies these latent directions.

However, a limitation: the VLM context encoder negatively impacts text-based QA performance when used for vision transfer. Using a VLM encoder degrades the hypernetwork's text processing capabilities, suggesting the visual and textual adapter-generation pathways interfere with each other.

---

Source: lora-doc-to-lora-hypernetworks-research-2026-03-02

Relevant Notes:
- [[Doc-to-LoRA hypernetwork generates LoRA adapters in a single forward pass via Perceiver cross-attention compressing documents into sub-50 MB weight updates]] -- the base system this extends cross-modally
- [[LoRA adaptation amplifies existing underemphasized directions in pre-trained weights rather than learning entirely new features]] -- suggests why cross-modal transfer is even possible
- [[pre-trained language models have low intrinsic dimension with larger models having even lower intrinsic dimension after pre-training]] -- low intrinsic dimension may include cross-modal directions from text describing visual concepts
- [[hypernetworks learn functions that generate weights for other networks amortizing per-task training cost into a single meta-training phase]] -- the VLM encoder is a hypernetwork mapping visual inputs to weight updates
- [[a generic cross-species autoencoder performs nearly as well as species-specific models suggesting shared vocalization structure]] -- both demonstrate cross-domain representation transfer: VLM-to-text via adapter weights, and cross-species via shared autoencoder features

Topics:
- [[model-adaptation]]

---
description: "Sakana AI companion to Doc-to-LoRA — task encoder + MLP blocks generate A,B matrices for Mistral-7B q_proj/v_proj rank-8; SFT-trained variant competitive zero-shot on benchmarks"
type: finding
confidence: likely
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
---

# Text-to-LoRA generates task-specific LoRA adapters from natural language descriptions in a single forward pass replacing fine-tuning pipelines

Text-to-LoRA is a companion system to Doc-to-LoRA that accepts a natural-language task description and generates a LoRA adapter in a single forward pass, replacing the entire fine-tuning pipeline for task adaptation. Where Doc-to-LoRA internalizes document *content*, Text-to-LoRA internalizes task *behavior*.

The architecture: a task encoder extracts vector representations from text descriptions. These are combined with learnable module and layer embeddings, then processed through MLP blocks to generate A and B low-rank matrices. The base model is Mistral-7B-Instruct, targeting q_proj and v_proj at rank 8 across all layers (~3.4M adapter parameters).

Two training approaches were explored: (1) **Reconstruction training** — matching existing task-specific LoRAs from the Lots-of-LoRAs dataset (479 diverse tasks from SNI); (2) **SFT training** — end-to-end optimization through downstream task loss. The SFT-trained variant can zero-shot generate adapters for benchmark tasks competitively, while reconstruction training does not generalize as well to truly novel tasks.

Performance scales with training dataset diversity, especially for larger Text-to-LoRA variants. This suggests the hypernetwork learns generalizable task→adapter mappings that improve with exposure to more diverse task descriptions.

The broader implication: since [[hypernetworks learn functions that generate weights for other networks amortizing per-task training cost into a single meta-training phase]], Text-to-LoRA extends this from document-conditioning to task-conditioning. Together with Doc-to-LoRA, this suggests that many forms of model adaptation can be amortized through hypernetworks.

---

Source: [[lora-doc-to-lora-hypernetworks-research-2026-03-02]]

Relevant Notes:
- [[Doc-to-LoRA hypernetwork generates LoRA adapters in a single forward pass via Perceiver cross-attention compressing documents into sub-50 MB weight updates]] -- the document-conditioning counterpart
- [[hypernetworks learn functions that generate weights for other networks amortizing per-task training cost into a single meta-training phase]] -- the foundational concept both systems share
- [[LoRA exploits low intrinsic rank of weight updates to match full fine-tuning with 10000x fewer trainable parameters]] -- the adapter format both systems generate
- [[multi-LoRA serving enables hundreds of concurrent adapters from a single base model with millisecond switching in production]] -- production infrastructure that could serve generated adapters
- [[ICL fails on specification-heavy tasks reaching less than half of fine-tuned performance due to inadequate schema comprehension]] -- Text-to-LoRA bridges ICL's speed with fine-tuning's depth for complex tasks
- [[the ICL to LoRA to Doc-to-LoRA progression represents a spectrum from implicit temporary to explicit persistent knowledge internalization]] -- Text-to-LoRA occupies the automated end of this spectrum alongside Doc-to-LoRA, generating persistent weight-based knowledge at ICL speed

Topics:
- [[model-adaptation]]

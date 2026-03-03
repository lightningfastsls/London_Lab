---
description: "Charakorn et al Feb 2026 (Sakana AI) — 309M param Perceiver with 8 cross-attention blocks on Gemma-2-2b; sub-second update vs 40s oracle distillation, 83.5% of full-context SQuAD"
type: finding
confidence: likely
created: 2026-03-02
topics:
  - "[[model-adaptation]]"
  - "[[context-management]]"
---

# Doc-to-LoRA hypernetwork generates LoRA adapters in a single forward pass via Perceiver cross-attention compressing documents into sub-50 MB weight updates

Doc-to-LoRA (Charakorn et al., Sakana AI, February 2026) applies the concept of since [[hypernetworks learn functions that generate weights for other networks amortizing per-task training cost into a single meta-training phase]] specifically to document internalization. The architecture has two modules: (1) a Perceiver-style cross-attention encoder consuming per-layer token activations from a frozen base LLM, and (2) output heads mapping latent queries to LoRA matrices. The hypernetwork has approximately 309M parameters with 8 cross-attention blocks, generating rank-8 LoRA adapters targeting MLP layers of Gemma-2-2b-it.

The training objective is teacher-student distillation: the hypernetwork minimizes the gap between a teacher (full document in context) and a student (LoRA-adapted, no context) response. This amortizes per-document training cost — expensive meta-training once, then document in → adapter out in sub-second time.

Performance on Reading Comprehension (SQuAD): 83.5% of the full-context upper bound, without the document in the query window. Compared to alternatives: oracle context distillation takes 40 seconds per document, traditional context distillation takes 100+ seconds. Doc-to-LoRA: <1 second.

The inference API is remarkably clean:
```python
model.internalize(doc)   # generates LoRA adapter from document
model.generate(...)       # answers questions without document in context
model.reset()             # clears internalized information
```

Since [[Doc-to-LoRA reduces KV-cache memory from 12-plus GB to constant sub-50 MB regardless of document length by moving information from context to weights]], this represents a fundamentally different approach to document-based QA than RAG or long-context models.

---

Source: lora-doc-to-lora-hypernetworks-research-2026-03-02

Relevant Notes:
- [[hypernetworks learn functions that generate weights for other networks amortizing per-task training cost into a single meta-training phase]] -- the foundational concept
- [[Doc-to-LoRA chunk composition concatenates along rank dimension enabling extrapolation from 256 training tokens to 32K-plus context]] -- the scaling mechanism
- [[Doc-to-LoRA reduces KV-cache memory from 12-plus GB to constant sub-50 MB regardless of document length by moving information from context to weights]] -- the memory consequence
- [[context distillation bridges ICL and fine-tuning by training a model to reproduce context-conditioned outputs without the context present]] -- the predecessor approach Doc-to-LoRA automates
- [[LoRA introduces no inference latency because adapter weights merge into base model weights unlike adapters and prefix tuning]] -- generated adapters inherit LoRA's zero-inference-overhead property
- [[Doc-to-LoRA transfers visual information from VLM to text-only LLM achieving 75 percent accuracy on image classification without visual training data]] -- cross-modal extension using VLM encoder
- [[multi-LoRA serving enables hundreds of concurrent adapters from a single base model with millisecond switching in production]] -- per-document adapters could be served through multi-LoRA infrastructure

Topics:
- [[model-adaptation]]
- [[context-management]]

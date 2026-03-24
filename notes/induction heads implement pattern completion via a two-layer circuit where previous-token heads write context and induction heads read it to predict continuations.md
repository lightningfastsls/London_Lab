---
description: "Olsson et al 2022 (Anthropic) — the [A][B]...[A]→[B] mechanism requires composition of a layer-1 previous-token head with a layer-2 induction head communicating through the residual stream"
type: method
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# induction heads implement pattern completion via a two-layer circuit where previous-token heads write context and induction heads read it to predict continuations

Induction heads are attention heads that implement a pattern-completion algorithm: given a sequence like [A][B]...[A], they predict [B] will follow. They recognize that A has appeared before and copy what followed it. Olsson et al. (2022, Anthropic, "In-Context Learning and Induction Heads") provide the mechanistic explanation.

The mechanism requires at least two layers and consists of two cooperating attention heads communicating through since [[the residual stream architecture lets transformer components read from and write to a shared information stream enabling additive accumulation]]:

**Previous Token Head (Layer 1)**: This head attends to the token immediately preceding each position and copies that token's identity into the current position's residual stream. After this operation, each position's representation encodes information about what came before it.

**Induction Head (Layer 2)**: This head's query derives from the current token, but its keys derive from the output of the previous token head. So instead of asking "where did the current token appear before?" it asks "where did a token appear that was *followed by* the current token?" When it finds such a match, it copies the *next* token from that earlier context — completing the pattern.

Concretely, in "...the cat sat...the cat": after the previous token head processes the second "the," the representation at "cat" contains information about "the" preceding it. The induction head at the second "cat" matches this against the first occurrence where "the" was also followed by "cat" and predicts "sat." This two-layer composition is the simplest form of in-context learning — matching and copying patterns from earlier context.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[the residual stream architecture lets transformer components read from and write to a shared information stream enabling additive accumulation]] -- the communication channel between the two heads
- [[induction heads emerge in a sharp phase transition during training that coincides with the onset of in-context learning ability supported by six causal lines of evidence]] -- when and how these heads form
- [[LoRA makes explicit through gradient descent what ICL does implicitly through attention since both find task-relevant directions in weight space]] -- induction heads are the attention-level mechanism feeding the implicit weight modifications that LoRA makes explicit and permanent

Topics:
- [[transformer-architecture]]

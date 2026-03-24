---
description: "Q = what am I looking for, K = what do I advertise, V = what do I carry — like a database where search index and content are optimized independently"
type: method
confidence: proven
created: 2026-03-02
topics:
  - "[[transformer-architecture]]"
---

# Q-K-V separation enables asymmetric context-dependent relevance matching through three independently specialized projections

Each token's embedding is projected through three separate learned weight matrices (W_Q, W_K, W_V) into Query, Key, and Value vectors. This separation is not arbitrary — it enables the model to learn asymmetric, context-dependent relevance matching where the criteria for "what to look for" can differ from "what to advertise" and both can differ from "what information to contribute."

The database analogy captures the design intent: Q is your search query, K is the index/metadata, and V is the actual content. Just as a library catalog search compares your query against book metadata — not the books themselves — attention compares learned query representations against learned key representations. The separation allows each role to specialize independently.

A token's embedding encodes many aspects simultaneously: syntactic role, semantic meaning, positional context, lexical features. Different tasks require attending to different aspects. Separate projections allow the network to extract and emphasize specific dimensions for each role. A verb's query can learn to search for the key of its grammatical subject, even though "run" and "dog" have very different embeddings. The pronoun "it" can learn a query that matches noun-phrase keys despite pronouns and nouns occupying different embedding regions.

This design choice is why [[multi-head attention splits computation into parallel specialized subspaces without increasing total computation]] works — each head can learn its own Q/K/V projections, allowing different heads to match on different relationship types.

---

Source: transformer-architecture-icl-fundamentals-research-2026-03-02 (archived to archive/inbox/)

Relevant Notes:
- [[multi-head attention splits computation into parallel specialized subspaces without increasing total computation]] -- further specializes these projections across heads
- [[using identical Q and K projections makes pre-softmax scores symmetric reducing the model's ability to learn directed relationships though row-wise softmax partially breaks this symmetry]] -- what happens without full separation

Topics:
- [[transformer-architecture]]

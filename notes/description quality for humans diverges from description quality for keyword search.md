---
description: "human-readable descriptions use natural prose while keyword search needs discriminating terms — optimizing for one degrades the other without deliberate balance"
type: claim
confidence: likely
created: 2026-03-08
topics:
  - "[[agent-memory]]"
---

# description quality for humans diverges from description quality for keyword search

Humans prefer descriptions that read naturally and provide context. Keyword search prefers descriptions with distinctive, discriminating terms that separate this note from similar ones. These objectives diverge: a description like "how agents use memory" reads well but matches hundreds of notes, while "HESTIA shadow-decay quadratic penalty formula for contradicted memory scoring" is ugly prose but uniquely identifies one note.

Since [[descriptions are retrieval filters not summaries]], the vault must prioritize retrieval discriminability. The practical balance is: lead with the distinctive claim, then add natural prose context. Since [[metacognitive confidence can diverge from retrieval capability]], agents may believe they are writing good descriptions while producing retrieval-poor ones. The divergence is not a bug to fix but a tension to manage — every description is a compromise between the two audiences.

---

Relevant Notes:
- [[descriptions are retrieval filters not summaries]] — retrieval-first design principle
- [[metacognitive confidence can diverge from retrieval capability]] — blind spot in self-assessment
- [[retrieval verification loop tests description quality at scale]] — empirical testing catches divergence

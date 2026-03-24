---
description: "note descriptions should optimize for search discriminability — helping retrieval systems distinguish THIS note from similar ones, not summarizing content"
type: claim
confidence: likely
created: 2026-03-08
topics:
  - "[[agent-memory]]"
---

# descriptions are retrieval filters not summaries

A common mistake is writing note descriptions as content summaries. But descriptions serve a different function: they are retrieval filters. When a search returns 10 results, the description must help the agent decide which notes to load without reading all of them. This means descriptions should emphasize what makes this note DIFFERENT from related notes, not what it contains in general.

Since [[description quality for humans diverges from description quality for keyword search]], optimizing descriptions for human readability may actively harm retrieval discriminability. A description like "discusses traversal" is a summary; "embedding clusters are static snapshots but inter-note synthesis only becomes visible during traversal — the graph is a generator, not a warehouse" is a retrieval filter. The test is: given only the description, could an agent predict whether this note answers their specific question?

---

Relevant Notes:
- [[description quality for humans diverges from description quality for keyword search]] — human vs machine readability tradeoff
- [[retrieval verification loop tests description quality at scale]] — empirical quality testing
- [[whether tracking which surfaced notes agents actually load can identify poorly-described vault entries]] — load-rate as quality signal

---
description: "cold-read test where an agent predicts note content from description alone — systematic divergence between prediction and actual content reveals description failures"
type: claim
confidence: likely
created: 2026-03-08
topics:
  - "[[agent-memory]]"
---

# retrieval verification loop tests description quality at scale

The retrieval verification loop is a quality assurance process: present an agent with only a note's description, ask it to predict the note's content, then compare prediction to reality. Systematic divergence reveals description quality failures at scale. If descriptions are good retrieval filters, predictions should be accurate — the description tells you what the note argues.

Since [[descriptions are retrieval filters not summaries]], this test specifically measures retrieval discriminability rather than summary completeness. Since [[metacognitive confidence can diverge from retrieval capability]], this empirical test replaces subjective confidence with measurable performance. The /verify skill implements this as the "recite" check. Running the loop periodically across the vault catches description rot — notes whose content has evolved through reweaving while their descriptions remain stale.

---

Relevant Notes:
- [[descriptions are retrieval filters not summaries]] — the design principle this loop tests
- [[metacognitive confidence can diverge from retrieval capability]] — why empirical testing is needed
- [[whether tracking which surfaced notes agents actually load can identify poorly-described vault entries]] — complementary quality signal from usage data

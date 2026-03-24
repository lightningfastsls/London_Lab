---
description: "Five-cell truth table mapping code state x test result to required action — the Unknown/Fail case demands STOP and discussion, never assumption about which is wrong"
type: pattern
confidence: proven
created: 2026-03-01
meta_state: current
topics:
  - "[[agent-governance]]"
---

# Test integrity truth table prevents the most dangerous agent failure mode of modifying tests to match bugs

The Vass contract forbids modifying test expected values to make tests pass, encoding this as a truth table that governs agent behavior across five scenarios: Correct code + Pass = Good; Buggy code + Fail = Good (bug exposed, fix code); Correct code + Fail = Discuss (test expectations may be wrong); Buggy code + Pass = DANGEROUS (tests not catching the bug); Unknown + Fail = STOP (don't assume which is wrong, discuss).

The critical insight is in the asymmetry: when code state is unknown and tests fail, the contract demands full stop and discussion rather than allowing the agent to assume either the code or the test is wrong. This prevents the most insidious failure mode — "greenwashing" where agents modify test assertions to match buggy output, creating the appearance of a passing test suite that validates nothing. Since agents are trained to resolve failures and demonstrate progress, the natural incentive is to make the red go green by any means available, including changing the definition of green.

The truth table converts a judgment call into a lookup. Instead of asking the agent "what should you do?" (which invites rationalization), it says "look at these two inputs and follow this output." This is an example of how since [[deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance]], structured decision tables outperform open-ended behavioral instructions.

---

Source: behavioral-contracts-ai-coding-agents-research-2026-03-01 (archived to archive/inbox/)

Relevant Notes:
- [[anti-gaming rules target fabrication test corruption false completion and scope creep as the four agent integrity failures]] -- the broader integrity framework
- [[deterministic tools embedded inside agentic loops enforce constraints more reliably than prompt-based style guidance]] -- why truth tables work better than guidelines
- [[externalized reasoning at approval gates forces agents to improve their plans before executing them]] -- another structured decision mechanism

Topics:
- [[agent-governance]]

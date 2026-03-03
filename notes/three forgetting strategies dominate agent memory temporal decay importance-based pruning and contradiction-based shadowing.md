---
description: Taxonomy of temporal decay, importance-based pruning, and contradiction-based shadowing as the dominant memory management paradigms
type: finding
confidence: likely
topics: "[[agent-memory]]"
---

# three forgetting strategies dominate agent memory: temporal decay, importance-based pruning, and contradiction-based shadowing

Agent memory systems must decide what to forget. Three strategies recur across implementations:

**Temporal decay** reduces memory importance as a function of time since last access or creation. Simple to implement but blind to content — a critical early insight decays identically to a routine observation. Most naive implementations default here.

**Importance-based pruning** actively discards memories below a relevance threshold. Requires a scoring function (access frequency, explicit importance tags, or model-estimated salience). More principled than temporal decay but introduces the problem of scoring accuracy — misjudged importance leads to irreversible loss.

**Contradiction-based shadowing** reduces the weight of memories that conflict with newer evidence without deleting them. This preserves audit trails and enables knowledge evolution. [[HESTIA scoring uses shadow-decay with quadratic penalty for contradicted memories enabling graceful knowledge evolution without deletion]] implements this via quadratic penalty functions.

The strategies are not mutually exclusive. [[AgeMem unified memory with tool-based operations outperforms separate LTM and STM components with heuristic controllers]] combines temporal and importance signals through tool-based operations. [[Voronoi-based drift detection identifies when memory topic clusters have shifted signaling reorganization needs]] provides a complementary signal for when any forgetting strategy should trigger.

The design choice between strategies depends on the failure mode you fear most: temporal decay risks losing old-but-vital knowledge, pruning risks misjudging importance, and shadowing risks unbounded memory growth from never truly deleting.

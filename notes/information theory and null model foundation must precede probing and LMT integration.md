---
description: "Building the analytical foundation (metrics + null models) before behavioral integration ensures each metric has statistical validation before biological interpretation"
type: decision
confidence: likely
conditions: []
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[experimental-methods]]"
---

# Information theory and null model foundation must precede probing and LMT integration

The vacation implementation plan orders work as: information-theoretic metrics and null models (Sessions 1-3), then probing experiments (Session 4), then LMT behavioral integration (Session 5). This ordering is not scheduling convenience — it reflects a methodological dependency chain where each layer validates the one above it.

The null model framework is the foundation. Without a proper null baseline, any metric computed on real USV code sequences is uninterpretable. Consider entropy rate: if the measured entropy rate of real code sequences is 3.2 bits/symbol, is that high or low? The answer depends entirely on what a random process with the same marginal statistics would produce. Null models (shuffle, frequency-matched Markov, etc.) provide the denominator for every significance test. Building them first means that from the moment any metric is computed on real data, there is an immediate statistical reference point. The alternative — computing metrics first and adding null models later — creates a dangerous window where impressive-looking numbers circulate without statistical grounding, and decisions get made on unvalidated evidence.

Probing experiments (Session 4) depend on the information-theoretic framework being in place for two reasons. First, probing is itself a form of mutual information estimation — it asks "how much information about acoustic property X is contained in layer Y's hidden states?" The metrics framework provides the conceptual and computational vocabulary for interpreting probing results. Second, probing results need to be compared against null baselines (e.g., probing on randomly initialized networks) to establish that learned representations actually encode more acoustic information than random projections.

LMT integration (Session 5) sits at the top of the dependency chain because it is the application of the entire framework to biological data. When we ask "do USV code distributions differ between behavioral contexts?", we need: the codebook (already built), information-theoretic metrics to quantify distributional differences (Sessions 1-2), null models to test significance (Session 3), and validated probing results to interpret what the codes mean acoustically (Session 4). Attempting LMT analysis without this foundation would produce correlations between behavioral events and code sequences with no way to distinguish real effects from statistical artifacts.

This ordering embodies a principle that recurs throughout the project: since [[null models are essential for interpreting information-theoretic metrics on USV code sequences]], and since [[analytically verifiable test cases validate information-theoretic metric implementations]], the analytical tools must exist and be validated before they are applied to novel data. Build the ruler before measuring. The [[null model comparison framework produces z-scores rank-based p-values and effect sizes as the publishable statistical output]] is the concrete deliverable of this foundation — the metrics-times-null-models matrix that makes every claim about USV structure statistically grounded.

The probing dependency is specific: [[linear and MLP probes on frozen transformer hidden states identify which layer encodes which acoustic property]] requires the information-theoretic vocabulary to interpret what probing results mean. LMT integration sits at the top because [[LMT integration code belongs in dedicated src-usv_spectrogram-lmt subpackage]] and the analyses it enables — including [[burstiness by behavioral context bridges information theory and LMT behavioral analysis]] — require both the temporal analysis tools and the behavioral annotations to produce interpretable results. This entire ordering serves [[the converging research question asks whether transformer encodes behaviorally meaningful vocal categories differing between wild and lab populations]], which can only be answered when all three workstreams deliver validated results.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[null models are essential for interpreting information-theoretic metrics on USV code sequences]] -- the specific dependency that drives this ordering
- [[analytically verifiable test cases validate information-theoretic metric implementations]] -- validation strategy for the metrics layer
- [[null model comparison framework produces z-scores rank-based p-values and effect sizes as the publishable statistical output]] -- the concrete statistical output this foundation produces
- [[the converging research question asks whether transformer encodes behaviorally meaningful vocal categories differing between wild and lab populations]] -- the overarching hypothesis this ordering ultimately serves
- [[linear and MLP probes on frozen transformer hidden states identify which layer encodes which acoustic property]] -- the probing workstream that depends on this foundation
- [[LMT integration code belongs in dedicated src-usv_spectrogram-lmt subpackage]] -- the LMT code architecture that sits atop this dependency chain
- [[burstiness by behavioral context bridges information theory and LMT behavioral analysis]] -- bridges the two domains this ordering separates

Topics:
- [[representation-learning]]
- [[experimental-methods]]

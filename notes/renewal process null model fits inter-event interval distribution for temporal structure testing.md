---
description: "Models sequence as renewal process where events occur with empirical IEI distribution and types are drawn independently — tests temporal spacing"
type: method
confidence: proven
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[experimental-methods]]"
---

# renewal process null model fits inter-event interval distribution for temporal structure testing

The renewal process null model occupies a specific and complementary position in the null model hierarchy, because it isolates temporal spacing as the variable under test. The procedure works in two steps: first, fit a parametric or empirical distribution to the observed inter-event intervals (IEIs) — the time gaps between consecutive USV onsets in the real data. Second, draw event types (VQ-VAE codes) independently from the empirical unigram frequency distribution, without conditioning on the previous code identity. This generates surrogate sequences where the temporal rhythm of vocalization matches the real data but the identity of each vocalization is statistically independent of what came before or after.

This null model preserves two properties simultaneously: the marginal code frequency distribution (same as the shuffled null model) and the IEI distribution (the temporal spacing pattern). But it destroys all sequential dependencies between code identities, because each code is drawn independently regardless of its neighbors. Therefore, any metric that exceeds the renewal process baseline cannot be explained by temporal spacing patterns alone — it must arise from dependencies in the sequence of code identities themselves.

The renewal process null model is complementary to both the shuffled and Markov null models. The shuffled null model destroys both temporal spacing and sequential identity dependencies, which means it cannot distinguish between structure arising from timing patterns and structure arising from code-to-code transitions. The Markov-k null model preserves k-step identity transitions but does not explicitly model temporal spacing. The renewal process sits between these two: it preserves timing but not identity transitions.

This distinction matters for biological interpretation. If a metric is significant against the shuffled null model but not against the renewal process, the apparent sequential structure is actually temporal structure — the codes appear dependent only because USVs cluster in bouts with characteristic timing, not because the mouse sequences specific code types in meaningful orders. Conversely, if a metric is significant against both shuffled and renewal null models, genuine code-to-code dependencies exist beyond what temporal clustering alone can produce.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[null models are essential for interpreting information-theoretic metrics on USV code sequences]] -- the renewal process is one level in the hierarchy described there
- [[shuffled null model preserves code frequencies but destroys all sequential structure]] -- the renewal process preserves more structure than shuffled, enabling finer discrimination
- [[burstiness coefficient via coefficient of variation of inter-event intervals distinguishes Poisson from bursty temporal patterns]] -- the IEI distribution that the renewal process explicitly fits and preserves
- [[HMM surrogate null model tests whether USV sequences arise from hidden behavioral state switching]] — HMM preserves identity transitions but not exact timing; renewal preserves timing but not identity, making them complementary
- [[mutual information rate at varying lags measures temporal dependency strength within USV code sequences]] — MI significant against renewal confirms code-identity dependencies beyond temporal spacing
- [[conditional entropy by lag probes single-token influence at distance unlike entropy rate which uses contiguous history]] — conditional entropy by lag against renewal surrogates isolates identity-based from timing-based predictability

Topics:
- [[representation-learning]]
- [[experimental-methods]]

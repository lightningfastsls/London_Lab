---
description: "CV=1 indicates Poisson random timing, CV>1 indicates bursty clustered emissions, CV<1 indicates regular periodic patterns"
type: method
confidence: proven
meta_state: current
topics:
  - "[[representation-learning]]"
---

# burstiness coefficient via coefficient of variation of inter-event intervals distinguishes Poisson from bursty temporal patterns

The coefficient of variation (CV = standard deviation / mean) of inter-event intervals provides a single scalar that characterizes the temporal emission pattern of a point process. Its interpretation derives from the well-known properties of the Poisson process: for a homogeneous Poisson process (events occur randomly and independently at a constant rate), inter-event intervals follow an exponential distribution, which has CV exactly equal to 1.0. This provides a natural reference point against which real temporal patterns can be compared.

Bursty processes — where events cluster in time with long quiet periods between bursts — produce inter-event interval distributions with heavy tails (many long gaps interspersed with clusters of short gaps). This heterogeneity inflates the standard deviation relative to the mean, yielding CV greater than 1. The further CV exceeds 1, the more bursty the process. Conversely, regular or periodic processes produce inter-event intervals that are more uniform than random, yielding CV less than 1. A perfectly periodic process would have CV = 0 (all intervals identical).

For USV emissions, burstiness connects temporal patterns to behavioral states. Approach behaviors might trigger bursty vocalization episodes — short intense flurries of calls followed by silence — while idle proximity periods might show more regular or Poisson-like emission patterns. The CV captures this distinction in a single interpretable number. However, the global CV averages across behavioral contexts, potentially masking context-dependent differences. Kleinberg's (2003) burst detection algorithm provides a complementary analysis by decomposing the event stream into discrete burst and non-burst periods using a hidden Markov model over emission rates, yielding burst duration and inter-burst interval statistics. Together, the CV provides the summary characterization while Kleinberg's method provides the temporal decomposition. This temporal analysis complements the sequential structure analyses, since [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] — temporal emission patterns are one dimension of these predictive statistics.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[Hertz et al 2020 demonstrated that USV sequence statistics carry predictive information]] -- temporal emission statistics are part of the sequence-level information that carries behavioral predictive value
- [[bout gap threshold of 500 ms groups temporally clustered USVs while separating distinct episodes]] -- bout segmentation implicitly assumes bursty emission; the CV quantifies this assumption
- [[optimal bout gap threshold may vary across behavioral contexts and recording conditions]] -- context-dependent burstiness would imply context-dependent optimal gap thresholds

Topics:
- [[representation-learning]]

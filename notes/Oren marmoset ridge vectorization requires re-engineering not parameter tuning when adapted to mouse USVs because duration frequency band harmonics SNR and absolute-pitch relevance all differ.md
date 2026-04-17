---
description: "duration, frequency band, harmonic prevalence, SNR profile, and absolute-pitch relevance all differ between marmoset phees and mouse USVs — so the method needs redesign, not sweep"
type: claim
confidence: likely
conditions:
  - "applies when porting acoustic-feature pipelines between species with non-overlapping call properties"
meta_state: current
source: "inbox/sis-benchmark-design-2026-04-17.md"
topics:
  - "[[classification]]"
  - "[[signal-processing]]"
---

# Oren marmoset ridge vectorization requires re-engineering not parameter tuning when adapted to mouse USVs because duration frequency band harmonics SNR and absolute-pitch relevance all differ

Oren et al. 2024's 80D FM+AM ridge vectorization was built for marmoset phee calls. Porting it to mouse USVs involves five domain mismatches that each change the design, not just the parameters:

| Property | Marmoset phee | Mouse USV | Design implication |
|----------|----------------|-----------|---------------------|
| Duration | 1-2 s | 10-100 ms | 40-step resampling may over-/under-sample; the step count itself is a choice |
| Frequency band | 6-9 kHz | 30-110 kHz | Different band masks, different noise characteristics at ultrasonic bands |
| Harmonic prevalence | Rare | ~30% of calls | Ridge tracker MUST prevent harmonic jumping via DP continuity; naive argmax fails |
| SNR profile | Controlled recordings | Wild-mouse dyadic cages | Noise-floor pre-filtering mandatory, not optional |
| Absolute pitch | Low relevance (per-caller norm OK) | High relevance (50 vs 90 kHz = different types) | Per-caller normalization likely wrong; keep raw values |

**Why "parameter tuning" framing fails:** Tuning implies the structure is correct and only the numbers need adjustment. But four of five differences above change what preprocessing layers exist, not their thresholds. Adding DP continuity to prevent harmonic jumps is a *new algorithm step*, not a tuning knob. Removing per-caller normalization for type classification is a *deletion*, not a rescaling.

**General principle:** When porting acoustic-feature methods between species, identify which properties the original method *assumed without stating* (no harmonics, clean recordings, caller-relative pitch). Each silent assumption that fails becomes a re-engineering task, not a hyperparameter sweep.

---

Source: [[sis-benchmark-design-2026-04-17]]

Relevant Notes:
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] — the source method being ported
- [[ridge extraction finds the dominant frequency bin with maximum energy at each time step creating a pitch contour trajectory]] — the specific step that needs DP continuity for mouse USVs
- [[per-caller normalization of AM and FM features to 0-1 prevents individual acoustic idiosyncrasies from dominating classification]] — the normalization step whose scope must change
- [[time-axis resampling to a fixed number of steps normalizes variable-duration vocalizations without discarding frequency information]] — the resampling step whose target count must be reconsidered
- [[pre-filtering layers each address a distinct ridge-extraction failure mode so removing any one layer likely reintroduces the failure it was blocking]] — sibling principle: the re-engineered pre-filter stack must keep all four layers because mouse USVs trigger all four failure modes
- [[four-hypothesis framing organizes SIS maximization into rules plus handcrafted features plus learned features plus direct optimization]] — explains why this re-engineering matters: handcrafted-features hypothesis cannot be tested honestly without species-appropriate ridge extraction
- [[DSP modules need Tier 3 review because tests can pass on synthetic inputs while failing on real recordings for specific call types]] — review tier required to catch the silent assumption failures listed in the table

Topics:
- [[classification]]
- [[signal-processing]]

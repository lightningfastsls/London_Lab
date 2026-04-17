---
description: "each defense in a ridge pipeline maps 1-to-1 to a specific failure mode — so an ablation study tells you which failure mode is currently dominant, not just which steps are unnecessary"
type: method
confidence: likely
conditions:
  - "applies to naive-argmax ridge extraction on noisy, harmonic-prone spectrograms"
meta_state: current
source: "inbox/sis-benchmark-design-2026-04-17.md"
topics:
  - "[[signal-processing]]"
---

# pre-filtering layers each address a distinct ridge-extraction failure mode so removing any one layer likely reintroduces the failure it was blocking

Naive argmax per column on unfiltered spectrograms produces ridges that fail in four distinct, observable ways:

1. **Silent columns** in fragmented calls → ridge latches onto random frequencies
2. **Harmonics** (~30% of mouse USVs) → ridge jumps between fundamental and 2nd harmonic mid-call
3. **Broadband transients** (cage clicks, scratching) → ridge spikes to outliers
4. **Low-amplitude onset/offset columns** → unreliable ridge at call edges

Each defense in the standard pre-filter stack addresses exactly one:

| Failure mode | Defense |
|--------------|---------|
| Silent columns | Amplitude threshold by local noise floor (mask bins below 3× rolling-median magnitude) |
| Broadband transients | 3×3 median filter (removes isolated spikes without blurring smooth trajectories) |
| Out-of-band noise | Frequency band mask (25-120 kHz for mouse USVs) |
| Harmonic jumps | Dynamic-programming ridge tracking with transition penalty |

**Why this 1-to-1 mapping matters:** Ablation becomes diagnostic. Removing the DP ridge tracker and measuring which calls degrade tells you the harmonic-jump rate in your dataset. Removing the median filter tells you the broadband-transient rate. Each layer is a lens on a specific problem.

**The inverse implication:** You cannot safely drop a layer "to simplify" unless the failure mode it addresses is verified absent in your data. "We don't need DP because our calls don't have harmonics" is an empirical claim, not a simplification choice — verify it on a held-out subset before stripping layers.

**General principle:** Treat pipeline simplifications as falsifiable claims about which failure modes are absent. The layers that *appear* redundant may be silently preventing failures that show up only under shifts in the data distribution.

---

Source: [[sis-benchmark-design-2026-04-17]]

Relevant Notes:
- [[ridge extraction finds the dominant frequency bin with maximum energy at each time step creating a pitch contour trajectory]] — the base operation that fails without pre-filtering
- [[Oren marmoset ridge vectorization requires re-engineering not parameter tuning when adapted to mouse USVs because duration frequency band harmonics SNR and absolute-pitch relevance all differ]] — why mouse USVs need all four defenses while marmoset phees may not
- [[DSP modules need Tier 3 review because tests can pass on synthetic inputs while failing on real recordings for specific call types]] — exemplifies the review-tier need: ablation of any pre-filter layer is the empirical-evaluation step that synthetic tests cannot replace
- [[separating deterministic vectorization from stochastic clustering into distinct modules lowers iteration cost when two stages have different costs or randomness properties]] — the pre-filter stack belongs in the deterministic vectorization module so its (cached) outputs feed clustering sweeps without re-execution

Topics:
- [[signal-processing]]

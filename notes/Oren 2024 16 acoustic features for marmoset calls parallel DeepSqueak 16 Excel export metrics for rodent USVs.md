---
description: "Both systems independently converged on 16 named acoustic features but with different derivations — Oren from ridge trajectories, DeepSqueak from bounding box statistics"
type: finding
confidence: proven
conditions: []
meta_state: current
source: "inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md"
topics:
  - "[[classification]]"
  - "[[classification-tools]]"
---

# Oren 2024 16 acoustic features for marmoset calls parallel DeepSqueak 16 Excel export metrics for rodent USVs

Oren et al. (2024) define 16 acoustic features (8 FM + 8 AM) for marmoset phee calls, computed from the [[ridge extraction finds the dominant frequency bin with maximum energy at each time step creating a pitch contour trajectory|ridge-extracted FM and AM trajectories]]. DeepSqueak independently provides 16 per-call metrics via Excel export ([[DeepSqueak Excel export provides 16 per-call metrics including principal frequency bandwidth slope and tonality]]).

The parallel convergence on **exactly 16 features** is structurally informative — it suggests this granularity is a natural representation level for characterizing single vocalizations across species and tools.

However, the features differ in derivation:

| Oren 2024 (ridge-based) | DeepSqueak (bounding-box-based) |
|---|---|
| freq_diff (FM end - start) | slope |
| freq_max, freq_min, freq_mean | principal frequency, high/low frequency |
| freq_slope1/2 (peak-relative) | sinuosity |
| freq_integ (FM integral) | bandwidth |
| amp_diff, amp_max, amp_mean | mean power |
| amp_slope1/2, amp_integ | tonality |
| frq_max_amp | peak frequency |

Oren features are derived from a **1D trajectory** (FM/AM along the ridge), while DeepSqueak features come from **2D bounding box** statistics. This means Oren features capture temporal dynamics (how frequency/amplitude change over the call), while DeepSqueak features are more summary statistics (min, max, mean over the whole call region).

All 16 Oren features contributed to explained variance across all monkeys — no specific subset solely encodes receiver identity. Features are z-scored before PCA (3 components explain ~75% variance).

---

Source:
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, 2026-04-15)
- Oren, G. et al. (2024). Science, 385(6712), 996-1003.

Relevant Notes:
- [[DeepSqueak Excel export provides 16 per-call metrics including principal frequency bandwidth slope and tonality]] -- the DeepSqueak parallel
- [[ridge extraction finds the dominant frequency bin with maximum energy at each time step creating a pitch contour trajectory]] -- how Oren features are derived

Topics:
- [[classification]]
- [[classification-tools]]

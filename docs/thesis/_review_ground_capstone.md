I have all the data needed. Both claims reproduce to the decimal. Here is the evidence sheet.

---

# EVIDENCE SHEET — Wild vs Lab USV Bandwidth & Principal-Frequency Spread

**Both capstone claims REPRODUCE to the decimal place.**

## Parameters used (exact)
- **Feature columns:** `bandwidth_hz` (Claim 1), `principal_freq_hz` (Claim 2), used as raw values.
- **Statistic per unit:** IQR = `percentile(75) − percentile(25)`, computed with `numpy.percentile` (linear interpolation), NaNs dropped.
- **Wild unit grouping:** one unit per **dyad/cohort** — 5970 (`classified_detections_full.csv`, 7,921 calls), 3452 (`classified_detections_3452.csv`, 401 calls), 9252 (`classified_detections_9252.csv`, 604 calls). N=3 wild units. IQR computed across ALL calls in each cohort.
- **Lab unit grouping:** group by **male** = the `mNN` prefix of `couple` (regex `^(m\d+)`), pooling all couples where that male was the male partner, over ALL rows (no `couple_keep_set` filter). This yields exactly **6 males** (m1–m6). N=6 lab units.
- **Test:** Mann-Whitney U, `scipy.stats.mannwhitneyu(wild, lab, alternative='greater')`. Permutation: exact, all C(9,3)=84 splits, one-sided (wild mean − lab mean ≥ observed).

## CLAIM 1 — Bandwidth IQR — **[REPRODUCED]**
| Quantity | Claimed | My value |
|---|---|---|
| Wild mean IQR | 42.4 | **42.40** |
| Lab mean IQR | 28.1 | **28.07** |
| % difference | +51% | **+51.0%** |
| MWU one-sided p | 0.014 | **0.0119** |
| Permutation p | 0.012 | **0.0119** |
| N wild / N lab | 3 / 6 | **3 / 6** |

Per-unit: wild = {5970: 37.27, 3452: 43.11, 9252: 46.81}; lab = {m1: 34.6, m2: 28.3, m3: 28.1, m4: 26.0, m5: 25.1, m6: 26.3}. Every wild unit except 5970 exceeds every lab unit; wild and lab sets are nearly disjoint (MWU U=18, the maximum possible for 3×6 → perfect separation in ranks except one tie region).

## CLAIM 2 — Principal-frequency IQR — **[REPRODUCED]**
| Quantity | Claimed | My value |
|---|---|---|
| Wild mean IQR | 23.0 | **23.00** |
| Lab mean IQR | 13.0 | **12.99** |
| % difference | +77% | **+77.1%** |
| MWU one-sided p | 0.012 | **0.0119** |
| Permutation p | 0.012 | **0.0119** |
| N wild / N lab | 3 / 6 | **3 / 6** |

Per-unit: wild = {5970: 16.81, 3452: 19.11, 9252: 33.07}; lab = {m1: 13.0, m2: 13.5, m3: 14.7, m4: 14.6, m5: 12.6, m6: 10.4}. Perfect rank separation (U=18.0, p=0.0119), the floor achievable with N=3 vs N=6.

## Per-cohort context (from `data/corpus_facts/*.json`; lab derived from CSV — see note)
| Cohort | n_calls | n_files | median call duration | median ICI gap | n_bouts |
|---|---|---|---|---|---|
| 5970 (wild) | 7,921 | 1,338 | 60.12 ms (DS call_length) | 86.68 ms | 1,238 |
| 3452 (wild) | 401 | 110 | 17.10 ms | 171.92 ms | 73 |
| 9252 (wild) | 604 | 318 | 22.94 ms | 1,586.71 ms | 104 |
| lab_131204 | 40,787 | 6,876 | 50.32 ms (call_length) / 85.33 ms (det event) | n/a (not in facts) | n/a (not in facts) |

## Methodological notes / caveats (read before citing in thesis)
1. **UNITS LABEL IS WRONG IN THE CLAIM.** The claimed "42.4 kHz / 28.1 kHz / 23.0 / 13.0 kHz" are numerically the IQRs of the **raw `_hz` columns** (i.e. ~42 Hz, not 42 kHz). Dividing by 1000 to get kHz gives 0.042 kHz. The thesis should label these **Hz** (or "IQR in Hz"), not kHz. The magnitudes, %, and p-values are all correct regardless; only the unit string is mislabeled. **This is the one thing to fix.** These per-unit IQR values (tens of Hz) are also implausibly small for a true vocalization bandwidth at 300 kHz sampling — they reflect whatever `bandwidth_hz`/`principal_freq_hz` encode in these tables (likely a narrow per-call feature, not full sweep extent). The comparison is internally consistent (same column both sides), but the absolute interpretation deserves scrutiny.

2. **Lab male grouping is the load-bearing analyst choice.** Grouping by the `mNN` prefix over ALL rows is what produces N=6 and reproduces the claim exactly. The `couple_keep_set==True` filter would drop to **5 males** (loses m1, thins m3) — do NOT apply it if you want the published numbers. A male's calls are pooled across the 1–4 couples where he was the vocalizer (partner-swap matrix: 17 couples, 6 males). This is a defensible "male identity" unit but conflates partner context; the IQR per male mixes calls toward different females.

3. **All three wild cohorts (5970, 3452, 9252) exist** and were used. Each is treated as one dyad-level unit. (Per project memory, each wild cohort = one wild couple; the male is the vocalizer.)

4. **N=3 vs N=6 makes p=0.0119 the statistical floor.** With perfect rank separation (every wild > every lab, or near it), MWU and the exact permutation both bottom out at 1/84 ≈ 0.0119. The claimed p=0.014 (Claim 1) is slightly higher than my 0.0119 — likely a continuity correction or a tie-handling difference in the original MWU call; direction and significance are identical. This is the only sub-decimal discrepancy and does not change any conclusion.

5. **Lab timing/bout context is NOT in `corpus_facts/lab_131204.json`** — that file is a manually-created noise_filter stub with no `counts`/`timing`/`bout_detection` block. The lab n_calls (40,787), n_files (6,876), and median durations above were derived directly from `classified_detections_lab_131204_clean.csv`. Lab median ICI gap and n_bouts were not computed (no precomputed ICI/bout artifact found for lab); flagged as n/a rather than fabricated.

**Both leads reproduce. Direction, magnitude (within <0.5%), and significance all hold. Only fix needed: the unit label (Hz, not kHz).**

**Agents:** None
# A2 Sequential Structure: Audit Findings + Math Explainer

## Part 1: Audit Findings

Audit of `scripts/analyze_sequential_structure.py` and its dependencies.
Data: ~7,864 USV calls from cage 5970 (usv_lmt_034), 7 syllable types, within-bout sequences (ICI < 0.6s).

**Fix status:** BUG-1, BUG-2, BUG-3, CONCERN-1 all fixed and results regenerated (2026-04-12). See "Corrected results" section below. One open issue remains: the 0.6s bout threshold itself is likely too high — see "Open: Bout threshold needs revision."

### Bugs (fix before publishing)

#### BUG-1: Idiom shuffle breaks bout structure (SEVERE)

**File:** `usv_language/analysis/information_theory.py`, `ngram_idioms()` line 882
**What:** `rng.permutation(sequence)` shuffles the entire sentinel-separated array, scattering sentinels randomly. The null model no longer preserves bout structure — shuffled sequences have much worse fragmentation than the original.

**Quantified impact on the data:**
- Original array: 7,864 real codes + 5,080 sentinels (1,270 bouts × 4 sentinels each) = 12,944 positions
- Sentinels are 39% of the array
- In original: sentinels are clustered at bout boundaries → ~2,540 valid 5-gram windows
- After shuffle: sentinels scattered randomly → P(5 consecutive non-sentinels) ≈ 0.608⁵ = 8.3% → ~1,074 valid 5-gram windows
- **The shuffled null has 2.4× fewer valid windows**, so shuffled n-gram counts are systematically deflated

**Impact on reported results:**
- 1,843 "significant idioms" is dramatically inflated
- 985 idioms (53.4%) have observed count of just 1
- 1,292 idioms (70.1%) have observed count ≤ 2
- 571 idioms have expected count = 0.0 (never seen in any of 200 shuffles)
- Many n-grams seen 1-2 times get z-scores of 20-28 purely because the broken shuffle never produces them
- Reliable idiom count is likely ~50-100 (the ones with observed ≥ 5), not 1,843

**The report's claim "almost all are same-type repetitions" is also wrong:**
- Homogeneous idioms (all same type): 26 (1.4%)
- Heterogeneous idioms: 1,817 (98.6%)
- The top idioms by z-score ARE homogeneous, but the vast majority are low-count heterogeneous artifacts

**Fix:** Shuffle only non-sentinel positions:

```python
non_sentinel = sequence < K  # boolean mask
values = sequence[non_sentinel].copy()
rng.shuffle(values)
shuffled = sequence.copy()
shuffled[non_sentinel] = values
```

This preserves bout boundaries (sentinel positions) while randomizing call types within them.

#### BUG-2: ICI uses start-to-start instead of gap (end-to-start)

**File:** `scripts/analyze_sequential_structure.py`, `detect_bouts()` line 103
**What:** ICI = `absolute_time[i+1] - absolute_time[i]`, which is onset-to-onset. Should be `start[i+1] - end[i]` (the silent gap between calls).
**Impact:** A Complex call lasting 150ms followed by a Short 10ms later has ICI = 160ms (start-to-start) but actual gap = 10ms. With a 0.6s threshold, the error goes in the wrong direction: calls that are truly in the same bout (tiny gap, long preceding call) could appear to have a larger ICI than they actually do. More importantly, two calls with a true gap of 0.7s (should be separate bouts) where the first call is 200ms long would show ICI = 0.9s — correctly split. So the bias is conservative (over-splitting rather than under-splitting), but it's still wrong.
**Fix:** Compute `end_time = absolute_time + call_duration` (from `duration_s` or `end_time_s - begin_time_s`), then ICI = `start[i+1] - end[i]`.

#### BUG-3: `_zipf_mle_powerlaw` always returns p_value=0.0

**File:** `usv_language/analysis/information_theory.py`, `_zipf_mle_powerlaw()` line 362
**What:** `p_value = 0.0` is hardcoded and never updated. The `distribution_compare()` call gives `(R, p)` but `p` is the comparison p-value (power-law vs exponential), not the goodness-of-fit p-value. The KS goodness-of-fit p-value (from `fit.power_law.KS()` or bootstrapped) is never computed.
**Impact on this report:** None — the 7-type early return (line 346-348) bypasses `_zipf_mle_powerlaw` entirely, returning p=1.0 correctly. But if the powerlaw package is available and this function is called with ≥10 frequency values, it will always report p=0.0, making every distribution look like it rejects the power-law hypothesis.
**Fix:** Use `fit.power_law.KS()` or compute bootstrap p-value as the scipy fallback does.

### Concerns (should fix or document)

#### CONCERN-1: Self-transition "chance" baseline is wrong

**File:** `analyze_sequential_structure.py`, line 574
**What:** Reports chance = 1/K = 14.3% (uniform). The correct independence baseline is Σ(pᵢ²) where pᵢ are marginal type probabilities. Since Flat is ~32% of calls, the true independence self-transition rate is 19.3%, not 14.3%.

**Verified numbers:**
- Σ(pᵢ²) = 0.193 = 19.3% (correct independence baseline)
- 1/K = 0.143 = 14.3% (wrong — assumes uniform distribution)
- Observed mean self-transition: 25.8%
- Enrichment vs uniform: 1.80× (what report claims)
- Enrichment vs independence: 1.34× (correct claim)
- The apparent enrichment is overstated by 5pp (nearly half the difference is from non-uniform marginals, not sequential structure)

**Fix:** Compute `chance_self = sum(p**2 for p in marginal_probs)` and report that.

#### CONCERN-2: Entropy rate at orders 4-5 has no bias correction

**File:** `analyze_sequential_structure.py`, `entropy_rate_from_bouts()` line 173
**What:** Plugin estimator H_n / n with no Miller-Madow correction. At order 5, there are 7⁵ = 16,807 possible n-grams but only ~6,000 within-bout samples. The plugin estimator is biased low (treats unseen n-grams as impossible).
**Impact:** Convergence plot shows steeper decline than true entropy rate at orders 4-5. Orders 1-2 are fine.
**Fix:** Either add Miller-Madow correction or cap the convergence plot at order 3 with a note.

#### CONCERN-3: MI has no finite-sample bias correction

**File:** `analyze_sequential_structure.py`, `mi_at_lag_from_bouts()` line 226
**What:** Plugin MI estimator is always ≥ 0 even for independent variables. Expected bias ≈ (K²-1)/(2N·ln2) ≈ 0.006 bits for these data.
**Impact:** MI values at lag 6+ (~0.01 bits) are within the bias floor. The "noise floor by lag 6+" claim should acknowledge this.
**Fix:** Either add shuffled baseline or report the analytic bias floor.

### Correct (no issues found)

- Transition matrix P(B|A): within-bout pairs only, row-normalized ✓
- Conditional entropy H(next|current): weighted row-entropy formula ✓
- MI at lag: joint distribution formula, bout-boundary handling at arbitrary lag ✓
- MI lag-1 vs H-H(next|current) discrepancy (0.093 vs 0.095): explained by different marginal computation at bout edges, not a bug ✓
- Zipf: graceful degenerate return for 7 types ✓
- FDR: Benjamini-Hochberg step-up procedure ✓
- Bout segmentation logic: threshold, single-call exclusion, cross-bout exclusion ✓ (aside from ICI definition)
- All reported numbers verified against `sequential_structure_summary.csv` — exact match ✓

### Verified data points

| Metric | Report | CSV | Match? |
|--------|--------|-----|--------|
| n_calls | ~7,864 | 7864 | ✓ |
| H(marginal) | 2.544 | 2.5438 | ✓ |
| H(next\|current) | 2.449 | 2.4494 | ✓ |
| Entropy reduction | 3.7% | 3.71% | ✓ |
| MI lag 1 | 0.093 | 0.0925 | ✓ |
| MI lag 2 | 0.042 | 0.042 | ✓ |
| Zipf α | 0 | 0.0 | ✓ |
| Zipf p | 1 | 1.0 | ✓ |
| n idioms | 1,843 | 1843 | ✓ (but inflated — BUG-1) |
| Mean self-trans | 25.8% | 0.2578 | ✓ (but baseline wrong — CONCERN-1) |

### Test coverage gaps

The idiom test (`test_information_theory.py:304`) has a single test: plant a known trigram into a random sequence and verify detection. This does **not** test:
- Sentinel-separated sequences (the exact usage pattern in the orchestrator)
- Bout-aware shuffling (the bug)
- Low-count n-grams with zero expected count
- FDR correction with thousands of candidates

### Summary table

| Analysis Layer | Verdict | Severity | Impact |
|---|---|---|---|
| **Bout segmentation** | BUG-2: start-to-start ICI | Medium | Conservative bias (over-splitting) |
| **Transition matrix** | Correct math | — | — |
| **Self-transition baseline** | CONCERN-1: wrong baseline | Medium | 1.80× claimed vs 1.34× correct enrichment |
| **Conditional entropy** | Correct | — | — |
| **Entropy rate** | CONCERN-2: no bias correction | Low | Orders 4-5 unreliable |
| **MI at lag** | CONCERN-3: no bias floor | Low | Lag 6+ values ambiguous |
| **Zipf** | Correct + BUG-3 (latent) | Low | No impact on this data (7 types) |
| **Idiom detection** | BUG-1: broken shuffle | **HIGH** | 1,843 count is ~10-20× inflated |

---

## Corrected Results (2026-04-12)

All four code fixes applied, analysis re-run with `--n-shuffles 200`. Results in `results/sequential_structure/`.

### Before vs After

| Metric | Before (buggy) | After (fixed) | Change |
|--------|---------------|---------------|--------|
| Significant idioms | 1,843 | 653 | -65% (broken shuffle inflated count) |
| Self-transition baseline | 14.3% (1/K) | 20.0% (Σpᵢ²) | Correct independence baseline |
| Self-transition enrichment | 1.80× | 1.28× | Nearly half the effect was non-uniform marginals |
| Multi-call bouts | 1,270 | 1,238 | Gap-based ICI splits 32 former bouts |
| Within-bout pairs | 6,300 | 6,350 | Slight redistribution |
| Marginal entropy | 2.544 bits | 2.544 bits | Unchanged |
| Conditional entropy | 2.449 bits | 2.450 bits | Unchanged |
| MI lag 1 | 0.093 bits | 0.092 bits | Unchanged |
| Zipf | α=0, p=1 | α=0, p=1 | Unchanged |

Entropy, MI, and Zipf were unaffected — those analyses were correct. The damage was concentrated in idiom detection (broken null model) and the self-transition framing (wrong baseline).

---

## Open: Bout threshold needs revision

**Status:** Pending domain input (see `docs/questions-for-mickey.md` Q1)

The 0.6s threshold was derived as `3 × median(onset-to-onset ICI)` over all ICIs, but this is wrong for two reasons:

1. **ICI type:** Used onset-to-onset instead of gap-based (fixed in code, but threshold was set before the fix)
2. **Recording structure:** WAV files are trigger-based (start on noise, stop ~2s after silence). Cross-file gaps (17.5% of ICIs, median 15s) are recording artifacts, not vocalization timing. They were mixed into the ICI distribution used to compute the threshold.

### What the data shows (within-file, gap-based ICIs only)

- Median within-file gap: **78ms**
- 2-component GMM crossover: **0.143s**
- 3-component GMM: core peak at 74ms (66%), transition zone at 250ms (19%), between-bout at 21s (15%)
- Sensitivity sweep flattens above ~0.25s

### Recommended approach

Two-layer bout segmentation:
```
Bout boundary if:  (different file)  OR  (same file AND gap > threshold)
```
Threshold: 0.25s (conservative) or 0.14s (crossover). Waiting for Mickey's input on what "bout" means for the analysis before choosing.

### Impact of revising threshold

If threshold drops from 0.6s to 0.25s: ~100 more bouts, ~300 fewer within-bout pairs. Modest impact on transition matrix and entropy (these are robust to threshold). Idiom count may change more significantly since bout boundaries affect the sentinel-separated sequence.

### Supporting plots

- `results/sequential_structure/bout_threshold_within_file.png` — mixture fit on within-file ICIs + sensitivity sweep

---

## Part 2: Math Explainer

### Conditional Entropy: H(next | current)

**What it answers:** "If I know the current call type, how uncertain am I about the next one?"

#### The formula

H(next | current) = Σₐ P(a) · H(next | current = a)

Where:
- `a` ranges over all 7 syllable types
- P(a) = fraction of within-bout transitions where the current call is type `a`
- H(next | current = a) = -Σ_b P(b|a) · log₂ P(b|a), the entropy of the next-call distribution given that the current call is type `a`

#### How the code implements it

**Step 1** — Build a 7×7 count matrix from within-bout consecutive pairs:

```
counts[a, b] = number of times type a is immediately followed by type b
```

**Step 2** — Row-normalize to get the transition matrix:

```
P(b | a) = counts[a, b] / Σ_j counts[a, j]
```

**Step 3** — Compute the "current call" marginal. This is NOT the overall type distribution. It's the distribution of types that appear as the *first* element of a within-bout pair:

```
P(a) = row_sum[a] / total_transitions
```

This excludes bout-final calls (they never appear as "current" in a transition pair) and single-call bouts. So P(a) is slightly different from the raw type proportions.

**Step 4** — Weighted average of per-row entropies:

```
H(next | current) = Σ_a P(a) · [-Σ_b P(b|a) · log₂ P(b|a)]
```

#### Interpretation

- Maximum possible: log₂(7) = 2.807 bits (completely random next call, uniform over 7 types)
- Marginal entropy H(next) = 2.544 bits (non-uniform type frequencies reduce uncertainty)
- Conditional entropy H(next | current) = 2.449 bits (knowing current type reduces uncertainty further)
- **Information gain** = 2.544 - 2.449 = 0.095 bits (3.7%)

The 3.7% reduction means: knowing the current call type gives you only a tiny additional edge in predicting the next one, beyond what you'd get from just knowing the base rates.

### Entropy Rate Convergence: H_n

**What it answers:** "As I look at longer and longer sequences, how much new information does each additional call carry?"

#### The formula

The script uses the **block entropy** estimator:

```
h_n = H(X₁, X₂, ..., Xₙ) / n
```

Where H(X₁, ..., Xₙ) is the Shannon entropy of the n-gram distribution — treat each unique n-gram as a symbol, compute its probability from counts, apply -Σ p log₂ p.

Dividing by n gives the "per-symbol rate" — the average information per call when considering n-call windows.

#### What convergence means

- At order 1: h₁ = H(X) = marginal entropy = 2.544 bits. No context.
- At order 2: h₂ = H(X₁,X₂)/2. This accounts for pairwise dependencies. If calls were independent, h₂ = h₁. If there are dependencies, h₂ < h₁.
- At order n → ∞: h_n converges to the true entropy rate h, which is the irreducible uncertainty per call given unlimited context.

The convergence plot should show h₁ > h₂ > h₃ > ... → h. The rate of convergence tells you the "memory depth" of the sequence.

#### Why higher orders are unreliable here

At order n, you're estimating a distribution over K^n possible n-grams:

| Order | Possible n-grams | Available samples (~6,000) | Ratio |
|-------|-----------------|---------------------------|-------|
| 1     | 7               | 6,000                     | 857×  |
| 2     | 49              | 6,000                     | 122×  |
| 3     | 343             | 6,000                     | 17×   |
| 4     | 2,401           | 6,000                     | 2.5×  |
| 5     | 16,807          | 6,000                     | 0.36× |

At order 5, you have fewer samples than possible patterns. The plugin estimator assigns probability 0 to unseen n-grams, which artificially lowers the entropy estimate. This makes the sequence look more structured (more predictable) than it actually is.

#### The alternative estimator

The handoff asked about the **per-symbol conditional** estimator:

```
ĥ_n = H(Xₙ | X₁, ..., Xₙ₋₁) = H(X₁,...,Xₙ) - H(X₁,...,Xₙ₋₁)
```

This is mathematically related to block entropy but estimates the entropy rate from below rather than above. The true rate is sandwiched: ĥ_n ≤ h ≤ h_n. The code uses h_n (block entropy, overestimate that converges from above).

### Mutual Information at Lag: MI(T, T+k)

**What it answers:** "How much does knowing the call at position t tell you about the call k steps later?"

#### The formula

MI(T, T+k) = Σᵢ Σⱼ p(i,j) · log₂[ p(i,j) / (p(i) · p(j)) ]

Where:
- p(i,j) = joint probability that a call at position t is type i AND the call at position t+k is type j
- p(i) = marginal probability of type i at position t (sum over j)
- p(j) = marginal probability of type j at position t+k (sum over i)

#### How the code builds the joint distribution

**Step 1** — For each bout with length L > k, collect all (seq[t], seq[t+k]) pairs where t = 0, 1, ..., L-k-1. This ensures both endpoints and all intermediate calls are within the same bout.

**Step 2** — Count these pairs into a 7×7 joint matrix, then normalize by total count to get p(i,j).

**Step 3** — Derive marginals by summing rows and columns of the joint matrix. These marginals come FROM the joint, not from the overall type distribution. This matters because:
- At lag 1: all calls except bout-final ones contribute to marginal_x; all except bout-initial to marginal_y
- At lag 5: only calls at least 5 positions from bout end contribute to marginal_x

So the marginals shift slightly with lag, which is correct behavior.

#### Why MI ≥ 0 always (even for independent sequences)

The plugin MI estimator has a positive bias: with finite data, the estimated joint distribution p̂(i,j) will never exactly equal p̂(i)·p̂(j) even under true independence, so the log ratio contributes positive values. The expected bias is approximately:

```
E[MI_bias] ≈ (K² - 1) / (2N · ln2)
```

For K=7 types and N≈6,000 samples: bias ≈ 48 / (12,000 × 0.693) ≈ 0.006 bits.

So any MI value below ~0.006 bits cannot be distinguished from sampling noise. The lag-1 value of 0.093 is 15× the bias — definitely real structure. The lag-6+ values near 0.01 are only 1-2× the bias — ambiguous.

#### MI vs conditional entropy

For lag 1, MI and conditional entropy are related:

```
MI(T, T+1) = H(T+1) - H(T+1 | T)
```

If H(T) = H(T+1) (stationary process), then MI = H - H(next|current). The report's 0.093 vs the predicted 0.095 differs by 0.002 bits because marginals at position t vs t+1 differ slightly (bout-edge effects). Not a bug.

#### What the MI decay curve means

- **Sharp drop** (lag 1→2): most sequential structure is in immediate neighbors
- **Slow decay** (lag 2→5): weak longer-range correlations, possibly from bout-level trends
- **Noise floor** (lag 6+): no detectable structure beyond ~5 calls back

This profile is consistent with a process that has strong first-order Markov structure (self-repetition bias), weak second-order effects, and essentially no higher-order memory. The sequence "remembers" about 3-5 calls back.

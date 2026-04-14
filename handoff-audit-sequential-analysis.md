# Handoff: Audit Sequential Structure Analysis Implementation

## Goal

Audit the code that produced the sequential structure analysis results in the USV progress report (Section 4). We need to verify that each analysis layer is implemented correctly — not just that it runs, but that the math and logic match what the analysis claims to do.

## Context

The progress report describes six analysis layers applied to ~7,864 USV calls from cage 5970 (usv_lmt_034), each classified into one of 7 syllable types (Flat, Down, Chevron, Short, Complex, Frequency_Jump, Up). The analyses were run on within-bout sequences only (ICI < 0.6s threshold).

The reported results are:
- Transition matrix: self-transitions enriched (25.8% observed vs 14.3% chance)
- Conditional entropy H(next|current) = 2.449 bits, marginal H = 2.544 bits → 0.095 bit reduction (3.7%)
- MI at lag 1 = 0.093 bits, lag 2 = 0.042 bits, noise floor by lag 6+
- Zipf: α=0, p=1 (not a power law)
- Idiom detection: 1,843 significant n-grams, almost all same-type repetitions

## Instructions

### Step 1: Find the code

Search the mickey-lab repo for the files that implement the sequential structure analysis. Look for:
- Scripts that compute transition matrices
- Entropy / mutual information calculations
- Zipf fitting
- Idiom / n-gram shuffle tests
- Bout segmentation logic (the 0.6s ICI threshold)

Show me the file paths and a brief description of what each file does. Don't proceed to Step 2 until I confirm which files to audit.

### Step 2: Audit bout segmentation

This is the foundation — if bout boundaries are wrong, everything downstream is wrong.

Check:
- How are bouts defined? Verify ICI < 0.6s is the threshold
- Are cross-bout transitions correctly excluded from all downstream analyses?
- How are single-call bouts handled? They should contribute zero transitions
- Is ICI computed correctly (start of next call minus end of previous call, or start-to-start)? The choice matters and should be documented

### Step 3: Audit the transition matrix

Check:
- Is P(B|A) computed as count(A→B) / count(A→anything), using only within-bout consecutive pairs?
- Is the "chance" baseline correct? It should be the marginal distribution of the *following* call (which might differ slightly from the overall distribution if bout-initial calls are excluded)
- The reported 25.8% average self-transition vs 14.3% by chance — verify both numbers. The 14.3% chance figure should be sum(p_i^2) where p_i are marginal probabilities (expected self-transition rate under independence)
- Are the matrix rows properly normalized (sum to 1.0)?

### Step 4: Audit entropy calculations

Check:
- Marginal entropy H = -Σ p_i log₂(p_i) — verify it equals 2.544 bits
- Conditional entropy H(next|current) = Σ_a P(a) * H(next|current=a) — verify it equals 2.449 bits
- The probabilities used: are they from the transition matrix (within-bout only)?
- Entropy rate convergence (the plot showing values at n-gram orders 1–5): verify the n-gram entropy estimator is H_n = H(X_1,...,X_n) / n or the per-symbol conditional H(X_n | X_1,...,X_{n-1}). These are different estimators and the report should be clear about which one is used
- **Important**: at higher n-gram orders (4, 5), check for undersampling bias. With 7 types, 5-grams have 7^5 = 16,807 possible patterns but only ~6,000 within-bout transitions. Naive n-gram entropy estimates are biased downward with insufficient data. Check if any correction (Miller-Madow, jackknife, NSB) is applied

### Step 5: Audit mutual information at lag

Check:
- MI(T, T+k) = H(T) + H(T+k) - H(T, T+k), computed from the joint distribution of call types at positions separated by lag k
- Are only within-bout pairs used? At lag k, a pair should only count if ALL k intermediate calls are also within the same bout (no bout boundary crossed anywhere in the span)
- At lag 1, MI should approximately equal H - H(next|current) = 2.544 - 2.449 = 0.095 bits. The report says 0.093 — check if the small discrepancy is a rounding issue or a different computation
- Is there a significance test or confidence interval? MI is always ≥ 0 even for independent variables due to finite-sample bias. Check if shuffled baselines are used

### Step 6: Audit Zipf fitting

Check:
- What fitting method is used? (MLE via powerlaw package, or least-squares on log-log?)
- The powerlaw package by Clauset et al. is the standard — verify it's used correctly
- α=0 and p=1 is a valid result for 7 types (too few for a meaningful fit), just confirm the code runs correctly

### Step 7: Audit idiom detection (shuffle test)

This is the most complex analysis and most likely to have subtle bugs.

Check:
- How are shuffles generated? They should permute the sequence while preserving marginal type frequencies. Are they shuffling within-bout sequences only, or the full sequence?
- **Critical**: if they shuffle the full sequence and then re-segment into bouts, that changes bout structure. The correct approach is to shuffle call types within each bout separately, OR shuffle the full within-bout concatenated sequence
- z-score computation: z = (observed_count - mean_shuffled) / std_shuffled. Check that std is not zero (which would cause division errors for rare n-grams)
- Multiple comparisons: with 7 types and n-grams up to length 5, there are ~19,000 possible n-grams. Are the 1,843 "significant" idioms corrected for multiple testing (Bonferroni, FDR)?
- The report says "almost all are same-type repetitions" — verify this by checking what fraction of the 1,843 idioms are homogeneous (all same type) vs heterogeneous

### Step 8: Report findings

For each analysis layer, report:
1. **Correct / Bug / Concern** — is the implementation faithful to the described method?
2. If there's a bug or concern, what is the expected impact on the reported results?
3. Any methodological improvements worth flagging (e.g., finite-sample MI bias correction)

Format the output as a structured audit report I can include in the project docs.

## Notes

- Do NOT modify any code. This is a read-only audit.
- If you find the code spread across notebooks (.ipynb), extract the relevant cells — don't try to run them.
- If any analysis was done interactively (in a notebook without clean functions), flag that as a reproducibility concern.

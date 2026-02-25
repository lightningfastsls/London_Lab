# Null Model Surrogate Generators

**Phase:** Extension of Phase 8.4 (user-supplied spec)
**ADRs:** None (no DSP/STFT changes)
**Tests:** `usv_language/tests/test_null_models.py` -- 16 tests across 10 test classes

## Purpose

Provides a hierarchy of statistical null models for testing whether VQ-VAE code sequences contain genuine language-like structure. Each generator preserves specific statistical properties while destroying others, enabling structured hypothesis testing:

- If real > shuffled: sequential structure exists
- If real > Markov-k: structure exceeds k-th order memory
- If real > renewal: structure is more than temporal spacing
- If real > HMM: structure exceeds hidden-state dynamics
- If real > phase-randomized: structure exceeds linear correlations

All generators accept 1D int64 ndarrays and return lists of ndarrays (same length as input).

## Public Interface

### Dataclasses (1 frozen)

| Dataclass | Key Fields | Used By |
|-----------|-----------|---------|
| `NullModelConfig` | n_surrogates, random_seed, markov_orders, hmm_n_states, hmm_n_iter, phase_rand_method | `NullModelGenerator` |

### NullModelGenerator Methods

| Method | Signature | What It Preserves |
|--------|-----------|-------------------|
| `shuffled` | `(ndarray) -> list[ndarray]` | Exact marginal distribution |
| `markov_order_k` | `(ndarray, k) -> list[ndarray]` | k-gram transition statistics |
| `renewal_process` | `(ndarray) -> list[ndarray]` | Per-code temporal spacing (IEIs) |
| `hmm_surrogate` | `(ndarray, n_states?) -> list[ndarray]` | Hidden-state dynamics |
| `phase_randomized` | `(ndarray) -> list[ndarray]` | Linear autocorrelation |
| `generate_all` | `(ndarray) -> dict[str, list[ndarray]]` | All of the above |

## Key Decisions

1. **Input = `np.ndarray`** (not `list[int]`): matches the entire existing codebase convention where VQ-VAE code sequences are always int64 ndarrays.

2. **`hmmlearn` optional**: Same import guard pattern as `powerlaw` in `information_theory.py`. `generate_all()` silently skips HMM if unavailable.

3. **Return = `list[np.ndarray]`**: Each surrogate is a full-length ndarray. Lists rather than stacked 2D arrays allow variable handling downstream.

4. **Renewal = per-code IEI**: More informative than pooled IEI (which collapses to shuffled for dense sequences). First-placed code wins collisions; unassigned positions filled from unigram.

5. **Phase randomization default = FFT**: Standard method. AAFT documented as better for discrete data (preserves exact marginal).

6. **Markov backoff chain**: k-gram -> (k-1)-gram -> ... -> unigram fallback for unseen contexts. Same principle as Katz backoff in NLP language models.

7. **No imports from information_theory.py**: `null_models.py` is standalone. Tests import from both for verification but the generator has no cross-module dependencies.

## Algorithm Details

### Shuffled
`rng.permutation(sequence)` -- trivial. Exact frequency preservation guaranteed.

### Markov order-k
Builds k-gram -> next-symbol transition tables from data. Backoff chain for unseen contexts: try k-gram, then (k-1)-gram, ..., then unigram. Seeds generation with a random observed k-gram.

### Renewal Process
1. Compute per-code inter-occurrence intervals (IEIs) from original positions
2. For each code (most frequent first): random start + shuffled IEIs -> positions (wrapping)
3. Collision handling: first-placed code wins
4. Unassigned positions filled from unigram distribution

### HMM Surrogate
Fits `CategoricalHMM` via EM (Baum-Welch), then samples n_surrogates sequences from the fitted model. Requires `hmmlearn`.

### Phase-Randomized
- **FFT**: randomize phases preserving magnitudes + conjugate symmetry, IFFT, round to nearest valid code
- **AAFT**: rank-match to Gaussian -> phase randomize -> re-rank to restore exact marginal distribution

## Integration Points

- **Imports from:** numpy, scipy.fft. Optional: hmmlearn
- **Called by:** Future `run_analysis.py` integration (not yet wired)
- **Feeds into:** Hypothesis testing comparisons with information_theory metrics
- **Dependencies:** numpy, scipy (already in requirements). `hmmlearn` optional.

## Edge Cases

| Method | Edge Case | Handling |
|--------|-----------|---------|
| All | Empty sequence | Return list of empty arrays |
| `markov_order_k` | k > len(seq) - 1 | Clamp k to len(seq) - 1 |
| `renewal_process` | Collision-heavy sequences | Overflow placed at random free positions |
| `hmm_surrogate` | hmmlearn not installed | Raises ImportError |
| `hmm_surrogate` | Degenerate sequence | `generate_all` catches exception, skips |
| `phase_randomized` | Single unique code | Returns constant arrays |

"""Null model surrogate generators for VQ-VAE code sequences.

Provides a hierarchy of statistical null models for hypothesis testing:
each generator preserves specific statistical properties while destroying
others, enabling structured inference about sequence complexity.

Hierarchy (weakest to strongest):
    1. Shuffled -- preserves marginal distribution, destroys all order
    2. Markov order-k -- preserves k-gram transition statistics
    3. Renewal process -- preserves per-code temporal spacing (IEIs)
    4. HMM surrogate -- preserves hidden-state dynamics (optional hmmlearn)
    5. Phase-randomized -- preserves linear autocorrelation (FFT/AAFT)

Interpretation guide:
    - real > shuffled  => sequential structure exists
    - real > markov-k  => structure exceeds k-th order memory
    - real > renewal   => structure is more than temporal spacing
    - real > hmm       => structure exceeds hidden-state dynamics
    - real > phase-rand => structure exceeds linear correlations

All generators accept 1D int64 ndarrays and return lists of ndarrays.

Dependencies: numpy, scipy (already in requirements).  ``hmmlearn`` optional.
"""

from __future__ import annotations

import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass

import numpy as np
from scipy import fft as sp_fft

# Optional hmmlearn (same guard pattern as powerlaw in information_theory.py)
try:
    from hmmlearn.hmm import CategoricalHMM as _CategoricalHMM

    _HAS_HMMLEARN = True
except ImportError:
    _HAS_HMMLEARN = False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NullModelConfig:
    """Configuration for null model surrogate generation.

    Parameters
    ----------
    n_surrogates : int
        Number of surrogates to generate per model type.
    random_seed : int
        Seed for reproducible generation.
    markov_orders : tuple[int, ...]
        Which Markov orders to generate (e.g. (1, 2, 3)).
    hmm_n_states : int
        Number of hidden states for HMM fitting.
    hmm_n_iter : int
        Maximum EM iterations for HMM fitting.
    phase_rand_method : str
        Phase randomization method: "fft" or "aaft".
    """

    n_surrogates: int = 100
    random_seed: int = 42
    markov_orders: tuple[int, ...] = (1, 2, 3)
    hmm_n_states: int = 8
    hmm_n_iter: int = 100
    phase_rand_method: str = "fft"

    def __post_init__(self) -> None:
        if self.n_surrogates <= 0:
            raise ValueError(
                f"n_surrogates must be positive, got {self.n_surrogates}"
            )
        if not (0 <= self.random_seed <= 2**32 - 1):
            raise ValueError(
                f"random_seed must be in [0, 2**32-1], got {self.random_seed}"
            )
        if not self.markov_orders:
            raise ValueError("markov_orders must be non-empty")
        for k in self.markov_orders:
            if k <= 0:
                raise ValueError(
                    f"markov_orders entries must be positive, got {k}"
                )
        if self.hmm_n_states <= 0:
            raise ValueError(
                f"hmm_n_states must be positive, got {self.hmm_n_states}"
            )
        if self.hmm_n_iter <= 0:
            raise ValueError(
                f"hmm_n_iter must be positive, got {self.hmm_n_iter}"
            )
        if self.phase_rand_method not in ("fft", "aaft"):
            raise ValueError(
                f"phase_rand_method must be 'fft' or 'aaft', "
                f"got '{self.phase_rand_method}'"
            )


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------


class NullModelGenerator:
    """Generates surrogate sequences under various null hypotheses.

    Parameters
    ----------
    config : NullModelConfig
        Generation parameters.

    Attributes
    ----------
    config : NullModelConfig
        Stored configuration.
    rng : np.random.RandomState
        Seeded random state for reproducibility.
    """

    def __init__(self, config: NullModelConfig | None = None) -> None:
        self.config = config or NullModelConfig()
        self.rng = np.random.RandomState(self.config.random_seed)

    # -------------------------------------------------------------------
    # 1. Shuffled
    # -------------------------------------------------------------------

    def shuffled(self, sequence: np.ndarray) -> list[np.ndarray]:
        """Generate surrogates by random permutation.

        Preserves exact marginal distribution (code frequencies).
        Destroys all sequential structure.

        Parameters
        ----------
        sequence : np.ndarray
            1D int64 array of code indices.

        Returns
        -------
        list[np.ndarray]
            n_surrogates shuffled copies.
        """
        result = []
        for _ in range(self.config.n_surrogates):
            surr = self.rng.permutation(sequence).astype(sequence.dtype)
            result.append(surr)
        return result

    # -------------------------------------------------------------------
    # 2. Markov order-k
    # -------------------------------------------------------------------

    def markov_order_k(
        self, sequence: np.ndarray, k: int
    ) -> list[np.ndarray]:
        """Generate surrogates from a k-th order Markov model.

        Builds k-gram -> next-symbol transition tables from the data,
        with backoff to (k-1)-gram -> ... -> unigram for unseen contexts.

        Parameters
        ----------
        sequence : np.ndarray
            1D int64 array of code indices.
        k : int
            Markov order (context length).

        Returns
        -------
        list[np.ndarray]
            n_surrogates Markov-k sequences of same length as input.
        """
        n = len(sequence)
        if n == 0:
            return [np.array([], dtype=sequence.dtype)
                    for _ in range(self.config.n_surrogates)]

        # Effective order clamped to sequence length - 1
        k_eff = min(k, n - 1)

        # Build transition tables for orders 1..k_eff + unigram (order 0)
        transitions: dict[int, dict[tuple, Counter]] = {}
        for order in range(k_eff + 1):
            transitions[order] = defaultdict(Counter)

        # Unigram (order 0)
        unigram = Counter(sequence.tolist())
        transitions[0][()] = unigram

        # Higher orders
        for order in range(1, k_eff + 1):
            for i in range(order, n):
                context = tuple(sequence[i - order: i].tolist())
                symbol = int(sequence[i])
                transitions[order][context][symbol] += 1

        # Precompute observed k-grams for starting seeds
        # Exclude the terminal k-gram (has no outgoing transitions)
        if k_eff > 0:
            start_kgrams = [
                tuple(sequence[i: i + k_eff].tolist())
                for i in range(max(1, n - k_eff))
            ]
        else:
            start_kgrams = [()]

        result = []
        for _ in range(self.config.n_surrogates):
            surr = self._generate_markov_sequence(
                n, k_eff, transitions, start_kgrams, sequence.dtype
            )
            result.append(surr)
        return result

    def _generate_markov_sequence(
        self,
        length: int,
        k: int,
        transitions: dict[int, dict[tuple, Counter]],
        start_kgrams: list[tuple],
        dtype: np.dtype,
    ) -> np.ndarray:
        """Generate a single Markov-k sequence with backoff."""
        surr = np.empty(length, dtype=dtype)

        # Seed with a random observed k-gram
        if k > 0:
            seed_idx = self.rng.randint(len(start_kgrams))
            seed = start_kgrams[seed_idx]
            for i, val in enumerate(seed):
                surr[i] = val
            start = k
        else:
            start = 0

        for i in range(start, length):
            # Try orders k, k-1, ..., 0 (backoff)
            sampled = False
            for order in range(k, -1, -1):
                if order == 0:
                    context = ()
                else:
                    context = tuple(surr[i - order: i].tolist())

                if context in transitions[order]:
                    counter = transitions[order][context]
                    symbols = list(counter.keys())
                    counts = np.array(list(counter.values()), dtype=np.float64)
                    probs = counts / counts.sum()
                    surr[i] = self.rng.choice(symbols, p=probs)
                    sampled = True
                    break

            if not sampled:
                # Absolute fallback: uniform over unigram vocabulary
                symbols = list(transitions[0][()].keys())
                surr[i] = self.rng.choice(symbols)

        return surr

    # -------------------------------------------------------------------
    # 3. Renewal process
    # -------------------------------------------------------------------

    def renewal_process(self, sequence: np.ndarray) -> list[np.ndarray]:
        """Generate surrogates preserving per-code temporal spacing.

        For each code, computes inter-occurrence intervals (IEIs) from
        the original, then places that code at random start + shuffled
        IEIs.  Collision handling: first-placed code wins; unassigned
        positions filled from the unigram distribution.

        Parameters
        ----------
        sequence : np.ndarray
            1D int64 array of code indices.

        Returns
        -------
        list[np.ndarray]
            n_surrogates renewal-process sequences.
        """
        n = len(sequence)
        if n == 0:
            return [np.array([], dtype=sequence.dtype)
                    for _ in range(self.config.n_surrogates)]

        # Compute per-code positions and IEIs
        code_positions: dict[int, list[int]] = defaultdict(list)
        for i, code in enumerate(sequence.tolist()):
            code_positions[code].append(i)

        code_ieis: dict[int, np.ndarray] = {}
        for code, positions in code_positions.items():
            if len(positions) > 1:
                code_ieis[code] = np.diff(positions)
            else:
                code_ieis[code] = np.array([], dtype=np.int64)

        # Unigram distribution for filling unassigned positions
        unigram = Counter(sequence.tolist())
        codes_uni = list(unigram.keys())
        probs_uni = np.array([unigram[c] for c in codes_uni], dtype=np.float64)
        probs_uni /= probs_uni.sum()

        # Sort codes by count descending (most frequent placed first)
        sorted_codes = sorted(
            code_positions.keys(),
            key=lambda c: len(code_positions[c]),
            reverse=True,
        )

        result = []
        for _ in range(self.config.n_surrogates):
            surr = np.full(n, -1, dtype=sequence.dtype)
            occupied = np.zeros(n, dtype=bool)

            for code in sorted_codes:
                count = len(code_positions[code])
                if count == 0:
                    continue

                ieis = code_ieis[code]
                if len(ieis) > 0:
                    shuffled_ieis = self.rng.permutation(ieis)
                else:
                    shuffled_ieis = np.array([], dtype=np.int64)

                # Random start position
                start = self.rng.randint(0, n)

                # Place code at start, then at cumulative IEI offsets
                pos = start % n
                placed = 0
                if not occupied[pos]:
                    surr[pos] = code
                    occupied[pos] = True
                    placed += 1

                for iei in shuffled_ieis:
                    if placed >= count:
                        break
                    pos = (pos + int(iei)) % n
                    if not occupied[pos]:
                        surr[pos] = code
                        occupied[pos] = True
                        placed += 1

                # If some copies couldn't be placed due to collisions,
                # try random unoccupied positions
                if placed < count:
                    free = np.where(~occupied)[0]
                    need = min(count - placed, len(free))
                    if need > 0:
                        chosen = self.rng.choice(free, size=need, replace=False)
                        for idx in chosen:
                            surr[idx] = code
                            occupied[idx] = True

            # Fill remaining unassigned positions from unigram
            unassigned = np.where(~occupied)[0]
            if len(unassigned) > 0:
                fills = self.rng.choice(codes_uni, size=len(unassigned),
                                        p=probs_uni)
                surr[unassigned] = fills

            result.append(surr)
        return result

    # -------------------------------------------------------------------
    # 4. HMM surrogate
    # -------------------------------------------------------------------

    def hmm_surrogate(
        self,
        sequence: np.ndarray,
        n_states: int | None = None,
    ) -> list[np.ndarray]:
        """Generate surrogates from a fitted Hidden Markov Model.

        Fits a CategoricalHMM via Baum-Welch EM, then samples new
        sequences from the fitted model.

        Parameters
        ----------
        sequence : np.ndarray
            1D int64 array of code indices.
        n_states : int, optional
            Number of hidden states.  Defaults to config.hmm_n_states.

        Returns
        -------
        list[np.ndarray]
            n_surrogates HMM-sampled sequences.

        Raises
        ------
        ImportError
            If hmmlearn is not installed.
        """
        if not _HAS_HMMLEARN:
            raise ImportError(
                "hmmlearn is required for HMM surrogates. "
                "Install with: pip install hmmlearn"
            )

        n = len(sequence)
        if n == 0:
            return [np.array([], dtype=sequence.dtype)
                    for _ in range(self.config.n_surrogates)]

        if n_states is None:
            n_states = self.config.hmm_n_states

        n_codes = int(np.max(sequence)) + 1

        # Fit CategoricalHMM
        model = _CategoricalHMM(
            n_components=n_states,
            n_iter=self.config.hmm_n_iter,
            random_state=self.config.random_seed,
            n_features=n_codes,
        )
        X = sequence.reshape(-1, 1)
        model.fit(X)

        result = []
        for _ in range(self.config.n_surrogates):
            samples, _ = model.sample(n)
            surr = samples.flatten().astype(sequence.dtype)
            result.append(surr)
        return result

    # -------------------------------------------------------------------
    # 5. Phase-randomized
    # -------------------------------------------------------------------

    def phase_randomized(self, sequence: np.ndarray) -> list[np.ndarray]:
        """Generate surrogates preserving linear autocorrelation.

        Two methods:
        - FFT: randomize Fourier phases (preserving conjugate symmetry),
          IFFT, round to nearest valid code.
        - AAFT (Amplitude Adjusted Fourier Transform): rank-match to
          Gaussian, phase-randomize, re-rank to restore exact marginal.
          Better for discrete data.

        Parameters
        ----------
        sequence : np.ndarray
            1D int64 array of code indices.

        Returns
        -------
        list[np.ndarray]
            n_surrogates phase-randomized sequences.
        """
        n = len(sequence)
        if n == 0:
            return [np.array([], dtype=sequence.dtype)
                    for _ in range(self.config.n_surrogates)]

        if self.config.phase_rand_method == "aaft":
            return self._phase_randomized_aaft(sequence)
        return self._phase_randomized_fft(sequence)

    def _phase_randomized_fft(self, sequence: np.ndarray) -> list[np.ndarray]:
        """FFT phase randomization: preserves power spectrum."""
        n = len(sequence)
        valid_codes = np.unique(sequence)
        x = sequence.astype(np.float64)
        X = sp_fft.rfft(x)

        result = []
        for _ in range(self.config.n_surrogates):
            # Random phases preserving conjugate symmetry
            # rfft output has length n//2 + 1
            # DC (index 0) and Nyquist (last, if n even) must stay real
            phases = self.rng.uniform(0, 2 * np.pi, size=len(X))
            phases[0] = 0.0  # DC component
            if n % 2 == 0:
                phases[-1] = 0.0  # Nyquist component

            X_rand = np.abs(X) * np.exp(1j * phases)
            x_rand = sp_fft.irfft(X_rand, n=n)

            # Map back to nearest valid code
            surr = self._nearest_codes(x_rand, valid_codes, sequence.dtype)
            result.append(surr)
        return result

    def _phase_randomized_aaft(
        self, sequence: np.ndarray
    ) -> list[np.ndarray]:
        """AAFT: preserves exact marginal distribution + autocorrelation."""
        n = len(sequence)
        x_sorted = np.sort(sequence.astype(np.float64))

        result = []
        for _ in range(self.config.n_surrogates):
            # Step 1: Generate Gaussian noise, rank-match to original
            gaussian = self.rng.randn(n)
            original_rank_idx = np.argsort(np.argsort(sequence))

            # Rank-matched Gaussian: same rank structure as original
            # sorted_gaussian[rank[i]] gives position i the Gaussian value
            # whose rank matches the original element's rank
            y = np.sort(gaussian)[original_rank_idx]

            # Step 2: Phase-randomize the rank-matched Gaussian
            Y = sp_fft.rfft(y)
            phases = self.rng.uniform(0, 2 * np.pi, size=len(Y))
            phases[0] = 0.0
            if n % 2 == 0:
                phases[-1] = 0.0
            Y_rand = np.abs(Y) * np.exp(1j * phases)
            y_rand = sp_fft.irfft(Y_rand, n=n)

            # Step 3: Re-rank to restore exact marginal distribution
            surr_rank = np.argsort(np.argsort(y_rand))
            surr = x_sorted[surr_rank].astype(sequence.dtype)
            result.append(surr)
        return result

    @staticmethod
    def _nearest_codes(
        values: np.ndarray,
        valid_codes: np.ndarray,
        dtype: np.dtype,
    ) -> np.ndarray:
        """Map continuous values to nearest valid discrete codes."""
        # For each value, find the closest code
        # Use broadcasting: |values[:, None] - codes[None, :]|
        # For large arrays, use searchsorted on sorted codes
        sorted_codes = np.sort(valid_codes)
        idx = np.searchsorted(sorted_codes, values, side="left")
        idx = np.clip(idx, 0, len(sorted_codes) - 1)

        # Check if left or right neighbor is closer
        left = np.clip(idx - 1, 0, len(sorted_codes) - 1)
        right = idx

        dist_left = np.abs(values - sorted_codes[left])
        dist_right = np.abs(values - sorted_codes[right])

        best = np.where(dist_left <= dist_right, left, right)
        return sorted_codes[best].astype(dtype)

    # -------------------------------------------------------------------
    # 6. generate_all
    # -------------------------------------------------------------------

    def generate_all(
        self, sequence: np.ndarray
    ) -> dict[str, list[np.ndarray]]:
        """Generate surrogates from all available null models.

        Returns
        -------
        dict[str, list[np.ndarray]]
            Keys: "shuffled", "markov_1", "markov_2", ..., "renewal",
            "hmm" (if hmmlearn available), "phase_randomized".
            Values: lists of n_surrogates ndarrays.
        """
        results: dict[str, list[np.ndarray]] = {}

        results["shuffled"] = self.shuffled(sequence)

        for k in self.config.markov_orders:
            results[f"markov_{k}"] = self.markov_order_k(sequence, k)

        results["renewal"] = self.renewal_process(sequence)

        if _HAS_HMMLEARN:
            try:
                results["hmm"] = self.hmm_surrogate(sequence)
            except Exception as e:
                warnings.warn(
                    f"HMM surrogate generation failed: {e}. "
                    f"Skipping 'hmm' key.",
                    RuntimeWarning,
                    stacklevel=2,
                )

        results["phase_randomized"] = self.phase_randomized(sequence)

        return results

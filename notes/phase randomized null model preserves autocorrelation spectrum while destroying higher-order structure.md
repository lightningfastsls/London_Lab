---
description: "FFT phase randomization preserves power spectrum and autocorrelation of code sequences but destroys phase relationships and higher-order temporal structure"
type: method
confidence: proven
meta_state: current
topics:
  - "[[representation-learning]]"
  - "[[signal-processing]]"
---

# phase randomized null model preserves autocorrelation spectrum while destroying higher-order structure

The phase randomization null model operates in the frequency domain: compute the FFT of the code sequence, replace each phase with a uniform random value while preserving the magnitude spectrum, then apply the inverse FFT. Because the power spectrum is the Fourier transform of the autocorrelation function (by the Wiener-Khinchin theorem), this procedure exactly preserves the second-order temporal statistics (autocorrelation at all lags) while destroying all higher-order structure and phase relationships. The resulting surrogate has the same "spectral fingerprint" as the original but is otherwise random.

This null model tests a specific hypothesis: whether linear temporal correlations alone explain the observed patterns. If a metric on real data does not significantly exceed the phase-randomized baseline, the structure is fully captured by the autocorrelation function — there is no need to invoke nonlinear dynamics, compositionality, or hierarchical organization. But if real data significantly exceeds this null, the structure depends on phase relationships or higher-order statistics that linear models cannot capture, which means something genuinely nonlinear is happening in USV sequence organization.

There is an important caveat for discrete code sequences. Phase randomization is naturally suited for continuous-valued signals, because the inverse FFT of randomized phases produces continuous values. For discrete VQ-VAE codes (integers from 0 to K-1), rounding the continuous output back to integers introduces approximation error that distorts both the preserved autocorrelation and the marginal distribution. The standard solution is Amplitude-Adjusted Fourier Transform (AAFT), which iteratively adjusts the surrogate to match both the power spectrum and the marginal distribution of the original. AAFT alternates between frequency-domain adjustments (matching the power spectrum) and amplitude-domain adjustments (matching the rank-ordered distribution), converging on a surrogate that preserves both properties simultaneously.

The phase randomization approach complements the Markov null models because it operates on a fundamentally different axis: Markov models test local transition structure, while phase randomization tests global spectral structure. A sequence could exceed the Markov-1 null but match the phase-randomized null (structure is in the autocorrelation), or vice versa (structure is in local transitions but not spectral). Therefore, the combination provides a more complete picture of what kind of temporal organization USV code sequences possess.

---

Source:
- vacation-master-plan-v2 (archived to archive/inbox/)

Relevant Notes:
- [[null models are essential for interpreting information-theoretic metrics on USV code sequences]] -- phase randomization is one level in the hierarchical null model framework

Topics:
- [[representation-learning]]
- [[signal-processing]]

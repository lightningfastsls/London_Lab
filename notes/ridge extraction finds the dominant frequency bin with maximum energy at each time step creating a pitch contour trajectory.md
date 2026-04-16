---
description: "Ridge extraction (argmax per spectrogram column with continuity constraints) produces a pitch contour vector from STFT output — core step of Omer vectorization"
type: method
confidence: proven
conditions:
  - MATLAB tfridge provides continuity-constrained version; Python argmax per column is simpler alternative
meta_state: current
source: "inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md"
topics:
  - "[[signal-processing]]"
  - "[[classification]]"
---

# ridge extraction finds the dominant frequency bin with maximum energy at each time step creating a pitch contour trajectory

Ridge extraction identifies the dominant frequency at each time step in a spectrogram by finding the frequency bin with maximum energy per column. In the Omer lab implementation (Oren et al. 2024), this is done via MATLAB's `tfridge` function, which adds **continuity constraints** — the ridge cannot jump arbitrarily between frequency bins, enforcing smooth pitch tracking across adjacent time steps.

For a spectrogram S(f, t) with F frequency bins and T time steps, the simplest ridge extraction is:

```
FM(t) = f[argmax_f S(f, t)]   for each time step t
```

This produces a T-element vector of frequencies — the frequency modulation (FM) trajectory or pitch contour. The corresponding amplitude modulation (AM) trajectory extracts the spectrogram magnitude at the ridge location:

```
AM(t) = |S(FM_index(t), t)|   for each time step t
```

This is exactly what Mickey described: "find in each pixel column what is the highest amplitude and take that one." The technique maps directly to mouse USV pitch contour extraction, where the dominant frequency within the 30-110 kHz band at each time step traces the USV's frequency modulation pattern.

**Continuity constraints matter** for harmonic calls where energy may briefly shift between harmonics or noise spikes. Without constraints, the ridge can jump to noise or harmonics. MATLAB's `tfridge` uses penalty-based tracking; a Python equivalent could use dynamic programming on the magnitude matrix or `scipy.signal` peak tracking.

The extracted ridge is then resampled to a fixed number of time steps: [[time-axis resampling to a fixed number of steps normalizes variable-duration vocalizations without discarding frequency information]].

---

Source:
- inbox/oren-2024-vocal-labeling-deep-read-2026-04-15.md (deep read, 2026-04-15)
- Oren, G. et al. (2024). Science, 385(6712), 996-1003.

Relevant Notes:
- [[Omer lab 80-dimensional FM plus AM ridge vectorization embeds each vocalization call in a fixed-length feature space]] -- the full vectorization pipeline this step feeds into
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- our STFT parameters determine the resolution of ridge extraction
- [[Hann window provides good sidelobe suppression for spectral analysis of USVs]] -- window choice affects ridge quality
- [[temporal resolution versus frequency resolution in STFT parameter selection]] -- tradeoff affects ridge extraction fidelity
- [[iMSA rule-based pitch-jump classification produces the highest SIS among compared methods despite lower label entropy]] -- iMSA's pitch-jump rules implicitly perform ridge extraction; its top SIS score validates that ridge-based features carry sequential predictive power

Topics:
- [[signal-processing]]
- [[classification]]

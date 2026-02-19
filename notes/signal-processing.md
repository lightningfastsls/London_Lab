---
description: STFT computation, frequency analysis, spectrogram parameters, and sample rate conventions at 300 kHz
type: moc
---

# signal-processing

The DSP foundation for everything else. All audio is recorded at 300 kHz (Nyquist to 150 kHz). STFT parameters balance temporal and frequency resolution for short, narrow-band USV signals. Detection and visualization use intentionally different STFT configs.

## Core Ideas
- [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]] -- canonical sample rate, Nyquist to 150 kHz
- [[512-point FFT at 300 kHz gives 1.7 ms temporal resolution with 586 Hz frequency bins]] -- core STFT parameter choice
- [[75 percent overlap with hop length 128 provides smooth temporal coverage for USV detection]] -- 0.427 ms hop duration
- [[visualization STFT uses different parameters than detection STFT by design]] -- n_fft=2048 (61 Hz/bin) vs n_fft=512 (586 Hz/bin)
- [[frequency resolution of 586 Hz per bin suffices to distinguish USV subtypes in the 20-120 kHz range]] -- ~171 bins in the USV band
- [[Hann window provides good sidelobe suppression for spectral analysis of USVs]] -- standard window for spectral analysis
- [[auto sample rate reading from WAV headers prevents silent frequency miscalculation]] -- robustness for varying recording setups
- [[temporal resolution versus frequency resolution in STFT parameter selection]] -- the fundamental uncertainty principle tradeoff
- [[per-frequency-bin normalization removes frequency-dependent energy bias in spectrogram input]] -- zero mean/unit variance per frequency bin for transformer input
- [[20-120 kHz detection range pads the mouse USV band to avoid clipping edge-case calls]] -- 30-110 kHz USV range with padding
- [[chevron calls expose the STFT time-frequency tradeoff because they require simultaneous temporal and spectral precision]] -- concrete example of the time-frequency tradeoff
- [[per-recording normalization compensates for varying noise floors across recording sessions]] -- dynamic vmin/vmax based on per-recording stats

## Artifact Patterns
- [[electrical interference at 60 kHz harmonics produces horizontal lines easily distinguishable from USVs]] -- narrowband horizontal artifacts from power line harmonics
- [[transient cage noises produce broadband vertical smears rejected by the minimum duration filter]] -- broadband vertical artifacts from cage impacts
- [[shared lab space without sound attenuation explains why noise robustness is a primary design constraint]] -- the recording environment that produces both artifact types

## Open Questions
- Whether alternative window functions could improve USV detection
- Optimal frequency resolution for USV subtype discrimination

## Related Areas
- [[detection]] -- energy detection depends on STFT output
- [[classification]] -- CNN input is STFT-derived spectrograms
- [[experimental-methods]] -- recording setup determines sample rate

---

Topics:
- [[index]]

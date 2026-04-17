---
description: DSP foundation — 300 kHz sample rate (Nyquist to 150 kHz), with intentionally different STFT configs for detection (temporal precision) vs visualization (frequency detail)
type: moc
topics: "[[index]]"
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

## Ridge-Based Feature Extraction
- [[ridge extraction finds the dominant frequency bin with maximum energy at each time step creating a pitch contour trajectory]] -- argmax per spectrogram column with optional continuity constraints (Oren 2024 / MATLAB `tfridge`)
- [[time-axis resampling to a fixed number of steps normalizes variable-duration vocalizations without discarding frequency information]] -- 2D interpolation to fixed time steps; contrasts with zero-pad/center-crop
- [[pre-filtering layers each address a distinct ridge-extraction failure mode so removing any one layer likely reintroduces the failure it was blocking]] -- 1-to-1 mapping between defenses and failures: amplitude threshold (silent columns), median filter (broadband transients), band mask (out-of-band noise), DP tracking (harmonic jumps)
- [[Oren marmoset ridge vectorization requires re-engineering not parameter tuning when adapted to mouse USVs because duration frequency band harmonics SNR and absolute-pitch relevance all differ]] -- five domain mismatches (duration, frequency band, harmonics, SNR, absolute pitch) each change pipeline structure rather than thresholds

## Acoustic Property Extraction
- [[acoustic property extraction from spectrogram data produces ground truth targets for probing experiments]] -- seven properties computed directly from spectrogram columns (170 bins, 20-120 kHz) as probe labels

## Autoencoder Preprocessing
- [[BCE loss with sigmoid output treats spectrogram pixels as independent probabilities requiring input normalization to 0-1 range]] -- loss function choice cascades to normalization and output activation
- [[per-spectrogram max normalization is the simplest effective preprocessing for BCE-based spectrogram reconstruction]] -- divide by max; no fitted parameters; discards absolute amplitude
- [[symmetric zero-padding for short USVs and center-cropping for long ones standardizes variable-duration inputs to fixed dimensions]] -- AMVOC pads/crops to 64 frames (128ms); generalizable method
- [[AMVOC uses 2ms non-overlapping spectrogram windows giving 0.5 kHz frequency resolution at the expense of temporal smoothness]] -- contrast with our 75% overlap; independent frames vs smooth coverage

## Cross-Tool STFT Comparisons
- [[MUPET operates at 250 kHz sample rate with minimum 90 kHz requirement covering the 25-125 kHz USV band]] -- 250 kHz standard; 8th-order Chebyshev at 25 kHz; Nyquist 125 kHz vs our 150 kHz
- [[MUPET 2 ms frame duration with 80 percent overlap prioritizes temporal resolution for capturing rapid USV frequency modulations]] -- 512-point FFT at 250 kHz: ~0.5 kHz bins, Hamming; convergent with our parameters
- [[MUPET gammatone filterbank with k-means discovers 100 to 140 data-driven syllable types as a handcrafted feature baseline]] -- 64-channel Gammatone filterbank as alternative to direct STFT
- [[25000-125000 Hz is the standard mouse USV frequency band used across bioacoustic tools for defining regions of interest]] -- cross-tool convention; slightly different from our 20-120 kHz
- [[perceptual loss outperforms pixel-level MSE for autoencoder spectrogram representation learning]] -- VGG-based loss preserves structural features over pixel reconstruction

## Classification-Specific STFT
- [[fine frequency resolution matters more than time resolution for CNN classification of USV spectrogram patches]] -- at least 512-point FFT, 1024 preferred for ~293 Hz resolution
- [[DeepSqueak 3.2 ms FFT window with 2.8 ms overlap translates to 960-sample FFTs at 300 kHz]] -- DeepSqueak's duration-based FFT specification accepts any sample rate
- [[DeepSqueak uses constant-duration FFT windows making it inherently sample-rate agnostic]] -- specifying windows in seconds means 250 kHz and 300 kHz recordings produce comparable spectrograms
- [[gammatone spectrograms outperform standard STFTs for USV classification according to BootSnap]] -- auditory-model-inspired alternative to uniform STFT bins
- [[entropy-based USV detection achieves 94.9 percent recall and 99.3 percent precision as a classical signal processing alternative]] -- entropy measures spectral complexity rather than raw energy

## Frequency-Domain Null Models
- [[phase randomized null model preserves autocorrelation spectrum while destroying higher-order structure]] -- FFT phase randomization tests whether linear temporal correlations explain code sequence patterns

## Sample Rate & Domain Transfer
- [[spectrogram-based SSL avoids the sample rate mismatch that limits waveform-based models for USV analysis]] -- waveform SSL expects 16-48 kHz; spectrograms abstract away the raw sample rate
- [[the 300 kHz USV sample rate creates a domain shift challenge for applying audio foundation models]] -- 10-19x gap with pretrained models; frequency shifting or spectrogram-as-image needed
- [[frequency shifting USVs into the audible range could enable classification with standard audio foundation models]] -- pitch-shift 50-90 kHz to 2-10 kHz preserving relative spectral structure
- [[PCEN normalization is more robust than log-mel spectrograms for few-shot bioacoustic scenarios]] -- per-channel energy normalization adapts to local noise; robust to varying recording conditions
- [[BirdVoxDetect PCEN reduced false alarm rates 50x near-field and 5x far-field]] -- strongest empirical evidence for PCEN: 50x FP reduction near-field, driven by adaptive normalization of distance-dependent energy variation

## Source Separation Signal Processing
- [[frequency separation provides a partial solution when overlapping USVs occupy different spectral bands]] -- spectral peak splitting when calls occupy different frequency ranges
- [[Conv-TasNet time-domain separation architecture could handle 300 kHz USV recordings directly but requires ultrasonic training data]] -- time-domain approach natively handles any sample rate; 19x compute cost vs speech
- [[A-MUD classical signal processing detector outperforms USVSEG and MUPET in true positive rate for USV detection]] -- 90.6%/80.0% precision/recall; best classical method; requires STx software

## Open Questions
- Whether alternative window functions could improve USV detection
- Optimal frequency resolution for USV subtype discrimination

## Related Areas
- [[detection]] -- energy detection depends on STFT output
- [[classification]] -- CNN input is STFT-derived spectrograms
- [[experimental-methods]] -- recording setup determines sample rate
- [[generative-modeling]] -- bounded gain stability and error amplification patterns transfer to signal processing pipelines; the notes on [[error amplification near targets is a general instability pattern in iterative refinement systems beyond diffusion models]] and [[bounded gain in iterative refinement prevents error amplification while unbounded gain creates structural instability regardless of domain]] are cross-listed here

---

Topics:
- [[index]]

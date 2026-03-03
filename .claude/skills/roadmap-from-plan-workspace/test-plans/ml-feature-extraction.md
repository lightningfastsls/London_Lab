# Audio Feature Extraction Pipeline

We discussed building a feature extraction pipeline for the USV analysis project. Here's what we came up with:

## Step 1: Feature Extractor Config

Create a configuration dataclass for the feature extraction parameters. We want to extract:
- Mel-frequency cepstral coefficients (MFCCs) — 13 coefficients
- Spectral centroid and bandwidth
- Zero-crossing rate
- Chroma features (12 bins)

The config should specify the sample rate (300kHz for USV), FFT window size, hop length, and which features to extract. Use the frozen dataclass pattern that the project already uses for SpectrogramConfig and DetectionConfig.

## Step 2: Feature Extraction Engine

Build the actual extraction engine that takes WAV segments and produces feature vectors. Should:
- Accept a list of candidate segments (start_time, end_time) from the detection pipeline
- Load each segment from the WAV file using the existing io_wav.py utilities
- Compute all requested features using librosa (remember to always specify sr=300000!)
- Stack into a feature matrix (n_segments x n_features)
- Support normalization (z-score per feature, optional)

Key design decision: We chose per-segment extraction over whole-file extraction because USV calls are short (10-100ms) and scattered. Processing the whole file would waste computation on silence. The detection pipeline already gives us precise segment boundaries.

## Step 3: Feature Cache

Add caching to avoid recomputing features when re-analyzing the same WAV file with the same config. Use a simple SQLite cache keyed on (wav_path_hash, config_hash, segment_hash). This is important because feature extraction is the bottleneck — each segment takes ~50ms on CPU, and a typical recording has 500-2000 detected segments.

## Step 4: CLI Entry Point

Create a script that takes a WAV file path and a detections JSON file, runs feature extraction, and saves the feature matrix as an NPZ file. Should support `--normalize` and `--features` flags to select which features to compute.

Usage should look like:
```
python scripts/extract_features.py "5970 USV/recording_001.wav" --detections detections.json --output features.npz --normalize --features mfcc,spectral_centroid
```

## Scientific Background

The MFCCs are computed using the discrete cosine transform of the log mel-spectrogram. The mel scale warps frequencies to approximate human perception, though for USV (20-120 kHz range) we're well above human hearing. The mel warping still helps because it compresses the frequency axis in a biologically-motivated way — many mammalian hearing systems show logarithmic frequency resolution.

For spectral centroid, we're computing the weighted mean of frequencies present in the signal, weighted by their magnitudes. This gives a single number characterizing the "brightness" of each USV call. Combined with bandwidth (the spread around the centroid), these features capture the frequency profile without needing the full spectrogram.

Zero-crossing rate is the simplest feature but surprisingly useful for USV classification — tonal calls (frequency modulated sweeps) have very different zero-crossing patterns compared to noisy/broadband calls.

The chroma features are less obviously useful for USV (they're designed for musical pitch analysis), but we include them as a hypothesis — if USV calls have harmonic structure, chroma features would capture it. This is speculative and we may drop them after initial experiments.

## References
- Riede (2011) "Subglottal pressure, tracheal airflow, and intrinsic laryngeal muscle activity during rat ultrasound vocalization" — physiology of USV production
- Coffey et al. (2019) "DeepSqueak: a deep learning-based system for detection and analysis of ultrasonic vocalizations" — feature extraction approach for USV

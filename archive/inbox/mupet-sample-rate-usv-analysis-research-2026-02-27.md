---
description: "MUPET uses 250 kHz sampling rate for USV analysis, supports any rate above 90 kHz, with 512-point FFT and 64-channel Gammatone filterbank"
source_type: paper
url: "https://pmc.ncbi.nlm.nih.gov/articles/PMC5939957/"
author: "Van Segbroeck, Knoll, Levitt, Narayanan"
date_accessed: "2026-02-27"
status: processed
research_tool: "web-search"
research_query: "MUPET sample rate USV analysis ultrasonic vocalization"
research_depth: "quick"
---

# MUPET Sample Rate and Signal Processing Parameters

MUPET (Mouse Ultrasonic Profile ExTraction) uses a 250 kHz sampling rate for its USV analysis, though the tool can process audio files recorded at any sampling rate above 90 kHz. This 250 kHz rate provides a Nyquist frequency of 125 kHz, which fully covers the 25-125 kHz ultrasonic range where mouse vocalizations occur.

---

## Sample Rate and Frequency Range

According to Van Segbroeck et al. (2017), audio files used with MUPET were collected at a sampling rate of 250 kHz. The tool's minimum requirement is a sampling rate above 90 kHz, which makes sense given that MUPET analyzes vocalizations in the 25-125 kHz range. An 8th-order Chebyshev filter with a 25 kHz corner frequency extracts the ultrasonic frequency range from the raw audio, effectively bandpassing the signal to isolate USV content from lower-frequency noise.

The 250 kHz rate is notably lower than the 300 kHz rate used in this project's pipeline. This has implications for frequency resolution and the usable bandwidth: at 250 kHz, the Nyquist limit is 125 kHz, while at 300 kHz it extends to 150 kHz.

## STFT Parameters

MUPET uses a 512-point STFT algorithm with the following parameters (at 250 kHz sampling rate):
- **Frame size**: 500 samples (2 ms duration)
- **Frame shift**: 400 samples (1.6 ms, giving 80% overlap)
- **Window**: Hamming
- **Frequency resolution**: approximately 0.5 kHz (250000 / 512)

These parameters prioritize temporal resolution (2 ms frames) over spectral resolution, which is appropriate for capturing the rapid frequency modulations typical of mouse USVs.

## Gammatone Filterbank (GF-USV)

Rather than operating directly on spectrograms, MUPET converts them into compact "GF-USV" representations using a 64-channel Gammatone filterbank. The filter bandwidths vary from 0.5 to 4 kHz, with narrower filters concentrated in the frequency regions containing the most acoustic events. This biologically-inspired representation (mimicking the auditory periphery) enables more robust syllable comparison than raw spectrograms.

## Comparison to This Project's Parameters

This project operates at 300 kHz sampling rate. Key differences from MUPET's approach:
- Higher Nyquist frequency (150 kHz vs 125 kHz)
- MUPET's frequency range (25-125 kHz) is fully representable at both rates
- MUPET's filterbank approach differs from this project's direct energy detection on STFT output
- MUPET uses unsupervised clustering (k-means with cosine distance) rather than energy-based detection

---

## Source Log

| # | URL | Status | Relevance | Key Finding |
|---|-----|--------|-----------|-------------|
| 1 | https://pmc.ncbi.nlm.nih.gov/articles/PMC5939957/ | fetched | high | Primary paper: 250 kHz sample rate, >90 kHz minimum, 512-point FFT, 64-channel Gammatone filterbank |
| 2 | https://sail.usc.edu/mupet/ | fetch failed (SSL cert) | medium | MUPET project homepage, could not retrieve due to certificate error |
| 3 | https://www.sciencedirect.com/science/article/pii/S0896627317302982 | search result, not fetched | medium | Same paper on ScienceDirect (duplicate of PMC source) |

## Research Context

- **Query**: what sample rate does MUPET use for USV analysis
- **Depth**: quick (auto-detected -- specific factual question about a single tool)
- **Existing vault knowledge**: No existing notes on MUPET found in the vault
- **Knowledge gap addressed**: MUPET's signal processing parameters, especially sample rate, which enables comparison with this project's 300 kHz pipeline

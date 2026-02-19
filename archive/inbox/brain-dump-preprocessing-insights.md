---
source: researcher brain-dump
topic: spectrogram preprocessing insights
date: 2026-02-19
method: structured interview (AskUserQuestion)
---

# Spectrogram Preprocessing Insights — Brain Dump

## Frequency Range Selection

Mouse USVs fall roughly in 30-110 kHz. The 20-120 kHz detection range is padded on both ends to avoid clipping edge-case calls. Below ~25 kHz is mostly environmental noise and audible-range sounds. Above 110-120 kHz there's very little signal and microphone sensitivity drops off. The exact boundaries aren't critical — what matters is capturing the full USV range without flooding the spectrogram with irrelevant low-frequency noise.

## STFT Parameter Lessons

n_fft=512 at 300 kHz gives ~1.7ms time resolution and ~585 Hz frequency resolution. Hop length of 128 samples (~0.43ms per frame) gives good temporal overlap.

Key lesson: short calls (<15ms) get somewhat smeared, but this is acceptable for binary detection.

The chevron call problem: needing both good time resolution (to see that the call is short) and good frequency resolution (to see the up-then-down trajectory). A moderate n_fft is a compromise; you accept some smearing on the shortest calls. This was an important conceptual lesson about the time-frequency tradeoff.

## Artifact Types

- Electrical interference at 60 kHz harmonics — appears as perfectly horizontal lines, easily distinguishable from USVs
- Transient cage noises — broadband vertical smears
- Minimum duration filter (~8-10ms) rejects transient artifacts

## The Big Preprocessing Lesson

The energy detector created selection bias by filtering out quiet USVs, which then biased CNN training. This was the most consequential preprocessing insight — it wasn't a spectrogram parameter issue, it was a pipeline design issue where upstream filtering silently shaped downstream training data. This is why multi-source negative sampling and training bias correction were so important.

## Normalization

Dynamic per-spectrogram normalization using vmin/vmax based on per-recording statistics. Different recordings have different noise floors — a recording from one session might have a higher ambient noise level than another. Per-sample normalization (min-max to [0,1] or standardization) works for the CNN. The key lesson: without per-recording normalization, the model's performance varies across recordings.

---
source: researcher brain-dump
topic: labeling expertise
date: 2026-02-19
method: structured interview (AskUserQuestion)
---

# USV Labeling Expertise — Brain Dump

## What makes a good vs bad USV training sample?

A good training sample is a spectrogram patch that clearly contains (or clearly doesn't contain) a USV. For positives, the ideal is a bright coherent frequency trace against a dark background — but critically, USVs embedded in noise are still valid positives. A USV doesn't stop being a USV just because there's background noise around it. The model needs to learn to detect USVs in clean and noisy conditions. For negatives, a good sample is something that's unambiguously not a USV — pure noise, electrical artifacts, silence, or transient cage sounds.

## What edge cases do you encounter when labeling?

- USVs inside noisy regions — still labeled positive, and important for training robustness
- Harmonics (e.g., 45 kHz fundamental + 90 kHz copy) — treated as one call
- Overlapping calls from 2 mice in the cage — labeled as positive (USV present)
- Long USVs with a noisy gap in the middle — the CNN currently splits these into two detections, which is a known behavior (not necessarily wrong, but worth documenting)
- Very short signals near the 8-10ms boundary

## How do you handle uncertain labels?

The detection system is a CNN running as a sliding window across the entire WAV file. Automatic detections save with their probability scores. Manually saved detections are marked as "user saved" but don't show probability — they represent cases where you spotted something the CNN missed or wanted to override. So there's an implicit two-tier system: CNN confidence (probabilistic) and human override (binary).

## What are the most common false positives and false negatives?

FPs: Mostly noise regions — though the CNN actually ignores most noise, it sometimes flags noisy patches, especially areas inside noise (the CNN may be picking up on energy patterns within noise that superficially resemble USV structure).

FNs: Quiet/silent USVs (low amplitude, near the noise floor) and small/short ones — these are the hardest to catch and were the original source of training bias in earlier iterations.

Known artifact: Long USVs with noise interrupting the middle get split into two separate detections. This inflates call counts but each detected segment is still a real USV fragment.

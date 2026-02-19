---
source: researcher brain-dump
topic: scientific hypotheses
date: 2026-02-19
method: structured interview (AskUserQuestion)
---

# Scientific Hypotheses — Brain Dump

## Core Hypothesis: Courtship Vocal Degradation in Lab Mice

A century of inbreeding and the absence of natural selection pressure for courtship success in lab environments has caused lab mice (e.g., C57BL/6) to degrade or lose courtship behavioral competence. USVs are one marker of this — lab mice don't need to "work" for breeding in captivity, so the vocal courtship repertoire may have atrophied.

The hypothesis is directional: wild mice retain richer, more complex courtship behavior, and lab mice have lost or reduced it. This is not just "different" — it's specifically a degradation story in the lab population.

Evidence already observed: wild mice show much more diverse USV repertoires than lab mice.

## Expected Differences

- Wild mice: richer USV repertoires (already observed), potentially more complex sequential structure, more varied call types
- Lab mice: reduced/simplified repertoire, possibly fewer call types or more stereotyped sequences

## Behavioral Context

Courtship — male-female pairs. USVs are one component of a multimodal courtship behavior suite that also includes mounting, approach, and other movements. The USV analysis will eventually be integrated with MiceCraft (video-based detection of known movement behaviors) to build a complete picture of courtship competence across modalities.

### USV-Behavior Correlation
We want to explore whether specific USV types correlate with specific behavioral outcomes — for example, whether a certain type of USV makes the female more likely to allow the male to mount her. This requires temporal alignment between USV detections and behavioral events from LMT/MiceCraft.

## Two-Tier Analysis Strategy

1. **Immediate (pre-VQ-VAE)**: Use DeepSqueak's built-in classification to categorize each USV call and compare repertoire distributions between wild and lab populations. The core courtship degradation finding can likely be demonstrated at this stage.

2. **Deeper investigation (VQ-VAE + Transformer)**: Investigate whether USV sequential structure has language-like properties. This is a separate, deeper question about the nature of USV communication that goes beyond the courtship degradation finding.

## Significance Criteria

A significant result: showing that wild mice use more call types, with different distributions and potentially more complex temporal patterning, consistent with courtship behavioral degradation in lab strains. Combined with MiceCraft movement data, this builds the case across multiple behavioral markers.

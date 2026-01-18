# USV Labeling Guide

This guide helps you consistently label USV (ultrasonic vocalization) candidates as either **USV** or **Not USV**.

## Quick Reference

| Label | Use When |
|-------|----------|
| **USV** | Clear tonal signal with coherent frequency structure in 25-110 kHz range |
| **Not USV** | Noise, artifacts, interference, or unclear signals |
| **Uncertain** | Cannot confidently decide (use sparingly, <10% of candidates) |

---

## What USVs Look Like

### Characteristic Features of Real USVs

1. **Narrow-band signal** - Energy concentrated at specific frequency (not spread across entire band)
2. **Coherent shape** - Smooth frequency contour (flat, rising, falling, or modulated)
3. **Clear edges** - Distinct start and end points
4. **Appropriate duration** - Typically 10-300 ms (most common: 30-100 ms)
5. **Within frequency range** - 25-110 kHz (most common: 40-80 kHz)

### Common USV Types in Mouse Recordings

**Flat calls:**
- Constant frequency throughout
- Appears as horizontal line on spectrogram
- Duration: 20-100 ms

**Frequency-modulated (FM) sweeps:**
- Frequency changes over time (upward or downward)
- Appears as diagonal line on spectrogram
- Can be short or long duration

**Chevrons/hooks:**
- Frequency rises then falls (or vice versa)
- Appears as inverted V or V shape
- Often 30-80 ms duration

**Complex calls:**
- Multiple frequency changes
- May have harmonics (energy at 2x fundamental frequency)
- Longer duration (50-200 ms)

---

## What is NOT a USV

### Noise Characteristics

1. **Broadband energy** - Vertical smear across many frequencies simultaneously
2. **No coherent shape** - Random or chaotic patterns
3. **Too short** - <10 ms is almost always artifact
4. **Too long** - >500 ms is usually interference
5. **Perfectly stable frequency** - Exactly 50/60 kHz suggests electrical interference

### Common False Positives

**Clicks/transients:**
- Very brief (<10 ms)
- Broadband (vertical line on spectrogram)
- Often from cage/equipment

**Electrical interference:**
- Perfectly constant frequency (often 50 or 60 kHz harmonics)
- Very long duration (seconds)
- Regular pattern

**Background noise:**
- Diffuse energy across frequency band
- No clear boundaries
- Low intensity throughout

---

## Labeling Decision Flowchart

```
1. Is there a visible signal in the candidate region?
   NO → Label: Not USV
   YES → Continue

2. Is the signal narrow-band (concentrated at specific frequency)?
   NO (broadband/vertical) → Label: Not USV
   YES → Continue

3. Does it have coherent frequency structure (smooth contour)?
   NO (chaotic/random) → Label: Not USV
   YES → Continue

4. Is duration appropriate (10-500 ms)?
   NO → Label: Not USV
   YES → Continue

5. Is it within 25-110 kHz?
   NO → Label: Not USV
   YES → Label: USV
```

---

## Edge Cases and How to Handle Them

### Faint but visible USVs
**Decision:** Label as USV
- If you can see the characteristic shape, it's a USV
- Faint USVs are valuable training data
- Don't penalize low amplitude

### Partial USVs (cut off at edge)
**Decision:** Label as USV if clearly identifiable
- If the visible portion clearly shows USV characteristics, label as USV
- If too little is visible to judge, mark Uncertain

### Multiple calls in one candidate
**Decision:** Label as USV
- Binary classification: "Is there at least one USV present?"
- Multiple overlapping calls still count as "USV present"

### USV with noise nearby
**Decision:** Label as USV
- Focus on whether a USV is present
- Background noise doesn't disqualify a real USV

### Harmonics
**Decision:** Count as one USV
- Some calls have energy at 2x fundamental (e.g., 40 kHz + 80 kHz)
- This is one call, not two
- Label as USV

---

## Reading the Spectrogram Display

### Axes
- **Y-axis:** Frequency in kHz (20-120 kHz range)
- **X-axis:** Time in milliseconds
- **Color:** Intensity in dB (brighter = louder)

### Markers
- **Green vertical lines:** Candidate region boundaries
- **Title:** Candidate ID and peak frequency

### Color Scale
- **Yellow/white:** High energy (loud signal)
- **Orange/red:** Medium energy
- **Dark purple/black:** Low energy (noise floor)

---

## Labeling Best Practices

1. **Be consistent** - Apply the same criteria throughout
2. **When in doubt, label USV** - False negatives are worse than false positives for training
3. **Take breaks** - Every 50-100 candidates, rest your eyes
4. **Don't overthink** - Most decisions should take <3 seconds
5. **Use Uncertain sparingly** - Target <10% uncertain rate
6. **Review periodically** - Every 100 labels, check a few previous decisions

---

## Calibration Exercise

Before starting, label these practice candidates and compare with the key:

| Candidate | Expected Label | Reasoning |
|-----------|---------------|-----------|
| Clear frequency sweep 50-70 kHz, 40ms | USV | Classic FM sweep |
| Vertical broadband spike, 5ms | Not USV | Click artifact |
| Faint horizontal line at 55 kHz, 60ms | USV | Flat call (faint is OK) |
| Diffuse energy, no clear shape | Not USV | Background noise |
| Perfectly stable 60 kHz, 2000ms | Not USV | Electrical interference |

---

## Questions?

If you encounter candidates that don't fit these guidelines, note the candidate ID and discuss before continuing. Consistency is more important than any individual decision.

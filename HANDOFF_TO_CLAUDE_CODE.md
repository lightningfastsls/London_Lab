# Handoff: Final assets needed for USV deck

The deck (`USV_presentation_v3.pptx`) now has all 27 slides built with Mickey's
feedback applied. Three placeholders remain that need real assets, plus one
clarification.

## 1. Challenging-clip waveform (slide 5)

**What's needed:** a second waveform PNG showing a clip where USVs are even
harder to spot than the first slide. Same styling as the existing
`presentation/figures/00_raw_waveform.png` so the pair on slides 4 → 5
visually rhymes.

- ~3.5 : 1 aspect, 2100 × 600 px, same matplotlib settings (dark slate line,
  white background, no decoration)
- 2-second window like slide 4
- Pick a clip with one or more of: lower-amplitude USVs, denser background
  artifacts (paw-scuffs / scratching / husbandry transients), or a USV near
  the noise floor — whatever makes the case for "the detector has to work in
  the wild" most viscerally
- Save as `presentation/figures/00_raw_waveform_challenging.png`

Bonus (optional): if the clip you pick has CNN-detected USVs, include their
time positions in the return so I can mark them with thin coral verticals on
the slide, reinforcing the point that the model finds calls the eye can't.

## 2. Spectrogram thumbnails for the 7 syllable types (slide 14)

**What's needed:** one canonical spectrogram exemplar per type, to sit above
each bar in the type-distribution chart.

Path pattern: `presentation/figures/08_classification/gallery/<TYPE>/01_*.png`

Types and order needed (left-to-right on the slide, to match the bar chart):

```
1. Short
2. Flat
3. Up
4. Down
5. Chevron
6. Complex
7. Frequency_Jump
```

For each: pick the `01_*.png` (canonical exemplar) from that type's gallery
folder. If 01 looks unrepresentative when you check it, swap for whichever of
the 5 in the folder reads most clearly as that type at thumbnail size.

Crop/resize each to roughly **120 × 140 px** (slightly taller than wide,
matching the slot dimensions on the slide). Strip axis labels and titles if
present — these are thumbnails, not full figures.

Save as:
```
presentation/figures/thumbs/01_short.png
presentation/figures/thumbs/02_flat.png
presentation/figures/thumbs/03_up.png
presentation/figures/thumbs/04_down.png
presentation/figures/thumbs/05_chevron.png
presentation/figures/thumbs/06_complex.png
presentation/figures/thumbs/07_frequency_jump.png
```

## 3. Recording-setup video (slide 3)

This one is a video file, not a figure. **No Claude Code action needed** —
just drop the video directly into the slide-3 placeholder in PowerPoint.

A short clip showing the LMT cage + mouse couple + overhead microphone setup
would work — anything 10-30 seconds that lets the audience see what "wild
mouse in a cage with a 300 kHz mic" actually looks like.

## 4. Clarification needed (not a fetch task)

Mickey referenced **fnbeh-09-00076-g001.jpg** (Chabout et al. 2015, "Male mice
song syntax depends on social contexts and influences female preferences,"
Frontiers in Behavioral Neuroscience).

I added a citation note on slide 22 (transitions) — "Compare to Chabout et al.
2015 — context-dependent syntax in lab mouse song." But I couldn't tell from
the comment whether Mickey wanted:

- (a) Just the citation, which I've added → **no further action**
- (b) An actual comparison panel showing their Figure 1 alongside ours
- (c) A dedicated slide referencing their result before our transition matrix

If Mickey wanted (b) or (c), let me know and I'll restructure that section.

## How to deliver

Drop the 8 new PNGs (`00_raw_waveform_challenging.png` + 7 thumbnails) plus
this handoff into a fresh web claude.ai session and ask it to update the deck
with the real assets in place of the placeholders. The build script is at
`/home/claude/build_deck_v3.js` in that session's context.

## What's already done (no action needed)

| Mickey's comment | What changed |
|---|---|
| Slide 2 — visual vs text | Replaced 3-paragraph block with two side-by-side timeline cards. Just the numbers + 4-word labels. |
| Slide 2.5 — movie | Added slide 3, "How we record", with video placeholder. |
| Slide 3 — show zoom region + challenging example | Slide 4 now has the red zoom-region box + dashed leader lines connecting to the zoomed panel. Slide 5 reserved for the challenging clip (placeholder). |
| Slide 8 — more visually driven | Slide 10 now uses real spectrogram thumbnail input + 3D feature-map stacks + FC nodes-and-edges + sigma output node. Modeled on the reference image you showed. |
| Slide 12 — thumbnails above bars | Slide 14 now has a 7-thumbnail strip above the chart (placeholders, awaiting Block 2 assets). |
| Slide 13 — simplify | Slide 15 rebuilt as a native pptx table: ms not s, no ±SD, dropped Tonality + Principal Freq, single one-line takeaway at the bottom. |
| Slide 14 — zoom in | Slide 16 now uses a cropped image (focus on main blob, drops outliers) + custom side legend. |
| fnbeh paper | Citation added to slide 22 (transitions); flagged for clarification above. |

I have everything I need. All 10 figures exist and all 10 are referenced in body text; the abstract is 267 words (over limit); and there is a figure-ordering inversion. Here is the prioritised critique.

---

# Prioritised Gap List — Atgar 2026 Thesis Review

## TIER 1 — MISSING REQUIRED ELEMENTS / HARD-LIMIT VIOLATIONS (fix first)

**1.1 — Abstract exceeds the 250-word cap (267 words). HARD RUBRIC LIMIT.**
Measured: 267 words. You must cut ≥17 words. Lowest-information sentences to trim:
- "The principled next step is a shape-only contour comparison." (10 words) — already implied by the Discussion's future-work; safe to delete from the abstract.
- "— these are window-level, not whole-file, metrics." (7 words) — the honesty caveat is valuable, but if you keep it, drop the trailing future-work sentence above and you land at ~257; you still need ~7 more. Tighten "I developed a Python-native, CNN-aligned PyQt6 labeling and visualization application" → "a CNN-aligned PyQt6 labeling application" (saves ~6). Combined, that lands you under 250.
Fix: delete the final sentence + compress the tooling sentence. Do not cut the "promising leads, not a proven difference" sentence — that is the rubric's honesty anchor.

**1.2 — Title page is incomplete (placeholder bracket present).**
Quote: "**Atgar Final Project — [School / Mentor: to fill]**". A title page is a required chapter and currently ships with an unfilled template field. Fix: populate the school/mentor before submission, or the title page reads as a draft.

**1.3 — Bibliography contains uncited references (story-coherence + rubric "referenced by name" spirit).**
The rubric demands ONE coherent story; the bibliography lists 17 works but several are never cited in the body. Unreferenced in-text: **Scattoni 2008 (#2), Scattoni 2009 (#3), Grimsley 2011 (#5), Hertz 2020 (#7), Stoumpou/AMVOC 2022 (#12), Abbasi/BootSnap 2022 (#13), Perrodin 2023 (#14), Boll 1979 (#16).** That is 8 of 17 entries with no anchor in the text. Fix: either cite each where it belongs (e.g., BootSnap and AMVOC belong in the Intro's "several automated approaches now exist" list alongside DeepSqueak/VocalMat/Goffinet; Hertz 2020 is your own lab's syllable-labeling work and should appear in the Intro) or remove them. A bibliography padded with uncited entries is the single most common Atgar coherence flag.

---

## TIER 2 — RESULTS STRUCTURE (sub-question → strategy → result → figure)

The four-beat pattern is present and explicit in **3.1, 3.3, 3.4, 3.5(a/b/c)** — those are clean (each opens with a bolded/italicized sub-question, states a strategy, gives a result, points at a figure). One subsection breaks the chain:

**2.1 — §3.2 "result → figure" link is broken; it points at Figure 6 instead of its own figure, and Figure 5 (its natural result figure) is deferred to §3.3.**
Quote (§3.2 result beat): "The result is that the detector fires on real USVs. **Figure 6** shows the production pipeline on a 2.5 s clip…". §3.2's sub-question is "can a CNN detect USVs directly from spectrograms"; its result figure should be the one that demonstrates detection, but Figure 5 (the three-generation detection demo, the actual evidence that it *detects*) is held back to §3.3, and §3.2 instead borrows Figure 6 from the post-processing section. Net effect: §3.2 has a strategy and a figure (4) for *architecture* but its *result* beat leans on a figure that belongs to a later sub-question. Fix: either (a) move the "fires on real USVs / Figure 6" demonstration so §3.2 closes on its own evidence, or (b) explicitly frame §3.2's result as "architecture defined; selectivity evidence follows in 3.3" so the reader isn't sent forward two figures to find the result. As written it reads as a forward-reference gap.

**2.2 — Figure presentation order is inverted (Figure 6 referenced before Figure 5).**
Figure 4 → then **Figure 6** (first textual reference, end of §3.2) → then **Figure 5** (first textual reference, §3.3). Figures should be numbered in order of first mention. Either renumber so the detection-demo figure precedes the pipeline-trace figure, or restructure §3.2/§3.3 so Figure 5 is mentioned first. This is a mechanical but visible rubric ding ("referenced by name" implies sequential introduction).

---

## TIER 3 — FIGURE CAPTION COMPLETENESS (axes / units / test / p-value)

All 10 figures are referenced in body text (verified) and all 10 files exist (verified). No figure is orphaned. Caption-content gaps:

**3.1 — Figure 3 caption: no axes/units.**
Quote: "The custom PyQt6 labeling/review application — a 20–120 kHz spectrogram panel…". This is a software screenshot, so strict axis-label scoring is softened, but the caption should still name the axes of the spectrogram/probability panels (time in s on x; frequency in kHz on y; probability 0–1 on the CNN panel). Add one clause. Acceptable as-is only if your rubric exempts UI screenshots.

**3.2 — Figure 4 caption: no axes (acceptable — it is an architecture schematic, not a data plot).** No fix required; flagging only so a grader doesn't dock it — consider adding "(schematic; no data axes)".

**3.3 — Figure 5 caption: axes present (freq kHz vs time s) ✓ but no statistical test/p-value.**
This is a qualitative count demo (1 / 17 / 4 detections), so no test is expected — that is fine. No fix needed; the counts are descriptive, not inferential.

**3.4 — Figure 7 caption: strong (n, AUC, AP, all four metrics) ✓.** No p-value, but ROC/PR curves don't require one. Good.

**3.5 — Figures 8 and 9: exemplary — units (kHz), means, % effect, BOTH tests with p-values ✓.** These are the rubric-model captions.

**3.6 — Figure 10 caption: MISSING the embedding/visualization specifics and has no quantitative support.**
Quote: "UMAP of a 32-D contour-VAE latent at matched sample sizes; wild (top row) and lab (bottom row) cohorts occupy the same call-space, indicating no gross repertoire reorganization." Two issues: (a) UMAP axes are unitless/non-interpretable — the caption should say so explicitly ("UMAP-1/UMAP-2, arbitrary units; distances not metric") so a reader doesn't over-read the geometry; (b) this is the figure backing a **null claim** ("same call-space") yet carries no quantitative overlap statistic (e.g., overlap index, kNN cross-group purity, or a silhouette/ANOSIM value). A null asserted from an unquantified UMAP is the weakest evidentiary link in the thesis. Fix: add the per-panel sample sizes to the caption (they're in body text but not the caption) and, ideally, one overlap statistic; at minimum, state in the caption that the claim is qualitative.

---

## TIER 4 — OVERCLAIM AUDIT (lab-vs-wild must stay "promising leads, not a proven difference")

The hedging is generally disciplined and repeated in Abstract, §2, §3.5, and §4. Three residual spots drift toward overclaim or undercut the hedge:

**4.1 — §3.5(a) "The rank separation was essentially perfect — nearly every wild unit exceeded every lab unit."**
Repeated in §3.5(b): "the wild and lab unit sets separated almost perfectly in rank." "Essentially/almost perfectly" reads as strong evidence and works *against* your own caveat that p ≈ 0.012 is a *floor* forced by tiny N, not power. Fix: reframe as "with N=3 vs 6, perfect rank separation is the *most* these tests can show, and it yields the floor p ≈ 0.012" — i.e., present clean ranking as a ceiling on what's knowable, not as strength. The Discussion (§4) already says this correctly ("the significance reflects clean ordering of nine aggregate per-unit IQRs, not deep statistical power") — make §3.5 consistent with §4 rather than letting §3.5 lean optimistic and §4 walk it back.

**4.2 — §3.5(c) and §5.8 cite Zala 2020 / Goffinet to make the null sound corroborated.**
Quote (§3.5c): "consistent with reports that wild-derived house mice modulate a shared repertoire by context rather than inventing distinct call types (Zala et al., 2020)." Borderline: a single UMAP overlap (no overlap statistic, n-imbalanced cohorts) being called "consistent with" published findings nudges a qualitative null toward a corroborated result. Keep the citation but soften to "is *not inconsistent with*" and tie it to the §3.6 caveat that the overlap is qualitative.

**4.3 — Abstract: "two statistically tested acoustic leads" is fine, but verify the bandwidth p-value pairing.**
Quote: "wild calls span a wider bandwidth (per-unit IQR +51%, one-sided p = 0.014)". Body and Methods report Mann–Whitney p = 0.014 **and** permutation p = 0.012 for bandwidth. The abstract cites only 0.014 for bandwidth but the bare "p = 0.012" for principal frequency — a reader may misread the two as the same test. Minor: add "(Mann–Whitney)" once so the abstract's two p-values are unambiguously the same test family. Not an overclaim, but a clarity fix that prevents an examiner inferring you cherry-picked the smaller p.

Otherwise the overclaim posture is sound: "promising leads, not a proven difference" appears verbatim in Abstract, §1, §2, and §4, and the cage/SNR confound is stated critically in §4 ("The wider wild bandwidth could therefore be partly a recording-SNR artifact rather than biology"). That confound paragraph is the strongest part of the Discussion — keep it.

---

## TIER 5 — COHERENCE, TRANSITIONS, REDUNDANCY

**5.1 — Heavy repetition of the "two detection paths" passage (stated ~4 times nearly verbatim).**
Appears in §3.1, §3.3 (full paragraph: "applying the batch FP filter inside the app context once wrongly suppressed real detections (8 → 0)"), §3.4, §4, and §5.6. The "8 → 0 on one file" anecdote appears in §3.3, §4, and §5.6. This is the most redundant thread in the thesis. Fix: state it once in full in Results (§3.3 is the right home), reference it in one clause elsewhere ("the two-path distinction, §3.3"), and keep the Methods (§5.6) statement since Methods is meant to be self-contained. Cutting two of the four full restatements also helps the length problem (Tier 6).

**5.2 — Repetition of the "invisible in waveform / 20–120 kHz / cage noise / 300 kHz tools unusable" framing.**
This exact triad appears in the Abstract, §1 (twice), §2, §3.1, §3.2, and §4. It is the thesis's load-bearing motivation, so *some* repetition is intentional and good. But §3.1 and §3.2 both re-derive it from scratch in their opening paragraphs. Fix: in §3.2 compress to "as in §3.1, the call is invisible in the waveform and shares its band with cage noise" rather than re-explaining. Saves ~60 words and removes the sense of a loop.

**5.3 — The STFT-constants paragraph is stated three times at near-full detail.**
§1 ("n_fft = 512 … 585.9 Hz/bin … 1.7 ms"), §3.2 (same numbers), §5.2 (same numbers). Methods (§5.2) is the correct canonical home. §1 should keep the *intuition* (time-vs-frequency tradeoff) but can drop the exact 585.9 Hz figure; §3.2 can cite "the locked corpus grid (§5.2)" instead of re-listing. Currently a reader sees the same six numbers three times.

**5.4 — Minor transition gap into §3.2.**
§3.1 ends pointing at "the detector described next"; §3.2 opens well. But Figure 6 is introduced in §3.2 and then re-used as the *header* figure of §5.5 — fine — yet §3.2's closing sentence ("Turning this probability trace into reliable, countable USV events is the subject of the next section") is duplicated almost word-for-word by §3.3's opening. Tighten one of the two so the hand-off isn't doubled.

---

## TIER 6 — LENGTH vs 10–20 pp / ~7000–8500 word TARGET

**6.1 — The body is at or slightly above the upper word target before front/back matter.**
Section-by-section the Results + Methods are dense and partially redundant (Tiers 5.1–5.3). The Methods chapter (§5) re-states the detector architecture (already fully given in §3.2), the post-processing chain (already in §3.3), and the two-path distinction (already 3×) at near-Results length — Methods should be reproducible but can *reference* Results figures rather than re-prose the architecture. Estimated reclaimable: ~400–600 words from de-duplicating Tiers 5.1–5.3 without losing any reproducibility content. Recommendation: do that consolidation first; it simultaneously fixes the redundancy flags and pulls you comfortably inside the 8,500 ceiling. If, after de-duplication, you are still near 20 pp, the figure-heavy layout (10 figures) likely pushes page count above 20 — confirm the *page* count after typesetting, since 10 full-width figures can add 4–6 pages on top of the text.

**6.2 — No length *shortfall* risk.** The thesis is comprehensive; the risk is entirely on the upper bound. No section is too thin.

---

## SUMMARY — DO THESE IN ORDER
1. Cut abstract to ≤250 (currently 267) — delete final sentence + compress tooling clause.
2. Fill the title-page placeholder `[School / Mentor: to fill]`.
3. Cite or cut the 8 uncited bibliography entries (#2,3,5,7,12,13,14,16).
4. Fix §3.2's result→figure beat and the Figure 5/6 numbering inversion.
5. Add overlap statistic (or explicit "qualitative") to Figure 10's caption; add UMAP-axis note.
6. Reconcile §3.5's "essentially perfect rank separation" with §4's "floor, not power" framing.
7. De-duplicate the two-path passage (4×→1×+refs) and the STFT-constants block (3×→1×) — also solves length.

**Strongest, do-not-touch elements:** the §4 cage/SNR confound paragraph; Figures 8 & 9 captions (rubric-model); the verbatim "promising leads, not a proven difference" anchor repeated across chapters; the window-level-not-whole-file honesty disclaimer.
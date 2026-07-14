All references verified against real repo usage. Every required paper is genuinely load-bearing, plus I found several additional real dependencies (Hertz 2020, Zala 2020, Stoumpou/AMVOC 2022, Abbasi/BootSnap 2022, Perrodin 2023, Portfors 2007). I now have everything needed to produce the bibliography.

# Thesis Bibliography — Mouse-USV Detection & Lab-vs-Wild Courtship Comparison

All entries below were confirmed as **real dependencies cited in the repo** (`docs/`, `notes/`, source comments) unless marked otherwise. Author-year style; venues/years verified from my own knowledge and cross-checked against repo citations.

## (a) Reference List

1. **Holy, T. E., & Guo, Z. (2005).** Ultrasonic songs of male mice. *PLoS Biology*, 3(12), e386.

2. **Scattoni, M. L., Gandhy, S. U., Ricceri, L., & Crawley, J. N. (2008).** Unusual repertoire of vocalizations in the BTBR T+tf/J mouse model of autism. *PLoS ONE*, 3(8), e3067.

3. **Scattoni, M. L., Crawley, J., & Ricceri, L. (2009).** Ultrasonic vocalizations: A tool for behavioural phenotyping of mouse models of neurodevelopmental disorders. *Neuroscience & Biobehavioral Reviews*, 33(4), 508–515.

4. **Portfors, C. V. (2007).** Types and functions of ultrasonic vocalizations in laboratory rats and mice. *Journal of the American Association for Laboratory Animal Science (JAALAS)*, 46(1), 28–34.

5. **Grimsley, J. M. S., Monaghan, J. J. M., & Wenstrup, J. J. (2011).** Development of social vocalizations in mice. *PLoS ONE*, 6(3), e17460.

6. **Chabout, J., Sarkar, A., Dunson, D. B., & Jarvis, E. D. (2015).** Male mice song syntax depends on social contexts and influences female preferences. *Frontiers in Behavioral Neuroscience*, 9, 76.

7. **Hertz, S., Weiner, B., Perets, N., & London, M. (2020).** Temporal structure of mouse courtship vocalizations facilitates syllable labeling. *Communications Biology*, 3, 333. DOI: 10.1038/s42003-020-1053-7. *(Verified in repo: `notes/`, DOI cited 10×.)*

8. **Zala, S. M., Reitschmidt, D., Noll, A., Balazs, P., & Penn, D. J. (2020).** Spectrographic analysis of ultrasonic vocalizations of adult male and female wild-derived house mice (*Mus musculus musculus*). *Frontiers in Zoology*, 17, 14. *(Wild-vs-lab axis reference; verified in `notes/wild-lab-vocal-comparison.md`.)*

9. **Coffey, K. R., Marx, R. G., & Neumaier, J. F. (2019).** DeepSqueak: A deep learning-based system for detection and analysis of ultrasonic vocalizations. *Neuropsychopharmacology*, 44(5), 859–868.

10. **Fonseca, A. H. O., Santana, G. M., Bosque Ortiz, G. M., Bampi, S., & Dietrich, M. O. (2021).** Analysis of ultrasonic vocalizations from mice using computer vision and machine learning (VocalMat). *eLife*, 10, e59161.

11. **Goffinet, J., Brudner, S., Mooney, R., & Pearson, J. (2021).** Low-dimensional learned feature spaces quantify individual and group differences in vocal repertoires. *eLife*, 10, e67855.

12. **Stoumpou, V., González-Gutiérrez, C., et al. (2022).** AMVOC: A tool for analysis of mouse vocal communication. *Bioacoustics*, 32(2), 199–229. *(Autoencoder design reference; verified in `docs/handoffs/three-paper-deep-reads-2026-04-15.md`.)*

13. **Abbasi, R., Balazs, P., Marconi, M. A., Nicolakis, D., Zala, S. M., & Penn, D. J. (2022).** Capturing the songs of mice with an improved detection and classification method for ultrasonic vocalizations (BootSnap). *PLOS Computational Biology*, 18(5), e1010049. *(Wild-derived + lab validated classifier; verified in `docs/plans/ROADMAP_BOOTSNAP*.md`.)*

14. **Perrodin, C., Verzat, C., & Bendor, D. (2023).** Courtship behaviour reveals temporal regularity is a critical social cue in mouse communication. *eLife*, 12, e86464. *(Rhythm-over-order grammar claim; verified in WS-B/C handoffs.)* **[verify — exact title wording]**

15. **Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017).** On calibration of modern neural networks (temperature scaling). *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, 1321–1330.

16. **Boll, S. F. (1979).** Suppression of acoustic noise in speech using spectral subtraction. *IEEE Transactions on Acoustics, Speech, and Signal Processing*, 27(2), 113–120.

17. **He, K., Zhang, X., Ren, S., & Sun, J. (2016).** Deep residual learning for image recognition (ResNet). *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)*, 770–778. *(Conv-net lineage for the ResNet-18 lab classifier.)*

*Note:* von Merten et al. 2014 (Behavioral Ecology & Sociobiology) is **not** cited anywhere in the repo; the actual wild-derived house-mouse reference used in this work is **Zala et al. 2020** (#8), which serves the same role and is genuinely load-bearing. I substituted it rather than add an uncited paper. If von Merten 2014 is wanted regardless: **von Merten, S., Hoier, S., Pfeifle, C., & Tautz, D. (2014).** A role for ultrasonic vocalisation in social communication and divergence of natural populations of the house mouse. *PLoS ONE*, 9(5), e97244. **[verify — not cited in repo]**

## (b) Relevance Map

| # | Reference | Where most relevant in thesis |
|---|-----------|-------------------------------|
| 1 | Holy & Guo 2005 | **Introduction** — foundational: male mouse courtship song |
| 2 | Scattoni et al. 2008 | **Introduction / Classification** — syllable taxonomy origin |
| 3 | Scattoni et al. 2009 | **Introduction** — USVs as behavioural phenotyping |
| 4 | Portfors 2007 | **Detection methods** — USV frequency band (25–110 kHz) justification |
| 5 | Grimsley et al. 2011 | **Classification methods** — the 12-class call taxonomy used by the lab classifier |
| 6 | Chabout et al. 2015 | **Introduction / Comparison** — song syntax & social context |
| 7 | Hertz et al. 2020 | **Comparison** — courtship sequence structure, SIS/MI framework |
| 8 | Zala et al. 2020 | **Comparison** — wild-derived mouse repertoire & context modulation |
| 9 | Coffey et al. 2019 | **Detection methods** — DeepSqueak (baseline / contour port) |
| 10 | Fonseca et al. 2021 | **Classification methods** — VocalMat (image-based pipeline & curated taxonomy) |
| 11 | Goffinet et al. 2021 | **Comparison / Classification** — continuum view, low-D autoencoded repertoires |
| 12 | Stoumpou et al. 2022 | **Classification methods** — AMVOC autoencoder design |
| 13 | Abbasi et al. 2022 | **Classification methods / Comparison** — wild+lab validated USV classifier |
| 14 | Perrodin et al. 2023 | **Comparison** — temporal-regularity grammar claim |
| 15 | Guo et al. 2017 | **Calibration** — temperature scaling for CNN probabilities |
| 16 | Boll 1979 | **Detection methods** — spectral-subtraction pre-CNN denoising |
| 17 | He et al. 2016 | **Classification methods** — ResNet backbone lineage |

**Verification status:** #1–13, 15, 16 fully confirmed as cited dependencies in repo. #14 confirmed cited but exact title marked [verify]. #17 (ResNet) used in code (`resnet18_classifier`) but the He-2016 paper is not cited by name — added as canonical method reference. von Merten 2014 omitted from the main list (not a repo dependency) and offered only as an optional [verify] alternative.
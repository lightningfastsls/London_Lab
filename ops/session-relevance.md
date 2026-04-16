# Session Relevance Brief
<!-- Generated: 2026-04-16 19:06 -->
<!-- Method: topic-map-traversal + ripgrep -->

## USV Analysis Stage
- "BootSnap includes an explicit false-positive class alongside 11 USV syllable categories" (finding) -- unified noise class competing in softmax as alternative to two-stage FP filtering
- "VocalMat two-stage morphological filtering plus CNN noise classification achieves over 98 percent detection rate" (finding) -- hand-engineered first stage versus our model-derived approach
- "DAS temporal convolutional network achieves 98 percent precision and 99 percent recall on mouse USVs but requires raw audio input" (finding) -- highest detection metrics but raw-audio-only

## Dataset 9252
- "DAS temporal convolutional network achieves 98 percent precision and 99 percent recall on mouse USVs but requires raw audio input" (finding) -- highest reported detection metrics (Python, TensorFlow)
- "WhisperSeg adapts OpenAI Whisper transformer for animal vocalization segmentation with positive cross-species transfer" (finding) -- outperforms DAS with cross-species transfer
- "DeepSqueak v3 switched from Faster R-CNN to YOLO v2 improving speed and accuracy for USV detection" (finding) -- DeepSqueak's MATLAB-only detection architecture evolution

## DeepSqueak Classification Bridge
- "HDBSCAN re-clustering of our 7864 USV calls found only 3 natural clusters with 96 percent collapsing into one continuous manifold" (finding) -- own-data result: HDBSCAN collapses 27 k-means clusters to 3 (96% in one), independently confirming GMM k<=2
- "BootSnap includes an explicit false-positive class alongside 11 USV syllable categories" (finding) -- unified noise class competing in softmax as alternative to two-stage FP filtering
- "VocalMat provides 12954 labeled USV spectrograms freely available as training data" (finding) -- largest freely available labeled USV dataset (10,871 USVs + 2,083 noise)

## Phase 5.3
- "DeepSqueak k-means clustering on USV contour shape frequency and duration yielded 20 optimal syllable types via elbow method" (finding) -- k-means finds k=20 but GMM finds k<=2 on learned representations
- "SqueakOut autoencoder segmentation achieves Dice 90.2 designed to feed downstream unsupervised clustering pipelines" (finding) -- upstream segmentation improves downstream clustering quality
- "supervised bioacoustic foundation models vastly outperform self-supervised for species-level clustering" (finding) -- Muenster 2025: supervised 0.418 AMI vs self-supervised 0.256



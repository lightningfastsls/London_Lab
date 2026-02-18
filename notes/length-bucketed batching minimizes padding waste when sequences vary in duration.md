---
description: Sorting variable-length bout spectrograms into 6-8 duration buckets pads only to longest-in-batch, reducing wasted computation from global padding.
type: method
confidence: proven
topics:
  - "[[experimental-methods]]"
---

# length-bucketed batching minimizes padding waste when sequences vary in duration

Bout-level spectrograms span a wide duration range: individual bouts can be as short as 50 ms or as long as 10,000 ms. A naive batching strategy would pad every sequence to the maximum length in the dataset, filling a 50 ms bout with 9,950 ms of zeros to match the longest example. The resulting attention computation over padding tokens is wasted work that slows training without contributing signal.

Length-bucketed batching addresses this by sorting training examples into 6-8 discrete duration buckets with boundaries at 64, 128, 192, 256, 384, and 512 frames. Within each batch, samples are drawn from the same bucket, so padding extends only to the longest sequence in that bucket rather than across the entire dataset. For the majority of short bouts, this dramatically reduces the fraction of padding. Each padded position carries an attention mask with value 0 (padding) or 1 (real data), allowing the transformer to ignore padding positions in self-attention.

Bouts that exceed the maximum bucket length (512 frames) are chunked into overlapping segments with 50% overlap before bucketing, consistent with the temporal coverage strategy described in [[75 percent overlap with hop length 128 provides smooth temporal coverage for USV detection]]. This ensures that no bout is silently truncated; all segments are processed and their representations aggregated.

Bucket boundaries were chosen to align with powers of two or multiples of 64, which are memory-efficient for GPU tensor operations. The choice of 6-8 buckets balances between fine-grained length matching (many buckets, little padding) and within-bucket batch diversity (few buckets, more varied lengths per batch). This method applies to the transformer training stage; the CNN training pipeline uses fixed-size windows rather than variable-length bouts. See [[bout-level spectrograms preserve inter-USV timing context for transformer training]] for the motivation for using full-bout representations in the first place.

---

Source: [[ROADMAP.md]], Phase 4

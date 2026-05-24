---
description: "A shared encoder plus a domain discriminator with a gradient-reversal layer lets the same network learn discriminative class features while becoming bad at distinguishing source domains — no paired or balanced batches required"
type: methodology
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[model-adaptation]]"
  - "[[representation-learning]]"
  - "[[training-methodology]]"
---

# DANN gradient-reversal enforces invariance without per-batch domain matching

Standard domain adaptation methods (CORAL, MMD, paired-sample contrastive losses) require some form of cross-domain alignment within each training batch — you have to ensure both domains are represented, often in balanced quantities, to compute the alignment loss meaningfully. DANN (Ganin & Lempitsky 2015, `arXiv:1409.7495`) sidesteps this with a clever architectural trick.

Architecture: a shared encoder feeds two heads. The class head predicts the supervised target (e.g., 12-way syllable type). The domain head predicts the source domain (e.g., 2-way: `lab_131204` vs `vocalmat`). The standard cross-entropy loss is applied to both heads. The trick is a **gradient-reversal layer** (GRL) inserted between the encoder and the domain head: forward pass is identity, backward pass multiplies the gradient by −λ. So the encoder receives positive gradient from the class head (improve class prediction) and *negative* gradient from the domain head (make domain prediction *worse*).

What this produces in equilibrium: the encoder learns features that are discriminative for the class task (because the class head wants them) but uninformative for the domain task (because the encoder is rewarded for tricking the domain head). The discriminator can be trained to its current best on whatever the encoder produces, in any batch composition, asynchronously from the encoder updates — there is no per-batch alignment requirement. λ is annealed from 0 to 1 over training so the encoder isn't pushed to discard discriminative features too early (see [[DANN lambda schedule 0 to 1 prevents encoder collapse from aggressive adversarial loss]]).

The ~50-line implementation cost (a custom autograd Function plus a 2-layer MLP domain head) is small compared to the cage-invariance guarantees it provides. For the lab CNN classifier, this is the cheapest way to add cage invariance: no curated paired data, no balanced batching, no special sampler — just architecture.

---

Source: [[lab_cnn_classifier_plan_2026-05-20]]

Relevant Notes:
- [[DANN lambda schedule 0 to 1 prevents encoder collapse from aggressive adversarial loss]] — the training schedule that prevents trivial solutions
- [[classifiers trained on lab mice generalize poorly to wild mice requiring population-specific training data]] — the cross-domain failure DANN is trying to mitigate
- [[cage acoustics drive between-cohort spectrogram separation more than biology]] — the specific confound DANN is configured against in our pipeline

Topics:
- [[model-adaptation]]
- [[representation-learning]]
- [[training-methodology]]

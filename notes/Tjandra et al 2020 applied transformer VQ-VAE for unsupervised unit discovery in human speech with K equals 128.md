---
description: "The closest architectural analog to our approach — applied to human speech not animal vocalizations, using K=128 discrete codes"
type: finding
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
---

# Tjandra et al 2020 applied transformer VQ-VAE for unsupervised unit discovery in human speech with K equals 128

Tjandra et al. (Interspeech 2020) applied a transformer VQ-VAE architecture for unsupervised unit discovery in human speech, using K=128 discrete codes. This is the closest architectural analog to our approach: a transformer for sequence modeling combined with VQ-VAE for discretization. The key difference is domain — their work is on human speech, while [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]]. Their K=128 provides a reference point for our [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] at K=64. The smaller codebook in our case reflects the expected lower complexity of mouse USV repertoire compared to human phonemic inventory, while still providing headroom beyond the ~10-15 traditional categories.

---

Source:
- Researcher brain-dump on literature context (2026-02-19)
- Tjandra et al. (Interspeech 2020)

Relevant Notes:
- [[no published work has applied VQ-VAE to animal vocalizations making this a genuine research gap]] -- what makes our application novel
- [[codebook size of 64 gives interpretable discrete vocabulary with headroom beyond traditional USV types]] -- our K=64 vs their K=128
- [[transformer-first then VQ-VAE avoids forcing premature discretization]] -- shared architectural philosophy

Topics:
- [[classification]]

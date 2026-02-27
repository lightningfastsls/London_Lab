---
description: "All trained models (CNN classifier, VQ-VAE checkpoints) saved as .pt files via torch.save -- native format, easy loading, no extra dependencies"
type: decision
confidence: proven
conditions: []
meta_state: current
topics:
  - "[[classification]]"
  - "[[representation-learning]]"
---

# PyTorch pt format is the standard model artifact format giving native save-load with no extra dependencies

All trained models in this project use PyTorch's native `.pt` format via `torch.save()`. The CNN classifier saves a single `.pt` file with `state_dict`. The VQ-VAE saves `.pt` checkpoints containing model `state_dict`, optimizer state, and training metadata. Checkpoints are saved every N epochs during training.

Loading is straightforward: `torch.load()` + `model.load_state_dict()`. The format is not portable to non-PyTorch frameworks, but this is acceptable since the entire pipeline is PyTorch-based. No extra serialization dependencies (ONNX, TorchScript, etc.) are needed.

---

Source:
- DECISIONS.md (ADR-009) -- USV Spectrogram project architecture decisions

Relevant Notes:
- [[three convolutional blocks with global average pooling suffice for USV classification on small datasets]] -- the CNN model saved in this format
- [[300 kHz sample rate provides comfortable Nyquist headroom for mouse USVs up to 120 kHz]] -- sample rate metadata preserved in checkpoints

Topics:
- [[classification]]
- [[representation-learning]]

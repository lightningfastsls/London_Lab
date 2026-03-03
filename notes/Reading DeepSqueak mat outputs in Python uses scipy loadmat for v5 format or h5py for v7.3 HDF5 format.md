---
description: "Python interop with DeepSqueak .mat files requires checking MATLAB format version — scipy.io.loadmat handles v5, h5py handles v7.3 HDF5"
type: method
confidence: proven
conditions:
  - DeepSqueak v3.x may use either format depending on file size and MATLAB settings
meta_state: current
source: "inbox/deepsqueak-usv-syllable-classification-practical-guide.md"
topics:
  - "[[classification]]"
---

# Reading DeepSqueak mat outputs in Python uses scipy loadmat for v5 format or h5py for v7.3 HDF5 format

Reading DeepSqueak outputs in Python is straightforward once the MATLAB format version is identified. Use `scipy.io.loadmat()` for MATLAB v5 format or `h5py` for v7.3 (HDF5) format. DeepSqueak v3.x may produce either depending on MATLAB settings and variable sizes (MATLAB automatically switches to HDF5 for variables exceeding 2 GB).

All bounding box, score, type, and audio data are accessible as NumPy arrays after loading. This means Python scripts can read DeepSqueak's native .mat output directly — the Raven selection table import is needed only for feeding data *into* DeepSqueak, not for reading results *out*.

Our existing `deepsqueak_import.py` module handles this conversion as part of the classification bridge pipeline.

---

Source:
- Compass synthesis: inbox/deepsqueak-usv-syllable-classification-practical-guide.md

Relevant Notes:
- [[DeepSqueak requires MATLAB 2020a plus seven toolboxes and has no Python port]] -- the MATLAB dependency that necessitates this interop
- [[timestamp proximity matching with configurable tolerance bridges detection systems that use different internal time representations]] -- how we re-associate DeepSqueak results with our detections
- [[DeepSqueak Excel export provides 16 per-call metrics including principal frequency bandwidth slope and tonality]] -- alternative structured export; .mat is native, Excel is more accessible
- [[DeepSqueak built-in classification enables pre-VQ-VAE repertoire comparison between wild and lab populations]] -- strategic context for reading DeepSqueak outputs back into Python

Topics:
- [[classification-tools]]

---
description: "DeepSqueak and VocalMat remain MATLAB-only while DAS, SqueakOut, OpenSoundscape, Koogu, vak, Whombat, and Crowsetta form a now-comprehensive Python ecosystem"
type: finding
confidence: likely
meta_state: current
topics:
  - "[[detection]]"
---

# the Python vs MATLAB divide in USV tools is shrinking but remains a practical barrier

The bioacoustic annotation landscape reveals a clear platform divide: legacy USV tools (DeepSqueak, VocalMat, USVSEG) are MATLAB-based, while the growing ecosystem of newer tools is Python-native. The Python side now includes detection (DAS, SqueakOut), annotation (Whombat), ML training (OpenSoundscape, Koogu, vak), and interoperability (Crowsetta) — a more comprehensive coverage than the MATLAB ecosystem.

However, the MATLAB tools are not easily replaceable because they represent years of community adoption, validated parameters, and published benchmarks. [[DeepSqueak requires MATLAB 2020a plus seven toolboxes and has no Python port]] remains the most widely cited USV detection tool despite its platform limitations. [[DeepSqueak is fundamentally GUI-centric with no officially supported headless or scriptable operation]] compounds the problem — even with MATLAB available, automation requires workarounds.

The practical barrier is real: integrating DeepSqueak into automated Python pipelines requires either MATLAB licenses and engine bridges or format-level interop through Raven tables. Our DeepSqueak-to-Python bridge via Raven export navigates this divide at the format level, but the divide constrains which tools can participate in automated workflows.

The trajectory is clear — new tools are almost exclusively Python — but the transition will take years because published results and comparisons anchor researchers to the MATLAB tools they already know. The Python ecosystem now offers stronger individual components (DAS for detection, SqueakOut for segmentation), but the network effects of DeepSqueak's community keep it dominant.

---

Source:
- bioacoustic-annotation-tools-landscape-2025-research-2026-02-28 (archived to archive/inbox/)

Relevant Notes:
- [[DeepSqueak requires MATLAB 2020a plus seven toolboxes and has no Python port]] — the specific MATLAB dependency barrier
- [[DeepSqueak is fundamentally GUI-centric with no officially supported headless or scriptable operation]] — compounding the automation challenge

Topics:
- [[detection]]

# Verification Notes

## Environment
OS:
Python version:
Virtual environment:

## Commands executed (verbatim)

## Results (pass/fail)

## Output validation
Files produced:
Sizes:
Sanity checks:

## Failures (if any)
Root cause:
Fix summary:
Rerun transcript:
# Verification Transcript

Task: 2026-01-08_document-real-input-path-and-verify-on-real-data
Environment: Windows PowerShell, repo root D:\mickey_london_lab

## Commands and results

1) Create output directory
`New-Item -ItemType Directory -Force -Path tasks\2026-01-08_document-real-input-path-and-verify-on-real-data\verify_artifacts`
- Result: directory created.

2) Run primary script on a real WAV (expected sample rate check)
`python scripts\make_spectrogram.py --input "D:\mickey_london_lab\5970 USV\2024-09-30_11-21-24_0000039.wav" --output "D:\mickey_london_lab\tasks\2026-01-08_document-real-input-path-and-verify-on-real-data\verify_artifacts\2024-09-30_11-21-24_0000039_spectrogram.png"`
- Result:
```
ValueError: Expected 250000 Hz, got 300000 Hz.
```

3) Scan for 250 kHz WAVs
```
@'
from __future__ import annotations

from pathlib import Path
import soundfile as sf

folder = Path(r"D:\mickey_london_lab\5970 USV")
for path in sorted(folder.glob("*.wav")):
    with sf.SoundFile(str(path)) as wav:
        sr = int(wav.samplerate)
    if sr == 250000:
        print(path)
        break
else:
    print("NO_250K_FOUND")
'@ | python -
```
- Output:
```
NO_250K_FOUND
```

4) Generate spectrogram with sample-rate enforcement disabled (verification-only script)
```
@'
from __future__ import annotations

from pathlib import Path
import sys

repo = Path(r"D:\mickey_london_lab")
src_root = repo / "src"
if str(src_root) not in sys.path:
    sys.path.insert(0, str(src_root))

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.io_wav import load_wav_mono
from usv_spectrogram.render_tiles import render_png
from usv_spectrogram.spectrogram import compute_spectrogram_db

input_path = Path(r"D:\mickey_london_lab\5970 USV\2024-09-30_11-21-24_0000039.wav")
output_path = Path(r"D:\mickey_london_lab\tasks\2026-01-08_document-real-input-path-and-verify-on-real-data\verify_artifacts\2024-09-30_11-21-24_0000039_spectrogram.png")

samples, sample_rate_hz = load_wav_mono(input_path)
cfg = SpectrogramConfig(enforce_sample_rate=False, expected_sample_rate_hz=sample_rate_hz)
spec_db, freqs_hz, times_s = compute_spectrogram_db(samples, sample_rate_hz, cfg)
render_png(spec_db, freqs_hz, times_s, output_path, cfg, title=input_path.name)
print(output_path)
'@ | python -
```
- Output:
```
D:\mickey_london_lab\tasks\2026-01-08_document-real-input-path-and-verify-on-real-data\verify_artifacts\2024-09-30_11-21-24_0000039_spectrogram.png
```

5) Validate output PNG exists and non-empty
`Get-Item -Path "D:\mickey_london_lab\tasks\2026-01-08_document-real-input-path-and-verify-on-real-data\verify_artifacts\2024-09-30_11-21-24_0000039_spectrogram.png" | Format-List FullName,Length`
- Output:
```
FullName : D:\mickey_london_lab\tasks\2026-01-08_document-real-input-path-and-verify-on-real-data\verify_artifacts\2024-09-30_11-21-24_0000039_spectrogram.png
Length   : 2425511
```

## Notes
- Real input WAVs appear to be 300 kHz; default config enforces 250 kHz.
- Verification output generated via a short script using the existing modules with enforcement disabled.

---

# Verification Rerun (Auto Sample Rate)

Environment: Windows PowerShell, repo root D:\mickey_london_lab

## Commands and results

1) `$env:PYTHONDONTWRITEBYTECODE='1'; python -m py_compile scripts\make_spectrogram.py`
- Result: exit code 0, no stderr.

2) Run CLI with auto sample rate on real WAV
`python scripts\make_spectrogram.py --input "D:\mickey_london_lab\5970 USV\2024-09-30_11-21-24_0000039.wav" --auto-sample-rate --output "D:\mickey_london_lab\tasks\2026-01-08_document-real-input-path-and-verify-on-real-data\verify_artifacts\2024-09-30_11-21-24_0000039_spectrogram_auto.png"`
- Output:
```
Detected sample rate: 300000 Hz
```

3) Validate output PNG exists and non-empty
`Get-Item -Path "D:\mickey_london_lab\tasks\2026-01-08_document-real-input-path-and-verify-on-real-data\verify_artifacts\2024-09-30_11-21-24_0000039_spectrogram_auto.png" | Format-List FullName,Length`
- Output:
```
FullName : D:\mickey_london_lab\tasks\2026-01-08_document-real-input-path-and-verify-on-real-data\verify_artifacts\2024-09-30_11-21-24_0000039_spectrogram_auto.png
Length   : 2475725
```

## Notes
- Auto sample-rate flow works on real 300 kHz input.

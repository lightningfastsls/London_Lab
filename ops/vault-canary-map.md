# Vault Canary Map

Registry of source files annotated with `# VAULT:` comments pointing to knowledge graph notes.
Used for periodic audits: are canaries pointing to current notes? Are high-regression files covered?

## HIGH Risk (canary + `/kcheck` mandatory)

| File | Referenced Notes |
|------|-----------------|
| `src/usv_spectrogram/app/core/saved_detection_tracker.py` | saved-previous ghost detections current editable and saved-current form three aligned detection state tiers in the app |
| `src/usv_spectrogram/app/core/detection_logic.py` | saved-previous ghost detections current editable and saved-current form three aligned detection state tiers in the app |
| `src/usv_spectrogram/app/widgets/spectrogram_view.py` | saved-previous ghost detections..., visualization STFT uses different parameters than detection STFT by design |
| `src/usv_spectrogram/classification/deepsqueak_import.py` | DeepSqueak import previously required exact subdirectory name matches..., timestamp proximity matching with configurable tolerance... |
| `src/usv_spectrogram/classification/raven_export.py` | Raven selection table format is the standard interchange format..., DeepSqueak regenerates its own spectrograms from raw audio... |

## MEDIUM Risk (canary only)

| File | Referenced Notes |
|------|-----------------|
| `src/usv_spectrogram/detection/energy_detector.py` | 512-point FFT at 300 kHz gives 1.7 ms temporal resolution..., visualization STFT uses different parameters than detection STFT by design |

## Audit Checklist

- [ ] All referenced note titles resolve to existing files in `notes/`
- [ ] No HIGH-risk file has had its canary removed
- [ ] Notes referenced by canaries have `meta_state: current` (not outdated/superseded)
- [ ] Files with recent regressions are evaluated for canary addition

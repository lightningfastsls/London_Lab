"""Extract spectrograms from priority recordings for easy visual comparison.

Extracts spectrograms from recordings that had the most changes,
organized to make before/after comparison easy.
"""

import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from usv_spectrogram.extraction.spectrogram_extractor import extract_candidate_spectrogram


def extract_priority_samples():
    """Extract spectrograms from recordings with most changes."""
    print("=" * 80)
    print("EXTRACTING COMPARISON SPECTROGRAMS")
    print("=" * 80)
    print()

    # Priority recordings (had biggest splits from comparison)
    priority_recordings = [
        '2024-09-30_11-21-26_0000040.wav',  # 7 -> 14 (+7)
        '2024-09-30_11-20-32_0000025.wav',  # 11 -> 15 (+4)
        '2024-09-30_11-20-56_0000033.wav',  # 3 -> 6 (+3)
        '2024-09-30_11-20-41_0000029.wav',  # 13 -> 16 (+3)
    ]

    # Load candidates
    new_df = pd.read_csv('candidates_v2.csv')

    # Output directories
    output_dir = Path('spectrograms_comparison')
    output_dir.mkdir(exist_ok=True)

    wav_dir = Path('5970 USV')

    total_extracted = 0

    for recording in priority_recordings:
        print(f"Processing: {recording}")

        # Get candidates from this recording
        candidates = new_df[new_df['source_file'] == recording].copy()
        candidates = candidates.sort_values('start_ms')

        print(f"  Found {len(candidates)} candidates")

        wav_path = wav_dir / recording
        if not wav_path.exists():
            print(f"  Warning: WAV file not found, skipping")
            continue

        # Extract spectrograms
        for _, row in candidates.iterrows():
            candidate_id = row['candidate_id']
            output_path = output_dir / f"{candidate_id}.png"

            try:
                extract_candidate_spectrogram(
                    wav_path=wav_path,
                    candidate_id=candidate_id,
                    start_ms=row['start_ms'],
                    end_ms=row['end_ms'],
                    output_path=output_path,
                    context_before_ms=row['context_before_ms'],
                    context_after_ms=row['context_after_ms']
                )
                total_extracted += 1
            except Exception as e:
                print(f"  Error extracting {candidate_id}: {e}")

        print(f"  Extracted {len(candidates)} spectrograms")
        print()

    print(f"Total spectrograms extracted: {total_extracted}")
    print(f"Output directory: {output_dir}")
    print()

    # Create a summary file
    summary_path = output_dir / 'REVIEW_INSTRUCTIONS.md'
    summary_path.write_text(f"""# Spectrogram Comparison Review

**Purpose:** Visual review of tuned detection results from priority recordings

---

## Recordings Included

These recordings had the most candidate splits (likely multi-syllable calls):

1. **2024-09-30_11-21-26_0000040.wav** (7 → 14 candidates, +7)
2. **2024-09-30_11-20-32_0000025.wav** (11 → 15 candidates, +4)
3. **2024-09-30_11-20-56_0000033.wav** (3 → 6 candidates, +3)
4. **2024-09-30_11-20-41_0000029.wav** (13 → 16 candidates, +3)

---

## What to Look For

For each spectrogram, verify:

### ✅ Good (Complete Single USV):
- Single continuous frequency sweep
- Clear start and end
- Energy concentrated in narrow frequency band
- Duration 15-100ms (typical single syllable)

### ⚠️ Concerning (Partial USV):
- Appears to be cut off mid-sweep
- Frequency trajectory seems incomplete
- Looks like "half" of a USV

### ❌ Bad (Multi-Syllable Still Present):
- Multiple distinct frequency sweeps in one image
- Clear gaps between syllables but not split
- Duration > 120ms with multiple calls visible

---

## Review Process

1. Open spectrograms in image viewer
2. Sort by filename (groups by recording)
3. Look at each candidate sequentially
4. Note any concerning patterns

**Expected outcome:** All spectrograms show complete single USVs

**Red flags:**
- Many partial USVs (over-splitting)
- Any spectrograms still showing multiple syllables (under-splitting)

---

## Total Spectrograms

**Extracted:** {total_extracted} spectrograms from 4 priority recordings

**Review estimate:** ~10-15 minutes for visual scan

---

**Next:** If quality looks good, proceed with full dataset extraction
""")

    print(f"Review instructions: {summary_path}")
    print()

    return total_extracted


def main():
    total = extract_priority_samples()

    print("=" * 80)
    print("EXTRACTION COMPLETE")
    print("=" * 80)
    print()
    print(f"Extracted {total} spectrograms for review")
    print()
    print("Next steps:")
    print("  1. Open spectrograms_comparison/ folder")
    print("  2. Review REVIEW_INSTRUCTIONS.md")
    print("  3. Visually inspect spectrograms")
    print("  4. Verify all show complete single USVs (not partial or multi-syllable)")
    print()


if __name__ == '__main__':
    main()

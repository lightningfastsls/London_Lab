# CLI Tool Consolidation

Currently we have three separate scripts that share a lot of boilerplate:
- `scripts/generate_spectrograms.py` — generates spectrogram PNGs from WAV files
- `scripts/run_detection.py` — runs the energy detector on WAV files
- `scripts/export_detections.py` — exports detection results to CSV/JSON/Raven format

They all: parse WAV file arguments, load config, set up logging, handle the same errors. Let's merge them into one unified CLI tool.

## Step 1: Unified CLI Framework

Create `scripts/usv_cli.py` with subcommands: `spectrogram`, `detect`, `export`. Use argparse with subparsers. Common arguments (--wav-dir, --config, --output-dir, --verbose) go on the parent parser so they're shared across all subcommands.

## Step 2: Migrate Each Script

Move the core logic from each existing script into the unified CLI as subcommand handlers. Each handler calls the existing library functions — this is just reorganizing entry points, not rewriting logic.

Keep the original scripts as thin wrappers that import from usv_cli and call the right subcommand, for backwards compatibility. Something like:

```python
# scripts/generate_spectrograms.py (after migration)
from usv_cli import main
import sys
sys.exit(main(["spectrogram"] + sys.argv[1:]))
```

## Step 3: Add `pipeline` Subcommand

A new command that chains spectrogram → detect → export in one call. Useful for batch processing entire directories. Should support --skip-existing to avoid reprocessing files that already have outputs.

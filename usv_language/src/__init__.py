"""VQ-VAE + Transformer for USV language structure discovery.

Bootstrap sys.path so we can import from the existing usv_spectrogram package.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add the main project's src/ directory to sys.path so we can reuse
# usv_spectrogram modules (io_wav, _stft_core, etc.)
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_SRC_ROOT = _PROJECT_ROOT / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

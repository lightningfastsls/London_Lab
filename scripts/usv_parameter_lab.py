"""DEPRECATED Streamlit entrypoint for the USV Parameter Lab.

Superseded by the PyQt6 desktop app for interactive exploration.

This module calls os._exit(1) at import time. Direct invocation
(`python scripts/usv_parameter_lab.py`) prints the banner and exits.
`streamlit run` will still start its server, but the user script aborts
the script-runner thread before any UI is rendered, so the page is
permanently blank — there is no working Parameter Lab to reach.

Launch the PyQt6 app instead:

    .venv/bin/python scripts/run_app.py
"""

from __future__ import annotations

import os
import sys


sys.stderr.write(
    "\n[DEPRECATED] The Streamlit USV Parameter Lab has been retired.\n"
    "Use the PyQt6 desktop app instead:\n"
    "    .venv/bin/python scripts/run_app.py\n"
    "\n"
    "If you genuinely need to resurrect this Streamlit tool, revert this\n"
    "file from git history and restore the run() body in\n"
    "src/usv_spectrogram/param_lab/app.py.\n\n"
)
sys.stderr.flush()
os._exit(1)

"""Repo-wide pytest bootstrap.

Windows in this environment can fail to initialize PyTorch DLLs if PyQt6
loads first in the same process. Preloading torch here makes test collection
order-independent without changing application code.
"""

from __future__ import annotations

import sys


if sys.platform == "win32":
    import torch  # noqa: F401

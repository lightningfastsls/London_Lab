"""Reproducible feature/param persistence for the shape-invariance bake-off.

save_features(method, X, params) writes
    features/shape_invariance/{method}__{paramhash}.npy   (the (N,d) features)
    features/shape_invariance/{method}__{paramhash}.json  (the params + meta)
where paramhash = first 8 hex chars of md5 over the canonically-sorted params,
so sweeps are reproducible and diffable.
"""
from __future__ import annotations

import hashlib
import json
import os

import numpy as np


def param_hash(params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(blob).hexdigest()[:8]


def save_features(method: str, X: np.ndarray, params: dict,
                  outdir: str = "features/shape_invariance") -> str:
    """Write {method}__{hash}.npy + sibling .json. Returns the .npy path."""
    os.makedirs(outdir, exist_ok=True)
    h = param_hash(params)
    stem = os.path.join(outdir, f"{method}__{h}")
    npy_path = f"{stem}.npy"
    json_path = f"{stem}.json"
    X = np.asarray(X)
    np.save(npy_path, X)
    meta = {
        "method": method,
        "paramhash": h,
        "params": params,
        "shape": list(X.shape),
        "dtype": str(X.dtype),
    }
    with open(json_path, "w") as fp:
        json.dump(meta, fp, indent=2, default=str)
    return npy_path

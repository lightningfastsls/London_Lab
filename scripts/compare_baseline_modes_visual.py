"""Side-by-side visual comparison of baseline_mode='percentile' vs 'median_envelope'.

Picks 30 s of a lab WAV (known to have USVs), runs the full 4-layer
cleaning stack twice (only baseline_mode differs), saves:
- Full cleaned spectrograms (overview)
- Six matched 0.22 s patches per mode at the same time positions
- An HTML page showing them side by side

Output: $CLAUDE_JOB_DIR/baseline_compare/ (overview PNGs + per-patch PNGs +
index.html). Diagnostic-only, does not modify any production code.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
SRC = REPO_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from usv_spectrogram.classifier import CleaningConfig, clean_spectrogram  # noqa: E402
import importlib.util  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "cnn_prepare_training_data", SCRIPTS / "cnn_prepare_training_data.py"
)
prep = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prep)


def _spec_to_pil(spec_2d: np.ndarray) -> Image.Image:
    lo, hi = float(spec_2d.min()), float(spec_2d.max())
    if hi - lo < 1e-9:
        scaled = np.zeros_like(spec_2d, dtype=np.uint8)
    else:
        scaled = ((spec_2d - lo) / (hi - lo) * 255.0).astype(np.uint8)
    return Image.fromarray(scaled[::-1])


def _patch_to_pil(spec_2d: np.ndarray) -> Image.Image:
    arr = prep._spec_to_uint8_patch(spec_2d)
    return Image.fromarray(arr[::-1])


def main() -> int:
    out_dir = Path(os.environ.get("CLAUDE_JOB_DIR", "/tmp")) / "baseline_compare"
    out_dir.mkdir(parents=True, exist_ok=True)

    lab_wav = Path("/home/shachar/projects/mickey_london_lab/USV_lab_131204/131204_1400_m1fm1.wav")
    print(f"loading {lab_wav.name} ...")
    samples, sr = sf.read(str(lab_wav), dtype="float32")
    if samples.ndim > 1:
        samples = samples[:, 0]
    chunk = samples[: int(30 * sr)]
    print(f"  using first 30s, samples={chunk.shape[0]:,}, sr={sr}")

    print("computing STFT (shared) ...")
    spec_db = prep._spectrogram_db(chunk, sr)
    print(f"  spec shape={spec_db.shape}")

    results = {}
    for mode in ("percentile", "median_envelope"):
        cfg = CleaningConfig(baseline_mode=mode)
        t0 = time.perf_counter()
        cleaned = clean_spectrogram(spec_db, cfg, lab_wav.stem)
        dt = time.perf_counter() - t0
        print(f"  [{dt:7.2f}s] {mode}: shape={cleaned.shape} mean={float(cleaned.mean()):+.3f} std={float(cleaned.std()):.3f} min={float(cleaned.min()):+.3f} max={float(cleaned.max()):+.3f}")
        results[mode] = cleaned

    # --- Save full-window overview spectrograms ---
    print("saving overview spectrograms ...")
    for mode, cleaned in results.items():
        _spec_to_pil(cleaned).save(out_dir / f"overview_{mode}.png")

    # --- Save matched patches at six evenly-spaced time positions ---
    patch_duration_s = 0.22
    frames_per_patch = max(
        1, int(round(patch_duration_s * sr / prep._VOCALMAT_STFT_HOP))
    )
    n_time = results["percentile"].shape[1]
    n_patches_available = max(1, n_time // frames_per_patch)
    print(f"  frames_per_patch={frames_per_patch}, total patches available={n_patches_available}")

    # Pick 6 patches: equally spaced, including ends.
    sample_idx = np.linspace(0, n_patches_available - 1, 6, dtype=int).tolist()
    print(f"  sample patch indices: {sample_idx}")

    rows = []
    for pi in sample_idx:
        t_start = pi * frames_per_patch
        t_end = min(t_start + frames_per_patch, n_time)
        t_sec = t_start * prep._VOCALMAT_STFT_HOP / sr
        per_mode_files = {}
        for mode, cleaned in results.items():
            slab = cleaned[:, t_start:t_end]
            fn = f"patch_{mode}_p{pi:04d}.png"
            _patch_to_pil(slab).save(out_dir / fn)
            per_mode_files[mode] = fn
        rows.append((pi, t_sec, per_mode_files))

    # --- HTML index ---
    print("writing index.html ...")
    html_path = out_dir / "index.html"
    html_path.write_text(_render_html(rows, results), encoding="utf-8")

    print(f"\nDONE. Output in: {out_dir}")
    print(f"HTML: {html_path}")
    print(f"WSL URL: file://wsl.localhost/Ubuntu{html_path}")
    return 0


def _render_html(rows, results) -> str:
    stats = {
        mode: {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        }
        for mode, arr in results.items()
    }
    rows_html = []
    for pi, t_sec, files in rows:
        rows_html.append(f"""
        <div class="row">
          <div class="t">t={t_sec:.2f}s (patch #{pi})</div>
          <div class="pair">
            <figure><img src="{files['percentile']}" alt="percentile p{pi}"><figcaption>percentile</figcaption></figure>
            <figure><img src="{files['median_envelope']}" alt="median_envelope p{pi}"><figcaption>median_envelope</figcaption></figure>
          </div>
        </div>""")
    rows_html = "\n".join(rows_html)
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>baseline_mode comparison</title>
<style>
body {{ font-family: -apple-system, system-ui, sans-serif; max-width: 1200px; margin: 2rem auto; padding: 1rem; background: #111; color: #eee; }}
h1 {{ font-size: 1.4rem; }} h2 {{ font-size: 1.1rem; margin-top: 2rem; border-bottom: 1px solid #444; padding-bottom: .2rem; }}
table {{ border-collapse: collapse; }} td, th {{ padding: .25rem .8rem; border-bottom: 1px solid #333; }}
.overview {{ display: flex; gap: 1rem; flex-wrap: wrap; }}
.overview figure {{ margin: 0; }}
.overview img {{ width: 560px; height: auto; border: 1px solid #555; image-rendering: pixelated; }}
.row {{ margin: 1rem 0; }} .t {{ color: #aaa; margin-bottom: .3rem; font-family: monospace; }}
.pair {{ display: flex; gap: 1rem; }}
.pair figure {{ margin: 0; }}
.pair img {{ width: 256px; height: 256px; border: 1px solid #555; image-rendering: pixelated; }}
figcaption {{ color: #888; font-size: .85rem; text-align: center; margin-top: .2rem; font-family: monospace; }}
</style></head><body>
<h1>Baseline-mode comparison &mdash; <code>131204_1400_m1fm1.wav</code> (first 30s)</h1>
<p>Identical input through the 4-layer cleaning stack. Only <code>baseline_mode</code> differs.
soft-notch is a no-op (no tonal library). global_mad and per_recording_zscore are identical.</p>

<h2>Output statistics</h2>
<table>
<tr><th>mode</th><th>mean</th><th>std</th><th>min</th><th>max</th></tr>
<tr><td>percentile</td><td>{stats['percentile']['mean']:+.3f}</td><td>{stats['percentile']['std']:.3f}</td><td>{stats['percentile']['min']:+.3f}</td><td>{stats['percentile']['max']:+.3f}</td></tr>
<tr><td>median_envelope</td><td>{stats['median_envelope']['mean']:+.3f}</td><td>{stats['median_envelope']['std']:.3f}</td><td>{stats['median_envelope']['min']:+.3f}</td><td>{stats['median_envelope']['max']:+.3f}</td></tr>
</table>

<h2>Full 30s overview (downsampled for display)</h2>
<div class="overview">
  <figure><img src="overview_percentile.png" alt="percentile"><figcaption>percentile (~1 s)</figcaption></figure>
  <figure><img src="overview_median_envelope.png" alt="median_envelope"><figcaption>median_envelope (~250 s)</figcaption></figure>
</div>

<h2>Matched 0.22 s patches at six time positions</h2>
{rows_html}

<h2>What to look for</h2>
<ul>
  <li><b>USV preservation</b>: are call traces still visible at both modes?</li>
  <li><b>Cage tone removal</b>: is the horizontal 50.4&ndash;51 kHz line gone in both?</li>
  <li><b>Noise floor</b>: does one mode leave more residual noise than the other?</li>
  <li><b>Contrast</b>: percentile has lower std (35 vs 55) &mdash; less dynamic range, but is that visible?</li>
</ul>
</body></html>
"""


if __name__ == "__main__":
    raise SystemExit(main())

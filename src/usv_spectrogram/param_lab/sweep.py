"""Sweep helpers for the USV Parameter Lab."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path
import time

from usv_spectrogram.config import SpectrogramConfig
from usv_spectrogram.io_wav import load_wav_segment_mono
from usv_spectrogram.render_tiles import render_png
from usv_spectrogram.spectrogram import compute_spectrogram_db
from usv_spectrogram.utils import ensure_dir

from .metrics import compute_metrics


def _sanitize_label(label: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in label)
    return cleaned.strip("_") or "sweep"


def run_sweep(
    path: str | Path,
    start_s: float,
    duration_s: float,
    cfgs: list[tuple[str, SpectrogramConfig]],
    display_gain_db: float,
    display_range_db: float,
    output_dir: str | Path,
) -> dict[str, object]:
    """Run a parameter sweep on a WAV segment and export images and a report."""
    output_dir = ensure_dir(output_dir)
    samples, sample_rate_hz = load_wav_segment_mono(path, start_s, duration_s)
    results: list[dict[str, object]] = []

    for label, cfg in cfgs:
        t0 = time.perf_counter()
        spec_db, freqs_hz, times_s = compute_spectrogram_db(samples, sample_rate_hz, cfg)
        elapsed_s = time.perf_counter() - t0
        metrics = compute_metrics(spec_db, freqs_hz, times_s)

        display_cfg = replace(cfg, gain_db=display_gain_db, range_db=display_range_db)
        safe_label = _sanitize_label(label)
        output_path = output_dir / f"{safe_label}.png"
        render_png(spec_db, freqs_hz, times_s, output_path, display_cfg, title=label)

        results.append(
            {
                "label": label,
                "output_png": str(output_path),
                "elapsed_s": float(elapsed_s),
                "metrics": metrics,
                "config": asdict(cfg),
            }
        )

    configs_path = output_dir / "configs.json"
    metrics_path = output_dir / "metrics.json"
    report_path = output_dir / "report.md"

    configs_payload = {
        "segment": {
            "path": str(path),
            "start_s": float(start_s),
            "duration_s": float(duration_s),
            "sample_rate_hz": float(sample_rate_hz),
        },
        "configs": [item["config"] for item in results],
    }
    configs_path.write_text(json.dumps(configs_payload, indent=2), encoding="utf-8")

    metrics_payload = [
        {
            "label": item["label"],
            "elapsed_s": item["elapsed_s"],
            "metrics": item["metrics"],
            "output_png": item["output_png"],
        }
        for item in results
    ]
    metrics_path.write_text(json.dumps(metrics_payload, indent=2), encoding="utf-8")

    lines = [
        "# USV Parameter Lab Sweep Report",
        "",
        f"Segment: {path}",
        f"Start (s): {start_s:.3f}",
        f"Duration (s): {duration_s:.3f}",
        "",
        "| Label | PNG | Noise Floor (dB) | Contrast (dB) | Frames | Time (s) |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for item in results:
        metrics = item["metrics"]
        lines.append(
            "| {label} | {png} | {noise:.2f} | {contrast:.2f} | {frames:.0f} | {elapsed:.3f} |".format(
                label=item["label"],
                png=Path(item["output_png"]).name,
                noise=metrics.get("noise_floor_db", 0.0),
                contrast=metrics.get("contrast_db", 0.0),
                frames=metrics.get("frames", 0.0),
                elapsed=item["elapsed_s"],
            )
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "report": str(report_path),
        "configs": str(configs_path),
        "metrics": str(metrics_path),
        "count": len(results),
    }

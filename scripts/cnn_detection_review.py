"""Human-judged detection review — APP-FAITHFUL (corrected 2026-05-27).

PRIOR BUG: the first version applied the FP-filter + temperature scaling +
energy_threshold=0.1 + postprocessing.hysteresis. The interactive app does NONE
of those — its Detect button is CNN -> hysteresis only. The FP-filter wrongly
killed hard_neg's real detections (8->0 on 0004954, which the app detects fine).

This version reproduces the app's EXACT Detect pipeline (verified in
app/main_window.py InferenceWorker + HysteresisDetector usage):
  SlidingInference(window_width_px=100, hop_px=10, energy_threshold=0.35,
                   enable_per_window_norm=False)  -> RAW probs (no temperature)
  HysteresisDetector(high=0.6, low=0.4, merge_gap_columns=3,
                     min_duration_ms=10, max_duration_ms=500,
                     min_sustained_prob=0.0, exclude_start/end_sec=0.0)
  NO FP-filter, NO temperature, NO soft-notch.

Each model on its own best extraction (per Shachar):
  - first CNN : native inferno/25-110, 150px window (its original deployment),
                global MAD via SlidingInference + plt.get_cmap->inferno monkeypatch.
  - matched   : app default magma/20-120, 100px.
  - hard_neg  : app default magma/20-120, 100px  (== exactly what the app runs).

Authorized legacy override (first CNN only): ExtractionConfig(25_000,110_000,
colormap='inferno'). Defaults / corpus.py untouched (CNN-FREEZE).
"""
import sys
from pathlib import Path

import numpy as np
import librosa
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

REPO = Path("/home/shachar/projects/mickey_london_lab")
sys.path.insert(0, str(REPO / "src"))
from usv_spectrogram.app.core.sliding_inference import SlidingInference          # noqa: E402
from usv_spectrogram.app.core.audio_loader import AudioLoader                     # noqa: E402
from usv_spectrogram.app.core.detection_logic import HysteresisDetector          # noqa: E402
from usv_spectrogram.detection.extraction_config import ExtractionConfig          # noqa: E402
from usv_spectrogram.corpus import SAMPLE_RATE_HZ                                 # noqa: E402

SR = SAMPLE_RATE_HZ
ENERGY = 0.35       # app value
FIRST_WIDTH = 150   # first CNN's original deployment window
LEG = ExtractionConfig(freq_min_hz=25_000, freq_max_hz=110_000, colormap="inferno")
OUTDIR = REPO / "presentation/figures/cnn_improvement/detection_review"

FILES = [  # all 12 (batch 1 + 2)
    "2024-10-01_05-57-44_0004783", "2024-10-01_01-03-15_0004134", "2024-10-01_08-58-52_0004954",
    "2024-09-30_12-28-58_0000551", "2024-09-30_17-11-57_0001700", "2024-09-30_17-42-51_0001930",
    "2024-09-30_21-34-29_0003189", "2024-09-30_12-19-37_0000491", "2024-09-30_12-19-52_0000494",
    "2024-09-30_17-03-56_0001628", "2024-09-30_17-45-49_0001960", "2024-09-30_18-30-47_0002273",
]


def resolve_wav(stem):
    p = REPO / "5970_manual_review_reviewed" / f"{stem}.wav"
    if p.exists():
        return p
    hits = list((REPO / "5970").rglob(f"{stem}.wav"))
    return hits[0] if hits else None


def detect(model_si, spectrogram_db, times, detector, native):
    if native:
        orig = plt.get_cmap
        plt.get_cmap = lambda *a, **k: orig("inferno")
        try:
            res = model_si.infer(spectrogram_db, times)
        finally:
            plt.get_cmap = orig
    else:
        res = model_si.infer(spectrogram_db, times)
    dr = detector.detect(res.probabilities, res.column_indices, res.times)
    return dr.usvs


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    print("=" * 78)
    print("DETECTION REVIEW — APP-FAITHFUL (CNN -> hysteresis only; NO FP-filter/temperature)")
    print(f"[params] SlidingInference: hop=10 energy={ENERGY} per_win_norm=OFF; "
          f"first CNN width={FIRST_WIDTH}px (native inferno/25-110), matched+hard_neg width=100px (magma/20-120)")
    print(f"[params] HysteresisDetector: high=0.6 low=0.4 merge_gap_cols=3 dur=10-500ms "
          f"min_sustained=0.0 exclude_start/end=0.0 (exactly the app)")

    si_first = SlidingInference(str(REPO / "models/production/best_model.pt"),
                                window_width_px=FIRST_WIDTH, hop_px=10, energy_threshold=ENERGY,
                                enable_per_window_norm=False)
    si_matched = SlidingInference(str(REPO / "models/matched_windows/best_model.pt"),
                                  window_width_px=100, hop_px=10, energy_threshold=ENERGY,
                                  enable_per_window_norm=False)
    si_hard = SlidingInference(str(REPO / "models/hard_neg_retrain/best_model.pt"),
                               window_width_px=100, hop_px=10, energy_threshold=ENERGY,
                               enable_per_window_norm=False)
    default_loader = AudioLoader()
    native_loader = AudioLoader(config=LEG)

    def new_detector():
        return HysteresisDetector(high_threshold=0.6, low_threshold=0.4, merge_gap_columns=3,
                                  min_duration_ms=10.0, max_duration_ms=500.0,
                                  min_sustained_prob=0.0, exclude_start_sec=0.0, exclude_end_sec=0.0)

    MODEL_ORDER = [("first CNN (native inferno/25-110, 150px)", "#d62728"),
                   ("matched_windows (app default)", "#5c8fb3"),
                   ("hard_neg / production (== the app)", "#1a7d1a")]
    verdict_rows = []

    for stem in FILES:
        wav = resolve_wav(stem)
        if wav is None:
            print(f"  !! missing {stem}"); continue
        samples, _ = librosa.load(str(wav), sr=SR, mono=True)
        audio = default_loader.load(str(wav))
        bd_db, bd_freqs, _ = default_loader._compute_spectrogram(samples, SR)
        dur_s = float(audio.times[-1]) if len(audio.times) else len(samples) / SR
        fmin_khz, fmax_khz = bd_freqs[0] / 1e3, bd_freqs[-1] / 1e3

        na = native_loader.load(str(wav))
        ev_first = detect(si_first, na.spectrogram_db, na.times, new_detector(), native=True)
        ev_matched = detect(si_matched, audio.spectrogram_db, audio.times, new_detector(), native=False)
        ev_hard = detect(si_hard, audio.spectrogram_db, audio.times, new_detector(), native=False)
        events = [ev_first, ev_matched, ev_hard]
        print(f"\n[{stem}] APP-FAITHFUL detections: first={len(ev_first)}  "
              f"matched={len(ev_matched)}  hard_neg={len(ev_hard)}")

        fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
        for ax, (label, color), evs in zip(axes, MODEL_ORDER, events):
            ax.imshow(bd_db, aspect="auto", origin="lower", cmap="magma",
                      extent=[0, dur_s, fmin_khz, fmax_khz],
                      vmin=np.percentile(bd_db, 5), vmax=np.percentile(bd_db, 99))
            for e in evs:
                ax.add_patch(Rectangle((e.start_time_s, fmin_khz),
                                       e.end_time_s - e.start_time_s, fmax_khz - fmin_khz,
                                       fill=False, edgecolor=color, lw=2.0))
                ax.text(e.start_time_s, fmax_khz * 0.96, f"{e.max_probability:.2f}",
                        color=color, fontsize=8, fontweight="bold", va="top")
            ax.set_ylabel("kHz", fontsize=9)
            ax.set_title(f"{label}  —  {len(evs)} detections", fontsize=10, color=color,
                         loc="left", fontweight="bold")
        axes[-1].set_xlabel("time (s)", fontsize=10)
        fig.suptitle(f"{stem}   (APP-FAITHFUL: CNN+hysteresis, no FP-filter)   "
                     "judge hits / FP / miss per panel", fontsize=12, fontweight="bold")
        fig.tight_layout(rect=[0, 0, 1, 0.97])
        out = OUTDIR / f"{stem}__3model_detections.png"
        fig.savefig(out, dpi=140, bbox_inches="tight")
        plt.close(fig)

        for mname, evs in zip(["first_cnn", "matched", "hard_neg"], events):
            for i, e in enumerate(evs):
                verdict_rows.append({"stem": stem, "model": mname, "det_idx": i,
                                     "start_s": round(e.start_time_s, 3), "end_s": round(e.end_time_s, 3),
                                     "peak_prob": round(e.max_probability, 3), "verdict_hit_or_fp": ""})

    vdf = pd.DataFrame(verdict_rows)
    vdf.to_csv(OUTDIR / "detection_review_verdicts.csv", index=False)
    print(f"\n[summary] APP-FAITHFUL per-model detection counts:")
    print(vdf.groupby("model").size().to_string())
    print(f"[done] corrected figures + CSV in {OUTDIR}")


if __name__ == "__main__":
    main()

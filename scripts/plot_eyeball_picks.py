"""Render stratified eyeball samples to test the duration filter.

Reads eyeball_picks.parquet, renders each event with the same plotting
function used for the longest-event spectrograms. Output organized into
three subfolders matching the stratum:

  - high_risk_slipped/  : 250-299 ms events in chunks with the strong 61.6 kHz tonal
                          (events that survived the <300ms filter)
  - medium_risk/         : 100-200 ms events in tonal-heavy chunks
  - clean_control/       : events in chunks WITHOUT any unmatched tonal — what
                          a real lab USV should look like
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from plot_long_event_spectrograms import _render, WAV_DIR  # noqa: E402

BATCH = Path("results/batch_lab_full_softnotch_20260513_1538")
DEFAULT_PICKS_PATH = BATCH / "eyeball_picks.parquet"
OUT_ROOT = BATCH / "eyeball_inspection"


def main() -> None:
    picks_path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PICKS_PATH
    picks = pd.read_parquet(picks_path)
    picks["unmatched_centers_khz"] = picks["unmatched_centers_khz"].apply(list)

    for _, ev in picks.iterrows():
        stratum_dir = OUT_ROOT / ev["stratum"]
        stratum_dir.mkdir(parents=True, exist_ok=True)
        wav_path = WAV_DIR / f"{ev['stem']}.wav"
        if not wav_path.exists():
            print(f"MISSING: {wav_path}")
            continue
        out_path = stratum_dir / f"{ev['stem']}_{int(ev['duration_ms'])}ms.png"
        ev_for_render = pd.Series({
            "duration_ms": float(ev["duration_ms"]),
            "start_s": float(ev["start_s"]),
            "end_s": float(ev["end_s"]),
            "max_prob": float(ev["max_prob"]),
            "unmatched_centers_khz": ev["unmatched_centers_khz"],
        })
        _render(wav_path, ev_for_render, out_path)
        print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
